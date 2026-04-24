"""
backtest_circuit_breaker_2025.py — Does the -1.5r circuit breaker mute wins?

Method:
  - Use 2025 games from feature_matrix_enriched_v2 (close_total + total_runs).
  - Treat Vegas close_total as the 'baseline projection' (no in-house model
    projections were persisted for 2025 so this is the honest substitute).
  - residual_i = close_total_i - total_runs_i   (+ => Vegas over-projected)
  - For each game in date order, compute point-in-time dynamic bias using the
    same BIAS_WINDOW_DAYS / BLEND / DAMPING constants as the live thermostat
    (from score_run_dist_today), but WITHOUT the 1.5r clamp (uncapped) and
    WITH the clamp (capped). Apply each offset to close_total to produce the
    capped and uncapped adjusted projections.
  - Over-betting rule: flag OVER when projected_total_adj > close_total + 0.5.
  - Score: hit rate = count(actual > close_total) among flagged OVER picks.

Question: In high-scoring / hot environments where the uncapped offset would
have been aggressive, does the 1.5r cap turn would-be winners into folds?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_run_dist_today as srd

FEATURE_MTX = Path("feature_matrix_enriched_v2.parquet")
OUT_CSV = Path("data/logs/circuit_breaker_backtest_2025.csv")

BASELINE_OVER_MARGIN = 0.5   # flag OVER when adj - vegas > 0.5


def pit_offset(park_rows: pd.DataFrame, window: pd.DataFrame,
                use_cap: bool) -> tuple[float, str]:
    """Replay the live thermostat at one point in time."""
    if len(window) < srd.BIAS_MIN_ROWS:
        return srd.GLOBAL_BIAS_FALLBACK, "fallback"
    recent = window.tail(srd.BIAS_MAX_ROWS)
    global_r = float(recent["residual"].mean())
    if len(park_rows) >= srd.BIAS_PARK_MIN_ROWS:
        park_r = float(park_rows["residual"].mean())
        # Safety valve
        if global_r < 0 and park_r > 0:
            raw = -park_r * srd.BIAS_DAMPING
            label = "safety_valve"
        else:
            blended = (1 - srd.BIAS_PARK_BLEND_WEIGHT) * global_r + \
                      srd.BIAS_PARK_BLEND_WEIGHT * park_r
            raw = -blended * srd.BIAS_DAMPING
            label = "blend"
    else:
        raw = -global_r * srd.BIAS_DAMPING
        label = "global"
    if use_cap:
        clamped = max(-srd.MAX_BIAS_ADJUSTMENT, min(srd.MAX_BIAS_ADJUSTMENT, raw))
        return clamped, label + ("_clamped" if clamped != raw else "")
    return raw, label


def main():
    fm = pd.read_parquet(FEATURE_MTX,
                          columns=["game_date", "home_team", "away_team",
                                   "close_total", "total_runs", "temp_f"])
    fm["game_date"] = pd.to_datetime(fm["game_date"])
    fm25 = fm[(fm["game_date"].dt.year == 2025)]\
              .dropna(subset=["close_total", "total_runs"])\
              .sort_values("game_date").reset_index(drop=True)
    print(f"2025 games with vegas+actual: {len(fm25):,}")

    # Build per-game residuals (baseline = vegas close). Walk forward.
    log = pd.DataFrame(columns=["date", "home_park_id", "residual"])
    rows = []
    for idx, r in fm25.iterrows():
        d = r["game_date"]
        park = r["home_team"]
        cutoff = d - pd.Timedelta(days=srd.BIAS_WINDOW_DAYS)
        past = log[log["date"] < d]
        window = past[past["date"] >= cutoff]
        park_rows = window[window["home_park_id"] == park]
        off_u, lab_u = pit_offset(park_rows, window, use_cap=False)
        off_c, lab_c = pit_offset(park_rows, window, use_cap=True)
        rows.append({
            "date": d, "home_team": park, "away_team": r["away_team"],
            "close_total": r["close_total"], "actual_total": r["total_runs"],
            "temp_f": r["temp_f"],
            "offset_uncapped": off_u, "offset_capped": off_c,
            "was_clamped": off_u != off_c,
            "label_uncapped": lab_u, "label_capped": lab_c,
            "adj_uncapped": r["close_total"] + off_u,
            "adj_capped":   r["close_total"] + off_c,
        })
        # Append this game's residual for future iterations
        log = pd.concat([log, pd.DataFrame([{
            "date": d, "home_park_id": park,
            "residual": r["close_total"] - r["total_runs"]
        }])], ignore_index=True)
        # Prune oldest to BIAS_MAX_ROWS * 3 for speed
        if len(log) > 1500:
            log = log.sort_values("date").tail(1500).reset_index(drop=True)

    bt = pd.DataFrame(rows)
    bt["actual_over_vegas"] = bt["actual_total"] > bt["close_total"]
    bt["uncapped_says_over"] = bt["offset_uncapped"] > BASELINE_OVER_MARGIN
    bt["capped_says_over"]   = bt["offset_capped"] > BASELINE_OVER_MARGIN
    bt["only_uncapped_over"] = bt["uncapped_says_over"] & ~bt["capped_says_over"]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    bt.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(bt)} rows -> {OUT_CSV}")

    # ─── Audit ─────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("CIRCUIT BREAKER BACKTEST — 2025 SEASON")
    print("=" * 64)
    n = len(bt)
    clamped = int(bt["was_clamped"].sum())
    print(f"Total games:            {n:,}")
    print(f"Games clamped by -1.5r: {clamped:,}  ({clamped/n*100:.2f}%)")

    def hit(sub, col="actual_over_vegas"):
        if not len(sub): return (0, 0, float("nan"))
        h = int(sub[col].sum()); t = len(sub)
        return (h, t, h/t)

    print("\nFlagged OVER hit rates (actual_total > close_total):")
    h, t, r = hit(bt[bt["capped_says_over"]]);   print(f"  capped recommends:    {h}/{t} = {r:.3%}")
    h, t, r = hit(bt[bt["uncapped_says_over"]]); print(f"  uncapped recommends:  {h}/{t} = {r:.3%}")
    sub = bt[bt["only_uncapped_over"]]
    h, t, r = hit(sub)
    print(f"  ONLY uncapped (cap mutes signal): {h}/{t} = {r:.3%}")

    print("\nHigh-scoring subset (actual_total >= 10, n=%d):" %
          int((bt["actual_total"] >= 10).sum()))
    hs = bt[bt["actual_total"] >= 10]
    h, t, r = hit(hs[hs["capped_says_over"]]);   print(f"  capped recommends:    {h}/{t} = {r:.3%}")
    h, t, r = hit(hs[hs["uncapped_says_over"]]); print(f"  uncapped recommends:  {h}/{t} = {r:.3%}")
    h, t, r = hit(hs[hs["only_uncapped_over"]]); print(f"  ONLY uncapped:        {h}/{t} = {r:.3%}")

    print("\nHot-weather subset (temp_f >= 85, n=%d):" %
          int((bt["temp_f"] >= 85).sum()))
    hw = bt[bt["temp_f"] >= 85]
    h, t, r = hit(hw[hw["capped_says_over"]]);   print(f"  capped recommends:    {h}/{t} = {r:.3%}")
    h, t, r = hit(hw[hw["uncapped_says_over"]]); print(f"  uncapped recommends:  {h}/{t} = {r:.3%}")
    h, t, r = hit(hw[hw["only_uncapped_over"]]); print(f"  ONLY uncapped:        {h}/{t} = {r:.3%}")

    # Clamped-game summary: when the cap fired, did actual_total beat vegas?
    if clamped:
        cl = bt[bt["was_clamped"]]
        h, t, r = hit(cl)
        print(f"\nWhen cap fired (n={t}): actual > vegas {h}/{t} = {r:.3%}")
        print(f"  Mean uncapped offset: {cl['offset_uncapped'].mean():+.2f}r")
        print(f"  Mean capped offset:   {cl['offset_capped'].mean():+.2f}r")


if __name__ == "__main__":
    main()
