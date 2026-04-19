"""
optimize_locks.py
=================
Grid-search optimal lock gate constants against historical outcomes.

Constants optimized
-------------------
  SANITY_THRESHOLD    |p_model - p_true| <= X   (Pinnacle sanity gate)
  EDGE_TIER2          minimum edge to qualify for any bet
  EDGE_TIER1          minimum edge for Tier-1 (quarter-Kelly) bet
  ODDS_FLOOR          minimum acceptable American odds

Phase 1 — Model calibration (2025 eval_predictions, 4796 games)
  No odds data needed.  Buckets model probability against actual cover rate.
  Tells us how well-calibrated the signal is and where it breaks down.

Phase 2 — Lock simulation (2026 daily cards + actuals, ~77 games)
  Full betting simulation across all four gate combinations.
  Reports ROI, win rate, bet count, and P&L per grid cell.
  Flags the current constants for comparison.

Usage
-----
  python optimize_locks.py              # both phases
  python optimize_locks.py --phase 1   # calibration only
  python optimize_locks.py --phase 2   # lock simulation only
  python optimize_locks.py --save      # also write results to optimize_locks_results.csv
"""

import argparse
import glob
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Current constants (for comparison)
# ---------------------------------------------------------------------------
CURRENT = {
    "sanity":     0.04,
    "edge_tier2": 0.010,
    "edge_tier1": 0.030,
    "odds_floor": -225,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal multiplier (profit per $1 staked)."""
    if odds >= 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def pnl(stake: float, odds: float, won: bool) -> float:
    return stake * american_to_decimal(odds) if won else -stake


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_prob)) ** 2))


# ---------------------------------------------------------------------------
# Phase 1 — model calibration
# ---------------------------------------------------------------------------

def phase1() -> None:
    print("=" * 65)
    print("  PHASE 1 — Model calibration (2025, eval_predictions)")
    print("=" * 65)

    ep = pd.read_csv("eval_predictions.csv")
    df = ep.dropna(subset=["home_covers_rl", "rl_stacked"]).copy()
    df["y"]   = df["home_covers_rl"].astype(float)
    df["p"]   = df["rl_stacked"].astype(float)
    df["mode"] = df["_eval_mode"]

    print(f"\n  {len(df)} games | base rate = {df['y'].mean():.4f}")
    print(f"  Overall Brier = {brier(df['y'], df['p']):.5f}")

    # Calibration by probability bucket
    print("\n  Calibration by model probability bucket:")
    print(f"  {'Bucket':>12}  {'N':>6}  {'Pred%':>7}  {'Actual%':>8}  {'Brier':>8}  {'Edge':>8}")
    bins = np.arange(0.20, 0.65, 0.05)
    for lo in bins:
        hi = lo + 0.05
        mask = (df["p"] >= lo) & (df["p"] < hi)
        sub = df[mask]
        if len(sub) < 10:
            continue
        pred_rate   = sub["p"].mean()
        actual_rate = sub["y"].mean()
        b = brier(sub["y"], sub["p"])
        edge = pred_rate - actual_rate
        bar = "▲" if edge > 0.015 else ("▼" if edge < -0.015 else " ")
        print(f"  {lo:.2f}–{hi:.2f}      {len(sub):>6}  {pred_rate:>6.1%}  "
              f"{actual_rate:>8.1%}  {b:>8.5f}  {edge:>+7.1%} {bar}")

    # Confidence filter: does restricting to high-confidence predictions help?
    print("\n  Effect of minimum model confidence threshold:")
    print(f"  {'Min prob':>9}  {'N':>6}  {'Brier':>8}  {'Win rate':>9}")
    for min_p in [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.55]:
        mask = (df["p"] >= min_p) | (df["p"] <= (1 - min_p))
        # Flip prob so we always bet the stronger side
        sub = df[mask].copy()
        sub["p_adj"] = np.where(sub["p"] >= 0.5, sub["p"], 1 - sub["p"])
        sub["y_adj"] = np.where(sub["p"] >= 0.5, sub["y"], 1 - sub["y"])
        if len(sub) == 0:
            continue
        b = brier(sub["y_adj"], sub["p_adj"])
        wr = sub["y_adj"].mean()
        marker = " ← current (no floor)" if min_p == 0.40 else ""
        print(f"  {min_p:>8.0%}   {len(sub):>6}  {b:>8.5f}  {wr:>9.1%}{marker}")

    # By eval mode
    print("\n  By eval_mode:")
    for mode in df["mode"].unique():
        sub = df[df["mode"] == mode]
        print(f"  [{mode}]  n={len(sub)}  Brier={brier(sub['y'], sub['p']):.5f}  "
              f"base_rate={sub['y'].mean():.4f}")


# ---------------------------------------------------------------------------
# Phase 2 — lock simulation
# ---------------------------------------------------------------------------

def load_cards_with_actuals() -> pd.DataFrame:
    """Load 2026 daily cards joined with actual outcomes."""
    card_files = sorted(glob.glob("daily_cards/daily_card_2026-*.csv"))
    if not card_files:
        return pd.DataFrame()

    cards = []
    for f in card_files:
        d = pd.read_csv(f)
        d["date"] = Path(f).stem.replace("daily_card_", "")
        cards.append(d)
    df = pd.concat(cards, ignore_index=True)

    # Load actuals
    act_path = Path("data/statcast/actuals_2026.parquet")
    if not act_path.exists():
        return df

    act = pd.read_parquet(act_path, engine="pyarrow",
                          columns=["game_date", "home_team", "away_team",
                                   "home_covers_rl", "home_score_final",
                                   "away_score_final"])
    act["date"] = pd.to_datetime(act["game_date"]).dt.strftime("%Y-%m-%d")
    act["actual_total"] = act["home_score_final"] + act["away_score_final"]
    act = act.drop(columns=["game_date"])

    merged = df.merge(act, on=["home_team", "away_team", "date"], how="left")
    return merged


def simulate_bets(df: pd.DataFrame, sanity: float, edge_tier2: float,
                  edge_tier1: float, odds_floor: float) -> dict:
    """
    Simulate betting outcomes for one parameter combination.
    Returns dict with bet_count, win_rate, roi, total_pnl, brier_filtered.
    """
    bet_rows = []

    for _, row in df.iterrows():
        # Evaluate all bet directions for this game
        candidates = []

        # RL home -1.5
        if pd.notna(row.get("lock_p_model")) and pd.notna(row.get("lock_p_true")) \
                and pd.notna(row.get("lock_retail_implied")) and pd.notna(row.get("rl_odds")):
            p_model  = float(row["lock_p_model"])
            p_true   = float(row["lock_p_true"])
            p_retail = float(row["lock_retail_implied"])
            odds     = float(row["rl_odds"])
            edge     = p_model - p_retail
            actual   = row.get("home_covers_rl")
            candidates.append(("RL_HOME", p_model, p_true, p_retail, odds, edge, actual))

        # RL away +1.5
        if pd.notna(row.get("away_lock_p_model")) and pd.notna(row.get("away_lock_p_true")) \
                and pd.notna(row.get("away_lock_retail_implied")) and pd.notna(row.get("rl_odds")):
            p_model  = float(row["away_lock_p_model"])
            p_true   = float(row["away_lock_p_true"])
            p_retail = float(row["away_lock_retail_implied"])
            odds     = -float(row["rl_odds"]) if float(row["rl_odds"]) < 0 else float(row["rl_odds"])
            edge     = p_model - p_retail
            actual_rl = row.get("home_covers_rl")
            actual   = (1 - float(actual_rl)) if pd.notna(actual_rl) else np.nan
            candidates.append(("RL_AWAY", p_model, p_true, p_retail, odds, edge, actual))

        # ML home
        if pd.notna(row.get("ml_lock_p_model")) and pd.notna(row.get("ml_lock_p_true")) \
                and pd.notna(row.get("ml_lock_retail_implied")) and pd.notna(row.get("vegas_ml_home")):
            p_model  = float(row["ml_lock_p_model"])
            p_true   = float(row["ml_lock_p_true"])
            p_retail = float(row["ml_lock_retail_implied"])
            odds     = float(row["vegas_ml_home"])
            edge     = p_model - p_retail
            actual_win = row.get("actual_home_win") if "actual_home_win" in row else np.nan
            candidates.append(("ML_HOME", p_model, p_true, p_retail, odds, edge, actual_win))

        # O/U
        if pd.notna(row.get("ou_p_model")) and pd.notna(row.get("ou_p_true")) \
                and pd.notna(row.get("ou_p_retail")) and pd.notna(row.get("ou_posted_line")):
            direction = row.get("ou_direction", "OVER")
            p_model  = float(row["ou_p_model"])
            p_true   = float(row["ou_p_true"])
            p_retail = float(row["ou_p_retail"])
            edge     = p_model - p_retail
            odds     = -110.0  # standard juice
            actual_total = row.get("actual_total")
            posted   = float(row["ou_posted_line"]) if pd.notna(row.get("ou_posted_line")) else np.nan
            if pd.notna(actual_total) and pd.notna(posted):
                if direction == "OVER":
                    actual = 1.0 if float(actual_total) > posted else 0.0
                else:
                    actual = 1.0 if float(actual_total) < posted else 0.0
            else:
                actual = np.nan
            candidates.append(("OU", p_model, p_true, p_retail, odds, edge, actual))

        for (bet_type, p_model, p_true, p_retail, odds, edge, actual) in candidates:
            # Gate 1: sanity
            if abs(p_model - p_true) > sanity:
                continue
            # Gate 2: odds floor
            if odds < odds_floor:
                continue
            # Gate 3: minimum edge
            if edge < edge_tier2:
                continue
            # Tier
            tier = 1 if edge >= edge_tier1 else 2
            # Kelly fraction
            frac = 0.25 if tier == 1 else 0.125
            # Stake (fixed $50 max)
            q = 1 - p_model
            kelly_full = (p_model * american_to_decimal(odds) - q) / american_to_decimal(odds)
            stake = min(round(max(kelly_full, 0) * frac * 2000, 2), 50.0)
            if stake <= 0:
                continue

            bet_rows.append({
                "bet_type": bet_type,
                "p_model":  p_model,
                "p_true":   p_true,
                "edge":     edge,
                "odds":     odds,
                "tier":     tier,
                "stake":    stake,
                "actual":   actual,
            })

    if not bet_rows:
        return {"bet_count": 0, "win_rate": np.nan, "roi": np.nan,
                "total_pnl": 0.0, "brier": np.nan, "staked": 0.0}

    bets = pd.DataFrame(bet_rows)
    resolved = bets.dropna(subset=["actual"])

    total_staked  = resolved["stake"].sum()
    total_pl = sum(
        pnl(r["stake"], r["odds"], bool(r["actual"]))
        for _, r in resolved.iterrows()
    )
    win_rate = resolved["actual"].mean() if len(resolved) > 0 else np.nan
    roi      = total_pl / total_staked if total_staked > 0 else np.nan
    br       = brier(resolved["actual"], resolved["p_model"]) if len(resolved) > 0 else np.nan

    return {
        "bet_count": len(resolved),
        "win_rate":  round(float(win_rate), 4) if not np.isnan(win_rate) else np.nan,
        "roi":       round(float(roi), 4)       if not np.isnan(roi) else np.nan,
        "total_pnl": round(total_pl, 2),
        "staked":    round(total_staked, 2),
        "brier":     round(float(br), 5)        if not np.isnan(br) else np.nan,
    }


def phase2(save: bool = False) -> None:
    print("\n" + "=" * 65)
    print("  PHASE 2 — Lock simulation (2026 daily cards + actuals)")
    print("=" * 65)

    df = load_cards_with_actuals()
    resolved = df.dropna(subset=["home_covers_rl"])
    print(f"\n  {len(df)} total games | {len(resolved)} with actual outcomes")

    if len(resolved) < 10:
        print("  Too few resolved games — need more history.")
        return

    # Current baseline
    base = simulate_bets(resolved, **CURRENT)
    print(f"\n  Current constants: sanity={CURRENT['sanity']:.0%}  "
          f"edge_t2={CURRENT['edge_tier2']:.1%}  "
          f"edge_t1={CURRENT['edge_tier1']:.1%}  "
          f"odds_floor={CURRENT['odds_floor']}")
    print(f"  Baseline → bets={base['bet_count']}  "
          f"win={base['win_rate']:.1%}  "
          f"ROI={base['roi']:+.1%}  "
          f"P&L=${base['total_pnl']:+.2f}  "
          f"staked=${base['staked']:.2f}")

    # Grid search
    grid = {
        "sanity":     [0.03, 0.04, 0.05, 0.06, 0.08],
        "edge_tier2": [0.005, 0.010, 0.015, 0.020],
        "edge_tier1": [0.020, 0.025, 0.030, 0.035, 0.040],
        "odds_floor": [-300, -250, -225, -200, -175, -150],
    }

    print(f"\n  Grid search ({np.prod([len(v) for v in grid.values()])} combinations)...")

    results = []
    for sanity, et2, et1, of in itertools.product(
            grid["sanity"], grid["edge_tier2"], grid["edge_tier1"], grid["odds_floor"]):
        if et2 >= et1:
            continue
        r = simulate_bets(resolved, sanity, et2, et1, of)
        r.update({"sanity": sanity, "edge_tier2": et2, "edge_tier1": et1, "odds_floor": of})
        results.append(r)

    res_df = pd.DataFrame(results).dropna(subset=["roi"])

    if res_df.empty:
        print("  No results with resolvable bets — sample too small.")
        return

    # Sort by ROI
    res_df = res_df.sort_values("roi", ascending=False)

    print(f"\n  Top 10 by ROI (min 3 bets):")
    print(f"  {'Sanity':>8}  {'ET2':>6}  {'ET1':>6}  {'Floor':>7}  "
          f"{'Bets':>5}  {'Win%':>6}  {'ROI':>8}  {'P&L':>8}")
    top = res_df[res_df["bet_count"] >= 3].head(10)
    for _, row in top.iterrows():
        curr = (abs(row["sanity"] - CURRENT["sanity"]) < 0.001 and
                abs(row["edge_tier2"] - CURRENT["edge_tier2"]) < 0.001 and
                abs(row["edge_tier1"] - CURRENT["edge_tier1"]) < 0.001 and
                abs(row["odds_floor"] - CURRENT["odds_floor"]) < 1)
        marker = " ← CURRENT" if curr else ""
        print(f"  {row['sanity']:>7.0%}   {row['edge_tier2']:>5.1%}   {row['edge_tier1']:>5.1%}"
              f"  {row['odds_floor']:>6.0f}   {row['bet_count']:>5}  "
              f"{row['win_rate']:>5.1%}  {row['roi']:>+7.1%}  ${row['total_pnl']:>+7.2f}{marker}")

    # Current rank
    curr_row = res_df[
        (res_df["sanity"].round(3) == CURRENT["sanity"]) &
        (res_df["edge_tier2"].round(4) == CURRENT["edge_tier2"]) &
        (res_df["edge_tier1"].round(3) == CURRENT["edge_tier1"]) &
        (res_df["odds_floor"] == CURRENT["odds_floor"])
    ]
    if not curr_row.empty:
        rank = res_df.reset_index(drop=True).index[
            (res_df["sanity"].round(3) == CURRENT["sanity"]).values &
            (res_df["edge_tier2"].round(4) == CURRENT["edge_tier2"]).values &
            (res_df["edge_tier1"].round(3) == CURRENT["edge_tier1"]).values &
            (res_df["odds_floor"] == CURRENT["odds_floor"]).values
        ]
        if len(rank):
            print(f"\n  Current constants rank: #{rank[0]+1} of {len(res_df)} combinations")

    best = res_df[res_df["bet_count"] >= 3].iloc[0] if len(res_df[res_df["bet_count"] >= 3]) else None
    if best is not None:
        print(f"\n  Best combination (ROI={best['roi']:+.1%}, n={best['bet_count']}):")
        print(f"    SANITY_THRESHOLD = {best['sanity']:.0%}")
        print(f"    EDGE_TIER2       = {best['edge_tier2']:.1%}")
        print(f"    EDGE_TIER1       = {best['edge_tier1']:.1%}")
        print(f"    ODDS_FLOOR       = {best['odds_floor']:.0f}")

    if len(resolved) < 150:
        print(f"\n  ⚠  Only {len(resolved)} resolved games — results directional only.")
        print(f"     Re-run at n=150+ for reliable optimization (target: mid-May).")

    if save:
        out = Path("optimize_locks_results.csv")
        res_df.to_csv(out, index=False)
        print(f"\n  Saved {len(res_df)} grid results to {out}")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2], default=0)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if args.phase in (0, 1):
        phase1()
    if args.phase in (0, 2):
        phase2(save=args.save)


if __name__ == "__main__":
    main()
