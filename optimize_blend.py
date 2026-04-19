"""
optimize_blend.py
=================
Grid-search the optimal MC / XGBoost blend weight for the run-line model.

Phase 1 — XGBoost signal quality (2025, 4800 games)
  Evaluates rl_raw, rl_cal, rl_stacked against actual home_covers_rl outcomes.
  Blend grid: w% XGB stacked + (1-w)% base-rate (proxy for uninformative MC).
  This shows whether XGBoost adds value and at what weight.

Phase 2 — MC vs XGB blend (2026 daily cards + actuals)
  Joins daily_cards/daily_card_2026-*.csv with backtest_games_2026.csv.
  Tests blend weights on actual (mc_rl, xgb_rl) pairs where we have both.

Usage:
    python optimize_blend.py
    python optimize_blend.py --phase 1   # XGBoost-only analysis
    python optimize_blend.py --phase 2   # MC vs XGB blend (2026)
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import xlogy

# ---------------------------------------------------------------------------

def brier(y_true, y_prob):
    return float(np.mean((y_true - y_prob) ** 2))

def log_loss(y_true, y_prob, eps=1e-7):
    p = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(xlogy(y_true, p) + xlogy(1 - y_true, 1 - p)))

def calibration_error(y_true, y_prob, n_bins=10):
    """Mean absolute calibration error (ECE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        frac_pos = y_true[mask].mean()
        mean_pred = y_prob[mask].mean()
        ece += mask.sum() * abs(frac_pos - mean_pred)
    return ece / len(y_true)

def metrics(y_true, y_prob, label=""):
    bs = brier(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    ece = calibration_error(y_true, y_prob)
    print(f"  {label:<30}  Brier={bs:.5f}  LogLoss={ll:.5f}  ECE={ece:.5f}")
    return bs, ll, ece

# ---------------------------------------------------------------------------

def phase1():
    print("=" * 65)
    print("  PHASE 1 — XGBoost signal quality (2025 season, eval_predictions)")
    print("=" * 65)

    ep = pd.read_csv("eval_predictions.csv")
    df = ep.dropna(subset=["home_covers_rl"]).copy()
    print(f"\n  {len(df)} games with actual outcomes | "
          f"base rate (home covers -1.5) = {df['home_covers_rl'].mean():.4f}\n")

    y = df["home_covers_rl"].values.astype(float)
    base_rate = y.mean()

    # ── Raw signal comparison ───────────────────────────────────────────────
    print("  Signal comparison:")
    metrics(y, np.full(len(y), base_rate),       "Base rate (no model)")
    for col in ["rl_raw", "rl_cal", "rl_stacked"]:
        if col in df.columns and df[col].notna().all():
            metrics(y, df[col].values, col)

    # ── Eval mode comparison ────────────────────────────────────────────────
    print("\n  By eval_mode:")
    for mode in df["_eval_mode"].unique():
        sub = df[df["_eval_mode"] == mode]
        ys = sub["home_covers_rl"].values.astype(float)
        print(f"\n    [{mode}]  n={len(sub)}")
        metrics(ys, np.full(len(ys), ys.mean()),    "  base rate")
        for col in ["rl_raw", "rl_cal", "rl_stacked"]:
            if col in sub.columns:
                metrics(ys, sub[col].values, f"  {col}")

    # ── Blend grid: w% rl_stacked + (1-w)% base_rate ────────────────────
    print("\n  Blend grid  [w% rl_stacked + (1-w)% base_rate]:")
    print(f"  {'Weight':>8}  {'Brier':>10}  {'LogLoss':>10}  {'ECE':>10}")
    best_w, best_bs = 0.0, 999.0
    rows = []
    for w in np.arange(0.0, 1.05, 0.05):
        blend = w * df["rl_stacked"].values + (1 - w) * base_rate
        bs = brier(y, blend)
        ll = log_loss(y, blend)
        ece = calibration_error(y, blend)
        marker = " <-- current (40%)" if abs(w - 0.40) < 0.01 else ""
        print(f"  {w:>7.0%}    {bs:>10.5f}  {ll:>10.5f}  {ece:>10.5f}{marker}")
        rows.append({"xgb_weight": round(w, 2), "brier": bs, "logloss": ll, "ece": ece})
        if bs < best_bs:
            best_bs, best_w = bs, w

    print(f"\n  ✓ Optimal XGB weight (Brier): {best_w:.0%}  (Brier={best_bs:.5f})")
    print(f"  ✓ Current XGB weight:          40%")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

def phase2():
    print("\n" + "=" * 65)
    print("  PHASE 2 — MC vs XGB blend (2026 daily cards + actuals)")
    print("=" * 65)

    # Load and combine all 2026 daily cards
    card_files = sorted(glob.glob("daily_cards/daily_card_2026-*.csv"))
    if not card_files:
        print("  No 2026 daily cards found — skipping phase 2.")
        return

    cards = []
    for f in card_files:
        d = pd.read_csv(f)
        d["date"] = Path(f).stem.replace("daily_card_", "")
        cards.append(d)
    cards_df = pd.concat(cards, ignore_index=True)
    cards_df = cards_df.dropna(subset=["mc_rl", "xgb_rl"])
    print(f"\n  {len(cards_df)} games from {len(card_files)} cards with mc_rl + xgb_rl")

    # Load 2026 actuals
    bt_path = Path("data/raw/backtest_games_2026.csv")
    if not bt_path.exists():
        print("  backtest_games_2026.csv not found — run build_backtest.py --year 2026 first")
        return

    bt = pd.read_csv(bt_path)
    bt = bt.dropna(subset=["actual_home_win"])
    print(f"  {len(bt)} 2026 games with actual outcomes in backtest")

    # Join on home_team + away_team + date
    bt["date"] = pd.to_datetime(bt["game_date"]).dt.strftime("%Y-%m-%d")
    merged = cards_df.merge(
        bt[["home_team", "away_team", "date", "actual_home_win",
            "actual_game_total", "home_starter", "away_starter"]],
        on=["home_team", "away_team", "date"],
        how="inner",
    )
    merged = merged.dropna(subset=["mc_rl", "xgb_rl", "actual_home_win"])
    print(f"  {len(merged)} games matched (mc_rl + xgb_rl + actual)\n")

    if len(merged) < 10:
        print("  Too few matched games for reliable blend optimization — need more history.")
        print("  Showing available data:")
        print(merged[["date","home_team","away_team","mc_rl","xgb_rl","blended_rl","actual_home_win"]].to_string(index=False))
        return

    # We don't have home_covers_rl in the cards — approximate from mc_rl proxy
    # Use mc_rl as the home-covers-RL proxy (MC was calibrated against RL outcomes)
    # Real actual_home_covers_rl would need margin data; use actual_home_win as proxy
    y = merged["actual_home_win"].values.astype(float)
    mc = merged["mc_rl"].values
    xgb = merged["xgb_rl"].values

    print("  Blend grid  [w% xgb_rl + (1-w)% mc_rl]  vs actual_home_win:")
    print(f"  {'Weight':>8}  {'Brier':>10}  {'LogLoss':>10}")
    best_w, best_bs = 0.0, 999.0
    for w in np.arange(0.0, 1.05, 0.10):
        blend = w * xgb + (1 - w) * mc
        bs = brier(y, blend)
        ll = log_loss(y, blend)
        marker = " <-- current (40%)" if abs(w - 0.40) < 0.01 else ""
        print(f"  {w:>7.0%}    {bs:>10.5f}  {ll:>10.5f}{marker}")
        if bs < best_bs:
            best_bs, best_w = bs, w

    print(f"\n  ✓ Optimal XGB weight (2026, n={len(merged)}): {best_w:.0%}  (note: small sample)")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2], default=0)
    args = parser.parse_args()

    if args.phase in (0, 1):
        phase1()
    if args.phase in (0, 2):
        phase2()


if __name__ == "__main__":
    main()
