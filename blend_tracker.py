"""
blend_tracker.py
================
Rolling blend-weight optimizer. Joins daily card predictions (mc_rl, xgb_rl)
with actual home_covers_rl outcomes from actuals_2026.parquet, then grid-
searches the optimal MC/XGB blend weight for the current season.

Run after each day's games complete to get an up-to-date recommendation.

Output
------
  blend_tracker_results.csv   — one row per day's cumulative grid search
  Prints current recommendation to stdout.

Usage
-----
  python blend_tracker.py              # evaluate and print
  python blend_tracker.py --update     # save results to CSV and print
"""

import argparse
import glob
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import xlogy


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_true - y_prob) ** 2))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(xlogy(y_true, p) + xlogy(1 - y_true, 1 - p)))


def calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 8) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece / max(len(y_true), 1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predictions() -> pd.DataFrame:
    """Load all 2026 daily cards with mc_rl and xgb_rl."""
    card_files = sorted(glob.glob("daily_cards/daily_card_2026-*.csv"))
    if not card_files:
        raise FileNotFoundError("No 2026 daily cards found in daily_cards/")

    rows = []
    for f in card_files:
        d = pd.read_csv(f)
        d["date"] = Path(f).stem.replace("daily_card_", "")
        rows.append(d)

    df = pd.concat(rows, ignore_index=True)
    df = df[["date", "home_team", "away_team", "mc_rl", "xgb_rl",
             "blended_rl", "lock_p_model"]].copy()
    df = df.dropna(subset=["mc_rl", "xgb_rl"])
    return df


def load_actuals() -> pd.DataFrame:
    """Load 2026 actuals with home_covers_rl from actuals_2026.parquet."""
    path = Path("data/statcast/actuals_2026.parquet")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run statcast_pull_2026.py first")

    df = pd.read_parquet(path, engine="pyarrow",
                         columns=["game_date", "home_team", "away_team",
                                  "home_covers_rl", "home_score_final",
                                  "away_score_final"])
    df["date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["home_covers_rl"])
    return df


# ---------------------------------------------------------------------------
# Blend optimization
# ---------------------------------------------------------------------------

def grid_search(y: np.ndarray, mc: np.ndarray, xgb: np.ndarray,
                step: float = 0.05) -> pd.DataFrame:
    rows = []
    for w in np.arange(0.0, 1.0 + step / 2, step):
        blend = (1 - w) * mc + w * xgb
        rows.append({
            "xgb_weight": round(w, 2),
            "mc_weight": round(1 - w, 2),
            "brier": brier(y, blend),
            "log_loss": log_loss(y, blend),
            "ece": calibration_error(y, blend),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling MC/XGB blend optimizer")
    parser.add_argument("--update", action="store_true",
                        help="Append today's result to blend_tracker_results.csv")
    args = parser.parse_args()

    preds = load_predictions()
    actuals = load_actuals()

    merged = preds.merge(
        actuals[["home_team", "away_team", "date", "home_covers_rl",
                 "home_score_final", "away_score_final"]],
        on=["home_team", "away_team", "date"],
        how="inner",
    ).dropna(subset=["mc_rl", "xgb_rl", "home_covers_rl"])

    n = len(merged)
    today_str = date.today().isoformat()

    print("=" * 65)
    print(f"  Blend Tracker — {today_str}")
    print("=" * 65)
    print(f"\n  {n} games matched (prediction + actual home_covers_rl)")
    print(f"  Date range: {merged['date'].min()} → {merged['date'].max()}")
    print(f"  Actual covers rate: {merged['home_covers_rl'].mean():.4f}")

    if n < 20:
        print(f"\n  ⚠  Only {n} games — results unreliable until n≥100.")
        print("     Continue collecting data and re-run each week.")

    y   = merged["home_covers_rl"].values.astype(float)
    mc  = merged["mc_rl"].values.astype(float)
    xgb = merged["xgb_rl"].values.astype(float)

    results = grid_search(y, mc, xgb)
    best = results.loc[results["brier"].idxmin()]
    current = results[results["xgb_weight"] == 0.40].iloc[0]

    print("\n  Grid search  [xgb_weight% × xgb_rl  +  (1-w)% × mc_rl]:")
    print(f"  {'XGB%':>6}  {'MC%':>6}  {'Brier':>10}  {'LogLoss':>10}  {'ECE':>8}")
    for _, row in results.iterrows():
        marker = ""
        if abs(row["xgb_weight"] - best["xgb_weight"]) < 0.001:
            marker = " ← OPTIMAL"
        if abs(row["xgb_weight"] - 0.40) < 0.001:
            marker += " ← CURRENT"
        print(f"  {row['xgb_weight']:>5.0%}   {row['mc_weight']:>5.0%}"
              f"  {row['brier']:>10.5f}  {row['log_loss']:>10.5f}"
              f"  {row['ece']:>8.5f}{marker}")

    improvement = (current["brier"] - best["brier"]) / current["brier"] * 100
    print(f"\n  Current  (40% XGB):  Brier = {current['brier']:.5f}")
    print(f"  Optimal ({best['xgb_weight']:.0%} XGB):  Brier = {best['brier']:.5f}  "
          f"({improvement:+.2f}% vs current)")

    if n < 100:
        print(f"\n  Recommendation: HOLD — only {n} games. Re-evaluate at n=100.")
    elif abs(best["xgb_weight"] - 0.40) <= 0.05:
        print(f"\n  Recommendation: KEEP current 40% weight — optimal is within ±5%.")
    else:
        direction = "INCREASE" if best["xgb_weight"] > 0.40 else "DECREASE"
        print(f"\n  Recommendation: {direction} XGB weight from 40% → {best['xgb_weight']:.0%}")
        print(f"    Change XGB_BLEND_WEIGHT in monte_carlo_runline.py line ~1403.")

    # ── Per-signal diagnostics ─────────────────────────────────────────────
    print("\n  Individual signal Brier scores:")
    for label, prob in [("mc_rl only (0% XGB)", mc),
                         ("xgb_rl only (100% XGB)", xgb),
                         ("current blend (40% XGB)", 0.60 * mc + 0.40 * xgb),
                         ("base rate", np.full(n, y.mean()))]:
        print(f"    {label:<35}  {brier(y, prob):.5f}")

    if args.update:
        out_path = Path("blend_tracker_results.csv")
        row = {
            "run_date": today_str,
            "n_games": n,
            "date_min": merged["date"].min(),
            "date_max": merged["date"].max(),
            "covers_rate": round(float(y.mean()), 4),
            "optimal_xgb_weight": float(best["xgb_weight"]),
            "optimal_brier": round(float(best["brier"]), 5),
            "current_brier_40pct": round(float(current["brier"]), 5),
            "improvement_pct": round(improvement, 3),
        }
        if out_path.exists():
            prev = pd.read_csv(out_path)
            updated = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
        else:
            updated = pd.DataFrame([row])
        updated.to_csv(out_path, index=False)
        print(f"\n  Saved to {out_path} ({len(updated)} total entries)")


if __name__ == "__main__":
    main()
