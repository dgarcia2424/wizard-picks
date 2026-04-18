"""
kprop_tracker.py
================
Append yesterday's K prop predictions (from daily card) against actuals
(from actuals_2026.parquet) into kprop_tracker_2026.csv.

Run daily after actuals are available (typically 8–10 AM next day).

Usage:
  python kprop_tracker.py                  # appends yesterday
  python kprop_tracker.py --date 2026-04-15  # specific date
  python kprop_tracker.py --rebuild        # rebuild from all daily cards
"""

import argparse
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR    = Path("./data/statcast")
CARDS_DIR   = Path("./daily_cards")
OUTPUT_CSV  = Path("kprop_tracker_2026.csv")


def _build_rows(card_date: str) -> pd.DataFrame:
    card_path = CARDS_DIR / f"daily_card_{card_date}.csv"
    if not card_path.exists():
        return pd.DataFrame()

    act_path = DATA_DIR / "actuals_2026.parquet"
    if not act_path.exists():
        return pd.DataFrame()

    try:
        card = pd.read_csv(card_path, encoding="utf-8")
    except Exception:
        return pd.DataFrame()

    act = pd.read_parquet(act_path)
    act["game_date"] = act["game_date"].astype(str)
    act_day = act[act["game_date"] == card_date]
    if act_day.empty:
        return pd.DataFrame()

    rows = []
    for side, sp_col, line_col, mean_col, over_col, impl_col, actual_col in [
        ("home", "home_sp", "home_k_line", "mc_home_sp_k_mean",
         "home_k_model_over", "home_k_implied_over", "home_sp_k"),
        ("away", "away_sp", "away_k_line", "mc_away_sp_k_mean",
         "away_k_model_over", "away_k_implied_over", "away_sp_k"),
    ]:
        required = [line_col, mean_col]
        if not all(c in card.columns for c in required):
            continue

        opt_cols = [over_col, impl_col, "ump_k_above_avg"]
        for c in opt_cols:
            if c not in card.columns:
                card[c] = float("nan")

        sub = card[[
            "home_team", "away_team", sp_col, line_col,
            mean_col, over_col, impl_col, "ump_k_above_avg",
        ]].dropna(subset=[line_col, mean_col]).copy()
        sub = sub.rename(columns={
            sp_col: "sp", line_col: "line",
            mean_col: "pred_mean", over_col: "model_over",
            impl_col: "implied_over",
        })
        sub["side"] = side
        sub["game_date"] = card_date

        merged = sub.merge(
            act_day[["home_team", "away_team", actual_col]],
            on=["home_team", "away_team"], how="inner",
        ).rename(columns={actual_col: "actual_k"})
        merged = merged.dropna(subset=["actual_k"])
        merged["actual_k"] = merged["actual_k"].astype(int)
        rows.append(merged)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    df["hit_over"]      = (df["actual_k"] > df["line"]).astype(int)
    df["model_pick_over"] = (df["model_over"] > 0.5).astype(int)
    df["model_correct"] = (df["model_pick_over"] == df["hit_over"]).astype(int)
    df["edge"]          = df["model_over"] - df["implied_over"]
    return df


def append_date(target_date: str) -> None:
    new_rows = _build_rows(target_date)
    if new_rows.empty:
        print(f"  [kprop_tracker] No data for {target_date} — skipping.")
        return

    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV)
        combined = (pd.concat([existing, new_rows], ignore_index=True)
                      .drop_duplicates(subset=["game_date", "home_team", "away_team", "side"])
                      .sort_values(["game_date", "home_team", "side"])
                      .reset_index(drop=True))
    else:
        combined = new_rows

    combined.to_csv(OUTPUT_CSV, index=False)
    n = len(new_rows)
    acc = new_rows["model_correct"].mean()
    bias = (new_rows["pred_mean"] - new_rows["actual_k"]).mean()
    print(f"  [kprop_tracker] {target_date}: +{n} rows | accuracy={acc:.1%} | bias={bias:+.2f}K | total={len(combined)}")


def rebuild() -> None:
    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()
    for card_path in sorted(CARDS_DIR.glob("daily_card_2026-*.csv")):
        d = card_path.stem.replace("daily_card_", "")
        append_date(d)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",    type=str, help="YYYY-MM-DD to process")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild from all cards")
    args = parser.parse_args()

    if args.rebuild:
        rebuild()
    else:
        target = args.date or (date.today() - timedelta(days=1)).isoformat()
        append_date(target)

    if OUTPUT_CSV.exists():
        df = pd.read_csv(OUTPUT_CSV)
        print(f"\n  === K Prop Tracker ({len(df)} starts) ===")
        print(f"  Overall accuracy : {df['model_correct'].mean():.1%}")
        edge = df[df['edge'] > 0.05]
        if len(edge):
            print(f"  Edge>5% accuracy : {edge['model_correct'].mean():.1%}  (n={len(edge)})")
        print(f"  Pred mean MAE    : {(df['pred_mean'] - df['actual_k']).abs().mean():.2f} Ks")
        print(f"  Bias (pred-act)  : {(df['pred_mean'] - df['actual_k']).mean():+.2f} Ks")


if __name__ == "__main__":
    main()
