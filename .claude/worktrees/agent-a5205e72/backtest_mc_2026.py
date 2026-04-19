"""
backtest_mc_2026.py
===================
Retroactively evaluate Monte Carlo model predictions against ground-truth
actuals for all completed 2026 games.

Workflow:
  1. Load actuals_2026.parquet  (F1 / F5 / K actuals, created by extract_actuals_2026.py)
  2. Load backtest_2026_results.csv  (MC & XGB predictions, run-line outcomes)
  3. Merge on game_date + home_team + away_team
  4. Compute accuracy metrics for every MC field that is present
  5. Save enriched results to backtest_mc_2026_results.csv
  6. Print a clean summary report

Graceful degradation:
  - If mc_f5_*, mc_nrfi_prob, mc_home_sp_k_mean, mc_away_sp_k_mean are absent
    from backtest_2026_results.csv, those sections are skipped cleanly.
  - Works with any subset of MC columns available.

Usage:
  python backtest_mc_2026.py           # merge & report
  python backtest_mc_2026.py --rebuild # re-extract actuals first, then merge
"""

import argparse
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
DATA_DIR       = Path("./data/statcast")
ACTUALS_PATH   = DATA_DIR / "actuals_2026.parquet"
BACKTEST_CSV   = Path("backtest_2026_results.csv")
OUTPUT_CSV     = Path("backtest_mc_2026_results.csv")

# MC columns that *may* be present in backtest_2026_results.csv.
# We never error if they are absent — we just skip those sections.
MC_F5_HOME_COL       = "mc_f5_home_runs"
MC_F5_AWAY_COL       = "mc_f5_away_runs"
MC_F5_TOTAL_COL      = "mc_f5_total"
MC_NRFI_PROB_COL     = "mc_nrfi_prob"
MC_HOME_K_MEAN_COL   = "mc_home_sp_k_mean"
MC_AWAY_K_MEAN_COL   = "mc_away_sp_k_mean"


# ---------------------------------------------------------------------------
# LOAD HELPERS
# ---------------------------------------------------------------------------

def load_actuals() -> pd.DataFrame:
    """Load actuals_2026.parquet, exit with a helpful message if absent."""
    if not ACTUALS_PATH.exists():
        print(f"\n  ERROR: {ACTUALS_PATH} not found.")
        print("  Run  python extract_actuals_2026.py  first, or use --rebuild.")
        sys.exit(1)

    df = pd.read_parquet(ACTUALS_PATH, engine="pyarrow")
    df["game_date"] = df["game_date"].astype(str)
    print(f"  Loaded actuals   : {len(df)} games  ({ACTUALS_PATH})")
    return df


def load_backtest() -> pd.DataFrame:
    """Load the main backtest tracker CSV."""
    if not BACKTEST_CSV.exists():
        print(f"\n  ERROR: {BACKTEST_CSV} not found. Run backtest_2026.py first.")
        sys.exit(1)

    df = pd.read_csv(BACKTEST_CSV)
    # Normalise the date column name — may be 'date' or 'game_date'
    if "date" in df.columns and "game_date" not in df.columns:
        df = df.rename(columns={"date": "game_date"})
    df["game_date"] = df["game_date"].astype(str)
    print(f"  Loaded backtest  : {len(df)} rows  ({BACKTEST_CSV})")
    return df


# ---------------------------------------------------------------------------
# MERGE
# ---------------------------------------------------------------------------

def merge_datasets(actuals: pd.DataFrame, backtest: pd.DataFrame) -> pd.DataFrame:
    """Inner-merge on game_date + home_team + away_team.

    The backtest CSV uses 'game' = 'AWAY @ HOME', so we parse that into
    home_team / away_team when those columns are absent.
    """
    # Ensure home_team / away_team exist in backtest
    if "home_team" not in backtest.columns or "away_team" not in backtest.columns:
        if "game" in backtest.columns:
            # Format: "AWAY @ HOME"
            split = backtest["game"].str.split(r"\s*@\s*", expand=True)
            backtest["away_team"] = split[0].str.strip()
            backtest["home_team"] = split[1].str.strip()
        else:
            print("  ERROR: Cannot determine home_team/away_team from backtest CSV.")
            sys.exit(1)

    merge_keys = ["game_date", "home_team", "away_team"]
    merged = actuals.merge(backtest, on=merge_keys, how="inner", suffixes=("", "_bt"))

    # Resolve duplicated home_covers_rl: prefer the actuals version
    if "home_covers_rl_bt" in merged.columns:
        merged = merged.drop(columns=["home_covers_rl_bt"])

    print(f"  Merged rows      : {len(merged)}  (actuals={len(actuals)}, backtest={len(backtest)})")
    if len(merged) < len(actuals):
        unmatched = len(actuals) - len(merged)
        print(f"  Note: {unmatched} actuals games had no matching backtest entry (predictions not yet run?)")

    return merged


# ---------------------------------------------------------------------------
# ACCURACY METRICS
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute all available accuracy metrics. Returns a results dict."""
    results = {}

    # ----------------------------------------------------------------
    # Run-line accuracy (always available since home_covers_rl + bet_win exist)
    # ----------------------------------------------------------------
    if "bet_win" in df.columns and "signal" in df.columns:
        signal_df = df[df["signal"].notna() & (df["signal"] != "")]
        if not signal_df.empty:
            wins  = signal_df["bet_win"].astype(float).sum()
            total = len(signal_df)
            results["rl_bets"]      = int(total)
            results["rl_wins"]      = int(wins)
            results["rl_win_pct"]   = round(wins / total, 4) if total else np.nan
            results["rl_roi"]       = round((wins * (100 / 110) - (total - wins)) / total, 4) if total else np.nan
        else:
            results["rl_bets"] = 0

    # ----------------------------------------------------------------
    # NRFI prediction accuracy
    # ----------------------------------------------------------------
    if MC_NRFI_PROB_COL in df.columns:
        nrfi_df = df[[MC_NRFI_PROB_COL, "f1_nrfi"]].dropna()
        if not nrfi_df.empty:
            predicted_nrfi = (nrfi_df[MC_NRFI_PROB_COL] > 0.5).astype(int)
            actual_nrfi    = nrfi_df["f1_nrfi"].astype(int)
            correct = (predicted_nrfi == actual_nrfi).sum()
            results["nrfi_n"]        = len(nrfi_df)
            results["nrfi_accuracy"] = round(correct / len(nrfi_df), 4)
            results["nrfi_prob_mean"]= round(nrfi_df[MC_NRFI_PROB_COL].mean(), 4)
            results["nrfi_actual_rate"] = round(actual_nrfi.mean(), 4)
        else:
            results["nrfi_n"] = 0
    else:
        results["nrfi_n"] = None  # column absent

    # ----------------------------------------------------------------
    # F5 total MAE
    # ----------------------------------------------------------------
    if MC_F5_TOTAL_COL in df.columns:
        f5_df = df[[MC_F5_TOTAL_COL, "f5_total"]].dropna()
        if not f5_df.empty:
            errors = (f5_df[MC_F5_TOTAL_COL] - f5_df["f5_total"]).abs()
            results["f5_total_n"]   = len(f5_df)
            results["f5_total_mae"] = round(errors.mean(), 3)
            results["f5_within_1"]  = round((errors <= 1.0).mean(), 4)
            results["f5_within_2"]  = round((errors <= 2.0).mean(), 4)
        else:
            results["f5_total_n"] = 0
    else:
        results["f5_total_n"] = None

    # ----------------------------------------------------------------
    # SP strikeout MAE
    # ----------------------------------------------------------------
    home_k_available = MC_HOME_K_MEAN_COL in df.columns
    away_k_available = MC_AWAY_K_MEAN_COL in df.columns

    if home_k_available or away_k_available:
        k_errors = []

        if home_k_available:
            hk = df[[MC_HOME_K_MEAN_COL, "home_sp_k"]].dropna()
            if not hk.empty:
                k_errors.append((hk[MC_HOME_K_MEAN_COL] - hk["home_sp_k"]).abs())
                results["home_sp_k_n"]   = len(hk)
                results["home_sp_k_mae"] = round((hk[MC_HOME_K_MEAN_COL] - hk["home_sp_k"]).abs().mean(), 3)
            else:
                results["home_sp_k_n"] = 0
        else:
            results["home_sp_k_n"] = None

        if away_k_available:
            ak = df[[MC_AWAY_K_MEAN_COL, "away_sp_k"]].dropna()
            if not ak.empty:
                k_errors.append((ak[MC_AWAY_K_MEAN_COL] - ak["away_sp_k"]).abs())
                results["away_sp_k_n"]   = len(ak)
                results["away_sp_k_mae"] = round((ak[MC_AWAY_K_MEAN_COL] - ak["away_sp_k"]).abs().mean(), 3)
            else:
                results["away_sp_k_n"] = 0
        else:
            results["away_sp_k_n"] = None

        if k_errors:
            all_k = pd.concat(k_errors)
            results["combined_sp_k_mae"] = round(all_k.mean(), 3)
    else:
        results["home_sp_k_n"] = None
        results["away_sp_k_n"] = None

    return results


# ---------------------------------------------------------------------------
# COLUMN-LEVEL ERROR COLUMNS
# ----------------------------------------------------------------

def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Append per-row error columns to the merged dataframe where possible."""

    # NRFI: predicted label vs actual
    if MC_NRFI_PROB_COL in df.columns:
        df["mc_nrfi_pred"]    = (df[MC_NRFI_PROB_COL] > 0.5).astype("Int8")
        df["nrfi_correct"]    = (df["mc_nrfi_pred"] == df["f1_nrfi"].astype("Int8")).astype("Int8")

    # F5 total error
    if MC_F5_TOTAL_COL in df.columns:
        df["f5_total_err"]    = (df[MC_F5_TOTAL_COL] - df["f5_total"]).round(2)
        df["f5_total_abs_err"]= df["f5_total_err"].abs().round(2)

    # Home SP K error
    if MC_HOME_K_MEAN_COL in df.columns:
        df["home_sp_k_err"]   = (df[MC_HOME_K_MEAN_COL] - df["home_sp_k"]).round(2)

    # Away SP K error
    if MC_AWAY_K_MEAN_COL in df.columns:
        df["away_sp_k_err"]   = (df[MC_AWAY_K_MEAN_COL] - df["away_sp_k"]).round(2)

    return df


# ---------------------------------------------------------------------------
# SUMMARY REPORT
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n  {'— ' + title + ' —':^56}")


def print_report(df: pd.DataFrame, metrics: dict) -> None:
    print()
    print("=" * 60)
    print("  backtest_mc_2026  — Monte Carlo accuracy report")
    print("=" * 60)

    g_min = df["game_date"].min()
    g_max = df["game_date"].max()
    print(f"  Games analysed: {len(df)}   ({g_min}  to  {g_max})")

    # ----------------------------------------------------------------
    # Run-line
    # ----------------------------------------------------------------
    _section("Run-Line Signals")
    rl_bets = metrics.get("rl_bets", 0)
    if rl_bets and rl_bets > 0:
        print(f"  {'Bets':<22}: {rl_bets}")
        print(f"  {'Wins':<22}: {metrics['rl_wins']}")
        print(f"  {'Win %':<22}: {metrics['rl_win_pct']:.1%}")
        print(f"  {'ROI':<22}: {metrics['rl_roi']:+.1%}")
    else:
        print("  No run-line signal bets found in dataset.")

    # ----------------------------------------------------------------
    # NRFI
    # ----------------------------------------------------------------
    _section("NRFI (First-Inning No-Run)")
    n = metrics.get("nrfi_n")
    if n is None:
        print(f"  mc_nrfi_prob column not present — skipping.")
    elif n == 0:
        print("  No matched NRFI predictions found.")
    else:
        actual_rate = metrics.get("nrfi_actual_rate", float("nan"))
        prob_mean   = metrics.get("nrfi_prob_mean", float("nan"))
        acc         = metrics.get("nrfi_accuracy", float("nan"))
        print(f"  {'Games with prediction':<28}: {n}")
        print(f"  {'Actual NRFI rate':<28}: {actual_rate:.1%}")
        print(f"  {'Model avg NRFI prob':<28}: {prob_mean:.1%}")
        print(f"  {'Prediction accuracy':<28}: {acc:.1%}")
        # Breakdown table
        if "nrfi_correct" in df.columns:
            sub = df.dropna(subset=["nrfi_correct"])
            by_actual = sub.groupby("f1_nrfi")["nrfi_correct"].agg(["mean", "count"])
            print()
            print(f"  {'Actual':<10} {'n':>5} {'Acc':>8}")
            print(f"  {'-'*25}")
            labels = {0: "YRFI (0)", 1: "NRFI (1)"}
            for val, row in by_actual.iterrows():
                lab = labels.get(int(val), str(val))
                print(f"  {lab:<10} {int(row['count']):>5} {row['mean']:>8.1%}")

    # ----------------------------------------------------------------
    # F5 total
    # ----------------------------------------------------------------
    _section("F5 Total Runs")
    n = metrics.get("f5_total_n")
    if n is None:
        print(f"  {MC_F5_TOTAL_COL} column not present — skipping.")
    elif n == 0:
        print("  No matched F5 predictions found.")
    else:
        mae = metrics.get("f5_total_mae", float("nan"))
        w1  = metrics.get("f5_within_1", float("nan"))
        w2  = metrics.get("f5_within_2", float("nan"))
        print(f"  {'Games with prediction':<28}: {n}")
        print(f"  {'MAE (runs)':<28}: {mae:.3f}")
        print(f"  {'Within 1 run':<28}: {w1:.1%}")
        print(f"  {'Within 2 runs':<28}: {w2:.1%}")
        # Actual vs predicted avg
        if MC_F5_TOTAL_COL in df.columns:
            sub = df[[MC_F5_TOTAL_COL, "f5_total"]].dropna()
            if not sub.empty:
                print(f"  {'Actual avg F5 total':<28}: {sub['f5_total'].mean():.2f}")
                print(f"  {'Model avg F5 total':<28}: {sub[MC_F5_TOTAL_COL].mean():.2f}")

    # ----------------------------------------------------------------
    # SP strikeouts
    # ----------------------------------------------------------------
    _section("SP Strikeouts")
    home_n = metrics.get("home_sp_k_n")
    away_n = metrics.get("away_sp_k_n")
    combined_mae = metrics.get("combined_sp_k_mae")

    if home_n is None and away_n is None:
        print(f"  mc_home_sp_k_mean / mc_away_sp_k_mean columns not present — skipping.")
    else:
        if home_n is not None and home_n > 0:
            print(f"  {'Home SP — n':<28}: {home_n}")
            print(f"  {'Home SP — MAE (Ks)':<28}: {metrics['home_sp_k_mae']:.3f}")
            if MC_HOME_K_MEAN_COL in df.columns:
                sub = df[[MC_HOME_K_MEAN_COL, "home_sp_k"]].dropna()
                print(f"  {'Home SP — actual avg K':<28}: {sub['home_sp_k'].mean():.2f}")
                print(f"  {'Home SP — model avg K':<28}: {sub[MC_HOME_K_MEAN_COL].mean():.2f}")
        elif home_n == 0:
            print("  No matched home SP K predictions.")
        else:
            print(f"  {MC_HOME_K_MEAN_COL} not present.")

        if away_n is not None and away_n > 0:
            print(f"  {'Away SP — n':<28}: {away_n}")
            print(f"  {'Away SP — MAE (Ks)':<28}: {metrics['away_sp_k_mae']:.3f}")
            if MC_AWAY_K_MEAN_COL in df.columns:
                sub = df[[MC_AWAY_K_MEAN_COL, "away_sp_k"]].dropna()
                print(f"  {'Away SP — actual avg K':<28}: {sub['away_sp_k'].mean():.2f}")
                print(f"  {'Away SP — model avg K':<28}: {sub[MC_AWAY_K_MEAN_COL].mean():.2f}")
        elif away_n == 0:
            print("  No matched away SP K predictions.")
        else:
            print(f"  {MC_AWAY_K_MEAN_COL} not present.")

        if combined_mae is not None:
            print(f"  {'Combined SP K MAE':<28}: {combined_mae:.3f}")

    # ----------------------------------------------------------------
    # Actuals sanity check
    # ----------------------------------------------------------------
    _section("Actuals Sanity Check")
    print(f"  {'NRFI rate':<28}: {df['f1_nrfi'].mean():.1%}")
    print(f"  {'F5 total avg':<28}: {df['f5_total'].mean():.2f}")
    print(f"  {'Home SP K avg':<28}: {df['home_sp_k'].mean():.2f}")
    print(f"  {'Away SP K avg':<28}: {df['away_sp_k'].mean():.2f}")
    print(f"  {'Home RL cover %':<28}: {df['home_covers_rl'].mean():.1%}")

    print()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MC model accuracy vs 2026 actuals")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-extract actuals from statcast_2026.parquet before merging",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  backtest_mc_2026.py")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Optionally rebuild actuals
    # ----------------------------------------------------------------
    if args.rebuild:
        print("\n  [--rebuild] Re-extracting actuals from statcast_2026.parquet ...")
        result = subprocess.run(
            [sys.executable, "extract_actuals_2026.py"],
            capture_output=False,
        )
        if result.returncode != 0:
            print("  ERROR: extract_actuals_2026.py failed. Aborting.")
            sys.exit(1)

    # ----------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------
    print()
    actuals  = load_actuals()
    backtest = load_backtest()

    # ----------------------------------------------------------------
    # Merge
    # ----------------------------------------------------------------
    merged = merge_datasets(actuals, backtest)

    if merged.empty:
        print("\n  No overlapping games found between actuals and backtest. Exiting.")
        sys.exit(0)

    # ----------------------------------------------------------------
    # Add per-row error columns
    # ----------------------------------------------------------------
    merged = add_error_columns(merged)

    # ----------------------------------------------------------------
    # Compute summary metrics
    # ----------------------------------------------------------------
    metrics = compute_metrics(merged)

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved -> {OUTPUT_CSV}  ({len(merged)} rows)")

    # ----------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------
    print_report(merged, metrics)


if __name__ == "__main__":
    main()
