"""
extract_actuals_2026.py
=======================
Extract ground-truth F1 (first inning), F5 (first 5 innings), and K
(pitcher strikeouts) actuals from statcast_2026.parquet for all 2026 games.

These actuals are used to validate model predictions in backtest_mc_2026.py.

Inputs:
  data/statcast/statcast_2026.parquet

Output:
  data/statcast/actuals_2026.parquet  — one row per game_pk

Column reference:
  F1  — first inning run-scoring outcomes (NRFI/YRFI)
  F5  — cumulative scores through 5 complete innings
  K   — starting pitcher strikeout totals for the full game
  RL  — full-game run-line outcome

Usage:
  python extract_actuals_2026.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
DATA_DIR      = Path("./data/statcast")
STATCAST_PATH = DATA_DIR / "statcast_2026.parquet"
OUTPUT_PATH   = DATA_DIR / "actuals_2026.parquet"


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _identify_sp(inning1_group: pd.DataFrame, side: str) -> int | None:
    """Return pitcher_id for the SP on the given side (Top=away, Bot=home).

    The SP is the pitcher who appears earliest in inning 1 for that half-inning,
    identified by the lowest at_bat_number.
    """
    half = inning1_group[inning1_group["inning_topbot"] == side]
    if half.empty:
        return None
    # Use at_bat_number to find the very first batter faced
    idx = half["at_bat_number"].idxmin()
    return int(half.loc[idx, "pitcher"])


def extract_f1(game_df: pd.DataFrame) -> dict:
    """Extract first-inning run actuals for a single game.

    Uses post_home_score / post_away_score at the end of each half-inning.
    In inning 1 the starting scores are always 0, so:
      f1_home_runs = max(post_home_score) in Bot of inning 1
      f1_away_runs = max(post_away_score) in Top of inning 1
    """
    inn1 = game_df[game_df["inning"] == 1]

    # Away bats in Top of 1st
    top1 = inn1[inn1["inning_topbot"] == "Top"]
    f1_away_runs = float(top1["post_away_score"].max()) if not top1.empty else 0.0

    # Home bats in Bot of 1st
    bot1 = inn1[inn1["inning_topbot"] == "Bot"]
    f1_home_runs = float(bot1["post_home_score"].max()) if not bot1.empty else 0.0

    f1_home_scored = int(f1_home_runs > 0)
    f1_away_scored = int(f1_away_runs > 0)
    f1_nrfi        = int(f1_home_runs == 0 and f1_away_runs == 0)

    return {
        "f1_home_scored": f1_home_scored,
        "f1_away_scored": f1_away_scored,
        "f1_nrfi":        f1_nrfi,
        "f1_home_runs":   f1_home_runs,
        "f1_away_runs":   f1_away_runs,
    }


def extract_f5(game_df: pd.DataFrame) -> dict:
    """Extract first-5-inning run totals for a single game.

    Uses max cumulative post_*_score across all plate appearances in innings
    1-5. This handles both complete and partial 5th innings correctly: if a
    team doesn't bat in the bottom of the 5th (walk-off or lead), we still
    have the max score through whatever half-innings completed.
    """
    f5 = game_df[game_df["inning"] <= 5]

    f5_home_runs = float(f5["post_home_score"].max()) if not f5.empty else 0.0
    f5_away_runs = float(f5["post_away_score"].max()) if not f5.empty else 0.0
    f5_total     = f5_home_runs + f5_away_runs
    f5_home_win  = int(f5_home_runs > f5_away_runs)

    return {
        "f5_home_runs": f5_home_runs,
        "f5_away_runs": f5_away_runs,
        "f5_total":     f5_total,
        "f5_home_win":  f5_home_win,
    }


def extract_sp_ks(game_df: pd.DataFrame, home_sp_id: int | None, away_sp_id: int | None) -> dict:
    """Extract SP strikeouts and estimated innings pitched for the full game.

    Counts every plate appearance with events == 'strikeout' (including
    strikeout_double_play) for each identified SP pitcher, across all innings.

    IP is estimated as (total pitches seen) / 15.5, which approximates the
    historical average pitches-per-inning for MLB starters.
    """
    strikeout_events = {"strikeout", "strikeout_double_play"}

    def _sp_stats(sp_id: int | None) -> tuple[int, float]:
        if sp_id is None:
            return 0, 0.0
        sp_pa = game_df[game_df["pitcher"] == sp_id]
        # Each row is a pitch; count rows where the at-bat ended in a strikeout
        k_count   = int(sp_pa["events"].isin(strikeout_events).sum())
        # Estimate IP via total pitches thrown (all rows, not just terminal ones)
        total_pitches = len(sp_pa)
        est_ip = round(total_pitches / 15.5, 1)
        return k_count, est_ip

    home_sp_k, home_sp_ip = _sp_stats(home_sp_id)
    away_sp_k, away_sp_ip = _sp_stats(away_sp_id)

    return {
        "home_sp_id": home_sp_id,
        "away_sp_id": away_sp_id,
        "home_sp_k":  home_sp_k,
        "away_sp_k":  away_sp_k,
        "home_sp_ip": home_sp_ip,
        "away_sp_ip": away_sp_ip,
    }


# ---------------------------------------------------------------------------
# MAIN EXTRACTION
# ---------------------------------------------------------------------------

def extract_actuals(statcast_path: Path = STATCAST_PATH) -> pd.DataFrame:
    """Load statcast data and extract per-game actuals. Returns a DataFrame."""

    print(f"  Loading {statcast_path} ...")
    sc = pd.read_parquet(statcast_path, engine="pyarrow")
    sc["game_date"] = pd.to_datetime(sc["game_date"]).dt.date.astype(str)

    # Cast pitcher id to nullable int for safe comparisons
    sc["pitcher"] = pd.to_numeric(sc["pitcher"], errors="coerce").astype("Int64")

    # Normalise events to str so membership tests work on both Arrow and object dtypes
    sc["events"] = sc["events"].astype(str)

    all_game_pks = sc["game_pk"].unique()
    print(f"  Found {len(all_game_pks)} unique games.")

    records = []

    for game_pk in sorted(all_game_pks):
        game_df = sc[sc["game_pk"] == game_pk]

        # ----------------------------------------------------------------
        # Game metadata
        # ----------------------------------------------------------------
        game_date = str(game_df["game_date"].iloc[0])
        home_team = str(game_df["home_team"].iloc[0])
        away_team = str(game_df["away_team"].iloc[0])

        # ----------------------------------------------------------------
        # Identify starting pitchers from inning 1
        # ----------------------------------------------------------------
        inn1 = game_df[game_df["inning"] == 1]

        # Away SP pitches in Top of 1st (faces home lineup)
        # Home SP pitches in Bot of 1st (faces away lineup)
        away_sp_id = _identify_sp(inn1, "Top")   # pitches to home batters
        home_sp_id = _identify_sp(inn1, "Bot")   # pitches to away batters

        # ----------------------------------------------------------------
        # Full-game final scores
        # ----------------------------------------------------------------
        home_score_final = int(game_df["post_home_score"].max())
        away_score_final = int(game_df["post_away_score"].max())
        home_margin      = home_score_final - away_score_final
        home_covers_rl   = int(home_margin >= 2)

        # ----------------------------------------------------------------
        # Metric extraction
        # ----------------------------------------------------------------
        f1_data = extract_f1(game_df)
        f5_data = extract_f5(game_df)
        sp_data = extract_sp_ks(game_df, home_sp_id, away_sp_id)

        record = {
            "game_pk":           game_pk,
            "game_date":         game_date,
            "home_team":         home_team,
            "away_team":         away_team,
            # F1
            **f1_data,
            # F5
            **f5_data,
            # SP Ks
            **sp_data,
            # Full game
            "home_score_final":  home_score_final,
            "away_score_final":  away_score_final,
            "home_covers_rl":    home_covers_rl,
        }
        records.append(record)

    actuals = pd.DataFrame(records)
    actuals = actuals.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    return actuals


# ---------------------------------------------------------------------------
# SUMMARY PRINTER
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    print()
    print("=" * 60)
    print("  actuals_2026  — extraction summary")
    print("=" * 60)
    print(f"  Games extracted  : {len(df)}")
    print(f"  Date range       : {df['game_date'].min()}  to  {df['game_date'].max()}")
    print()

    # NRFI rate
    nrfi_rate = df["f1_nrfi"].mean()
    print(f"  NRFI rate        : {nrfi_rate:.1%}  (league ~55-58%)")

    # F5 averages
    f5_avg = df["f5_total"].mean()
    print(f"  F5 total avg     : {f5_avg:.2f} runs")

    # SP K averages
    home_k_avg = df["home_sp_k"].mean()
    away_k_avg = df["away_sp_k"].mean()
    print(f"  Home SP K avg    : {home_k_avg:.2f}")
    print(f"  Away SP K avg    : {away_k_avg:.2f}")

    # RL cover rate
    rl_rate = df["home_covers_rl"].mean()
    print(f"  Home RL cover %  : {rl_rate:.1%}  (historical ~35.7%)")

    # Missing SP counts
    missing_home_sp = df["home_sp_id"].isna().sum()
    missing_away_sp = df["away_sp_id"].isna().sum()
    if missing_home_sp or missing_away_sp:
        print(f"\n  WARNING: {missing_home_sp} games missing home SP id, "
              f"{missing_away_sp} missing away SP id")

    print()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  extract_actuals_2026.py")
    print("=" * 60)

    actuals = extract_actuals(STATCAST_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    actuals.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")
    print(f"\n  Saved -> {OUTPUT_PATH}  ({len(actuals)} rows)")

    print_summary(actuals)


if __name__ == "__main__":
    main()
