"""
build_bullpen_avail.py
======================
Compute trailing bullpen-availability features from statcast pitch-level data.

For each (team, game_date):
  - Sum reliever pitches thrown in the previous 1 day  (rest_1d)
  - Sum reliever pitches thrown in the previous 2 days (rest_2d)
  - Count distinct relievers who pitched yesterday      (n_used_yday)
  - "Depleted" flag: rest_1d >= 80 (rough proxy for a blown bullpen)

A reliever is any pitcher who started pitching in inning >= 2, OR who pitched
in inning 1 but only went through the order once (n_thruorder_pitcher == 1
and appearance is very short).

Outputs: data/statcast/bullpen_avail_{year}.parquet
Columns: team, game_date, bp_pitches_rest1d, bp_pitches_rest2d,
         bp_n_used_yday, bp_depleted_flag
"""

import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("./data/statcast")
YEARS    = [2023, 2024, 2025]


# ---------------------------------------------------------------------------
def _classify_relievers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'team' column (which side is the pitcher on) and 'is_reliever' flag.
    Returns pitcher-game rows with is_reliever=True only.
    """
    # Assign pitcher's team:
    #   inning_topbot == 'Top'  → away batting  → pitcher is HOME team
    #   inning_topbot == 'Bot'  → home batting  → pitcher is AWAY team
    df = df.copy()
    df["pitcher_team"] = np.where(
        df["inning_topbot"] == "Top",
        df["home_team"],
        df["away_team"],
    )

    # Per (game_pk, pitcher): aggregate to determine starter vs reliever
    grp = df.groupby(["game_pk", "game_date", "pitcher", "pitcher_team"]).agg(
        pitches          = ("pitch_type",              "count"),
        n_thruorder      = ("n_thruorder_pitcher",     "max"),
        inning_min       = ("inning",                  "min"),
    ).reset_index()

    # Starter: enters in inning 1 AND goes through the order at least twice
    grp["is_starter"] = (grp["inning_min"] == 1) & (grp["n_thruorder"] >= 2)
    grp["is_reliever"] = ~grp["is_starter"]

    return grp[grp["is_reliever"]]


def _build_year(year: int, verbose: bool = True) -> pd.DataFrame:
    sc_path = DATA_DIR / f"statcast_{year}.parquet"
    if not sc_path.exists():
        print(f"  {year}: statcast file not found — skip")
        return pd.DataFrame()

    df = pd.read_parquet(sc_path, engine="pyarrow",
                         columns=["game_pk", "game_date", "pitcher", "pitch_type",
                                  "home_team", "away_team", "inning", "inning_topbot",
                                  "n_thruorder_pitcher"])
    df["game_date"] = pd.to_datetime(df["game_date"])

    rel = _classify_relievers(df)

    # Sum pitches per team per game date (relievers only)
    daily = (
        rel.groupby(["pitcher_team", "game_date"])
        .agg(
            bp_pitches_day  = ("pitches",   "sum"),
            bp_n_used       = ("pitcher",   "nunique"),
        )
        .reset_index()
        .rename(columns={"pitcher_team": "team"})
        .sort_values(["team", "game_date"])
    )

    # Lag: pitches thrown in prior 1 day and prior 2 days
    out_rows = []
    for team, grp in daily.groupby("team"):
        grp = grp.set_index("game_date").sort_index()
        # For each game date in this year, look back
        # We'll use a rolling sum on a date-indexed series
        s_pitches = grp["bp_pitches_day"]
        s_n_used  = grp["bp_n_used"]

        # Build a full daily index to handle missing dates
        full_idx = pd.date_range(s_pitches.index.min(), s_pitches.index.max(), freq="D")
        s_pitches = s_pitches.reindex(full_idx, fill_value=0)
        s_n_used  = s_n_used.reindex(full_idx, fill_value=0)

        # 1-day lag sum (yesterday only)
        rest1d   = s_pitches.shift(1).fillna(0)
        n_yday   = s_n_used.shift(1).fillna(0)
        # 2-day lag sum (yesterday + day before)
        rest2d   = s_pitches.shift(1).fillna(0) + s_pitches.shift(2).fillna(0)

        tbl = pd.DataFrame({
            "team":               team,
            "game_date":          full_idx,
            "bp_pitches_rest1d":  rest1d.values,
            "bp_pitches_rest2d":  rest2d.values,
            "bp_n_used_yday":     n_yday.values.astype(int),
        })
        # Keep only dates that actually had a game for this team
        game_dates = set(grp.index)
        tbl = tbl[tbl["game_date"].isin(game_dates)]
        out_rows.append(tbl)

    result = pd.concat(out_rows, ignore_index=True)
    result["bp_depleted_flag"] = (result["bp_pitches_rest1d"] >= 80).astype("int8")

    out_path = DATA_DIR / f"bullpen_avail_{year}.parquet"
    result.to_parquet(out_path, engine="pyarrow", index=False)

    if verbose:
        print(f"  {year}: {len(result)} team-game rows | "
              f"avg rest1d pitches={result['bp_pitches_rest1d'].mean():.1f} | "
              f"depleted={result['bp_depleted_flag'].mean():.1%} -> {out_path.name}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Build bullpen availability features")
    parser.add_argument("--years", nargs="+", type=int, default=YEARS)
    args = parser.parse_args()

    print("=" * 55)
    print("  build_bullpen_avail.py - Bullpen Availability Features")
    print("=" * 55)
    for yr in args.years:
        _build_year(yr)
    print("\n  Done.")


if __name__ == "__main__":
    main()
