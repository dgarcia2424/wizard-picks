"""
build_backtest.py — Processes raw Statcast Parquet files into game-level summaries
                    for model backtesting and weight calibration.

Reads from: ./statcast_data/statcast_YYYY.parquet  (produced by statcast_pull.py)
Writes to:  ./backtest_games_YYYY.csv

Per-game output columns:
  game_date, home_team, away_team,
  home_starter, away_starter        (FIRST LAST uppercase — matches pipeline)
  actual_game_total, actual_f5_total, actual_f3_total, actual_f1_total
  actual_home_win

Usage:
    # Step 1 — download raw data (one-time, ~30-40 min per season):
    python statcast_pull.py

    # Step 2 — process into game summaries (fast, re-runnable):
    python build_backtest.py              # processes 2024 + 2025
    python build_backtest.py --year 2025  # single season
"""

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from pybaseball import playerid_reverse_lookup, cache

cache.enable()

STATCAST_DIR = Path("./data/statcast")   # where statcast_pull.py saves files
RAW_DIR      = Path("./data/raw")

# ─── SEASON DATE RANGES ───────────────────────────────────────────────────────

SEASONS = {
    2024: ("2024-03-20", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-28"),
    2026: ("2026-03-27", date.today().strftime("%Y-%m-%d")),
}

# ─── TEAM ABBREVIATION FIXES ─────────────────────────────────────────────────
# Statcast uses slightly different codes than the pipeline in some cases

TEAM_FIXES = {
    "WSN": "WSH",
    "TBR": "TB",
    "SDP": "SD",
    "SFG": "SF",
    "KCR": "KC",
    "CHW": "CWS",
    "ANA": "LAA",
    "FLA": "MIA",
    "MON": "WSH",
}

# ─── STATCAST COLUMNS WE ACTUALLY NEED ───────────────────────────────────────
# Drop everything else immediately to keep memory manageable

KEEP_COLS = [
    "game_pk", "game_date", "home_team", "away_team",
    "pitcher",
    "inning", "inning_topbot",
    "at_bat_number", "pitch_number",
    "post_home_score", "post_away_score",
]


# ─── DATA LOAD ────────────────────────────────────────────────────────────────

def load_parquet(year):
    """Load a season's Parquet file (produced by statcast_pull.py), keep only needed cols."""
    path = STATCAST_DIR / f"statcast_{year}.parquet"
    if not path.exists():
        print(f"  ERROR: {path} not found.")
        print(f"  Run statcast_pull.py first to download the raw data.")
        return None

    print(f"  Loading {path} ...", end="", flush=True)
    df = pd.read_parquet(path)

    # Keep only the columns we need
    keep = [c for c in KEEP_COLS if c in df.columns]
    df = df[keep].copy()

    # Normalize inning_topbot casing
    if "inning_topbot" in df.columns:
        df["inning_topbot"] = df["inning_topbot"].str.capitalize()

    print(f" {len(df):,} pitches  ({df['game_pk'].nunique()} games)")
    return df


# ─── GAME PROCESSING ─────────────────────────────────────────────────────────

def score_after_inning(gdf, n):
    """
    Returns total runs scored (home + away) after the bottom of inning N.
    Handles walk-off games where inning N has no bottom half.
    """
    bot = gdf[(gdf["inning"] == n) & (gdf["inning_topbot"] == "Bot")]
    if not bot.empty:
        last = bot.iloc[-1]
        return float(last["post_home_score"]) + float(last["post_away_score"])

    # No bottom half (e.g. home team already winning at end of top in extras)
    top = gdf[(gdf["inning"] == n) & (gdf["inning_topbot"] == "Top")]
    if not top.empty:
        last = top.iloc[-1]
        return float(last["post_home_score"]) + float(last["post_away_score"])

    return None


def process_game(game_pk, gdf):
    """
    Convert one game's pitch-level rows into a single summary dict.

    Starter identification:
      - Home starter pitches in the TOP of inning 1  (away team bats)
      - Away starter pitches in the BOTTOM of inning 1 (home team bats)
    """
    # Sort: inning asc, Top before Bot, then at-bat and pitch order
    gdf = gdf.sort_values(
        ["inning", "inning_topbot", "at_bat_number", "pitch_number"],
        ascending=[True, False, True, True],   # False on topbot → Top < Bot alphabetically reversed
    )

    first = gdf.iloc[0]
    last  = gdf.iloc[-1]

    home_team = TEAM_FIXES.get(str(first["home_team"]), str(first["home_team"]))
    away_team = TEAM_FIXES.get(str(first["away_team"]), str(first["away_team"]))

    # Starting pitchers — first pitcher to appear on each side in inning 1
    top1 = gdf[(gdf["inning"] == 1) & (gdf["inning_topbot"] == "Top")]   # home pitcher faces away batters
    bot1 = gdf[(gdf["inning"] == 1) & (gdf["inning_topbot"] == "Bot")]   # away pitcher faces home batters

    home_starter_id = int(top1["pitcher"].iloc[0]) if not top1.empty else None
    away_starter_id = int(bot1["pitcher"].iloc[0]) if not bot1.empty else None

    # Actual totals
    f1    = score_after_inning(gdf, 1)
    f3    = score_after_inning(gdf, 3)
    f5    = score_after_inning(gdf, 5)
    total = float(last["post_home_score"]) + float(last["post_away_score"])
    home_win = float(last["post_home_score"]) > float(last["post_away_score"])

    return {
        "game_pk":           int(game_pk),
        "game_date":         str(first["game_date"])[:10],
        "home_team":         home_team,
        "away_team":         away_team,
        "home_starter_id":   home_starter_id,
        "away_starter_id":   away_starter_id,
        "actual_game_total": round(total, 1),
        "actual_f5_total":   round(f5, 1) if f5 is not None else None,
        "actual_f3_total":   round(f3, 1) if f3 is not None else None,
        "actual_f1_total":   round(f1, 1) if f1 is not None else None,
        "actual_home_win":   home_win,
    }


# ─── PITCHER NAME RESOLUTION ─────────────────────────────────────────────────

def resolve_names(games_df):
    """
    Batch-convert MLBAM pitcher IDs → 'FIRST LAST' uppercase names.
    Matches the format score_models.py uses for pitcher lookups.
    """
    ids = set()
    for col in ["home_starter_id", "away_starter_id"]:
        ids |= set(games_df[col].dropna().astype(int).tolist())

    if not ids:
        return games_df

    print(f"  Resolving {len(ids)} pitcher IDs ... ", end="", flush=True)
    try:
        lu = playerid_reverse_lookup(list(ids), key_type="mlbam")
        name_map = {}
        for _, row in lu.iterrows():
            pid  = int(row["key_mlbam"])
            name = f"{str(row['name_first']).strip()} {str(row['name_last']).strip()}".upper()
            name_map[pid] = name
        print(f"resolved {len(name_map)}")
    except Exception as e:
        print(f"WARNING — name lookup failed ({e}), keeping IDs")
        name_map = {}

    out = games_df.copy()
    out["home_starter"] = out["home_starter_id"].apply(
        lambda x: name_map.get(int(x), f"ID:{int(x)}") if pd.notna(x) else None
    )
    out["away_starter"] = out["away_starter_id"].apply(
        lambda x: name_map.get(int(x), f"ID:{int(x)}") if pd.notna(x) else None
    )
    return out.drop(columns=["home_starter_id", "away_starter_id"])


# ─── SEASON BUILDER ───────────────────────────────────────────────────────────

def build_season(year):
    if year not in SEASONS:
        print(f"No date range configured for {year}")
        return

    start_str, end_str = SEASONS[year]
    start_dt = datetime.strptime(start_str, "%Y-%m-%d").date()
    end_dt   = min(datetime.strptime(end_str, "%Y-%m-%d").date(), date.today())

    print(f"\n{'='*60}")
    print(f"  Season {year}   {start_dt} → {end_dt}")
    print(f"{'='*60}")

    # Build list of months to fetch
    months, cur = [], date(start_dt.year, start_dt.month, 1)
    while cur <= end_dt:
        months.append(cur.month)
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)

    pitches = load_parquet(year)
    if pitches is None:
        return
    n_games = pitches["game_pk"].nunique()
    print(f"\n  Processing {len(pitches):,} pitches across {n_games} games...")

    rows, errors = [], 0
    for gk, gdf in pitches.groupby("game_pk"):
        try:
            rows.append(process_game(gk, gdf))
        except Exception as e:
            errors += 1
    if errors:
        print(f"  {errors} games skipped due to errors")

    games_df = pd.DataFrame(rows)
    games_df = resolve_names(games_df)

    # Final column order
    cols = [
        "game_date", "home_team", "away_team",
        "home_starter", "away_starter",
        "actual_game_total", "actual_f5_total", "actual_f3_total", "actual_f1_total",
        "actual_home_win", "game_pk",
    ]
    games_df = games_df[[c for c in cols if c in games_df.columns]]
    games_df = games_df.sort_values("game_date").reset_index(drop=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"backtest_games_{year}.csv"
    games_df.to_csv(out_path, index=False)

    print(f"\n  {len(games_df)} games saved → {out_path}")
    print(f"  F5 coverage : {games_df['actual_f5_total'].notna().sum()} / {len(games_df)}")
    print(f"  F3 coverage : {games_df['actual_f3_total'].notna().sum()} / {len(games_df)}")
    print(f"  Sample:")
    print(games_df.head(5).to_string(index=False))


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build historical game backtest dataset")
    parser.add_argument("--year", type=int, help="Single year to build (default: 2024 + 2025)")
    args = parser.parse_args()

    years = [args.year] if args.year else [2024, 2025]
    for year in years:
        build_season(year)
    print("\nDone.")


if __name__ == "__main__":
    main()
