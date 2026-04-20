"""
backfill_lineup_quality.py
==========================
Retroactively compute lineup quality (wRC+, xwOBA vs LHP/RHP) for every
game in 2023, 2024, and 2025.

Steps per year:
  1. Pull historical lineups via lineup_pull.py --historical (MLB Stats API)
     — boxscore endpoint used for completed games → actual batting orders
  2. Iterate over every game date in lineups_{year}.parquet
  3. Call build_lineup_quality.build(date_str) for each date
  4. Consolidate into data/statcast/lineup_quality_{year}.parquet

Outputs:
  data/statcast/lineups_{year}.parquet          (lineup pull — wide format)
  data/statcast/lineup_quality_{year}.parquet   (one row per game-team)

These files are read by enrich_feature_matrix_v2.py when it looks up
lineup_wrc_plus for historical games.

Usage:
  python backfill_lineup_quality.py                     # all years 2023-2025
  python backfill_lineup_quality.py --years 2024 2025   # specific years
  python backfill_lineup_quality.py --quality-only      # skip lineup pull, just rebuild quality
  python backfill_lineup_quality.py --year 2025 --start 2025-04-01 --end 2025-09-28
"""

import argparse
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("data/statcast")

# Season date ranges (Opening Day → last regular-season day)
SEASON_DATES = {
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-20", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-28"),
}

DEFAULT_YEARS = [2023, 2024, 2025]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], label: str) -> int:
    print(f"\n>>> {label}")
    result = subprocess.run(
        [sys.executable, "-X", "utf8"] + cmd,
        cwd=Path(__file__).parent,
        text=True, encoding="utf-8", errors="replace",
    )
    return result.returncode


def pull_lineups_for_year(year: int, start: str, end: str) -> bool:
    """
    Run lineup_pull.py --historical for [start, end].
    Returns True if lineups_{year}.parquet now exists.
    """
    out_path = OUTPUT_DIR / f"lineups_{year}.parquet"
    if out_path.exists():
        df = pd.read_parquet(out_path)
        print(f"  lineups_{year}.parquet already exists: {len(df)} games")
        # Check coverage — if it has most of the season, skip
        n_dates = df["game_date"].nunique() if "game_date" in df.columns else 0
        expected_dates = (
            pd.to_datetime(end) - pd.to_datetime(start)
        ).days * 0.55  # ~55% of days have games
        if n_dates >= expected_dates * 0.9:
            print(f"  Coverage OK ({n_dates} dates) — skipping re-pull")
            return True
        print(f"  Coverage low ({n_dates} dates < {expected_dates:.0f} expected) — re-pulling")

    rc = _run(
        ["lineup_pull.py", "--historical", "--start", start, "--end", end],
        f"Pulling lineups {year} ({start} → {end})"
    )
    return out_path.exists()


def build_quality_for_year(year: int, start: str, end: str,
                            quality_only: bool = False) -> pd.DataFrame:
    """
    For each game date in lineups_{year}.parquet, call build_lineup_quality.build()
    and accumulate results.  Saves to lineup_quality_{year}.parquet.
    """
    from build_lineup_quality import build as lq_build
    from build_batter_splits import main as batter_splits_main

    out_path = OUTPUT_DIR / f"lineup_quality_{year}.parquet"

    # Load lineups to get all game dates
    lineup_path = OUTPUT_DIR / f"lineups_{year}.parquet"
    if not lineup_path.exists():
        print(f"  [WARN] lineups_{year}.parquet not found — cannot compute quality")
        return pd.DataFrame()

    lineups = pd.read_parquet(lineup_path)
    lineups["game_date"] = pd.to_datetime(lineups["game_date"])
    all_dates = sorted(lineups["game_date"].dt.strftime("%Y-%m-%d").unique())
    print(f"\n  {year}: {len(all_dates)} game dates to process "
          f"({all_dates[0]} → {all_dates[-1]})")

    # Load existing quality file to skip already-processed dates
    existing_dates: set[str] = set()
    existing_rows: list[pd.DataFrame] = []
    if out_path.exists():
        try:
            existing = pd.read_parquet(out_path)
            existing["game_date"] = pd.to_datetime(existing["game_date"])
            existing_dates = set(existing["game_date"].dt.strftime("%Y-%m-%d").unique())
            existing_rows.append(existing)
            print(f"  Already have {len(existing_dates)} dates — will skip those")
        except Exception as e:
            print(f"  [WARN] Could not read existing quality file: {e}")

    dates_to_process = [d for d in all_dates if d not in existing_dates]
    print(f"  Dates to process: {len(dates_to_process)}")

    if not dates_to_process:
        print(f"  All dates already processed — {out_path.name} is current")
        return pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()

    # Ensure batter splits are built for this year before computing quality
    print(f"  Ensuring batter splits exist for {year} ...")
    _run(["build_batter_splits.py", "--years", str(year)], f"batter_splits_{year}")

    new_rows: list[pd.DataFrame] = []
    n = len(dates_to_process)
    for i, date_str in enumerate(dates_to_process, 1):
        if i % 25 == 0 or i == 1 or i == n:
            print(f"  [{i:4d}/{n}] {date_str} ...", flush=True)
        try:
            lq_build(date_str, verbose=False)
            # Read the saved parquet for this date
            lq_path = OUTPUT_DIR / f"lineup_quality_{date_str}.parquet"
            if lq_path.exists():
                df = pd.read_parquet(lq_path)
                new_rows.append(df)
        except Exception as e:
            if i <= 5 or i % 50 == 0:
                print(f"  [WARN] {date_str}: {e}")

    # Consolidate all rows
    all_rows = existing_rows + new_rows
    if not all_rows:
        print(f"  No quality rows produced for {year}")
        return pd.DataFrame()

    combined = (
        pd.concat(all_rows, ignore_index=True)
        .drop_duplicates(subset=["game_pk", "team"], keep="last")
        .sort_values(["game_date", "game_pk", "team"])
        .reset_index(drop=True)
    )

    combined.to_parquet(out_path, engine="pyarrow", index=False)
    n_games = combined["game_pk"].nunique()
    n_teams  = len(combined)
    avg_wrc  = combined["lineup_wrc_plus"].mean() if "lineup_wrc_plus" in combined.columns else None
    print(f"\n  {year}: {n_games} games, {n_teams} team-rows | "
          f"avg wRC+={avg_wrc:.1f}" if avg_wrc else f"\n  {year}: {n_games} games, {n_teams} team-rows")
    print(f"  Saved → {out_path}")
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Backfill historical lineup quality (wRC+, xwOBA) for 2023-2025"
    )
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS,
                        help="Years to process (default: 2023 2024 2025)")
    parser.add_argument("--year", type=int, default=None,
                        help="Single year override (sets --years to just this year)")
    parser.add_argument("--start", type=str, default=None,
                        help="Override start date for lineup pull (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,
                        help="Override end date for lineup pull (YYYY-MM-DD)")
    parser.add_argument("--quality-only", action="store_true",
                        help="Skip lineup pull step — only rebuild quality from existing lineups")
    args = parser.parse_args()

    years = [args.year] if args.year else args.years

    print("=" * 60)
    print("  backfill_lineup_quality.py")
    print(f"  Years: {years}")
    print("=" * 60)

    for year in years:
        if year not in SEASON_DATES and not (args.start and args.end):
            print(f"  [SKIP] No season dates defined for {year} — pass --start/--end")
            continue

        start = args.start or SEASON_DATES[year][0]
        end   = args.end   or SEASON_DATES[year][1]

        print(f"\n{'='*60}")
        print(f"  YEAR {year}  ({start} → {end})")
        print(f"{'='*60}")

        # Step 1: Pull historical lineups
        if not args.quality_only:
            ok = pull_lineups_for_year(year, start, end)
            if not ok:
                print(f"  [ERROR] Lineup pull failed for {year} — skipping quality build")
                continue
        else:
            lineup_path = OUTPUT_DIR / f"lineups_{year}.parquet"
            if not lineup_path.exists():
                print(f"  [ERROR] lineups_{year}.parquet not found and --quality-only set")
                continue

        # Step 2: Build lineup quality for each game date
        build_quality_for_year(year, start, end, quality_only=args.quality_only)

    print("\n" + "=" * 60)
    print("  Done. Rebuild feature matrix to incorporate new lineup quality:")
    print("    python enrich_feature_matrix.py")
    print("    python enrich_feature_matrix_v2.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
