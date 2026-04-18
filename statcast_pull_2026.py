"""
statcast_pull_2026.py
=====================
Daily incremental append of Statcast pitch-level data for the current season.

Reads the current max date in statcast_2026.parquet, pulls from (max_date + 1)
through yesterday via pybaseball.statcast(), and appends deduplicated rows.
After a successful append, re-runs extract_actuals_2026.py to refresh actuals.

Statcast data is typically available from Baseball Savant by ~3 AM ET the
following morning, so the 4 AM refresh slot picks up the prior day reliably.

Usage:
  python statcast_pull_2026.py              # pull yesterday forward
  python statcast_pull_2026.py --from 2026-04-01  # pull from specific date
  python statcast_pull_2026.py --dry-run    # show what would be pulled, no write
"""

import argparse
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR      = Path("./data/statcast")
OUT_PATH      = DATA_DIR / "statcast_2026.parquet"
SEASON_START  = "2026-03-20"   # Opening Day 2026 (update each year)


def _current_max_date() -> str:
    """Return the latest game_date already in statcast_2026.parquet, or season start."""
    if not OUT_PATH.exists():
        return SEASON_START
    df = pd.read_parquet(OUT_PATH, columns=["game_date"], engine="pyarrow")
    if df.empty:
        return SEASON_START
    max_dt = pd.to_datetime(df["game_date"]).max()
    return max_dt.strftime("%Y-%m-%d")


def pull_and_append(start_date: str, end_date: str, dry_run: bool = False) -> int:
    """
    Pull Statcast data for [start_date, end_date] and append to parquet.
    Returns number of new rows added.
    """
    try:
        import pybaseball
        pybaseball.cache.enable()
    except ImportError:
        print("  ERROR: pybaseball not installed — run: pip install pybaseball")
        return 0

    print(f"  Pulling Statcast {start_date} → {end_date} ...")
    try:
        df_new = pybaseball.statcast(start_dt=start_date, end_dt=end_date)
    except Exception as e:
        print(f"  ERROR pulling Statcast: {e}")
        return 0

    if df_new is None or df_new.empty:
        print("  No data returned (off-day or too early for yesterday's data).")
        return 0

    df_new["game_date"] = pd.to_datetime(df_new["game_date"])

    if dry_run:
        print(f"  [dry-run] Would add {len(df_new)} rows covering {df_new['game_date'].min().date()} → {df_new['game_date'].max().date()}")
        return 0

    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH, engine="pyarrow")
        existing["game_date"] = pd.to_datetime(existing["game_date"])
        combined = (
            pd.concat([existing, df_new], ignore_index=True)
            .drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
            .sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])
            .reset_index(drop=True)
        )
        new_rows = len(combined) - len(existing)
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        combined = df_new.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
        new_rows = len(combined)

    combined.to_parquet(OUT_PATH, engine="pyarrow", index=False)
    max_date = combined["game_date"].max().strftime("%Y-%m-%d")
    print(f"  statcast_2026.parquet: +{new_rows} rows | total={len(combined)} | max_date={max_date}")
    return new_rows


def run_extract_actuals() -> None:
    """Re-run extract_actuals_2026.py after a successful statcast update."""
    try:
        import extract_actuals_2026 as ea
        actuals = ea.extract_actuals(OUT_PATH)
        out = DATA_DIR / "actuals_2026.parquet"
        actuals.to_parquet(out, index=False, engine="pyarrow")
        print(f"  actuals_2026.parquet refreshed: {len(actuals)} games | "
              f"max_date={actuals['game_date'].max()}")
    except Exception as e:
        print(f"  WARNING: extract_actuals failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Statcast append for 2026 season")
    parser.add_argument("--from",   dest="from_date", help="Start date YYYY-MM-DD (default: day after current max)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be pulled without writing")
    args = parser.parse_args()

    yesterday = (date.today() - timedelta(days=1)).isoformat()

    if args.from_date:
        start = args.from_date
    else:
        current_max = _current_max_date()
        start = (pd.to_datetime(current_max) + timedelta(days=1)).strftime("%Y-%m-%d")

    if start > yesterday:
        print(f"  statcast_2026 already current through {current_max} — nothing to pull.")
        return

    new_rows = pull_and_append(start, yesterday, dry_run=args.dry_run)

    if new_rows > 0:
        print("  Re-extracting actuals ...")
        run_extract_actuals()
    else:
        print("  No new rows — skipping actuals refresh.")


if __name__ == "__main__":
    main()
