"""
ump_pull.py
===========
Pull home-plate umpire assignments for every MLB game via the MLB Stats API
(no auth required) and save to data/statcast/umpire_assignments_{year}.parquet.

Runs in two modes:
  Historical  -- pulls every gamePk found in schedule_all_{year}.parquet
  Daily       -- pulls today's (or a specified date's) games

The home-plate umpire is the single most actionable umpire signal:
  - K%-friendly zones inflate strikeout rates → fewer baserunners → fewer runs
  - BB%-friendly zones inflate walk rates     → more baserunners → more runs
  These effects are consistent across umpires and known before first pitch.

Usage
-----
  python ump_pull.py --years 2023 2024 2025      # historical backfill
  python ump_pull.py --date 2026-04-15           # today / specific date
  python ump_pull.py --years 2026 --date 2026-04-15  # both
"""

import argparse
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

DATA_DIR    = Path("./data/statcast")
API_BASE    = "https://statsapi.mlb.com/api/v1"
RATE_SLEEP  = 0.07   # seconds between API calls (~14 req/s, well under MLB limit)
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict = None, retries: int = MAX_RETRIES) -> dict:
    """GET with simple retry + exponential backoff."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}] {exc} — waiting {wait}s")
            time.sleep(wait)
    return {}


def _hp_ump_from_boxscore(game_pk: int) -> dict | None:
    """
    Return {ump_id, ump_name} for the home-plate umpire of game_pk,
    or None if unavailable.
    """
    url = f"{API_BASE}/game/{game_pk}/boxscore"
    try:
        data = _get(url)
    except Exception:
        return None

    for official in data.get("officials", []):
        if official.get("officialType") == "Home Plate":
            o = official.get("official", {})
            return {
                "ump_hp_id":   o.get("id"),
                "ump_hp_name": o.get("fullName", ""),
            }
    return None


def _game_pks_from_schedule(year: int) -> list[int]:
    """Load unique gamePks from the local schedule_all_{year}.parquet."""
    path = DATA_DIR / f"schedule_all_{year}.parquet"
    if not path.exists():
        print(f"  [WARN] schedule_all_{year}.parquet not found — skipping")
        return []
    sched = pd.read_parquet(path, engine="pyarrow")
    pks = (
        pd.to_numeric(sched["gamePk"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    # Only regular season games (gamePk is sequential; filter out obvious
    # spring-training PKs by checking schedule status if available)
    if "status" in sched.columns:
        valid = sched[sched["status"].isin(["Final", "Completed Early"])]["gamePk"]
        valid = pd.to_numeric(valid, errors="coerce").dropna().astype(int).unique()
        pks = [p for p in pks if p in set(valid)]
    return sorted(set(pks))


def _game_pks_from_date(date_str: str) -> list[int]:
    """
    Fetch gamePks for a specific date from the live schedule API.
    Returns list of int gamePks.
    """
    url    = f"{API_BASE}/schedule"
    params = {"sportId": 1, "date": date_str}
    try:
        data = _get(url, params)
    except Exception as exc:
        print(f"  [WARN] Schedule API failed for {date_str}: {exc}")
        return []

    pks = []
    for day in data.get("dates", []):
        for game in day.get("games", []):
            status = game.get("status", {}).get("abstractGameState", "")
            if status in ("Final", "Live", "Preview"):
                pks.append(int(game["gamePk"]))
    return pks


# ---------------------------------------------------------------------------
# Historical pull
# ---------------------------------------------------------------------------

def pull_year(year: int, force: bool = False) -> pd.DataFrame:
    """
    Pull HP umpire for every completed game in `year`.

    Saves to data/statcast/umpire_assignments_{year}.parquet and returns the
    DataFrame.  If the file already exists and force=False, loads and returns
    the cached version.
    """
    out_path = DATA_DIR / f"umpire_assignments_{year}.parquet"

    if out_path.exists() and not force:
        print(f"  {year}: loaded cached {out_path.name}")
        return pd.read_parquet(out_path, engine="pyarrow")

    pks = _game_pks_from_schedule(year)
    if not pks:
        return pd.DataFrame()

    print(f"  {year}: pulling {len(pks)} games from MLB Stats API ...")

    rows = []
    for i, pk in enumerate(pks):
        if i % 200 == 0 and i > 0:
            print(f"    {i}/{len(pks)} ...")
        ump = _hp_ump_from_boxscore(pk)
        if ump:
            rows.append({"game_pk": pk, **ump})
        time.sleep(RATE_SLEEP)

    if not rows:
        print(f"  {year}: no umpire data retrieved")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["year"] = year
    df.to_parquet(out_path, engine="pyarrow", index=False)
    n_found = df["ump_hp_name"].notna().sum()
    print(f"  {year}: {n_found}/{len(pks)} umpires saved -> {out_path.name}")
    return df


# ---------------------------------------------------------------------------
# Daily pull
# ---------------------------------------------------------------------------

def pull_date(date_str: str) -> pd.DataFrame:
    """
    Pull HP umpire assignments for a specific date.

    Appends to (or creates) data/statcast/umpire_assignments_2026.parquet
    so the daily features are available for run_today.py.
    """
    year     = int(date_str[:4])
    out_path = DATA_DIR / f"umpire_assignments_{year}.parquet"

    pks = _game_pks_from_date(date_str)
    if not pks:
        print(f"  No games found for {date_str}")
        return pd.DataFrame()

    print(f"  {date_str}: pulling {len(pks)} games ...")
    rows = []
    for pk in pks:
        ump = _hp_ump_from_boxscore(pk)
        if ump:
            rows.append({"game_pk": pk, "game_date": date_str, **ump})
        time.sleep(RATE_SLEEP)

    if not rows:
        return pd.DataFrame()

    new_df = pd.DataFrame(rows)
    new_df["year"] = year

    # Merge with existing file, dedup by game_pk
    if out_path.exists():
        existing = pd.read_parquet(out_path, engine="pyarrow")
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["game_pk"], keep="last")
    else:
        combined = new_df

    combined.to_parquet(out_path, engine="pyarrow", index=False)
    print(f"  Saved {len(new_df)} assignments -> {out_path.name}")
    return new_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pull MLB home-plate umpire assignments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--years", type=int, nargs="+",
                   help="Historical years to pull (e.g. 2023 2024 2025)")
    p.add_argument("--date", type=str,
                   help="Specific date YYYY-MM-DD (daily mode)")
    p.add_argument("--force", action="store_true",
                   help="Re-pull even if cached file exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 55)
    print("  ump_pull.py  - MLB Home-Plate Umpire Assignments")
    print("=" * 55)

    if args.years:
        for year in args.years:
            pull_year(year, force=args.force)

    if args.date:
        pull_date(args.date)

    if not args.years and not args.date:
        # Default: pull today
        today = date.today().isoformat()
        print(f"  No args — defaulting to today ({today})")
        pull_date(today)

    print("\n  Done.")


if __name__ == "__main__":
    main()
