"""
pull_odds_history_api.py
========================
Pull closing MLB moneyline + O/U totals from The Odds API historical endpoint
for all game dates in the training matrix (2023-2025).

Cost: ~20 credits per date * 572 dates = ~11,440 credits out of 20,000/month.

Two-pass strategy
-----------------
Pass 1 (T23:00:00Z):  Captures evening/night games (~7pm+ EDT) as their
  pre-game closing lines. Cached as {date}.json.

Pass 2 (T14:00:00Z):  For dates where T23z missed games (afternoon day games
  already finished), fetch the snapshot at 14:00 UTC (~10am EDT) which
  shows pre-game lines before those games start.
  Cached as {date}_14z.json.

Union logic: T23z is preferred (closer to first pitch for most games). T14z
  fills in matchups not covered by T23z.

Doubleheader handling
---------------------
After unioning both snapshots, join to feature matrix on (game_date, home,
away). For doubleheader twins (same date+matchup appearing twice in FM), the
second row inherits the odds from the first.

Outputs: data/statcast/odds_api_hist_{year}.parquet
Columns: game_date, home_team, away_team,
         close_total, close_ml_home, close_ml_away,
         implied_home_prob, implied_away_prob, true_home_prob, true_away_prob,
         game_hour_et

Usage:
    python pull_odds_history_api.py               # all years
    python pull_odds_history_api.py --year 2025   # single year
    python pull_odds_history_api.py --dry-run     # show dates + credit estimate
    python pull_odds_history_api.py --gaps-only   # only fetch gap dates at T14z
    python pull_odds_history_api.py --regen       # rebuild parquets from cache (no API calls)
"""

import argparse
import json
import os
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_DIR       = Path("./data/statcast")
CACHE_DIR      = DATA_DIR / "odds_api_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

API_KEY        = os.getenv("ODDS_API_KEY", "")
BASE_URL       = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds-history/"
QUERY_TIME_23  = "T23:00:00Z"   # closing line (11pm UTC ~7pm EDT)
QUERY_TIME_14  = "T14:00:00Z"   # morning snapshot (~10am EDT) for day games
BOOKMAKERS     = "draftkings,fanduel,pinnacle"
MARKETS        = "totals,h2h"
ODDS_FORMAT    = "american"
SLEEP_BETWEEN  = 0.25           # seconds between API calls

YEARS = [2023, 2024, 2025]

# ── Team name → pipeline abbreviation ─────────────────────────────────────
TEAM_MAP = {
    "Arizona Diamondbacks": "AZ",   # pipeline uses AZ not ARI
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "ATH",     # pipeline uses ATH all years
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
    "Sacramento Athletics": "ATH",  # 2025 relocation
    "Athletics": "ATH",
}


def _abbr(name: str) -> str:
    return TEAM_MAP.get(name, name[:3].upper() if name else "")


def _commence_to_et_hour(commence_time: str) -> float | None:
    """
    Convert an ISO-8601 UTC commence_time string to Eastern Time hour of day.

    Returns a float (e.g. 13.75 = 1:45 PM ET) or None on parse failure.
    Uses the zoneinfo module when available; falls back to a fixed EDT/EST
    offset derived from the month (EDT = UTC-4 for months 3-11, EST = UTC-5).
    """
    if not commence_time:
        return None
    try:
        # Parse UTC timestamp — accept both trailing Z and +00:00
        ts = commence_time.rstrip("Z").replace("+00:00", "")
        dt_utc = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        try:
            from zoneinfo import ZoneInfo
            dt_et = dt_utc.astimezone(ZoneInfo("America/New_York"))
        except Exception:
            # Fallback: EDT (UTC-4) Apr-Oct, EST (UTC-5) Nov-Mar
            offset_h = -4 if 3 <= dt_utc.month <= 11 else -5
            dt_et = dt_utc + timedelta(hours=offset_h)
        return float(dt_et.hour) + dt_et.minute / 60.0
    except Exception:
        return None


def _american_to_implied(odds: float | None) -> float | None:
    """Convert American odds to raw implied probability (not de-vigged)."""
    if odds is None:
        return None
    if odds >= 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def _parse_events(events: list, date_str: str, filter_to_date: bool = False) -> list[dict]:
    """
    Parse a list of API event objects into game dicts.

    Args:
        events:         List of event dicts from the API response.
        date_str:       The date being queried (YYYY-MM-DD). Used to tag game_date.
        filter_to_date: If True, only include events whose commence_time starts
                        on date_str.  Use this for T14z to avoid picking up
                        next-day games.
    """
    games = []
    for event in events:
        home_name = event.get("home_team", "")
        away_name = event.get("away_team", "")
        home_abbr = _abbr(home_name)
        away_abbr = _abbr(away_name)
        if not home_abbr or not away_abbr:
            continue

        # For T14z: skip events whose commence_time is on a different day
        if filter_to_date:
            commence = event.get("commence_time", "")
            if not commence.startswith(date_str):
                continue

        # Collect lines from each bookmaker; prefer pinnacle for sharpness,
        # then draftkings, then fanduel.
        book_priority = {"pinnacle": 0, "draftkings": 1, "fanduel": 2}

        close_total = close_ml_home = close_ml_away = None
        best_book_total = best_book_ml = 99

        for book in event.get("bookmakers", []):
            bk = book.get("key", "")
            pri = book_priority.get(bk, 99)
            for market in book.get("markets", []):
                if market["key"] == "totals" and pri < best_book_total:
                    for o in market.get("outcomes", []):
                        if o["name"] == "Over":
                            close_total = o.get("point")
                            best_book_total = pri
                elif market["key"] == "h2h" and pri < best_book_ml:
                    for o in market.get("outcomes", []):
                        if o["name"] == home_name:
                            close_ml_home = o.get("price")
                        elif o["name"] == away_name:
                            close_ml_away = o.get("price")
                    if close_ml_home is not None:
                        best_book_ml = pri

        if close_total is None and close_ml_home is None:
            continue  # no useful data for this event

        commence = event.get("commence_time", "")
        games.append({
            "game_date":        date_str,
            "home_team":        home_abbr,
            "away_team":        away_abbr,
            "close_total":      close_total,
            "close_ml_home":    close_ml_home,
            "close_ml_away":    close_ml_away,
            "game_hour_et":     _commence_to_et_hour(commence),
        })

    return games


def _fetch_date_t23(date_str: str) -> list[dict]:
    """
    Fetch T23:00:00Z snapshot for a single date.
    Cached as {date_str}.json.
    """
    cache_file = CACHE_DIR / f"{date_str}.json"

    if cache_file.exists():
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        url = (
            f"{BASE_URL}"
            f"?apiKey={API_KEY}"
            f"&regions=us,eu"
            f"&markets={MARKETS}"
            f"&bookmakers={BOOKMAKERS}"
            f"&oddsFormat={ODDS_FORMAT}"
            f"&date={date_str}{QUERY_TIME_23}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "MLB-Model/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = json.loads(r.read())
        cache_file.write_text(json.dumps(raw), encoding="utf-8")
        time.sleep(SLEEP_BETWEEN)

    events = raw.get("data", raw) if isinstance(raw, dict) else raw
    return _parse_events(events, date_str, filter_to_date=False)


def _fetch_date_t14(date_str: str) -> list[dict]:
    """
    Fetch T14:00:00Z snapshot for a single date (morning pre-game lines).
    Cached as {date_str}_14z.json.
    Only returns events whose commence_time is on date_str (filters out next-day games).
    """
    cache_file = CACHE_DIR / f"{date_str}_14z.json"

    if cache_file.exists():
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        url = (
            f"{BASE_URL}"
            f"?apiKey={API_KEY}"
            f"&regions=us,eu"
            f"&markets={MARKETS}"
            f"&bookmakers={BOOKMAKERS}"
            f"&oddsFormat={ODDS_FORMAT}"
            f"&date={date_str}{QUERY_TIME_14}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "MLB-Model/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = json.loads(r.read())
        cache_file.write_text(json.dumps(raw), encoding="utf-8")
        time.sleep(SLEEP_BETWEEN)

    events = raw.get("data", raw) if isinstance(raw, dict) else raw
    # filter_to_date=True ensures we only get games actually on this calendar day
    return _parse_events(events, date_str, filter_to_date=True)


def _compute_de_vig(df: pd.DataFrame) -> pd.DataFrame:
    """Add implied/true probability columns from American moneylines."""
    df = df.copy()
    df["implied_home_prob"] = df["close_ml_home"].apply(_american_to_implied)
    df["implied_away_prob"] = df["close_ml_away"].apply(_american_to_implied)
    total_implied = df["implied_home_prob"] + df["implied_away_prob"]
    df["true_home_prob"] = df["implied_home_prob"] / total_implied
    df["true_away_prob"] = df["implied_away_prob"] / total_implied
    return df


def _get_gap_dates(year: int, fm: pd.DataFrame) -> list[str]:
    """
    Return regular-season dates (>=Apr 1) where the current odds parquet
    does not have a matched line for every FM game on that date.

    Uses matchup-level join (not row counts), so dates where T23z grabbed the
    right number of games but for the WRONG matchups are also captured.
    """
    parquet_path = DATA_DIR / f"odds_api_hist_{year}.parquet"
    if not parquet_path.exists():
        fm_yr = fm[(fm["year"] == year) & (fm["game_date"] >= f"{year}-04-01")]
        return sorted(fm_yr["game_date"].dt.strftime("%Y-%m-%d").unique())

    df_odds = pd.read_parquet(parquet_path)
    df_odds["game_date"] = pd.to_datetime(df_odds["game_date"])

    fm_yr = fm[(fm["year"] == year) & (fm["game_date"] >= f"{year}-04-01")].copy()

    merged = fm_yr.merge(
        df_odds[["game_date", "home_team", "away_team", "close_ml_home"]],
        on=["game_date", "home_team", "away_team"],
        how="left"
    )
    # Dates where any FM game is unmatched
    date_joined = merged.groupby("game_date")["close_ml_home"].count()
    date_total  = merged.groupby("game_date").size()
    gap_dates   = date_total[date_joined < date_total].index
    return sorted(d.strftime("%Y-%m-%d") for d in gap_dates)


def pull_gap_dates(year: int, fm: pd.DataFrame, dry_run: bool = False) -> pd.DataFrame:
    """
    Fetch T14z snapshots for all gap dates in year, then merge with the
    existing T23z parquet and save.

    Returns the updated DataFrame.
    """
    gap_dates = _get_gap_dates(year, fm)
    if not gap_dates:
        print(f"  {year}: no gap dates — skipping T14z pass.")
        parquet_path = DATA_DIR / f"odds_api_hist_{year}.parquet"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        return pd.DataFrame()

    already_cached = sum(1 for d in gap_dates if (CACHE_DIR / f"{d}_14z.json").exists())
    to_fetch = len(gap_dates) - already_cached
    est_cost = to_fetch * 20

    print(f"  {year} T14z pass: {len(gap_dates)} gap dates | "
          f"{already_cached} cached | {to_fetch} to fetch | est. {est_cost:,} credits")

    if dry_run:
        return pd.DataFrame()

    # Fetch T14z for each gap date
    t14_games: list[dict] = []
    errors = 0
    for i, date_str in enumerate(gap_dates):
        try:
            games = _fetch_date_t14(date_str)
            t14_games.extend(games)
            if (i + 1) % 25 == 0:
                print(f"    T14z: {i+1}/{len(gap_dates)} dates processed ...")
        except urllib.error.HTTPError as e:
            errors += 1
            print(f"    HTTPError {e.code} on {date_str} T14z: {e.read().decode()[:100]}")
        except Exception as e:
            errors += 1
            print(f"    Error on {date_str} T14z: {e}")

    print(f"  T14z fetched {len(t14_games)} game-lines ({errors} errors)")

    if not t14_games:
        parquet_path = DATA_DIR / f"odds_api_hist_{year}.parquet"
        return pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()

    t14_df = pd.DataFrame(t14_games)
    t14_df["game_date"] = pd.to_datetime(t14_df["game_date"])
    t14_df = t14_df.drop_duplicates(subset=["game_date", "home_team", "away_team"], keep="last")

    # Load existing T23z parquet
    parquet_path = DATA_DIR / f"odds_api_hist_{year}.parquet"
    if parquet_path.exists():
        t23_df = pd.read_parquet(parquet_path)
        t23_df["game_date"] = pd.to_datetime(t23_df["game_date"])
    else:
        t23_df = pd.DataFrame()

    # Union: prefer T23z, fill from T14z where missing
    if not t23_df.empty:
        combined = pd.concat([t23_df, t14_df], ignore_index=True)
        # Keep T23z rows when there's a conflict (it was loaded first, drop_duplicates keeps first)
        combined = combined.drop_duplicates(subset=["game_date", "home_team", "away_team"],
                                            keep="first")
    else:
        combined = t14_df

    combined = combined.sort_values(["game_date", "home_team", "away_team"]).reset_index(drop=True)
    combined = _compute_de_vig(combined)

    combined.to_parquet(parquet_path, engine="pyarrow", index=False)

    n_t14_new = len(t14_df)
    n_total = combined["close_total"].notna().sum()
    n_ml = combined["close_ml_home"].notna().sum()
    print(f"  {year} after T14z: {len(combined)} games total | "
          f"+{n_t14_new} T14z added | "
          f"total={n_total} ({100*n_total/len(combined):.1f}%) | "
          f"ml={n_ml} ({100*n_ml/len(combined):.1f}%)")

    return combined


def _fill_doubleheader_odds(df_odds: pd.DataFrame, fm: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Doubleheader fix: feature_matrix has some (game_date, home, away) pairs
    with 2 rows (2 games in a day). The odds parquet has at most 1 row per
    matchup per day.  The second game inherits the same line as the first.

    Returns a new DataFrame with the extra rows appended.
    """
    fm_yr = fm[fm["year"] == year][["game_date", "home_team", "away_team"]].copy()
    fm_yr["game_date"] = pd.to_datetime(fm_yr["game_date"])

    # Find double-counted FM rows (same date+home+away, count=2)
    counts = fm_yr.groupby(["game_date", "home_team", "away_team"]).size().reset_index(name="n")
    doubles = counts[counts["n"] > 1]

    if doubles.empty:
        return df_odds

    # For each doubleheader pair, check if odds has exactly 1 entry
    df_odds = df_odds.copy()
    df_odds["game_date"] = pd.to_datetime(df_odds["game_date"])

    extra_rows = []
    for _, row in doubles.iterrows():
        subset = df_odds[
            (df_odds["game_date"] == row["game_date"]) &
            (df_odds["home_team"] == row["home_team"]) &
            (df_odds["away_team"] == row["away_team"])
        ]
        if len(subset) == 1:
            # Duplicate the row for the second game
            extra_rows.append(subset.iloc[0].to_dict())

    if extra_rows:
        df_extra = pd.DataFrame(extra_rows)
        df_odds = pd.concat([df_odds, df_extra], ignore_index=True)
        print(f"  {year}: +{len(extra_rows)} doubleheader rows inherited")

    return df_odds


def pull_year(year: int, game_dates: list[str], dry_run: bool = False) -> pd.DataFrame:
    """Pull T23z for all dates in a year and return a DataFrame."""
    cached    = sum(1 for d in game_dates if (CACHE_DIR / f"{d}.json").exists())
    to_fetch  = len(game_dates) - cached
    est_cost  = to_fetch * 20

    print(f"  {year}: {len(game_dates)} dates | {cached} cached | "
          f"{to_fetch} to fetch | est. {est_cost:,} credits")

    if dry_run:
        return pd.DataFrame()

    all_games = []
    fetched = errors = 0
    for i, date_str in enumerate(sorted(game_dates)):
        try:
            games = _fetch_date_t23(date_str)
            all_games.extend(games)
            fetched += 1
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(game_dates)} dates processed ...")
        except urllib.error.HTTPError as e:
            errors += 1
            print(f"    HTTPError {e.code} on {date_str}: {e.read().decode()[:100]}")
        except Exception as e:
            errors += 1
            print(f"    Error on {date_str}: {e}")

    df = pd.DataFrame(all_games)
    if df.empty:
        return df

    df["game_date"] = pd.to_datetime(df["game_date"])

    df = _compute_de_vig(df)

    # Drop duplicates (keep last for T23z; doubleheaders handled later)
    df = df.drop_duplicates(subset=["game_date", "home_team", "away_team"], keep="last")

    out_path = DATA_DIR / f"odds_api_hist_{year}.parquet"
    df.to_parquet(out_path, engine="pyarrow", index=False)

    n_total   = df["close_total"].notna().sum()
    n_ml      = df["close_ml_home"].notna().sum()
    print(f"  {year}: {len(df)} games | "
          f"total={n_total} ({100*n_total/len(df):.1f}%) | "
          f"ml={n_ml} ({100*n_ml/len(df):.1f}%) | "
          f"errors={errors} -> {out_path.name}")

    return df


def regen_from_cache(years: list[int] | None = None) -> None:
    """
    Rebuild odds_api_hist_{year}.parquet files entirely from disk cache.

    Reads every cached {date}.json and {date}_14z.json, re-parses them
    (picking up any new columns like game_hour_et), recomputes de-vig
    probabilities, and saves the updated parquets.

    No API calls are made — uses only the local cache.
    """
    target_years = years or YEARS
    print("=" * 60)
    print("  regen_from_cache — rebuilding parquets from disk")
    print("=" * 60)

    for year in target_years:
        # Collect all T23z cache files for this year
        t23_games: list[dict] = []
        t14_games: list[dict] = []

        cache_files_23 = sorted(CACHE_DIR.glob(f"{year}-*.json"))
        # Exclude _14z variants
        cache_files_23 = [f for f in cache_files_23 if "_14z" not in f.name]
        cache_files_14 = sorted(CACHE_DIR.glob(f"{year}-*_14z.json"))

        print(f"\n  {year}: {len(cache_files_23)} T23z files, {len(cache_files_14)} T14z files")

        for cache_file in cache_files_23:
            date_str = cache_file.stem  # e.g. "2024-07-01"
            try:
                raw = json.loads(cache_file.read_text(encoding="utf-8"))
                events = raw.get("data", raw) if isinstance(raw, dict) else raw
                games = _parse_events(events, date_str, filter_to_date=False)
                t23_games.extend(games)
            except Exception as e:
                print(f"    Warning: could not parse {cache_file.name}: {e}")

        for cache_file in cache_files_14:
            # stem is e.g. "2024-07-01_14z"
            date_str = cache_file.stem.replace("_14z", "")
            try:
                raw = json.loads(cache_file.read_text(encoding="utf-8"))
                events = raw.get("data", raw) if isinstance(raw, dict) else raw
                games = _parse_events(events, date_str, filter_to_date=True)
                t14_games.extend(games)
            except Exception as e:
                print(f"    Warning: could not parse {cache_file.name}: {e}")

        if not t23_games and not t14_games:
            print(f"    No cache data found for {year} — skipping")
            continue

        t23_df = pd.DataFrame(t23_games)
        t14_df = pd.DataFrame(t14_games)

        # Union: T23z preferred; T14z fills gaps
        if not t23_df.empty and not t14_df.empty:
            t23_df["game_date"] = pd.to_datetime(t23_df["game_date"])
            t14_df["game_date"] = pd.to_datetime(t14_df["game_date"])
            t23_df = t23_df.drop_duplicates(subset=["game_date","home_team","away_team"], keep="last")
            t14_df = t14_df.drop_duplicates(subset=["game_date","home_team","away_team"], keep="last")
            combined = pd.concat([t23_df, t14_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["game_date","home_team","away_team"], keep="first")
        elif not t23_df.empty:
            t23_df["game_date"] = pd.to_datetime(t23_df["game_date"])
            combined = t23_df.drop_duplicates(subset=["game_date","home_team","away_team"], keep="last")
        else:
            t14_df["game_date"] = pd.to_datetime(t14_df["game_date"])
            combined = t14_df.drop_duplicates(subset=["game_date","home_team","away_team"], keep="last")

        combined = combined.sort_values(["game_date","home_team","away_team"]).reset_index(drop=True)
        combined = _compute_de_vig(combined)

        n_with_hour = combined["game_hour_et"].notna().sum() if "game_hour_et" in combined.columns else 0
        out_path = DATA_DIR / f"odds_api_hist_{year}.parquet"
        combined.to_parquet(out_path, engine="pyarrow", index=False)
        print(f"    {year}: {len(combined)} games saved | "
              f"game_hour_et coverage: {n_with_hour}/{len(combined)} "
              f"({100*n_with_hour/len(combined):.1f}%)")

    print("\n  regen_from_cache complete.")


def main():
    parser = argparse.ArgumentParser(description="Pull historical MLB odds from The Odds API")
    parser.add_argument("--year",       type=int, help="Single year to pull (default: all)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Estimate credits without pulling")
    parser.add_argument("--gaps-only",  action="store_true",
                        help="Skip T23z pass; only fetch T14z for gap dates")
    parser.add_argument("--dh-only",    action="store_true",
                        help="Only run doubleheader fill (no API calls)")
    parser.add_argument("--regen",      action="store_true",
                        help="Rebuild parquets from disk cache (no API calls)")
    args = parser.parse_args()

    # --regen: rebuild parquets from disk cache, no API calls needed
    if args.regen:
        years = [args.year] if args.year else None
        regen_from_cache(years)
        return

    if not API_KEY and not args.dry_run:
        print("ERROR: ODDS_API_KEY not set. Add to .env file.")
        return

    print("=" * 60)
    print("  pull_odds_history_api.py  -  Historical MLB Odds")
    print("=" * 60)

    # Load game dates from feature matrix
    fm = pd.read_parquet(DATA_DIR / ".." / ".." / "feature_matrix.parquet",
                         engine="pyarrow",
                         columns=["game_date", "home_team", "away_team", "year"])
    fm["game_date"] = pd.to_datetime(fm["game_date"])

    years = [args.year] if args.year else YEARS
    total_fetch = 0

    # ── Pass 1: T23z (standard closing lines) ────────────────────────────
    if not args.gaps_only and not args.dh_only:
        print("\n[Pass 1] T23:00Z closing lines")
        for yr in years:
            dates = sorted(
                fm[fm["year"] == yr]["game_date"].dt.strftime("%Y-%m-%d").unique()
            )
            pull_year(yr, dates, dry_run=args.dry_run)
            cached = sum(1 for d in dates if (CACHE_DIR / f"{d}.json").exists())
            total_fetch += max(0, len(dates) - cached)

    # ── Pass 2: T14z gap-date fills ──────────────────────────────────────
    if not args.dh_only:
        print("\n[Pass 2] T14:00Z gap fills (afternoon day games)")
        for yr in years:
            pull_gap_dates(yr, fm, dry_run=args.dry_run)

    # ── Pass 3: Doubleheader fill ─────────────────────────────────────────
    if not args.dry_run:
        print("\n[Pass 3] Doubleheader row inheritance")
        for yr in years:
            parquet_path = DATA_DIR / f"odds_api_hist_{yr}.parquet"
            if not parquet_path.exists():
                continue
            df_odds = pd.read_parquet(parquet_path)
            df_odds["game_date"] = pd.to_datetime(df_odds["game_date"])
            df_odds = _fill_doubleheader_odds(df_odds, fm, yr)
            df_odds = df_odds.sort_values(["game_date", "home_team", "away_team"])
            df_odds.to_parquet(parquet_path, engine="pyarrow", index=False)

    # ── Summary ──────────────────────────────────────────────────────────
    if not args.dry_run:
        print("\n[Summary] Final join rates against feature matrix")
        odds_all = []
        for yr in years:
            p = DATA_DIR / f"odds_api_hist_{yr}.parquet"
            if p.exists():
                odds_all.append(pd.read_parquet(p))
        if odds_all:
            odds = pd.concat(odds_all)
            odds["game_date"] = pd.to_datetime(odds["game_date"])
            merged = fm.merge(
                odds[["game_date", "home_team", "away_team", "close_ml_home"]],
                on=["game_date", "home_team", "away_team"],
                how="left"
            )
            joined = merged["close_ml_home"].notna().sum()
            print(f"  Overall join rate: {joined}/{len(merged)} = {100*joined/len(merged):.1f}%")
            for yr in years:
                yr_merged = merged[merged["year"] == yr]
                yr_joined = yr_merged["close_ml_home"].notna().sum()
                print(f"    {yr}: {yr_joined}/{len(yr_merged)} = "
                      f"{100*yr_joined/len(yr_merged):.1f}%")
    else:
        print(f"\n  Total to fetch: {total_fetch} dates | "
              f"Est. credits: {total_fetch * 20:,}")

    print("\n  Done.")
    if not args.dry_run:
        print("  Next: python build_feature_matrix.py && python train_xgboost.py --ncv")


if __name__ == "__main__":
    main()
