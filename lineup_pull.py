"""
lineup_pull.py
--------------
Pull confirmed MLB starting lineups from the MLB Stats API.

Run modes:
    python lineup_pull.py                                        # today's lineups
    python lineup_pull.py --date 2026-04-11                      # specific date
    python lineup_pull.py --historical --start 2024-04-01 --end 2024-09-29

Output (today / single date):
    statcast_data/lineups_today.parquet
    statcast_data/lineups_today.csv
    statcast_data/lineups_today_long.parquet

Output (historical):
    statcast_data/lineups_{year}.parquet
"""

import argparse
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("./data/statcast")
LINEUP_CACHE_DIR = OUTPUT_DIR / "lineup_cache"

MLB_SCHEDULE_BASE = "https://statsapi.mlb.com/api/v1/schedule"
MLB_LINESCORE_BASE = "https://statsapi.mlb.com/api/v1/game/{gamePk}/linescore"
MLB_PEOPLE_BASE   = "https://statsapi.mlb.com/api/v1/people/{personId}"

HANDEDNESS_CACHE_PATH = OUTPUT_DIR / "handedness_cache.json"

REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_REQUESTS = 0.2  # seconds — be polite to the free API

# ---------------------------------------------------------------------------
# Team ID → abbreviation mapping
# (matches score_models.py / fetch_data.py conventions)
# "AZ" for Arizona, "ATH" for Sacramento Athletics
# ---------------------------------------------------------------------------

TEAM_ID_TO_ABBR: dict[int, str] = {
    108: "LAA",
    109: "AZ",
    110: "BAL",
    111: "BOS",
    112: "CHC",
    113: "CIN",
    114: "CLE",
    115: "COL",
    116: "DET",
    117: "HOU",
    118: "KC",
    119: "LAD",
    120: "WSH",
    121: "NYM",
    133: "ATH",   # Sacramento Athletics (formerly OAK)
    134: "PIT",
    135: "SD",
    136: "SEA",
    137: "SF",
    138: "STL",
    139: "TB",
    140: "TEX",
    141: "TOR",
    142: "MIN",
    143: "PHI",
    144: "ATL",
    145: "CWS",
    146: "MIA",
    147: "NYY",
    158: "MIL",
}

TEAM_NAME_TO_ABBR: dict[str, str] = {
    "Arizona Diamondbacks": "AZ",
    "Atlanta Braves":       "ATL",
    "Baltimore Orioles":    "BAL",
    "Boston Red Sox":       "BOS",
    "Chicago Cubs":         "CHC",
    "Chicago White Sox":    "CWS",
    "Cincinnati Reds":      "CIN",
    "Cleveland Guardians":  "CLE",
    "Colorado Rockies":     "COL",
    "Detroit Tigers":       "DET",
    "Houston Astros":       "HOU",
    "Kansas City Royals":   "KC",
    "Los Angeles Angels":   "LAA",
    "Los Angeles Dodgers":  "LAD",
    "Miami Marlins":        "MIA",
    "Milwaukee Brewers":    "MIL",
    "Minnesota Twins":      "MIN",
    "New York Mets":        "NYM",
    "New York Yankees":     "NYY",
    "Athletics":            "ATH",
    "Oakland Athletics":    "ATH",
    "Philadelphia Phillies":"PHI",
    "Pittsburgh Pirates":   "PIT",
    "San Diego Padres":     "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners":     "SEA",
    "St. Louis Cardinals":  "STL",
    "Tampa Bay Rays":       "TB",
    "Texas Rangers":        "TEX",
    "Toronto Blue Jays":    "TOR",
    "Washington Nationals": "WSH",
}


def team_abbr(team_data: dict) -> str:
    """
    Resolve a team abbreviation from the MLB API team dict.
    Tries team ID first, then full name, then falls back to the
    API's own abbreviation field.
    """
    tid = team_data.get("id")
    if tid and tid in TEAM_ID_TO_ABBR:
        return TEAM_ID_TO_ABBR[tid]
    name = team_data.get("name", "")
    if name in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[name]
    # Last resort: use whatever abbreviation the API provided
    return team_data.get("abbreviation", team_data.get("teamCode", "???")).upper()


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LINEUP_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def daily_cache_path(date_str: str) -> Path:
    return LINEUP_CACHE_DIR / f"{date_str}.json"


def load_daily_cache(date_str: str) -> dict | None:
    p = daily_cache_path(date_str)
    if p.exists():
        try:
            with p.open() as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_daily_cache(date_str: str, data: dict) -> None:
    p = daily_cache_path(date_str)
    try:
        with p.open("w") as f:
            json.dump(data, f)
    except OSError as exc:
        print(f"  [WARN] Could not write lineup cache for {date_str}: {exc}")


# ---------------------------------------------------------------------------
# MLB Stats API fetchers
# ---------------------------------------------------------------------------

def fetch_schedule(date_str: str, use_cache: bool = True) -> dict | None:
    """
    Fetch the MLB schedule for a single date, hydrated with lineups and
    probable pitchers.  Returns the parsed JSON or None on failure.
    Falls back to cache on network errors.
    """
    if use_cache:
        cached = load_daily_cache(date_str)
        if cached is not None:
            return cached

    params = {
        "sportId": 1,
        "gameType": "R",
        "date": date_str,
        "hydrate": "lineups,probablePitcher,team",
    }
    try:
        resp = requests.get(MLB_SCHEDULE_BASE, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        save_daily_cache(date_str, data)
        return data
    except requests.RequestException as exc:
        print(f"  [ERROR] MLB schedule fetch failed for {date_str}: {exc}")
        # Try stale cache as fallback
        if use_cache:
            stale = load_daily_cache(date_str)
            if stale:
                print(f"  [INFO] Using stale cache for {date_str}")
                return stale
        return None


# ---------------------------------------------------------------------------
# Lineup extraction
# ---------------------------------------------------------------------------

def extract_lineup_slots(lineup_list: list) -> list[dict]:
    """
    Parse the 'lineups' array from the MLB API into a list of dicts with:
        batting_order, player_id, player_name, position

    The MLB Stats API hydrate=lineups returns player objects directly (not
    nested under a 'player' key), with position under 'primaryPosition'.
    battingOrder is not included — order is implied by list position (1-indexed).
    """
    slots = []
    if not isinstance(lineup_list, list):
        return slots
    for idx, item in enumerate(lineup_list, start=1):
        try:
            # API returns player fields at top level: id, fullName, primaryPosition
            player_id   = item.get("id") or item.get("player", {}).get("id")
            player_name = (item.get("fullName") or item.get("player", {}).get("fullName", ""))
            # battingOrder may be present in live/boxscore; for schedule hydrate it's the list index
            batting_order = item.get("battingOrder") or idx
            # Position: primaryPosition (schedule hydrate) or position (boxscore)
            pos_obj  = item.get("primaryPosition") or item.get("position") or {}
            position = pos_obj.get("abbreviation", "")
            slots.append({
                "batting_order": batting_order,
                "player_id":     player_id,
                "player_name":   player_name,
                "position":      position,
            })
        except (AttributeError, KeyError):
            continue
    # Sort by batting order (should already be ordered, but be safe)
    slots.sort(key=lambda x: (x["batting_order"] is None, x["batting_order"]))
    return slots


def parse_game(game: dict) -> dict:
    """
    Extract all relevant fields from a single game dict returned by the
    schedule endpoint.

    Returns a dict representing one row of the 'wide' output format.
    """
    game_pk   = game.get("gamePk")
    game_date = game.get("officialDate", game.get("gameDate", "")[:10])
    status    = game.get("status", {}).get("abstractGameState", "")

    # Game time — parse UTC ISO string and convert to ET
    game_time_et = ""
    raw_dt = game.get("gameDate", "")
    start_time_tbd = game.get("status", {}).get("startTimeTBD", False)
    if raw_dt and not start_time_tbd:
        try:
            from datetime import datetime, timezone, timedelta
            utc_dt = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
            et_offset = timedelta(hours=-4)   # EDT (UTC-4); adjust to -5 for EST in Nov
            et_dt = utc_dt + et_offset
            # Cross-platform hour formatting (no leading zero)
            hour = et_dt.strftime("%I").lstrip("0") or "12"
            minute = et_dt.strftime("%M")
            ampm = et_dt.strftime("%p").lower()
            game_time_et = f"{hour}:{minute} {ampm} ET"
        except Exception:
            game_time_et = ""

    teams = game.get("teams", {})
    home_raw = teams.get("home", {})
    away_raw = teams.get("away", {})

    home_team_data = home_raw.get("team", {})
    away_team_data = away_raw.get("team", {})

    home_abbr = team_abbr(home_team_data)
    away_abbr = team_abbr(away_team_data)

    # Probable pitchers
    def probable(side_raw: dict) -> tuple[int | None, str]:
        pp = side_raw.get("probablePitcher", {})
        if pp:
            return pp.get("id"), pp.get("fullName", "")
        return None, ""

    home_sp_id, home_sp_name = probable(home_raw)
    away_sp_id, away_sp_name = probable(away_raw)

    # Lineups
    lineups_data = game.get("lineups", {})
    home_lineup_raw = lineups_data.get("homePlayers", [])
    away_lineup_raw = lineups_data.get("awayPlayers", [])

    home_slots = extract_lineup_slots(home_lineup_raw)
    away_slots  = extract_lineup_slots(away_lineup_raw)

    home_confirmed = len(home_slots) >= 8
    away_confirmed  = len(away_slots)  >= 8

    # Build JSON-serialisable batting order lists (for wide format)
    def slots_to_order(slots: list[dict]) -> str:
        return json.dumps([
            {
                "order":    s["batting_order"],
                "id":       s["player_id"],
                "name":     s["player_name"],
                "position": s["position"],
            }
            for s in slots
        ])

    return {
        "game_date":               game_date,
        "game_pk":                 game_pk,
        "game_status":             status,
        "game_time_et":            game_time_et,
        "home_team":               home_abbr,
        "away_team":               away_abbr,
        "home_starter_id":         home_sp_id,
        "home_starter_name":       home_sp_name,
        "away_starter_id":         away_sp_id,
        "away_starter_name":       away_sp_name,
        "home_lineup_confirmed":   home_confirmed,
        "away_lineup_confirmed":    away_confirmed,
        "home_batting_order":      slots_to_order(home_slots),
        "away_batting_order":       slots_to_order(away_slots),
        # Store slot lists for long-format expansion
        "_home_slots":             home_slots,
        "_away_slots":              away_slots,
    }


def parse_schedule_response(data: dict) -> list[dict]:
    """
    Iterate over all dates / games in a schedule API response and
    return a list of parsed game dicts.
    """
    records = []
    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            try:
                records.append(parse_game(game))
            except Exception as exc:
                pk = game.get("gamePk", "?")
                print(f"  [WARN] Could not parse gamePk={pk}: {exc}")
    return records


# ---------------------------------------------------------------------------
# Wide and Long DataFrames
# ---------------------------------------------------------------------------

WIDE_COLUMNS = [
    "game_date", "game_pk", "game_status", "game_time_et",
    "home_team", "away_team",
    "home_starter_id", "home_starter_name",
    "away_starter_id", "away_starter_name",
    "home_lineup_confirmed", "away_lineup_confirmed",
    "home_batting_order", "away_batting_order",
]


def records_to_wide(records: list[dict]) -> pd.DataFrame:
    """Build the wide DataFrame (one row per game) from parsed records."""
    rows = []
    for r in records:
        row = {k: r[k] for k in WIDE_COLUMNS if k in r}
        rows.append(row)
    df = pd.DataFrame(rows, columns=WIDE_COLUMNS)
    return df


# ---------------------------------------------------------------------------
# Handedness enrichment (MLB /people/{id} -> batSide.code)
# ---------------------------------------------------------------------------

def _load_handedness_cache() -> dict[str, str]:
    if HANDEDNESS_CACHE_PATH.exists():
        try:
            return json.loads(HANDEDNESS_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_handedness_cache(cache: dict[str, str]) -> None:
    try:
        HANDEDNESS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        HANDEDNESS_CACHE_PATH.write_text(json.dumps(cache, sort_keys=True),
                                          encoding="utf-8")
    except Exception as exc:
        print(f"[handedness] cache save failed: {exc}")


def fetch_handedness(player_id: int,
                      cache: dict[str, str] | None = None) -> str:
    """Return bat-side code: 'L' | 'R' | 'S' | '' (unknown).

    Hits MLB Stats API /people/{id} for a single player and parses
    batSide.code. Uses the provided cache dict in-memory; caller is
    responsible for persisting after the batch.
    """
    key = str(int(player_id))
    if cache is not None and key in cache:
        return cache[key]
    url = MLB_PEOPLE_BASE.format(personId=key)
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        people = data.get("people") or []
        code = ""
        if people:
            code = (people[0].get("batSide") or {}).get("code", "") or ""
        if cache is not None:
            cache[key] = code
        return code
    except Exception as exc:
        print(f"[handedness] fetch failed for {player_id}: {exc}")
        if cache is not None:
            cache.setdefault(key, "")
        return ""


def enrich_long_with_handedness(long_df: pd.DataFrame) -> pd.DataFrame:
    """Attach a 'stand' column to a lineups long DataFrame.

    Uses a persistent JSON cache to avoid re-hitting /people/{id} for
    previously-seen players (handedness is effectively static).
    """
    if long_df.empty or "player_id" not in long_df.columns:
        long_df["stand"] = ""
        return long_df

    cache = _load_handedness_cache()
    unique_ids = sorted({int(p) for p in long_df["player_id"].dropna().unique()})
    missing = [pid for pid in unique_ids if str(pid) not in cache]
    print(f"[handedness] {len(unique_ids)} unique players; "
          f"{len(missing)} need API fetch.")
    for pid in missing:
        fetch_handedness(pid, cache=cache)
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    _save_handedness_cache(cache)

    long_df = long_df.copy()
    long_df["stand"] = long_df["player_id"].astype("Int64").map(
        lambda p: cache.get(str(int(p)), "") if pd.notna(p) else ""
    )
    return long_df


def records_to_long(records: list[dict]) -> pd.DataFrame:
    """Build the long DataFrame (one row per lineup slot)."""
    long_rows = []
    for r in records:
        base = {
            "game_date": r["game_date"],
            "game_pk":   r["game_pk"],
        }
        for side, slot_key, abbr_key in [
            ("home", "_home_slots", "home_team"),
            ("away",  "_away_slots",  "away_team"),
        ]:
            team = r.get(abbr_key, "")
            for slot in r.get(slot_key, []):
                long_rows.append({
                    **base,
                    "side":          side,
                    "team":          team,
                    "batting_order": slot["batting_order"],
                    "player_id":     slot["player_id"],
                    "player_name":   slot["player_name"],
                    "position":      slot["position"],
                })
    long_cols = [
        "game_date", "game_pk", "side", "team",
        "batting_order", "player_id", "player_name", "position",
    ]
    df = pd.DataFrame(long_rows, columns=long_cols) if long_rows else pd.DataFrame(columns=long_cols)
    # Enrich with handedness (cached; hits MLB /people/{id} only for new ids).
    df = enrich_long_with_handedness(df)
    return df


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_today_outputs(wide_df: pd.DataFrame, long_df: pd.DataFrame,
                       date_str: str = None) -> None:
    """Save wide and long DataFrames to output files.

    Always writes the canonical 'today' files for backward compatibility.
    When date_str is provided, also writes date-stamped files so multi-day
    pipelines (today + tomorrow) can coexist without overwriting each other.
    """
    import datetime as _dt
    today_str = _dt.date.today().isoformat()
    is_today  = (date_str is None) or (date_str == today_str)

    # Canonical 'today' files — always written (or only when it IS today)
    if is_today:
        wide_parquet = OUTPUT_DIR / "lineups_today.parquet"
        wide_csv     = OUTPUT_DIR / "lineups_today.csv"
        long_parquet = OUTPUT_DIR / "lineups_today_long.parquet"

        try:
            wide_df.to_parquet(wide_parquet, engine="pyarrow", index=False)
            print(f"Saved {wide_parquet}  shape={wide_df.shape}")
        except Exception as exc:
            print(f"[ERROR] Could not save {wide_parquet}: {exc}")

        try:
            wide_df.to_csv(wide_csv, index=False)
            print(f"Saved {wide_csv}  shape={wide_df.shape}")
        except Exception as exc:
            print(f"[ERROR] Could not save {wide_csv}: {exc}")

        try:
            long_df.to_parquet(long_parquet, engine="pyarrow", index=False)
            print(f"Saved {long_parquet}  shape={long_df.shape}")
        except Exception as exc:
            print(f"[ERROR] Could not save {long_parquet}: {exc}")

    # Date-stamped files — always written when date_str is provided
    if date_str:
        dated_wide = OUTPUT_DIR / f"lineups_{date_str}.parquet"
        dated_long = OUTPUT_DIR / f"lineups_{date_str}_long.parquet"
        try:
            wide_df.to_parquet(dated_wide, engine="pyarrow", index=False)
            print(f"Saved {dated_wide}  shape={wide_df.shape}")
        except Exception as exc:
            print(f"[ERROR] Could not save {dated_wide}: {exc}")
        try:
            long_df.to_parquet(dated_long, engine="pyarrow", index=False)
            print(f"Saved {dated_long}  shape={long_df.shape}")
        except Exception as exc:
            print(f"[ERROR] Could not save {dated_long}: {exc}")


def save_historical_year(year: int, records: list[dict]) -> None:
    """Append / merge records into the annual parquet file."""
    path = OUTPUT_DIR / f"lineups_{year}.parquet"
    wide_df = records_to_wide(records)

    # If file already exists, merge (deduplicate on game_pk)
    if path.exists():
        try:
            existing = pd.read_parquet(path, engine="pyarrow")
            combined = (
                pd.concat([existing, wide_df], ignore_index=True)
                .drop_duplicates(subset=["game_pk"], keep="last")
                .sort_values(["game_date", "game_pk"])
                .reset_index(drop=True)
            )
        except Exception as exc:
            print(f"  [WARN] Could not merge with existing {path}: {exc}")
            combined = wide_df
    else:
        combined = wide_df

    try:
        combined.to_parquet(path, engine="pyarrow", index=False)
        print(f"  Saved {path}  shape={combined.shape}")
    except Exception as exc:
        print(f"  [ERROR] Could not save {path}: {exc}")


# ---------------------------------------------------------------------------
# Boxscore lineup fetcher (completed games)
# ---------------------------------------------------------------------------

def fetch_boxscore_lineups(game_pk: int, game_date: str,
                            home_abbr: str, away_abbr: str,
                            game_time_et: str = "",
                            verbose: bool = False) -> dict | None:
    """
    Fetch actual batting orders from the boxscore endpoint for a completed game.
    Returns a record dict compatible with parse_game() output, or None on failure.

    The boxscore endpoint has the real confirmed batting orders for any game
    that has started or completed — unlike the schedule/lineups hydration which
    only works pre-game.
    """
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        if verbose:
            print(f"  [WARN] Boxscore fetch failed for game {game_pk}: {exc}")
        return None

    teams = data.get("teams", {})

    def extract_side(side_key: str) -> tuple[list[dict], str, str]:
        """Return (slots, starter_name, starter_id) for home or away."""
        side = teams.get(side_key, {})
        batters = side.get("batters", [])
        players = side.get("players", {})
        pitchers = side.get("pitchers", [])

        slots = []
        for order_idx, pid in enumerate(batters, start=1):
            player = players.get(f"ID{pid}", {})
            name = player.get("person", {}).get("fullName", "")
            pos_obj = player.get("position", {})
            pos = pos_obj.get("abbreviation", "")
            batting_order = player.get("battingOrder", order_idx * 100)
            # battingOrder from boxscore is like 100,200,...900 (multiply of 100)
            real_order = int(batting_order) // 100 if batting_order else order_idx
            slots.append({
                "batting_order": real_order,
                "player_id":     pid,
                "player_name":   name,
                "position":      pos,
            })

        # Starting pitcher = first pitcher listed
        sp_id, sp_name = None, ""
        if pitchers:
            sp_pid = pitchers[0]
            sp_player = players.get(f"ID{sp_pid}", {})
            sp_name = sp_player.get("person", {}).get("fullName", "")
            sp_id = sp_pid

        return slots, sp_name, sp_id

    home_slots, home_sp_name, home_sp_id = extract_side("home")
    away_slots, away_sp_name, away_sp_id = extract_side("away")

    home_confirmed = len(home_slots) >= 8
    away_confirmed  = len(away_slots) >= 8

    def slots_to_order(slots):
        return json.dumps([{
            "order":    s["batting_order"],
            "id":       s["player_id"],
            "name":     s["player_name"],
            "position": s["position"],
        } for s in slots])

    return {
        "game_date":             game_date,
        "game_pk":               game_pk,
        "game_status":           "Final",
        "game_time_et":          game_time_et,
        "home_team":             home_abbr,
        "away_team":             away_abbr,
        "home_starter_id":       home_sp_id,
        "home_starter_name":     home_sp_name,
        "away_starter_id":       away_sp_id,
        "away_starter_name":     away_sp_name,
        "home_lineup_confirmed": home_confirmed,
        "away_lineup_confirmed": away_confirmed,
        "home_batting_order":    slots_to_order(home_slots),
        "away_batting_order":    slots_to_order(away_slots),
        "_home_slots":           home_slots,
        "_away_slots":           away_slots,
    }


def pull_date_with_boxscore(date_str: str, verbose: bool = True) -> list[dict]:
    """
    Pull lineups for date_str. For completed games, enriches with actual
    batting orders from the boxscore endpoint. For future/live games, falls
    back to the schedule hydration (probable pitchers only).
    """
    if verbose:
        print(f"Fetching lineups for {date_str} ...")

    data = fetch_schedule(date_str, use_cache=False)
    if data is None:
        print(f"  [ERROR] No schedule data returned for {date_str}.")
        return []

    records = parse_schedule_response(data)
    if not records:
        return []

    enriched = []
    for r in records:
        status = r.get("game_status", "")
        game_pk = r.get("game_pk")

        # For completed or in-progress games, fetch real lineups from boxscore
        if status in ("Final", "Live", "In Progress") and game_pk:
            if verbose:
                print(f"  Fetching boxscore for {r['away_team']} @ {r['home_team']} (pk={game_pk})")
            bs = fetch_boxscore_lineups(
                game_pk=game_pk,
                game_date=date_str,
                home_abbr=r["home_team"],
                away_abbr=r["away_team"],
                game_time_et=r.get("game_time_et", ""),
                verbose=verbose,
            )
            if bs:
                enriched.append(bs)
                time.sleep(SLEEP_BETWEEN_REQUESTS)
                continue

        # Future/preview games — use schedule data as-is
        enriched.append(r)

    if verbose:
        confirmed = sum(1 for r in enriched if r.get("home_lineup_confirmed") or r.get("away_lineup_confirmed"))
        print(f"  {len(enriched)} games  |  {confirmed} with confirmed lineups")

    return enriched


# ---------------------------------------------------------------------------
# Single-date pull
# ---------------------------------------------------------------------------

def pull_date(date_str: str, verbose: bool = True) -> list[dict]:
    """
    Fetch and parse all games for `date_str`.
    Returns a list of parsed game dicts.
    """
    if verbose:
        print(f"Fetching lineups for {date_str} ...")

    data = fetch_schedule(date_str)
    if data is None:
        print(f"  [ERROR] No data returned for {date_str}.")
        return []

    records = parse_schedule_response(data)

    if verbose:
        total      = len(records)
        confirmed  = sum(1 for r in records if r["home_lineup_confirmed"] or r["away_lineup_confirmed"])
        pitchers   = sum(1 for r in records if r["home_starter_id"] or r["away_starter_id"])
        print(f"  {total} games  |  {confirmed} with lineup data  |  {pitchers} with probable pitcher")

    return records


# ---------------------------------------------------------------------------
# Historical pull
# ---------------------------------------------------------------------------

def dates_already_cached(start: date, end: date) -> set[str]:
    """Return set of date strings that already have a lineup cache file."""
    cached = set()
    cur = start
    while cur <= end:
        ds = cur.strftime("%Y-%m-%d")
        if daily_cache_path(ds).exists():
            cached.add(ds)
        cur += timedelta(days=1)
    return cached


def pull_historical_lineups(start_date: str, end_date: str) -> None:
    """
    Loop over [start_date, end_date] (inclusive), pull each day's schedule
    with lineups, and save annual parquet files.

    Already-cached dates are skipped unless the cache is stale.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end   = datetime.strptime(end_date,   "%Y-%m-%d").date()

    if start > end:
        print("[ERROR] start_date must be <= end_date.")
        return

    already_cached = dates_already_cached(start, end)

    cur = start
    # Accumulate records by year
    year_records: dict[int, list[dict]] = {}

    total_days = (end - start).days + 1
    day_num = 0

    while cur <= end:
        day_num += 1
        date_str = cur.strftime("%Y-%m-%d")
        year     = cur.year

        if date_str in already_cached:
            print(f"  [{day_num}/{total_days}] {date_str} — using cache")
            # Still need to parse it for the parquet
            data = load_daily_cache(date_str)
            if data:
                records = parse_schedule_response(data)
                year_records.setdefault(year, []).extend(records)
        else:
            print(f"  [{day_num}/{total_days}] {date_str} — fetching ...")
            records = pull_date(date_str, verbose=False)
            n_games = len(records)
            n_conf  = sum(1 for r in records if r["home_lineup_confirmed"] or r["away_lineup_confirmed"])
            print(f"    {n_games} games, {n_conf} with lineup data")
            year_records.setdefault(year, []).extend(records)
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        # Flush to disk every time the year changes or on the last date
        next_date  = cur + timedelta(days=1)
        next_year  = next_date.year if cur < end else None

        if cur == end or (next_year is not None and next_year != year):
            if year in year_records and year_records[year]:
                print(f"\nSaving lineups_{year}.parquet ...")
                save_historical_year(year, year_records[year])
                del year_records[year]

        cur = next_date

    # Flush any remaining years
    for year, records in year_records.items():
        if records:
            print(f"\nSaving lineups_{year}.parquet ...")
            save_historical_year(year, records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull MLB starting lineups from the MLB Stats API."
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--date", type=str, metavar="YYYY-MM-DD",
        help="Pull lineups for a specific date (default: today).",
    )
    mode.add_argument(
        "--historical", action="store_true",
        help="Backfill historical lineups over a date range.",
    )
    mode.add_argument(
        "--recent", action="store_true",
        help="Pull lineups for today + last 72 hours using boxscore data for completed games.",
    )

    # Historical options
    parser.add_argument(
        "--start", type=str, metavar="YYYY-MM-DD",
        help="Start date for historical pull.",
    )
    parser.add_argument(
        "--end", type=str, metavar="YYYY-MM-DD",
        help="End date for historical pull.",
    )

    return parser.parse_args()


def main() -> None:
    ensure_dirs()
    args = parse_args()

    if args.historical:
        # --- Historical backfill ---
        if not args.start or not args.end:
            print("[ERROR] --historical requires --start and --end dates.")
            return
        try:
            datetime.strptime(args.start, "%Y-%m-%d")
            datetime.strptime(args.end,   "%Y-%m-%d")
        except ValueError as exc:
            print(f"[ERROR] Invalid date format: {exc}")
            return

        print(f"=== Historical lineup pull: {args.start} → {args.end} ===")
        pull_historical_lineups(args.start, args.end)
        return

    if args.recent:
        # --- Last 72 hours + today ---
        today_d = date.today()
        dates = [(today_d - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(3, -1, -1)]
        print(f"=== Pulling lineups for last 72h + today: {dates[0]} to {dates[-1]} ===")
        for date_str in dates:
            print(f"\n--- {date_str} ---")
            records = pull_date_with_boxscore(date_str, verbose=True)
            if not records:
                print(f"  No games found for {date_str}.")
                continue
            wide_df = records_to_wide(records)
            long_df = records_to_long(records)
            save_today_outputs(wide_df, long_df, date_str=date_str)
            # Also update canonical today files
            if date_str == today_d.strftime("%Y-%m-%d"):
                save_today_outputs(wide_df, long_df, date_str=None)
        return

    # --- Single-date or today ---
    if args.date:
        try:
            datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            print(f"[ERROR] --date must be YYYY-MM-DD, got '{args.date}'")
            return
        date_str = args.date
    else:
        date_str = date.today().strftime("%Y-%m-%d")

    print(f"=== Pulling lineups for {date_str} ===")
    records = pull_date_with_boxscore(date_str, verbose=True)

    if not records:
        print("No games found. Nothing to save.")
        return

    wide_df = records_to_wide(records)
    long_df = records_to_long(records)

    # Print summary
    print("\n--- Wide format preview ---")
    preview_cols = [
        "game_date", "game_pk", "home_team", "away_team",
        "home_starter_name", "away_starter_name",
        "home_lineup_confirmed", "away_lineup_confirmed",
    ]
    preview = wide_df[[c for c in preview_cols if c in wide_df.columns]]
    print(preview.to_string(index=False))

    print()
    save_today_outputs(wide_df, long_df, date_str=date_str)

    # Print lineup details for confirmed games
    confirmed = [r for r in records if r["home_lineup_confirmed"] or r["away_lineup_confirmed"]]
    if confirmed:
        print(f"\n--- Confirmed lineup details ({len(confirmed)} games) ---")
        for r in confirmed:
            print(f"\n  {r['away_team']} @ {r['home_team']}  (gamePk={r['game_pk']})")
            if r["away_starter_name"]:
                print(f"    Away SP: {r['away_starter_name']} (id={r['away_starter_id']})")
            if r["home_starter_name"]:
                print(f"    Home SP: {r['home_starter_name']} (id={r['home_starter_id']})")

            if r["away_lineup_confirmed"]:
                print(f"    Away lineup ({r['away_team']}):")
                for slot in r.get("_away_slots", []):
                    order = slot['batting_order'] if slot['batting_order'] is not None else "?"
                    print(f"      {str(order):>3}. {slot['player_name']} ({slot['position']})")
            else:
                print(f"    Away lineup ({r['away_team']}): not yet posted")

            if r["home_lineup_confirmed"]:
                print(f"    Home lineup ({r['home_team']}):")
                for slot in r.get("_home_slots", []):
                    order = slot['batting_order'] if slot['batting_order'] is not None else "?"
                    print(f"      {str(order):>3}. {slot['player_name']} ({slot['position']})")
            else:
                print(f"    Home lineup ({r['home_team']}): not yet posted")
    else:
        print("\nNo confirmed lineups yet (lineups are typically posted 1-3 hours before first pitch).")
        print("Probable pitchers are available above if the API returned them.")


if __name__ == "__main__":
    main()
