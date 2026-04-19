"""
odds_current_pull.py
====================
Daily odds pull implementing a dual-region ingestion strategy:

  REGION 1 — US  (DraftKings, FanDuel, BetMGM)
    Retail reference lines.  De-vigged to produce Retail_Implied_Prob.

  REGION 2 — EU  (Pinnacle only)
    Sharp benchmark.  De-vigged to produce P_true — the structural True Market
    Probability baseline fed into the Three-Part Lock execution gate.

  FALLBACK WATERFALL (for historical / offline dates):
    1. OddsPortal historical cache (read-only parquets from odds_historical_pull.py)
    2. The Odds API US region (quota-guarded, current season)
    3. ActionNetwork (scraper fallback — sole source for public betting %)

Usage:
    python odds_current_pull.py                   # pull today's odds
    python odds_current_pull.py --date 2026-04-11 # specific date

Requires:
    pip install requests beautifulsoup4 pandas pyarrow python-dotenv

Environment variables (in .env or system):
    ODDS_API_KEY=<premium key from the-odds-api.com>
"""

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("./data/statcast")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ODDS_API_CACHE_DIR = OUTPUT_DIR / "odds_api_cache"
ODDS_API_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ODDS_API_KEY: Optional[str] = os.getenv("ODDS_API_KEY")
ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
ODDS_API_QUOTA_THRESHOLD = 50
ODDS_API_CACHE_TTL_HOURS = 2          # re-fetch if cache is older than this

ACTIONNETWORK_URL = "https://www.actionnetwork.com/mlb/odds"

QUOTA_LOG   = OUTPUT_DIR / "odds_api_quota.log"
MISSING_LOG = OUTPUT_DIR / "odds_missing.log"

HISTORICAL_YEARS = [2023, 2024, 2025]

# US retail books (in preference order for line selection)
ODDS_API_US_BOOKS = ["draftkings", "fanduel", "betmgm"]
# EU sharp books (Pinnacle is the structural sharp benchmark)
ODDS_API_EU_BOOKS = ["pinnacle"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "odds_current_pull.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Team name normalisation
# ---------------------------------------------------------------------------
TEAM_ABBR_MAP: dict[str, str] = {
    "arizona diamondbacks": "AZ", "diamondbacks": "AZ", "arizona": "AZ", "ari": "AZ",
    "atlanta braves": "ATL", "braves": "ATL", "atlanta": "ATL",
    "baltimore orioles": "BAL", "orioles": "BAL", "baltimore": "BAL",
    "boston red sox": "BOS", "red sox": "BOS", "boston": "BOS",
    "chicago cubs": "CHC", "cubs": "CHC",
    "chicago white sox": "CWS", "white sox": "CWS",
    "cincinnati reds": "CIN", "reds": "CIN", "cincinnati": "CIN",
    "cleveland guardians": "CLE", "guardians": "CLE", "cleveland": "CLE",
    "cleveland indians": "CLE",
    "colorado rockies": "COL", "rockies": "COL", "colorado": "COL",
    "detroit tigers": "DET", "tigers": "DET", "detroit": "DET",
    "houston astros": "HOU", "astros": "HOU", "houston": "HOU",
    "kansas city royals": "KC", "royals": "KC", "kansas city": "KC",
    "los angeles angels": "LAA", "angels": "LAA", "la angels": "LAA",
    "los angeles dodgers": "LAD", "dodgers": "LAD", "la dodgers": "LAD",
    "miami marlins": "MIA", "marlins": "MIA", "miami": "MIA",
    "milwaukee brewers": "MIL", "brewers": "MIL", "milwaukee": "MIL",
    "minnesota twins": "MIN", "twins": "MIN", "minnesota": "MIN",
    "new york mets": "NYM", "mets": "NYM", "ny mets": "NYM",
    "new york yankees": "NYY", "yankees": "NYY", "ny yankees": "NYY",
    "athletics": "ATH", "oakland athletics": "ATH", "oakland": "ATH",
    "las vegas athletics": "ATH", "oak": "ATH",
    "philadelphia phillies": "PHI", "phillies": "PHI", "philadelphia": "PHI",
    "pittsburgh pirates": "PIT", "pirates": "PIT", "pittsburgh": "PIT",
    "san diego padres": "SD", "padres": "SD", "san diego": "SD",
    "san francisco giants": "SF", "giants": "SF", "san francisco": "SF",
    "seattle mariners": "SEA", "mariners": "SEA", "seattle": "SEA",
    "st. louis cardinals": "STL", "st louis cardinals": "STL", "cardinals": "STL",
    "st. louis": "STL", "st louis": "STL",
    "tampa bay rays": "TB", "rays": "TB", "tampa bay": "TB",
    "texas rangers": "TEX", "rangers": "TEX", "texas": "TEX",
    "toronto blue jays": "TOR", "blue jays": "TOR", "toronto": "TOR",
    "washington nationals": "WSH", "nationals": "WSH", "washington": "WSH",
}


def normalize_team(raw: str) -> str:
    cleaned = raw.strip().lower()
    if cleaned in TEAM_ABBR_MAP:
        return TEAM_ABBR_MAP[cleaned]
    for key, abbr in sorted(TEAM_ABBR_MAP.items(), key=lambda x: -len(x[0])):
        if key in cleaned:
            return abbr
    log.warning("Unknown team name: %r", raw)
    return raw.upper()[:3]


# ---------------------------------------------------------------------------
# Output schema — base retail columns + Pinnacle/P_true extensions
# ---------------------------------------------------------------------------
SCHEMA_COLS = [
    "game_date", "game_pk", "home_team", "away_team",
    # Retail (US) moneyline / total / runline
    "open_ml_home", "close_ml_home", "open_ml_away", "close_ml_away",
    "open_total", "close_total", "runline_home", "runline_home_odds",
    # Public-betting metadata (ActionNetwork only)
    "public_pct_home", "public_pct_over",
    # Pinnacle (EU sharp) raw lines
    "pinnacle_ml_home", "pinnacle_ml_away",
    "pinnacle_total_line", "pinnacle_total_over_odds", "pinnacle_total_under_odds",
    "pinnacle_rl_home", "pinnacle_rl_home_odds", "pinnacle_rl_away_odds",
    # De-vigged True Market Probabilities  (from Pinnacle)
    "P_true_home", "P_true_away",           # moneyline
    "P_true_over", "P_true_under",           # totals
    "P_true_rl_home", "P_true_rl_away",     # run-line
    # De-vigged Retail Implied Probabilities (from best US book)
    "retail_implied_home", "retail_implied_away",   # moneyline
    "retail_implied_over", "retail_implied_under",   # totals
    "retail_implied_rl_home", "retail_implied_rl_away",  # run-line
    # Best retail book that provided the retail line used above
    "retail_book_used",
    "source", "pull_timestamp",
    # Game metadata
    "game_hour_et",    # scheduled start hour in Eastern Time (float, e.g. 13.75 = 1:45 PM)
]


def _empty_odds_dict(game_date: str, home_team: str, away_team: str) -> dict:
    """Return a blank odds dict with all nullable fields as None."""
    return {col: None for col in SCHEMA_COLS} | {
        "game_date":    game_date,
        "home_team":    home_team,
        "away_team":    away_team,
        "runline_home": -1.5,
    }


# ---------------------------------------------------------------------------
# De-vigging engine
# ---------------------------------------------------------------------------

def american_to_decimal(american: int) -> float:
    """Convert American odds to decimal (European) odds."""
    if american > 0:
        return (american / 100.0) + 1.0
    return (100.0 / abs(american)) + 1.0


def devigify_two_way(price_a: int, price_b: int) -> tuple[float, float]:
    """
    Multiplicative de-vig for a symmetric two-way market.

    Removes the bookmaker's overround proportionally from both sides so that
    the resulting probabilities sum to exactly 1.0.

    Parameters
    ----------
    price_a, price_b : American odds for side A and side B.

    Returns
    -------
    (fair_prob_a, fair_prob_b) — de-vigged win probabilities.
    """
    dec_a = american_to_decimal(price_a)
    dec_b = american_to_decimal(price_b)
    imp_a = 1.0 / dec_a
    imp_b = 1.0 / dec_b
    total = imp_a + imp_b
    if total <= 0:
        return 0.5, 0.5
    return imp_a / total, imp_b / total


def compute_p_true(
    ml_home: Optional[int],
    ml_away: Optional[int],
    total_over_odds: Optional[int] = None,
    total_under_odds: Optional[int] = None,
    rl_home_odds: Optional[int] = None,
    rl_away_odds: Optional[int] = None,
) -> dict:
    """
    Compute all de-vigged True Market Probabilities from Pinnacle lines.

    Returns a dict with keys:
        P_true_home, P_true_away
        P_true_over, P_true_under
        P_true_rl_home, P_true_rl_away
    Any market with missing odds returns None for that pair.
    """
    result = {
        "P_true_home": None, "P_true_away": None,
        "P_true_over": None, "P_true_under": None,
        "P_true_rl_home": None, "P_true_rl_away": None,
    }

    if ml_home is not None and ml_away is not None:
        try:
            h, a = devigify_two_way(int(ml_home), int(ml_away))
            result["P_true_home"] = round(h, 6)
            result["P_true_away"] = round(a, 6)
        except Exception as exc:
            log.debug("P_true ML devigify failed: %s", exc)

    if total_over_odds is not None and total_under_odds is not None:
        try:
            o, u = devigify_two_way(int(total_over_odds), int(total_under_odds))
            result["P_true_over"]  = round(o, 6)
            result["P_true_under"] = round(u, 6)
        except Exception as exc:
            log.debug("P_true total devigify failed: %s", exc)

    if rl_home_odds is not None and rl_away_odds is not None:
        try:
            rh, ra = devigify_two_way(int(rl_home_odds), int(rl_away_odds))
            result["P_true_rl_home"] = round(rh, 6)
            result["P_true_rl_away"] = round(ra, 6)
        except Exception as exc:
            log.debug("P_true RL devigify failed: %s", exc)

    return result


def compute_retail_implied(
    ml_home: Optional[int],
    ml_away: Optional[int],
    total_over_odds: Optional[int] = None,
    total_under_odds: Optional[int] = None,
    rl_home_odds: Optional[int] = None,
    rl_away_odds: Optional[int] = None,
) -> dict:
    """
    Compute de-vigged Retail Implied Probabilities from the best US retail book.
    Identical algorithm to compute_p_true — multiplicative de-vig.

    Returns a dict with keys:
        retail_implied_home, retail_implied_away
        retail_implied_over, retail_implied_under
        retail_implied_rl_home, retail_implied_rl_away
    """
    result = {
        "retail_implied_home": None, "retail_implied_away": None,
        "retail_implied_over": None, "retail_implied_under": None,
        "retail_implied_rl_home": None, "retail_implied_rl_away": None,
    }

    if ml_home is not None and ml_away is not None:
        try:
            h, a = devigify_two_way(int(ml_home), int(ml_away))
            result["retail_implied_home"] = round(h, 6)
            result["retail_implied_away"] = round(a, 6)
        except Exception as exc:
            log.debug("Retail implied ML devigify failed: %s", exc)

    if total_over_odds is not None and total_under_odds is not None:
        try:
            o, u = devigify_two_way(int(total_over_odds), int(total_under_odds))
            result["retail_implied_over"]  = round(o, 6)
            result["retail_implied_under"] = round(u, 6)
        except Exception as exc:
            log.debug("Retail implied total devigify failed: %s", exc)

    if rl_home_odds is not None and rl_away_odds is not None:
        try:
            rh, ra = devigify_two_way(int(rl_home_odds), int(rl_away_odds))
            result["retail_implied_rl_home"] = round(rh, 6)
            result["retail_implied_rl_away"] = round(ra, 6)
        except Exception as exc:
            log.debug("Retail implied RL devigify failed: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Source 1: Historical OddsPortal cache
# ---------------------------------------------------------------------------
_historical_cache: dict[int, pd.DataFrame] = {}


def _load_historical_cache(year: int) -> pd.DataFrame:
    """Load and memoize the OddsPortal historical parquet for a given year."""
    if year not in _historical_cache:
        path = OUTPUT_DIR / f"odds_historical_{year}.parquet"
        if path.exists():
            df = pd.read_parquet(path, engine="pyarrow")
            df["home_team"] = df["home_team"].astype(str)
            df["away_team"] = df["away_team"].astype(str)
            df["game_date"] = df["game_date"].astype(str)
            _historical_cache[year] = df
            log.info("Loaded historical cache for %d: %d rows", year, len(df))
        else:
            log.info("No historical cache found for %d at %s", year, path)
            _historical_cache[year] = pd.DataFrame()
    return _historical_cache[year]


def get_odds_from_historical(game_date: str, home_team: str, away_team: str) -> Optional[dict]:
    """Look up a game in the OddsPortal historical cache."""
    year = int(game_date[:4])
    if year not in HISTORICAL_YEARS:
        return None
    df = _load_historical_cache(year)
    if df.empty:
        return None
    mask = (
        (df["game_date"] == game_date) &
        (df["home_team"] == home_team) &
        (df["away_team"] == away_team)
    )
    rows = df[mask]
    if rows.empty:
        return None
    row = rows.iloc[0].to_dict()
    row["source"] = "oddsportal"
    return row


# ---------------------------------------------------------------------------
# Source 2: The Odds API — generic regional fetcher
# ---------------------------------------------------------------------------
_odds_api_quota_remaining: Optional[int] = None
_odds_api_quota_used: Optional[int] = None

# Separate in-memory caches per region to avoid cross-contamination
_odds_api_us_cache: Optional[list[dict]] = None
_odds_api_eu_cache: Optional[list[dict]] = None


def _log_quota(remaining: Optional[int], used: Optional[int]) -> None:
    line = (
        f"{datetime.utcnow().isoformat()} | "
        f"requests_remaining: {remaining} | requests_used: {used}\n"
    )
    with open(QUOTA_LOG, "a", encoding="utf-8") as fh:
        fh.write(line)
    log.info("Odds API quota — remaining: %s | used: %s", remaining, used)


def fetch_odds_api_region(
    target_date: str,
    region: str,
    bookmakers: list[str],
    force: bool = False,
) -> Optional[list[dict]]:
    """
    Fetch odds from The Odds API for a specific geographic region and bookmaker set.

    Parameters
    ----------
    target_date : ISO date string, e.g. "2026-04-14"
    region      : "us" for retail books, "eu" for Pinnacle/sharp books.
    bookmakers  : List of Odds API bookmaker keys to request.
    force       : If True, bypass cache and re-fetch even if a cache file exists.

    Returns
    -------
    Raw list of game dicts from the API, or None on failure / quota exhaustion.
    Caches the raw JSON to ODDS_API_CACHE_DIR/{date}_{region}.json.
    Cache is reused only if it is less than ODDS_API_CACHE_TTL_HOURS old.
    """
    global _odds_api_quota_remaining, _odds_api_quota_used

    if not ODDS_API_KEY:
        log.warning("ODDS_API_KEY not set — skipping Odds API (%s region)", region)
        return None

    cache_file = ODDS_API_CACHE_DIR / f"{target_date}_{region}.json"
    if cache_file.exists() and not force:
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < ODDS_API_CACHE_TTL_HOURS:
            log.info(
                "Loading Odds API %s-region response from cache (%.1fh old): %s",
                region, age_hours, cache_file,
            )
            with open(cache_file, encoding="utf-8") as fh:
                return json.load(fh)
        else:
            log.info(
                "Cache %s is %.1fh old (TTL=%dh) — re-fetching %s region",
                cache_file, age_hours, ODDS_API_CACHE_TTL_HOURS, region,
            )

    if (
        _odds_api_quota_remaining is not None
        and _odds_api_quota_remaining < ODDS_API_QUOTA_THRESHOLD
    ):
        log.warning(
            "Odds API quota low (%d remaining) — skipping %s region",
            _odds_api_quota_remaining,
            region,
        )
        _log_quota(_odds_api_quota_remaining, _odds_api_quota_used)
        return None

    params = {
        "apiKey":            ODDS_API_KEY,
        "regions":           region,
        "bookmakers":        ",".join(bookmakers),
        "markets":           "h2h,totals,spreads",
        "oddsFormat":        "american",
        "dateFormat":        "iso",
        "commenceTimeFrom":  f"{target_date}T00:00:00Z",
        "commenceTimeTo":    f"{target_date}T23:59:59Z",
    }

    try:
        resp = requests.get(ODDS_API_BASE, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.error("Odds API %s-region request failed: %s", region, exc)
        return None

    remaining = resp.headers.get("x-requests-remaining")
    used      = resp.headers.get("x-requests-used")
    if remaining is not None:
        _odds_api_quota_remaining = int(remaining)
    if used is not None:
        _odds_api_quota_used = int(used)
    _log_quota(_odds_api_quota_remaining, _odds_api_quota_used)

    if (
        _odds_api_quota_remaining is not None
        and _odds_api_quota_remaining < ODDS_API_QUOTA_THRESHOLD
    ):
        log.warning(
            "Odds API quota now low (%d remaining) after %s-region pull",
            _odds_api_quota_remaining,
            region,
        )

    data = resp.json()
    with open(cache_file, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    log.info(
        "Odds API %s-region cached: %s (%d games)", region, cache_file, len(data)
    )
    return data


def fetch_odds_api(target_date: str, force: bool = False) -> Optional[list[dict]]:
    """
    Fetch the US-region odds (DraftKings, FanDuel, BetMGM).
    Populates the module-level US cache.
    Kept for backward compatibility with existing callers.
    """
    global _odds_api_us_cache
    if _odds_api_us_cache is None or force:
        _odds_api_us_cache = fetch_odds_api_region(
            target_date, "us", ODDS_API_US_BOOKS, force=force
        )
    return _odds_api_us_cache


def fetch_pinnacle_data(target_date: str, force: bool = False) -> Optional[list[dict]]:
    """
    Fetch the EU/Pinnacle region odds — the sharp benchmark.
    Populates the module-level EU cache.
    """
    global _odds_api_eu_cache
    if _odds_api_eu_cache is None or force:
        _odds_api_eu_cache = fetch_odds_api_region(
            target_date, "eu", ODDS_API_EU_BOOKS, force=force
        )
    return _odds_api_eu_cache


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _commence_to_et_hour(commence_time: str) -> float | None:
    """
    Convert an ISO-8601 UTC commence_time string to Eastern Time hour of day.

    Returns a float (e.g. 13.75 = 1:45 PM ET) or None on parse failure.
    """
    if not commence_time:
        return None
    try:
        ts = commence_time.rstrip("Z").replace("+00:00", "")
        dt_utc = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        try:
            from zoneinfo import ZoneInfo
            dt_et = dt_utc.astimezone(ZoneInfo("America/New_York"))
        except Exception:
            offset_h = -4 if 3 <= dt_utc.month <= 11 else -5
            dt_et = dt_utc + timedelta(hours=offset_h)
        return float(dt_et.hour) + dt_et.minute / 60.0
    except Exception:
        return None


def _parse_retail_game(game: dict) -> dict:
    """
    Convert a single US-region Odds API game object to a standardised retail dict.

    Selects the best available book in ODDS_API_US_BOOKS preference order
    (DraftKings → FanDuel → BetMGM → first available).

    Returns a dict with keys from SCHEMA_COLS that are retail-specific, plus
    de-vigged Retail_Implied_Prob fields populated inline.
    """
    home_raw = game.get("home_team", "")
    away_raw = game.get("away_team", "")
    home_team = normalize_team(home_raw)
    away_team = normalize_team(away_raw)

    commence_time = game.get("commence_time", "")
    game_date = commence_time[:10] if commence_time else None

    result = _empty_odds_dict(game_date or "", home_team, away_team)
    result["source"] = "odds_api"
    result["game_hour_et"] = _commence_to_et_hour(commence_time)

    bookmakers = game.get("bookmakers", [])

    # Pick the best available retail book
    preferred = None
    for bk_name in ODDS_API_US_BOOKS:
        for bk in bookmakers:
            if bk.get("key", "").lower() == bk_name:
                preferred = bk
                break
        if preferred:
            break
    if not preferred and bookmakers:
        preferred = bookmakers[0]
    if not preferred:
        return result

    result["retail_book_used"] = preferred.get("key")

    # Collect raw retail odds for de-vigging
    retail_ml_home: Optional[int] = None
    retail_ml_away: Optional[int] = None
    retail_total_over_odds: Optional[int] = None
    retail_total_under_odds: Optional[int] = None
    retail_rl_home_odds: Optional[int] = None
    retail_rl_away_odds: Optional[int] = None

    for market in preferred.get("markets", []):
        key      = market.get("key", "")
        outcomes = market.get("outcomes", [])

        if key == "h2h":
            for out in outcomes:
                name  = normalize_team(out.get("name", ""))
                price = out.get("price")
                if name == home_team:
                    result["close_ml_home"] = int(price) if price is not None else None
                    retail_ml_home = result["close_ml_home"]
                elif name == away_team:
                    result["close_ml_away"] = int(price) if price is not None else None
                    retail_ml_away = result["close_ml_away"]

        elif key == "totals":
            over_price  = None
            under_price = None
            for out in outcomes:
                side = out.get("name", "").lower()
                if side == "over":
                    result["close_total"] = float(out["point"]) if "point" in out else None
                    over_price  = int(out["price"]) if "price" in out else None
                elif side == "under":
                    under_price = int(out["price"]) if "price" in out else None
            retail_total_over_odds  = over_price
            retail_total_under_odds = under_price

        elif key == "spreads":
            away_rl_odds: Optional[int] = None
            for out in outcomes:
                name = normalize_team(out.get("name", ""))
                if name == home_team:
                    result["runline_home"] = (
                        float(out["point"]) if "point" in out else -1.5
                    )
                    result["runline_home_odds"] = (
                        int(out["price"]) if "price" in out else None
                    )
                    retail_rl_home_odds = result["runline_home_odds"]
                elif name == away_team:
                    away_rl_odds = int(out["price"]) if "price" in out else None
            retail_rl_away_odds = away_rl_odds

    # Compute de-vigged Retail Implied Probabilities
    retail_implied = compute_retail_implied(
        ml_home=retail_ml_home,
        ml_away=retail_ml_away,
        total_over_odds=retail_total_over_odds,
        total_under_odds=retail_total_under_odds,
        rl_home_odds=retail_rl_home_odds,
        rl_away_odds=retail_rl_away_odds,
    )
    result.update(retail_implied)

    return result


def _parse_pinnacle_game(game: dict) -> Optional[dict]:
    """
    Extract Pinnacle's raw lines from an EU-region Odds API game object and
    compute de-vigged P_true values.

    Returns a dict keyed by (home_team, away_team) with Pinnacle-specific
    fields, or None if Pinnacle is not present in the bookmakers list.
    """
    home_team = normalize_team(game.get("home_team", ""))
    away_team = normalize_team(game.get("away_team", ""))

    # Locate Pinnacle in the bookmakers list
    pinnacle_bk = None
    for bk in game.get("bookmakers", []):
        if bk.get("key", "").lower() == "pinnacle":
            pinnacle_bk = bk
            break

    if not pinnacle_bk:
        log.debug("Pinnacle not found in EU response for %s @ %s", away_team, home_team)
        return None

    out: dict = {
        "home_team": home_team,
        "away_team": away_team,
        "pinnacle_ml_home":         None,
        "pinnacle_ml_away":         None,
        "pinnacle_total_line":      None,
        "pinnacle_total_over_odds": None,
        "pinnacle_total_under_odds":None,
        "pinnacle_rl_home":         None,
        "pinnacle_rl_home_odds":    None,
        "pinnacle_rl_away_odds":    None,
    }

    pin_ml_home: Optional[int]  = None
    pin_ml_away: Optional[int]  = None
    pin_tot_over: Optional[int] = None
    pin_tot_under: Optional[int]= None
    pin_rl_home_odds: Optional[int] = None
    pin_rl_away_odds: Optional[int] = None

    for market in pinnacle_bk.get("markets", []):
        key      = market.get("key", "")
        outcomes = market.get("outcomes", [])

        if key == "h2h":
            for o in outcomes:
                name  = normalize_team(o.get("name", ""))
                price = o.get("price")
                if name == home_team:
                    out["pinnacle_ml_home"] = int(price) if price is not None else None
                    pin_ml_home = out["pinnacle_ml_home"]
                elif name == away_team:
                    out["pinnacle_ml_away"] = int(price) if price is not None else None
                    pin_ml_away = out["pinnacle_ml_away"]

        elif key == "totals":
            for o in outcomes:
                side = o.get("name", "").lower()
                if side == "over":
                    out["pinnacle_total_line"]      = (
                        float(o["point"]) if "point" in o else None
                    )
                    out["pinnacle_total_over_odds"] = (
                        int(o["price"]) if "price" in o else None
                    )
                    pin_tot_over = out["pinnacle_total_over_odds"]
                elif side == "under":
                    out["pinnacle_total_under_odds"] = (
                        int(o["price"]) if "price" in o else None
                    )
                    pin_tot_under = out["pinnacle_total_under_odds"]

        elif key == "spreads":
            for o in outcomes:
                name = normalize_team(o.get("name", ""))
                if name == home_team:
                    out["pinnacle_rl_home"]      = (
                        float(o["point"]) if "point" in o else None
                    )
                    out["pinnacle_rl_home_odds"] = (
                        int(o["price"]) if "price" in o else None
                    )
                    pin_rl_home_odds = out["pinnacle_rl_home_odds"]
                elif name == away_team:
                    out["pinnacle_rl_away_odds"] = (
                        int(o["price"]) if "price" in o else None
                    )
                    pin_rl_away_odds = out["pinnacle_rl_away_odds"]

    # Compute de-vigged True Market Probabilities from Pinnacle lines
    p_true = compute_p_true(
        ml_home=pin_ml_home,
        ml_away=pin_ml_away,
        total_over_odds=pin_tot_over,
        total_under_odds=pin_tot_under,
        rl_home_odds=pin_rl_home_odds,
        rl_away_odds=pin_rl_away_odds,
    )
    out.update(p_true)

    return out


def get_odds_from_api(game_date: str, home_team: str, away_team: str) -> Optional[dict]:
    """
    Look up a specific game from the US-region Odds API data for that date.
    Returns a retail-parsed dict with Retail_Implied_Prob fields populated.
    """
    global _odds_api_us_cache
    if _odds_api_us_cache is None:
        _odds_api_us_cache = fetch_odds_api(game_date) or []

    for game in _odds_api_us_cache:
        ht       = normalize_team(game.get("home_team", ""))
        at       = normalize_team(game.get("away_team", ""))
        commence = game.get("commence_time", "")[:10]
        if commence == game_date and ht == home_team and at == away_team:
            return _parse_retail_game(game)

    return None


def get_pinnacle_lines(
    target_date: str,
    home_team: str,
    away_team: str,
) -> Optional[dict]:
    """
    Look up Pinnacle's parsed + de-vigged lines for a specific game.

    Returns a dict with pinnacle_* and P_true_* keys, or None if not found.
    The caller is responsible for merging this into the retail odds dict.
    """
    global _odds_api_eu_cache
    if _odds_api_eu_cache is None:
        _odds_api_eu_cache = fetch_pinnacle_data(target_date) or []

    for game in _odds_api_eu_cache:
        ht       = normalize_team(game.get("home_team", ""))
        at       = normalize_team(game.get("away_team", ""))
        commence = game.get("commence_time", "")[:10]
        if commence == target_date and ht == home_team and at == away_team:
            return _parse_pinnacle_game(game)

    log.debug(
        "Pinnacle lines not found for %s @ %s on %s",
        away_team, home_team, target_date,
    )
    return None


# ---------------------------------------------------------------------------
# Pitcher strikeout props (per-event endpoint)
# ---------------------------------------------------------------------------

def fetch_pitcher_strikeout_props(target_date: str) -> pd.DataFrame:
    """
    Fetch pitcher strikeout over/under props from The Odds API.
    Calls the per-event endpoint for each game on target_date.

    Returns DataFrame with columns:
        game_date, event_id, home_team, away_team,
        pitcher_name, line, over_odds, under_odds, book, pull_timestamp

    Saves to data/statcast/k_props_{target_date}.parquet
    """
    global _odds_api_quota_remaining, _odds_api_quota_used

    if not ODDS_API_KEY:
        log.warning("ODDS_API_KEY not set — skipping K props fetch")
        return pd.DataFrame()

    if (
        _odds_api_quota_remaining is not None
        and _odds_api_quota_remaining < ODDS_API_QUOTA_THRESHOLD + 20
    ):
        log.warning(
            "Odds API quota too low for K props (%d remaining)",
            _odds_api_quota_remaining or 0,
        )
        return pd.DataFrame()

    cache_path = OUTPUT_DIR / f"k_props_{target_date}.parquet"
    if cache_path.exists():
        age_hours = (datetime.utcnow().timestamp() - cache_path.stat().st_mtime) / 3600
        if age_hours < 4:
            log.info("Loading K props from cache: %s", cache_path)
            return pd.read_parquet(cache_path, engine="pyarrow")

    global _odds_api_us_cache
    if _odds_api_us_cache is None:
        _odds_api_us_cache = fetch_odds_api(target_date) or []

    if not _odds_api_us_cache:
        log.warning("No game-level US odds available — cannot fetch K props")
        return pd.DataFrame()

    props_rows  = []
    events_tried = 0

    for game in _odds_api_us_cache:
        event_id = game.get("id", "")
        if not event_id:
            continue

        home_team = normalize_team(game.get("home_team", ""))
        away_team = normalize_team(game.get("away_team", ""))

        url    = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds"
        params = {
            "apiKey":       ODDS_API_KEY,
            "regions":      "us",
            "markets":      "pitcher_strikeouts",
            "bookmakers":   "draftkings,fanduel,betmgm",
            "oddsFormat":   "american",
        }

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 404:
                log.info(
                    "K props not yet posted for %s @ %s (404 — props post ~2h before game)",
                    away_team, home_team,
                )
            else:
                log.warning(
                    "K props fetch failed for %s @ %s: %s", away_team, home_team, exc
                )
            continue
        except requests.RequestException as exc:
            log.warning(
                "K props fetch failed for %s @ %s: %s", away_team, home_team, exc
            )
            continue

        remaining = resp.headers.get("x-requests-remaining")
        used      = resp.headers.get("x-requests-used")
        if remaining is not None:
            _odds_api_quota_remaining = int(remaining)
        if used is not None:
            _odds_api_quota_used = int(used)
        events_tried += 1

        if (
            _odds_api_quota_remaining is not None
            and _odds_api_quota_remaining < ODDS_API_QUOTA_THRESHOLD
        ):
            log.warning(
                "K props: quota hit threshold after %d events", events_tried
            )
            _log_quota(_odds_api_quota_remaining, _odds_api_quota_used)
            break

        try:
            data = resp.json()
        except Exception:
            continue

        for bk in data.get("bookmakers", []):
            book_key = bk.get("key", "")
            for market in bk.get("markets", []):
                if market.get("key") != "pitcher_strikeouts":
                    continue
                pitcher_lines: dict = {}
                for out in market.get("outcomes", []):
                    pitcher = out.get("description", out.get("name", ""))
                    side    = out.get("name", "").lower()
                    price   = out.get("price")
                    point   = out.get("point")
                    if not pitcher or price is None or point is None:
                        continue
                    if pitcher not in pitcher_lines:
                        pitcher_lines[pitcher] = {"line": float(point)}
                    if "over" in side:
                        pitcher_lines[pitcher]["over_odds"] = int(price)
                    elif "under" in side:
                        pitcher_lines[pitcher]["under_odds"] = int(price)

                for pitcher, info in pitcher_lines.items():
                    props_rows.append({
                        "game_date":      target_date,
                        "event_id":       event_id,
                        "home_team":      home_team,
                        "away_team":      away_team,
                        "pitcher_name":   pitcher,
                        "line":           info.get("line"),
                        "over_odds":      info.get("over_odds"),
                        "under_odds":     info.get("under_odds"),
                        "book":           book_key,
                        "pull_timestamp": datetime.utcnow(),
                    })

        time.sleep(0.3)

    _log_quota(_odds_api_quota_remaining, _odds_api_quota_used)
    log.info(
        "K props: fetched %d rows from %d events", len(props_rows), events_tried
    )

    if not props_rows:
        return pd.DataFrame()

    df = pd.DataFrame(props_rows)
    df["line"]       = pd.to_numeric(df["line"],       errors="coerce")
    df["over_odds"]  = pd.to_numeric(df["over_odds"],  errors="coerce")
    df["under_odds"] = pd.to_numeric(df["under_odds"], errors="coerce")

    try:
        df.to_parquet(cache_path, engine="pyarrow", index=False)
        log.info("K props saved: %s (%d rows)", cache_path, len(df))
    except Exception as exc:
        log.warning("Could not save K props: %s", exc)

    return df


def get_k_prop_for_pitcher(pitcher_name: str, target_date: str) -> Optional[dict]:
    """
    Look up the consensus K prop line for a specific pitcher on target_date.
    Returns dict with: line, over_odds, under_odds, book — or None if not found.
    """
    cache_path = OUTPUT_DIR / f"k_props_{target_date}.parquet"
    if not cache_path.exists():
        return None

    try:
        df = pd.read_parquet(cache_path, engine="pyarrow")
    except Exception:
        return None

    if df.empty:
        return None

    def _norm(s: object) -> str:
        return str(s).upper().strip() if s else ""

    pitcher_norm = _norm(pitcher_name)
    df["name_norm"] = df["pitcher_name"].apply(_norm)

    matches = df[df["name_norm"] == pitcher_norm]
    if matches.empty:
        last    = pitcher_norm.split()[-1] if pitcher_norm else ""
        matches = (
            df[df["name_norm"].str.contains(last, na=False)] if last else pd.DataFrame()
        )
    if matches.empty:
        return None

    for book in ["draftkings", "fanduel", "betmgm"]:
        book_rows = matches[matches["book"] == book]
        if not book_rows.empty:
            row = book_rows.iloc[0]
            return {
                "pitcher_name": row["pitcher_name"],
                "line":         row["line"],
                "over_odds":    row["over_odds"],
                "under_odds":   row["under_odds"],
                "book":         row["book"],
            }

    row = matches.iloc[0]
    return {
        "pitcher_name": row["pitcher_name"],
        "line":         row["line"],
        "over_odds":    row["over_odds"],
        "under_odds":   row["under_odds"],
        "book":         row["book"],
    }


# ---------------------------------------------------------------------------
# Source 3: ActionNetwork scraper
# ---------------------------------------------------------------------------
_actionnetwork_cache: Optional[list[dict]] = None


def fetch_actionnetwork() -> list[dict]:
    """Scrape ActionNetwork MLB odds page for public betting % data."""
    global _actionnetwork_cache
    if _actionnetwork_cache is not None:
        return _actionnetwork_cache

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer":         "https://www.actionnetwork.com/",
    }

    games = []
    try:
        resp = requests.get(ACTIONNETWORK_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        soup  = BeautifulSoup(resp.text, "html.parser")
        games = _parse_actionnetwork_html(soup)
        log.info("ActionNetwork: scraped %d games", len(games))
    except requests.RequestException as exc:
        log.error("ActionNetwork request failed: %s", exc)
    except Exception as exc:
        log.error("ActionNetwork parse error: %s", exc, exc_info=True)

    _actionnetwork_cache = games
    return games


def _parse_actionnetwork_html(soup: BeautifulSoup) -> list[dict]:
    games     = []
    today_str = date.today().strftime("%Y-%m-%d")
    pull_ts   = datetime.utcnow()

    for script in soup.find_all("script", {"id": "__NEXT_DATA__"}):
        try:
            data       = json.loads(script.string or "")
            page_props = data.get("props", {}).get("pageProps", {})
            game_list  = (
                page_props.get("games")
                or page_props.get("mlbGames")
                or page_props.get("matchups")
                or []
            )
            for g in game_list:
                parsed = _parse_actionnetwork_game(g, today_str, pull_ts)
                if parsed:
                    games.append(parsed)
            if games:
                return games
        except (json.JSONDecodeError, AttributeError):
            continue

    game_rows = soup.find_all(
        "div", class_=re.compile(r"game-line|matchup|odds-table|event-row", re.I)
    )
    for row in game_rows:
        parsed = _parse_actionnetwork_html_row(row, today_str, pull_ts)
        if parsed:
            games.append(parsed)

    return games


def _parse_actionnetwork_game(
    g: dict, game_date: str, pull_ts: datetime
) -> Optional[dict]:
    try:
        home_raw = (
            g.get("home_team", {}).get("full_name")
            or g.get("home", {}).get("full_name")
            or g.get("homeTeam", {}).get("name", "")
        )
        away_raw = (
            g.get("away_team", {}).get("full_name")
            or g.get("away", {}).get("full_name")
            or g.get("awayTeam", {}).get("name", "")
        )
        if not home_raw or not away_raw:
            return None

        home_team = normalize_team(home_raw)
        away_team = normalize_team(away_raw)

        odds       = g.get("odds", [{}])
        first_book = odds[0] if odds else {}

        close_ml_home = first_book.get("ml_home") or first_book.get("moneyline_home")
        close_ml_away = first_book.get("ml_away") or first_book.get("moneyline_away")
        close_total   = first_book.get("total") or first_book.get("over_under")

        public          = g.get("public_betting", {}) or g.get("publicBetting", {})
        public_pct_home = None
        public_pct_over = None
        if public:
            home_pct = public.get("home_ml_percent") or public.get("homeMlPercent")
            over_pct = public.get("over_percent")    or public.get("overPercent")
            public_pct_home = float(home_pct) / 100 if home_pct is not None else None
            public_pct_over = float(over_pct) / 100 if over_pct is not None else None

        rl_home      = first_book.get("spread_home") or first_book.get("run_line_home", -1.5)
        rl_home_odds = (
            first_book.get("spread_home_odds") or first_book.get("run_line_home_odds")
        )

        scheduled = g.get("scheduled") or g.get("start_time") or ""
        gd = scheduled[:10] if scheduled else game_date

        return {
            "game_date":          gd,
            "game_pk":            None,
            "home_team":          home_team,
            "away_team":          away_team,
            "open_ml_home":       None,
            "close_ml_home":      int(close_ml_home) if close_ml_home is not None else None,
            "open_ml_away":       None,
            "close_ml_away":      int(close_ml_away) if close_ml_away is not None else None,
            "open_total":         None,
            "close_total":        float(close_total) if close_total is not None else None,
            "runline_home":       float(rl_home) if rl_home is not None else -1.5,
            "runline_home_odds":  int(rl_home_odds) if rl_home_odds is not None else None,
            "public_pct_home":    public_pct_home,
            "public_pct_over":    public_pct_over,
            "source":             "actionnetwork",
            "pull_timestamp":     pull_ts,
        }
    except Exception as exc:
        log.debug("Failed to parse ActionNetwork game JSON: %s", exc)
        return None


def _parse_actionnetwork_html_row(
    row, game_date: str, pull_ts: datetime
) -> Optional[dict]:
    try:
        team_els = row.find_all(
            class_=re.compile(r"team-name|team_name|teamName|participant", re.I)
        )
        if len(team_els) < 2:
            return None

        home_raw  = team_els[1].get_text(strip=True)
        away_raw  = team_els[0].get_text(strip=True)
        if not home_raw or not away_raw:
            return None

        home_team = normalize_team(home_raw)
        away_team = normalize_team(away_raw)

        ml_els        = row.find_all(class_=re.compile(r"moneyline|money-line|ml-odds", re.I))
        close_ml_away = None
        close_ml_home = None
        if len(ml_els) >= 2:
            close_ml_away = _text_to_int(ml_els[0].get_text(strip=True))
            close_ml_home = _text_to_int(ml_els[1].get_text(strip=True))
        elif ml_els:
            close_ml_home = _text_to_int(ml_els[0].get_text(strip=True))

        total_el    = row.find(class_=re.compile(r"total|over-under", re.I))
        close_total = None
        if total_el:
            raw_total = re.search(r"[\d.]+", total_el.get_text())
            close_total = float(raw_total.group()) if raw_total else None

        public_els      = row.find_all(class_=re.compile(r"public|consensus|bets-pct", re.I))
        public_pct_home = None
        public_pct_over = None
        if len(public_els) >= 2:
            hp = re.search(r"([\d.]+)%?", public_els[-1].get_text())
            op = re.search(r"([\d.]+)%?", public_els[0].get_text())
            public_pct_home = float(hp.group(1)) / 100 if hp else None
            public_pct_over = float(op.group(1)) / 100 if op else None

        return {
            "game_date":         game_date,
            "game_pk":           None,
            "home_team":         home_team,
            "away_team":         away_team,
            "open_ml_home":      None,
            "close_ml_home":     close_ml_home,
            "open_ml_away":      None,
            "close_ml_away":     close_ml_away,
            "open_total":        None,
            "close_total":       close_total,
            "runline_home":      -1.5,
            "runline_home_odds": None,
            "public_pct_home":   public_pct_home,
            "public_pct_over":   public_pct_over,
            "source":            "actionnetwork",
            "pull_timestamp":    pull_ts,
        }
    except Exception as exc:
        log.debug("Failed to parse ActionNetwork HTML row: %s", exc)
        return None


def _text_to_int(text: str) -> Optional[int]:
    text = text.strip().replace("+", "").replace(",", "")
    try:
        return int(text)
    except ValueError:
        return None


def get_odds_from_actionnetwork(
    game_date: str, home_team: str, away_team: str
) -> Optional[dict]:
    """Look up a game from the ActionNetwork scrape cache."""
    games = fetch_actionnetwork()
    for g in games:
        if g["home_team"] == home_team and g["away_team"] == away_team:
            g["game_date"] = game_date
            return g
    return None


# ---------------------------------------------------------------------------
# Waterfall function — retail lines
# ---------------------------------------------------------------------------

def get_odds_for_game(
    game_date: str,
    home_team: str,
    away_team: str,
    year: Optional[int] = None,
) -> Optional[dict]:
    """
    Return a standardised retail odds dict via waterfall priority:
      1. OddsPortal historical cache  (completed games, 2023–2025)
      2. Odds API US region            (current season, quota-guarded)
      3. ActionNetwork                 (scraper fallback; public % data)

    Pinnacle P_true fields are NOT populated here — they are merged in
    pull_odds_for_date() after a separate EU-region fetch.
    """
    if year is None:
        year = int(game_date[:4])

    log.debug("Waterfall lookup: %s %s @ %s", game_date, away_team, home_team)

    result = get_odds_from_historical(game_date, home_team, away_team)
    if result:
        return result

    result = get_odds_from_api(game_date, home_team, away_team)
    if result:
        return result

    result = get_odds_from_actionnetwork(game_date, home_team, away_team)
    if result:
        return result

    log.warning("No retail odds found: %s %s @ %s", game_date, away_team, home_team)
    _log_missing(game_date, home_team, away_team)
    return None


def _log_missing(game_date: str, home_team: str, away_team: str) -> None:
    line = f"{datetime.utcnow().isoformat()} | {game_date} | {away_team} @ {home_team}\n"
    with open(MISSING_LOG, "a", encoding="utf-8") as fh:
        fh.write(line)


# ---------------------------------------------------------------------------
# Schedule loader
# ---------------------------------------------------------------------------

def load_schedule_for_date(target_date: str) -> list[dict]:
    """Load game schedule for the target date from available mlb_schedule parquets."""
    year          = int(target_date[:4])
    schedule_path = OUTPUT_DIR / f"mlb_schedule_{year}.parquet"
    if not schedule_path.exists():
        schedule_path = OUTPUT_DIR / f"schedule_all_{year}.parquet"
    if not schedule_path.exists():
        log.warning("No schedule file found for year %d", year)
        return []

    df = pd.read_parquet(schedule_path, engine="pyarrow")
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if not date_cols:
        log.warning("No date column in schedule file")
        return []

    date_col        = date_cols[0]
    df[date_col]    = df[date_col].astype(str).str[:10]
    day_df          = df[df[date_col] == target_date]

    games = []
    for _, row in day_df.iterrows():
        # Support both snake_case (home_team/away_team) and the MLB Stats API
        # camelCase schema (home_team_name/away_team_name) used in mlb_schedule_YYYY.parquet
        home = str(
            row.get("home_team_name") or row.get("home_team") or
            row.get("home") or ""
        )
        away = str(
            row.get("away_team_name") or row.get("away_team") or
            row.get("away") or ""
        )
        pk = (row.get("gamePk") or row.get("game_pk") or
              row.get("game_id") or row.get("gamePK"))
        if home and away and home != "None" and away != "None":
            games.append({
                "game_date": target_date,
                "home_team": normalize_team(home),
                "away_team": normalize_team(away),
                "game_pk":   int(pk) if pk is not None and str(pk) != "nan" else None,
            })
    log.info("Schedule: found %d games for %s", len(games), target_date)
    return games


# ---------------------------------------------------------------------------
# Main pull logic
# ---------------------------------------------------------------------------

def _to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Convert list of odds dicts to a fully typed, schema-aligned DataFrame."""
    if not records:
        return pd.DataFrame(columns=SCHEMA_COLS)

    df = pd.DataFrame(records)
    for col in SCHEMA_COLS:
        if col not in df.columns:
            df[col] = None
    df = df[SCHEMA_COLS]

    df["game_date"] = df["game_date"].astype(str)

    int_cols = [
        "game_pk", "open_ml_home", "close_ml_home", "open_ml_away", "close_ml_away",
        "runline_home_odds",
        "pinnacle_ml_home", "pinnacle_ml_away",
        "pinnacle_total_over_odds", "pinnacle_total_under_odds",
        "pinnacle_rl_home_odds", "pinnacle_rl_away_odds",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    float_cols = [
        "open_total", "close_total", "runline_home",
        "public_pct_home", "public_pct_over",
        "pinnacle_total_line", "pinnacle_rl_home",
        "P_true_home", "P_true_away",
        "P_true_over", "P_true_under",
        "P_true_rl_home", "P_true_rl_away",
        "retail_implied_home", "retail_implied_away",
        "retail_implied_over", "retail_implied_under",
        "retail_implied_rl_home", "retail_implied_rl_away",
        "game_hour_et",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["pull_timestamp"] = pd.to_datetime(df["pull_timestamp"])
    return df


def pull_odds_for_date(target_date: str, force: bool = False) -> None:
    """
    Pull odds for all games on target_date and save to parquet.

    Execution order
    ---------------
    1. Warm the US-region cache (retail reference lines).
    2. Warm the EU/Pinnacle cache (sharp benchmark).
    3. For each game, fetch retail via waterfall.
    4. Merge Pinnacle lines into the retail row (P_true + Retail_Implied already
       computed at parse time; we just stamp in the Pinnacle raw fields + P_true here).
    5. Write the enriched DataFrame to odds_current_{date}.parquet.

    Parameters
    ----------
    force : If True, bypass the API cache and re-fetch from the Odds API even if
            a cache file already exists. Use this after West Coast lines post.
    """
    log.info("Pulling odds for date: %s (force=%s)", target_date, force)
    output_path = OUTPUT_DIR / f"odds_current_{target_date.replace('-', '_')}.parquet"

    # ── Step 1 & 2: Warm both region caches before the per-game loop ──────────
    log.info("Fetching US-region (retail) odds …")
    fetch_odds_api(target_date, force=force)

    log.info("Fetching EU/Pinnacle (sharp benchmark) odds …")
    fetch_pinnacle_data(target_date, force=force)

    # ── Step 3: Build game list ───────────────────────────────────────────────
    games = load_schedule_for_date(target_date)

    if not games:
        log.info("No schedule file — pulling game list from Odds API + ActionNetwork")
        api_data = _odds_api_us_cache or []
        for g in api_data:
            ht = normalize_team(g.get("home_team", ""))
            at = normalize_team(g.get("away_team", ""))
            games.append({
                "game_date": target_date,
                "home_team": ht,
                "away_team": at,
                "game_pk":   None,
            })
        an_data      = fetch_actionnetwork()
        existing_keys = {(g["home_team"], g["away_team"]) for g in games}
        for g in an_data:
            key = (g["home_team"], g["away_team"])
            if key not in existing_keys:
                games.append({
                    "game_date": target_date,
                    "home_team": g["home_team"],
                    "away_team": g["away_team"],
                    "game_pk":   None,
                })

    if not games:
        log.warning("No games found for %s — nothing to pull", target_date)
        return

    log.info("Processing %d games for %s", len(games), target_date)

    # ── Step 4: Per-game fetch + Pinnacle merge ───────────────────────────────
    results = []
    for g in games:
        retail = get_odds_for_game(
            game_date=g["game_date"],
            home_team=g["home_team"],
            away_team=g["away_team"],
        )
        if retail is None:
            retail = _empty_odds_dict(g["game_date"], g["home_team"], g["away_team"])

        if g.get("game_pk"):
            retail["game_pk"] = g["game_pk"]

        # Merge Pinnacle sharp lines + P_true into the retail row
        pinnacle = get_pinnacle_lines(
            target_date=g["game_date"],
            home_team=g["home_team"],
            away_team=g["away_team"],
        )
        if pinnacle:
            pinnacle_fields = {
                k: v for k, v in pinnacle.items()
                if k not in ("home_team", "away_team")
            }
            retail.update(pinnacle_fields)
            log.debug(
                "Pinnacle merged for %s @ %s — P_true_home=%.4f  P_true_away=%.4f",
                g["away_team"],
                g["home_team"],
                pinnacle.get("P_true_home") or 0,
                pinnacle.get("P_true_away") or 0,
            )
        else:
            log.info(
                "No Pinnacle lines for %s @ %s — P_true will be null",
                g["away_team"],
                g["home_team"],
            )

        results.append(retail)

    # ── Step 5: Write output ──────────────────────────────────────────────────
    df = _to_dataframe(results)
    df = df.drop_duplicates(subset=["game_date", "home_team", "away_team"], keep="first")
    df = df.sort_values(["game_date", "home_team"]).reset_index(drop=True)

    # Merge with previous save: preserve non-null Pinnacle/retail values from earlier
    # pulls so that lines captured at game time aren't wiped by post-game re-runs
    # (Pinnacle removes lines once a game starts; later pulls would overwrite with NaN).
    if output_path.exists():
        try:
            prev = pd.read_parquet(output_path, engine="pyarrow")
            _key = ["home_team", "away_team"]
            _fill_cols = [c for c in prev.columns if c not in _key
                          and c in df.columns and c != "pull_timestamp"]
            prev_idx = prev.set_index(_key)
            df = df.set_index(_key)
            for col in _fill_cols:
                df[col] = df[col].fillna(prev_idx[col].reindex(df.index))
            df = df.reset_index()
        except Exception as exc:
            log.warning("Could not merge with previous odds file: %s", exc)

    df.to_parquet(output_path, engine="pyarrow", index=False)
    log.info("Saved %s — shape: %s", output_path.name, df.shape)
    print(f"Saved {output_path.name} — shape: {df.shape}")

    # ── Step 5b: Append snapshot to history file (for CLV tracking) ──────────
    # Each run appends a timestamped copy so we can compare morning vs closing odds.
    history_path = OUTPUT_DIR / f"odds_history_{target_date.replace('-', '_')}.parquet"
    snap = df.copy()
    snap["snapshot_time"] = datetime.utcnow()
    if history_path.exists():
        try:
            prev = pd.read_parquet(history_path, engine="pyarrow")
            snap = pd.concat([prev, snap], ignore_index=True)
        except Exception:
            pass
    snap.to_parquet(history_path, engine="pyarrow", index=False)
    log.info("Odds history snapshot appended → %s (%d total rows)", history_path.name, len(snap))

    # Summary
    with_retail   = df[df["close_ml_home"].notna()]
    with_pinnacle = df[df["P_true_home"].notna()]
    missing       = df[df["close_ml_home"].isna()]
    print(
        f"  Retail ML    : {len(with_retail)}/{len(df)}"
        f"  |  Pinnacle P_true : {len(with_pinnacle)}/{len(df)}"
        f"  |  Missing    : {len(missing)}"
        f"  |  Sources    : {df['source'].value_counts().to_dict()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily MLB odds pull — dual-region (US retail + EU Pinnacle)"
    )
    parser.add_argument(
        "--date",
        default=date.today().strftime("%Y-%m-%d"),
        help="Date to pull odds for (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help=(
            "Bypass the API cache and re-fetch from The Odds API even if a "
            "cache file already exists. Useful after West Coast lines post."
        ),
    )
    args = parser.parse_args()
    date_str = args.date

    pull_odds_for_date(date_str, force=args.force)

    # Pitcher K props — US region, per-event endpoint, quota-aware
    k_props = fetch_pitcher_strikeout_props(date_str)
    if not k_props.empty:
        log.info(
            "Pitcher K props: %d rows fetched for %d pitchers",
            len(k_props),
            k_props["pitcher_name"].nunique(),
        )


if __name__ == "__main__":
    main()
