"""
odds_current_pull.py
====================
Daily odds pull using a 3-source waterfall:
  1. OddsPortal historical cache (read-only parquets from odds_historical_pull.py)
  2. The Odds API (quota-guarded, current season)
  3. ActionNetwork (scraper fallback, sole source for public betting %)

Usage:
    python odds_current_pull.py                   # pull today's odds
    python odds_current_pull.py --date 2026-04-11 # specific date

Requires:
    pip install requests beautifulsoup4 pandas pyarrow python-dotenv

Environment variables (in .env file or system):
    ODDS_API_KEY=<your key from the-odds-api.com>
"""

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime, date, timedelta
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
OUTPUT_DIR = Path("./statcast_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ODDS_API_CACHE_DIR = OUTPUT_DIR / "odds_api_cache"
ODDS_API_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ODDS_API_KEY: Optional[str] = os.getenv("ODDS_API_KEY")
ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
ODDS_API_QUOTA_THRESHOLD = 50

ACTIONNETWORK_URL = "https://www.actionnetwork.com/mlb/odds"

QUOTA_LOG = OUTPUT_DIR / "odds_api_quota.log"
MISSING_LOG = OUTPUT_DIR / "odds_missing.log"

HISTORICAL_YEARS = [2023, 2024, 2025]

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
# Team name normalisation (shared with odds_historical_pull.py)
# ---------------------------------------------------------------------------
TEAM_ABBR_MAP: dict[str, str] = {
    "arizona diamondbacks": "ARI", "diamondbacks": "ARI", "arizona": "ARI",
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
    "athletics": "OAK", "oakland athletics": "OAK", "oakland": "OAK",
    "las vegas athletics": "OAK",
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
    # Odds API uses full names
    "chicago cubs": "CHC",
    "chicago white sox": "CWS",
    "new york mets": "NYM",
    "new york yankees": "NYY",
}

# Odds API bookmaker → internal key mapping
ODDS_API_BOOKS = ["draftkings", "fanduel"]

SCHEMA_COLS = [
    "game_date", "game_pk", "home_team", "away_team",
    "open_ml_home", "close_ml_home", "open_ml_away", "close_ml_away",
    "open_total", "close_total", "runline_home", "runline_home_odds",
    "public_pct_home", "public_pct_over", "source", "pull_timestamp",
]


def normalize_team(raw: str) -> str:
    cleaned = raw.strip().lower()
    if cleaned in TEAM_ABBR_MAP:
        return TEAM_ABBR_MAP[cleaned]
    for key, abbr in sorted(TEAM_ABBR_MAP.items(), key=lambda x: -len(x[0])):
        if key in cleaned:
            return abbr
    log.warning("Unknown team name: %r", raw)
    return raw.upper()[:3]


def _empty_odds_dict(game_date: str, home_team: str, away_team: str) -> dict:
    """Return a blank odds dict with all nullable fields as None."""
    return {
        "game_date": game_date,
        "game_pk": None,
        "home_team": home_team,
        "away_team": away_team,
        "open_ml_home": None,
        "close_ml_home": None,
        "open_ml_away": None,
        "close_ml_away": None,
        "open_total": None,
        "close_total": None,
        "runline_home": -1.5,
        "runline_home_odds": None,
        "public_pct_home": None,
        "public_pct_over": None,
        "source": None,
        "pull_timestamp": datetime.utcnow(),
    }


# ---------------------------------------------------------------------------
# Source 1: Historical OddsPortal cache
# ---------------------------------------------------------------------------
_historical_cache: dict[int, pd.DataFrame] = {}


def _load_historical_cache(year: int) -> pd.DataFrame:
    """Load and cache the OddsPortal historical parquet for a given year."""
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
            _historical_cache[year] = pd.DataFrame(columns=SCHEMA_COLS)
    return _historical_cache[year]


def get_odds_from_historical(game_date: str, home_team: str, away_team: str) -> Optional[dict]:
    """Look up a game in the OddsPortal historical cache."""
    year = int(game_date[:4])
    if year not in HISTORICAL_YEARS:
        return None
    df = _load_historical_cache(year)
    if df.empty:
        return None
    mask = (df["game_date"] == game_date) & (df["home_team"] == home_team) & (df["away_team"] == away_team)
    rows = df[mask]
    if rows.empty:
        return None
    row = rows.iloc[0].to_dict()
    row["source"] = "oddsportal"
    return row


# ---------------------------------------------------------------------------
# Source 2: The Odds API
# ---------------------------------------------------------------------------
_odds_api_quota_remaining: Optional[int] = None
_odds_api_quota_used: Optional[int] = None
_odds_api_data_cache: Optional[list[dict]] = None


def _log_quota(remaining: Optional[int], used: Optional[int]) -> None:
    """Append quota info to odds_api_quota.log."""
    line = f"{datetime.utcnow().isoformat()} | requests_remaining: {remaining} | requests_used: {used}\n"
    with open(QUOTA_LOG, "a", encoding="utf-8") as fh:
        fh.write(line)
    log.info("Odds API quota — remaining: %s | used: %s", remaining, used)


def fetch_odds_api(target_date: str) -> Optional[list[dict]]:
    """
    Fetch today's odds from The Odds API.
    Returns list of raw game dicts, or None if quota exhausted / key missing / error.
    Caches the raw response to ./statcast_data/odds_api_cache/{date}.json.
    """
    global _odds_api_quota_remaining, _odds_api_quota_used, _odds_api_data_cache

    if not ODDS_API_KEY:
        log.warning("ODDS_API_KEY not set — skipping Odds API")
        return None

    # Check cached file first
    cache_file = ODDS_API_CACHE_DIR / f"{target_date}.json"
    if cache_file.exists():
        log.info("Loading Odds API response from cache: %s", cache_file)
        with open(cache_file, encoding="utf-8") as fh:
            return json.load(fh)

    # Pre-check quota if we already know it
    if _odds_api_quota_remaining is not None and _odds_api_quota_remaining < ODDS_API_QUOTA_THRESHOLD:
        log.warning(
            "Odds API quota low (%d remaining) — skipping Odds API", _odds_api_quota_remaining
        )
        _log_quota(_odds_api_quota_remaining, _odds_api_quota_used)
        return None

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,totals,spreads",
        "bookmakers": ",".join(ODDS_API_BOOKS),
        "oddsFormat": "american",
        "dateFormat": "iso",
        "commenceTimeFrom": f"{target_date}T00:00:00Z",
        "commenceTimeTo": f"{target_date}T23:59:59Z",
    }

    try:
        resp = requests.get(ODDS_API_BASE, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.error("Odds API request failed: %s", exc)
        return None

    # Extract quota headers
    remaining = resp.headers.get("x-requests-remaining")
    used = resp.headers.get("x-requests-used")
    if remaining is not None:
        _odds_api_quota_remaining = int(remaining)
    if used is not None:
        _odds_api_quota_used = int(used)
    _log_quota(_odds_api_quota_remaining, _odds_api_quota_used)

    # Post-fetch quota guard
    if _odds_api_quota_remaining is not None and _odds_api_quota_remaining < ODDS_API_QUOTA_THRESHOLD:
        log.warning(
            "Odds API quota now low (%d remaining) — results returned but future calls blocked",
            _odds_api_quota_remaining,
        )

    data = resp.json()

    # Cache raw response
    with open(cache_file, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    log.info("Odds API response cached: %s (%d games)", cache_file, len(data))

    return data


def _parse_odds_api_game(game: dict) -> dict:
    """
    Convert a single Odds API game object to the standardised odds dict.
    Prefers DraftKings, falls back to FanDuel.
    """
    home_raw = game.get("home_team", "")
    away_raw = game.get("away_team", "")
    home_team = normalize_team(home_raw)
    away_team = normalize_team(away_raw)

    commence_time = game.get("commence_time", "")
    game_date = commence_time[:10] if commence_time else None

    result = _empty_odds_dict(game_date or "", home_team, away_team)
    result["source"] = "odds_api"

    bookmakers = game.get("bookmakers", [])
    # Prefer DraftKings, then FanDuel
    preferred = None
    for bk_name in ODDS_API_BOOKS:
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

    for market in preferred.get("markets", []):
        key = market.get("key", "")
        outcomes = market.get("outcomes", [])

        if key == "h2h":
            for out in outcomes:
                name = normalize_team(out.get("name", ""))
                price = out.get("price")
                if name == home_team:
                    result["close_ml_home"] = int(price) if price is not None else None
                elif name == away_team:
                    result["close_ml_away"] = int(price) if price is not None else None

        elif key == "totals":
            for out in outcomes:
                if out.get("name", "").lower() == "over":
                    result["close_total"] = float(out["point"]) if "point" in out else None
                    break

        elif key == "spreads":
            for out in outcomes:
                name = normalize_team(out.get("name", ""))
                if name == home_team:
                    result["runline_home"] = float(out["point"]) if "point" in out else -1.5
                    result["runline_home_odds"] = int(out["price"]) if "price" in out else None
                    break

    return result


def get_odds_from_api(game_date: str, home_team: str, away_team: str) -> Optional[dict]:
    """Look up a specific game from the Odds API data for that date."""
    global _odds_api_data_cache

    if _odds_api_data_cache is None:
        _odds_api_data_cache = fetch_odds_api(game_date) or []

    for game in _odds_api_data_cache:
        ht = normalize_team(game.get("home_team", ""))
        at = normalize_team(game.get("away_team", ""))
        commence = game.get("commence_time", "")[:10]
        if commence == game_date and ht == home_team and at == away_team:
            return _parse_odds_api_game(game)

    return None


# ---------------------------------------------------------------------------
# Source 3: ActionNetwork scraper
# ---------------------------------------------------------------------------
_actionnetwork_cache: Optional[list[dict]] = None


def fetch_actionnetwork() -> list[dict]:
    """
    Scrape ActionNetwork MLB odds page.
    Returns list of raw game dicts with moneyline, total, and public %.
    """
    global _actionnetwork_cache
    if _actionnetwork_cache is not None:
        return _actionnetwork_cache

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.actionnetwork.com/",
    }

    games = []
    try:
        resp = requests.get(ACTIONNETWORK_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        games = _parse_actionnetwork_html(soup)
        log.info("ActionNetwork: scraped %d games", len(games))
    except requests.RequestException as exc:
        log.error("ActionNetwork request failed: %s", exc)
    except Exception as exc:
        log.error("ActionNetwork parse error: %s", exc, exc_info=True)

    _actionnetwork_cache = games
    return games


def _parse_actionnetwork_html(soup: BeautifulSoup) -> list[dict]:
    """
    Parse ActionNetwork odds HTML.
    ActionNetwork frequently updates its markup — this targets both the legacy
    and current layouts using class heuristics and JSON-LD script tags.
    """
    games = []
    today_str = date.today().strftime("%Y-%m-%d")
    pull_ts = datetime.utcnow()

    # Attempt 1: JSON-LD / __NEXT_DATA__ embedded script (most reliable)
    for script in soup.find_all("script", {"id": "__NEXT_DATA__"}):
        try:
            data = json.loads(script.string or "")
            page_props = data.get("props", {}).get("pageProps", {})
            game_list = (
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

    # Attempt 2: HTML structure — game rows
    game_rows = soup.find_all(
        "div", class_=re.compile(r"game-line|matchup|odds-table|event-row", re.I)
    )
    for row in game_rows:
        parsed = _parse_actionnetwork_html_row(row, today_str, pull_ts)
        if parsed:
            games.append(parsed)

    return games


def _parse_actionnetwork_game(g: dict, game_date: str, pull_ts: datetime) -> Optional[dict]:
    """Parse a game dict from ActionNetwork's __NEXT_DATA__ JSON."""
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

        # Moneyline
        odds = g.get("odds", [{}])
        first_book = odds[0] if odds else {}

        close_ml_home = first_book.get("ml_home") or first_book.get("moneyline_home")
        close_ml_away = first_book.get("ml_away") or first_book.get("moneyline_away")
        close_total = first_book.get("total") or first_book.get("over_under")

        # Public betting %
        public_pct_home = None
        public_pct_over = None
        public = g.get("public_betting", {}) or g.get("publicBetting", {})
        if public:
            home_pct = public.get("home_ml_percent") or public.get("homeMlPercent")
            over_pct = public.get("over_percent") or public.get("overPercent")
            public_pct_home = float(home_pct) / 100 if home_pct is not None else None
            public_pct_over = float(over_pct) / 100 if over_pct is not None else None

        # Runline
        rl_home = first_book.get("spread_home") or first_book.get("run_line_home", -1.5)
        rl_home_odds = first_book.get("spread_home_odds") or first_book.get("run_line_home_odds")

        # Scheduled date from API
        scheduled = g.get("scheduled") or g.get("start_time") or ""
        gd = scheduled[:10] if scheduled else game_date

        return {
            "game_date": gd,
            "game_pk": None,
            "home_team": home_team,
            "away_team": away_team,
            "open_ml_home": None,
            "close_ml_home": int(close_ml_home) if close_ml_home is not None else None,
            "open_ml_away": None,
            "close_ml_away": int(close_ml_away) if close_ml_away is not None else None,
            "open_total": None,
            "close_total": float(close_total) if close_total is not None else None,
            "runline_home": float(rl_home) if rl_home is not None else -1.5,
            "runline_home_odds": int(rl_home_odds) if rl_home_odds is not None else None,
            "public_pct_home": public_pct_home,
            "public_pct_over": public_pct_over,
            "source": "actionnetwork",
            "pull_timestamp": pull_ts,
        }
    except Exception as exc:
        log.debug("Failed to parse ActionNetwork game JSON: %s", exc)
        return None


def _parse_actionnetwork_html_row(row, game_date: str, pull_ts: datetime) -> Optional[dict]:
    """Fallback HTML row parser for ActionNetwork."""
    try:
        team_els = row.find_all(
            class_=re.compile(r"team-name|team_name|teamName|participant", re.I)
        )
        if len(team_els) < 2:
            return None

        home_raw = team_els[1].get_text(strip=True)
        away_raw = team_els[0].get_text(strip=True)
        if not home_raw or not away_raw:
            return None

        home_team = normalize_team(home_raw)
        away_team = normalize_team(away_raw)

        # Moneyline odds
        ml_els = row.find_all(class_=re.compile(r"moneyline|money-line|ml-odds", re.I))
        close_ml_away = None
        close_ml_home = None
        if len(ml_els) >= 2:
            close_ml_away = _text_to_int(ml_els[0].get_text(strip=True))
            close_ml_home = _text_to_int(ml_els[1].get_text(strip=True))
        elif ml_els:
            close_ml_home = _text_to_int(ml_els[0].get_text(strip=True))

        # Total
        total_el = row.find(class_=re.compile(r"total|over-under", re.I))
        close_total = None
        if total_el:
            raw_total = re.search(r"[\d.]+", total_el.get_text())
            close_total = float(raw_total.group()) if raw_total else None

        # Public %
        public_els = row.find_all(class_=re.compile(r"public|consensus|bets-pct", re.I))
        public_pct_home = None
        public_pct_over = None
        if len(public_els) >= 2:
            hp = re.search(r"([\d.]+)%?", public_els[-1].get_text())
            op = re.search(r"([\d.]+)%?", public_els[0].get_text())
            public_pct_home = float(hp.group(1)) / 100 if hp else None
            public_pct_over = float(op.group(1)) / 100 if op else None

        return {
            "game_date": game_date,
            "game_pk": None,
            "home_team": home_team,
            "away_team": away_team,
            "open_ml_home": None,
            "close_ml_home": close_ml_home,
            "open_ml_away": None,
            "close_ml_away": close_ml_away,
            "open_total": None,
            "close_total": close_total,
            "runline_home": -1.5,
            "runline_home_odds": None,
            "public_pct_home": public_pct_home,
            "public_pct_over": public_pct_over,
            "source": "actionnetwork",
            "pull_timestamp": pull_ts,
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


def get_odds_from_actionnetwork(game_date: str, home_team: str, away_team: str) -> Optional[dict]:
    """Look up a game from the ActionNetwork scrape cache."""
    games = fetch_actionnetwork()
    for g in games:
        if g["home_team"] == home_team and g["away_team"] == away_team:
            g["game_date"] = game_date  # Override with requested date
            return g
    return None


# ---------------------------------------------------------------------------
# Waterfall function
# ---------------------------------------------------------------------------
def get_odds_for_game(
    game_date: str,
    home_team: str,
    away_team: str,
    year: Optional[int] = None,
) -> Optional[dict]:
    """
    Returns standardised odds dict. Falls through sources in priority order:
      1. OddsPortal historical cache (2023-2025 completed games)
      2. The Odds API (quota-guarded, current/recent games)
      3. ActionNetwork (scraper fallback; only source for public %)
    Logs to odds_missing.log if all sources exhausted.
    """
    if year is None:
        year = int(game_date[:4])

    log.debug("Waterfall lookup: %s %s @ %s", game_date, away_team, home_team)

    # --- Source 1: Historical OddsPortal cache ---
    result = get_odds_from_historical(game_date, home_team, away_team)
    if result:
        log.debug("Source 1 (historical) hit: %s %s @ %s", game_date, away_team, home_team)
        return result

    # --- Source 2: Odds API ---
    result = get_odds_from_api(game_date, home_team, away_team)
    if result:
        log.debug("Source 2 (odds_api) hit: %s %s @ %s", game_date, away_team, home_team)
        return result

    # --- Source 3: ActionNetwork ---
    result = get_odds_from_actionnetwork(game_date, home_team, away_team)
    if result:
        log.debug("Source 3 (actionnetwork) hit: %s %s @ %s", game_date, away_team, home_team)
        return result

    # All sources exhausted — log and return None
    log.warning("No odds found: %s %s @ %s", game_date, away_team, home_team)
    _log_missing(game_date, home_team, away_team)
    return None


def _log_missing(game_date: str, home_team: str, away_team: str) -> None:
    line = f"{datetime.utcnow().isoformat()} | {game_date} | {away_team} @ {home_team}\n"
    with open(MISSING_LOG, "a", encoding="utf-8") as fh:
        fh.write(line)


# ---------------------------------------------------------------------------
# Schedule loader (to get list of games for a given date)
# ---------------------------------------------------------------------------
def load_schedule_for_date(target_date: str) -> list[dict]:
    """
    Load game schedule for the target date from available mlb_schedule parquets.
    Returns list of {game_date, home_team, away_team, game_pk} dicts.
    """
    year = int(target_date[:4])
    schedule_path = OUTPUT_DIR / f"mlb_schedule_{year}.parquet"
    if not schedule_path.exists():
        # Fallback: try schedule_all_{year}.parquet
        schedule_path = OUTPUT_DIR / f"schedule_all_{year}.parquet"
    if not schedule_path.exists():
        log.warning("No schedule file found for year %d", year)
        return []

    df = pd.read_parquet(schedule_path, engine="pyarrow")
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if not date_cols:
        log.warning("No date column in schedule file")
        return []

    date_col = date_cols[0]
    df[date_col] = df[date_col].astype(str).str[:10]
    day_df = df[df[date_col] == target_date]

    games = []
    for _, row in day_df.iterrows():
        home = str(row.get("home_team") or row.get("home") or "")
        away = str(row.get("away_team") or row.get("away") or "")
        pk = row.get("game_pk") or row.get("gamePk") or row.get("game_id")
        if home and away:
            games.append({
                "game_date": target_date,
                "home_team": normalize_team(home),
                "away_team": normalize_team(away),
                "game_pk": int(pk) if pk is not None and str(pk) != "nan" else None,
            })
    return games


# ---------------------------------------------------------------------------
# Main pull logic
# ---------------------------------------------------------------------------
def _to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Convert list of odds dicts to standardised DataFrame."""
    if not records:
        return pd.DataFrame(columns=SCHEMA_COLS)
    df = pd.DataFrame(records)
    for col in SCHEMA_COLS:
        if col not in df.columns:
            df[col] = None
    df = df[SCHEMA_COLS]
    df["game_date"] = df["game_date"].astype(str)
    for int_col in ["game_pk", "open_ml_home", "close_ml_home", "open_ml_away", "close_ml_away",
                    "runline_home_odds"]:
        df[int_col] = pd.to_numeric(df[int_col], errors="coerce").astype("Int64")
    for float_col in ["open_total", "close_total", "runline_home", "public_pct_home", "public_pct_over"]:
        df[float_col] = pd.to_numeric(df[float_col], errors="coerce")
    df["pull_timestamp"] = pd.to_datetime(df["pull_timestamp"])
    return df


def pull_odds_for_date(target_date: str) -> None:
    """Pull odds for all games on target_date and save to parquet."""
    log.info("Pulling odds for date: %s", target_date)
    output_path = OUTPUT_DIR / f"odds_current_{target_date.replace('-', '_')}.parquet"

    games = load_schedule_for_date(target_date)

    # If no schedule file, try to pull all available from Odds API / ActionNetwork directly
    if not games:
        log.info("No schedule file — pulling all available games from Odds API + ActionNetwork")
        api_data = fetch_odds_api(target_date) or []
        for g in api_data:
            ht = normalize_team(g.get("home_team", ""))
            at = normalize_team(g.get("away_team", ""))
            games.append({"game_date": target_date, "home_team": ht, "away_team": at, "game_pk": None})

        an_data = fetch_actionnetwork()
        existing_keys = {(g["home_team"], g["away_team"]) for g in games}
        for g in an_data:
            key = (g["home_team"], g["away_team"])
            if key not in existing_keys:
                games.append({
                    "game_date": target_date,
                    "home_team": g["home_team"],
                    "away_team": g["away_team"],
                    "game_pk": None,
                })

    if not games:
        log.warning("No games found for %s — nothing to pull", target_date)
        return

    log.info("Processing %d games for %s", len(games), target_date)

    results = []
    for g in games:
        odds = get_odds_for_game(
            game_date=g["game_date"],
            home_team=g["home_team"],
            away_team=g["away_team"],
        )
        if odds:
            if g.get("game_pk"):
                odds["game_pk"] = g["game_pk"]
            results.append(odds)
        else:
            # Include a skeleton row with no odds data
            skeleton = _empty_odds_dict(g["game_date"], g["home_team"], g["away_team"])
            skeleton["game_pk"] = g.get("game_pk")
            results.append(skeleton)

    df = _to_dataframe(results)
    df = df.drop_duplicates(subset=["game_date", "home_team", "away_team"], keep="first")
    df = df.sort_values(["game_date", "home_team"]).reset_index(drop=True)

    df.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"Saved {output_path.name} — shape: {df.shape}")
    log.info("Saved %s — shape: %s", output_path.name, df.shape)

    # Summary
    sourced = df[df["source"].notna() & (df["close_ml_home"].notna())]
    missing = df[df["close_ml_home"].isna()]
    print(
        f"  With ML odds : {len(sourced)}/{len(df)}"
        f"  |  Missing: {len(missing)}"
        f"  |  Sources: {df['source'].value_counts().to_dict()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily MLB odds pull (3-source waterfall)")
    parser.add_argument(
        "--date",
        default=date.today().strftime("%Y-%m-%d"),
        help="Date to pull odds for (YYYY-MM-DD). Defaults to today.",
    )
    args = parser.parse_args()
    pull_odds_for_date(args.date)


if __name__ == "__main__":
    main()
