"""
build_lines_backtest.py — Adds historical betting lines to backtest game files.

Data source priority:
  1. The Odds API historical endpoint  (requires paid "Extras" plan)
     — upgrade at https://the-odds-api.com/#pricing
  2. SportsBookReviewsOnline (SBRO) Excel download  (free, if accessible)
  3. Manual CSV fallback  (place sbro_cache/manual_lines_YYYY.csv)

Adds columns to backtest files:
  market_total_game   closing game O/U total
  market_total_f5     estimated as 55.5% of game total
  market_total_f3     estimated as 32.0% of game total
  market_total_f1     estimated as 11.0% of game total
  market_ml_home      home team moneyline (American odds)
  market_ml_away      away team moneyline (American odds)
  lines_source        "odds_api", "sbro", "manual", or "missing"

After running, re-run calibrate_weights.py to calibrate batter-sensitive models
(MFull, MF3i, MF1i) using actual market lines.

Usage:
    python build_lines_backtest.py              # 2024 + 2025
    python build_lines_backtest.py --year 2025  # single season
    python build_lines_backtest.py --source sbro      # force SBRO
    python build_lines_backtest.py --source odds_api  # force Odds API
    python build_lines_backtest.py --no-download      # use cached files only
"""

import argparse
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

# ─── ODDS API (HISTORICAL) ────────────────────────────────────────────────────
# Requires paid "Extras" plan at https://the-odds-api.com/#pricing
# Free tier returns 401. Once upgraded, this fetches closing lines per game date.

ODDS_API_HIST_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds-history/"
ODDS_API_CACHE_DIR = Path("./odds_api_cache")

# Game time to query (23:00 UTC ≈ closing line, covers all US time zones)
ODDS_API_QUERY_TIME = "T23:00:00Z"

# ─── SBRO FILE URLS ──────────────────────────────────────────────────────────
# SBRO changed their URL structure in 2025. Try multiple patterns.

SBRO_URLS = {
    2024: [
        "https://www.sportsbookreviewsonline.com/scoresoddsarchives/mlb/mlb-odds-2024.xlsx",
        "https://www.sportsbookreviewsonline.com/scoresoddsarchives/mlb/mlb%20odds%202024.xlsx",
    ],
    2025: [
        "https://www.sportsbookreviewsonline.com/scoresoddsarchives/mlb/mlb-odds-2025.xlsx",
        "https://www.sportsbookreviewsonline.com/scoresoddsarchives/mlb/mlb%20odds%202025.xlsx",
    ],
}

SBRO_CACHE_DIR = Path("./sbro_cache")

BACKTEST_FILES = {
    2024: "backtest_games_2024.csv",
    2025: "backtest_games_2025.csv",
}

# F-inning total ratios (same as score_models.py)
F5_RATIO = 0.555
F3_RATIO = 0.320
F1_RATIO = 0.110

# ─── TEAM NAME → ABBREVIATION MAP ────────────────────────────────────────────
# SBRO uses full team names; we need our pipeline abbreviations.

SBRO_TEAM_MAP = {
    # Full names
    "Arizona Diamondbacks":    "ARI",
    "Atlanta Braves":          "ATL",
    "Baltimore Orioles":       "BAL",
    "Boston Red Sox":          "BOS",
    "Chicago Cubs":            "CHC",
    "Chicago White Sox":       "CWS",
    "Cincinnati Reds":         "CIN",
    "Cleveland Guardians":     "CLE",
    "Colorado Rockies":        "COL",
    "Detroit Tigers":          "DET",
    "Houston Astros":          "HOU",
    "Kansas City Royals":      "KC",
    "Los Angeles Angels":      "LAA",
    "Los Angeles Dodgers":     "LAD",
    "Miami Marlins":           "MIA",
    "Milwaukee Brewers":       "MIL",
    "Minnesota Twins":         "MIN",
    "New York Mets":           "NYM",
    "New York Yankees":        "NYY",
    "Oakland Athletics":       "OAK",
    "Philadelphia Phillies":   "PHI",
    "Pittsburgh Pirates":      "PIT",
    "San Diego Padres":        "SD",
    "San Francisco Giants":    "SF",
    "Seattle Mariners":        "SEA",
    "St. Louis Cardinals":     "STL",
    "Tampa Bay Rays":          "TB",
    "Texas Rangers":           "TEX",
    "Toronto Blue Jays":       "TOR",
    "Washington Nationals":    "WSH",
    # Athletics relocated to Sacramento for 2025
    "Sacramento Athletics":    "OAK",
    "Athletics":               "OAK",
    # Shortened versions SBRO sometimes uses
    "Arizona":                 "ARI",
    "Atlanta":                 "ATL",
    "Baltimore":               "BAL",
    "Boston":                  "BOS",
    "Chi Cubs":                "CHC",
    "Chi White Sox":           "CWS",
    "Cincinnati":              "CIN",
    "Cleveland":               "CLE",
    "Colorado":                "COL",
    "Detroit":                 "DET",
    "Houston":                 "HOU",
    "Kansas City":             "KC",
    "LA Angels":               "LAA",
    "LA Dodgers":              "LAD",
    "Miami":                   "MIA",
    "Milwaukee":               "MIL",
    "Minnesota":               "MIN",
    "NY Mets":                 "NYM",
    "NY Yankees":              "NYY",
    "Oakland":                 "OAK",
    "Philadelphia":            "PHI",
    "Pittsburgh":              "PIT",
    "San Diego":               "SD",
    "San Francisco":           "SF",
    "Seattle":                 "SEA",
    "St. Louis":               "STL",
    "Tampa Bay":               "TB",
    "Texas":                   "TEX",
    "Toronto":                 "TOR",
    "Washington":              "WSH",
    "Sacramento":              "OAK",
}


# ─── ODDS API TEAM NAME MAP ───────────────────────────────────────────────────
# The Odds API uses full city+team names

ODDS_API_TEAM_MAP = {
    "Arizona Diamondbacks":    "ARI",
    "Atlanta Braves":          "ATL",
    "Baltimore Orioles":       "BAL",
    "Boston Red Sox":          "BOS",
    "Chicago Cubs":            "CHC",
    "Chicago White Sox":       "CWS",
    "Cincinnati Reds":         "CIN",
    "Cleveland Guardians":     "CLE",
    "Colorado Rockies":        "COL",
    "Detroit Tigers":          "DET",
    "Houston Astros":          "HOU",
    "Kansas City Royals":      "KC",
    "Los Angeles Angels":      "LAA",
    "Los Angeles Dodgers":     "LAD",
    "Miami Marlins":           "MIA",
    "Milwaukee Brewers":       "MIL",
    "Minnesota Twins":         "MIN",
    "New York Mets":           "NYM",
    "New York Yankees":        "NYY",
    "Oakland Athletics":       "OAK",
    "Philadelphia Phillies":   "PHI",
    "Pittsburgh Pirates":      "PIT",
    "San Diego Padres":        "SD",
    "San Francisco Giants":    "SF",
    "Seattle Mariners":        "SEA",
    "St. Louis Cardinals":     "STL",
    "Tampa Bay Rays":          "TB",
    "Texas Rangers":           "TEX",
    "Toronto Blue Jays":       "TOR",
    "Washington Nationals":    "WSH",
    "Sacramento Athletics":    "OAK",
}


# ─── ODDS API HISTORICAL FETCH ────────────────────────────────────────────────

def fetch_odds_api_historical(games_df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """
    Fetch closing lines from The Odds API historical endpoint for each game date.

    Requires paid "Extras" plan. Free tier returns 401.
    Caches each date's response to odds_api_cache/YYYY-MM-DD.json.

    Returns DataFrame with same schema as parse_sbro():
      game_date, home_abbr, away_abbr, market_total_game, market_ml_home, market_ml_away
    """
    ODDS_API_CACHE_DIR.mkdir(exist_ok=True)

    dates = sorted(games_df["game_date"].astype(str).str[:10].unique())
    print(f"  Odds API: fetching {len(dates)} game dates ...")

    all_lines = []
    fetched = cached = errors = 0

    for date_str in dates:
        cache_file = ODDS_API_CACHE_DIR / f"{date_str}.json"

        if cache_file.exists():
            raw = json.loads(cache_file.read_text())
            cached += 1
        else:
            url = (
                f"{ODDS_API_HIST_URL}"
                f"?apiKey={api_key}"
                f"&regions=us"
                f"&markets=totals,h2h"
                f"&bookmakers=draftkings,fanduel"
                f"&oddsFormat=american"
                f"&date={date_str}{ODDS_API_QUERY_TIME}"
            )
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "MLB-Model/1.0"})
                with urllib.request.urlopen(req, timeout=15) as r:
                    remaining = r.headers.get("x-requests-remaining", "?")
                    raw = json.loads(r.read())
                cache_file.write_text(json.dumps(raw))
                fetched += 1
                if fetched % 20 == 0:
                    print(f"    {fetched}/{len(dates) - cached} fetched, {remaining} credits remaining")
                time.sleep(0.2)  # be polite
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    print(f"\n  Odds API 401 — historical data requires a paid Extras plan.")
                    print(f"  Upgrade at: https://the-odds-api.com/#pricing")
                    return pd.DataFrame()
                errors += 1
                continue
            except Exception as e:
                errors += 1
                continue

        # Parse the response (data key for historical endpoint)
        events = raw if isinstance(raw, list) else raw.get("data", [])
        for event in events:
            home_name = event.get("home_team", "")
            away_name = event.get("away_team", "")
            home_abbr = ODDS_API_TEAM_MAP.get(home_name, "")
            away_abbr = ODDS_API_TEAM_MAP.get(away_name, "")
            if not home_abbr or not away_abbr:
                continue

            total = ml_home = ml_away = None
            for book in event.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market["key"] == "totals":
                        for o in market.get("outcomes", []):
                            if o["name"] == "Over" and total is None:
                                total = o.get("point")
                    elif market["key"] == "h2h":
                        for o in market.get("outcomes", []):
                            if o["name"] == home_name and ml_home is None:
                                ml_home = o.get("price")
                            elif o["name"] == away_name and ml_away is None:
                                ml_away = o.get("price")

            all_lines.append({
                "game_date":         date_str,
                "home_abbr":         home_abbr,
                "away_abbr":         away_abbr,
                "market_total_game": total,
                "market_ml_home":    ml_home,
                "market_ml_away":    ml_away,
            })

    if errors:
        print(f"  {errors} dates failed")
    print(f"  Odds API: {fetched} fetched, {cached} from cache, {len(all_lines)} game lines")
    return pd.DataFrame(all_lines) if all_lines else pd.DataFrame()


def normalize_team(name: str) -> str:
    """Map SBRO team name to pipeline abbreviation."""
    if not isinstance(name, str):
        return ""
    name = name.strip()
    if name in SBRO_TEAM_MAP:
        return SBRO_TEAM_MAP[name]
    # Fuzzy: check if any key is a substring
    name_lower = name.lower()
    for k, v in SBRO_TEAM_MAP.items():
        if k.lower() in name_lower or name_lower in k.lower():
            return v
    return name.upper()


# ─── SBRO DOWNLOAD ───────────────────────────────────────────────────────────

def download_sbro(year: int, force: bool = False) -> Path:
    """Download SBRO Excel file; tries multiple URLs; caches locally."""
    SBRO_CACHE_DIR.mkdir(exist_ok=True)
    cache_path = SBRO_CACHE_DIR / f"mlb-odds-{year}.xlsx"

    if cache_path.exists() and not force:
        # Verify it's actually an Excel file (not an HTML redirect we cached before)
        data = cache_path.read_bytes()
        if data[:4] in (b'PK\x03\x04', b'\xd0\xcf\x11\xe0') or b'<html' not in data[:200].lower():
            print(f"  Using cached {cache_path}")
            return cache_path
        else:
            print(f"  Cached file is HTML — re-downloading")

    urls = SBRO_URLS.get(year, [])
    if not urls:
        raise ValueError(f"No SBRO URL configured for {year}")

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    for url in urls:
        print(f"  Trying {url} ...", end="", flush=True)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            # Reject HTML responses
            if b'<html' in data[:500].lower() or b'<!doctype' in data[:500].lower():
                print(f" HTML redirect (skipping)")
                continue
            cache_path.write_bytes(data)
            print(f" {len(data):,} bytes")
            return cache_path
        except Exception as e:
            print(f" failed ({e})")

    raise RuntimeError(
        f"All SBRO URLs failed for {year}.\n"
        f"Download manually from sportsbookreviewsonline.com and save to:\n"
        f"  {cache_path}"
    )


# ─── SBRO PARSER ─────────────────────────────────────────────────────────────

def parse_sbro(path: Path) -> pd.DataFrame:
    """
    Parse SBRO Excel file into a normalized DataFrame.

    SBRO format: two rows per game (visitor row, then home row).
    Typical columns: Date, Rot, VH, Team, 1st, 2nd, 3rd, 4th, Final, Open, Close, ML, 2H

    Returns DataFrame with one row per game:
      game_date, home_abbr, away_abbr,
      market_total_game (closing total),
      market_ml_home, market_ml_away
    """
    print(f"  Parsing {path.name} ...", end="", flush=True)
    try:
        raw = pd.read_excel(path, header=0, dtype=str)
    except Exception as e:
        print(f"\n  ERROR reading Excel: {e}")
        return pd.DataFrame()

    raw.columns = [str(c).strip().lower() for c in raw.columns]
    print(f" {len(raw)} rows, columns: {list(raw.columns)}")

    # Detect key columns (SBRO column names vary slightly by year)
    col_map = _detect_sbro_columns(raw.columns.tolist())
    if not col_map:
        print("  ERROR: Could not detect required SBRO columns")
        print(f"  Available columns: {raw.columns.tolist()}")
        return pd.DataFrame()

    date_col  = col_map["date"]
    vh_col    = col_map["vh"]
    team_col  = col_map["team"]
    close_col = col_map["close"]
    ml_col    = col_map["ml"]

    games = []
    i = 0
    while i + 1 < len(raw):
        row_v = raw.iloc[i]
        row_h = raw.iloc[i + 1]

        # Expect V row followed by H row
        vh_v = str(row_v.get(vh_col, "")).strip().upper()
        vh_h = str(row_h.get(vh_col, "")).strip().upper()

        if vh_v not in ("V", "VISITOR") or vh_h not in ("H", "HOME"):
            i += 1
            continue

        # Parse date
        raw_date = str(row_v.get(date_col, "")).strip()
        game_date = _parse_sbro_date(raw_date)
        if not game_date:
            i += 2
            continue

        away_abbr = normalize_team(str(row_v.get(team_col, "")))
        home_abbr = normalize_team(str(row_h.get(team_col, "")))

        if not away_abbr or not home_abbr:
            i += 2
            continue

        # Closing O/U total (use home row Close; both rows usually same)
        total_raw = str(row_h.get(close_col, row_v.get(close_col, ""))).strip()
        total = _parse_numeric(total_raw)

        # Moneylines
        ml_away = _parse_numeric(str(row_v.get(ml_col, "")).strip())
        ml_home = _parse_numeric(str(row_h.get(ml_col, "")).strip())

        games.append({
            "game_date":          game_date,
            "home_abbr":          home_abbr,
            "away_abbr":          away_abbr,
            "market_total_game":  total,
            "market_ml_home":     ml_home,
            "market_ml_away":     ml_away,
        })
        i += 2

    df = pd.DataFrame(games)
    print(f"  Parsed {len(df)} games from SBRO")
    return df


def _detect_sbro_columns(cols: list) -> dict:
    """Return mapping of logical → actual column names."""
    result = {}

    # Date
    for c in cols:
        if "date" in c:
            result["date"] = c
            break

    # VH (visitor/home indicator)
    for c in cols:
        if c in ("vh", "v/h", "side"):
            result["vh"] = c
            break

    # Team name
    for c in cols:
        if "team" in c:
            result["team"] = c
            break

    # Closing total — "close" preferred over "open"
    for c in cols:
        if "close" in c:
            result["close"] = c
            break
    if "close" not in result:
        for c in cols:
            if "open" in c:
                result["close"] = c
                break

    # Moneyline
    for c in cols:
        if c in ("ml", "money line", "moneyline", "money"):
            result["ml"] = c
            break

    required = {"date", "vh", "team", "close", "ml"}
    missing = required - set(result.keys())
    if missing:
        print(f"\n  Missing SBRO columns: {missing}")
        return {}
    return result


def _parse_sbro_date(raw: str) -> str:
    """Convert SBRO date formats (20240401, 4/1/2024, 2024-04-01) to YYYY-MM-DD."""
    raw = raw.strip()
    # Remove any time portion
    raw = raw.split(" ")[0].split("T")[0]
    try:
        if len(raw) == 8 and raw.isdigit():
            return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
        for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y"):
            try:
                from datetime import datetime
                return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
            except ValueError:
                pass
    except Exception:
        pass
    return ""


def _parse_numeric(val: str):
    """Parse numeric string; return None if not parseable."""
    if not val or val.upper() in ("NL", "PK", "N/A", "NAN", "", "-"):
        return None
    val = val.replace("+", "").strip()
    try:
        return float(val)
    except ValueError:
        return None


# ─── MERGE ───────────────────────────────────────────────────────────────────

def merge_lines(backtest_df: pd.DataFrame, lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join lines onto backtest by (game_date, home_team, away_team).
    Estimates F5/F3/F1 totals from game total.
    """
    if lines_df.empty:
        backtest_df["market_total_game"] = None
        backtest_df["market_total_f5"]   = None
        backtest_df["market_total_f3"]   = None
        backtest_df["market_total_f1"]   = None
        backtest_df["market_ml_home"]    = None
        backtest_df["market_ml_away"]    = None
        backtest_df["lines_source"]      = "missing"
        return backtest_df

    # Normalize keys for joining
    bt = backtest_df.copy()
    bt["_home"] = bt["home_team"].str.upper().str.strip()
    bt["_away"] = bt["away_team"].str.upper().str.strip()
    bt["_date"] = bt["game_date"].astype(str).str[:10]

    ln = lines_df.copy()
    ln["_home"] = ln["home_abbr"].str.upper().str.strip()
    ln["_away"] = ln["away_abbr"].str.upper().str.strip()
    ln["_date"] = ln["game_date"].astype(str).str[:10]
    ln = ln.drop_duplicates(subset=["_date", "_home", "_away"])

    merged = bt.merge(
        ln[["_date", "_home", "_away",
            "market_total_game", "market_ml_home", "market_ml_away"]],
        on=["_date", "_home", "_away"],
        how="left",
    )

    # F-inning estimates from game total
    has_total = merged["market_total_game"].notna()
    merged.loc[has_total, "market_total_f5"] = (
        merged.loc[has_total, "market_total_game"] * F5_RATIO
    ).round(2)
    merged.loc[has_total, "market_total_f3"] = (
        merged.loc[has_total, "market_total_game"] * F3_RATIO
    ).round(2)
    merged.loc[has_total, "market_total_f1"] = (
        merged.loc[has_total, "market_total_game"] * F1_RATIO
    ).round(2)

    merged["lines_source"] = merged["market_total_game"].apply(
        lambda x: "sbro" if pd.notna(x) else "missing"
    )

    return merged.drop(columns=["_home", "_away", "_date"])


# ─── SEASON BUILDER ──────────────────────────────────────────────────────────

def build_season_lines(year: int, no_download: bool = False,
                       source: str = "auto", api_key: str = ""):
    backtest_path = BACKTEST_FILES.get(year)
    if not backtest_path:
        print(f"  No backtest file configured for {year}")
        return

    try:
        games_df = pd.read_csv(backtest_path)
        print(f"  Loaded {len(games_df)} games from {backtest_path}")
    except FileNotFoundError:
        print(f"  {backtest_path} not found — run build_backtest.py first")
        return

    lines_df = pd.DataFrame()

    # ── Try The Odds API historical endpoint first ────────────────────────────
    if source in ("auto", "odds_api") and api_key:
        print(f"  Trying The Odds API historical endpoint ...")
        lines_df = fetch_odds_api_historical(games_df, api_key)
        if not lines_df.empty:
            print(f"  Odds API: {len(lines_df)} game lines fetched")

    # ── Fall back to SBRO ─────────────────────────────────────────────────────
    if lines_df.empty and source in ("auto", "sbro"):
        try:
            sbro_path = download_sbro(year, force=False)
            lines_df  = parse_sbro(sbro_path)
        except Exception as e:
            print(f"  SBRO unavailable: {e}")

    # ── Manual CSV fallback ───────────────────────────────────────────────────
    if lines_df.empty:
        manual = SBRO_CACHE_DIR / f"manual_lines_{year}.csv"
        if manual.exists():
            print(f"  Loading manual lines from {manual}")
            lines_df = pd.read_csv(manual)
        else:
            print(f"\n  No lines source available for {year}.")
            print(f"  Options:")
            print(f"    1. Upgrade The Odds API plan: https://the-odds-api.com/#pricing")
            print(f"    2. Place a CSV at: {manual}")
            print(f"       Columns: game_date, home_abbr, away_abbr, market_total_game, market_ml_home, market_ml_away")
            return

    merged = merge_lines(games_df, lines_df)

    matched   = (merged["lines_source"] == "sbro").sum()
    missing   = (merged["lines_source"] == "missing").sum()
    match_pct = matched / len(merged) * 100 if len(merged) else 0

    print(f"\n  Match rate : {matched}/{len(merged)} games ({match_pct:.1f}%)")
    if missing:
        print(f"  Unmatched  : {missing} games (no market line — will use None in calibration)")

    # Diagnostics for unmatched games
    if missing and missing < 50:
        unmatched = merged[merged["lines_source"] == "missing"][
            ["game_date", "home_team", "away_team"]
        ].head(10)
        print(f"\n  Sample unmatched games:")
        print(unmatched.to_string(index=False))

    # Show sample totals
    sample = merged[merged["lines_source"] == "sbro"].head(5)
    print(f"\n  Sample lines:")
    print(sample[["game_date","home_team","away_team",
                  "market_total_game","market_ml_home","market_ml_away"]].to_string(index=False))

    merged.to_csv(backtest_path, index=False)
    print(f"\n  Saved {len(merged)} games with lines → {backtest_path}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Add historical betting lines to backtest files")
    parser.add_argument("--year",        type=int,   help="Single year (default: 2024 + 2025)")
    parser.add_argument("--no-download", action="store_true",
                        help="Use only cached files, no downloads")
    parser.add_argument("--source",      default="auto",
                        choices=["auto", "odds_api", "sbro"],
                        help="Data source (default: try odds_api then sbro)")
    args = parser.parse_args()

    # Load Odds API key from settings if available
    api_key = ""
    try:
        import sys as _sys; _sys.path.insert(0, "wizard_agents")
        from config.settings import ODDS_API_KEY
        api_key = ODDS_API_KEY or ""
    except Exception:
        pass

    years = [args.year] if args.year else [2024, 2025]
    for year in years:
        print(f"\n{'='*60}")
        print(f"  Season {year}")
        print(f"{'='*60}")
        build_season_lines(year, no_download=args.no_download,
                           source=args.source, api_key=api_key)

    print("\nDone.")
    print("\nNext step: re-run calibrate_weights.py to calibrate MFull/MF3i/MF1i")


if __name__ == "__main__":
    main()
