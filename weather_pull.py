"""
weather_pull.py
---------------
Pull per-game weather for every MLB game in the backtest/season data
using the Open-Meteo historical archive API (free, no key required).

Run modes:
    python weather_pull.py                    # all years from backtest files
    python weather_pull.py --year 2025        # single year
    python weather_pull.py --date 2026-04-11  # single date (all games that day)
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
CACHE_DIR = OUTPUT_DIR / "weather_cache"

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# 7 PM local hour index in the hourly arrays (0-based, so index 19 = 19:00)
FIRST_PITCH_HOUR_INDEX = 19

# Backtest CSV / parquet patterns
BACKTEST_CSV_PATTERN = "backtest_games_{year}.csv"
SCHEDULE_PARQUET_PATTERN = "mlb_schedule_{year}.parquet"

# ---------------------------------------------------------------------------
# Stadium coordinates — 30 stadiums + Sacramento Athletics
# ---------------------------------------------------------------------------

STADIUM_COORDS = {
    "ARI": (33.4453, -112.0667, "Chase Field"),
    "AZ":  (33.4453, -112.0667, "Chase Field"),          # alias used in existing data
    "ATL": (33.8908, -84.4678,  "Truist Park"),
    "BAL": (39.2838, -76.6218,  "Camden Yards"),
    "BOS": (42.3467, -71.0972,  "Fenway Park"),
    "CHC": (41.9484, -87.6553,  "Wrigley Field"),
    "CWS": (41.8299, -87.6338,  "Guaranteed Rate"),
    "CIN": (39.0979, -84.5082,  "Great American Ball Park"),
    "CLE": (41.4958, -81.6852,  "Progressive Field"),
    "COL": (39.7559, -104.9942, "Coors Field"),
    "DET": (42.3390, -83.0485,  "Comerica Park"),
    "HOU": (29.7573, -95.3555,  "Minute Maid Park"),
    "KC":  (39.0517, -94.4803,  "Kauffman Stadium"),
    "LAA": (33.8003, -117.8827, "Angel Stadium"),
    "LAD": (34.0739, -118.2400, "Dodger Stadium"),
    "MIA": (25.7781, -80.2197,  "loanDepot Park"),
    "MIL": (43.0280, -87.9712,  "American Family Field"),
    "MIN": (44.9817, -93.2781,  "Target Field"),
    "NYM": (40.7571, -73.8458,  "Citi Field"),
    "NYY": (40.8296, -73.9262,  "Yankee Stadium"),
    "OAK": (37.7516, -122.2005, "Oakland Coliseum"),
    "PHI": (39.9061, -75.1665,  "Citizens Bank Park"),
    "PIT": (40.4469, -80.0057,  "PNC Park"),
    "SD":  (32.7076, -117.1570, "Petco Park"),
    "SF":  (37.7786, -122.3893, "Oracle Park"),
    "SEA": (47.5914, -122.3325, "T-Mobile Park"),
    "STL": (38.6226, -90.1928,  "Busch Stadium"),
    "TB":  (27.7682, -82.6534,  "Tropicana Field"),
    "TEX": (32.7512, -97.0832,  "Globe Life Field"),
    "TOR": (43.6414, -79.3894,  "Rogers Centre"),
    "WSH": (38.8730, -77.0074,  "Nationals Park"),
    # 2025 Sacramento Athletics
    "ATH": (38.5802, -121.4996, "Sutter Health Park"),
    "SAC": (38.5802, -121.4996, "Sutter Health Park"),   # alias
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_path(game_date: str, team: str) -> Path:
    """Return path for a cached weather JSON file."""
    return CACHE_DIR / f"{game_date}_{team}.json"


def load_cached(game_date: str, team: str) -> dict | None:
    """Return parsed JSON from cache if it exists, else None."""
    p = cache_path(game_date, team)
    if p.exists():
        try:
            with p.open() as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_cache(game_date: str, team: str, data: dict) -> None:
    """Persist API response JSON to cache."""
    p = cache_path(game_date, team)
    try:
        with p.open("w") as f:
            json.dump(data, f)
    except OSError as exc:
        print(f"  [WARN] Could not write cache for {game_date}/{team}: {exc}")


def fetch_open_meteo(lat: float, lon: float, date_str: str) -> dict | None:
    """
    Call the Open-Meteo archive endpoint for a single date.
    Returns the parsed JSON response or None on failure.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": "temperature_2m,windspeed_10m,winddirection_10m,relativehumidity_2m,precipitation,dewpoint_2m",
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "timezone": "America/New_York",
    }
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        print(f"  [ERROR] Open-Meteo request failed ({date_str}, lat={lat}, lon={lon}): {exc}")
        return None


def extract_weather(data: dict, team: str) -> dict:
    """
    Pull index-19 (7 PM) values out of the hourly arrays.
    Returns a flat dict of weather fields.
    """
    hourly = data.get("hourly", {})
    idx = FIRST_PITCH_HOUR_INDEX

    def safe_get(key: str) -> float | None:
        arr = hourly.get(key, [])
        if len(arr) > idx:
            val = arr[idx]
            # Open-Meteo may return None for missing values
            return float(val) if val is not None else None
        return None

    _, _, stadium_name = STADIUM_COORDS.get(team.upper(), (None, None, "Unknown"))

    return {
        "temp_f":        safe_get("temperature_2m"),
        "wind_mph":      safe_get("windspeed_10m"),
        "wind_bearing":  safe_get("winddirection_10m"),
        "humidity":      safe_get("relativehumidity_2m"),
        "precip_mm":     safe_get("precipitation"),
        "dew_point_f":   safe_get("dewpoint_2m"),
        "stadium_name":  stadium_name,
    }


def pull_weather_for_pair(game_date: str, home_team: str, sleep: bool = True) -> dict | None:
    """
    Fetch (or load from cache) weather for a single (date, team) pair.
    Returns a dict with weather fields, or None on failure.
    """
    team = home_team.upper().strip()

    # Lookup stadium coordinates
    coords = STADIUM_COORDS.get(team)
    if coords is None:
        print(f"  [WARN] No coordinates for team '{team}' — skipping.")
        return None

    lat, lon, _ = coords

    # Try cache first
    cached = load_cached(game_date, team)
    if cached is not None:
        return extract_weather(cached, team)

    # Fetch from API
    if sleep:
        time.sleep(0.1)

    data = fetch_open_meteo(lat, lon, game_date)
    if data is None:
        return None

    save_cache(game_date, team, data)
    return extract_weather(data, team)


# ---------------------------------------------------------------------------
# Public helper — used by the daily pipeline
# ---------------------------------------------------------------------------

def pull_weather_for_date(game_date: str, home_team: str) -> dict | None:
    """
    Return a weather dict for a single game (date + home team).
    Suitable for calling from the daily pipeline.

    Parameters
    ----------
    game_date : str
        ISO date string, e.g. '2026-04-11'
    home_team : str
        Team abbreviation, e.g. 'NYY'

    Returns
    -------
    dict with keys: temp_f, wind_mph, wind_bearing, humidity, precip_mm,
    stadium_name — or None on failure.
    """
    result = pull_weather_for_pair(game_date, home_team, sleep=False)
    if result is not None:
        result["game_date"] = game_date
        result["home_team"] = home_team.upper()
    return result


# ---------------------------------------------------------------------------
# Game list loaders
# ---------------------------------------------------------------------------

def load_game_list_for_year(year: int) -> pd.DataFrame | None:
    """
    Try to load a game list for `year` from:
      1. backtest_games_{year}.csv   (columns: game_date, home_team, ...)
      2. mlb_schedule_{year}.parquet (columns: game_date, home_team, ...)
    Returns a DataFrame with at least [game_date, home_team], or None.
    """
    base = Path(".")

    # Option 1: backtest CSV
    csv_path = base / BACKTEST_CSV_PATTERN.format(year=year)
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, usecols=lambda c: c in {"game_date", "home_team"})
            df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
            print(f"Loaded {len(df)} rows from {csv_path}")
            return df[["game_date", "home_team"]].drop_duplicates()
        except Exception as exc:
            print(f"[WARN] Could not read {csv_path}: {exc}")

    # Option 2: schedule_all parquet (generated by supplemental_pull.py — has game_date + home_team)
    all_sched_path = OUTPUT_DIR / f"schedule_all_{year}.parquet"
    if all_sched_path.exists():
        try:
            df = pd.read_parquet(all_sched_path, engine="pyarrow")
            # Keep only home-team rows to avoid duplicating games
            if "home_away" in df.columns:
                df = df[df["home_away"].str.lower() == "home"]
            df = df.rename(columns={"gameDate": "game_date"})
            df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
            # Resolve home_team column name variants
            if "home_team" not in df.columns:
                for alt in ("home_team_name", "home", "team"):
                    if alt in df.columns:
                        df = df.rename(columns={alt: "home_team"})
                        break
            print(f"Loaded {len(df)} rows from {all_sched_path}")
            return df[["game_date", "home_team"]].drop_duplicates()
        except Exception as exc:
            print(f"[WARN] Could not read {all_sched_path}: {exc}")

    # Option 3: per-team mlb_schedule parquet (gameDate column — handle rename)
    parquet_path = OUTPUT_DIR / SCHEDULE_PARQUET_PATTERN.format(year=year)
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path, engine="pyarrow")
            # Normalize column names
            df = df.rename(columns={"gameDate": "game_date", "home_team_name": "home_team"})
            df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
            if "home_team" not in df.columns:
                for alt in ("home", "team"):
                    if alt in df.columns:
                        df = df.rename(columns={alt: "home_team"})
                        break
            print(f"Loaded {len(df)} rows from {parquet_path}")
            return df[["game_date", "home_team"]].drop_duplicates()
        except Exception as exc:
            print(f"[WARN] Could not read {parquet_path}: {exc}")

    print(f"[WARN] No game list found for year {year}.")
    return None


def available_years() -> list[int]:
    """Return years that have a backtest CSV or schedule parquet."""
    base = Path(".")
    years = set()
    for p in base.glob("backtest_games_*.csv"):
        try:
            years.add(int(p.stem.split("_")[-1]))
        except ValueError:
            pass
    for p in OUTPUT_DIR.glob("mlb_schedule_*.parquet"):
        try:
            years.add(int(p.stem.split("_")[-1]))
        except ValueError:
            pass
    return sorted(years)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_game_list(df: pd.DataFrame, year: int) -> None:
    """
    Given a DataFrame with [game_date, home_team], fetch weather for each
    unique pair and save to weather_{year}.parquet.
    """
    pairs = (
        df[["game_date", "home_team"]]
        .drop_duplicates()
        .sort_values(["game_date", "home_team"])
        .reset_index(drop=True)
    )

    total = len(pairs)
    print(f"Processing {total} unique (date, team) pairs for year {year} ...")

    records: list[dict] = []

    for i, row in pairs.iterrows():
        game_date = str(row["game_date"])
        home_team = str(row["home_team"]).upper().strip()

        print(f"  [{i+1}/{total}] {game_date} {home_team}", end="", flush=True)

        weather = pull_weather_for_pair(game_date, home_team, sleep=True)
        if weather is None:
            print(" — FAILED")
            continue

        record = {
            "game_date":   game_date,
            "home_team":   home_team,
            **weather,
        }
        records.append(record)
        temp   = weather.get("temp_f")
        wind   = weather.get("wind_mph")
        print(f" — temp={temp}°F  wind={wind}mph")

    if not records:
        print(f"[WARN] No weather records collected for year {year}.")
        return

    out_df = pd.DataFrame(records)

    # Ensure column order
    col_order = [
        "game_date", "home_team", "temp_f", "wind_mph",
        "wind_bearing", "humidity", "precip_mm", "dew_point_f", "stadium_name",
    ]
    for col in col_order:
        if col not in out_df.columns:
            out_df[col] = None
    out_df = out_df[col_order]

    out_path = OUTPUT_DIR / f"weather_{year}.parquet"
    try:
        if out_path.exists():
            existing = pd.read_parquet(out_path, engine="pyarrow")
            out_df = (pd.concat([existing, out_df], ignore_index=True)
                        .drop_duplicates(subset=["game_date", "home_team"], keep="last")
                        .sort_values(["game_date", "home_team"])
                        .reset_index(drop=True))
        out_df.to_parquet(out_path, engine="pyarrow", index=False)
        print(f"\nSaved {out_path}  shape={out_df.shape}")
    except Exception as exc:
        print(f"[ERROR] Could not save {out_path}: {exc}")


def process_single_date(date_str: str) -> None:
    """
    Fetch weather for all teams that have a game on `date_str` based on
    any available backtest CSVs / schedule parquets.  Also works if you
    just want to pull weather for all 30 stadiums on a given date.
    """
    # Collect all game lists
    years = available_years()
    frames = []
    for year in years:
        df = load_game_list_for_year(year)
        if df is not None:
            frames.append(df)

    if frames:
        all_games = pd.concat(frames, ignore_index=True)
        day_games = all_games[all_games["game_date"] == date_str].copy()
    else:
        day_games = pd.DataFrame(columns=["game_date", "home_team"])

    if day_games.empty:
        print(f"No games found in backtest files for {date_str}.")
        print("Pulling weather for all 30 stadiums as a fallback ...")
        teams = list(STADIUM_COORDS.keys())
        rows = [{"game_date": date_str, "home_team": t} for t in teams]
        day_games = pd.DataFrame(rows)

    year = int(date_str[:4])
    process_game_list(day_games, year)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull per-game weather from Open-Meteo for MLB games."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--year", type=int, metavar="YYYY",
        help="Pull weather for a single season year."
    )
    group.add_argument(
        "--date", type=str, metavar="YYYY-MM-DD",
        help="Pull weather for all games on a specific date."
    )
    return parser.parse_args()


def main() -> None:
    ensure_dirs()
    args = parse_args()

    if args.date:
        # Validate date format
        try:
            datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            print(f"[ERROR] --date must be in YYYY-MM-DD format, got '{args.date}'")
            return
        print(f"=== Pulling weather for date: {args.date} ===")
        process_single_date(args.date)

    elif args.year:
        print(f"=== Pulling weather for year: {args.year} ===")
        df = load_game_list_for_year(args.year)
        if df is not None:
            process_game_list(df, args.year)

    else:
        # All available years
        years = available_years()
        if not years:
            print("[ERROR] No backtest CSVs or schedule parquets found. "
                  "Run with --year or --date, or place backtest_games_{year}.csv "
                  "in the working directory.")
            return
        print(f"=== Pulling weather for all years: {years} ===")
        for year in years:
            df = load_game_list_for_year(year)
            if df is not None:
                process_game_list(df, year)


if __name__ == "__main__":
    main()
