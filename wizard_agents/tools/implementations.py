"""
tools/implementations.py

Pure Python tool functions. No framework inheritance needed.
Each function takes **kwargs matching the tool's input_schema and returns a JSON string.

These are called by tool_executor() in each agent module, which dispatches
by tool name. Claude decides WHEN to call them — you just implement WHAT they do.
"""
from __future__ import annotations

import json
import smtplib
from datetime import date, datetime, timedelta
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from config.settings import (
    FILES, STATIC_FILES, STALE_THRESHOLD_DAYS, PIPELINE_DIR,
    ODDS_API_KEY, F5_ESTIMATE_RATIO, F3_ESTIMATE_RATIO, F1_ESTIMATE_RATIO,
    GMAIL_FROM, GMAIL_PASSWORD, EMAIL_RECIPIENTS,
)

# ── Ballpark coordinates ──────────────────────────────────────────────────────
# Full name → abbreviation mapping (Odds API returns full names)
TEAM_NAME_TO_ABBR: dict[str, str] = {
    'Arizona Diamondbacks': 'ARI', 'Atlanta Braves': 'ATL',
    'Baltimore Orioles': 'BAL', 'Boston Red Sox': 'BOS',
    'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CWS',
    'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE',
    'Colorado Rockies': 'COL', 'Detroit Tigers': 'DET',
    'Houston Astros': 'HOU', 'Kansas City Royals': 'KC',
    'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD',
    'Miami Marlins': 'MIA', 'Milwaukee Brewers': 'MIL',
    'Minnesota Twins': 'MIN', 'New York Mets': 'NYM',
    'New York Yankees': 'NYY', 'Oakland Athletics': 'OAK',
    'Philadelphia Phillies': 'PHI', 'Pittsburgh Pirates': 'PIT',
    'San Diego Padres': 'SD', 'San Francisco Giants': 'SF',
    'Seattle Mariners': 'SEA', 'St. Louis Cardinals': 'STL',
    'Tampa Bay Rays': 'TB', 'Texas Rangers': 'TEX',
    'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSH',
    'Athletics': 'OAK',
}

BALLPARK_COORDS: dict[str, tuple[float, float]] = {
    "ARI": (33.4455, -112.0667), "ATL": (33.8908, -84.4678),
    "BAL": (39.2838, -76.6217),  "BOS": (42.3467, -71.0972),
    "CHC": (41.9484, -87.6553),  "CWS": (41.8300, -87.6339),
    "CIN": (39.0979, -84.5082),  "CLE": (41.4959, -81.6854),
    "COL": (39.7559, -104.9942), "DET": (42.3390, -83.0485),
    "HOU": (29.7573, -95.3555),  "KC":  (39.0517, -94.4803),
    "LAA": (33.8003, -117.8827), "LAD": (34.0739, -118.2400),
    "MIA": (25.7781, -80.2197),  "MIL": (43.0280, -87.9712),
    "MIN": (44.9817, -93.2776),  "NYM": (40.7571, -73.8458),
    "NYY": (40.8296, -73.9262),  "OAK": (37.7516, -122.2005),
    "PHI": (39.9061, -75.1665),  "PIT": (40.4469, -80.0057),
    "SD":  (32.7076, -117.1570), "SF":  (37.7786, -122.3893),
    "SEA": (47.5914, -122.3325), "STL": (38.6226, -90.1928),
    "TB":  (27.7683, -82.6534),  "TEX": (32.7473, -97.0822),
    "TOR": (43.6414, -79.3894),  "WSH": (38.8730, -77.0074),
}

BET_TRACKER_COLUMNS = [
    "id", "date", "game", "model", "bet_type", "model_prob",
    "market_line", "book", "units", "result", "actual_total",
    "profit_loss", "logged_at", "notes",
]


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — Data Ingestion Tools
# ══════════════════════════════════════════════════════════════════════════════

def check_stale_files(threshold_days: int = STALE_THRESHOLD_DAYS) -> str:
    cutoff = datetime.now() - timedelta(days=threshold_days)
    results, stale, missing = {}, [], []

    for key in STATIC_FILES:
        path = FILES.get(key)
        if not path or not Path(path).exists():
            results[key] = {"status": "MISSING"}
            missing.append(key)
            continue
        mtime    = datetime.fromtimestamp(Path(path).stat().st_mtime)
        age_days = (datetime.now() - mtime).days
        if mtime < cutoff:
            results[key] = {"status": "STALE", "age_days": age_days,
                            "warning": f"⚠️ STALE: {key} last updated {age_days} days ago. Upload fresh data Sunday."}
            stale.append(key)
        else:
            results[key] = {"status": "OK", "age_days": age_days}

    return json.dumps({"stale_files": stale, "missing_files": missing, "details": results})


def fetch_odds_api(game_date: str = "") -> str:
    target = game_date or date.today().isoformat()
    url    = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
    params = {
        "apiKey": ODDS_API_KEY, "regions": "us",
        "markets": "h2h,totals", "oddsFormat": "american", "dateFormat": "iso",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        return json.dumps({"status": "HALT", "error": str(e), "games": []})

    if not raw:
        return json.dumps({
            "status": "HALT",
            "error":  "No lines available from Odds API. Aborting pipeline run. Try again after 10 AM.",
            "games":  [],
        })

    games = []
    for event in raw:
        home, away = event.get("home_team", ""), event.get("away_team", "")
        dk_total = fd_total = dk_home_ml = dk_away_ml = fd_home_ml = fd_away_ml = None

        for bm in event.get("bookmakers", []):
            key = bm.get("key", "")
            if key not in ("draftkings", "fanduel"):
                continue
            for mkt in bm.get("markets", []):
                if mkt["key"] == "totals":
                    for o in mkt.get("outcomes", []):
                        if o["name"] == "Over":
                            if key == "draftkings": dk_total = o.get("point")
                            else:                   fd_total = o.get("point")
                elif mkt["key"] == "h2h":
                    for o in mkt.get("outcomes", []):
                        if o["name"] == home:
                            if key == "draftkings": dk_home_ml = o.get("price")
                            else:                   fd_home_ml = o.get("price")
                        else:
                            if key == "draftkings": dk_away_ml = o.get("price")
                            else:                   fd_away_ml = o.get("price")

        game_total = dk_total or fd_total
        games.append({
            "game_id":         event.get("id"),
            "home_team":       home,
            "away_team":       away,
            "commence":        event.get("commence_time"),
            "game_total":      game_total,
            "dk_total":        dk_total,
            "fd_total":        fd_total,
            "f5_total":        round(game_total * F5_ESTIMATE_RATIO, 1) if game_total else None,
            "f3_total":        round(game_total * F3_ESTIMATE_RATIO, 1) if game_total else None,
            "f1_total":        round(game_total * F1_ESTIMATE_RATIO, 1) if game_total else None,
            "f5_estimated":    True,
            "f3_estimated":    True,
            "f1_estimated":    True,
            "dk_home_ml":      dk_home_ml,
            "dk_away_ml":      dk_away_ml,
            "fd_home_ml":      fd_home_ml,
            "fd_away_ml":      fd_away_ml,
            "best_over_book":  "DK" if (dk_total or 0) >= (fd_total or 0) else "FD",
            "best_under_book": "FD" if (fd_total or 0) <= (dk_total or 0) else "DK",
            "is_coors":        home in ("COL",) or away in ("COL",),
        })

    return json.dumps({"status": "OK", "games_fetched": len(games), "games": games})


def fetch_mlb_starters(game_date: str = "") -> str:
    target = game_date or date.today().isoformat()
    url    = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1, "date": target,
        "hydrate": "probablePitcher(note),team",
        "fields":  "dates,games,gamePk,teams,home,away,team,name,abbreviation,probablePitcher,fullName,id",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return json.dumps({"status": "WARNING", "error": str(e), "starters": {}})

    starters, excluded = {}, []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            gk       = str(game.get("gamePk"))
            home_d   = game.get("teams", {}).get("home", {})
            away_d   = game.get("teams", {}).get("away", {})
            home_abbr = home_d.get("team", {}).get("abbreviation", "")
            away_abbr = away_d.get("team", {}).get("abbreviation", "")
            hp        = home_d.get("probablePitcher", {})
            ap        = away_d.get("probablePitcher", {})

            if not hp or not ap:
                excluded.append(f"{away_abbr} @ {home_abbr} (TBD starter)")
                continue
            starters[gk] = {
                "home_team":    home_abbr, "away_team":    away_abbr,
                "home_starter": hp.get("fullName"), "home_pitcher_id": hp.get("id"),
                "away_starter": ap.get("fullName"), "away_pitcher_id": ap.get("id"),
            }

    return json.dumps({"status": "OK", "confirmed_games": len(starters),
                       "excluded_games": excluded, "starters": starters})


def fetch_weather(games_json: str) -> str:
    try:
        games = json.loads(games_json)
    except Exception as e:
        return json.dumps({"status": "WARNING", "error": f"Invalid games_json: {e}", "weather": {}})

    weather = {}
    for game in games:
        home    = game.get("home_team", "")
        game_id = game.get("game_id", home)
        home_key = TEAM_NAME_TO_ABBR.get(home, home)
        coords  = BALLPARK_COORDS.get(home_key)
        if not coords:
            weather[game_id] = {"status": "WARNING", "error": f"No coords for {home}"}
            continue
        lat, lon = coords
        try:
            resp = requests.get("https://api.open-meteo.com/v1/forecast", timeout=10, params={
                "latitude": lat, "longitude": lon,
                "current":  "temperature_2m,wind_speed_10m,wind_direction_10m,precipitation_probability,weather_code",
                "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "forecast_days": 1,
            })
            resp.raise_for_status()
            w = resp.json().get("current", {})
            weather[game_id] = {
                "temperature_f":             round(w.get("temperature_2m", 0)),
                "wind_speed_mph":            round(w.get("wind_speed_10m", 0)),
                "wind_direction":            w.get("wind_direction_10m"),
                "precipitation_probability": w.get("precipitation_probability"),
            }
        except Exception as e:
            weather[game_id] = {"status": "WARNING", "error": str(e)}

    return json.dumps({"status": "OK", "weather": weather})


# ══════════════════════════════════════════════════════════════════════════════
# SHARED — File I/O Tools
# ══════════════════════════════════════════════════════════════════════════════

def read_csv(file_key: str, max_rows: Optional[int] = None) -> str:
    file_key = file_key.replace(".csv", "").replace(".CSV", "")
    path = FILES.get(file_key)
    if not path or not Path(path).exists():
        return json.dumps({"status": "ERROR", "error": f"File not found: {file_key} → {path}"})
    try:
        # Cap at 500 rows by default to prevent context window overflow
        cap = max_rows if max_rows is not None else 500
        total_rows = sum(1 for _ in open(path)) - 1  # exclude header
        df = pd.read_csv(path, nrows=cap)
        result = {"status": "OK", "rows": len(df), "total_rows": total_rows,
                  "columns": list(df.columns), "records": df.to_dict(orient="records")}
        if total_rows > cap:
            result["warning"] = f"Showing {cap} of {total_rows} rows. Pass max_rows to get more."
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"status": "ERROR", "error": str(e)})


def read_csv_filtered(
    file_key: str,
    name_filter: Optional[str] = None,
    team_filter: Optional[str] = None,
    name_col: str = "Name",
    team_col: str = "Team",
) -> str:
    """Return only rows matching pitcher/batter names or team abbreviations.
    Cuts token payload by ~90% vs full read_csv on fangraphs/savant files."""
    file_key = file_key.replace(".csv", "").replace(".CSV", "")
    path = FILES.get(file_key)
    if not path or not Path(path).exists():
        return json.dumps({"status": "ERROR", "error": f"File not found: {file_key} → {path}"})
    try:
        df         = pd.read_csv(path)
        total_rows = len(df)
        names      = json.loads(name_filter) if name_filter else []
        teams      = json.loads(team_filter) if team_filter else []

        mask = pd.Series([False] * len(df), index=df.index)
        if names and name_col in df.columns:
            mask |= df[name_col].isin(names)
        if teams and team_col in df.columns:
            mask |= df[team_col].isin(teams)
        if not names and not teams:
            mask = pd.Series([True] * len(df), index=df.index)

        filtered = df[mask]
        return json.dumps({
            "status":     "OK",
            "rows":       len(filtered),
            "total_rows": total_rows,
            "filtered":   True,
            "columns":    list(filtered.columns),
            "records":    filtered.to_dict(orient="records"),
        })
    except Exception as e:
        return json.dumps({"status": "ERROR", "error": str(e)})


def write_csv(file_key: str, records_json: str, mode: str = "w") -> str:
    file_key = file_key.replace(".csv", "").replace(".CSV", "")
    if file_key == "bet_tracker":
        return json.dumps({"status": "REJECTED",
                           "error": "bet_tracker.csv is exclusively managed by the Bet Tracker Agent."})
    path = FILES.get(file_key)
    if not path:
        return json.dumps({"status": "ERROR", "error": f"Unknown file key: {file_key}"})
    try:
        records = json.loads(records_json)
        df      = pd.DataFrame(records)
        header  = (mode == "w") or not Path(path).exists()
        df.to_csv(path, mode=mode, index=False, header=header)
        return json.dumps({"status": "OK", "file": str(path), "rows_written": len(df)})
    except Exception as e:
        return json.dumps({"status": "ERROR", "error": str(e)})


def write_html(file_key: str, html_content: str) -> str:
    path = FILES.get(file_key)
    if not path:
        return json.dumps({"status": "ERROR", "error": f"Unknown file key: {file_key}"})
    try:
        Path(path).write_text(html_content, encoding="utf-8")
        return json.dumps({"status": "OK", "file": str(path), "bytes": len(html_content)})
    except Exception as e:
        return json.dumps({"status": "ERROR", "error": str(e)})


def validate_static_file(
    file_key: str,
    expected_columns: str,
    min_rows: int = 10,
    expected_size_kb: Optional[int] = None,
) -> str:
    # Normalize key — Claude sometimes passes savant_pitchers.csv instead of savant_pitchers
    file_key = file_key.replace(".csv", "").replace(".CSV", "")
    path = FILES.get(file_key)
    if not path or not Path(path).exists():
        return json.dumps({"status": "REJECTED", "error": f"File not found: {file_key}"})

    size_kb = Path(path).stat().st_size // 1024

    if size_kb == 0:
        return json.dumps({"status": "REJECTED", "error": f"File is empty (0KB): {file_key}. Re-export and re-upload."})

    # Size guard for savant files
    # Note: early season both savant files may be similar size — no size guard needed
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return json.dumps({"status": "REJECTED", "error": f"Cannot read CSV: {e}"})

    required = json.loads(expected_columns)
    missing  = [c for c in required if c not in df.columns]
    if missing:
        return json.dumps({"status": "REJECTED", "error": f"Missing columns: {missing}"})
    if len(df) < min_rows:
        return json.dumps({"status": "REJECTED",
                           "error": f"Only {len(df)} rows — expected ≥ {min_rows}."})

    return json.dumps({"status": "OK", "rows": len(df), "size_kb": size_kb,
                       "columns": list(df.columns)})


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — Scoring Engine (ML inference + Three-Part Lock)
# ══════════════════════════════════════════════════════════════════════════════
#
# generate_ml_scores() replaces the legacy heuristic scorers.  It:
#   1. Invokes score_ml_today.predict_games()        → full-game moneyline L2 prob
#   2. Invokes score_f5_today.predict_games()        → F5 home-cover L2 prob (informational)
#   3. Invokes score_run_dist_today.predict_games()  → totals + runline L2 probs
#   4. Joins model output with games.csv on (home_team, away_team)
#   5. Applies the Three-Part Lock (sanity / odds-floor / edge) per market
#   6. Computes Kelly dollar stake ($2,000 bank, cap $50, $1 floor)
#   7. Writes model_scores.csv and returns ONLY a compact actionable summary
#
# Design invariant: the heavy DataFrame work happens inside this tool; the LLM
# only ever sees a small JSON payload (counts + actionable pick list).

_KELLY_BANK       = 2000.0
_KELLY_CAP        = 50.0
_TIER1_EDGE       = 0.030
_TIER2_EDGE       = 0.010
_SANITY_THRESHOLD = 0.04
_ODDS_FLOOR       = -225


def _american_to_decimal(odds: float) -> float:
    return 1.0 + (odds / 100.0) if odds >= 0 else 1.0 + (100.0 / abs(odds))


def _kelly_stake(model_prob: float, american_odds: float, tier: int) -> int:
    """Quarter-Kelly (tier 1) / Eighth-Kelly (tier 2), $2000 bank, $50 cap, $1 floor."""
    if american_odds is None or model_prob is None:
        return 0
    b = _american_to_decimal(float(american_odds)) - 1.0
    if b <= 0:
        return 0
    f_star = (b * model_prob - (1.0 - model_prob)) / b
    if f_star <= 0:
        return 0
    mult    = 0.25 if tier == 1 else 0.125
    raw     = _KELLY_BANK * f_star * mult
    capped  = min(raw, _KELLY_CAP)
    rounded = int(round(capped))
    return max(rounded, 1)


def _classify_edge(edge: float) -> Optional[int]:
    if edge is None:
        return None
    if edge >= _TIER1_EDGE:
        return 1
    if edge >= _TIER2_EDGE:
        return 2
    return None


def _safe_float(v) -> Optional[float]:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _invoke_scorer(module_name: str, date_str: str) -> pd.DataFrame:
    """Import a score_*_today module fresh and run predict_games(date_str).
    Scoring scripts use cwd-relative paths, so we chdir into PIPELINE_DIR for the call."""
    import sys, os, io, contextlib
    prev_cwd = os.getcwd()
    prev_path = list(sys.path)
    try:
        os.chdir(str(PIPELINE_DIR))
        if str(PIPELINE_DIR) not in sys.path:
            sys.path.insert(0, str(PIPELINE_DIR))
        sys.modules.pop(module_name, None)
        with contextlib.redirect_stdout(io.StringIO()):
            mod = __import__(module_name)
            df  = mod.predict_games(date_str)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    finally:
        os.chdir(prev_cwd)
        sys.path[:] = prev_path


def _evaluate_pick(
    model_prob: float,
    p_true:     Optional[float],
    retail_imp: Optional[float],
    retail_odds: Optional[float],
) -> dict:
    """Apply Three-Part Lock and compute tier + stake.  Returns all gate flags."""
    sanity_pass = False if p_true is None else (abs(model_prob - p_true) <= _SANITY_THRESHOLD)
    odds_pass   = False if retail_odds is None else (retail_odds >= _ODDS_FLOOR)
    edge        = None if retail_imp is None else (model_prob - retail_imp)
    tier        = _classify_edge(edge) if edge is not None else None
    stake       = 0
    if sanity_pass and odds_pass and tier is not None:
        stake = _kelly_stake(model_prob, retail_odds, tier)
    actionable  = bool(sanity_pass and odds_pass and tier is not None and stake >= 1)
    return {
        "model_prob":          round(model_prob, 4),
        "P_true":              None if p_true is None else round(p_true, 4),
        "Retail_Implied_Prob": None if retail_imp is None else round(retail_imp, 4),
        "edge":                None if edge is None else round(edge, 4),
        "retail_american_odds": retail_odds,
        "sanity_check_pass":   bool(sanity_pass),
        "odds_floor_pass":     bool(odds_pass),
        "tier":                tier,
        "dollar_stake":        int(stake) if stake >= 1 else None,
        "actionable":          actionable,
    }


def generate_ml_scores(game_date: str = "") -> str:
    """
    Run all three ML scoring scripts for game_date, join with games.csv,
    apply the Three-Part Lock + Kelly staking, write model_scores.csv,
    and return a compact actionable summary.

    Returns JSON with keys:
      status, games_scored, actionable, tier1, tier2,
      failed_sanity, failed_odds_floor, failed_edge,
      actionable_picks (list of ≤N compact dicts)
    """
    target = game_date or date.today().isoformat()

    # 1. Games + market context
    games_path = FILES.get("games")
    if not games_path or not Path(games_path).exists():
        return json.dumps({"status": "ERROR",
                           "error": "games.csv not found — Agent 1 must run first."})
    games_df = pd.read_csv(games_path)
    if len(games_df) == 0:
        return json.dumps({"status": "ERROR", "error": "games.csv is empty."})

    # 2. Invoke all three scorers (ML, F5, Run-Dist)
    try:
        ml_df  = _invoke_scorer("score_ml_today",       target)
        rd_df  = _invoke_scorer("score_run_dist_today", target)
        f5_df  = _invoke_scorer("score_f5_today",       target)
    except Exception as e:
        return json.dumps({"status": "ERROR",
                           "error": f"Scoring inference failed: {type(e).__name__}: {e}"})

    def _keyed(df: pd.DataFrame, cols: list[str]) -> dict:
        """Index a scoring frame by (home_team, away_team) for O(1) lookup."""
        out = {}
        if df is None or len(df) == 0:
            return out
        for _, r in df.iterrows():
            key = (str(r.get("home_team", "")).strip(),
                   str(r.get("away_team", "")).strip())
            out[key] = {c: _safe_float(r.get(c)) for c in cols}
        return out

    ml_by = _keyed(ml_df, ["stacker_l2"])                             # p_home_win
    rd_by = _keyed(rd_df, ["p_over_final", "p_home_cover_final",
                           "lam_home",     "lam_away",     "total_line"])
    f5_by = _keyed(f5_df, ["stacker_l2"])                             # p_f5_home_cover

    # 3. Build pick rows — one per market per game
    picks: list[dict] = []
    counters = {"sanity_fail": 0, "odds_fail": 0, "edge_fail": 0,
                "tier1": 0, "tier2": 0, "actionable": 0}

    for _, g in games_df.iterrows():
        home       = str(g.get("home_team", "")).strip()
        away       = str(g.get("away_team", "")).strip()
        game_label = g.get("game_label") or f"{away} @ {home}"
        key        = (home, away)

        # ─── MODEL 1: ML (full-game moneyline) ──────────────────────────────
        ml_row = ml_by.get(key)
        if ml_row and ml_row.get("stacker_l2") is not None:
            p_home = float(ml_row["stacker_l2"])
            # Both directions evaluated — pick side with higher model prob.
            for side, prob, p_true_col, imp_col, odds_col in [
                ("HOME", p_home,      "P_true_home", "Retail_Implied_Prob_home", "retail_ml_home_odds"),
                ("AWAY", 1.0 - p_home, "P_true_away", "Retail_Implied_Prob_away", "retail_ml_away_odds"),
            ]:
                ev = _evaluate_pick(prob,
                                    _safe_float(g.get(p_true_col)),
                                    _safe_float(g.get(imp_col)),
                                    _safe_float(g.get(odds_col)))
                picks.append({"date": target, "game": game_label,
                              "model": "ML", "bet_type": "Moneyline",
                              "pick_direction": side, **ev})

        # ─── MODEL 2: Totals (Over / Under) ─────────────────────────────────
        rd_row = rd_by.get(key)
        if rd_row and rd_row.get("p_over_final") is not None:
            p_over = float(rd_row["p_over_final"])
            for side, prob, p_true_col, imp_col, odds_col in [
                ("OVER",  p_over,        "P_true_over",  "Retail_Implied_Prob_over",  "retail_over_odds"),
                ("UNDER", 1.0 - p_over,  "P_true_under", "Retail_Implied_Prob_under", "retail_under_odds"),
            ]:
                ev = _evaluate_pick(prob,
                                    _safe_float(g.get(p_true_col)),
                                    _safe_float(g.get(imp_col)),
                                    _safe_float(g.get(odds_col)))
                picks.append({"date": target, "game": game_label,
                              "model": "Totals",
                              "bet_type": f"Total {_safe_float(g.get('retail_total_line')) or rd_row.get('total_line')}",
                              "pick_direction": side, **ev})

        # ─── MODEL 3: Runline (Pinnacle-only — no retail odds yet) ──────────
        if rd_row and rd_row.get("p_home_cover_final") is not None:
            p_rl_home = float(rd_row["p_home_cover_final"])
            for side, prob, p_true_col in [
                ("HOME -1.5", p_rl_home,        "P_true_rl_home"),
                ("AWAY +1.5", 1.0 - p_rl_home,  "P_true_rl_away"),
            ]:
                # No retail_rl_*_odds in games.csv schema yet → non-actionable.
                ev = _evaluate_pick(prob, _safe_float(g.get(p_true_col)), None, None)
                picks.append({"date": target, "game": game_label,
                              "model": "Runline", "bet_type": "Runline -1.5",
                              "pick_direction": side, **ev})

        # ─── MODEL 4: F5 (informational — no F5 market lines in games.csv) ──
        f5_row = f5_by.get(key)
        if f5_row and f5_row.get("stacker_l2") is not None:
            p_f5 = float(f5_row["stacker_l2"])
            ev = _evaluate_pick(p_f5, None, None, None)
            picks.append({"date": target, "game": game_label,
                          "model": "F5", "bet_type": "F5 Home +0.5",
                          "pick_direction": "HOME" if p_f5 >= 0.5 else "AWAY", **ev})

    # 4. Tally gate-failure breakdown + tier counts
    for p in picks:
        if p["actionable"]:
            counters["actionable"] += 1
            if p["tier"] == 1: counters["tier1"] += 1
            if p["tier"] == 2: counters["tier2"] += 1
            continue
        if not p["sanity_check_pass"]: counters["sanity_fail"] += 1
        if not p["odds_floor_pass"]:   counters["odds_fail"]   += 1
        if p["tier"] is None and p["edge"] is not None: counters["edge_fail"] += 1

    # 5. Persist full frame to model_scores.csv (all picks, gates + stakes)
    output_cols = ["date", "game", "model", "bet_type", "pick_direction",
                   "model_prob", "P_true", "Retail_Implied_Prob", "edge",
                   "retail_american_odds",
                   "sanity_check_pass", "odds_floor_pass",
                   "tier", "dollar_stake", "actionable"]
    out_df = pd.DataFrame(picks, columns=output_cols) if picks else pd.DataFrame(columns=output_cols)
    out_path = FILES["model_scores"]
    out_df.to_csv(out_path, index=False)

    # 6. Return compact actionable summary (LLM-safe payload)
    actionable_picks = [
        {k: p[k] for k in ("game", "model", "bet_type", "pick_direction",
                           "model_prob", "P_true", "Retail_Implied_Prob",
                           "edge", "retail_american_odds", "tier", "dollar_stake")}
        for p in picks if p["actionable"]
    ]
    actionable_picks.sort(key=lambda r: (r["tier"], -(r["edge"] or 0)))

    return json.dumps({
        "status":             "OK",
        "date":               target,
        "games_scored":       int(len(games_df)),
        "total_picks":        int(len(picks)),
        "actionable":         counters["actionable"],
        "tier1":              counters["tier1"],
        "tier2":              counters["tier2"],
        "failed_sanity":      counters["sanity_fail"],
        "failed_odds_floor":  counters["odds_fail"],
        "failed_edge":        counters["edge_fail"],
        "model_scores_path":  str(out_path),
        "actionable_picks":   actionable_picks,
    })


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 5 — Notification Tool
# ══════════════════════════════════════════════════════════════════════════════

def send_email(subject: str, body: str, attach_report: bool = True) -> str:
    msg           = MIMEMultipart()
    msg["From"]   = GMAIL_FROM
    msg["To"]     = ", ".join(EMAIL_RECIPIENTS)
    msg["Subject"]= subject
    msg.attach(MIMEText(body, "plain"))

    if attach_report:
        rp = FILES.get("model_report")
        if rp and Path(rp).exists():
            with open(rp, "rb") as f:
                part = MIMEApplication(f.read(), Name="model_report.html")
            part["Content-Disposition"] = 'attachment; filename="model_report.html"'
            msg.attach(part)

    last_err = None
    for attempt in range(1, 3):
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
                s.login(GMAIL_FROM, GMAIL_PASSWORD)
                s.sendmail(GMAIL_FROM, EMAIL_RECIPIENTS, msg.as_string())
            return json.dumps({"status": "OK", "recipients": EMAIL_RECIPIENTS,
                               "subject": subject, "attempt": attempt})
        except Exception as e:
            last_err = str(e)

    return json.dumps({"status": "WARNING",
                       "error": f"Email failed after 2 attempts: {last_err}. Report available locally."})


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 6 — Bet Tracker Tools  (ONLY these functions touch bet_tracker.csv)
# ══════════════════════════════════════════════════════════════════════════════

def _load_tracker() -> pd.DataFrame:
    p = Path(FILES["bet_tracker"])
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=BET_TRACKER_COLUMNS)
    return pd.read_csv(p)


def _save_tracker(df: pd.DataFrame) -> None:
    df.to_csv(FILES["bet_tracker"], index=False)


def append_bet(date: str, game: str, model: str, bet_type: str,
               model_prob: float, market_line: str, book: str,
               units: float, notes: str = "") -> str:
    df     = _load_tracker()
    new_id = int(df["id"].max()) + 1 if len(df) > 0 and "id" in df.columns else 1
    row    = {
        "id": new_id, "date": date, "game": game, "model": model,
        "bet_type": bet_type, "model_prob": model_prob, "market_line": market_line,
        "book": book, "units": units, "result": "PENDING",
        "actual_total": None, "profit_loss": None,
        "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "notes": notes,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save_tracker(df)
    return json.dumps({"status": "logged", "bet_id": new_id})


def log_result(bet_id: int, result: str, actual_total: Optional[float] = None) -> str:
    result = result.upper()
    if result not in ("WIN", "LOSS", "PUSH"):
        return json.dumps({"status": "REJECTED", "error": f"Invalid result: {result}."})

    df   = _load_tracker()
    mask = df["id"] == bet_id
    if not mask.any():
        return json.dumps({"status": "REJECTED", "error": f"No bet found with id={bet_id}."})

    existing = str(df.loc[mask, "result"].values[0])
    if existing != "PENDING":
        return json.dumps({"status": "REJECTED",
                           "error": f"❌ Result already recorded as {existing}. Cannot modify a finalized bet."})

    units = float(df.loc[mask, "units"].values[0])
    line  = str(df.loc[mask, "market_line"].values[0])

    if result == "PUSH":
        pl = 0.0
    elif result == "LOSS":
        pl = -units
    else:
        try:
            l = float(line)
            pl = round(units * (100 / abs(l)), 3) if l < 0 else round(units * (l / 100), 3)
        except Exception:
            pl = round(units * 0.909, 3)  # Default -110

    df.loc[mask, "result"]       = result
    df.loc[mask, "profit_loss"]  = pl
    df.loc[mask, "actual_total"] = actual_total
    _save_tracker(df)
    return json.dumps({"status": "result_recorded", "bet_id": bet_id,
                       "result": result, "profit_loss": pl})


def read_tracker_stats() -> str:
    df = _load_tracker()
    if df.empty:
        return json.dumps({"overall": {}, "by_model": {}})

    fin = df[df["result"].isin(["WIN", "LOSS", "PUSH"])]

    def stats(sub: pd.DataFrame) -> dict:
        w = (sub["result"] == "WIN").sum()
        l = (sub["result"] == "LOSS").sum()
        p = (sub["result"] == "PUSH").sum()
        t = w + l
        pl      = float(sub["profit_loss"].sum())
        wagered = float(sub["units"].sum())
        return {
            "record":   f"{w}-{l}-{p}",
            "win_rate": round(w / t, 3) if t > 0 else 0.0,
            "units_pl": round(pl, 2),
            "roi":      round(pl / wagered, 3) if wagered > 0 else 0.0,
            "pending":  int((df["result"] == "PENDING").sum()),
        }

    return json.dumps({
        "overall":  stats(fin),
        "by_model": {m: stats(fin[fin["model"] == m])
                     for m in ["ML", "Totals", "Runline", "F5"]},
    })
