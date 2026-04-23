"""
models/feature_engineering.py

Environmental + interaction features for the MLB Totals / Run-Distribution model.

Public API
----------
- PARK_CF_AZIMUTH                   : per-park CF compass bearing (deg true north)
- SEASON_OPENING_DAY                : per-season opening-day date
- compute_wind_vector_out(...)      : scalar wind projected onto the CF axis
- compute_days_since_opening_day(...)
- compute_league_rpg_rolling_7d(...)
- create_interaction_features(df)   : adds aero_impact, sp_environment_vulnerability,
                                      thermal_aging per side + diffs
- NEW_FEATURE_COLUMNS               : headers appended for the next training cycle

All pure-function helpers. Call sites: orchestrator/daily_pipeline.py (Step 1 ETL)
and build_feature_matrix.py (training parity).
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Per-park center-field azimuth (compass degrees from true north, home-plate
# → CF wall). Sources: public park-orientation references; values are good
# to ±5° which is well within the noise on 10m wind data.
# ---------------------------------------------------------------------------
PARK_CF_AZIMUTH: dict[str, float] = {
    "ARI":  23.0, "AZ":  23.0,
    "ATL": 151.0,
    "BAL":  60.0,
    "BOS":  45.0,
    "CHC":  42.0,
    "CWS":  38.0,
    "CIN":  75.0,
    "CLE":   0.0,
    "COL":   1.0,
    "DET": 148.0,
    "HOU": 345.0,
    "KC":   45.0,
    "LAA":  68.0,
    "LAD":  25.0,
    "MIA": 130.0,
    "MIL": 130.0,
    "MIN":  90.0,
    "NYM":  25.0,
    "NYY":  75.0,
    "OAK":  57.0,
    "PHI":  14.0,
    "PIT": 118.0,
    "SD":    0.0,
    "SF":   90.0,
    "SEA": 123.0,
    "STL":  72.0,
    "TB":   60.0,
    "TEX":   0.0,
    "TOR":   0.0,
    "WSH":  58.0,
    "ATH":  10.0, "SAC": 10.0,
}

# Season opening-day dates (regular-season, North America). Used for
# days_since_opening_day — a continuous linear warming signal that
# replaces the categorical game_month at the next retrain.
SEASON_OPENING_DAY: dict[int, date] = {
    2023: date(2023, 3, 30),
    2024: date(2024, 3, 28),
    2025: date(2025, 3, 27),
    2026: date(2026, 3, 26),
    2027: date(2027, 3, 25),
}


# ---------------------------------------------------------------------------
# 1. Wind vector on the CF axis
# ---------------------------------------------------------------------------

def compute_wind_vector_out(
    wind_mph: Optional[float],
    wind_bearing_deg: Optional[float],
    home_team: str,
    roof_closed: Optional[float] = None,
) -> Optional[float]:
    """
    Project the wind vector onto the home-plate→CF axis.

        wind_vector_out = wind_mph * cos(bearing_from - CF_azimuth)

    Meteorological convention: `wind_bearing_deg` is the direction the wind
    is coming FROM. A wind blowing FROM CF is blowing IN (toward home plate).
    A wind FROM home plate is blowing OUT. So the sign flips: a wind FROM
    the CF bearing should produce a negative (blowing-in) vector.

        angle = bearing_from - (CF_azimuth + 180)
        wind_out = wind_mph * cos(angle)

    Equivalently: wind_out = -wind_mph * cos(bearing_from - CF_azimuth).

    Returns None when inputs are missing or the park is unknown.
    Returns 0.0 if the roof is closed (wind can't affect flight).
    """
    if roof_closed is not None:
        try:
            if float(roof_closed) >= 0.5:
                return 0.0
        except (TypeError, ValueError):
            pass

    if wind_mph is None or wind_bearing_deg is None:
        return None

    cf_az = PARK_CF_AZIMUTH.get(str(home_team).upper().strip())
    if cf_az is None:
        return None

    try:
        ws = float(wind_mph)
        wb = float(wind_bearing_deg)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(ws) or not np.isfinite(wb):
        return None

    # Wind blowing OUT to CF = wind coming FROM home-plate side of park.
    # That is bearing_from ≈ CF_azimuth + 180. cos() of offset is max (+1)
    # when wind is perfectly "out", -1 when perfectly "in".
    angle_rad = math.radians(wb - (cf_az + 180.0))
    return float(ws * math.cos(angle_rad))


# ---------------------------------------------------------------------------
# 2. Days since opening day
# ---------------------------------------------------------------------------

def _to_date(d) -> Optional[date]:
    if d is None:
        return None
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    try:
        return datetime.strptime(str(d)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def compute_days_since_opening_day(game_date) -> Optional[int]:
    """
    Continuous linear calendar feature: days elapsed from that season's
    opening day. Captures gradual warming of the offensive environment
    better than a categorical game_month.
    """
    gd = _to_date(game_date)
    if gd is None:
        return None
    opener = SEASON_OPENING_DAY.get(gd.year)
    if opener is None:
        # Fallback: estimate opener as last Thursday of March
        opener = date(gd.year, 3, 28)
    return int((gd - opener).days)


# ---------------------------------------------------------------------------
# 3. League RPG, trailing 7 days
# ---------------------------------------------------------------------------

# Cold-start baseline: 2025 full-season league RPG (approx 4.39); we round
# to 4.5 as a stable seed for 2026 opening-week windows that haven't yet
# accumulated 7 days of completed games.
LEAGUE_RPG_BOOTSTRAP: float = 4.5


def compute_league_rpg_rolling_7d(
    target_date,
    history_df: pd.DataFrame,
    total_col: str = "actual_game_total",
    date_col: str = "game_date",
    bootstrap: Optional[float] = LEAGUE_RPG_BOOTSTRAP,
) -> Optional[float]:
    """
    Mean total runs per game across the whole league in the 7 days PRIOR
    to `target_date` (exclusive — avoids leakage for same-day prediction).

    `history_df` must have columns [date_col, total_col] with completed games.
    Returns `bootstrap` if the window is empty or inputs are unusable —
    prevents None propagating into downstream multipliers during opening week.
    Pass bootstrap=None to preserve the pre-seed behaviour (returns None on
    empty window).
    """
    td = _to_date(target_date)
    if td is None or history_df is None or history_df.empty:
        return bootstrap
    if total_col not in history_df.columns or date_col not in history_df.columns:
        return bootstrap

    h = history_df[[date_col, total_col]].copy()
    h[date_col] = pd.to_datetime(h[date_col], errors="coerce").dt.date
    h = h.dropna(subset=[date_col, total_col])

    window_start = td - timedelta(days=7)
    mask = (h[date_col] >= window_start) & (h[date_col] < td)
    vals = pd.to_numeric(h.loc[mask, total_col], errors="coerce").dropna()
    if vals.empty:
        return bootstrap
    return float(vals.mean())


def load_league_rpg_history(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Concatenate backtest_games_{year}.csv files into a single
    (game_date, actual_game_total) frame, used as the source of truth
    for league RPG rolling windows. Returns empty DataFrame if no files
    are found.
    """
    root = Path(data_dir) if data_dir is not None else Path("data/raw")
    frames = []
    for p in sorted(root.glob("backtest_games_*.csv")):
        try:
            df = pd.read_csv(p, usecols=["game_date", "actual_game_total"])
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["game_date", "actual_game_total"])
    out = pd.concat(frames, ignore_index=True)
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.date
    return out.dropna(subset=["game_date"])


# ---------------------------------------------------------------------------
# 4. Interaction features
# ---------------------------------------------------------------------------

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds explicit cross-product features that help the gradient-boosting
    stacker find interactions the deep-feature list might bury.

    Adds (when inputs are present):
      - aero_impact
            air_density_rho * home_park_factor
      - home_sp_environment_vulnerability, away_sp_environment_vulnerability
            (1 - sp_gb_pct) * wind_vector_out
            (FB-leaning pitcher exposure to wind-out parks)
      - sp_environment_vulnerability_diff
            home - away (kept signed so the model sees asymmetry)
      - home_sp_thermal_aging, away_sp_thermal_aging
            sp_age_pit * temp_f
            (older pitchers tend to degrade in cold temps)
      - sp_thermal_aging_diff
            home - away

    Missing inputs → NaN in the output column (no row drops). Operates on
    a copy; returns the new DataFrame.
    """
    out = df.copy()

    # aero_impact
    if {"air_density_rho", "home_park_factor"}.issubset(out.columns):
        out["aero_impact"] = (
            pd.to_numeric(out["air_density_rho"], errors="coerce")
            * pd.to_numeric(out["home_park_factor"], errors="coerce")
        )

    # sp_environment_vulnerability per side + diff
    if {"home_sp_gb_pct", "wind_vector_out"}.issubset(out.columns):
        home_fb = 1.0 - pd.to_numeric(out["home_sp_gb_pct"], errors="coerce")
        out["home_sp_environment_vulnerability"] = (
            home_fb * pd.to_numeric(out["wind_vector_out"], errors="coerce")
        )
    if {"away_sp_gb_pct", "wind_vector_out"}.issubset(out.columns):
        away_fb = 1.0 - pd.to_numeric(out["away_sp_gb_pct"], errors="coerce")
        out["away_sp_environment_vulnerability"] = (
            away_fb * pd.to_numeric(out["wind_vector_out"], errors="coerce")
        )
    if {"home_sp_environment_vulnerability",
        "away_sp_environment_vulnerability"}.issubset(out.columns):
        out["sp_environment_vulnerability_diff"] = (
            out["home_sp_environment_vulnerability"]
            - out["away_sp_environment_vulnerability"]
        )

    # sp_thermal_aging per side + diff
    if {"home_sp_age_pit", "temp_f"}.issubset(out.columns):
        out["home_sp_thermal_aging"] = (
            pd.to_numeric(out["home_sp_age_pit"], errors="coerce")
            * pd.to_numeric(out["temp_f"], errors="coerce")
        )
    if {"away_sp_age_pit", "temp_f"}.issubset(out.columns):
        out["away_sp_thermal_aging"] = (
            pd.to_numeric(out["away_sp_age_pit"], errors="coerce")
            * pd.to_numeric(out["temp_f"], errors="coerce")
        )
    if {"home_sp_thermal_aging", "away_sp_thermal_aging"}.issubset(out.columns):
        out["sp_thermal_aging_diff"] = (
            out["home_sp_thermal_aging"] - out["away_sp_thermal_aging"]
        )

    return out


# ---------------------------------------------------------------------------
# 5. Canonical list of new headers (additive)
# ---------------------------------------------------------------------------
# Order matters only for the v2 training JSON — inference of existing pkl
# models is unaffected because those models read the original
# run_dist_feature_cols.json which remains unchanged.

NEW_FEATURE_COLUMNS: list[str] = [
    "wind_vector_out",
    "dew_point_f",
    "days_since_opening_day",
    "league_rpg_rolling_7d",
    "aero_impact",
    "home_sp_environment_vulnerability",
    "away_sp_environment_vulnerability",
    "sp_environment_vulnerability_diff",
    "home_sp_thermal_aging",
    "away_sp_thermal_aging",
    "sp_thermal_aging_diff",
]
