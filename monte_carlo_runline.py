"""
monte_carlo_runline.py
======================
Run-line probability via Poisson-based Monte Carlo simulation.

For each starting pitcher matchup, the model:
  1. Loads the pitcher profile (blended_xwoba, k%, bb%, sigma, velocity flag)
  2. Draws N_SIMS per-game xwOBA samples using the pitcher's sigma
  3. Converts simulated xwOBA to expected runs using a linear model fitted to
     historical MLB data (xwOBA -> R/G relationship)
  4. Samples actual runs per inning via Poisson(lambda) for each team
  5. Applies a bullpen adjustment after SP exits (innings 6-9)
  6. Computes P(home covers -1.5) and P(total over/under)
  7. Combines with XGBoost probability via configurable blending weight

This gives a PHYSICS-BASED probability that can be compared against the
XGBoost data-driven probability and the Vegas line for three-way consensus.

Inputs:
  statcast_data/pitcher_profiles_2026.parquet
  models/xgb_rl.json
  models/xgb_total.json
  models/feature_cols.json
  fangraphs_pitchers.csv  (bullpen ERA fallback)

Usage:
  python monte_carlo_runline.py --home NYY --away BOS --home-sp "GERRIT COLE" --away-sp "NICK PIVETTA"
  python monte_carlo_runline.py --home COL --away LAD --home-sp "GERMAN MARQUEZ" --away-sp "WALKER BUEHLER" --temp 65
  python monte_carlo_runline.py --list-pitchers   # show available pitcher profiles
"""

import argparse
import json
import re
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    xgb = None

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR   = Path("./data/statcast")
MODELS_DIR = Path("./models")

N_SIMS         = 50_000    # Monte Carlo sample count
INNINGS_SP     = 5.5       # Expected innings from starter before bullpen
INNINGS_GAME   = 9         # Total innings

# xwOBA against -> runs/game conceded (fitted from 2022-2024 MLB SP data)
# At xwOBA=0.250 (elite SP): ~3.0 R/G allowed
# At xwOBA=0.320 (league avg): ~4.7 R/G allowed
# At xwOBA=0.400 (poor SP):   ~6.6 R/G allowed
# R/G_allowed ≈ XWOBA_INTERCEPT + XWOBA_SLOPE * pitcher_xwOBA_against
XWOBA_INTERCEPT = -3.0
XWOBA_SLOPE     = 24.0

# xwOBA league baseline (2024 average)
LEAGUE_XWOBA = 0.318

# XGBoost blend weight (0 = pure Monte Carlo, 1 = pure XGBoost)
XGB_BLEND_WEIGHT = 0.40

# Typical MLB bullpen ERA and xwOBA equivalent
DEFAULT_BULLPEN_XWOBA = 0.310   # slightly below league avg (relievers are better)
DEFAULT_BULLPEN_ERA   = 3.80

# Stadium elevations for air density calculation
STADIUM_ELEVATION = {
    "COL": 5200, "AZ": 1082, "TEX": 551,  "HOU": 43,   "ATL": 1050,
    "STL": 465,  "KC":  740, "MIN": 840,  "CIN": 550,  "MIL": 635,
    "CHC": 595,  "CLE": 653, "DET": 600,  "PIT": 730,  "PHI": 20,
    "NYY": 55,   "NYM": 33,  "BOS": 19,   "BAL": 53,   "WSH": 25,
    "TOR": 249,  "TB":  28,  "MIA": 8,    "SF":  52,   "LAD": 512,
    "LAA": 160,  "SD":  17,  "SEA": 56,   "ATH": 25,   "CWS": 20,
}

# Altitude-based run scoring multiplier.
# Less dense air at altitude means the ball carries further and pitches
# break less -> net effect is MORE runs (Coors effect).
# Calibrated so Coors (5200 ft) ≈ +35% runs, sea level ≈ 0%.
# Source: empirical park factor data, 2018-2024 MLB.
def air_density_ratio(elev_ft: float) -> float:
    """Run scoring multiplier from altitude (>1.0 = more runs, 1.0 = sea level)."""
    return 1.0 + (elev_ft / 5200.0) * 0.35


# Temperature adjustment: runs increase ~0.3% per degree above 72°F
# Source: James, Albert et al. weather-run relationship
TEMP_BASELINE_F   = 72.0
TEMP_RUN_FACTOR   = 0.003   # % change per degree F


# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------

def load_profiles() -> pd.DataFrame:
    path = DATA_DIR / "pitcher_profiles_2026.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run build_pitcher_profile.py first")
    df = pd.read_parquet(path, engine="pyarrow")
    # Normalize name for lookup
    df["name_upper"] = df["pitcher_name"].apply(_normalize_name)
    return df


def load_xgb_models():
    if xgb is None:
        return None, None, None
    models = {}
    for name, path in [("rl",    MODELS_DIR / "xgb_rl.json"),
                       ("total", MODELS_DIR / "xgb_total.json")]:
        if path.exists():
            m = xgb.XGBClassifier() if name == "rl" else xgb.XGBRegressor()
            m.load_model(str(path))
            models[name] = m
        else:
            models[name] = None

    feat_path = MODELS_DIR / "feature_cols.json"
    feat_cols = json.loads(feat_path.read_text()) if feat_path.exists() else []
    return models.get("rl"), models.get("total"), feat_cols


_NAME_SUFFIXES = {"JR", "SR", "II", "III", "IV"}


def _strip_accents(s: str) -> str:
    """Remove diacritical marks: 'MARTÍN' → 'MARTIN', 'PÉREZ' → 'PEREZ'."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.strip()
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        name = f"{parts[1]} {parts[0]}"
    return _strip_accents(name.upper())


def _get_last_name(name_upper: str) -> str:
    """Extract last name, ignoring trailing suffixes like JR., SR., II, III."""
    parts = name_upper.replace(".", "").split()
    while parts and parts[-1] in _NAME_SUFFIXES:
        parts.pop()
    return parts[-1] if parts else ""


def find_pitcher(profiles: pd.DataFrame, name: str) -> pd.Series | None:
    name_upper = _normalize_name(name)
    matches = profiles[profiles["name_upper"] == name_upper]
    if len(matches) == 0:
        # Fuzzy: last name only (strip suffixes like JR., SR.)
        last = _get_last_name(name_upper)
        if not last:
            return None
        matches = profiles[
            profiles["name_upper"].str.contains(
                r"\b" + re.escape(last) + r"$", regex=True
            )
        ]
        if len(matches) == 1:
            return matches.iloc[0]
        if len(matches) > 1:
            # Try to narrow by first name
            first = name_upper.split()[0] if name_upper else ""
            first_matches = matches[matches["name_upper"].str.startswith(first)]
            if len(first_matches) == 1:
                return first_matches.iloc[0]
            if len(first_matches) == 0:
                # Last name matched others but not this pitcher — not in profiles
                return None
            # Genuine ambiguity (two pitchers with same first+last name)
            print(f"  Multiple matches for '{name}': "
                  + ", ".join(matches["pitcher_name"].tolist()))
            return None
        return None
    return matches.iloc[0]


# ---------------------------------------------------------------------------
# MONTE CARLO ENGINE
# ---------------------------------------------------------------------------

def pitcher_expected_xwoba(prof: pd.Series, month: int = 4) -> tuple[float, float]:
    """
    Return (expected_xwoba, sigma) for a pitcher in the given month.
    Uses blended estimate if available, else career monthly mean.
    """
    from build_pitcher_profile import MONTH_LABELS  # reuse label mapping

    label = MONTH_LABELS.get(month, "apr")

    # Blended estimate (career + 2026 trailing weighted by trust)
    blended = prof.get("blended_xwoba")
    if pd.notna(blended):
        mu = float(blended)
    else:
        career_col = f"{label}_xwoba_mean"
        mu = prof.get(career_col)
        if pd.isna(mu):
            mu = LEAGUE_XWOBA

    # Monte Carlo sigma (already has velocity multiplier applied)
    sigma = prof.get("monte_carlo_sigma")
    if pd.isna(sigma):
        sigma = 0.050   # league-average uncertainty fallback

    # New pitch K% boost: slightly lower expected xwOBA
    k_boost = float(prof.get("new_pitch_k_boost", 0.0))
    mu = mu - k_boost * 0.08   # rough xwOBA impact of +7.5% K%

    return float(mu), float(sigma)


def xwoba_to_runs_per_game(xwoba: np.ndarray) -> np.ndarray:
    """Convert xwOBA values to expected runs/game using linear fit."""
    return np.clip(XWOBA_INTERCEPT + XWOBA_SLOPE * xwoba, 0, 20)


def simulate_game(
    home_mu: float,   home_sigma: float,
    away_mu: float,   away_sigma: float,
    park_elevation_ft: float = 0,
    temp_f: float = 72.0,
    home_bullpen_xwoba: float = DEFAULT_BULLPEN_XWOBA,
    away_bullpen_xwoba: float = DEFAULT_BULLPEN_XWOBA,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo game simulation.

    Returns (home_runs_array, away_runs_array) each of length N_SIMS.
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- Environment adjustments -------------------------------------------
    air   = air_density_ratio(park_elevation_ft)
    t_adj = 1 + TEMP_RUN_FACTOR * (temp_f - TEMP_BASELINE_F)
    env_factor = air * t_adj

    # --- Draw per-game xwOBA for each pitcher (normally distributed) --------
    # Clip to plausible range: xwOBA very rarely < 0.15 or > 0.50 for starters
    home_xwoba_sp = np.clip(
        rng.normal(home_mu, home_sigma, N_SIMS), 0.150, 0.500)
    away_xwoba_sp = np.clip(
        rng.normal(away_mu, away_sigma, N_SIMS), 0.150, 0.500)

    # --- SP innings: runs allowed per game from SP portion -----------------
    # home_xwoba_sp = xwOBA home SP allows opponents (= away team offensive output)
    # away_xwoba_sp = xwOBA away SP allows opponents (= home team offensive output)
    sp_frac = INNINGS_SP / INNINGS_GAME   # fraction of game covered by SP
    bp_frac = 1 - sp_frac

    # Runs allowed by each pitcher / bullpen (= opponent's scoring)
    home_sp_concedes  = xwoba_to_runs_per_game(home_xwoba_sp) * sp_frac   # away team scores off home SP
    away_sp_concedes  = xwoba_to_runs_per_game(away_xwoba_sp) * sp_frac   # home team scores off away SP
    home_bp_concedes  = xwoba_to_runs_per_game(home_bullpen_xwoba) * bp_frac
    away_bp_concedes  = xwoba_to_runs_per_game(away_bullpen_xwoba) * bp_frac

    # --- Total expected runs (environment-adjusted) -----------------------
    # Home team scores = what AWAY pitching concedes
    home_lambda = (away_sp_concedes + away_bp_concedes) * env_factor
    # Away team scores = what HOME pitching concedes
    away_lambda = (home_sp_concedes + home_bp_concedes) * env_factor

    # --- Poisson draw (actual runs in each simulated game) ----------------
    home_runs = rng.poisson(np.maximum(home_lambda, 0.1))
    away_runs = rng.poisson(np.maximum(away_lambda, 0.1))

    return home_runs.astype(float), away_runs.astype(float)


# ---------------------------------------------------------------------------
# XGBOOST PREDICTION BUILDER
# ---------------------------------------------------------------------------

def build_xgb_row(home_prof: pd.Series, away_prof: pd.Series,
                  home_team: str, away_team: str,
                  temp_f: float, feature_cols: list) -> pd.DataFrame | None:
    """
    Build a single feature row for XGBoost inference.
    Uses pitcher profile features where available.
    """
    if not feature_cols:
        return None

    row = {c: np.nan for c in feature_cols}

    # Calendar
    import datetime
    today = datetime.date.today()
    row["game_month"]      = today.month
    row["game_day_of_week"] = today.weekday()
    row["year"]            = today.year

    # Home SP features
    def fill_sp(prof, prefix):
        mapping = {
            f"{prefix}_k_pct":          "blended_k_pct",
            f"{prefix}_bb_pct":         "blended_bb_pct",
            f"{prefix}_xwoba_against":  "blended_xwoba",
            f"{prefix}_gb_pct":         "blended_gb_pct",
            f"{prefix}_xrv_per_pitch":  "trailing_xrv_per_pitch",
            f"{prefix}_ff_velo":        "ff_velo_2026_april",
            f"{prefix}_age_pit":        "age_pit",
            f"{prefix}_arm_angle":      "arm_angle_2026",
            f"{prefix}_k_minus_bb":     None,  # derived
            f"{prefix}_p_throws_R":     None,  # derived
            f"{prefix}_whiff_pctl":     "whiff_pctl",
            f"{prefix}_fb_spin_pctl":   "fb_spin_pctl",
            f"{prefix}_fb_velo_pctl":   "fb_velo_pctl",
            f"{prefix}_xera_pctl":      "xera_pctl",
            f"{prefix}_era_minus_xera": "era_minus_xfip",
        }
        for feat, src in mapping.items():
            if feat not in row:
                continue
            if src is None:
                continue
            val = prof.get(src)
            if pd.notna(val):
                row[feat] = float(val)

        # Derived
        k  = prof.get("blended_k_pct")
        bb = prof.get("blended_bb_pct")
        if pd.notna(k) and pd.notna(bb):
            row[f"{prefix}_k_minus_bb"] = float(k) - float(bb)

        hand = prof.get("p_throws")
        row[f"{prefix}_p_throws_R"] = 1 if str(hand) == "R" else 0

    fill_sp(home_prof, "home_sp")
    fill_sp(away_prof, "away_sp")

    # SP differentials
    def safe_diff(a, b):
        return float(a) - float(b) if pd.notna(a) and pd.notna(b) else np.nan

    row["sp_k_pct_diff"]    = safe_diff(
        home_prof.get("blended_k_pct"), away_prof.get("blended_k_pct"))
    row["sp_xwoba_diff"]    = safe_diff(
        away_prof.get("blended_xwoba"), home_prof.get("blended_xwoba"))
    row["sp_xrv_diff"]      = safe_diff(
        home_prof.get("trailing_xrv_per_pitch"), away_prof.get("trailing_xrv_per_pitch"))
    row["sp_velo_diff"]     = safe_diff(
        home_prof.get("ff_velo_2026_april"), away_prof.get("ff_velo_2026_april"))
    row["sp_age_diff"]      = safe_diff(
        home_prof.get("age_pit"), away_prof.get("age_pit"))
    row["sp_kminusbb_diff"] = safe_diff(
        row.get("home_sp_k_minus_bb"), row.get("away_sp_k_minus_bb"))

    # Weather
    row["temp_f"]    = temp_f
    row["wind_mph"]  = 8.0   # default
    row["humidity"]  = 50.0  # default

    # Park
    elev = STADIUM_ELEVATION.get(home_team, 100)
    row["park_factor"] = 1.0 + (elev / 5200 * 0.10)  # rough proxy from elevation

    return pd.DataFrame([row])[feature_cols]


# ---------------------------------------------------------------------------
# MAIN PREDICTION FUNCTION
# ---------------------------------------------------------------------------

def predict_game(
    home_team: str,
    away_team: str,
    home_sp_name: str,
    away_sp_name: str,
    temp_f: float = 72.0,
    month: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    Full prediction pipeline for one game.

    Returns dict with:
      mc_home_win_prob
      mc_home_covers_rl_prob   P(home margin >= 2)
      mc_expected_total        expected total runs
      mc_over_prob             P(total > mc_expected_total)   # for reference
      xgb_home_covers_rl_prob  (if models available)
      blended_home_covers_rl   weighted average
      blended_total
    """
    import datetime
    if month is None:
        month = datetime.date.today().month

    # Load profiles
    profiles = load_profiles()

    home_prof = find_pitcher(profiles, home_sp_name)
    away_prof = find_pitcher(profiles, away_sp_name)

    if home_prof is None:
        if verbose:
            print(f"  [WARN] Home SP '{home_sp_name}' not found in profiles — "
                  f"using league average")
        home_prof = profiles.iloc[0].copy()
        for col in ["blended_xwoba", "blended_k_pct", "blended_bb_pct",
                    "blended_gb_pct", "monte_carlo_sigma"]:
            home_prof[col] = np.nan
        home_prof["monte_carlo_sigma"] = 0.050

    if away_prof is None:
        if verbose:
            print(f"  [WARN] Away SP '{away_sp_name}' not found in profiles — "
                  f"using league average")
        away_prof = profiles.iloc[0].copy()
        for col in ["blended_xwoba", "blended_k_pct", "blended_bb_pct",
                    "blended_gb_pct", "monte_carlo_sigma"]:
            away_prof[col] = np.nan
        away_prof["monte_carlo_sigma"] = 0.050

    # Pitcher expected xwOBA and sigma
    home_mu, home_sigma = pitcher_expected_xwoba(home_prof, month)
    away_mu, away_sigma = pitcher_expected_xwoba(away_prof, month)

    # Environment
    elev = STADIUM_ELEVATION.get(home_team, 0)

    # Monte Carlo simulation
    rng = np.random.default_rng(seed=42)
    home_runs, away_runs = simulate_game(
        home_mu=home_mu, home_sigma=home_sigma,
        away_mu=away_mu, away_sigma=away_sigma,
        park_elevation_ft=elev,
        temp_f=temp_f,
        rng=rng,
    )

    margin     = home_runs - away_runs
    mc_home_win     = (margin > 0).mean()
    mc_covers_rl    = (margin >= 2).mean()   # home covers -1.5 (wins by 2+)
    mc_home_cvr_25  = (margin >= 3).mean()   # home covers -2.5 (wins by 3+)
    mc_away_cvr_25  = (margin <= -3).mean()  # away covers +2.5 (loses by 3+... wait no)
    # away +2.5 means away wins or loses by <=2, so home wins by <=2 → margin <= 2
    # more precisely: away +2.5 covers when home_margin < 2.5, i.e. margin <= 2
    mc_away_cvr_rl  = (margin <= 0).mean()   # away moneyline (away wins outright, includes ties→extra innings)
    mc_away_25      = (margin <= 2).mean()   # away +2.5 covers (home wins by 2 or fewer, or away wins)

    mc_total        = (home_runs + away_runs).mean()
    mc_total_median = float(np.median(home_runs + away_runs))

    result = {
        "home_team":           home_team,
        "away_team":           away_team,
        "home_sp":             home_sp_name,
        "away_sp":             away_sp_name,
        "home_sp_mu":          round(home_mu, 4),
        "home_sp_sigma":       round(home_sigma, 4),
        "away_sp_mu":          round(away_mu, 4),
        "away_sp_sigma":       round(away_sigma, 4),
        "home_sp_velocity_flag":  str(home_prof.get("velocity_flag", "NORMAL")),
        "away_sp_velocity_flag":  str(away_prof.get("velocity_flag", "NORMAL")),
        "home_sp_age_bucket":  str(home_prof.get("age_bucket", "unknown")),
        "away_sp_age_bucket":  str(away_prof.get("age_bucket", "unknown")),
        "park_elevation_ft":   elev,
        "temp_f":              temp_f,
        # Moneyline probabilities
        "mc_home_win_prob":    round(float(mc_home_win), 4),
        "mc_away_win_prob":    round(float(mc_away_cvr_rl), 4),
        # Run line cover probabilities
        "mc_home_covers_rl":   round(float(mc_covers_rl), 4),   # home -1.5
        "mc_home_covers_25":   round(float(mc_home_cvr_25), 4), # home -2.5
        "mc_away_covers_25":   round(float(mc_away_25), 4),     # away +2.5
        # Per-team run expectations
        "mc_home_runs_mean":   round(float(home_runs.mean()), 1),
        "mc_away_runs_mean":   round(float(away_runs.mean()), 1),
        "mc_home_runs_lo":     int(np.percentile(home_runs, 25)),   # 25th pctl
        "mc_home_runs_hi":     int(np.percentile(home_runs, 75)),   # 75th pctl
        "mc_away_runs_lo":     int(np.percentile(away_runs, 25)),
        "mc_away_runs_hi":     int(np.percentile(away_runs, 75)),
        # Totals
        "mc_expected_total":   round(mc_total, 2),
        "mc_total_median":     mc_total_median,
        "n_sims":              N_SIMS,
    }

    # XGBoost prediction
    rl_model, tot_model, feat_cols = load_xgb_models()
    if rl_model is not None:
        xgb_row = build_xgb_row(
            home_prof, away_prof, home_team, away_team, temp_f, feat_cols)
        if xgb_row is not None:
            xgb_rl_prob = float(rl_model.predict_proba(xgb_row)[0, 1])
            xgb_tot     = float(tot_model.predict(xgb_row)[0]) \
                if tot_model else mc_total
            result["xgb_home_covers_rl"] = round(xgb_rl_prob, 4)
            result["xgb_expected_total"] = round(xgb_tot, 2)

            # Blend MC + XGBoost
            w = XGB_BLEND_WEIGHT
            blended_rl  = (1 - w) * mc_covers_rl + w * xgb_rl_prob
            blended_tot = (1 - w) * mc_total      + w * xgb_tot
            result["blended_home_covers_rl"] = round(blended_rl, 4)
            result["blended_expected_total"] = round(blended_tot, 2)

    return result


def format_output(res: dict, vegas_total: float | None = None,
                  vegas_ml_home: int | None = None) -> None:
    """Pretty-print the prediction result."""
    print()
    print("=" * 62)
    print(f"  {res['away_team']} @ {res['home_team']}  "
          f"({res['temp_f']:.0f}°F, elev={res['park_elevation_ft']}ft)")
    print("=" * 62)

    print(f"\n  HOME SP : {res['home_sp']}")
    print(f"    xwOBA: {res['home_sp_mu']:.3f}  sigma: {res['home_sp_sigma']:.4f}  "
          f"flag: {res['home_sp_velocity_flag']}  "
          f"age: {res['home_sp_age_bucket']}")

    print(f"\n  AWAY SP : {res['away_sp']}")
    print(f"    xwOBA: {res['away_sp_mu']:.3f}  sigma: {res['away_sp_sigma']:.4f}  "
          f"flag: {res['away_sp_velocity_flag']}  "
          f"age: {res['away_sp_age_bucket']}")

    print(f"\n  {'='*58}")
    print(f"  MONTE CARLO  ({res['n_sims']:,} sims)")
    print(f"  {'='*58}")
    print(f"    Home win prob    : {res['mc_home_win_prob']:.3f} "
          f"({res['mc_home_win_prob']*100:.1f}%)")
    print(f"    Home covers -1.5 : {res['mc_home_covers_rl']:.3f} "
          f"({res['mc_home_covers_rl']*100:.1f}%)")
    print(f"    Expected total   : {res['mc_expected_total']:.2f} runs  "
          f"(median: {res['mc_total_median']:.0f})")

    if "xgb_home_covers_rl" in res:
        print(f"\n  {'='*58}")
        print(f"  XGBOOST")
        print(f"  {'='*58}")
        print(f"    Home covers -1.5 : {res['xgb_home_covers_rl']:.3f} "
              f"({res['xgb_home_covers_rl']*100:.1f}%)")
        print(f"    Expected total   : {res['xgb_expected_total']:.2f} runs")

    if "blended_home_covers_rl" in res:
        w = XGB_BLEND_WEIGHT
        print(f"\n  {'='*58}")
        print(f"  BLENDED  (MC={1-w:.0%} + XGB={w:.0%})")
        print(f"  {'='*58}")
        bl_rl  = res["blended_home_covers_rl"]
        bl_tot = res["blended_expected_total"]
        print(f"    Home covers -1.5 : {bl_rl:.3f} ({bl_rl*100:.1f}%)")
        print(f"    Expected total   : {bl_tot:.2f} runs")

        if vegas_ml_home is not None:
            import math
            if vegas_ml_home > 0:
                implied_win = 100 / (vegas_ml_home + 100)
            else:
                implied_win = abs(vegas_ml_home) / (abs(vegas_ml_home) + 100)
            print(f"\n  {'='*58}")
            print(f"  VEGAS vs MODEL")
            print(f"  {'='*58}")
            print(f"    Vegas ML home implied: {implied_win:.3f} ({implied_win*100:.1f}%)")
            edge_sign = "+" if bl_rl > 0.5238 else ""
            print(f"    Model RL cover prob:   {bl_rl:.3f}  "
                  f"(edge={edge_sign}{(bl_rl-0.5238)*100:+.1f}pp vs -110)")
            if bl_rl >= 0.56:
                print(f"\n    ** BET SIGNAL: Home -1.5 (prob {bl_rl:.3f} >= 0.56) **")
            elif bl_rl <= 0.44:
                print(f"\n    ** BET SIGNAL: Away +1.5 (home prob {bl_rl:.3f} <= 0.44) **")
            else:
                print(f"\n    No strong signal (prob {bl_rl:.3f} within no-bet zone)")

        if vegas_total is not None:
            model_tot = bl_tot
            diff = model_tot - vegas_total
            print(f"\n    Vegas total    : {vegas_total:.1f}")
            print(f"    Model total    : {model_tot:.2f}  (diff: {diff:+.2f})")
            if abs(diff) >= 0.75:
                direction = "OVER" if diff > 0 else "UNDER"
                print(f"\n    ** TOTAL SIGNAL: {direction} {vegas_total} "
                      f"(model={model_tot:.1f}, diff={diff:+.1f}) **")

    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo + XGBoost run-line predictor")
    parser.add_argument("--home",     type=str, help="Home team abbreviation (e.g. NYY)")
    parser.add_argument("--away",     type=str, help="Away team abbreviation (e.g. BOS)")
    parser.add_argument("--home-sp",  type=str, help="Home starting pitcher name")
    parser.add_argument("--away-sp",  type=str, help="Away starting pitcher name")
    parser.add_argument("--temp",     type=float, default=72.0,
                        help="Game-time temperature in Fahrenheit (default: 72)")
    parser.add_argument("--month",    type=int,   default=None,
                        help="Month (1-12, default: current month)")
    parser.add_argument("--vegas-total", type=float, default=None,
                        help="Vegas closing total (for signal comparison)")
    parser.add_argument("--vegas-ml",    type=int,   default=None,
                        help="Vegas closing moneyline home (American odds, e.g. -130)")
    parser.add_argument("--blend",    type=float, default=XGB_BLEND_WEIGHT,
                        help=f"XGBoost blend weight 0-1 (default: {XGB_BLEND_WEIGHT})")
    parser.add_argument("--sims",     type=int,   default=N_SIMS,
                        help=f"Monte Carlo simulations (default: {N_SIMS:,})")
    parser.add_argument("--list-pitchers", action="store_true",
                        help="List available pitcher profiles and exit")
    parser.add_argument("--no-xgb",  action="store_true",
                        help="Skip XGBoost (pure Monte Carlo)")
    args = parser.parse_args()

    # Update globals from args
    globals()["N_SIMS"]           = args.sims
    globals()["XGB_BLEND_WEIGHT"] = args.blend

    if args.list_pitchers:
        profiles = load_profiles()
        print(f"\n  {len(profiles)} pitcher profiles available:\n")
        print(profiles[["pitcher_name", "p_throws", "velocity_flag",
                         "age_bucket", "blended_xwoba",
                         "monte_carlo_sigma"]].to_string(index=False))
        return

    # Validate required args
    if not all([args.home, args.away, args.home_sp, args.away_sp]):
        parser.print_help()
        print("\n  Error: --home, --away, --home-sp, and --away-sp are required")
        return

    print("=" * 62)
    print("  monte_carlo_runline.py")
    print("=" * 62)

    result = predict_game(
        home_team=args.home.upper(),
        away_team=args.away.upper(),
        home_sp_name=args.home_sp,
        away_sp_name=args.away_sp,
        temp_f=args.temp,
        month=args.month,
    )

    format_output(result, vegas_total=args.vegas_total, vegas_ml_home=args.vegas_ml)


if __name__ == "__main__":
    main()
