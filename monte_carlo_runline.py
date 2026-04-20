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

try:
    import lightgbm as lgb_mc
    _LGBM_MC = True
except ImportError:
    lgb_mc = None
    _LGBM_MC = False

try:
    import catboost as cb_mc
    _CATBOOST_MC = True
except ImportError:
    cb_mc = None
    _CATBOOST_MC = False

# ── GPU backend: CuPy with transparent numpy fallback ──────────────────────
# All simulation arrays are generated on the RTX 5080 when CuPy is available.
# If CuPy is absent or no CUDA device is present, every cp.* call silently
# falls back to the equivalent numpy operation — zero code changes required.
try:
    import cupy as cp
    _GPU = cp.cuda.is_available()
    if _GPU:
        # Probe cuRAND — curand*.dll may be absent even when CUDA device exists
        try:
            _probe = cp.random.standard_normal(1)
            del _probe
        except (ImportError, Exception):
            _GPU = False
    if not _GPU:
        import numpy as cp          # no cuRAND or no device → CPU fallback
except ImportError:
    import numpy as cp              # CuPy not installed → CPU fallback
    _GPU = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR   = Path("./data/statcast")
MODELS_DIR = Path("./models")

# Columns negated when building the away-perspective feature row for xgb_rl_team.
# Must match _DIFF_COLS_TO_NEGATE in train_xgboost.py exactly.
_TEAM_DIFF_COLS = [
    "sp_k_pct_diff", "sp_xwoba_diff", "sp_xrv_diff", "sp_velo_diff",
    "sp_age_diff", "sp_kminusbb_diff", "sp_k_pct_10d_diff", "sp_xwoba_10d_diff",
    "sp_bb_pct_10d_diff", "batting_matchup_edge", "batting_matchup_edge_10d",
    "bp_era_diff", "bp_k9_diff", "bp_whip_diff", "circadian_edge",
]

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

# ── Bivariate run-scoring parameters (Poisson-LogNormal copula) ─────────────
#
# Model:  H | V_h ~ Poisson(μ_h · V_h),  A | V_a ~ Poisson(μ_a · V_a)
#         [log V_h, log V_a] ~ BVN(−σ²/2·1, σ²·[[1,ρ],[ρ,1]])
#
# σ_NB = 0.50:  log-normal mixing SD, equivalent to NB dispersion r ≈ 4.
#   Var(X) = μ + μ²·(e^σ² − 1) = 4.4 + 4.4²·0.284 = 9.9  →  σ_runs ≈ 3.14  ✓
#
# ρ_COPULA = 0.14: Gaussian copula correlation.
#   Corr(H,A) = μ_h·μ_a·(e^(σ²·ρ)−1) / √(Var_h·Var_a)
#             = 19.36·(e^0.0175−1) / 9.9 ≈ 0.070  (validated 2019-2024 MLB)
#
RUN_SIGMA_NB   : float = 0.50   # log-normal mixing SD  → overdispersion r ≈ 4
RUN_RHO_COPULA : float = 0.14   # copula corr → run-level Corr(H,A) ≈ 0.07

# ── K-prop Negative Binomial dispersion ──────────────────────────────────────
# MLE on 470 pitcher-starts (2026): actual K distribution is near-Poisson
# (r_mle → ∞; var=5.67 vs mean=4.75).  r=20 adds minimal overdispersion for
# early-exit tail; Gamma-Poisson mixture: λ ~ Γ(r_k, μ/r_k), K ~ Poisson(λ)
KPROP_NB_R: float = 20.0

# ── K-prop mean calibration ──────────────────────────────────────────────────
# blended_k_pct overestimates by ~1.8pp and expected_ip by ~0.25 inn.
# MLE on 470 starts: pred μ=5.29, actual μ=4.75 → multiplier=4.75/5.29=0.898
# Apply before simulation to remove systematic overestimation bias.
KPROP_MEAN_CALIB: float = 0.899

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


def load_pitcher_10d() -> dict:
    """
    Load trailing-10-day SP stats → dict keyed by pitcher_name_upper.
    Returns {} if file not present (graceful degradation).
    """
    path = DATA_DIR / "pitcher_10d_2026.parquet"
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path, engine="pyarrow")
        return {
            row["pitcher_name_upper"]: {
                "k_pct_10d": row.get("k_pct_10d"),
                "xwoba_10d": row.get("xwoba_10d"),
            }
            for _, row in df.iterrows()
            if pd.notna(row.get("pitcher_name_upper"))
        }
    except Exception:
        return {}


def _shadow_predict_lgbm(model, X_df) -> float:
    """
    Unified LightGBM shadow inference.
    Handles both lgb.Booster (native .json) and LGBMClassifier (sklearn pkl).
    lgb.Booster.predict() returns the sigmoid probability directly for binary tasks.
    """
    if hasattr(model, "predict_proba"):
        # LGBMClassifier (pkl fallback)
        return float(model.predict_proba(X_df)[0, 1])
    # lgb.Booster (native json — loaded via lgb_mc.Booster)
    arr = X_df.values.astype("float64") if hasattr(X_df, "values") else X_df
    return float(model.predict(arr)[0])


def _shadow_predict_catboost(model, X_arr) -> float:
    """
    Unified CatBoost shadow inference.
    Both native .cbm (CatBoostClassifier.load_model) and pkl use predict_proba.
    X_arr must already be a float64 numpy array (pd.NA → np.nan).
    """
    return float(model.predict_proba(X_arr)[0, 1])


def load_xgb_models():
    """
    Load official XGBoost models + Level-2 stacking LR plus the two Shadow
    models (LightGBM, CatBoost) used exclusively for variance estimation.

    Shadow model load order:
      1. Native format  — lgb_shadow.json / cat_shadow.cbm   (preferred)
      2. Sklearn pkl    — lgbm_rl.pkl / cat_rl.pkl           (legacy fallback)

    Returns
    -------
    rl_model, tot_model, ml_model, feat_cols, lgbm_shadow, cat_shadow, stacking
    """
    import pickle as _pkl

    if xgb is None:
        return None, None, None, [], None, None, None, None, None

    # ── XGBoost (required — official model) ──────────────────────────────
    models = {}
    for name, path in [("rl",    MODELS_DIR / "xgb_rl.json"),
                       ("total", MODELS_DIR / "xgb_total.json"),
                       ("ml",    MODELS_DIR / "xgb_ml.json")]:
        if path.exists():
            m = xgb.XGBRegressor() if name == "total" else xgb.XGBClassifier()
            m.load_model(str(path))
            if _GPU:
                m.set_params(device="cuda")
            models[name] = m
        else:
            models[name] = None

    feat_path = MODELS_DIR / "feature_cols.json"
    feat_cols = json.loads(feat_path.read_text()) if feat_path.exists() else []

    # ── Team-perspective RL model (optional — doubles dataset with is_home col) ─
    team_rl_model = None
    team_feat_cols = None
    team_rl_path   = MODELS_DIR / "xgb_rl_team.json"
    team_feat_path = MODELS_DIR / "feature_cols_team.json"
    if team_rl_path.exists():
        try:
            _tm = xgb.XGBClassifier()
            _tm.load_model(str(team_rl_path))
            if _GPU:
                _tm.set_params(device="cuda")
            team_rl_model = _tm
            print(f"  [MC] Team RL model loaded: {team_rl_path.name}")
        except Exception as e:
            print(f"  [WARN] Could not load {team_rl_path.name}: {e}")
    if team_feat_path.exists():
        try:
            team_feat_cols = json.loads(team_feat_path.read_text())
        except Exception as e:
            print(f"  [WARN] Could not load {team_feat_path.name}: {e}")

    # ── LightGBM Shadow (optional) ────────────────────────────────────────
    # Prefer native .json (lgb.Booster); fall back to sklearn pkl.
    lgbm_shadow = None
    if _LGBM_MC:
        for p, loader in [
            (MODELS_DIR / "lgb_shadow.json",
             lambda p: lgb_mc.Booster(model_file=str(p))),
            (MODELS_DIR / "lgbm_rl.pkl",
             lambda p: _pkl.loads(p.read_bytes())),
        ]:
            if p.exists():
                try:
                    lgbm_shadow = loader(p)
                    print(f"  [MC] LightGBM shadow: {p.name}")
                    break
                except Exception as e:
                    print(f"  [WARN] Could not load {p.name}: {e}")

    # ── CatBoost Shadow (optional) ────────────────────────────────────────
    # Prefer native .cbm (CatBoostClassifier.load_model); fall back to pkl.
    cat_shadow = None
    if _CATBOOST_MC:
        for p, fmt in [
            (MODELS_DIR / "cat_shadow.cbm", "cbm"),
            (MODELS_DIR / "cat_rl.pkl",     None),
        ]:
            if p.exists():
                try:
                    if fmt == "cbm":
                        m = cb_mc.CatBoostClassifier()
                        m.load_model(str(p), format="cbm")
                        cat_shadow = m
                    else:
                        cat_shadow = _pkl.loads(p.read_bytes())
                    print(f"  [MC] CatBoost shadow: {p.name}")
                    break
                except Exception as e:
                    print(f"  [WARN] Could not load {p.name}: {e}")

    # ── Stacking model (official — XGBoost-only input) ────────────────────
    # Supports both new BayesianStacker (v5.1) and legacy StackingModel pickles.
    stacking = None
    stk_path = MODELS_DIR / "stacking_lr_rl.pkl"
    if stk_path.exists():
        try:
            import sys, train_xgboost as _txgb
            # Register both classes so pickle can deserialise either format
            for _cls_name in ("BayesianStacker", "StackingModel"):
                if hasattr(_txgb, _cls_name):
                    _cls = getattr(_txgb, _cls_name)
                    if not hasattr(sys.modules.get("__main__"), _cls_name):
                        sys.modules["__main__"].__dict__[_cls_name] = _cls
                    # Also register under __main__ module path that pickle uses
                    sys.modules[__name__].__dict__[_cls_name] = _cls
            stacking = _pkl.loads(stk_path.read_bytes())
            _stk_type = type(stacking).__name__
            print(f"  [MC] Stacking model loaded: {_stk_type}")
        except Exception as e:
            print(f"  [WARN] Could not load stacking_lr_rl.pkl: {e}")

    n_shadow = (lgbm_shadow is not None) + (cat_shadow is not None)
    print(f"  [MC] Official: XGBoost -> Bayesian Stacker"
          f"  |  Shadow: {n_shadow}/2 loaded"
          f"{' (LGBM' if lgbm_shadow else ''}"
          f"{'+CAT' if lgbm_shadow and cat_shadow else 'CAT' if cat_shadow else ''}"
          f"{')' if n_shadow else ''}")

    return (models.get("rl"), models.get("total"), models.get("ml"), feat_cols,
            lgbm_shadow, cat_shadow, stacking, team_rl_model, team_feat_cols)


def _build_away_xgb_row(row_df: pd.DataFrame) -> pd.DataFrame:
    """Build away-perspective feature row: swap home/away cols, negate diffs, set is_home=0."""
    away = row_df.copy()
    cols = set(away.columns)
    for col in list(cols):
        if col.startswith("home_"):
            pair = "away_" + col[5:]
            if pair in cols:
                away[col]  = row_df[pair].values
                away[pair] = row_df[col].values
    for h_col, a_col in [("home_bat_vs_away_sp", "away_bat_vs_home_sp"),
                          ("home_bat_vs_away_sp_10d", "away_bat_vs_home_sp_10d")]:
        if h_col in cols and a_col in cols:
            away[h_col] = row_df[a_col].values
            away[a_col] = row_df[h_col].values
    if "true_home_prob" in cols and "true_away_prob" in cols:
        away["true_home_prob"] = row_df["true_away_prob"].values
        away["true_away_prob"] = row_df["true_home_prob"].values
    for col in _TEAM_DIFF_COLS:
        if col in cols:
            away[col] = -row_df[col].values
    if "is_home" in cols:
        away["is_home"] = 0
    return away


# Calibrator cache — loaded once per process
_calibrators: dict = {}


def load_calibrator(name: str):
    """
    Load an isotonic regression calibrator from models/calibrator_{name}.pkl.
    Returns the calibrator object, or None if the file doesn't exist.
    Results are cached so the file is only read once per process.
    """
    if name in _calibrators:
        return _calibrators[name]
    import pickle
    path = MODELS_DIR / f"calibrator_{name}.pkl"
    if path.exists():
        try:
            cal = pickle.loads(path.read_bytes())
            _calibrators[name] = cal
            return cal
        except Exception:
            pass
    _calibrators[name] = None
    return None


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

    # ── Thin-sample shrinkage ─────────────────────────────────────────────────
    # When a pitcher has very few 2026 starts (trailing_trust < 0.30), the
    # blended_xwoba is heavily weighted by career data which may not reflect
    # current form (e.g. post-TJ, age, innings limits, reduced velocity).
    # Shrink the estimate toward league average to avoid over-confidence.
    trailing_trust = float(prof.get("trailing_trust_xwoba", 0.5) or 0.5)
    if pd.isna(trailing_trust):
        trailing_trust = 0.5
    if trailing_trust < 0.30:
        # shrink_weight: 0 at trust=0.30, up to 0.20 at trust=0 (20% pull toward league avg)
        shrink_weight = (0.30 - trailing_trust) / 0.30 * 0.20
        mu = (1 - shrink_weight) * mu + shrink_weight * LEAGUE_XWOBA
        # Also widen sigma to capture added uncertainty from small sample
        sigma = min(sigma * (1 + shrink_weight * 0.5), 0.110)

    # New pitch K% boost: slightly lower expected xwOBA
    k_boost = float(prof.get("new_pitch_k_boost", 0.0))
    mu = mu - k_boost * 0.08   # rough xwOBA impact of +7.5% K%

    return float(mu), float(sigma)


def xwoba_to_runs_per_game(xwoba: np.ndarray) -> np.ndarray:
    """Convert xwOBA values to expected runs/game using linear fit."""
    return np.clip(XWOBA_INTERCEPT + XWOBA_SLOPE * xwoba, 0, 20)


LEAGUE_RS_PER_GAME = 4.38   # 2024 MLB average runs scored per team per game
BF_PER_INNING      = 4.3    # Average batters faced per inning (for K prop model)


def simulate_game(
    home_mu: float,   home_sigma: float,
    away_mu: float,   away_sigma: float,
    park_elevation_ft: float = 0,
    temp_f: float = 72.0,
    home_bullpen_xwoba: float = DEFAULT_BULLPEN_XWOBA,
    away_bullpen_xwoba: float = DEFAULT_BULLPEN_XWOBA,
    home_team_rs_per_game: float | None = None,
    away_team_rs_per_game: float | None = None,
    home_expected_ip: float = INNINGS_SP,
    away_expected_ip: float = INNINGS_SP,
    rng=None,                   # accepted but ignored — GPU uses its own RNG
) -> tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated Monte Carlo game simulation.

    Run-scoring model: Poisson-LogNormal compound with Gaussian Copula.
      H | V_h ~ Poisson(μ_h · V_h)
      A | V_a ~ Poisson(μ_a · V_a)
      [log V_h, log V_a] ~ BVN(−σ²/2·𝟏,  σ²·[[1, ρ], [ρ, 1]])

    This correctly captures:
      • Overdispersion:  Var(X) = μ + μ²(e^σ² − 1) ≈ 9.9  vs Poisson 4.4
      • Scoring correlation:  Corr(H, A) ≈ 0.07  (shared park/weather/umpire effects)

    All array operations run on the RTX 5080 via CuPy.  Results are returned
    as numpy arrays so the rest of the call stack requires zero changes.

    Returns (home_runs_array, away_runs_array) each of length N_SIMS.
    """
    # ── Team offensive quality factors ─────────────────────────────────────
    def off_factor(rs_pg):
        if rs_pg is None or (isinstance(rs_pg, float) and np.isnan(rs_pg)):
            return 1.0
        raw = float(rs_pg) / LEAGUE_RS_PER_GAME
        return 0.50 * raw + 0.50          # 50% blend toward league average

    home_off = off_factor(home_team_rs_per_game)
    away_off = off_factor(away_team_rs_per_game)

    # ── Environment multiplier (altitude × temperature) ────────────────────
    env_factor = air_density_ratio(park_elevation_ft) * (
        1.0 + TEMP_RUN_FACTOR * (temp_f - TEMP_BASELINE_F)
    )

    # ── Pitcher xwOBA draws (CPU, scalar → GPU lambda arrays) ──────────────
    # Use numpy for the small (N_SIMS,) normal draw; xwOBA per-game sigma
    # is the pitcher's career-spring dispersion, not the run-scoring overdispersion.
    _rng = np.random.default_rng() if rng is None else rng
    home_xwoba_sp = np.clip(_rng.normal(home_mu, home_sigma, N_SIMS), 0.150, 0.500)
    away_xwoba_sp = np.clip(_rng.normal(away_mu, away_sigma, N_SIMS), 0.150, 0.500)

    # ── SP / Bullpen innings split ─────────────────────────────────────────
    home_sp_frac = float(np.clip(home_expected_ip, 1.0, INNINGS_GAME)) / INNINGS_GAME
    away_sp_frac = float(np.clip(away_expected_ip, 1.0, INNINGS_GAME)) / INNINGS_GAME
    home_bp_frac = 1.0 - home_sp_frac
    away_bp_frac = 1.0 - away_sp_frac

    # Expected runs conceded (= opponent's expected scoring rate) per game
    # Computed on CPU (vectorised numpy) — only N_SIMS floats, negligible cost
    _xwoba_to_rpg = lambda x: np.clip(XWOBA_INTERCEPT + XWOBA_SLOPE * x, 0.0, 20.0)
    mu_home = ((_xwoba_to_rpg(away_xwoba_sp) * away_sp_frac
                + _xwoba_to_rpg(away_bullpen_xwoba) * away_bp_frac)
               * env_factor * home_off)
    mu_away = ((_xwoba_to_rpg(home_xwoba_sp) * home_sp_frac
                + _xwoba_to_rpg(home_bullpen_xwoba) * home_bp_frac)
               * env_factor * away_off)

    # Clamp lambdas to a sane floor before GPU transfer
    mu_home = np.maximum(mu_home, 0.10)
    mu_away = np.maximum(mu_away, 0.10)

    # ── GPU: Poisson-LogNormal correlated run draws ────────────────────────
    # Move expected-run arrays to GPU (no-op if cp == np / CPU fallback)
    mu_h_gpu = cp.asarray(mu_home, dtype=cp.float32)   # [N_SIMS]
    mu_a_gpu = cp.asarray(mu_away, dtype=cp.float32)   # [N_SIMS]

    # Correlated standard normals via Cholesky:
    #   L = [[1, 0], [ρ, √(1−ρ²)]]  so that L Lᵀ = [[1,ρ],[ρ,1]]
    rho   = float(RUN_RHO_COPULA)
    sigma = float(RUN_SIGMA_NB)
    z1    = cp.random.standard_normal(N_SIMS).astype(cp.float32)
    z2    = cp.random.standard_normal(N_SIMS).astype(cp.float32)
    eps_h = z1                                                   # [N_SIMS]
    eps_a = rho * z1 + float(np.sqrt(1.0 - rho ** 2)) * z2      # [N_SIMS]

    # Log-normal mixing: V = exp(σ·ε − σ²/2)  →  E[V]=1, Var(V)=e^σ²−1
    V_h = cp.exp(sigma * eps_h - 0.5 * sigma * sigma)   # [N_SIMS]
    V_a = cp.exp(sigma * eps_a - 0.5 * sigma * sigma)   # [N_SIMS]

    # Compound Poisson draw:  runs | V ~ Poisson(μ · V)
    home_runs_gpu = cp.random.poisson(mu_h_gpu * V_h).astype(cp.float32)
    away_runs_gpu = cp.random.poisson(mu_a_gpu * V_a).astype(cp.float32)

    # Return as CPU numpy arrays (cp.asnumpy = np.asarray when cp == np)
    if _GPU:
        home_runs = cp.asnumpy(home_runs_gpu)
        away_runs = cp.asnumpy(away_runs_gpu)
    else:
        home_runs = np.asarray(home_runs_gpu)
        away_runs = np.asarray(away_runs_gpu)

    return home_runs.astype(float), away_runs.astype(float)


def simulate_k_prop_nb(
    mu_k: float,
    r_k: float = KPROP_NB_R,
    n_sims: int = N_SIMS,
) -> np.ndarray:
    """
    Sample K-prop counts using a Negative Binomial via Gamma-Poisson mixture.

    MLE on 470 2026 pitcher-starts shows the actual K distribution is near-
    Poisson (r_mle → ∞).  r=20 retains a small overdispersion allowance for
    early-exit games without distorting tail probabilities.  mu_k must already
    include the KPROP_MEAN_CALIB multiplier applied at the call site.

    Implementation:
      λ ~ Gamma(r_k, μ_k / r_k)    [shape=r_k, scale=μ_k/r_k]
      K ~ Poisson(λ)                 [exact NB marginal]

    Parameters
    ----------
    mu_k    : expected K count (already calibrated via KPROP_MEAN_CALIB)
    r_k     : NB dispersion (default 20.0 → near-Poisson)
    n_sims  : number of Monte Carlo draws

    Returns
    -------
    np.ndarray of shape (n_sims,) — integer K counts
    """
    mu_k = max(float(mu_k), 0.01)
    scale = mu_k / r_k   # Gamma scale parameter

    # GPU Gamma → Poisson (NB Gamma-Poisson mixture)
    lam_k = cp.random.gamma(r_k, scale, size=n_sims).astype(cp.float32)
    k_gpu = cp.random.poisson(lam_k).astype(cp.float32)

    return cp.asnumpy(k_gpu) if _GPU else np.asarray(k_gpu)


# ---------------------------------------------------------------------------
# XGBOOST PREDICTION BUILDER
# ---------------------------------------------------------------------------

def build_xgb_row(home_prof: pd.Series, away_prof: pd.Series,
                  home_team: str, away_team: str,
                  temp_f: float, feature_cols: list,
                  wind_mph: float = 8.0,
                  home_lineup_platoon: dict | None = None,
                  away_lineup_platoon: dict | None = None,
                  home_team_stats: pd.Series | None = None,
                  away_team_stats: pd.Series | None = None,
                  pitcher_10d: dict | None = None,
                  market_odds: dict | None = None,
                  home_lineup_wrc: float | None = None,
                  away_lineup_wrc: float | None = None,
                  game_hour_et: float | None = None) -> pd.DataFrame | None:
    """
    Build a single feature row for XGBoost inference.
    Uses pitcher profile features where available.

    market_odds (optional) dict keys:
        true_home_prob  -- de-vigged home win probability (Pinnacle preferred)
        true_away_prob  -- de-vigged away win probability
        close_total     -- closing O/U total (for context only, not a trained feature)

    home_lineup_wrc / away_lineup_wrc (optional):
        Today's actual lineup wRC+ (100 = league average).  When provided, all
        batting xwOBA features for that team are scaled by (wrc / 100) to
        reflect lineup quality vs. the season-average batting profile.
        No retraining is needed — this is a multiplicative adjustment applied
        only at inference time.

    game_hour_et (optional):
        Scheduled start hour in Eastern Time (float, e.g. 13.75 = 1:45 PM ET).
        Used to compute circadian features:
          game_hour_et          — raw ET start hour
          away_game_local_hour  — what time the away team's body thinks it is
          is_day_game           — 1 if before 5 PM ET
          circadian_edge        — home minus away local hour (positive = home advantage)
        Defaults to 19.0 (7 PM ET, a neutral evening game) if not provided.
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

    # Trailing-10-day SP stats (k_pct_10d, xwoba_10d) from pitcher_10d lookup
    if pitcher_10d:
        home_sp_name = home_prof.get("pitcher_name_upper") or home_prof.get("name_upper", "")
        away_sp_name = away_prof.get("pitcher_name_upper") or away_prof.get("name_upper", "")
        for sp_name, prefix in [(home_sp_name, "home_sp"), (away_sp_name, "away_sp")]:
            sp_10d = pitcher_10d.get(sp_name, {})
            for src, col in [("k_pct_10d",  f"{prefix}_k_pct_10d"),
                              ("xwoba_10d",  f"{prefix}_xwoba_10d")]:
                if col in row and pd.notna(sp_10d.get(src)):
                    row[col] = float(sp_10d[src])

    # SP differentials
    def safe_diff(a, b):
        return float(a) - float(b) if pd.notna(a) and pd.notna(b) else np.nan

    row["sp_k_pct_diff"]    = safe_diff(
        home_prof.get("blended_k_pct"), away_prof.get("blended_k_pct"))
    row["sp_xwoba_diff"]    = safe_diff(
        away_prof.get("blended_xwoba"), home_prof.get("blended_xwoba"))
    # 10d diffs — only set if both SP 10d values are present
    row["sp_k_pct_10d_diff"] = safe_diff(
        row.get("home_sp_k_pct_10d"), row.get("away_sp_k_pct_10d"))
    row["sp_xwoba_10d_diff"] = safe_diff(
        row.get("away_sp_xwoba_10d"), row.get("home_sp_xwoba_10d"))   # away − home
    row["sp_xrv_diff"]      = safe_diff(
        home_prof.get("trailing_xrv_per_pitch"), away_prof.get("trailing_xrv_per_pitch"))
    row["sp_velo_diff"]     = safe_diff(
        home_prof.get("ff_velo_2026_april"), away_prof.get("ff_velo_2026_april"))
    row["sp_age_diff"]      = safe_diff(
        home_prof.get("age_pit"), away_prof.get("age_pit"))
    row["sp_kminusbb_diff"] = safe_diff(
        row.get("home_sp_k_minus_bb"), row.get("away_sp_k_minus_bb"))

    # Team batting splits (season-to-date + trailing-10-day)
    def fill_batting(stats, prefix):
        if stats is None:
            return
        for col in ["bat_xwoba_vs_rhp", "bat_xwoba_vs_lhp",
                    "bat_k_vs_rhp",     "bat_k_vs_lhp",
                    "bat_bb_vs_rhp",    "bat_bb_vs_lhp",
                    "bat_xwoba_vs_rhp_10d", "bat_xwoba_vs_lhp_10d"]:
            feat = f"{prefix}_{col}"
            if feat in row and pd.notna(stats.get(col)):
                row[feat] = float(stats.get(col))

    fill_batting(home_team_stats, "home")
    fill_batting(away_team_stats, "away")

    # Bullpen stats from team_stats (era / whip / k9)
    def fill_bullpen(stats, prefix):
        if stats is None:
            return
        for col, src in [
            (f"{prefix}_bp_era",  "bullpen_era"),
            (f"{prefix}_bp_whip", "bullpen_whip"),
            (f"{prefix}_bp_k9",   "bullpen_k9"),
        ]:
            if col in row and pd.notna(stats.get(src)):
                row[col] = float(stats.get(src))

    fill_bullpen(home_team_stats, "home")
    fill_bullpen(away_team_stats, "away")

    # Bullpen differentials (home minus away; negative = home bullpen advantage)
    row["bp_era_diff"]  = safe_diff(row.get("home_bp_era"),  row.get("away_bp_era"))
    row["bp_whip_diff"] = safe_diff(row.get("home_bp_whip"), row.get("away_bp_whip"))
    row["bp_k9_diff"]   = safe_diff(row.get("home_bp_k9"),   row.get("away_bp_k9"))

    # Rolling 15-game momentum features (v2 model additions)
    for prefix, ts in [("home", home_team_stats), ("away", away_team_stats)]:
        if ts is not None:
            rd  = ts.get("rolling_rd_15g")
            pyth = ts.get("pyth_win_pct_15g")
            if pd.notna(rd)   and f"{prefix}_rolling_rd_15g"   in row:
                row[f"{prefix}_rolling_rd_15g"]   = float(rd)
            if pd.notna(pyth) and f"{prefix}_pyth_win_pct_15g" in row:
                row[f"{prefix}_pyth_win_pct_15g"] = float(pyth)
    row["rolling_rd_diff"]   = safe_diff(row.get("home_rolling_rd_15g"),   row.get("away_rolling_rd_15g"))
    row["pyth_win_pct_diff"] = safe_diff(row.get("home_pyth_win_pct_15g"), row.get("away_pyth_win_pct_15g"))

    # Lineup wRC+ as direct ML features (v2 model additions — from today's actual lineup)
    if home_lineup_wrc is not None and pd.notna(float(home_lineup_wrc)):
        if "home_lineup_wrc_plus" in row:
            row["home_lineup_wrc_plus"] = float(home_lineup_wrc)
    if away_lineup_wrc is not None and pd.notna(float(away_lineup_wrc)):
        if "away_lineup_wrc_plus" in row:
            row["away_lineup_wrc_plus"] = float(away_lineup_wrc)
    if "lineup_wrc_plus_diff" in row:
        row["lineup_wrc_plus_diff"] = safe_diff(
            row.get("home_lineup_wrc_plus"), row.get("away_lineup_wrc_plus"))

    # Matchup-specific batting edge (home team vs away SP handedness)
    away_sp_rhp = row.get("away_sp_p_throws_R", 1)
    home_sp_rhp = row.get("home_sp_p_throws_R", 1)
    if home_team_stats is not None and pd.notna(away_sp_rhp):
        home_bat_vs_sp = (home_team_stats.get("bat_xwoba_vs_rhp")
                          if away_sp_rhp == 1
                          else home_team_stats.get("bat_xwoba_vs_lhp"))
        if "home_bat_vs_away_sp" in row and pd.notna(home_bat_vs_sp):
            row["home_bat_vs_away_sp"] = float(home_bat_vs_sp)
    if away_team_stats is not None and pd.notna(home_sp_rhp):
        away_bat_vs_sp = (away_team_stats.get("bat_xwoba_vs_rhp")
                          if home_sp_rhp == 1
                          else away_team_stats.get("bat_xwoba_vs_lhp"))
        if "away_bat_vs_home_sp" in row and pd.notna(away_bat_vs_sp):
            row["away_bat_vs_home_sp"] = float(away_bat_vs_sp)
    if ("home_bat_vs_away_sp" in row and "away_bat_vs_home_sp" in row
            and pd.notna(row["home_bat_vs_away_sp"])
            and pd.notna(row["away_bat_vs_home_sp"])):
        row["batting_matchup_edge"] = (row["home_bat_vs_away_sp"]
                                       - row["away_bat_vs_home_sp"])

    # Trailing-10-day matchup edge — same handedness logic, 10d batting splits
    if home_team_stats is not None and pd.notna(away_sp_rhp):
        home_bat_10d = (home_team_stats.get("bat_xwoba_vs_rhp_10d")
                        if away_sp_rhp == 1
                        else home_team_stats.get("bat_xwoba_vs_lhp_10d"))
        if pd.notna(home_bat_10d):
            row["home_bat_vs_away_sp_10d"] = float(home_bat_10d)
    if away_team_stats is not None and pd.notna(home_sp_rhp):
        away_bat_10d = (away_team_stats.get("bat_xwoba_vs_rhp_10d")
                        if home_sp_rhp == 1
                        else away_team_stats.get("bat_xwoba_vs_lhp_10d"))
        if pd.notna(away_bat_10d):
            row["away_bat_vs_home_sp_10d"] = float(away_bat_10d)
    if ("home_bat_vs_away_sp_10d" in row and "away_bat_vs_home_sp_10d" in row
            and pd.notna(row.get("home_bat_vs_away_sp_10d"))
            and pd.notna(row.get("away_bat_vs_home_sp_10d"))):
        row["batting_matchup_edge_10d"] = (row["home_bat_vs_away_sp_10d"]
                                           - row["away_bat_vs_home_sp_10d"])

    # Lineup wRC+ scaling — adjust batting xwOBA features by today's lineup quality.
    # Multiplier = wrc / 100  (e.g. 110 wRC+ → 1.10x, i.e. 10% better than season avg).
    # Applied after all batting/matchup features are set so derived edges are also scaled.
    # K/BB rates are not scaled (those reflect pitcher traits more than lineup composition).
    _xwoba_bat_feats_home = [
        "home_bat_xwoba_vs_rhp", "home_bat_xwoba_vs_lhp",
        "home_bat_xwoba_vs_rhp_10d", "home_bat_xwoba_vs_lhp_10d",
        "home_bat_vs_away_sp", "home_bat_vs_away_sp_10d",
    ]
    _xwoba_bat_feats_away = [
        "away_bat_xwoba_vs_rhp", "away_bat_xwoba_vs_lhp",
        "away_bat_xwoba_vs_rhp_10d", "away_bat_xwoba_vs_lhp_10d",
        "away_bat_vs_home_sp", "away_bat_vs_home_sp_10d",
    ]
    _LEAGUE_XWOBA = 0.315
    _away_hand = str(away_prof.get("p_throws", "R")).upper()[:1]
    _home_hand = str(home_prof.get("p_throws", "R")).upper()[:1]

    # Home batting scale: prefer platoon xwOBA vs opposing starter's hand
    if home_lineup_platoon is not None:
        _xw_key = f"xwoba_vs_{'rhp' if _away_hand == 'R' else 'lhp'}"
        _xw = home_lineup_platoon.get(_xw_key)
        if _xw and not np.isnan(float(_xw)):
            scale_h = float(_xw) / _LEAGUE_XWOBA
            for feat in _xwoba_bat_feats_home:
                if feat in row and pd.notna(row.get(feat)):
                    row[feat] = float(row[feat]) * scale_h
    elif home_lineup_wrc is not None and not np.isnan(float(home_lineup_wrc)):
        scale_h = float(home_lineup_wrc) / 100.0
        for feat in _xwoba_bat_feats_home:
            if feat in row and pd.notna(row.get(feat)):
                row[feat] = float(row[feat]) * scale_h

    # Away batting scale: prefer platoon xwOBA vs opposing starter's hand
    if away_lineup_platoon is not None:
        _xw_key = f"xwoba_vs_{'rhp' if _home_hand == 'R' else 'lhp'}"
        _xw = away_lineup_platoon.get(_xw_key)
        if _xw and not np.isnan(float(_xw)):
            scale_a = float(_xw) / _LEAGUE_XWOBA
            for feat in _xwoba_bat_feats_away:
                if feat in row and pd.notna(row.get(feat)):
                    row[feat] = float(row[feat]) * scale_a
    elif away_lineup_wrc is not None and not np.isnan(float(away_lineup_wrc)):
        scale_a = float(away_lineup_wrc) / 100.0
        for feat in _xwoba_bat_feats_away:
            if feat in row and pd.notna(row.get(feat)):
                row[feat] = float(row[feat]) * scale_a
    # Recompute derived matchup edges from scaled values
    if pd.notna(row.get("home_bat_vs_away_sp")) and pd.notna(row.get("away_bat_vs_home_sp")):
        row["batting_matchup_edge"] = row["home_bat_vs_away_sp"] - row["away_bat_vs_home_sp"]
    if pd.notna(row.get("home_bat_vs_away_sp_10d")) and pd.notna(row.get("away_bat_vs_home_sp_10d")):
        row["batting_matchup_edge_10d"] = row["home_bat_vs_away_sp_10d"] - row["away_bat_vs_home_sp_10d"]

    # Weather
    row["temp_f"]    = temp_f
    row["wind_mph"]  = wind_mph
    row["humidity"]  = 50.0  # default

    # Park factor (static 3-yr average, same dict used in build_feature_matrix.py)
    _PARK_FACTORS = {
        "COL": 1.281, "BOS": 1.078, "CIN": 1.062, "TEX": 1.050,
        "PHI": 1.040, "BAL": 1.038, "ATL": 1.034, "MIL": 1.029,
        "DET": 1.024, "CHC": 1.023, "HOU": 1.018, "STL": 1.014,
        "PIT": 1.012, "TB":  1.010, "KC":  1.005, "TOR": 1.000,
        "NYY": 0.998, "LAA": 0.996, "MIN": 0.995, "AZ":  0.993,
        "ARI": 0.993, "WSH": 0.990, "WAS": 0.990, "CLE": 0.988,
        "NYM": 0.984, "MIA": 0.978, "CWS": 0.975, "SF":  0.972,
        "LAD": 0.970, "SEA": 0.968, "OAK": 0.962, "ATH": 0.962,
        "SD":  0.960,
    }
    _league_avg_park = sum(_PARK_FACTORS.values()) / len(_PARK_FACTORS)
    elev = STADIUM_ELEVATION.get(home_team, 100)
    row["park_factor"]      = 1.0 + (elev / 5200 * 0.10)   # elevation proxy (legacy col)
    row["home_park_factor"] = _PARK_FACTORS.get(home_team, _league_avg_park)

    # Circadian / game-time features (same team timezone dict as build_feature_matrix.py)
    _TEAM_ET_OFFSET = {
        "NYY": 0, "NYM": 0, "BOS": 0, "TOR": 0, "DET": 0,
        "CLE": 0, "BAL": 0, "PHI": 0, "PIT": 0, "WSH": 0, "WAS": 0,
        "MIA": 0, "ATL": 0, "TB":  0, "CIN": 0,
        "CHC": -1, "CWS": -1, "MIL": -1, "MIN": -1,
        "STL": -1, "KC":  -1, "HOU": -1, "TEX": -1,
        "COL": -2, "AZ": -2, "ARI": -2,
        "LAD": -3, "LAA": -3, "SEA": -3, "SF": -3,
        "SD":  -3, "ATH": -3, "OAK": -3,
    }
    _gh = float(game_hour_et) if game_hour_et is not None else 19.0  # default: 7pm ET (neutral)
    _away_offset = _TEAM_ET_OFFSET.get(away_team, 0)
    _home_offset = _TEAM_ET_OFFSET.get(home_team, 0)
    row["game_hour_et"]          = _gh
    row["is_day_game"]           = float(_gh < 17.0)
    row["away_game_local_hour"]  = _gh + _away_offset
    row["home_game_local_hour"]  = _gh + _home_offset
    row["circadian_edge"]        = row["home_game_local_hour"] - row["away_game_local_hour"]

    # Market odds — top-2 features by XGBoost gain; must be populated at inference.
    # true_home_prob / true_away_prob = de-vigged win probability from sharp market.
    # Prefer Pinnacle (P_true_home from odds_current_pull) over retail de-vig.
    if market_odds:
        thp = market_odds.get("true_home_prob")
        tap = market_odds.get("true_away_prob")
        if thp is not None and not np.isnan(float(thp)):
            row["true_home_prob"] = float(thp)
        if tap is not None and not np.isnan(float(tap)):
            row["true_away_prob"] = float(tap)

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
    wind_mph: float = 8.0,
    month: int | None = None,
    verbose: bool = True,
    home_team_stats: pd.Series | None = None,   # row from team_stats_2026
    away_team_stats: pd.Series | None = None,
    home_lineup_wrc: float | None = None,   # today's lineup wRC+ (100 = league avg)
    away_lineup_wrc: float | None = None,
    home_lineup_platoon: dict | None = None,  # {xwoba_vs_rhp, xwoba_vs_lhp} for home batters
    away_lineup_platoon: dict | None = None,  # same for away batters
    pitcher_10d: dict | None = None,   # {pitcher_name_upper: {k_pct_10d, xwoba_10d}}
    posted_total: float | None = None, # Vegas O/U line — used to compute mc_over_prob
    market_odds: dict | None = None,   # {true_home_prob, true_away_prob} for XGBoost
    game_hour_et: float | None = None, # scheduled start hour in ET (e.g. 13.75 = 1:45 PM)
    ump_k_above_avg: float = 0.0,      # HP ump trailing K% vs league avg (from ump_features)
    home_pp_k_line: float | None = None,  # PrizePicks standard K line for home SP
    away_pp_k_line: float | None = None,  # PrizePicks standard K line for away SP
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

    # Team quality inputs for simulation
    home_rs_pg = float(home_team_stats["team_rs_per_game"]) \
        if home_team_stats is not None and pd.notna(home_team_stats.get("team_rs_per_game")) \
        else None
    away_rs_pg = float(away_team_stats["team_rs_per_game"]) \
        if away_team_stats is not None and pd.notna(away_team_stats.get("team_rs_per_game")) \
        else None

    # Lineup quality → implied RS/G.
    # Priority: platoon xwOBA (vs specific starter handedness) > generic wRC+ > team avg
    _LEAGUE_XWOBA = 0.315

    away_hand = str(away_prof.get("p_throws", "R")).upper()[:1]  # handedness of away SP
    home_hand = str(home_prof.get("p_throws", "R")).upper()[:1]  # handedness of home SP

    # Home batters face away SP → use home_lineup_platoon vs away_hand
    if home_lineup_platoon is not None:
        xw_key = f"xwoba_vs_{'rhp' if away_hand == 'R' else 'lhp'}"
        xw = home_lineup_platoon.get(xw_key)
        if xw and not pd.isna(xw):
            home_rs_pg = (float(xw) / _LEAGUE_XWOBA) * LEAGUE_RS_PER_GAME
    elif home_lineup_wrc is not None and not pd.isna(float(home_lineup_wrc)):
        home_rs_pg = float(home_lineup_wrc) / 100.0 * LEAGUE_RS_PER_GAME

    # Away batters face home SP → use away_lineup_platoon vs home_hand
    if away_lineup_platoon is not None:
        xw_key = f"xwoba_vs_{'rhp' if home_hand == 'R' else 'lhp'}"
        xw = away_lineup_platoon.get(xw_key)
        if xw and not pd.isna(xw):
            away_rs_pg = (float(xw) / _LEAGUE_XWOBA) * LEAGUE_RS_PER_GAME
    elif away_lineup_wrc is not None and not pd.isna(float(away_lineup_wrc)):
        away_rs_pg = float(away_lineup_wrc) / 100.0 * LEAGUE_RS_PER_GAME

    home_bp_xwoba = float(home_team_stats["bullpen_xwoba"]) \
        if home_team_stats is not None and pd.notna(home_team_stats.get("bullpen_xwoba")) \
        else DEFAULT_BULLPEN_XWOBA
    away_bp_xwoba = float(away_team_stats["bullpen_xwoba"]) \
        if away_team_stats is not None and pd.notna(away_team_stats.get("bullpen_xwoba")) \
        else DEFAULT_BULLPEN_XWOBA

    # Per-pitcher expected innings (from 2026 actual IP/start data)
    home_ip = float(home_prof.get("expected_ip") or INNINGS_SP)
    away_ip = float(away_prof.get("expected_ip") or INNINGS_SP)
    if pd.isna(home_ip): home_ip = INNINGS_SP
    if pd.isna(away_ip): away_ip = INNINGS_SP

    # Monte Carlo simulation
    rng = np.random.default_rng(seed=42)
    home_runs, away_runs = simulate_game(
        home_mu=home_mu, home_sigma=home_sigma,
        away_mu=away_mu, away_sigma=away_sigma,
        park_elevation_ft=elev,
        temp_f=temp_f,
        home_bullpen_xwoba=home_bp_xwoba,
        away_bullpen_xwoba=away_bp_xwoba,
        home_team_rs_per_game=home_rs_pg,
        away_team_rs_per_game=away_rs_pg,
        home_expected_ip=home_ip,
        away_expected_ip=away_ip,
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

    # -----------------------------------------------------------------------
    # SHARED HELPERS for F5 / F1 calculations
    # -----------------------------------------------------------------------
    env_factor = air_density_ratio(elev) * (1 + TEMP_RUN_FACTOR * (temp_f - TEMP_BASELINE_F))

    def _off_factor(rs_pg):
        if rs_pg is None:
            return 1.0
        return 0.50 * (float(rs_pg) / LEAGUE_RS_PER_GAME) + 0.50

    # SP runs/game allowed (used by both F5 and F1)
    away_sp_rpg = float(np.clip(XWOBA_INTERCEPT + XWOBA_SLOPE * away_mu, 0, 20))
    home_sp_rpg = float(np.clip(XWOBA_INTERCEPT + XWOBA_SLOPE * home_mu, 0, 20))
    bp_rpg_home = float(np.clip(XWOBA_INTERCEPT + XWOBA_SLOPE * home_bp_xwoba, 0, 20))
    bp_rpg_away = float(np.clip(XWOBA_INTERCEPT + XWOBA_SLOPE * away_bp_xwoba, 0, 20))

    # -----------------------------------------------------------------------
    # F5 PREDICTIONS (First 5 Innings — analytical Poisson)
    # -----------------------------------------------------------------------
    home_f5_ip    = min(home_ip, 5.0)
    away_f5_ip    = min(away_ip, 5.0)
    home_f5_bp_ip = max(0.0, 5.0 - home_f5_ip)
    away_f5_bp_ip = max(0.0, 5.0 - away_f5_ip)

    # Home team's F5 expected runs = what away pitching gives up over 5 innings
    home_f5_lambda = (
        away_sp_rpg * away_f5_ip / 9.0
        + bp_rpg_away * away_f5_bp_ip / 9.0
    ) * env_factor * _off_factor(home_rs_pg)
    # Away team's F5 expected runs = what home pitching gives up over 5 innings
    away_f5_lambda = (
        home_sp_rpg * home_f5_ip / 9.0
        + bp_rpg_home * home_f5_bp_ip / 9.0
    ) * env_factor * _off_factor(away_rs_pg)

    f5_home_runs = rng.poisson(max(home_f5_lambda, 0.05), size=N_SIMS).astype(float)
    f5_away_runs = rng.poisson(max(away_f5_lambda, 0.05), size=N_SIMS).astype(float)
    f5_margin    = f5_home_runs - f5_away_runs

    # -----------------------------------------------------------------------
    # F1 / NRFI PREDICTIONS (First Inning — analytical Poisson)
    # -----------------------------------------------------------------------
    import math as _math

    home_f1_lambda = (away_sp_rpg / 9.0) * env_factor * _off_factor(home_rs_pg)
    away_f1_lambda = (home_sp_rpg / 9.0) * env_factor * _off_factor(away_rs_pg)

    p_home_scoreless_f1 = _math.exp(-home_f1_lambda)
    p_away_scoreless_f1 = _math.exp(-away_f1_lambda)
    p_nrfi = p_home_scoreless_f1 * p_away_scoreless_f1

    f1_home_runs = rng.poisson(max(home_f1_lambda, 0.01), size=N_SIMS).astype(float)
    f1_away_runs = rng.poisson(max(away_f1_lambda, 0.01), size=N_SIMS).astype(float)

    # -----------------------------------------------------------------------
    # PITCHER K PREDICTIONS (Strikeout props — Negative Binomial model)
    # NB(r=20) Gamma-Poisson mixture; mu scaled by KPROP_MEAN_CALIB=0.899
    # to correct +0.54K mean bias (IP overestimate + k_pct overestimate).
    # Umpire adjustment: ump_k_above_avg (K% vs league avg) × expected BF.
    # 10th–90th percentile ump range = ±1.2 Ks per SP (2025 data).
    # -----------------------------------------------------------------------
    home_k_rate = float(home_prof.get("blended_k_pct") or 0.225)
    away_k_rate = float(away_prof.get("blended_k_pct") or 0.225)
    if pd.isna(home_k_rate): home_k_rate = 0.225
    if pd.isna(away_k_rate): away_k_rate = 0.225

    home_expected_bf = home_ip * BF_PER_INNING
    away_expected_bf = away_ip * BF_PER_INNING

    ump_adj = float(ump_k_above_avg) if not pd.isna(ump_k_above_avg) else 0.0
    mc_home_k_mean = home_k_rate * home_expected_bf * KPROP_MEAN_CALIB + ump_adj * home_expected_bf
    mc_away_k_mean = away_k_rate * away_expected_bf * KPROP_MEAN_CALIB + ump_adj * away_expected_bf

    # Blend PrizePicks standard line as a market prior (45% weight).
    # PP line represents sharp market consensus; pulling MC toward it improves calibration
    # when our K rate estimate diverges significantly from the market.
    # Weight bumped 35%→45% after 2026 early-season audit: MC K correlation ≈0.02,
    # suggesting the physics model adds noise; revisit at n≥200 starts.
    _PP_K_WEIGHT = 0.45
    if home_pp_k_line is not None and not pd.isna(home_pp_k_line):
        mc_home_k_mean = (1 - _PP_K_WEIGHT) * mc_home_k_mean + _PP_K_WEIGHT * float(home_pp_k_line)
    if away_pp_k_line is not None and not pd.isna(away_pp_k_line):
        mc_away_k_mean = (1 - _PP_K_WEIGHT) * mc_away_k_mean + _PP_K_WEIGHT * float(away_pp_k_line)

    home_k_sims = simulate_k_prop_nb(mc_home_k_mean).astype(float)
    away_k_sims = simulate_k_prop_nb(mc_away_k_mean).astype(float)

    def k_over_prob(k_sims, line):
        return float((k_sims > line).mean())

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
        "home_sp_expected_ip": round(home_ip, 1),
        "away_sp_expected_ip": round(away_ip, 1),
        "park_elevation_ft":   elev,
        "temp_f":              temp_f,
        "home_lineup_wrc":     home_lineup_wrc,
        "away_lineup_wrc":     away_lineup_wrc,
        # Moneyline probabilities
        "mc_home_win_prob":    round(float(mc_home_win), 4),
        "mc_away_win_prob":    round(float(1 - mc_home_win), 4),
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
        "mc_total_lo":         int(np.percentile(home_runs + away_runs, 25)),
        "mc_total_hi":         int(np.percentile(home_runs + away_runs, 75)),
        "mc_total_std":        round(float((home_runs + away_runs).std()), 2),
        "n_sims":              N_SIMS,
        # F5 predictions
        "mc_f5_home_runs":      round(float(f5_home_runs.mean()), 2),
        "mc_f5_away_runs":      round(float(f5_away_runs.mean()), 2),
        "mc_f5_total":          round(float((f5_home_runs + f5_away_runs).mean()), 2),
        "mc_f5_home_runs_lo":   int(np.percentile(f5_home_runs, 25)),
        "mc_f5_home_runs_hi":   int(np.percentile(f5_home_runs, 75)),
        "mc_f5_away_runs_lo":   int(np.percentile(f5_away_runs, 25)),
        "mc_f5_away_runs_hi":   int(np.percentile(f5_away_runs, 75)),
        "mc_f5_total_lo":       int(np.percentile(f5_home_runs + f5_away_runs, 25)),
        "mc_f5_total_hi":       int(np.percentile(f5_home_runs + f5_away_runs, 75)),
        "mc_f5_home_win_prob":  round(float((f5_margin > 0).mean()), 4),
        "mc_f5_away_win_prob":  round(float((f5_margin < 0).mean()), 4),
        "mc_f5_home_covers_rl": round(float((f5_margin >= 2).mean()), 4),
        # F1 / NRFI predictions
        "mc_nrfi_prob":         round(p_nrfi, 4),
        "mc_p_home_scores_f1":  round(1 - p_home_scoreless_f1, 4),
        "mc_p_away_scores_f1":  round(1 - p_away_scoreless_f1, 4),
        "mc_f1_home_runs":      round(float(f1_home_runs.mean()), 3),
        "mc_f1_away_runs":      round(float(f1_away_runs.mean()), 3),
        # K prop predictions
        "mc_home_sp_k_mean":    round(mc_home_k_mean, 2),
        "mc_away_sp_k_mean":    round(mc_away_k_mean, 2),
        "mc_home_sp_k_std":     round(float(home_k_sims.std()), 2),
        "mc_away_sp_k_std":     round(float(away_k_sims.std()), 2),
        "mc_home_sp_k_over_35": round(k_over_prob(home_k_sims, 3.5), 4),
        "mc_home_sp_k_over_45": round(k_over_prob(home_k_sims, 4.5), 4),
        "mc_home_sp_k_over_55": round(k_over_prob(home_k_sims, 5.5), 4),
        "mc_home_sp_k_over_65": round(k_over_prob(home_k_sims, 6.5), 4),
        "mc_away_sp_k_over_35": round(k_over_prob(away_k_sims, 3.5), 4),
        "mc_away_sp_k_over_45": round(k_over_prob(away_k_sims, 4.5), 4),
        "mc_away_sp_k_over_55": round(k_over_prob(away_k_sims, 5.5), 4),
        "mc_away_sp_k_over_65": round(k_over_prob(away_k_sims, 6.5), 4),
        "ump_k_above_avg":      round(ump_adj, 4),
    }

    # O/U probability vs posted line
    if posted_total is not None:
        total_sims = home_runs + away_runs
        result["mc_over_prob"]  = round(float((total_sims > float(posted_total)).mean()), 4)
        result["mc_under_prob"] = round(float((total_sims < float(posted_total)).mean()), 4)

    # Level-1 XGBoost + Level-2 stacking LR (official signal)
    # + Shadow inference for ensemble_min / ensemble_max / model_spread
    rl_model, tot_model, ml_model, feat_cols, lgbm_shadow, cat_shadow, stacking, team_rl_model, team_feat_cols = load_xgb_models()
    if rl_model is not None:
        xgb_row = build_xgb_row(
            home_prof, away_prof, home_team, away_team, temp_f, feat_cols,
            wind_mph=wind_mph,
            home_lineup_platoon=home_lineup_platoon,
            away_lineup_platoon=away_lineup_platoon,
            home_team_stats=home_team_stats,
            away_team_stats=away_team_stats,
            pitcher_10d=pitcher_10d,
            market_odds=market_odds,
            home_lineup_wrc=home_lineup_wrc,
            away_lineup_wrc=away_lineup_wrc,
            game_hour_et=game_hour_et)
        if xgb_row is not None:
            xgb_rl_prob_raw = float(rl_model.predict_proba(xgb_row)[0, 1])

            # ── Quantile total regression (v5.1) ──────────────────────────
            # tot_model outputs shape (1, 3): [Q10_floor, Q50_median, Q90_ceiling].
            # Older single-output models return shape (1,) — handled via fallback.
            xgb_tot_floor   = None
            xgb_tot_ceiling = None
            if tot_model:
                _tot_raw = tot_model.predict(xgb_row)
                if _tot_raw.ndim == 2 and _tot_raw.shape[1] == 3:
                    xgb_tot_floor   = float(np.clip(_tot_raw[0, 0], 0, 30))
                    xgb_tot         = float(np.clip(_tot_raw[0, 1], 0, 30))
                    xgb_tot_ceiling = float(np.clip(_tot_raw[0, 2], 0, 30))
                else:
                    # Legacy single-output fallback (reg:squarederror model)
                    xgb_tot = float(_tot_raw.flat[0])
            else:
                xgb_tot = mc_total

            # ── Shadow inference (variance bounds only — NOT fed to stacker) ──
            lgbm_rl_prob_raw = None
            if lgbm_shadow is not None:
                try:
                    lgbm_rl_prob_raw = _shadow_predict_lgbm(lgbm_shadow, xgb_row)
                except Exception as e:
                    pass  # shadow miss is non-fatal

            cat_rl_prob_raw = None
            if cat_shadow is not None:
                try:
                    _xgb_row_cat = xgb_row.astype("float64").values  # pd.NA → np.nan
                    cat_rl_prob_raw = _shadow_predict_catboost(cat_shadow, _xgb_row_cat)
                except Exception as e:
                    pass  # shadow miss is non-fatal

            result["lgbm_rl_prob_raw"] = round(lgbm_rl_prob_raw, 4) if lgbm_rl_prob_raw is not None else None
            result["cat_rl_prob_raw"]  = round(cat_rl_prob_raw,  4) if cat_rl_prob_raw  is not None else None
            result["_stacking_model"]  = stacking   # pass through for run_today.py

            # ── Team-perspective model: Step 1 (raw) + Step 2 (L1 normalize) ────────
            xgb_team_rl_home_norm = None
            xgb_team_rl_away_norm = None
            if team_rl_model is not None and team_feat_cols is not None:
                try:
                    # Build home and away feature rows aligned to team model cols
                    X_home_team = pd.DataFrame(index=[0])
                    for c in team_feat_cols:
                        X_home_team[c] = xgb_row[c].values[0] if c in xgb_row.columns else np.nan
                    X_home_team["is_home"] = 1.0

                    X_away_team = _build_away_xgb_row(X_home_team)
                    X_away_team["is_home"] = 0.0

                    raw_home = float(team_rl_model.predict_proba(X_home_team)[0, 1])
                    raw_away = float(team_rl_model.predict_proba(X_away_team)[0, 1])
                    p_sum = max(raw_home + raw_away, 1e-9)
                    xgb_team_rl_home_norm = raw_home / p_sum
                    xgb_team_rl_away_norm = raw_away / p_sum
                except Exception as _te:
                    print(f"  [WARN] Team RL model inference failed: {_te}")
            result["xgb_team_rl_home_norm"] = round(xgb_team_rl_home_norm, 4) if xgb_team_rl_home_norm is not None else None
            result["xgb_team_rl_away_norm"] = round(xgb_team_rl_away_norm, 4) if xgb_team_rl_away_norm is not None else None

            # ── ML model raw prob (for stacking gap feature) ─────────────────
            # ml_model_vs_vegas_gap = XGB ML raw prob − Pinnacle closing implied.
            # Trained signal: gap > 10% → 65.1% win rate vs 57.2% when aligned.
            xgb_ml_raw_prob = None
            if ml_model is not None:
                try:
                    xgb_ml_raw_prob = float(ml_model.predict_proba(xgb_row)[0, 1])
                except Exception:
                    pass  # ML model miss is non-fatal

            # Export stacking features for run_today.py Three-Part Lock.
            # All 11 domain features + ml_model_vs_vegas_gap are computable at inference:
            #   sp_*/batting_matchup_edge  → pitcher profiles + team batting splits
            #   bp_*_diff                  → team_stats bullpen (era / whip / k9)
            #   *_10d                      → trailing-10d batting + pitcher_10d file
            # il_return flags remain at fill_value=0 (absent from profiles)
            _stk_feat_cols = [
                "sp_k_pct_diff", "sp_xwoba_diff", "sp_kminusbb_diff",
                "batting_matchup_edge",
                "bp_era_diff", "bp_whip_diff", "bp_k9_diff",
                "batting_matchup_edge_10d",
                "sp_k_pct_10d_diff", "sp_xwoba_10d_diff",
            ]
            _stk_feats = {
                col: float(xgb_row[col].iloc[0])
                for col in _stk_feat_cols
                if col in xgb_row.columns and pd.notna(xgb_row[col].iloc[0])
            }
            # SP handedness columns — needed by BayesianStacker to derive segment_id.
            # 1 = RHP, 0 = LHP.  Default RvR (both 1) when columns are absent.
            for _sp_col in ("home_sp_p_throws_R", "away_sp_p_throws_R"):
                if _sp_col in xgb_row.columns and pd.notna(xgb_row[_sp_col].iloc[0]):
                    _stk_feats[_sp_col] = float(xgb_row[_sp_col].iloc[0])
                else:
                    _stk_feats[_sp_col] = 1.0   # default RHP
            # Compute ml_model_vs_vegas_gap: ML model raw prob − Pinnacle implied
            if xgb_ml_raw_prob is not None and market_odds is not None:
                _true_hp = market_odds.get("true_home_prob")
                if _true_hp is not None:
                    try:
                        _stk_feats["ml_model_vs_vegas_gap"] = float(xgb_ml_raw_prob) - float(_true_hp)
                    except (TypeError, ValueError):
                        pass  # leave gap absent → stacker fill_value
            result["stacking_feats"] = _stk_feats

            # Apply calibration — Platt (LogisticRegression) uses predict_proba;
            # legacy isotonic regression uses predict.  Handle both interfaces.
            cal_rl = load_calibrator("rl")
            result["xgb_home_covers_rl_raw"] = round(xgb_rl_prob_raw, 4)
            if cal_rl is not None:
                try:
                    # Platt scaling (LogisticRegression) — explicit (1,1) array
                    _cal_input = np.array([[float(xgb_rl_prob_raw)]])
                    xgb_rl_prob = float(cal_rl.predict_proba(_cal_input)[0, 1])
                except (AttributeError, ValueError):
                    try:
                        # Isotonic regression or reshape fallback
                        _cal_input = np.array([[float(xgb_rl_prob_raw)]])
                        xgb_rl_prob = float(cal_rl.predict(_cal_input)[0])
                    except Exception:
                        xgb_rl_prob = xgb_rl_prob_raw
            else:
                xgb_rl_prob = xgb_rl_prob_raw

            result["xgb_home_covers_rl"] = round(xgb_rl_prob, 4)
            result["xgb_expected_total"] = round(xgb_tot, 2)

            # Quantile bounds (None for legacy single-output model)
            result["total_floor_10th"]    = round(xgb_tot_floor,   2) if xgb_tot_floor   is not None else None
            result["total_ceiling_90th"]  = round(xgb_tot_ceiling, 2) if xgb_tot_ceiling is not None else None
            result["total_variance_spread"] = (
                round(xgb_tot_ceiling - xgb_tot_floor, 2)
                if xgb_tot_floor is not None and xgb_tot_ceiling is not None
                else None
            )

            # Blend MC + (calibrated) XGBoost — use Q50 median for total blend
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
