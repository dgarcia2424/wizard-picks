"""
run_today.py
============
Fully automated daily prediction card.

Pulls today's starters, Vegas lines, and weather from files already
on disk (refreshed by the data pipeline), runs Monte Carlo + XGBoost
for every game, and prints a ranked bet card.

Usage:
  python run_today.py               # today's card
  python run_today.py --date 2026-04-12
  python run_today.py --min-edge 0.56   # only print games above threshold
  python run_today.py --csv             # also write daily_card.csv
"""

import argparse
import datetime
import os
import traceback as _traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Load .env file so GMAIL_APP_PASSWORD etc. are available
def _load_dotenv():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

_load_dotenv()

DATA_DIR   = Path("./data/statcast")
MODELS_DIR = Path("./models")

# Bet signal thresholds (legacy — kept for backtest compatibility)
BET_THRESHOLD_HIGH      = 0.58
BET_THRESHOLD_HIGH_LEAN = 0.54
BET_THRESHOLD_LOW       = 0.34
BET_THRESHOLD_LOW_LEAN  = 0.40

# ---------------------------------------------------------------------------
# Three-Part Lock execution gate constants
# ---------------------------------------------------------------------------
SYNTHETIC_BANKROLL       = 2000   # dollar bankroll used for Kelly sizing
MAX_BET_DOLLARS          = 50     # hard cap per bet
ODDS_FLOOR               = -225   # reject any line worse than -225
EDGE_TIER1               = 0.030  # >= 3.0% CLV edge   → Tier 1 (quarter-Kelly)
EDGE_TIER2               = 0.010  # >= 1.0% CLV edge   → Tier 2 (eighth-Kelly)
SANITY_THRESHOLD         = 0.04   # |P_model - P_true| must be <= 4%  (Pinnacle)
SANITY_THRESHOLD_RETAIL  = 0.08   # |P_model - P_retail| must be <= 8% (fallback)

# ---------------------------------------------------------------------------
# Correlated Parlay constants
# ---------------------------------------------------------------------------
_PARLAY_TARGET_MIN   = 280   # American odds lower bound (inclusive)
_PARLAY_TARGET_MAX   = 450   # American odds upper bound (inclusive)
_PARLAY_MIN_LEG_PROB = 0.35  # drop any leg with model prob below this (or above 1-this)
_PARLAY_MIN_EDGE     = -0.02 # drop legs with edge below this (more than 2% under water)
_PARLAY_MIN_COMBO_EDGE = 0.0 # require positive combined model edge
_PARLAY_MIN_BEST_LEG_EDGE = 0.02  # at least one leg must have ≥ 2% edge
_PARLAY_TOP_N        = 5     # max combos to surface

# Pearson ρ between same-game leg types.
# "subset" means the smaller-prob event ⊆ the larger (joint = min of the two).
_LEG_CORR: dict[tuple, object] = {
    ("RL_HOME", "ML_HOME"): "subset",   # RL_HOME ⊂ ML_HOME (win by 2 ⊂ win)
    ("RL_AWAY", "ML_AWAY"): "subset",   # RL_AWAY ⊂ ML_AWAY
    ("ML_HOME", "OVER"):    0.10,
    ("ML_HOME", "UNDER"):  -0.10,
    ("ML_AWAY", "OVER"):    0.06,
    ("ML_AWAY", "UNDER"):  -0.06,
    ("RL_HOME", "UNDER"):   0.08,   # home dominates → low-run game possible
    ("RL_HOME", "OVER"):    0.04,
    ("RL_AWAY", "OVER"):    0.06,
    ("RL_AWAY", "UNDER"):   0.04,
    ("K_HOME",  "ML_HOME"): 0.15,   # SP dominates → team wins
    ("K_AWAY",  "ML_AWAY"): 0.15,
    ("K_HOME",  "UNDER"):   0.18,   # SP dominates → low-scoring game
    ("K_AWAY",  "UNDER"):   0.18,
    ("K_HOME",  "OVER"):   -0.05,
    ("K_AWAY",  "OVER"):   -0.05,
    ("K_HOME",  "ML_AWAY"):-0.08,   # home SP dominates ↔ home wins
    ("K_AWAY",  "ML_HOME"):-0.08,
}


# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------

def load_lineups(date_str: str) -> pd.DataFrame:
    """Load starters — prefers date-stamped file, falls back to lineups_today.parquet."""
    dated = DATA_DIR / f"lineups_{date_str}.parquet"
    path  = dated if dated.exists() else DATA_DIR / "lineups_today.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No lineup file found for {date_str}")
    df = pd.read_parquet(path, engine="pyarrow")
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)

    today = df[df["game_date"] == date_str]
    if len(today) == 0:
        # Fall back to most recent date in file
        latest = df["game_date"].max()
        print(f"  [WARN] No lineups for {date_str}, using most recent: {latest}")
        today = df[df["game_date"] == latest]

    cols = ["game_pk", "game_date", "home_team", "away_team",
            "home_starter_name", "away_starter_name",
            "home_lineup_confirmed", "away_lineup_confirmed"]
    if "game_time_et" in today.columns:
        cols.append("game_time_et")
    return today[cols]


def load_odds(date_str: str) -> pd.DataFrame:
    """Load today's Vegas odds — tries dated file first, then most recent."""
    # Try exact date file
    dated = DATA_DIR / f"odds_current_{date_str}.parquet"
    if dated.exists():
        df = pd.read_parquet(dated, engine="pyarrow")
    else:
        # Fall back to any odds_current file, most recent
        candidates = sorted(DATA_DIR.glob("odds_current_*.parquet"), reverse=True)
        if not candidates:
            print("  [WARN] No odds_current file found — running without Vegas lines")
            return pd.DataFrame()
        df = pd.read_parquet(candidates[0], engine="pyarrow")
        print(f"  [WARN] Using odds file: {candidates[0].name}")

    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
    today = df[df["game_date"] == date_str]
    if len(today) == 0:
        today = df   # use all rows if date doesn't match

    numeric_cols = [
        "close_ml_home", "close_ml_away", "close_total", "runline_home_odds",
        # Pinnacle sharp benchmark (populated by odds_current_pull.py)
        "P_true_home", "P_true_away", "P_true_rl_home", "P_true_rl_away",
        "P_true_over", "P_true_under",
        # US retail implied probabilities (vig-adjusted)
        "retail_implied_home", "retail_implied_away",
        "retail_implied_rl_home", "retail_implied_rl_away",
        "retail_implied_over", "retail_implied_under",
    ]
    for col in numeric_cols:
        if col in today.columns:
            today[col] = pd.to_numeric(today[col], errors="coerce")
    return today


def load_weather(date_str: str) -> pd.DataFrame:
    """Load weather for today's games."""
    path = DATA_DIR / "weather_2026.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path, engine="pyarrow")
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
    today = df[df["game_date"] == date_str]
    if len(today) == 0:
        return pd.DataFrame()
    return today[["home_team", "temp_f", "wind_mph", "humidity"]]


def load_kprops(date_str: str) -> dict:
    """
    Load K prop lines for today's starters.
    Returns dict: PITCHER_NAME_UPPER -> {line, implied_over, implied_under, over_odds, under_odds}
    De-vigs the market and averages across books for each (pitcher, line) pair.
    """
    path = DATA_DIR / f"k_props_{date_str}.parquet"
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path, engine="pyarrow")
        if df.empty:
            return {}

        def _norm(name):
            name = str(name).strip()
            if "," in name:
                parts = [p.strip() for p in name.split(",", 1)]
                return f"{parts[1]} {parts[0]}".upper()
            return name.upper()

        def _implied(odds):
            if pd.isna(odds):
                return np.nan
            o = float(odds)
            return 100 / (o + 100) if o > 0 else abs(o) / (abs(o) + 100)

        df["name_key"]  = df["pitcher_name"].apply(_norm)
        df["raw_over"]  = df["over_odds"].apply(_implied)
        df["raw_under"] = df["under_odds"].apply(_implied)
        df["total_impl"] = df["raw_over"] + df["raw_under"]
        # De-vig
        df = df[df["total_impl"] > 0].copy()
        df["imp_over"]  = df["raw_over"]  / df["total_impl"]
        df["imp_under"] = df["raw_under"] / df["total_impl"]

        # Average implied probabilities across books per (pitcher, line)
        grp = (df.groupby(["name_key", "line"])
                 .agg(imp_over=("imp_over", "mean"),
                      imp_under=("imp_under", "mean"),
                      over_odds=("over_odds", "first"),
                      under_odds=("under_odds", "first"))
                 .reset_index())

        # For each pitcher keep the single most-common line
        # (choose line closest to 50/50 implied — least juice)
        result = {}
        for name_key, sub in grp.groupby("name_key"):
            best = sub.loc[(sub["imp_over"] - 0.5).abs().idxmin()]
            result[str(name_key)] = {
                "line":          float(best["line"]),
                "implied_over":  round(float(best["imp_over"]),  4),
                "implied_under": round(float(best["imp_under"]), 4),
                "over_odds":     (int(best["over_odds"])
                                  if not pd.isna(best["over_odds"]) else None),
                "under_odds":    (int(best["under_odds"])
                                  if not pd.isna(best["under_odds"]) else None),
            }
        return result
    except Exception as e:
        print(f"  [WARN] Could not load k_props: {e}")
        return {}


def ml_to_prob(ml):
    if pd.isna(ml) or ml is None:
        return np.nan
    ml = float(ml)
    return 100 / (ml + 100) if ml > 0 else abs(ml) / (abs(ml) + 100)


def _edge_vs_line(model_prob: float, market_odds: float) -> float:
    """Edge = model probability minus implied probability from market odds."""
    implied = ml_to_prob(market_odds)
    if np.isnan(implied):
        return np.nan
    return model_prob - implied


def _is_missing(v) -> bool:
    """True if v is None, NaN, pandas NA, or any missing sentinel."""
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Stacking feature helpers for away-perspective normalization
# ---------------------------------------------------------------------------
# Stacking feature columns that get negated for the away perspective.
_STK_DIFF_COLS_AWAY = [
    "sp_k_pct_diff", "sp_xwoba_diff", "sp_kminusbb_diff",
    "bp_era_diff", "bp_whip_diff", "batting_matchup_edge",
    "sp_k_pct_10d_diff", "sp_xwoba_10d_diff", "batting_matchup_edge_10d",
    "ml_model_vs_vegas_gap",
]

def _flip_stacking_feats_dict(sf: dict) -> dict:
    """Return away-perspective copy of stacking features dict."""
    flipped = dict(sf)
    for col in _STK_DIFF_COLS_AWAY:
        if col in flipped:
            flipped[col] = -flipped[col]
    h_il = sf.get("home_sp_il_return_flag", 0.0)
    a_il = sf.get("away_sp_il_return_flag", 0.0)
    flipped["home_sp_il_return_flag"] = a_il
    flipped["away_sp_il_return_flag"] = h_il
    h_throws = sf.get("home_sp_p_throws_R", 1.0)
    a_throws = sf.get("away_sp_p_throws_R", 1.0)
    flipped["home_sp_p_throws_R"] = a_throws
    flipped["away_sp_p_throws_R"] = h_throws
    return flipped


# ---------------------------------------------------------------------------
# Posterior trace cache  (BayesianStacker p_std — Variance Gate)
# ---------------------------------------------------------------------------
_POSTERIOR_TRACE: dict | None = None   # loaded once per process


def _load_posterior_trace() -> dict | None:
    """
    Load NUTS posterior samples from stacking_lr_rl.npz (cached per-process).
    Returns dict {param: np.ndarray} or None if file not found.
    """
    global _POSTERIOR_TRACE
    if _POSTERIOR_TRACE is not None:
        return _POSTERIOR_TRACE
    trace_path = MODELS_DIR / "stacking_lr_rl.npz"
    if not trace_path.exists():
        return None
    try:
        data = np.load(str(trace_path))
        _POSTERIOR_TRACE = {k: data[k] for k in data.files}
        n_samples = _POSTERIOR_TRACE["alpha"].shape[0]
        print(f"  [Bayes] Posterior trace loaded: {n_samples} samples")
        return _POSTERIOR_TRACE
    except Exception as e:
        print(f"  [WARN] Could not load posterior trace: {e}")
        return None


def _compute_p_std(stk_model, xgb_rl_raw: float,
                   stacking_feats: dict | None) -> float | None:
    """
    Posterior predictive standard deviation of P(home covers RL).

    Vectorises all NUTS posterior samples in one matrix multiply:
      theta_s = alpha_s + beta_s * logit(p_xgb) + delta_s[seg] + gamma_s @ x
      p_s     = sigmoid(theta_s)
      p_std   = std(p_s)

    High p_std => posterior is spread across the probability space for this
    specific game — model is genuinely uncertain.  Use as a Variance Gate:
    only act on bets where p_std is below a threshold (e.g. 0.035).

    Returns None when trace is unavailable or stk_model is not BayesianStacker.
    """
    if not hasattr(stk_model, "delta"):
        return None
    trace = _load_posterior_trace()
    if trace is None or _is_missing(xgb_rl_raw):
        return None
    try:
        # Build domain feature vector aligned to training feature order
        sf = stacking_feats or {}
        x_dom = np.array(
            [sf.get(c, stk_model.fill_values.get(c, 0.0))
             for c in stk_model.stacking_feature_names],
            dtype=float,
        )   # (n_features,)

        # SP handedness segment  (0=LvL 1=LvR 2=RvL 3=RvR)
        seg = int(sf.get("home_sp_p_throws_R", 1)) * 2 + \
              int(sf.get("away_sp_p_throws_R", 1))

        # Posterior arrays
        alpha_s = trace["alpha"]        # (S,)
        beta_s  = trace["beta"]         # (S,)
        delta_s = trace["delta"]        # (S, 4)
        gamma_s = trace["gamma"]        # (S, n_features)

        logit_p = (np.log(np.clip(float(xgb_rl_raw), 1e-6, 1.0 - 1e-6)) -
                   np.log(1.0 - np.clip(float(xgb_rl_raw), 1e-6, 1.0 - 1e-6)))

        # Linear predictor for every posterior sample  →  (S,)
        theta_s = (alpha_s
                   + beta_s   * logit_p
                   + delta_s[:, seg]
                   + gamma_s  @ x_dom)

        p_s = 1.0 / (1.0 + np.exp(-theta_s))
        return float(np.std(p_s))
    except Exception as e:
        print(f"  [WARN] p_std computation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Three-Part Lock helpers
# ---------------------------------------------------------------------------

def _load_stacking_model():
    """
    Load stacking LR (preferred) or fall back to Platt calibrator.

    Returns an object with a .predict_proba(raw_prob, feat_df) interface,
    or None if neither artifact exists.
    """
    import pickle, sys
    stk_path = MODELS_DIR / "stacking_lr_rl.pkl"
    cal_path  = MODELS_DIR / "calibrator_rl.pkl"

    if stk_path.exists():
        try:
            # Both BayesianStacker (v5.1) and legacy StackingModel may be serialised
            # from train_xgboost.py running as __main__.  Register both classes so
            # pickle can resolve the reference regardless of which version is on disk.
            import train_xgboost as _txgb
            for _cls_name in ("BayesianStacker", "StackingModel"):
                if hasattr(_txgb, _cls_name):
                    _cls = getattr(_txgb, _cls_name)
                    if not hasattr(sys.modules.get("__main__"), _cls_name):
                        sys.modules["__main__"].__dict__[_cls_name] = _cls
            with open(stk_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"  [WARN] Could not load stacking model: {e}")

    if cal_path.exists():
        try:
            with open(cal_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"  [WARN] Could not load calibrator: {e}")

    return None


def _get_p_model(stk_model, xgb_rl_raw: float, stacking_feats: dict | None,
                 lgbm_raw: float = None, cat_raw: float = None) -> float:
    """
    Apply stacking model (or calibrator) to get calibrated P_model.

    BayesianStacker (new, v5.1) — identified by .delta attribute:
      .predict(xgb_raw, feat_df, segment_id=...)
      segment_id derived from home_sp_p_throws_R / away_sp_p_throws_R in
      stacking_feats.

    Legacy StackingModel — identified by .stacking_feature_names but no .delta:
      .predict(xgb_raw, feat_df, lgbm_raw=None, cat_raw=None)

    Plain Platt/isotonic calibrators use only xgb_raw (single-model path).
    Falls back to raw XGBoost probability on any error.
    """
    if stk_model is None or _is_missing(xgb_rl_raw):
        return xgb_rl_raw

    # ── BayesianStacker path ────────────────────────────────────────────────
    if hasattr(stk_model, "delta"):
        try:
            row = {}
            for col in stk_model.stacking_feature_names:
                row[col] = (stacking_feats or {}).get(
                    col, stk_model.fill_values.get(col, 0.0))
            feat_df = pd.DataFrame([row])
            # Derive SP handedness segment from stacking_feats (default RvR=3)
            sf = stacking_feats or {}
            h = int(sf.get("home_sp_p_throws_R", 1))
            a = int(sf.get("away_sp_p_throws_R", 1))
            seg_id = np.array([h * 2 + a], dtype=np.int32)
            return float(stk_model.predict(
                np.array([xgb_rl_raw]), feat_df, segment_id=seg_id
            )[0])
        except Exception as e:
            print(f"  [WARN] BayesianStacker.predict failed: {e} — using raw XGB prob")
            return float(xgb_rl_raw)

    # ── Legacy StackingModel path ───────────────────────────────────────────
    if hasattr(stk_model, "stacking_feature_names"):
        try:
            row = {}
            for col in stk_model.stacking_feature_names:
                row[col] = (stacking_feats or {}).get(
                    col, stk_model.fill_values.get(col, 0.0))
            feat_df  = pd.DataFrame([row])
            lgbm_arr = np.array([float(lgbm_raw)]) if lgbm_raw is not None else None
            cat_arr  = np.array([float(cat_raw)])  if cat_raw  is not None else None
            return float(stk_model.predict(
                np.array([xgb_rl_raw]), feat_df,
                lgbm_raw=lgbm_arr, cat_raw=cat_arr,
            )[0])
        except Exception as e:
            print(f"  [WARN] StackingModel.predict failed: {e} — using raw XGB prob")
            return float(xgb_rl_raw)

    # ── Plain sklearn calibrator (Platt LogisticRegression or isotonic) ─────
    try:
        return float(stk_model.predict_proba([[xgb_rl_raw]])[0, 1])
    except AttributeError:
        try:
            return float(stk_model.predict([xgb_rl_raw])[0])
        except Exception:
            pass

    return float(xgb_rl_raw)


def _apply_three_part_lock(
    p_model: float,
    p_true,        # Pinnacle implied prob — may be NaN/None
    retail_implied_prob,   # US retail implied prob — may be NaN/None
    retail_ml_odds,        # US retail ML odds (American) — may be NaN/None
) -> dict:
    """
    Three-Part Lock execution gate.

    Always returns a dict with gate results.  ``tier`` is None when no lock.

    Gates:
    1. Sanity:     abs(p_model - p_true) <= SANITY_THRESHOLD (4%)
                   Pinnacle preferred. If Pinnacle missing, falls back to retail
                   with a looser threshold: abs(p_model - p_retail) <= 8%.
                   If both are missing, sanity gate fails conservatively.
    2. Odds floor: retail_ml_odds >= ODDS_FLOOR (-225)
                   If retail_ml_odds missing, defaults to pass (no data = no veto).
    3. Edge:       p_model - retail_implied_prob >= EDGE_TIER2 (1%)
                   If retail_implied_prob missing, edge gate fails.
    """
    result = {
        "tier": None,
        "dollar_stake": None,
        "sanity_pass": False,
        "odds_floor_pass": False,
        "edge_pass": False,
        "edge": np.nan,
        "p_true": p_true,
        "retail_implied": retail_implied_prob,
    }

    # Gate 1 — Sanity
    # Prefer Pinnacle (sharp book) with tight 4% threshold.
    # For early games where Pinnacle hasn't posted yet, fall back to retail
    # with a wider 8% threshold (retail lines are noisier / more juice).
    if not _is_missing(p_true):
        sanity_ok = abs(p_model - float(p_true)) <= SANITY_THRESHOLD
        result["sanity_source"] = "pinnacle"
    elif not _is_missing(retail_implied_prob):
        sanity_ok = abs(p_model - float(retail_implied_prob)) <= SANITY_THRESHOLD_RETAIL
        result["sanity_source"] = "retail_fallback"
    else:
        return result   # no price reference at all — conservative fail
    result["sanity_pass"] = sanity_ok
    if not sanity_ok:
        return result

    # Gate 2 — Odds floor
    if _is_missing(retail_ml_odds):
        odds_floor_ok = True   # no data → don't veto
    else:
        odds_floor_ok = float(retail_ml_odds) >= ODDS_FLOOR
    result["odds_floor_pass"] = odds_floor_ok
    if not odds_floor_ok:
        return result

    # Gate 3 — Edge vs retail
    if _is_missing(retail_implied_prob):
        return result
    edge = p_model - float(retail_implied_prob)
    result["edge"] = round(edge, 4)
    if edge < EDGE_TIER2:
        return result
    result["edge_pass"] = True

    # All gates passed — determine tier
    tier = 1 if edge >= EDGE_TIER1 else 2

    # Kelly staking: f* = (b*p - q) / b
    fraction = 0.25 if tier == 1 else 0.125
    if not _is_missing(retail_ml_odds):
        mo = float(retail_ml_odds)
        b  = (100 / abs(mo)) if mo < 0 else (mo / 100)
        p, q = p_model, 1 - p_model
        kelly_f = max(0.0, (b * p - q) / b)
    else:
        kelly_f = max(0.0, p_model - 0.524)   # rough vs -110 breakeven

    result["tier"] = tier
    result["dollar_stake"] = round(min(fraction * kelly_f * SYNTHETIC_BANKROLL, MAX_BET_DOLLARS), 2)
    return result


def _blend_ml_prob(model_prob: float, market_odds) -> tuple[float, float]:
    """
    Blend model ML win probability with Vegas implied probability.

    The Monte Carlo only sees starting pitcher quality. Vegas bakes in lineup,
    bullpen, park factors, and roster depth — all the things our model misses.

    Blend: 35% model (pitcher matchup signal) + 65% Vegas (everything else)

    Returns (blended_prob, market_implied_prob)
    """
    if _is_missing(market_odds):
        return model_prob, np.nan
    implied = ml_to_prob(float(market_odds))
    if np.isnan(implied):
        return model_prob, implied
    blended = 0.35 * model_prob + 0.65 * implied
    return round(blended, 4), round(implied, 4)


# ---------------------------------------------------------------------------
# Parlay helper functions
# ---------------------------------------------------------------------------

def _amt_to_dec(o: float) -> float:
    """American odds → decimal odds."""
    return (o / 100 + 1) if o > 0 else (100 / abs(o) + 1)


def _dec_to_amt(d: float) -> int:
    """Decimal odds → American odds (integer)."""
    return int(round((d - 1) * 100)) if d >= 2.0 else int(round(-100 / (d - 1)))


def _implied_to_amt(p: float):
    """Implied probability → approximate American odds, or None if invalid."""
    if not (0.01 < p < 0.99):
        return None
    return (int(round(-(p / (1 - p)) * 100)) if p >= 0.5
            else int(round(((1 - p) / p) * 100)))


def _leg_corr_val(t1: str, t2: str):
    """
    Return Pearson ρ (float) or "subset" for a same-game leg pair.
    Returns 0.0 if no known correlation (cross-game or unregistered pair).
    """
    key = (t1, t2) if (t1, t2) in _LEG_CORR else (t2, t1) if (t2, t1) in _LEG_CORR else None
    return _LEG_CORR[key] if key else 0.0


def _joint_prob_two(p1: float, p2: float, corr) -> float:
    """
    Joint probability of two events given Pearson correlation ρ, or
    "subset" when one event is a strict subset of the other.
    """
    if corr == "subset":
        return min(p1, p2)
    rho = float(corr)
    s1  = (p1 * (1 - p1)) ** 0.5
    s2  = (p2 * (1 - p2)) ** 0.5
    return max(0.001, min(0.999, p1 * p2 + rho * s1 * s2))


def _build_parlay_legs(r: dict) -> list[dict]:
    """
    Extract all parlay-eligible legs from one game result row.
    Returns list of leg dicts with: game_key, home, away, leg_type,
    label, model_prob, american_odds, decimal_odds, market_prob, edge.
    """
    home = r["home_team"]
    away = r["away_team"]
    gkey = f"{away}@{home}"
    legs: list[dict] = []

    def _add(leg_type: str, label: str, model_prob, american_odds):
        if _is_missing(model_prob) or _is_missing(american_odds):
            return
        mp = float(model_prob)
        ao = float(american_odds)
        # Skip extreme probabilities and extreme prices
        if mp < _PARLAY_MIN_LEG_PROB or mp > (1 - _PARLAY_MIN_LEG_PROB):
            return
        if ao < -600 or ao > 800:
            return
        mkt_p = ml_to_prob(ao)
        if np.isnan(mkt_p):
            return
        edge  = mp - mkt_p
        if edge < _PARLAY_MIN_EDGE:
            return
        legs.append({
            "game_key":    gkey,
            "home":        home,
            "away":        away,
            "leg_type":    leg_type,
            "label":       label,
            "model_prob":  mp,
            "american_odds": int(round(ao)),
            "decimal_odds":  _amt_to_dec(ao),
            "market_prob": round(mkt_p, 4),
            "edge":        round(edge, 4),
        })

    # ── Moneyline ─────────────────────────────────────────────────────────────
    ml_home_p = r.get("ml_lock_p_model")
    _add("ML_HOME", f"{home} ML", ml_home_p, r.get("vegas_ml_home"))
    if not _is_missing(ml_home_p):
        _add("ML_AWAY", f"{away} ML", 1 - float(ml_home_p), r.get("vegas_ml_away"))

    # ── Run line ──────────────────────────────────────────────────────────────
    _add("RL_HOME", f"{home} -1.5", r.get("lock_p_model"), r.get("rl_odds"))
    rl_away_odds = None
    if not _is_missing(r.get("rl_odds")):
        ho = float(r["rl_odds"])
        rl_away_odds = int(round((-ho - 20) if ho < 0 else (-ho + 20)))
    _add("RL_AWAY", f"{away} +1.5", r.get("away_lock_p_model"), rl_away_odds)

    # ── Over / Under ──────────────────────────────────────────────────────────
    mc_over   = r.get("mc_over_prob")
    ri_over   = r.get("retail_implied_over")
    ri_under  = r.get("retail_implied_under")
    ou_line   = r.get("ou_posted_line") or r.get("vegas_total")
    line_str  = f"{float(ou_line):.1f}" if not _is_missing(ou_line) else ""
    ov_odds   = _implied_to_amt(float(ri_over))  if not _is_missing(ri_over)  else -110
    un_odds   = _implied_to_amt(float(ri_under)) if not _is_missing(ri_under) else -110
    if not _is_missing(mc_over):
        _add("OVER",  f"OVER {line_str}",  float(mc_over),         ov_odds)
        _add("UNDER", f"UNDER {line_str}", 1 - float(mc_over),     un_odds)

    # ── K props ───────────────────────────────────────────────────────────────
    for side_type, pfx, sp_name in [
        ("K_HOME", "home", str(r.get("home_sp") or "SP").split()[-1].title()),
        ("K_AWAY", "away", str(r.get("away_sp") or "SP").split()[-1].title()),
    ]:
        kp_mp  = r.get(f"{pfx}_k_model_over")
        kp_ao  = r.get(f"{pfx}_k_over_odds")
        kp_ln  = r.get(f"{pfx}_k_line")
        if not _is_missing(kp_mp) and not _is_missing(kp_ao) and not _is_missing(kp_ln):
            _add(side_type, f"{sp_name} K O{float(kp_ln):.1f}", kp_mp, kp_ao)

    return legs


def _find_parlay_combos(
    results: list[dict],
    target_min: int = _PARLAY_TARGET_MIN,
    target_max: int = _PARLAY_TARGET_MAX,
    top_n: int = _PARLAY_TOP_N,
) -> list[dict]:
    """
    Find 2- and 3-leg combos whose combined American odds fall in
    [target_min, target_max].  Correlation between same-game legs is
    applied to the model joint probability.  Market odds multiply naively
    (sportsbook applies its own SGP haircut for same-game legs).

    Returns top_n combos sorted by combined model edge (descending).
    """
    import itertools

    all_legs: list[dict] = []
    for r in results:
        all_legs.extend(_build_parlay_legs(r))

    if len(all_legs) < 2:
        return []

    combos: list[dict] = []

    for n in (2, 3):
        for idxs in itertools.combinations(range(len(all_legs)), n):
            legs = [all_legs[i] for i in idxs]

            # ── Skip degenerate same-game same-leg-type duplicates ─────────
            leg_keys = [f"{l['game_key']}:{l['leg_type']}" for l in legs]
            if len(set(leg_keys)) < n:
                continue
            # Skip same-game pairs of exactly opposing probabilities (ML both sides)
            skip = False
            for i in range(n):
                for j in range(i + 1, n):
                    if (legs[i]["game_key"] == legs[j]["game_key"] and
                            {legs[i]["leg_type"], legs[j]["leg_type"]} in (
                                {"ML_HOME", "ML_AWAY"}, {"OVER", "UNDER"},
                                {"RL_HOME", "RL_AWAY"})):
                        skip = True
                        break
                if skip:
                    break
            if skip:
                continue

            # ── Combined market decimal odds ───────────────────────────────
            combined_dec = 1.0
            for lg in legs:
                combined_dec *= lg["decimal_odds"]
            combined_amt = _dec_to_amt(combined_dec)
            if not (target_min <= combined_amt <= target_max):
                continue

            # ── Model joint probability (with same-game correlations) ──────
            # Start with independent product
            jp = 1.0
            for lg in legs:
                jp *= lg["model_prob"]

            # Adjust for same-game correlated pairs
            pair_idxs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            for i, j in pair_idxs:
                if legs[i]["game_key"] != legs[j]["game_key"]:
                    continue
                corr = _leg_corr_val(legs[i]["leg_type"], legs[j]["leg_type"])
                pi, pj = legs[i]["model_prob"], legs[j]["model_prob"]
                p_pair_ind = pi * pj
                p_pair_adj = _joint_prob_two(pi, pj, corr)
                if p_pair_ind > 1e-9:
                    jp *= p_pair_adj / p_pair_ind
            jp = max(0.001, min(0.999, jp))

            # ── Edge and quality gates ─────────────────────────────────────
            market_joint = 1.0 / combined_dec
            combo_edge   = jp - market_joint
            max_leg_edge = max(lg["edge"] for lg in legs)

            if max_leg_edge < _PARLAY_MIN_BEST_LEG_EDGE:
                continue
            if combo_edge < _PARLAY_MIN_COMBO_EDGE:
                continue

            same_game_pairs = [
                (legs[i]["game_key"], legs[i]["leg_type"], legs[j]["leg_type"])
                for i, j in pair_idxs if legs[i]["game_key"] == legs[j]["game_key"]
            ]

            combos.append({
                "legs":               legs,
                "combined_dec":       round(combined_dec, 3),
                "combined_amt":       combined_amt,
                "model_joint_prob":   round(jp, 4),
                "market_joint_prob":  round(market_joint, 4),
                "combo_edge":         round(combo_edge, 4),
                "n_legs":             n,
                "same_game_pairs":    same_game_pairs,
                "is_sgp":             len(same_game_pairs) > 0,
            })

    # Sort by combo edge; de-duplicate to avoid showing same legs repeatedly
    combos.sort(key=lambda c: c["combo_edge"], reverse=True)
    seen: set[tuple] = set()
    out: list[dict] = []
    for c in combos:
        fingerprint = tuple(sorted(f"{l['game_key']}|{l['leg_type']}" for l in c["legs"]))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        out.append(c)
        if len(out) >= top_n:
            break

    return out


def _best_bet(
    mc_home_win: float,
    mc_home_covers_rl: float,
    mc_home_covers_25: float,
    mc_away_covers_25: float,
    blended_rl: float,
    vegas_ml_home,
    vegas_ml_away,
    rl_home_odds,
    alt_home_25_odds,
    alt_away_25_odds,
    rl_home_line: float = -1.5,
) -> dict:
    """
    Evaluate edge across all available lines and return the best bet.

    ML probabilities are blended 35% model / 65% Vegas to account for
    team-level factors (lineup, bullpen, depth) the pitcher-only model misses.
    RL probabilities use the existing blended MC+XGB model as-is.

    Returns dict with keys: line, model_prob, market_prob, market_odds, edge,
                            tier, signal, raw_model_prob, market_deviation
    """
    candidates = []

    def add(line, model_prob, market_odds, raw_model_prob=None):
        if _is_missing(market_odds):
            return
        market_odds_f = float(market_odds)
        edge = _edge_vs_line(model_prob, market_odds_f)
        if not np.isnan(edge):
            market_implied = ml_to_prob(market_odds_f)
            candidates.append({
                "line":           line,
                "model_prob":     round(model_prob, 3),
                "raw_model_prob": round(raw_model_prob or model_prob, 3),
                "market_implied": round(market_implied, 3),
                "market_odds":    int(market_odds_f),
                "edge":           round(edge, 4),
            })

    # Moneyline — blend with Vegas so team factors are accounted for
    if not _is_missing(vegas_ml_home):
        home_blended, _ = _blend_ml_prob(mc_home_win, vegas_ml_home)
        add("ML (home)", home_blended, vegas_ml_home, raw_model_prob=mc_home_win)
    if not _is_missing(vegas_ml_away):
        away_blended, _ = _blend_ml_prob(1 - mc_home_win, vegas_ml_away)
        add("ML (away)", away_blended, vegas_ml_away, raw_model_prob=1 - mc_home_win)

    # Run line candidates.
    # rl_home_line is the actual spread for the home team from the odds file:
    #   -1.5 means home team is favored (giving runs)
    #   +1.5 means home team is the underdog (getting runs)
    # We label bets correctly based on the actual line, not the hardcoded -1.5.
    if not _is_missing(rl_home_odds):
        ho = float(rl_home_odds)
        # Estimate the other side: flip odds sign and subtract 20 cents juice
        other_rl_est = int(round((-ho - 20) if ho < 0 else (-ho + 20)))

        if float(rl_home_line) < 0:
            # Home team is giving runs (-1.5): away team is getting runs (+1.5)
            home_label = "RL -1.5 (home)"
            away_label = "RL +1.5 (away)"
            home_model_prob = blended_rl
            away_model_prob = 1 - blended_rl
        else:
            # Home team is getting runs (+1.5): away team is giving runs (-1.5)
            home_label = "RL +1.5 (home)"
            away_label = "RL -1.5 (away)"
            home_model_prob = 1 - blended_rl   # home covering +1.5 = away NOT covering -1.5
            away_model_prob = blended_rl        # away covering -1.5 = blended_rl

        add(home_label, home_model_prob, rl_home_odds)
        add(away_label, away_model_prob, other_rl_est)

    # Alternate run line candidates
    add("RL -2.5 (home)", mc_home_covers_25, alt_home_25_odds)
    add("RL +2.5 (away)", mc_away_covers_25, alt_away_25_odds)

    if not candidates:
        return {"line": "", "model_prob": blended_rl, "market_odds": None,
                "edge": 0.0, "tier": ""}

    # Pick highest edge candidate
    best = max(candidates, key=lambda c: c["edge"])

    if best["edge"] >= 0.08:
        tier = "**"
    elif best["edge"] >= 0.04:
        tier = "*"
    else:
        tier = ""

    # ── Guardrail: cap tier at * (lean) for extreme underdog ML bets ──────────
    # When the market prices a team at > +400, there's high uncertainty and
    # our pitcher-only model can't see roster/bullpen depth factors that explain
    # those big odds. Cap at lean so we don't label them "STRONG" bets.
    if "ML" in best.get("line", ""):
        mo = best.get("market_odds", 0) or 0
        if mo > 400:
            if tier == "**":
                tier = "*"   # downgrade strong → lean
                best["tier_capped"] = True  # flag for display

    best["tier"] = tier

    # How far does the raw model diverge from the market? Flag large deviations.
    raw   = best.get("raw_model_prob", best["model_prob"])
    mkt   = best.get("market_implied", best["model_prob"])
    best["market_deviation"] = round(abs(raw - mkt), 3) if not np.isnan(mkt) else 0.0

    # Build a clean signal string like "HOME -1.5 **" or "ML (away) *"
    if tier:
        best["signal"] = f"{best['line']} {tier}"
    else:
        best["signal"] = ""

    return best


# ---------------------------------------------------------------------------
# RUN PREDICTIONS
# ---------------------------------------------------------------------------

def _load_team_stats() -> pd.DataFrame:
    """Load 2026 team batting/bullpen stats, rebuilding if stale (>6 hrs old)."""
    import datetime as _dt
    from build_team_stats_2026 import build as _build_team_stats
    path = Path("data/statcast/team_stats_2026.parquet")
    if not path.exists():
        print("  Building team stats 2026 ...")
        _build_team_stats(verbose=False)
    else:
        age_hours = (_dt.datetime.now().timestamp() - path.stat().st_mtime) / 3600
        if age_hours > 6:
            print("  Refreshing team stats 2026 (stale) ...")
            _build_team_stats(verbose=False)
    return pd.read_parquet(path, engine="pyarrow").set_index("batting_team")


def _load_lineup_quality(date_str: str) -> dict:
    """
    Load today's lineup wRC+ quality scores per (game_pk, team).
    Returns dict: {(game_pk, team): wrc_plus} or {} if not available.
    Falls back gracefully — team RS/G is still used if lineup data missing.
    """
    try:
        from build_lineup_quality import build as _build_lq
        return _build_lq(date_str, verbose=False)
    except Exception as e:
        print(f"  [WARN] Lineup quality unavailable: {e}")
        return {}


def run_card(date_str: str, min_edge: float = 0.0) -> list[dict]:
    from monte_carlo_runline import predict_game, load_profiles, load_pitcher_10d

    # Load all data sources
    lineups    = load_lineups(date_str)
    odds       = load_odds(date_str)
    weather    = load_weather(date_str)
    profiles   = load_profiles()
    pitcher_10d = load_pitcher_10d()
    kprops     = load_kprops(date_str)
    stk_model  = _load_stacking_model()
    try:
        team_stats = _load_team_stats()
    except Exception as e:
        print(f"  [WARN] Could not load team stats: {e} — using pitcher-only model")
        team_stats = None

    lineup_quality = _load_lineup_quality(date_str)

    if len(lineups) == 0:
        print("  No games found for today.")
        return []

    # Merge odds and weather onto lineups
    if not odds.empty:
        odds_cols = ["home_team", "away_team", "close_ml_home",
                     "close_ml_away", "close_total", "runline_home", "runline_home_odds"]
        # Three-Part Lock columns — include if present
        for lock_col in [
            "P_true_home", "P_true_away", "P_true_rl_home", "P_true_rl_away",
            "P_true_over", "P_true_under",
            "retail_implied_home", "retail_implied_away",
            "retail_implied_rl_home", "retail_implied_rl_away",
            "retail_implied_over", "retail_implied_under",
        ]:
            if lock_col in odds.columns:
                odds_cols.append(lock_col)
        # Alternate line odds
        for alt_col in ["alt_rl_home_25_odds", "alt_rl_away_25_odds",
                        "alt_rl_home_ml_odds", "alt_rl_away_ml_odds"]:
            if alt_col in odds.columns:
                odds_cols.append(alt_col)
        # Game start time (for circadian features)
        if "game_hour_et" in odds.columns:
            odds_cols.append("game_hour_et")
        odds_merge = odds[odds_cols].drop_duplicates(subset=["home_team", "away_team"])
        lineups = lineups.merge(odds_merge, on=["home_team", "away_team"],
                                how="left")

    if not weather.empty:
        lineups = lineups.merge(weather, on="home_team", how="left")

    # Fill defaults
    if "temp_f" not in lineups.columns:
        lineups["temp_f"] = 72.0
    lineups["temp_f"] = lineups["temp_f"].fillna(72.0)

    results = []
    today_month = int(date_str.split("-")[1])

    print(f"\n  Running predictions for {len(lineups)} games on {date_str} ...\n")

    for _, game in lineups.iterrows():
        home = str(game["home_team"])
        away = str(game["away_team"])
        home_sp = str(game.get("home_starter_name", ""))
        away_sp = str(game.get("away_starter_name", ""))
        temp    = float(game.get("temp_f", 72.0))

        if not home_sp or home_sp == "nan" or not away_sp or away_sp == "nan":
            print(f"  {away} @ {home}: missing starter(s) — adding as TBD row (Vegas-only pick)")
            # Use Vegas ML implied prob as best available pick
            vegas_ml_h = game.get("close_ml_home")
            vegas_ml_a = game.get("close_ml_away")
            vegas_pick = None
            if not _is_missing(vegas_ml_h):
                home_imp = ml_to_prob(float(vegas_ml_h))
                vegas_pick = home if home_imp >= 0.5 else away
            results.append({
                "game":          f"{away} @ {home}",
                "home_team":     home,
                "away_team":     away,
                "home_sp":       home_sp.upper() if home_sp and home_sp != "nan" else "TBD",
                "away_sp":       away_sp.upper() if away_sp and away_sp != "nan" else "TBD",
                "game_time_et":  game.get("game_time_et", ""),
                "lineup_confirmed": False,
                "vegas_ml_home": vegas_ml_h,
                "vegas_ml_away": vegas_ml_a,
                "vegas_pick":    vegas_pick,   # best guess based on odds alone
                "rl_signal":     "",
                "total_signal":  "",
                "best_line":     "",
                "best_tier":     "",
            })
            continue

        # Normalize name format: "Last, First" -> "FIRST LAST"
        def norm(name):
            name = str(name).strip()
            if "," in name:
                parts = [p.strip() for p in name.split(",", 1)]
                return f"{parts[1]} {parts[0]}".upper()
            return name.upper()

        home_sp_norm = norm(home_sp)
        away_sp_norm = norm(away_sp)

        vegas_ml      = game.get("close_ml_home")
        vegas_tot     = game.get("close_total")
        rl_odds       = game.get("runline_home_odds")
        rl_home_line  = game.get("runline_home", -1.5)  # actual spread for home team
        if _is_missing(rl_home_line):
            rl_home_line = -1.5

        # Fetch team stats rows (None if team not found)
        home_ts = team_stats.loc[home] if (team_stats is not None and home in team_stats.index) else None
        away_ts = team_stats.loc[away] if (team_stats is not None and away in team_stats.index) else None

        game_pk = game.get("game_pk")
        home_lq = lineup_quality.get((game_pk, home)) or lineup_quality.get((str(game_pk), home))
        away_lq = lineup_quality.get((game_pk, away)) or lineup_quality.get((str(game_pk), away))

        # Build market_odds dict for XGBoost features (true_home_prob, true_away_prob).
        # Prefer Pinnacle de-vigged (P_true_home) — sharpest signal.
        # Fall back to retail de-vig from close_ml_home / close_ml_away.
        def _devig(ml_h, ml_a):
            """Simple de-vig: convert American odds pair to true win probabilities."""
            def _raw(ml):
                if _is_missing(ml):
                    return None
                ml = float(ml)
                return 100 / (ml + 100) if ml >= 0 else abs(ml) / (abs(ml) + 100)
            rh, ra = _raw(ml_h), _raw(ml_a)
            if rh is None or ra is None:
                return None, None
            total = rh + ra
            return rh / total, ra / total

        p_true_h = game.get("P_true_home")
        p_true_a = game.get("P_true_away")
        if _is_missing(p_true_h) or _is_missing(p_true_a):
            # Fall back to retail de-vig
            p_true_h, p_true_a = _devig(game.get("close_ml_home"), game.get("close_ml_away"))

        market_odds_xgb = None
        if not _is_missing(p_true_h) and not _is_missing(p_true_a):
            market_odds_xgb = {
                "true_home_prob": float(p_true_h),
                "true_away_prob": float(p_true_a),
            }

        # Game start hour in ET — from odds pull commence_time
        _game_hour_raw = game.get("game_hour_et")
        game_hour_et_val = float(_game_hour_raw) if not _is_missing(_game_hour_raw) else None

        try:
            res = predict_game(
                home_team=home, away_team=away,
                home_sp_name=home_sp_norm, away_sp_name=away_sp_norm,
                temp_f=temp, month=today_month,
                verbose=False,
                home_team_stats=home_ts,
                away_team_stats=away_ts,
                home_lineup_wrc=home_lq,
                away_lineup_wrc=away_lq,
                pitcher_10d=pitcher_10d,
                posted_total=(float(vegas_tot) if not _is_missing(vegas_tot) else None),
                market_odds=market_odds_xgb,
                game_hour_et=game_hour_et_val,
            )
        except Exception as e:
            print(f"  {away} @ {home}: ERROR — {e}")
            print(_traceback.format_exc())
            continue

        blended_rl  = res.get("blended_home_covers_rl",
                               res.get("mc_home_covers_rl"))
        blended_tot = res.get("blended_expected_total",
                               res.get("mc_expected_total"))
        # Quantile total bounds (populated when quantile model is active)
        total_floor_10th     = res.get("total_floor_10th")
        total_ceiling_90th   = res.get("total_ceiling_90th")
        total_variance_spread = res.get("total_variance_spread")
        xgb_rl_raw  = res.get("xgb_home_covers_rl_raw",
                               res.get("xgb_home_covers_rl", np.nan))
        xgb_rl      = res.get("xgb_home_covers_rl", np.nan)
        mc_rl       = res["mc_home_covers_rl"]

        # --- Three-Part Lock (RL home direction) ---
        # P_model: stacking LR calibration using available domain features.
        # sp_k_pct_diff, sp_xwoba_diff, sp_kminusbb_diff, and batting_matchup_edge
        # are extracted from the XGB feature row inside predict_game().
        # Features not computable at inference (10d stats, bullpen) default to their
        # training fill_values (≈ 0, symmetric diffs) in the stacking LR.
        stacking_feats  = res.get("stacking_feats") or {}
        lgbm_rl_raw     = res.get("lgbm_rl_prob_raw")   # Shadow only — NOT fed to stacker
        cat_rl_raw      = res.get("cat_rl_prob_raw")     # Shadow only — NOT fed to stacker
        # If predict_game() embedded the stacking model, prefer it over the
        # separately loaded stk_model (ensures model version consistency).
        _embedded_stk   = res.get("_stacking_model")
        _active_stk     = _embedded_stk if _embedded_stk is not None else stk_model
        # CRITICAL: Shadow probs are intentionally NOT passed to _get_p_model.
        # The official stacker is XGBoost-only.  lgbm_rl_raw / cat_rl_raw are
        # used below exclusively for ensemble_min / ensemble_max / model_spread.

        # ── 5-Step Double-Normalization Flow (Team Perspective Model) ──────────
        _team_home_norm = res.get("xgb_team_rl_home_norm")
        _team_away_norm = res.get("xgb_team_rl_away_norm")
        stk_feats_away  = _flip_stacking_feats_dict(stacking_feats)

        p_model_rl = p_model_rl_away = None
        p_std_rl   = p_std_rl_away   = None

        if _team_home_norm is not None and _team_away_norm is not None:
            # Steps 3 & 4: Stacker on both normalized Level-1 probs
            stk_raw_home = _get_p_model(_active_stk, _team_home_norm, stacking_feats)
            stk_raw_away = _get_p_model(_active_stk, _team_away_norm, stk_feats_away)
            # Step 5: L2 normalize stacker outputs
            if stk_raw_home is not None and stk_raw_away is not None:
                _stk_sum = max(float(stk_raw_home) + float(stk_raw_away), 1e-9)
                p_model_rl      = float(stk_raw_home) / _stk_sum
                p_model_rl_away = float(stk_raw_away) / _stk_sum
            else:
                p_model_rl      = stk_raw_home
                p_model_rl_away = stk_raw_away
            # p_std for both directions
            p_std_rl      = _compute_p_std(_active_stk, _team_home_norm, stacking_feats)
            p_std_rl_away = _compute_p_std(_active_stk, _team_away_norm, stk_feats_away)

        # ── Shadow Ensemble bounds (variance estimation, does not affect signal) ──
        _shadow_pool = [
            p for p in [xgb_rl_raw, lgbm_rl_raw, cat_rl_raw]
            if p is not None and not _is_missing(p)
        ]
        if len(_shadow_pool) >= 2:
            ensemble_min  = round(float(min(_shadow_pool)), 4)
            ensemble_max  = round(float(max(_shadow_pool)), 4)
            model_spread  = round(ensemble_max - ensemble_min, 4)
        else:
            # Only XGBoost available — bounds collapse to the single estimate
            ensemble_min  = round(float(xgb_rl_raw), 4) if not _is_missing(xgb_rl_raw) else None
            ensemble_max  = ensemble_min
            model_spread  = 0.0

        # Pinnacle and retail columns for RL home
        p_true_rl          = game.get("P_true_rl_home")
        retail_implied_rl  = game.get("retail_implied_rl_home")
        retail_odds_rl     = game.get("runline_home_odds")   # retail RL odds (American)

        lock_result = _apply_three_part_lock(
            p_model         = p_model_rl,
            p_true          = p_true_rl,
            retail_implied_prob = retail_implied_rl,
            retail_ml_odds  = retail_odds_rl,
        )

        # Away-side RL (away team +1.5)
        p_true_rl_away       = game.get("P_true_rl_away")
        retail_implied_rl_away = game.get("retail_implied_rl_away")
        # Estimate away retail RL odds by flipping home odds minus juice
        if not _is_missing(retail_odds_rl):
            _ho = float(retail_odds_rl)
            retail_odds_rl_away = int(round((-_ho - 20) if _ho < 0 else (-_ho + 20)))
        else:
            retail_odds_rl_away = None

        away_lock_result = _apply_three_part_lock(
            p_model             = p_model_rl_away,
            p_true              = p_true_rl_away,
            retail_implied_prob = retail_implied_rl_away,
            retail_ml_odds      = retail_odds_rl_away,
        )

        # --- ML gate (home win direction, blended 35% model / 65% Vegas) ---
        mc_home_win_prob = res.get("mc_home_win_prob", 0.5)
        p_model_ml, _    = _blend_ml_prob(mc_home_win_prob, vegas_ml)
        p_true_ml        = game.get("P_true_home")
        retail_impl_ml   = game.get("retail_implied_home")
        ml_lock_result   = _apply_three_part_lock(
            p_model             = p_model_ml,
            p_true              = p_true_ml,
            retail_implied_prob = retail_impl_ml,
            retail_ml_odds      = vegas_ml,
        )

        # --- O/U gate (evaluate both over and under, surface the better direction) ---
        mc_over_prob     = res.get("mc_over_prob")
        p_true_over      = game.get("P_true_over")
        p_true_under     = game.get("P_true_under")
        retail_impl_over = game.get("retail_implied_over")
        retail_impl_under= game.get("retail_implied_under")

        ou_lock_result  = {"tier": None, "edge": np.nan, "sanity_pass": False,
                           "odds_floor_pass": True, "edge_pass": False,
                           "p_true": p_true_over, "retail_implied": retail_impl_over}
        ou_direction    = "OVER"
        if not _is_missing(mc_over_prob):
            over_lock  = _apply_three_part_lock(
                float(mc_over_prob), p_true_over,  retail_impl_over,  None)
            under_lock = _apply_three_part_lock(
                1 - float(mc_over_prob), p_true_under, retail_impl_under, None)
            # Pick direction with a lock; otherwise best edge
            if over_lock["tier"] is not None:
                ou_lock_result, ou_direction = over_lock,  "OVER"
            elif under_lock["tier"] is not None:
                ou_lock_result, ou_direction = under_lock, "UNDER"
            else:
                oe = over_lock.get("edge") or -99
                ue = under_lock.get("edge") or -99
                if not _is_missing(oe) and not _is_missing(ue):
                    if float(oe) >= float(ue):
                        ou_lock_result, ou_direction = over_lock,  "OVER"
                    else:
                        ou_lock_result, ou_direction = under_lock, "UNDER"
                else:
                    ou_lock_result, ou_direction = over_lock, "OVER"

        # --- K prop lookup ---
        def _k_prop(sp_norm, side):
            """Return (line, model_p_over, implied_over, edge, over_odds, under_odds) or Nones."""
            kp = kprops.get(sp_norm)
            if not kp:
                return None, None, None, None, None, None
            line = kp["line"]
            # Map to precomputed bucket key  e.g. 4.5 -> mc_home_sp_k_over_45
            bucket_key = f"mc_{side}_sp_k_over_{int(line * 10)}"
            model_p = res.get(bucket_key)
            if model_p is None:
                return line, None, kp["implied_over"], None, kp["over_odds"], kp["under_odds"]
            edge = round(float(model_p) - kp["implied_over"], 4)
            return line, round(float(model_p), 4), kp["implied_over"], edge, kp["over_odds"], kp["under_odds"]

        hk_line, hk_model, hk_impl, hk_edge, hk_over_odds, hk_under_odds = _k_prop(home_sp_norm, "home")
        ak_line, ak_model, ak_impl, ak_edge, ak_over_odds, ak_under_odds = _k_prop(away_sp_norm, "away")

        # Gather alternate line odds (only present if odds file has them)
        vegas_ml_away    = game.get("close_ml_away")
        alt_home_25_odds = game.get("alt_rl_home_25_odds")
        alt_away_25_odds = game.get("alt_rl_away_25_odds")

        # Best-bet selection across all available lines
        best = _best_bet(
            mc_home_win      = res.get("mc_home_win_prob", 0.5),
            mc_home_covers_rl= blended_rl,
            mc_home_covers_25= res.get("mc_home_covers_25", 0.0),
            mc_away_covers_25= res.get("mc_away_covers_25", 0.0),
            blended_rl       = blended_rl,
            vegas_ml_home    = vegas_ml,
            vegas_ml_away    = vegas_ml_away,
            rl_home_odds     = rl_odds,
            alt_home_25_odds = alt_home_25_odds,
            alt_away_25_odds = alt_away_25_odds,
            rl_home_line     = rl_home_line,
        )

        signal = best.get("signal", "")

        # Also keep the legacy rl_signal for backwards compat (backtest etc.)
        if blended_rl >= BET_THRESHOLD_HIGH:
            rl_signal_legacy = "HOME -1.5 **"
        elif blended_rl >= BET_THRESHOLD_HIGH_LEAN:
            rl_signal_legacy = "HOME -1.5 *"
        elif blended_rl <= BET_THRESHOLD_LOW:
            rl_signal_legacy = "AWAY +1.5 **"
        elif blended_rl <= BET_THRESHOLD_LOW_LEAN:
            rl_signal_legacy = "AWAY +1.5 *"
        else:
            rl_signal_legacy = ""

        # Total signal
        total_signal = ""
        if not pd.isna(vegas_tot) and not pd.isna(blended_tot):
            diff = blended_tot - float(vegas_tot)
            if abs(diff) >= 0.75:
                total_signal = f"{'OVER' if diff > 0 else 'UNDER'} {vegas_tot}"

        results.append({
            "game":           f"{away} @ {home}",
            "home_team":      home,
            "away_team":      away,
            "home_sp":        home_sp_norm,
            "away_sp":        away_sp_norm,
            "home_sp_flag":   res.get("home_sp_velocity_flag", ""),
            "away_sp_flag":   res.get("away_sp_velocity_flag", ""),
            "home_sp_xwoba":  res.get("home_sp_mu"),
            "away_sp_xwoba":  res.get("away_sp_mu"),
            "temp_f":         temp,
            "mc_rl":          round(mc_rl, 3),
            "xgb_rl":         round(xgb_rl, 3) if not pd.isna(xgb_rl) else None,
            "blended_rl":     round(blended_rl, 3),
            "mc_home_win":    round(res.get("mc_home_win_prob", 0.5), 3),
            "mc_home_cvr25":  round(res.get("mc_home_covers_25", 0.0), 3),
            "mc_away_cvr25":  round(res.get("mc_away_covers_25", 0.0), 3),
            "home_runs_mean": res.get("mc_home_runs_mean"),
            "away_runs_mean": res.get("mc_away_runs_mean"),
            "home_runs_lo":   res.get("mc_home_runs_lo"),
            "home_runs_hi":   res.get("mc_home_runs_hi"),
            "away_runs_lo":   res.get("mc_away_runs_lo"),
            "away_runs_hi":   res.get("mc_away_runs_hi"),
            "mc_total":       round(res["mc_expected_total"], 2),
            "blended_total":  round(blended_tot, 2) if blended_tot else None,
            "vegas_ml_home":  vegas_ml,
            "vegas_ml_away":  vegas_ml_away,
            "vegas_total":    vegas_tot,
            "rl_odds":        rl_odds,
            "alt_home_25_odds": alt_home_25_odds,
            "alt_away_25_odds": alt_away_25_odds,
            "vegas_implied":  round(ml_to_prob(vegas_ml), 3) if not pd.isna(vegas_ml) else None,
            # Best bet recommendation
            "best_line":         best.get("line", ""),
            "best_model_prob":   best.get("model_prob", blended_rl),
            "best_raw_model":    best.get("raw_model_prob", blended_rl),
            "best_market_implied": best.get("market_implied"),
            "best_market_odds":  best.get("market_odds"),
            "best_edge":         best.get("edge", 0.0),
            "best_tier":         best.get("tier", ""),
            "best_tier_capped":  best.get("tier_capped", False),
            "market_deviation":  best.get("market_deviation", 0.0),
            # Three-Part Lock fields — RL home (-1.5)
            "lock_tier":         lock_result["tier"],
            "lock_dollars":      lock_result["dollar_stake"],
            "lock_edge":         lock_result["edge"],
            "lock_p_model":      round(p_model_rl, 4),
            "lock_p_true":       p_true_rl,
            "lock_retail_implied": retail_implied_rl,
            "lock_sanity_pass":  lock_result["sanity_pass"],
            "lock_odds_floor_pass": lock_result["odds_floor_pass"],
            "lock_edge_pass":    lock_result["edge_pass"],
            "xgb_rl_raw":        round(xgb_rl_raw, 4) if not _is_missing(xgb_rl_raw) else None,
            # Shadow Ensemble variance bounds (inference-only — not in stacker)
            "ensemble_min":      ensemble_min,
            "ensemble_max":      ensemble_max,
            "model_spread":      model_spread,
            # Posterior predictive uncertainty (Variance Gate)
            # p_std < 0.030 => tight posterior => high conviction
            # p_std >= 0.030 => spread posterior => model uncertain about this game
            "p_std":             round(p_std_rl, 4) if p_std_rl is not None else None,
            "variance_gate":     (p_std_rl < 0.030) if p_std_rl is not None else None,
            "p_std_away":        round(p_std_rl_away, 4) if p_std_rl_away is not None else None,
            "variance_gate_away": (p_std_rl_away < 0.030) if p_std_rl_away is not None else None,
            # Three-Part Lock fields — RL away (+1.5)
            "away_lock_tier":          away_lock_result["tier"],
            "away_lock_dollars":       away_lock_result["dollar_stake"],
            "away_lock_edge":          away_lock_result["edge"],
            "away_lock_p_model":       round(p_model_rl_away, 4),
            "away_lock_p_true":        p_true_rl_away,
            "away_lock_retail_implied": retail_implied_rl_away,
            "away_lock_sanity_pass":   away_lock_result["sanity_pass"],
            "away_lock_odds_floor_pass": away_lock_result["odds_floor_pass"],
            "away_lock_edge_pass":     away_lock_result["edge_pass"],
            # Three-Part Lock fields — ML
            "ml_lock_tier":      ml_lock_result["tier"],
            "ml_lock_dollars":   ml_lock_result["dollar_stake"],
            "ml_lock_edge":      ml_lock_result["edge"],
            "ml_lock_p_model":   round(p_model_ml, 4),
            "ml_lock_p_true":    p_true_ml,
            "ml_lock_retail_implied": retail_impl_ml,
            "ml_lock_sanity_pass":    ml_lock_result["sanity_pass"],
            "ml_lock_edge_pass":      ml_lock_result["edge_pass"],
            # Three-Part Lock fields — O/U
            "ou_lock_tier":      ou_lock_result["tier"],
            "ou_lock_dollars":   ou_lock_result.get("dollar_stake"),
            "ou_lock_edge":      ou_lock_result["edge"],
            "ou_direction":      ou_direction,
            "ou_posted_line":    vegas_tot,
            "ou_p_model":        round(float(mc_over_prob), 4) if not _is_missing(mc_over_prob) else None,
            "ou_p_true":         p_true_over if ou_direction == "OVER" else p_true_under,
            "ou_p_retail":       retail_impl_over if ou_direction == "OVER" else retail_impl_under,
            "ou_lock_sanity_pass":  ou_lock_result["sanity_pass"],
            "ou_lock_edge_pass":    ou_lock_result["edge_pass"],
            # Raw O/U probs for display
            "mc_over_prob":      round(float(mc_over_prob), 4) if not _is_missing(mc_over_prob) else None,
            "p_true_over":       p_true_over,
            "p_true_under":      p_true_under,
            "retail_implied_over": retail_impl_over,
            "retail_implied_under": retail_impl_under,
            # Total confidence band (MC simulation)
            "mc_total_lo":       res.get("mc_total_lo"),
            "mc_total_hi":       res.get("mc_total_hi"),
            # Quantile regression bounds (XGBoost multi-quantile model)
            "total_floor_10th":      total_floor_10th,
            "total_ceiling_90th":    total_ceiling_90th,
            "total_variance_spread": total_variance_spread,
            # K prop fields
            "home_k_line":       hk_line,
            "home_k_model_over": hk_model,
            "home_k_implied_over": hk_impl,
            "home_k_edge":       hk_edge,
            "home_k_over_odds":  hk_over_odds,
            "home_k_under_odds": hk_under_odds,
            "away_k_line":       ak_line,
            "away_k_model_over": ak_model,
            "away_k_implied_over": ak_impl,
            "away_k_edge":       ak_edge,
            "away_k_over_odds":  ak_over_odds,
            "away_k_under_odds": ak_under_odds,
            # Legacy fields (used by backtest)
            "rl_signal":      rl_signal_legacy,
            "total_signal":   total_signal,
            "lineup_confirmed": bool(game.get("home_lineup_confirmed", False)
                                     and game.get("away_lineup_confirmed", False)),
            "game_time_et":   game.get("game_time_et", ""),
            "home_lineup_wrc": round(home_lq, 1) if home_lq else None,
            "away_lineup_wrc": round(away_lq, 1) if away_lq else None,
            # F5 predictions
            "mc_f5_home_runs":      res.get("mc_f5_home_runs"),
            "mc_f5_away_runs":      res.get("mc_f5_away_runs"),
            "mc_f5_total":          res.get("mc_f5_total"),
            "mc_f5_home_win_prob":  res.get("mc_f5_home_win_prob"),
            "mc_f5_home_covers_rl": res.get("mc_f5_home_covers_rl"),
            # F1 / NRFI
            "mc_nrfi_prob":         res.get("mc_nrfi_prob"),
            "mc_p_home_scores_f1":  res.get("mc_p_home_scores_f1"),
            "mc_p_away_scores_f1":  res.get("mc_p_away_scores_f1"),
            "mc_f1_home_runs":      res.get("mc_f1_home_runs"),
            "mc_f1_away_runs":      res.get("mc_f1_away_runs"),
            # K props
            "mc_home_sp_k_mean":    res.get("mc_home_sp_k_mean"),
            "mc_away_sp_k_mean":    res.get("mc_away_sp_k_mean"),
            "mc_home_sp_k_over_35": res.get("mc_home_sp_k_over_35"),
            "mc_home_sp_k_over_45": res.get("mc_home_sp_k_over_45"),
            "mc_home_sp_k_over_55": res.get("mc_home_sp_k_over_55"),
            "mc_home_sp_k_over_65": res.get("mc_home_sp_k_over_65"),
            "mc_away_sp_k_over_35": res.get("mc_away_sp_k_over_35"),
            "mc_away_sp_k_over_45": res.get("mc_away_sp_k_over_45"),
            "mc_away_sp_k_over_55": res.get("mc_away_sp_k_over_55"),
            "mc_away_sp_k_over_65": res.get("mc_away_sp_k_over_65"),
        })

    # ── Sort results: locks first → near-miss → has Pinnacle → no Pinnacle ──
    def _sort_key(r):
        has_lock = any(r.get(k) is not None
                       for k in ("lock_tier", "ml_lock_tier", "ou_lock_tier"))
        has_pinnacle = not _is_missing(r.get("lock_p_true"))
        best_edge = max(
            float(r.get("lock_edge")    or -99) if not _is_missing(r.get("lock_edge"))    else -99,
            float(r.get("ml_lock_edge") or -99) if not _is_missing(r.get("ml_lock_edge")) else -99,
            float(r.get("ou_lock_edge") or -99) if not _is_missing(r.get("ou_lock_edge")) else -99,
        )
        if has_lock:
            return (0, -best_edge)
        if has_pinnacle:
            return (1, -best_edge)   # near-miss games float to top of this tier
        return (2, 0)

    results.sort(key=_sort_key)
    return results


# ---------------------------------------------------------------------------
# FORMATTING
# ---------------------------------------------------------------------------

def print_card(results: list[dict], min_edge: float = 0.0) -> None:
    if not results:
        return

    import datetime as _dt
    date_str = results[0].get("game_time_et", "")
    try:
        day_label = _dt.date.today().strftime("%A, %B %-d, %Y")
    except ValueError:
        day_label = str(_dt.date.today())

    tier1 = [r for r in results if r.get("lock_tier") == 1]
    tier2 = [r for r in results if r.get("lock_tier") == 2]
    # A game is a "lock" only when tier is not None

    # Sort each tier by edge descending
    tier1 = sorted(tier1, key=lambda r: r.get("lock_edge") or 0, reverse=True)
    tier2 = sorted(tier2, key=lambda r: r.get("lock_edge") or 0, reverse=True)

    print("=" * 72)
    print("  WIZARD REPORT — MLB PICKS")
    print(f"  {day_label}")
    print("=" * 72)

    total_locks = len(tier1) + len(tier2)
    if total_locks == 0:
        print()
        print("  WIZARD REPORT: 0 Actionable Edges")
        print("  No game passed all three gates today.")
        print("  (Sanity / Odds Floor / CLV Edge)")
        print()
        if not min_edge:
            # Still show all games in compact form
            print("  --- ALL GAMES (no lock signal) ---\n")
            for r in results:
                _print_game_row_compact(r)
        return

    if tier1:
        print(f"\n  --- TIER 1: STRONG EDGE ({len(tier1)} lock{'s' if len(tier1)>1 else ''}) ---\n")
        for r in tier1:
            _print_lock_row(r)

    if tier2:
        print(f"\n  --- TIER 2: MEDIUM EDGE ({len(tier2)} lock{'s' if len(tier2)>1 else ''}) ---\n")
        for r in tier2:
            _print_lock_row(r)

    if not min_edge:
        no_lock = [r for r in results if r.get("lock_tier") is None]
        if no_lock:
            print(f"\n  --- NO LOCK ({len(no_lock)} games) ---\n")
            for r in no_lock:
                _print_game_row_compact(r)

    print()
    print(f"  Three-Part Lock: |P_model - P_pinnacle| <= 4%  |  "
          f"odds >= -225  |  edge >= 1.0%")
    print(f"  Kelly sizing: Tier 1 = quarter-Kelly  |  Tier 2 = eighth-Kelly  "
          f"(max ${MAX_BET_DOLLARS})")
    print()


def _print_lock_row(r: dict) -> None:
    """Print a full detail row for a game that passed the Three-Part Lock."""
    home, away = r["home_team"], r["away_team"]
    conf  = "" if r.get("lineup_confirmed") else " [PROJ]"
    tier  = r.get("lock_tier", "?")
    edge  = r.get("lock_edge", 0.0) or 0.0
    stake = r.get("lock_dollars", 0.0) or 0.0
    p_mdl = r.get("lock_p_model", 0.0) or 0.0
    p_tru = r.get("lock_p_true")
    p_ret = r.get("lock_retail_implied")
    rl_odds = r.get("rl_odds")
    ml_home = r.get("vegas_ml_home")

    flag_h = f" [{r['home_sp_flag']}]" if r.get("home_sp_flag","") not in ("NORMAL","UNKNOWN","") else ""
    flag_a = f" [{r['away_sp_flag']}]" if r.get("away_sp_flag","") not in ("NORMAL","UNKNOWN","") else ""

    rl_odds_str = f"  RL {int(rl_odds):+d}" if not _is_missing(rl_odds) else ""
    ml_str      = f"  ML {int(ml_home):+d}" if not _is_missing(ml_home) else ""

    print(f"  {away} @ {home}{conf}{ml_str}{rl_odds_str}")
    print(f"    HOME SP: {r.get('home_sp','?'):<28} xwOBA={r.get('home_sp_xwoba',0.0):.3f}{flag_h}")
    print(f"    AWAY SP: {r.get('away_sp','?'):<28} xwOBA={r.get('away_sp_xwoba',0.0):.3f}{flag_a}")

    p_true_str = f"{float(p_tru):.3f}" if not _is_missing(p_tru) else "N/A"
    p_ret_str  = f"{float(p_ret):.3f}" if not _is_missing(p_ret) else "N/A"
    print(f"    P_model={p_mdl:.3f}  P_pinnacle={p_true_str}  P_retail={p_ret_str}  edge={edge:+.1%}")
    print(f"    >>> TIER {tier} LOCK — BET HOME RL -1.5   stake=${stake:.2f}")
    print()


def _print_game_row_compact(r: dict) -> None:
    """Print a compact one-line summary for games with no lock signal."""
    home, away = r["home_team"], r["away_team"]
    bl   = r.get("blended_rl")
    ml_h = r.get("vegas_ml_home")
    rl_o = r.get("rl_odds")

    # TBD row
    if bl is None or _is_missing(bl):
        vp = r.get("vegas_pick", "—")
        print(f"  {away} @ {home}  [TBD starters]  Vegas pick: {vp}")
        return

    ml_str = f"ML {int(ml_h):+d}" if not _is_missing(ml_h) else "no ML"
    rl_str = f"RL {int(rl_o):+d}" if not _is_missing(rl_o) else ""
    # Show which gate failed (first failure wins)
    if _is_missing(r.get("lock_p_true")):
        fail = "no P_true"
    elif not r.get("lock_sanity_pass"):
        fail = "sanity"
    elif not r.get("lock_odds_floor_pass"):
        fail = "odds-floor"
    elif not r.get("lock_edge_pass"):
        fail = f"edge({r.get('lock_edge',0.0):+.1%})" if not _is_missing(r.get("lock_edge")) else "edge"
    else:
        fail = "?"

    print(f"  {away} @ {home}  blend={float(bl):.3f}  {ml_str}  {rl_str}  (no lock: {fail})")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def _edge_pct(blended_rl: float, signal: str) -> str:
    """Return edge vs -110 breakeven (52.4%) as a string like +18%."""
    breakeven = 100 / 210  # 52.38%
    if "HOME" in signal:
        edge = blended_rl - breakeven
    else:
        edge = (1 - blended_rl) - breakeven
    return f"{edge:+.0%}"


def _why(r: dict) -> str:
    """One-line reason for the bet."""
    parts = []
    home_x = r.get("home_sp_xwoba") or 0
    away_x = r.get("away_sp_xwoba") or 0
    signal = r["rl_signal"]

    # Flag the favored pitcher
    if "AWAY" in signal:
        sp_name = r["away_sp"].title()
        sp_x = away_x
        opp_name = r["home_sp"].title()
        opp_x = home_x
    else:
        sp_name = r["home_sp"].title()
        sp_x = home_x
        opp_name = r["away_sp"].title()
        opp_x = away_x

    if sp_x < 0.280:
        parts.append(f"{sp_name} elite (xwOBA {sp_x:.3f})")
    elif sp_x < 0.305:
        parts.append(f"{sp_name} strong (xwOBA {sp_x:.3f})")

    if opp_x > 0.360:
        parts.append(f"{opp_name} struggling (xwOBA {opp_x:.3f})")
    elif opp_x > 0.330:
        parts.append(f"{opp_name} below avg (xwOBA {opp_x:.3f})")

    # Velocity flags
    if "AWAY" in signal and r.get("home_sp_flag") in ("VOLATILE",):
        parts.append(f"{r['home_sp'].title()} velocity declining [VOLATILE]")
    if "HOME" in signal and r.get("away_sp_flag") in ("VOLATILE",):
        parts.append(f"{r['away_sp'].title()} velocity declining [VOLATILE]")
    if "AWAY" in signal and r.get("away_sp_flag") == "GAINER":
        parts.append(f"{r['away_sp'].title()} velo trending up [GAINER]")

    # Temperature
    temp = r.get("temp_f", 72)
    if temp and temp > 80:
        parts.append(f"warm weather ({temp:.0f}F)")
    elif temp and temp < 50:
        parts.append(f"cold weather ({temp:.0f}F, suppresses scoring)")

    return " | ".join(parts) if parts else "model edge on run differential"


def _why_lock(r: dict) -> str:
    """One-line reason for a Three-Part Lock bet (home RL -1.5 direction)."""
    parts = []
    home_x = r.get("home_sp_xwoba") or 0
    away_x = r.get("away_sp_xwoba") or 0

    # Pitcher quality
    if home_x < 0.280:
        parts.append(f"{r.get('home_sp','?').title()} elite (xwOBA {home_x:.3f})")
    elif home_x < 0.305:
        parts.append(f"{r.get('home_sp','?').title()} strong (xwOBA {home_x:.3f})")

    if away_x > 0.360:
        parts.append(f"{r.get('away_sp','?').title()} struggling (xwOBA {away_x:.3f})")
    elif away_x > 0.330:
        parts.append(f"{r.get('away_sp','?').title()} below avg (xwOBA {away_x:.3f})")

    # Velocity flags
    if r.get("away_sp_flag") in ("VOLATILE",):
        parts.append(f"{r.get('away_sp','?').title()} velocity declining")
    if r.get("home_sp_flag") == "GAINER":
        parts.append(f"{r.get('home_sp','?').title()} velo trending up")

    edge = r.get("lock_edge", 0.0) or 0.0
    parts.append(f"CLV edge {edge:+.1%} vs retail")

    return " | ".join(parts) if parts else "model edge on run differential"


def _build_email_body(results: list[dict], date_str: str) -> str:
    """Plain-text email fallback: all locks (ML/RL/O/U) + full game list."""
    import datetime as dt
    try:
        day_label = dt.datetime.strptime(date_str, "%Y-%m-%d").strftime("%A, %B %-d, %Y")
    except Exception:
        day_label = date_str

    lines = []
    lines.append("THE WIZARD REPORT — MLB PICKS")
    lines.append(day_label)
    lines.append("=" * 48)

    # Collect all locks across all three markets
    all_locks = []
    for r in results:
        home, away = r["home_team"], r["away_team"]
        for mkt, tier_key, edge_key, dol_key, pmod_key, ptrue_key, pret_key in [
            ("RL",       "lock_tier",      "lock_edge",      "lock_dollars",      "lock_p_model",      "lock_p_true",          "lock_retail_implied"),
            ("RL_AWAY",  "away_lock_tier", "away_lock_edge", "away_lock_dollars", "away_lock_p_model", "away_lock_p_true",     "away_lock_retail_implied"),
            ("ML",       "ml_lock_tier",   "ml_lock_edge",   "ml_lock_dollars",   "ml_lock_p_model",   "ml_lock_p_true",       "ml_lock_retail_implied"),
            ("O/U",      "ou_lock_tier",   "ou_lock_edge",   "ou_lock_dollars",   "ou_p_model",        "ou_p_true",            "ou_p_retail"),
        ]:
            tier = r.get(tier_key)
            if tier is not None:
                ou_dir = r.get("ou_direction", "OVER") if mkt == "O/U" else ""
                line_label = (f"O/U {ou_dir} {r.get('ou_posted_line','')} " if mkt == "O/U"
                              else f"RL {away} +1.5 " if mkt == "RL_AWAY"
                              else f"RL {home} -1.5 " if mkt == "RL" else "ML WIN ")
                all_locks.append({
                    "game": f"{away} @ {home}", "mkt": mkt, "tier": tier,
                    "label": line_label.strip(),
                    "edge":  r.get(edge_key), "dollars": r.get(dol_key),
                    "p_model": r.get(pmod_key), "p_true": r.get(ptrue_key),
                    "p_ret": r.get(pret_key),
                })

    all_locks.sort(key=lambda x: (x["tier"], -(x["edge"] or 0)))

    if not all_locks:
        lines.append("")
        lines.append("0 Actionable Edges — No game passed all three gates today.")
        lines.append("(Sanity / Odds Floor / CLV Edge >= 1%)")
    else:
        t1 = [x for x in all_locks if x["tier"] == 1]
        t2 = [x for x in all_locks if x["tier"] == 2]
        lines.append(f"\nTODAY'S LOCKS  ({len(t1)} Tier-1  |  {len(t2)} Tier-2)")

        def fmt_lock(lk, n):
            e  = f"{float(lk['edge']):+.1%}" if not _is_missing(lk.get("edge")) else "—"
            pm = f"{float(lk['p_model']):.3f}" if not _is_missing(lk.get("p_model")) else "—"
            pt = f"{float(lk['p_true']):.3f}"  if not _is_missing(lk.get("p_true"))  else "—"
            pr = f"{float(lk['p_ret']):.3f}"   if not _is_missing(lk.get("p_ret"))   else "—"
            dol = f"  stake=${lk['dollars']:.0f}" if lk.get("dollars") else ""
            return (f"\n[{n}] TIER {lk['tier']} — {lk['game']}  {lk['mkt']} {lk['label']}{dol}\n"
                    f"    Model={pm}  Pinnacle={pt}  Retail={pr}  edge={e}")

        if t1:
            lines.append("\nTIER 1 — STRONG EDGE")
            lines.append("-" * 40)
            for i, lk in enumerate(t1, 1):
                lines.append(fmt_lock(lk, i))
        if t2:
            lines.append("\nTIER 2 — MEDIUM EDGE")
            lines.append("-" * 40)
            for i, lk in enumerate(t2, len(t1) + 1):
                lines.append(fmt_lock(lk, i))

    # Full game list
    lines.append("\n" + "-" * 48)
    lines.append(f"ALL GAMES ({len(results)})")
    lines.append("-" * 48)
    for r in results:
        home, away = r["home_team"], r["away_team"]
        gtime = r.get("game_time_et", "")
        ml_h = r.get("vegas_ml_home")
        rl_o = r.get("rl_odds")
        ml_str = f"ML {int(float(ml_h)):+d}" if not _is_missing(ml_h) else ""
        rl_str = f"RL {int(float(rl_o)):+d}" if not _is_missing(rl_o) else ""
        odds = "  ".join(x for x in [ml_str, rl_str] if x)
        hsp = str(r.get("home_sp","TBD")).title()
        asp = str(r.get("away_sp","TBD")).title()
        hx  = f"{r.get('home_sp_xwoba',0):.3f}" if not _is_missing(r.get("home_sp_xwoba")) else "—"
        ax  = f"{r.get('away_sp_xwoba',0):.3f}" if not _is_missing(r.get("away_sp_xwoba")) else "—"
        tot = r.get("mc_total") or r.get("blended_total")
        tot_str = f"  pred {float(tot):.1f}" if not _is_missing(tot) else ""
        line = r.get("ou_posted_line") or r.get("vegas_total")
        line_str = f" vs {float(line):.1f}" if not _is_missing(line) else ""
        lines.append(f"\n  {away} @ {home}  {gtime}  {odds}")
        lines.append(f"    {home} SP: {hsp} ({hx})  |  {away} SP: {asp} ({ax})")
        lines.append(f"    Total:{tot_str}{line_str}")

    # Parlay suggestions
    parlay_combos_email = _find_parlay_combos(results)
    lines.append("\n" + "=" * 48)
    lines.append(f"CORRELATED PARLAY SUGGESTIONS (+{_PARLAY_TARGET_MIN}–+{_PARLAY_TARGET_MAX})")
    lines.append("=" * 48)
    if not parlay_combos_email:
        lines.append("  No combos with positive model edge found in target range today.")
    else:
        for i, c in enumerate(parlay_combos_email, 1):
            sgp_tag = " [SGP]" if c["is_sgp"] else ""
            lines.append(
                f"\n[{i}] {c['n_legs']}-Leg Parlay{sgp_tag}  "
                f"Combined: {c['combined_amt']:+d}  "
                f"Model: {c['model_joint_prob']:.1%}  "
                f"Mkt: {c['market_joint_prob']:.1%}  "
                f"Edge: {c['combo_edge']:+.1%}"
            )
            for lg in c["legs"]:
                lines.append(
                    f"    • {lg['game_key']:20s}  {lg['label']:<20s}  "
                    f"{lg['american_odds']:+5d}  model={lg['model_prob']:.0%}  edge={lg['edge']:+.1%}"
                )
        lines.append(
            "\n  SGP = Same-Game Parlay. Sportsbook applies own correlation discount to payout."
        )

    lines.append("\n" + "=" * 48)
    lines.append("Three-Part Lock: Sanity / Odds Floor / CLV Edge >= 1%")
    lines.append("=" * 48)

    return "\n".join(lines)



# legacy — keep for console output formatting
def _build_email_body_old(results: list[dict], date_str: str) -> str:
    lines = []
    lines.append(f"MLB Run Line Card — {date_str}")
    lines.append("=" * 60)

    strong = [r for r in results if "**" in r["rl_signal"]]
    lean   = [r for r in results if "*"  in r["rl_signal"] and "**" not in r["rl_signal"]]
    watch  = [r for r in results if not r["rl_signal"]]

    if strong:
        lines.append(f"\n** STRONG SIGNALS ({len(strong)} games)")
        lines.append("-" * 40)
        for r in strong:
            lines.append(fmt(r))
            lines.append("")

    if lean:
        lines.append(f"\n*  LEAN SIGNALS ({len(lean)} games)")
        lines.append("-" * 40)
        for r in lean:
            lines.append(fmt(r))
            lines.append("")

    if watch:
        lines.append(f"\n-- NO SIGNAL ({len(watch)} games)")
        lines.append("-" * 40)
        for r in watch:
            lines.append(fmt(r))
            lines.append("")

    lines.append(f"Thresholds: ** HOME>=0.58 | * HOME>=0.54 | ** AWAY<=0.34 | * AWAY<=0.40")
    lines.append(f"Blend: 60% Monte Carlo + 40% XGBoost")
    return "\n".join(lines)


def send_card_email(results: list[dict], date_str: str) -> None:
    """
    Send the daily card as a multipart email via Gmail.
    - HTML body: full daily_card.html (renders in Gmail / Outlook / Apple Mail)
    - Plain-text fallback: Tier 1 / Tier 2 lock summary
    """
    import os
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    gmail_from = os.getenv("GMAIL_FROM", "garcia.dan24@gmail.com")
    gmail_pass = os.getenv("GMAIL_APP_PASSWORD", "")
    recipients = os.getenv("EMAIL_RECIPIENTS", gmail_from).split(",")

    if not gmail_pass:
        print("  [WARN] GMAIL_APP_PASSWORD not set — skipping email")
        return

    # Count locks across all three markets
    all_locks = [r for r in results
                 if any(r.get(k) is not None
                        for k in ("lock_tier", "ml_lock_tier", "ou_lock_tier"))]
    n_locks = len(all_locks)
    t1_rl = sum(1 for r in results if r.get("lock_tier") == 1)
    t2_rl = sum(1 for r in results if r.get("lock_tier") == 2)

    if n_locks == 0:
        subject = f"Wizard MLB {date_str} — 0 locks today"
    elif n_locks == 1:
        subject = f"Wizard MLB {date_str} — 1 lock"
    else:
        subject = f"Wizard MLB {date_str} — {n_locks} locks"

    # ── Plain-text fallback ──────────────────────────────────────────────────
    plain_body = _build_email_body(results, date_str)

    # ── HTML body: read the generated daily_card.html ────────────────────────
    html_path = Path("daily_card.html")
    if html_path.exists():
        html_body = html_path.read_text(encoding="utf-8")
    else:
        # Fallback: generate inline if file missing
        write_html_card(results, date_str)
        html_body = html_path.read_text(encoding="utf-8") if html_path.exists() else f"<pre>{plain_body}</pre>"

    # ── Build multipart/alternative message ─────────────────────────────────
    msg = MIMEMultipart("alternative")
    msg["From"]    = gmail_from
    msg["To"]      = ", ".join(recipients)
    msg["Subject"] = subject
    # Plain text first (fallback), HTML second (preferred)
    msg.attach(MIMEText(plain_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body,  "html",  "utf-8"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(gmail_from, gmail_pass)
            s.sendmail(gmail_from, recipients, msg.as_string())
        print(f"  Email sent -> {', '.join(recipients)}")
    except Exception as e:
        print(f"  [ERROR] Email failed: {e}")


def write_html_card(results: list[dict], date_str: str) -> None:
    """
    Write a self-contained daily_card.html — open in any browser after triage.

    Layout (card per game, sorted by actionability):
      • Header: date + lock summary
      • Per-game cards: predicted score · starters + K props · ML / RL / O/U markets
    """
    import datetime as _dt

    try:
        day_label = _dt.datetime.strptime(date_str, "%Y-%m-%d").strftime("%A, %B %-d, %Y")
    except Exception:
        day_label = date_str

    # Count locks across all three markets
    all_locks = [r for r in results
                 if any(r.get(k) is not None
                        for k in ("lock_tier", "ml_lock_tier", "ou_lock_tier"))]
    total_locks = len(all_locks)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _prob(v, decimals=3):
        if _is_missing(v): return "—"
        return f"{float(v):.{decimals}f}"

    def _pct(v, sign=True):
        if _is_missing(v): return "—"
        return f"{float(v):+.1%}" if sign else f"{float(v):.1%}"

    def _edge_cls(e):
        if _is_missing(e): return "val-dim"
        e = float(e)
        if e >= 0.01:  return "val-green"
        if e >= -0.005: return "val-yellow"
        return "val-red"

    def _gate_badge(p_true, sanity_pass, odds_floor_pass, edge_pass, edge, lock_tier,
                    lock_dollars=None, prefix=""):
        if lock_tier in (1, 2):
            cls  = "badge-green" if lock_tier == 1 else "badge-amber"
            lbl  = f"TIER {lock_tier}"
            stk  = f' <span class="stake">${lock_dollars:.0f}</span>' if lock_dollars else ""
            return f'<span class="badge {cls}">{prefix}{lbl}</span>{stk}'
        if _is_missing(p_true):
            return '<span class="badge badge-gray">No Pinnacle</span>'
        if not sanity_pass:
            gap = abs((float(p_true) if not _is_missing(p_true) else 0))
            return f'<span class="badge badge-red">Sanity fail</span>'
        if not odds_floor_pass:
            return '<span class="badge badge-red">Odds Floor</span>'
        if _is_missing(edge):
            return '<span class="badge badge-gray">—</span>'
        e = float(edge)
        if e >= -0.005:   cls = "badge-yellow"
        elif e >= -0.03:  cls = "badge-orange"
        else:             cls = "badge-gray"
        return f'<span class="badge {cls}">Edge {e:+.1%}</span>'

    def _mkt_block(label, p_model, p_pin, p_retail, edge,
                   lock_tier, lock_dollars, p_true, sanity_pass,
                   odds_floor_pass, edge_pass, extra_rows=""):
        """Render one market widget (ML, RL, or O/U)."""
        edge_c   = _edge_cls(edge)
        edge_str = _pct(edge) if not _is_missing(edge) else "—"
        badge    = _gate_badge(p_true, sanity_pass, odds_floor_pass, edge_pass,
                               edge, lock_tier, lock_dollars)
        return f"""<div class="mkt">
  <div class="mkt-label">{label}</div>
  <table class="mkt-tbl">
    <tr><td class="ml">Model</td><td class="mv bright">{_prob(p_model)}</td></tr>
    <tr><td class="ml">Pinnacle</td><td class="mv pin">{_prob(p_pin)}</td></tr>
    <tr><td class="ml">Retail</td><td class="mv ret">{_prob(p_retail)}</td></tr>
    {extra_rows}
    <tr><td class="ml">Edge</td><td class="mv {edge_c}">{edge_str}</td></tr>
  </table>
  <div class="mkt-badge">{badge}</div>
</div>"""

    def _ou_block(r):
        """Render the O/U market widget."""
        ou_dir    = r.get("ou_direction", "OVER")
        line      = r.get("ou_posted_line")
        mc_over   = r.get("mc_over_prob")
        p_pin_over= r.get("p_true_over")
        p_pin_und = r.get("p_true_under")
        p_ret_over= r.get("retail_implied_over")
        p_ret_und = r.get("retail_implied_under")
        total_mean= r.get("mc_total") or r.get("blended_total")
        total_lo  = r.get("mc_total_lo")
        total_hi  = r.get("mc_total_hi")
        edge      = r.get("ou_lock_edge")
        lock_tier = r.get("ou_lock_tier")
        lock_dol  = r.get("ou_lock_dollars")
        p_true    = r.get("ou_p_true")

        line_str = f"{float(line):.1f}" if not _is_missing(line) else "—"
        band_str = (f'<span class="band">({total_lo}–{total_hi})</span>'
                    if not _is_missing(total_lo) and not _is_missing(total_hi) else "")
        pred_str = f"{float(total_mean):.1f} {band_str}" if not _is_missing(total_mean) else "—"

        if ou_dir == "OVER":
            p_model_show = mc_over
            p_pin_show   = p_pin_over
            p_ret_show   = p_ret_over
        else:
            p_model_show = (1 - float(mc_over)) if not _is_missing(mc_over) else None
            p_pin_show   = p_pin_und
            p_ret_show   = p_ret_und

        edge_c   = _edge_cls(edge)
        edge_str = _pct(edge) if not _is_missing(edge) else "—"
        badge    = _gate_badge(p_true,
                               r.get("ou_lock_sanity_pass", False),
                               True,   # no odds-floor gate for O/U
                               r.get("ou_lock_edge_pass", False),
                               edge, lock_tier, lock_dol)

        # Quantile band row (only shown when quantile model produced bounds)
        _q_floor   = r.get("total_floor_10th")
        _q_ceiling = r.get("total_ceiling_90th")
        _q_spread  = r.get("total_variance_spread")
        if not _is_missing(_q_floor) and not _is_missing(_q_ceiling):
            _sp = float(_q_spread) if not _is_missing(_q_spread) else 0
            _q_cls = ("qband-tight" if _sp < 4.0
                      else "qband-wide" if _sp > 6.0
                      else "qband-mid")
            _q_row = (f'<tr><td class="ml">Q10–Q90</td>'
                      f'<td class="mv"><span class="{_q_cls}">'
                      f'{float(_q_floor):.1f}–{float(_q_ceiling):.1f}'
                      f' <span class="spread-pill">±{_sp:.1f}</span>'
                      f'</span></td></tr>')
        else:
            _q_row = ""

        return f"""<div class="mkt">
  <div class="mkt-label">O/U · LINE {line_str}</div>
  <table class="mkt-tbl">
    <tr><td class="ml">Pred Total</td><td class="mv bright">{pred_str}</td></tr>
    {_q_row}
    <tr><td class="ml">P({ou_dir})</td><td class="mv bright">{_prob(p_model_show)}</td></tr>
    <tr><td class="ml">Pin P({ou_dir})</td><td class="mv pin">{_prob(p_pin_show)}</td></tr>
    <tr><td class="ml">Ret P({ou_dir})</td><td class="mv ret">{_prob(p_ret_show)}</td></tr>
    <tr><td class="ml">Edge</td><td class="mv {edge_c}">{edge_str}</td></tr>
  </table>
  <div class="mkt-badge">{badge}</div>
</div>"""

    def _k_badge(model_p, implied_p, edge, line):
        if _is_missing(model_p) or _is_missing(line):
            return ""
        line_str = f"{float(line):.1f}"
        mp_str   = f"{float(model_p):.0%}"
        e_str    = _pct(edge) if not _is_missing(edge) else "—"
        e_cls    = _edge_cls(edge)
        return (f'<span class="k-line">K {line_str}</span>'
                f'&nbsp;<span class="k-pover">P(O)&nbsp;{mp_str}</span>'
                f'&nbsp;<span class="{e_cls} k-edge">{e_str}</span>')

    def _sp_cell(team, name, xwoba, flag, k_line, k_model, k_impl, k_edge):
        name_str  = str(name or "TBD").title()
        flag_html = (f' <span class="sp-flag">{flag}</span>'
                     if flag and str(flag) not in ("NORMAL","UNKNOWN","") else "")
        xwoba_str = f"{float(xwoba):.3f}" if not _is_missing(xwoba) else "—"
        k_html    = _k_badge(k_model, k_impl, k_edge, k_line)
        return (f'<span class="sp-team">{team}</span> '
                f'<span class="sp-name">{name_str}</span>{flag_html} '
                f'<span class="xwoba">{xwoba_str}</span>'
                + (f'<br><span class="k-props">{k_html}</span>' if k_html else ""))

    def _score_row(r):
        home  = r["home_team"]
        away  = r["away_team"]
        hm    = r.get("home_runs_mean")
        am    = r.get("away_runs_mean")
        hlo   = r.get("home_runs_lo")
        hhi   = r.get("home_runs_hi")
        alo   = r.get("away_runs_lo")
        ahi   = r.get("away_runs_hi")
        total = r.get("mc_total") or r.get("blended_total")
        tlo   = r.get("mc_total_lo")
        thi   = r.get("mc_total_hi")
        line  = r.get("ou_posted_line") or r.get("vegas_total")

        def fmt_team(mean, lo, hi):
            if _is_missing(mean): return "—"
            band = (f'<span class="band">({lo}–{hi})</span>'
                    if not _is_missing(lo) and not _is_missing(hi) else "")
            return f'<span class="score-val">{float(mean):.1f}</span> {band}'

        home_s = fmt_team(hm, hlo, hhi)
        away_s = fmt_team(am, alo, ahi)
        tot_s  = ""
        if not _is_missing(total):
            tband = (f'<span class="band">({tlo}–{thi})</span>'
                     if not _is_missing(tlo) and not _is_missing(thi) else "")
            tot_s = f'<span class="score-val">{float(total):.1f}</span> {tband}'
            if not _is_missing(line):
                tot_s += f' <span class="ou-line">vs {float(line):.1f}</span>'

        return (f'<span class="score-lbl">{home}</span>&nbsp;{home_s}'
                f'&nbsp;&nbsp;<span class="score-sep">·</span>&nbsp;&nbsp;'
                f'<span class="score-lbl">{away}</span>&nbsp;{away_s}'
                + (f'&nbsp;&nbsp;<span class="score-sep">|</span>&nbsp;&nbsp;'
                   f'<span class="score-lbl">Total</span>&nbsp;{tot_s}' if tot_s else ""))

    # ── game card HTML ────────────────────────────────────────────────────────
    def _game_card(r):
        home   = r["home_team"]
        away   = r["away_team"]
        gtime  = r.get("game_time_et", "")
        conf   = r.get("lineup_confirmed", False)
        ml_h   = r.get("vegas_ml_home")
        ml_a   = r.get("vegas_ml_away")
        rl_o   = r.get("rl_odds")

        # Determine best lock across all markets (home RL, away RL, ML, O/U)
        best_tier = None
        for t in (r.get("lock_tier"), r.get("away_lock_tier"),
                  r.get("ml_lock_tier"), r.get("ou_lock_tier")):
            if t is not None:
                best_tier = min(t, best_tier) if best_tier else t

        card_cls = ""
        if best_tier == 1:    card_cls = "card-t1"
        elif best_tier == 2:  card_cls = "card-t2"
        elif not _is_missing(r.get("lock_p_true")) or not _is_missing(r.get("away_lock_p_true")):
            # near-miss: any market passed sanity and got close on edge
            edges = [r.get("lock_edge"), r.get("away_lock_edge"),
                     r.get("ml_lock_edge"), r.get("ou_lock_edge")]
            best_e = max((float(e) for e in edges if not _is_missing(e)), default=-99)
            if best_e >= -0.02:
                card_cls = "card-near"
        else:
            card_cls = "card-npin"

        odds_parts = []
        if not _is_missing(ml_h): odds_parts.append(f"ML {int(float(ml_h)):+d}")
        if not _is_missing(ml_a): odds_parts.append(f"/ {int(float(ml_a)):+d}")
        if not _is_missing(rl_o): odds_parts.append(f"· RL {int(float(rl_o)):+d}")
        odds_str  = " ".join(odds_parts) if odds_parts else ""
        conf_str  = "✓ confirmed" if conf else "~ projected"
        gtime_str = f" · {gtime}" if gtime else ""

        # Best overall badge for header (includes both RL sides)
        lock_tiers = {
            f"RL {home} -1.5": (r.get("lock_tier"),      r.get("lock_dollars")),
            f"RL {away} +1.5": (r.get("away_lock_tier"), r.get("away_lock_dollars")),
            "ML":              (r.get("ml_lock_tier"),    r.get("ml_lock_dollars")),
            "O/U":             (r.get("ou_lock_tier"),    r.get("ou_lock_dollars")),
        }
        header_badge = ""
        for mkt, (tier, dol) in lock_tiers.items():
            if tier is not None:
                cls = "badge-green" if tier == 1 else "badge-amber"
                dol_str = f" ${dol:.0f}" if dol else ""
                header_badge += f'<span class="badge {cls}">{mkt} TIER {tier}{dol_str}</span> '

        # Shadow ensemble range row for RL block
        _ens_min = r.get("ensemble_min")
        _ens_max = r.get("ensemble_max")
        _spread  = r.get("model_spread")
        if not _is_missing(_ens_min) and not _is_missing(_ens_max) and not _is_missing(_spread):
            _sp = float(_spread)
            _spread_cls = ("shadow-agree" if _sp < 0.10
                           else "shadow-warn" if _sp < 0.20
                           else "shadow-split")
            _shadow_row = (
                f'<tr><td class="ml">Shadow</td>'
                f'<td class="mv"><span class="{_spread_cls}">'
                f'{float(_ens_min):.0%}–{float(_ens_max):.0%}'
                f' <span class="spread-pill">±{_sp:.0%}</span>'
                f'</span></td></tr>'
            )
        else:
            _shadow_row = ""

        # RL market
        rl_block = _mkt_block(
            label        = f"RL · {home} -1.5",
            p_model      = r.get("lock_p_model"),
            p_pin        = r.get("lock_p_true"),
            p_retail     = r.get("lock_retail_implied"),
            edge         = r.get("lock_edge"),
            lock_tier    = r.get("lock_tier"),
            lock_dollars = r.get("lock_dollars"),
            p_true       = r.get("lock_p_true"),
            sanity_pass  = r.get("lock_sanity_pass", False),
            odds_floor_pass = r.get("lock_odds_floor_pass", True),
            edge_pass    = r.get("lock_edge_pass", False),
            extra_rows   = _shadow_row,
        )

        # RL away market (+1.5)
        rl_away_block = _mkt_block(
            label        = f"RL · {away} +1.5",
            p_model      = r.get("away_lock_p_model"),
            p_pin        = r.get("away_lock_p_true"),
            p_retail     = r.get("away_lock_retail_implied"),
            edge         = r.get("away_lock_edge"),
            lock_tier    = r.get("away_lock_tier"),
            lock_dollars = r.get("away_lock_dollars"),
            p_true       = r.get("away_lock_p_true"),
            sanity_pass  = r.get("away_lock_sanity_pass", False),
            odds_floor_pass = r.get("away_lock_odds_floor_pass", True),
            edge_pass    = r.get("away_lock_edge_pass", False),
        )

        # ML market
        ml_label = (f"ML · {home} WIN" if not _is_missing(ml_h) and float(ml_h) < 0
                    else f"ML · {away} WIN" if not _is_missing(ml_a) and float(ml_a) < 0
                    else f"ML · {home} WIN")
        ml_block = _mkt_block(
            label        = ml_label,
            p_model      = r.get("ml_lock_p_model"),
            p_pin        = r.get("ml_lock_p_true"),
            p_retail     = r.get("ml_lock_retail_implied"),
            edge         = r.get("ml_lock_edge"),
            lock_tier    = r.get("ml_lock_tier"),
            lock_dollars = r.get("ml_lock_dollars"),
            p_true       = r.get("ml_lock_p_true"),
            sanity_pass  = r.get("ml_lock_sanity_pass", False),
            odds_floor_pass = True,
            edge_pass    = r.get("ml_lock_edge_pass", False),
        )

        # O/U market
        ou_block = _ou_block(r)

        # Starters
        hk = (r.get("home_k_line"), r.get("home_k_model_over"),
              r.get("home_k_implied_over"), r.get("home_k_edge"))
        ak = (r.get("away_k_line"), r.get("away_k_model_over"),
              r.get("away_k_implied_over"), r.get("away_k_edge"))
        home_sp_cell = _sp_cell(home, r.get("home_sp"), r.get("home_sp_xwoba"),
                                r.get("home_sp_flag",""), *hk)
        away_sp_cell = _sp_cell(away, r.get("away_sp"), r.get("away_sp_xwoba"),
                                r.get("away_sp_flag",""), *ak)

        tbd = r.get("home_sp","TBD") == "TBD" or r.get("away_sp","TBD") == "TBD"
        tbd_notice = '<div class="tbd-notice">⚠ Starters not yet confirmed</div>' if tbd else ""

        score_html = _score_row(r)

        return f"""<div class="gc {card_cls}">
  <div class="gc-head">
    <div class="gc-head-left">
      <span class="gc-teams">{away} @ {home}</span>
      <span class="gc-time">{gtime_str.strip(" ·")}</span>
      <span class="gc-conf">{conf_str}</span>
    </div>
    <div class="gc-head-right">
      <span class="gc-odds">{odds_str}</span>
      {header_badge}
    </div>
  </div>
  <div class="gc-score">{score_html}</div>
  <div class="gc-body">
    <div class="gc-starters">
      <div class="gc-sp">{home_sp_cell}</div>
      <div class="gc-sp">{away_sp_cell}</div>
      {tbd_notice}
    </div>
    <div class="gc-markets">
      {ml_block}
      {rl_block}
      {rl_away_block}
      {ou_block}
    </div>
  </div>
</div>"""

    # ── assemble HTML ─────────────────────────────────────────────────────────
    gen_time = _dt.datetime.now().strftime("%H:%M")

    if total_locks == 0:
        header_sub = "0 Actionable Edges — No game passed all three gates today"
        header_cls = "hdr-none"
    elif total_locks == 1:
        header_sub = "1 Lock"
        header_cls = "hdr-lock"
    else:
        header_sub = f"{total_locks} Locks"
        header_cls = "hdr-lock"

    cards_html = "\n".join(_game_card(r) for r in results)

    # ── Parlay suggestions section ────────────────────────────────────────────
    def _parlay_html(combos: list[dict]) -> str:
        if not combos:
            return (
                '<div class="pl-section">'
                '<div class="pl-title">🎯 Correlated Parlay Suggestions</div>'
                '<div class="pl-note">No combinations with positive combined edge '
                f'found in the +{_PARLAY_TARGET_MIN}–+{_PARLAY_TARGET_MAX} range today.</div>'
                '</div>'
            )

        combo_blocks = []
        for c in combos:
            sgp_tag = '<span class="pl-sgp">SGP</span>' if c["is_sgp"] else ""
            edge_cls = "edge-pos" if c["combo_edge"] >= 0.03 else "edge-med"
            head = (
                f'<div class="pl-combo-head">'
                f'<span class="pl-n">{c["n_legs"]}-LEG PARLAY</span> {sgp_tag}'
                f'<span class="pl-price">{c["combined_amt"]:+d}</span>'
                f'<span class="pl-stats">'
                f'Model: {c["model_joint_prob"]:.1%} &nbsp;·&nbsp; '
                f'Mkt: {c["market_joint_prob"]:.1%} &nbsp;·&nbsp; '
                f'Edge: <span class="{edge_cls}">{c["combo_edge"]:+.1%}</span>'
                f'</span></div>'
            )

            rows = ""
            for lg in c["legs"]:
                ec  = "edge-pos" if lg["edge"] >= 0.02 else "edge-med" if lg["edge"] >= 0 else "edge-neg"
                # Check if this leg is in a same-game correlated pair
                corr_note = ""
                if c["is_sgp"]:
                    for gk, lt1, lt2 in c["same_game_pairs"]:
                        if lg["game_key"] == gk and lg["leg_type"] in (lt1, lt2):
                            corr_note = '<span class="pl-corr"> corr</span>'
                            break
                rows += (
                    f"<tr>"
                    f'<td class="pl-game">{lg["game_key"]}</td>'
                    f'<td class="pl-leg">{lg["label"]}{corr_note}</td>'
                    f'<td class="pl-odds">{lg["american_odds"]:+d}</td>'
                    f'<td class="pl-prob">{lg["model_prob"]:.0%}</td>'
                    f'<td class="{ec}">{lg["edge"]:+.1%}</td>'
                    f"</tr>"
                )

            tbl = (
                '<table class="pl-tbl">'
                '<tr><th>Game</th><th>Leg</th><th>Odds</th>'
                '<th>Model P</th><th>Edge</th></tr>'
                f"{rows}</table>"
            )
            combo_blocks.append(f'<div class="pl-combo">{head}{tbl}</div>')

        inner = "\n".join(combo_blocks)
        return (
            '<div class="pl-section">'
            '<div class="pl-title">🎯 Correlated Parlay Suggestions</div>'
            f'<div class="pl-note">Top {len(combos)} combo(s) with positive combined model edge '
            f'in the +{_PARLAY_TARGET_MIN}–+{_PARLAY_TARGET_MAX} range. '
            'SGP = Same-Game Parlay — sportsbook applies its own correlation haircut to the payout. '
            'Market odds shown are the raw multiplication; actual SGP payout will be lower.</div>'
            f'<div class="pl-combos">{inner}</div>'
            '</div>'
        )

    parlay_combos = _find_parlay_combos(results)
    parlay_section = _parlay_html(parlay_combos)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wizard Report — {date_str}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
  background: #0d1117; color: #c9d1d9; padding: 16px; font-size: 13px;
}}

/* ── Header ─────────────────────────────────────── */
.hdr {{
  background: #161b22; border: 1px solid #21262d;
  border-left: 4px solid #388bfd;
  border-radius: 8px; padding: 18px 22px; margin-bottom: 16px;
}}
.hdr h1 {{ font-size: 20px; font-weight: 800; color: #ffffff; letter-spacing: .5px; }}
.hdr .date {{ color: #8b949e; font-size: 12px; margin-top: 2px; }}
.hdr .sub  {{ font-size: 15px; font-weight: 700; margin-top: 8px; }}
.hdr-none .sub {{ color: #6e7681; }}
.hdr-lock .sub {{ color: #3fb950; }}
.hdr .meta {{ color: #6e7681; font-size: 11px; margin-top: 6px; }}

/* ── Game cards ──────────────────────────────────── */
.gc {{
  background: #161b22; border: 1px solid #21262d;
  border-radius: 8px; margin-bottom: 12px; overflow: hidden;
}}
.card-t1   {{ border-left: 3px solid #3fb950; }}
.card-t2   {{ border-left: 3px solid #d29922; }}
.card-near {{ border-left: 3px solid #388bfd; }}
.card-npin {{ border-left: 3px solid #30363d; }}

/* Card header */
.gc-head {{
  display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 8px;
  padding: 10px 16px; background: #1c2128; border-bottom: 1px solid #21262d;
}}
.gc-head-left  {{ display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }}
.gc-head-right {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
.gc-teams {{ font-size: 16px; font-weight: 800; color: #ffffff; }}
.gc-time  {{ font-size: 11px; color: #8b949e; }}
.gc-conf  {{ font-size: 10px; color: #6e7681; }}
.gc-odds  {{ font-size: 12px; color: #8b949e; }}

/* Score prediction row */
.gc-score {{
  padding: 6px 16px; background: #0d1117;
  font-size: 12px; color: #8b949e;
  border-bottom: 1px solid #21262d;
}}
.score-val {{ color: #ffffff; font-weight: 700; }}
.score-lbl {{ color: #6e7681; font-size: 11px; }}
.score-sep {{ color: #30363d; }}
.ou-line   {{ color: #6e7681; font-size: 11px; }}
.band      {{ color: #6e7681; font-size: 11px; }}

/* Body: starters + markets side by side */
.gc-body {{
  display: flex; gap: 0;
}}
.gc-starters {{
  padding: 12px 16px; min-width: 220px; max-width: 280px;
  border-right: 1px solid #21262d; flex-shrink: 0;
}}
.gc-sp {{ margin-bottom: 10px; line-height: 1.5; }}
.gc-sp:last-of-type {{ margin-bottom: 0; }}
.sp-team {{ color: #6e7681; font-size: 10px; font-weight: 700;
            text-transform: uppercase; letter-spacing: .5px; }}
.sp-name {{ color: #ffffff; font-weight: 600; font-size: 13px; }}
.xwoba   {{ color: #8b949e; font-size: 11px; margin-left: 4px; }}
.sp-flag {{ font-size: 9px; font-weight: 700; color: #d29922;
            border: 1px solid #d29922; border-radius: 3px; padding: 1px 4px; margin-left: 4px; }}
.k-props {{ font-size: 11px; }}
.k-line  {{ color: #6e7681; }}
.k-pover {{ color: #8b949e; }}
.k-edge  {{ font-weight: 600; }}
.tbd-notice {{ font-size: 11px; color: #d29922; margin-top: 6px; }}

/* Market grid */
.gc-markets {{
  display: flex; flex: 1; flex-wrap: wrap;
}}
.mkt {{
  flex: 1; min-width: 160px;
  padding: 12px 14px;
  border-right: 1px solid #21262d;
}}
.mkt:last-child {{ border-right: none; }}
.mkt-label {{
  font-size: 10px; font-weight: 800; color: #8b949e;
  text-transform: uppercase; letter-spacing: .6px;
  margin-bottom: 8px;
}}
.mkt-tbl {{ border-collapse: collapse; width: 100%; }}
.mkt-tbl .ml {{
  color: #6e7681; font-size: 11px; padding: 2px 0; width: 55%;
}}
.mkt-tbl .mv {{
  font-size: 13px; font-weight: 600; padding: 2px 0; text-align: right;
}}
.mkt-badge {{ margin-top: 8px; }}

/* Value colours */
.bright    {{ color: #ffffff; }}
.pin       {{ color: #58a6ff; }}
.ret       {{ color: #8b949e; }}
.val-green {{ color: #3fb950; }}
.val-yellow{{ color: #d29922; }}
.val-red   {{ color: #f85149; }}
.val-dim   {{ color: #6e7681; }}

/* Shadow ensemble range colours */
.shadow-agree {{ color: #3fb950; font-size: 12px; }}
.shadow-warn  {{ color: #d29922; font-size: 12px; }}
.shadow-split {{ color: #f85149; font-size: 12px; }}
.spread-pill  {{
  display: inline-block; font-size: 10px; font-weight: 700;
  background: rgba(255,255,255,0.08); border-radius: 3px;
  padding: 0 4px; margin-left: 3px; vertical-align: middle;
}}

/* Quantile band colours (total Q10–Q90 range) */
.qband-tight {{ color: #3fb950; font-size: 12px; }}   /* spread < 4.0 — narrow, confident */
.qband-mid   {{ color: #d29922; font-size: 12px; }}   /* spread 4–6   — moderate */
.qband-wide  {{ color: #f85149; font-size: 12px; }}   /* spread > 6.0 — high variance */

/* Badges */
.badge {{
  display: inline-block; font-size: 10px; font-weight: 700;
  letter-spacing: .4px; padding: 2px 7px; border-radius: 4px;
}}
.badge-green  {{ background: #0f2d1a; color: #3fb950; border: 1px solid #3fb950; }}
.badge-amber  {{ background: #2d1b00; color: #d29922; border: 1px solid #d29922; }}
.badge-yellow {{ background: #2d2800; color: #e3b341; border: 1px solid #e3b341; }}
.badge-orange {{ background: #2d1600; color: #db6d28; border: 1px solid #db6d28; }}
.badge-red    {{ background: #2d0d0d; color: #f85149; border: 1px solid #f85149; }}
.badge-gray   {{ background: #161b22; color: #6e7681; border: 1px solid #30363d; }}
.stake {{ color: #3fb950; font-weight: 800; margin-left: 4px; }}

/* Glossary */
.defs {{
  margin-top: 16px;
  background: #161b22; border: 1px solid #21262d; border-radius: 8px;
  padding: 16px 20px;
}}
.defs-title {{
  font-size: 11px; font-weight: 700; color: #6e7681;
  letter-spacing: .8px; text-transform: uppercase; margin-bottom: 14px;
}}
.defs-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 20px;
}}
.def-section {{ }}
.def-heading {{
  font-size: 10px; font-weight: 700; color: #388bfd;
  letter-spacing: .6px; text-transform: uppercase;
  margin-bottom: 8px; padding-bottom: 4px;
  border-bottom: 1px solid #21262d;
}}
.def-row {{
  display: flex; gap: 8px; margin-bottom: 6px; align-items: baseline;
}}
.def-term {{
  font-size: 11px; font-weight: 700; color: #c9d1d9;
  white-space: nowrap; min-width: 90px; flex-shrink: 0;
}}
.def-desc {{
  font-size: 11px; color: #6e7681; line-height: 1.4;
}}

@media (max-width: 700px) {{
  .gc-body {{ flex-direction: column; }}
  .gc-starters {{ max-width: 100%; border-right: none;
                  border-bottom: 1px solid #21262d; }}
  .mkt {{ min-width: 130px; }}
}}

/* ── Parlay suggestions ────────────────────────────── */
.pl-section {{
  background: #161b22; border: 1px solid #21262d;
  border-left: 4px solid #8957e5;
  border-radius: 8px; padding: 16px 20px; margin-bottom: 16px;
}}
.pl-title {{
  font-size: 14px; font-weight: 800; color: #d2a8ff;
  letter-spacing: .3px; margin-bottom: 4px;
}}
.pl-note {{
  font-size: 11px; color: #6e7681; margin-bottom: 12px; line-height: 1.5;
}}
.pl-combos {{ display: flex; flex-direction: column; gap: 12px; }}
.pl-combo {{
  background: #0d1117; border: 1px solid #21262d;
  border-radius: 6px; overflow: hidden;
}}
.pl-combo-head {{
  display: flex; align-items: center; flex-wrap: wrap; gap: 10px;
  padding: 8px 14px; background: #1c2128; border-bottom: 1px solid #21262d;
}}
.pl-n {{
  font-size: 10px; font-weight: 800; color: #6e7681;
  letter-spacing: 1px; text-transform: uppercase;
}}
.pl-sgp {{
  font-size: 9px; font-weight: 700; color: #d29922;
  background: #2d2a1f; border: 1px solid #d29922;
  border-radius: 3px; padding: 1px 5px; letter-spacing: .5px;
}}
.pl-price {{
  font-size: 18px; font-weight: 800; color: #d2a8ff;
}}
.pl-stats {{
  font-size: 11px; color: #8b949e; margin-left: auto;
}}
.pl-tbl {{
  width: 100%; border-collapse: collapse; font-size: 12px;
}}
.pl-tbl th {{
  padding: 5px 14px; text-align: left;
  font-size: 10px; font-weight: 700; color: #6e7681;
  letter-spacing: .5px; text-transform: uppercase;
  border-bottom: 1px solid #21262d;
}}
.pl-tbl td {{
  padding: 6px 14px; border-bottom: 1px solid #161b22;
  vertical-align: middle;
}}
.pl-tbl tr:last-child td {{ border-bottom: none; }}
.pl-game  {{ color: #8b949e; font-size: 11px; white-space: nowrap; }}
.pl-leg   {{ color: #c9d1d9; font-weight: 600; }}
.pl-odds  {{ color: #ffffff; font-weight: 700; white-space: nowrap; }}
.pl-prob  {{ color: #8b949e; }}
.pl-corr  {{ font-size: 10px; color: #d29922; font-style: italic; }}
.edge-pos {{ color: #3fb950; font-weight: 700; }}
.edge-med {{ color: #d29922; font-weight: 700; }}
.edge-neg {{ color: #f85149; font-weight: 700; }}
.pl-empty {{
  font-size: 12px; color: #6e7681; font-style: italic; padding: 8px 0;
}}
</style>
</head>
<body>

<div class="hdr {header_cls}">
  <h1>THE WIZARD REPORT</h1>
  <div class="date">{day_label}</div>
  <div class="sub">{header_sub}</div>
  <div class="meta">Generated {gen_time} ET &nbsp;·&nbsp;
    Three-Part Lock: |P_model − P_Pinnacle| ≤ 4% &nbsp;·&nbsp; odds ≥ −225 &nbsp;·&nbsp; edge ≥ 1.0% vs retail
  </div>
</div>

{cards_html}

{parlay_section}

<div class="defs">
  <div class="defs-title">&#9432; GLOSSARY</div>
  <div class="defs-grid">

    <div class="def-section">
      <div class="def-heading">PROBABILITIES</div>
      <div class="def-row"><span class="def-term" style="color:#ffffff">Model</span><span class="def-desc">Our simulation's estimated win/cover/over probability for this game</span></div>
      <div class="def-row"><span class="def-term" style="color:#58a6ff">Pinnacle</span><span class="def-desc">Sharp offshore book — de-vigged implied probability. Best market signal available. Used as the sanity check</span></div>
      <div class="def-row"><span class="def-term" style="color:#8b949e">Retail</span><span class="def-desc">US sportsbook (DraftKings / FanDuel / BetMGM) — de-vigged implied probability. This is what you actually bet into</span></div>
      <div class="def-row"><span class="def-term">Edge</span><span class="def-desc">Model prob minus Retail implied prob. Positive = model likes this side more than the market does</span></div>
      <div class="def-row"><span class="def-term">Band (3–7)</span><span class="def-desc">25th–75th percentile range from the simulation. The game lands inside this range 50% of the time</span></div>
    </div>

    <div class="def-section">
      <div class="def-heading">BET MARKETS</div>
      <div class="def-row"><span class="def-term">ML</span><span class="def-desc">Moneyline — straight win/loss bet. No spread. Model uses a 35% model / 65% Vegas blend since our sim only models pitching</span></div>
      <div class="def-row"><span class="def-term">RL</span><span class="def-desc">Run Line — home team -1.5 spread. Home team must win by 2+ runs to cover. Primary model bet</span></div>
      <div class="def-row"><span class="def-term">O/U</span><span class="def-desc">Over/Under — total combined runs scored. Model picks the better edge direction (OVER or UNDER)</span></div>
      <div class="def-row"><span class="def-term">K prop</span><span class="def-desc">Strikeout prop for the starting pitcher. Line = posted K total. P(O) = model's probability of going over that line. Edge = P(O) minus market implied</span></div>
    </div>

    <div class="def-section">
      <div class="def-heading">THREE-PART LOCK GATES</div>
      <div class="def-row"><span class="def-term">Gate 1: Sanity</span><span class="def-desc">|Model − Pinnacle| must be ≤ 4%. If the model disagrees with the sharp market by more than 4 points, something is wrong — skip the game</span></div>
      <div class="def-row"><span class="def-term">Gate 2: Odds Floor</span><span class="def-desc">Retail ML must be better than −225. Avoids heavy favorites where you risk too much to win too little</span></div>
      <div class="def-row"><span class="def-term">Gate 3: Edge</span><span class="def-desc">Model must beat the retail implied probability by ≥ 1.0%. This is the actual closing line value (CLV) edge you're betting on</span></div>
      <div class="def-row"><span class="def-term">TIER 1</span><span class="def-desc">Edge ≥ 3.0% — strong signal. Quarter-Kelly sizing</span></div>
      <div class="def-row"><span class="def-term">TIER 2</span><span class="def-desc">Edge ≥ 1.0% — medium signal. Eighth-Kelly sizing</span></div>
    </div>

    <div class="def-section">
      <div class="def-heading">STARTER FLAGS</div>
      <div class="def-row"><span class="def-term" style="color:#d29922">VOLATILE</span><span class="def-desc">Pitcher's recent velocity is trending down — higher variance, less reliable projection</span></div>
      <div class="def-row"><span class="def-term" style="color:#3fb950">GAINER</span><span class="def-desc">Pitcher's velocity is trending up — potentially outperforming their season stats</span></div>
      <div class="def-row"><span class="def-term">xwOBA</span><span class="def-desc">Expected weighted on-base average allowed. Measures pitcher quality based on contact quality, not outcomes. Lower = better pitcher. League avg ≈ 0.315</span></div>
      <div class="def-row"><span class="def-term">✓ confirmed</span><span class="def-desc">Official lineup confirmed by the team</span></div>
      <div class="def-row"><span class="def-term">~ projected</span><span class="def-desc">Probable starter based on rotation — not yet officially confirmed</span></div>
    </div>

  </div>
</div>

</body>
</html>"""

    out = Path("daily_card.html")
    out.write_text(html, encoding="utf-8")
    cards_dir = Path("daily_cards")
    cards_dir.mkdir(exist_ok=True)
    dated = cards_dir / f"daily_card_{date_str}.html"
    dated.write_text(html, encoding="utf-8")
    print(f"  Saved -> {out}  ({out.stat().st_size // 1024}KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Run automated MLB daily prediction card")
    parser.add_argument("--date",      type=str,   default=None,
                        help="Date (YYYY-MM-DD, default: today)")
    parser.add_argument("--min-edge",  type=float, default=0.0,
                        help="Only show games above this RL probability threshold")
    parser.add_argument("--csv",       action="store_true",
                        help="Write results to daily_card.csv")
    parser.add_argument("--email",     action="store_true",
                        help="Send the card as an email via Gmail")
    args = parser.parse_args()

    date_str = args.date or str(datetime.date.today())

    print("=" * 72)
    print("  run_today.py — automated daily card")
    print(f"  Date: {date_str}")
    print("=" * 72)

    results = run_card(date_str, min_edge=args.min_edge)

    if args.csv and results:
        # Always save date-stamped file so today + tomorrow can coexist
        cards_dir = Path("daily_cards")
        cards_dir.mkdir(exist_ok=True)
        dated_out = cards_dir / f"daily_card_{date_str}.csv"
        pd.DataFrame(results).to_csv(dated_out, index=False)
        print(f"  Saved -> {dated_out}")
        # Also update the canonical daily_card.csv when running for today
        if date_str == str(datetime.date.today()):
            pd.DataFrame(results).to_csv("daily_card.csv", index=False)
            print(f"  Saved -> daily_card.csv")
        # Always write the HTML card alongside the CSV
        write_html_card(results, date_str)

    print_card(results, min_edge=args.min_edge)

    if args.email and results:
        send_card_email(results, date_str)


if __name__ == "__main__":
    main()
