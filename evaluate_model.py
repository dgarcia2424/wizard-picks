"""
evaluate_model.py
=================
Walk-forward backtesting framework for The Wizard Report MLB pipeline.

Evaluates the current XGBoost + Platt/Stacking pipeline on the 2025 val set
and reports per-month metrics: AUC, Brier score, log-loss, accuracy, and
cover rate.  Optionally compares against a Bayesian empirical shrinkage
variant that pulls early-season pitcher stats toward career priors when the
current-season sample is small.

Why Bayesian shrinkage matters here
-------------------------------------
The feature matrix uses time-based EWMA (halflife=30 days) per pitcher.  In
April a starter may only have 1-3 starts (~20-60 batters faced) — the EWMA
is essentially just that thin slice of 2026 data.  Empirical Bayes shrinkage
adds a career-weighted floor:

    posterior = (n_est × ewma_curr + n_prior × career_mean)
                ─────────────────────────────────────────────
                         (n_est + n_prior)

where n_prior = 200 PA (≈10 starts).  By June/July when n_est >> n_prior the
posterior converges to the current EWMA; in April it leans heavily on the
career mean, reducing early-season noise.

Usage
-----
  python evaluate_model.py                   # current model on 2025 val set
  python evaluate_model.py --bayes           # add Bayesian shrinkage comparison
  python evaluate_model.py --n-prior 150     # tune shrinkage strength (default 200)
  python evaluate_model.py --year 2024       # evaluate on a different year
  python evaluate_model.py --save-preds      # write eval_predictions.csv
  python evaluate_model.py --bayes --save-preds --n-prior 200

Outputs
-------
  eval_metrics.csv          per-month metrics table
  eval_predictions.csv      per-game predictions (--save-preds only)
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# BayesianStacker (v5.1) and legacy StackingModel must be importable so pickle
# can reconstruct whichever format is on disk.
try:
    from train_xgboost import BayesianStacker  # noqa: F401 — needed by pickle
except ImportError:
    BayesianStacker = None  # type: ignore

try:
    from train_xgboost import StackingModel    # noqa: F401 — legacy pickle support
except ImportError:
    StackingModel = None  # type: ignore — removed in v5.1, legacy fallback

# ── GPU availability flag (used by load_models to enable CUDA inference) ───
try:
    import cupy as _cp
    _GPU = _cp.cuda.is_available()
    if _GPU:
        try:
            _probe = _cp.random.standard_normal(1); del _probe
        except Exception:
            _GPU = False
except ImportError:
    _GPU = False

try:
    import lightgbm as _lgb_eval
    _LGBM_EVAL = True
except ImportError:
    _lgb_eval = None
    _LGBM_EVAL = False

try:
    import catboost as _cb_eval
    _CATBOOST_EVAL = True
except ImportError:
    _cb_eval = None
    _CATBOOST_EVAL = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS_DIR   = Path("./models")
MATRIX_PATH  = Path("feature_matrix.parquet")

# Pitcher EWMA features that benefit from early-season shrinkage
SP_SHRINK_COLS = [
    "k_pct", "bb_pct", "xwoba_against", "gb_pct", "xrv_per_pitch", "k_minus_bb",
]

# Differential features that must be recomputed after shrinkage
SP_DIFF_MAP = {
    # (result_col,  home_col,                  away_col,              direction)
    # direction = +1 means "home - away"; -1 means "away - home"
    "sp_k_pct_diff":    ("home_sp_k_pct",         "away_sp_k_pct",         +1),
    "sp_xwoba_diff":    ("away_sp_xwoba_against",  "home_sp_xwoba_against", +1),
    "sp_xrv_diff":      ("home_sp_xrv_per_pitch",  "away_sp_xrv_per_pitch", +1),
    "sp_kminusbb_diff": ("home_sp_k_minus_bb",     "away_sp_k_minus_bb",    +1),
}

# Estimated batters-faced by calendar month at time of a typical start.
# April: first 2-3 starts ≈ 40 BF; October: ~20 starts ≈ 400 BF.
MONTH_BF = {4: 40, 5: 100, 6: 180, 7: 250, 8: 320, 9: 380, 10: 400}

N_PRIOR_DEFAULT = 200   # effective prior batters-faced (≈ 10 full starts)

# Signed diff columns to negate when building the away-perspective row.
# Mirrors _TEAM_DIFF_COLS in monte_carlo_runline.py.
_TEAM_DIFF_COLS = [
    "sp_k_pct_diff", "sp_xwoba_diff", "sp_xrv_diff", "sp_velo_diff",
    "sp_age_diff", "sp_kminusbb_diff", "sp_k_pct_10d_diff", "sp_xwoba_10d_diff",
    "sp_bb_pct_10d_diff", "batting_matchup_edge", "batting_matchup_edge_10d",
    "bp_era_diff", "bp_k9_diff", "bp_whip_diff", "circadian_edge",
]

MONTH_NAMES = {
    4: "Apr", 5: "May", 6: "Jun", 7: "Jul",
    8: "Aug", 9: "Sep", 10: "Oct",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(models_dir: Path = MODELS_DIR) -> dict:
    """
    Load XGBoost models (RL / ML / Total), Platt calibrators, stacking LR,
    and the feature column list.

    Returns
    -------
    dict with keys: feature_cols, xgb_rl, xgb_ml, xgb_tot,
                    cal_rl, cal_ml, stacking (or None)
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("pip install xgboost")

    feature_cols = json.loads((models_dir / "feature_cols.json").read_text())

    xgb_rl = xgb.XGBClassifier()
    xgb_rl.load_model(str(models_dir / "xgb_rl.json"))
    if _GPU:
        xgb_rl.set_params(device="cuda")

    xgb_ml = xgb.XGBClassifier()
    xgb_ml.load_model(str(models_dir / "xgb_ml.json"))
    if _GPU:
        xgb_ml.set_params(device="cuda")

    xgb_tot = xgb.XGBRegressor()
    xgb_tot.load_model(str(models_dir / "xgb_total.json"))
    if _GPU:
        xgb_tot.set_params(device="cuda")

    cal_rl = pickle.loads((models_dir / "calibrator_rl.pkl").read_bytes())
    cal_ml = pickle.loads((models_dir / "calibrator_ml.pkl").read_bytes())

    stk = None
    stk_path = models_dir / "stacking_lr_rl.pkl"
    if stk_path.exists():
        stk = pickle.loads(stk_path.read_bytes())

    # ── LightGBM Level-1 models (optional — graceful fallback if absent) ──
    lgbm_rl = lgbm_ml = lgbm_tot = None
    if _LGBM_EVAL:
        for attr, fname in [("lgbm_rl", "lgbm_rl.pkl"),
                             ("lgbm_ml", "lgbm_ml.pkl"),
                             ("lgbm_tot", "lgbm_total.pkl")]:
            p = models_dir / fname
            if p.exists():
                try:
                    obj = pickle.loads(p.read_bytes())
                    if attr == "lgbm_rl":  lgbm_rl  = obj
                    elif attr == "lgbm_ml": lgbm_ml = obj
                    else:                   lgbm_tot = obj
                except Exception as e:
                    print(f"  [WARN] Could not load {fname}: {e}")

    # ── CatBoost Level-1 models (optional — graceful fallback if absent) ──
    cat_rl = cat_ml = cat_tot = None
    if _CATBOOST_EVAL:
        for attr, fname in [("cat_rl", "cat_rl.pkl"),
                             ("cat_ml", "cat_ml.pkl"),
                             ("cat_tot", "cat_total.pkl")]:
            p = models_dir / fname
            if p.exists():
                try:
                    obj = pickle.loads(p.read_bytes())
                    if attr == "cat_rl":  cat_rl  = obj
                    elif attr == "cat_ml": cat_ml = obj
                    else:                  cat_tot = obj
                except Exception as e:
                    print(f"  [WARN] Could not load {fname}: {e}")

    n_l1 = 1 + (lgbm_rl is not None) + (cat_rl is not None)
    print(f"  Level-1 models loaded: XGBoost"
          f"{' + LightGBM' if lgbm_rl else ''}"
          f"{' + CatBoost' if cat_rl else ''}"
          f"  ({n_l1}/3)")

    # ── Team-perspective RL model (5-step flow) ───────────────────────────
    team_rl = None
    team_feat_cols = None
    team_path = models_dir / "xgb_rl_team.json"
    feat_team_path = models_dir / "feature_cols_team.json"
    if team_path.exists() and feat_team_path.exists():
        try:
            _tm = xgb.XGBClassifier()
            _tm.load_model(str(team_path))
            if _GPU:
                _tm.set_params(device="cuda")
            team_rl = _tm
            team_feat_cols = json.loads(feat_team_path.read_text())
            print(f"  Team RL  : loaded ({len(team_feat_cols)} features) — 5-step flow active")
        except Exception as e:
            print(f"  Team RL  : load failed ({e}) — falling back to raw probs for stacker")
    else:
        print(f"  Team RL  : not found — falling back to raw probs for stacker")

    return {
        "feature_cols":    feature_cols,
        "xgb_rl":          xgb_rl,
        "xgb_ml":          xgb_ml,
        "xgb_tot":         xgb_tot,
        "lgbm_rl":         lgbm_rl,
        "lgbm_ml":         lgbm_ml,
        "lgbm_tot":        lgbm_tot,
        "cat_rl":          cat_rl,
        "cat_ml":          cat_ml,
        "cat_tot":         cat_tot,
        "cal_rl":          cal_rl,
        "cal_ml":          cal_ml,
        "stacking":        stk,
        "team_rl":         team_rl,
        "team_feat_cols":  team_feat_cols,
    }


# ---------------------------------------------------------------------------
# Team-perspective helpers (5-step inference flow)
# ---------------------------------------------------------------------------

def _build_team_X(df: pd.DataFrame, team_feat_cols: list, is_home: int) -> pd.DataFrame:
    """
    Build the team-model feature matrix for one perspective (home or away).

    For the away perspective, all home_* / away_* column pairs are swapped,
    true_home_prob ↔ true_away_prob is swapped, bat_vs columns are swapped,
    and all signed diff columns are negated.  is_home is set accordingly.
    """
    cols = set(df.columns)

    if is_home:
        X = pd.DataFrame(index=df.index)
        for c in team_feat_cols:
            X[c] = df[c] if c in cols else np.nan
        X["is_home"] = 1.0
        return X

    # Away perspective: start with a copy, then swap
    tmp = df.copy()

    # Swap all home_* ↔ away_* pairs present in team_feat_cols
    processed: set[str] = set()
    for c in team_feat_cols:
        if c in processed:
            continue
        if c.startswith("home_"):
            partner = "away_" + c[5:]
            if partner in cols and c in cols:
                tmp[c]       = df[partner]
                tmp[partner] = df[c]
                processed.add(c)
                processed.add(partner)
        elif c.startswith("away_"):
            partner = "home_" + c[5:]
            if partner in cols and c in cols:
                tmp[c]       = df[partner]
                tmp[partner] = df[c]
                processed.add(c)
                processed.add(partner)

    # Swap bat_vs columns (not covered by generic home_/away_ prefix swap above
    # because the partner name has a different structure)
    for h_col, a_col in [("home_bat_vs_away_sp",     "away_bat_vs_home_sp"),
                          ("home_bat_vs_away_sp_10d", "away_bat_vs_home_sp_10d")]:
        if h_col in cols and a_col in cols:
            tmp[h_col] = df[a_col]
            tmp[a_col] = df[h_col]

    # Swap true_home_prob ↔ true_away_prob
    if "true_home_prob" in cols and "true_away_prob" in cols:
        tmp["true_home_prob"] = df["true_away_prob"]
        tmp["true_away_prob"] = df["true_home_prob"]

    # Negate signed diff columns
    for c in _TEAM_DIFF_COLS:
        if c in cols:
            tmp[c] = -df[c]

    X = pd.DataFrame(index=df.index)
    for c in team_feat_cols:
        X[c] = tmp[c] if c in tmp.columns else np.nan
    X["is_home"] = 0.0
    return X


def _team_l1_probs(team_model, df: pd.DataFrame,
                   team_feat_cols: list) -> np.ndarray | None:
    """
    Steps 1+2: run team model on both perspectives, return L1-normalised home prob.
    Returns None if inference fails.
    """
    try:
        X_home = _build_team_X(df, team_feat_cols, is_home=1)
        X_away = _build_team_X(df, team_feat_cols, is_home=0)
        raw_home = team_model.predict_proba(X_home)[:, 1]
        raw_away = team_model.predict_proba(X_away)[:, 1]
        p_sum = np.maximum(raw_home + raw_away, 1e-9)
        return raw_home / p_sum   # shape (n,) — L1-normalised home prob
    except Exception as exc:
        print(f"  [WARN] Team model inference failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_df(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """
    Score a feature matrix DataFrame through the full inference stack:
      XGBoost raw → Platt calibration → Stacking LR (if available)

    Adds columns: rl_raw, rl_cal, rl_stacked*, ml_raw, ml_cal, tot_pred
    (* only if stacking model is present)

    Parameters
    ----------
    df     : feature matrix with the same column structure as training
    models : dict returned by load_models()

    Returns
    -------
    Copy of df with prediction columns appended.
    """
    feature_cols = models["feature_cols"]

    # Build X aligned to training order; fill unknown cols with NaN
    X = pd.DataFrame(index=df.index)
    for col in feature_cols:
        X[col] = df[col] if col in df.columns else np.nan

    # ── XGBoost raw probabilities ─────────────────────────────────────────
    rl_raw   = models["xgb_rl"].predict_proba(X)[:, 1]
    ml_raw   = models["xgb_ml"].predict_proba(X)[:, 1]
    _tot_raw = models["xgb_tot"].predict(X)
    tot_pred = _tot_raw[:, 1] if _tot_raw.ndim == 2 else _tot_raw   # quantile model -> take median col

    # ── LightGBM raw probabilities (optional) ────────────────────────────
    lgbm_rl_raw = lgbm_ml_raw = lgbm_tot_pred = None
    if models.get("lgbm_rl") is not None:
        try:
            lgbm_rl_raw  = models["lgbm_rl"].predict_proba(X)[:, 1]
        except Exception as e:
            print(f"  [WARN] LGBM RL scoring failed: {e}")
    if models.get("lgbm_ml") is not None:
        try:
            lgbm_ml_raw = models["lgbm_ml"].predict_proba(X)[:, 1]
        except Exception as e:
            print(f"  [WARN] LGBM ML scoring failed: {e}")
    if models.get("lgbm_tot") is not None:
        try:
            lgbm_tot_pred = models["lgbm_tot"].predict(X)
        except Exception as e:
            print(f"  [WARN] LGBM TOT scoring failed: {e}")

    # ── CatBoost raw probabilities (optional) ────────────────────────────
    cat_rl_raw = cat_ml_raw = cat_tot_pred = None
    _X_cat = X.astype("float64").values   # CatBoost cannot handle pd.NA
    if models.get("cat_rl") is not None:
        try:
            cat_rl_raw  = models["cat_rl"].predict_proba(_X_cat)[:, 1]
        except Exception as e:
            print(f"  [WARN] CatBoost RL scoring failed: {e}")
    if models.get("cat_ml") is not None:
        try:
            cat_ml_raw = models["cat_ml"].predict_proba(_X_cat)[:, 1]
        except Exception as e:
            print(f"  [WARN] CatBoost ML scoring failed: {e}")
    if models.get("cat_tot") is not None:
        try:
            cat_tot_pred = models["cat_tot"].predict(_X_cat)
        except Exception as e:
            print(f"  [WARN] CatBoost TOT scoring failed: {e}")

    # ── Platt calibration (XGBoost only — single-model calibrator) ────────
    rl_cal = models["cal_rl"].predict_proba(rl_raw.reshape(-1, 1))[:, 1]
    ml_cal = models["cal_ml"].predict_proba(ml_raw.reshape(-1, 1))[:, 1]

    out = df.copy()
    out["rl_raw"]   = rl_raw
    out["rl_cal"]   = rl_cal
    out["ml_raw"]   = ml_raw
    out["ml_cal"]   = ml_cal
    out["tot_pred"] = tot_pred
    if lgbm_rl_raw  is not None: out["lgbm_rl_raw"]   = lgbm_rl_raw
    if cat_rl_raw   is not None: out["cat_rl_raw"]    = cat_rl_raw
    if lgbm_tot_pred is not None: out["lgbm_tot_pred"] = lgbm_tot_pred
    if cat_tot_pred  is not None: out["cat_tot_pred"]  = cat_tot_pred

    # ── Stacking (Level-2) — 5-step team flow when team model is available ──
    if models["stacking"] is not None:
        try:
            team_rl      = models.get("team_rl")
            team_feat_cols = models.get("team_feat_cols")

            if team_rl is not None and team_feat_cols is not None:
                # Steps 1+2: team model inference + L1 normalisation
                home_norm = _team_l1_probs(team_rl, df, team_feat_cols)
            else:
                home_norm = None

            # Compute SP handedness segment: home_R*2 + away_R (0=LvL,1=LvR,2=RvL,3=RvR)
            _h_r = (df["home_sp_p_throws_R"].fillna(1).astype(int)
                    if "home_sp_p_throws_R" in df.columns
                    else pd.Series(1, index=df.index))
            _a_r = (df["away_sp_p_throws_R"].fillna(1).astype(int)
                    if "away_sp_p_throws_R" in df.columns
                    else pd.Series(1, index=df.index))
            segment_ids = (_h_r * 2 + _a_r).values.astype(np.int32)

            if home_norm is not None:
                # Step 3: feed L1-normalised home prob to stacker (production path)
                stk_probs = models["stacking"].predict(home_norm, df,
                                                       segment_id=segment_ids)
                out["team_l1_home_norm"] = home_norm   # expose for diagnostics
            else:
                # Fallback: raw xgb_rl → stacker (same as pre-team-model behaviour)
                stk_probs = models["stacking"].predict(
                    rl_raw, df,
                    segment_id=segment_ids,
                    lgbm_raw=lgbm_rl_raw,
                    cat_raw=cat_rl_raw,
                )
            out["rl_stacked"] = stk_probs
        except Exception as exc:
            print(f"  [WARN] Stacking model scoring failed: {exc}")
            out["rl_stacked"] = np.nan

    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(y_true: pd.Series, y_pred: pd.Series) -> dict | None:
    """
    AUC, Brier, LogLoss, accuracy@50%, and cover rate.
    Returns None if < 20 samples or only one class present.
    """
    mask = y_true.notna() & y_pred.notna()
    yt   = y_true[mask].astype(int)
    yp   = y_pred[mask].astype(float)
    n    = len(yt)

    if n < 20 or yt.nunique() < 2:
        return None

    return {
        "n":          n,
        "auc":        round(float(roc_auc_score(yt, yp)), 4),
        "brier":      round(float(brier_score_loss(yt, yp)), 4),
        "logloss":    round(float(log_loss(yt, yp)), 4),
        "accuracy":   round(float(((yp > 0.50) == yt).mean()), 4),
        "cover_rate": round(float(yt.mean()), 4),
    }


def compute_monthly_metrics(
    scored_df: pd.DataFrame,
    label: str = "current",
) -> pd.DataFrame:
    """
    Compute per-month and overall metrics for both the RL and ML models.

    Returns a tidy DataFrame with columns:
      dataset, model, period, month_num, n, auc, brier, logloss, accuracy, cover_rate
    """
    df = scored_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["month"]     = df["game_date"].dt.month

    # Prefer stacked probs over Platt for RL when available
    rl_col = "rl_stacked" if "rl_stacked" in df.columns else "rl_cal"

    eval_targets = [
        ("RL", rl_col,  "home_covers_rl"),
        ("ML", "ml_cal", "actual_home_win"),
    ]

    rows = []
    for model_name, prob_col, label_col in eval_targets:
        if label_col not in df.columns or prob_col not in df.columns:
            continue

        sub = df[df[label_col].notna() & df[prob_col].notna()].copy()
        if len(sub) == 0:
            continue

        # Per-month
        for month_num, grp in sub.groupby("month"):
            m = _metrics(grp[label_col], grp[prob_col])
            if m is None:
                continue
            rows.append({
                "dataset":   label,
                "model":     model_name,
                "period":    f"{int(month_num):02d}-{MONTH_NAMES.get(int(month_num), '?')}",
                "month_num": int(month_num),
                **m,
            })

        # Overall
        m_all = _metrics(sub[label_col], sub[prob_col])
        if m_all:
            rows.append({
                "dataset":   label,
                "model":     model_name,
                "period":    "OVERALL",
                "month_num": 99,
                **m_all,
            })

    return (
        pd.DataFrame(rows)
        .sort_values(["model", "month_num"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Bayesian empirical shrinkage
# ---------------------------------------------------------------------------

def _build_career_priors(
    train_df: pd.DataFrame,
) -> dict:
    """
    Compute per-pitcher career mean for each SP_SHRINK_COL by pooling
    home_sp_* and away_sp_* rows from the training data.

    Also computes the league-wide mean as a fallback for unknown pitchers.

    Returns
    -------
    {stat: {pitcher_name_upper: career_mean, '__league__': league_mean}}
    """
    priors: dict[str, dict] = {}

    for stat in SP_SHRINK_COLS:
        # Gather (name, ewma_value) pairs from both home and away starters
        pairs: list[tuple[str, float]] = []

        for prefix, name_col in [("home_sp", "home_starter_name"),
                                   ("away_sp", "away_starter_name")]:
            feat_col = f"{prefix}_{stat}"
            if feat_col not in train_df.columns or name_col not in train_df.columns:
                continue
            tmp = (
                train_df[[name_col, feat_col]]
                .dropna()
                .rename(columns={name_col: "pitcher", feat_col: "val"})
            )
            pairs.extend(tmp.itertuples(index=False, name=None))

        if not pairs:
            priors[stat] = {}
            continue

        tmp_df = pd.DataFrame(pairs, columns=["pitcher", "val"])
        career_means = (
            tmp_df.groupby("pitcher")["val"]
            .mean()
            .to_dict()
        )
        career_means["__league__"] = float(tmp_df["val"].mean())
        priors[stat] = career_means

    return priors


def apply_bayes_shrinkage(
    val_df: pd.DataFrame,
    train_df: pd.DataFrame,
    n_prior: int = N_PRIOR_DEFAULT,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Apply empirical Bayes shrinkage to pitcher EWMA features in val_df.

    For each pitcher EWMA feature at game G in month M:

        n_est = MONTH_BF[M]               # estimated batters faced so far
        prior = career_mean from 2023+2024 training data

        posterior = (n_est * ewma_curr + n_prior * prior_mean)
                    ─────────────────────────────────────────
                              (n_est + n_prior)

    Shrinkage weight = n_prior / (n_est + n_prior):
        April  (n_est=40):  weight ≈ 0.83  → heavy shrinkage toward career
        June   (n_est=180): weight ≈ 0.53  → moderate
        August (n_est=320): weight ≈ 0.38  → light
        Oct    (n_est=400): weight ≈ 0.33  → minimal

    Differential features (sp_k_pct_diff etc.) are recomputed after shrinkage.

    Parameters
    ----------
    val_df   : validation year feature matrix rows
    train_df : training year feature matrix rows (career priors source)
    n_prior  : effective prior batters-faced (default: 200)
    verbose  : print shrinkage weight table

    Returns
    -------
    Modified copy of val_df.
    """
    df = val_df.copy()
    df["_month"] = pd.to_datetime(df["game_date"]).dt.month

    if verbose:
        print(f"\n  Shrinkage weights by month (n_prior={n_prior}):")
        print(f"  {'Month':<8}  {'n_est':>6}  {'weight':>7}")
        print(f"  {'-'*26}")
        for m, n_est in sorted(MONTH_BF.items()):
            w = n_prior / (n_est + n_prior)
            print(f"  {MONTH_NAMES.get(m, str(m)):<8}  {n_est:>6}  {w:>7.3f}")

    # Build career priors from train data
    priors = _build_career_priors(train_df)

    for prefix, name_col in [("home_sp", "home_starter_name"),
                               ("away_sp", "away_starter_name")]:
        if name_col not in df.columns:
            continue

        for stat in SP_SHRINK_COLS:
            feat_col = f"{prefix}_{stat}"
            if feat_col not in df.columns:
                continue
            if stat not in priors or not priors[stat]:
                continue

            stat_priors = priors[stat]
            league_mean = stat_priors.get("__league__", np.nan)

            # Vectorised shrinkage per game row
            n_est_vec  = df["_month"].map(MONTH_BF).fillna(400).astype(float)
            weight_vec = n_prior / (n_est_vec + n_prior)      # toward prior
            curr_vec   = df[feat_col]

            # Look up career mean per pitcher (fallback to league mean)
            prior_vec = (
                df[name_col]
                .map(stat_priors)
                .fillna(league_mean)
            )

            # posterior = (1 - w) * curr + w * prior  [equivalent formula]
            df[feat_col] = (
                (1.0 - weight_vec) * curr_vec + weight_vec * prior_vec
            )

    # Recompute derived differential features
    for diff_col, (col_a, col_b, direction) in SP_DIFF_MAP.items():
        if col_a in df.columns and col_b in df.columns:
            df[diff_col] = direction * (df[col_a] - df[col_b])

    df = df.drop(columns=["_month"])
    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_metrics_table(
    metrics_df: pd.DataFrame,
    datasets: list[str],
) -> None:
    """Print a formatted per-month metrics table (single or dual dataset)."""

    for model_name in metrics_df["model"].unique():
        sub = metrics_df[metrics_df["model"] == model_name].copy()

        print(f"\n{'='*74}")
        print(f"  {model_name}")
        print(f"{'='*74}")

        if len(datasets) == 1:
            ds    = datasets[0]
            rows  = sub.sort_values("month_num")
            hdr   = f"  {'Period':<10}  {'n':>5}  {'AUC':>7}  {'Brier':>7}  {'Acc':>7}  {'Cover%':>7}"
            sep   = f"  {'-'*56}"
            print(hdr)
            print(sep)
            for _, r in rows.iterrows():
                pfx = "  >> " if r["period"] == "OVERALL" else "     "
                print(
                    f"{pfx}{r['period']:<10}  {int(r['n']):>5}  "
                    f"{r['auc']:>7.4f}  {r['brier']:>7.4f}  "
                    f"{r['accuracy']:>7.4f}  {r['cover_rate']:>7.3f}"
                )

        else:
            # Side-by-side comparison: current vs bayes
            d1, d2     = datasets[0], datasets[1]
            idx1 = sub[sub["dataset"] == d1].set_index("period")
            idx2 = sub[sub["dataset"] == d2].set_index("period")
            all_periods = sub.sort_values("month_num")["period"].unique()

            hdr = (
                f"  {'Period':<10}  {'n':>5}  "
                f"{'AUC(curr)':>10}  {'AUC(bayes)':>11}  {'dAUC':>7}  "
                f"{'Brier(c)':>9}  {'Brier(b)':>9}  {'dBrier':>7}"
            )
            sep = f"  {'-'*82}"
            print(hdr)
            print(sep)

            for period in all_periods:
                if period not in idx1.index or period not in idx2.index:
                    continue
                r1 = idx1.loc[period]
                r2 = idx2.loc[period]
                if isinstance(r1, pd.DataFrame): r1 = r1.iloc[0]
                if isinstance(r2, pd.DataFrame): r2 = r2.iloc[0]

                d_auc   = r2["auc"]   - r1["auc"]
                d_brier = r1["brier"] - r2["brier"]   # positive = Bayes better

                auc_flag   = " [+]" if d_auc   > +0.004 else (" [-]" if d_auc   < -0.004 else "    ")
                brier_flag = " [+]" if d_brier > +0.002 else (" [-]" if d_brier < -0.002 else "    ")

                pfx = "  >> " if period == "OVERALL" else "     "
                print(
                    f"{pfx}{period:<10}  {int(r1['n']):>5}  "
                    f"{r1['auc']:>10.4f}  {r2['auc']:>11.4f}  {d_auc:>+7.4f}{auc_flag}  "
                    f"{r1['brier']:>9.4f}  {r2['brier']:>9.4f}  {d_brier:>+7.4f}{brier_flag}"
                )

            print()
            print(f"  [+] = Bayesian version is better | [-] = current model is better")
            print(f"  AUC: higher is better.  dBrier: positive = Bayes improves calibration.")


def _print_edge_analysis(
    scored_df: pd.DataFrame,
    label_col: str = "home_covers_rl",
    thresholds: tuple = (0.48, 0.50, 0.52, 0.525, 0.54, 0.56, 0.58, 0.60),
) -> None:
    """Simulated ROI at various probability thresholds (-110 standard juice)."""
    # Prefer stacked > Platt > raw
    for col in ("rl_stacked", "rl_cal", "rl_raw"):
        if col in scored_df.columns:
            prob_col = col
            break
    else:
        return

    sub = scored_df[scored_df[label_col].notna()].copy()
    sub[label_col] = sub[label_col].astype(int)

    print(f"\n  Edge analysis ({prob_col}, -110 juice):")
    print(f"  {'Threshold':>10}  {'n_bets':>7}  {'win_rate':>9}  {'roi':>7}  {'units':>8}")
    print(f"  {'-'*48}")
    for thresh in thresholds:
        bets = sub[sub[prob_col] >= thresh]
        n    = len(bets)
        if n < 15:
            continue
        wins   = int(bets[label_col].sum())
        wr     = wins / n
        roi    = (wins * (100 / 110) - (n - wins)) / n
        units  = wins * (100 / 110) - (n - wins)
        print(
            f"  {thresh:>10.3f}  {n:>7}  {wr:>9.3f}  {roi:>+7.3f}  {units:>+8.1f}"
        )


def _print_monthly_edge_analysis(
    scored_df: pd.DataFrame,
    thresh: float = 0.53,
    label_col: str = "home_covers_rl",
) -> None:
    """Win rate and ROI at a fixed threshold, broken down by month."""
    for col in ("rl_stacked", "rl_cal", "rl_raw"):
        if col in scored_df.columns:
            prob_col = col
            break
    else:
        return

    df = scored_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["month"]     = df["game_date"].dt.month
    sub = df[df[label_col].notna() & (df[prob_col] >= thresh)].copy()
    sub[label_col] = sub[label_col].astype(int)

    if len(sub) < 10:
        return

    print(f"\n  Monthly edge breakdown (prob >= {thresh:.3f}, -110 juice):")
    print(f"  {'Month':<8}  {'n_bets':>6}  {'win_rate':>9}  {'roi':>7}  {'units':>8}")
    print(f"  {'-'*44}")

    for month_num, grp in sub.groupby("month"):
        n    = len(grp)
        if n < 5:
            continue
        wins  = int(grp[label_col].sum())
        wr    = wins / n
        roi   = (wins * (100 / 110) - (n - wins)) / n
        units = wins * (100 / 110) - (n - wins)
        mname = MONTH_NAMES.get(int(month_num), str(month_num))
        print(
            f"  {mname:<8}  {n:>6}  {wr:>9.3f}  {roi:>+7.3f}  {units:>+8.1f}"
        )

    # Overall
    n     = len(sub)
    wins  = int(sub[label_col].sum())
    wr    = wins / n
    roi   = (wins * (100 / 110) - (n - wins)) / n
    units = wins * (100 / 110) - (n - wins)
    print(f"  {'-'*44}")
    print(
        f"  {'TOTAL':<8}  {n:>6}  {wr:>9.3f}  {roi:>+7.3f}  {units:>+8.1f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtesting — current model vs Bayesian shrinkage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--year", type=int, default=2025,
        help="Validation year (default: 2025)",
    )
    parser.add_argument(
        "--matrix", default=str(MATRIX_PATH),
        help=f"Feature matrix parquet path (default: {MATRIX_PATH})",
    )
    parser.add_argument(
        "--models-dir", default=str(MODELS_DIR),
        help=f"Models directory (default: {MODELS_DIR})",
    )
    parser.add_argument(
        "--bayes", action="store_true",
        help="Add Bayesian empirical shrinkage comparison",
    )
    parser.add_argument(
        "--n-prior", type=int, default=N_PRIOR_DEFAULT,
        help=f"Effective prior batters-faced for shrinkage (default: {N_PRIOR_DEFAULT})",
    )
    parser.add_argument(
        "--save-preds", action="store_true",
        help="Save per-game predictions to eval_predictions.csv",
    )
    parser.add_argument(
        "--edge-thresh", type=float, default=0.53,
        help="Monthly edge breakdown threshold (default: 0.53)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args       = parse_args()
    models_dir = Path(args.models_dir)
    matrix_path = Path(args.matrix)

    print("=" * 60)
    print("  evaluate_model.py  - Walk-Forward Backtesting Framework")
    print(f"  Val year : {args.year}")
    print(f"  Bayes    : {'ON  (n_prior=' + str(args.n_prior) + ')' if args.bayes else 'OFF'}")
    print("=" * 60)

    # ── Load feature matrix ───────────────────────────────────────────────
    if not matrix_path.exists():
        print(f"\n  ERROR: feature matrix not found at {matrix_path}")
        print("  Run: python build_feature_matrix.py")
        return

    df = pd.read_parquet(matrix_path, engine="pyarrow")
    print(f"\n  Loaded {len(df)} rows x {len(df.columns)} cols from {matrix_path}")

    # Normalise year column
    if "year" not in df.columns and "season" in df.columns:
        df["year"] = df["season"].astype(int)

    if "year" not in df.columns:
        print("  ERROR: feature matrix must contain 'year' or 'season' column.")
        return

    # Derive actual_home_win from home_margin if not present
    if "actual_home_win" not in df.columns and "home_margin" in df.columns:
        df["actual_home_win"] = np.where(
            df["home_margin"].isna(), np.nan,
            (df["home_margin"] > 0).astype(float),
        )

    val_df   = df[df["year"] == args.year].copy()
    train_df = df[df["year"].isin([2023, 2024])].copy()

    if len(val_df) == 0:
        print(f"\n  ERROR: no games found for year {args.year}.")
        print(f"  Years available: {sorted(df['year'].dropna().unique().astype(int))}")
        return

    n_rl = val_df["home_covers_rl"].notna().sum()
    n_ml = val_df["actual_home_win"].notna().sum() if "actual_home_win" in val_df.columns else 0

    print(f"  Val set  : {len(val_df)} games | RL labeled: {n_rl} | ML labeled: {n_ml}")
    print(f"  Train set: {len(train_df)} games")

    if n_rl < 20:
        print(f"\n  WARNING: only {n_rl} labeled RL games in {args.year}. "
              f"Metrics may be unreliable.")

    # ── Load models ───────────────────────────────────────────────────────
    print(f"\n  Loading models from {models_dir} ...")
    try:
        models = load_models(models_dir)
    except FileNotFoundError as exc:
        print(f"  ERROR: {exc}")
        print("  Run: python train_xgboost.py --ncv")
        return

    print(f"  Features : {len(models['feature_cols'])} columns")
    stk_status = "loaded" if models["stacking"] else "not found (using Platt only)"
    print(f"  Stacking : {stk_status}")

    all_metrics: list[pd.DataFrame] = []
    all_preds:   list[pd.DataFrame] = []

    # ── Current model ─────────────────────────────────────────────────────
    print(f"\n  Scoring {args.year} val set (current model) ...")
    scored_current = score_df(val_df, models)
    metrics_current = compute_monthly_metrics(scored_current, label="current")
    all_metrics.append(metrics_current)
    all_preds.append(scored_current.assign(_eval_mode="current"))

    # ── Bayesian shrinkage comparison ─────────────────────────────────────
    if args.bayes:
        print(f"\n  Applying Bayesian empirical shrinkage  (n_prior={args.n_prior}) ...")
        print(f"  Career priors from: {sorted(train_df['year'].dropna().unique().astype(int))}")

        val_bayes    = apply_bayes_shrinkage(val_df, train_df, n_prior=args.n_prior)
        scored_bayes = score_df(val_bayes, models)
        metrics_bayes = compute_monthly_metrics(scored_bayes, label="bayes")
        all_metrics.append(metrics_bayes)
        all_preds.append(scored_bayes.assign(_eval_mode="bayes"))

        # Show shrinkage magnitude per month for k_pct as a sanity-check
        if "home_sp_k_pct" in val_df.columns and "home_sp_k_pct" in val_bayes.columns:
            val_m   = val_df.copy()
            bayes_m = val_bayes.copy()
            val_m["_month"]   = pd.to_datetime(val_m["game_date"]).dt.month
            bayes_m["_month"] = pd.to_datetime(bayes_m["game_date"]).dt.month

            print(f"\n  Shrinkage effect - home_sp_k_pct mean by month:")
            print(f"  {'Month':<7}  {'Current':>9}  {'Bayes':>9}  {'Delta':>8}")
            print(f"  {'-'*38}")
            for m in sorted(val_m["_month"].dropna().unique()):
                m  = int(m)
                c  = val_m.loc[val_m["_month"] == m, "home_sp_k_pct"].mean()
                b  = bayes_m.loc[bayes_m["_month"] == m, "home_sp_k_pct"].mean()
                mn = MONTH_NAMES.get(m, str(m))
                print(f"  {mn:<7}  {c:>9.4f}  {b:>9.4f}  {b - c:>+8.4f}")

    # ── Print results ─────────────────────────────────────────────────────
    combined = pd.concat(all_metrics, ignore_index=True)
    datasets = combined["dataset"].unique().tolist()

    print(f"\n\n{'='*74}")
    print(f"  EVALUATION RESULTS - {args.year} validation set")
    print(f"  NOTE: Final XGBoost trained on 2023+2024+2025 (NCV mode).")
    print(f"        2025 metrics are IN-SAMPLE for XGBoost; use for Bayesian")
    print(f"        comparison only. For true OOS, run train_xgboost.py (standard)")
    print(f"        on 2023+2024 then re-evaluate.")
    print(f"{'='*74}")
    _print_metrics_table(combined, datasets)

    # ── Edge analysis (current model) ─────────────────────────────────────
    if "home_covers_rl" in scored_current.columns:
        print(f"\n\n{'='*60}")
        print(f"  EDGE ANALYSIS - {args.year} (current model)")
        print(f"{'='*60}")
        _print_edge_analysis(scored_current)
        _print_monthly_edge_analysis(scored_current, thresh=args.edge_thresh)

    # ── Save predictions ──────────────────────────────────────────────────
    if args.save_preds:
        combined_preds = pd.concat(all_preds, ignore_index=True)

        keep_cols = [
            "game_date", "home_team", "away_team", "year",
            "home_starter_name", "away_starter_name",
            "home_score", "away_score", "home_margin",
            "home_covers_rl", "total_runs",
            "rl_raw", "rl_cal", "rl_stacked",
            "ml_raw", "ml_cal", "actual_home_win",
            "tot_pred", "_eval_mode",
        ]
        save_cols = [c for c in keep_cols if c in combined_preds.columns]
        out_path  = Path("eval_predictions.csv")
        combined_preds[save_cols].to_csv(out_path, index=False)
        print(f"\n  Saved predictions -> {out_path}  ({len(combined_preds)} rows)")

    # Always save metrics
    metrics_path = Path("eval_metrics.csv")
    combined.to_csv(metrics_path, index=False)
    print(f"  Saved metrics     -> {metrics_path}")

    # ── Summary recommendation ────────────────────────────────────────────
    if args.bayes and len(all_metrics) == 2:
        curr_overall = (
            all_metrics[0]
            .query("model == 'RL' and period == 'OVERALL'")
        )
        bayes_overall = (
            all_metrics[1]
            .query("model == 'RL' and period == 'OVERALL'")
        )
        if len(curr_overall) > 0 and len(bayes_overall) > 0:
            d_auc   = (bayes_overall["auc"].values[0]
                       - curr_overall["auc"].values[0])
            d_brier = (curr_overall["brier"].values[0]
                       - bayes_overall["brier"].values[0])   # positive = Bayes better

            # Detect if we're almost certainly in-sample (AUC > 0.97)
            curr_auc   = curr_overall["auc"].values[0]
            in_sample  = curr_auc > 0.97

            print(f"\n{'='*60}")
            print(f"  RECOMMENDATION")
            print(f"{'='*60}")
            print(f"  RL model - overall {args.year}:")
            print(f"    dAUC   = {d_auc:+.4f}  "
                  f"({'Bayesian better' if d_auc > 0 else 'Current better'})")
            print(f"    dBrier = {d_brier:+.4f}  "
                  f"({'Bayesian better' if d_brier > 0 else 'Current better'})")

            if in_sample:
                print(f"\n  *** IN-SAMPLE WARNING ***")
                print(f"  AUC = {curr_auc:.4f} signals the XGBoost saw {args.year} data")
                print(f"  during training (NCV final model). This comparison is biased:")
                print(f"  the model memorised {args.year} pitcher stats, so Bayesian")
                print(f"  shrinkage (which perturbs those stats) looks worse regardless")
                print(f"  of its true OOS value.")
                print(f"\n  For a valid comparison, run a standard (non-NCV) model")
                print(f"  trained on 2023+2024 only, save it to models_oos/, then:")
                print(f"    python train_xgboost.py --no-early-stop")
                print(f"      (edit train/val split to train=2023+2024, val=2025)")
                print(f"    mkdir models_oos && cp models/*.json models_oos/")
                print(f"    python evaluate_model.py --bayes --models-dir models_oos")
            elif d_auc > 0.005 or d_brier > 0.003:
                print(f"\n  => Bayesian shrinkage shows meaningful OOS improvement.")
                print(f"     Next step: integrate shrinkage into build_pitcher_profile.py")
                print(f"     and rebuild the feature matrix with shrinkage baked in.")
            elif d_auc < -0.005:
                print(f"\n  => Current EWMA model outperforms Bayesian shrinkage OOS.")
                print(f"     Time-based EWMA alone is sufficient for this dataset size.")
            else:
                print(f"\n  => Results are statistically similar OOS.")
                print(f"     Try tuning: --n-prior 100, --n-prior 300")
                print(f"     Also check the April/May breakdown — shrinkage may help")
                print(f"     early-season even if the full-year OVERALL is flat.")

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
