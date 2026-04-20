"""
train_xgboost.py
================
Shadow Ensemble architecture (v5.1)
====================================
Three tree algorithms are trained on the same feature matrix and splits:

  Official pipeline
  -----------------
  • XGBoost  — Level-1 model whose OOF predictions feed the Level-2 stacker.
  • Bayesian Stacker (Level-2) — Hierarchical model using NUTS MCMC (NumPyro/JAX).
    Input: XGBoost raw prob + domain features + SP handedness segment (0–3).
    CRITICAL: LightGBM and CatBoost OOFs are intentionally NOT fed into the
    stacker.  The official betting signal is XGBoost → Bayesian Stacker only.

  Shadow models (inference-only, variance estimation)
  ----------------------------------------------------
  • LightGBM — leaf-wise trees (num_leaves=63), saved as lgb_shadow.json.
  • CatBoost — symmetric trees, saved as cat_shadow.cbm.
  These models generate independent raw probabilities at inference time.
  Together with the XGBoost raw prob they produce:
    ensemble_min   = min(xgb_raw, lgbm_raw, cat_raw)
    ensemble_max   = max(xgb_raw, lgbm_raw, cat_raw)
    model_spread   = ensemble_max − ensemble_min
  A wide spread signals high model uncertainty; a narrow spread signals
  consensus.  Neither value alters the official stacker output.

Models trained:
  1. home_covers_rl   (binary: home team covers -1.5 run line)
  2. total_runs       (regression: total runs scored)
  3. actual_home_win  (binary: home team wins outright — auxiliary)

Calibration — Platt Scaling (Logistic / Sigmoid)
-------------------------------------------------
Isotonic regression has high capacity and overfits on our 3-season dataset
(~2000–2500 labeled games).  A sigmoid (logistic) calibration layer with only
2 parameters eliminates overfit while preserving rank order (AUC unchanged).

Level-2 Bayesian Hierarchical Stacker  (XGBoost-only input)
------------------------------------------------------------
Model:  y ~ Bernoulli(σ(α + β·logit(p_xgb) + δ_j + γᵀ·x))
Segment j: SP handedness matchup {LvL=0, LvR=1, RvL=2, RvR=3}
Priors: α~N(0,1)  β~N(1,.5)  σ_δ~HalfCauchy(1)  δ~N(0,σ_δ)  γ~N(0,.3)
Input: 1 raw prob + 11 domain features + 1 segment

Training modes
--------------
Default (no --ncv):
  Train 2023+2024, validate 2025.

Nested Cross-Validation (--ncv):
  Fold 1: train 2023       → validate 2024
  Fold 2: train 2023+2024  → validate 2025
  Final : train 2023+2024+2025 (2026 held out entirely)

Outputs:
  models/xgb_rl.json           XGBoost run-line model          [official]
  models/xgb_total.json        XGBoost total runs model         [official]
  models/xgb_ml.json           XGBoost moneyline model          [official]
  models/lgb_shadow.json       LightGBM shadow (native format)  [shadow]
  models/cat_shadow.cbm        CatBoost shadow (native format)  [shadow]
  models/lgbm_rl.pkl           LightGBM sklearn wrapper         [shadow/compat]
  models/lgbm_total.pkl        LightGBM total runs wrapper      [shadow/compat]
  models/lgbm_ml.pkl           LightGBM moneyline wrapper       [shadow/compat]
  models/cat_rl.pkl            CatBoost sklearn wrapper         [shadow/compat]
  models/cat_total.pkl         CatBoost total runs wrapper      [shadow/compat]
  models/cat_ml.pkl            CatBoost moneyline wrapper       [shadow/compat]
  models/feature_cols.json     ordered feature column list
  models/calibrator_rl.pkl     Platt sigmoid calibrator (run-line)
  models/calibrator_ml.pkl     Platt sigmoid calibrator (moneyline)
  models/stacking_lr_rl.pkl    Level-2 Bayesian hierarchical stacker (XGBoost-only)
  models/stacking_lr_rl.npz   Full NUTS posterior trace (alpha, beta, delta, gamma, sigma_delta)
  xgb_val_predictions.csv      validation set predictions
  xgb_feature_importance.csv   gain-based feature importances
  xgb_ncv_results.csv          per-fold NCV metrics (--ncv only)

Usage:
  python train_xgboost.py
  python train_xgboost.py --matrix feature_matrix.parquet
  python train_xgboost.py --no-early-stop
  python train_xgboost.py --ncv
  python train_xgboost.py --ncv --no-early-stop
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss,
    mean_absolute_error, mean_squared_error,
)

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("pip install xgboost")

try:
    import lightgbm as lgb
    _LGBM = True
except ImportError:
    lgb = None
    _LGBM = False

try:
    import catboost as cb
    _CATBOOST = True
except ImportError:
    cb = None
    _CATBOOST = False

# ── GPU availability (used to set device flags for all three frameworks) ────
try:
    import cupy as _cp_train
    _GPU = _cp_train.cuda.is_available()
    if _GPU:
        try:
            _probe = _cp_train.random.standard_normal(1); del _probe
        except Exception:
            _GPU = False
except ImportError:
    _GPU = False

# ── JAX / NumPyro (Bayesian hierarchical Level-2 stacker) ───────────────────
# Routes JAX to the RTX 5080 only when JAX's *own* CUDA backend is present
# (requires jax[cuda12_pip]).  CuPy's _GPU flag is not sufficient — CuPy and
# JAX have independent CUDA installations.  Falls back to JAX-CPU (or plain
# LR stacker) when NumPyro or JAX is unavailable.
try:
    import jax as _jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    # Probe whether JAX's own GPU backend exists (distinct from CuPy)
    _jax_gpu_ok = False
    if _GPU:
        try:
            _jax.config.update("jax_platform_name", "gpu")
            jnp.zeros(1)          # triggers backend initialisation
            _jax_gpu_ok = True
        except RuntimeError:
            _jax.config.update("jax_platform_name", "cpu")
    _JAX_PLATFORM = "gpu" if _jax_gpu_ok else "cpu"
    _NUMPYRO = True
except ImportError:
    jnp = None; numpyro = None; dist = None; MCMC = None; NUTS = None
    _JAX_PLATFORM = "cpu"
    _NUMPYRO = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(exist_ok=True)

# Columns that are NOT features (labels, identifiers, administrative)
NON_FEATURE_COLS = {
    "game_date", "game_pk", "home_team", "away_team", "season", "year",
    "home_starter_name", "away_starter_name", "split",
    # Labels
    "actual_home_win", "actual_game_total", "actual_f5_total",
    "actual_f3_total", "actual_f1_total",
    "home_score", "away_score", "home_margin",
    "home_covers_rl", "away_covers_rl", "total_runs",
    # Pipeline metadata
    "source", "pull_timestamp",
    # Vegas / market lines — excluded: MC engine handles this signal
    "close_ml_home", "close_ml_away", "open_total", "close_total",
    "vegas_implied_home", "vegas_implied_away",
}

# ── Pass 1 feature drops ──────────────────────────────────────────────────────
# Applied before any training.  Two categories:
#
# (A) 100% NULL — pybaseball functions not available in the installed version.
#     Keeping these wastes colsample_bytree budget and adds no signal.
#     mc_expected_runs is intentionally excluded from this list: it stays as a
#     feature because it is populated at inference time (NaN during training is
#     handled by XGBoost's default-branch routing).
#
# (B) Collinear derivations — ratios / products of features already in the
#     matrix.  For GBDTs the redundancy doesn't cause instability, but it burns
#     ~8 colsample slots that signal features could occupy.
_PASS1_DROP = {
    # ── (A) 100% null: pitcher run value / pitch movement (pybaseball unavailable)
    "home_sp_swing_rv_per100",    "away_sp_swing_rv_per100",    "sp_swing_rv_diff",
    "home_sp_take_rv_per100",     "away_sp_take_rv_per100",     "sp_take_rv_diff",
    "home_sp_ff_h_break_inch",    "away_sp_ff_h_break_inch",    "sp_ff_h_break_diff",
    "home_sp_ff_v_break_inch",    "away_sp_ff_v_break_inch",    "sp_ff_v_break_diff",
    # ── (A) 100% null: arsenal stats (unavailable — all derived interactions also null)
    "home_sp_arsenal_weighted_rv", "away_sp_arsenal_weighted_rv", "sp_arsenal_rv_diff",
    "home_sp_primary_whiff_pct",   "away_sp_primary_whiff_pct",   "sp_primary_whiff_diff",
    "home_sp_primary_putaway_pct", "away_sp_primary_putaway_pct", "sp_primary_putaway_diff",
    "home_sp_arsenal_quality_ratio","away_sp_arsenal_quality_ratio","sp_arsenal_quality_ratio_diff",
    # ── (B) Collinear: K/BB ratio is a ratio of k_pct / bb_pct, which the model
    #    already sees directly alongside k_minus_bb.  All 6 ratio cols are redundant.
    "home_sp_k_bb_ratio",    "away_sp_k_bb_ratio",    "sp_k_bb_ratio_diff",
    "home_sp_k_bb_ratio_10d","away_sp_k_bb_ratio_10d","sp_k_bb_ratio_10d_diff",
    # ── (B) Collinear: local_hour is a simple offset of game_hour_et; circadian_edge
    #    already captures the timezone mismatch.  Raw hour adds no independent signal.
    "home_game_local_hour",  "away_game_local_hour",
    # ── (C) Zero gain across all 3 models (ML, RL, Total) in NCV run — dead weight
    "away_sp_il_return_flag", "away_sp_starts_since_il", "home_sp_starts_since_il",
    "is_day_game",            "mc_expected_runs",
}
NON_FEATURE_COLS = NON_FEATURE_COLS | _PASS1_DROP

# Diff columns (home - away) that must be negated when building the away perspective.
# Covers all *_diff and edge columns that are signed relative to home team.
_DIFF_COLS_TO_NEGATE = [
    "sp_k_pct_diff", "sp_xwoba_diff", "sp_xrv_diff", "sp_velo_diff",
    "sp_age_diff", "sp_kminusbb_diff", "sp_k_pct_10d_diff", "sp_xwoba_10d_diff",
    "sp_bb_pct_10d_diff", "batting_matchup_edge", "batting_matchup_edge_10d",
    "bp_era_diff", "bp_k9_diff", "bp_whip_diff", "circadian_edge",
]

# XGBoost base parameters
XGB_BASE = {
    "tree_method":       "hist",
    "device":            "cuda" if _GPU else "cpu",
    "n_jobs":            -1,
    "random_state":      42,
    "learning_rate":     0.04,
    "max_depth":         5,
    "min_child_weight":  20,
    "subsample":         0.80,
    "colsample_bytree":  0.75,
    "reg_alpha":         0.5,
    "reg_lambda":        2.0,
    "n_estimators":      600,
}

# LightGBM base parameters — leaf-wise growth, GPU when available
LGBM_BASE = {
    "n_estimators":      600,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "max_depth":         -1,
    "min_child_samples": 20,
    "subsample":         0.80,
    "colsample_bytree":  0.75,
    "reg_alpha":         0.5,
    "reg_lambda":        2.0,
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
    **({"device": "gpu"} if _GPU else {}),
}

# CatBoost base parameters — symmetric trees, native categoricals
CATBOOST_BASE = {
    "iterations":    600,
    "learning_rate": 0.05,
    "depth":         6,
    "l2_leaf_reg":   3.0,
    "random_seed":   42,
    "verbose":       0,
    **({"task_type": "GPU"} if _GPU else {}),
}

# ---------------------------------------------------------------------------
# ABS-era feature weights
# ---------------------------------------------------------------------------
ABS_STUFF_WEIGHT: float = 1.35
ABS_STUFF_COLS = [
    "home_sp_whiff_pctl", "away_sp_whiff_pctl",
    "home_sp_xera_pctl",  "away_sp_xera_pctl",
]

# Year-decay weights — upweight recent seasons to combat roster drift.
# Pattern: model over-trusts stale 2023/2024 team identities (BAL, PIT, LAA).
# 2025 data is 2.1x more influential than 2023 after both multipliers apply.
YEAR_DECAY_WEIGHTS: dict[int, float] = {
    2023: 0.70,
    2024: 1.00,
    2025: 1.50,
}
YEAR_DECAY_DEFAULT: float = 1.00   # any unseen year gets neutral weight

EARLY_STOPPING_ROUNDS = 40

# Multi-quantile targets for total-runs regression.
# The model produces 3 outputs per row: (floor, median, ceiling).
TOT_QUANTILE_ALPHAS = [0.10, 0.50, 0.90]   # 10th / 50th / 90th percentiles
TOT_Q_IDX = {"floor": 0, "median": 1, "ceiling": 2}
BREAKEVEN_PROB        = 110 / 210

NCV_FOLDS = [
    {"train_years": [2023],       "val_year": 2024},
    {"train_years": [2023, 2024], "val_year": 2025},
]
NCV_ALL_YEARS = [2023, 2024, 2025]

STACKING_FEATURES = [
    "sp_k_pct_diff", "sp_xwoba_diff", "sp_kminusbb_diff",
    "bp_era_diff", "bp_whip_diff", "batting_matchup_edge",
    "home_sp_il_return_flag", "away_sp_il_return_flag",
    "sp_k_pct_10d_diff", "sp_xwoba_10d_diff", "batting_matchup_edge_10d",
    # Vegas-model gap: when model disagrees with closing ML by >10%,
    # empirical win rate jumps from 57% → 65% (2025 val set analysis).
    # Making this explicit lets the stacker LR learn to boost conviction
    # on those contrarian calls.  Computed as xgb_ml_raw - true_home_prob
    # at training time; at inference it comes from the stacking_feats dict.
    "ml_model_vs_vegas_gap",
]

# ---------------------------------------------------------------------------
# SP HANDEDNESS MATCHUP SEGMENTS  (used by Bayesian hierarchical stacker)
# ---------------------------------------------------------------------------
# Encoding  home_sp_p_throws_R * 2 + away_sp_p_throws_R  → 4 groups:
#   0 = LvL  (home LHP vs away LHP)
#   1 = LvR  (home LHP vs away RHP)
#   2 = RvL  (home RHP vs away LHP)
#   3 = RvR  (home RHP vs away RHP)  ← most common matchup
_SP_HANDEDNESS_N_SEGMENTS = 4
_SP_SEG_LABELS            = ["LvL", "LvR", "RvL", "RvR"]


def _derive_segment_id(df: pd.DataFrame) -> np.ndarray:
    """
    Compute per-game SP handedness matchup segment (0–3).
    Falls back to 3 (RvR — most common) when columns are missing.
    """
    if ("home_sp_p_throws_R" not in df.columns or
            "away_sp_p_throws_R" not in df.columns):
        return np.full(len(df), 3, dtype=np.int32)
    h = df["home_sp_p_throws_R"].fillna(1).astype(int).values
    a = df["away_sp_p_throws_R"].fillna(1).astype(int).values
    return (h * 2 + a).astype(np.int32)


_STACKING_DIFF_COLS = [
    "sp_k_pct_diff", "sp_xwoba_diff", "sp_kminusbb_diff",
    "bp_era_diff", "bp_whip_diff", "batting_matchup_edge",
    "sp_k_pct_10d_diff", "sp_xwoba_10d_diff", "batting_matchup_edge_10d",
    "ml_model_vs_vegas_gap",
]

def _flip_stacking_feats(feat_df: pd.DataFrame) -> pd.DataFrame:
    away = feat_df.copy()
    for col in _STACKING_DIFF_COLS:
        if col in away.columns:
            away[col] = -feat_df[col].values
    if "home_sp_il_return_flag" in away.columns and "away_sp_il_return_flag" in away.columns:
        away["home_sp_il_return_flag"] = feat_df["away_sp_il_return_flag"].values
        away["away_sp_il_return_flag"] = feat_df["home_sp_il_return_flag"].values
    return away

def _derive_segment_id_away(df: pd.DataFrame) -> np.ndarray:
    if "home_sp_p_throws_R" not in df.columns or "away_sp_p_throws_R" not in df.columns:
        return np.full(len(df), 3, dtype=np.int32)
    h = df["home_sp_p_throws_R"].fillna(1).astype(int).values
    a = df["away_sp_p_throws_R"].fillna(1).astype(int).values
    return (a * 2 + h).astype(np.int32)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def build_sample_weights(X: pd.DataFrame,
                         year_series: pd.Series | None = None) -> np.ndarray:
    """
    Combined sample weight = ABS_STUFF_WEIGHT × YEAR_DECAY_WEIGHT.

    ABS layer  : rows with non-null Statcast stuff metrics get 1.35x.
    Year-decay : 2023=0.70x  2024=1.00x  2025=1.50x — combats roster drift
                 so recent seasons dominate (BAL/PIT/LAA regression pattern).

    Both multipliers stack: a 2025 ABS row → 1.35 × 1.50 = 2.025x weight.

    Args:
        X           : feature DataFrame (year column stripped — pass separately).
        year_series : pd.Series of year values aligned to X's index.
    """
    weights = np.ones(len(X), dtype=np.float32)

    # ── Year-decay layer ──────────────────────────────────────────────────
    if year_series is not None and len(year_series) == len(X):
        year_w = (year_series
                  .map(YEAR_DECAY_WEIGHTS)
                  .fillna(YEAR_DECAY_DEFAULT)
                  .astype(np.float32)
                  .values)
        weights *= year_w
        unique_yrs = sorted(year_series.dropna().unique())
        yr_summary = "  ".join(
            f"{int(y)}={YEAR_DECAY_WEIGHTS.get(int(y), YEAR_DECAY_DEFAULT):.2f}x"
            for y in unique_yrs
        )
        print(f"    Year-decay weights: {yr_summary}")

    # ── ABS-era Statcast layer ────────────────────────────────────────────
    present = [c for c in ABS_STUFF_COLS if c in X.columns]
    if present:
        has_stuff = X[present].notna().any(axis=1).values
        weights[has_stuff] *= ABS_STUFF_WEIGHT
        n_up = int(has_stuff.sum())
        print(f"    ABS sample weights: {n_up}/{len(X)} rows "
              f"({100*n_up/max(len(X),1):.1f}%) upweighted to {ABS_STUFF_WEIGHT}x")

    return weights


def prep_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == object:
            try:
                X[col] = pd.to_numeric(X[col], errors="coerce")
            except Exception:
                X = X.drop(columns=[col])
                feature_cols = [c for c in feature_cols if c != col]
    return X, feature_cols


def log_metrics(name, y_true, y_pred_prob=None, y_pred_reg=None):
    print(f"\n  --- {name} ---")
    if y_pred_reg is not None and np.ndim(y_pred_reg) == 2:
        y_pred_reg = y_pred_reg[:, y_pred_reg.shape[1] // 2]   # quantile model -> take median col
    if y_pred_prob is not None:
        auc   = roc_auc_score(y_true, y_pred_prob)
        ll    = log_loss(y_true, y_pred_prob)
        brier = brier_score_loss(y_true, y_pred_prob)
        acc   = ((y_pred_prob > 0.50) == y_true).mean()
        print(f"    AUC-ROC     : {auc:.4f}")
        print(f"    Log-loss    : {ll:.4f}")
        print(f"    Brier score : {brier:.4f}")
        print(f"    Accuracy@50%: {acc:.4f}")
    if y_pred_reg is not None:
        mae  = mean_absolute_error(y_true, y_pred_reg)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_reg))
        print(f"    MAE         : {mae:.4f}")
        print(f"    RMSE        : {rmse:.4f}")


def edge_analysis(val_df, prob_col, label_col="home_covers_rl",
                  thresholds=(0.50, 0.525, 0.55, 0.575, 0.60)):
    print(f"\n  Edge analysis (model_prob >= threshold => bet, -110 juice):")
    print(f"  {'threshold':>10}  {'n_bets':>7}  {'win_rate':>9}  {'roi':>7}  {'units':>7}")
    print(f"  {'-'*45}")
    for thresh in thresholds:
        bets = val_df[val_df[prob_col] >= thresh]
        n = len(bets)
        if n < 20:
            continue
        wins = bets[label_col].sum()
        wr   = wins / n
        roi  = (wins * (100 / 110) - (n - wins)) / n
        print(f"  {thresh:>10.3f}  {n:>7}  {wr:>9.3f}  {roi:>+7.3f}  "
              f"{(wins*(100/110)-(n-wins)):>+7.1f}")


# ---------------------------------------------------------------------------
# MODEL TRAINING — XGBoost
# ---------------------------------------------------------------------------

def train_binary(X_train, y_train, X_val, y_val,
                 label, early_stop, scale_pos=1.0, year_series=None):
    params = {**XGB_BASE, "objective": "binary:logistic",
              "eval_metric": ["logloss", "auc"], "scale_pos_weight": scale_pos}
    callbacks = ([xgb.callback.EarlyStopping(rounds=EARLY_STOPPING_ROUNDS,
                  metric_name="auc", maximize=True, save_best=True)]
                 if early_stop else [])
    sw = build_sample_weights(X_train, year_series)
    model = xgb.XGBClassifier(**params, callbacks=callbacks)
    model.fit(X_train, y_train, sample_weight=sw,
              eval_set=[(X_val, y_val)], verbose=False)
    best = getattr(model, "best_iteration", -1)
    print(f"    XGB {label}: best_iter={best} | scale_pos={scale_pos:.2f}")
    return model


def train_regression(X_train, y_train, X_val, y_val, label, early_stop,
                     year_series=None):
    """
    Multi-quantile regression for total runs (v5.1).

    Objective: reg:quantileerror  with quantile_alpha=[0.10, 0.50, 0.90]
    Strategy:  multi_output_tree — one tree per round outputs all 3 quantiles
               simultaneously (GPU-accelerated, 3x more efficient than
               one_output_per_tree at equal round count).

    Output shape: predict() → np.ndarray (n_samples, 3)
        col 0 = 10th-pct floor   (TOT_Q_IDX["floor"])
        col 1 = 50th-pct median  (TOT_Q_IDX["median"])  ← primary prediction
        col 2 = 90th-pct ceiling (TOT_Q_IDX["ceiling"])

    Early stopping monitors the mean pinball loss ("quantile" metric).
    """
    params = {
        **XGB_BASE,
        "objective":       "reg:quantileerror",
        "quantile_alpha":  TOT_QUANTILE_ALPHAS,
        "multi_strategy":  "multi_output_tree",
        "eval_metric":     "quantile",
    }
    params.pop("scale_pos_weight", None)
    callbacks = (
        [xgb.callback.EarlyStopping(rounds=EARLY_STOPPING_ROUNDS,
                                    metric_name="quantile",
                                    maximize=False, save_best=True)]
        if early_stop else []
    )
    sw = build_sample_weights(X_train, year_series)
    model = xgb.XGBRegressor(**params, callbacks=callbacks)
    model.fit(X_train, y_train, sample_weight=sw,
              eval_set=[(X_val, y_val)], verbose=False)
    print(f"    XGB {label}: best_iter={getattr(model,'best_iteration',-1)}")
    return model


# ---------------------------------------------------------------------------
# MODEL TRAINING — LightGBM
# ---------------------------------------------------------------------------

def train_binary_lgbm(X_train, y_train, X_val, y_val,
                      label, early_stop, scale_pos=1.0, year_series=None):
    if not _LGBM:
        return None
    sw = build_sample_weights(X_train, year_series)
    try:
        params = {**LGBM_BASE, "objective": "binary", "metric": "auc",
                  "scale_pos_weight": scale_pos}
        cb_list = (
            [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
             lgb.log_evaluation(-1)]
            if early_stop else [lgb.log_evaluation(-1)]
        )
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sw,
                  eval_set=[(X_val, y_val)], callbacks=cb_list)
        best = getattr(model, "best_iteration_", -1)
        print(f"    LGBM {label}: best_iter={best} | scale_pos={scale_pos:.2f}")
        return model
    except Exception as e:
        print(f"    [WARN] LGBM {label} failed ({e}), skipping.")
        return None


def train_regression_lgbm(X_train, y_train, X_val, y_val, label, early_stop,
                          year_series=None):
    if not _LGBM:
        return None
    sw = build_sample_weights(X_train, year_series)
    try:
        params = {**LGBM_BASE, "objective": "regression", "metric": "rmse"}
        cb_list = (
            [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
             lgb.log_evaluation(-1)]
            if early_stop else [lgb.log_evaluation(-1)]
        )
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, sample_weight=sw,
                  eval_set=[(X_val, y_val)], callbacks=cb_list)
        print(f"    LGBM {label}: best_iter={getattr(model,'best_iteration_',-1)}")
        return model
    except Exception as e:
        print(f"    [WARN] LGBM {label} failed ({e}), skipping.")
        return None


# ---------------------------------------------------------------------------
# MODEL TRAINING — CatBoost
# ---------------------------------------------------------------------------

def _to_cat_array(X) -> np.ndarray:
    """
    Convert a pandas DataFrame to a float64 numpy array safe for CatBoost.
    Replaces pd.NA / pd.NaT with np.nan (CatBoost cannot handle pd.NA sentinel).
    """
    return X.astype("float64").values


def train_binary_catboost(X_train, y_train, X_val, y_val,
                           label, early_stop, scale_pos=1.0, year_series=None):
    if not _CATBOOST:
        return None
    sw = build_sample_weights(X_train, year_series)
    try:
        # class_weights goes to the CONSTRUCTOR, not fit() — CatBoost 1.x API
        neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
        cw = [1.0, float(neg / max(pos, 1)) * scale_pos] if scale_pos != 1.0 else None

        params = {**CATBOOST_BASE, "loss_function": "Logloss", "eval_metric": "AUC"}
        if cw is not None:
            params["class_weights"] = cw
        es = EARLY_STOPPING_ROUNDS if early_stop else None

        model = cb.CatBoostClassifier(**params, early_stopping_rounds=es)
        model.fit(
            _to_cat_array(X_train), y_train.values,
            sample_weight=sw,
            eval_set=(_to_cat_array(X_val), y_val.values),
            use_best_model=early_stop,
            verbose=False,
        )
        best = model.get_best_iteration() if early_stop else -1
        print(f"    CAT {label}: best_iter={best} | scale_pos={scale_pos:.2f}")
        return model
    except Exception as e:
        print(f"    [WARN] CatBoost {label} failed ({e}), skipping.")
        return None


def train_regression_catboost(X_train, y_train, X_val, y_val, label, early_stop,
                              year_series=None):
    if not _CATBOOST:
        return None
    sw = build_sample_weights(X_train, year_series)
    try:
        params = {**CATBOOST_BASE, "loss_function": "RMSE", "eval_metric": "RMSE"}
        es = EARLY_STOPPING_ROUNDS if early_stop else None
        model = cb.CatBoostRegressor(**params, early_stopping_rounds=es)
        model.fit(
            _to_cat_array(X_train), y_train.values,
            sample_weight=sw,
            eval_set=(_to_cat_array(X_val), y_val.values),
            use_best_model=early_stop,
            verbose=False,
        )
        print(f"    CAT {label}: best_iter={model.get_best_iteration() if early_stop else -1}")
        return model
    except Exception as e:
        print(f"    [WARN] CatBoost {label} failed ({e}), skipping.")
        return None


# ---------------------------------------------------------------------------
# PLATT SCALING
# ---------------------------------------------------------------------------

def fit_platt_calibrator(raw_probs, true_labels, out_path, label=""):
    platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=500)
    platt.fit(raw_probs.reshape(-1, 1), true_labels)
    out_path.write_bytes(pickle.dumps(platt))
    cal = platt.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    print(f"  Saved Platt [{label}] -> {out_path}  "
          f"slope={platt.coef_[0][0]:.4f}  "
          f"mean_raw={raw_probs.mean():.4f} -> mean_cal={cal.mean():.4f}")
    return platt


def calibration_report(raw_probs, true_labels, calibrator, label=""):
    cal = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    bins   = [0, .30, .35, .40, .45, .50, .55, .60, .65, 1.0]
    labels = ["<.30",".30-.35",".35-.40",".40-.45",".45-.50",
              ".50-.55",".55-.60",".60-.65",">.65"]
    print(f"\n  Calibration [{label}]")
    print(f"  {'bucket':>10}  {'n':>6}  {'actual':>8}  {'raw':>8}  {'cal':>8}")
    print(f"  {'-'*48}")
    idx = np.clip(np.digitize(raw_probs, bins) - 1, 0, len(labels)-1)
    raw_errs, cal_errs = [], []
    for i, lbl in enumerate(labels):
        m = idx == i
        if m.sum() < 5: continue
        act = true_labels[m].mean()
        rm  = raw_probs[m].mean()
        cm  = cal[m].mean()
        raw_errs.append(abs(act-rm)); cal_errs.append(abs(act-cm))
        print(f"  {lbl:>10}  {m.sum():>6}  {act:>8.3f}  {rm:>8.3f}  {cm:>8.3f}")
    if raw_errs:
        imp = (np.mean(raw_errs)-np.mean(cal_errs))/np.mean(raw_errs)*100
        print(f"  Calibration improved {imp:.1f}%  "
              f"({np.mean(raw_errs):.3f} -> {np.mean(cal_errs):.3f})")


# ---------------------------------------------------------------------------
# STACKING MODEL  (Level-2 Bayesian Hierarchical — NumPyro / JAX backend)
# ---------------------------------------------------------------------------
# Model:
#   y_i ~ Bernoulli(σ(α + β·logit(p_global_i) + δ_{j(i)} + γᵀ·x_i))
#
# Priors:
#   α     ~ Normal(0, 1)           intercept
#   β     ~ Normal(1, 0.5)         centered at 1 → trust XGBoost by default;
#                                  σ=0.5 lets data move β ±1 unit from prior
#   σ_δ  ~ HalfCauchy(0, 1)       Polson-Scott hyperprior: heavy tail allows
#                                  large group effects when data justifies,
#                                  but mass near 0 → complete pooling when n_j ↓
#   δ_j  ~ Normal(0, σ_δ)         per-segment offset (partial pooling)
#                                  Shrinkage: λ_j = n_j·σ_δ²/(n_j·σ_δ²+1)→0 as n_j→0
#   γ_k  ~ Normal(0, 0.3)         domain feature weights (weakly regularised;
#                                  σ=0.3 ≈ 3× typical standardised feature effect)
#
# Segment j: SP handedness matchup {0=LvL, 1=LvR, 2=RvL, 3=RvR}
# ---------------------------------------------------------------------------


def _numpyro_stacker_model(p_global, segment_id, X_domain,
                            n_segments, n_domain, y_obs=None):
    """NumPyro generative model for the Bayesian hierarchical stacker."""
    alpha = numpyro.sample("alpha", dist.Normal(0.0, 1.0))
    beta  = numpyro.sample("beta",  dist.Normal(1.0, 0.5))

    sigma_delta = numpyro.sample("sigma_delta", dist.HalfCauchy(1.0))
    with numpyro.plate("segments", n_segments):
        delta = numpyro.sample("delta", dist.Normal(0.0, sigma_delta))

    with numpyro.plate("features", n_domain):
        gamma = numpyro.sample("gamma", dist.Normal(0.0, 0.3))

    logit_p = jnp.log(jnp.clip(p_global, 1e-6, 1.0 - 1e-6)) - \
              jnp.log(1.0 - jnp.clip(p_global, 1e-6, 1.0 - 1e-6))
    theta = alpha + beta * logit_p + delta[segment_id] + X_domain @ gamma

    with numpyro.plate("data", len(p_global)):
        numpyro.sample("y", dist.Bernoulli(logits=theta), obs=y_obs)


class BayesianStacker:
    """
    Level-2 Bayesian Hierarchical Model, fitted on OOF data via NUTS MCMC.

    Stores posterior means for closed-form inference:
        P = σ(α̂ + β̂·logit(p_global) + δ̂_j + γ̂ᵀ·x)

    Backward-compatible interface:
      • .stacking_feature_names  — recognised by _get_p_model in run_today.py
      • .fill_values             — imputation dict for missing domain features
      • .predict(xgb_raw, feat_df, segment_id=...)  → P array
      • .predict_proba(...)      → [[1-P, P]] array
    """

    def __init__(self, alpha, beta, delta, gamma,
                 stacking_feature_names, fill_values,
                 n_segments=4, posterior_path=None):
        self.alpha              = float(alpha)
        self.beta               = float(beta)
        self.delta              = np.asarray(delta,  dtype=float)   # (n_segments,)
        self.gamma              = np.asarray(gamma,  dtype=float)   # (n_domain,)
        self.stacking_feature_names = stacking_feature_names
        self.fill_values        = fill_values
        self.n_segments         = n_segments
        self.posterior_path     = posterior_path   # path to .npz trace file

    # ------------------------------------------------------------------
    def _build_X_domain(self, feat_df: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(index=feat_df.index)
        for col in self.stacking_feature_names:
            if col in feat_df.columns:
                X[col] = feat_df[col]
            else:
                X[col] = self.fill_values.get(col, 0.0)   # missing at eval time — use training fill
        for col, val in self.fill_values.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)
        return X.values.astype(float)

    # ------------------------------------------------------------------
    def predict(self, xgb_raw, feat_df, segment_id=None,
                lgbm_raw=None, cat_raw=None) -> np.ndarray:
        """
        Returns calibrated P(home covers RL) for each row.

        xgb_raw    : 1-D array of XGBoost raw probs, shape (n,)
        feat_df    : DataFrame with domain features, n rows
        segment_id : int array (n,) or scalar; None defaults to RvR=3
        lgbm_raw, cat_raw : accepted for interface parity, not used
        """
        xgb_raw = np.asarray(xgb_raw, dtype=float).ravel()
        n = len(xgb_raw)

        if segment_id is None:
            seg = np.full(n, self.n_segments - 1, dtype=int)   # RvR default
        else:
            seg = np.asarray(segment_id, dtype=int).ravel()
            if len(seg) == 1 and n > 1:
                seg = np.full(n, int(seg[0]), dtype=int)

        X_dom = self._build_X_domain(feat_df)

        logit_p = (np.log(np.clip(xgb_raw, 1e-6, 1.0 - 1e-6)) -
                   np.log(1.0 - np.clip(xgb_raw, 1e-6, 1.0 - 1e-6)))
        theta = (self.alpha
                 + self.beta    * logit_p
                 + self.delta[seg]
                 + X_dom @ self.gamma)
        return 1.0 / (1.0 + np.exp(-theta))

    def predict_proba(self, xgb_raw, feat_df, segment_id=None,
                      lgbm_raw=None, cat_raw=None) -> np.ndarray:
        p = self.predict(xgb_raw, feat_df, segment_id, lgbm_raw, cat_raw)
        return np.column_stack([1.0 - p, p])


# ---------------------------------------------------------------------------

def _train_bayes_fallback_lr(oof_xgb_probs, X_domain, oof_labels,
                              feat_names, fill_vals, out_path):
    """Plain LR fallback used when NumPyro is unavailable."""
    X_stack = np.hstack([oof_xgb_probs.reshape(-1, 1), X_domain])
    lr = LogisticRegression(C=10, solver="lbfgs", max_iter=1000)
    lr.fit(X_stack, oof_labels)
    model = BayesianStacker(
        alpha  = float(lr.intercept_[0]),
        beta   = float(lr.coef_[0, 0]),
        delta  = np.zeros(_SP_HANDEDNESS_N_SEGMENTS),
        gamma  = lr.coef_[0, 1:],
        stacking_feature_names = feat_names,
        fill_values            = fill_vals,
    )
    out_path.write_bytes(pickle.dumps(model))
    print(f"  [Fallback LR] BayesianStacker saved -> {out_path}")
    return model


def train_bayesian_stacker(
    oof_xgb_probs:    np.ndarray,
    oof_feat_df:      pd.DataFrame,
    oof_labels:       np.ndarray,
    oof_segment_ids:  np.ndarray,
    out_path:         Path,
    n_segments:       int = _SP_HANDEDNESS_N_SEGMENTS,
    num_warmup:       int = 500,
    num_samples:      int = 1_000,
    num_chains:       int = 2,
) -> BayesianStacker:
    """
    Fit Level-2 Bayesian Hierarchical stacker via NUTS MCMC on OOF data.
    Falls back to logistic regression when JAX / NumPyro is unavailable.

    Parameters
    ----------
    oof_xgb_probs   : XGBoost raw OOF probabilities, shape (n,)
    oof_feat_df     : domain feature DataFrame, n rows
    oof_labels      : binary labels (0/1), shape (n,)
    oof_segment_ids : SP handedness segment (0-3), shape (n,)
    out_path        : destination path for the pickled BayesianStacker
    n_segments      : number of partial-pooling groups (default 4)
    num_warmup      : NUTS warm-up steps per chain
    num_samples     : posterior samples per chain
    num_chains      : parallel chains (requires JAX)
    """
    feat_names = [c for c in STACKING_FEATURES if c in oof_feat_df.columns]
    fill_vals  = {c: float(oof_feat_df[c].median()) for c in feat_names}

    X_feat = oof_feat_df[feat_names].copy()
    for col, val in fill_vals.items():
        X_feat[col] = X_feat[col].fillna(val)
    X_domain = X_feat.values.astype(float)
    n_domain = X_domain.shape[1]

    if not _NUMPYRO:
        print("  [WARN] NumPyro not available — using LR fallback stacker")
        return _train_bayes_fallback_lr(
            oof_xgb_probs, X_domain, oof_labels, feat_names, fill_vals, out_path
        )

    seg_counts = {j: int((oof_segment_ids == j).sum()) for j in range(n_segments)}
    print(f"\n{'='*60}")
    print(f"  BAYESIAN HIERARCHICAL STACKER  (NumPyro / NUTS — {_JAX_PLATFORM.upper()})")
    print(f"{'='*60}")
    print(f"  OOF rows : {len(oof_xgb_probs):,}  |  domain features : {n_domain}")
    print(f"  Segments : " + "  ".join(
        f"{_SP_SEG_LABELS[j]}={seg_counts[j]}" for j in range(n_segments)
    ))
    print(f"  MCMC     : {num_chains} chains × ({num_warmup} warmup + {num_samples} samples)")

    import jax as _jax_local
    p_jax  = jnp.array(oof_xgb_probs.astype(float))
    seg_jx = jnp.array(oof_segment_ids.astype(int))
    X_jax  = jnp.array(X_domain)
    y_jax  = jnp.array(oof_labels.astype(float))

    kernel = NUTS(_numpyro_stacker_model)
    mcmc   = MCMC(kernel,
                  num_warmup  = num_warmup,
                  num_samples = num_samples,
                  num_chains  = num_chains,
                  progress_bar= True)
    mcmc.run(
        _jax_local.random.PRNGKey(42),
        p_jax, seg_jx, X_jax, n_segments, n_domain,
        y_obs = y_jax,
    )
    samples = mcmc.get_samples()

    # ── Posterior means ────────────────────────────────────────────────
    alpha_hat = float(np.mean(samples["alpha"]))
    beta_hat  = float(np.mean(samples["beta"]))
    delta_hat = np.mean(samples["delta"],  axis=0)   # (n_segments,)
    gamma_hat = np.mean(samples["gamma"],  axis=0)   # (n_domain,)

    print(f"\n  Posterior parameter summary (mean +/- std):")
    print(f"    {'alpha (intercept)':<32} {alpha_hat:+.4f} +/- "
          f"{float(np.std(samples['alpha'])):.4f}")
    print(f"    {'beta  (XGB trust)':<32} {beta_hat:+.4f} +/- "
          f"{float(np.std(samples['beta'])):.4f}")
    print(f"    {'sigma_delta (HalfCauchy)':<32} {float(np.mean(samples['sigma_delta'])):+.4f} +/- "
          f"{float(np.std(samples['sigma_delta'])):.4f}")
    for j in range(n_segments):
        lbl = _SP_SEG_LABELS[j] if j < len(_SP_SEG_LABELS) else str(j)
        print(f"    delta[{j}] ({lbl:>3})  n={seg_counts[j]:>4}       "
              f"{delta_hat[j]:+.4f} +/- {float(np.std(samples['delta'][:, j])):.4f}")
    for k, feat in enumerate(feat_names):
        print(f"    gamma  {feat:<32} {gamma_hat[k]:+.4f} +/- "
              f"{float(np.std(samples['gamma'][:, k])):.4f}")

    # ── Save full posterior trace to disk (.npz) ───────────────────────
    trace_path = out_path.with_suffix(".npz")
    np.savez(str(trace_path), **{k: np.array(v) for k, v in samples.items()})
    print(f"\n  Posterior trace -> {trace_path}")

    # ── Build, evaluate and save stacker ──────────────────────────────
    model = BayesianStacker(
        alpha  = alpha_hat,
        beta   = beta_hat,
        delta  = delta_hat,
        gamma  = gamma_hat,
        stacking_feature_names = feat_names,
        fill_values            = fill_vals,
        n_segments             = n_segments,
        posterior_path         = str(trace_path),
    )
    out_path.write_bytes(pickle.dumps(model))
    print(f"  BayesianStacker -> {out_path}")

    stk_probs = model.predict(oof_xgb_probs, oof_feat_df, oof_segment_ids)
    auc_xgb   = roc_auc_score(oof_labels, oof_xgb_probs)
    auc_stk   = roc_auc_score(oof_labels, stk_probs)
    ll_xgb    = log_loss(oof_labels, oof_xgb_probs)
    ll_stk    = log_loss(oof_labels, stk_probs)
    print(f"\n  OOF performance:")
    print(f"    {'Model':<28}  {'AUC':>7}  {'LogLoss':>9}")
    print(f"    {'-'*48}")
    print(f"    {'XGBoost (raw)':<28}  {auc_xgb:>7.4f}  {ll_xgb:>9.4f}")
    print(f"    {'Bayesian Stacker':<28}  {auc_stk:>7.4f}  {ll_stk:>9.4f}  <<<")
    print(f"    Net AUC delta: {auc_stk - auc_xgb:+.4f}")

    return model


# ---------------------------------------------------------------------------
# NESTED CROSS-VALIDATION
# ---------------------------------------------------------------------------

def _fold_masks(df, train_years, val_year):
    return df["year"].isin(train_years), df["year"] == val_year


def _proba(model, X):
    """Unified predict-proba: works for XGB, LGBM, CatBoost."""
    if model is None:
        return None
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)   # regressor fallback


def run_ncv(df, X_all, feature_cols, early_stop):
    """Walk-forward NCV. Returns OOF arrays for all three Level-1 models."""
    print(f"\n{'='*60}")
    print(f"  NESTED CROSS-VALIDATION  (2026 held out)")
    print(f"{'='*60}")

    ncv_rows = []
    oof_xgb_rl  = [];  oof_lgbm_rl = [];  oof_cat_rl  = []
    oof_labels_rl = []
    oof_xgb_ml  = [];  oof_labels_ml = []
    oof_feat_rows    = []
    oof_segment_rows = []   # SP handedness segment IDs aligned to oof_feat_rows
    oof_stk_probs    = []
    oof_stk_labels   = []
    oof_stk_feat_rows = []
    oof_stk_seg_rows  = []
    oof_stk_is_home  = []

    for fold_idx, fold in enumerate(NCV_FOLDS, 1):
        train_mask, val_mask = _fold_masks(df, fold["train_years"], fold["val_year"])
        print(f"\n  {'-'*56}")
        print(f"  Fold {fold_idx}  |  Train: {fold['train_years']}  ->  "
              f"Validate: {fold['val_year']}")
        print(f"  {'-'*56}")

        # ── Run-line ──────────────────────────────────────────────────────
        rl_tr = train_mask & df["home_covers_rl"].notna()
        rl_vl = val_mask   & df["home_covers_rl"].notna()
        X_tr = X_all[rl_tr];  y_tr = df.loc[rl_tr,"home_covers_rl"].astype(int)
        X_vl = X_all[rl_vl];  y_vl = df.loc[rl_vl,"home_covers_rl"].astype(int)
        scale = (y_tr==0).sum() / max((y_tr==1).sum(),1)
        print(f"\n  [RL] n_train={rl_tr.sum()} | n_val={rl_vl.sum()} | "
              f"scale={scale:.2f}")

        yr_tr = df.loc[rl_tr, "year"]   # year series aligned to training rows

        # XGBoost
        xgb_m = train_binary(X_tr, y_tr, X_vl, y_vl,
                              f"Fold{fold_idx}-RL", early_stop, scale,
                              year_series=yr_tr)
        xgb_p = xgb_m.predict_proba(X_vl)[:, 1]

        # LightGBM
        lgbm_m = train_binary_lgbm(X_tr, y_tr, X_vl, y_vl,
                                    f"Fold{fold_idx}-RL", early_stop, scale,
                                    year_series=yr_tr)
        lgbm_p = _proba(lgbm_m, X_vl)

        # CatBoost
        cat_m  = train_binary_catboost(X_tr, y_tr, X_vl, y_vl,
                                        f"Fold{fold_idx}-RL", early_stop, scale,
                                        year_series=yr_tr)
        cat_p  = _proba(cat_m, X_vl)

        oof_xgb_rl.extend(xgb_p.tolist())
        oof_labels_rl.extend(y_vl.tolist())
        if lgbm_p is not None: oof_lgbm_rl.extend(lgbm_p.tolist())
        if cat_p  is not None: oof_cat_rl.extend(cat_p.tolist())

        # ── Team-perspective model (per fold, zero-leakage OOF) ──────────
        print(f"\n  [TEAM-RL] Fold {fold_idx}: training doubled dataset model ...")
        _team_feat_ncv = feature_cols + ["is_home"]
        _team_nonfeat  = NON_FEATURE_COLS | {"covers_rl"}
        df_team_tr_f = build_team_perspective_df(df[rl_tr])
        df_team_vl_f = build_team_perspective_df(df[rl_vl])
        n_vl_f = int(rl_vl.sum())

        def _prep_tm(dft):
            Xt = dft[_team_feat_ncv].copy()
            for c in Xt.columns:
                if Xt[c].dtype == object:
                    Xt[c] = pd.to_numeric(Xt[c], errors="coerce")
            return Xt

        X_tm_tr_f = _prep_tm(df_team_tr_f); y_tm_tr_f = df_team_tr_f["covers_rl"].astype(int)
        X_tm_vl_f = _prep_tm(df_team_vl_f); y_tm_vl_f = df_team_vl_f["covers_rl"].astype(int)
        team_fold_m = train_binary(X_tm_tr_f, y_tm_tr_f, X_tm_vl_f, y_tm_vl_f,
                                    f"Fold{fold_idx}-TEAM-RL", early_stop, 1.0,
                                    year_series=df.loc[rl_tr, "year"])
        # Score home and away halves (first n_vl = home rows, next n_vl = away rows)
        _p_home_raw_f = team_fold_m.predict_proba(X_tm_vl_f.iloc[:n_vl_f])[:, 1]
        _p_away_raw_f = team_fold_m.predict_proba(X_tm_vl_f.iloc[n_vl_f:])[:, 1]

        # Stacking feature collection deferred to after ML fold (needs ml_model_vs_vegas_gap)

        # Ensemble average for logging
        probs_list = [xgb_p]
        if lgbm_p is not None: probs_list.append(lgbm_p)
        if cat_p  is not None: probs_list.append(cat_p)
        ens_p = np.mean(probs_list, axis=0)

        auc_xgb = roc_auc_score(y_vl, xgb_p)
        auc_ens = roc_auc_score(y_vl, ens_p)
        print(f"    XGB AUC={auc_xgb:.4f} | "
              f"ENS AUC={auc_ens:.4f} | delta={auc_ens-auc_xgb:+.4f}")

        vdf = df[rl_vl].copy(); vdf["rl_prob"] = xgb_p
        edge_analysis(vdf, "rl_prob", label_col="home_covers_rl",
                      thresholds=[0.48,0.50,0.52,0.525,0.54,0.56,0.58,0.60])

        ll    = log_loss(y_vl, xgb_p)
        brier = brier_score_loss(y_vl, xgb_p)
        acc   = ((xgb_p > 0.50) == y_vl).mean()
        cr    = float(y_vl.mean())
        ncv_rows.append(dict(
            fold=fold_idx, model="run_line",
            train_years=str(fold["train_years"]), val_year=fold["val_year"],
            n_train=int(rl_tr.sum()), n_val=int(rl_vl.sum()),
            auc_xgb=round(auc_xgb,4), auc_ens=round(auc_ens,4),
            log_loss=round(ll,4), brier=round(brier,4),
            accuracy=round(acc,4), base_cover_rate=round(cr,4),
        ))

        # ── Total runs ─────────────────────────────────────────────────────
        tot_tr = train_mask & df["total_runs"].notna()
        tot_vl = val_mask   & df["total_runs"].notna()
        Xt_tr = X_all[tot_tr]; yt_tr = df.loc[tot_tr,"total_runs"]
        Xt_vl = X_all[tot_vl]; yt_vl = df.loc[tot_vl,"total_runs"]
        yr_tot_tr = df.loc[tot_tr, "year"]
        print(f"\n  [TOT] n_train={tot_tr.sum()} | n_val={tot_vl.sum()}")
        xgb_tot = train_regression(Xt_tr, yt_tr, Xt_vl, yt_vl,
                                   f"Fold{fold_idx}-TOTAL", early_stop,
                                   year_series=yr_tot_tr)
        train_regression_lgbm(Xt_tr, yt_tr, Xt_vl, yt_vl,
                              f"Fold{fold_idx}-TOTAL", early_stop,
                              year_series=yr_tot_tr)
        train_regression_catboost(Xt_tr, yt_tr, Xt_vl, yt_vl,
                                  f"Fold{fold_idx}-TOTAL", early_stop,
                                  year_series=yr_tot_tr)
        # Multi-quantile output: shape (n_val, 3) → use col 1 (median) for error metrics
        tot_preds_q = xgb_tot.predict(Xt_vl)
        tot_preds   = (tot_preds_q[:, TOT_Q_IDX["median"]]
                       if tot_preds_q.ndim == 2 else tot_preds_q)
        mae  = mean_absolute_error(yt_vl, tot_preds)
        rmse = np.sqrt(mean_squared_error(yt_vl, tot_preds))
        if tot_preds_q.ndim == 2:
            q10_mean = tot_preds_q[:, TOT_Q_IDX["floor"]].mean()
            q50_mean = tot_preds_q[:, TOT_Q_IDX["median"]].mean()
            q90_mean = tot_preds_q[:, TOT_Q_IDX["ceiling"]].mean()
            print(f"    MAE={mae:.4f}  RMSE={rmse:.4f}  "
                  f"[Q10={q10_mean:.2f} | Q50={q50_mean:.2f} | Q90={q90_mean:.2f}]")
        else:
            print(f"    MAE={mae:.4f}  RMSE={rmse:.4f}")
        ncv_rows.append(dict(
            fold=fold_idx, model="total_runs",
            train_years=str(fold["train_years"]), val_year=fold["val_year"],
            n_train=int(tot_tr.sum()), n_val=int(tot_vl.sum()),
            mae=round(mae,4), rmse=round(rmse,4),
        ))

        # ── Moneyline ─────────────────────────────────────────────────────
        ml_tr = train_mask & df["actual_home_win"].notna()
        ml_vl = val_mask   & df["actual_home_win"].notna()
        Xm_tr = X_all[ml_tr]; ym_tr = df.loc[ml_tr,"actual_home_win"].astype(int)
        Xm_vl = X_all[ml_vl]; ym_vl = df.loc[ml_vl,"actual_home_win"].astype(int)
        ml_scale = (ym_tr==0).sum() / max((ym_tr==1).sum(),1)
        yr_ml_tr = df.loc[ml_tr, "year"]
        print(f"\n  [ML] n_train={ml_tr.sum()} | n_val={ml_vl.sum()}")
        xgb_ml_m = train_binary(Xm_tr, ym_tr, Xm_vl, ym_vl,
                                 f"Fold{fold_idx}-ML", early_stop, ml_scale,
                                 year_series=yr_ml_tr)
        train_binary_lgbm(Xm_tr, ym_tr, Xm_vl, ym_vl,
                          f"Fold{fold_idx}-ML", early_stop, ml_scale,
                          year_series=yr_ml_tr)
        train_binary_catboost(Xm_tr, ym_tr, Xm_vl, ym_vl,
                               f"Fold{fold_idx}-ML", early_stop, ml_scale,
                               year_series=yr_ml_tr)
        ml_p = xgb_ml_m.predict_proba(Xm_vl)[:, 1]
        if ml_tr.sum() > 0:
            oof_xgb_ml.extend(ml_p.tolist())
            oof_labels_ml.extend(ym_vl.tolist())
        ml_auc = roc_auc_score(ym_vl, ml_p)
        print(f"    XGB ML AUC={ml_auc:.4f}")
        ncv_rows.append(dict(
            fold=fold_idx, model="moneyline",
            train_years=str(fold["train_years"]), val_year=fold["val_year"],
            n_train=int(ml_tr.sum()), n_val=int(ml_vl.sum()),
            auc=round(ml_auc,4),
        ))

        # ── Stacking feature collection (deferred — requires ML OOF probs) ──
        # ml_model_vs_vegas_gap = XGB ML raw prob − Pinnacle closing implied.
        # When gap > 10%, empirical win rate on 2025 val: 65.1% vs 57.2% aligned.
        stk_base_cols = [c for c in STACKING_FEATURES
                         if c in df.columns and c != "ml_model_vs_vegas_gap"]
        feat_block = df.loc[rl_vl, stk_base_cols].copy()

        # Predict ML probability on the RL validation set (may differ from ml_vl)
        ml_probs_on_rl_val = xgb_ml_m.predict_proba(X_all[rl_vl])[:, 1]
        true_home_probs    = df.loc[rl_vl, "true_home_prob"].fillna(0.5).values
        feat_block["ml_model_vs_vegas_gap"] = ml_probs_on_rl_val - true_home_probs
        feat_block = feat_block.reset_index(drop=True)
        oof_feat_rows.append(feat_block)

        # SP handedness segment IDs (aligned to rl_vl rows)
        seg_ids = _derive_segment_id(df.loc[rl_vl])
        oof_segment_rows.append(seg_ids)

        seg_dist = {_SP_SEG_LABELS[j]: int((seg_ids == j).sum())
                    for j in range(_SP_HANDEDNESS_N_SEGMENTS)}
        print(f"    Stacking feats collected: {len(feat_block)} rows | "
              f"gap mean={feat_block['ml_model_vs_vegas_gap'].mean():+.4f} | "
              f"segs={seg_dist}")

        # ── Doubled stacker OOF collection (team model, zero-leakage) ────
        _p_sum_f = np.maximum(_p_home_raw_f + _p_away_raw_f, 1e-9)
        _norm_home_f = _p_home_raw_f / _p_sum_f
        _norm_away_f = _p_away_raw_f / _p_sum_f
        _team_auc_f = roc_auc_score(y_vl.values, _norm_home_f)
        print(f"    TEAM-RL AUC (norm home, fold {fold_idx})={_team_auc_f:.4f}")
        # Home rows
        _n_vl = len(_norm_home_f)
        oof_stk_probs.extend(_norm_home_f.tolist())
        oof_stk_labels.extend(y_vl.values.tolist())
        oof_stk_feat_rows.append(feat_block.reset_index(drop=True))
        oof_stk_seg_rows.append(_derive_segment_id(df.loc[rl_vl]))
        oof_stk_is_home.extend([True] * _n_vl)
        # Away rows (flipped stacking feats, flipped segment, flipped label)
        oof_stk_probs.extend(_norm_away_f.tolist())
        oof_stk_labels.extend((1.0 - y_vl.values).tolist())
        oof_stk_feat_rows.append(_flip_stacking_feats(feat_block.reset_index(drop=True)))
        oof_stk_seg_rows.append(_derive_segment_id_away(df.loc[rl_vl]))
        oof_stk_is_home.extend([False] * _n_vl)

    # NCV summary
    rl_rows = [r for r in ncv_rows if r["model"] == "run_line"]
    print(f"\n{'='*60}")
    print(f"  NCV SUMMARY — Run-Line Model")
    print(f"{'='*60}")
    print(f"  {'Fold':>4}  {'Val':>5}  {'n_val':>6}  "
          f"{'XGB_AUC':>8}  {'ENS_AUC':>8}  {'delta':>7}  {'Cover%':>7}")
    print(f"  {'-'*55}")
    for r in rl_rows:
        delta = r["auc_ens"] - r["auc_xgb"]
        print(f"  {r['fold']:>4}  {r['val_year']:>5}  {r['n_val']:>6}  "
              f"{r['auc_xgb']:>8.4f}  {r['auc_ens']:>8.4f}  "
              f"{delta:>+7.4f}  {r['base_cover_rate']:>7.3f}")
    avg_xgb = np.mean([r["auc_xgb"] for r in rl_rows])
    avg_ens = np.mean([r["auc_ens"] for r in rl_rows])
    print(f"  {'MEAN':>4}  {'':>5}  {'':>6}  "
          f"{avg_xgb:>8.4f}  {avg_ens:>8.4f}  {avg_ens-avg_xgb:>+7.4f}")
    print(f"\n  2026 was NOT used in any fold above.")

    pd.DataFrame(ncv_rows).to_csv("xgb_ncv_results.csv", index=False)
    print(f"  Saved NCV results to xgb_ncv_results.csv")

    oof_feat_df     = (pd.concat(oof_feat_rows, ignore_index=True)
                       if oof_feat_rows else pd.DataFrame())
    oof_segment_ids = (np.concatenate(oof_segment_rows)
                       if oof_segment_rows else np.array([], dtype=np.int32))

    team_oof = {
        "probs":   np.array(oof_stk_probs),
        "labels":  np.array(oof_stk_labels, dtype=float),
        "feat_df": (pd.concat(oof_stk_feat_rows, ignore_index=True)
                    if oof_stk_feat_rows else pd.DataFrame()),
        "seg_ids": (np.concatenate(oof_stk_seg_rows).astype(np.int32)
                    if oof_stk_seg_rows else np.array([], dtype=np.int32)),
        "is_home": np.array(oof_stk_is_home, dtype=bool),
    }

    return (
        ncv_rows,
        np.array(oof_xgb_rl),
        np.array(oof_labels_rl),
        np.array(oof_xgb_ml) if oof_xgb_ml else np.array([]),
        np.array(oof_labels_ml) if oof_labels_ml else np.array([]),
        oof_feat_df,
        oof_segment_ids,
        np.array(oof_lgbm_rl) if oof_lgbm_rl else None,
        np.array(oof_cat_rl)  if oof_cat_rl  else None,
        team_oof,
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def build_team_perspective_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a doubled dataset for the team-perspective RL model.

    Each labeled game produces two rows:
      Row A  is_home=1 : original home-team perspective, label = home_covers_rl
      Row B  is_home=0 : away-team perspective (columns swapped), label = 1-home_covers_rl

    Transformations applied to Row B:
      - All paired home_* <-> away_* columns are swapped
      - home_bat_vs_away_sp <-> away_bat_vs_home_sp  (cross-matchup pair)
      - true_home_prob <-> true_away_prob
      - All _DIFF_COLS_TO_NEGATE are negated (home-away sign flips to away-home)
      - home_park_factor is unchanged (same ballpark regardless of perspective)

    The resulting DataFrame has a 'covers_rl' label column and 'is_home' feature.
    All NON_FEATURE_COLS remain present for split/year filtering but are excluded
    from features by prep_features().
    """
    labeled = df[df["home_covers_rl"].notna()].copy()
    all_cols = set(labeled.columns)

    # -- Row A: home perspective, unchanged --
    row_a = labeled.copy()
    row_a["is_home"]   = 1
    row_a["covers_rl"] = row_a["home_covers_rl"]

    # -- Row B: away perspective --
    row_b = labeled.copy()

    # Swap all paired home_* <-> away_* columns
    for col in list(all_cols):
        if col.startswith("home_"):
            away_col = "away_" + col[5:]
            if away_col in all_cols:
                row_b[col]      = labeled[away_col].values
                row_b[away_col] = labeled[col].values

    # Swap asymmetric cross-matchup columns
    for h_col, a_col in [
        ("home_bat_vs_away_sp",     "away_bat_vs_home_sp"),
        ("home_bat_vs_away_sp_10d", "away_bat_vs_home_sp_10d"),
    ]:
        if h_col in all_cols and a_col in all_cols:
            row_b[h_col] = labeled[a_col].values
            row_b[a_col] = labeled[h_col].values

    # Swap true_home_prob <-> true_away_prob
    if "true_home_prob" in all_cols and "true_away_prob" in all_cols:
        row_b["true_home_prob"] = labeled["true_away_prob"].values
        row_b["true_away_prob"] = labeled["true_home_prob"].values

    # Negate signed diff columns (home-away -> away-home)
    for col in _DIFF_COLS_TO_NEGATE:
        if col in all_cols:
            row_b[col] = -labeled[col].values

    row_b["is_home"]   = 0
    row_b["covers_rl"] = 1.0 - labeled["home_covers_rl"].values

    return pd.concat([row_a, row_b], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Train 3-model stacked ensemble (XGBoost + LightGBM + CatBoost)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--matrix",       default="feature_matrix.parquet")
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--val-year",     type=int, default=2025)
    parser.add_argument("--ncv",          action="store_true")
    parser.add_argument("--extra-features", type=str, default=None,
                        help="Comma-separated extra features to add from a v2 matrix. "
                             "When set, restricts features to models/feature_cols_v1.json + these extras.")
    args       = parser.parse_args()
    early_stop = not args.no_early_stop

    print("=" * 60)
    print("  train_xgboost.py  v5.0  (3-model stacked ensemble)")
    print(f"  Level-1 : XGBoost{'  LightGBM' if _LGBM else ''}"
          f"{'  CatBoost' if _CATBOOST else ''}")
    print(f"  GPU     : {'CUDA (RTX 5080)' if _GPU else 'CPU'}")
    if args.ncv:
        print("  Mode    : Nested Cross-Validation (2026 held out)")
    else:
        print(f"  Mode    : Standard (train=2023-2024 | val={args.val_year})")
    print("=" * 60)

    df = pd.read_parquet(args.matrix, engine="pyarrow")
    print(f"\n  Loaded {len(df)} rows x {len(df.columns)} cols")

    for c in ("year", "season"):
        if c in df.columns and c != "year":
            df["year"] = df[c]; break
    if "year" not in df.columns:
        raise ValueError("Feature matrix must contain 'year' or 'season'.")

    X_all, feature_cols = prep_features(df)

    # --extra-features: restrict to v1 baseline + whitelisted extras
    if args.extra_features:
        extra_list = [f.strip() for f in args.extra_features.split(",") if f.strip()]
        v1_path = MODELS_DIR / "feature_cols_v1.json"
        if v1_path.exists():
            v1_set = set(json.loads(v1_path.read_text()))
            print(f"\n  --extra-features mode: v1={len(v1_set)} + extra={len(extra_list)}")
            feature_cols = [c for c in feature_cols if c in v1_set or c in extra_list]
            present_extra = [c for c in extra_list if c in feature_cols]
            missing_extra = [c for c in extra_list if c not in feature_cols]
            print(f"  Extra features present : {present_extra}")
            if missing_extra:
                print(f"  [WARN] Extra features not in matrix: {missing_extra}")
            X_all = X_all[[c for c in feature_cols]] if hasattr(X_all, 'columns') else X_all
        else:
            print(f"  [WARN] --extra-features: v1 baseline not found at {v1_path}")

    print(f"  Feature columns: {len(feature_cols)}")
    (MODELS_DIR / "feature_cols.json").write_text(json.dumps(feature_cols, indent=2))

    # =========================================================================
    # NCV MODE
    # =========================================================================
    if args.ncv:
        years_available = sorted(df["year"].dropna().unique().astype(int))
        print(f"  Years in matrix: {years_available}")
        missing = set(NCV_ALL_YEARS) - set(years_available)
        if missing:
            raise ValueError(f"Missing NCV years: {missing}")

        (ncv_rows,
         oof_xgb_rl, oof_labels_rl,
         oof_xgb_ml, oof_labels_ml,
         oof_feat_df,
         oof_segment_ids,
         oof_lgbm_rl, oof_cat_rl,
         team_oof) = run_ncv(df, X_all, feature_cols, early_stop)

        # ── Final models on 2023+2024+2025 ────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  FINAL MODEL TRAINING  (2023+2024 train | 2025 val | 2026 held out)")
        print(f"{'='*60}")

        # Train on 2023+2024; validate on 2025 (clean holdout — no overlap).
        # Previously final_mask included 2025, causing early stopping to monitor
        # training data → models ran to the n_estimators cap.
        final_mask = df["year"].isin([2023, 2024])
        val_mask   = df["year"] == 2025

        # ── Run-line
        rl_tr = final_mask & df["home_covers_rl"].notna()
        rl_vl = val_mask   & df["home_covers_rl"].notna()
        X_tr = X_all[rl_tr]; y_tr = df.loc[rl_tr,"home_covers_rl"].astype(int)
        X_vl = X_all[rl_vl]; y_vl = df.loc[rl_vl,"home_covers_rl"].astype(int)
        rl_scale = (y_tr==0).sum() / max((y_tr==1).sum(),1)
        yr_rl_tr = df.loc[rl_tr, "year"]
        print(f"\n  [RL] n_train={rl_tr.sum()} | scale={rl_scale:.2f}")
        rl_model    = train_binary(X_tr, y_tr, X_vl, y_vl, "Final-RL", early_stop, rl_scale,
                                   year_series=yr_rl_tr)
        lgbm_rl_m   = train_binary_lgbm(X_tr, y_tr, X_vl, y_vl, "Final-RL", early_stop, rl_scale,
                                         year_series=yr_rl_tr)
        cat_rl_m    = train_binary_catboost(X_tr, y_tr, X_vl, y_vl, "Final-RL", early_stop, rl_scale,
                                            year_series=yr_rl_tr)

        # ── Total runs
        tot_tr = final_mask & df["total_runs"].notna()
        tot_vl = val_mask   & df["total_runs"].notna()
        Xt_tr = X_all[tot_tr]; yt_tr = df.loc[tot_tr,"total_runs"]
        Xt_vl = X_all[tot_vl]; yt_vl = df.loc[tot_vl,"total_runs"]
        yr_tot_tr = df.loc[tot_tr, "year"]
        print(f"\n  [TOT] n_train={tot_tr.sum()}")
        tot_model   = train_regression(Xt_tr, yt_tr, Xt_vl, yt_vl, "Final-TOT", early_stop,
                                       year_series=yr_tot_tr)
        lgbm_tot_m  = train_regression_lgbm(Xt_tr, yt_tr, Xt_vl, yt_vl, "Final-TOT", early_stop,
                                            year_series=yr_tot_tr)
        cat_tot_m   = train_regression_catboost(Xt_tr, yt_tr, Xt_vl, yt_vl, "Final-TOT", early_stop,
                                               year_series=yr_tot_tr)

        # ── Moneyline
        ml_tr = final_mask & df["actual_home_win"].notna()
        ml_vl = val_mask   & df["actual_home_win"].notna()
        Xm_tr = X_all[ml_tr]; ym_tr = df.loc[ml_tr,"actual_home_win"].astype(int)
        Xm_vl = X_all[ml_vl]; ym_vl = df.loc[ml_vl,"actual_home_win"].astype(int)
        ml_scale = (ym_tr==0).sum() / max((ym_tr==1).sum(),1)
        yr_ml_tr = df.loc[ml_tr, "year"]
        print(f"\n  [ML] n_train={ml_tr.sum()} | scale={ml_scale:.2f}")
        ml_model    = train_binary(Xm_tr, ym_tr, Xm_vl, ym_vl, "Final-ML", early_stop, ml_scale,
                                   year_series=yr_ml_tr)
        lgbm_ml_m   = train_binary_lgbm(Xm_tr, ym_tr, Xm_vl, ym_vl, "Final-ML", early_stop, ml_scale,
                                         year_series=yr_ml_tr)
        cat_ml_m    = train_binary_catboost(Xm_tr, ym_tr, Xm_vl, ym_vl, "Final-ML", early_stop, ml_scale,
                                            year_series=yr_ml_tr)

        # Val predictions for CSV
        # tot_model outputs (n, 3) for multi-quantile; extract median (col 1).
        rl_probs    = rl_model.predict_proba(X_vl)[:, 1]
        _tot_q      = tot_model.predict(Xt_vl)
        tot_preds   = (_tot_q[:, TOT_Q_IDX["median"]]
                       if _tot_q.ndim == 2 else _tot_q)
        ml_probs    = ml_model.predict_proba(X_vl)[:, 1]
        val_df_edge = df[rl_vl].copy()
        val_df_edge["rl_prob"]  = rl_probs
        val_df_edge["tot_pred"] = tot_preds
        val_df_edge["ml_prob"]  = ml_probs

        # ── Platt calibration (pooled OOF)
        print(f"\n{'='*60}")
        print(f"  PLATT CALIBRATION  (pooled OOF — zero leakage)")
        print(f"{'='*60}")
        cal_rl = fit_platt_calibrator(oof_xgb_rl, oof_labels_rl,
                                      MODELS_DIR/"calibrator_rl.pkl", "run_line")
        calibration_report(oof_xgb_rl, oof_labels_rl, cal_rl, "run_line")
        if len(oof_xgb_ml) > 0:
            cal_ml = fit_platt_calibrator(oof_xgb_ml, oof_labels_ml,
                                          MODELS_DIR/"calibrator_ml.pkl", "moneyline")
            calibration_report(oof_xgb_ml, oof_labels_ml, cal_ml, "moneyline")

        # ── Bayesian Hierarchical Stacker (Level-2)
        # Now trained on doubled team-perspective OOF (home + away rows, zero-leakage).
        if not team_oof["feat_df"].empty and len(team_oof["probs"]) == len(team_oof["feat_df"]):
            stk_model = train_bayesian_stacker(
                oof_xgb_probs   = team_oof["probs"],
                oof_feat_df     = team_oof["feat_df"],
                oof_labels      = team_oof["labels"],
                oof_segment_ids = team_oof["seg_ids"],
                out_path        = MODELS_DIR / "stacking_lr_rl.pkl",
            )

            # Comparison table: home-only slice (use is_home mask to correctly select)
            _hm = team_oof.get("is_home", np.ones(len(team_oof["probs"]), dtype=bool))
            _home_probs   = team_oof["probs"][_hm]
            _home_labels  = team_oof["labels"][_hm]
            _home_feat_df = team_oof["feat_df"].loc[_hm].reset_index(drop=True)
            _home_segs    = team_oof["seg_ids"][_hm]

            stk_p   = stk_model.predict(_home_probs, _home_feat_df, _home_segs)
            platt_p = cal_rl.predict_proba(oof_xgb_rl.reshape(-1, 1))[:, 1]
            # Use _home_labels (aligned to team OOF home rows) as ground truth.
            y       = _home_labels

            auc_xgb   = roc_auc_score(y, oof_xgb_rl)
            auc_platt = roc_auc_score(y, platt_p)
            auc_stk   = roc_auc_score(y, stk_p)
            ll_xgb    = log_loss(y, oof_xgb_rl)
            ll_stk    = log_loss(y, np.clip(stk_p, 1e-7, 1-1e-7))
            acc_xgb   = ((oof_xgb_rl > .5) == y).mean()
            acc_stk   = ((stk_p > .5) == y).mean()

            rows_cmp = [("XGBoost (raw)",         auc_xgb,   ll_xgb,  acc_xgb),
                        ("XGBoost (Platt)",        auc_platt, None,    None)]
            if oof_lgbm_rl is not None:
                rows_cmp.append(("LightGBM (raw)",
                                 roc_auc_score(y, oof_lgbm_rl), None, None))
            if oof_cat_rl is not None:
                if len(oof_cat_rl) == len(y):
                    rows_cmp.append(("CatBoost (raw)",
                                     roc_auc_score(y, oof_cat_rl),  None, None))
                else:
                    print(f"  [SKIP CatBoost report] OOF size mismatch "
                          f"({len(oof_cat_rl)} vs {len(y)})")
            rows_cmp.append(("Bayesian Stacker", auc_stk, ll_stk, acc_stk))

            print(f"  {'Model':<24}  {'AUC':>7}  {'LogLoss':>9}  {'Acc@50%':>8}")
            print(f"  {'-'*54}")
            for nm, auc, ll, acc in rows_cmp:
                ll_s  = f"{ll:.4f}"  if ll  is not None else "      -"
                acc_s = f"{acc:.4f}" if acc is not None else "      -"
                marker = " <<<" if nm == "Bayesian Stacker" else ""
                print(f"  {nm:<24}  {auc:>7.4f}  {ll_s:>9}  {acc_s:>8}{marker}")
            print(f"\n  Net AUC gain (Stacker vs XGBoost raw): {auc_stk - auc_xgb:+.4f}")
        elif not oof_feat_df.empty and len(oof_xgb_rl) == len(oof_feat_df):
            # Fallback: train stacker on home-only OOF if team OOF not available
            stk_model = train_bayesian_stacker(
                oof_xgb_probs   = oof_xgb_rl,
                oof_feat_df     = oof_feat_df,
                oof_labels      = np.array(oof_labels_rl),
                oof_segment_ids = oof_segment_ids,
                out_path        = MODELS_DIR / "stacking_lr_rl.pkl",
            )

    # =========================================================================
    # STANDARD MODE
    # =========================================================================
    else:
        train_mask = df["split"] == "train"
        val_mask   = df["split"] == "val"

        rl_tr = train_mask & df["home_covers_rl"].notna()
        rl_vl = val_mask   & df["home_covers_rl"].notna()
        X_rl_tr = X_all[rl_tr]; y_rl_tr = df.loc[rl_tr,"home_covers_rl"].astype(int)
        X_rl_vl = X_all[rl_vl]; y_rl_vl = df.loc[rl_vl,"home_covers_rl"].astype(int)

        tot_tr = train_mask & df["total_runs"].notna()
        tot_vl = val_mask   & df["total_runs"].notna()
        X_tot_tr = X_all[tot_tr]; y_tot_tr = df.loc[tot_tr,"total_runs"]
        X_tot_vl = X_all[tot_vl]; y_tot_vl = df.loc[tot_vl,"total_runs"]

        ml_tr = train_mask & df["actual_home_win"].notna()
        ml_vl = val_mask   & df["actual_home_win"].notna()
        X_ml_tr = X_all[ml_tr]; y_ml_tr = df.loc[ml_tr,"actual_home_win"].astype(int)
        X_ml_vl = X_all[ml_vl]; y_ml_vl = df.loc[ml_vl,"actual_home_win"].astype(int)

        rl_scale = (y_rl_tr==0).sum() / max((y_rl_tr==1).sum(),1)
        ml_scale = (y_ml_tr==0).sum() / max((y_ml_tr==1).sum(),1)
        print(f"\n  Train: {rl_tr.sum()} | Val: {rl_vl.sum()}")

        print(f"\n{'='*60}")
        print(f"  Training Level-1 models ({'early stop ON' if early_stop else 'OFF'})")
        print(f"{'='*60}")

        print("\n  [RL] Run-line ...")
        rl_model   = train_binary(X_rl_tr, y_rl_tr, X_rl_vl, y_rl_vl,
                                   "RL", early_stop, rl_scale)
        lgbm_rl_m  = train_binary_lgbm(X_rl_tr, y_rl_tr, X_rl_vl, y_rl_vl,
                                        "RL", early_stop, rl_scale)
        cat_rl_m   = train_binary_catboost(X_rl_tr, y_rl_tr, X_rl_vl, y_rl_vl,
                                            "RL", early_stop, rl_scale)

        print("\n  [TOT] Total runs ...")
        tot_model  = train_regression(X_tot_tr, y_tot_tr, X_tot_vl, y_tot_vl,
                                       "TOT", early_stop)
        lgbm_tot_m = train_regression_lgbm(X_tot_tr, y_tot_tr, X_tot_vl, y_tot_vl,
                                            "TOT", early_stop)
        cat_tot_m  = train_regression_catboost(X_tot_tr, y_tot_tr, X_tot_vl, y_tot_vl,
                                                "TOT", early_stop)

        print("\n  [ML] Moneyline ...")
        ml_model   = train_binary(X_ml_tr, y_ml_tr, X_ml_vl, y_ml_vl,
                                   "ML", early_stop, ml_scale)
        lgbm_ml_m  = train_binary_lgbm(X_ml_tr, y_ml_tr, X_ml_vl, y_ml_vl,
                                        "ML", early_stop, ml_scale)
        cat_ml_m   = train_binary_catboost(X_ml_tr, y_ml_tr, X_ml_vl, y_ml_vl,
                                            "ML", early_stop, ml_scale)

        rl_probs  = rl_model.predict_proba(X_rl_vl)[:, 1]
        _tot_raw  = tot_model.predict(X_tot_vl)
        tot_preds = _tot_raw[:, _tot_raw.shape[1]//2] if np.ndim(_tot_raw)==2 else _tot_raw
        ml_probs  = ml_model.predict_proba(X_ml_vl)[:, 1]

        # Ensemble average for logging
        ens_parts = [rl_probs]
        lgbm_vl_p = _proba(lgbm_rl_m, X_rl_vl)
        cat_vl_p  = _proba(cat_rl_m,  X_rl_vl)
        if lgbm_vl_p is not None: ens_parts.append(lgbm_vl_p)
        if cat_vl_p  is not None: ens_parts.append(cat_vl_p)
        ens_p = np.mean(ens_parts, axis=0)

        print(f"\n{'='*60}")
        print(f"  Validation Metrics ({args.val_year})")
        print(f"{'='*60}")
        log_metrics("Run-Line (XGBoost)", y_rl_vl, y_pred_prob=rl_probs)
        if lgbm_vl_p is not None:
            log_metrics("Run-Line (LightGBM)", y_rl_vl, y_pred_prob=lgbm_vl_p)
        if cat_vl_p is not None:
            log_metrics("Run-Line (CatBoost)", y_rl_vl, y_pred_prob=cat_vl_p)
        log_metrics("Run-Line (Ensemble avg)", y_rl_vl, y_pred_prob=ens_p)
        log_metrics("Total Runs", y_tot_vl, y_pred_reg=tot_preds)
        log_metrics("Moneyline", y_ml_vl, y_pred_prob=ml_probs)

        edge_analysis(df[rl_vl].assign(rl_prob=rl_probs), "rl_prob",
                      thresholds=[0.48,0.50,0.52,0.525,0.54,0.56,0.58,0.60])

        val_df_edge = df[rl_vl].copy()
        val_df_edge["rl_prob"]  = rl_probs
        val_df_edge["tot_pred"] = tot_preds
        val_df_edge["ml_prob"]  = ml_probs

        # Platt calibration on val set (minor leakage — use --ncv for zero leakage)
        print(f"\n{'='*60}")
        print(f"  PLATT CALIBRATION  (val set — use --ncv for zero leakage)")
        print(f"{'='*60}")
        cal_rl = fit_platt_calibrator(rl_probs, y_rl_vl.values,
                                      MODELS_DIR/"calibrator_rl.pkl", "run_line")
        calibration_report(rl_probs, y_rl_vl.values, cal_rl, "run_line")
        if len(ml_probs) == len(y_ml_vl):
            cal_ml = fit_platt_calibrator(ml_probs, y_ml_vl.values,
                                          MODELS_DIR/"calibrator_ml.pkl", "moneyline")
            calibration_report(ml_probs, y_ml_vl.values, cal_ml, "moneyline")

        # Comparison table
        print(f"\n{'='*60}")
        print(f"  ENSEMBLE COMPARISON (val set)")
        print(f"{'='*60}")
        y = y_rl_vl.values
        rows_cmp = [("XGBoost", roc_auc_score(y, rl_probs))]
        if lgbm_vl_p is not None:
            rows_cmp.append(("LightGBM", roc_auc_score(y, lgbm_vl_p)))
        if cat_vl_p is not None:
            rows_cmp.append(("CatBoost", roc_auc_score(y, cat_vl_p)))
        rows_cmp.append(("Ensemble avg", roc_auc_score(y, ens_p)))
        print(f"  {'Model':<20}  {'AUC':>8}")
        print(f"  {'-'*30}")
        for nm, auc in rows_cmp:
            marker = " <<<" if nm == "Ensemble avg" else ""
            print(f"  {nm:<20}  {auc:>8.4f}{marker}")

    # =========================================================================
    # TEAM-PERSPECTIVE RL MODEL  (parallel to xgb_rl.json)
    # Doubles the dataset: each game -> home row + away row (features swapped).
    # Adds 'is_home' as a feature so the model learns home/away asymmetry.
    # Output: xgb_rl_team.json  — predicts P(featured team covers RL)
    # At inference: run twice per game (home feats → P_home, away feats → P_away)
    # replaces the 1-p approximation for the away direction.
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  [TEAM-RL] Building doubled team-perspective dataset ...")
    print(f"{'='*60}")

    try:
        df_team = build_team_perspective_df(df)
        print(f"  Doubled dataset: {len(df_team)} rows "
              f"({len(df_team)//2} games x 2 perspectives)")

        # prep_features excludes NON_FEATURE_COLS; 'is_home' and 'covers_rl' are
        # new columns not in that set, so we exclude 'covers_rl' manually.
        _team_non_feat = NON_FEATURE_COLS | {"covers_rl"}
        _team_feat_cols = [c for c in df_team.columns if c not in _team_non_feat]
        X_team_all = df_team[_team_feat_cols].copy()
        for col in X_team_all.columns:
            if X_team_all[col].dtype == object:
                X_team_all[col] = pd.to_numeric(X_team_all[col], errors="coerce")

        # Train/val split: mirror the original split column
        team_train = df_team["split"] == "train"
        team_val   = df_team["split"] == "val"
        X_tm_tr = X_team_all[team_train];  y_tm_tr = df_team.loc[team_train, "covers_rl"].astype(int)
        X_tm_vl = X_team_all[team_val];    y_tm_vl = df_team.loc[team_val,   "covers_rl"].astype(int)

        tm_scale = (y_tm_tr == 0).sum() / max((y_tm_tr == 1).sum(), 1)
        print(f"  Train: {len(X_tm_tr)}  Val: {len(X_tm_vl)}  "
              f"scale_pos_weight={tm_scale:.3f}")

        print(f"  [TEAM-RL] Training ...")
        team_rl_model = train_binary(X_tm_tr, y_tm_tr, X_tm_vl, y_tm_vl,
                                     "TEAM-RL", early_stop, tm_scale)

        team_rl_probs = team_rl_model.predict_proba(X_tm_vl)[:, 1]
        team_auc      = roc_auc_score(y_tm_vl.values, team_rl_probs)
        team_acc      = ((team_rl_probs > 0.5) == y_tm_vl.values).mean()

        # Compare: home-only rows use same label as xgb_rl — extract for apple-to-apple
        home_val_mask = team_val & (df_team["is_home"] == 1)
        X_tm_vl_home  = X_team_all[home_val_mask]
        y_tm_vl_home  = df_team.loc[home_val_mask, "covers_rl"].astype(int)
        home_only_auc = roc_auc_score(y_tm_vl_home,
                                      team_rl_model.predict_proba(X_tm_vl_home)[:, 1])

        print(f"\n  --- Team-RL vs Home-only RL ---")
        print(f"  Team model (all rows)   AUC: {team_auc:.4f}  Acc: {team_acc:.4f}")
        print(f"  Team model (home rows)  AUC: {home_only_auc:.4f}  (comparable to xgb_rl)")
        _cmp_y = locals().get("y_rl_vl", locals().get("y_vl", None))
        if _cmp_y is not None and "rl_probs" in locals():
            _cmp_probs = locals()["rl_probs"]
            orig_auc = roc_auc_score(_cmp_y.values, _cmp_probs)
            print(f"  Original xgb_rl        AUC: {orig_auc:.4f}")
            print(f"  Delta (team home vs orig): {home_only_auc - orig_auc:+.4f}")

        # Save feature list for team model
        (MODELS_DIR / "feature_cols_team.json").write_text(
            json.dumps(_team_feat_cols, indent=2))
        print(f"  Saved: feature_cols_team.json  ({len(_team_feat_cols)} features)")

    except Exception as _e:
        print(f"  [WARN] Team-RL training failed: {_e}")
        import traceback; traceback.print_exc()
        team_rl_model = None

    # =========================================================================
    # SAVE MODELS
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  Saving models ...")
    print(f"{'='*60}")

    rl_model.save_model(str(MODELS_DIR / "xgb_rl.json"))
    tot_model.save_model(str(MODELS_DIR / "xgb_total.json"))
    ml_model.save_model(str(MODELS_DIR / "xgb_ml.json"))
    print(f"  Saved: xgb_rl.json | xgb_total.json | xgb_ml.json")

    if team_rl_model is not None:
        team_rl_model.save_model(str(MODELS_DIR / "xgb_rl_team.json"))
        print(f"  Saved: xgb_rl_team.json  (team-perspective, parallel to xgb_rl.json)")

    def _save_pkl(obj, name):
        if obj is None: return
        path = MODELS_DIR / name
        path.write_bytes(pickle.dumps(obj))
        print(f"  Saved: {name}")

    _save_pkl(lgbm_rl_m,  "lgbm_rl.pkl")
    _save_pkl(lgbm_tot_m, "lgbm_total.pkl")
    _save_pkl(lgbm_ml_m,  "lgbm_ml.pkl")
    _save_pkl(cat_rl_m,   "cat_rl.pkl")
    _save_pkl(cat_tot_m,  "cat_total.pkl")
    _save_pkl(cat_ml_m,   "cat_ml.pkl")

    # ── Shadow models — native formats for fast inference ────────────────
    # These are the run-line shadow models only (primary betting market).
    # They are NOT inputs to the stacker; they produce ensemble_min/max/spread.
    if lgbm_rl_m is not None:
        try:
            shadow_lgb_path = MODELS_DIR / "lgb_shadow.json"
            lgbm_rl_m.booster_.save_model(str(shadow_lgb_path))
            print(f"  Saved: lgb_shadow.json  (shadow — excluded from stacker)")
        except Exception as e:
            print(f"  [WARN] Could not save lgb_shadow.json: {e}")

    if cat_rl_m is not None:
        try:
            shadow_cat_path = MODELS_DIR / "cat_shadow.cbm"
            cat_rl_m.save_model(str(shadow_cat_path), format="cbm")
            print(f"  Saved: cat_shadow.cbm   (shadow — excluded from stacker)")
        except Exception as e:
            print(f"  [WARN] Could not save cat_shadow.cbm: {e}")

    # Feature importances
    fi_rl = pd.DataFrame({"feature": feature_cols,
                           "gain": rl_model.feature_importances_,
                           "model": "run_line"})
    fi_tot = pd.DataFrame({"feature": feature_cols,
                            "gain": tot_model.feature_importances_,
                            "model": "total"})
    fi_ml = pd.DataFrame({"feature": feature_cols,
                           "gain": ml_model.feature_importances_,
                           "model": "moneyline"})
    fi_all = (pd.concat([fi_rl, fi_tot, fi_ml], ignore_index=True)
              .sort_values(["model","gain"], ascending=[True,False]))
    fi_all.to_csv("xgb_feature_importance.csv", index=False)

    print(f"\n  Top-10 features (run-line, XGBoost gain):")
    for _, row in fi_rl.sort_values("gain", ascending=False).head(10).iterrows():
        print(f"    {row['feature']:<45} {row['gain']:.5f}")

    val_df_edge.to_csv("xgb_val_predictions.csv", index=False)
    print(f"\n  Saved xgb_val_predictions.csv ({len(val_df_edge)} rows)")
    print(f"  Saved xgb_feature_importance.csv")

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
