"""
train_ml_model.py
=================
Train an XGBoost Full-Game Moneyline (ML) classifier + Bayesian Hierarchical
Stacker to predict P(home team wins outright).

Architecture — Full Two-Level Stack (mirrors train_f5_model.py)
---------------------------------------------------------------
  Level 1 — XGBoost:
    Label   : home_win  — 1 if home_score > away_score (no ties full-game)
    Features: SP stats, batting matchup, bullpens, park/ump/schedule
              + MC residual features (Poisson physics baseline)
    Training: 2023+2024 train, 2025 validate (LOYO across all years for final)

  Level 2 — Bayesian Hierarchical Stacker (NumPyro / NUTS MCMC):
    Model:  y ~ Bernoulli(σ(α + β·logit(p_xgb) + δ_j + γᵀ·x))
    Input:  XGBoost OOF raw probs
            + domain features (SP diffs, matchup edge, bullpen signals,
              team log-odds, rolling ML form)
            + SP handedness segment j ∈ {LvL, LvR, RvL, RvR}
    Falls back to logistic regression when NumPyro unavailable.

Key differences from the F5 pipeline:
  - Target is binary home_win (no +0.5 line, no tie-inflation math).
  - Bullpen features are INCLUDED in XGBoost (relief is decisive full-game).
  - No dual-Poisson sidecar (full-game distributional gain too small).
  - Stacker consumes bullpen_vulnerability_diff / bp_fatigue_diff as domain
    features to capture the L1-residual bullpen signal.

Outputs
-------
  models/xgb_ml.json              XGBoost ML classifier
  models/xgb_ml_calibrator.pkl    Platt sigmoid calibrator
  models/stacking_lr_ml.pkl       Bayesian Hierarchical Stacker
  models/stacking_lr_ml.npz       Full NUTS posterior trace
  models/ml_feature_cols.json     Ordered XGBoost feature column list
  models/team_ml_model.json       Doubled-dataset team perspective XGBoost
  models/team_ml_feat_cols.json   Team model feature manifest
  ml_val_predictions.csv          Validation set predictions

Usage
-----
  python train_ml_model.py --matrix feature_matrix_enriched_v2.parquet
  python train_ml_model.py --matrix feature_matrix_enriched_v2.parquet --with-2026
"""

import argparse
import gc
import json
import pickle
import sys
import warnings
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 encode errors
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import xgboost as xgb

# ── NumPyro / JAX for Bayesian Hierarchical Stacker ─────────────────────────
try:
    import jax as _jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    _NUMPYRO = True
    _JAX_PLATFORM = "cpu"
    try:
        _jax.config.update("jax_platform_name", "gpu")
        jnp.zeros(1)
        _JAX_PLATFORM = "gpu"
    except Exception:
        _jax.config.update("jax_platform_name", "cpu")
except ImportError:
    jnp = numpyro = dist = MCMC = NUTS = None
    _NUMPYRO = False
    _JAX_PLATFORM = "cpu"

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR      = Path(".")
DATA_DIR      = BASE_DIR / "data" / "statcast"
MODELS_DIR    = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEAT_MATRIX   = BASE_DIR / "feature_matrix_enriched_v2.parquet"
ACTUALS_2026  = DATA_DIR / "actuals_2026.parquet"

OUTPUT_MODEL          = MODELS_DIR / "xgb_ml.json"
OUTPUT_CALIB          = MODELS_DIR / "xgb_ml_calibrator.pkl"
OUTPUT_FEAT_COLS      = MODELS_DIR / "ml_feature_cols.json"
OUTPUT_VAL_PREDS      = BASE_DIR  / "ml_val_predictions.csv"
OUTPUT_STACKER        = MODELS_DIR / "stacking_lr_ml.pkl"
OUTPUT_STACKER_NPZ    = MODELS_DIR / "stacking_lr_ml.npz"
OUTPUT_TEAM_MODEL     = MODELS_DIR / "team_ml_model.json"
OUTPUT_TEAM_FEAT_COLS = MODELS_DIR / "team_ml_feat_cols.json"

# ---------------------------------------------------------------------------
# FEATURE CONFIG
# ---------------------------------------------------------------------------
NON_FEATURE_COLS = {
    "game_pk", "game_date", "home_team", "away_team",
    "home_starter_name", "away_starter_name", "season", "year", "split",
    # Labels
    "actual_home_win", "actual_game_total", "actual_f5_total",
    "actual_f3_total", "actual_f1_total",
    "home_score", "away_score", "home_margin",
    "home_covers_rl", "away_covers_rl", "total_runs",
    # Vegas / market lines
    "close_ml_home", "close_ml_away", "open_total", "close_total",
    "true_home_prob", "true_away_prob",
    "vegas_implied_home", "vegas_implied_away",
    # Full-game MC (scalar) — not used directly as a feature here
    "mc_expected_runs",
    # Pipeline metadata
    "source", "pull_timestamp",
    # Weather / venue (keep ones that exist in ML model; air_density + roof kept)
    "temp_f", "wind_mph", "wind_bearing",
    # 1st-inning features not in the ML manifest
    "home_1st_inn_run_rate", "away_1st_inn_run_rate",
    "home_sp_1st_k_pct", "home_sp_1st_bb_pct", "home_sp_1st_xwoba",
    "away_sp_1st_k_pct", "away_sp_1st_bb_pct", "away_sp_1st_xwoba",
    "sp_1st_k_pct_diff", "sp_1st_xwoba_diff",
}

# Pass 1 drops (mirror of F5 — 100% null pybaseball gaps + collinear + zero-gain)
_PASS1_DROP = {
    "home_sp_swing_rv_per100",    "away_sp_swing_rv_per100",    "sp_swing_rv_diff",
    "home_sp_take_rv_per100",     "away_sp_take_rv_per100",     "sp_take_rv_diff",
    "home_sp_ff_h_break_inch",    "away_sp_ff_h_break_inch",    "sp_ff_h_break_diff",
    "home_sp_ff_v_break_inch",    "away_sp_ff_v_break_inch",    "sp_ff_v_break_diff",
    "home_sp_arsenal_weighted_rv","away_sp_arsenal_weighted_rv","sp_arsenal_rv_diff",
    "home_sp_primary_whiff_pct",  "away_sp_primary_whiff_pct",  "sp_primary_whiff_diff",
    "home_sp_primary_putaway_pct","away_sp_primary_putaway_pct","sp_primary_putaway_diff",
    "home_sp_arsenal_quality_ratio","away_sp_arsenal_quality_ratio","sp_arsenal_quality_ratio_diff",
    "home_sp_k_bb_ratio",    "away_sp_k_bb_ratio",    "sp_k_bb_ratio_diff",
    "home_sp_k_bb_ratio_10d","away_sp_k_bb_ratio_10d","sp_k_bb_ratio_10d_diff",
    "home_game_local_hour",  "away_game_local_hour",
    "away_sp_il_return_flag", "away_sp_starts_since_il", "home_sp_starts_since_il",
    "is_day_game",
}
NON_FEATURE_COLS = NON_FEATURE_COLS | _PASS1_DROP

# Bullpen features INCLUDED for ML (unlike F5).  These feed both the L1 XGBoost
# and — via bp_fatigue_diff / bullpen_vulnerability_diff — the stacker.
BULLPEN_FEATURES = [
    "home_bp_era", "home_bp_k9", "home_bp_bb9", "home_bp_hr9",
    "home_bp_whip", "home_bp_gb_pct",
    "away_bp_era", "away_bp_k9", "away_bp_bb9", "away_bp_hr9",
    "away_bp_whip", "away_bp_gb_pct",
    "bp_era_diff", "bp_k9_diff", "bp_whip_diff",
    "home_bp_fatigue_72h", "away_bp_fatigue_72h", "bp_fatigue_diff",
    "home_bp_top3_fatigue_3d", "away_bp_top3_fatigue_3d", "bp_top3_fatigue_diff",
    "home_bullpen_vulnerability", "away_bullpen_vulnerability",
    "bullpen_vulnerability_diff",
    "home_bp_cluster", "away_bp_cluster",
]

# MC residual columns used as features (F5 Monte Carlo outputs are informative
# for ML too — low-total-environment games correlate with home-win variance).
MC_RESIDUAL_COLS = [
    "mc_f5_home_win_pct",
    "mc_f5_away_win_pct",
    "mc_f5_tie_pct",
    "mc_f5_expected_total",
    "mc_f5_home_cover_pct",
]

XGB_PARAMS = {
    "tree_method":      "hist",
    "n_jobs":           -1,
    "random_state":     42,
    "learning_rate":    0.04,
    "max_depth":        4,
    "min_child_weight": 15,
    "subsample":        0.80,
    "colsample_bytree": 0.70,
    "reg_alpha":        0.10,
    "reg_lambda":       1.50,
    "gamma":            0.05,
    "n_estimators":     600,
    "eval_metric":      "logloss",
    "objective":        "binary:logistic",
}

YEAR_DECAY = {2023: 0.70, 2024: 1.00, 2025: 1.50, 2026: 2.00}

# ---------------------------------------------------------------------------
# BAYESIAN STACKER CONFIG
# ---------------------------------------------------------------------------
_N_SEGMENTS   = 4
_SEG_LABELS   = ["LvL", "LvR", "RvL", "RvR"]

# Domain features fed into the stacker.  No Poisson sidecar features — ML uses
# the team-log-odds, rolling-form, and bullpen-diff signals instead.
ML_STACKING_FEATURES = [
    "sp_k_pct_diff",
    "sp_xwoba_diff",
    "sp_kminusbb_diff",
    "batting_matchup_edge",
    "batting_matchup_edge_10d",
    "home_sp_il_return_flag",
    "away_sp_il_return_flag",
    "sp_k_pct_10d_diff",
    "sp_xwoba_10d_diff",
    "bp_fatigue_diff",                   # bullpen fatigue residual
    "bullpen_vulnerability_diff",        # bullpen quality residual
    "team_ml_log_odds",                  # logit(p_home) - logit(p_away)
    "rolling_ml_form_diff",              # home minus away 15-game form residual
    "mc_f5_expected_total",              # scoring environment carries ML signal
    # v8.0 script probability signals
    "p_script_a2",                       # A2 Away Dominance proxy
    "p_script_b",                        # B Explosion (high-total) proxy
    "p_script_c",                        # C Elite Duel (low-total) proxy
]

CALIB_BINS   = [0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.0]
CALIB_LABELS = ["<45", "45-50", "50-55", "55-60", "60-65", "65-70", "70-75", "75-80", ">80"]


# ---------------------------------------------------------------------------
# BAYESIAN STACKER INFRASTRUCTURE
# ---------------------------------------------------------------------------

def _derive_segment_id(df: pd.DataFrame) -> np.ndarray:
    """SP handedness matchup segment: home_throws_R*2 + away_throws_R → 0–3."""
    if "home_sp_p_throws_R" not in df.columns or "away_sp_p_throws_R" not in df.columns:
        return np.full(len(df), 3, dtype=np.int32)
    h = df["home_sp_p_throws_R"].fillna(1).astype(int).values
    a = df["away_sp_p_throws_R"].fillna(1).astype(int).values
    return (h * 2 + a).astype(np.int32)


def _numpyro_stacker_model(p_global, segment_id, X_domain,
                            n_segments, n_domain, y_obs=None):
    alpha = numpyro.sample("alpha", dist.Normal(0.0, 1.0))
    beta  = numpyro.sample("beta",  dist.Normal(1.0, 0.5))

    sigma_delta = numpyro.sample("sigma_delta", dist.HalfCauchy(1.0))
    with numpyro.plate("segments", n_segments):
        delta = numpyro.sample("delta", dist.Normal(0.0, sigma_delta))

    with numpyro.plate("features", n_domain):
        gamma = numpyro.sample("gamma", dist.Normal(0.0, 0.3))

    logit_p = (jnp.log(jnp.clip(p_global, 1e-6, 1-1e-6)) -
               jnp.log(1 - jnp.clip(p_global, 1e-6, 1-1e-6)))
    theta = alpha + beta * logit_p + delta[segment_id] + X_domain @ gamma

    with numpyro.plate("data", len(p_global)):
        numpyro.sample("y", dist.Bernoulli(logits=theta), obs=y_obs)


class BayesianStackerML:
    """
    Level-2 Bayesian Hierarchical Stacker for full-game moneyline predictions.

    Mirrors BayesianStackerF5 — stores posterior means and does closed-form
    inference (no MCMC at test time).
    """
    def __init__(self, alpha, beta, delta, gamma,
                 stacking_feature_names, fill_values, n_segments=4,
                 posterior_path=None):
        self.alpha                  = float(alpha)
        self.beta                   = float(beta)
        self.delta                  = np.asarray(delta, dtype=float)
        self.gamma                  = np.asarray(gamma, dtype=float)
        self.stacking_feature_names = stacking_feature_names
        self.fill_values            = fill_values
        self.n_segments             = n_segments
        self.posterior_path         = posterior_path

    def _build_X_domain(self, feat_df: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(index=feat_df.index)
        for col in self.stacking_feature_names:
            X[col] = feat_df[col] if col in feat_df.columns else self.fill_values.get(col, 0.0)
        for col, val in self.fill_values.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)
        return X.values.astype(float)

    def predict(self, xgb_raw, feat_df, segment_id=None) -> np.ndarray:
        xgb_raw = np.asarray(xgb_raw, dtype=float).ravel()
        n = len(xgb_raw)
        seg = (np.full(n, self.n_segments - 1, dtype=int) if segment_id is None
               else np.asarray(segment_id, dtype=int).ravel())
        if len(seg) == 1 and n > 1:
            seg = np.full(n, int(seg[0]), dtype=int)
        X_dom   = self._build_X_domain(feat_df)
        logit_p = (np.log(np.clip(xgb_raw, 1e-6, 1-1e-6)) -
                   np.log(1 - np.clip(xgb_raw, 1e-6, 1-1e-6)))
        theta = self.alpha + self.beta * logit_p + self.delta[seg] + X_dom @ self.gamma
        return 1.0 / (1.0 + np.exp(-theta))

    def predict_proba(self, xgb_raw, feat_df, segment_id=None) -> np.ndarray:
        p = self.predict(xgb_raw, feat_df, segment_id)
        return np.column_stack([1.0 - p, p])


def train_ml_stacker(
    oof_probs:   np.ndarray,
    oof_feat_df: pd.DataFrame,
    oof_labels:  np.ndarray,
    oof_segs:    np.ndarray,
    num_warmup:  int = 500,
    num_samples: int = 1_000,
    num_chains:  int = 2,
) -> BayesianStackerML:
    feat_names = [c for c in ML_STACKING_FEATURES if c in oof_feat_df.columns]
    fill_vals  = {c: float(oof_feat_df[c].median()) for c in feat_names}

    X_feat = oof_feat_df[feat_names].copy()
    for col, val in fill_vals.items():
        X_feat[col] = X_feat[col].fillna(val)
    X_domain = X_feat.values.astype(float)
    n_domain = X_domain.shape[1]

    if not _NUMPYRO:
        print("  [WARN] NumPyro not available — using LR fallback stacker")
        X_stack = np.hstack([oof_probs.reshape(-1, 1), X_domain])
        lr = LogisticRegression(C=10, solver="lbfgs", max_iter=1000)
        lr.fit(X_stack, oof_labels)
        model = BayesianStackerML(
            alpha=float(lr.intercept_[0]),
            beta=float(lr.coef_[0, 0]),
            delta=np.zeros(_N_SEGMENTS),
            gamma=lr.coef_[0, 1:],
            stacking_feature_names=feat_names,
            fill_values=fill_vals,
        )
        OUTPUT_STACKER.write_bytes(pickle.dumps(model))
        print(f"  Fallback stacker saved -> {OUTPUT_STACKER}")
        return model

    seg_counts = {j: int((oof_segs == j).sum()) for j in range(_N_SEGMENTS)}
    print(f"\n{'='*60}")
    print(f"  ML BAYESIAN HIERARCHICAL STACKER  ({_JAX_PLATFORM.upper()})")
    print(f"{'='*60}")
    print(f"  OOF rows : {len(oof_probs):,}  |  domain features: {n_domain}")
    print(f"  Segments : " + "  ".join(
        f"{_SEG_LABELS[j]}={seg_counts[j]}" for j in range(_N_SEGMENTS)
    ))
    print(f"  MCMC     : {num_chains} chains x ({num_warmup} warmup + {num_samples} samples)")

    p_jax  = jnp.array(oof_probs.astype(float))
    seg_jx = jnp.array(oof_segs.astype(int))
    X_jax  = jnp.array(X_domain)
    y_jax  = jnp.array(oof_labels.astype(float))

    kernel = NUTS(_numpyro_stacker_model)
    mcmc   = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                  num_chains=num_chains, progress_bar=True)
    mcmc.run(_jax.random.PRNGKey(42),
             p_jax, seg_jx, X_jax, _N_SEGMENTS, n_domain, y_obs=y_jax)
    samples = mcmc.get_samples()

    alpha_hat = float(np.mean(samples["alpha"]))
    beta_hat  = float(np.mean(samples["beta"]))
    delta_hat = np.mean(samples["delta"], axis=0)
    gamma_hat = np.mean(samples["gamma"], axis=0)

    print(f"\n  Posterior means:")
    print(f"    alpha  {alpha_hat:+.4f} +/- {float(np.std(samples['alpha'])):.4f}")
    print(f"    beta   {beta_hat:+.4f} +/- {float(np.std(samples['beta'])):.4f}")
    print(f"    sigma_delta {float(np.mean(samples['sigma_delta'])):+.4f}")
    for j in range(_N_SEGMENTS):
        print(f"    delta[{_SEG_LABELS[j]}] n={seg_counts[j]:>4}  "
              f"{delta_hat[j]:+.4f}")
    for k, feat in enumerate(feat_names):
        print(f"    gamma  {feat:<32} {gamma_hat[k]:+.4f}")

    np.savez(str(OUTPUT_STACKER_NPZ), **{k: np.array(v) for k, v in samples.items()})
    print(f"\n  Posterior trace -> {OUTPUT_STACKER_NPZ}")

    model = BayesianStackerML(
        alpha=alpha_hat, beta=beta_hat, delta=delta_hat, gamma=gamma_hat,
        stacking_feature_names=feat_names, fill_values=fill_vals,
        n_segments=_N_SEGMENTS, posterior_path=str(OUTPUT_STACKER_NPZ),
    )
    OUTPUT_STACKER.write_bytes(pickle.dumps(model))
    print(f"  BayesianStackerML -> {OUTPUT_STACKER}")

    stk_probs = model.predict(oof_probs, oof_feat_df, oof_segs)
    auc_xgb   = roc_auc_score(oof_labels, oof_probs)
    auc_stk   = roc_auc_score(oof_labels, stk_probs)
    ll_xgb    = log_loss(oof_labels, oof_probs)
    ll_stk    = log_loss(oof_labels, stk_probs)
    print(f"\n  OOF performance:")
    print(f"    XGBoost L1 (raw)       AUC={auc_xgb:.4f}  LL={ll_xgb:.4f}")
    print(f"    Bayesian Stacker L2    AUC={auc_stk:.4f}  LL={ll_stk:.4f}")
    print(f"    Net AUC delta: {auc_stk - auc_xgb:+.4f}")
    return model


# ---------------------------------------------------------------------------
# ROLLING ML FORM (opp-adjusted 15-game team form)
# ---------------------------------------------------------------------------

def _compute_rolling_adj_ml_form(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling 15-game opp-adjusted ML form.  Per-game score (home perspective):
      adj_home = (home_win - mc_f5_home_win_pct)
    Rolling mean of that residual over the last 15 team-games, shift(1) to
    exclude the current game.  Falls back to 0.0 when <5 prior games.
    """
    required = ["mc_f5_home_win_pct", "mc_f5_away_win_pct", "home_win"]
    missing = [c for c in required if c not in merged.columns]
    if missing:
        print(f"  [WARN] Missing columns for rolling_adj_ml_form: {missing} — skipping")
        for c in ["home_rolling_adj_ml_form", "away_rolling_adj_ml_form",
                  "rolling_ml_form_diff"]:
            merged[c] = 0.0
        return merged

    mc_home = merged["mc_f5_home_win_pct"].fillna(0.5)
    mc_away = merged["mc_f5_away_win_pct"].fillna(0.5)
    # Residual: actual home_win (0/1) minus physics P(home wins thru 5)
    adj_home = merged["home_win"].astype(float) - mc_home
    # Mirror away perspective: away won iff home_win==0 (no ties full-game)
    adj_away = (1.0 - merged["home_win"].astype(float)) - mc_away

    home_long = pd.DataFrame({
        "game_date": merged["game_date"],
        "team":      merged["home_team"],
        "adj_score": adj_home.values,
    })
    away_long = pd.DataFrame({
        "game_date": merged["game_date"],
        "team":      merged["away_team"],
        "adj_score": adj_away.values,
    })
    long_df = pd.concat([home_long, away_long], ignore_index=True)

    parts = []
    for team, grp in long_df.groupby("team"):
        g = grp.sort_values("game_date").reset_index(drop=True)
        form = g["adj_score"].rolling(15, min_periods=5).mean().shift(1)
        parts.append(pd.DataFrame({
            "game_date": g["game_date"],
            "team":      team,
            "form":      form,
        }))

    team_rolling = pd.concat(parts, ignore_index=True)
    team_rolling["game_date"] = pd.to_datetime(team_rolling["game_date"])
    global_mean = float(team_rolling["form"].mean(skipna=True))
    team_rolling["form"] = team_rolling["form"].fillna(global_mean)
    team_rolling = team_rolling.drop_duplicates(subset=["game_date", "team"], keep="last")

    merged = merged.merge(
        team_rolling.rename(columns={
            "form": "home_rolling_adj_ml_form",
            "team": "home_team",
        }),
        on=["game_date", "home_team"], how="left",
    )
    merged = merged.merge(
        team_rolling.rename(columns={
            "form": "away_rolling_adj_ml_form",
            "team": "away_team",
        }),
        on=["game_date", "away_team"], how="left",
    )

    merged["home_rolling_adj_ml_form"] = merged["home_rolling_adj_ml_form"].fillna(global_mean)
    merged["away_rolling_adj_ml_form"] = merged["away_rolling_adj_ml_form"].fillna(global_mean)
    merged["rolling_ml_form_diff"] = (
        merged["home_rolling_adj_ml_form"] - merged["away_rolling_adj_ml_form"]
    )

    print(f"    rolling_adj_ml_form: "
          f"home_mean={merged['home_rolling_adj_ml_form'].mean():.4f}  "
          f"diff_std={merged['rolling_ml_form_diff'].std():.4f}  "
          f"fallback={global_mean:.4f}")

    return merged


# ---------------------------------------------------------------------------
# LABELS
# ---------------------------------------------------------------------------

def load_ml_labels(fm: pd.DataFrame) -> pd.DataFrame:
    """
    Return DataFrame with [game_pk, home_win, year].

    Label source priority:
      1. Feature matrix has `actual_home_win` — use directly.
      2. Fallback: derive from home_score/away_score when present.
    """
    if "actual_home_win" in fm.columns:
        lbl = fm[["game_pk", "actual_home_win"]].copy()
        lbl = lbl.rename(columns={"actual_home_win": "home_win"})
    elif "home_score" in fm.columns and "away_score" in fm.columns:
        lbl = fm[["game_pk", "home_score", "away_score"]].copy()
        lbl["home_win"] = (lbl["home_score"] > lbl["away_score"]).astype("Int64")
        lbl = lbl[["game_pk", "home_win"]]
    else:
        raise RuntimeError(
            "Feature matrix must contain either 'actual_home_win' "
            "or 'home_score'/'away_score' to derive the ML label."
        )
    return lbl


# ---------------------------------------------------------------------------
# SCRIPT PROBABILITY INJECTION (v8.0)
# ---------------------------------------------------------------------------

_SGP_DIR = BASE_DIR / "data/sgp"

_SCRIPT_MAP = {
    "A2_Dominance": "p_script_a2",
    "B_Explosion":  "p_script_b",
    "C_EliteDuel":  "p_script_c",
}


def _load_sgp_real_probs() -> pd.DataFrame:
    """
    Scan data/sgp/sgp_live_edge_*.csv and return a wide-format DataFrame:
      (game_date, home_team) -> p_script_a2, p_script_b, p_script_c

    Each CSV contains one row per (game, script); we pivot to wide.
    """
    if not _SGP_DIR.exists():
        return pd.DataFrame()

    frames = []
    for csv_path in sorted(_SGP_DIR.glob("sgp_live_edge_*.csv")):
        # Parse date from filename: sgp_live_edge_2026_04_24.csv -> 2026-04-24
        stem = csv_path.stem  # sgp_live_edge_2026_04_24
        parts = stem.split("_")
        if len(parts) >= 6 and parts[3].isdigit() and len(parts[3]) == 4:
            try:
                game_date = f"{parts[3]}-{parts[4]}-{parts[5]}"
            except IndexError:
                continue
        else:
            continue

        try:
            df = pd.read_csv(csv_path, usecols=["home_team", "script", "p_joint_copula"])
        except Exception:
            continue

        df = df[df["script"].isin(_SCRIPT_MAP)].copy()
        df["game_date"] = pd.to_datetime(game_date)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    all_sgp = pd.concat(frames, ignore_index=True)
    wide = all_sgp.pivot_table(
        index=["game_date", "home_team"],
        columns="script",
        values="p_joint_copula",
        aggfunc="max",
    ).reset_index()
    wide.columns.name = None

    # Rename script -> p_script_*
    wide = wide.rename(columns={s: c for s, c in _SCRIPT_MAP.items()
                                  if s in wide.columns})
    return wide


def _inject_script_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add p_script_a2, p_script_b, p_script_c to the training dataframe.

    Priority:
      1. Real p_joint_copula from sgp_live_edge CSVs (2026 games only)
      2. Proxy derivations from feature-matrix stats (all years)

    Proxy formulas
    --------------
    p_script_a2  (A2 Away Dominance: away SP better, game goes under)
        sigmoid(10 * (away_sp_k_pct_10d - home_sp_k_pct_10d - 0.02))

    p_script_b   (B Explosion: high-total environment, offenses dominant)
        sigmoid(0.5 * (close_total - 8.8))   — centred at league-avg

    p_script_c   (C Elite Duel: both SPs elite, low total)
        sigmoid(-0.5 * (close_total - 8.2))   — inverse of B
    """
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    # --- proxy scores (all rows) ---
    away_k = df.get("away_sp_k_pct_10d", pd.Series(0.22, index=df.index))
    home_k = df.get("home_sp_k_pct_10d", pd.Series(0.22, index=df.index))
    total  = df.get("close_total",        pd.Series(8.8,  index=df.index))

    df["p_script_a2"] = _sigmoid(10 * (away_k.fillna(0.22) - home_k.fillna(0.22) - 0.02)).round(5)
    df["p_script_b"]  = _sigmoid(0.5 * (total.fillna(8.8)  - 8.8)).round(5)
    df["p_script_c"]  = _sigmoid(-0.5 * (total.fillna(8.2) - 8.2)).round(5)

    # --- override with real SGP probs where available ---
    sgp = _load_sgp_real_probs()
    if not sgp.empty:
        df["game_date"] = pd.to_datetime(df["game_date"])
        sgp["game_date"] = pd.to_datetime(sgp["game_date"])
        merged_sgp = df[["game_pk", "game_date", "home_team"]].merge(
            sgp, on=["game_date", "home_team"], how="left", suffixes=("", "_real"))

        for proxy_col in ["p_script_a2", "p_script_b", "p_script_c"]:
            real_col = proxy_col + "_real"
            if real_col in merged_sgp.columns:
                real_vals = merged_sgp.set_index("game_pk")[real_col]
                mask = df["game_pk"].map(real_vals).notna()
                df.loc[mask, proxy_col] = df.loc[mask, "game_pk"].map(real_vals)

        n_real = merged_sgp[["p_script_a2_real", "p_script_b_real", "p_script_c_real"]
                             ].notna().any(axis=1).sum() if all(
            c in merged_sgp.columns for c in
            ["p_script_a2_real", "p_script_b_real", "p_script_c_real"]
        ) else 0
        if n_real:
            print(f"    [script injection] {n_real} rows with real SGP probs; "
                  f"remainder use proxies.")

    return df


# ---------------------------------------------------------------------------
# DATASET BUILD
# ---------------------------------------------------------------------------

def build_dataset(include_2026: bool = False) -> tuple[pd.DataFrame, list[str]]:
    print("\n[1] Loading feature matrix …")
    fm = pd.read_parquet(FEAT_MATRIX)
    print(f"    Feature matrix: {fm.shape[0]} rows × {fm.shape[1]} cols")

    missing_mc = [c for c in MC_RESIDUAL_COLS if c not in fm.columns]
    if missing_mc:
        print(f"    [WARN] MC residual columns missing: {missing_mc}")
    else:
        mc_nan_pct = fm[MC_RESIDUAL_COLS].isna().mean().mean()
        print(f"    MC residual cols present | NaN rate: {mc_nan_pct:.1%}")

    print("\n[2] Extracting ML labels …")
    lbl = load_ml_labels(fm)
    # Drop rows without labels BEFORE merge to preserve row count clarity
    lbl = lbl.dropna(subset=["home_win"]).copy()
    lbl["home_win"] = lbl["home_win"].astype(int)

    fm = fm.drop(columns=[c for c in ["actual_home_win"] if c in fm.columns])
    merged = fm.merge(lbl, on="game_pk", how="inner")
    print(f"    After label join: {len(merged)} rows")

    # Ensure year column
    if "year" not in merged.columns:
        merged["year"] = pd.to_datetime(merged["game_date"]).dt.year
    merged["year"] = merged["year"].astype(int)

    if not include_2026:
        merged = merged[merged["year"].isin([2023, 2024, 2025])].copy()
        print(f"    After dropping 2026: {len(merged)} rows")

    merged["game_date"] = pd.to_datetime(merged["game_date"])
    merged = merged.sort_values("game_date").reset_index(drop=True)

    print("    Computing opponent-quality-adjusted ML form …")
    merged = _compute_rolling_adj_ml_form(merged)

    print("    Injecting script probability features (v8.0) …")
    merged = _inject_script_features(merged)

    print(f"    Label breakdown: home_win_rate={merged['home_win'].mean():.3f}")

    exclude = NON_FEATURE_COLS | {
        "home_win",
        # Rolling-form diff is a stacker feature; the per-team rolling columns
        # remain available as model features (captured below via inclusion).
        "rolling_ml_form_diff",
    }
    # The per-team rolling_adj_ml_form columns ARE included as XGB features
    # (they're zero-leakage — shift(1) excludes the current game).
    feat_cols = [c for c in merged.columns if c not in exclude]

    print(f"    Total features: {len(feat_cols)} columns")
    return merged, feat_cols


# ---------------------------------------------------------------------------
# TRAIN HELPERS
# ---------------------------------------------------------------------------

def _sample_weights(df: pd.DataFrame) -> np.ndarray:
    return df["year"].map(YEAR_DECAY).fillna(1.0).values


def _train_xgb(X_tr, y_tr, sw_tr, X_val, y_val) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=40)
    model.fit(X_tr, y_tr, sample_weight=sw_tr,
              eval_set=[(X_val, y_val)], verbose=False)
    return model


def _print_metrics(label: str, y_true, y_prob):
    auc  = roc_auc_score(y_true, y_prob)
    ll   = log_loss(y_true, y_prob)
    bs   = brier_score_loss(y_true, y_prob)
    base = log_loss(y_true, np.full(len(y_true), y_true.mean()))
    print(f"  {label:30s}  AUC={auc:.4f}  LogLoss={ll:.4f}  "
          f"Brier={bs:.4f}  BaseLL={base:.4f}")


# ---------------------------------------------------------------------------
# TEAM-PERSPECTIVE MODEL
# ---------------------------------------------------------------------------

def _flip_team_perspective(
    df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str = "home_win",
) -> pd.DataFrame:
    """
    Build away-perspective mirror of a game DataFrame.
    - Swaps home_X ↔ away_X column pairs
    - Negates _diff and matchup_edge columns
    - Swaps mc_f5_home_* ↔ mc_f5_away_* and recomputes mc_f5_home_cover_pct
    - Flips label: away_win = 1 - home_win (no ties full-game)
    """
    df_flip = df.copy()
    feat_set = set(feat_cols)

    swapped: set[str] = set()
    for hc in list(feat_set):
        if not hc.startswith("home_"):
            continue
        ac = "away_" + hc[5:]
        if ac in feat_set and hc not in swapped and ac not in swapped:
            tmp         = df_flip[hc].copy()
            df_flip[hc] = df_flip[ac]
            df_flip[ac] = tmp
            swapped.add(hc); swapped.add(ac)

    mc_home = [c for c in feat_set if c.startswith("mc_f5_home_") and not c.endswith("_cover_pct")]
    for hc in mc_home:
        ac = "mc_f5_away_" + hc[len("mc_f5_home_"):]
        if ac in feat_set:
            tmp         = df_flip[hc].copy()
            df_flip[hc] = df_flip[ac]
            df_flip[ac] = tmp

    if ("mc_f5_home_win_pct" in df_flip.columns and
            "mc_f5_tie_pct" in df_flip.columns and
            "mc_f5_home_cover_pct" in df_flip.columns):
        df_flip["mc_f5_home_cover_pct"] = (
            df_flip["mc_f5_home_win_pct"] + df_flip["mc_f5_tie_pct"]
        )

    for c in feat_cols:
        if (c.endswith("_diff") or "matchup_edge" in c) and c in df_flip.columns:
            df_flip[c] = -df_flip[c]

    if label_col in df_flip.columns:
        df_flip[label_col] = 1 - df_flip[label_col].astype(int)
    return df_flip


def _build_team_dataset(
    df: pd.DataFrame,
    feat_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    home_rows = df.copy()
    home_rows["is_home"] = 1
    away_rows = _flip_team_perspective(df, feat_cols)
    away_rows["is_home"] = 0
    doubled = pd.concat([home_rows, away_rows], ignore_index=True)
    team_feat_cols = feat_cols + ["is_home"]
    return doubled, team_feat_cols


def _compute_log_odds_ratio(p_home: np.ndarray, p_away: np.ndarray) -> np.ndarray:
    eps = 1e-6
    lo_h = np.log(np.clip(p_home, eps, 1-eps)) - np.log(np.clip(1-p_home, eps, 1-eps))
    lo_a = np.log(np.clip(p_away, eps, 1-eps)) - np.log(np.clip(1-p_away, eps, 1-eps))
    return lo_h - lo_a


# ---------------------------------------------------------------------------
# LOYO OOF GENERATOR
# ---------------------------------------------------------------------------

def _generate_oof_for_stacker(
    df: pd.DataFrame,
    feat_cols: list[str],
    years: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Leave-One-Year-Out CV that produces zero-leakage OOF predictions for:
      - L1 single-perspective XGB prob (oof_probs)
      - Team model log-odds (team_ml_log_odds)
    """
    required_cols = ["home_win", "year"]
    full = df.dropna(subset=required_cols).reset_index(drop=True).copy()
    full["year"] = full["year"].astype(int)

    if years is None:
        years = sorted(full["year"].unique().tolist())
    years = [int(y) for y in years if int(y) in set(full["year"].unique())]

    eligible = [y for y in years if len(set(years) - {y}) >= 1]
    skipped  = [y for y in years if y not in eligible]
    if skipped:
        print(f"  [OOF][WARN] Skipping singleton years: {skipped}")
    if len(eligible) == 0:
        raise ValueError("LOYO requires at least 2 distinct years in df.")

    oof_mask = full["year"].isin(eligible).values
    oof_df   = full.loc[oof_mask].reset_index(drop=True).copy()
    n_oof    = len(oof_df)

    oof_probs        = np.zeros(n_oof, dtype=float)
    oof_team_logodds = np.zeros(n_oof, dtype=float)
    oof_labels       = oof_df["home_win"].values.astype(int)

    year_pos = {yr: np.where(oof_df["year"].values == yr)[0] for yr in eligible}

    print(f"\n  [OOF] LOYO across {eligible} for stacker training")
    print(f"  [OOF] Pool rows: {len(full):,} total  |  OOF rows: {n_oof:,}")

    for val_yr in eligible:
        tr = full[full["year"] != val_yr].reset_index(drop=True)
        va = full[full["year"] == val_yr].reset_index(drop=True)
        if len(tr) == 0 or len(va) == 0:
            print(f"    [skip] year={val_yr}  tr={len(tr)} va={len(va)}")
            continue

        X_tr  = tr[feat_cols].fillna(0).values.astype(np.float32)
        y_tr  = tr["home_win"].values.astype(int)
        sw_tr = _sample_weights(tr)
        X_va  = va[feat_cols].fillna(0).values.astype(np.float32)
        y_va  = va["home_win"].values.astype(int)

        # L1 XGB
        m   = _train_xgb(X_tr, y_tr, sw_tr, X_va, y_va)
        raw = m.predict_proba(X_va)[:, 1]

        # Team model (doubled)
        tr_doubled, team_feat_cols = _build_team_dataset(tr, feat_cols)
        X_tm_tr = tr_doubled[team_feat_cols].fillna(0).values.astype(np.float32)
        y_tm_tr = tr_doubled["home_win"].values.astype(int)
        sw_tm   = _sample_weights(tr_doubled)
        va_es   = va.copy(); va_es["is_home"] = 1
        X_tm_es = va_es[team_feat_cols].fillna(0).values.astype(np.float32)
        tm_m = _train_xgb(X_tm_tr, y_tm_tr, sw_tm, X_tm_es, y_va)

        va_home = va.copy(); va_home["is_home"] = 1
        p_h = tm_m.predict_proba(
            va_home[team_feat_cols].fillna(0).values.astype(np.float32))[:, 1]
        va_flip = _flip_team_perspective(va, feat_cols); va_flip["is_home"] = 0
        p_a = tm_m.predict_proba(
            va_flip[team_feat_cols].fillna(0).values.astype(np.float32))[:, 1]
        lo_fold = _compute_log_odds_ratio(p_h, p_a)

        pos = year_pos[val_yr]
        oof_probs[pos]        = raw
        oof_team_logodds[pos] = lo_fold

        def _safe_auc(y, p):
            return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
        print(f"    Fold {val_yr}: tr={len(tr):>5,}  va={len(va):>5,}  "
              f"XGB-AUC={_safe_auc(y_va, raw):.4f}  "
              f"team-logodds-AUC={_safe_auc(y_va, lo_fold):.4f}")

        del m, raw, tm_m, tr_doubled, X_tm_tr, y_tm_tr, sw_tm, va_es, va_home, va_flip
        del X_tr, y_tr, sw_tr, X_va, y_va, tr, va, lo_fold, p_h, p_a
        gc.collect()

    assert len(oof_df) == n_oof == oof_probs.shape[0], \
        f"OOF length mismatch: oof_df={len(oof_df)}  arrays={oof_probs.shape[0]}"

    def _safe_auc(y, p):
        return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    print(f"  [OOF] Combined: n={n_oof:,}  "
          f"XGB={_safe_auc(oof_labels, oof_probs):.4f}  "
          f"team-logodds={_safe_auc(oof_labels, oof_team_logodds):.4f}")

    oof_df["team_ml_log_odds"] = oof_team_logodds
    oof_segs = _derive_segment_id(oof_df)
    return oof_probs, oof_labels, oof_df, oof_segs


# ---------------------------------------------------------------------------
# TRAIN DEFAULT (2023+2024 → 2025)
# ---------------------------------------------------------------------------

def _fixed_bin_calibration(y_true: np.ndarray, y_prob: np.ndarray, label: str = ""):
    df_c = pd.DataFrame({"pred": y_prob, "actual": y_true})
    df_c["band"] = pd.cut(df_c["pred"], bins=CALIB_BINS, labels=CALIB_LABELS)
    print(f"\n  Fixed-bin calibration{' — ' + label if label else ''}:")
    print(f"  {'Band':>8}  {'n':>5}  {'Pred':>7}  {'Actual':>7}  {'Diff':>8}")
    print(f"  {'-'*46}")
    for band, grp in df_c.groupby("band", observed=True):
        if len(grp) == 0:
            continue
        diff = grp["actual"].mean() - grp["pred"].mean()
        print(f"  {str(band):>8}  {len(grp):>5}  "
              f"{grp['pred'].mean():>7.3f}  {grp['actual'].mean():>7.3f}  {diff:>+8.3f}")


def train_default(df: pd.DataFrame, feat_cols: list[str], val_year: int = 2025):
    train_years = [y for y in [2023, 2024, 2025] if y < val_year]
    print(f"\n[3] Default split: train {train_years} / validate {val_year}  [label: home_win]")
    train = df[df["year"].isin(train_years)]
    val   = df[df["year"] == val_year]
    print(f"    Train: {len(train)} | Val: {len(val)}")

    X_tr  = train[feat_cols].fillna(0).values.astype(np.float32)
    y_tr  = train["home_win"].values.astype(int)
    sw_tr = _sample_weights(train)
    X_val = val[feat_cols].fillna(0).values.astype(np.float32)
    y_val = val["home_win"].values.astype(int)

    model   = _train_xgb(X_tr, y_tr, sw_tr, X_val, y_val)
    raw_val = model.predict_proba(X_val)[:, 1]

    print("\n  Raw XGBoost:")
    _print_metrics("Validation 2025", y_val, raw_val)

    print(f"\n[3b] Generating OOF predictions (LOYO on {train_years}) …")
    oof_probs, oof_labels, oof_df, oof_segs = _generate_oof_for_stacker(
        df, feat_cols, years=train_years,
    )

    cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    cal.fit(oof_probs.reshape(-1, 1), oof_labels)
    cal_val = cal.predict_proba(raw_val.reshape(-1, 1))[:, 1]
    print("  Platt calibrated (OOF-fitted):")
    _print_metrics("Validation 2025", y_val, cal_val)
    _fixed_bin_calibration(y_val, cal_val, "OOF-Platt, 2025 val")

    # Team log-odds on 2025 val
    print("\n[3c] Team model: training doubled-dataset on 2023+2024 for val log-odds …")
    train_reset = train.reset_index(drop=True)
    val_reset   = val.reset_index(drop=True)
    train_doubled_tm, team_feat_cols_tm = _build_team_dataset(train_reset, feat_cols)
    X_tm_tr  = train_doubled_tm[team_feat_cols_tm].fillna(0).values.astype(np.float32)
    y_tm_tr  = train_doubled_tm["home_win"].values.astype(int)
    sw_tm    = _sample_weights(train_doubled_tm)
    val_es   = val_reset.copy(); val_es["is_home"] = 1
    X_tm_es  = val_es[team_feat_cols_tm].fillna(0).values.astype(np.float32)
    tm_model_val = _train_xgb(X_tm_tr, y_tm_tr, sw_tm, X_tm_es, y_val)

    val_home_tm = val_reset.copy(); val_home_tm["is_home"] = 1
    p_h_val = tm_model_val.predict_proba(
        val_home_tm[team_feat_cols_tm].fillna(0).values.astype(np.float32))[:, 1]
    val_flip_tm = _flip_team_perspective(val_reset, feat_cols); val_flip_tm["is_home"] = 0
    p_a_val = tm_model_val.predict_proba(
        val_flip_tm[team_feat_cols_tm].fillna(0).values.astype(np.float32))[:, 1]
    log_odds_val = _compute_log_odds_ratio(p_h_val, p_a_val)

    val = val.copy()
    val["team_ml_log_odds"] = log_odds_val
    print(f"  Team log-odds on 2025: AUC={roc_auc_score(y_val, log_odds_val):.4f}")

    print("\n[3d] Training Bayesian Hierarchical Stacker (Level-2) …")
    stacker = train_ml_stacker(oof_probs, oof_df, oof_labels, oof_segs)

    val_segs = _derive_segment_id(val)
    stk_val  = stacker.predict(raw_val, val, val_segs)

    print("\n  Validation 2025 — full comparison:")
    for lbl, probs in [
        ("XGBoost L1 (raw)",        raw_val),
        ("XGBoost L1 (OOF-Platt)",  cal_val),
        ("Bayesian Stacker L2",     stk_val),
    ]:
        auc = roc_auc_score(y_val, probs)
        ll  = log_loss(y_val, probs)
        bs  = brier_score_loss(y_val, probs)
        tag = "  <<<" if lbl.startswith("Bayesian") else ""
        print(f"  {lbl:<36}  AUC={auc:.4f}  LL={ll:.4f}  Brier={bs:.4f}{tag}")

    val_df = val[["game_pk", "game_date", "home_team", "away_team", "home_win"]].copy()
    val_df["xgb_raw_ml"]     = raw_val
    val_df["xgb_cal_ml"]     = cal_val
    val_df["team_ml_log_odds"] = log_odds_val
    val_df["stacker_ml"]     = stk_val
    val_df.to_csv(OUTPUT_VAL_PREDS, index=False)
    print(f"\n  Saved validation predictions → {OUTPUT_VAL_PREDS}")

    return model, cal, stacker, feat_cols, val_df, log_odds_val


# ---------------------------------------------------------------------------
# TRAIN FINAL (full data)
# ---------------------------------------------------------------------------

def train_final(df: pd.DataFrame, feat_cols: list[str], with_2026: bool = False):
    years = [2023, 2024, 2025]
    if with_2026:
        years.append(2026)
    final_df = df[df["year"].isin(years)].reset_index(drop=True)
    print(f"\n[4] Final model: training on {years} ({len(final_df)} games) …")

    X  = final_df[feat_cols].fillna(0).values.astype(np.float32)
    y  = final_df["home_win"].values.astype(int)
    sw = _sample_weights(final_df)

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y, sample_weight=sw, verbose=False)
    print(f"  XGBoost trained on {len(final_df)} games  ({len(feat_cols)} features)")

    # LOYO OOF for Platt
    print("  [4a] LOYO OOF for calibrator …")
    n       = len(final_df)
    oof_raw = np.zeros(n, dtype=float)
    oof_lbl = y.copy()
    for held_yr in years:
        tr_mask = (final_df["year"] != held_yr).values
        va_mask = (final_df["year"] == held_yr).values
        tr_sub  = final_df[tr_mask]
        va_sub  = final_df[va_mask]
        if len(va_sub) == 0:
            continue
        X_tr  = tr_sub[feat_cols].fillna(0).values.astype(np.float32)
        y_tr  = tr_sub["home_win"].values.astype(int)
        sw_tr = _sample_weights(tr_sub)
        X_va  = va_sub[feat_cols].fillna(0).values.astype(np.float32)
        y_va  = va_sub["home_win"].values.astype(int)
        m_oof = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=40)
        m_oof.fit(X_tr, y_tr, sample_weight=sw_tr,
                  eval_set=[(X_va, y_va)], verbose=False)
        raw_va = m_oof.predict_proba(X_va)[:, 1]
        oof_raw[va_mask] = raw_va
        print(f"    Held-out {held_yr}: n={len(va_sub):>4}  AUC={roc_auc_score(y_va, raw_va):.4f}")

    cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    cal.fit(oof_raw.reshape(-1, 1), oof_lbl)
    print(f"  Platt OOF-fitted — overall OOF AUC={roc_auc_score(oof_lbl, oof_raw):.4f}")

    # Team model (doubled) — persisted for score_ml_today.py
    print("  [4b] Training team model on all data (doubled dataset) …")
    doubled, team_feat_cols = _build_team_dataset(final_df, feat_cols)
    X_tm  = doubled[team_feat_cols].fillna(0).values.astype(np.float32)
    y_tm  = doubled["home_win"].values.astype(int)
    sw_tm = _sample_weights(doubled)

    last_yr = years[-1]
    es_sub  = final_df[final_df["year"] == last_yr].copy()
    es_sub["is_home"] = 1
    X_tm_es = es_sub[team_feat_cols].fillna(0).values.astype(np.float32)
    y_tm_es = es_sub["home_win"].values.astype(int)

    team_model = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=40)
    team_model.fit(X_tm, y_tm, sample_weight=sw_tm,
                   eval_set=[(X_tm_es, y_tm_es)], verbose=False)

    team_model.save_model(str(OUTPUT_TEAM_MODEL))
    json.dump(team_feat_cols, open(OUTPUT_TEAM_FEAT_COLS, "w"), indent=2)
    print(f"  Team model     → {OUTPUT_TEAM_MODEL}")
    print(f"  Team feat cols → {OUTPUT_TEAM_FEAT_COLS}")

    # Full-year LOYO OOF + stacker refit (Step-2 parity with F5)
    print("  [4c] Generating LOYO OOF across all training years for stacker …")
    oof_probs, oof_labels, oof_df, oof_segs = _generate_oof_for_stacker(
        final_df, feat_cols, years=years,
    )

    print("  [4c] Refitting Bayesian Hierarchical Stacker on full-year OOF …")
    stacker = train_ml_stacker(oof_probs, oof_df, oof_labels, oof_segs)
    print(f"  Stacker artefacts → {OUTPUT_STACKER}")
    print(f"                    → {OUTPUT_STACKER_NPZ}")

    return model, cal, stacker


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost ML classifier + Bayesian stacker")
    parser.add_argument("--with-2026", action="store_true",
                        help="Include 2026 partial data in final training")
    parser.add_argument("--matrix", type=str, default="feature_matrix_enriched_v2.parquet",
                        help="Feature matrix parquet path")
    parser.add_argument("--val-year", type=int, default=2025,
                        help="Holdout year for validation (default: 2025)")
    args = parser.parse_args()

    print("=" * 70)
    print("ML XGBoost Model Training — Two-Level Stack")
    print("=" * 70)

    global FEAT_MATRIX
    FEAT_MATRIX = Path(args.matrix)

    df, feat_cols = build_dataset(include_2026=args.with_2026)

    json.dump(feat_cols, open(OUTPUT_FEAT_COLS, "w"), indent=2)
    print(f"\n  Saved feature list → {OUTPUT_FEAT_COLS} ({len(feat_cols)} features)")

    # Default split
    model_val, cal_val, stacker, _, val_df, _log_odds_val = train_default(df, feat_cols, val_year=args.val_year)

    # Final model on all data
    model_final, cal_final, stacker_final = train_final(df, feat_cols, with_2026=args.with_2026)
    model_final.save_model(str(OUTPUT_MODEL))
    pickle.dump(cal_final, open(OUTPUT_CALIB, "wb"))
    print(f"\n  Saved final model → {OUTPUT_MODEL}")
    print(f"  Saved calibrator  → {OUTPUT_CALIB}")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
