"""
train_f5_model.py
=================
Train an XGBoost F5 (first-5-innings) classifier + Bayesian Hierarchical Stacker
to predict P(home team covers F5 +0.5 run line).

Architecture — Full Two-Level Stack (mirrors full-game ML/RL pipeline)
-----------------------------------------------------------------------
  Level 1 — XGBoost:
    Label   : f5_home_cover  — 1 if home >= away after 5 innings (+0.5)
    Features: SP stats, batting matchup, park/ump/schedule
              + MC F5 residual features (Poisson physics baseline)
    Training: 2023+2024 train, 2025 validate

  Level 2 — Bayesian Hierarchical Stacker (NumPyro / NUTS MCMC):
    Model:  y ~ Bernoulli(σ(α + β·logit(p_xgb) + δ_j + γᵀ·x))
    Input:  XGBoost OOF raw probs (generated via NCV folds)
            + 11 domain features (SP diffs, matchup edge, MC physics)
            + SP handedness segment j ∈ {LvL, LvR, RvL, RvR}
    Priors: α~N(0,1)  β~N(1,0.5)  σ_δ~HalfCauchy(1)  δ~N(0,σ_δ)  γ~N(0,0.3)
    Partial pooling across 4 handedness segments (LvL is rare → shrunk to mean)
    Falls back to logistic regression when NumPyro unavailable.

  Residual MC features:
    mc_f5_home_cover_pct  — P(home covers +0.5) from Poisson simulation
    mc_f5_expected_total  — E[F5 total] — scoring environment drives tie rate

Pre-requisite
-------------
  python backfill_mc.py --matrix feature_matrix_with_2026.parquet --force

Outputs
-------
  models/xgb_f5.json              XGBoost F5 classifier
  models/xgb_f5_calibrator.pkl    Platt sigmoid calibrator
  models/stacking_lr_f5.pkl       Bayesian Hierarchical Stacker
  models/stacking_lr_f5.npz       Full NUTS posterior trace
  models/f5_feature_cols.json     Ordered XGBoost feature column list
  f5_val_predictions.csv          Validation set predictions

Usage
-----
  python train_f5_model.py --matrix feature_matrix_with_2026.parquet
  python train_f5_model.py --matrix feature_matrix_with_2026.parquet --ncv
  python train_f5_model.py --matrix feature_matrix_with_2026.parquet --with-2026
"""

import argparse
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
# Same detection logic as train_xgboost.py — GPU if JAX has its own CUDA
# backend; falls back to CPU-JAX or plain LR when unavailable.
try:
    import jax as _jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    _NUMPYRO = True
    # Probe JAX GPU (separate from CuPy)
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

FEAT_MATRIX   = BASE_DIR / "feature_matrix.parquet"
ACTUALS_2026  = DATA_DIR / "actuals_2026.parquet"

OUTPUT_MODEL      = MODELS_DIR / "xgb_f5.json"
OUTPUT_CALIB      = MODELS_DIR / "xgb_f5_calibrator.pkl"
OUTPUT_FEAT_COLS  = MODELS_DIR / "f5_feature_cols.json"
OUTPUT_VAL_PREDS  = BASE_DIR  / "f5_val_predictions.csv"

# ---------------------------------------------------------------------------
# FEATURE CONFIG
# ---------------------------------------------------------------------------
# Columns that are NOT features (labels, identifiers, market lines)
NON_FEATURE_COLS = {
    "game_pk", "game_date", "home_team", "away_team",
    "home_starter_name", "away_starter_name", "season", "year", "split",
    # Labels (full-game)
    "actual_home_win", "actual_game_total", "actual_f5_total",
    "actual_f3_total", "actual_f1_total",
    "home_score", "away_score", "home_margin",
    "home_covers_rl", "away_covers_rl", "total_runs",
    # Vegas / market — MC handles the market signal
    "close_ml_home", "close_ml_away", "open_total", "close_total",
    "true_home_prob", "true_away_prob",
    "vegas_implied_home", "vegas_implied_away",
    # Full-game MC output (not a F5 feature)
    "mc_expected_runs",
    # Pipeline metadata
    "source", "pull_timestamp",
}

# Bullpen features excluded — relief accounts for only 5–8% of F5 runs;
# in 67% of games both SPs go full 5 innings and bullpen = 0%
BULLPEN_FEATURES = {
    "home_bp_era", "home_bp_k9", "home_bp_bb9", "home_bp_hr9",
    "home_bp_whip", "home_bp_gb_pct",
    "away_bp_era", "away_bp_k9", "away_bp_bb9", "away_bp_hr9",
    "away_bp_whip", "away_bp_gb_pct",
    "bp_era_diff", "bp_k9_diff", "bp_whip_diff",
    "home_bp_cluster", "away_bp_cluster",
}

# MC F5 simulation columns used as residual-learning features
# These must exist in the feature matrix (run backfill_mc.py first)
MC_F5_RESIDUAL_COLS = [
    "mc_f5_home_win_pct",
    "mc_f5_away_win_pct",
    "mc_f5_tie_pct",
    "mc_f5_expected_total",
    "mc_f5_home_cover_pct",   # = home_win + tie  ← the +0.5 physics baseline
]

# XGBoost hyperparameters (tuned for ~5,000-row dataset)
XGB_PARAMS = {
    "tree_method":      "hist",
    "n_jobs":           -1,
    "random_state":     42,
    "learning_rate":    0.04,
    "max_depth":        4,          # shallower than main model (less data)
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

# Year sample weights
YEAR_DECAY = {2023: 0.70, 2024: 1.00, 2025: 1.50, 2026: 2.00}

# ---------------------------------------------------------------------------
# BAYESIAN STACKER CONFIG
# ---------------------------------------------------------------------------
# SP handedness segments: home_throws_R * 2 + away_throws_R
#   0=LvL  1=LvR  2=RvL  3=RvR
_N_SEGMENTS   = 4
_SEG_LABELS   = ["LvL", "LvR", "RvL", "RvR"]

# Domain features fed into the stacker (on top of XGBoost raw prob).
# Mirrors full-game STACKING_FEATURES but F5-adapted:
#   - Removed: bp_era_diff, bp_whip_diff (bullpen not relevant for F5)
#   - Replaced: ml_model_vs_vegas_gap → mc_f5_home_cover_pct (F5 physics baseline)
#   - Added:    mc_f5_expected_total   (scoring environment drives tie rate)
F5_STACKING_FEATURES = [
    "sp_k_pct_diff",
    "sp_xwoba_diff",
    "sp_kminusbb_diff",
    "batting_matchup_edge",
    "batting_matchup_edge_10d",
    "home_sp_il_return_flag",
    "away_sp_il_return_flag",
    "sp_k_pct_10d_diff",
    "sp_xwoba_10d_diff",
    "mc_f5_home_cover_pct",    # Poisson physics P(+0.5 cover) — replaces vegas gap
    "mc_f5_expected_total",    # low-total games → high tie rate → cover probability up
    "team_f5_log_odds",        # logit(p_home) - logit(p_away): tie-safe relative strength
    "rolling_f5_tie_rate",     # observed 30-day F5 tie rate: environmental base-rate signal
]

# Fixed probability bins for calibration reliability diagrams.
# Using pd.cut (equal-width) rather than pd.qcut (equal-frequency) so that
# each band represents a true probability range, not a model-distribution artifact.
CALIB_BINS   = [0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.0]
CALIB_LABELS = ["<45", "45-50", "50-55", "55-60", "60-65", "65-70", "70-75", "75-80", ">80"]

OUTPUT_STACKER     = MODELS_DIR / "stacking_lr_f5.pkl"
OUTPUT_STACKER_NPZ = MODELS_DIR / "stacking_lr_f5.npz"
OUTPUT_TEAM_MODEL     = MODELS_DIR / "team_f5_model.json"
OUTPUT_TEAM_FEAT_COLS = MODELS_DIR / "team_f5_feat_cols.json"


# ---------------------------------------------------------------------------
# BAYESIAN STACKER INFRASTRUCTURE
# (Ported from train_xgboost.py — same model, F5-adapted domain features)
# ---------------------------------------------------------------------------

def _derive_segment_id(df: pd.DataFrame) -> np.ndarray:
    """SP handedness matchup segment: home_throws_R*2 + away_throws_R → 0–3."""
    if "home_sp_p_throws_R" not in df.columns or "away_sp_p_throws_R" not in df.columns:
        return np.full(len(df), 3, dtype=np.int32)   # default RvR
    h = df["home_sp_p_throws_R"].fillna(1).astype(int).values
    a = df["away_sp_p_throws_R"].fillna(1).astype(int).values
    return (h * 2 + a).astype(np.int32)


def _numpyro_stacker_model(p_global, segment_id, X_domain,
                            n_segments, n_domain, y_obs=None):
    """NumPyro generative model: y ~ Bernoulli(σ(α + β·logit(p) + δ_j + γᵀ·x))."""
    alpha = numpyro.sample("alpha", dist.Normal(0.0, 1.0))
    beta  = numpyro.sample("beta",  dist.Normal(1.0, 0.5))   # trust XGB by default

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


class BayesianStackerF5:
    """
    Level-2 Bayesian Hierarchical Stacker for F5 +0.5 predictions.
    Stores posterior means; inference is closed-form (no MCMC at test time).

    Interface mirrors the full-game BayesianStacker in train_xgboost.py.
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


def train_f5_stacker(
    oof_probs:   np.ndarray,
    oof_feat_df: pd.DataFrame,
    oof_labels:  np.ndarray,
    oof_segs:    np.ndarray,
    num_warmup:  int = 500,
    num_samples: int = 1_000,
    num_chains:  int = 2,
) -> BayesianStackerF5:
    """
    Fit the Level-2 Bayesian Hierarchical Stacker on OOF predictions.
    Saves model to OUTPUT_STACKER and trace to OUTPUT_STACKER_NPZ.
    """
    feat_names = [c for c in F5_STACKING_FEATURES if c in oof_feat_df.columns]
    fill_vals  = {c: float(oof_feat_df[c].median()) for c in feat_names}

    X_feat = oof_feat_df[feat_names].copy()
    for col, val in fill_vals.items():
        X_feat[col] = X_feat[col].fillna(val)
    X_domain = X_feat.values.astype(float)
    n_domain = X_domain.shape[1]

    # ── Fallback: plain LR when NumPyro unavailable ───────────────────────
    if not _NUMPYRO:
        print("  [WARN] NumPyro not available — using LR fallback stacker")
        X_stack = np.hstack([oof_probs.reshape(-1, 1), X_domain])
        lr = LogisticRegression(C=10, solver="lbfgs", max_iter=1000)
        lr.fit(X_stack, oof_labels)
        model = BayesianStackerF5(
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

    # ── Full NUTS run ─────────────────────────────────────────────────────
    seg_counts = {j: int((oof_segs == j).sum()) for j in range(_N_SEGMENTS)}
    print(f"\n{'='*60}")
    print(f"  F5 BAYESIAN HIERARCHICAL STACKER  ({_JAX_PLATFORM.upper()})")
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
    print(f"    alpha (intercept)         {alpha_hat:+.4f} +/- {float(np.std(samples['alpha'])):.4f}")
    print(f"    beta  (XGB trust)         {beta_hat:+.4f} +/- {float(np.std(samples['beta'])):.4f}")
    print(f"    sigma_delta               {float(np.mean(samples['sigma_delta'])):+.4f} +/- "
          f"{float(np.std(samples['sigma_delta'])):.4f}")
    for j in range(_N_SEGMENTS):
        print(f"    delta[{_SEG_LABELS[j]}]  n={seg_counts[j]:>4}    "
              f"{delta_hat[j]:+.4f} +/- {float(np.std(samples['delta'][:, j])):.4f}")
    for k, feat in enumerate(feat_names):
        print(f"    gamma  {feat:<32} {gamma_hat[k]:+.4f} +/- "
              f"{float(np.std(samples['gamma'][:, k])):.4f}")

    # Save posterior trace
    np.savez(str(OUTPUT_STACKER_NPZ), **{k: np.array(v) for k, v in samples.items()})
    print(f"\n  Posterior trace -> {OUTPUT_STACKER_NPZ}")

    model = BayesianStackerF5(
        alpha=alpha_hat, beta=beta_hat, delta=delta_hat, gamma=gamma_hat,
        stacking_feature_names=feat_names, fill_values=fill_vals,
        n_segments=_N_SEGMENTS, posterior_path=str(OUTPUT_STACKER_NPZ),
    )
    OUTPUT_STACKER.write_bytes(pickle.dumps(model))
    print(f"  BayesianStackerF5 -> {OUTPUT_STACKER}")

    # OOF performance comparison
    stk_probs = model.predict(oof_probs, oof_feat_df, oof_segs)
    auc_xgb   = roc_auc_score(oof_labels, oof_probs)
    auc_stk   = roc_auc_score(oof_labels, stk_probs)
    ll_xgb    = log_loss(oof_labels, oof_probs)
    ll_stk    = log_loss(oof_labels, stk_probs)
    print(f"\n  OOF performance:")
    print(f"    {'Model':<28}  {'AUC':>7}  {'LogLoss':>9}")
    print(f"    {'-'*48}")
    print(f"    {'XGBoost L1 (raw)':<28}  {auc_xgb:>7.4f}  {ll_xgb:>9.4f}")
    print(f"    {'Bayesian Stacker L2':<28}  {auc_stk:>7.4f}  {ll_stk:>9.4f}  <<<")
    print(f"    Net AUC delta: {auc_stk - auc_xgb:+.4f}")

    return model

# ---------------------------------------------------------------------------
# STEP 1: Extract F5 outcomes from statcast
# ---------------------------------------------------------------------------

def extract_f5_from_statcast(year: int) -> pd.DataFrame:
    """Return DataFrame with [game_pk, f5_home_runs, f5_away_runs, f5_home_win]."""
    path = DATA_DIR / f"statcast_{year}.parquet"
    if not path.exists():
        print(f"  [WARN] {path} not found — skipping year {year}")
        return pd.DataFrame()

    cols = ["game_pk", "inning", "inning_topbot", "post_home_score", "post_away_score"]
    sc = pd.read_parquet(path, columns=cols)

    # F5 score = max cumulative post_*_score across all pitches in innings 1-5.
    # Using max() is correct because post_home/away_score are monotonically
    # increasing; max() handles incomplete partial innings, rain-shortened games,
    # and any out-of-order rows in the parquet file.
    f5_grp = sc[sc["inning"] <= 5].groupby("game_pk", as_index=False).agg(
        f5_home_runs=("post_home_score", "max"),
        f5_away_runs=("post_away_score", "max"),
    )

    out = pd.DataFrame({
        "game_pk":       f5_grp["game_pk"],
        "f5_home_runs":  f5_grp["f5_home_runs"].astype(float),
        "f5_away_runs":  f5_grp["f5_away_runs"].astype(float),
    })
    out["f5_home_win"] = (out["f5_home_runs"] > out["f5_away_runs"]).astype(int)
    out["f5_tie"]      = (out["f5_home_runs"] == out["f5_away_runs"]).astype(int)
    out["f5_total"]    = out["f5_home_runs"] + out["f5_away_runs"]
    out["year"]        = year
    print(f"  Year {year}: {len(out)} games | home win {out['f5_home_win'].mean():.3f} "
          f"| tie {out['f5_tie'].mean():.3f} | avg total {out['f5_total'].mean():.2f}")
    return out


def load_f5_labels() -> pd.DataFrame:
    """Combine F5 outcomes for 2023–2025 from statcast + 2026 from actuals file."""
    frames = []
    for yr in [2023, 2024, 2025]:
        df = extract_f5_from_statcast(yr)
        if len(df):
            frames.append(df)

    # 2026 actuals (already extracted, richer data)
    if ACTUALS_2026.exists():
        act = pd.read_parquet(ACTUALS_2026,
                              columns=["game_pk", "f5_home_runs", "f5_away_runs",
                                       "f5_total", "f5_home_win"])
        act["f5_tie"] = (act["f5_home_runs"] == act["f5_away_runs"]).astype(int)
        act["year"]   = 2026
        print(f"  Year 2026: {len(act)} games from actuals file "
              f"| home win {act['f5_home_win'].mean():.3f} "
              f"| tie {act['f5_tie'].mean():.3f}")
        frames.append(act)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# STEP 2: Build training dataset
# ---------------------------------------------------------------------------

def build_dataset(include_2026: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """
    Join F5 labels to feature matrix, return (df, feature_cols).

    Label: f5_home_cover = 1 if home_runs >= away_runs after 5 innings
           (home wins OR tie — matches the +0.5 run-line bet)

    Residual features: MC F5 simulation outputs (mc_f5_home_cover_pct, etc.)
    are kept as features so XGBoost learns what the physics engine misses.
    """
    print("\n[1] Loading feature matrix …")
    fm = pd.read_parquet(FEAT_MATRIX)
    print(f"    Feature matrix: {fm.shape[0]} rows × {fm.shape[1]} cols")

    # Check whether MC F5 residual columns are present
    missing_mc = [c for c in MC_F5_RESIDUAL_COLS if c not in fm.columns]
    if missing_mc:
        print(f"    [WARN] MC F5 residual columns missing: {missing_mc}")
        print(f"    Run: python backfill_mc.py --matrix {FEAT_MATRIX} --force")
    else:
        mc_nan_pct = fm[MC_F5_RESIDUAL_COLS].isna().mean().mean()
        print(f"    MC F5 residual cols present | NaN rate: {mc_nan_pct:.1%}")

    print("\n[2] Extracting F5 labels from statcast …")
    f5 = load_f5_labels()

    # Join on game_pk
    merged = fm.merge(
        f5[["game_pk", "f5_home_runs", "f5_away_runs", "f5_home_win", "f5_tie", "f5_total"]],
        on="game_pk",
        how="inner",
    )
    print(f"\n    After join: {len(merged)} rows "
          f"(dropped {len(fm) - len(merged)} without statcast match)")

    if not include_2026:
        merged = merged[merged["year"].isin([2023, 2024, 2025])]
        print(f"    After dropping 2026: {len(merged)} rows")

    # Build the +0.5 label: home wins OR tie
    merged["f5_home_cover"] = (
        (merged["f5_home_runs"] >= merged["f5_away_runs"]).astype(int)
    )

    # ── Opponent-quality-adjusted F5 offensive performance ────────────────────
    print("    Computing opponent-quality-adjusted F5 offense features …")
    merged = _compute_rolling_adj_f5_form(merged)

    cover_rate = merged["f5_home_cover"].mean()
    tie_rate   = merged["f5_tie"].mean()
    win_rate   = merged["f5_home_win"].mean()
    print(f"    Label breakdown: home cover={cover_rate:.3f} "
          f"(wins={win_rate:.3f} + ties={tie_rate:.3f})")

    # ── Rolling 30-day F5 tie rate (stacker environmental feature) ───────────
    # Uses only games BEFORE each game's date (closed='left') so there is
    # no look-ahead leakage. Falls back to the global tie mean for early games
    # where fewer than 10 prior games exist in the window.
    merged = merged.sort_values("game_date").reset_index(drop=True)
    merged["game_date"] = pd.to_datetime(merged["game_date"])
    _global_tie_mean = float(merged["f5_tie"].mean())
    _tie_rolling = (
        merged.set_index("game_date")["f5_tie"]
        .rolling("30D", min_periods=10, closed="left")
        .mean()
        .reset_index(drop=True)
    )
    merged["rolling_f5_tie_rate"] = _tie_rolling.fillna(_global_tie_mean).values
    print(f"    Rolling 30-day tie rate: "
          f"mean={merged['rolling_f5_tie_rate'].mean():.4f}  "
          f"min={merged['rolling_f5_tie_rate'].min():.4f}  "
          f"max={merged['rolling_f5_tie_rate'].max():.4f}")

    # Determine XGBoost feature columns.
    # rolling_f5_tie_rate is a STACKER-ONLY feature — excluded from XGBoost
    # because it would require computing the rolling rate at inference time
    # and is more useful as an environmental signal for the Bayesian layer.
    exclude = NON_FEATURE_COLS | BULLPEN_FEATURES | {
        "f5_home_runs", "f5_away_runs", "f5_home_win", "f5_tie",
        "f5_total", "f5_home_cover",
        "rolling_f5_tie_rate",   # stacker feature only
    }
    feat_cols = [c for c in merged.columns if c not in exclude]

    # Log which residual cols made it into features
    present_mc = [c for c in MC_F5_RESIDUAL_COLS if c in feat_cols]
    print(f"    MC residual features included: {present_mc}")
    print(f"    Total features: {len(feat_cols)} columns")

    return merged, feat_cols


# ---------------------------------------------------------------------------
# STEP 3: Train helpers
# ---------------------------------------------------------------------------

def _sample_weights(df: pd.DataFrame) -> np.ndarray:
    return df["year"].map(YEAR_DECAY).fillna(1.0).values


def _train_xgb(X_tr, y_tr, sw_tr, X_val, y_val) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=40)
    model.fit(
        X_tr, y_tr,
        sample_weight=sw_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def _platt_calibrate(raw_probs_tr: np.ndarray, y_tr: np.ndarray,
                     raw_probs_val: np.ndarray) -> tuple:
    """Fit Platt scaling on training OOF probs, return (calibrator, cal_val_probs)."""
    cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    cal.fit(raw_probs_tr.reshape(-1, 1), y_tr)
    cal_val = cal.predict_proba(raw_probs_val.reshape(-1, 1))[:, 1]
    return cal, cal_val


def _print_metrics(label: str, y_true, y_prob):
    auc  = roc_auc_score(y_true, y_prob)
    ll   = log_loss(y_true, y_prob)
    bs   = brier_score_loss(y_true, y_prob)
    base = log_loss(y_true, np.full(len(y_true), y_true.mean()))
    print(f"  {label:30s}  AUC={auc:.4f}  LogLoss={ll:.4f}  "
          f"Brier={bs:.4f}  BaseLL={base:.4f}")


# ---------------------------------------------------------------------------
# TEAM-PERSPECTIVE MODEL (doubled-dataset symmetrization)
# ---------------------------------------------------------------------------
# Each game → 2 rows: home-perspective (is_home=1) + away-perspective (is_home=0)
# All home_X ↔ away_X features are swapped for the away row; diff features are
# negated; batting_matchup_edge is negated.
#
# Relative-strength feature: logit(p_home) - logit(p_away)
#   Replaces the old p_home/(p_home+p_away) normalization which incorrectly
#   suppressed the tie state: on ties both p_home and p_away are high, so
#   p_home+p_away > 1 and the division pushed everything toward 0.5.
#   The log-odds difference is symmetric around 0, unbounded, and tie-safe.
# ---------------------------------------------------------------------------

def _flip_team_perspective(
    df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str = "f5_home_cover",
) -> pd.DataFrame:
    """
    Build the away-perspective mirror of a game DataFrame.
    - Swaps all home_X ↔ away_X column pairs that both exist in feat_cols
    - Negates _diff columns  (home - away → away - home = negative)
    - Negates batting_matchup_edge_* columns
    - Recomputes mc_f5_home_cover_pct from the swapped win-pct columns
    - Sets label = f5_away_cover = (away_runs >= home_runs)
    """
    df_flip = df.copy()

    feat_set = set(feat_cols)

    # ── Swap home_X ↔ away_X column pairs ───────────────────────────────
    swapped: set[str] = set()
    for hc in list(feat_set):
        if not hc.startswith("home_"):
            continue
        ac = "away_" + hc[5:]
        if ac in feat_set and hc not in swapped and ac not in swapped:
            tmp            = df_flip[hc].copy()
            df_flip[hc]    = df_flip[ac]
            df_flip[ac]    = tmp
            swapped.add(hc)
            swapped.add(ac)

    # ── Swap mc_f5_home_* ↔ mc_f5_away_* (win/tie probabilities) ────────
    mc_home = [c for c in feat_set if c.startswith("mc_f5_home_") and not c.endswith("_cover_pct")]
    mc_away = [c for c in feat_set if c.startswith("mc_f5_away_")]
    mc_map = {}
    for hc in mc_home:
        ac = "mc_f5_away_" + hc[len("mc_f5_home_"):]
        if ac in feat_set:
            mc_map[hc] = ac
    for hc, ac in mc_map.items():
        tmp            = df_flip[hc].copy()
        df_flip[hc]    = df_flip[ac]
        df_flip[ac]    = tmp

    # Recompute cover_pct from the now-swapped win_pct + tie_pct
    # (from away perspective: "home cover" = away team covers = away_win + tie)
    if ("mc_f5_home_win_pct" in df_flip.columns and
            "mc_f5_tie_pct" in df_flip.columns and
            "mc_f5_home_cover_pct" in df_flip.columns):
        df_flip["mc_f5_home_cover_pct"] = (
            df_flip["mc_f5_home_win_pct"] + df_flip["mc_f5_tie_pct"]
        )

    # ── Negate diff and matchup-edge columns ─────────────────────────────
    for c in feat_cols:
        if c.endswith("_diff") or "matchup_edge" in c:
            df_flip[c] = -df_flip[c]

    # ── Flip label: away_cover = (away_runs >= home_runs) ────────────────
    # On ties BOTH home and away cover (+0.5), so both labels can be 1.
    if "f5_away_runs" in df_flip.columns and "f5_home_runs" in df_flip.columns:
        df_flip[label_col] = (
            (df_flip["f5_away_runs"] >= df_flip["f5_home_runs"]).astype(int)
        )
    elif label_col in df_flip.columns:
        # fallback: 1 - f5_home_win (not the +0.5 label but close)
        df_flip[label_col] = 1 - df_flip[label_col]

    return df_flip


def _build_team_dataset(
    df: pd.DataFrame,
    feat_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build doubled dataset: each game → home row (is_home=1) + away row (is_home=0).
    Returns (doubled_df, team_feat_cols) where team_feat_cols = feat_cols + ['is_home'].
    """
    home_rows = df.copy()
    home_rows["is_home"] = 1

    away_rows = _flip_team_perspective(df, feat_cols)
    away_rows["is_home"] = 0

    doubled = pd.concat([home_rows, away_rows], ignore_index=True)
    team_feat_cols = feat_cols + ["is_home"]
    return doubled, team_feat_cols


def _compute_log_odds_ratio(p_home: np.ndarray, p_away: np.ndarray) -> np.ndarray:
    """
    Log odds ratio: logit(p_home) - logit(p_away).

    Mathematically correct relative-strength feature for +0.5 lines:
      - Ties make both p_home and p_away high simultaneously
        (p_home+p_away > 1), so p_home/(p_home+p_away) incorrectly
        suppresses the tie state and compresses predictions toward 0.5.
      - The log-odds difference is symmetric around 0, unbounded, and
        unaffected by the tie inflation in either probability.
      - Positive values → home has stronger consensus; 0 → even; negative → away.
    """
    eps = 1e-6
    lo_h = np.log(np.clip(p_home, eps, 1-eps)) - np.log(np.clip(1-p_home, eps, 1-eps))
    lo_a = np.log(np.clip(p_away, eps, 1-eps)) - np.log(np.clip(1-p_away, eps, 1-eps))
    return lo_h - lo_a


def train_team_model(df: pd.DataFrame, feat_cols: list[str]):
    """
    Train the doubled-dataset team perspective model and compare to single-perspective.

    Architecture
    ------------
    1. Build doubled dataset (each game = 2 rows)
    2. Train 2023+2024 → validate 2025 (same split as default model)
    3. At inference: run model on home features AND flipped-away features,
       normalize: p_norm = p_h / (p_h + p_a)
    4. Print side-by-side AUC comparison

    Returns
    -------
    team_model    : fitted XGBClassifier (on doubled 2023+2024 data)
    team_feat_cols: feat_cols + ['is_home']
    """
    print("\n[TEAM] Building doubled-dataset team perspective model …")

    # ── Training split ────────────────────────────────────────────────────
    train_df_raw = df[df["year"].isin([2023, 2024])].reset_index(drop=True)
    val_df_raw   = df[df["year"] == 2025].reset_index(drop=True)
    print(f"  Single-perspective: train={len(train_df_raw)}  val={len(val_df_raw)}")

    # Build doubled training set
    train_doubled, team_feat_cols = _build_team_dataset(train_df_raw, feat_cols)
    print(f"  Doubled training:   {len(train_doubled)} rows  ({len(team_feat_cols)} features)")

    X_tr  = train_doubled[team_feat_cols].fillna(0).values.astype(np.float32)
    y_tr  = train_doubled["f5_home_cover"].values.astype(int)
    sw_tr = _sample_weights(train_doubled)

    # For early stopping use the 2025 val set (home perspective)
    val_home = val_df_raw.copy()
    val_home["is_home"] = 1
    X_val_es = val_home[team_feat_cols].fillna(0).values.astype(np.float32)
    y_val_es = val_home["f5_home_cover"].values.astype(int)

    team_model = _train_xgb(X_tr, y_tr, sw_tr, X_val_es, y_val_es)

    # ── Inference: home + away perspectives, then normalize ───────────────
    val_home["is_home"] = 1
    X_h = val_home[team_feat_cols].fillna(0).values.astype(np.float32)
    p_h = team_model.predict_proba(X_h)[:, 1]

    val_away = _flip_team_perspective(val_df_raw, feat_cols)
    val_away["is_home"] = 0
    X_a = val_away[team_feat_cols].fillna(0).values.astype(np.float32)
    p_a = team_model.predict_proba(X_a)[:, 1]

    log_odds = _compute_log_odds_ratio(p_h, p_a)
    y_val    = val_df_raw["f5_home_cover"].values.astype(int)

    # ── Reference: single-perspective XGB ────────────────────────────────
    X_tr_single  = train_df_raw[feat_cols].fillna(0).values.astype(np.float32)
    y_tr_single  = train_df_raw["f5_home_cover"].values.astype(int)
    sw_single    = _sample_weights(train_df_raw)
    X_val_single = val_df_raw[feat_cols].fillna(0).values.astype(np.float32)
    single_model = _train_xgb(X_tr_single, y_tr_single, sw_single, X_val_single, y_val)
    p_single     = single_model.predict_proba(X_val_single)[:, 1]

    # ── Comparison table ──────────────────────────────────────────────────
    print(f"\n  Validation 2025 (n={len(y_val)}) — single vs team model (AUC):")
    print(f"  {'Model':<36}  {'AUC':>7}  {'Note'}")
    print(f"  {'-'*58}")
    for lbl, probs in [
        ("Single-perspective XGB",       p_single),
        ("Team model  (home view only)",  p_h),
        ("Team model  (away view only)",  1 - p_a),
    ]:
        auc = roc_auc_score(y_val, probs)
        print(f"  {lbl:<36}  {auc:>7.4f}")

    # Log-odds ratio is not a probability — rank-order AUC still meaningful
    auc_lo = roc_auc_score(y_val, log_odds)
    print(f"  {'Team model  (log-odds ratio)':<36}  {auc_lo:>7.4f}  (stacker feature)")
    print(f"\n  log-odds range: [{log_odds.min():.2f}, {log_odds.max():.2f}]  "
          f"mean={log_odds.mean():.4f}  (0 = even, +ve = home favoured)")

    return team_model, team_feat_cols, log_odds


def _generate_oof_for_stacker(
    df: pd.DataFrame,
    feat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Year-swap cross-validation on 2023+2024 training data.
    Generates zero-leakage OOF XGBoost predictions for stacker training.

    Fold A: train=2023 → predict 2024
    Fold B: train=2024 → predict 2023
    Combined: OOF covering all 2023+2024 rows.

    Returns
    -------
    oof_probs     : np.ndarray, shape (n_train,)  — XGBoost OOF probabilities
    oof_labels    : np.ndarray, shape (n_train,)  — f5_home_cover labels
    train_df      : pd.DataFrame                  — 2023+2024 rows + team_f5_log_odds col
    oof_segs      : np.ndarray, shape (n_train,)  — SP handedness segment IDs
    """
    train_df = df[df["year"].isin([2023, 2024])].reset_index(drop=True)
    n = len(train_df)
    oof_probs      = np.zeros(n, dtype=float)
    oof_team_logodds = np.zeros(n, dtype=float)
    oof_labels     = train_df["f5_home_cover"].values.astype(int)

    # year_to_positions: integer row indices into train_df for each year
    year_pos = {yr: np.where(train_df["year"].values == yr)[0]
                for yr in [2023, 2024]}

    swap_folds = [
        ([2023], 2024, "Fold A: train=2023 → val=2024"),
        ([2024], 2023, "Fold B: train=2024 → val=2023"),
    ]

    print("\n  [OOF] Year-swap CV on 2023+2024 for stacker training:")
    for train_yrs, val_yr, label in swap_folds:
        tr = train_df[train_df["year"].isin(train_yrs)].reset_index(drop=True)
        va = train_df[train_df["year"] == val_yr].reset_index(drop=True)

        X_tr  = tr[feat_cols].fillna(0).values.astype(np.float32)
        y_tr  = tr["f5_home_cover"].values.astype(int)
        sw_tr = _sample_weights(tr)
        X_va  = va[feat_cols].fillna(0).values.astype(np.float32)
        y_va  = va["f5_home_cover"].values.astype(int)

        # ── Single-perspective XGB OOF ────────────────────────────────────
        m   = _train_xgb(X_tr, y_tr, sw_tr, X_va, y_va)
        raw = m.predict_proba(X_va)[:, 1]

        # ── Team model OOF (doubled training, normalized val predictions) ──
        tr_doubled, team_feat_cols = _build_team_dataset(tr, feat_cols)
        X_tm_tr  = tr_doubled[team_feat_cols].fillna(0).values.astype(np.float32)
        y_tm_tr  = tr_doubled["f5_home_cover"].values.astype(int)
        sw_tm    = _sample_weights(tr_doubled)
        # Early-stop monitor: home-perspective val
        va_es = va.copy(); va_es["is_home"] = 1
        X_tm_es = va_es[team_feat_cols].fillna(0).values.astype(np.float32)
        tm_m = _train_xgb(X_tm_tr, y_tm_tr, sw_tm, X_tm_es, y_va)

        va_home = va.copy(); va_home["is_home"] = 1
        p_h = tm_m.predict_proba(
            va_home[team_feat_cols].fillna(0).values.astype(np.float32))[:, 1]

        va_flip = _flip_team_perspective(va, feat_cols); va_flip["is_home"] = 0
        p_a = tm_m.predict_proba(
            va_flip[team_feat_cols].fillna(0).values.astype(np.float32))[:, 1]

        lo_fold = _compute_log_odds_ratio(p_h, p_a)

        # Write both back to aligned positions in the combined OOF arrays
        pos = year_pos[val_yr]
        oof_probs[pos]        = raw
        oof_team_logodds[pos] = lo_fold

        auc_xgb  = roc_auc_score(y_va, raw)
        auc_team = roc_auc_score(y_va, lo_fold)   # rank-order AUC on log-odds
        print(f"    {label}: n={len(va):,}  XGB-AUC={auc_xgb:.4f}  team-logodds-AUC={auc_team:.4f}")

    overall_auc  = roc_auc_score(oof_labels, oof_probs)
    overall_team = roc_auc_score(oof_labels, oof_team_logodds)
    print(f"  [OOF] Combined: n={n:,}  XGB={overall_auc:.4f}  team-logodds={overall_team:.4f}")

    # Attach log-odds column so stacker can pick up "team_f5_log_odds"
    train_df = train_df.copy()
    train_df["team_f5_log_odds"] = oof_team_logodds

    oof_segs = _derive_segment_id(train_df)
    return oof_probs, oof_labels, train_df, oof_segs


# ---------------------------------------------------------------------------
# STEP 4: Main training
# ---------------------------------------------------------------------------

def _fixed_bin_calibration(y_true: np.ndarray, y_prob: np.ndarray, label: str = ""):
    """
    Print a calibration reliability table using fixed probability bands.
    Uses CALIB_BINS/CALIB_LABELS so bin edges are fixed regardless of the
    model's output distribution (no pd.qcut floating-edge artifact).
    """
    df_c = pd.DataFrame({"pred": y_prob, "actual": y_true})
    df_c["band"] = pd.cut(df_c["pred"], bins=CALIB_BINS, labels=CALIB_LABELS)
    hdr = f"\n  Fixed-bin calibration{' — ' + label if label else ''} (bins={CALIB_BINS[1:-1]}):"
    print(hdr)
    print(f"  {'Band':>8}  {'n':>5}  {'Pred':>7}  {'Actual':>7}  {'Diff':>8}")
    print(f"  {'-'*46}")
    for band, grp in df_c.groupby("band", observed=True):
        if len(grp) == 0:
            continue
        diff = grp["actual"].mean() - grp["pred"].mean()
        print(f"  {str(band):>8}  {len(grp):>5}  "
              f"{grp['pred'].mean():>7.3f}  {grp['actual'].mean():>7.3f}  {diff:>+8.3f}")
    empty = [str(b) for b in CALIB_LABELS
             if df_c[df_c["band"] == b].empty]
    if empty:
        print(f"  (empty bins: {', '.join(empty)})")


def _compute_rolling_adj_f5_form(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling 15-game capped opponent-adjusted F5 form for each team.

    Per-game score (from a team's perspective):
      1. Raw_Margin   = f5_runs_scored - f5_runs_allowed  (home POV; negated for away)
      2. Capped       = clip(Raw_Margin, -4, +4)          (blowout cap)
      3. MC_Expected  = mc_f5_home_win_pct - mc_f5_away_win_pct (from home POV)
      4. Adj_Score    = Capped - MC_Expected              (outperformance vs Poisson)

    Rolling: 15-game mean shifted by 1 (current game excluded — no look-ahead).
    Falls back to 0.0 (neutral) when fewer than 5 qualifying games exist.

    Adds columns:
      home_rolling_adj_f5_form  — home team's 15-game adj F5 form
      away_rolling_adj_f5_form  — away team's 15-game adj F5 form
      rolling_adj_f5_form_diff  — home minus away
    """
    required = ["mc_f5_home_win_pct", "mc_f5_away_win_pct",
                "f5_home_runs", "f5_away_runs"]
    missing = [c for c in required if c not in merged.columns]
    if missing:
        print(f"  [WARN] Missing columns for rolling_adj_f5_form: {missing} — skipping")
        for c in ["home_rolling_adj_f5_form", "away_rolling_adj_f5_form",
                  "rolling_adj_f5_form_diff"]:
            merged[c] = 0.0
        return merged

    # Per-game adjusted score — home perspective
    raw_margin = (merged["f5_home_runs"] - merged["f5_away_runs"]).astype(float)
    capped     = raw_margin.clip(-4, 4)
    mc_exp     = (merged["mc_f5_home_win_pct"].fillna(0.0)
                  - merged["mc_f5_away_win_pct"].fillna(0.0))
    adj_home   = capped - mc_exp     # home team outperformance
    adj_away   = -adj_home           # symmetric: away team's score = negation

    # Long format: one row per team per game
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

    # Per-team rolling 15-game mean, shift(1) to exclude current game
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

    # Global fallback ≈ 0 (adj scores are zero-sum across home/away by construction)
    global_mean = float(team_rolling["form"].mean(skipna=True))
    team_rolling["form"] = team_rolling["form"].fillna(global_mean)

    # Deduplicate edge-case doubleheaders with identical timestamps
    team_rolling = team_rolling.drop_duplicates(subset=["game_date", "team"], keep="last")

    # Merge back: home team
    merged = merged.merge(
        team_rolling.rename(columns={
            "form": "home_rolling_adj_f5_form",
            "team": "home_team",
        }),
        on=["game_date", "home_team"],
        how="left",
    )
    # Merge back: away team
    merged = merged.merge(
        team_rolling.rename(columns={
            "form": "away_rolling_adj_f5_form",
            "team": "away_team",
        }),
        on=["game_date", "away_team"],
        how="left",
    )

    merged["home_rolling_adj_f5_form"] = merged["home_rolling_adj_f5_form"].fillna(global_mean)
    merged["away_rolling_adj_f5_form"] = merged["away_rolling_adj_f5_form"].fillna(global_mean)
    merged["rolling_adj_f5_form_diff"] = (
        merged["home_rolling_adj_f5_form"] - merged["away_rolling_adj_f5_form"]
    )

    print(f"    rolling_adj_f5_form: "
          f"home_mean={merged['home_rolling_adj_f5_form'].mean():.4f}  "
          f"away_mean={merged['away_rolling_adj_f5_form'].mean():.4f}  "
          f"diff_std={merged['rolling_adj_f5_form_diff'].std():.4f}  "
          f"global_fallback={global_mean:.4f}")

    return merged


def train_default(df: pd.DataFrame, feat_cols: list[str]):
    """
    Train 2023+2024 → validate 2025.
    Label: f5_home_cover (+0.5) = home wins or ties after 5 innings.

    Calibration fix: Platt layer is now fitted on TRUE OOF predictions
    (year-swap CV on 2023+2024), not on in-sample training scores.
    """
    print("\n[3] Default split: train 2023+2024 / validate 2025  [label: f5_home_cover +0.5]")
    train = df[df["year"].isin([2023, 2024])]
    val   = df[df["year"] == 2025]
    print(f"    Train: {len(train)} | Val: {len(val)}")

    X_tr  = train[feat_cols].fillna(0).values.astype(np.float32)
    y_tr  = train["f5_home_cover"].values.astype(int)
    sw_tr = _sample_weights(train)
    X_val = val[feat_cols].fillna(0).values.astype(np.float32)
    y_val = val["f5_home_cover"].values.astype(int)

    model   = _train_xgb(X_tr, y_tr, sw_tr, X_val, y_val)
    raw_val = model.predict_proba(X_val)[:, 1]

    print("\n  Raw XGBoost:")
    _print_metrics("Validation 2025", y_val, raw_val)

    # ── OOF generation (moved before calibration fitting) ────────────────────
    # The Platt layer must be fitted on TRUE OOF predictions so it sees the
    # same score distribution the model produces on held-out data.
    # This is also needed by the team model per fold, so generate once here.
    print("\n[3b] Generating OOF predictions (year-swap on 2023+2024) …")
    oof_probs, oof_labels, oof_df, oof_segs = _generate_oof_for_stacker(df, feat_cols)

    # ── Platt calibration fitted on OOF probs (not in-sample raw_tr) ─────────
    cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    cal.fit(oof_probs.reshape(-1, 1), oof_labels)
    cal_val = cal.predict_proba(raw_val.reshape(-1, 1))[:, 1]
    print("  Platt calibrated (OOF-fitted):")
    _print_metrics("Validation 2025", y_val, cal_val)

    # Fixed-bin calibration reliability diagram
    _fixed_bin_calibration(y_val, cal_val, "OOF-Platt, 2025 val")

    # Top feature importances
    imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print("\n  Top 20 feature importances (gain):")
    for feat, score in imp.head(20).items():
        print(f"    {feat:<45s} {score:.4f}")

    # ── Team model: compute log-odds ratio for 2025 val set ──────────────────
    # This feeds into the stacker as the "team_f5_log_odds" domain feature.
    print("\n[3c] Team model: building 2023+2024 doubled dataset for val log-odds …")
    train_reset = train.reset_index(drop=True)
    val_reset   = val.reset_index(drop=True)
    train_doubled_tm, team_feat_cols_tm = _build_team_dataset(train_reset, feat_cols)
    X_tm_tr  = train_doubled_tm[team_feat_cols_tm].fillna(0).values.astype(np.float32)
    y_tm_tr  = train_doubled_tm["f5_home_cover"].values.astype(int)
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

    # Inject into val (copy to avoid SettingWithCopyWarning)
    val = val.copy()
    val["team_f5_log_odds"] = log_odds_val
    auc_team_val = roc_auc_score(y_val, log_odds_val)
    print(f"  Team log-odds on 2025: AUC={auc_team_val:.4f}  "
          f"range=[{log_odds_val.min():.2f}, {log_odds_val.max():.2f}]")

    # ── Bayesian Hierarchical Stacker (Level-2) ──────────────────────────────
    print("\n[3d] Training Bayesian Hierarchical Stacker (Level-2) …")
    stacker = train_f5_stacker(oof_probs, oof_df, oof_labels, oof_segs)

    # Evaluate stacker on 2025 validation set
    val_segs = _derive_segment_id(val)
    stk_val  = stacker.predict(raw_val, val, val_segs)

    print("\n  Validation 2025 — full comparison:")
    print(f"  {'Model':<36}  {'AUC':>7}  {'LogLoss':>9}  {'Brier':>8}")
    print(f"  {'-'*68}")
    for lbl, probs in [
        ("XGBoost L1 (raw)",               raw_val),
        ("XGBoost L1 (OOF-Platt)",         cal_val),
        ("Bayesian Stacker L2",            stk_val),
    ]:
        auc = roc_auc_score(y_val, probs)
        ll  = log_loss(y_val, probs)
        bs  = brier_score_loss(y_val, probs)
        tag = "  <<<" if lbl.startswith("Bayesian") else ""
        print(f"  {lbl:<36}  {auc:>7.4f}  {ll:>9.4f}  {bs:>8.4f}{tag}")

    delta_auc = roc_auc_score(y_val, stk_val) - roc_auc_score(y_val, raw_val)
    print(f"\n  Net AUC delta (L2 vs L1): {delta_auc:+.4f}")

    # Save validation predictions
    val_df = val[["game_pk", "game_date", "home_team", "away_team",
                  "f5_home_runs", "f5_away_runs", "f5_home_win",
                  "f5_tie", "f5_home_cover"]].copy()
    val_df["xgb_raw_f5_cover"]     = raw_val
    val_df["xgb_cal_f5_cover"]     = cal_val      # OOF-Platt calibrated
    val_df["team_f5_log_odds"]     = log_odds_val
    val_df["stacker_f5_cover"]     = stk_val
    val_df.to_csv(OUTPUT_VAL_PREDS, index=False)
    print(f"\n  Saved validation predictions → {OUTPUT_VAL_PREDS}")

    return model, cal, stacker, feat_cols, val_df, log_odds_val


def train_final(df: pd.DataFrame, feat_cols: list[str], with_2026: bool = False):
    """
    Train final models on all available data (2023+2024+2025 + optionally 2026).

    Calibration fix: Platt layer is fitted on leave-one-year-out OOF predictions
    (not in-sample training scores) — same principled approach as train_default().

    Team model: doubled-dataset XGBoost trained on all data and saved to disk
    so score_f5_today.py can compute the team_f5_log_odds stacker feature.
    """
    years = [2023, 2024, 2025]
    if with_2026:
        years.append(2026)
    final_df = df[df["year"].isin(years)].reset_index(drop=True)
    print(f"\n[4] Final model: training on {years} ({len(final_df)} games) …")

    X  = final_df[feat_cols].fillna(0).values.astype(np.float32)
    y  = final_df["f5_home_cover"].values.astype(int)
    sw = _sample_weights(final_df)

    # Main XGBoost: train on all data without early stopping (use all 600 trees)
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y, sample_weight=sw, verbose=False)
    print(f"  XGBoost trained on {len(final_df)} games  ({len(feat_cols)} features)")

    # ── [4a] Leave-one-year-out OOF for Platt calibration ─────────────────
    # For each held-out year: train XGB on remaining years → predict held-out.
    # Platt is then fitted on the combined OOF probs — same distribution shift
    # correction as the year-swap OOF in train_default().
    print("  [4a] Leave-one-year-out OOF for calibrator …")
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
        y_tr  = tr_sub["f5_home_cover"].values.astype(int)
        sw_tr = _sample_weights(tr_sub)
        X_va  = va_sub[feat_cols].fillna(0).values.astype(np.float32)
        y_va  = va_sub["f5_home_cover"].values.astype(int)

        # Use held-out year as early-stop monitor (mildly optimistic but practical
        # here — the OOF probs are only used for Platt calibration, not final scores)
        m_oof = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=40)
        m_oof.fit(X_tr, y_tr, sample_weight=sw_tr,
                  eval_set=[(X_va, y_va)], verbose=False)
        raw_va = m_oof.predict_proba(X_va)[:, 1]
        oof_raw[va_mask] = raw_va
        print(f"    Held-out {held_yr}: n={len(va_sub):>4}  "
              f"AUC={roc_auc_score(y_va, raw_va):.4f}")

    # Fit Platt on OOF
    cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    cal.fit(oof_raw.reshape(-1, 1), oof_lbl)
    print(f"  Platt OOF-fitted — overall OOF AUC={roc_auc_score(oof_lbl, oof_raw):.4f}")

    # ── [4b] Team model on all data — save for score_f5_today.py ──────────
    # score_f5_today.py needs to compute team_f5_log_odds per game, which
    # requires running the team model on home + flipped-away features.
    print("  [4b] Training team model on all data (doubled dataset) …")
    doubled, team_feat_cols = _build_team_dataset(final_df, feat_cols)
    X_tm  = doubled[team_feat_cols].fillna(0).values.astype(np.float32)
    y_tm  = doubled["f5_home_cover"].values.astype(int)
    sw_tm = _sample_weights(doubled)

    # Early-stop eval set: home-perspective rows from the last (most recent) year
    last_yr = years[-1]
    es_sub  = final_df[final_df["year"] == last_yr].copy()
    es_sub["is_home"] = 1
    X_tm_es = es_sub[team_feat_cols].fillna(0).values.astype(np.float32)
    y_tm_es = es_sub["f5_home_cover"].values.astype(int)

    team_model = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=40)
    team_model.fit(X_tm, y_tm, sample_weight=sw_tm,
                   eval_set=[(X_tm_es, y_tm_es)], verbose=False)

    team_model.save_model(str(OUTPUT_TEAM_MODEL))
    json.dump(team_feat_cols, open(OUTPUT_TEAM_FEAT_COLS, "w"), indent=2)
    print(f"  Team model     → {OUTPUT_TEAM_MODEL}")
    print(f"  Team feat cols → {OUTPUT_TEAM_FEAT_COLS}")

    return model, cal


# ---------------------------------------------------------------------------
# STEP 5: NCV mode
# ---------------------------------------------------------------------------

def train_ncv(df: pd.DataFrame, feat_cols: list[str]):
    """Nested cross-validation across years."""
    print("\n[3] NCV mode")
    folds = [
        ([2023],       2024, "Fold 1: 2023→2024"),
        ([2023, 2024], 2025, "Fold 2: 2023+2024→2025"),
    ]
    ncv_rows = []
    for train_yrs, val_yr, label in folds:
        tr = df[df["year"].isin(train_yrs)]
        va = df[df["year"] == val_yr]
        if len(va) == 0:
            continue
        X_tr  = tr[feat_cols].fillna(0).values.astype(np.float32)
        y_tr  = tr["f5_home_cover"].values.astype(int)
        sw_tr = _sample_weights(tr)
        X_va  = va[feat_cols].fillna(0).values.astype(np.float32)
        y_va  = va["f5_home_cover"].values.astype(int)
        m     = _train_xgb(X_tr, y_tr, sw_tr, X_va, y_va)
        raw   = m.predict_proba(X_va)[:, 1]
        auc   = roc_auc_score(y_va, raw)
        ll    = log_loss(y_va, raw)
        print(f"  {label}: AUC={auc:.4f}  LogLoss={ll:.4f}")
        ncv_rows.append({"fold": label, "val_year": val_yr, "auc": auc, "logloss": ll})
    return ncv_rows


# ---------------------------------------------------------------------------
# STEP 6: Predict today's games
# ---------------------------------------------------------------------------

def predict_today(model, cal, feat_cols: list[str]):
    """Apply F5 model to today's feature matrix and compare to actuals if available."""
    import glob
    from datetime import date

    today_str = date.today().isoformat()  # "2026-04-19"
    print(f"\n[5] Predicting today's games ({today_str}) …")

    # Look for today's feature matrix file (run_today.py writes it)
    today_fm_paths = sorted(glob.glob(f"data/statcast/odds_current_{today_str.replace('-','_')}.parquet"))
    if not today_fm_paths:
        # Try to build features from feature_matrix if today's games are in it
        fm = pd.read_parquet(FEAT_MATRIX)
        today_rows = fm[fm["game_date"].astype(str).str.startswith(today_str)]
        if len(today_rows) == 0:
            print(f"  No feature matrix rows for {today_str}. "
                  "Run build_feature_matrix.py first.")
            return
        df_today = today_rows
    else:
        df_today = pd.read_parquet(today_fm_paths[-1])

    # Align columns
    missing = [c for c in feat_cols if c not in df_today.columns]
    for c in missing:
        df_today[c] = 0.0

    X_today = df_today[feat_cols].fillna(0).values.astype(np.float32)
    raw     = model.predict_proba(X_today)[:, 1]
    cal_p   = cal.predict_proba(raw.reshape(-1, 1))[:, 1]

    # Build output table
    id_cols = ["game_pk", "home_team", "away_team"]
    available_id = [c for c in id_cols if c in df_today.columns]
    out = df_today[available_id].copy()
    out["xgb_f5_home_win_prob"] = cal_p.round(4)
    out["xgb_f5_away_win_prob"] = (1 - cal_p).round(4)

    # Compare to actuals if yesterday's data is present
    if ACTUALS_2026.exists():
        act = pd.read_parquet(ACTUALS_2026)
        yesterday = act[act["game_date"].astype(str).str.startswith("2026-04-18")]
        if len(yesterday) and "game_pk" in out.columns:
            merged = out.merge(
                yesterday[["game_pk","f5_home_win","f5_home_runs","f5_away_runs"]],
                on="game_pk", how="left"
            )
            print("\n  Today's F5 predictions (with yesterday's actuals where matched):")
            print(f"  {'Away':6s}  {'@':1s}  {'Home':6s}  {'xgb_home%':9s}  "
                  f"{'xgb_away%':9s}  {'Actual':7s}  {'Score':12s}")
            print("  " + "-" * 68)
            for _, row in merged.iterrows():
                actual = ""
                score  = ""
                if pd.notna(row.get("f5_home_win")):
                    hw = int(row["f5_home_win"])
                    hr = int(row["f5_home_runs"])
                    ar = int(row["f5_away_runs"])
                    actual = "HW" if hw == 1 else ("TIE" if hr == ar else "AW")
                    score  = f"{hr}–{ar}"
                home  = row.get("home_team", "?")
                away  = row.get("away_team", "?")
                hp    = row["xgb_f5_home_win_prob"]
                ap    = row["xgb_f5_away_win_prob"]
                print(f"  {away:6s}  @  {home:6s}  {hp:9.1%}  {ap:9.1%}  "
                      f"{actual:7s}  {score:12s}")
            return

    # No actuals match — just print predictions
    print("\n  Today's F5 predictions:")
    print(f"  {'Away':6s}  {'@':1s}  {'Home':6s}  {'xgb_home%':9s}  {'xgb_away%':9s}")
    print("  " + "-" * 44)
    for _, row in out.iterrows():
        home = row.get("home_team", "?")
        away = row.get("away_team", "?")
        hp   = row["xgb_f5_home_win_prob"]
        ap   = row["xgb_f5_away_win_prob"]
        print(f"  {away:6s}  @  {home:6s}  {hp:9.1%}  {ap:9.1%}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost F5 classifier")
    parser.add_argument("--ncv",          action="store_true",
                        help="Run nested cross-validation")
    parser.add_argument("--with-2026",    action="store_true",
                        help="Include 2026 partial data in final training")
    parser.add_argument("--predict-today", action="store_true",
                        help="Score today's games after training")
    parser.add_argument("--compare-yesterday", action="store_true",
                        help="Compare model predictions to yesterday's actuals")
    parser.add_argument("--team-model",   action="store_true",
                        help="Also train doubled-dataset team perspective model and compare")
    parser.add_argument("--matrix", type=str, default="feature_matrix.parquet",
                        help="Feature matrix parquet path (default: feature_matrix.parquet)")
    parser.add_argument("--extra-features", type=str, default=None,
                        help="Comma-separated list of EXTRA features to allow from a v2 matrix. "
                             "When set, features are restricted to the v1 baseline list "
                             "(models/f5_feature_cols_v1.json) plus these extra columns.")
    args = parser.parse_args()

    print("=" * 70)
    print("F5 XGBoost Model Training")
    print("=" * 70)

    # Override feature matrix path if supplied
    global FEAT_MATRIX
    FEAT_MATRIX = Path(args.matrix)

    # Build dataset
    df, feat_cols = build_dataset(include_2026=args.with_2026)

    # If --extra-features is set, restrict features to v1 baseline + whitelisted extras
    if args.extra_features:
        extra_list = [f.strip() for f in args.extra_features.split(",") if f.strip()]
        v1_path = MODELS_DIR / "f5_feature_cols_v1.json"
        if v1_path.exists():
            v1_cols = set(json.load(open(v1_path)))
            print(f"\n  --extra-features mode: v1 baseline={len(v1_cols)} + "
                  f"extra={len(extra_list)} features")
            feat_cols = [c for c in feat_cols if c in v1_cols or c in extra_list]
            # Report which extras are actually in the matrix
            present_extra = [c for c in extra_list if c in feat_cols]
            missing_extra = [c for c in extra_list if c not in feat_cols]
            print(f"  Extra features present: {present_extra}")
            if missing_extra:
                print(f"  [WARN] Extra features NOT found in matrix: {missing_extra}")
        else:
            print(f"  [WARN] --extra-features: v1 baseline file not found at {v1_path}")
            print(f"         Snapshot current feat_cols to {v1_path} first.")
        print(f"  Final feature count: {len(feat_cols)}")

    # Save feature column list
    json.dump(feat_cols, open(OUTPUT_FEAT_COLS, "w"), indent=2)
    print(f"\n  Saved feature list → {OUTPUT_FEAT_COLS} ({len(feat_cols)} features)")

    if args.ncv:
        train_ncv(df, feat_cols)

    # Default validation split (includes stacker training + L1 vs L2 comparison)
    model_val, cal_val, stacker, _, val_df, _log_odds_val = train_default(df, feat_cols)

    # Final model on all data
    model_final, cal_final = train_final(df, feat_cols, with_2026=args.with_2026)
    model_final.save_model(str(OUTPUT_MODEL))
    pickle.dump(cal_final, open(OUTPUT_CALIB, "wb"))
    print(f"\n  Saved final model → {OUTPUT_MODEL}")
    print(f"  Saved calibrator  → {OUTPUT_CALIB}")

    # Team-perspective model (optional but recommended)
    if args.team_model:
        train_team_model(df, feat_cols)

    # Compare to yesterday's actuals (default: always show)
    _compare_yesterday(model_val, cal_val, feat_cols, val_df, stacker=stacker)

    if args.predict_today:
        predict_today(model_val, cal_val, feat_cols)

    print("\n" + "=" * 70)
    print("Done.")


# ---------------------------------------------------------------------------
# BONUS: Compare model to yesterday's 2026 games using the validation model
# ---------------------------------------------------------------------------

def _compare_yesterday(model, cal, feat_cols: list[str], val_df: pd.DataFrame,
                        stacker=None):
    """Score yesterday's 2026 games and compare to actual F5 outcomes."""
    print("\n" + "=" * 70)
    print("Yesterday's 2026 Game Comparison (2026-04-18)")
    print("=" * 70)

    if not ACTUALS_2026.exists():
        print("  actuals_2026.parquet not found — skipping comparison.")
        return

    act = pd.read_parquet(ACTUALS_2026)
    yesterday = act[act["game_date"].astype(str).str.startswith("2026-04-18")].copy()
    if len(yesterday) == 0:
        print("  No 2026-04-18 actuals found.")
        return

    fm = pd.read_parquet(FEAT_MATRIX)
    # Try to find 2026 game rows
    # feature_matrix may not have 2026 rows unless built with --years 2026
    # Use actuals game_pk list to check
    pks = yesterday["game_pk"].tolist()
    rows = fm[fm["game_pk"].isin(pks)]

    if len(rows) == 0:
        print(f"  No feature matrix rows for yesterday's {len(pks)} games.")
        print("  (Run build_feature_matrix.py --years 2023 2024 2025 2026 to include 2026)")
        # Still show actuals
        print("\n  Yesterday's actual F5 results:")
        print(f"  {'Away':6s}  @  {'Home':6s}  {'F5 Score':12s}  {'Winner':6s}")
        print("  " + "-" * 40)
        for _, row in yesterday.iterrows():
            hr = int(row["f5_home_runs"])
            ar = int(row["f5_away_runs"])
            winner = "HOME" if row["f5_home_win"] == 1 else ("TIE" if hr == ar else "AWAY")
            print(f"  {row['away_team']:6s}  @  {row['home_team']:6s}  "
                  f"{ar}–{hr} (F5)        {winner}")
        return

    # We have feature rows — score them
    missing = [c for c in feat_cols if c not in rows.columns]
    for c in missing:
        rows = rows.copy()
        rows[c] = 0.0

    X = rows[feat_cols].fillna(0).values.astype(np.float32)
    raw   = model.predict_proba(X)[:, 1]
    cal_p = cal.predict_proba(raw.reshape(-1, 1))[:, 1]
    rows  = rows.copy()
    rows["xgb_f5_home_win_prob"] = cal_p

    # Apply stacker if available
    if stacker is not None:
        rows_seg  = _derive_segment_id(rows)
        stk_p     = stacker.predict(raw, rows, rows_seg)
        rows["stacker_f5_cover_prob"] = stk_p
    else:
        rows["stacker_f5_cover_prob"] = cal_p   # fallback

    sel_cols = ["game_pk", "home_team", "away_team",
                "xgb_f5_home_win_prob", "stacker_f5_cover_prob"]
    merged = rows[sel_cols].merge(
        yesterday[["game_pk","f5_home_win","f5_home_runs","f5_away_runs","f5_total"]],
        on="game_pk"
    )

    # Metrics + fixed-bin calibration
    if len(merged) >= 5:
        merged["f5_home_cover_act"] = (
            (merged["f5_home_runs"] >= merged["f5_away_runs"]).astype(int)
        )
        auc_l1  = roc_auc_score(merged["f5_home_cover_act"], merged["xgb_f5_home_win_prob"])
        auc_stk = roc_auc_score(merged["f5_home_cover_act"], merged["stacker_f5_cover_prob"])
        bs_l1   = brier_score_loss(merged["f5_home_cover_act"], merged["xgb_f5_home_win_prob"])
        bs_stk  = brier_score_loss(merged["f5_home_cover_act"], merged["stacker_f5_cover_prob"])
        print(f"  XGB L1   AUC={auc_l1:.4f}  Brier={bs_l1:.4f}  (n={len(merged)} games)")
        if stacker is not None:
            print(f"  Stack L2 AUC={auc_stk:.4f}  Brier={bs_stk:.4f}  "
                  f"delta={auc_stk-auc_l1:+.4f}")
        if len(merged) >= 10:
            use_col = "stacker_f5_cover_prob" if stacker is not None else "xgb_f5_home_win_prob"
            _fixed_bin_calibration(
                merged["f5_home_cover_act"].values,
                merged[use_col].values,
                f"2026-04-18 ({len(merged)} games)"
            )

    # Output table — show both L1 and L2 probs
    use_stacker = stacker is not None
    hdr = (f"\n  {'Away':6s}  @  {'Home':6s}  {'XGB%':7s}  "
           + (f"{'Stk%':7s}  " if use_stacker else "")
           + f"{'F5 Score':10s}  {'Result':6s}  {'+0.5':4s}")
    sep_len = 68 + (10 if use_stacker else 0)
    print(hdr)
    print("  " + "-" * sep_len)

    correct_l1 = 0
    correct_stk = 0
    sort_col = "stacker_f5_cover_prob" if use_stacker else "xgb_f5_home_win_prob"
    for _, row in merged.sort_values(sort_col, ascending=False).iterrows():
        hr     = int(row["f5_home_runs"])
        ar     = int(row["f5_away_runs"])
        hw     = int(row["f5_home_win"])
        covers = int(hr >= ar)
        result = "HOME" if hw == 1 else ("TIE" if hr == ar else "AWAY")

        prob_l1  = row["xgb_f5_home_win_prob"]
        prob_stk = row["stacker_f5_cover_prob"]
        pred_l1  = "HOME" if prob_l1  >= 0.50 else "AWAY"
        pred_stk = "HOME" if prob_stk >= 0.50 else "AWAY"
        call_l1  = "+" if (pred_l1  == "HOME") == bool(covers) else "-"
        call_stk = "+" if (pred_stk == "HOME") == bool(covers) else "-"
        correct_l1  += int((pred_l1  == "HOME") == bool(covers))
        correct_stk += int((pred_stk == "HOME") == bool(covers))

        stk_fld = f"{prob_stk:7.1%}  " if use_stacker else ""
        call_fld = call_stk if use_stacker else call_l1
        print(f"  {row['away_team']:6s}  @  {row['home_team']:6s}  {prob_l1:7.1%}  "
              f"{stk_fld}{ar}-{hr} (F5)  {result:6s}  {call_fld}")

    n = len(merged)
    print(f"\n  XGB L1  +0.5 correct: {correct_l1}/{n}")
    if use_stacker:
        print(f"  Stack L2 +0.5 correct: {correct_stk}/{n}")


if __name__ == "__main__":
    main()
