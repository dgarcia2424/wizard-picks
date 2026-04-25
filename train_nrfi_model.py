"""
train_nrfi_model.py
===================
Train an XGBoost NRFI (No Runs First Inning) classifier + Bayesian Hierarchical
Stacker to predict P(NRFI) = P(0 total runs scored in the 1st inning).

Architecture mirrors train_f5_model.py exactly:
  L1: XGBoost binary classifier on feature_matrix.parquet
  Team model: doubled/flipped dataset → team_nrfi_log_odds
  Dual Poisson sidecar on f1_home_runs / f1_away_runs → pois_p_nrfi
  L2: NumPyro Bayesian Hierarchical Stacker (NUTS MCMC)
  LOYO (Leave-One-Year-Out) CV across 2023, 2024, 2025
  Labels: f1_nrfi directly from actuals_2026 parquet; derived from statcast for 2023-2025.

NRFI_STACKING_FEATURES notes (verified against feature_matrix.parquet columns):
  - sp_whiff_diff: NOT PRESENT in matrix (only pctl versions exist) — DROPPED.
  - sp_home_fip / sp_away_fip: NOT PRESENT — replaced with home_sp_xera_pctl / away_sp_xera_pctl.
"""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import xgboost as xgb

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

FEAT_MATRIX      = BASE_DIR / "feature_matrix.parquet"
ACTUALS_2026     = DATA_DIR / "actuals_2026.parquet"
ABS_FEATURES_2026 = DATA_DIR / "abs_features_2026.parquet"

OUTPUT_MODEL      = MODELS_DIR / "xgb_nrfi.json"
OUTPUT_CALIB      = MODELS_DIR / "xgb_nrfi_calibrator.pkl"
OUTPUT_FEAT_COLS  = MODELS_DIR / "nrfi_feature_cols.json"
OUTPUT_VAL_PREDS  = BASE_DIR  / "nrfi_val_predictions.csv"

# ---------------------------------------------------------------------------
# FEATURE CONFIG
# ---------------------------------------------------------------------------
NON_FEATURE_COLS = {
    "game_pk", "game_date", "home_team", "away_team",
    "home_starter_name", "away_starter_name", "season", "year", "split",
    "actual_home_win", "actual_game_total", "actual_f5_total",
    "actual_f3_total", "actual_f1_total",
    "home_score", "away_score", "home_margin",
    "home_covers_rl", "away_covers_rl", "total_runs",
    "close_ml_home", "close_ml_away", "open_total", "close_total",
    "true_home_prob", "true_away_prob",
    "vegas_implied_home", "vegas_implied_away",
    "mc_expected_runs",
    "source", "pull_timestamp",
}

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

# Bullpen features aren't relevant for 1st inning — excluded
BULLPEN_FEATURES = {
    "home_bp_era", "home_bp_k9", "home_bp_bb9", "home_bp_hr9",
    "home_bp_whip", "home_bp_gb_pct",
    "away_bp_era", "away_bp_k9", "away_bp_bb9", "away_bp_hr9",
    "away_bp_whip", "away_bp_gb_pct",
    "bp_era_diff", "bp_k9_diff", "bp_whip_diff",
    "home_bp_cluster", "away_bp_cluster",
}

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

# Domain features fed into the stacker (on top of XGBoost raw prob).
# Substitutions vs. user spec:
#   sp_whiff_diff    → NOT in matrix; DROPPED.
#   sp_home_fip      → NOT in matrix; replaced with home_sp_xera_pctl.
#   sp_away_fip      → NOT in matrix; replaced with away_sp_xera_pctl.
NRFI_STACKING_FEATURES = [
    "sp_k_pct_diff",
    "sp_kminusbb_diff",
    "home_sp_xera_pctl",
    "away_sp_xera_pctl",
    "batting_matchup_edge",
    "team_nrfi_log_odds",
    "rolling_nrfi_base_rate",
    "pois_lam_f1_home",
    "pois_lam_f1_away",
    "pois_p_nrfi",
    # ABS-regime features (2026 only; NaN for 2023-2025 — no backfill)
    "home_sp_fi_whiff_pct",
    "away_sp_fi_whiff_pct",
    "sp_fi_whiff_diff",
    "home_sp_zone_pct",
    "away_sp_zone_pct",
    "sp_zone_pct_diff",
]

POISSON_XGB_PARAMS = {
    "objective":        "count:poisson",
    "eval_metric":      "poisson-nloglik",
    "tree_method":      "hist",
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 5,
    "reg_lambda":       1.0,
    "max_delta_step":   0.7,
}
POISSON_N_ROUNDS_HOME = 85
POISSON_N_ROUNDS_AWAY = 215


def _train_poisson_booster(X_tr: np.ndarray, y_tr: np.ndarray,
                           n_rounds: int) -> "xgb.Booster":
    import xgboost as _xgb_lib
    dtr = _xgb_lib.DMatrix(X_tr, label=y_tr)
    return _xgb_lib.train(POISSON_XGB_PARAMS, dtr, num_boost_round=n_rounds,
                          verbose_eval=False)


def _poisson_predict(booster, X: np.ndarray) -> np.ndarray:
    import xgboost as _xgb_lib
    return booster.predict(_xgb_lib.DMatrix(X))


def _prob_nrfi_from_lambdas(lam_home: np.ndarray,
                            lam_away: np.ndarray) -> np.ndarray:
    """P(NRFI) = P(H=0)*P(A=0) = exp(-(lam_h + lam_a)) under independent Poissons."""
    return np.exp(-(lam_home + lam_away))


CALIB_BINS   = [0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.0]
CALIB_LABELS = ["<45", "45-50", "50-55", "55-60", "60-65", "65-70", "70-75", "75-80", ">80"]

OUTPUT_STACKER        = MODELS_DIR / "stacking_lr_nrfi.pkl"
OUTPUT_STACKER_NPZ    = MODELS_DIR / "stacking_lr_nrfi.npz"
OUTPUT_TEAM_MODEL     = MODELS_DIR / "team_nrfi_model.json"
OUTPUT_TEAM_FEAT_COLS = MODELS_DIR / "team_nrfi_feat_cols.json"
OUTPUT_POIS_HOME      = MODELS_DIR / "xgb_pois_f1_home.json"
OUTPUT_POIS_AWAY      = MODELS_DIR / "xgb_pois_f1_away.json"


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
    """NumPyro generative model: y ~ Bernoulli(σ(α + β·logit(p) + δ_j + γᵀ·x))."""
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


class BayesianStackerNRFI:
    """Level-2 Bayesian Hierarchical Stacker for NRFI predictions."""
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
                fill = val if not pd.isna(val) else 0.0
                X[col] = X[col].fillna(fill)
        X = X.fillna(0.0)  # final safety: catch any remaining NaN
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


def train_nrfi_stacker(
    oof_probs:   np.ndarray,
    oof_feat_df: pd.DataFrame,
    oof_labels:  np.ndarray,
    oof_segs:    np.ndarray,
    num_warmup:  int = 500,
    num_samples: int = 1_000,
    num_chains:  int = 2,
) -> BayesianStackerNRFI:
    feat_names = [c for c in NRFI_STACKING_FEATURES if c in oof_feat_df.columns]
    fill_vals  = {
        c: (m if not pd.isna(m := float(oof_feat_df[c].median())) else 0.0)
        for c in feat_names
    }

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
        model = BayesianStackerNRFI(
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
    print(f"  NRFI BAYESIAN HIERARCHICAL STACKER  ({_JAX_PLATFORM.upper()})")
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

    np.savez(str(OUTPUT_STACKER_NPZ), **{k: np.array(v) for k, v in samples.items()})
    print(f"\n  Posterior trace -> {OUTPUT_STACKER_NPZ}")

    model = BayesianStackerNRFI(
        alpha=alpha_hat, beta=beta_hat, delta=delta_hat, gamma=gamma_hat,
        stacking_feature_names=feat_names, fill_values=fill_vals,
        n_segments=_N_SEGMENTS, posterior_path=str(OUTPUT_STACKER_NPZ),
    )
    OUTPUT_STACKER.write_bytes(pickle.dumps(model))
    print(f"  BayesianStackerNRFI -> {OUTPUT_STACKER}")

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
# STEP 1: Extract NRFI (F1) outcomes from statcast
# ---------------------------------------------------------------------------

def extract_nrfi_from_statcast(year: int) -> pd.DataFrame:
    """Return DataFrame with [game_pk, f1_home_runs, f1_away_runs, f1_nrfi, year]."""
    path = DATA_DIR / f"statcast_{year}.parquet"
    if not path.exists():
        print(f"  [WARN] {path} not found — skipping year {year}")
        return pd.DataFrame()

    cols = ["game_pk", "inning", "inning_topbot", "post_home_score", "post_away_score"]
    sc = pd.read_parquet(path, columns=cols)

    # First inning starts 0-0; post_*_score is monotonically increasing so max()
    # over all inning==1 rows gives the runs scored in the first frame.
    f1_grp = sc[sc["inning"] == 1].groupby("game_pk", as_index=False).agg(
        f1_home_runs=("post_home_score", "max"),
        f1_away_runs=("post_away_score", "max"),
    )

    out = pd.DataFrame({
        "game_pk":       f1_grp["game_pk"],
        "f1_home_runs":  f1_grp["f1_home_runs"].astype(float),
        "f1_away_runs":  f1_grp["f1_away_runs"].astype(float),
    })
    out["f1_total"] = out["f1_home_runs"] + out["f1_away_runs"]
    out["f1_nrfi"]  = (out["f1_total"] == 0).astype(int)
    out["year"]     = year
    print(f"  Year {year}: {len(out)} games | NRFI rate {out['f1_nrfi'].mean():.3f} "
          f"| avg F1 total {out['f1_total'].mean():.2f}")
    return out


def load_nrfi_labels() -> pd.DataFrame:
    """Combine NRFI outcomes for 2023–2025 from statcast + 2026 from actuals file."""
    frames = []
    for yr in [2023, 2024, 2025]:
        df = extract_nrfi_from_statcast(yr)
        if len(df):
            frames.append(df)

    if ACTUALS_2026.exists():
        act_all = pd.read_parquet(ACTUALS_2026)
        keep = ["game_pk", "f1_home_runs", "f1_away_runs", "f1_nrfi"]
        keep = [c for c in keep if c in act_all.columns]
        act = act_all[keep].copy()
        act["f1_total"] = act["f1_home_runs"] + act["f1_away_runs"]
        act["year"]     = 2026
        print(f"  Year 2026: {len(act)} games from actuals file "
              f"| NRFI rate {act['f1_nrfi'].mean():.3f}")
        frames.append(act)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# STEP 2: Build training dataset
# ---------------------------------------------------------------------------

def build_dataset(include_2026: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Join NRFI labels to feature matrix, return (df, feature_cols)."""
    print("\n[1] Loading feature matrix …")
    fm = pd.read_parquet(FEAT_MATRIX)
    print(f"    Feature matrix: {fm.shape[0]} rows × {fm.shape[1]} cols")

    # Merge ABS features (2026 only; NaN for prior years = temporal leakage guard)
    if ABS_FEATURES_2026.exists():
        abs_cols = [
            "game_pk",
            "home_sp_fi_whiff_pct", "away_sp_fi_whiff_pct",
            "home_sp_zone_pct",     "away_sp_zone_pct",
            "sp_zone_pct_diff",
        ]
        abs_df = pd.read_parquet(ABS_FEATURES_2026, columns=abs_cols)
        abs_df["sp_fi_whiff_diff"] = (
            abs_df["home_sp_fi_whiff_pct"] - abs_df["away_sp_fi_whiff_pct"]
        )
        fm = fm.merge(abs_df, on="game_pk", how="left")
        n_matched = fm[fm["home_sp_fi_whiff_pct"].notna()].shape[0]
        print(f"    ABS features merged: {n_matched} rows with 2026 whiff/zone data "
              f"({fm.shape[0] - n_matched} rows NaN — correct for 2023-2025)")
    else:
        print(f"    [WARN] {ABS_FEATURES_2026} not found — ABS features skipped")

    print("\n[2] Extracting NRFI labels …")
    f1 = load_nrfi_labels()

    merged = fm.merge(
        f1[["game_pk", "f1_home_runs", "f1_away_runs", "f1_nrfi", "f1_total"]],
        on="game_pk",
        how="inner",
    )
    print(f"\n    After join: {len(merged)} rows "
          f"(dropped {len(fm) - len(merged)} without label match)")

    if not include_2026:
        merged = merged[merged["year"].isin([2023, 2024, 2025])]
        print(f"    After dropping 2026: {len(merged)} rows")

    nrfi_rate = merged["f1_nrfi"].mean()
    print(f"    Label breakdown: NRFI={nrfi_rate:.3f}  YRFI={1-nrfi_rate:.3f}")

    # Rolling 30-day no-look-ahead NRFI base rate
    merged = merged.sort_values("game_date").reset_index(drop=True)
    merged["game_date"] = pd.to_datetime(merged["game_date"])
    _global_nrfi_mean = float(merged["f1_nrfi"].mean())
    _nrfi_rolling = (
        merged.set_index("game_date")["f1_nrfi"]
        .rolling("30D", min_periods=10, closed="left")
        .mean()
        .reset_index(drop=True)
    )
    merged["rolling_nrfi_base_rate"] = _nrfi_rolling.fillna(_global_nrfi_mean).values
    print(f"    Rolling 30-day NRFI base rate: "
          f"mean={merged['rolling_nrfi_base_rate'].mean():.4f}  "
          f"min={merged['rolling_nrfi_base_rate'].min():.4f}  "
          f"max={merged['rolling_nrfi_base_rate'].max():.4f}")

    exclude = NON_FEATURE_COLS | BULLPEN_FEATURES | {
        "f1_home_runs", "f1_away_runs", "f1_nrfi", "f1_total",
        "rolling_nrfi_base_rate",   # stacker feature only
    }
    feat_cols = [c for c in merged.columns if c not in exclude]

    print(f"    Total features: {len(feat_cols)} columns")
    return merged, feat_cols


# ---------------------------------------------------------------------------
# STEP 3: Train helpers
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
# TEAM-PERSPECTIVE MODEL (doubled-dataset symmetrization)
# ---------------------------------------------------------------------------

def _flip_team_perspective(
    df: pd.DataFrame,
    feat_cols: list[str],
    label_col: str = "f1_nrfi",
) -> pd.DataFrame:
    """Away-perspective mirror: swap home_X/away_X, negate diffs & matchup_edge.
    NRFI label is team-symmetric (0 runs total) → label_col is preserved as-is."""
    df_flip = df.copy()
    feat_set = set(feat_cols)

    swapped: set[str] = set()
    for hc in list(feat_set):
        if not hc.startswith("home_"):
            continue
        ac = "away_" + hc[5:]
        if ac in feat_set and hc not in swapped and ac not in swapped:
            tmp            = df_flip[hc].copy()
            df_flip[hc]    = df_flip[ac]
            df_flip[ac]    = tmp
            swapped.add(hc); swapped.add(ac)

    for c in feat_cols:
        if (c.endswith("_diff") or "matchup_edge" in c) and c in df_flip.columns:
            df_flip[c] = -df_flip[c]

    # NRFI is symmetric → label unchanged on flip.
    return df_flip


def _build_team_dataset(
    df: pd.DataFrame,
    feat_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Doubled dataset: each game → home row (is_home=1) + away row (is_home=0)."""
    home_rows = df.copy()
    home_rows["is_home"] = 1

    away_rows = _flip_team_perspective(df, feat_cols)
    away_rows["is_home"] = 0

    doubled = pd.concat([home_rows, away_rows], ignore_index=True)
    team_feat_cols = feat_cols + ["is_home"]
    return doubled, team_feat_cols


def _compute_log_odds_ratio(p_home: np.ndarray, p_away: np.ndarray) -> np.ndarray:
    """Log odds ratio: logit(p_home) - logit(p_away)."""
    eps = 1e-6
    lo_h = np.log(np.clip(p_home, eps, 1-eps)) - np.log(np.clip(1-p_home, eps, 1-eps))
    lo_a = np.log(np.clip(p_away, eps, 1-eps)) - np.log(np.clip(1-p_away, eps, 1-eps))
    return lo_h - lo_a


# ---------------------------------------------------------------------------
# OOF generation (LOYO)
# ---------------------------------------------------------------------------

def _generate_oof_for_stacker(
    df: pd.DataFrame,
    feat_cols: list[str],
    years: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """LOYO CV → OOF L1 probs + team_nrfi_log_odds + Poisson sidecar features."""
    import gc

    required_cols = ["f1_nrfi", "f1_home_runs", "f1_away_runs", "year"]
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
    oof_lam_home     = np.zeros(n_oof, dtype=float)
    oof_lam_away     = np.zeros(n_oof, dtype=float)
    oof_p_nrfi_poi   = np.zeros(n_oof, dtype=float)
    oof_labels       = oof_df["f1_nrfi"].values.astype(int)

    year_pos = {yr: np.where(oof_df["year"].values == yr)[0] for yr in eligible}

    print(f"\n  [OOF] Leave-One-Year-Out CV across {eligible}")
    print(f"  [OOF] Pool rows: {len(full):,} total  |  OOF rows: {n_oof:,}")

    for val_yr in eligible:
        tr = full[full["year"] != val_yr].reset_index(drop=True)
        va = full[full["year"] == val_yr].reset_index(drop=True)
        if len(tr) == 0 or len(va) == 0:
            continue

        X_tr  = tr[feat_cols].fillna(0).values.astype(np.float32)
        y_tr  = tr["f1_nrfi"].values.astype(int)
        sw_tr = _sample_weights(tr)
        X_va  = va[feat_cols].fillna(0).values.astype(np.float32)
        y_va  = va["f1_nrfi"].values.astype(int)

        m   = _train_xgb(X_tr, y_tr, sw_tr, X_va, y_va)
        raw = m.predict_proba(X_va)[:, 1]

        tr_doubled, team_feat_cols = _build_team_dataset(tr, feat_cols)
        X_tm_tr = tr_doubled[team_feat_cols].fillna(0).values.astype(np.float32)
        y_tm_tr = tr_doubled["f1_nrfi"].values.astype(int)
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

        del tr_doubled, X_tm_tr, y_tm_tr, sw_tm, X_tm_es, tm_m, va_home, va_flip, va_es
        gc.collect()

        # Dual-Poisson sidecar on f1_home_runs / f1_away_runs
        y_hr_tr = tr["f1_home_runs"].astype(float).values
        y_ar_tr = tr["f1_away_runs"].astype(float).values
        bst_hr  = _train_poisson_booster(X_tr, y_hr_tr, POISSON_N_ROUNDS_HOME)
        bst_ar  = _train_poisson_booster(X_tr, y_ar_tr, POISSON_N_ROUNDS_AWAY)
        lam_h_fold = _poisson_predict(bst_hr, X_va)
        lam_a_fold = _poisson_predict(bst_ar, X_va)
        p_nrfi_fold = _prob_nrfi_from_lambdas(lam_h_fold, lam_a_fold)

        pos = year_pos[val_yr]
        oof_probs[pos]        = raw
        oof_team_logodds[pos] = lo_fold
        oof_lam_home[pos]     = lam_h_fold
        oof_lam_away[pos]     = lam_a_fold
        oof_p_nrfi_poi[pos]   = p_nrfi_fold

        def _safe_auc(y, p):
            return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
        print(f"    Fold {val_yr}: tr={len(tr):>5,}  va={len(va):>5,}  "
              f"XGB-AUC={_safe_auc(y_va, raw):.4f}  "
              f"team-logodds-AUC={_safe_auc(y_va, lo_fold):.4f}  "
              f"poi-AUC={_safe_auc(y_va, p_nrfi_fold):.4f}")

        del m, raw, bst_hr, bst_ar, lam_h_fold, lam_a_fold, p_nrfi_fold
        del X_tr, y_tr, sw_tr, X_va, y_va, tr, va, lo_fold, p_h, p_a
        gc.collect()

    assert len(oof_df) == n_oof == oof_probs.shape[0]

    def _safe_auc(y, p):
        return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    print(f"  [OOF] Combined: n={n_oof:,}  "
          f"XGB={_safe_auc(oof_labels, oof_probs):.4f}  "
          f"team-logodds={_safe_auc(oof_labels, oof_team_logodds):.4f}  "
          f"poi={_safe_auc(oof_labels, oof_p_nrfi_poi):.4f}")

    oof_df["team_nrfi_log_odds"]   = oof_team_logodds
    oof_df["pois_lam_f1_home"]     = oof_lam_home
    oof_df["pois_lam_f1_away"]     = oof_lam_away
    oof_df["pois_p_nrfi"]          = oof_p_nrfi_poi

    oof_segs = _derive_segment_id(oof_df)
    return oof_probs, oof_labels, oof_df, oof_segs


# ---------------------------------------------------------------------------
# STEP 4: Main training
# ---------------------------------------------------------------------------

def _fixed_bin_calibration(y_true: np.ndarray, y_prob: np.ndarray, label: str = ""):
    df_c = pd.DataFrame({"pred": y_prob, "actual": y_true})
    df_c["band"] = pd.cut(df_c["pred"], bins=CALIB_BINS, labels=CALIB_LABELS)
    hdr = f"\n  Fixed-bin calibration{' — ' + label if label else ''}:"
    print(hdr)
    print(f"  {'Band':>8}  {'n':>5}  {'Pred':>7}  {'Actual':>7}  {'Diff':>8}")
    print(f"  {'-'*46}")
    for band, grp in df_c.groupby("band", observed=True):
        if len(grp) == 0:
            continue
        diff = grp["actual"].mean() - grp["pred"].mean()
        print(f"  {str(band):>8}  {len(grp):>5}  "
              f"{grp['pred'].mean():>7.3f}  {grp['actual'].mean():>7.3f}  {diff:>+8.3f}")


def train_default(df: pd.DataFrame, feat_cols: list[str], val_year: int = 2025):
    """Train on years before val_year → validate on val_year, L1 + calibrator + team + Poisson + stacker."""
    train_years = [y for y in [2023, 2024, 2025] if y < val_year]
    print(f"\n[3] Default split: train {train_years} / validate {val_year}  [label: f1_nrfi]")
    train = df[df["year"].isin(train_years)]
    val   = df[df["year"] == val_year]
    print(f"    Train: {len(train)} | Val: {len(val)}")

    X_tr  = train[feat_cols].fillna(0).values.astype(np.float32)
    y_tr  = train["f1_nrfi"].values.astype(int)
    sw_tr = _sample_weights(train)
    X_val = val[feat_cols].fillna(0).values.astype(np.float32)
    y_val = val["f1_nrfi"].values.astype(int)

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

    imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print("\n  Top 20 feature importances (gain):")
    for feat, score in imp.head(20).items():
        print(f"    {feat:<45s} {score:.4f}")

    # Team model for 2025 val log-odds
    print("\n[3c] Team model: building 2023+2024 doubled dataset for val log-odds …")
    train_reset = train.reset_index(drop=True)
    val_reset   = val.reset_index(drop=True)
    train_doubled_tm, team_feat_cols_tm = _build_team_dataset(train_reset, feat_cols)
    X_tm_tr  = train_doubled_tm[team_feat_cols_tm].fillna(0).values.astype(np.float32)
    y_tm_tr  = train_doubled_tm["f1_nrfi"].values.astype(int)
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
    val["team_nrfi_log_odds"] = log_odds_val

    # Dual-Poisson sidecar
    print("\n[3c2] Dual-Poisson sidecar (F1 runs) …")
    y_hr_tr_p = train["f1_home_runs"].astype(float).values
    y_ar_tr_p = train["f1_away_runs"].astype(float).values
    bst_hr_full = _train_poisson_booster(X_tr, y_hr_tr_p, POISSON_N_ROUNDS_HOME)
    bst_ar_full = _train_poisson_booster(X_tr, y_ar_tr_p, POISSON_N_ROUNDS_AWAY)
    lam_h_val = _poisson_predict(bst_hr_full, X_val)
    lam_a_val = _poisson_predict(bst_ar_full, X_val)
    p_nrfi_val = _prob_nrfi_from_lambdas(lam_h_val, lam_a_val)
    val["pois_lam_f1_home"] = lam_h_val
    val["pois_lam_f1_away"] = lam_a_val
    val["pois_p_nrfi"]      = p_nrfi_val
    auc_poi_val = roc_auc_score(y_val, p_nrfi_val)
    print(f"  Poisson p_nrfi on 2025: AUC={auc_poi_val:.4f}  "
          f"mean lam_home={lam_h_val.mean():.3f}  lam_away={lam_a_val.mean():.3f}")

    bst_hr_full.save_model(str(OUTPUT_POIS_HOME))
    bst_ar_full.save_model(str(OUTPUT_POIS_AWAY))
    print(f"  Poisson boosters → {OUTPUT_POIS_HOME}")
    print(f"                   → {OUTPUT_POIS_AWAY}")

    # Stacker
    print("\n[3d] Training Bayesian Hierarchical Stacker (Level-2) …")
    stacker = train_nrfi_stacker(oof_probs, oof_df, oof_labels, oof_segs)

    val_segs = _derive_segment_id(val)
    stk_val  = stacker.predict(raw_val, val, val_segs)

    print("\n  Validation 2025 — full comparison:")
    print(f"  {'Model':<36}  {'AUC':>7}  {'LogLoss':>9}  {'Brier':>8}")
    print(f"  {'-'*68}")
    for lbl, probs in [
        ("XGBoost L1 (raw)",        raw_val),
        ("XGBoost L1 (OOF-Platt)",  cal_val),
        ("Bayesian Stacker L2",     stk_val),
    ]:
        auc = roc_auc_score(y_val, probs)
        ll  = log_loss(y_val, probs)
        bs  = brier_score_loss(y_val, probs)
        tag = "  <<<" if lbl.startswith("Bayesian") else ""
        print(f"  {lbl:<36}  {auc:>7.4f}  {ll:>9.4f}  {bs:>8.4f}{tag}")

    val_df_out = val[["game_pk", "game_date", "home_team", "away_team",
                       "f1_home_runs", "f1_away_runs", "f1_nrfi"]].copy()
    val_df_out["xgb_raw_nrfi"]   = raw_val
    val_df_out["xgb_cal_nrfi"]   = cal_val
    val_df_out["team_nrfi_log_odds"] = log_odds_val
    val_df_out["stacker_nrfi"]   = stk_val
    val_df_out.to_csv(OUTPUT_VAL_PREDS, index=False)
    print(f"\n  Saved validation predictions → {OUTPUT_VAL_PREDS}")

    return model, cal, stacker, feat_cols, val_df_out, log_odds_val


def train_final(df: pd.DataFrame, feat_cols: list[str], with_2026: bool = False):
    """Train final models on all data (2023+2024+2025 + optional 2026)."""
    years = [2023, 2024, 2025]
    if with_2026:
        years.append(2026)
    final_df = df[df["year"].isin(years)].reset_index(drop=True)
    print(f"\n[4] Final model: training on {years} ({len(final_df)} games) …")

    X  = final_df[feat_cols].fillna(0).values.astype(np.float32)
    y  = final_df["f1_nrfi"].values.astype(int)
    sw = _sample_weights(final_df)

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y, sample_weight=sw, verbose=False)
    print(f"  XGBoost trained on {len(final_df)} games  ({len(feat_cols)} features)")

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
        y_tr  = tr_sub["f1_nrfi"].values.astype(int)
        sw_tr = _sample_weights(tr_sub)
        X_va  = va_sub[feat_cols].fillna(0).values.astype(np.float32)
        y_va  = va_sub["f1_nrfi"].values.astype(int)

        m_oof = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=40)
        m_oof.fit(X_tr, y_tr, sample_weight=sw_tr,
                  eval_set=[(X_va, y_va)], verbose=False)
        raw_va = m_oof.predict_proba(X_va)[:, 1]
        oof_raw[va_mask] = raw_va
        print(f"    Held-out {held_yr}: n={len(va_sub):>4}  "
              f"AUC={roc_auc_score(y_va, raw_va):.4f}")

    cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    cal.fit(oof_raw.reshape(-1, 1), oof_lbl)
    print(f"  Platt OOF-fitted — overall OOF AUC={roc_auc_score(oof_lbl, oof_raw):.4f}")

    # Team model on all data
    print("  [4b] Training team model on all data (doubled dataset) …")
    doubled, team_feat_cols = _build_team_dataset(final_df, feat_cols)
    X_tm  = doubled[team_feat_cols].fillna(0).values.astype(np.float32)
    y_tm  = doubled["f1_nrfi"].values.astype(int)
    sw_tm = _sample_weights(doubled)

    last_yr = years[-1]
    es_sub  = final_df[final_df["year"] == last_yr].copy()
    es_sub["is_home"] = 1
    X_tm_es = es_sub[team_feat_cols].fillna(0).values.astype(np.float32)
    y_tm_es = es_sub["f1_nrfi"].values.astype(int)

    team_model = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=40)
    team_model.fit(X_tm, y_tm, sample_weight=sw_tm,
                   eval_set=[(X_tm_es, y_tm_es)], verbose=False)

    team_model.save_model(str(OUTPUT_TEAM_MODEL))
    json.dump(team_feat_cols, open(OUTPUT_TEAM_FEAT_COLS, "w"), indent=2)
    print(f"  Team model     → {OUTPUT_TEAM_MODEL}")
    print(f"  Team feat cols → {OUTPUT_TEAM_FEAT_COLS}")

    # Poisson sidecar on all data
    print("  [4c] Training Poisson sidecars on all data …")
    y_hr = final_df["f1_home_runs"].astype(float).values
    y_ar = final_df["f1_away_runs"].astype(float).values
    bst_hr_final = _train_poisson_booster(X, y_hr, POISSON_N_ROUNDS_HOME)
    bst_ar_final = _train_poisson_booster(X, y_ar, POISSON_N_ROUNDS_AWAY)
    bst_hr_final.save_model(str(OUTPUT_POIS_HOME))
    bst_ar_final.save_model(str(OUTPUT_POIS_AWAY))
    print(f"  Poisson boosters → {OUTPUT_POIS_HOME}")
    print(f"                   → {OUTPUT_POIS_AWAY}")

    # LOYO OOF + stacker refit on all years
    print("  [4d] Generating LOYO OOF across all training years for stacker …")
    oof_probs, oof_labels, oof_df, oof_segs = _generate_oof_for_stacker(
        final_df, feat_cols, years=years,
    )

    print("  [4d] Refitting Bayesian Hierarchical Stacker on full-year OOF …")
    stacker = train_nrfi_stacker(oof_probs, oof_df, oof_labels, oof_segs)
    print(f"  Stacker artefacts → {OUTPUT_STACKER}")
    print(f"                    → {OUTPUT_STACKER_NPZ}")

    return model, cal, stacker


# ---------------------------------------------------------------------------
# STEP 5: NCV mode
# ---------------------------------------------------------------------------

def train_ncv(df: pd.DataFrame, feat_cols: list[str]):
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
        y_tr  = tr["f1_nrfi"].values.astype(int)
        sw_tr = _sample_weights(tr)
        X_va  = va[feat_cols].fillna(0).values.astype(np.float32)
        y_va  = va["f1_nrfi"].values.astype(int)
        m     = _train_xgb(X_tr, y_tr, sw_tr, X_va, y_va)
        raw   = m.predict_proba(X_va)[:, 1]
        auc   = roc_auc_score(y_va, raw)
        ll    = log_loss(y_va, raw)
        print(f"  {label}: AUC={auc:.4f}  LogLoss={ll:.4f}")
        ncv_rows.append({"fold": label, "val_year": val_yr, "auc": auc, "logloss": ll})
    return ncv_rows


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost NRFI classifier")
    parser.add_argument("--ncv",          action="store_true",
                        help="Run nested cross-validation")
    parser.add_argument("--with-2026",    action="store_true",
                        help="Include 2026 partial data in final training")
    parser.add_argument("--matrix", type=str, default="feature_matrix.parquet",
                        help="Feature matrix parquet path")
    parser.add_argument("--val-year", type=int, default=2025,
                        help="Holdout year for validation (default: 2025)")
    parser.add_argument("--val-preds-out", type=str, default=None,
                        help="Override path for val predictions CSV")
    parser.add_argument("--no-save-model", action="store_true",
                        help="Skip saving model files (only writes val predictions)")
    args = parser.parse_args()

    print("=" * 70)
    print("NRFI XGBoost Model Training")
    print("=" * 70)

    global FEAT_MATRIX
    FEAT_MATRIX = Path(args.matrix)

    df, feat_cols = build_dataset(include_2026=args.with_2026)

    json.dump(feat_cols, open(OUTPUT_FEAT_COLS, "w"), indent=2)
    print(f"\n  Saved feature list → {OUTPUT_FEAT_COLS} ({len(feat_cols)} features)")

    if args.ncv:
        train_ncv(df, feat_cols)

    if args.val_preds_out:
        global OUTPUT_VAL_PREDS
        OUTPUT_VAL_PREDS = Path(args.val_preds_out)

    model_val, cal_val, stacker, _, val_df, _log_odds_val = train_default(df, feat_cols, val_year=args.val_year)

    if not args.no_save_model:
        model_final, cal_final, stacker_final = train_final(df, feat_cols, with_2026=args.with_2026)
        model_final.save_model(str(OUTPUT_MODEL))
        pickle.dump(cal_final, open(OUTPUT_CALIB, "wb"))
        print(f"\n  Saved final model -> {OUTPUT_MODEL}")
        print(f"  Saved calibrator  -> {OUTPUT_CALIB}")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
