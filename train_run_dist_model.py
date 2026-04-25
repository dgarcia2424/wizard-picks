"""
train_run_dist_model.py
=======================
Unified Run Distribution trainer — replaces the split Runline (RL) and Totals
pipelines with a single joint-distribution model.

Architecture — Two L1 signals, Two L2 stackers
----------------------------------------------
  Level 1:
    (a) DixonColesMLB — Poisson attack/defense panel fit on
        (home_team, away_team, home_runs, away_runs).  Yields:
          p_over_dc, p_cover_dc, total_dc, diff_dc

    (b) Dual-Poisson XGBoost — two xgb.XGBRegressor(objective='count:poisson')
        predict lam_home, lam_away on the standard feature matrix.  Analytic
        convolution (K=25) yields:
          p_over_xgb, p_cover_xgb

  Level 2 (Bayesian, NumPyro NUTS):
    BayesianStackerTotals — anchors on logit(p_over_dc), augments with
      [logit(p_over_xgb), lam_sum_centered, total_dc_centered].
    BayesianStackerRL — anchors on logit(p_cover_dc), augments with
      [logit(p_cover_xgb), lam_diff_centered, diff_dc_centered].
    Both are flat logistic models (no segment hierarchy — handedness signal
    is absorbed by the L1 lambdas).

Labels:
  total_line column: prefers 'close_total' from the feature matrix, falls
    back to 8.5 where missing.  over = int((home_runs + away_runs) > total_line)
  runline line: fixed at -1.5.  home_cover = int((home_runs - away_runs) > -1.5)

Outputs
-------
  models/dc_model_run_dist.pkl
  models/xgb_run_dist_lam_home.json
  models/xgb_run_dist_lam_away.json
  models/stacker_totals.pkl, models/stacker_totals.npz
  models/stacker_rl.pkl,     models/stacker_rl.npz
  models/run_dist_feature_cols.json

Usage
-----
  python train_run_dist_model.py --matrix feature_matrix_enriched_v2.parquet
"""

from __future__ import annotations

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

from dixon_coles_mlb import DixonColesMLB

# ── NumPyro / JAX ────────────────────────────────────────────────────────────
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
BASE_DIR   = Path(".")
DATA_DIR   = BASE_DIR / "data" / "statcast"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEAT_MATRIX = BASE_DIR / "feature_matrix_enriched_v2.parquet"

OUT_DC              = MODELS_DIR / "dc_model_run_dist.pkl"
OUT_XGB_LAM_HOME    = MODELS_DIR / "xgb_run_dist_lam_home.json"
OUT_XGB_LAM_AWAY    = MODELS_DIR / "xgb_run_dist_lam_away.json"
OUT_STACKER_TOTALS  = MODELS_DIR / "stacker_totals.pkl"
OUT_STACKER_TOTALS_NPZ = MODELS_DIR / "stacker_totals.npz"
OUT_STACKER_RL      = MODELS_DIR / "stacker_rl.pkl"
OUT_STACKER_RL_NPZ  = MODELS_DIR / "stacker_rl.npz"
OUT_FEAT_COLS       = MODELS_DIR / "run_dist_feature_cols.json"

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
FALLBACK_TOTAL_LINE = 8.5
RUNLINE = -1.5
K_TRUNC = 25  # analytic convolution truncation for XGB lambdas

YEAR_DECAY = {2023: 0.70, 2024: 1.00, 2025: 1.50, 2026: 2.00}

XGB_POIS_PARAMS = {
    "objective":        "count:poisson",
    "tree_method":      "hist",
    "n_jobs":           -1,
    "random_state":     42,
    "learning_rate":    0.05,
    "max_depth":        5,
    "min_child_weight": 5,
    "subsample":        0.85,
    "colsample_bytree": 0.85,
    "reg_lambda":       1.0,
    "max_delta_step":   0.7,
    "n_estimators":     800,
}


# ---------------------------------------------------------------------------
# ANALYTIC POISSON CONVOLUTION (for XGB side)
# ---------------------------------------------------------------------------
def _poisson_pmf(lam: float, k_max: int = K_TRUNC) -> np.ndarray:
    from scipy.special import gammaln
    k = np.arange(k_max + 1)
    logp = k * np.log(max(lam, 1e-9)) - lam - gammaln(k + 1)
    p = np.exp(logp)
    s = p.sum()
    if s > 0:
        p = p / s
    return p


def p_over_poisson(lam_h: float, lam_a: float, total_line: float,
                    k_max: int = K_TRUNC) -> float:
    ph = _poisson_pmf(lam_h, k_max)
    pa = _poisson_pmf(lam_a, k_max)
    idx_h = np.arange(k_max + 1).reshape(-1, 1)
    idx_a = np.arange(k_max + 1).reshape(1, -1)
    mask = (idx_h + idx_a) > total_line
    M = np.outer(ph, pa)
    return float(M[mask].sum())


def p_cover_poisson(lam_h: float, lam_a: float, rl_line: float = RUNLINE,
                     k_max: int = K_TRUNC) -> float:
    ph = _poisson_pmf(lam_h, k_max)
    pa = _poisson_pmf(lam_a, k_max)
    idx_h = np.arange(k_max + 1).reshape(-1, 1)
    idx_a = np.arange(k_max + 1).reshape(1, -1)
    mask = (idx_h - idx_a) > rl_line
    M = np.outer(ph, pa)
    return float(M[mask].sum())


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log(1 - p)


# ---------------------------------------------------------------------------
# BAYESIAN STACKER MODELS
# ---------------------------------------------------------------------------
def _numpyro_flat_stacker_model(l1_prob, X_aux, n_aux, y_obs=None):
    alpha = numpyro.sample("alpha", dist.Normal(0.0, 1.0))
    beta  = numpyro.sample("beta",  dist.Normal(0.0, 1.5))

    with numpyro.plate("aux_feats", n_aux):
        gamma = numpyro.sample("gamma", dist.Normal(0.0, 0.5))

    lp = jnp.clip(l1_prob, 1e-6, 1 - 1e-6)
    logit_p = jnp.log(lp) - jnp.log(1 - lp)
    theta = alpha + beta * logit_p + X_aux @ gamma

    with numpyro.plate("data", len(l1_prob)):
        numpyro.sample("y", dist.Bernoulli(logits=theta), obs=y_obs)


class _BayesianStackerFlat:
    """Shared base for the two flat run-dist stackers."""
    aux_feature_names: list[str] = []

    def __init__(self, alpha: float, beta: float, gamma: np.ndarray,
                 aux_feature_names: list[str], aux_centers: dict[str, float],
                 posterior_path: str | None = None):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = np.asarray(gamma, dtype=float)
        self.aux_feature_names = list(aux_feature_names)
        self.aux_centers = dict(aux_centers)
        self.posterior_path = posterior_path

    def _build_X_aux(self, aux_df: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(index=aux_df.index)
        for col in self.aux_feature_names:
            if col in aux_df.columns:
                X[col] = aux_df[col].astype(float)
            else:
                X[col] = 0.0
            if col in self.aux_centers:
                X[col] = X[col] - self.aux_centers[col]
            X[col] = X[col].fillna(0.0)
        return X.values.astype(float)

    def predict(self, l1_prob: np.ndarray, aux_df: pd.DataFrame) -> np.ndarray:
        l1_prob = np.asarray(l1_prob, dtype=float).ravel()
        X_aux = self._build_X_aux(aux_df)
        logit_p = _logit(l1_prob)
        theta = self.alpha + self.beta * logit_p + X_aux @ self.gamma
        return 1.0 / (1.0 + np.exp(-theta))

    def predict_proba(self, l1_prob: np.ndarray, aux_df: pd.DataFrame) -> np.ndarray:
        p = self.predict(l1_prob, aux_df)
        return np.column_stack([1.0 - p, p])


class BayesianStackerTotals(_BayesianStackerFlat):
    """Totals (over/under) stacker. Anchor: p_over_dc."""
    pass


class BayesianStackerRL(_BayesianStackerFlat):
    """Runline (home -1.5 cover) stacker. Anchor: p_cover_dc."""
    pass


def _fit_flat_stacker(
    klass,
    l1_prob: np.ndarray,
    aux_df: pd.DataFrame,
    aux_feature_names: list[str],
    aux_centers: dict[str, float],
    y: np.ndarray,
    out_pkl: Path,
    out_npz: Path,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 2,
    tag: str = "",
):
    # Build centered X_aux
    X_df = pd.DataFrame(index=aux_df.index)
    for col in aux_feature_names:
        if col in aux_df.columns:
            X_df[col] = aux_df[col].astype(float)
        else:
            X_df[col] = 0.0
        X_df[col] = X_df[col] - aux_centers.get(col, 0.0)
        X_df[col] = X_df[col].fillna(0.0)
    X_aux = X_df.values.astype(float)
    n_aux = X_aux.shape[1]

    if not _NUMPYRO:
        print(f"  [WARN] NumPyro unavailable — LR fallback for {tag}")
        Xstk = np.hstack([_logit(l1_prob).reshape(-1, 1), X_aux])
        lr = LogisticRegression(C=10, solver="lbfgs", max_iter=1000)
        lr.fit(Xstk, y.astype(int))
        model = klass(
            alpha=float(lr.intercept_[0]),
            beta=float(lr.coef_[0, 0]),
            gamma=lr.coef_[0, 1:],
            aux_feature_names=aux_feature_names,
            aux_centers=aux_centers,
        )
        out_pkl.write_bytes(pickle.dumps(model))
        print(f"  Fallback stacker saved -> {out_pkl}")
        return model

    print(f"\n{'='*60}")
    print(f"  RUN-DIST BAYESIAN STACKER — {tag}  ({_JAX_PLATFORM.upper()})")
    print(f"{'='*60}")
    print(f"  n={len(l1_prob):,}  aux_features={n_aux}  "
          f"MCMC: {num_chains} chains x ({num_warmup}+{num_samples})")

    p_jax = jnp.array(np.clip(l1_prob, 1e-6, 1 - 1e-6))
    X_jax = jnp.array(X_aux)
    y_jax = jnp.array(y.astype(float))

    kernel = NUTS(_numpyro_flat_stacker_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=True)
    mcmc.run(_jax.random.PRNGKey(42), p_jax, X_jax, n_aux, y_obs=y_jax)
    samples = mcmc.get_samples()

    alpha_hat = float(np.mean(samples["alpha"]))
    beta_hat  = float(np.mean(samples["beta"]))
    gamma_hat = np.mean(samples["gamma"], axis=0)

    print(f"\n  Posterior means [{tag}]:")
    print(f"    alpha {alpha_hat:+.4f} +/- {float(np.std(samples['alpha'])):.4f}")
    print(f"    beta  {beta_hat:+.4f} +/- {float(np.std(samples['beta'])):.4f}")
    for k, feat in enumerate(aux_feature_names):
        print(f"    gamma  {feat:<28} {gamma_hat[k]:+.4f}")

    np.savez(str(out_npz), **{k: np.array(v) for k, v in samples.items()})
    print(f"  Posterior trace -> {out_npz}")

    model = klass(
        alpha=alpha_hat, beta=beta_hat, gamma=gamma_hat,
        aux_feature_names=aux_feature_names, aux_centers=aux_centers,
        posterior_path=str(out_npz),
    )
    out_pkl.write_bytes(pickle.dumps(model))
    print(f"  {klass.__name__} -> {out_pkl}")

    # Quick OOF fit diagnostics
    p_stk = model.predict(l1_prob, aux_df)
    try:
        auc_l1 = roc_auc_score(y, l1_prob)
        auc_s  = roc_auc_score(y, p_stk)
        ll_l1  = log_loss(y, np.clip(l1_prob, 1e-6, 1 - 1e-6))
        ll_s   = log_loss(y, np.clip(p_stk, 1e-6, 1 - 1e-6))
        print(f"\n  OOF perf [{tag}]:")
        print(f"    L1 anchor  AUC={auc_l1:.4f}  LL={ll_l1:.4f}")
        print(f"    Stacker L2 AUC={auc_s:.4f}  LL={ll_s:.4f}  "
              f"(delta {auc_s - auc_l1:+.4f})")
    except Exception as e:
        print(f"  (diagnostics skipped: {e})")
    return model


# ---------------------------------------------------------------------------
# LABELS + TOTAL LINE
# ---------------------------------------------------------------------------
def _derive_scores(fm: pd.DataFrame) -> pd.DataFrame:
    """Ensure home_runs, away_runs available. Fall back to home_score/away_score."""
    out = fm.copy()
    if "home_runs" not in out.columns:
        if "home_score" in out.columns:
            out["home_runs"] = out["home_score"]
        elif "home_score_final" in out.columns:
            out["home_runs"] = out["home_score_final"]
        else:
            out["home_runs"] = np.nan
    if "away_runs" not in out.columns:
        if "away_score" in out.columns:
            out["away_runs"] = out["away_score"]
        elif "away_score_final" in out.columns:
            out["away_runs"] = out["away_score_final"]
        else:
            out["away_runs"] = np.nan
    return out


def _resolve_total_line(fm: pd.DataFrame) -> tuple[pd.Series, str]:
    """Prefer close_total / open_total / vegas_total; else fallback 8.5."""
    for col in ("close_total", "open_total", "vegas_total", "total_line",
                "closing_total", "ou_line"):
        if col in fm.columns:
            s = pd.to_numeric(fm[col], errors="coerce")
            if s.notna().sum() > 0:
                filled = s.fillna(FALLBACK_TOTAL_LINE)
                print(f"  [total_line] source column: '{col}'  "
                      f"coverage={s.notna().mean():.1%}  "
                      f"fallback_{FALLBACK_TOTAL_LINE} for missing")
                return filled, col
    print(f"  [total_line] no Vegas line column found — using fixed "
          f"{FALLBACK_TOTAL_LINE} for all rows")
    return pd.Series([FALLBACK_TOTAL_LINE] * len(fm), index=fm.index), "FALLBACK_8.5"


# ---------------------------------------------------------------------------
# DATASET BUILD
# ---------------------------------------------------------------------------
NON_FEATURE_COLS = {
    "game_pk", "game_date", "home_team", "away_team",
    "home_starter_name", "away_starter_name", "season", "year", "split",
    "home_score", "away_score", "home_margin", "home_runs", "away_runs",
    "home_covers_rl", "away_covers_rl", "total_runs",
    "actual_home_win", "actual_game_total", "actual_f5_total",
    "actual_f3_total", "actual_f1_total",
    "close_ml_home", "close_ml_away", "open_total", "close_total",
    "true_home_prob", "true_away_prob",
    "vegas_implied_home", "vegas_implied_away",
    "mc_expected_runs",
    "source", "pull_timestamp",
    "temp_f", "wind_mph", "wind_bearing",
    "home_1st_inn_run_rate", "away_1st_inn_run_rate",
}


def build_dataset(matrix_path: Path, include_2026: bool = False,
                   feat_cols_override: list[str] | None = None
                   ) -> tuple[pd.DataFrame, list[str], str]:
    print("\n[1] Loading feature matrix …")
    fm = pd.read_parquet(matrix_path)
    print(f"    Matrix: {fm.shape[0]} rows x {fm.shape[1]} cols")

    fm = _derive_scores(fm)
    fm = fm.dropna(subset=["home_runs", "away_runs",
                            "home_team", "away_team"]).reset_index(drop=True)
    fm["home_runs"] = fm["home_runs"].astype(float)
    fm["away_runs"] = fm["away_runs"].astype(float)

    print("\n[2] Resolving total line …")
    total_line, total_line_src = _resolve_total_line(fm)
    fm["total_line"] = total_line.values

    fm["over"] = ((fm["home_runs"] + fm["away_runs"]) > fm["total_line"]).astype(int)
    fm["home_cover"] = ((fm["home_runs"] - fm["away_runs"]) > RUNLINE).astype(int)
    # Drop exact-push totals rows
    push_mask = (fm["home_runs"] + fm["away_runs"]) == fm["total_line"]
    if push_mask.any():
        print(f"    Dropping {int(push_mask.sum())} total-push rows")
        fm = fm.loc[~push_mask].reset_index(drop=True)

    if "year" not in fm.columns:
        fm["year"] = pd.to_datetime(fm["game_date"]).dt.year
    fm["year"] = fm["year"].astype(int)

    if not include_2026:
        fm = fm[fm["year"].isin([2023, 2024, 2025])].reset_index(drop=True)
        print(f"    After dropping 2026: {len(fm)} rows")

    fm["game_date"] = pd.to_datetime(fm["game_date"])
    fm = fm.sort_values("game_date").reset_index(drop=True)

    if feat_cols_override is not None:
        feat_cols = [c for c in feat_cols_override if c in fm.columns]
        missing = [c for c in feat_cols_override if c not in fm.columns]
        if missing:
            print(f"    [WARN] {len(missing)} manifest features missing from matrix: "
                  f"{missing[:8]}{'…' if len(missing) > 8 else ''}")
    else:
        excl = NON_FEATURE_COLS | {"total_line", "over", "home_cover"}
        feat_cols = [c for c in fm.columns if c not in excl]

    print(f"    Features: {len(feat_cols)}")
    print(f"    Over rate: {fm['over'].mean():.3f}  |  "
          f"Home-cover rate: {fm['home_cover'].mean():.3f}")
    return fm, feat_cols, total_line_src


# ---------------------------------------------------------------------------
# DC SIGNALS for a va slice (given a DC fit on tr)
# ---------------------------------------------------------------------------
def _dc_signals_for_slice(dc: DixonColesMLB, va: pd.DataFrame,
                           K: int = K_TRUNC) -> pd.DataFrame:
    p_over = np.zeros(len(va), dtype=float)
    p_cover = np.zeros(len(va), dtype=float)
    total_dc = np.zeros(len(va), dtype=float)
    diff_dc = np.zeros(len(va), dtype=float)

    home_teams = va["home_team"].values
    away_teams = va["away_team"].values
    total_lines = va["total_line"].values

    idx_h = np.arange(K + 1).reshape(-1, 1)
    idx_a = np.arange(K + 1).reshape(1, -1)

    for i in range(len(va)):
        M = dc.predict_match_matrix(home_teams[i], away_teams[i])
        # Constrain to K+1 just in case DC has bigger max_runs
        M = M[:K + 1, :K + 1]
        Msum = M.sum()
        if Msum > 0:
            M = M / Msum
        line = float(total_lines[i])
        p_over[i] = float(M[(idx_h + idx_a) > line].sum())
        p_cover[i] = float(M[(idx_h - idx_a) > RUNLINE].sum())
        total_dc[i] = float((M * (idx_h + idx_a)).sum())
        diff_dc[i] = float((M * (idx_h - idx_a)).sum())

    return pd.DataFrame({
        "p_over_dc":  p_over,
        "p_cover_dc": p_cover,
        "total_dc":   total_dc,
        "diff_dc":    diff_dc,
    }, index=va.index)


# ---------------------------------------------------------------------------
# LOYO OOF GENERATOR
# ---------------------------------------------------------------------------
def _sample_weights(df: pd.DataFrame) -> np.ndarray:
    return df["year"].map(YEAR_DECAY).fillna(1.0).values


def _fit_poisson_xgb(X_tr, y_tr, sw_tr, X_va, y_va) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**XGB_POIS_PARAMS, early_stopping_rounds=50)
    model.fit(X_tr, y_tr, sample_weight=sw_tr,
              eval_set=[(X_va, y_va)], verbose=False)
    return model


def _generate_oof(
    df: pd.DataFrame,
    feat_cols: list[str],
    years: list[int] | None = None,
) -> pd.DataFrame:
    """
    Produces a DataFrame aligned to df.index containing:
       p_over_dc, p_cover_dc, total_dc, diff_dc,
       lam_home, lam_away, p_over_xgb, p_cover_xgb,
       total_line, over, home_cover, year
    """
    full = df.dropna(subset=["over", "home_cover", "year"]).reset_index(drop=True).copy()
    full["year"] = full["year"].astype(int)
    if years is None:
        years = sorted(full["year"].unique().tolist())
    years = [int(y) for y in years if int(y) in set(full["year"].unique())]

    eligible = [y for y in years if len(set(years) - {y}) >= 1]
    if len(eligible) < 2:
        raise ValueError("LOYO requires >=2 distinct years")
    oof_mask = full["year"].isin(eligible).values
    oof_df = full.loc[oof_mask].reset_index(drop=True).copy()
    n = len(oof_df)

    out = pd.DataFrame(index=oof_df.index, data={
        "p_over_dc":   np.zeros(n),
        "p_cover_dc":  np.zeros(n),
        "total_dc":    np.zeros(n),
        "diff_dc":     np.zeros(n),
        "lam_home":    np.zeros(n),
        "lam_away":    np.zeros(n),
        "p_over_xgb":  np.zeros(n),
        "p_cover_xgb": np.zeros(n),
    })

    year_pos = {yr: np.where(oof_df["year"].values == yr)[0] for yr in eligible}

    print(f"\n  [OOF] LOYO across {eligible}  |  pool={len(full):,}  oof={n:,}")

    for val_yr in eligible:
        tr = full[full["year"] != val_yr].reset_index(drop=True)
        va = oof_df[oof_df["year"] == val_yr].copy()
        if len(tr) == 0 or len(va) == 0:
            continue

        # ── DC phase ────────────────────────────────────────────────
        dc = DixonColesMLB(max_runs=K_TRUNC)
        dc.fit(tr[["home_team", "away_team", "home_runs", "away_runs"]])
        dc_signals = _dc_signals_for_slice(dc, va)

        del dc
        gc.collect()

        # ── XGB Poisson phase ───────────────────────────────────────
        X_tr = tr[feat_cols].fillna(0).values.astype(np.float32)
        sw_tr = _sample_weights(tr)
        y_tr_h = tr["home_runs"].astype(float).values
        y_tr_a = tr["away_runs"].astype(float).values

        X_va = va[feat_cols].fillna(0).values.astype(np.float32)
        y_va_h = va["home_runs"].astype(float).values
        y_va_a = va["away_runs"].astype(float).values

        m_home = _fit_poisson_xgb(X_tr, y_tr_h, sw_tr, X_va, y_va_h)
        m_away = _fit_poisson_xgb(X_tr, y_tr_a, sw_tr, X_va, y_va_a)

        lam_h = np.clip(m_home.predict(X_va), 0.1, 15.0)
        lam_a = np.clip(m_away.predict(X_va), 0.1, 15.0)

        total_lines = va["total_line"].values.astype(float)
        p_over_xgb  = np.array([p_over_poisson(lh, la, tl)
                                  for lh, la, tl in zip(lam_h, lam_a, total_lines)])
        p_cover_xgb = np.array([p_cover_poisson(lh, la, RUNLINE)
                                  for lh, la in zip(lam_h, lam_a)])

        # Scatter into oof arrays
        pos = year_pos[val_yr]
        out.iloc[pos, out.columns.get_loc("p_over_dc")]   = dc_signals["p_over_dc"].values
        out.iloc[pos, out.columns.get_loc("p_cover_dc")]  = dc_signals["p_cover_dc"].values
        out.iloc[pos, out.columns.get_loc("total_dc")]    = dc_signals["total_dc"].values
        out.iloc[pos, out.columns.get_loc("diff_dc")]     = dc_signals["diff_dc"].values
        out.iloc[pos, out.columns.get_loc("lam_home")]    = lam_h
        out.iloc[pos, out.columns.get_loc("lam_away")]    = lam_a
        out.iloc[pos, out.columns.get_loc("p_over_xgb")]  = p_over_xgb
        out.iloc[pos, out.columns.get_loc("p_cover_xgb")] = p_cover_xgb

        def _safe_auc(y, p):
            return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")

        y_over_va = va["over"].astype(int).values
        y_cov_va  = va["home_cover"].astype(int).values
        print(f"    Fold {val_yr}: tr={len(tr):>5,}  va={len(va):>5,}  "
              f"over[DC={_safe_auc(y_over_va, dc_signals['p_over_dc'].values):.3f} "
              f"XGB={_safe_auc(y_over_va, p_over_xgb):.3f}]  "
              f"cover[DC={_safe_auc(y_cov_va, dc_signals['p_cover_dc'].values):.3f} "
              f"XGB={_safe_auc(y_cov_va, p_cover_xgb):.3f}]")

        del m_home, m_away, X_tr, X_va, sw_tr, y_tr_h, y_tr_a, y_va_h, y_va_a
        del tr, va, dc_signals, lam_h, lam_a, p_over_xgb, p_cover_xgb
        gc.collect()

    assert len(out) == n, f"OOF length mismatch: out={len(out)} expected={n}"

    out["total_line"] = oof_df["total_line"].values
    out["over"]       = oof_df["over"].astype(int).values
    out["home_cover"] = oof_df["home_cover"].astype(int).values
    out["year"]       = oof_df["year"].astype(int).values
    out["lam_sum"]    = out["lam_home"] + out["lam_away"]
    out["lam_diff"]   = out["lam_home"] - out["lam_away"]
    return out


# ---------------------------------------------------------------------------
# STACKER FITS (both Totals + RL)
# ---------------------------------------------------------------------------
def _fit_totals_stacker(oof: pd.DataFrame, out_pkl: Path, out_npz: Path
                          ) -> BayesianStackerTotals:
    l1 = oof["p_over_dc"].values
    aux_df = pd.DataFrame({
        "logit_p_over_xgb":     _logit(oof["p_over_xgb"].values),
        "lam_sum":              oof["lam_sum"].values.astype(float),
        "total_dc":             oof["total_dc"].values.astype(float),
    })
    centers = {
        "logit_p_over_xgb": 0.0,
        "lam_sum":          float(aux_df["lam_sum"].mean()),
        "total_dc":         float(aux_df["total_dc"].mean()),
    }
    aux_names = list(aux_df.columns)
    return _fit_flat_stacker(
        BayesianStackerTotals, l1, aux_df, aux_names, centers,
        oof["over"].values.astype(int), out_pkl, out_npz, tag="TOTALS",
    )


def _fit_rl_stacker(oof: pd.DataFrame, out_pkl: Path, out_npz: Path
                      ) -> BayesianStackerRL:
    l1 = oof["p_cover_dc"].values
    aux_df = pd.DataFrame({
        "logit_p_cover_xgb": _logit(oof["p_cover_xgb"].values),
        "lam_diff":          oof["lam_diff"].values.astype(float),
        "diff_dc":           oof["diff_dc"].values.astype(float),
    })
    centers = {
        "logit_p_cover_xgb": 0.0,
        "lam_diff":          float(aux_df["lam_diff"].mean()),
        "diff_dc":           float(aux_df["diff_dc"].mean()),
    }
    aux_names = list(aux_df.columns)
    return _fit_flat_stacker(
        BayesianStackerRL, l1, aux_df, aux_names, centers,
        oof["home_cover"].values.astype(int), out_pkl, out_npz, tag="RUNLINE",
    )


# ---------------------------------------------------------------------------
# DEFAULT SPLIT DIAGNOSTICS (2023+2024 → 2025)
# ---------------------------------------------------------------------------
def train_default(df: pd.DataFrame, feat_cols: list[str], val_year: int = 2025):
    train_years = [y for y in [2023, 2024, 2025] if y < val_year]
    print(f"\n[3] Default split: train {train_years} / validate {val_year}")
    train = df[df["year"].isin(train_years)].reset_index(drop=True)
    val   = df[df["year"] == val_year].reset_index(drop=True)
    print(f"    Train: {len(train)}  Val: {len(val)}")
    if len(val) == 0:
        print(f"    [skip] no {val_year} rows")
        return None

    # DC fit on train
    dc = DixonColesMLB(max_runs=K_TRUNC)
    dc.fit(train[["home_team", "away_team", "home_runs", "away_runs"]])
    dc_val = _dc_signals_for_slice(dc, val)

    # XGBs
    X_tr = train[feat_cols].fillna(0).values.astype(np.float32)
    sw_tr = _sample_weights(train)
    X_val = val[feat_cols].fillna(0).values.astype(np.float32)

    m_home = _fit_poisson_xgb(X_tr, train["home_runs"].astype(float).values,
                               sw_tr, X_val, val["home_runs"].astype(float).values)
    m_away = _fit_poisson_xgb(X_tr, train["away_runs"].astype(float).values,
                               sw_tr, X_val, val["away_runs"].astype(float).values)

    lam_h = np.clip(m_home.predict(X_val), 0.1, 15.0)
    lam_a = np.clip(m_away.predict(X_val), 0.1, 15.0)
    tot_lines = val["total_line"].values.astype(float)
    p_over_xgb  = np.array([p_over_poisson(h, a, tl)
                             for h, a, tl in zip(lam_h, lam_a, tot_lines)])
    p_cover_xgb = np.array([p_cover_poisson(h, a, RUNLINE)
                             for h, a in zip(lam_h, lam_a)])

    y_over = val["over"].astype(int).values
    y_cov  = val["home_cover"].astype(int).values

    def _safe_auc(y, p):
        return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")

    print(f"\n  L1 Val-2025 diagnostics:")
    print(f"    Actual total mean={val['home_runs'].mean() + val['away_runs'].mean():.3f}  "
          f"XGB lam_sum mean={(lam_h + lam_a).mean():.3f}  "
          f"DC total mean={dc_val['total_dc'].mean():.3f}")
    print(f"    Over  AUC: DC={_safe_auc(y_over, dc_val['p_over_dc'].values):.4f}  "
          f"XGB={_safe_auc(y_over, p_over_xgb):.4f}")
    print(f"    Cover AUC: DC={_safe_auc(y_cov, dc_val['p_cover_dc'].values):.4f}  "
          f"XGB={_safe_auc(y_cov, p_cover_xgb):.4f}")

    print(f"\n[3b] LOYO OOF on {train_years} for default-split stackers …")
    oof_23_24 = _generate_oof(df, feat_cols, years=train_years)
    print(f"    OOF rows: {len(oof_23_24)}")

    st_totals = _fit_totals_stacker(oof_23_24,
                                      MODELS_DIR / "_tmp_stacker_totals_default.pkl",
                                      MODELS_DIR / "_tmp_stacker_totals_default.npz")
    st_rl     = _fit_rl_stacker(oof_23_24,
                                  MODELS_DIR / "_tmp_stacker_rl_default.pkl",
                                  MODELS_DIR / "_tmp_stacker_rl_default.npz")

    # Apply on val
    val_aux_tot = pd.DataFrame({
        "logit_p_over_xgb": _logit(p_over_xgb),
        "lam_sum":          lam_h + lam_a,
        "total_dc":         dc_val["total_dc"].values,
    })
    p_over_final = st_totals.predict(dc_val["p_over_dc"].values, val_aux_tot)

    val_aux_rl = pd.DataFrame({
        "logit_p_cover_xgb": _logit(p_cover_xgb),
        "lam_diff":          lam_h - lam_a,
        "diff_dc":           dc_val["diff_dc"].values,
    })
    p_cover_final = st_rl.predict(dc_val["p_cover_dc"].values, val_aux_rl)

    print(f"\n  L2 Val-2025 (stackers trained on 2023+2024 OOF):")
    for lbl, probs, y in [
        ("Totals DC (L1)",        dc_val["p_over_dc"].values, y_over),
        ("Totals XGB (L1)",       p_over_xgb,                 y_over),
        ("Totals Stacker (L2)",   p_over_final,               y_over),
        ("Runline DC (L1)",       dc_val["p_cover_dc"].values, y_cov),
        ("Runline XGB (L1)",      p_cover_xgb,                y_cov),
        ("Runline Stacker (L2)",  p_cover_final,              y_cov),
    ]:
        try:
            auc = roc_auc_score(y, probs)
            ll  = log_loss(y, np.clip(probs, 1e-6, 1 - 1e-6))
            bs  = brier_score_loss(y, probs)
            print(f"    {lbl:<28}  AUC={auc:.4f}  LL={ll:.4f}  Brier={bs:.4f}")
        except Exception as e:
            print(f"    {lbl:<28}  [metric error: {e}]")

    del m_home, m_away, oof_23_24, dc
    gc.collect()


# ---------------------------------------------------------------------------
# FINAL FIT (full data)
# ---------------------------------------------------------------------------
def train_final(df: pd.DataFrame, feat_cols: list[str], with_2026: bool = False):
    years = [2023, 2024, 2025]
    if with_2026:
        years.append(2026)
    final_df = df[df["year"].isin(years)].reset_index(drop=True)
    print(f"\n[4] Final fit on {years} ({len(final_df)} games)")

    # --- Fit DC on ALL years ---
    print("  [4a] Fitting DixonColesMLB on all years …")
    dc = DixonColesMLB(max_runs=K_TRUNC)
    dc.fit(final_df[["home_team", "away_team", "home_runs", "away_runs"]])
    OUT_DC.write_bytes(pickle.dumps(dc))
    print(f"        {len(dc.teams_)} teams  |  saved -> {OUT_DC}")

    # --- Fit XGB Poisson boosters on ALL years ---
    print("  [4b] Fitting Poisson XGBs on all years …")
    X = final_df[feat_cols].fillna(0).values.astype(np.float32)
    sw = _sample_weights(final_df)
    y_h = final_df["home_runs"].astype(float).values
    y_a = final_df["away_runs"].astype(float).values

    # Use last year as eval-set for early stopping
    last_yr = years[-1]
    es_mask = (final_df["year"] == last_yr).values
    X_es = final_df.loc[es_mask, feat_cols].fillna(0).values.astype(np.float32)
    y_h_es = final_df.loc[es_mask, "home_runs"].astype(float).values
    y_a_es = final_df.loc[es_mask, "away_runs"].astype(float).values

    m_home = xgb.XGBRegressor(**XGB_POIS_PARAMS, early_stopping_rounds=50)
    m_home.fit(X, y_h, sample_weight=sw, eval_set=[(X_es, y_h_es)], verbose=False)
    m_away = xgb.XGBRegressor(**XGB_POIS_PARAMS, early_stopping_rounds=50)
    m_away.fit(X, y_a, sample_weight=sw, eval_set=[(X_es, y_a_es)], verbose=False)

    m_home.save_model(str(OUT_XGB_LAM_HOME))
    m_away.save_model(str(OUT_XGB_LAM_AWAY))
    print(f"        -> {OUT_XGB_LAM_HOME}")
    print(f"        -> {OUT_XGB_LAM_AWAY}")

    del m_home, m_away
    gc.collect()

    # --- LOYO OOF across all years to refit stackers ---
    print("  [4c] Generating LOYO OOF across all years for final stackers …")
    oof_full = _generate_oof(df, feat_cols, years=years)
    print(f"        OOF rows: {len(oof_full)}")

    print("  [4d] Refitting TOTALS Bayesian stacker on full OOF …")
    _fit_totals_stacker(oof_full, OUT_STACKER_TOTALS, OUT_STACKER_TOTALS_NPZ)

    print("  [4e] Refitting RUNLINE Bayesian stacker on full OOF …")
    _fit_rl_stacker(oof_full, OUT_STACKER_RL, OUT_STACKER_RL_NPZ)

    print(f"\n  Stackers → {OUT_STACKER_TOTALS}, {OUT_STACKER_RL}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train unified Run Distribution stack")
    parser.add_argument("--with-2026", action="store_true")
    parser.add_argument("--matrix", type=str,
                        default="feature_matrix_enriched_v2.parquet")
    parser.add_argument("--val-year", type=int, default=2025,
                        help="Holdout year for validation (default: 2025)")
    parser.add_argument("--feat-cols", type=str, default=None,
                        help="Path to feature-column manifest JSON. Default: "
                             "run_dist_feature_cols.json (live model). Pass "
                             "models/run_dist_feature_cols_next.json to retrain "
                             "with the v2 env/interaction feature set.")
    args = parser.parse_args()

    print("=" * 70)
    print("Run Distribution Training — DC + Dual-Poisson XGB + Bayesian Stackers")
    print("=" * 70)

    matrix_path = Path(args.matrix)

    # Use persisted manifest if present; otherwise derive
    feat_cols_override = None
    manifest_path = Path(args.feat_cols) if args.feat_cols else OUT_FEAT_COLS
    if manifest_path.exists():
        try:
            feat_cols_override = json.load(open(manifest_path))
            print(f"  Using feature manifest: {manifest_path} "
                  f"({len(feat_cols_override)} features)")
        except Exception:
            feat_cols_override = None

    df, feat_cols, total_line_src = build_dataset(
        matrix_path, include_2026=args.with_2026,
        feat_cols_override=feat_cols_override,
    )

    # Persist the effective feature list
    json.dump(feat_cols, open(OUT_FEAT_COLS, "w"), indent=2)
    print(f"\n  Feature manifest saved -> {OUT_FEAT_COLS} ({len(feat_cols)} cols)")
    print(f"  Total-line source: {total_line_src}")

    # Default split diagnostics
    train_default(df, feat_cols, val_year=args.val_year)

    # Final artifacts
    train_final(df, feat_cols, with_2026=args.with_2026)

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
