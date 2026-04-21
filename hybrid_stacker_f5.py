"""
hybrid_stacker_f5.py
====================
Test whether the distributional challenger signal adds value on top of a
binary F5 classifier (option 3 from the design doc).

Design:
  1. Retrain a binary XGB classifier on 2024 (5-fold OOF) using the same
     feature set as the incumbent. Call this "xgb_binary".
     This is a clean, reproducible reference — we do NOT rely on the
     production incumbent's internal state.
  2. Retrain the dual-Poisson challenger on 2024 (5-fold OOF).
     Call its outputs lam_home, lam_away, p_cover_raw.
  3. Fit a logistic regression on 2024 OOF:
        y_cover ~ logit(p_binary) + logit(p_cover_raw) + lam_home + lam_away
     This is the "hybrid" — binary prob plus distributional features.
  4. On 2025 val: retrain both base models on full 2024, produce val preds,
     apply the LR hybrid. Compare against:
       - xgb_binary alone
       - challenger raw alone
       - hybrid
       - the production incumbent outputs (xgb_cal_f5_cover, stacker_f5_cover)

If the hybrid beats xgb_binary (which holds the "same info but binary
target" position), we have evidence the distributional signal carries
independent information.

Inputs:
  feature_matrix.parquet
  data/statcast/statcast_2024.parquet
  data/statcast/statcast_2025.parquet
  models/f5_challenger_meta.json   (reuse feature list + xgb params)
  f5_val_predictions.csv           (for head-to-head vs production)

Outputs:
  models/f5_hybrid_lr.pkl
  f5_hybrid_eval.txt / .json

Usage:
  python hybrid_stacker_f5.py
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

ROOT             = Path(__file__).parent
MATRIX_PATH      = ROOT / "feature_matrix.parquet"
STATCAST_DIR     = ROOT / "data" / "statcast"
META_PATH        = ROOT / "models" / "f5_challenger_meta.json"
INCUMBENT_PREDS  = ROOT / "f5_val_predictions.csv"

OUT_LR           = ROOT / "models" / "f5_hybrid_lr.pkl"
OUT_TXT          = ROOT / "f5_hybrid_eval.txt"
OUT_JSON         = ROOT / "f5_hybrid_eval.json"
OUT_VAL_PREDS    = ROOT / "f5_hybrid_val_predictions.csv"

N_FOLDS  = 5
MAX_RUNS = 20
SEED     = 42

BINARY_XGB_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    "tree_method":      "hist",
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 5,
    "reg_lambda":       1.0,
}
BINARY_N_ROUNDS = 300  # matches challenger best iters roughly


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def extract_f5_splits(year: int) -> pd.DataFrame:
    sc = pd.read_parquet(
        STATCAST_DIR / f"statcast_{year}.parquet",
        columns=["game_pk", "inning", "post_home_score", "post_away_score"],
    )
    f5 = sc[sc["inning"] <= 5]
    agg = f5.groupby("game_pk", as_index=False).agg(
        f5_home_runs=("post_home_score", "max"),
        f5_away_runs=("post_away_score", "max"),
    )
    agg["f5_home_runs"] = agg["f5_home_runs"].fillna(0).astype(float)
    agg["f5_away_runs"] = agg["f5_away_runs"].fillna(0).astype(float)
    return agg


def prob_home_cover(lam_home: np.ndarray, lam_away: np.ndarray) -> np.ndarray:
    k = np.arange(MAX_RUNS + 1)
    pmf_home = poisson.pmf(k[None, :], mu=lam_home[:, None])
    pmf_away = poisson.pmf(k[None, :], mu=lam_away[:, None])
    cdf_away = np.cumsum(pmf_away, axis=1)
    return (pmf_home * cdf_away).sum(axis=1)


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def scores(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "brier":   float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p)),
        "auc":     float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "mean_p":  float(p.mean()),
        "mean_y":  float(y.mean()),
        "n":       int(len(y)),
    }


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
def load_data():
    with open(META_PATH) as fh:
        meta = json.load(fh)
    feats      = meta["features"]
    poi_params = meta["xgb_params"]
    home_iter  = meta["home_best_iter"] + 1
    away_iter  = meta["away_best_iter"] + 1

    df = pd.read_parquet(MATRIX_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["year"] = df["game_date"].dt.year

    splits = pd.concat([extract_f5_splits(2024), extract_f5_splits(2025)],
                       ignore_index=True)
    df = df.merge(splits, on="game_pk", how="left")
    df["f5_home_cover"] = (df["f5_home_runs"] >= df["f5_away_runs"]).astype("Int64")

    mask_tr = (df["year"] == 2024) & df["f5_home_runs"].notna()
    mask_vl = (df["year"] == 2025) & df["f5_home_runs"].notna()
    print(f"  Train rows: {mask_tr.sum()}   Val rows: {mask_vl.sum()}")

    return df, feats, poi_params, home_iter, away_iter, mask_tr, mask_vl


# ---------------------------------------------------------------------------
# OOF BASE MODELS ON 2024
# ---------------------------------------------------------------------------
def oof_predictions(df, feats, poi_params, home_iter, away_iter, mask_tr):
    sub = df.loc[mask_tr].reset_index(drop=True)
    X   = sub[feats].astype(float).values
    y_c = sub["f5_home_cover"].astype(int).values
    y_h = sub["f5_home_runs"].values.astype(float)
    y_a = sub["f5_away_runs"].values.astype(float)

    oof_bin  = np.zeros(len(sub))
    oof_lamh = np.zeros(len(sub))
    oof_lama = np.zeros(len(sub))

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr, vl) in enumerate(kf.split(X), start=1):
        # Binary classifier
        dtr = xgb.DMatrix(X[tr], label=y_c[tr])
        dvl = xgb.DMatrix(X[vl])
        bst = xgb.train(BINARY_XGB_PARAMS, dtr, num_boost_round=BINARY_N_ROUNDS,
                        verbose_eval=False)
        oof_bin[vl] = bst.predict(dvl)

        # Dual Poisson
        dtr_h = xgb.DMatrix(X[tr], label=y_h[tr])
        dtr_a = xgb.DMatrix(X[tr], label=y_a[tr])
        bst_h = xgb.train(poi_params, dtr_h, num_boost_round=home_iter,
                          verbose_eval=False)
        bst_a = xgb.train(poi_params, dtr_a, num_boost_round=away_iter,
                          verbose_eval=False)
        oof_lamh[vl] = bst_h.predict(dvl)
        oof_lama[vl] = bst_a.predict(dvl)
        print(f"  Fold {fold}/{N_FOLDS} done")

    oof_p_poi = prob_home_cover(oof_lamh, oof_lama)
    return sub, y_c, oof_bin, oof_lamh, oof_lama, oof_p_poi


# ---------------------------------------------------------------------------
# VAL BASE MODELS (retrain on full 2024)
# ---------------------------------------------------------------------------
def val_predictions(df, feats, poi_params, home_iter, away_iter, mask_tr, mask_vl):
    tr = df.loc[mask_tr]
    vl = df.loc[mask_vl].reset_index(drop=True)

    X_tr = tr[feats].astype(float).values
    X_vl = vl[feats].astype(float).values

    dtr_c = xgb.DMatrix(X_tr, label=tr["f5_home_cover"].astype(int).values)
    dvl   = xgb.DMatrix(X_vl)
    bst_c = xgb.train(BINARY_XGB_PARAMS, dtr_c, num_boost_round=BINARY_N_ROUNDS,
                      verbose_eval=False)
    p_bin = bst_c.predict(dvl)

    dtr_h = xgb.DMatrix(X_tr, label=tr["f5_home_runs"].astype(float).values)
    dtr_a = xgb.DMatrix(X_tr, label=tr["f5_away_runs"].astype(float).values)
    bst_h = xgb.train(poi_params, dtr_h, num_boost_round=home_iter, verbose_eval=False)
    bst_a = xgb.train(poi_params, dtr_a, num_boost_round=away_iter, verbose_eval=False)
    lamh  = bst_h.predict(dvl)
    lama  = bst_a.predict(dvl)
    p_poi = prob_home_cover(lamh, lama)

    return vl, p_bin, lamh, lama, p_poi


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def build_stacker_X(p_bin, p_poi, lamh, lama):
    """Feature matrix for the hybrid LR: logits + lambdas."""
    return np.column_stack([
        _logit(p_bin),
        _logit(p_poi),
        lamh,
        lama,
    ])


def main():
    print("Loading data...")
    df, feats, poi_params, home_iter, away_iter, mask_tr, mask_vl = load_data()

    print("\nGenerating 2024 OOF predictions (binary XGB + dual Poisson)...")
    sub, y_tr, oof_bin, oof_lamh, oof_lama, oof_poi = oof_predictions(
        df, feats, poi_params, home_iter, away_iter, mask_tr,
    )

    # OOF Brier reference
    oof_scores = {
        "xgb_binary_oof": scores(oof_bin, y_tr),
        "poisson_oof":    scores(oof_poi, y_tr),
    }
    print("\n  OOF Brier (2024):")
    for k, v in oof_scores.items():
        print(f"    {k:<20} brier={v['brier']:.4f}  logloss={v['logloss']:.4f}  auc={v['auc']:.4f}")

    print("\nFitting hybrid LR on 2024 OOF...")
    X_oof = build_stacker_X(oof_bin, oof_poi, oof_lamh, oof_lama)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(X_oof, y_tr)
    oof_hybrid = lr.predict_proba(X_oof)[:, 1]
    print(f"  LR coefs: p_bin_logit={lr.coef_[0,0]:+.3f}  p_poi_logit={lr.coef_[0,1]:+.3f}  "
          f"lam_home={lr.coef_[0,2]:+.3f}  lam_away={lr.coef_[0,3]:+.3f}  "
          f"intercept={lr.intercept_[0]:+.3f}")
    print(f"  Hybrid OOF Brier: {brier_score_loss(y_tr, oof_hybrid):.4f}")

    with open(OUT_LR, "wb") as fh:
        pickle.dump(lr, fh)
    print(f"  Saved hybrid LR -> {OUT_LR}")

    # Also fit a simpler "logits-only" variant for diagnostic
    lr_simple = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    X_oof_simple = X_oof[:, :2]  # just two logits
    lr_simple.fit(X_oof_simple, y_tr)
    print(f"  [Simple LR (logits only)] coefs={lr_simple.coef_[0]}  "
          f"intercept={lr_simple.intercept_[0]:+.3f}")

    print("\nRetraining base models on full 2024 -> scoring 2025 val...")
    vl, p_bin_v, lamh_v, lama_v, p_poi_v = val_predictions(
        df, feats, poi_params, home_iter, away_iter, mask_tr, mask_vl,
    )
    X_val_full   = build_stacker_X(p_bin_v, p_poi_v, lamh_v, lama_v)
    X_val_simple = X_val_full[:, :2]
    p_hybrid        = lr.predict_proba(X_val_full)[:, 1]
    p_hybrid_simple = lr_simple.predict_proba(X_val_simple)[:, 1]

    y_val = vl["f5_home_cover"].astype(int).values

    # Align with incumbent
    inc = pd.read_csv(INCUMBENT_PREDS)
    vl_out = vl[["game_pk", "game_date", "home_team", "away_team",
                 "f5_home_runs", "f5_away_runs", "f5_home_cover"]].copy()
    vl_out["p_xgb_binary"]   = p_bin_v
    vl_out["lam_home"]       = lamh_v
    vl_out["lam_away"]       = lama_v
    vl_out["p_cover_raw"]    = p_poi_v
    vl_out["p_hybrid"]       = p_hybrid
    vl_out["p_hybrid_simple"] = p_hybrid_simple
    merged = inc[["game_pk", "xgb_cal_f5_cover", "stacker_f5_cover"]].merge(
        vl_out, on="game_pk", how="inner",
    )
    merged.to_csv(OUT_VAL_PREDS, index=False)
    print(f"  Joined val rows: {len(merged)}")

    # Report
    report = []
    metrics = {}

    def log(s: str = ""):
        print(s)
        report.append(s)

    log("=" * 70)
    log("F5 HYBRID STACKER - 2025 holdout")
    log("=" * 70)
    log(f"Val rows (joined):   {len(merged)}")
    log(f"Base rate f5_cover:  {merged['f5_home_cover'].mean():.4f}")
    log("")

    y = merged["f5_home_cover"].astype(int).values
    models = [
        ("production_incumbent_xgb_cal",   merged["xgb_cal_f5_cover"].values),
        ("production_incumbent_stacker",   merged["stacker_f5_cover"].values),
        ("retrained_xgb_binary",           merged["p_xgb_binary"].values),
        ("retrained_poisson_raw",          merged["p_cover_raw"].values),
        ("hybrid_simple (bin+poi logits)", merged["p_hybrid_simple"].values),
        ("hybrid_full (logits+lambdas)",   merged["p_hybrid"].values),
    ]
    log(f"{'Model':<36}{'Brier':>10}{'LogLoss':>10}{'AUC':>8}{'MeanP':>9}")
    for name, p in models:
        s = scores(p, y)
        metrics[name] = s
        log(f"{name:<36}{s['brier']:>10.4f}{s['logloss']:>10.4f}"
            f"{s['auc']:>8.4f}{s['mean_p']:>9.4f}")

    log("")
    log("-" * 70)
    log("INTERPRETATION")
    log("-" * 70)
    b_bin   = metrics["retrained_xgb_binary"]["brier"]
    b_poi   = metrics["retrained_poisson_raw"]["brier"]
    b_hyb_s = metrics["hybrid_simple (bin+poi logits)"]["brier"]
    b_hyb_f = metrics["hybrid_full (logits+lambdas)"]["brier"]
    log(f"  Same features, same 2024 training data, three targets:")
    log(f"    binary classifier:  Brier={b_bin:.4f}")
    log(f"    dual Poisson:       Brier={b_poi:.4f}")
    log(f"    hybrid (binary+poi logits):            {b_hyb_s:.4f}  "
        f"(delta vs binary {b_hyb_s - b_bin:+.4f})")
    log(f"    hybrid (binary+poi logits + lambdas):  {b_hyb_f:.4f}  "
        f"(delta vs binary {b_hyb_f - b_bin:+.4f})")
    log("")
    log("  If hybrid Brier < binary Brier, the distributional signal adds")
    log("  independent information beyond what the binary classifier captured.")

    with open(OUT_TXT, "w") as fh:
        fh.write("\n".join(report) + "\n")
    with open(OUT_JSON, "w") as fh:
        json.dump({"oof": oof_scores, "val": metrics,
                   "lr_coef": lr.coef_.tolist(),
                   "lr_intercept": lr.intercept_.tolist(),
                   "lr_simple_coef": lr_simple.coef_.tolist(),
                   "lr_simple_intercept": lr_simple.intercept_.tolist()}, fh, indent=2)
    log(f"\nReport -> {OUT_TXT}")
    log(f"Metrics -> {OUT_JSON}")


if __name__ == "__main__":
    main()
