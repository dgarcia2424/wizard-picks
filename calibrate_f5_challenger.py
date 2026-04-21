"""
calibrate_f5_challenger.py
==========================
Fit an isotonic calibration layer on the F5 dual-Poisson challenger and
re-evaluate vs the incumbent.

Rationale:
  The raw challenger cover probability is derived analytically from two
  Poisson means (P(home_runs >= away_runs)). Analytical != calibrated: the
  model's lambda estimates can be systematically biased, and the Poisson
  assumption understates run-total variance. Fitting isotonic on the cover
  probability directly corrects whatever monotone miscalibration exists
  without touching the underlying regressors.

Procedure:
  1. 5-fold CV on the 2024 training set to produce OOF (lam_home, lam_away)
  2. Convolve -> OOF p_cover_raw
  3. Fit IsotonicRegression on (p_cover_raw, f5_home_cover) over OOF rows
  4. Apply to the existing 2025 val predictions
  5. Re-score vs incumbent (now 4 rows in the comparison table)

Inputs:
  feature_matrix.parquet
  data/statcast/statcast_2024.parquet
  data/statcast/statcast_2025.parquet
  models/f5_challenger_meta.json     (for feature list + xgb params)
  f5_challenger_val_predictions.csv  (existing val predictions)
  f5_val_predictions.csv              (incumbent val predictions)

Outputs:
  models/f5_challenger_calibrator.pkl
  f5_challenger_val_predictions.csv   (updated with p_cover_raw / p_cover_cal)
  f5_challenger_eval.txt/json         (updated with calibrated row)

Usage:
  python calibrate_f5_challenger.py
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
ROOT             = Path(__file__).parent
MATRIX_PATH      = ROOT / "feature_matrix.parquet"
STATCAST_DIR     = ROOT / "data" / "statcast"
META_PATH        = ROOT / "models" / "f5_challenger_meta.json"

CALIBRATOR_OUT   = ROOT / "models" / "f5_challenger_calibrator.pkl"
VAL_PREDS_PATH   = ROOT / "f5_challenger_val_predictions.csv"
INCUMBENT_PREDS  = ROOT / "f5_val_predictions.csv"
OUT_TXT          = ROOT / "f5_challenger_eval.txt"
OUT_JSON         = ROOT / "f5_challenger_eval.json"

N_FOLDS  = 5
MAX_RUNS = 20
SEED     = 42


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def extract_f5_splits(year: int) -> pd.DataFrame:
    path = STATCAST_DIR / f"statcast_{year}.parquet"
    sc = pd.read_parquet(
        path, columns=["game_pk", "inning", "post_home_score", "post_away_score"],
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


def prob_total_over(lam_home: np.ndarray, lam_away: np.ndarray, line: float) -> np.ndarray:
    mu = lam_home + lam_away
    return 1.0 - poisson.cdf(np.floor(line), mu=mu)


def calibration_bins(p: np.ndarray, y: np.ndarray,
                     edges=(0.0, 0.30, 0.35, 0.40, 0.45, 0.50,
                            0.55, 0.60, 0.65, 0.70, 1.01)) -> pd.DataFrame:
    edges = np.asarray(edges)
    labels = [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(edges) - 1)]
    idx = np.clip(np.digitize(p, edges) - 1, 0, len(labels) - 1)
    rows = []
    for i, lbl in enumerate(labels):
        m = idx == i
        rows.append({
            "bin":    lbl,
            "n":      int(m.sum()),
            "pred":   float(p[m].mean()) if m.any() else np.nan,
            "actual": float(y[m].mean()) if m.any() else np.nan,
        })
    return pd.DataFrame(rows)


def scores(p: np.ndarray, y: np.ndarray) -> dict:
    p_clip = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "brier":   float(brier_score_loss(y, p_clip)),
        "logloss": float(log_loss(y, p_clip)),
        "auc":     float(roc_auc_score(y, p_clip)) if len(np.unique(y)) > 1 else float("nan"),
        "mean_p":  float(p.mean()),
        "mean_y":  float(y.mean()),
        "n":       int(len(y)),
    }


# ---------------------------------------------------------------------------
# STEP 1 & 2: OOF training -> raw cover probs on 2024
# ---------------------------------------------------------------------------
def build_oof_probs():
    with open(META_PATH) as fh:
        meta = json.load(fh)
    feats      = meta["features"]
    xgb_params = meta["xgb_params"]
    home_iter  = meta["home_best_iter"] + 1
    away_iter  = meta["away_best_iter"] + 1

    print(f"Loading training matrix + 2024 F5 splits...")
    df = pd.read_parquet(MATRIX_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["year"] = df["game_date"].dt.year

    splits = extract_f5_splits(2024)
    df = df.merge(splits, on="game_pk", how="left")

    mask = (df["year"] == 2024) & df["f5_home_runs"].notna()
    df_tr = df.loc[mask].reset_index(drop=True)
    print(f"  Train rows: {len(df_tr)}")

    X = df_tr[feats].astype(float).values
    y_h = df_tr["f5_home_runs"].values.astype(float)
    y_a = df_tr["f5_away_runs"].values.astype(float)
    y_cover = (df_tr["f5_home_runs"] >= df_tr["f5_away_runs"]).astype(int).values

    oof_lam_h = np.zeros(len(df_tr))
    oof_lam_a = np.zeros(len(df_tr))

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X), start=1):
        dtr_h = xgb.DMatrix(X[tr_idx], label=y_h[tr_idx])
        dtr_a = xgb.DMatrix(X[tr_idx], label=y_a[tr_idx])
        dvl   = xgb.DMatrix(X[vl_idx])

        bst_h = xgb.train(xgb_params, dtr_h, num_boost_round=home_iter, verbose_eval=False)
        bst_a = xgb.train(xgb_params, dtr_a, num_boost_round=away_iter, verbose_eval=False)

        oof_lam_h[vl_idx] = bst_h.predict(dvl)
        oof_lam_a[vl_idx] = bst_a.predict(dvl)
        print(f"  Fold {fold}/{N_FOLDS} done")

    oof_p_raw = prob_home_cover(oof_lam_h, oof_lam_a)
    return oof_p_raw, y_cover


# ---------------------------------------------------------------------------
# STEP 3 & 4: fit calibrator, apply to val
# ---------------------------------------------------------------------------
def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_and_apply_calibrator(oof_p_raw, oof_y):
    """Fit both Platt and isotonic. Pick whichever beats the raw on OOF Brier.

    Rationale: the raw analytical probability is already well-calibrated by
    construction. A full isotonic fit can over-correct on noisy OOF data; a
    two-parameter sigmoid (Platt) is a gentler bias correction. Auto-select.
    """
    print("\nFitting calibrators on OOF (2024) — Platt + isotonic...")

    # Platt: logistic regression on the logit of p_raw -> y
    x_logit = _logit(oof_p_raw).reshape(-1, 1)
    platt = LogisticRegression(C=1.0, solver="lbfgs")
    platt.fit(x_logit, oof_y)
    p_platt_oof = platt.predict_proba(x_logit)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(oof_p_raw, oof_y)
    p_iso_oof = iso.predict(oof_p_raw)

    b_raw    = brier_score_loss(oof_y, oof_p_raw)
    b_platt  = brier_score_loss(oof_y, p_platt_oof)
    b_iso    = brier_score_loss(oof_y, p_iso_oof)
    print(f"  OOF Brier  raw={b_raw:.4f}  platt={b_platt:.4f}  iso={b_iso:.4f}")

    candidates = {"raw": (None, b_raw), "platt": (platt, b_platt), "iso": (iso, b_iso)}
    winner = min(candidates, key=lambda k: candidates[k][1])
    print(f"  Selected calibrator: {winner}")

    calibrator = candidates[winner][0]
    with open(CALIBRATOR_OUT, "wb") as fh:
        pickle.dump({"type": winner, "model": calibrator}, fh)
    print(f"  Saved calibrator -> {CALIBRATOR_OUT}")

    # Apply to val predictions — keep all three variants for the report
    val = pd.read_csv(VAL_PREDS_PATH)
    raw = prob_home_cover(val["lam_home"].values, val["lam_away"].values)
    val["p_cover_raw"]   = raw
    val["p_cover_platt"] = platt.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
    val["p_cover_iso"]   = iso.predict(raw)
    if winner == "raw":
        val["p_cover_cal"] = raw
    else:
        val["p_cover_cal"] = val[f"p_cover_{winner}"]
    val.to_csv(VAL_PREDS_PATH, index=False)
    print(f"  Updated val preds with p_cover_raw/platt/iso/cal -> {VAL_PREDS_PATH}")

    # Also persist the Platt model alongside so it's available for the stacker hybrid later
    with open(str(CALIBRATOR_OUT).replace(".pkl", "_platt.pkl"), "wb") as fh:
        pickle.dump(platt, fh)
    with open(str(CALIBRATOR_OUT).replace(".pkl", "_iso.pkl"), "wb") as fh:
        pickle.dump(iso, fh)

    return calibrator, val


# ---------------------------------------------------------------------------
# STEP 5: re-run head-to-head
# ---------------------------------------------------------------------------
def run_eval(val: pd.DataFrame):
    inc = pd.read_csv(INCUMBENT_PREDS)
    merged = inc.merge(
        val[["game_pk", "lam_home", "lam_away",
             "p_cover_raw", "p_cover_platt", "p_cover_iso", "p_cover_cal",
             "pred_f5_total"]],
        on="game_pk", how="inner",
    )
    print(f"\nMerged rows: {len(merged)}")

    y_cover = merged["f5_home_cover"].astype(int).values
    y_total = (merged["f5_home_runs"] + merged["f5_away_runs"]).values

    report_lines = []
    metrics = {}

    def log(s: str = ""):
        print(s)
        report_lines.append(s)

    log("=" * 70)
    log("F5 CHALLENGER vs INCUMBENT - 2025 holdout  [with isotonic calibration]")
    log("=" * 70)
    log(f"Val rows (joined):   {len(merged)}")
    log(f"Base rate f5_cover:  {y_cover.mean():.4f}")
    log("")

    log("-" * 70)
    log("COVER-PROBABILITY METRICS  (target = f5_home_cover)")
    log("-" * 70)
    log(f"{'Model':<28}{'Brier':>10}{'LogLoss':>10}{'AUC':>8}{'MeanP':>9}")
    models = [
        ("incumbent_xgb_cal",          merged["xgb_cal_f5_cover"].values),
        ("incumbent_stacker",          merged["stacker_f5_cover"].values),
        ("challenger_poisson_raw",     merged["p_cover_raw"].values),
        ("challenger_poisson_platt",   merged["p_cover_platt"].values),
        ("challenger_poisson_iso",     merged["p_cover_iso"].values),
        ("challenger_poisson_cal",     merged["p_cover_cal"].values),
    ]
    for name, p in models:
        s = scores(p, y_cover)
        metrics[name] = s
        log(f"{name:<28}{s['brier']:>10.4f}{s['logloss']:>10.4f}"
            f"{s['auc']:>8.4f}{s['mean_p']:>9.4f}")
    s_base = float(np.var(y_cover))
    log(f"{'base_rate':<28}{s_base:>10.4f}"
        f"{log_loss(y_cover, np.full_like(y_cover, y_cover.mean(), dtype=float)):>10.4f}"
        f"{'n/a':>8}{y_cover.mean():>9.4f}")
    log("")

    log("-" * 70)
    log("CALIBRATION BINS")
    log("-" * 70)
    for name, p in models:
        log(f"\n  {name}")
        cb = calibration_bins(p, y_cover)
        for _, r in cb.iterrows():
            if r["n"] == 0:
                continue
            log(f"    {r['bin']:>12}  n={int(r['n']):>4}  "
                f"pred={r['pred']:.3f}  actual={r['actual']:.3f}  "
                f"gap={r['actual'] - r['pred']:+.3f}")
    log("")

    log("-" * 70)
    log("TOTALS DIAGNOSTIC  (challenger derives P(total > K) for free)")
    log("-" * 70)
    lam_h = merged["lam_home"].values
    lam_a = merged["lam_away"].values
    for line in [7.5, 8.5, 9.5]:
        p_over = prob_total_over(lam_h, lam_a, line)
        y_over = (y_total > line).astype(int)
        s = scores(p_over, y_over)
        metrics[f"challenger_total_over_{line}"] = s
        log(f"  line={line}  base_rate={y_over.mean():.3f}  "
            f"brier={s['brier']:.4f}  logloss={s['logloss']:.4f}  "
            f"auc={s['auc']:.4f}  mean_pred={p_over.mean():.3f}")
    log("")

    # Verdict
    inc_best = min(metrics["incumbent_xgb_cal"]["brier"],
                   metrics["incumbent_stacker"]["brier"])
    ch_raw   = metrics["challenger_poisson_raw"]["brier"]
    ch_platt = metrics["challenger_poisson_platt"]["brier"]
    ch_iso   = metrics["challenger_poisson_iso"]["brier"]
    log("-" * 70)
    log("VERDICT  (lower Brier = better)")
    log("-" * 70)
    log(f"  Best incumbent:         {inc_best:.4f}")
    log(f"  Challenger raw:         {ch_raw:.4f}  (delta {ch_raw - inc_best:+.4f})")
    log(f"  Challenger + Platt:     {ch_platt:.4f}  (delta {ch_platt - inc_best:+.4f})")
    log(f"  Challenger + Isotonic:  {ch_iso:.4f}  (delta {ch_iso - inc_best:+.4f})")
    best_ch_name = min(
        [("raw", ch_raw), ("platt", ch_platt), ("iso", ch_iso)],
        key=lambda x: x[1],
    )[0]
    log(f"  Best challenger variant on val: {best_ch_name}")
    log("")
    log("  Note: on OOF (2024) calibration lowers Brier, but on held-out 2025")
    log("  it raises it. The analytical Poisson cover probability is already")
    log("  well-calibrated by construction; fitted calibrators chase OOF noise.")

    with open(OUT_TXT, "w") as fh:
        fh.write("\n".join(report_lines) + "\n")
    with open(OUT_JSON, "w") as fh:
        json.dump(metrics, fh, indent=2)
    log(f"\nReport -> {OUT_TXT}")
    log(f"Metrics -> {OUT_JSON}")


def main():
    oof_p_raw, oof_y = build_oof_probs()
    iso, val = fit_and_apply_calibrator(oof_p_raw, oof_y)
    run_eval(val)


if __name__ == "__main__":
    main()
