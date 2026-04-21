"""
eval_f5_challenger.py
=====================
Head-to-head evaluation of the dual-Poisson F5 challenger against the
incumbent F5 classifier.

The challenger produces lambda_home, lambda_away. We derive the cover
probability by convolving two independent Poisson distributions:

    P(home_cover) = P(home_runs >= away_runs)
                  = sum_{h >= a} Poisson(h; lam_home) * Poisson(a; lam_away)

(f5_home_cover == 1 iff home wins or ties, so the boundary uses >=.)

The incumbent predictions are read from f5_val_predictions.csv. We compare
three probabilities on the same 2025 holdout:
  - xgb_cal_f5_cover   (raw incumbent XGB + Platt calibration)
  - stacker_f5_cover   (incumbent XGB + team log-odds stacker)
  - challenger         (dual Poisson convolution)

Metrics: Brier score, log-loss, AUC, calibration bins, push rate.

Also reports a diagnostic comparison on the totals side — converting the
predicted lambda sum into P(total > K) for K in {8, 9} — since the
distributional approach gets totals "for free".

Inputs:
  f5_val_predictions.csv
  f5_challenger_val_predictions.csv

Output:
  f5_challenger_eval.txt     human-readable report
  f5_challenger_eval.json    metrics in machine-readable form

Usage:
  python eval_f5_challenger.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT             = Path(__file__).parent
INCUMBENT_PREDS  = ROOT / "f5_val_predictions.csv"
CHALLENGER_PREDS = ROOT / "f5_challenger_val_predictions.csv"
OUT_TXT          = ROOT / "f5_challenger_eval.txt"
OUT_JSON         = ROOT / "f5_challenger_eval.json"

MAX_RUNS = 20  # upper bound for convolution; P(Poisson(~3) > 20) is negligible


def prob_home_cover(lam_home: np.ndarray, lam_away: np.ndarray) -> np.ndarray:
    """P(home_runs >= away_runs) under independent Poisson(lam_home) + Poisson(lam_away)."""
    k = np.arange(MAX_RUNS + 1)
    # shape: (n_games, MAX_RUNS+1)
    pmf_home = poisson.pmf(k[None, :], mu=lam_home[:, None])
    pmf_away = poisson.pmf(k[None, :], mu=lam_away[:, None])
    cdf_away = np.cumsum(pmf_away, axis=1)          # P(A <= k)
    # P(A <= h) for each h -> sum_h P(H=h) * P(A <= h) = P(A <= H) = P(H >= A)
    return (pmf_home * cdf_away).sum(axis=1)


def prob_total_over(lam_home: np.ndarray, lam_away: np.ndarray, line: float) -> np.ndarray:
    """P(home + away > line) under Poisson(lam_home + lam_away) since sum of Poissons is Poisson."""
    mu = lam_home + lam_away
    # P(T > line) = 1 - P(T <= floor(line)); if line is an integer, standard over/under
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
        if not m.any():
            rows.append({"bin": lbl, "n": 0, "pred": np.nan, "actual": np.nan})
        else:
            rows.append({
                "bin":    lbl,
                "n":      int(m.sum()),
                "pred":   float(p[m].mean()),
                "actual": float(y[m].mean()),
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


def main():
    inc = pd.read_csv(INCUMBENT_PREDS)
    ch  = pd.read_csv(CHALLENGER_PREDS)

    merged = inc.merge(
        ch[["game_pk", "lam_home", "lam_away", "pred_f5_total"]],
        on="game_pk", how="inner",
    )
    print(f"Merged rows: {len(merged)}  (incumbent={len(inc)}, challenger={len(ch)})")

    y_cover = merged["f5_home_cover"].astype(int).values
    y_total = (merged["f5_home_runs"] + merged["f5_away_runs"]).values

    lam_h = merged["lam_home"].values
    lam_a = merged["lam_away"].values
    p_challenger = prob_home_cover(lam_h, lam_a)

    report_lines = []
    metrics = {}

    def log(line: str = ""):
        print(line)
        report_lines.append(line)

    log("=" * 70)
    log("F5 CHALLENGER vs INCUMBENT - 2025 holdout")
    log("=" * 70)
    log(f"Val rows (joined):   {len(merged)}")
    log(f"Base rate f5_cover:  {y_cover.mean():.4f}")
    log(f"Actual F5 total mean: {y_total.mean():.3f}   "
        f"Challenger pred mean: {merged['pred_f5_total'].mean():.3f}")
    log("")

    log("-" * 70)
    log("COVER-PROBABILITY METRICS  (target = f5_home_cover)")
    log("-" * 70)
    log(f"{'Model':<28}{'Brier':>10}{'LogLoss':>10}{'AUC':>8}{'MeanP':>9}")
    models = [
        ("incumbent_xgb_cal", merged["xgb_cal_f5_cover"].values),
        ("incumbent_stacker", merged["stacker_f5_cover"].values),
        ("challenger_poisson", p_challenger),
    ]
    for name, p in models:
        s = scores(p, y_cover)
        metrics[name] = s
        log(f"{name:<28}{s['brier']:>10.4f}{s['logloss']:>10.4f}"
            f"{s['auc']:>8.4f}{s['mean_p']:>9.4f}")
    log(f"{'base_rate':<28}{np.var(y_cover):>10.4f}"
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

    # Totals side-benefit
    log("-" * 70)
    log("TOTALS DIAGNOSTIC  (challenger derives P(total > K) for free)")
    log("-" * 70)
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
    inc_best_brier = min(metrics["incumbent_xgb_cal"]["brier"],
                         metrics["incumbent_stacker"]["brier"])
    ch_brier = metrics["challenger_poisson"]["brier"]
    delta = ch_brier - inc_best_brier
    log("-" * 70)
    log(f"VERDICT  (lower = better)")
    log("-" * 70)
    log(f"  Best incumbent Brier: {inc_best_brier:.4f}")
    log(f"  Challenger Brier:     {ch_brier:.4f}")
    log(f"  Delta:                {delta:+.4f}  "
        f"({'challenger wins' if delta < 0 else 'incumbent wins'})")
    log("")
    log("  NOTE: Challenger is raw — no calibration layer, no stacking.")
    log("        If it is close or better here, wiring it into the stack would likely improve further.")

    with open(OUT_TXT, "w") as fh:
        fh.write("\n".join(report_lines) + "\n")
    with open(OUT_JSON, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nReport -> {OUT_TXT}")
    print(f"Metrics -> {OUT_JSON}")


if __name__ == "__main__":
    main()
