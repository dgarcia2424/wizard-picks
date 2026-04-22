"""
diagnose_nrfi_calibration.py
----------------------------
Calibration diagnostic for the NRFI stacker.

MODE: Re-runs LOYO via `_generate_oof_for_stacker` from `train_nrfi_model.py`
to produce genuine OOF L1 raw probs + OOF feature DataFrame (with
team_nrfi_log_odds and Poisson sidecar features). Those OOF probs are then
fed through:
  - the saved Platt calibrator (`models/xgb_nrfi_calibrator.pkl`)  -> L1 probs
  - the saved BayesianStackerNRFI (`models/stacking_lr_nrfi.pkl`)   -> L2 probs

Note: the stacker is trained ON these OOF rows, so L2 reliability here is
slightly optimistic (in-sample for the stacker fit). This is still the most
honest look available without a full nested-CV refit, since the L1 stage is
fully held out via LOYO.

Outputs:
  - text reliability tables for L1 (Platt) and L2 (stacker)
  - Brier + ECE per stage
  - a calibration verdict
  - nrfi_calibration_buckets.csv
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

REPO = Path(r"C:\Users\garci\OneDrive\Documents\Claude\mlb_model_pipeline_dan")
sys.path.insert(0, str(REPO))

from train_nrfi_model import (  # noqa: E402
    build_dataset,
    _generate_oof_for_stacker,
    BayesianStackerNRFI,  # needed for pickle unpickling
)

MODELS   = REPO / "models"
CAL_PKL  = MODELS / "xgb_nrfi_calibrator.pkl"
STK_PKL  = MODELS / "stacking_lr_nrfi.pkl"
OUT_CSV  = REPO / "nrfi_calibration_buckets.csv"

BINS = np.linspace(0.0, 1.0, 11)  # 10 equal-width buckets


def reliability_table(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    # digitize: assign bucket 0..9; rightmost edge inclusive in bucket 9
    idx = np.clip(np.digitize(y_prob, BINS, right=False) - 1, 0, 9)
    rows = []
    for b in range(10):
        lo, hi = BINS[b], BINS[b + 1]
        mask = (idx == b)
        n = int(mask.sum())
        if n == 0:
            rows.append({
                "bucket": f"[{lo:.1f},{hi:.1f})",
                "n": 0, "predicted": np.nan, "actual": np.nan, "diff": np.nan,
            })
            continue
        pred = float(y_prob[mask].mean())
        actual = float(y_true[mask].mean())
        rows.append({
            "bucket": f"[{lo:.1f},{hi:.1f})",
            "n": n,
            "predicted": pred,
            "actual": actual,
            "diff": actual - pred,
        })
    return pd.DataFrame(rows)


def ece(df: pd.DataFrame) -> float:
    pop = df.dropna(subset=["predicted", "actual"])
    if pop["n"].sum() == 0:
        return float("nan")
    w = pop["n"] / pop["n"].sum()
    return float((w * (pop["actual"] - pop["predicted"]).abs()).sum())


def print_table(name: str, df: pd.DataFrame, brier: float, ece_val: float) -> None:
    print(f"\n=== {name} ===")
    print(f"{'bucket':<14}{'n':>7}{'predicted':>12}{'actual':>10}{'diff':>10}")
    print("-" * 53)
    for _, r in df.iterrows():
        if r["n"] == 0:
            print(f"{r['bucket']:<14}{0:>7}{'—':>12}{'—':>10}{'—':>10}")
        else:
            print(f"{r['bucket']:<14}{int(r['n']):>7}"
                  f"{r['predicted']:>12.4f}{r['actual']:>10.4f}"
                  f"{r['diff']:>+10.4f}")
    print(f"Brier: {brier:.5f}   ECE (n-weighted): {ece_val:.5f}")


def verdict(name: str, df: pd.DataFrame) -> list[str]:
    msgs = []
    for _, r in df.iterrows():
        if r["n"] >= 50 and not np.isnan(r["diff"]) and abs(r["diff"]) > 0.05:
            msgs.append(
                f"[WARN] {name} MISCALIBRATED in bucket {r['bucket']}: "
                f"model says {r['predicted']*100:.1f}%, "
                f"reality is {r['actual']*100:.1f}% "
                f"(n={int(r['n'])}, diff={r['diff']*100:+.1f} pp)"
            )
    if not msgs:
        msgs.append(f"[OK] {name} well-calibrated across populated buckets.")
    return msgs


def main() -> None:
    print(f"[1/5] Loading dataset via build_dataset()…")
    df, feat_cols = build_dataset(include_2026=False)
    print(f"      rows={len(df):,}  feats={len(feat_cols)}")

    print(f"\n[2/5] Running LOYO OOF (this takes a few minutes)…")
    oof_raw, oof_lbl, oof_df, oof_segs = _generate_oof_for_stacker(df, feat_cols)
    print(f"      OOF rows={len(oof_raw):,}")

    print(f"\n[3/5] Loading saved Platt calibrator and Bayesian stacker…")
    cal = pickle.loads(CAL_PKL.read_bytes())
    stk = pickle.loads(STK_PKL.read_bytes())

    p_l1 = cal.predict_proba(oof_raw.reshape(-1, 1))[:, 1]
    p_l2 = stk.predict(oof_raw, oof_df, oof_segs)

    print(f"      L1 range: [{p_l1.min():.3f}, {p_l1.max():.3f}]  mean={p_l1.mean():.3f}")
    print(f"      L2 range: [{p_l2.min():.3f}, {p_l2.max():.3f}]  mean={p_l2.mean():.3f}")
    print(f"      base rate (actual NRFI): {oof_lbl.mean():.4f}")

    print(f"\n[4/5] Building reliability tables…")
    tbl_l1 = reliability_table(oof_lbl, p_l1)
    tbl_l2 = reliability_table(oof_lbl, p_l2)

    brier_l1 = brier_score_loss(oof_lbl, p_l1)
    brier_l2 = brier_score_loss(oof_lbl, p_l2)
    ece_l1 = ece(tbl_l1)
    ece_l2 = ece(tbl_l2)

    print_table("L1 — XGBoost + Platt", tbl_l1, brier_l1, ece_l1)
    print_table("L2 — Bayesian Stacker", tbl_l2, brier_l2, ece_l2)

    print("\n=== VERDICT ===")
    for m in verdict("L1 Platt", tbl_l1):
        print(m)
    for m in verdict("L2 stacker", tbl_l2):
        print(m)

    print(f"\n[5/5] Saving per-bucket CSV…")
    tbl_l1_out = tbl_l1.copy(); tbl_l1_out.insert(0, "stage", "L1_platt")
    tbl_l2_out = tbl_l2.copy(); tbl_l2_out.insert(0, "stage", "L2_stacker")
    out = pd.concat([tbl_l1_out, tbl_l2_out], ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"      -> {OUT_CSV}")

    print("\n=== SUMMARY ===")
    print(f"  OOF n              = {len(oof_lbl):,}")
    print(f"  base rate          = {oof_lbl.mean():.4f}")
    print(f"  L1 Brier / ECE     = {brier_l1:.5f} / {ece_l1:.5f}")
    print(f"  L2 Brier / ECE     = {brier_l2:.5f} / {ece_l2:.5f}")
    print(f"  L1 prob spread     = [{p_l1.min():.3f}, {p_l1.max():.3f}]")
    print(f"  L2 prob spread     = [{p_l2.min():.3f}, {p_l2.max():.3f}]")


if __name__ == "__main__":
    main()
