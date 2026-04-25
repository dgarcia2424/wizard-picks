"""
decile_calibration_audit.py  --  Section 3.4: Calibration Diagnostic
======================================================================
Evaluates stacker L2 output probabilities against actual outcomes from
xgb_val_predictions.csv.

Targets evaluated:
  - rl_prob  vs  home_covers_rl   (BayesianStacker / run-line path)
  - ml_prob  vs  actual_home_win  (BayesianStackerML / moneyline path)

Output sections per target:
  1. Global Brier Score and ECE
  2. Full decile calibration table (10 equal-frequency bins)
  3. HIGH-PROB SUBSET (p >= 0.7143) -- subset Brier, ECE, win rate,
     mean predicted prob, and granular bin table within the subset

Usage:
    python -m wizard_agents.audit.decile_calibration_audit
    -- or --
    python wizard_agents/audit/decile_calibration_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT    = Path(__file__).resolve().parents[2]
_VALFILE = _ROOT / "xgb_val_predictions.csv"

# -250 American odds implied probability = 250 / (250 + 100)
HIGH_PROB_THRESHOLD: float = 250.0 / (250.0 + 100.0)   # 0.71429

N_DECILES       = 10
N_HIGHPROB_BINS = 5   # granular bins within the high-prob subset


# ── helpers ───────────────────────────────────────────────────────────────────

def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-width bins over [0,1])."""
    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    total  = len(y)
    err    = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            continue
        frac  = mask.sum() / total
        err  += frac * abs(p[mask].mean() - y[mask].mean())
    return float(err)


def decile_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10,
                 equal_freq: bool = True, label: str = "bin") -> pd.DataFrame:
    """
    Build a calibration bin table.
    equal_freq=True  -> quantile-based bins (equal-count deciles)
    equal_freq=False -> equal-width bins
    """
    if equal_freq:
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges     = np.unique(np.quantile(p, quantiles))
        bin_ids   = np.digitize(p, edges[1:-1])   # 0-indexed bins
    else:
        edges   = np.linspace(p.min(), p.max(), n_bins + 1)
        bin_ids = np.digitize(p, edges[1:-1])

    rows = []
    for i in range(n_bins):
        mask = (bin_ids == i)
        n    = int(mask.sum())
        if n == 0:
            continue
        mean_p   = float(p[mask].mean())
        actual_r = float(y[mask].mean())
        bs_bin   = float(np.mean((p[mask] - y[mask]) ** 2))
        rows.append({
            label:          i + 1,
            "n":            n,
            "p_lo":         float(p[mask].min()),
            "p_hi":         float(p[mask].max()),
            "mean_pred":    round(mean_p,   4),
            "actual_rate":  round(actual_r, 4),
            "gap":          round(mean_p - actual_r, 4),
            "brier_bin":    round(bs_bin, 5),
        })
    return pd.DataFrame(rows)


def run_target(label: str, prob_col: str, outcome_col: str,
               df: pd.DataFrame) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  TARGET: {label}")
    print(f"  prob={prob_col!r}  outcome={outcome_col!r}")
    print(sep)

    sub = df[[prob_col, outcome_col]].dropna()
    if sub.empty:
        print("  [ERROR] No data after dropna — skipping.")
        return

    p = sub[prob_col].to_numpy(dtype=float)
    y = sub[outcome_col].to_numpy(dtype=float)
    n = len(p)

    gs_brier = brier_score(y, p)
    gs_ece   = ece(y, p, n_bins=N_DECILES)

    print(f"\n  n={n}  |  Global Brier Score={gs_brier:.6f}  |  ECE={gs_ece:.6f}")
    print(f"  Overall actual rate={y.mean():.4f}  mean predicted={p.mean():.4f}")

    # ── Full decile table ─────────────────────────────────────────────────────
    print(f"\n  FULL DECILE TABLE (equal-frequency, {N_DECILES} bins)")
    print("  " + "-" * 66)
    tbl = decile_table(y, p, n_bins=N_DECILES, equal_freq=True, label="decile")
    tbl_str = tbl.to_string(index=False)
    for line in tbl_str.splitlines():
        print("  " + line)

    # ── High-prob subset ──────────────────────────────────────────────────────
    hp_mask = p >= HIGH_PROB_THRESHOLD
    n_hp    = int(hp_mask.sum())

    print(f"\n  HIGH-PROB SUBSET (p >= {HIGH_PROB_THRESHOLD:.4f})  n={n_hp}")
    print("  " + "-" * 66)

    if n_hp == 0:
        print("  [WARN] No samples in high-prob subset.")
        return

    p_hp = p[hp_mask]
    y_hp = y[hp_mask]

    hp_brier    = brier_score(y_hp, p_hp)
    hp_ece      = ece(y_hp, p_hp, n_bins=N_HIGHPROB_BINS)
    hp_win_rate = float(y_hp.mean())
    hp_mean_p   = float(p_hp.mean())
    hp_gap      = hp_mean_p - hp_win_rate

    print(f"  Subset Brier Score  : {hp_brier:.6f}")
    print(f"  Subset ECE          : {hp_ece:.6f}")
    print(f"  Actual win rate     : {hp_win_rate:.4f}")
    print(f"  Mean predicted prob : {hp_mean_p:.4f}")
    print(f"  Mean gap (pred-act) : {hp_gap:+.4f}  "
          f"{'[OVER-CONFIDENT]' if hp_gap > 0.02 else '[UNDER-CONFIDENT]' if hp_gap < -0.02 else '[WELL-CALIBRATED]'}")

    print(f"\n  HIGH-PROB BIN DETAIL ({N_HIGHPROB_BINS} equal-frequency bins within subset)")
    print("  " + "-" * 66)
    hp_tbl = decile_table(y_hp, p_hp, n_bins=N_HIGHPROB_BINS,
                          equal_freq=True, label="hp_bin")
    for line in hp_tbl.to_string(index=False).splitlines():
        print("  " + line)

    # Flag any bin where actual rate < 50% (model should never be this confident)
    bad = hp_tbl[hp_tbl["actual_rate"] < 0.50]
    if not bad.empty:
        print(f"\n  *** CALIBRATION FLAG: {len(bad)} high-prob bin(s) with actual_rate < 0.50 ***")
        for _, row in bad.iterrows():
            print(f"      bin {int(row['hp_bin'])}: mean_pred={row['mean_pred']:.4f}  "
                  f"actual_rate={row['actual_rate']:.4f}  n={int(row['n'])}")
    else:
        print(f"\n  All {N_HIGHPROB_BINS} high-prob bins have actual_rate >= 0.50.")


def main() -> None:
    if not _VALFILE.exists():
        print(f"[ERROR] Validation file not found: {_VALFILE}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(_VALFILE)

    # Filter to validation split only
    if "split" in df.columns:
        df_val = df[df["split"] == "val"].copy()
        print(f"Loaded {_VALFILE.name}  --  {len(df_val)} val rows "
              f"(of {len(df)} total; filtered on split=='val')")
    else:
        df_val = df.copy()
        print(f"Loaded {_VALFILE.name}  --  {len(df_val)} rows (no 'split' column)")

    print(f"Date range: {df_val['game_date'].min()} -> {df_val['game_date'].max()}")

    # Evaluate run-line stacker
    run_target(
        label       = "RUN-LINE STACKER (BayesianStacker / train_xgboost.py)",
        prob_col    = "rl_prob",
        outcome_col = "home_covers_rl",
        df          = df_val,
    )

    # Evaluate moneyline stacker
    run_target(
        label       = "MONEYLINE STACKER (BayesianStackerML / train_ml_model.py)",
        prob_col    = "ml_prob",
        outcome_col = "actual_home_win",
        df          = df_val,
    )

    print(f"\n{'=' * 70}")
    print("  END OF CALIBRATION AUDIT")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
