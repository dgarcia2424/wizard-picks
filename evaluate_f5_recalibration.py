"""
evaluate_f5_recalibration.py
============================
Post-training evaluation for the F5 Bayesian L2 Stacker after the LOYO
recalibration.  Two independent sections:

A. Posterior inspection
   Loads models/stacking_lr_f5.pkl (a pickled BayesianStackerF5) and prints
   alpha, beta, delta[0..3], gamma[feat] posterior means in tabular form.
   Optionally loads models/stacking_lr_f5.npz for posterior std bands.

B. Performance differential
   Compares legacy vs. new predictions on common game_ids.  Reports:
     - Brier, LogLoss, AUC (sklearn.metrics)
     - Brier improvement (%)
     - Expected Calibration Error (ECE, 10 equal-width bins)
     - Mean predicted prob vs. mean actual cover

Usage
-----
    # Posterior only
    python evaluate_f5_recalibration.py --posterior-only

    # Posterior + performance diff
    python evaluate_f5_recalibration.py \
        --legacy legacy_predictions.csv \
        --new    new_predictions.csv

    # Custom paths
    python evaluate_f5_recalibration.py \
        --stacker models/stacking_lr_f5.pkl \
        --npz     models/stacking_lr_f5.npz \
        --legacy  legacy_predictions.csv \
        --new     new_predictions.csv \
        --outcome-col f5_home_cover --prob-col p_stk --id-col game_id

CSV schema expected (both legacy and new):
    game_id, f5_home_cover (0/1), p_stk (float in [0,1])
Missing columns raise a clear error and skip the performance section.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
except ImportError as e:
    print(f"FATAL: sklearn is required ({e})", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# SECTION A: POSTERIOR INSPECTION
# ---------------------------------------------------------------------------

def load_stacker(pkl_path: Path):
    """Unpickle the BayesianStackerF5 object, injecting the class into
    __main__ if necessary (the stacker is saved by train_f5_model running
    as __main__)."""
    if not pkl_path.exists():
        raise FileNotFoundError(f"Stacker pickle not found: {pkl_path}")

    # Inject the class so pickle's __main__.BayesianStackerF5 ref resolves
    try:
        sys.path.insert(0, str(pkl_path.parent.parent))
        from train_f5_model import BayesianStackerF5  # noqa: F401
        import __main__
        if not hasattr(__main__, "BayesianStackerF5"):
            __main__.BayesianStackerF5 = BayesianStackerF5
    except Exception as e:
        print(f"[WARN] Could not import BayesianStackerF5 from train_f5_model: {e}")
        print("       Will attempt a bare pickle.load — may fail if the class "
              "reference cannot be resolved.")

    with open(pkl_path, "rb") as fh:
        return pickle.load(fh)


def maybe_load_npz(npz_path: Optional[Path]) -> Optional[dict]:
    if npz_path is None or not npz_path.exists():
        return None
    try:
        data = np.load(npz_path)
        return {k: data[k] for k in data.files}
    except Exception as e:
        print(f"[WARN] Could not read posterior trace {npz_path}: {e}")
        return None


_SEG_LABELS = ["LvL", "LvR", "RvL", "RvR"]


def _posterior_std(trace: dict, key: str) -> Optional[np.ndarray]:
    if trace is None or key not in trace:
        return None
    return np.asarray(trace[key]).std(axis=0)


def print_posterior(stacker, trace: Optional[dict]) -> pd.DataFrame:
    """Build a flat DataFrame of every posterior-mean parameter and print it.
    Returns the DataFrame for programmatic inspection."""

    rows = []

    # alpha
    a_std = trace["alpha"].std() if trace and "alpha" in trace else np.nan
    rows.append({"param": "alpha", "index": "-", "feature": "(intercept)",
                 "mean": float(stacker.alpha), "std": float(a_std)})

    # beta
    b_std = trace["beta"].std() if trace and "beta" in trace else np.nan
    rows.append({"param": "beta", "index": "-", "feature": "logit(xgb_raw)",
                 "mean": float(stacker.beta), "std": float(b_std)})

    # delta per segment
    d_std = _posterior_std(trace, "delta") if trace else None
    delta_arr = np.asarray(stacker.delta, dtype=float).ravel()
    for j, dv in enumerate(delta_arr):
        seg = _SEG_LABELS[j] if j < len(_SEG_LABELS) else f"seg{j}"
        std_j = float(d_std[j]) if d_std is not None and j < len(d_std) else np.nan
        rows.append({"param": "delta", "index": j, "feature": f"segment[{seg}]",
                     "mean": float(dv), "std": std_j})

    # gamma per domain feature
    feat_names = list(stacker.stacking_feature_names)
    g_std = _posterior_std(trace, "gamma") if trace else None
    gamma_arr = np.asarray(stacker.gamma, dtype=float).ravel()
    for k, gv in enumerate(gamma_arr):
        name = feat_names[k] if k < len(feat_names) else f"gamma_{k}"
        std_k = float(g_std[k]) if g_std is not None and k < len(g_std) else np.nan
        rows.append({"param": "gamma", "index": k, "feature": name,
                     "mean": float(gv), "std": std_k})

    df = pd.DataFrame(rows)

    # Sort gamma block by absolute mean magnitude
    gamma_mask = df["param"] == "gamma"
    gamma_sorted = (df[gamma_mask]
                    .assign(_abs=lambda d: d["mean"].abs())
                    .sort_values("_abs", ascending=False)
                    .drop(columns="_abs"))
    df_print = pd.concat([df[~gamma_mask], gamma_sorted], ignore_index=True)

    # Pretty print
    print("=" * 78)
    print("  F5 L2 Bayesian Stacker — Posterior Means")
    print("=" * 78)
    print(f"  # segments      : {getattr(stacker, 'n_segments', '?')}")
    print(f"  # domain feats  : {len(feat_names)}")
    print(f"  posterior trace : {'available' if trace else 'not available'}")
    print("")
    with pd.option_context("display.max_rows", None,
                           "display.width", 140,
                           "display.float_format", "{:+.4f}".format):
        print(df_print[["param", "index", "feature", "mean", "std"]].to_string(index=False))

    # Flag the two features the user asked about explicitly
    print("\n  Highlighted stacker features:")
    for target in ("rolling_f5_tie_rate", "pois_p_cover", "pois_lam_home",
                   "pois_lam_away"):
        if target in feat_names:
            k = feat_names.index(target)
            mean = float(gamma_arr[k])
            std_k = float(g_std[k]) if g_std is not None and k < len(g_std) else np.nan
            # Wald-style |z| > 2 as a rule-of-thumb signal detector
            z = mean / std_k if std_k and not np.isnan(std_k) and std_k != 0 else np.nan
            print(f"    gamma[{target:<22s}]  mean={mean:+.4f}  "
                  f"std={std_k:.4f}  |z|={'n/a' if np.isnan(z) else f'{abs(z):.2f}'}")
        else:
            print(f"    gamma[{target:<22s}]  — NOT IN stacking_feature_names")

    return df_print


# ---------------------------------------------------------------------------
# SECTION B: PERFORMANCE DIFFERENTIAL
# ---------------------------------------------------------------------------

REQUIRED_COLS_DEFAULT = ("game_id", "f5_home_cover", "p_stk")


def _load_preds(path: Path, id_col: str, outcome_col: str, prob_col: str,
                label: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[WARN] {label} predictions CSV not found: {path}")
        return None
    df = pd.read_csv(path)
    missing = [c for c in (id_col, outcome_col, prob_col) if c not in df.columns]
    if missing:
        print(f"[WARN] {label} CSV missing columns: {missing} "
              f"(have {list(df.columns)})")
        return None
    df = df[[id_col, outcome_col, prob_col]].copy()
    df = df.dropna(subset=[outcome_col, prob_col])
    df[outcome_col] = df[outcome_col].astype(int)
    df[prob_col] = df[prob_col].astype(float).clip(1e-6, 1 - 1e-6)
    return df


def _scores(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "n":        int(len(y)),
        "mean_y":   float(y.mean()),
        "mean_p":   float(p.mean()),
        "brier":    float(brier_score_loss(y, p)),
        "logloss":  float(log_loss(y, p, labels=[0, 1])),
        "auc":      float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
    }


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    """10-bin equal-width Expected Calibration Error."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx  = np.clip(np.digitize(p, bins, right=True) - 1, 0, n_bins - 1)
    rows = []
    ece_sum = 0.0
    N = len(y)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            rows.append({"bin": f"[{bins[b]:.2f},{bins[b+1]:.2f})",
                         "n": 0, "mean_p": np.nan, "mean_y": np.nan, "gap": np.nan})
            continue
        mean_p = float(p[mask].mean())
        mean_y = float(y[mask].mean())
        gap    = mean_y - mean_p
        w      = mask.sum() / N
        ece_sum += w * abs(gap)
        rows.append({"bin": f"[{bins[b]:.2f},{bins[b+1]:.2f})",
                     "n": int(mask.sum()),
                     "mean_p": mean_p, "mean_y": mean_y, "gap": gap})
    return float(ece_sum), pd.DataFrame(rows)


def compare_predictions(legacy: pd.DataFrame, new: pd.DataFrame,
                        id_col: str, outcome_col: str, prob_col: str) -> None:
    merged = legacy.merge(new, on=id_col, how="inner",
                          suffixes=("_legacy", "_new"))
    if len(merged) == 0:
        print(f"[FAIL] Inner join on '{id_col}' produced 0 rows — cannot compare.")
        return

    # Cross-check outcome consistency
    y_leg = merged[f"{outcome_col}_legacy"].to_numpy(dtype=int)
    y_new = merged[f"{outcome_col}_new"].to_numpy(dtype=int)
    if not np.array_equal(y_leg, y_new):
        n_diff = int((y_leg != y_new).sum())
        print(f"[WARN] {n_diff} rows disagree on {outcome_col} between CSVs — "
              f"using the legacy outcome column for scoring.")
    y = y_leg
    p_leg = merged[f"{prob_col}_legacy"].to_numpy(dtype=float)
    p_new = merged[f"{prob_col}_new"].to_numpy(dtype=float)

    s_leg = _scores(y, p_leg)
    s_new = _scores(y, p_new)
    ece_leg, bin_leg = _ece(y, p_leg)
    ece_new, bin_new = _ece(y, p_new)

    print("\n" + "=" * 78)
    print("  Performance Differential")
    print("=" * 78)
    print(f"  Joined rows : {len(merged):,}   (legacy={len(legacy):,}  new={len(new):,})")
    print(f"  Base rate   : {y.mean():.4f}\n")

    summary = pd.DataFrame({
        "legacy": s_leg,
        "new":    s_new,
    }).T
    summary["ece_10bin"] = [ece_leg, ece_new]
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary.to_string())

    # Deltas
    brier_delta = s_new["brier"] - s_leg["brier"]
    brier_improve_pct = -100.0 * brier_delta / s_leg["brier"] if s_leg["brier"] > 0 else 0.0
    logloss_delta = s_new["logloss"] - s_leg["logloss"]
    auc_delta = s_new["auc"] - s_leg["auc"]

    print("\n  Differentials  (new − legacy):")
    print(f"    Δ Brier      : {brier_delta:+.5f}   "
          f"({brier_improve_pct:+.2f}% improvement — positive = better)")
    print(f"    Δ LogLoss    : {logloss_delta:+.5f}   (negative = better)")
    print(f"    Δ AUC        : {auc_delta:+.5f}       (positive = better)")
    print(f"    Δ ECE (10b)  : {ece_new - ece_leg:+.5f}  (negative = better)")

    verdict = "IMPROVED" if brier_delta < 0 else ("DEGRADED" if brier_delta > 0 else "NEUTRAL")
    print(f"\n  Verdict (Brier) : {verdict}")

    # Per-bin calibration side-by-side
    bins = bin_leg[["bin", "n", "mean_p", "mean_y", "gap"]].rename(
        columns={"n": "n_leg", "mean_p": "mp_leg", "mean_y": "my_leg", "gap": "gap_leg"}
    ).merge(
        bin_new[["bin", "n", "mean_p", "mean_y", "gap"]].rename(
            columns={"n": "n_new", "mean_p": "mp_new", "mean_y": "my_new", "gap": "gap_new"}
        ),
        on="bin", how="outer",
    )
    print("\n  Calibration curve (10 equal-width bins):")
    with pd.option_context("display.float_format", "{:+.4f}".format,
                           "display.width", 160):
        print(bins.to_string(index=False))


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> int:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="F5 L2 stacker recalibration evaluator")
    ap.add_argument("--stacker", type=Path,
                    default=here / "models" / "stacking_lr_f5.pkl",
                    help="Path to pickled BayesianStackerF5")
    ap.add_argument("--npz", type=Path,
                    default=here / "models" / "stacking_lr_f5.npz",
                    help="Path to NUTS posterior trace (optional, for std)")
    ap.add_argument("--legacy", type=Path, default=None,
                    help="Legacy predictions CSV (game_id, f5_home_cover, p_stk)")
    ap.add_argument("--new", type=Path, default=None,
                    help="New predictions CSV (same schema)")
    ap.add_argument("--id-col", default="game_id")
    ap.add_argument("--outcome-col", default="f5_home_cover")
    ap.add_argument("--prob-col", default="p_stk")
    ap.add_argument("--posterior-only", action="store_true",
                    help="Skip the performance-diff section")
    args = ap.parse_args()

    # ── SECTION A ──────────────────────────────────────────────────────────
    try:
        stacker = load_stacker(args.stacker)
    except Exception as e:
        print(f"FAIL: Could not load stacker pickle: {e}", file=sys.stderr)
        return 1

    trace = maybe_load_npz(args.npz)
    try:
        print_posterior(stacker, trace)
    except Exception as e:
        print(f"[WARN] posterior tabulation failed: {e}")

    # ── SECTION B ──────────────────────────────────────────────────────────
    if args.posterior_only:
        return 0

    if args.legacy is None or args.new is None:
        print("\n[INFO] --legacy and/or --new not provided; skipping "
              "performance differential.  Pass both CSVs to enable it.")
        return 0

    legacy_df = _load_preds(args.legacy, args.id_col, args.outcome_col,
                            args.prob_col, "legacy")
    new_df    = _load_preds(args.new,    args.id_col, args.outcome_col,
                            args.prob_col, "new")
    if legacy_df is None or new_df is None:
        print("\n[INFO] One or both prediction files unusable — differential skipped.")
        return 0

    compare_predictions(legacy_df, new_df, args.id_col, args.outcome_col, args.prob_col)
    return 0


if __name__ == "__main__":
    sys.exit(main())
