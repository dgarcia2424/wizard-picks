"""
analyze_f5_signals.py
=====================
Retrospective accuracy analysis of every F5 +0.5 cover signal.

Answers the question: which signal (MC simulation, XGB L1, XGB L2 stacker,
Pinnacle-estimated F5) is most predictive, at what probability threshold does
each become actionable, and what's the optimal blend weight?

Data source: f5_val_predictions.csv (2025 season, 2,436 games)
             joined with feature_matrix_enriched_v2.parquet for MC + Pinnacle.

Usage:
    python analyze_f5_signals.py               # full analysis + thresholds
    python analyze_f5_signals.py --save        # also write results to CSV
    python analyze_f5_signals.py --year 2025   # default; extend when more years available
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss, accuracy_score
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data" / "statcast"
MODELS_DIR = BASE_DIR / "models"

# Pinnacle ML → estimated F5 scaling factor (fitted from historical data)
PIN_F5_K = 1.289

# Action thresholds: signal must exceed these to generate a call
# (tuned below; defaults used for display before analysis runs)
DEFAULT_ACTION_THRESH = 0.55   # above → HOME call; below (1 - thresh) → AWAY call


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """
    Load and join:
      - f5_val_predictions.csv  → XGB L1, L2, team log-odds, actual F5 outcome
      - feature_matrix_enriched_v2.parquet  → MC F5 signals + Pinnacle ML prob
    Returns merged DataFrame with one row per game.
    """
    val_path = BASE_DIR / "f5_val_predictions.csv"
    fm_path  = BASE_DIR / "feature_matrix_enriched_v2.parquet"

    if not val_path.exists():
        raise FileNotFoundError(f"Missing {val_path} — run train_f5_model.py first")
    if not fm_path.exists():
        raise FileNotFoundError(f"Missing {fm_path} — run enrich_feature_matrix_v2.py first")

    val = pd.read_csv(val_path)
    val["game_date"] = pd.to_datetime(val["game_date"])

    fm_cols = [
        "game_pk", "game_date", "home_team", "away_team",
        "mc_f5_home_cover_pct", "mc_f5_home_win_pct", "mc_f5_tie_pct",
        "mc_f5_expected_total",
        "true_home_prob", "true_away_prob",
    ]
    fm = pd.read_parquet(fm_path, columns=fm_cols, engine="pyarrow")
    fm["game_date"] = pd.to_datetime(fm["game_date"])

    # Join on game_pk (primary); fall back to date+teams if needed
    merged = val.merge(
        fm[["game_pk", "mc_f5_home_cover_pct", "mc_f5_home_win_pct",
            "mc_f5_tie_pct", "true_home_prob", "true_away_prob"]],
        on="game_pk", how="left"
    )

    # Pinnacle estimated F5 probability
    # Formula: Pin F5 = 0.5 + K * (PinML_home - 0.5)
    merged["pin_f5_cover"] = (
        0.5 + PIN_F5_K * (merged["true_home_prob"].fillna(np.nan) - 0.5)
    )

    n_total  = len(merged)
    n_pin    = merged["pin_f5_cover"].notna().sum()
    n_mc     = merged["mc_f5_home_cover_pct"].notna().sum()
    n_xgb    = merged["xgb_cal_f5_cover"].notna().sum()
    n_stk    = merged["stacker_f5_cover"].notna().sum()

    print(f"  Loaded {n_total} games (2025 validation)")
    print(f"  Signal coverage: XGB={n_xgb}  Stacker={n_stk}  MC={n_mc}  Pin={n_pin}")
    print(f"  Positive rate (home covers +0.5): {merged['f5_home_cover'].mean():.1%}")

    return merged


# ---------------------------------------------------------------------------
# Per-signal accuracy analysis
# ---------------------------------------------------------------------------

SIGNALS = {
    "MC Sim (cover%)":    "mc_f5_home_cover_pct",
    "MC Sim (win%)":      "mc_f5_home_win_pct",
    "XGB L1 (Platt)":     "xgb_cal_f5_cover",
    "XGB L2 (Stacker)":   "stacker_f5_cover",
    "Pinnacle Est F5":    "pin_f5_cover",
    "Team Log-Odds":      "team_f5_log_odds",  # log-odds scale, needs sigmoid
}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def signal_stats(df: pd.DataFrame, sig_col: str) -> dict:
    """Compute accuracy metrics for one signal column against f5_home_cover."""
    sub = df[["f5_home_cover", sig_col]].dropna()
    if len(sub) < 100:
        return {"n": len(sub), "auc": None, "brier": None,
                "acc_50": None, "acc_55": None, "acc_60": None,
                "thresh_55acc": None, "n_bets_55": None}

    y    = sub["f5_home_cover"].values
    raw  = sub[sig_col].values

    # Team log-odds uses log-odds scale → convert to probability
    if sig_col == "team_f5_log_odds":
        probs = sigmoid(raw)
    else:
        probs = raw.clip(0.001, 0.999)

    auc    = roc_auc_score(y, probs)
    brier  = brier_score_loss(y, probs)
    ll     = log_loss(y, probs)
    acc_50 = accuracy_score(y, (probs >= 0.50).astype(int))

    # Accuracy at ≥0.55 threshold (home bets only)
    mask55 = probs >= 0.55
    acc_55 = accuracy_score(y[mask55], (probs >= 0.50)[mask55]) if mask55.sum() >= 20 else None
    n_55   = int(mask55.sum())

    # Accuracy at ≥0.60 threshold
    mask60 = probs >= 0.60
    acc_60 = accuracy_score(y[mask60], (probs >= 0.50)[mask60]) if mask60.sum() >= 20 else None

    # Find threshold where accuracy first hits 55% (on high-confidence end)
    thresh_55acc = None
    for t in np.arange(0.50, 0.80, 0.01):
        mk = probs >= t
        if mk.sum() >= 20:
            a = accuracy_score(y[mk], np.ones(mk.sum()))
            if a >= 0.55:
                thresh_55acc = round(t, 2)
                break

    return {
        "n": len(sub), "auc": auc, "brier": brier, "logloss": ll,
        "acc_50": acc_50, "acc_55": acc_55, "acc_60": acc_60,
        "thresh_55acc": thresh_55acc, "n_bets_55": n_55,
    }


def threshold_sweep(df: pd.DataFrame, sig_col: str,
                    thresholds=None) -> pd.DataFrame:
    """
    Sweep probability thresholds 0.50–0.75 and compute accuracy + sample size
    for bets taken at each threshold (both HOME and AWAY sides).
    Returns DataFrame with columns: thresh, side, n, acc, roi_flat.
    """
    if thresholds is None:
        thresholds = np.arange(0.50, 0.76, 0.01)

    sub = df[["f5_home_cover", sig_col]].dropna()
    y = sub["f5_home_cover"].values

    if sig_col == "team_f5_log_odds":
        probs = sigmoid(sub[sig_col].values)
    else:
        probs = sub[sig_col].values.clip(0.001, 0.999)

    rows = []
    for t in thresholds:
        # Home bets (signal favours home)
        mh = probs >= t
        if mh.sum() >= 10:
            acc_h = y[mh].mean()
            rows.append({"thresh": round(t, 2), "side": "HOME",
                         "n": int(mh.sum()), "acc": acc_h})
        # Away bets (signal favours away)
        ma = probs <= (1 - t)
        if ma.sum() >= 10:
            acc_a = (1 - y[ma]).mean()
            rows.append({"thresh": round(t, 2), "side": "AWAY",
                         "n": int(ma.sum()), "acc": acc_a})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Blend model
# ---------------------------------------------------------------------------

def fit_blend(df: pd.DataFrame) -> tuple[LogisticRegression, list[str], float]:
    """
    Fit a logistic regression blend of available signals.
    Uses 5-fold cross-validation to report OOF AUC.
    Returns (model, feature_cols, oof_auc).
    """
    blend_cols = []
    for name, col in SIGNALS.items():
        if col in df.columns and df[col].notna().sum() > 100:
            if col == "team_f5_log_odds":
                df[f"_sig_{col}"] = sigmoid(df[col])
                blend_cols.append(f"_sig_{col}")
            else:
                blend_cols.append(col)

    sub = df[["f5_home_cover"] + blend_cols].dropna()
    if len(sub) < 200:
        print("  [WARN] Not enough data for blend fitting")
        return None, blend_cols, None

    X = sub[blend_cols].values
    y = sub["f5_home_cover"].values

    # 5-fold OOF AUC
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr, va in kf.split(X, y):
        m = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    oof_auc = roc_auc_score(y, oof)

    # Final model on all data
    final = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    final.fit(X, y)

    return final, blend_cols, oof_auc


# ---------------------------------------------------------------------------
# Calibration check
# ---------------------------------------------------------------------------

def calibration_check(df: pd.DataFrame, sig_col: str, n_bins: int = 8) -> str:
    """Return a compact string showing avg predicted vs actual per probability bin."""
    sub = df[["f5_home_cover", sig_col]].dropna()
    if len(sub) < 100:
        return "  (insufficient data)"
    y = sub["f5_home_cover"].values
    p = sub[sig_col].values if sig_col != "team_f5_log_odds" else sigmoid(sub[sig_col].values)
    p = p.clip(0.001, 0.999)
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    lines = []
    for mp, fp in zip(mean_pred, frac_pos):
        bar = "#" * int(fp * 20)
        lines.append(f"    pred={mp:.2f}  actual={fp:.2f}  {'✓' if abs(mp-fp)<0.05 else '△'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Consensus action report
# ---------------------------------------------------------------------------

def consensus_actions(df: pd.DataFrame,
                       thresholds: dict[str, float]) -> pd.DataFrame:
    """
    For each game, determine the action call from each signal and the consensus.
    thresholds: {signal_col: float} — minimum probability for a HOME call.
    """
    rows = []
    for _, r in df.iterrows():
        calls = {}
        for name, col in SIGNALS.items():
            t = thresholds.get(col, DEFAULT_ACTION_THRESH)
            if col not in df.columns or pd.isna(r.get(col)):
                calls[name] = "—"
                continue
            p = sigmoid(r[col]) if col == "team_f5_log_odds" else float(r[col])
            if p >= t:
                calls[name] = f"HOME ({p:.0%})"
            elif p <= (1 - t):
                calls[name] = f"AWAY ({1-p:.0%})"
            else:
                calls[name] = "—"

        home_votes = sum(1 for v in calls.values() if v.startswith("HOME"))
        away_votes = sum(1 for v in calls.values() if v.startswith("AWAY"))
        total_votes = home_votes + away_votes
        if home_votes >= 3:
            consensus = f"STRONG HOME ({home_votes}/{total_votes})"
        elif away_votes >= 3:
            consensus = f"STRONG AWAY ({away_votes}/{total_votes})"
        elif home_votes == 2:
            consensus = f"LEAN HOME ({home_votes}/{total_votes})"
        elif away_votes == 2:
            consensus = f"LEAN AWAY ({away_votes}/{total_votes})"
        else:
            consensus = "SPLIT"

        rows.append({
            "game_pk":   r["game_pk"],
            "game_date": r["game_date"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "actual":    int(r["f5_home_cover"]),
            "consensus": consensus,
            **calls,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="F5 signal blend accuracy analysis"
    )
    parser.add_argument("--save", action="store_true",
                        help="Save results to f5_signal_analysis.csv")
    parser.add_argument("--sweep", action="store_true",
                        help="Print full threshold sweep table per signal")
    args = parser.parse_args()

    bar = "=" * 68
    print(bar)
    print("  F5 +0.5 Cover Signal Analysis")
    print(bar)

    df = load_data()

    # ── 1. Per-signal standalone accuracy ─────────────────────────────────
    print(f"\n{'─'*68}")
    print("  SIGNAL STANDALONE ACCURACY  (2025, n=2,436 games)")
    print(f"{'─'*68}")
    print(f"  {'Signal':<22}  {'AUC':>6}  {'Brier':>6}  "
          f"{'Acc@50%':>8}  {'Acc@≥55%':>9}  {'n@≥55%':>7}  "
          f"{'55%Acc thresh':>13}")
    print(f"  {'-'*22}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*13}")

    stats_all = {}
    for name, col in SIGNALS.items():
        if col not in df.columns:
            continue
        s = signal_stats(df, col)
        stats_all[name] = (col, s)
        auc_s   = f"{s['auc']:.4f}"    if s["auc"]   is not None else "  N/A "
        brier_s = f"{s['brier']:.4f}"  if s["brier"] is not None else "  N/A "
        acc50_s = f"{s['acc_50']:.1%}" if s["acc_50"] is not None else "  N/A "
        acc55_s = f"{s['acc_55']:.1%}" if s["acc_55"] is not None else "  N/A "
        n55_s   = str(s["n_bets_55"])  if s["n_bets_55"] is not None else "N/A"
        t55_s   = f"{s['thresh_55acc']:.2f}" if s["thresh_55acc"] is not None else " none"
        print(f"  {name:<22}  {auc_s:>6}  {brier_s:>6}  "
              f"{acc50_s:>8}  {acc55_s:>9}  {n55_s:>7}  {t55_s:>13}")

    # ── 2. Threshold sweep — find where each signal becomes actionable ─────
    print(f"\n{'─'*68}")
    print("  THRESHOLD SWEEP — accuracy + sample size per cut")
    print(f"  (Home bets only; side/threshold giving ≥55% acc highlighted)")
    print(f"{'─'*68}")

    optimal_thresholds = {}  # col → float threshold for HOME call
    for name, (col, _) in stats_all.items():
        sweep = threshold_sweep(df, col)
        home_sweep = sweep[sweep["side"] == "HOME"].sort_values("thresh")
        # Find lowest threshold giving ≥55% accuracy with ≥50 games
        act_rows = home_sweep[(home_sweep["acc"] >= 0.55) & (home_sweep["n"] >= 50)]
        if len(act_rows) > 0:
            best_t = float(act_rows.iloc[0]["thresh"])
            best_n = int(act_rows.iloc[0]["n"])
            best_acc = float(act_rows.iloc[0]["acc"])
            optimal_thresholds[col] = best_t
            marker = f"  *** ACTION THRESHOLD: {best_t:.2f}  ({best_n} bets, {best_acc:.1%} acc)"
        else:
            optimal_thresholds[col] = DEFAULT_ACTION_THRESH
            marker = "  (no threshold reached 55% acc with n≥50)"

        if args.sweep:
            print(f"\n  {name}")
            print(f"  {'Thresh':>7}  {'n HOME':>7}  {'Acc HOME':>9}")
            for _, row in home_sweep[home_sweep["thresh"].between(0.50, 0.72)].iterrows():
                hi = " <--" if abs(row["thresh"] - optimal_thresholds.get(col, 99)) < 0.005 else ""
                print(f"  {row['thresh']:>7.2f}  {int(row['n']):>7}  {row['acc']:>9.1%}{hi}")
            print(f"  {marker}")
        else:
            print(f"  {name:<22}  opt threshold={optimal_thresholds.get(col, DEFAULT_ACTION_THRESH):.2f}"
                  f"  {marker.strip()}")

    # ── 3. Blend model ────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("  BLEND MODEL  (logistic regression over all signals)")
    print(f"{'─'*68}")
    blend_model, blend_cols, blend_auc = fit_blend(df)
    if blend_model is not None and blend_auc is not None:
        print(f"  5-fold OOF AUC: {blend_auc:.4f}")
        print(f"  Feature weights (log-odds scale):")
        for col, coef in zip(blend_cols, blend_model.coef_[0]):
            clean = col.replace("_sig_", "")
            print(f"    {clean:<30}  {coef:+.4f}")
        print(f"  Intercept: {blend_model.intercept_[0]:+.4f}")

        # Apply blend to full dataset
        sub = df[["f5_home_cover"] + blend_cols].dropna()
        X_all = sub[blend_cols].values
        blend_probs = blend_model.predict_proba(X_all)[:, 1]
        blend_auc_full = roc_auc_score(sub["f5_home_cover"].values, blend_probs)
        print(f"  In-sample AUC (all data): {blend_auc_full:.4f}")

        # Blend action threshold
        sweep_blend = []
        y_b = sub["f5_home_cover"].values
        for t in np.arange(0.50, 0.76, 0.01):
            mk = blend_probs >= t
            if mk.sum() >= 20:
                sweep_blend.append((round(t, 2), int(mk.sum()),
                                    float(y_b[mk].mean())))
        print(f"\n  Blend threshold sweep (HOME bets):")
        print(f"  {'Thresh':>7}  {'n':>5}  {'Acc':>6}")
        for t, n, acc in sweep_blend:
            hi = " <-- 55%" if 0.54 <= acc <= 0.57 else (" <-- 60%" if 0.59 <= acc <= 0.62 else "")
            print(f"  {t:>7.2f}  {n:>5}  {acc:>6.1%}{hi}")
    else:
        print("  Blend fitting failed — insufficient overlapping data")

    # ── 4. Signal correlation matrix ─────────────────────────────────────
    print(f"\n{'─'*68}")
    print("  SIGNAL CORRELATIONS  (Pearson r — how much do signals agree?)")
    print(f"{'─'*68}")
    sig_cols_present = [c for _, c in SIGNALS.items()
                        if c in df.columns and df[c].notna().sum() > 100
                        and c != "team_f5_log_odds"]
    corr_df = df[sig_cols_present].dropna().corr()
    short_names = {c: c.replace("mc_f5_home_", "mc_").replace("_f5_cover", "").replace("xgb_cal", "xgb_l1").replace("stacker", "l2").replace("pin", "pin") for c in sig_cols_present}
    corr_df = corr_df.rename(index=short_names, columns=short_names)
    print(corr_df.round(3).to_string())

    # ── 5. Recommended action thresholds ─────────────────────────────────
    print(f"\n{'─'*68}")
    print("  RECOMMENDED ACTION THRESHOLDS  (HOME call / AWAY call)")
    print(f"{'─'*68}")
    print(f"  {'Signal':<22}  {'HOME if ≥':>10}  {'AWAY if ≤':>10}  {'Note'}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*30}")
    for name, (col, s) in stats_all.items():
        t = optimal_thresholds.get(col, DEFAULT_ACTION_THRESH)
        note = (f"AUC={s['auc']:.3f}" if s["auc"] else "N/A")
        print(f"  {name:<22}  {t:>10.2f}  {(1-t):>10.2f}  {note}")

    if args.save:
        # Save consensus action calls per game
        out = consensus_actions(df, optimal_thresholds)
        out_path = BASE_DIR / "f5_signal_analysis.csv"
        out.to_csv(out_path, index=False)
        print(f"\n  Saved per-game consensus actions -> {out_path}")

    print(f"\n{bar}")
    print("  Done.")
    print(bar)


if __name__ == "__main__":
    main()
