"""
train_xgboost.py
================
Train XGBoost models on the feature matrix.

Models trained:
  1. home_covers_rl   (binary: home team covers -1.5 run line)
  2. total_runs       (regression: total runs scored)
  3. actual_home_win  (binary: home team wins outright — auxiliary)

Training modes
--------------
Default (no --ncv):
  Training split  : 2023 + 2024 games   (df["split"] == "train")
  Validation split: 2025 games           (df["split"] == "val")

Nested Cross-Validation (--ncv):
  Walk-forward outer folds:
    Fold 1 : Train 2023        → Validate 2024
    Fold 2 : Train 2023+2024   → Validate 2025
  Final model saved to disk:
    Train on 2023+2024+2025 (all non-2026 data)
  2026 is held out entirely — never seen during NCV or final training.

Outputs:
  models/xgb_rl.json           XGBoost run-line model (binary)
  models/xgb_total.json        XGBoost total runs model (regression)
  models/xgb_ml.json           XGBoost moneyline model (binary)
  models/feature_cols.json     ordered feature column list
  xgb_val_predictions.csv      validation set predictions (last fold in NCV mode)
  xgb_feature_importance.csv   gain-based feature importances
  xgb_ncv_results.csv          per-fold NCV metrics (--ncv only)
  models/calibrator_rl.pkl     isotonic calibrator for run-line model
  models/calibrator_ml.pkl     isotonic calibrator for moneyline model

Usage:
  python train_xgboost.py
  python train_xgboost.py --matrix feature_matrix.parquet
  python train_xgboost.py --no-early-stop      # fixed 500 estimators
  python train_xgboost.py --ncv                # walk-forward NCV + final model
  python train_xgboost.py --ncv --no-early-stop

Calibration (isotonic regression):
  NCV mode : fitted on pooled OOF predictions (2024 from Fold1 + 2025 from Fold2)
             — zero leakage, each prediction was made by a model that never saw it
  Standard : fitted on the 2025 val set (same data used for early stopping — minor
             leakage, but calibration layer is low-capacity so impact is small)
  Output   : models/calibrator_rl.pkl  (applied automatically at inference time)
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss,
    mean_absolute_error, mean_squared_error
)

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("pip install xgboost")

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(exist_ok=True)

# Columns that are NOT features (labels, identifiers, administrative)
NON_FEATURE_COLS = {
    "game_date", "game_pk", "home_team", "away_team", "season", "year",
    "home_starter_name", "away_starter_name", "split",
    # Labels
    "actual_home_win", "actual_game_total", "actual_f5_total",
    "actual_f3_total", "actual_f1_total",
    "home_score", "away_score", "home_margin",
    "home_covers_rl", "away_covers_rl", "total_runs",
    # Raw odds (keep implied prob and close ML as features, skip raw ML)
    "open_total", "close_total",  # 100% null
    "source", "pull_timestamp",
}

# Columns with > threshold nulls that should be imputed (not dropped)
IMPUTE_MEDIAN_COLS = [
    "vegas_implied_home", "vegas_implied_away",
    "close_ml_home", "close_ml_away",
]

# XGBoost base parameters (shared across all models)
XGB_BASE = {
    "tree_method":    "hist",
    "device":         "cpu",
    "n_jobs":         -1,
    "random_state":   42,
    "learning_rate":  0.04,
    "max_depth":      5,
    "min_child_weight": 20,   # at least 20 samples per leaf (avoids tiny splits)
    "subsample":      0.80,
    "colsample_bytree": 0.75,
    "reg_alpha":      0.5,    # L1
    "reg_lambda":     2.0,    # L2
    "n_estimators":   600,
}

# Early stopping patience (rounds without improvement on eval set)
EARLY_STOPPING_ROUNDS = 40

# Betting break-even at -110 juice
BREAKEVEN_PROB = 110 / 210   # ~0.5238

# Walk-forward NCV fold definitions
NCV_FOLDS = [
    {"train_years": [2023],       "val_year": 2024},
    {"train_years": [2023, 2024], "val_year": 2025},
]
# All years used in NCV (for the final model trained on everything except 2026)
NCV_ALL_YEARS = [2023, 2024, 2025]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def prep_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Extract and clean feature columns.
    Returns (X_df, feature_cols).
    """
    drop = set(NON_FEATURE_COLS)
    feature_cols = [c for c in df.columns if c not in drop]

    X = df[feature_cols].copy()

    # Median imputation for key columns that have structural nulls
    for col in IMPUTE_MEDIAN_COLS:
        if col in X.columns:
            median = X[col].median()
            X[col] = X[col].fillna(median)

    # Convert any boolean-ish columns
    for col in X.columns:
        if X[col].dtype == object:
            try:
                X[col] = pd.to_numeric(X[col], errors="coerce")
            except Exception:
                X = X.drop(columns=[col])

    return X, feature_cols


def log_metrics(name: str, y_true, y_pred_prob=None, y_pred_class=None,
                y_pred_reg=None):
    print(f"\n  --- {name} ---")
    if y_pred_prob is not None:
        auc   = roc_auc_score(y_true, y_pred_prob)
        ll    = log_loss(y_true, y_pred_prob)
        brier = brier_score_loss(y_true, y_pred_prob)
        acc   = ((y_pred_prob > 0.50) == y_true).mean()
        print(f"    AUC-ROC     : {auc:.4f}")
        print(f"    Log-loss    : {ll:.4f}")
        print(f"    Brier score : {brier:.4f}")
        print(f"    Accuracy@50%: {acc:.4f}")
    if y_pred_reg is not None:
        mae  = mean_absolute_error(y_true, y_pred_reg)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_reg))
        print(f"    MAE         : {mae:.4f}")
        print(f"    RMSE        : {rmse:.4f}")


def edge_analysis(val_df: pd.DataFrame, prob_col: str,
                  label_col: str = "home_covers_rl",
                  thresholds=(0.50, 0.525, 0.55, 0.575, 0.60)):
    """
    Show win rate and simulated ROI at various probability thresholds.
    Assumes standard -110 juice on run line.
    """
    print(f"\n  Edge analysis (model_prob >= threshold -> bet, -110 juice):")
    print(f"  {'threshold':>10}  {'n_bets':>7}  {'win_rate':>9}  {'roi':>7}  {'units':>7}")
    print(f"  {'-'*45}")

    has_vegas = "vegas_implied_home" in val_df.columns
    if has_vegas:
        val_df = val_df.copy()
        val_df["edge_vs_vegas"] = val_df[prob_col] - val_df["vegas_implied_home"]

    for thresh in thresholds:
        bets = val_df[val_df[prob_col] >= thresh]
        n    = len(bets)
        if n < 20:
            continue
        wins = bets[label_col].sum()
        wr   = wins / n
        roi  = (wins * (100 / 110) - (n - wins)) / n  # ROI per unit at -110
        marker = " <-- breakeven" if abs(thresh - BREAKEVEN_PROB) < 0.01 else ""
        print(f"  {thresh:>10.3f}  {n:>7}  {wr:>9.3f}  {roi:>+7.3f}  "
              f"{(wins*(100/110) - (n-wins)):>+7.1f}{marker}")


# ---------------------------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------------------------

def train_binary(X_train, y_train, X_val, y_val,
                 label: str, early_stop: bool,
                 scale_pos: float = 1.0) -> xgb.XGBClassifier:
    """Train a binary XGBoost classifier with optional early stopping."""
    params = {**XGB_BASE,
              "objective":       "binary:logistic",
              "eval_metric":     ["logloss", "auc"],
              "scale_pos_weight": scale_pos}

    callbacks = []
    if early_stop:
        callbacks = [xgb.callback.EarlyStopping(
            rounds=EARLY_STOPPING_ROUNDS,
            metric_name="auc",
            maximize=True,
            save_best=True,
        )]

    model = xgb.XGBClassifier(**params, callbacks=callbacks)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    best_iter = model.best_iteration if hasattr(model, "best_iteration") else -1
    print(f"    {label}: best_iteration={best_iter} | "
          f"scale_pos_weight={scale_pos:.2f}")
    return model


def train_regression(X_train, y_train, X_val, y_val,
                     label: str, early_stop: bool) -> xgb.XGBRegressor:
    """Train an XGBoost regressor."""
    params = {**XGB_BASE,
              "objective":   "reg:squarederror",
              "eval_metric": ["rmse", "mae"]}
    params.pop("scale_pos_weight", None)

    callbacks = []
    if early_stop:
        callbacks = [xgb.callback.EarlyStopping(
            rounds=EARLY_STOPPING_ROUNDS,
            metric_name="rmse",
            maximize=False,
            save_best=True,
        )]

    model = xgb.XGBRegressor(**params, callbacks=callbacks)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    best_iter = model.best_iteration if hasattr(model, "best_iteration") else -1
    print(f"    {label}: best_iteration={best_iter}")
    return model


# ---------------------------------------------------------------------------
# NESTED CROSS-VALIDATION
# ---------------------------------------------------------------------------

def _fold_masks(df: pd.DataFrame, train_years: list[int], val_year: int):
    """Return boolean masks for a single NCV fold."""
    train_mask = df["year"].isin(train_years)
    val_mask   = df["year"] == val_year
    return train_mask, val_mask


def run_ncv(df: pd.DataFrame, X_all: pd.DataFrame,
            feature_cols: list[str], early_stop: bool
            ) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Walk-forward Nested Cross-Validation over NCV_FOLDS.

    Fold 1 : Train 2023        -> Validate 2024
    Fold 2 : Train 2023+2024   -> Validate 2025
    2026 is never touched.

    Returns
    -------
    (ncv_rows, oof_probs_rl, oof_labels_rl, oof_probs_ml, oof_labels_ml)
      ncv_rows       : list of per-fold metric dicts
      oof_probs_rl   : pooled raw XGBoost run-line probs across all folds
      oof_labels_rl  : corresponding true home_covers_rl labels
      oof_probs_ml   : pooled raw XGBoost moneyline probs across all folds
      oof_labels_ml  : corresponding true actual_home_win labels
    """
    print(f"\n{'='*60}")
    print(f"  NESTED CROSS-VALIDATION  (2026 held out)")
    print(f"{'='*60}")

    ncv_rows = []        # one dict per fold x model type
    oof_probs_rl  = []   # raw XGB RL probs (for calibration)
    oof_labels_rl = []   # true home_covers_rl
    oof_probs_ml  = []   # raw XGB ML probs (for calibration)
    oof_labels_ml = []   # true actual_home_win

    for fold_idx, fold in enumerate(NCV_FOLDS, 1):
        train_years = fold["train_years"]
        val_year    = fold["val_year"]
        train_mask, val_mask = _fold_masks(df, train_years, val_year)

        print(f"\n  {'-'*56}")
        print(f"  Fold {fold_idx}  |  Train: {train_years}  ->  Validate: {val_year}")
        print(f"  {'-'*56}")

        # ── Run-line binary ───────────────────────────────────────────────
        rl_tr = train_mask & df["home_covers_rl"].notna()
        rl_vl = val_mask   & df["home_covers_rl"].notna()
        X_tr  = X_all[rl_tr];  y_tr = df.loc[rl_tr, "home_covers_rl"].astype(int)
        X_vl  = X_all[rl_vl];  y_vl = df.loc[rl_vl, "home_covers_rl"].astype(int)

        neg = (y_tr == 0).sum();  pos = (y_tr == 1).sum()
        scale = neg / max(pos, 1)

        print(f"\n  [RL] n_train={rl_tr.sum()} | n_val={rl_vl.sum()} | "
              f"scale_pos_weight={scale:.2f}")
        rl_model = train_binary(X_tr, y_tr, X_vl, y_vl,
                                label=f"Fold{fold_idx}-RL",
                                early_stop=early_stop, scale_pos=scale)
        rl_probs = rl_model.predict_proba(X_vl)[:, 1]

        # Collect OOF predictions for calibration (zero leakage per fold)
        oof_probs_rl.extend(rl_probs.tolist())
        oof_labels_rl.extend(y_vl.tolist())

        auc   = roc_auc_score(y_vl, rl_probs)
        ll    = log_loss(y_vl, rl_probs)
        brier = brier_score_loss(y_vl, rl_probs)
        acc   = ((rl_probs > 0.50) == y_vl).mean()
        cover_rate = y_vl.mean()

        print(f"    AUC={auc:.4f}  LogLoss={ll:.4f}  "
              f"Brier={brier:.4f}  Acc={acc:.4f}  "
              f"BaseCoverRate={cover_rate:.3f}")

        ncv_rows.append(dict(
            fold=fold_idx, model="run_line",
            train_years=str(train_years), val_year=val_year,
            n_train=int(rl_tr.sum()), n_val=int(rl_vl.sum()),
            auc=round(auc, 4), log_loss=round(ll, 4),
            brier=round(brier, 4), accuracy=round(acc, 4),
            base_cover_rate=round(cover_rate, 4),
        ))

        # Edge analysis for this fold
        vdf_fold = df[rl_vl].copy()
        vdf_fold["rl_prob"] = rl_probs
        edge_analysis(vdf_fold, "rl_prob", label_col="home_covers_rl",
                      thresholds=[0.48, 0.50, 0.52, 0.525, 0.54, 0.56, 0.58, 0.60])

        # ── Total runs regressor ──────────────────────────────────────────
        tot_tr = train_mask & df["total_runs"].notna()
        tot_vl = val_mask   & df["total_runs"].notna()
        Xt_tr  = X_all[tot_tr];  yt_tr = df.loc[tot_tr, "total_runs"]
        Xt_vl  = X_all[tot_vl];  yt_vl = df.loc[tot_vl, "total_runs"]

        print(f"\n  [TOT] n_train={tot_tr.sum()} | n_val={tot_vl.sum()}")
        tot_model = train_regression(Xt_tr, yt_tr, Xt_vl, yt_vl,
                                     label=f"Fold{fold_idx}-TOTAL",
                                     early_stop=early_stop)
        tot_preds = tot_model.predict(Xt_vl)
        mae  = mean_absolute_error(yt_vl, tot_preds)
        rmse = np.sqrt(mean_squared_error(yt_vl, tot_preds))
        within_1_5 = (np.abs(tot_preds - yt_vl.values) <= 1.5).mean()
        print(f"    MAE={mae:.4f}  RMSE={rmse:.4f}  Within1.5={within_1_5:.3f}")

        ncv_rows.append(dict(
            fold=fold_idx, model="total_runs",
            train_years=str(train_years), val_year=val_year,
            n_train=int(tot_tr.sum()), n_val=int(tot_vl.sum()),
            mae=round(mae, 4), rmse=round(rmse, 4),
            within_1_5=round(within_1_5, 4),
        ))

        # ── Moneyline binary ──────────────────────────────────────────────
        ml_tr = train_mask & df["actual_home_win"].notna()
        ml_vl = val_mask   & df["actual_home_win"].notna()
        Xm_tr = X_all[ml_tr];  ym_tr = df.loc[ml_tr, "actual_home_win"].astype(int)
        Xm_vl = X_all[ml_vl];  ym_vl = df.loc[ml_vl, "actual_home_win"].astype(int)

        ml_scale = (ym_tr == 0).sum() / max((ym_tr == 1).sum(), 1)
        print(f"\n  [ML] n_train={ml_tr.sum()} | n_val={ml_vl.sum()} | "
              f"scale_pos_weight={ml_scale:.2f}")
        ml_model = train_binary(Xm_tr, ym_tr, Xm_vl, ym_vl,
                                label=f"Fold{fold_idx}-ML",
                                early_stop=early_stop, scale_pos=ml_scale)
        ml_probs = ml_model.predict_proba(Xm_vl)[:, 1]

        # Collect ML OOF predictions for calibration (only when n_train > 0)
        if ml_tr.sum() > 0:
            oof_probs_ml.extend(ml_probs.tolist())
            oof_labels_ml.extend(ym_vl.tolist())

        ml_auc = roc_auc_score(ym_vl, ml_probs)
        ml_acc = ((ml_probs > 0.50) == ym_vl).mean()
        print(f"    AUC={ml_auc:.4f}  Acc={ml_acc:.4f}")

        ncv_rows.append(dict(
            fold=fold_idx, model="moneyline",
            train_years=str(train_years), val_year=val_year,
            n_train=int(ml_tr.sum()), n_val=int(ml_vl.sum()),
            auc=round(ml_auc, 4), accuracy=round(ml_acc, 4),
        ))

    # ── NCV Summary ───────────────────────────────────────────────────────
    rl_rows = [r for r in ncv_rows if r["model"] == "run_line"]
    print(f"\n{'='*60}")
    print(f"  NCV SUMMARY — Run-Line Model")
    print(f"{'='*60}")
    print(f"  {'Fold':>4}  {'Train':>15}  {'Val':>5}  {'n_val':>6}  "
          f"{'AUC':>7}  {'LogLoss':>8}  {'Acc':>7}  {'Cover%':>7}")
    print(f"  {'-'*63}")
    for r in rl_rows:
        print(f"  {r['fold']:>4}  {r['train_years']:>15}  {r['val_year']:>5}  "
              f"{r['n_val']:>6}  {r['auc']:>7.4f}  {r['log_loss']:>8.4f}  "
              f"{r['accuracy']:>7.4f}  {r['base_cover_rate']:>7.3f}")

    avg_auc = np.mean([r["auc"] for r in rl_rows])
    avg_ll  = np.mean([r["log_loss"] for r in rl_rows])
    avg_acc = np.mean([r["accuracy"] for r in rl_rows])
    print(f"  {'MEAN':>4}  {'':>15}  {'':>5}  {'':>6}  "
          f"{avg_auc:>7.4f}  {avg_ll:>8.4f}  {avg_acc:>7.4f}")
    print()
    print(f"  Interpretation:")
    print(f"    Mean AUC  {avg_auc:.4f} — cross-validated discriminative power")
    print(f"    Mean Acc  {avg_acc:.4f} — at 50% probability threshold")
    print(f"    2026 was NOT used in any fold above.")

    # Save NCV results CSV
    ncv_df = pd.DataFrame(ncv_rows)
    ncv_path = "xgb_ncv_results.csv"
    ncv_df.to_csv(ncv_path, index=False)
    print(f"\n  Saved NCV results to {ncv_path}")

    return (
        ncv_rows,
        np.array(oof_probs_rl),
        np.array(oof_labels_rl),
        np.array(oof_probs_ml) if oof_probs_ml else np.array([]),
        np.array(oof_labels_ml) if oof_labels_ml else np.array([]),
    )


# ---------------------------------------------------------------------------
# CALIBRATION
# ---------------------------------------------------------------------------

def fit_and_save_calibrator(raw_probs: np.ndarray, true_labels: np.ndarray,
                            out_path: Path, label: str = "") -> IsotonicRegression:
    """
    Fit an isotonic regression calibrator on (raw_probs, true_labels) and
    save it to out_path as a pickle.

    Isotonic regression is monotonic (preserves ranking = preserves AUC) and
    non-parametric (learns any shape, unlike Platt's logistic).  With 2000+
    samples it generalises well without overfitting.

    Returns the fitted calibrator.
    """
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(raw_probs, true_labels)
    out_path.write_bytes(pickle.dumps(cal))
    print(f"  Saved calibrator [{label}] -> {out_path}  "
          f"(n={len(raw_probs)}, mean_raw={raw_probs.mean():.3f}, "
          f"mean_label={true_labels.mean():.3f})")
    return cal


def calibration_report(raw_probs: np.ndarray, true_labels: np.ndarray,
                        calibrator: IsotonicRegression, label: str = "") -> None:
    """
    Print a before/after calibration table.
    Columns: prob_bucket | n | actual_cover | raw_mean | cal_mean
    """
    cal_probs = calibrator.predict(raw_probs)
    bins   = [0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 1.0]
    labels = ["<.30", ".30-.35", ".35-.40", ".40-.45",
              ".45-.50", ".50-.55", ".55-.60", ".60-.65", ">.65"]

    print(f"\n  Calibration report [{label}] — raw vs isotonic-calibrated")
    print(f"  {'bucket':>10}  {'n':>6}  {'actual':>8}  "
          f"{'raw_mean':>9}  {'cal_mean':>9}  {'raw_err':>8}  {'cal_err':>8}")
    print(f"  {'-'*68}")

    raw_errs = []
    cal_errs = []

    bucket_idx = np.digitize(raw_probs, bins) - 1
    bucket_idx = np.clip(bucket_idx, 0, len(labels) - 1)

    for i, lbl in enumerate(labels):
        mask = bucket_idx == i
        n = mask.sum()
        if n < 5:
            continue
        actual   = true_labels[mask].mean()
        raw_mean = raw_probs[mask].mean()
        cal_mean = cal_probs[mask].mean()
        raw_err  = abs(actual - raw_mean)
        cal_err  = abs(actual - cal_mean)
        raw_errs.append(raw_err)
        cal_errs.append(cal_err)
        flag = " <-- improved" if cal_err < raw_err - 0.01 else ""
        print(f"  {lbl:>10}  {n:>6}  {actual:>8.3f}  "
              f"{raw_mean:>9.3f}  {cal_mean:>9.3f}  "
              f"{raw_err:>8.3f}  {cal_err:>8.3f}{flag}")

    if raw_errs:
        print(f"  {'-'*68}")
        print(f"  {'MEAN':>10}  {'':>6}  {'':>8}  {'':>9}  {'':>9}  "
              f"{np.mean(raw_errs):>8.3f}  {np.mean(cal_errs):>8.3f}")
        improvement = (np.mean(raw_errs) - np.mean(cal_errs)) / np.mean(raw_errs) * 100
        print(f"\n  Mean calibration error improved by {improvement:.1f}% "
              f"({np.mean(raw_errs):.3f} -> {np.mean(cal_errs):.3f})")

    # Show updated bet thresholds in calibrated space
    print(f"\n  Threshold translation (raw XGBoost -> calibrated probability):")
    for raw_thresh in [0.34, 0.40, 0.54, 0.58]:
        cal_val = float(calibrator.predict([raw_thresh])[0])
        print(f"    raw {raw_thresh:.2f}  ->  calibrated {cal_val:.3f}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost models for the MLB pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--matrix", type=str, default="feature_matrix.parquet",
                        help="Feature matrix parquet file")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping (use all n_estimators)")
    parser.add_argument("--val-year", type=int, default=2025,
                        help="Validation year for non-NCV mode (default: 2025)")
    parser.add_argument("--ncv", action="store_true",
                        help=(
                            "Run walk-forward Nested Cross-Validation "
                            "(Fold1: train 2023/val 2024, Fold2: train 2023-2024/val 2025) "
                            "then train final model on 2023+2024+2025. "
                            "2026 is held out entirely."
                        ))
    args = parser.parse_args()
    early_stop = not args.no_early_stop

    print("=" * 60)
    print("  train_xgboost.py")
    if args.ncv:
        print("  Mode: Nested Cross-Validation (2026 held out)")
    else:
        print(f"  Mode: Standard  (train=2023-2024 | val={args.val_year})")
    print("=" * 60)

    # --- Load data -----------------------------------------------------------
    df = pd.read_parquet(args.matrix, engine="pyarrow")
    print(f"\n  Loaded {len(df)} rows x {len(df.columns)} cols")

    # Year column detection
    year_col = None
    for c in ("year", "season"):
        if c in df.columns:
            year_col = c
            break

    if args.ncv and year_col is None:
        raise ValueError(
            "--ncv requires a 'year' or 'season' column in the feature matrix "
            "to identify which rows belong to which season."
        )

    if year_col and year_col != "year":
        df["year"] = df[year_col]

    # Prepare features
    X_all, feature_cols = prep_features(df)
    print(f"  Feature columns: {len(feature_cols)}")

    # Save feature column order for inference (same regardless of mode)
    (MODELS_DIR / "feature_cols.json").write_text(
        json.dumps(feature_cols, indent=2))

    # =========================================================================
    # NCV MODE
    # =========================================================================
    if args.ncv:
        # Validate year coverage
        years_available = sorted(df["year"].dropna().unique().astype(int))
        print(f"  Years in matrix: {years_available}")
        required = set(NCV_ALL_YEARS)
        missing  = required - set(years_available)
        if missing:
            raise ValueError(
                f"Feature matrix is missing required NCV years: {missing}. "
                f"Available: {years_available}"
            )

        # ── Step 1: Run the NCV ────────────────────────────────────────────
        ncv_rows, oof_probs_rl, oof_labels_rl, oof_probs_ml, oof_labels_ml = \
            run_ncv(df, X_all, feature_cols, early_stop)

        # ── Step 2: Train final production model on 2023+2024+2025 ────────
        print(f"\n{'='*60}")
        print(f"  FINAL MODEL TRAINING  (2023 + 2024 + 2025)")
        print(f"  2026 remains held out.")
        print(f"{'='*60}")

        final_train_mask = df["year"].isin(NCV_ALL_YEARS)

        # We still need an eval set for early stopping — use 2025 (last fold)
        # so the early stopping "validation" signal is still meaningful.
        final_val_mask = df["year"] == 2025

        # ── Final RL model ─────────────────────────────────────────────────
        rl_tr = final_train_mask & df["home_covers_rl"].notna()
        rl_vl = final_val_mask   & df["home_covers_rl"].notna()
        X_tr = X_all[rl_tr];  y_tr = df.loc[rl_tr, "home_covers_rl"].astype(int)
        X_vl = X_all[rl_vl];  y_vl = df.loc[rl_vl, "home_covers_rl"].astype(int)

        neg = (y_tr == 0).sum();  pos = (y_tr == 1).sum()
        rl_scale = neg / max(pos, 1)
        print(f"\n  [1/3] Run-line | n_train={rl_tr.sum()} | "
              f"scale_pos_weight={rl_scale:.2f}")
        rl_model = train_binary(X_tr, y_tr, X_vl, y_vl,
                                label="Final-RL", early_stop=early_stop,
                                scale_pos=rl_scale)

        # ── Final Total model ──────────────────────────────────────────────
        tot_tr = final_train_mask & df["total_runs"].notna()
        tot_vl = final_val_mask   & df["total_runs"].notna()
        Xt_tr = X_all[tot_tr];  yt_tr = df.loc[tot_tr, "total_runs"]
        Xt_vl = X_all[tot_vl];  yt_vl = df.loc[tot_vl, "total_runs"]
        print(f"\n  [2/3] Total runs | n_train={tot_tr.sum()}")
        tot_model = train_regression(Xt_tr, yt_tr, Xt_vl, yt_vl,
                                     label="Final-TOTAL", early_stop=early_stop)

        # ── Final ML model ─────────────────────────────────────────────────
        ml_tr = final_train_mask & df["actual_home_win"].notna()
        ml_vl = final_val_mask   & df["actual_home_win"].notna()
        Xm_tr = X_all[ml_tr];  ym_tr = df.loc[ml_tr, "actual_home_win"].astype(int)
        Xm_vl = X_all[ml_vl];  ym_vl = df.loc[ml_vl, "actual_home_win"].astype(int)
        ml_scale = (ym_tr == 0).sum() / max((ym_tr == 1).sum(), 1)
        print(f"\n  [3/3] Moneyline | n_train={ml_tr.sum()} | "
              f"scale_pos_weight={ml_scale:.2f}")
        ml_model = train_binary(Xm_tr, ym_tr, Xm_vl, ym_vl,
                                label="Final-ML", early_stop=early_stop,
                                scale_pos=ml_scale)

        # Predictions on 2025 (last available labeled year) for the val CSV
        rl_probs = rl_model.predict_proba(X_vl)[:, 1]
        tot_preds = tot_model.predict(Xt_vl)
        ml_probs  = ml_model.predict_proba(X_all[rl_vl])[:, 1]

        val_df_for_edge = df[rl_vl].copy()
        val_df_for_edge["rl_prob"]  = rl_probs
        val_df_for_edge["tot_pred"] = tot_preds
        val_df_for_edge["ml_prob"]  = ml_probs

        print(f"\n  Final model NCV-unbiased metrics come from xgb_ncv_results.csv.")
        print(f"  The 2025 eval set above was used only for early-stopping signal.")

        # ── Step 3: Fit calibrators on pooled OOF predictions ─────────────
        # OOF predictions have zero leakage — each game was predicted by a
        # model that was trained on strictly earlier seasons.
        print(f"\n{'='*60}")
        print(f"  ISOTONIC CALIBRATION  (pooled OOF: 2024 + 2025 fold predictions)")
        print(f"{'='*60}")

        cal_rl = fit_and_save_calibrator(
            oof_probs_rl, oof_labels_rl,
            MODELS_DIR / "calibrator_rl.pkl", label="run_line")
        calibration_report(oof_probs_rl, oof_labels_rl, cal_rl, label="run_line")

        if len(oof_probs_ml) > 0:
            cal_ml = fit_and_save_calibrator(
                oof_probs_ml, oof_labels_ml,
                MODELS_DIR / "calibrator_ml.pkl", label="moneyline")
            calibration_report(oof_probs_ml, oof_labels_ml, cal_ml, label="moneyline")

    # =========================================================================
    # STANDARD MODE (original behaviour)
    # =========================================================================
    else:
        train_mask = df["split"] == "train"
        val_mask   = df["split"] == "val"

        # RL model
        rl_mask_train = train_mask & df["home_covers_rl"].notna()
        rl_mask_val   = val_mask   & df["home_covers_rl"].notna()
        X_rl_train = X_all[rl_mask_train]
        y_rl_train = df.loc[rl_mask_train, "home_covers_rl"].astype(int)
        X_rl_val   = X_all[rl_mask_val]
        y_rl_val   = df.loc[rl_mask_val,   "home_covers_rl"].astype(int)

        # Total model
        tot_mask_train = train_mask & df["total_runs"].notna()
        tot_mask_val   = val_mask   & df["total_runs"].notna()
        X_tot_train = X_all[tot_mask_train]
        y_tot_train = df.loc[tot_mask_train, "total_runs"]
        X_tot_val   = X_all[tot_mask_val]
        y_tot_val   = df.loc[tot_mask_val,   "total_runs"]

        # ML model
        ml_mask_train = train_mask & df["actual_home_win"].notna()
        ml_mask_val   = val_mask   & df["actual_home_win"].notna()
        X_ml_train = X_all[ml_mask_train]
        y_ml_train = df.loc[ml_mask_train, "actual_home_win"].astype(int)
        X_ml_val   = X_all[ml_mask_val]
        y_ml_val   = df.loc[ml_mask_val,   "actual_home_win"].astype(int)

        print(f"\n  Train: {rl_mask_train.sum()} games (RL labeled)")
        print(f"  Val  : {rl_mask_val.sum()} games (RL labeled)")
        print(f"  Class balance: home_covers_rl="
              f"{y_rl_train.mean():.3f} train | {y_rl_val.mean():.3f} val")

        neg = (y_rl_train == 0).sum();  pos = (y_rl_train == 1).sum()
        rl_scale = neg / max(pos, 1)
        print(f"  RL scale_pos_weight: {rl_scale:.2f}")

        print(f"\n{'='*60}")
        print(f"  Training models (early_stop={'ON' if early_stop else 'OFF'})")
        print(f"{'='*60}")

        print("\n  [1/3] Run-line (home -1.5 covers) classifier ...")
        rl_model = train_binary(
            X_rl_train, y_rl_train, X_rl_val, y_rl_val,
            label="RL", early_stop=early_stop, scale_pos=rl_scale)

        print("\n  [2/3] Total runs regressor ...")
        tot_model = train_regression(
            X_tot_train, y_tot_train, X_tot_val, y_tot_val,
            label="TOTAL", early_stop=early_stop)

        ml_scale = (y_ml_train == 0).sum() / max((y_ml_train == 1).sum(), 1)
        print("\n  [3/3] Moneyline (home win) classifier ...")
        ml_model = train_binary(
            X_ml_train, y_ml_train, X_ml_val, y_ml_val,
            label="ML", early_stop=early_stop, scale_pos=ml_scale)

        # Assign val aliases for shared code below
        X_rl_val_ = X_rl_val;  y_rl_val_ = y_rl_val
        X_tot_val_ = X_tot_val; y_tot_val_ = y_tot_val
        X_ml_val_ = X_ml_val;  y_ml_val_ = y_ml_val
        val_mask_for_edge = val_mask

        # --- Validation metrics ----------------------------------------------
        print(f"\n{'='*60}")
        print(f"  Validation Metrics ({args.val_year} games)")
        print(f"{'='*60}")

        rl_probs = rl_model.predict_proba(X_rl_val_)[:, 1]
        log_metrics("Run-Line (home covers -1.5)", y_rl_val_, y_pred_prob=rl_probs)

        tot_preds = tot_model.predict(X_tot_val_)
        log_metrics("Total Runs", y_tot_val_, y_pred_reg=tot_preds)

        ml_probs_full = ml_model.predict_proba(X_ml_val_)[:, 1]
        log_metrics("Moneyline (home wins)", y_ml_val_, y_pred_prob=ml_probs_full)

        ml_probs = ml_model.predict_proba(X_rl_val_)[:, 1]

        val_df_for_edge = df[val_mask_for_edge].copy()
        val_df_for_edge = val_df_for_edge[val_df_for_edge["home_covers_rl"].notna()].copy()
        val_df_for_edge["rl_prob"]  = rl_probs
        val_df_for_edge["tot_pred"] = tot_preds
        val_df_for_edge["ml_prob"]  = ml_probs

        if "vegas_implied_home" in val_df_for_edge.columns:
            vi = val_df_for_edge["vegas_implied_home"]
            vi_valid = vi.dropna()
            if len(vi_valid) > 100:
                merged_ml = val_df_for_edge.dropna(
                    subset=["vegas_implied_home", "actual_home_win"])
                merged_ml["actual_home_win"] = (
                    pd.to_numeric(merged_ml["actual_home_win"], errors="coerce")
                    .fillna(0).astype(int))
                if len(merged_ml) > 100:
                    vegas_auc = roc_auc_score(
                        merged_ml["actual_home_win"],
                        merged_ml["vegas_implied_home"])
                    print(f"\n  Vegas implied prob AUC (moneyline): {vegas_auc:.4f} "
                          f"(baseline to beat)")

        print(f"\n{'='*60}")
        print(f"  Run-line edge analysis ({args.val_year} validation)")
        print(f"{'='*60}")
        edge_analysis(
            val_df_for_edge, "rl_prob",
            label_col="home_covers_rl",
            thresholds=[0.48, 0.50, 0.52, 0.525, 0.54, 0.56, 0.58, 0.60])

        print(f"\n  Total runs accuracy (within 1.5 runs): "
              f"{(np.abs(tot_preds - y_tot_val_.values) <= 1.5).mean():.3f}")

        # Fit calibrators on 2025 val set (standard mode)
        print(f"\n{'='*60}")
        print(f"  ISOTONIC CALIBRATION  (2025 val set)")
        print(f"{'='*60}")
        print(f"  Note: In standard mode the val set was also used for early stopping.")
        print(f"  Use --ncv for zero-leakage OOF calibration.")
        cal_rl = fit_and_save_calibrator(
            rl_probs, y_rl_val_.values,
            MODELS_DIR / "calibrator_rl.pkl", label="run_line")
        calibration_report(rl_probs, y_rl_val_.values, cal_rl, label="run_line")

        if len(ml_probs_full) == len(y_ml_val_.values):
            cal_ml = fit_and_save_calibrator(
                ml_probs_full, y_ml_val_.values,
                MODELS_DIR / "calibrator_ml.pkl", label="moneyline")
            calibration_report(ml_probs_full, y_ml_val_.values, cal_ml, label="moneyline")

    # =========================================================================
    # SHARED — Feature importances + save models + val predictions
    # =========================================================================

    print(f"\n{'='*60}")
    print(f"  Feature Importances (Run-line model, gain)")
    print(f"{'='*60}")

    fi_rl = pd.DataFrame({
        "feature": feature_cols,
        "gain":    rl_model.feature_importances_,
    }).sort_values("gain", ascending=False)

    fi_tot = pd.DataFrame({
        "feature": feature_cols,
        "gain":    tot_model.feature_importances_,
    }).sort_values("gain", ascending=False)

    fi_rl["model"]  = "run_line"
    fi_tot["model"] = "total"

    fi_combined = pd.concat([fi_rl, fi_tot], ignore_index=True)
    fi_combined.to_csv("xgb_feature_importance.csv", index=False)

    print(f"\n  Top-15 run-line features:")
    for _, row in fi_rl.head(15).iterrows():
        print(f"    {row['feature']:<45} {row['gain']:.4f}")

    print(f"\n  Top-15 total-runs features:")
    for _, row in fi_tot.head(15).iterrows():
        print(f"    {row['feature']:<45} {row['gain']:.4f}")

    # --- Save models ---------------------------------------------------------
    rl_path  = MODELS_DIR / "xgb_rl.json"
    tot_path = MODELS_DIR / "xgb_total.json"
    ml_path  = MODELS_DIR / "xgb_ml.json"

    rl_model.save_model(str(rl_path))
    tot_model.save_model(str(tot_path))
    ml_model.save_model(str(ml_path))

    print(f"\n  Saved:")
    print(f"    {rl_path}")
    print(f"    {tot_path}")
    print(f"    {ml_path}")

    # --- Save val predictions ------------------------------------------------
    val_preds_path = "xgb_val_predictions.csv"

    _save_cols = [
        "game_date", "home_team", "away_team",
        "home_starter_name", "away_starter_name",
        "home_score", "away_score", "home_margin",
        "home_covers_rl", "actual_home_win", "total_runs",
        "rl_prob", "tot_pred", "ml_prob",
        "vegas_implied_home",
    ]
    _save_cols = [c for c in _save_cols if c in val_df_for_edge.columns]
    val_df_for_edge[_save_cols].to_csv(val_preds_path, index=False)
    print(f"    {val_preds_path}")

    # --- Calibration summary (shared) ----------------------------------------
    print(f"\n{'='*60}")
    print(f"  Calibration check (model prob buckets vs actual RL cover %)")
    print(f"{'='*60}")
    print(f"  {'prob_bucket':>12}  {'n':>6}  {'actual_cover':>12}  {'model_mean':>10}")
    print(f"  {'-'*45}")
    vdf = val_df_for_edge.copy()
    vdf["prob_bucket"] = pd.cut(vdf["rl_prob"],
                                bins=[0, 0.3, 0.35, 0.4, 0.45, 0.5,
                                      0.55, 0.6, 0.65, 1.0],
                                labels=["<.30", ".30-.35", ".35-.40", ".40-.45",
                                        ".45-.50", ".50-.55", ".55-.60",
                                        ".60-.65", ">.65"])
    for bucket, grp in vdf.groupby("prob_bucket", observed=True):
        n   = len(grp)
        act = grp["home_covers_rl"].mean()
        mod = grp["rl_prob"].mean()
        flag = " <-- miscalibrated" if abs(act - mod) > 0.08 else ""
        print(f"  {str(bucket):>12}  {n:>6}  {act:>12.3f}  {mod:>10.3f}{flag}")

    if args.ncv:
        print(f"\n{'='*60}")
        print(f"  NCV COMPLETE")
        print(f"  Cross-validated metrics are in: xgb_ncv_results.csv")
        print(f"  Production models trained on: 2023 + 2024 + 2025")
        print(f"  2026 performance = true out-of-sample test")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
