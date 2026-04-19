"""
train_xgboost.py
================
Train XGBoost models on the feature matrix.

Models trained:
  1. home_covers_rl   (binary: home team covers -1.5 run line)
  2. total_runs       (regression: total runs scored)
  3. actual_home_win  (binary: home team wins outright — auxiliary)

Training split  : 2023 + 2024 games
Validation split: 2025 games

Outputs:
  models/xgb_rl.json           XGBoost run-line model (binary)
  models/xgb_total.json        XGBoost total runs model (regression)
  models/xgb_ml.json           XGBoost moneyline model (binary)
  models/feature_cols.json     ordered feature column list
  xgb_val_predictions.csv      validation set predictions
  xgb_feature_importance.csv   gain-based feature importances

Usage:
  python train_xgboost.py
  python train_xgboost.py --matrix feature_matrix.parquet
  python train_xgboost.py --no-early-stop  # fixed 500 estimators
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
BREAKEVEN_PROB = 110 / 210   # ≈ 0.5238


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
        # Compute model vs Vegas deviation
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
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=str, default="feature_matrix.parquet",
                        help="Feature matrix parquet file")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping (use all n_estimators)")
    parser.add_argument("--val-year", type=int, default=2025,
                        help="Validation year (default: 2025)")
    args = parser.parse_args()

    early_stop = not args.no_early_stop

    print("=" * 60)
    print("  train_xgboost.py")
    print("=" * 60)

    # --- Load data -----------------------------------------------------------
    df = pd.read_parquet(args.matrix, engine="pyarrow")
    print(f"\n  Loaded {len(df)} rows x {len(df.columns)} cols")

    # Prepare features
    X_all, feature_cols = prep_features(df)
    print(f"  Feature columns: {len(feature_cols)}")

    # Save feature column order for inference
    (MODELS_DIR / "feature_cols.json").write_text(
        json.dumps(feature_cols, indent=2))

    # Train / val splits (drop rows with NaN labels)
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

    # ML (moneyline) model
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

    # Scale pos weight for imbalanced RL target
    neg = (y_rl_train == 0).sum()
    pos = (y_rl_train == 1).sum()
    rl_scale = neg / max(pos, 1)
    print(f"  RL scale_pos_weight: {rl_scale:.2f}")

    # --- Train models --------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Training models (early_stop={'ON' if early_stop else 'OFF'})")
    print(f"{'='*60}")

    # Run-line model
    print("\n  [1/3] Run-line (home -1.5 covers) classifier ...")
    rl_model = train_binary(
        X_rl_train, y_rl_train, X_rl_val, y_rl_val,
        label="RL", early_stop=early_stop, scale_pos=rl_scale)

    # Total runs model
    print("\n  [2/3] Total runs regressor ...")
    tot_model = train_regression(
        X_tot_train, y_tot_train, X_tot_val, y_tot_val,
        label="TOTAL", early_stop=early_stop)

    # Moneyline model (auxiliary — for comparison)
    ml_scale = (y_ml_train == 0).sum() / max((y_ml_train == 1).sum(), 1)
    print("\n  [3/3] Moneyline (home win) classifier ...")
    ml_model = train_binary(
        X_ml_train, y_ml_train, X_ml_val, y_ml_val,
        label="ML", early_stop=early_stop, scale_pos=ml_scale)

    # --- Validation metrics --------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Validation Metrics (2025 games)")
    print(f"{'='*60}")

    rl_probs = rl_model.predict_proba(X_rl_val)[:, 1]
    log_metrics("Run-Line (home covers -1.5)",
                y_rl_val, y_pred_prob=rl_probs)

    tot_preds = tot_model.predict(X_tot_val)
    log_metrics("Total Runs",
                y_tot_val, y_pred_reg=tot_preds)

    ml_probs_full = ml_model.predict_proba(X_ml_val)[:, 1]
    log_metrics("Moneyline (home wins)",
                y_ml_val, y_pred_prob=ml_probs_full)
    # For combined val dataframe, predict ML on the RL val rows
    ml_probs = ml_model.predict_proba(X_rl_val)[:, 1]

    # Vegas baseline: predict using Vegas implied prob
    val_df_for_edge = df[val_mask].copy()
    val_df_for_edge = val_df_for_edge[val_df_for_edge["home_covers_rl"].notna()].copy()
    val_df_for_edge["rl_prob"] = rl_probs
    val_df_for_edge["tot_pred"] = tot_preds
    val_df_for_edge["ml_prob"]  = ml_probs

    if "vegas_implied_home" in val_df_for_edge.columns:
        vi = val_df_for_edge["vegas_implied_home"]
        vi_valid = vi.dropna()
        if len(vi_valid) > 100:
            # Vegas as predictor of home win
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

    # --- Edge analysis -------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Run-line edge analysis (2025 validation)")
    print(f"{'='*60}")
    edge_analysis(
        val_df_for_edge, "rl_prob",
        label_col="home_covers_rl",
        thresholds=[0.48, 0.50, 0.52, 0.525, 0.54, 0.56, 0.58, 0.60])

    # Total prediction accuracy
    print(f"\n  Total runs accuracy (within 1.5 runs): "
          f"{(np.abs(tot_preds - y_tot_val.values) <= 1.5).mean():.3f}")

    # --- Feature importances -------------------------------------------------
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
    val_df_for_edge[[
        "game_date", "home_team", "away_team",
        "home_starter_name", "away_starter_name",
        "home_score", "away_score", "home_margin",
        "home_covers_rl", "actual_home_win", "total_runs",
        "rl_prob", "tot_pred", "ml_prob",
        "vegas_implied_home",
    ]].to_csv(val_preds_path, index=False)
    print(f"    {val_preds_path}")

    # --- Calibration summary -------------------------------------------------
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


if __name__ == "__main__":
    main()
