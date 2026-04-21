"""
train_f5_challenger.py
======================
Challenger to the current F5 ML classifier.

The incumbent predicts P(home covers F5) as a binary classification target
(f5_home_cover = home wins F5 or ties). This challenger takes a distributional
regression approach: fit two independent XGBoost Poisson regressors for
home-F5-runs and away-F5-runs, then derive the cover probability by convolving
the two predicted Poisson distributions:

    P(home_cover) = sum_{h,a: h >= a} Poisson(h; lambda_home) * Poisson(a; lambda_away)

This preserves margin-of-variance information that the binary target discards
and lets the model learn the run-scoring environment directly. It is trained
and evaluated standalone (no stacking, no incumbent dependencies), then
compared head-to-head in eval_f5_challenger.py.

Split:
  - Train: 2024 regular season
  - Val:   2025 regular season  (matches incumbent's f5_val_predictions.csv)

Inputs:
  feature_matrix.parquet             (train_xgboost.py's canonical matrix)
  data/statcast/statcast_2024.parquet
  data/statcast/statcast_2025.parquet
  models/f5_feature_cols.json        (feature set used by incumbent F5 model)

Outputs:
  models/f5_challenger_home.json     Poisson regressor for home F5 runs
  models/f5_challenger_away.json     Poisson regressor for away F5 runs
  models/f5_challenger_meta.json     feature list + metadata
  f5_challenger_val_predictions.csv  per-game predictions on 2025 holdout

Usage:
  python train_f5_challenger.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ROOT           = Path(__file__).parent
MATRIX_PATH    = ROOT / "feature_matrix.parquet"
STATCAST_DIR   = ROOT / "data" / "statcast"
FEAT_COLS_PATH = ROOT / "models" / "f5_feature_cols.json"

OUT_HOME       = ROOT / "models" / "f5_challenger_home.json"
OUT_AWAY       = ROOT / "models" / "f5_challenger_away.json"
OUT_META       = ROOT / "models" / "f5_challenger_meta.json"
OUT_VAL_PREDS  = ROOT / "f5_challenger_val_predictions.csv"

TRAIN_YEARS = [2024]
VAL_YEARS   = [2025]

XGB_PARAMS = {
    "objective":        "count:poisson",
    "eval_metric":      "poisson-nloglik",
    "tree_method":      "hist",
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 5,
    "reg_lambda":       1.0,
    "max_delta_step":   0.7,  # helps stability of Poisson regression
}
N_ROUNDS    = 2000
EARLY_STOP  = 75


# ---------------------------------------------------------------------------
# F5 SPLIT EXTRACTION
# ---------------------------------------------------------------------------
def extract_f5_splits(year: int) -> pd.DataFrame:
    """Return per-game f5_home_runs / f5_away_runs for a season."""
    path = STATCAST_DIR / f"statcast_{year}.parquet"
    sc = pd.read_parquet(
        path,
        columns=["game_pk", "inning", "post_home_score", "post_away_score"],
    )
    f5 = sc[sc["inning"] <= 5]
    agg = f5.groupby("game_pk", as_index=False).agg(
        f5_home_runs=("post_home_score", "max"),
        f5_away_runs=("post_away_score", "max"),
    )
    agg["f5_home_runs"] = agg["f5_home_runs"].fillna(0).astype(float)
    agg["f5_away_runs"] = agg["f5_away_runs"].fillna(0).astype(float)
    agg["season"] = year
    return agg


def build_labeled_matrix() -> pd.DataFrame:
    df = pd.read_parquet(MATRIX_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["year"] = df["game_date"].dt.year

    splits = pd.concat(
        [extract_f5_splits(y) for y in TRAIN_YEARS + VAL_YEARS],
        ignore_index=True,
    )
    print(f"  Extracted F5 splits for {len(splits):,} games")

    df = df.merge(splits[["game_pk", "f5_home_runs", "f5_away_runs"]],
                  on="game_pk", how="left")
    return df


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
def load_features() -> list[str]:
    with open(FEAT_COLS_PATH) as fh:
        feats = json.load(fh)
    # Drop features that aren't present in feature_matrix.parquet.
    df = pd.read_parquet(MATRIX_PATH, columns=None)
    available = [c for c in feats if c in df.columns]
    dropped = [c for c in feats if c not in df.columns]
    if dropped:
        print(f"  [note] dropping {len(dropped)} features not in matrix: {dropped}")
    return available


def train_poisson(X_tr, y_tr, X_vl, y_vl, label: str) -> xgb.Booster:
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dvl = xgb.DMatrix(X_vl, label=y_vl)
    booster = xgb.train(
        XGB_PARAMS,
        dtr,
        num_boost_round=N_ROUNDS,
        evals=[(dtr, "train"), (dvl, "val")],
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=100,
    )
    print(f"  [{label}] best_iter={booster.best_iteration} "
          f"best_val_nll={booster.best_score:.4f}")
    return booster


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("Loading feature matrix and F5 splits...")
    df = build_labeled_matrix()

    feats = load_features()
    print(f"  Using {len(feats)} features")

    mask_tr = df["year"].isin(TRAIN_YEARS) & df["f5_home_runs"].notna()
    mask_vl = df["year"].isin(VAL_YEARS)   & df["f5_home_runs"].notna()
    print(f"  Train rows: {mask_tr.sum()}   Val rows: {mask_vl.sum()}")

    X_tr = df.loc[mask_tr, feats].astype(float)
    X_vl = df.loc[mask_vl, feats].astype(float)

    y_home_tr = df.loc[mask_tr, "f5_home_runs"].astype(float)
    y_home_vl = df.loc[mask_vl, "f5_home_runs"].astype(float)
    y_away_tr = df.loc[mask_tr, "f5_away_runs"].astype(float)
    y_away_vl = df.loc[mask_vl, "f5_away_runs"].astype(float)

    print("\nTraining HOME Poisson regressor...")
    mdl_home = train_poisson(X_tr, y_home_tr, X_vl, y_home_vl, "home")
    print("\nTraining AWAY Poisson regressor...")
    mdl_away = train_poisson(X_tr, y_away_tr, X_vl, y_away_vl, "away")

    # Predictions on validation set
    dvl = xgb.DMatrix(X_vl)
    lam_home = mdl_home.predict(dvl, iteration_range=(0, mdl_home.best_iteration + 1))
    lam_away = mdl_away.predict(dvl, iteration_range=(0, mdl_away.best_iteration + 1))

    val_df = df.loc[mask_vl, ["game_pk", "game_date", "home_team", "away_team",
                              "f5_home_runs", "f5_away_runs"]].reset_index(drop=True)
    val_df["lam_home"] = lam_home
    val_df["lam_away"] = lam_away
    val_df["pred_f5_total"] = lam_home + lam_away
    val_df["actual_f5_total"] = val_df["f5_home_runs"] + val_df["f5_away_runs"]

    val_df.to_csv(OUT_VAL_PREDS, index=False)
    print(f"\n  Wrote val predictions -> {OUT_VAL_PREDS}")

    mdl_home.save_model(str(OUT_HOME))
    mdl_away.save_model(str(OUT_AWAY))

    meta = {
        "features":          feats,
        "train_years":       TRAIN_YEARS,
        "val_years":         VAL_YEARS,
        "xgb_params":        XGB_PARAMS,
        "home_best_iter":    int(mdl_home.best_iteration),
        "home_best_val_nll": float(mdl_home.best_score),
        "away_best_iter":    int(mdl_away.best_iteration),
        "away_best_val_nll": float(mdl_away.best_score),
        "train_rows":        int(mask_tr.sum()),
        "val_rows":          int(mask_vl.sum()),
    }
    with open(OUT_META, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  Wrote metadata -> {OUT_META}")

    print("\nVal-set sanity check:")
    print(f"  Actual home runs mean: {val_df['f5_home_runs'].mean():.3f}   "
          f"pred lam_home mean: {val_df['lam_home'].mean():.3f}")
    print(f"  Actual away runs mean: {val_df['f5_away_runs'].mean():.3f}   "
          f"pred lam_away mean: {val_df['lam_away'].mean():.3f}")
    print(f"  Actual total mean:     {val_df['actual_f5_total'].mean():.3f}   "
          f"pred total mean:     {val_df['pred_f5_total'].mean():.3f}")


if __name__ == "__main__":
    main()
