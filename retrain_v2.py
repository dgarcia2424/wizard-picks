"""
retrain_v2.py — Non-destructive v2 retrain with alpha features.

Joins historical_alpha_matrix_24_25.parquet onto feature_matrix_enriched_v2.parquet
(restricted to 2024+2025), trains:
  - Totals  XGBoost regressor on (home_score + away_score)
  - ML      XGBoost classifier on (home_score > away_score)

Artifacts:  models/totals_v2.pkl , models/ml_v2.pkl
Baseline:   trains v1 (no alpha features) on same split for RMSE / AUC delta.

Usage:  python retrain_v2.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
import xgboost as xgb

_ROOT = Path(__file__).resolve().parent
_FM   = _ROOT / "feature_matrix_enriched_v2.parquet"
_ALPHA= _ROOT / "data" / "raw" / "historical_alpha_matrix_24_25.parquet"
_MODELS = _ROOT / "models"

ALPHA_COLS = [
    "wind_vector_out",
    "thermal_aging",
    "bullpen_xfip_diff",
    "days_since_opening_day",
]

# Columns to exclude from the feature set (labels, ids, text).
NON_FEATURE = {
    "game_pk", "game_date", "date", "home_team", "away_team",
    "home_score", "away_score", "home_win", "away_win",
    "actual_game_total", "actual_home_win", "actual_f5_total", "actual_f3_total",
    "actual_f1_total", "home_starter", "away_starter", "season", "year",
    "season_x", "season_y", "home_team_x", "home_team_y", "away_team_x", "away_team_y",
    "game_date_x", "game_date_y",
    # Leakage guards — direct post-game outcomes baked into the enriched matrix.
    "total_runs", "home_covers_rl", "away_covers_rl",
    "home_margin", "away_margin", "margin",
}

XGB_REG_PARAMS = dict(
    n_estimators=600, max_depth=5, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.8, min_child_weight=5,
    objective="reg:squarederror", tree_method="hist",
    random_state=42,
)
XGB_CLF_PARAMS = dict(
    n_estimators=600, max_depth=5, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.8, min_child_weight=5,
    objective="binary:logistic", eval_metric="logloss", tree_method="hist",
    random_state=42,
)


def _load_joined() -> pd.DataFrame:
    fm    = pd.read_parquet(_FM)
    alpha = pd.read_parquet(_ALPHA)

    fm = fm[fm["season"].isin([2024, 2025])].copy()
    fm = fm.dropna(subset=["home_score", "away_score"]).copy()
    print(f"[load] feature_matrix 2024-25 rows = {len(fm):,}")
    print(f"[load] alpha matrix rows           = {len(alpha):,}")

    join = fm.merge(alpha[["game_pk"] + ALPHA_COLS], on="game_pk", how="left")
    print(f"[load] after join                  = {len(join):,}")
    cov = join[ALPHA_COLS].notna().all(axis=1).mean()
    print(f"[load] alpha feature coverage      = {cov:.1%}")
    return join


def _feature_columns(df: pd.DataFrame, include_alpha: bool) -> list[str]:
    cols = []
    for c in df.columns:
        if c in NON_FEATURE:
            continue
        if not include_alpha and c in ALPHA_COLS:
            continue
        if df[c].dtype in (np.float32, np.float64, np.int32, np.int64, "float64", "int64", "bool"):
            cols.append(c)
    return cols


def _split(df: pd.DataFrame):
    tr = df[df["season"] == 2024].copy()
    te = df[df["season"] == 2025].copy()
    print(f"[split] train (2024) = {len(tr):,} | test (2025) = {len(te):,}")
    return tr, te


def _fit_totals(df_tr, df_te, feats):
    y_tr = (df_tr["home_score"] + df_tr["away_score"]).astype(float).values
    y_te = (df_te["home_score"] + df_te["away_score"]).astype(float).values
    m = xgb.XGBRegressor(**XGB_REG_PARAMS)
    m.fit(df_tr[feats].values, y_tr, verbose=False)
    pred = m.predict(df_te[feats].values)
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    return m, rmse


def _fit_ml(df_tr, df_te, feats):
    y_tr = (df_tr["home_score"] > df_tr["away_score"]).astype(int).values
    y_te = (df_te["home_score"] > df_te["away_score"]).astype(int).values
    m = xgb.XGBClassifier(**XGB_CLF_PARAMS)
    m.fit(df_tr[feats].values, y_tr, verbose=False)
    prob = m.predict_proba(df_te[feats].values)[:, 1]
    auc   = float(roc_auc_score(y_te, prob))
    ll    = float(log_loss(y_te, prob))
    return m, auc, ll


def _report_alpha_importance(model, feats):
    gains = model.get_booster().get_score(importance_type="gain")
    # xgboost names features f0, f1, ... by default when fed arrays
    mapped = {feats[int(k[1:])]: v for k, v in gains.items()}
    total_gain = sum(mapped.values()) or 1.0
    rows = []
    for c in ALPHA_COLS:
        g = mapped.get(c, 0.0)
        rows.append({"feature": c, "gain": round(g, 2), "gain_pct": round(100 * g / total_gain, 2)})
    return pd.DataFrame(rows).sort_values("gain", ascending=False)


def main():
    _MODELS.mkdir(exist_ok=True)
    df = _load_joined()
    tr, te = _split(df)

    feats_v1 = _feature_columns(df, include_alpha=False)
    feats_v2 = _feature_columns(df, include_alpha=True)
    print(f"[feats] v1 = {len(feats_v1)} | v2 = {len(feats_v2)} (+{len(feats_v2)-len(feats_v1)} alpha)")

    # -------- TOTALS --------
    print("\n========== TOTALS ==========")
    tot_v1, rmse_v1 = _fit_totals(tr, te, feats_v1)
    tot_v2, rmse_v2 = _fit_totals(tr, te, feats_v2)
    print(f"Totals RMSE  v1 = {rmse_v1:.4f}")
    print(f"Totals RMSE  v2 = {rmse_v2:.4f}  (Δ = {rmse_v2 - rmse_v1:+.4f})")
    imp_totals = _report_alpha_importance(tot_v2, feats_v2)
    print("\nAlpha feature gain (Totals v2):")
    print(imp_totals.to_string(index=False))

    # -------- ML --------
    print("\n========== MONEYLINE ==========")
    ml_v1, auc_v1, ll_v1 = _fit_ml(tr, te, feats_v1)
    ml_v2, auc_v2, ll_v2 = _fit_ml(tr, te, feats_v2)
    print(f"ML AUC     v1 = {auc_v1:.4f}  | LogLoss = {ll_v1:.4f}")
    print(f"ML AUC     v2 = {auc_v2:.4f}  | LogLoss = {ll_v2:.4f}")
    print(f"ΔAUC = {auc_v2-auc_v1:+.4f} | ΔLogLoss = {ll_v2-ll_v1:+.4f}")
    imp_ml = _report_alpha_importance(ml_v2, feats_v2)
    print("\nAlpha feature gain (ML v2):")
    print(imp_ml.to_string(index=False))

    # -------- SAVE v2 ARTIFACTS (do NOT touch v1) --------
    with open(_MODELS / "totals_v2.pkl", "wb") as f:
        pickle.dump({"model": tot_v2, "features": feats_v2,
                     "rmse": rmse_v2, "rmse_baseline": rmse_v1}, f)
    with open(_MODELS / "ml_v2.pkl", "wb") as f:
        pickle.dump({"model": ml_v2, "features": feats_v2,
                     "auc": auc_v2, "logloss": ll_v2}, f)

    meta = {
        "train_season": 2024, "test_season": 2025,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "totals_rmse_v1": rmse_v1, "totals_rmse_v2": rmse_v2,
        "ml_auc_v1": auc_v1, "ml_auc_v2": auc_v2,
        "ml_logloss_v1": ll_v1, "ml_logloss_v2": ll_v2,
        "alpha_gain_totals": imp_totals.to_dict("records"),
        "alpha_gain_ml":     imp_ml.to_dict("records"),
        "alpha_features":    ALPHA_COLS,
    }
    with open(_MODELS / "v2_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[save] totals_v2.pkl, ml_v2.pkl, v2_meta.json → {_MODELS}")


if __name__ == "__main__":
    main()
