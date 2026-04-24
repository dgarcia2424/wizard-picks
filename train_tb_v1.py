"""
train_tb_v1.py
--------------
First training run of the Wizard TB (Total Bases) stacker.

Target: P(total_bases > 1.5).
Algorithm: XGBoost binary classifier, time-ordered train/valid split.

Features actually used (what's in the matrix today):
    - pull_side_wind_vector        (physics)
    - projected_total_adj          (grounded env, PIT backfill + live)
    - bias_offset                  (env, signed bias)
    - wind_mph, wind_bearing       (raw wind)
    - batting_order                (lineup slot — stand-in for lineup_fragility)
    - stand_L, stand_R, stand_S    (handedness)
    - pa_count                     (exposure proxy)
    - Statcast xStats (prior season, leakage-safe):
        ba, est_ba, slg, est_slg, woba, est_woba, avg_hit_angle,
        anglesweetspotpercent, max_hit_speed, avg_hit_speed, ev50, fbld,
        ev95percent, brl_percent, brl_pa

Features requested but NOT in the matrix (flagged honestly):
    - velocity_decay_risk  (needs per-start pitcher velo arc; not built)
    - lineup_fragility     (needs injury/rest proxy; not built)

Outputs:
    models/tb_stacker_v1.json           — trained booster
    data/logs/tb_v1_feature_importance.png
    data/logs/tb_v1_metrics.txt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MATRIX = Path("data/batter_features/final_training_matrix.parquet")
OUT_MODEL = Path("models/tb_stacker_v1.json")
OUT_PNG = Path("data/logs/tb_v1_feature_importance.png")
OUT_TXT = Path("data/logs/tb_v1_metrics.txt")

NUMERIC_FEATURES = [
    "pull_side_wind_vector", "projected_total_adj", "bias_offset",
    "wind_mph", "wind_bearing",
    "batting_order", "pa_count",
    "ba", "est_ba", "slg", "est_slg", "woba", "est_woba",
    "avg_hit_angle", "anglesweetspotpercent",
    "max_hit_speed", "avg_hit_speed", "ev50", "fbld",
    "ev95percent", "brl_percent", "brl_pa",
]


def main():
    df = pd.read_parquet(MATRIX)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    # One-hot handedness
    for h in ("L", "R", "S"):
        df[f"stand_{h}"] = (df["stand"].fillna("") == h).astype("int8")

    feat_cols = NUMERIC_FEATURES + ["stand_L", "stand_R", "stand_S"]
    for c in feat_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    y = (df["total_bases"].astype(float) > 1.5).astype("int8")

    # Time-ordered split: 2024+2025 train, 2026 validation.
    is_valid = df["year"] == 2026
    X_train, X_valid = df.loc[~is_valid, feat_cols], df.loc[is_valid, feat_cols]
    y_train, y_valid = y[~is_valid], y[is_valid]
    print(f"Train rows: {len(X_train):,}  |  Valid rows: {len(X_valid):,}")
    print(f"Train pos rate: {y_train.mean():.4f}  |  Valid pos rate: {y_valid.mean():.4f}")

    import xgboost as xgb
    from sklearn.metrics import (roc_auc_score, log_loss, brier_score_loss,
                                  average_precision_score)

    dtr = xgb.DMatrix(X_train, label=y_train, feature_names=feat_cols)
    dva = xgb.DMatrix(X_valid, label=y_valid, feature_names=feat_cols)

    params = dict(
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
        tree_method="hist",
        max_depth=5,
        eta=0.05,
        subsample=0.85,
        colsample_bytree=0.80,
        min_child_weight=10,
        reg_lambda=1.0,
        seed=42,
    )
    booster = xgb.train(
        params, dtr,
        num_boost_round=600,
        evals=[(dtr, "train"), (dva, "valid")],
        early_stopping_rounds=40,
        verbose_eval=50,
    )

    pred = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
    metrics = {
        "rows_train": int(len(X_train)),
        "rows_valid": int(len(X_valid)),
        "auc":    float(roc_auc_score(y_valid, pred)),
        "logloss": float(log_loss(y_valid, pred)),
        "brier":   float(brier_score_loss(y_valid, pred)),
        "ap":      float(average_precision_score(y_valid, pred)),
        "best_iter": int(booster.best_iteration),
        "base_rate_valid": float(y_valid.mean()),
    }
    print("\n=== Validation metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(OUT_MODEL))
    print(f"Saved booster -> {OUT_MODEL}")

    # Feature importance (gain)
    imp = booster.get_score(importance_type="gain")
    imp_df = (pd.DataFrame({"feature": list(imp.keys()),
                             "gain": list(imp.values())})
              .sort_values("gain", ascending=False)
              .reset_index(drop=True))
    imp_df["rank"] = imp_df.index + 1
    print("\n=== Feature importance (gain) ===")
    print(imp_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    pull_rank = int(imp_df.loc[imp_df["feature"] == "pull_side_wind_vector",
                                 "rank"].iloc[0]) if "pull_side_wind_vector" in imp.keys() else -1
    print(f"\npull_side_wind_vector rank: {pull_rank} / {len(imp_df)}")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TXT.open("w") as f:
        f.write("TB Stacker v1 — validation metrics\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nFeature importance (gain)\n")
        f.write(imp_df.to_string(index=False))
        f.write(f"\n\npull_side_wind_vector rank: {pull_rank}\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 0.35 * len(imp_df) + 1))
        ax.barh(imp_df["feature"][::-1], imp_df["gain"][::-1])
        ax.set_xlabel("XGBoost gain")
        ax.set_title(f"TB Stacker v1 — feature importance "
                      f"(AUC={metrics['auc']:.3f})")
        plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=120)
        print(f"Saved importance plot -> {OUT_PNG}")
    except Exception as exc:
        print(f"[warn] plot skipped: {exc}")


if __name__ == "__main__":
    main()
