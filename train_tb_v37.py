"""
train_tb_v37.py — v3.7 retrain with full v3.6 feature set.
Adds opp_sp_ff_velo directly as a feature alongside velocity_decay_risk.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

MATRIX    = Path("data/batter_features/final_training_matrix.parquet")
OUT_MODEL = Path("models/tb_stacker_v37.json")
OUT_PNG   = Path("data/logs/tb_v37_feature_importance.png")
OUT_TXT   = Path("data/logs/tb_v37_metrics.txt")

NUMERIC_FEATURES = [
    "pull_side_wind_vector", "projected_total_adj", "bias_offset",
    "wind_mph", "wind_bearing", "temp_f",
    "velocity_decay_risk", "opp_sp_ff_velo",
    "lineup_fragility", "platoon_same_hand",
    "batting_order", "exp_pa_heuristic",
    "ba", "est_ba", "slg", "est_slg", "woba", "est_woba",
    "avg_hit_angle", "anglesweetspotpercent",
    "max_hit_speed", "avg_hit_speed", "ev50", "fbld",
    "ev95percent", "brl_percent", "brl_pa",
]


def main(val_year: int = 2025):
    df = pd.read_parquet(MATRIX)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    for h in ("L", "R", "S"):
        df[f"stand_{h}"] = (df["stand"].fillna("") == h).astype("int8")

    feat_cols = NUMERIC_FEATURES + ["stand_L", "stand_R", "stand_S"]
    for c in feat_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    assert "pa_count" not in feat_cols
    y = (df["total_bases"].astype(float) > 1.5).astype("int8")
    is_valid = df["year"] == val_year
    X_tr, X_va = df.loc[~is_valid, feat_cols], df.loc[is_valid, feat_cols]
    y_tr, y_va = y[~is_valid], y[is_valid]
    print(f"Train {len(X_tr):,} | Valid {len(X_va):,}")
    print(f"Pos rate  train={y_tr.mean():.4f}  valid={y_va.mean():.4f}")

    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, average_precision_score

    dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_cols)
    dva = xgb.DMatrix(X_va, label=y_va, feature_names=feat_cols)

    params = dict(
        objective="binary:logistic", eval_metric=["logloss", "auc"],
        tree_method="hist", max_depth=5, eta=0.05,
        subsample=0.85, colsample_bytree=0.80,
        min_child_weight=10, reg_lambda=1.0, seed=42,
    )
    booster = xgb.train(
        params, dtr, num_boost_round=800,
        evals=[(dtr, "train"), (dva, "valid")],
        early_stopping_rounds=40, verbose_eval=50,
    )

    pred = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
    metrics = dict(
        rows_train=int(len(X_tr)), rows_valid=int(len(X_va)),
        auc=float(roc_auc_score(y_va, pred)),
        logloss=float(log_loss(y_va, pred)),
        brier=float(brier_score_loss(y_va, pred)),
        ap=float(average_precision_score(y_va, pred)),
        best_iter=int(booster.best_iteration),
        base_rate_valid=float(y_va.mean()),
    )
    print("\n=== v3.7 Validation ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(OUT_MODEL))

    imp = booster.get_score(importance_type="gain")
    imp_df = (pd.DataFrame({"feature": list(imp), "gain": list(imp.values())})
              .sort_values("gain", ascending=False).reset_index(drop=True))
    imp_df["rank"] = imp_df.index + 1
    print("\n=== Feature Importance (gain) ===")
    print(imp_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    with OUT_TXT.open("w") as f:
        f.write("TB Stacker v3.7 metrics\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")
        f.write(imp_df.to_string(index=False))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 0.35 * len(imp_df) + 1))
        ax.barh(imp_df["feature"][::-1], imp_df["gain"][::-1])
        ax.set_xlabel("gain")
        ax.set_title(f"TB v3.7 (AUC={metrics['auc']:.3f})")
        plt.tight_layout(); plt.savefig(OUT_PNG, dpi=120)
        print(f"Plot -> {OUT_PNG}")
    except Exception as e:
        print(f"[warn] plot: {e}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TB Stacker v3.7")
    parser.add_argument("--val-year", type=int, default=2025,
                        help="Holdout year (default: 2025)")
    args = parser.parse_args()
    main(val_year=args.val_year)
