"""
train_script_b.py — Script B: The Offensive Explosion.

Joint SGP: Home Team Scores >= 5  AND  Game Total Over  AND  Home ML Win.

Thesis: when park/wind/matchup strongly favors offense, all three legs correlate
positively. Books price as independent legs: their SGP is systematically cheap.

Note: "Star hitter TB/Hits Over" would require batter-grain scoring (see
score_sgp_today.py); at game-grain we proxy with home_score >= 5.

Outputs:
    models/script_b_v1.json
    data/logs/script_b_v1_metrics.txt
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

LABELS_FILE = Path("data/batter_features/script_labels.parquet")
FEATURE_MTX = Path("feature_matrix_enriched_v2.parquet")
OUT_MODEL   = Path("models/script_b_v1.json")
OUT_TXT     = Path("data/logs/script_b_v1_metrics.txt")

FEATURES = [
    # Offensive environment
    "home_park_factor", "temp_f", "air_density_rho", "roof_closed_flag",
    "wind_mph",
    # Home lineup quality
    "home_lineup_wrc_plus", "home_lineup_xwoba_vs_rhp", "home_lineup_xwoba_vs_lhp",
    "home_team_barrel_pct_15g", "home_team_woba_15g", "home_team_xwoba_off_15g",
    # Away pitcher (who home team faces)
    "away_sp_k_pct", "away_sp_xwoba_against", "away_sp_gb_pct",
    "away_sp_ff_velo", "away_sp_whiff_pctl", "away_sp_bb_pct",
    # Away bullpen (deteriorates as game goes on)
    "away_bullpen_vulnerability", "away_bp_fatigue_72h",
    # Game state
    "close_total", "elo_diff",
    # Home team form
    "home_pyth_win_pct_15g", "home_rolling_rd_15g",
    # Umpire
    "ump_rpg_above_avg",
    # Market signal
    "home_catcher_framing_runs",
]


def _load_features() -> pd.DataFrame:
    labels = pd.read_parquet(LABELS_FILE)
    labels = labels.dropna(subset=["y_b"])

    fm = pd.read_parquet(FEATURE_MTX)
    fm["game_date"] = pd.to_datetime(fm["game_date"])
    fm = fm.drop_duplicates("game_pk")

    data = labels.merge(fm, on="game_pk", how="inner", suffixes=("_lbl", "_fm"))
    for col in ("game_date", "year"):
        if f"{col}_lbl" in data.columns:
            data[col] = data[f"{col}_lbl"]
            data = data.drop(columns=[f"{col}_lbl", f"{col}_fm"], errors="ignore")
    data["year"] = pd.to_datetime(data["game_date"]).dt.year

    feat_cols = FEATURES
    for c in feat_cols:
        if c not in data.columns:
            data[c] = np.nan
        data[c] = pd.to_numeric(data[c], errors="coerce")

    return data, feat_cols


def main():
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

    print("=" * 60)
    print("  train_script_b.py — Script B: Offensive Explosion")
    print("=" * 60)

    data, feat_cols = _load_features()

    y_team = data["y_b_team"]
    y_over = data["y_b_over"]
    y_win  = data["y_b_win"]
    y      = data["y_b"]
    p_indep = y_team.mean() * y_over.mean() * y_win.mean()
    p_joint = y.mean()
    corr_ratio = p_joint / p_indep if p_indep > 0 else float("nan")
    print(f"\n  Correlation Stats:")
    print(f"    P(Home Score>=5): {y_team.mean():.4f}")
    print(f"    P(Game Over):     {y_over.mean():.4f}")
    print(f"    P(Home Win):      {y_win.mean():.4f}")
    print(f"    P(joint,actual):  {p_joint:.4f}")
    print(f"    P(joint,indep):   {p_indep:.4f}")
    print(f"    Corr ratio:       {corr_ratio:.3f}  "
          f"({'EDGE' if corr_ratio > 1 else 'AVOID'})")

    is_valid = data["year"] == 2025
    X_tr = data.loc[~is_valid, feat_cols]
    X_va = data.loc[is_valid,  feat_cols]
    y_tr = y[~is_valid]
    y_va = y[is_valid]
    print(f"\n  Train {len(X_tr):,} pos={y_tr.mean():.3f}  "
          f"Valid {len(X_va):,} pos={y_va.mean():.3f}")

    dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_cols)
    dva = xgb.DMatrix(X_va, label=y_va, feature_names=feat_cols)
    params = dict(objective="binary:logistic", eval_metric=["logloss", "auc"],
                  tree_method="hist", max_depth=4, eta=0.05,
                  subsample=0.80, colsample_bytree=0.80,
                  min_child_weight=5, reg_lambda=1.0, seed=42)
    booster = xgb.train(params, dtr, num_boost_round=600,
                        evals=[(dtr, "train"), (dva, "valid")],
                        early_stopping_rounds=40, verbose_eval=50)

    pred = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
    metrics = dict(
        script="B_Explosion",
        legs="Home_Score>=5 | Game_Total_Over | Home_ML_Win",
        rows_train=int(len(X_tr)), rows_valid=int(len(X_va)),
        auc=float(roc_auc_score(y_va, pred)),
        logloss=float(log_loss(y_va, pred)),
        brier=float(brier_score_loss(y_va, pred)),
        best_iter=int(booster.best_iteration),
        base_rate_valid=float(y_va.mean()),
        corr_ratio=float(corr_ratio),
        p_joint_actual=float(p_joint),
        p_joint_indep=float(p_indep),
    )

    print(f"\n=== Script B Validation ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    imp = booster.get_score(importance_type="gain")
    imp_df = (pd.DataFrame({"feature": list(imp), "gain": list(imp.values())})
              .sort_values("gain", ascending=False).reset_index(drop=True))
    print("\n=== Feature Importance ===")
    print(imp_df.head(10).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(OUT_MODEL))
    with OUT_TXT.open("w") as f:
        f.write("Script B: Offensive Explosion\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")
        f.write(imp_df.to_string(index=False))
    print(f"\n  Model -> {OUT_MODEL}")
    return metrics


if __name__ == "__main__":
    main()
