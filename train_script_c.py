"""
train_script_c.py — Script C: The Elite Duel.

Joint SGP: Game Total Under  AND  Both SP F5 K>=3  AND  Close Game (|margin|<=1).

Thesis: when two elite arms face off in a neutral or slightly favoring park,
the probability that all three legs hit is substantially higher than the
product of individual probabilities. The "Underdog RL +1.5 covers" leg
(proxied by |margin|<=1) is underpriced because books don't model the
correlation between pitcher dominance and game tightness.

Outputs:
    models/script_c_v1.json
    data/logs/script_c_v1_metrics.txt
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

LABELS_FILE = Path("data/batter_features/script_labels.parquet")
FEATURE_MTX = Path("feature_matrix_enriched_v2.parquet")
OUT_MODEL   = Path("models/script_c_v1.json")
OUT_TXT     = Path("data/logs/script_c_v1_metrics.txt")

FEATURES = [
    # Both SP quality — Script C requires BOTH starters to be elite
    "home_sp_k_pct", "home_sp_whiff_pctl", "home_sp_ff_velo",
    "home_sp_k_pct_10d", "home_sp_xwoba_against", "home_sp_gb_pct",
    "away_sp_k_pct", "away_sp_whiff_pctl", "away_sp_ff_velo",
    "away_sp_k_pct_10d", "away_sp_xwoba_against", "away_sp_gb_pct",
    # SP matchup edge
    "sp_k_pct_diff", "sp_xwoba_diff", "sp_velo_diff",
    # Competitive game indicators
    "elo_diff", "batting_matchup_edge",
    # Environment (pitcher-friendly = under + tight)
    "home_park_factor", "temp_f", "air_density_rho", "roof_closed_flag",
    "wind_mph",
    # Human element
    "ump_k_above_avg", "ump_called_strike_above_avg",
    "home_catcher_framing_runs", "away_catcher_framing_runs",
    # Bullpen quality (both fresh = more likely duel carries into late game)
    "home_bullpen_vulnerability", "away_bullpen_vulnerability",
    # Opponent lineup strikeout rate
    "away_bat_k_vs_rhp", "away_bat_k_vs_lhp",
    "home_bat_k_vs_rhp", "home_bat_k_vs_lhp",
]


def _load_features() -> pd.DataFrame:
    labels = pd.read_parquet(LABELS_FILE)
    labels = labels.dropna(subset=["y_c"])

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


def main(val_year: int = 2025):
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

    print("=" * 60)
    print("  train_script_c.py — Script C: Elite Duel")
    print("=" * 60)

    data, feat_cols = _load_features()

    y_under = data["y_c_under"]
    y_both  = data["y_c_both_k"]
    y_close = data["y_c_close"]
    y       = data["y_c"]
    p_indep = y_under.mean() * y_both.mean() * y_close.mean()
    p_joint = y.mean()
    corr_ratio = p_joint / p_indep if p_indep > 0 else float("nan")
    print(f"\n  Correlation Stats:")
    print(f"    P(Under line):      {y_under.mean():.4f}")
    print(f"    P(Both SP K_F5>=3): {y_both.mean():.4f}")
    print(f"    P(Close game <=1):  {y_close.mean():.4f}")
    print(f"    P(joint,actual):    {p_joint:.4f}")
    print(f"    P(joint,indep):     {p_indep:.4f}")
    print(f"    Corr ratio:         {corr_ratio:.3f}  "
          f"({'EDGE' if corr_ratio > 1 else 'AVOID'})")

    is_valid = data["year"] == val_year
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
        script="C_EliteDuel",
        legs="Game_Under | Both_SP_F5_K>=3 | Close_Game(|margin|<=1)",
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

    print(f"\n=== Script C Validation ===")
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
        f.write("Script C: Elite Duel\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")
        f.write(imp_df.to_string(index=False))
    print(f"\n  Model -> {OUT_MODEL}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Script C — Elite Duel SGP")
    parser.add_argument("--val-year", type=int, default=2025)
    args = parser.parse_args()
    main(val_year=args.val_year)
