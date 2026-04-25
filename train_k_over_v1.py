"""
train_k_over_v1.py — SP K-Over binary classifier.

Target: P(SP Ks >= threshold).  Default threshold = 5.5 (market standard).
Features: SP-level K%, whiff%, BB%, velo, age, arm angle, 10-day rolling K%,
          opposing team K rate, umpire K factor, park factor, temp.
Grain: one row per (game, SP).  Train on 2024, validate on 2025.

Outputs:
    models/k_over_v1.json
    data/logs/k_over_v1_metrics.txt
    data/logs/k_over_v1_importance.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SP_LABELS  = Path("data/batter_features/sp_k_labels.parquet")
FEATURE_MTX = Path("feature_matrix_enriched_v2.parquet")
OUT_MODEL      = Path("models/k_over_v1.json")
OUT_TXT        = Path("data/logs/k_over_v1_metrics.txt")
OUT_PNG        = Path("data/logs/k_over_v1_importance.png")
OUT_VAL_PREDS  = Path("k_val_predictions.csv")
K_THRESHOLD = 5.5

SP_FEATURES = [
    "sp_k_pct", "sp_bb_pct", "sp_whiff_pctl", "sp_ff_velo",
    "sp_age_pit", "sp_arm_angle", "sp_k_pct_10d", "sp_bb_pct_10d",
    "sp_k_minus_bb", "sp_k_bb_ratio", "sp_k_bb_ratio_10d",
    "sp_1st_k_pct", "sp_xwoba_against", "sp_gb_pct",
    "opp_bat_k_rate",   # opposing lineup K rate vs. SP hand
    "ump_k_above_avg",
    "home_park_factor",
    "temp_f",
]


def build_sp_grain(labels: pd.DataFrame, fm: pd.DataFrame) -> pd.DataFrame:
    """Join FM game-level context to SP label rows."""
    fm = fm.copy()
    fm["game_date"] = pd.to_datetime(fm["game_date"]).dt.strftime("%Y-%m-%d")

    # For each SP-game row, determine if they are the home or away SP
    # by matching game_pk. Merge both home and away contexts.
    fm_home = fm.rename(columns={
        c: c.replace("home_sp_", "sp_").replace("away_bat_", "opp_bat_")
        for c in fm.columns
    })
    fm_away = fm.rename(columns={
        c: c.replace("away_sp_", "sp_").replace("home_bat_", "opp_bat_")
        for c in fm.columns
    })

    # Dedupe SP features for home SPs
    home_cols = ["game_pk", "game_date",
                 "sp_k_pct", "sp_bb_pct", "sp_whiff_pctl", "sp_ff_velo",
                 "sp_age_pit", "sp_arm_angle", "sp_k_pct_10d", "sp_bb_pct_10d",
                 "sp_k_minus_bb", "sp_k_bb_ratio", "sp_k_bb_ratio_10d",
                 "sp_1st_k_pct", "sp_xwoba_against", "sp_gb_pct",
                 "ump_k_above_avg", "home_park_factor", "temp_f"]
    # build opp_bat_k_rate per side
    fm_h = fm[["game_pk", "game_date",
               "home_sp_k_pct", "home_sp_bb_pct", "home_sp_whiff_pctl",
               "home_sp_ff_velo", "home_sp_age_pit", "home_sp_arm_angle",
               "home_sp_k_pct_10d", "home_sp_bb_pct_10d", "home_sp_k_minus_bb",
               "home_sp_k_bb_ratio", "home_sp_k_bb_ratio_10d",
               "home_sp_1st_k_pct", "home_sp_xwoba_against", "home_sp_gb_pct",
               "ump_k_above_avg", "home_park_factor", "temp_f",
               "away_bat_k_vs_rhp", "away_bat_k_vs_lhp",
               "home_sp_p_throws_R"]].copy()
    fm_h["opp_bat_k_rate"] = np.where(
        fm_h["home_sp_p_throws_R"] == 1,
        fm_h["away_bat_k_vs_rhp"], fm_h["away_bat_k_vs_lhp"])
    fm_h = fm_h.rename(columns={c: c.replace("home_sp_", "sp_") for c in fm_h.columns})
    fm_h["_side"] = "home"

    fm_a = fm[["game_pk", "game_date",
               "away_sp_k_pct", "away_sp_bb_pct", "away_sp_whiff_pctl",
               "away_sp_ff_velo", "away_sp_age_pit", "away_sp_arm_angle",
               "away_sp_k_pct_10d", "away_sp_bb_pct_10d", "away_sp_k_minus_bb",
               "away_sp_k_bb_ratio", "away_sp_k_bb_ratio_10d",
               "away_sp_1st_k_pct", "away_sp_xwoba_against", "away_sp_gb_pct",
               "ump_k_above_avg", "home_park_factor", "temp_f",
               "home_bat_k_vs_rhp", "home_bat_k_vs_lhp",
               "away_sp_p_throws_R"]].copy()
    fm_a["opp_bat_k_rate"] = np.where(
        fm_a["away_sp_p_throws_R"] == 1,
        fm_a["home_bat_k_vs_rhp"], fm_a["home_bat_k_vs_lhp"])
    fm_a = fm_a.rename(columns={c: c.replace("away_sp_", "sp_") for c in fm_a.columns})
    fm_a["_side"] = "away"

    sp_fm = pd.concat([fm_h, fm_a], ignore_index=True)

    # Labels also need a side indicator to join correctly
    # SP who pitched most PAs: home SP pitches in "top" innings (top = away batting)
    # so home_team in labels == home_team in FM
    labels = labels.copy()
    labels["game_date"] = labels["game_date"].astype(str)
    labels["_side"] = np.where(
        labels["p_throws"].notna(),
        # Determine by pitcher-team mapping is not directly available, use PA stats
        "home", "away"
    )
    # Instead: join on game_pk only, then pick best match on pa_total (most PAs = likely SP)
    joined = labels.merge(
        sp_fm[sp_fm["_side"] == "home"][["game_pk"] + SP_FEATURES],
        on="game_pk", how="left", suffixes=("", "_home")
    )
    # For rows where SP features are null, try away side
    null_mask = joined["sp_k_pct"].isna()
    away_feat = sp_fm[sp_fm["_side"] == "away"][["game_pk"] + SP_FEATURES].copy()
    joined.update(joined.loc[null_mask, ["game_pk"]].merge(
        away_feat, on="game_pk", how="left").set_index(joined.index[null_mask]))

    return joined


def main(val_year: int = 2025, matrix: Path | None = None,
         val_preds_out: Path | None = None, no_save_model: bool = False):
    labels = pd.read_parquet(SP_LABELS)
    labels["year"] = pd.to_datetime(labels["game_date"]).dt.year
    print(f"SP-game labels: {len(labels):,}  K mean={labels['k_total'].mean():.2f}")

    fm_cols = ["game_pk", "game_date", "home_team", "away_team",
               "home_sp_k_pct", "home_sp_bb_pct", "home_sp_whiff_pctl",
               "home_sp_ff_velo", "home_sp_age_pit", "home_sp_arm_angle",
               "home_sp_k_pct_10d", "home_sp_bb_pct_10d", "home_sp_k_minus_bb",
               "home_sp_k_bb_ratio", "home_sp_k_bb_ratio_10d",
               "home_sp_1st_k_pct", "home_sp_xwoba_against", "home_sp_gb_pct",
               "home_sp_p_throws_R",
               "away_sp_k_pct", "away_sp_bb_pct", "away_sp_whiff_pctl",
               "away_sp_ff_velo", "away_sp_age_pit", "away_sp_arm_angle",
               "away_sp_k_pct_10d", "away_sp_bb_pct_10d", "away_sp_k_minus_bb",
               "away_sp_k_bb_ratio", "away_sp_k_bb_ratio_10d",
               "away_sp_1st_k_pct", "away_sp_xwoba_against", "away_sp_gb_pct",
               "away_sp_p_throws_R",
               "home_bat_k_vs_rhp", "home_bat_k_vs_lhp",
               "away_bat_k_vs_rhp", "away_bat_k_vs_lhp",
               "ump_k_above_avg", "home_park_factor", "temp_f"]
    fm_cols = [c for c in fm_cols if c in pd.read_parquet(
        FEATURE_MTX, columns=["game_pk"]).columns or True]
    fm = pd.read_parquet(matrix if matrix is not None else FEATURE_MTX)
    fm["game_date"] = pd.to_datetime(fm["game_date"]).dt.strftime("%Y-%m-%d")

    # Simple join: match on game_pk, determine side by home_team == home_team in labels
    sp = labels.copy()
    sp["game_date"] = sp["game_date"].astype(str)

    fm_h = fm[["game_pk",
               "home_sp_k_pct","home_sp_bb_pct","home_sp_whiff_pctl",
               "home_sp_ff_velo","home_sp_age_pit","home_sp_arm_angle",
               "home_sp_k_pct_10d","home_sp_bb_pct_10d","home_sp_k_minus_bb",
               "home_sp_k_bb_ratio","home_sp_k_bb_ratio_10d",
               "home_sp_1st_k_pct","home_sp_xwoba_against","home_sp_gb_pct",
               "away_bat_k_vs_rhp","away_bat_k_vs_lhp","home_sp_p_throws_R",
               "ump_k_above_avg","home_park_factor","temp_f",
               "home_team"]].copy()
    fm_h["opp_bat_k_rate"] = np.where(fm_h["home_sp_p_throws_R"]==1,
        fm_h["away_bat_k_vs_rhp"], fm_h["away_bat_k_vs_lhp"])
    fm_h = fm_h.rename(columns={c: c.replace("home_sp_","sp_") for c in fm_h.columns})

    fm_a = fm[["game_pk",
               "away_sp_k_pct","away_sp_bb_pct","away_sp_whiff_pctl",
               "away_sp_ff_velo","away_sp_age_pit","away_sp_arm_angle",
               "away_sp_k_pct_10d","away_sp_bb_pct_10d","away_sp_k_minus_bb",
               "away_sp_k_bb_ratio","away_sp_k_bb_ratio_10d",
               "away_sp_1st_k_pct","away_sp_xwoba_against","away_sp_gb_pct",
               "home_bat_k_vs_rhp","home_bat_k_vs_lhp","away_sp_p_throws_R",
               "ump_k_above_avg","home_park_factor","temp_f",
               "away_team"]].copy()
    fm_a["opp_bat_k_rate"] = np.where(fm_a["away_sp_p_throws_R"]==1,
        fm_a["home_bat_k_vs_rhp"], fm_a["home_bat_k_vs_lhp"])
    fm_a = fm_a.rename(columns={c: c.replace("away_sp_","sp_") for c in fm_a.columns})

    # Add away_team to labels via FM join
    sp = sp.merge(fm[["game_pk","home_team","away_team"]].rename(
        columns={"home_team":"fm_home","away_team":"fm_away"}),
        on="game_pk", how="left")

    # Determine pitcher side: home if pitcher's team == fm_home; else away
    sp["sp_is_home"] = sp["home_team"] == sp["fm_home"]

    sp_home = sp[sp["sp_is_home"]].merge(fm_h, on="game_pk", how="left")
    sp_away = sp[~sp["sp_is_home"]].merge(fm_a, on="game_pk", how="left")
    data = pd.concat([sp_home, sp_away], ignore_index=True)
    print(f"Joined rows: {len(data):,}")

    feat_cols = SP_FEATURES
    for c in feat_cols:
        if c not in data.columns:
            data[c] = np.nan
        data[c] = pd.to_numeric(data[c], errors="coerce")

    y = (data["k_total"] >= K_THRESHOLD).astype("int8")
    data["year"] = pd.to_datetime(data["game_date"]).dt.year
    is_valid = data["year"] == val_year
    X_tr, X_va = data.loc[~is_valid, feat_cols], data.loc[is_valid, feat_cols]
    y_tr, y_va = y[~is_valid], y[is_valid]
    print(f"Train {len(X_tr):,}  Valid {len(X_va):,}")
    print(f"Pos rate  train={y_tr.mean():.3f}  valid={y_va.mean():.3f}")

    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, average_precision_score

    dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_cols)
    dva = xgb.DMatrix(X_va, label=y_va, feature_names=feat_cols)
    params = dict(objective="binary:logistic", eval_metric=["logloss","auc"],
                  tree_method="hist", max_depth=4, eta=0.05,
                  subsample=0.80, colsample_bytree=0.80,
                  min_child_weight=5, reg_lambda=1.0, seed=42)
    booster = xgb.train(params, dtr, num_boost_round=600,
                         evals=[(dtr,"train"),(dva,"valid")],
                         early_stopping_rounds=40, verbose_eval=50)

    pred = booster.predict(dva, iteration_range=(0, booster.best_iteration+1))
    metrics = dict(
        rows_train=int(len(X_tr)), rows_valid=int(len(X_va)),
        k_threshold=K_THRESHOLD,
        auc=float(roc_auc_score(y_va, pred)),
        logloss=float(log_loss(y_va, pred)),
        brier=float(brier_score_loss(y_va, pred)),
        ap=float(average_precision_score(y_va, pred)),
        best_iter=int(booster.best_iteration),
        base_rate_valid=float(y_va.mean()),
    )
    print("\n=== K-Over v1 Validation ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    if not no_save_model:
        booster.save_model(str(OUT_MODEL))

    imp = booster.get_score(importance_type="gain")
    imp_df = (pd.DataFrame({"feature": list(imp), "gain": list(imp.values())})
              .sort_values("gain", ascending=False).reset_index(drop=True))
    imp_df["rank"] = imp_df.index + 1
    print("\n=== Feature Importance ===")
    print(imp_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    with OUT_TXT.open("w") as f:
        f.write("K-Over v1 metrics\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n"); f.write(imp_df.to_string(index=False))

    # Save validation predictions
    out_path = val_preds_out if val_preds_out is not None else OUT_VAL_PREDS
    val_df = data.loc[is_valid, ["game_pk", "game_date", "home_team", "away_team",
                                  "sp_is_home", "k_total"]].copy()
    val_df["k_line"]       = K_THRESHOLD
    val_df["k_over_actual"] = y_va.values
    val_df["k_over_pred"]  = pred
    val_df.to_csv(out_path, index=False)
    print(f"\n  Saved validation predictions -> {out_path}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 0.35*len(imp_df)+1))
        ax.barh(imp_df["feature"][::-1], imp_df["gain"][::-1])
        ax.set_xlabel("gain")
        ax.set_title(f"K-Over v1 (AUC={metrics['auc']:.3f}, K>={K_THRESHOLD})")
        plt.tight_layout(); plt.savefig(OUT_PNG, dpi=120)
        print(f"Plot -> {OUT_PNG}")
    except Exception as e:
        print(f"[warn] plot: {e}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train K-Over v1 classifier")
    parser.add_argument("--val-year", type=int, default=2025,
                        help="Holdout year for validation (default: 2025)")
    parser.add_argument("--matrix", type=str, default=None,
                        help="Feature matrix parquet path (overrides default)")
    parser.add_argument("--val-preds-out", type=str, default=None,
                        help="Path to save val predictions CSV (default: k_val_predictions.csv)")
    parser.add_argument("--no-save-model", action="store_true",
                        help="Skip saving model files (only writes val predictions)")
    args = parser.parse_args()
    main(val_year=args.val_year,
         matrix=Path(args.matrix) if args.matrix else None,
         val_preds_out=Path(args.val_preds_out) if args.val_preds_out else None,
         no_save_model=args.no_save_model)
