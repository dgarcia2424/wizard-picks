"""
train_script_a.py — Script A: The Dominance.

Joint SGP: SP K >= 5.5 (Over)  AND  Game Total < 8 (Under).

Rationale: A dominant SP correlates with BOTH the K-Over AND the Game Under.
This correlation is systematically underpriced by books that treat each leg
independently. The joint probability > product of marginals.

Method:
  - Label: y = 1 if BOTH sp_k>=5.5 AND game_total<8.0 in the same game.
  - Features: same SP K features + game-env features (temp, park, bullpen).
  - Train 2024, validate 2025.

Outputs:
    models/script_a_v1.json
    data/logs/script_a_v1_metrics.txt
    data/logs/script_a_correlation_stats.txt
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SP_LABELS   = Path("data/batter_features/sp_k_labels.parquet")
FEATURE_MTX = Path("feature_matrix_enriched_v2.parquet")
OUT_MODEL   = Path("models/script_a_v1.json")
OUT_TXT     = Path("data/logs/script_a_v1_metrics.txt")
OUT_CORR    = Path("data/logs/script_a_correlation_stats.txt")

K_THRESH    = 5.5
TOTAL_UNDER = 8.0

FEATURES = [
    "sp_k_pct", "sp_bb_pct", "sp_whiff_pctl", "sp_ff_velo",
    "sp_k_pct_10d", "sp_k_bb_ratio", "sp_1st_k_pct",
    "sp_xwoba_against", "sp_gb_pct", "opp_bat_k_rate",
    "ump_k_above_avg", "home_park_factor", "temp_f",
    "home_bullpen_vulnerability", "away_bullpen_vulnerability", "elo_diff",
    "close_total",
]


def main(val_year: int = 2025, matrix: Path | None = None, with_2026: bool = False):
    labels = pd.read_parquet(SP_LABELS)
    fm = pd.read_parquet(matrix if matrix is not None else FEATURE_MTX)
    fm["game_date"] = pd.to_datetime(fm["game_date"]).dt.strftime("%Y-%m-%d")

    if with_2026:
        act_path = Path("data/statcast/actuals_2026.parquet")
        if act_path.exists():
            act = pd.read_parquet(act_path)[["game_pk", "home_score_final", "away_score_final"]].dropna()
            act["actual_game_total_2026"] = act["home_score_final"] + act["away_score_final"]
            fm = fm.merge(act[["game_pk", "actual_game_total_2026"]], on="game_pk", how="left")
            mask = fm["actual_game_total"].isna() & fm["actual_game_total_2026"].notna()
            fm.loc[mask, "actual_game_total"] = fm.loc[mask, "actual_game_total_2026"]
            fm = fm.drop(columns=["actual_game_total_2026"])
            print(f"  [script_a] Enriched {mask.sum()} 2026 rows with actual_game_total")

    # Rename home/away SP features to side-neutral for each row
    fm_h = fm[["game_pk", "home_team",
               "home_sp_k_pct","home_sp_bb_pct","home_sp_whiff_pctl",
               "home_sp_ff_velo","home_sp_k_pct_10d","home_sp_k_bb_ratio",
               "home_sp_1st_k_pct","home_sp_xwoba_against","home_sp_gb_pct",
               "home_sp_p_throws_R",
               "away_bat_k_vs_rhp","away_bat_k_vs_lhp",
               "ump_k_above_avg","home_park_factor","temp_f",
               "home_bullpen_vulnerability","away_bullpen_vulnerability","elo_diff",
               "close_total","actual_game_total"]].copy()
    fm_h["opp_bat_k_rate"] = np.where(fm_h["home_sp_p_throws_R"]==1,
        fm_h["away_bat_k_vs_rhp"], fm_h["away_bat_k_vs_lhp"])
    fm_h = fm_h.rename(columns={c: c.replace("home_sp_","sp_") for c in fm_h.columns})

    fm_a = fm[["game_pk", "away_team",
               "away_sp_k_pct","away_sp_bb_pct","away_sp_whiff_pctl",
               "away_sp_ff_velo","away_sp_k_pct_10d","away_sp_k_bb_ratio",
               "away_sp_1st_k_pct","away_sp_xwoba_against","away_sp_gb_pct",
               "away_sp_p_throws_R",
               "home_bat_k_vs_rhp","home_bat_k_vs_lhp",
               "ump_k_above_avg","home_park_factor","temp_f",
               "home_bullpen_vulnerability","away_bullpen_vulnerability","elo_diff",
               "close_total","actual_game_total"]].copy()
    fm_a["opp_bat_k_rate"] = np.where(fm_a["away_sp_p_throws_R"]==1,
        fm_a["home_bat_k_vs_rhp"], fm_a["home_bat_k_vs_lhp"])
    fm_a = fm_a.rename(columns={c: c.replace("away_sp_","sp_") for c in fm_a.columns})

    labels = labels.copy()
    labels["game_date"] = labels["game_date"].astype(str)
    fm_ids = fm[["game_pk","home_team","away_team","actual_game_total"]]\
               .drop_duplicates("game_pk")\
               .rename(columns={"home_team":"fm_home","away_team":"fm_away"})
    labels = labels.merge(fm_ids, on="game_pk", how="left")
    # SP is home if labels.home_team == fm_home
    labels["sp_is_home"] = labels["home_team"] == labels["fm_home"]

    sp_home = labels[labels["sp_is_home"]]\
                    .merge(fm_h.rename(columns={"home_team":"fm_home"}),
                           on=["game_pk","fm_home"], how="inner")
    sp_away = labels[~labels["sp_is_home"]]\
                    .merge(fm_a.rename(columns={"away_team":"fm_away"}),
                           on=["game_pk","fm_away"], how="inner")
    data = pd.concat([sp_home, sp_away], ignore_index=True)
    data["year"] = pd.to_datetime(data["game_date"]).dt.year

    # actual_game_total: use _x version if present from first merge
    if "actual_game_total_x" in data.columns:
        data["actual_game_total"] = data["actual_game_total_x"].fillna(
            data.get("actual_game_total_y", np.nan))

    data = data.dropna(subset=["actual_game_total"])
    print(f"Joined rows with actual total: {len(data):,}")

    y_k    = (data["k_total"] >= K_THRESH).astype(int)
    y_tot  = (data["actual_game_total"] < TOTAL_UNDER).astype(int)
    y_joint = (y_k & y_tot).astype("int8")

    # ── Correlation stats ───────────────────────────────────────────────
    n = len(data)
    p_k     = y_k.mean()
    p_tot   = y_tot.mean()
    p_joint = y_joint.mean()
    p_indep = p_k * p_tot
    corr_ratio = p_joint / p_indep if p_indep > 0 else float("nan")
    stats = (
        f"Script A Correlation Stats  (K>={K_THRESH} AND Total<{TOTAL_UNDER})\n"
        f"  n games:           {n:,}\n"
        f"  P(K over):         {p_k:.4f}\n"
        f"  P(Total under):    {p_tot:.4f}\n"
        f"  P(joint, actual):  {p_joint:.4f}\n"
        f"  P(joint, indep):   {p_indep:.4f}\n"
        f"  Correlation ratio: {corr_ratio:.3f}  "
        f"({'POSITIVE corr — script has edge' if corr_ratio > 1 else 'NEGATIVE corr — avoid'})\n"
    )
    print(stats)

    OUT_CORR.parent.mkdir(parents=True, exist_ok=True)
    OUT_CORR.write_text(stats)

    # ── Train ────────────────────────────────────────────────────────────
    feat_cols = FEATURES
    for c in feat_cols:
        if c not in data.columns: data[c] = np.nan
        data[c] = pd.to_numeric(data[c], errors="coerce")

    is_valid = data["year"] == val_year
    X_tr, X_va = data.loc[~is_valid, feat_cols], data.loc[is_valid, feat_cols]
    y_tr, y_va = y_joint[~is_valid], y_joint[is_valid]
    print(f"Train {len(X_tr):,} pos={y_tr.mean():.3f}  Valid {len(X_va):,} pos={y_va.mean():.3f}")

    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

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
        script="A_Dominance",
        legs=f"SP_K>={K_THRESH} AND Game_Total<{TOTAL_UNDER}",
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
    print("\n=== Script A Validation ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(OUT_MODEL))

    imp = booster.get_score(importance_type="gain")
    imp_df = (pd.DataFrame({"feature": list(imp), "gain": list(imp.values())})
              .sort_values("gain", ascending=False).reset_index(drop=True))
    imp_df["rank"] = imp_df.index + 1
    print("\n=== Feature Importance ===")
    print(imp_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    with OUT_TXT.open("w") as f:
        f.write("Script A: The Dominance\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")
        f.write(stats)
        f.write("\nFeature importance\n")
        f.write(imp_df.to_string(index=False))

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Script A — SP K + Under SGP")
    parser.add_argument("--val-year", type=int, default=2025)
    parser.add_argument("--matrix", type=str, default=None)
    parser.add_argument("--with-2026", action="store_true",
                        help="Load actuals_2026.parquet to fill 2026 actual_game_total")
    args = parser.parse_args()
    main(val_year=args.val_year,
         matrix=Path(args.matrix) if args.matrix else None,
         with_2026=args.with_2026)
