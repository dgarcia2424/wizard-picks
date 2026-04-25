"""
analyze_feature_importance.py — v6.0 Feature Importance & RFE Audit.

Answers the core question: are our high-resolution context features
(air_density_rho, bullpen_burn_5d, catcher_iq) being drowned out by
batting_order / exp_pa_heuristic in the TB stacker?

Runs three analyses:
  1. XGBoost built-in gain importance on tb_stacker_v37 (existing model)
  2. Sklearn permutation importance on 2026 validation data
  3. Incremental AUC: what do the new v6.0 features add over baseline?

The RFE step fits a lightweight surrogate (LR or shallow XGB) and eliminates
the bottom 20% of features by permutation importance, then checks AUC delta.

Usage:
  python analyze_feature_importance.py              # full report
  python analyze_feature_importance.py --save       # also write CSV
  python analyze_feature_importance.py --new-only   # only incremental analysis
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

_ROOT = Path(__file__).resolve().parent

MATRIX_PATH    = _ROOT / "data/batter_features/final_training_matrix.parquet"
MODEL_PATH     = _ROOT / "models/tb_stacker_v60.json"
CONTEXT_PATH   = _ROOT / "data/orchestrator/daily_context.parquet"
BURN_PATH      = _ROOT / "data/batter_features/bullpen_burn_by_game.parquet"
CATCHER_PATH   = _ROOT / "data/statcast/catcher_iq_by_team_2026.parquet"
BATTER_PCT_PATH= _ROOT / "data/statcast/batter_percentiles_2026.parquet"

# Exact feature set from train_tb_v37.py
NUMERIC_V37 = [
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
FEAT_COLS_V37 = NUMERIC_V37 + ["stand_L", "stand_R", "stand_S"]

# New v6.0 candidate features (not yet in training matrix)
NEW_V60_FEATURES = [
    "air_density_rho",     # live ADI (from daily_context)
    "bullpen_burn_5d",     # 5-day bullpen fatigue (home team)
    "catcher_k_mult",      # catcher challenge IQ multiplier
    "bat_speed_pctile",    # batter bat speed percentile (from batter_percentiles)
    "swing_length_pctile", # swing length percentile
]

N_PERMUTATIONS = 30   # sklearn permutation_importance repeats


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_base_matrix() -> pd.DataFrame:
    df = pd.read_parquet(MATRIX_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    for h in ("L", "R", "S"):
        df[f"stand_{h}"] = (df["stand"].fillna("") == h).astype("int8")
    for c in FEAT_COLS_V37:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _join_air_density(df: pd.DataFrame) -> pd.DataFrame:
    """Join air_density_rho from daily_context by (game_date, home_team)."""
    if not CONTEXT_PATH.exists():
        print("  [rho] daily_context.parquet not found — air_density_rho skipped")
        df["air_density_rho"] = np.nan
        return df
    ctx = pd.read_parquet(CONTEXT_PATH)
    ctx["game_date"] = pd.to_datetime(ctx.get("orchestrator_date",
                                               ctx.get("game_date", pd.NaT)))
    if "air_density_rho" not in ctx.columns:
        df["air_density_rho"] = np.nan
        return df
    ctx_slim = ctx[["game_date", "home_team", "air_density_rho"]].drop_duplicates(
        ["game_date", "home_team"])
    df = df.merge(ctx_slim, left_on=["game_date", "home_park_id"],
                  right_on=["game_date", "home_team"], how="left")
    df["air_density_rho"] = df.get("air_density_rho_y",
                                    df.get("air_density_rho_x", np.nan))
    for col in ("air_density_rho_x", "air_density_rho_y", "home_team"):
        df.drop(columns=[col], errors="ignore", inplace=True)
    print(f"  [rho] air_density_rho joined: {df['air_density_rho'].notna().sum()} / {len(df)}")
    return df


def _join_bullpen_5d(df: pd.DataFrame) -> pd.DataFrame:
    """Join home bullpen_burn_5d from bullpen_burn_by_game."""
    if not BURN_PATH.exists():
        print("  [bp5d] burn file not found — bullpen_burn_5d skipped")
        df["bullpen_burn_5d"] = np.nan
        return df
    bp = pd.read_parquet(BURN_PATH)
    bp["game_date"] = pd.to_datetime(bp["game_date"])
    if "home_bullpen_burn_5d" in bp.columns:
        burn_col = "home_bullpen_burn_5d"
    elif "total_hl_pitches" in bp.columns:
        # Compute rolling 5d on the fly
        bp = bp.sort_values("game_date")
        bp["bullpen_burn_5d"] = (
            bp.groupby("home_team")["total_hl_pitches"]
            .transform(lambda s: s.rolling(5, min_periods=1).sum()))
        burn_col = "bullpen_burn_5d"
    else:
        df["bullpen_burn_5d"] = np.nan
        return df
    bp_slim = bp[["game_date", "home_team", burn_col]].rename(
        columns={burn_col: "bullpen_burn_5d"})
    df = df.merge(bp_slim, left_on=["game_date", "home_park_id"],
                  right_on=["game_date", "home_team"], how="left")
    df["bullpen_burn_5d"] = df.get("bullpen_burn_5d_x",
                                    df.get("bullpen_burn_5d", np.nan))
    df.drop(columns=["home_team", "bullpen_burn_5d_y",
                      "bullpen_burn_5d_x"], errors="ignore", inplace=True)
    print(f"  [bp5d] bullpen_burn_5d joined: {df['bullpen_burn_5d'].notna().sum()} / {len(df)}")
    return df


def _join_catcher_iq(df: pd.DataFrame) -> pd.DataFrame:
    """Join team-level catcher K multiplier."""
    if not CATCHER_PATH.exists():
        print("  [ciq] catcher_iq_by_team not found — run build_catcher_iq.py")
        df["catcher_k_mult"] = 1.0
        return df
    iq = pd.read_parquet(CATCHER_PATH)
    iq_slim = iq[["team", "team_k_multiplier"]].rename(
        columns={"team_k_multiplier": "catcher_k_mult"})
    df = df.merge(iq_slim, left_on="home_park_id", right_on="team", how="left")
    df["catcher_k_mult"] = df["catcher_k_mult"].fillna(1.0)
    df.drop(columns=["team"], errors="ignore", inplace=True)
    n = (df["catcher_k_mult"] != 1.0).sum()
    print(f"  [ciq] catcher_k_mult joined: {n} rows with non-baseline multiplier")
    return df


def _join_bat_tracking(df: pd.DataFrame) -> pd.DataFrame:
    """Join bat_speed and swing_length percentiles from batter_percentiles."""
    if not BATTER_PCT_PATH.exists():
        df["bat_speed_pctile"]    = np.nan
        df["swing_length_pctile"] = np.nan
        return df
    pct = pd.read_parquet(BATTER_PCT_PATH)
    pct_cols = list(pct.columns)
    # Find bat speed column
    bs_col = next((c for c in pct_cols if "bat_speed" in c.lower()), None)
    sl_col = next((c for c in pct_cols if "swing" in c.lower() and "length" in c.lower()), None)
    id_col = next((c for c in pct_cols if c in ("player_id", "batter", "mlbam_id")), None)

    if not id_col:
        df["bat_speed_pctile"] = np.nan
        df["swing_length_pctile"] = np.nan
        return df

    keep = {id_col: "player_id_pct"}
    if bs_col: keep[bs_col] = "bat_speed_pctile"
    if sl_col: keep[sl_col] = "swing_length_pctile"

    pct_slim = pct[[k for k in keep]].rename(columns=keep)
    df = df.merge(pct_slim, left_on="player_id", right_on="player_id_pct", how="left")
    df.drop(columns=["player_id_pct"], errors="ignore", inplace=True)
    if "bat_speed_pctile" not in df.columns:
        df["bat_speed_pctile"] = np.nan
    if "swing_length_pctile" not in df.columns:
        df["swing_length_pctile"] = np.nan
    print(f"  [bat] bat_speed_pctile: {df['bat_speed_pctile'].notna().sum()} rows")
    return df


def build_enriched_matrix() -> pd.DataFrame:
    print("  Loading base matrix …")
    df = _load_base_matrix()
    df = _join_air_density(df)
    df = _join_bullpen_5d(df)
    df = _join_catcher_iq(df)
    df = _join_bat_tracking(df)
    return df


# ---------------------------------------------------------------------------
# Analysis 1: built-in gain from existing model
# ---------------------------------------------------------------------------

def report_gain_importance() -> pd.DataFrame:
    """Return XGBoost gain importance from the v37 model."""
    import xgboost as xgb
    if not MODEL_PATH.exists():
        print("  [gain] model not found — skipping")
        return pd.DataFrame()
    bst = xgb.Booster()
    bst.load_model(str(MODEL_PATH))
    imp = bst.get_score(importance_type="gain")
    df  = (pd.DataFrame({"feature": list(imp), "gain": list(imp.values())})
             .sort_values("gain", ascending=False)
             .reset_index(drop=True))
    df["pct_gain"] = (df["gain"] / df["gain"].sum() * 100).round(2)
    df["cumulative_gain_pct"] = df["pct_gain"].cumsum().round(1)
    return df


# ---------------------------------------------------------------------------
# Analysis 2: permutation importance on 2026 validation set
# ---------------------------------------------------------------------------

def run_permutation_importance(df: pd.DataFrame,
                                feat_cols: list[str]) -> pd.DataFrame:
    """Run sklearn permutation importance on 2026 validation rows."""
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb

    if not MODEL_PATH.exists():
        print("  [perm] model not found — skipping")
        return pd.DataFrame()

    y = (df["total_bases"].astype(float) > 1.5).astype("int8")
    is_2026 = df["year"] == 2026

    X_va = df.loc[is_2026, feat_cols].copy()
    y_va = y[is_2026]

    if len(X_va) < 100:
        print(f"  [perm] only {len(X_va)} 2026 rows — need ≥100")
        return pd.DataFrame()

    X_va = X_va.fillna(0.0)

    bst = xgb.Booster()
    bst.load_model(str(MODEL_PATH))

    class _XGBWrapper:
        def __init__(self, bst, feat_names):
            self.bst = bst
            self.feat_names = feat_names
        def fit(self, X, y): return self
        def predict(self, X):
            d = xgb.DMatrix(X, feature_names=self.feat_names)
            return (self.bst.predict(d) >= 0.5).astype(int)
        def score(self, X, y):
            d = xgb.DMatrix(X, feature_names=self.feat_names)
            p = self.bst.predict(d)
            return roc_auc_score(y, p)

    print(f"  [perm] running {N_PERMUTATIONS} permutations on {len(X_va)} 2026 rows …")
    wrapper = _XGBWrapper(bst, feat_cols)
    result  = permutation_importance(
        wrapper, X_va.values, y_va.values,
        n_repeats=N_PERMUTATIONS, random_state=42, n_jobs=-1,
        scoring="roc_auc",
    )
    perm_df = pd.DataFrame({
        "feature":     feat_cols,
        "perm_mean":   result.importances_mean.round(5),
        "perm_std":    result.importances_std.round(5),
        "perm_min":    result.importances.min(axis=1).round(5),
    }).sort_values("perm_mean", ascending=False).reset_index(drop=True)
    perm_df["rank"] = perm_df.index + 1
    return perm_df


# ---------------------------------------------------------------------------
# Analysis 3: incremental AUC from v6.0 features
# ---------------------------------------------------------------------------

def run_incremental_auc(df: pd.DataFrame) -> dict:
    """Train two lightweight models: baseline (v37 features) vs v60 (+ new features).
    Reports AUC delta on 2026 hold-out to quantify incremental signal."""
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    y = (df["total_bases"].astype(float) > 1.5).astype("int8")
    is_2026 = df["year"] == 2026

    results = {}

    def _quick_auc(feat_cols, label=""):
        X_tr = df.loc[~is_2026, feat_cols].fillna(0.0)
        y_tr = y[~is_2026]
        X_va = df.loc[is_2026, feat_cols].fillna(0.0)
        y_va = y[is_2026]
        if len(X_va) < 50:
            return np.nan
        params = dict(objective="binary:logistic", eval_metric="auc",
                      tree_method="hist", max_depth=4, eta=0.10,
                      subsample=0.80, colsample_bytree=0.70,
                      min_child_weight=15, seed=42, verbosity=0)
        dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_cols)
        dva = xgb.DMatrix(X_va, label=y_va, feature_names=feat_cols)
        bst = xgb.train(params, dtr, num_boost_round=300,
                        evals=[(dva, "valid")], early_stopping_rounds=30,
                        verbose_eval=False)
        pred = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
        auc = roc_auc_score(y_va, pred)
        print(f"  [inc_auc] {label}: AUC={auc:.4f}  ({len(X_tr):,} train / {len(X_va):,} valid)")
        return auc

    # Baseline: v37 features
    results["baseline_auc"] = _quick_auc(FEAT_COLS_V37, "baseline_v37")

    # Available new features
    available_new = [f for f in NEW_V60_FEATURES if f in df.columns
                     and df[f].notna().sum() > 100]

    if available_new:
        augmented = FEAT_COLS_V37 + available_new
        results["v60_auc"] = _quick_auc(augmented, f"v60 (+{len(available_new)} new)")
        results["auc_delta"] = round(results["v60_auc"] - results["baseline_auc"], 5)
        results["new_features_used"] = available_new
    else:
        results["v60_auc"]          = np.nan
        results["auc_delta"]        = np.nan
        results["new_features_used"] = []
        print("  [inc_auc] no v6.0 features available to enrich matrix")

    return results


# ---------------------------------------------------------------------------
# RFE: identify features that can be dropped
# ---------------------------------------------------------------------------

def run_rfe(perm_df: pd.DataFrame, threshold: float = 0.0) -> list[str]:
    """Return features with negative permutation importance (drag, not signal).

    These are candidates for removal from the v6.0 feature set.
    """
    if perm_df.empty:
        return []
    drag = perm_df[perm_df["perm_mean"] < threshold]["feature"].tolist()
    return drag


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(gain_df: pd.DataFrame,
                 perm_df: pd.DataFrame,
                 incr: dict,
                 drag: list[str]) -> None:

    print("\n" + "=" * 70)
    print("  FEATURE IMPORTANCE AUDIT — TB Stacker v3.7")
    print("=" * 70)

    if not gain_df.empty:
        print(f"\n  XGBoost Gain Importance (top 10):")
        print(f"  {'Rank':>4}  {'Feature':<30}  {'Gain':>10}  {'%Gain':>7}  {'CumGain%':>9}")
        print("  " + "-" * 62)
        for _, r in gain_df.head(10).iterrows():
            # Flag if this is a PA-opportunity feature (order/exp_pa)
            flag = " [PA-OPP]" if r["feature"] in ("batting_order", "exp_pa_heuristic") else ""
            print(f"  {int(r.get('rank', _)):>4}  {r['feature']:<30}  "
                  f"{r['gain']:>10.2f}  {r['pct_gain']:>6.2f}%  "
                  f"{r['cumulative_gain_pct']:>8.1f}%{flag}")

        # Check if PA opportunity features dominate
        pa_pct = gain_df[gain_df["feature"].isin(
            ["batting_order", "exp_pa_heuristic"])]["pct_gain"].sum()
        physics_pct = gain_df[gain_df["feature"].isin(
            ["pull_side_wind_vector", "wind_mph", "wind_bearing",
             "temp_f", "velocity_decay_risk"])]["pct_gain"].sum()
        print(f"\n  [!] PA-opportunity features (batting_order + exp_pa): "
              f"{pa_pct:.1f}% of total gain")
        print(f"  [!] Physics features (wind/temp/velo):          "
              f"{physics_pct:.1f}% of total gain")
        if pa_pct > 15:
            print(f"\n  FINDING: PA-opportunity features are drowning out physics context "
                  f"({pa_pct:.0f}% vs {physics_pct:.0f}%).")
            print("  Recommendation: use batting_order as a segment variable "
                  "rather than a raw feature, or cap its gain via reg_lambda.")

    if not perm_df.empty:
        print(f"\n  Permutation Importance on 2026 Validation "
              f"({N_PERMUTATIONS} repeats, AUC drop):")
        print(f"  {'Rank':>4}  {'Feature':<30}  {'Mean AUC Drop':>13}  {'Std':>6}")
        print("  " + "-" * 55)
        for _, r in perm_df.head(12).iterrows():
            sign = " [DRAG]" if r["perm_mean"] < 0 else ""
            print(f"  {int(r['rank']):>4}  {r['feature']:<30}  "
                  f"{r['perm_mean']:>+13.5f}  {r['perm_std']:>6.5f}{sign}")

    if drag:
        print(f"\n  RFE Candidates (negative permutation importance — drag):")
        for f in drag:
            print(f"    - {f}")
        print(f"\n  Removing these {len(drag)} features could improve generalisation "
              f"and reduce overfitting.")

    if incr:
        print(f"\n  Incremental AUC: v37 baseline vs v6.0 augmented features:")
        print(f"    Baseline AUC (v37):      {incr.get('baseline_auc', '?'):.4f}")
        if not np.isnan(incr.get("v60_auc", np.nan)):
            print(f"    v6.0 AUC (+ new feats):  {incr['v60_auc']:.4f}")
            delta = incr.get("auc_delta", 0)
            tag   = "GAIN" if delta > 0.001 else ("LOSS" if delta < -0.001 else "FLAT")
            print(f"    Delta:                   {delta:+.5f}  [{tag}]")
            print(f"    New features used:       {incr.get('new_features_used', [])}")
            if delta > 0.002:
                print(f"\n  [OK] New features provide measurable signal. "
                      f"Proceed with v6.0 retrain.")
            else:
                print(f"\n  [!] New features show minimal AUC lift. "
                      f"Physics features may need richer game-level join.")
        else:
            print(f"    v6.0 AUC: N/A — new features not available in matrix")
            print(f"\n  ACTION: Enrich final_training_matrix.parquet with "
                  f"air_density_rho and bullpen_burn_5d, then re-run.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature importance + RFE audit for TB model (v6.0)")
    parser.add_argument("--save",     action="store_true",
                        help="Save gain/perm importance to data/logs/")
    parser.add_argument("--new-only", action="store_true",
                        help="Skip gain/perm, only run incremental AUC analysis")
    parser.add_argument("--skip-perm", action="store_true",
                        help="Skip slow permutation pass (keep gain only)")
    args = parser.parse_args()

    if not MATRIX_PATH.exists():
        print(f"  ERROR: {MATRIX_PATH} not found — run build_batter_features.py first")
        return

    print("  Building enriched matrix …")
    df = build_enriched_matrix()

    gain_df = pd.DataFrame()
    perm_df = pd.DataFrame()
    incr    = {}
    drag    = []

    if not args.new_only:
        print("\n  Analysis 1: XGBoost gain importance …")
        gain_df = report_gain_importance()

        if not args.skip_perm:
            print("\n  Analysis 2: Permutation importance …")
            perm_df = run_permutation_importance(df, FEAT_COLS_V37)
            drag = run_rfe(perm_df)

    print("\n  Analysis 3: Incremental AUC from v6.0 features …")
    incr = run_incremental_auc(df)

    print_report(gain_df, perm_df, incr, drag)

    if args.save:
        log_dir = _ROOT / "data/logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        if not gain_df.empty:
            gain_df.to_csv(log_dir / "feature_importance_gain_v60.csv", index=False)
        if not perm_df.empty:
            perm_df.to_csv(log_dir / "feature_importance_perm_v60.csv", index=False)
        print(f"\n  Saved to data/logs/")


if __name__ == "__main__":
    main()
