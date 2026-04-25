"""
generate_2024_predictions.py
============================
Generate RL and ML model predictions for the 2024 season using the
feature_matrix_enriched_v2.parquet and current trained models.

NOTE: These predictions are IN-SAMPLE (training data) since the current
models (xgb_ml_v2.json, xgb_rl.json) were trained on 2024.  Accuracy
metrics from this file should be interpreted as upper-bound estimates;
OOF estimates are only available for 2025 (eval_predictions.csv).

Output: eval_predictions_2024.csv  (same schema as eval_predictions.csv)

Usage: python generate_2024_predictions.py
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# BayesianStacker and StackingModel must be importable for pickle to reconstruct
try:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_xgboost import BayesianStacker, StackingModel  # noqa: F401
except ImportError:
    try:
        from train_xgboost import BayesianStacker  # noqa: F401
    except ImportError:
        pass

BASE      = Path(__file__).resolve().parent
MATRIX    = BASE / "feature_matrix_enriched_v2.parquet"
MODELS    = BASE / "models"
OUT_FILE  = BASE / "eval_predictions_2024.csv"

YEAR = 2024


def load_models() -> dict:
    import xgboost as xgb

    feat_cols_rl = json.loads((MODELS / "feature_cols.json").read_text())

    # ML v2 uses a larger feature set
    ml_feat_path = MODELS / "ml_feature_cols_v2.json"
    if not ml_feat_path.exists():
        ml_feat_path = MODELS / "ml_feature_cols.json"
    feat_cols_ml = json.loads(ml_feat_path.read_text())

    xgb_rl = xgb.XGBClassifier()
    xgb_rl.load_model(str(MODELS / "xgb_rl.json"))

    # Prefer v2 ML model; fall back to v1
    ml_path = MODELS / "xgb_ml_v2.json"
    if not ml_path.exists():
        ml_path = MODELS / "xgb_ml.json"
    xgb_ml = xgb.XGBClassifier()
    xgb_ml.load_model(str(ml_path))

    cal_rl = pickle.loads((MODELS / "calibrator_rl.pkl").read_bytes())
    cal_ml = pickle.loads((MODELS / "calibrator_ml.pkl").read_bytes())

    # LightGBM level-1 (optional)
    lgbm_rl = lgbm_ml = None
    try:
        import lightgbm as lgb
        if (MODELS / "lgbm_rl.pkl").exists():
            lgbm_rl = pickle.loads((MODELS / "lgbm_rl.pkl").read_bytes())
        if (MODELS / "lgbm_ml.pkl").exists():
            lgbm_ml = pickle.loads((MODELS / "lgbm_ml.pkl").read_bytes())
    except Exception:
        pass

    # CatBoost level-1 (optional)
    cat_rl = cat_ml = None
    try:
        import catboost as cb
        if (MODELS / "cat_rl.pkl").exists():
            cat_rl = pickle.loads((MODELS / "cat_rl.pkl").read_bytes())
        if (MODELS / "cat_ml.pkl").exists():
            cat_ml = pickle.loads((MODELS / "cat_ml.pkl").read_bytes())
    except Exception:
        pass

    stk = None
    stk_path = MODELS / "stacking_lr_rl.pkl"
    if stk_path.exists():
        stk = pickle.loads(stk_path.read_bytes())

    # Team RL (optional)
    team_rl = team_feat_cols = None
    team_path = MODELS / "xgb_rl_team.json"
    feat_team_path = MODELS / "feature_cols_team.json"
    if team_path.exists() and feat_team_path.exists():
        try:
            _tm = xgb.XGBClassifier()
            _tm.load_model(str(team_path))
            team_rl = _tm
            team_feat_cols = json.loads(feat_team_path.read_text())
        except Exception:
            pass

    return {
        "feature_cols":    feat_cols_rl,
        "feature_cols_ml": feat_cols_ml,
        "xgb_rl":       xgb_rl,
        "xgb_ml":       xgb_ml,
        "lgbm_rl":      lgbm_rl,
        "lgbm_ml":      lgbm_ml,
        "cat_rl":       cat_rl,
        "cat_ml":       cat_ml,
        "cal_rl":       cal_rl,
        "cal_ml":       cal_ml,
        "stacking":     stk,
        "team_rl":      team_rl,
        "team_feat_cols": team_feat_cols,
    }


def _team_l1_probs(team_rl, df: pd.DataFrame, team_feat_cols: list) -> np.ndarray:
    X_team = pd.DataFrame(index=df.index)
    for col in team_feat_cols:
        X_team[col] = df[col] if col in df.columns else np.nan
    probs = team_rl.predict_proba(X_team)[:, 1]
    # L1 normalise: home_norm = home / (home + (1-home))  → same as raw prob
    return probs


def score_df(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    # RL uses feature_cols.json (163 features)
    X_rl = pd.DataFrame(index=df.index)
    for col in models["feature_cols"]:
        X_rl[col] = df[col] if col in df.columns else np.nan

    # ML v2 uses ml_feature_cols_v2.json (211 features)
    X_ml = pd.DataFrame(index=df.index)
    for col in models["feature_cols_ml"]:
        X_ml[col] = df[col] if col in df.columns else np.nan

    rl_raw = models["xgb_rl"].predict_proba(X_rl)[:, 1]
    ml_raw = models["xgb_ml"].predict_proba(X_ml)[:, 1]

    # Optional L1 models for calibrator input
    lgbm_rl_raw = lgbm_ml_raw = cat_rl_raw = cat_ml_raw = None
    _X_cat = X_rl.astype("float64").values
    _X_cat_ml = X_ml.astype("float64").values
    if models.get("lgbm_rl") is not None:
        try: lgbm_rl_raw = models["lgbm_rl"].predict_proba(X_rl)[:, 1]
        except Exception: pass
    if models.get("lgbm_ml") is not None:
        try: lgbm_ml_raw = models["lgbm_ml"].predict_proba(X_ml)[:, 1]
        except Exception: pass
    if models.get("cat_rl") is not None:
        try: cat_rl_raw = models["cat_rl"].predict_proba(_X_cat)[:, 1]
        except Exception: pass
    if models.get("cat_ml") is not None:
        try: cat_ml_raw = models["cat_ml"].predict_proba(_X_cat_ml)[:, 1]
        except Exception: pass

    # Calibrator: expects 3 features [xgb, lgbm, cat]
    def _calibrate(cal, xgb_p, lgbm_p, cat_p):
        n_feats = getattr(cal, "n_features_in_", 1)
        if n_feats == 1:
            return cal.predict_proba(xgb_p.reshape(-1, 1))[:, 1]
        # Build multi-column input; fall back to xgb if optional absent
        cols = [xgb_p,
                lgbm_p if lgbm_p is not None else xgb_p,
                cat_p  if cat_p  is not None else xgb_p]
        X_cal = np.column_stack(cols[:n_feats])
        return cal.predict_proba(X_cal)[:, 1]

    rl_cal = _calibrate(models["cal_rl"], rl_raw, lgbm_rl_raw, cat_rl_raw)
    ml_cal = _calibrate(models["cal_ml"], ml_raw, lgbm_ml_raw, cat_ml_raw)

    out = df.copy()
    out["rl_raw"] = rl_raw
    out["rl_cal"] = rl_cal
    out["ml_raw"] = ml_raw
    out["ml_cal"] = ml_cal

    if models["stacking"] is not None:
        try:
            team_rl        = models.get("team_rl")
            team_feat_cols = models.get("team_feat_cols")

            if team_rl is not None and team_feat_cols is not None:
                home_norm = _team_l1_probs(team_rl, df, team_feat_cols)
            else:
                home_norm = None

            _h_r = (df["home_sp_p_throws_R"].fillna(1).astype(int)
                    if "home_sp_p_throws_R" in df.columns
                    else pd.Series(1, index=df.index))
            _a_r = (df["away_sp_p_throws_R"].fillna(1).astype(int)
                    if "away_sp_p_throws_R" in df.columns
                    else pd.Series(1, index=df.index))
            segment_ids = (_h_r * 2 + _a_r).values.astype(np.int32)

            if home_norm is not None:
                stk_probs = models["stacking"].predict(
                    home_norm, df, segment_id=segment_ids,
                    lgbm_raw=lgbm_rl_raw, cat_raw=cat_rl_raw)
            else:
                stk_probs = models["stacking"].predict(
                    rl_raw, X_rl, segment_id=segment_ids,
                    lgbm_raw=lgbm_rl_raw, cat_raw=cat_rl_raw)
            out["rl_stacked"] = stk_probs
        except Exception as exc:
            print(f"  [WARN] Stacking failed: {exc}")
            out["rl_stacked"] = rl_cal   # fallback
    else:
        out["rl_stacked"] = rl_cal

    return out


def main() -> None:
    print("=" * 60)
    print("  generate_2024_predictions.py")
    print("  NOTE: 2024 = TRAINING data (in-sample)")
    print("=" * 60)

    if not MATRIX.exists():
        raise SystemExit(f"Matrix not found: {MATRIX}")

    fm = pd.read_parquet(MATRIX)
    fm["game_date"] = pd.to_datetime(fm["game_date"])

    # Normalise year column
    if "year" not in fm.columns and "season" in fm.columns:
        fm["year"] = fm["season"].astype(int)

    df24 = fm[fm["year"] == YEAR].copy()
    print(f"  2024 rows: {len(df24)}")

    # Derive actuals if needed
    if "actual_home_win" not in df24.columns and "home_margin" in df24.columns:
        df24["actual_home_win"] = np.where(
            df24["home_margin"].isna(), np.nan,
            (df24["home_margin"] > 0).astype(float))

    n_rl = df24["home_covers_rl"].notna().sum()
    n_ml = df24["actual_home_win"].notna().sum() if "actual_home_win" in df24.columns else 0
    print(f"  RL actuals: {n_rl}  |  ML actuals: {n_ml}")

    print("  Loading models...")
    models = load_models()
    print(f"  Feature cols: {len(models['feature_cols'])}")
    print(f"  Stacker: {'loaded' if models['stacking'] else 'not found'}")

    print("  Scoring 2024 data...")
    scored = score_df(df24, models)

    # Build output in same schema as eval_predictions.csv
    keep_cols = ["game_date", "home_team", "away_team", "year",
                 "home_starter_name", "away_starter_name",
                 "home_score", "away_score",
                 "home_covers_rl", "actual_home_win",
                 "rl_raw", "rl_cal", "rl_stacked", "ml_raw", "ml_cal"]

    actual_cols = [c for c in keep_cols if c in scored.columns]
    out = scored[actual_cols].copy()
    out["home_margin"] = (out["home_score"] - out["away_score"]
                          if "home_score" in out.columns and "away_score" in out.columns
                          else np.nan)
    out["_eval_mode"] = "train_is"  # in-sample flag

    out.to_csv(OUT_FILE, index=False)
    print(f"\n  Saved {len(out)} rows -> {OUT_FILE}")

    # Quick accuracy preview
    for prob_col, label_col in [("rl_stacked", "home_covers_rl"),
                                 ("ml_raw",    "actual_home_win")]:
        if prob_col not in out.columns or label_col not in out.columns:
            continue
        sub = out.dropna(subset=[prob_col, label_col])
        thresh = 0.54
        hi = sub[sub[prob_col] >= thresh]
        print(f"\n  {prob_col} >= {thresh}: n={len(hi)}, "
              f"acc={hi[label_col].mean():.3f}" if len(hi) > 0 else
              f"  {prob_col} >= {thresh}: n=0")
        lo = sub[sub[prob_col] < thresh]
        print(f"  {prob_col} < {thresh}:  n={len(lo)}, "
              f"acc={lo[label_col].mean():.3f}" if len(lo) > 0 else
              f"  {prob_col} < {thresh}:  n=0")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
