"""
score_nrfi_today.py
===================
Score today's games with the full two-level NRFI stack:

  L1: XGBoost + OOF-fitted Platt calibration
  L2: Bayesian Hierarchical Stacker
        inputs: XGB raw prob (logit), SP diffs, xERA pctl, matchup edge,
                team log-odds ratio, rolling NRFI base rate, dual-Poisson features

Outputs L1 and L2 probabilities side-by-side, and compares to actual NRFI
outcomes if actuals_2026.parquet contains today's data.

Usage:
  python score_nrfi_today.py
  python score_nrfi_today.py --date 2026-04-19
"""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from train_nrfi_model import (
    BayesianStackerNRFI,
    NRFI_STACKING_FEATURES,
    _compute_log_odds_ratio,
    _derive_segment_id,
    _flip_team_perspective,
)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR        = Path(".")
DATA_DIR        = BASE_DIR / "data" / "statcast"
MODELS_DIR      = BASE_DIR / "models"
FEAT_MATRIX     = BASE_DIR / "feature_matrix_enriched_v2.parquet"

FEAT_COLS_PATH  = MODELS_DIR / "nrfi_feature_cols.json"
XGB_NRFI_PATH   = MODELS_DIR / "xgb_nrfi.json"
XGB_CAL_PATH    = MODELS_DIR / "xgb_nrfi_calibrator.pkl"
STACKER_PATH    = MODELS_DIR / "stacking_lr_nrfi.pkl"
TEAM_MODEL_PATH = MODELS_DIR / "team_nrfi_model.json"
TEAM_FEAT_PATH  = MODELS_DIR / "team_nrfi_feat_cols.json"
POIS_HOME_PATH  = MODELS_DIR / "xgb_pois_f1_home.json"
POIS_AWAY_PATH  = MODELS_DIR / "xgb_pois_f1_away.json"
ACTUALS_2026    = DATA_DIR   / "actuals_2026.parquet"


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

def load_models():
    for p in [FEAT_COLS_PATH, XGB_NRFI_PATH, XGB_CAL_PATH, STACKER_PATH,
              TEAM_MODEL_PATH, TEAM_FEAT_PATH, POIS_HOME_PATH, POIS_AWAY_PATH]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing model file: {p}\n"
                "Run: python train_nrfi_model.py --matrix feature_matrix_with_2026.parquet --with-2026"
            )

    feat_cols      = json.load(open(FEAT_COLS_PATH))
    team_feat_cols = json.load(open(TEAM_FEAT_PATH))

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(XGB_NRFI_PATH))

    team_model = xgb.XGBClassifier()
    team_model.load_model(str(TEAM_MODEL_PATH))

    cal = pickle.load(open(XGB_CAL_PATH, "rb"))

    # Inject class into __main__ so the pickle (originally saved with
    # train_nrfi_model as __main__) resolves regardless of context.
    import __main__ as _main
    if not hasattr(_main, "BayesianStackerNRFI"):
        _main.BayesianStackerNRFI = BayesianStackerNRFI
    stacker = pickle.load(open(STACKER_PATH, "rb"))

    pois_home = xgb.Booster()
    pois_home.load_model(str(POIS_HOME_PATH))
    pois_away = xgb.Booster()
    pois_away.load_model(str(POIS_AWAY_PATH))

    return (xgb_model, cal, feat_cols, stacker,
            team_model, team_feat_cols, pois_home, pois_away)


# ---------------------------------------------------------------------------
# ROLLING NRFI BASE RATE
# ---------------------------------------------------------------------------

def get_rolling_nrfi_base_rate(date_str: str) -> float:
    """30-day NRFI rate using games BEFORE date_str. Fallback ~0.56 (MLB baseline)."""
    _FALLBACK = 0.56

    if not ACTUALS_2026.exists():
        return _FALLBACK

    act = pd.read_parquet(ACTUALS_2026)
    act["game_date"] = pd.to_datetime(act["game_date"])
    cutoff           = pd.to_datetime(date_str)
    window_start     = cutoff - pd.Timedelta(days=30)
    window           = act[(act["game_date"] >= window_start) &
                           (act["game_date"] <  cutoff)]

    if len(window) < 10:
        return _FALLBACK

    if "f1_nrfi" in window.columns:
        return float(window["f1_nrfi"].mean())
    if "f1_home_runs" in window.columns and "f1_away_runs" in window.columns:
        return float(((window["f1_home_runs"] + window["f1_away_runs"]) == 0).mean())

    return _FALLBACK


# ---------------------------------------------------------------------------
# LINEUP LOADING
# ---------------------------------------------------------------------------

def get_todays_games(date_str: str) -> pd.DataFrame:
    lineup_path = DATA_DIR / f"lineups_{date_str}.parquet"
    if lineup_path.exists():
        lineups = pd.read_parquet(lineup_path)
    else:
        fallback = DATA_DIR / "lineups_today.parquet"
        if not fallback.exists():
            raise FileNotFoundError(
                f"No lineup file found for {date_str}.\nExpected: {lineup_path}"
            )
        lineups = pd.read_parquet(fallback)
    lineups["game_date"] = date_str
    return lineups


# ---------------------------------------------------------------------------
# FEATURE BUILDING
# ---------------------------------------------------------------------------

def build_game_feature_row(
    home_team:    str,
    away_team:    str,
    home_sp_name: str,
    away_sp_name: str,
    date_str:     str,
    fm:           pd.DataFrame,
    feat_cols:    list,
) -> "pd.Series | None":
    """Build feature vector using most-recent home-team row + SP overwrites."""
    fm_sorted = fm[fm["home_team"] == home_team].sort_values("game_date")
    if len(fm_sorted) == 0:
        return None

    base = fm_sorted.iloc[-1].copy()

    if home_sp_name and not pd.isna(home_sp_name):
        sp_up = home_sp_name.upper().strip()
        rows  = fm[fm["home_starter_name"].str.upper().str.strip() == sp_up]
        if len(rows) == 0:
            rows = fm[fm["away_starter_name"].str.upper().str.strip() == sp_up]
        if len(rows) > 0:
            sp_row = rows.sort_values("game_date").iloc[-1]
            if sp_row.get("home_starter_name", "").upper().strip() == sp_up:
                for c in feat_cols:
                    if c.startswith("home_sp_") and not c.endswith("diff"):
                        base[c] = sp_row.get(c, base.get(c))
            else:
                for c in feat_cols:
                    if c.startswith("home_sp_") and not c.endswith("diff"):
                        base[c] = sp_row.get("away_sp_" + c[8:], base.get(c))

    if away_sp_name and not pd.isna(away_sp_name):
        sp_up = away_sp_name.upper().strip()
        rows  = fm[fm["away_starter_name"].str.upper().str.strip() == sp_up]
        if len(rows) == 0:
            rows = fm[fm["home_starter_name"].str.upper().str.strip() == sp_up]
        if len(rows) > 0:
            sp_row = rows.sort_values("game_date").iloc[-1]
            if sp_row.get("away_starter_name", "").upper().strip() == sp_up:
                for c in feat_cols:
                    if c.startswith("away_sp_") and not c.endswith("diff"):
                        base[c] = sp_row.get(c, base.get(c))
            else:
                for c in feat_cols:
                    if c.startswith("away_sp_") and not c.endswith("diff"):
                        base[c] = sp_row.get("home_sp_" + c[8:], base.get(c))

    for dc in ["sp_k_pct_diff", "sp_xwoba_diff", "sp_xrv_diff", "sp_velo_diff",
               "sp_age_diff", "sp_kminusbb_diff", "sp_k_pct_10d_diff",
               "sp_xwoba_10d_diff", "sp_bb_pct_10d_diff"]:
        h_c = "home_" + dc.replace("_diff", "")
        a_c = "away_" + dc.replace("_diff", "")
        if h_c in base.index and a_c in base.index:
            try:
                base[dc] = float(base[h_c]) - float(base[a_c])
            except (TypeError, ValueError):
                pass

    import datetime as _dt
    d = _dt.datetime.strptime(date_str, "%Y-%m-%d")
    if "game_month" in base.index:
        base["game_month"] = d.month
    if "game_day_of_week" in base.index:
        base["game_day_of_week"] = d.weekday()

    return base


# ---------------------------------------------------------------------------
# TEAM LOG-ODDS
# ---------------------------------------------------------------------------

def compute_team_log_odds(
    feat_row:       "pd.Series",
    feat_cols:      list,
    team_model:     xgb.XGBClassifier,
    team_feat_cols: list,
) -> float:
    """Compute team_nrfi_log_odds = logit(p_home) - logit(p_away) via doubled model."""
    row_df = pd.DataFrame([feat_row])

    row_home = row_df.copy()
    row_home["is_home"] = 1
    for c in team_feat_cols:
        if c not in row_home.columns:
            row_home[c] = 0.0
    p_h = team_model.predict_proba(
        row_home[team_feat_cols].fillna(0).values.astype(np.float32)
    )[0, 1]

    row_flip = _flip_team_perspective(row_df, feat_cols)
    row_flip["is_home"] = 0
    for c in team_feat_cols:
        if c not in row_flip.columns:
            row_flip[c] = 0.0
    p_a = team_model.predict_proba(
        row_flip[team_feat_cols].fillna(0).values.astype(np.float32)
    )[0, 1]

    return float(_compute_log_odds_ratio(np.array([p_h]), np.array([p_a]))[0])


# ---------------------------------------------------------------------------
# MAIN PREDICTION LOOP
# ---------------------------------------------------------------------------

def predict_games(date_str: str) -> pd.DataFrame:
    print(f"Loading models …")
    (xgb_model, cal, feat_cols, stacker,
     team_model, team_feat_cols, pois_home, pois_away) = load_models()

    print(f"Loading feature matrix from {FEAT_MATRIX} …")
    fm = pd.read_parquet(FEAT_MATRIX)
    fm["game_date"] = pd.to_datetime(fm["game_date"])

    rolling_nrfi = get_rolling_nrfi_base_rate(date_str)
    print(f"Rolling 30-day NRFI base rate: {rolling_nrfi:.4f}")

    lineups = get_todays_games(date_str)
    print(f"\n{len(lineups)} games scheduled for {date_str}\n")

    actuals = {}
    if ACTUALS_2026.exists():
        act = pd.read_parquet(ACTUALS_2026)
        for _, r in act[act["game_date"].astype(str).str.startswith(date_str)].iterrows():
            actuals[r["game_pk"]] = r

    rows = []
    for _, g in lineups.iterrows():
        home    = g["home_team"]
        away    = g["away_team"]
        home_sp = g.get("home_starter_name", "")
        away_sp = g.get("away_starter_name", "")
        gk      = g.get("game_pk")

        feat = build_game_feature_row(home, away, home_sp, away_sp, date_str, fm, feat_cols)
        if feat is None:
            rows.append({
                "home_team": home, "away_team": away,
                "home_sp": home_sp, "away_sp": away_sp,
                "xgb_l1": None, "p_stk_nrfi": None,
            })
            continue

        # ── L1: XGBoost + OOF-fitted Platt ───────────────────────────────
        feat_df = pd.DataFrame([feat[feat_cols].fillna(0)], columns=feat_cols)

        # --- START VALIDATION BLOCK ---
        expected_cols = json.load(open(FEAT_COLS_PATH))
        actual_cols = list(feat_df.columns)
        assert expected_cols == actual_cols, "Feature list mismatch after pruning. Check for missing or extra columns."
        assert not feat_df[expected_cols].isnull().any().any(), "Pruning introduced unexpected NaNs."
        # --- END VALIDATION BLOCK ---

        X_l1 = feat_df.values.astype(np.float32)
        raw   = xgb_model.predict_proba(X_l1)[0, 1]
        cal_p = cal.predict_proba([[raw]])[0, 1]

        # Team log-odds
        team_lo = compute_team_log_odds(feat, feat_cols, team_model, team_feat_cols)

        # Dual-Poisson sidecar → lam_home, lam_away, pois_p_nrfi
        dmat   = xgb.DMatrix(X_l1)
        lam_h  = float(pois_home.predict(dmat)[0])
        lam_a  = float(pois_away.predict(dmat)[0])
        p_nrfi = float(np.exp(-(lam_h + lam_a)))

        # L2 stacker
        feat_aug = feat.copy()
        feat_aug["team_nrfi_log_odds"]     = team_lo
        feat_aug["rolling_nrfi_base_rate"] = rolling_nrfi
        feat_aug["pois_lam_f1_home"]       = lam_h
        feat_aug["pois_lam_f1_away"]       = lam_a
        feat_aug["pois_p_nrfi"]            = p_nrfi
        feat_row_df = pd.DataFrame([feat_aug])

        seg   = _derive_segment_id(feat_row_df)
        stk_p = float(stacker.predict(np.array([raw]), feat_row_df, seg)[0])

        act_row = actuals.get(gk, {})
        rows.append({
            "home_team":       home,
            "away_team":       away,
            "home_sp":         home_sp,
            "away_sp":         away_sp,
            "xgb_l1":          round(cal_p, 3),
            "p_stk_nrfi":      round(stk_p, 3),
            "team_log_odds":   round(team_lo, 3),
            "pois_p_nrfi":     round(p_nrfi, 3),
            "actual_f1_home":  act_row.get("f1_home_runs"),
            "actual_f1_away":  act_row.get("f1_away_runs"),
            "actual_nrfi":     act_row.get("f1_nrfi"),
        })

    df = pd.DataFrame(rows)

    has_actuals = df["actual_nrfi"].notna().any()
    print(f"  {'Away':6s}  @  {'Home':6s}  {'Away SP':22s}  {'Home SP':22s}  "
          f"{'L1%':6s}  {'Stk%':6s}  {'LogOdds':>8s}",
          end="  Actual\n" if has_actuals else "\n")
    print("  " + "-" * (95 if has_actuals else 82))

    correct_l1 = correct_stk = total_games = 0
    sort_col   = "p_stk_nrfi" if df["p_stk_nrfi"].notna().any() else "xgb_l1"
    for _, r in df.sort_values(sort_col, ascending=False, na_position="last").iterrows():
        if r["p_stk_nrfi"] is None:
            print(f"  {r['away_team']:6s}  @  {r['home_team']:6s}  — no features available —")
            continue

        l1 = r["xgb_l1"]
        l2 = r["p_stk_nrfi"]
        lo = r["team_log_odds"]
        pred_l1  = "NRFI" if l1 >= 0.50 else "YRFI"
        pred_stk = "NRFI" if l2 >= 0.50 else "YRFI"

        line = (f"  {r['away_team']:6s}  @  {r['home_team']:6s}  "
                f"{str(r['away_sp']):22s}  {str(r['home_sp']):22s}  "
                f"{l1:6.1%}  {l2:6.1%}  {lo:>+8.3f}")

        if has_actuals and r["actual_nrfi"] is not None:
            hr     = int(r["actual_f1_home"]) if pd.notna(r["actual_f1_home"]) else 0
            ar     = int(r["actual_f1_away"]) if pd.notna(r["actual_f1_away"]) else 0
            nrfi   = int(r["actual_nrfi"])
            result = "NRFI" if nrfi == 1 else "YRFI"
            c1 = "+" if (pred_l1  == "NRFI") == bool(nrfi) else "-"
            c2 = "+" if (pred_stk == "NRFI") == bool(nrfi) else "-"
            total_games += 1
            correct_l1  += int((pred_l1  == "NRFI") == bool(nrfi))
            correct_stk += int((pred_stk == "NRFI") == bool(nrfi))
            line += f"  {ar}–{hr} F1  {result:4s}  L1:{c1} Stk:{c2}"
        print(line)

    if total_games > 0:
        print(f"\n  L1  correct: {correct_l1}/{total_games}  ({correct_l1/total_games:.1%})")
        print(f"  Stk correct: {correct_stk}/{total_games}  ({correct_stk/total_games:.1%})")

    return df


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score today's games with the NRFI XGBoost + Bayesian Stacker"
    )
    parser.add_argument("--date", default=None,
                        help="Date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    from datetime import date
    date_str = args.date or date.today().isoformat()
    predict_games(date_str)


if __name__ == "__main__":
    main()
