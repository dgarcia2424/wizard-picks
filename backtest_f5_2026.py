"""
backtest_f5_2026.py
===================
Evaluate the full F5 XGBoost + Bayesian Stacker pipeline on 2026 games.

For each 2026 game in feature_matrix_enriched.parquet:
  1. Fill the 3 rolling-adj-form features using only PRIOR game actuals
     (no look-ahead — mirrors score_f5_today.py logic)
  2. Run L1: XGBoost → OOF-fitted Platt calibration
  3. Run L2: Bayesian Hierarchical Stacker
  4. Compare to actual F5 outcome from actuals_2026.parquet

Outputs:
  - AUC for L1 and L2
  - Fixed-bin calibration table
  - Signal performance at >60%, >65%, >70% cutoffs
  - Saves backtest_f5_2026_results.csv

NOTE: The 313 games with feature matrix rows are IN-SAMPLE for the L1
      model (trained with with_2026=True). AUC here is optimistic —
      treat it as an upper bound. True OOS eval requires LOO-year CV
      which is baked into train_final()'s OOF Platt fitting.
"""

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import logit
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from train_f5_model import (
    BayesianStackerF5,
    F5_STACKING_FEATURES,
    _compute_log_odds_ratio,
    _derive_segment_id,
    _flip_team_perspective,
)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR    = Path(".")
DATA_DIR    = BASE_DIR / "data" / "statcast"
MODELS_DIR  = BASE_DIR / "models"

FEAT_MATRIX  = BASE_DIR / "feature_matrix_enriched.parquet"
ACTUALS_2026 = DATA_DIR / "actuals_2026.parquet"
OUTPUT_CSV   = BASE_DIR / "backtest_f5_2026_results.csv"

FEAT_COLS_PATH  = MODELS_DIR / "f5_feature_cols.json"
XGB_F5_PATH     = MODELS_DIR / "xgb_f5.json"
XGB_CAL_PATH    = MODELS_DIR / "xgb_f5_calibrator.pkl"
STACKER_PATH    = MODELS_DIR / "stacking_lr_f5.pkl"
TEAM_MODEL_PATH = MODELS_DIR / "team_f5_model.json"
TEAM_FEAT_PATH  = MODELS_DIR / "team_f5_feat_cols.json"


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

def load_models():
    feat_cols      = json.load(open(FEAT_COLS_PATH))
    team_feat_cols = json.load(open(TEAM_FEAT_PATH))

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(XGB_F5_PATH))

    team_model = xgb.XGBClassifier()
    team_model.load_model(str(TEAM_MODEL_PATH))

    cal     = pickle.load(open(XGB_CAL_PATH, "rb"))
    stacker = pickle.load(open(STACKER_PATH, "rb"))

    return xgb_model, cal, feat_cols, stacker, team_model, team_feat_cols


# ---------------------------------------------------------------------------
# ROLLING ADJ FORM (no-leakage per-game computation)
# ---------------------------------------------------------------------------

def compute_form_for_date(date_str: str, actuals: pd.DataFrame,
                          fm_mc: pd.DataFrame) -> dict:
    """
    Given a game date, compute rolling 15-game capped adj form for each team
    using only games STRICTLY BEFORE that date.
    Returns {team: form_value}.
    """
    cutoff = pd.to_datetime(date_str)
    hist   = actuals[actuals["game_date"] < cutoff].copy()
    if len(hist) == 0:
        return {}

    hist = hist.merge(fm_mc, on="game_pk", how="left")
    hist["mc_f5_home_win_pct"] = hist["mc_f5_home_win_pct"].fillna(0.5)
    hist["mc_f5_away_win_pct"] = hist["mc_f5_away_win_pct"].fillna(0.5)

    raw_margin   = (hist["f5_home_runs"] - hist["f5_away_runs"]).astype(float)
    capped       = raw_margin.clip(-4, 4)
    mc_exp       = hist["mc_f5_home_win_pct"] - hist["mc_f5_away_win_pct"]
    hist["adj_home"] = capped - mc_exp
    hist["adj_away"] = -hist["adj_home"]

    home_long = pd.DataFrame({
        "game_date": hist["game_date"],
        "team":      hist["home_team"],
        "adj":       hist["adj_home"].values,
    })
    away_long = pd.DataFrame({
        "game_date": hist["game_date"],
        "team":      hist["away_team"],
        "adj":       hist["adj_away"].values,
    })
    long_df = pd.concat([home_long, away_long], ignore_index=True)

    global_mean = float(long_df["adj"].mean()) if len(long_df) > 0 else 0.0
    result: dict = {}
    for team, grp in long_df.groupby("team"):
        recent = grp.sort_values("game_date").tail(15)
        result[team] = float(recent["adj"].mean()) if len(recent) >= 5 else global_mean
    return result


# ---------------------------------------------------------------------------
# ROLLING TIE RATE
# ---------------------------------------------------------------------------

def get_rolling_tie_rate(date_str: str, actuals: pd.DataFrame) -> float:
    _FALLBACK = 0.095
    cutoff       = pd.to_datetime(date_str)
    window_start = cutoff - pd.Timedelta(days=30)
    window       = actuals[(actuals["game_date"] >= window_start) &
                           (actuals["game_date"] <  cutoff)]
    if len(window) < 10:
        return _FALLBACK
    if "f5_tie" in window.columns:
        return float(window["f5_tie"].mean())
    if "f5_home_runs" in window.columns and "f5_away_runs" in window.columns:
        return float((window["f5_home_runs"] == window["f5_away_runs"]).mean())
    return _FALLBACK


# ---------------------------------------------------------------------------
# TEAM LOG-ODDS
# ---------------------------------------------------------------------------

def compute_team_log_odds(feat_row: pd.Series, feat_cols: list,
                          team_model, team_feat_cols: list) -> float:
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
# MAIN BACKTEST
# ---------------------------------------------------------------------------

def run_backtest():
    print("Loading models …")
    xgb_model, cal, feat_cols, stacker, team_model, team_feat_cols = load_models()

    print(f"Loading feature matrix ({FEAT_MATRIX.name}) …")
    fm = pd.read_parquet(FEAT_MATRIX)
    fm["game_date"] = pd.to_datetime(fm["game_date"])

    print(f"Loading actuals ({ACTUALS_2026.name}) …")
    actuals = pd.read_parquet(ACTUALS_2026)
    actuals["game_date"] = pd.to_datetime(actuals["game_date"])

    # MC expectations for adj-form computation
    mc_cols = ["game_pk", "mc_f5_home_win_pct", "mc_f5_away_win_pct"]
    fm_mc   = fm[[c for c in mc_cols if c in fm.columns]].drop_duplicates("game_pk")

    # Filter to 2026 rows in feature matrix that have actuals
    fm26 = fm[fm["game_date"].dt.year == 2026].copy()
    fm26 = fm26.merge(
        actuals[["game_pk", "f5_home_runs", "f5_away_runs", "f5_home_win"]],
        on="game_pk", how="inner"
    )
    print(f"  2026 games with features+actuals: {len(fm26)}")

    # Pre-compute per-date form dicts (avoid recomputing for every game)
    dates_sorted = sorted(fm26["game_date"].dt.strftime("%Y-%m-%d").unique())
    form_cache: dict[str, dict] = {}
    tie_cache:  dict[str, float] = {}
    print(f"  Computing rolling form for {len(dates_sorted)} unique dates …")
    for d in dates_sorted:
        form_cache[d] = compute_form_for_date(d, actuals, fm_mc)
        tie_cache[d]  = get_rolling_tie_rate(d, actuals)

    print(f"  Scoring {len(fm26)} games …")
    results = []
    fm26 = fm26.sort_values("game_date").reset_index(drop=True)

    for idx, row in fm26.iterrows():
        date_str   = row["game_date"].strftime("%Y-%m-%d")
        home_team  = row["home_team"]
        away_team  = row["away_team"]

        form_dict  = form_cache[date_str]
        global_f   = float(np.mean(list(form_dict.values()))) if form_dict else 0.0
        home_form  = form_dict.get(home_team, global_f)
        away_form  = form_dict.get(away_team, global_f)
        rolling_tie = tie_cache[date_str]

        feat = row.copy()
        feat["home_rolling_adj_f5_form"] = home_form
        feat["away_rolling_adj_f5_form"] = away_form
        feat["rolling_adj_f5_form_diff"] = home_form - away_form

        # L1
        X_l1 = feat[feat_cols].fillna(0).values.reshape(1, -1).astype(np.float32)
        raw   = xgb_model.predict_proba(X_l1)[0, 1]
        cal_p = cal.predict_proba([[raw]])[0, 1]

        # Team log-odds
        team_lo = compute_team_log_odds(feat, feat_cols, team_model, team_feat_cols)

        # L2
        feat_aug = feat.copy()
        feat_aug["team_f5_log_odds"]    = team_lo
        feat_aug["rolling_f5_tie_rate"] = rolling_tie
        feat_row_df = pd.DataFrame([feat_aug])

        seg   = _derive_segment_id(feat_row_df)
        stk_p = stacker.predict(np.array([raw]), feat_row_df, seg)[0]

        actual_win = int(row["f5_home_win"])
        hr         = row["f5_home_runs"]
        ar         = row["f5_away_runs"]
        tie        = int(hr == ar)

        results.append({
            "date":         date_str,
            "game_pk":      int(row["game_pk"]),
            "home_team":    home_team,
            "away_team":    away_team,
            "home_sp":      row.get("home_starter_name", ""),
            "away_sp":      row.get("away_starter_name", ""),
            "l1_xgb_raw":   round(float(raw), 4),
            "l1_cal":       round(float(cal_p), 4),
            "l2_stacker":   round(float(stk_p), 4),
            "team_log_odds":round(float(team_lo), 4),
            "rolling_tie":  round(rolling_tie, 4),
            "home_form":    round(home_form, 4),
            "away_form":    round(away_form, 4),
            "f5_home_runs": hr,
            "f5_away_runs": ar,
            "f5_tie":       tie,
            "actual_win":   actual_win,
        })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved → {OUTPUT_CSV}")
    return df


# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------

def print_report(df: pd.DataFrame):
    y_true = df["actual_win"].values

    auc_l1 = roc_auc_score(y_true, df["l1_cal"].values)
    auc_l1_raw = roc_auc_score(y_true, df["l1_xgb_raw"].values)
    auc_l2 = roc_auc_score(y_true, df["l2_stacker"].values)
    tie_rate = df["f5_tie"].mean()
    cover_rate = df["actual_win"].mean()

    print("\n" + "=" * 65)
    print(f"  F5 BACKTEST 2026  ({df['date'].min()} → {df['date'].max()})")
    print("=" * 65)
    print(f"\n  Games           : {len(df)}")
    print(f"  Home cover rate : {cover_rate:.3f}  (home wins F5 +0.5)")
    print(f"  F5 tie rate     : {tie_rate:.3f}")
    print(f"\n  AUC  L1-raw (XGB)  : {auc_l1_raw:.4f}")
    print(f"  AUC  L1-cal (Platt): {auc_l1:.4f}")
    print(f"  AUC  L2 (Stacker)  : {auc_l2:.4f}")
    print()

    # ── Calibration ──────────────────────────────────────────────────────
    BINS   = [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.0]
    LABELS = ["<.40", ".40-.45", ".45-.50", ".50-.55", ".55-.60",
              ".60-.65", ".65-.70", ".70-.80", ">.80"]

    for label, col in [("L1-cal", "l1_cal"), ("L2-stacker", "l2_stacker")]:
        print(f"  --- {label} Calibration ---")
        print(f"  {'Bucket':>10}  {'n':>5}  {'Pred%':>7}  {'Actual%':>8}  {'Diff':>7}")
        print(f"  {'-'*45}")
        df["_bkt"] = pd.cut(df[col], bins=BINS, labels=LABELS)
        cal = df.groupby("_bkt", observed=True).agg(
            n      =(col,          "count"),
            pred   =(col,          "mean"),
            actual =("actual_win", "mean"),
        )
        for bkt, r in cal.iterrows():
            diff = r["actual"] - r["pred"]
            flag = "  <-- skewed" if r["n"] >= 10 and abs(diff) > 0.08 else ""
            print(f"  {bkt:>10}  {int(r['n']):>5}  {r['pred']:>7.3f}  {r['actual']:>8.3f}  "
                  f"{diff:>+7.3f}{flag}")
        df.drop(columns=["_bkt"], inplace=True)
        print()

    # ── Signal performance ────────────────────────────────────────────────
    print("  --- Signal Performance (directional, Pred >= threshold) ---")
    print(f"  {'Threshold':>12}  {'Col':>12}  {'Bets':>5}  {'W':>4}  {'Win%':>7}  {'ROI':>7}  {'Edge':>7}")
    print(f"  {'-'*65}")

    for thresh in [0.60, 0.65, 0.70, 0.75]:
        for col, lbl in [("l1_cal", "L1"), ("l2_stacker", "L2")]:
            # Home bets
            sub_h = df[df[col] >= thresh]
            if len(sub_h) > 0:
                w = sub_h["actual_win"].sum()
                n = len(sub_h)
                wr = w / n
                roi = (w * (100/110) - (n - w)) / n
                edge = wr - sub_h[col].mean()
                print(f"  {f'>={thresh:.0%}':>12}  {f'{lbl} home':>12}  {n:>5}  "
                      f"{int(w):>4}  {wr:>7.1%}  {roi:>+7.1%}  {edge:>+7.3f}")

            # Away bets (model says AWAY cover, i.e., pred_home < 1-thresh)
            away_thresh = 1.0 - thresh
            sub_a = df[df[col] <= away_thresh]
            if len(sub_a) > 0:
                w = (1 - sub_a["actual_win"]).sum()  # away covers
                n = len(sub_a)
                wr = w / n
                roi = (w * (100/110) - (n - w)) / n
                edge = wr - (1 - sub_a[col].mean())
                print(f"  {f'<={away_thresh:.0%}':>12}  {f'{lbl} away':>12}  {n:>5}  "
                      f"{int(w):>4}  {wr:>7.1%}  {roi:>+7.1%}  {edge:>+7.3f}")
    print()

    # ── By month ──────────────────────────────────────────────────────────
    df["month"] = pd.to_datetime(df["date"]).dt.month
    print("  --- AUC by Month ---")
    print(f"  {'Month':>8}  {'N':>5}  {'L1-AUC':>8}  {'L2-AUC':>8}")
    print(f"  {'-'*34}")
    for m, mdf in df.groupby("month"):
        if len(mdf) < 20:
            continue
        try:
            a1 = roc_auc_score(mdf["actual_win"], mdf["l1_cal"])
            a2 = roc_auc_score(mdf["actual_win"], mdf["l2_stacker"])
            print(f"  {m:>8}  {len(mdf):>5}  {a1:>8.4f}  {a2:>8.4f}")
        except Exception:
            pass
    print()

    # ── Overall accuracy ─────────────────────────────────────────────────
    l1_correct = ((df["l1_cal"] >= 0.5) == df["actual_win"].astype(bool)).mean()
    l2_correct = ((df["l2_stacker"] >= 0.5) == df["actual_win"].astype(bool)).mean()
    print(f"  Directional accuracy (>50% threshold):")
    print(f"    L1: {l1_correct:.1%}")
    print(f"    L2: {l2_correct:.1%}")
    print()

    # ── Top confident calls review ────────────────────────────────────────
    print("  --- Top 20 L2 Confident Calls (all time) ---")
    top = df.nlargest(20, "l2_stacker")[
        ["date", "away_team", "home_team", "l1_cal", "l2_stacker", "actual_win",
         "f5_home_runs", "f5_away_runs"]
    ]
    for _, r in top.iterrows():
        result = "WIN" if r["actual_win"] == 1 else "LOSS"
        tie    = " (TIE)" if r["f5_home_runs"] == r["f5_away_runs"] else ""
        print(f"    {r['date']}  {r['away_team']:5s} @ {r['home_team']:5s}  "
              f"L1={r['l1_cal']:.1%}  L2={r['l2_stacker']:.1%}  "
              f"F5:{int(r['f5_away_runs'])}-{int(r['f5_home_runs'])}  {result}{tie}")
    print()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true",
                        help="Print summary only from existing CSV")
    args = parser.parse_args()

    if args.summary and OUTPUT_CSV.exists():
        df = pd.read_csv(OUTPUT_CSV)
    else:
        df = run_backtest()

    print_report(df)
