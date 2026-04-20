"""
score_f5_today.py
=================
Score today's games with the full two-level F5 stack:

  L1: XGBoost + OOF-fitted Platt calibration
  L2: Bayesian Hierarchical Stacker
        inputs: XGB raw prob (logit), SP diffs, matchup edge,
                MC physics, team log-odds ratio, rolling tie rate

Outputs L1 and L2 probabilities side-by-side, and compares to actual
F5 outcomes if actuals_2026.parquet contains today's data.

Usage:
  python score_f5_today.py
  python score_f5_today.py --date 2026-04-19
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

# Import helpers from train_f5_model (stacker class, flip utils, log-odds)
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
BASE_DIR        = Path(".")
DATA_DIR        = BASE_DIR / "data" / "statcast"
MODELS_DIR      = BASE_DIR / "models"
FEAT_MATRIX     = BASE_DIR / "feature_matrix_enriched_v2.parquet"

FEAT_COLS_PATH  = MODELS_DIR / "f5_feature_cols.json"
XGB_F5_PATH     = MODELS_DIR / "xgb_f5.json"
XGB_CAL_PATH    = MODELS_DIR / "xgb_f5_calibrator.pkl"
STACKER_PATH    = MODELS_DIR / "stacking_lr_f5.pkl"
TEAM_MODEL_PATH = MODELS_DIR / "team_f5_model.json"
TEAM_FEAT_PATH  = MODELS_DIR / "team_f5_feat_cols.json"
ACTUALS_2026    = DATA_DIR   / "actuals_2026.parquet"


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

def load_models():
    """Load all model artifacts — raises clear errors if anything is missing."""
    for p in [FEAT_COLS_PATH, XGB_F5_PATH, XGB_CAL_PATH, STACKER_PATH,
              TEAM_MODEL_PATH, TEAM_FEAT_PATH]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing model file: {p}\n"
                "Run: python train_f5_model.py --matrix feature_matrix_with_2026.parquet --with-2026"
            )

    feat_cols      = json.load(open(FEAT_COLS_PATH))
    team_feat_cols = json.load(open(TEAM_FEAT_PATH))

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(XGB_F5_PATH))

    team_model = xgb.XGBClassifier()
    team_model.load_model(str(TEAM_MODEL_PATH))

    cal = pickle.load(open(XGB_CAL_PATH, "rb"))

    # Stacker pickle was saved when train_f5_model was __main__, so
    # BayesianStackerF5 is stored as __main__.BayesianStackerF5.
    # Inject the class into __main__ before unpickling so it resolves
    # regardless of which module is currently __main__.
    import sys as _sys
    import __main__ as _main
    if not hasattr(_main, "BayesianStackerF5"):
        _main.BayesianStackerF5 = BayesianStackerF5
    stacker = pickle.load(open(STACKER_PATH, "rb"))

    return xgb_model, cal, feat_cols, stacker, team_model, team_feat_cols


# ---------------------------------------------------------------------------
# ROLLING TIE RATE
# ---------------------------------------------------------------------------

def get_rolling_tie_rate(date_str: str) -> float:
    """
    Compute the 30-day F5 tie rate using games BEFORE date_str (no look-ahead).
    Falls back to historical mean (~0.095) when actuals are unavailable or sparse.
    """
    _FALLBACK = 0.095

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

    if "f5_tie" in window.columns:
        return float(window["f5_tie"].mean())
    if "f5_home_runs" in window.columns and "f5_away_runs" in window.columns:
        return float((window["f5_home_runs"] == window["f5_away_runs"]).mean())

    return _FALLBACK


def compute_rolling_adj_f5_form_today(date_str: str, fm: pd.DataFrame) -> dict[str, float]:
    """
    Compute rolling 15-game capped opp-adjusted F5 form for each team as of date_str.
    Mirrors _compute_rolling_adj_f5_form() in train_f5_model.py.
    Uses actuals_2026.parquet for F5 outcomes + feature matrix for MC expectations.
    Returns {team: form_value}. Fallback = 0.0 (neutral).
    """
    if not ACTUALS_2026.exists():
        return {}

    act = pd.read_parquet(ACTUALS_2026)
    act["game_date"] = pd.to_datetime(act["game_date"])
    cutoff = pd.to_datetime(date_str)

    # Only games before today
    hist = act[act["game_date"] < cutoff].copy()
    if len(hist) == 0:
        return {}

    # Get MC expectation columns from feature matrix
    mc_cols = [c for c in ["game_pk", "mc_f5_home_win_pct", "mc_f5_away_win_pct"]
               if c in fm.columns]
    if len(mc_cols) == 3:
        fm_mc = fm[mc_cols].drop_duplicates("game_pk")
        hist  = hist.merge(fm_mc, on="game_pk", how="left")
    else:
        hist["mc_f5_home_win_pct"] = 0.5
        hist["mc_f5_away_win_pct"] = 0.5

    hist["mc_f5_home_win_pct"] = hist["mc_f5_home_win_pct"].fillna(0.5)
    hist["mc_f5_away_win_pct"] = hist["mc_f5_away_win_pct"].fillna(0.5)

    # Per-game adj scores
    raw_margin   = (hist["f5_home_runs"] - hist["f5_away_runs"]).astype(float)
    capped       = raw_margin.clip(-4, 4)
    mc_exp       = hist["mc_f5_home_win_pct"] - hist["mc_f5_away_win_pct"]
    hist["adj_home"] = capped - mc_exp
    hist["adj_away"] = -hist["adj_home"]

    # Long format
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
    result: dict[str, float] = {}
    for team, grp in long_df.groupby("team"):
        recent = grp.sort_values("game_date").tail(15)
        if len(recent) >= 5:
            result[team] = float(recent["adj"].mean())
        else:
            result[team] = global_mean

    return result


# ---------------------------------------------------------------------------
# NEW ENRICHED-FEATURE HELPERS
# ---------------------------------------------------------------------------

def build_sp_lookup(fm: pd.DataFrame) -> dict:
    """
    Build a lookup: SP name (upper) → most recent 1st-inning + durability stats.
    Uses both home and away starter columns from the enriched feature matrix.
    """
    lookup = {}
    fm_sorted = fm.sort_values("game_date")

    SP_COLS_HOME = ["home_sp_avg_ip", "home_sp_1st_k_pct", "home_sp_1st_bb_pct", "home_sp_1st_xwoba"]
    SP_COLS_AWAY = ["away_sp_avg_ip", "away_sp_1st_k_pct", "away_sp_1st_bb_pct", "away_sp_1st_xwoba"]

    home_cols_ok = all(c in fm_sorted.columns for c in SP_COLS_HOME)
    away_cols_ok = all(c in fm_sorted.columns for c in SP_COLS_AWAY)

    if home_cols_ok:
        for _, row in fm_sorted.iterrows():
            name = str(row.get("home_starter_name", "")).upper().strip()
            if name:
                lookup[name] = {
                    "avg_ip":      row["home_sp_avg_ip"],
                    "1st_k_pct":   row["home_sp_1st_k_pct"],
                    "1st_bb_pct":  row["home_sp_1st_bb_pct"],
                    "1st_xwoba":   row["home_sp_1st_xwoba"],
                }
    if away_cols_ok:
        for _, row in fm_sorted.iterrows():
            name = str(row.get("away_starter_name", "")).upper().strip()
            if name:
                lookup[name] = {
                    "avg_ip":      row["away_sp_avg_ip"],
                    "1st_k_pct":   row["away_sp_1st_k_pct"],
                    "1st_bb_pct":  row["away_sp_1st_bb_pct"],
                    "1st_xwoba":   row["away_sp_1st_xwoba"],
                }
    return lookup


def get_weather_today(date_str: str, fm: pd.DataFrame) -> dict:
    """Load today's weather from weather_2026.parquet. Falls back to fm median if missing."""
    _DEFAULTS = {
        "temp_f":       float(fm["temp_f"].median())       if "temp_f"       in fm.columns else 72.0,
        "wind_mph":     float(fm["wind_mph"].median())     if "wind_mph"     in fm.columns else 7.0,
        "wind_bearing": float(fm["wind_bearing"].median()) if "wind_bearing" in fm.columns else 180.0,
    }
    weather_path = DATA_DIR / "weather_2026.parquet"
    if not weather_path.exists():
        return {}
    w = pd.read_parquet(weather_path)
    w["game_date"] = pd.to_datetime(w["game_date"])
    today = w[w["game_date"].dt.strftime("%Y-%m-%d") == date_str]
    result = {}
    for _, row in today.iterrows():
        result[row["home_team"]] = {
            "temp_f":       float(row.get("temp_f",       _DEFAULTS["temp_f"])),
            "wind_mph":     float(row.get("wind_mph",     _DEFAULTS["wind_mph"])),
            "wind_bearing": float(row.get("wind_bearing", _DEFAULTS["wind_bearing"])),
        }
    return result


def get_bullpen_avail_today(date_str: str, fm: pd.DataFrame) -> dict:
    """Load today's bullpen availability. Falls back to 0 (fresh) if missing."""
    bp_path = DATA_DIR / "bullpen_avail_2026.parquet"
    if not bp_path.exists():
        return {}
    bp = pd.read_parquet(bp_path)
    bp["game_date"] = pd.to_datetime(bp["game_date"])
    today = bp[bp["game_date"].dt.strftime("%Y-%m-%d") == date_str]
    result = {}
    for _, row in today.iterrows():
        result[row["team"]] = {
            "bp_depleted_flag":   int(row.get("bp_depleted_flag",   0)),
            "bp_pitches_rest1d":  float(row.get("bp_pitches_rest1d", 0.0)),
        }
    return result


def compute_sp_days_rest(date_str: str, sp_name: str, fm: pd.DataFrame) -> float:
    """Find how many days since this SP's last start. Returns 5.0 if unknown."""
    if not sp_name or pd.isna(sp_name):
        return 5.0
    sp_up  = sp_name.upper().strip()
    cutoff = pd.to_datetime(date_str)

    home_dates = fm.loc[
        fm["home_starter_name"].str.upper().str.strip() == sp_up, "game_date"
    ]
    away_dates = fm.loc[
        fm["away_starter_name"].str.upper().str.strip() == sp_up, "game_date"
    ]
    all_dates = pd.concat([home_dates, away_dates])
    all_dates = pd.to_datetime(all_dates)
    prior = all_dates[all_dates < cutoff]

    if len(prior) == 0:
        return 5.0
    last_start = prior.max()
    days = int((cutoff - last_start).days)
    return float(np.clip(days, 3, 15))


# ---------------------------------------------------------------------------
# LINEUP LOADING
# ---------------------------------------------------------------------------

def get_todays_games(date_str: str) -> pd.DataFrame:
    """Load today's lineup parquet (falls back to lineups_today.parquet)."""
    lineup_path = DATA_DIR / f"lineups_{date_str}.parquet"
    if lineup_path.exists():
        lineups = pd.read_parquet(lineup_path)
    else:
        fallback = DATA_DIR / "lineups_today.parquet"
        if not fallback.exists():
            raise FileNotFoundError(
                f"No lineup file found for {date_str}.\n"
                f"Expected: {lineup_path}"
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
    """
    Build a feature vector for today's game using the most recent historical
    row for the home team, with SP features overwritten from each pitcher's
    most recent appearance.
    """
    fm_sorted = fm[fm["home_team"] == home_team].sort_values("game_date")
    if len(fm_sorted) == 0:
        return None

    base = fm_sorted.iloc[-1].copy()

    # ── Home SP features ─────────────────────────────────────────────────
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
                # SP appeared as away in the historical row — mirror away→home
                for c in feat_cols:
                    if c.startswith("home_sp_") and not c.endswith("diff"):
                        base[c] = sp_row.get("away_sp_" + c[8:], base.get(c))

    # ── Away SP features ─────────────────────────────────────────────────
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
                # SP appeared as home — mirror home→away
                for c in feat_cols:
                    if c.startswith("away_sp_") and not c.endswith("diff"):
                        base[c] = sp_row.get("home_sp_" + c[8:], base.get(c))

    # ── Recompute SP diff features ────────────────────────────────────────
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

    # ── Schedule features ─────────────────────────────────────────────────
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
    """
    Compute team_f5_log_odds = logit(p_home) - logit(p_away).

    Runs the doubled-dataset team model on:
      - home perspective (is_home=1, original features)
      - away perspective (is_home=0, flipped home/away features)

    The log-odds difference is tie-safe: on +0.5 lines both p_home and
    p_away are elevated by ties, but the log-odds difference cancels out
    the tie inflation and isolates the relative home/away strength signal.
    """
    row_df = pd.DataFrame([feat_row])

    # Home perspective
    row_home = row_df.copy()
    row_home["is_home"] = 1
    for c in team_feat_cols:
        if c not in row_home.columns:
            row_home[c] = 0.0
    p_h = team_model.predict_proba(
        row_home[team_feat_cols].fillna(0).values.astype(np.float32)
    )[0, 1]

    # Away perspective — flip home_X ↔ away_X, negate diffs
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
    """Score all games on date_str and print L1 + L2 predictions."""
    print(f"Loading models …")
    xgb_model, cal, feat_cols, stacker, team_model, team_feat_cols = load_models()

    print(f"Loading feature matrix from {FEAT_MATRIX} …")
    fm = pd.read_parquet(FEAT_MATRIX)
    fm["game_date"] = pd.to_datetime(fm["game_date"])

    rolling_tie = get_rolling_tie_rate(date_str)
    print(f"Rolling 30-day F5 tie rate: {rolling_tie:.4f}")

    form_dict   = compute_rolling_adj_f5_form_today(date_str, fm)
    form_global = float(np.mean(list(form_dict.values()))) if form_dict else 0.0
    print(f"Rolling adj F5 form: {len(form_dict)} teams  global_mean={form_global:.4f}")

    # Pre-compute today's lookup tables
    sp_lookup    = build_sp_lookup(fm)
    weather_dict = get_weather_today(date_str, fm)
    bp_dict      = get_bullpen_avail_today(date_str, fm)

    # Global defaults from feature matrix medians
    _default_avg_ip    = float(fm["home_sp_avg_ip"].median())    if "home_sp_avg_ip"    in fm.columns else 5.1
    _default_1st_k     = float(fm["home_sp_1st_k_pct"].median()) if "home_sp_1st_k_pct" in fm.columns else 0.231
    _default_1st_bb    = float(fm["home_sp_1st_bb_pct"].median())if "home_sp_1st_bb_pct"in fm.columns else 0.085
    _default_1st_xwoba = float(fm["home_sp_1st_xwoba"].median()) if "home_sp_1st_xwoba"  in fm.columns else 0.330

    print(f"SP lookup: {len(sp_lookup)} pitchers")
    print(f"Weather:   {len(weather_dict)} home parks")
    print(f"Bullpen:   {len(bp_dict)} teams")

    lineups = get_todays_games(date_str)
    print(f"\n{len(lineups)} games scheduled for {date_str}\n")

    # Load actual outcomes if available (for post-game comparison)
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

        feat = build_game_feature_row(
            home, away, home_sp, away_sp, date_str, fm, feat_cols
        )
        if feat is None:
            rows.append({
                "home_team": home, "away_team": away,
                "home_sp": home_sp, "away_sp": away_sp,
                "xgb_l1": None, "stacker_l2": None,
            })
            continue

        # Inject rolling adj F5 form features
        home_form = form_dict.get(home, form_global)
        away_form = form_dict.get(away, form_global)
        feat = feat.copy()
        feat["home_rolling_adj_f5_form"] = home_form
        feat["away_rolling_adj_f5_form"] = away_form
        feat["rolling_adj_f5_form_diff"] = home_form - away_form

        # ── Weather ──────────────────────────────────────────────────────────
        wx = weather_dict.get(home, {})
        if wx:
            feat["temp_f"]       = wx.get("temp_f",       72.0)
            feat["wind_mph"]     = wx.get("wind_mph",     7.0)
            feat["wind_bearing"] = wx.get("wind_bearing", 180.0)
        else:
            feat["temp_f"]       = float(fm["temp_f"].median())       if "temp_f"       in fm.columns else 72.0
            feat["wind_mph"]     = float(fm["wind_mph"].median())     if "wind_mph"     in fm.columns else 7.0
            feat["wind_bearing"] = float(fm["wind_bearing"].median()) if "wind_bearing" in fm.columns else 180.0

        # ── Bullpen availability ──────────────────────────────────────────────
        bp_home = bp_dict.get(home, {})
        bp_away = bp_dict.get(away, {})
        feat["home_bp_depleted_flag"]  = bp_home.get("bp_depleted_flag",  0)
        feat["home_bp_pitches_rest1d"] = bp_home.get("bp_pitches_rest1d", 0.0)
        feat["away_bp_depleted_flag"]  = bp_away.get("bp_depleted_flag",  0)
        feat["away_bp_pitches_rest1d"] = bp_away.get("bp_pitches_rest1d", 0.0)

        # ── SP days of rest ───────────────────────────────────────────────────
        feat["home_sp_days_rest"] = compute_sp_days_rest(date_str, home_sp, fm)
        feat["away_sp_days_rest"] = compute_sp_days_rest(date_str, away_sp, fm)

        # ── SP avg IP + 1st inning splits ────────────────────────────────────
        h_sp_up    = str(home_sp).upper().strip()
        a_sp_up    = str(away_sp).upper().strip()
        h_sp_stats = sp_lookup.get(h_sp_up, {})
        a_sp_stats = sp_lookup.get(a_sp_up, {})

        feat["home_sp_avg_ip"]      = h_sp_stats.get("avg_ip",    _default_avg_ip)
        feat["home_sp_1st_k_pct"]   = h_sp_stats.get("1st_k_pct", _default_1st_k)
        feat["home_sp_1st_bb_pct"]  = h_sp_stats.get("1st_bb_pct",_default_1st_bb)
        feat["home_sp_1st_xwoba"]   = h_sp_stats.get("1st_xwoba", _default_1st_xwoba)

        feat["away_sp_avg_ip"]      = a_sp_stats.get("avg_ip",    _default_avg_ip)
        feat["away_sp_1st_k_pct"]   = a_sp_stats.get("1st_k_pct", _default_1st_k)
        feat["away_sp_1st_bb_pct"]  = a_sp_stats.get("1st_bb_pct",_default_1st_bb)
        feat["away_sp_1st_xwoba"]   = a_sp_stats.get("1st_xwoba", _default_1st_xwoba)

        # Diff features (recompute from injected values)
        feat["sp_1st_k_pct_diff"]  = feat["home_sp_1st_k_pct"]  - feat["away_sp_1st_k_pct"]
        feat["sp_1st_xwoba_diff"]  = feat["home_sp_1st_xwoba"]  - feat["away_sp_1st_xwoba"]

        # ── L1: XGBoost + OOF-fitted Platt ───────────────────────────────
        X_l1 = feat[feat_cols].fillna(0).values.reshape(1, -1).astype(np.float32)
        raw   = xgb_model.predict_proba(X_l1)[0, 1]
        cal_p = cal.predict_proba([[raw]])[0, 1]

        # ── Team log-odds (stacker domain feature) ────────────────────────
        team_lo = compute_team_log_odds(feat, feat_cols, team_model, team_feat_cols)

        # ── L2: Bayesian Hierarchical Stacker ─────────────────────────────
        # Augment the feature Series with the two stacker-only columns,
        # then build a 1-row DataFrame for the stacker's _build_X_domain().
        feat_aug = feat.copy()
        feat_aug["team_f5_log_odds"]    = team_lo
        feat_aug["rolling_f5_tie_rate"] = rolling_tie
        feat_row_df = pd.DataFrame([feat_aug])

        seg   = _derive_segment_id(feat_row_df)
        stk_p = stacker.predict(np.array([raw]), feat_row_df, seg)[0]

        act_row = actuals.get(gk, {})
        rows.append({
            "home_team":           home,
            "away_team":           away,
            "home_sp":             home_sp,
            "away_sp":             away_sp,
            "xgb_l1":              round(cal_p,  3),
            "stacker_l2":          round(stk_p,  3),
            "team_log_odds":       round(team_lo, 3),
            "actual_f5_home":      act_row.get("f5_home_runs"),
            "actual_f5_away":      act_row.get("f5_away_runs"),
            "actual_f5_win":       act_row.get("f5_home_win"),
            # Diagnostic: lookup hit booleans
            "home_sp_lookup_hit":  bool(h_sp_stats),
            "away_sp_lookup_hit":  bool(a_sp_stats),
            "weather_hit":         bool(wx),
            "home_bp_hit":         bool(bp_home),
            "away_bp_hit":         bool(bp_away),
            # Injected values for reporting
            "temp_f":              feat["temp_f"],
            "wind_mph":            feat["wind_mph"],
            "wind_bearing":        feat["wind_bearing"],
            "home_bp_depleted":    feat["home_bp_depleted_flag"],
            "home_bp_rest1d":      feat["home_bp_pitches_rest1d"],
            "away_bp_depleted":    feat["away_bp_depleted_flag"],
            "away_bp_rest1d":      feat["away_bp_pitches_rest1d"],
        })

    df = pd.DataFrame(rows)

    # ── Print table ───────────────────────────────────────────────────────
    has_actuals = df["actual_f5_win"].notna().any()
    print(f"  {'Away':6s}  @  {'Home':6s}  {'Away SP':22s}  {'Home SP':22s}  "
          f"{'L1%':6s}  {'L2%':6s}  {'LogOdds':>8s}",
          end="  Actual\n" if has_actuals else "\n")
    print("  " + "-" * (95 if has_actuals else 82))

    correct_l1 = correct_l2 = total_games = 0
    sort_col   = "stacker_l2" if df["stacker_l2"].notna().any() else "xgb_l1"
    for _, r in df.sort_values(sort_col, ascending=False, na_position="last").iterrows():
        if r["stacker_l2"] is None:
            print(f"  {r['away_team']:6s}  @  {r['home_team']:6s}  — no features available —")
            continue

        l1 = r["xgb_l1"]
        l2 = r["stacker_l2"]
        lo = r["team_log_odds"]
        pred_l1 = "HOME" if l1 >= 0.50 else "AWAY"
        pred_l2 = "HOME" if l2 >= 0.50 else "AWAY"

        line = (f"  {r['away_team']:6s}  @  {r['home_team']:6s}  "
                f"{str(r['away_sp']):22s}  {str(r['home_sp']):22s}  "
                f"{l1:6.1%}  {l2:6.1%}  {lo:>+8.3f}")

        if has_actuals and r["actual_f5_win"] is not None:
            hr     = int(r["actual_f5_home"])
            ar     = int(r["actual_f5_away"])
            covers = int(hr >= ar)   # home covers +0.5 iff home_runs >= away_runs
            result = "HOME" if r["actual_f5_win"] == 1 else ("TIE" if hr == ar else "AWAY")
            c1 = "+" if (pred_l1 == "HOME") == bool(covers) else "-"
            c2 = "+" if (pred_l2 == "HOME") == bool(covers) else "-"
            total_games += 1
            correct_l1  += int((pred_l1 == "HOME") == bool(covers))
            correct_l2  += int((pred_l2 == "HOME") == bool(covers))
            line += f"  {ar}–{hr} F5  {result:4s}  L1:{c1} L2:{c2}"
        print(line)

    if total_games > 0:
        print(f"\n  L1 correct: {correct_l1}/{total_games}  ({correct_l1/total_games:.1%})")
        print(f"  L2 correct: {correct_l2}/{total_games}  ({correct_l2/total_games:.1%})")

    return df


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score today's games with the F5 XGBoost + Bayesian Stacker"
    )
    parser.add_argument("--date", default=None,
                        help="Date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    from datetime import date
    date_str = args.date or date.today().isoformat()
    predict_games(date_str)


if __name__ == "__main__":
    main()
