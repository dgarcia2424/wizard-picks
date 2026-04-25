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
    _prob_home_cover_from_lambdas,
    _prob_tie_from_lambdas,
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
POIS_HOME_PATH    = MODELS_DIR / "f5_pois_home.json"
POIS_AWAY_PATH    = MODELS_DIR / "f5_pois_away.json"
TTO_PROFILES_2026 = DATA_DIR   / "pitcher_profiles_2026.parquet"
ACTUALS_2026      = DATA_DIR   / "actuals_2026.parquet"


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

    # Poisson boosters for Skellam sidecar (λ_home, λ_away → P(tie), P(cover))
    pois_home = pois_away = None
    if POIS_HOME_PATH.exists() and POIS_AWAY_PATH.exists():
        import xgboost as _xgb
        pois_home = _xgb.Booster()
        pois_home.load_model(str(POIS_HOME_PATH))
        pois_away = _xgb.Booster()
        pois_away.load_model(str(POIS_AWAY_PATH))
    else:
        print(f"[WARN] Poisson boosters not found — Skellam features will use training medians")

    # Stacker pickle was saved when train_f5_model was __main__, so
    # BayesianStackerF5 is stored as __main__.BayesianStackerF5.
    # Inject the class into __main__ before unpickling so it resolves
    # regardless of which module is currently __main__.
    import sys as _sys
    import __main__ as _main
    if not hasattr(_main, "BayesianStackerF5"):
        _main.BayesianStackerF5 = BayesianStackerF5
    stacker = pickle.load(open(STACKER_PATH, "rb"))

    return xgb_model, cal, feat_cols, stacker, team_model, team_feat_cols, pois_home, pois_away


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


def load_tto_lookup() -> dict[str, dict]:
    """
    Build pitcher_name_upper → {tto1_xwoba, tto2_xwoba, tto_xwoba_climb} lookup
    from pitcher_profiles_2026.parquet. Returns {} if file missing.
    """
    if not TTO_PROFILES_2026.exists():
        return {}
    tto_cols = ["pitcher_name_upper", "tto1_xwoba", "tto2_xwoba", "tto_xwoba_climb"]
    pp = pd.read_parquet(TTO_PROFILES_2026, columns=tto_cols).dropna(subset=["pitcher_name_upper"])
    pp["pitcher_name_upper"] = pp["pitcher_name_upper"].str.upper().str.strip()
    lookup = {}
    for _, row in pp.iterrows():
        lookup[row["pitcher_name_upper"]] = {
            "tto1_xwoba":  row["tto1_xwoba"],
            "tto2_xwoba":  row["tto2_xwoba"],
            "tto_xwoba_climb": row["tto_xwoba_climb"],
        }
    return lookup


def load_abs_pitcher_history() -> dict[str, pd.DataFrame]:
    """
    Build pitcher_name_upper → DataFrame of (game_date, whiff_pct, abs_vulnerability)
    from abs_features_2026.parquet. Used to compute trailing rolling ABS stats at inference.
    Returns {} if file missing.
    """
    abs_path = DATA_DIR / "abs_features_2026.parquet"
    if not abs_path.exists():
        return {}
    want_cols = ["game_date", "home_sp", "away_sp",
                 "home_sp_whiff_pct", "away_sp_whiff_pct",
                 "home_sp_abs_vulnerability", "away_sp_abs_vulnerability",
                 "home_sp_f5_whiff_pct", "away_sp_f5_whiff_pct",
                 "home_sp_f5_abs_vulnerability", "away_sp_f5_abs_vulnerability"]
    try:
        df_all = pd.read_parquet(abs_path)
        df = df_all[[c for c in want_cols if c in df_all.columns]]
    except Exception:
        return {}
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Build long-format: one row per SP per game
    has_f5 = "home_sp_f5_whiff_pct" in df.columns
    h_cols = ["game_date", "home_sp", "home_sp_whiff_pct", "home_sp_abs_vulnerability"]
    a_cols = ["game_date", "away_sp", "away_sp_whiff_pct", "away_sp_abs_vulnerability"]
    out_cols = ["game_date", "sp_name", "whiff_pct", "abs_vulnerability"]
    if has_f5:
        h_cols += ["home_sp_f5_whiff_pct", "home_sp_f5_abs_vulnerability"]
        a_cols += ["away_sp_f5_whiff_pct", "away_sp_f5_abs_vulnerability"]
        out_cols += ["f5_whiff_pct", "f5_abs_vulnerability"]
    home_rows = df[h_cols].copy()
    home_rows.columns = out_cols
    away_rows = df[a_cols].copy()
    away_rows.columns = out_cols

    long = pd.concat([home_rows, away_rows], ignore_index=True)
    long = long.dropna(subset=["sp_name"])
    long["sp_name"] = long["sp_name"].str.upper().str.strip()
    long = long.sort_values("game_date")

    lookup = {}
    for name, grp in long.groupby("sp_name"):
        lookup[name] = grp[["game_date", "whiff_pct", "abs_vulnerability"]].reset_index(drop=True)
    return lookup


def get_rolling_abs_for_pitcher(name: str, date_str: str,
                                 abs_history: dict, window_days: int = 30) -> dict:
    """Return rolling trailing ABS stats for a pitcher before date_str."""
    key = name.upper().strip() if name and not pd.isna(name) else ""
    hist = abs_history.get(key)
    if hist is None or hist.empty:
        return {}
    cutoff = pd.Timestamp(date_str)
    window_start = cutoff - pd.Timedelta(days=window_days)
    mask = (hist["game_date"] >= window_start) & (hist["game_date"] < cutoff)
    recent = hist[mask]
    if recent.empty:
        return {}
    result = {
        "whiff_pct":         float(recent["whiff_pct"].mean()),
        "abs_vulnerability": float(recent["abs_vulnerability"].mean()),
    }
    if "f5_whiff_pct" in recent.columns:
        result["f5_whiff_pct"]         = float(recent["f5_whiff_pct"].mean())
        result["f5_abs_vulnerability"] = float(recent["f5_abs_vulnerability"].mean())
    return result


def inject_abs_features(feat: "pd.Series", home_sp: str, away_sp: str,
                         date_str: str, abs_history: dict) -> "pd.Series":
    """Inject rolling ABS features into a feature Series."""
    feat = feat.copy()
    h = get_rolling_abs_for_pitcher(home_sp, date_str, abs_history)
    a = get_rolling_abs_for_pitcher(away_sp, date_str, abs_history)
    feat["home_sp_whiff_pct"]          = h.get("whiff_pct",           np.nan)
    feat["away_sp_whiff_pct"]          = a.get("whiff_pct",           np.nan)
    feat["home_sp_abs_vulnerability"]  = h.get("abs_vulnerability",   np.nan)
    feat["away_sp_abs_vulnerability"]  = a.get("abs_vulnerability",   np.nan)
    feat["home_sp_f5_whiff_pct"]       = h.get("f5_whiff_pct",        np.nan)
    feat["away_sp_f5_whiff_pct"]       = a.get("f5_whiff_pct",        np.nan)
    feat["home_sp_f5_abs_vulnerability"] = h.get("f5_abs_vulnerability", np.nan)
    feat["away_sp_f5_abs_vulnerability"] = a.get("f5_abs_vulnerability", np.nan)

    def _diff(k1, k2):
        v1, v2 = feat[k1], feat[k2]
        return v1 - v2 if pd.notna(v1) and pd.notna(v2) else np.nan

    feat["sp_whiff_pct_diff"]             = _diff("home_sp_whiff_pct",       "away_sp_whiff_pct")
    feat["sp_abs_vulnerability_diff"]     = _diff("home_sp_abs_vulnerability","away_sp_abs_vulnerability")
    feat["sp_f5_whiff_pct_diff"]          = _diff("home_sp_f5_whiff_pct",    "away_sp_f5_whiff_pct")
    feat["sp_f5_abs_vulnerability_diff"]  = _diff("home_sp_f5_abs_vulnerability","away_sp_f5_abs_vulnerability")
    return feat


def inject_tto_features(feat: "pd.Series", home_sp: str, away_sp: str,
                        tto_lookup: dict) -> "pd.Series":
    """Inject TTO features into a feature Series from the lookup dict."""
    feat = feat.copy()
    h_key = home_sp.upper().strip() if home_sp and not pd.isna(home_sp) else ""
    a_key = away_sp.upper().strip() if away_sp and not pd.isna(away_sp) else ""
    h_tto = tto_lookup.get(h_key, {})
    a_tto = tto_lookup.get(a_key, {})
    feat["home_sp_tto1_xwoba"]      = h_tto.get("tto1_xwoba",       np.nan)
    feat["away_sp_tto1_xwoba"]      = a_tto.get("tto1_xwoba",       np.nan)
    feat["home_sp_tto2_xwoba"]      = h_tto.get("tto2_xwoba",       np.nan)
    feat["away_sp_tto2_xwoba"]      = a_tto.get("tto2_xwoba",       np.nan)
    feat["home_sp_tto_xwoba_climb"] = h_tto.get("tto_xwoba_climb",  np.nan)
    feat["away_sp_tto_xwoba_climb"] = a_tto.get("tto_xwoba_climb",  np.nan)
    h2 = feat["home_sp_tto2_xwoba"]
    a2 = feat["away_sp_tto2_xwoba"]
    hc = feat["home_sp_tto_xwoba_climb"]
    ac = feat["away_sp_tto_xwoba_climb"]
    feat["sp_tto2_xwoba_diff"]      = h2 - a2 if pd.notna(h2) and pd.notna(a2) else np.nan
    feat["sp_tto_xwoba_climb_diff"] = hc - ac if pd.notna(hc) and pd.notna(ac) else np.nan
    return feat


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
    xgb_model, cal, feat_cols, stacker, team_model, team_feat_cols, pois_home, pois_away = load_models()

    print(f"Loading feature matrix from {FEAT_MATRIX} …")
    fm = pd.read_parquet(FEAT_MATRIX)
    fm["game_date"] = pd.to_datetime(fm["game_date"])

    tto_lookup  = load_tto_lookup()
    print(f"TTO lookup: {len(tto_lookup)} pitchers loaded")
    abs_history = load_abs_pitcher_history()
    print(f"ABS history: {len(abs_history)} pitchers loaded")

    rolling_tie = get_rolling_tie_rate(date_str)
    print(f"Rolling 30-day F5 tie rate: {rolling_tie:.4f}")

    form_dict   = compute_rolling_adj_f5_form_today(date_str, fm)
    form_global = float(np.mean(list(form_dict.values()))) if form_dict else 0.0
    print(f"Rolling adj F5 form: {len(form_dict)} teams  global_mean={form_global:.4f}")

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

        # Inject rolling ABS features (whiff_pct, abs_vulnerability) from game-level history
        feat = inject_abs_features(feat, home_sp, away_sp, date_str, abs_history)

        # Inject TTO features from pitcher_profiles_2026 lookup
        feat = inject_tto_features(feat, home_sp, away_sp, tto_lookup)

        # Ensure feat has all feat_cols (ABS/TTO features absent from enriched matrix → 0.0)
        feat = feat.copy()
        for _c in feat_cols:
            if _c not in feat.index:
                feat[_c] = 0.0

        # ── L1: XGBoost + OOF-fitted Platt ───────────────────────────────
        feat_df = pd.DataFrame([feat.reindex(feat_cols, fill_value=0.0).fillna(0.0)], columns=feat_cols)

        # --- START VALIDATION BLOCK ---
        expected_cols = json.load(open(FEAT_COLS_PATH))
        if expected_cols != list(feat_df.columns):
            print(f"[WARN] F5 scorer: feature list mismatch for {home}@{away} — cols may have drifted")
        nan_cols = feat_df.columns[feat_df.isnull().any()].tolist()
        if nan_cols:
            print(f"[WARN] F5 scorer: NaNs after reindex for {home}@{away} in {nan_cols} — filling 0")
            feat_df = feat_df.fillna(0.0)
        # --- END VALIDATION BLOCK ---

        X_l1 = feat_df.values.astype(np.float32)
        raw   = xgb_model.predict_proba(X_l1)[0, 1]
        cal_p = cal.predict_proba([[raw]])[0, 1]

        # ── Team log-odds (stacker domain feature) ────────────────────────
        team_lo = compute_team_log_odds(feat, feat_cols, team_model, team_feat_cols)

        # ── Skellam sidecar: compute λ_home, λ_away → P(cover), P(tie) ───────
        pois_lam_home = pois_lam_away = pois_p_cover = pois_p_tie = None
        if pois_home is not None and pois_away is not None:
            import xgboost as _xgb
            _dmat = _xgb.DMatrix(X_l1)
            pois_lam_home = float(pois_home.predict(_dmat)[0])
            pois_lam_away = float(pois_away.predict(_dmat)[0])
            pois_p_cover  = float(_prob_home_cover_from_lambdas(
                np.array([pois_lam_home]), np.array([pois_lam_away]))[0])
            pois_p_tie    = float(_prob_tie_from_lambdas(
                np.array([pois_lam_home]), np.array([pois_lam_away]))[0])

        # ── L2: Bayesian Hierarchical Stacker ─────────────────────────────
        # Augment the feature Series with stacker-only columns, including
        # real-time Skellam features (not training-median fill).
        feat_aug = feat.copy()
        feat_aug["team_f5_log_odds"]    = team_lo
        feat_aug["rolling_f5_tie_rate"] = rolling_tie
        if pois_lam_home is not None:
            feat_aug["pois_lam_home"] = pois_lam_home
            feat_aug["pois_lam_away"] = pois_lam_away
            feat_aug["pois_p_cover"]  = pois_p_cover
            feat_aug["pois_p_tie"]    = pois_p_tie
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
