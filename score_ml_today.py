"""
score_ml_today.py
=================
Score today's games with the full two-level ML (moneyline) stack:

  L1: XGBoost + OOF-fitted Platt calibration
  L2: Bayesian Hierarchical Stacker
        inputs: XGB raw prob (logit), SP diffs, matchup edge,
                bullpen diffs, team log-odds ratio, rolling ML form

Outputs L1 and L2 probabilities side-by-side, and compares to actual
home-win outcomes when actuals_2026.parquet contains today's data.

Usage:
  python score_ml_today.py
  python score_ml_today.py --date 2026-04-21
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

# Re-use helpers from train_ml_model (stacker class, flip utils, log-odds)
sys.path.insert(0, str(Path(__file__).parent))
from train_ml_model import (
    BayesianStackerML,
    ML_STACKING_FEATURES,
    _compute_log_odds_ratio,
    _derive_segment_id,
    _flip_team_perspective,
)

# ---------------------------------------------------------------------------
# MARKET-MISS SHRINKAGE CONSTANTS
# ---------------------------------------------------------------------------
# Applied when the closing market line is absent and ml_model_vs_vegas_gap
# cannot be populated with a real value.
#
# Mathematical derivation of _MARKET_MISS_LOGIT_SHRINK = 0.50:
#   For stacker output P to cross 0.7143 (logit = 0.916) after shrinkage:
#       0.50 * theta >= 0.916  →  theta >= 1.832
#   With beta ≈ 1.0 (posterior prior centre), theta ≈ logit(xgb_raw),
#   so xgb_raw must satisfy logit(xgb_raw) >= 1.832  →  xgb_raw >= 0.862.
#   This aligns with the stated standalone conviction threshold of 0.85.
_MARKET_MISS_LOGIT_SHRINK: float       = 0.50
_STANDALONE_CONVICTION_THRESHOLD: float = 0.85
_THRESHOLD_250: float                   = 250.0 / (250.0 + 100.0)   # ≈ 0.71429

# Features that encode model-vs-market gaps and must NOT be zero-filled.
# Zero-fill hallucinate perfect agreement; NaN or neutral-0 with shrinkage
# is the correct treatment.
_MARKET_GAP_FEATURES: frozenset[str] = frozenset({"ml_model_vs_vegas_gap"})


def _detect_market_missing(feat: "pd.Series") -> bool:
    """
    Return True when the closing market line is absent from the feature row.

    true_home_prob is populated by pull_odds_history_api.py from the vig-
    stripped closing Pinnacle line.  When it is NaN or absent, the stacker
    cannot receive a real ml_model_vs_vegas_gap — market data is missing.
    """
    val = feat.get("true_home_prob") if hasattr(feat, "get") else None
    if val is None:
        return True
    try:
        return not np.isfinite(float(val))
    except (TypeError, ValueError):
        return True


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR        = Path(".")
DATA_DIR        = BASE_DIR / "data" / "statcast"
CONTEXT_PATH    = BASE_DIR / "data/orchestrator/daily_context.parquet"

# v8.0 — 90th percentile of historical bullpen_burn_5d (from bullpen_burn_by_game.parquet)
BP_BURN_THRESH_90  = 360.0   # raw high-leverage pitch count
BP_BURN_PENALTY    = 0.05    # absolute win-prob adjustment when a side is gassed
MODELS_DIR      = BASE_DIR / "models"
FEAT_MATRIX     = BASE_DIR / "feature_matrix_enriched_v2.parquet"

# ── ML stack: wired to train_ml_model.py outputs (regularized XGB_PARAMS) ──
# v2 artifacts (retrain_v2.py, stale since 2026-04-23) are retired.
# All paths now match OUTPUT_* constants in train_ml_model.py.
FEAT_COLS_PATH  = MODELS_DIR / "ml_feature_cols.json"
XGB_ML_PATH     = MODELS_DIR / "xgb_ml.json"
XGB_CAL_PATH    = MODELS_DIR / "xgb_ml_calibrator.pkl"
STACKER_PATH    = MODELS_DIR / "stacking_lr_ml.pkl"
TEAM_MODEL_PATH = MODELS_DIR / "team_ml_model.json"
TEAM_FEAT_PATH  = MODELS_DIR / "team_ml_feat_cols.json"
ACTUALS_2026    = DATA_DIR   / "actuals_2026.parquet"


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

def load_models():
    for p in [FEAT_COLS_PATH, XGB_ML_PATH, XGB_CAL_PATH, STACKER_PATH,
              TEAM_MODEL_PATH, TEAM_FEAT_PATH]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing model file: {p}\n"
                "Run: python train_ml_model.py --matrix feature_matrix_enriched_v2.parquet --with-2026"
            )

    feat_cols      = json.load(open(FEAT_COLS_PATH))
    team_feat_cols = json.load(open(TEAM_FEAT_PATH))

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(XGB_ML_PATH))

    team_model = xgb.XGBClassifier()
    team_model.load_model(str(TEAM_MODEL_PATH))

    cal = pickle.load(open(XGB_CAL_PATH, "rb"))

    # Stacker pickle may reference __main__.BayesianStackerML (if saved while
    # train_ml_model was __main__). Inject the class before unpickling.
    import __main__ as _main
    if not hasattr(_main, "BayesianStackerML"):
        _main.BayesianStackerML = BayesianStackerML
    stacker = pickle.load(open(STACKER_PATH, "rb"))

    return xgb_model, cal, feat_cols, stacker, team_model, team_feat_cols


# ---------------------------------------------------------------------------
# ROLLING ML FORM
# ---------------------------------------------------------------------------

def compute_rolling_adj_ml_form_today(date_str: str, fm: pd.DataFrame) -> dict[str, float]:
    """
    Rolling 15-game opp-adjusted ML form per team as of date_str.
    Mirrors _compute_rolling_adj_ml_form() in train_ml_model.py.
    Returns {team: form_value}. Fallback = 0.0 (neutral).
    """
    if not ACTUALS_2026.exists():
        return {}

    act = pd.read_parquet(ACTUALS_2026)
    act["game_date"] = pd.to_datetime(act["game_date"])
    cutoff = pd.to_datetime(date_str)

    hist = act[act["game_date"] < cutoff].copy()
    if len(hist) == 0:
        return {}

    # Derive home_win from final scores
    if "home_score_final" in hist.columns and "away_score_final" in hist.columns:
        hist["home_win"] = (hist["home_score_final"] > hist["away_score_final"]).astype(int)
    else:
        return {}

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

    # Per-game adj scores (residual vs Poisson expectation)
    adj_home = hist["home_win"].astype(float) - hist["mc_f5_home_win_pct"]
    adj_away = (1.0 - hist["home_win"].astype(float)) - hist["mc_f5_away_win_pct"]

    home_long = pd.DataFrame({
        "game_date": hist["game_date"],
        "team":      hist["home_team"],
        "adj":       adj_home.values,
    })
    away_long = pd.DataFrame({
        "game_date": hist["game_date"],
        "team":      hist["away_team"],
        "adj":       adj_away.values,
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
    most recent appearance.  Bullpen features carry over from the base row
    since they're pre-computed in feature_matrix_enriched_v2.parquet.
    """
    fm_sorted = fm[fm["home_team"] == home_team].sort_values("game_date")
    if len(fm_sorted) == 0:
        return None

    base = fm_sorted.iloc[-1].copy()

    # Home SP
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

    # Away SP
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

    # Recompute SP diff features
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

    # Schedule features
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
    Compute team_ml_log_odds = logit(p_home) - logit(p_away) via the
    doubled-dataset team model run on both perspectives.
    """
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
    xgb_model, cal, feat_cols, stacker, team_model, team_feat_cols = load_models()

    print(f"Loading feature matrix from {FEAT_MATRIX} …")
    fm = pd.read_parquet(FEAT_MATRIX)
    fm["game_date"] = pd.to_datetime(fm["game_date"])

    form_dict   = compute_rolling_adj_ml_form_today(date_str, fm)
    form_global = float(np.mean(list(form_dict.values()))) if form_dict else 0.0
    print(f"Rolling adj ML form: {len(form_dict)} teams  global_mean={form_global:.4f}")

    lineups = get_todays_games(date_str)
    print(f"\n{len(lineups)} games scheduled for {date_str}\n")

    # Load actuals for post-game comparison
    actuals = {}
    if ACTUALS_2026.exists():
        act = pd.read_parquet(ACTUALS_2026)
        act_today = act[act["game_date"].astype(str).str.startswith(date_str)].copy()
        if "home_score_final" in act_today.columns and "away_score_final" in act_today.columns:
            act_today["home_win"] = (
                act_today["home_score_final"] > act_today["away_score_final"]
            ).astype(int)
        for _, r in act_today.iterrows():
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

        # Inject rolling ML form
        home_form = form_dict.get(home, form_global)
        away_form = form_dict.get(away, form_global)
        feat = feat.copy()
        feat["home_rolling_adj_ml_form"] = home_form
        feat["away_rolling_adj_ml_form"] = away_form
        feat["rolling_ml_form_diff"]     = home_form - away_form

        # ── Detect market data availability before any inference ─────────
        market_missing = _detect_market_missing(feat)
        if market_missing:
            print(
                f"  [MARKET-MISS] {away}@{home}: true_home_prob absent — "
                f"logit shrinkage ({_MARKET_MISS_LOGIT_SHRINK}x) will be applied."
            )

        # ── L1 feature slice + validation ────────────────────────────────
        # Two-pass reindex: non-market features filled 0.0 for XGBoost's
        # default-branch router; market-gap features left as NaN so they
        # never silently encode false model-market agreement at L1.
        # (ml_model_vs_vegas_gap is a stacking feature and not in feat_cols,
        # so this guard is forward-compatible if feature sets drift.)
        feat_reindexed = feat.reindex(feat_cols)
        non_market_l1  = [c for c in feat_cols if c not in _MARKET_GAP_FEATURES]
        feat_reindexed[non_market_l1] = feat_reindexed[non_market_l1].fillna(0.0)
        feat_df = pd.DataFrame([feat_reindexed], columns=feat_cols)

        # --- START VALIDATION BLOCK ---
        expected_cols = json.load(open(FEAT_COLS_PATH))
        if expected_cols != list(feat_df.columns):
            print(f"[WARN] ML scorer: feature list mismatch for {home}@{away} — cols may have drifted")
        nan_cols = [c for c in feat_df.columns
                    if feat_df[c].isnull().any() and c not in _MARKET_GAP_FEATURES]
        if nan_cols:
            print(f"[WARN] ML scorer: NaNs after reindex for {home}@{away} in {nan_cols} — filling 0")
            feat_df[nan_cols] = feat_df[nan_cols].fillna(0.0)
        # --- END VALIDATION BLOCK ---

        X_l1  = feat_df.values.astype(np.float32)
        raw   = xgb_model.predict_proba(X_l1)[0, 1]
        cal_p = cal.predict_proba([[raw]])[0, 1]

        # ── Team log-odds (stacker domain feature) ───────────────────────
        team_lo = compute_team_log_odds(feat, feat_cols, team_model, team_feat_cols)

        # ── L2 Bayesian Stacker ──────────────────────────────────────────
        feat_aug = feat.copy()
        feat_aug["team_ml_log_odds"]    = team_lo
        feat_aug["rolling_ml_form_diff"] = home_form - away_form
        feat_row_df = pd.DataFrame([feat_aug])

        seg   = _derive_segment_id(feat_row_df)
        stk_p = stacker.predict(np.array([raw]), feat_row_df, seg)[0]

        # ── Market-miss logit shrinkage (applied externally for BayesianStackerML) ──
        # When the closing market line is absent, the stacker's ml_model_vs_vegas_gap
        # cannot be populated with a real value.  Shrinking the stacker's output
        # logit by lambda=0.50 prevents false positives at the -250 threshold
        # unless the base XGBoost conviction is overwhelming (>= 0.85).
        if market_missing:
            logit_stk = (
                np.log(np.clip(stk_p, 1e-6, 1.0 - 1e-6))
                - np.log(1.0 - np.clip(stk_p, 1e-6, 1.0 - 1e-6))
            )
            logit_stk = logit_stk * _MARKET_MISS_LOGIT_SHRINK
            stk_p     = float(1.0 / (1.0 + np.exp(-logit_stk)))

            # Ceiling gate: block -250 crossing unless standalone conviction
            if raw < _STANDALONE_CONVICTION_THRESHOLD:
                stk_p = min(stk_p, _THRESHOLD_250 - 1e-4)

        act_row = actuals.get(gk, {})
        rows.append({
            "home_team":           home,
            "away_team":           away,
            "home_sp":             home_sp,
            "away_sp":             away_sp,
            "xgb_l1":              round(cal_p,  3),
            "stacker_l2":          round(stk_p,  3),
            "team_log_odds":       round(team_lo, 3),
            "actual_home_win":     act_row.get("home_win"),
            "market_data_missing": market_missing,
        })

    df = pd.DataFrame(rows)

    # ── Print table ───────────────────────────────────────────────────────
    has_actuals = df["actual_home_win"].notna().any()
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

        l1 = r["xgb_l1"]; l2 = r["stacker_l2"]; lo = r["team_log_odds"]
        pred_l1 = "HOME" if l1 >= 0.50 else "AWAY"
        pred_l2 = "HOME" if l2 >= 0.50 else "AWAY"

        line = (f"  {r['away_team']:6s}  @  {r['home_team']:6s}  "
                f"{str(r['away_sp']):22s}  {str(r['home_sp']):22s}  "
                f"{l1:6.1%}  {l2:6.1%}  {lo:>+8.3f}")

        if has_actuals and r["actual_home_win"] is not None and not pd.isna(r["actual_home_win"]):
            hw = int(r["actual_home_win"])
            result = "HOME" if hw == 1 else "AWAY"
            c1 = "+" if (pred_l1 == "HOME") == bool(hw) else "-"
            c2 = "+" if (pred_l2 == "HOME") == bool(hw) else "-"
            total_games += 1
            correct_l1  += int((pred_l1 == "HOME") == bool(hw))
            correct_l2  += int((pred_l2 == "HOME") == bool(hw))
            line += f"  {result:4s}  L1:{c1} L2:{c2}"
        print(line)

    if total_games > 0:
        print(f"\n  L1 correct: {correct_l1}/{total_games}  ({correct_l1/total_games:.1%})")
        print(f"  L2 correct: {correct_l2}/{total_games}  ({correct_l2/total_games:.1%})")

    # ── Alpha feature attachment ─────────────────────────────────────────────
    # Bullpen xFIP/ERA/WAR per side + home-away diff. Current L1/L2 models
    # were trained without these columns, so they do not enter inference;
    # attached here so model_scores.csv carries them for the renderer, the
    # auto-reconciler, and the next retrain cycle.
    try:
        from bullpen_features import append_bullpen_features
        df = append_bullpen_features(df)
    except Exception as e:
        print(f"[BullpenFeatures] ML scorer attach failed: {e}")

    # Manual bullpen adjustment bridge: nudge home_win_prob (L2 stacker)
    # by ±1.5pp when bullpen xFIP gap is decisive (|diff| > 0.45).
    # Thermal penalty: in extreme cold (<46°F) both bullpens see velocity/movement
    # degradation, so the xFIP edge is less reliable — halve the impact.
    # Live signal until bullpen cols enter the trained feature_cols.json.
    try:
        # Pull per-game temp from today's games.csv so the thermal gate has data.
        # games.csv keys by full name; ml_df keys by abbr — key temp by both.
        temp_by_home: dict = {}
        try:
            _games = pd.read_csv("games.csv")
            if "home_team" in _games.columns and "temp_f" in _games.columns:
                sys.path.insert(0, str(Path(__file__).parent / "wizard_agents"))
                try:
                    from tools.implementations import TEAM_NAME_TO_ABBR
                except Exception:
                    TEAM_NAME_TO_ABBR = {}
                for _, r in _games.iterrows():
                    full = r.get("home_team")
                    t    = pd.to_numeric(r.get("temp_f"), errors="coerce")
                    if pd.isna(t):
                        continue
                    temp_by_home[full] = float(t)
                    abbr = TEAM_NAME_TO_ABBR.get(full)
                    if abbr:
                        temp_by_home[abbr] = float(t)
        except Exception:
            pass

        # ── Bullpen Pivot RETIRED (v2 physics promotion, 2026-04-23) ───
        # The ±1.5pp manual bullpen bridge is deprecated — v2 ML stacker
        # now learns bullpen_xfip_diff natively (gain 7.84, rank #1 among
        # alpha features). Keeping a manual pivot on top double-counts
        # the signal. adj_home_win_prob is now identity with stacker_l2;
        # downstream ledger keeps the column for schema compatibility.
        if "stacker_l2" in df.columns:
            base = pd.to_numeric(df["stacker_l2"], errors="coerce")

            # ── v8.0 Bullpen Burn Penalty ─────────────────────────────────
            # When a team's 5-day cumulative high-leverage pitch count
            # exceeds the 90th-pctile historical threshold, apply a flat
            # ±BP_BURN_PENALTY shift to adj_home_win_prob.
            #   home gassed -> -0.05 (hurts home)
            #   away gassed -> +0.05 (hurts away / helps home)
            # Burns are loaded from daily_context (pre-computed by orchestrator).
            burn_delta = pd.Series(0.0, index=df.index)
            try:
                if CONTEXT_PATH.exists():
                    ctx = pd.read_parquet(CONTEXT_PATH, columns=[
                        "home_team", "home_bullpen_burn_5d", "away_bullpen_burn_5d"])
                    ctx_map = ctx.set_index("home_team")
                    h_burn = df["home_team"].map(ctx_map["home_bullpen_burn_5d"])
                    a_burn = df["home_team"].map(ctx_map["away_bullpen_burn_5d"])
                    h_gassed = h_burn.fillna(0) > BP_BURN_THRESH_90
                    a_gassed = a_burn.fillna(0) > BP_BURN_THRESH_90
                    burn_delta = burn_delta - h_gassed * BP_BURN_PENALTY
                    burn_delta = burn_delta + a_gassed * BP_BURN_PENALTY
                    df["home_burn_5d"]        = h_burn.round(0)
                    df["away_burn_5d"]        = a_burn.round(0)
                    df["home_burn_gassed"]    = h_gassed.astype(int)
                    df["away_burn_gassed"]    = a_gassed.astype(int)
                    n_flags = int((h_gassed | a_gassed).sum())
                    if n_flags:
                        print(f"  [BurnPenalty] {n_flags} game(s) flagged "
                              f"(thresh={BP_BURN_THRESH_90:.0f}): "
                              f"home_gassed={h_gassed.sum()}, "
                              f"away_gassed={a_gassed.sum()}")
            except Exception as _bp_exc:
                print(f"  [BurnPenalty] skipped: {_bp_exc}")

            adj = (base + burn_delta).clip(lower=0.01, upper=0.99).round(4)
            df["adj_home_win_prob"]         = adj
            df["bp_burn_delta"]             = burn_delta.round(4)
            df["bullpen_thermal_penalty"]   = 0
            df["bullpen_xfip_diff_effective"] = pd.to_numeric(
                df.get("bullpen_xfip_diff", 0.0), errors="coerce").fillna(0.0).round(3)
            df["signal_flags"] = ""
    except Exception as e:
        print(f"[BullpenAdj] failed: {e}")

    return df


# ---------------------------------------------------------------------------
# POISSON RUN LINE INTEGRATION (v8.0 TASK 4)
# ---------------------------------------------------------------------------

def _prob_to_american(p: float) -> int:
    """Convert win probability to American moneyline odds (nearest integer)."""
    p = max(0.01, min(0.99, p))
    if p >= 0.5:
        return int(round(-100 * p / (1.0 - p)))   # negative (favourite)
    else:
        return int(round(100 * (1.0 - p) / p))    # positive (underdog)


def _attach_poisson_rl(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """
    Join Poisson run-distribution outputs (lam_home, lam_away,
    p_home_cover_final) from score_run_dist_today onto the ML output.

    Adds:
      rl_lam_home, rl_lam_away     — Poisson lambdas
      rl_p_home_cover              — P(home covers -1.5) from Poisson stacker
      fair_rl_home_price           — American odds for home -1.5
      fair_rl_away_price           — American odds for away +1.5
      rl_ml_diverge_flag           — 1 when ML likes home ML but RL stacker
                                     says away covers +1.5 (or vice versa)
    """
    try:
        import score_run_dist_today as srd
        rl_df = srd.predict_games(date_str)
    except Exception as exc:
        print(f"  [PoissonRL] could not load RL scores: {exc}")
        return df

    if rl_df is None or rl_df.empty:
        return df

    keep = ["home_team", "lam_home", "lam_away",
            "p_home_cover_final", "projected_total_adj"]
    keep = [c for c in keep if c in rl_df.columns]
    rl_slim = rl_df[keep].copy().rename(columns={
        "lam_home":           "rl_lam_home",
        "lam_away":           "rl_lam_away",
        "p_home_cover_final": "rl_p_home_cover",
        "projected_total_adj":"rl_projected_total",
    })

    df = df.merge(rl_slim, on="home_team", how="left")

    if "rl_p_home_cover" in df.columns:
        p_cover = pd.to_numeric(df["rl_p_home_cover"], errors="coerce").fillna(0.5)
        df["fair_rl_home_price"] = p_cover.apply(_prob_to_american)
        df["fair_rl_away_price"] = (1.0 - p_cover).apply(_prob_to_american)

        # Divergence: ML model likes home win (> 0.55) but Poisson says away
        # covers the -1.5 spread (rl_p_home_cover < 0.45) — or vice versa
        ml_home = pd.to_numeric(df.get("adj_home_win_prob",
                                        df.get("stacker_l2", 0.5)),
                                errors="coerce").fillna(0.5)
        ml_likes_home = ml_home > 0.55
        rl_home_covers = p_cover > 0.55
        df["rl_ml_diverge_flag"] = (
            (ml_likes_home & ~rl_home_covers) |
            (~ml_likes_home & rl_home_covers)
        ).astype(int)

        n_div = int(df["rl_ml_diverge_flag"].sum())
        n_rl  = df["rl_lam_home"].notna().sum()
        print(f"  [PoissonRL] Joined {n_rl} RL scores | "
              f"{n_div} ML/RL divergence flag(s)")

    return df


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score today's games with the ML XGBoost + Bayesian Stacker"
    )
    parser.add_argument("--date", default=None,
                        help="Date YYYY-MM-DD (default: today)")
    parser.add_argument("--with-rl", action="store_true",
                        help="Attach Poisson RL lambdas and fair RL prices (v8.0)")
    args = parser.parse_args()

    from datetime import date
    date_str = args.date or date.today().isoformat()
    df = predict_games(date_str)

    if args.with_rl and df is not None and not df.empty:
        df = _attach_poisson_rl(df, date_str)
        rl_cols = ["home_team", "away_team", "rl_lam_home", "rl_lam_away",
                   "rl_p_home_cover", "fair_rl_home_price", "fair_rl_away_price",
                   "rl_ml_diverge_flag"]
        rl_cols = [c for c in rl_cols if c in df.columns]
        if rl_cols:
            print("\n  [RL Summary]")
            print(df[rl_cols].to_string(index=False))


if __name__ == "__main__":
    main()
