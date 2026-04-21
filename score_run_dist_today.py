"""
score_run_dist_today.py
=======================
Sterile scorer for the unified Run Distribution stack.

Given today's lineups (data/statcast/lineups_<DATE>.parquet), produce per-game:
   lam_home, lam_away, total_line,
   p_over_dc,  p_over_xgb,  p_over_final,
   p_cover_dc, p_cover_xgb, p_home_cover_final

The "final" columns are the L2 Bayesian-stacker outputs.

Artifacts required (written by train_run_dist_model.py):
   models/dc_model_run_dist.pkl
   models/xgb_run_dist_lam_home.json
   models/xgb_run_dist_lam_away.json
   models/stacker_totals.pkl
   models/stacker_rl.pkl
   models/run_dist_feature_cols.json

Usage:
   python score_run_dist_today.py
   python score_run_dist_today.py --date 2026-04-21
"""

from __future__ import annotations

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

# DRY: reuse feature-row builder and DC class from their source modules
sys.path.insert(0, str(Path(__file__).parent))
from score_ml_today import build_game_feature_row, get_todays_games
from dixon_coles_mlb import DixonColesMLB
from train_run_dist_model import (
    BayesianStackerTotals,
    BayesianStackerRL,
    p_over_poisson,
    p_cover_poisson,
    _logit,
    FALLBACK_TOTAL_LINE,
    RUNLINE,
    K_TRUNC,
)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR    = Path(".")
DATA_DIR    = BASE_DIR / "data" / "statcast"
MODELS_DIR  = BASE_DIR / "models"
FEAT_MATRIX = BASE_DIR / "feature_matrix_enriched_v2.parquet"

FEAT_COLS_PATH       = MODELS_DIR / "run_dist_feature_cols.json"
DC_PATH              = MODELS_DIR / "dc_model_run_dist.pkl"
XGB_LAM_HOME_PATH    = MODELS_DIR / "xgb_run_dist_lam_home.json"
XGB_LAM_AWAY_PATH    = MODELS_DIR / "xgb_run_dist_lam_away.json"
STACKER_TOTALS_PATH  = MODELS_DIR / "stacker_totals.pkl"
STACKER_RL_PATH      = MODELS_DIR / "stacker_rl.pkl"
ACTUALS_2026         = DATA_DIR   / "actuals_2026.parquet"


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------
def load_models():
    for p in [FEAT_COLS_PATH, DC_PATH, XGB_LAM_HOME_PATH, XGB_LAM_AWAY_PATH,
              STACKER_TOTALS_PATH, STACKER_RL_PATH]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing artifact: {p}\n"
                "Run: python train_run_dist_model.py "
                "--matrix feature_matrix_enriched_v2.parquet --with-2026"
            )
    feat_cols = json.load(open(FEAT_COLS_PATH))

    dc = pickle.loads(DC_PATH.read_bytes())

    m_home = xgb.XGBRegressor()
    m_home.load_model(str(XGB_LAM_HOME_PATH))
    m_away = xgb.XGBRegressor()
    m_away.load_model(str(XGB_LAM_AWAY_PATH))

    # Inject stacker classes under __main__ in case they were saved from there.
    import __main__ as _main
    if not hasattr(_main, "BayesianStackerTotals"):
        _main.BayesianStackerTotals = BayesianStackerTotals
    if not hasattr(_main, "BayesianStackerRL"):
        _main.BayesianStackerRL = BayesianStackerRL

    st_tot = pickle.loads(STACKER_TOTALS_PATH.read_bytes())
    st_rl  = pickle.loads(STACKER_RL_PATH.read_bytes())
    return feat_cols, dc, m_home, m_away, st_tot, st_rl


# ---------------------------------------------------------------------------
# TOTAL LINE LOOKUP
# ---------------------------------------------------------------------------
def _resolve_total_line_for_game(game_pk, date_str: str, fm: pd.DataFrame) -> float:
    """Prefer 'close_total' or equivalent from today's feature row / actuals."""
    # Check actuals first (market moves get ingested there)
    if ACTUALS_2026.exists():
        try:
            act = pd.read_parquet(ACTUALS_2026)
            if "game_pk" in act.columns:
                row = act[act["game_pk"] == game_pk]
                for col in ("close_total", "open_total", "vegas_total",
                            "total_line", "ou_line"):
                    if col in row.columns and len(row) > 0:
                        v = pd.to_numeric(row[col], errors="coerce").dropna()
                        if len(v) > 0:
                            return float(v.iloc[0])
        except Exception:
            pass

    # Fall back to feature matrix (same game_pk)
    if "game_pk" in fm.columns:
        row = fm[fm["game_pk"] == game_pk]
        for col in ("close_total", "open_total", "vegas_total",
                    "total_line", "ou_line"):
            if col in row.columns and len(row) > 0:
                v = pd.to_numeric(row[col], errors="coerce").dropna()
                if len(v) > 0:
                    return float(v.iloc[0])
    return float(FALLBACK_TOTAL_LINE)


# ---------------------------------------------------------------------------
# MAIN PREDICTION LOOP
# ---------------------------------------------------------------------------
def predict_games(date_str: str) -> pd.DataFrame:
    print("Loading models …")
    feat_cols, dc, m_home, m_away, st_tot, st_rl = load_models()

    print(f"Loading feature matrix from {FEAT_MATRIX} …")
    fm = pd.read_parquet(FEAT_MATRIX)
    fm["game_date"] = pd.to_datetime(fm["game_date"])

    lineups = get_todays_games(date_str)
    print(f"\n{len(lineups)} games scheduled for {date_str}\n")

    rows = []
    for _, g in lineups.iterrows():
        home    = g["home_team"]
        away    = g["away_team"]
        home_sp = g.get("home_starter_name", "")
        away_sp = g.get("away_starter_name", "")
        gk      = g.get("game_pk")

        feat = build_game_feature_row(home, away, home_sp, away_sp,
                                        date_str, fm, feat_cols)
        if feat is None:
            rows.append({
                "home_team": home, "away_team": away,
                "home_sp":   home_sp, "away_sp":   away_sp,
                "lam_home": None, "lam_away": None,
                "total_line": None,
                "p_over_dc": None, "p_over_xgb": None, "p_over_final": None,
                "p_cover_dc": None, "p_cover_xgb": None,
                "p_home_cover_final": None,
            })
            continue

        # ── Build feature frame + VALIDATION BLOCK ───────────────────────
        feat_df = pd.DataFrame([feat[feat_cols].fillna(0)], columns=feat_cols)

        # --- START VALIDATION BLOCK ---
        expected_cols = json.load(open(FEAT_COLS_PATH))
        actual_cols = list(feat_df.columns)
        assert expected_cols == actual_cols, "Feature list mismatch. Check manifest vs feat_df."
        assert not feat_df[expected_cols].isnull().any().any(), "Unexpected NaNs in feature frame."
        # --- END VALIDATION BLOCK ---

        X = feat_df.values.astype(np.float32)

        # ── Dual-Poisson XGB lambdas ────────────────────────────────────
        lam_h = float(np.clip(m_home.predict(X)[0], 0.1, 15.0))
        lam_a = float(np.clip(m_away.predict(X)[0], 0.1, 15.0))

        # ── Vegas total line ────────────────────────────────────────────
        total_line = _resolve_total_line_for_game(gk, date_str, fm)

        # ── XGB-side probs (analytic convolution) ───────────────────────
        p_over_xgb  = p_over_poisson(lam_h, lam_a, total_line, k_max=K_TRUNC)
        p_cover_xgb = p_cover_poisson(lam_h, lam_a, RUNLINE, k_max=K_TRUNC)

        # ── DC-side probs ───────────────────────────────────────────────
        M = dc.predict_match_matrix(home, away)
        K = M.shape[0]
        idx_h = np.arange(K).reshape(-1, 1)
        idx_a = np.arange(K).reshape(1, -1)
        p_over_dc  = float(M[(idx_h + idx_a) > total_line].sum())
        p_cover_dc = float(M[(idx_h - idx_a) > RUNLINE].sum())
        total_dc   = float((M * (idx_h + idx_a)).sum())
        diff_dc    = float((M * (idx_h - idx_a)).sum())

        # ── L2 Stackers ─────────────────────────────────────────────────
        aux_tot = pd.DataFrame({
            "logit_p_over_xgb": [float(_logit(np.array([p_over_xgb]))[0])],
            "lam_sum":          [lam_h + lam_a],
            "total_dc":         [total_dc],
        })
        p_over_final = float(st_tot.predict(np.array([p_over_dc]), aux_tot)[0])

        aux_rl = pd.DataFrame({
            "logit_p_cover_xgb": [float(_logit(np.array([p_cover_xgb]))[0])],
            "lam_diff":          [lam_h - lam_a],
            "diff_dc":           [diff_dc],
        })
        p_home_cover_final = float(st_rl.predict(np.array([p_cover_dc]), aux_rl)[0])

        rows.append({
            "home_team":          home,
            "away_team":          away,
            "home_sp":            home_sp,
            "away_sp":            away_sp,
            "lam_home":           round(lam_h, 3),
            "lam_away":           round(lam_a, 3),
            "total_line":         round(total_line, 2),
            "p_over_dc":          round(p_over_dc,  4),
            "p_over_xgb":         round(p_over_xgb, 4),
            "p_over_final":       round(p_over_final, 4),
            "p_cover_dc":         round(p_cover_dc,  4),
            "p_cover_xgb":        round(p_cover_xgb, 4),
            "p_home_cover_final": round(p_home_cover_final, 4),
        })

    df = pd.DataFrame(rows)

    # ── Pretty-print table sorted by largest edge ────────────────────────
    if len(df) > 0 and df["p_over_final"].notna().any():
        df["over_edge"]  = (df["p_over_final"].fillna(0.5) - 0.5).abs()
        df["cover_edge"] = (df["p_home_cover_final"].fillna(0.5) - 0.5).abs()
        df["max_edge"]   = df[["over_edge", "cover_edge"]].max(axis=1)
        df_sorted = df.sort_values("max_edge", ascending=False)
    else:
        df_sorted = df

    print(f"  {'Away':>6s} @ {'Home':<6s}  {'Line':>5s}  "
          f"{'lam_h':>6s} {'lam_a':>6s}  "
          f"{'Over_DC':>8s} {'Over_XGB':>9s} {'OverFin':>8s}  "
          f"{'Cov_DC':>7s} {'Cov_XGB':>8s} {'CovFin':>7s}")
    print("  " + "-" * 104)
    for _, r in df_sorted.iterrows():
        if r.get("lam_home") is None:
            print(f"  {r['away_team']:>6s} @ {r['home_team']:<6s}   [no features]")
            continue
        print(
            f"  {r['away_team']:>6s} @ {r['home_team']:<6s}  "
            f"{r['total_line']:>5.1f}  "
            f"{r['lam_home']:>6.2f} {r['lam_away']:>6.2f}  "
            f"{r['p_over_dc']:>8.3f} {r['p_over_xgb']:>9.3f} {r['p_over_final']:>8.3f}  "
            f"{r['p_cover_dc']:>7.3f} {r['p_cover_xgb']:>8.3f} {r['p_home_cover_final']:>7.3f}"
        )

    # Drop helper edge cols for the returned frame
    return df.drop(columns=[c for c in ("over_edge", "cover_edge", "max_edge")
                              if c in df.columns], errors="ignore")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Score today's games with the Run Distribution stack"
    )
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    args = parser.parse_args()
    from datetime import date
    date_str = args.date or date.today().isoformat()
    predict_games(date_str)


if __name__ == "__main__":
    main()
