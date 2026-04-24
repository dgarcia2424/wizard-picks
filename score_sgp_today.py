"""
score_sgp_today.py — SGP Scorer (Script A: The Dominance).

Scores today's slate for the joint SGP:
  Leg 1: SP K >= 5.5 (K-Over)
  Leg 2: Game Total < 8.0 (Under)

The correlation thesis: books price each leg independently.
  Edge = P(joint, model) − P(joint, independent)

Outputs:
    data/sgp/sgp_picks_today.csv   — all games ranked by edge
    data/sgp/sgp_picks_today.json  — machine-readable

NOTE: FanDuel/book odds not in any local data source.
      Book_Odds column is a stub.  Edge is measured against the
      independence assumption (the structural mispricing), not
      against an observed book line.

Usage:
    python score_sgp_today.py [--date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

_ROOT         = Path(__file__).resolve().parent
CONTEXT_FILE  = _ROOT / "data/orchestrator/daily_context.parquet"
FEATURE_MTX   = _ROOT / "feature_matrix_enriched_v2.parquet"
K_MODEL_PATH  = _ROOT / "models/k_over_v1.json"
A_MODEL_PATH  = _ROOT / "models/script_a_v1.json"
OUT_DIR       = _ROOT / "data/sgp"

K_THRESHOLD   = 5.5
TOTAL_UNDER   = 8.0
MIN_EDGE      = 0.01   # minimum correlation edge to surface a pick

# ── Feature lists (must match training) ────────────────────────────────
K_FEATURES = [
    "sp_k_pct", "sp_bb_pct", "sp_whiff_pctl", "sp_ff_velo",
    "sp_age_pit", "sp_arm_angle", "sp_k_pct_10d", "sp_bb_pct_10d",
    "sp_k_minus_bb", "sp_k_bb_ratio", "sp_k_bb_ratio_10d",
    "sp_1st_k_pct", "sp_xwoba_against", "sp_gb_pct",
    "opp_bat_k_rate", "ump_k_above_avg", "home_park_factor", "temp_f",
]

A_FEATURES = [
    "sp_k_pct", "sp_bb_pct", "sp_whiff_pctl", "sp_ff_velo",
    "sp_k_pct_10d", "sp_k_bb_ratio", "sp_1st_k_pct",
    "sp_xwoba_against", "sp_gb_pct", "opp_bat_k_rate",
    "ump_k_above_avg", "home_park_factor", "temp_f",
    "home_bullpen_vulnerability", "away_bullpen_vulnerability",
    "elo_diff", "close_total",
]


def _load_fm_latest() -> pd.DataFrame:
    """Load the most recent row per home_team from the feature matrix."""
    fm = pd.read_parquet(FEATURE_MTX)
    fm["game_date"] = pd.to_datetime(fm["game_date"])
    return (fm.sort_values("game_date")
              .groupby("home_team", as_index=False)
              .last())


def _fill_from_fm(ctx: pd.DataFrame, fm_latest: pd.DataFrame,
                  side: str, prefix: str) -> pd.DataFrame:
    """
    Fill missing SP/game columns from the latest FM row for this team.
    side  = 'home' | 'away'
    prefix = 'home_sp_' | 'away_sp_'
    """
    team_col = f"{side}_team"
    want = [c for c in fm_latest.columns
            if c.startswith(prefix) or c in (
                "home_bullpen_vulnerability", "away_bullpen_vulnerability",
                "elo_diff",
            )]
    sub = fm_latest[["home_team"] + want].copy()
    sub = sub.rename(columns={"home_team": team_col})
    filled = ctx.merge(sub, on=team_col, how="left", suffixes=("", "_fm"))
    for col in want:
        if col in filled.columns and f"{col}_fm" in filled.columns:
            filled[col] = filled[col].fillna(filled[f"{col}_fm"])
            filled.drop(columns=[f"{col}_fm"], inplace=True)
        elif f"{col}_fm" in filled.columns:
            filled[col] = filled[f"{col}_fm"]
            filled.drop(columns=[f"{col}_fm"], inplace=True)
    return filled


def build_sp_rows(ctx: pd.DataFrame, fm_latest: pd.DataFrame) -> pd.DataFrame:
    """
    Explode each game into two SP rows (home SP, away SP).
    Each row has side-neutral 'sp_*' feature columns + game context.
    """
    ctx = _fill_from_fm(ctx, fm_latest, "home", "home_sp_")
    ctx = _fill_from_fm(ctx, fm_latest, "away", "away_sp_")

    # ── Home SP rows ────────────────────────────────────────────────────
    home_rename = {c: c.replace("home_sp_", "sp_") for c in ctx.columns
                   if c.startswith("home_sp_")}
    fh = ctx.rename(columns=home_rename).copy()
    fh["_side"]       = "home"
    fh["sp_team"]     = fh["home_team"]
    fh["starter_name"] = fh.get("home_starter_name", "")
    # opp_bat_k_rate: away batters vs home SP hand (after rename, col is sp_p_throws_R)
    if "sp_p_throws_R" in fh.columns:
        fh["opp_bat_k_rate"] = np.where(
            fh["sp_p_throws_R"] == 1,
            fh["away_bat_k_vs_rhp"], fh["away_bat_k_vs_lhp"])
    else:
        fh["opp_bat_k_rate"] = fh.get("away_bat_k_vs_rhp", np.nan)

    # ── Away SP rows ────────────────────────────────────────────────────
    away_rename = {c: c.replace("away_sp_", "sp_") for c in ctx.columns
                   if c.startswith("away_sp_")}
    fa = ctx.rename(columns=away_rename).copy()
    fa["_side"]        = "away"
    fa["sp_team"]      = fa["away_team"]
    fa["starter_name"] = fa.get("away_starter_name", "")
    # after rename, away_sp_p_throws_R → sp_p_throws_R
    if "sp_p_throws_R" in fa.columns:
        fa["opp_bat_k_rate"] = np.where(
            fa["sp_p_throws_R"] == 1,
            fa["home_bat_k_vs_rhp"], fa["home_bat_k_vs_lhp"])
    else:
        fa["opp_bat_k_rate"] = fa.get("home_bat_k_vs_rhp", np.nan)

    rows = pd.concat([fh, fa], ignore_index=True)

    # Ensure all feature columns exist (fill with NaN if absent)
    for col in set(K_FEATURES + A_FEATURES):
        if col not in rows.columns:
            rows[col] = np.nan
        rows[col] = pd.to_numeric(rows[col], errors="coerce")

    return rows


def score_models(rows: pd.DataFrame,
                 k_booster: xgb.Booster,
                 a_booster: xgb.Booster) -> pd.DataFrame:
    """Score each SP-game row with both models."""
    # K-Over
    dk = xgb.DMatrix(rows[K_FEATURES], feature_names=K_FEATURES)
    rows["p_k_over"] = k_booster.predict(
        dk, iteration_range=(0, k_booster.best_iteration + 1))

    # Script A
    da = xgb.DMatrix(rows[A_FEATURES], feature_names=A_FEATURES)
    rows["p_joint_model"] = a_booster.predict(
        da, iteration_range=(0, a_booster.best_iteration + 1))

    # Independence baseline
    p_under_baseline = 0.4454   # empirical base rate from training corpus
    rows["p_total_under_approx"] = p_under_baseline  # no game-total model yet
    rows["p_joint_indep"] = rows["p_k_over"] * rows["p_total_under_approx"]

    # Correlation edge (structural mispricing vs independence assumption)
    rows["corr_edge"] = rows["p_joint_model"] - rows["p_joint_indep"]

    return rows


def format_picks(rows: pd.DataFrame) -> pd.DataFrame:
    """Build output picks table sorted by edge."""
    picks = rows[[
        "game_pk", "home_team", "away_team",
        "sp_team", "_side", "starter_name",
        "close_total", "temp_f",
        "p_k_over", "p_joint_model", "p_joint_indep", "corr_edge",
    ]].copy()

    picks["game_label"]  = picks["away_team"] + " @ " + picks["home_team"]
    picks["narrative"]   = (
        "Script A — The Dominance: "
        + picks["starter_name"].fillna(picks["sp_team"])
        + f" K>={K_THRESHOLD} + Game Under {TOTAL_UNDER}"
    )
    picks["legs"] = f"SP_K>={K_THRESHOLD} | Game_Total<{TOTAL_UNDER}"

    # Edge interpretation
    picks["edge_note"] = picks["corr_edge"].apply(
        lambda e: f"+{e:.3f} vs independence (book underpricing)"
        if e > 0 else f"{e:.3f} vs independence (avoid)"
    )

    # Book odds: STUB — FanDuel not in local data
    picks["book_odds"]       = "STUB"
    picks["book_odds_note"]  = "FanDuel/BetMGM line not in local data — use correlation edge only"

    picks = picks.sort_values("corr_edge", ascending=False).reset_index(drop=True)
    picks["rank"] = picks.index + 1

    return picks[[
        "rank", "game_label", "starter_name", "sp_team", "_side",
        "close_total", "temp_f",
        "p_k_over", "p_joint_model", "p_joint_indep", "corr_edge",
        "narrative", "legs", "edge_note", "book_odds", "book_odds_note",
    ]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None,
                        help="Score date YYYY-MM-DD (default: today from context file)")
    args = parser.parse_args()

    if not CONTEXT_FILE.exists():
        raise SystemExit(
            f"[ERROR] {CONTEXT_FILE} not found — run data_orchestrator.py first")
    if not K_MODEL_PATH.exists():
        raise SystemExit(
            f"[ERROR] {K_MODEL_PATH} not found — run train_k_over_v1.py first")
    if not A_MODEL_PATH.exists():
        raise SystemExit(
            f"[ERROR] {A_MODEL_PATH} not found — run train_script_a.py first")

    ctx = pd.read_parquet(CONTEXT_FILE)
    score_date = args.date or ctx["orchestrator_date"].iloc[0]
    ctx = ctx[ctx["orchestrator_date"] == score_date].copy()
    if ctx.empty:
        raise SystemExit(f"[ERROR] No games in context for {score_date}")

    print(f"\n{'='*60}")
    print(f" score_sgp_today.py  [{score_date}]")
    print(f"{'='*60}")
    print(f"  Games in slate: {len(ctx)}")

    # Load FM fallback for missing sticky features
    print("  Loading feature matrix for FM fallback...")
    fm_latest = _load_fm_latest()

    print("  Building SP rows...")
    rows = build_sp_rows(ctx, fm_latest)
    print(f"  SP rows: {len(rows)}")

    # Load models
    k_booster = xgb.Booster()
    k_booster.load_model(str(K_MODEL_PATH))
    a_booster = xgb.Booster()
    a_booster.load_model(str(A_MODEL_PATH))
    print(f"  Models loaded: K-Over (best_iter={k_booster.best_iteration}), "
          f"Script A (best_iter={a_booster.best_iteration})")

    rows = score_models(rows, k_booster, a_booster)
    picks = format_picks(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path  = OUT_DIR / "sgp_picks_today.csv"
    json_path = OUT_DIR / "sgp_picks_today.json"
    picks.to_csv(csv_path, index=False)
    picks.to_json(json_path, orient="records", indent=2)

    # ── Console report ──────────────────────────────────────────────────
    print(f"\n{'-'*60}")
    print(f"  SGP PICKS -- {score_date}")
    print(f"{'-'*60}")

    top = picks[picks["corr_edge"] >= MIN_EDGE]
    if top.empty:
        print("  No games exceed minimum edge threshold today.")
    else:
        for _, r in top.iterrows():
            print(f"\n  #{int(r['rank'])}  {r['game_label']}")
            print(f"      SP:           {r['starter_name']} ({r['sp_team']}, {r['_side']})")
            print(f"      Close total:  {r['close_total']}")
            print(f"      Temp:         {r['temp_f']}°F")
            print(f"      P(K-Over):    {r['p_k_over']:.3f}")
            print(f"      P(Joint/mdl): {r['p_joint_model']:.3f}")
            print(f"      P(Joint/ind): {r['p_joint_indep']:.3f}")
            print(f"      Corr edge:    {r['edge_note']}")
            print(f"      Book odds:    {r['book_odds']}  [{r['book_odds_note']}]")

    print(f"\n  All {len(picks)} SP-game rows -> {csv_path}")
    print(f"  JSON                          -> {json_path}")
    print("\n=== DONE ===\n")

    return picks


if __name__ == "__main__":
    main()
