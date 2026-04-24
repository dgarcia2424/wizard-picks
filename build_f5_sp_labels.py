"""
build_f5_sp_labels.py — F5 SP K Labels + Script Label Factory.

Builds:
    data/batter_features/f5_sp_labels.parquet      — SP Ks through inning 5
    data/batter_features/script_labels.parquet     — Joint labels for A2/B/C

F5 SP K grain: one row per (game_pk, side).
    - Top inning pitching = home SP.  Bot = away SP.
    - SP = pitcher with most plate appearances in innings 1-5 per game-side.

Script labels (game-grain, one row per game_pk):
    y_a2 : SP F5 K>=4  AND  F5 total < 4.5  AND  game total < close_total
    y_b  : home_score >= 5  AND  home_score > close_total/2  AND  home win
    y_c  : game total < close_total  AND  both SP F5 K>=3  AND  |margin|<=1
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

STATCAST_YEARS = [2024, 2025, 2026]
STATCAST_DIR   = Path("data/statcast")
FEATURE_MTX    = Path("feature_matrix_enriched_v2.parquet")
OUT_DIR        = Path("data/batter_features")

# ── Thresholds ──────────────────────────────────────────────────────────────
F5_K_THRESH_SP1  = 4      # Script A2 / Script C (SP dominant in F5)
F5_K_THRESH_BOTH = 3      # Script C (BOTH SPs, lower bar)
F5_TOTAL_UNDER   = 4.5    # Script A2 F5-total leg
TEAM_SCORE_FLOOR = 5      # Script B: home team "exploded"


def _load_statcast_f5(year: int) -> pd.DataFrame:
    path = STATCAST_DIR / f"statcast_{year}.parquet"
    if not path.exists():
        print(f"  [warn] {path} not found")
        return pd.DataFrame()
    cols = ["game_pk", "game_date", "pitcher", "home_team", "away_team",
            "inning", "inning_topbot", "events", "p_throws"]
    sc = pd.read_parquet(path, engine="pyarrow", columns=cols)
    return sc[sc["inning"] <= 5].copy()


def build_f5_sp_labels() -> pd.DataFrame:
    """
    For each (game_pk, side) compute the SP's K total in innings 1-5.
    Returns DataFrame with one row per (game_pk, side).
    """
    frames = []
    for yr in STATCAST_YEARS:
        sc5 = _load_statcast_f5(yr)
        if sc5.empty:
            continue
        sc5["is_k"] = sc5["events"].isin(
            ["strikeout", "strikeout_double_play"]).astype(int)
        agg = (sc5.groupby(
                   ["game_pk", "game_date", "home_team", "away_team",
                    "pitcher", "p_throws", "inning_topbot"])
               .agg(pa_f5=("events", "count"), k_f5=("is_k", "sum"))
               .reset_index())
        # SP = most PAs per (game_pk, side)
        sp = agg.loc[agg.groupby(["game_pk", "inning_topbot"])["pa_f5"].idxmax()]
        sp["year"] = yr
        frames.append(sp)
        print(f"  {yr}: {len(sp)} SP-game-sides  "
              f"K>=4 rate={( sp['k_f5'] >= F5_K_THRESH_SP1).mean():.3f}")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["game_date"] = pd.to_datetime(out["game_date"])
    out["sp_is_home"] = out["inning_topbot"] == "Top"
    return out


def build_script_labels(sp_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Merge SP labels with FM game-level actuals to build per-game script labels.
    """
    print("  Loading feature matrix for script labels ...")
    fm_cols = ["game_pk", "game_date", "home_team", "away_team",
               "actual_game_total", "actual_f5_total", "actual_home_win",
               "close_total", "home_score", "away_score", "elo_diff",
               "close_ml_home", "close_ml_away"]
    fm_cols = [c for c in fm_cols
               if c in pd.read_parquet(FEATURE_MTX, columns=["game_pk"]).columns
               or True]
    fm = pd.read_parquet(FEATURE_MTX)
    fm_keep = [c for c in fm_cols if c in fm.columns]
    fm = fm[fm_keep].drop_duplicates("game_pk")
    fm["game_date"] = pd.to_datetime(fm["game_date"])

    # Pivot SP labels to game-grain: home_sp and away_sp columns
    home_sp = sp_labels[sp_labels["sp_is_home"]].copy()
    away_sp = sp_labels[~sp_labels["sp_is_home"]].copy()

    home_sp = home_sp[["game_pk", "pitcher", "k_f5", "pa_f5", "p_throws"]]\
                     .rename(columns={"pitcher": "home_sp_id",
                                      "k_f5":    "home_sp_k_f5",
                                      "pa_f5":   "home_sp_pa_f5",
                                      "p_throws":"home_sp_p_throws"})
    away_sp = away_sp[["game_pk", "pitcher", "k_f5", "pa_f5", "p_throws"]]\
                     .rename(columns={"pitcher": "away_sp_id",
                                      "k_f5":    "away_sp_k_f5",
                                      "pa_f5":   "away_sp_pa_f5",
                                      "p_throws":"away_sp_p_throws"})

    game = fm.merge(home_sp, on="game_pk", how="inner")\
             .merge(away_sp, on="game_pk", how="inner")
    game = game.dropna(subset=["actual_game_total", "close_total"])
    game["year"] = game["game_date"].dt.year
    game["home_margin"] = game.get("home_score", pd.Series(dtype=float)) \
                        - game.get("away_score", pd.Series(dtype=float))

    # ── Script A2 — Dominance (3 legs) ─────────────────────────────────────
    # Home SP dominant in F5 AND F5 under AND game total under line
    game["y_a2_k"]   = (game["home_sp_k_f5"] >= F5_K_THRESH_SP1).astype("int8")
    game["y_a2_f5"]  = (game["actual_f5_total"] < F5_TOTAL_UNDER).astype("int8")
    game["y_a2_tot"] = (game["actual_game_total"] < game["close_total"]).astype("int8")
    game["y_a2"]     = (game["y_a2_k"] & game["y_a2_f5"] & game["y_a2_tot"]).astype("int8")

    # Away SP version (same legs, away starter perspective)
    game["y_a2_away_k"]  = (game["away_sp_k_f5"] >= F5_K_THRESH_SP1).astype("int8")
    game["y_a2_away"]    = (game["y_a2_away_k"] & game["y_a2_f5"] & game["y_a2_tot"]).astype("int8")

    # ── Script B — Offensive Explosion (3 legs) ─────────────────────────────
    # Home team scores big AND game goes over AND home wins
    if "home_score" in game.columns:
        game["y_b_team"] = (game["home_score"] >= TEAM_SCORE_FLOOR).astype("int8")
        game["y_b_team_gt_half"] = (
            game["home_score"] > game["close_total"] / 2.0).astype("int8")
    else:
        game["y_b_team"] = 0
        game["y_b_team_gt_half"] = 0
    game["y_b_over"] = (game["actual_game_total"] > game["close_total"]).astype("int8")
    game["y_b_win"]  = game.get("actual_home_win", pd.Series(0, index=game.index)).fillna(0).astype("int8")
    game["y_b"] = (game["y_b_team"] & game["y_b_over"] & game["y_b_win"]).astype("int8")

    # ── Script C — Elite Duel (3 legs) ──────────────────────────────────────
    # Game under line AND both SPs K>=3 in F5 AND close game (underdog covers +1.5)
    game["y_c_under"] = (game["actual_game_total"] < game["close_total"]).astype("int8")
    game["y_c_both_k"] = (
        (game["home_sp_k_f5"] >= F5_K_THRESH_BOTH) &
        (game["away_sp_k_f5"] >= F5_K_THRESH_BOTH)).astype("int8")
    if "home_margin" in game.columns:
        game["y_c_close"] = (game["home_margin"].abs() <= 1).astype("int8")
    else:
        game["y_c_close"] = 0
    game["y_c"] = (game["y_c_under"] & game["y_c_both_k"] & game["y_c_close"]).astype("int8")

    print(f"\n  Script label rates (n={len(game):,}):")
    for col, name in [("y_a2","A2 Dominance"), ("y_b","B Explosion"), ("y_c","C Elite Duel")]:
        print(f"    {name}: {game[col].mean():.4f}  ({game[col].sum()} games)")

    return game


def main():
    print("=" * 60)
    print("  build_f5_sp_labels.py")
    print("=" * 60)

    print("\n[1/2] Building F5 SP K labels from Statcast ...")
    sp = build_f5_sp_labels()
    if sp.empty:
        print("  [ERROR] No statcast data found.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sp_path = OUT_DIR / "f5_sp_labels.parquet"
    sp.to_parquet(sp_path, index=False)
    print(f"  Saved -> {sp_path}  ({len(sp):,} rows)")

    print("\n[2/2] Building script joint labels ...")
    labels = build_script_labels(sp)
    if labels.empty:
        print("  [ERROR] Could not build script labels — FM join failed.")
        return

    lbl_path = OUT_DIR / "script_labels.parquet"
    labels.to_parquet(lbl_path, index=False)
    print(f"  Saved -> {lbl_path}  ({len(labels):,} rows)")
    print("\nDone.")
    return sp, labels


if __name__ == "__main__":
    main()
