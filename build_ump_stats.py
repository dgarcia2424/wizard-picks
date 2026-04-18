"""
build_ump_stats.py
==================
Compute per-home-plate-umpire trailing tendency stats for the feature matrix.

For each game, the HP umpire's historical K% and BB% tendencies (relative to
the league average on those same game days) are computed using an EWMA with
the same 30-day halflife used for pitcher stats.  The shift(1) ensures no
look-ahead leakage — the feature for game G reflects only games 1..G-1.

Features produced
-----------------
  ump_k_above_avg    HP umpire's trailing K% minus league-avg K% that season
                     Positive = tight zone (fewer calls) → pitcher-friendly
  ump_bb_above_avg   HP umpire's trailing BB% minus league-avg BB%
                     Positive = generous zone → more walks
  ump_rpg_above_avg  HP umpire's trailing runs/game minus league avg
                     Positive = high-scoring environment umpire

Output
------
  data/statcast/ump_features_{year}.parquet   (one row per game_pk)
  Columns: game_pk, year, ump_hp_id, ump_hp_name,
           ump_k_above_avg, ump_bb_above_avg, ump_rpg_above_avg

Usage
-----
  python build_ump_stats.py                   # builds 2023-2025
  python build_ump_stats.py --years 2026      # builds current year
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path("./data/statcast")
EWMA_HALFLIFE_UMP_DAYS = 30   # same as SP — ~10 games for a daily umpire


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _time_ewm(df: pd.DataFrame, group_col: str, date_col: str,
              col: str, halflife_days: int) -> pd.Series:
    """Time-based EWMA with shift(1) to prevent look-ahead leakage."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    hl  = pd.Timedelta(days=halflife_days)
    for _, grp in df.groupby(group_col, sort=False):
        grp  = grp.sort_values(date_col)
        ewma = grp[col].ewm(halflife=hl, times=grp[date_col]).mean().shift(1)
        out.loc[ewma.index] = ewma.values
    return out


def build_ump_features(years: list[int], verbose: bool = True) -> None:
    """
    Build and save ump_features_{year}.parquet for each year.

    Requires:
      data/statcast/umpire_assignments_{year}.parquet  (from ump_pull.py)
      data/statcast/statcast_{year}.parquet            (pitch-level data)
      data/statcast/schedule_all_{year}.parquet        (scores per game)
    """
    for year in years:
        asgn_path = DATA_DIR / f"umpire_assignments_{year}.parquet"
        sc_path   = DATA_DIR / f"statcast_{year}.parquet"
        sched_path = DATA_DIR / f"schedule_all_{year}.parquet"
        out_path  = DATA_DIR / f"ump_features_{year}.parquet"

        if not asgn_path.exists():
            if verbose:
                print(f"  {year}: umpire_assignments not found — run ump_pull.py first")
            continue
        if not sc_path.exists():
            if verbose:
                print(f"  {year}: statcast_{year}.parquet not found — skipping")
            continue

        if verbose:
            print(f"  {year}: computing umpire tendency stats ...")

        # ── Load assignments ─────────────────────────────────────────────
        asgn = pd.read_parquet(asgn_path, engine="pyarrow")
        asgn["game_pk"] = _to_num(asgn["game_pk"])
        asgn = asgn.dropna(subset=["game_pk"]).copy()

        # ── Load statcast pitch data — K and BB events per game ──────────
        sc_cols = ["game_pk", "game_date", "events", "woba_denom"]
        sc = pd.read_parquet(sc_path, engine="pyarrow", columns=sc_cols)
        sc["game_pk"]  = _to_num(sc["game_pk"])
        sc["game_date"] = pd.to_datetime(sc["game_date"])
        sc["is_pa"] = (_to_num(sc["woba_denom"]) > 0).astype(float)
        sc["is_k"]  = (sc["events"] == "strikeout").astype(float)
        sc["is_bb"] = (sc["events"] == "walk").astype(float)

        # Per-game totals (regular season only: n_pa >= 10 guards vs spring)
        game_stats = (
            sc.groupby("game_pk")
            .agg(
                game_date = ("game_date", "first"),
                n_pa      = ("is_pa",     "sum"),
                n_k       = ("is_k",      "sum"),
                n_bb      = ("is_bb",     "sum"),
            )
            .reset_index()
        )
        game_stats = game_stats[game_stats["n_pa"] >= 10].copy()
        game_stats["k_pct"]  = game_stats["n_k"]  / game_stats["n_pa"].clip(lower=1)
        game_stats["bb_pct"] = game_stats["n_bb"] / game_stats["n_pa"].clip(lower=1)

        # ── Load scores for runs/game ────────────────────────────────────
        if sched_path.exists():
            sched = pd.read_parquet(sched_path, engine="pyarrow")
            sched = sched[sched["home_away"] == "Home"][
                ["gamePk", "home_score", "away_score"]
            ].rename(columns={"gamePk": "game_pk"})
            sched["game_pk"] = _to_num(sched["game_pk"])
            sched = sched.drop_duplicates("game_pk")
            sched["total_runs"] = (
                _to_num(sched["home_score"]) + _to_num(sched["away_score"])
            )
            game_stats = game_stats.merge(
                sched[["game_pk", "total_runs"]], on="game_pk", how="left"
            )
        else:
            game_stats["total_runs"] = np.nan

        # ── Join umpire to game stats ────────────────────────────────────
        ump_games = asgn.merge(game_stats, on="game_pk", how="inner")
        ump_games = ump_games.sort_values(["ump_hp_id", "game_date"]).reset_index(drop=True)

        if len(ump_games) == 0:
            if verbose:
                print(f"  {year}: no ump-game rows after join — skipping")
            continue

        # ── League-average K%, BB%, runs/game for the season ────────────
        lg_k   = float(game_stats["k_pct"].mean())
        lg_bb  = float(game_stats["bb_pct"].mean())
        lg_rpg = float(game_stats["total_runs"].mean()) if "total_runs" in game_stats else np.nan

        ump_games["k_above_avg"]   = ump_games["k_pct"]   - lg_k
        ump_games["bb_above_avg"]  = ump_games["bb_pct"]  - lg_bb
        ump_games["rpg_above_avg"] = (
            ump_games["total_runs"] - lg_rpg
            if "total_runs" in ump_games.columns else np.nan
        )

        # ── EWMA tendency per umpire (shift=1 for no leakage) ────────────
        for stat in ["k_above_avg", "bb_above_avg", "rpg_above_avg"]:
            if stat in ump_games.columns:
                ump_games[f"ump_{stat}_ewma"] = _time_ewm(
                    ump_games, "ump_hp_id", "game_date", stat,
                    EWMA_HALFLIFE_UMP_DAYS
                )

        # ── Output: one row per game_pk with EWMA tendency features ──────
        feat_cols = ["game_pk", "ump_hp_id", "ump_hp_name",
                     "ump_k_above_avg_ewma",
                     "ump_bb_above_avg_ewma",
                     "ump_rpg_above_avg_ewma"]
        feat_cols = [c for c in feat_cols if c in ump_games.columns]

        out = ump_games[feat_cols].rename(columns={
            "ump_k_above_avg_ewma":   "ump_k_above_avg",
            "ump_bb_above_avg_ewma":  "ump_bb_above_avg",
            "ump_rpg_above_avg_ewma": "ump_rpg_above_avg",
        })
        out.to_parquet(out_path, engine="pyarrow", index=False)

        n_umps  = out["ump_hp_id"].nunique()
        n_games = len(out)
        null_k  = out["ump_k_above_avg"].isna().mean() * 100
        if verbose:
            print(f"  {year}: {n_games} games | {n_umps} umpires | "
                  f"ump_k_above_avg null={null_k:.1f}% "
                  f"| mean={out['ump_k_above_avg'].mean():.4f} "
                  f"-> {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build umpire tendency features")
    parser.add_argument("--years", type=int, nargs="+",
                        default=[2023, 2024, 2025])
    args = parser.parse_args()
    print("=" * 55)
    print("  build_ump_stats.py  - Umpire Tendency Features")
    print("=" * 55)
    build_ump_features(args.years)
    print("\n  Done.")


if __name__ == "__main__":
    main()
