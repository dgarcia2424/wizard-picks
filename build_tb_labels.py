"""
build_tb_labels.py
------------------
Derive per-batter-per-game Total Bases (TB) labels from Statcast pitch-level
parquets. Foundation dataset for the Wizard v3.3 TB prop stacker.

Grain: [date, player_id, game_id, home_park_id, total_bases]

TB rubric (standard MLB definition):
    single      -> 1
    double      -> 2
    triple      -> 3
    home_run    -> 4
    walk / HBP / out / K / sac / etc -> 0

Usage:
    python build_tb_labels.py                       # 2024, 2025, 2026
    python build_tb_labels.py --years 2024 2025
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

STATCAST_DIR = Path("data/statcast")
OUT_DIR      = Path("data/batter_labels")
OUT_FILE     = OUT_DIR / "tb_by_game.parquet"

TB_MAP: dict[str, int] = {
    "single":   1,
    "double":   2,
    "triple":   3,
    "home_run": 4,
}
ZERO_TB_EVENTS: set[str] = {
    "walk", "hit_by_pitch", "intent_walk",
    "strikeout", "strikeout_double_play",
    "field_out", "force_out", "grounded_into_double_play", "double_play",
    "fielders_choice", "fielders_choice_out", "field_error",
    "sac_fly", "sac_bunt", "sac_fly_double_play", "sac_bunt_double_play",
    "catcher_interf", "truncated_pa", "other_out",
}


def _load_year(year: int) -> pd.DataFrame | None:
    f = STATCAST_DIR / f"statcast_{year}.parquet"
    if not f.exists():
        print(f"  [skip] {f} not found")
        return None
    cols = ["game_date", "game_pk", "batter", "events",
            "home_team", "inning_topbot"]
    df = pd.read_parquet(f, columns=cols)
    df = df[df["events"].notna()].copy()
    df["year"] = year
    return df


def _load_batter_name_map() -> dict[int, str]:
    """Build player_id -> player_name map from all lineup long parquets.
    Statcast's 'player_name' column is the PITCHER's name, not the batter's,
    so we use the lineup files (which are batter-centric) for display names.
    """
    name_map: dict[int, str] = {}
    for f in sorted(STATCAST_DIR.glob("lineups_*_long.parquet")):
        try:
            d = pd.read_parquet(f, columns=["player_id", "player_name"])
            for pid, pname in zip(d["player_id"], d["player_name"]):
                if pd.notna(pid) and pd.notna(pname):
                    name_map[int(pid)] = str(pname)
        except Exception:
            continue
    return name_map


def build(years: list[int]) -> pd.DataFrame:
    frames = [f for f in (_load_year(y) for y in years) if f is not None]
    if not frames:
        raise SystemExit("No statcast years loaded.")
    pa = pd.concat(frames, ignore_index=True)

    pa["tb"] = pa["events"].map(TB_MAP).fillna(0).astype("int8")
    # Sanity: flag unclassified events so we don't silently zero them.
    unknown_mask = (~pa["events"].isin(TB_MAP)) & (~pa["events"].isin(ZERO_TB_EVENTS))
    if unknown_mask.any():
        unk = pa.loc[unknown_mask, "events"].value_counts()
        print(f"\n[warn] {unknown_mask.sum()} PAs with unmapped events "
              f"(counted as 0 TB). Top offenders:")
        print(unk.head(10).to_string())

    labels = (pa.groupby(["game_date", "game_pk", "batter", "home_team"],
                          as_index=False)
                .agg(total_bases=("tb", "sum"),
                     pa_count   =("tb", "size")))
    labels = labels.rename(columns={"batter": "player_id",
                                     "game_pk": "game_id",
                                     "home_team": "home_park_id",
                                     "game_date": "date"})
    labels["year"] = pd.to_datetime(labels["date"]).dt.year.astype("int16")
    labels["total_bases"] = labels["total_bases"].astype("int8")
    labels["pa_count"]    = labels["pa_count"].astype("int8")

    # Attach batter names from the lineup parquets (statcast's player_name is
    # the pitcher — unusable for batter labelling).
    name_map = _load_batter_name_map()
    labels["player_name"] = labels["player_id"].map(name_map).fillna("")

    return labels[["date", "year", "player_id", "player_name",
                   "game_id", "home_park_id", "total_bases", "pa_count"]]


def audit(labels: pd.DataFrame) -> None:
    print("\n" + "="*70)
    print("TB LABEL AUDIT")
    print("="*70)
    print(f"Total player-game rows: {len(labels):,}")
    print(f"Unique players:          {labels['player_id'].nunique():,}")
    print(f"Unique games:            {labels['game_id'].nunique():,}")
    print(f"Date range:              {labels['date'].min()}  ..  {labels['date'].max()}")

    print("\nMean TB per player-game (min 1 PA):")
    for yr, g in labels.groupby("year"):
        print(f"  {yr}: mean={g['total_bases'].mean():.3f}  "
              f"median={g['total_bases'].median():.0f}  "
              f"p95={g['total_bases'].quantile(0.95):.0f}  "
              f"n={len(g):,}")

    # High-variance batters (min 40 games for stability).
    MIN_GAMES = 40
    by_p = labels.groupby("player_id").agg(
        games=("total_bases", "size"),
        mean_tb=("total_bases", "mean"),
        std_tb =("total_bases", "std"),
        player_name=("player_name", "last"),
    ).reset_index()
    qualifying = by_p[by_p["games"] >= MIN_GAMES]
    top5 = qualifying.sort_values("std_tb", ascending=False).head(5)
    print(f"\nTop 5 high-variance batters (min {MIN_GAMES} games, all years):")
    print(top5[["player_name", "games", "mean_tb", "std_tb"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int,
                    default=[2024, 2025, 2026])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading statcast for years: {args.years}")
    labels = build(args.years)

    labels.to_parquet(OUT_FILE, index=False)
    print(f"\nWrote {len(labels):,} rows -> {OUT_FILE}")

    audit(labels)


if __name__ == "__main__":
    main()
