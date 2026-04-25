"""
build_sp_k_labels.py — Append new season rows to sp_k_labels.parquet.

Sources:
  data/statcast/actuals_{year}.parquet  → game-level SP K/IP outcomes
  data/statcast/pitcher_profiles_{year}.parquet  → p_throws lookup

Logic:
  - Pivot each game into two SP rows (home, away)
  - pa_total approximated as round(sp_ip * 3.5), clipped to [10, 37]
  - Skips rows already in the existing file (dedup on game_pk + pitcher)
  - Appends to data/batter_features/sp_k_labels.parquet

Usage:
    python build_sp_k_labels.py [--year 2026] [--replace]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE       = Path(__file__).resolve().parent
OUT_PATH   = BASE / "data/batter_features/sp_k_labels.parquet"
STATCAST   = BASE / "data/statcast"


def load_p_throws(year: int) -> dict[int, str]:
    """Build pitcher_id -> p_throws lookup from pitcher_profiles."""
    profile_path = STATCAST / f"pitcher_profiles_{year}.parquet"
    if not profile_path.exists():
        print(f"  [WARN] {profile_path.name} not found — p_throws will be inferred from handedness cache")
        return {}
    df = pd.read_parquet(profile_path, columns=["pitcher", "p_throws"])
    df = df.dropna(subset=["pitcher", "p_throws"])
    return dict(zip(df["pitcher"].astype(int), df["p_throws"].astype(str)))


def build_year_rows(year: int) -> pd.DataFrame:
    """Build sp-grain rows from actuals_{year}.parquet."""
    actuals_path = STATCAST / f"actuals_{year}.parquet"
    if not actuals_path.exists():
        raise FileNotFoundError(f"actuals_{year}.parquet not found at {actuals_path}")

    act = pd.read_parquet(actuals_path)
    act["game_date"] = pd.to_datetime(act["game_date"]).dt.strftime("%Y-%m-%d")

    # Only games with SP data
    act = act.dropna(subset=["home_sp_id", "home_sp_k"]).copy()
    print(f"  actuals_{year}: {len(act)} games with SP data")

    p_throws = load_p_throws(year)

    rows = []
    for side, sp_id_col, k_col, ip_col, team_col in [
        ("home", "home_sp_id", "home_sp_k", "home_sp_ip", "home_team"),
        ("away", "away_sp_id", "away_sp_k", "away_sp_ip", "home_team"),
    ]:
        sub = act[["game_pk", "game_date", "home_team",
                   sp_id_col, k_col, ip_col]].copy()
        sub = sub.dropna(subset=[sp_id_col, k_col])
        sub = sub.rename(columns={
            sp_id_col: "pitcher",
            k_col:     "k_total",
            ip_col:    "sp_ip",
        })
        sub["pitcher"]   = sub["pitcher"].astype(int)
        sub["k_total"]   = sub["k_total"].astype(int)
        sub["pa_total"]  = (sub["sp_ip"].fillna(5.5) * 3.5).round().clip(10, 37).astype(int)
        sub["p_throws"]  = sub["pitcher"].map(p_throws).fillna("R")  # R default if unknown
        sub["is_sp"]     = True
        rows.append(sub)

    out = pd.concat(rows, ignore_index=True)
    out = out[["game_pk", "game_date", "pitcher", "p_throws",
               "home_team", "k_total", "pa_total", "is_sp"]]
    out["game_pk"] = out["game_pk"].astype("Int64")
    out["pitcher"] = out["pitcher"].astype("Int64")
    return out


def main():
    parser = argparse.ArgumentParser(description="Append SP K labels for a season")
    parser.add_argument("--year",    type=int, default=2026)
    parser.add_argument("--replace", action="store_true",
                        help="Replace existing rows for this year instead of skipping")
    args = parser.parse_args()

    print(f"Building sp_k_labels for {args.year} ...")
    new_rows = build_year_rows(args.year)
    print(f"  Built {len(new_rows)} SP-game rows for {args.year}")

    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        print(f"  Existing file: {len(existing)} rows")

        if args.replace:
            existing = existing[
                pd.to_datetime(existing["game_date"]).dt.year != args.year
            ].copy()
            print(f"  Dropped existing {args.year} rows -> {len(existing)} remain")

        # Dedup: skip rows already present
        existing_keys = set(zip(existing["game_pk"].astype(int),
                                existing["pitcher"].astype(int)))
        new_rows = new_rows[
            ~new_rows.apply(
                lambda r: (int(r["game_pk"]), int(r["pitcher"])) in existing_keys,
                axis=1
            )
        ]
        print(f"  New rows after dedup: {len(new_rows)}")

        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined = combined.sort_values(["game_date", "game_pk", "pitcher"]).reset_index(drop=True)
    combined.to_parquet(OUT_PATH, index=False)

    year_counts = pd.to_datetime(combined["game_date"]).dt.year.value_counts().sort_index()
    print(f"\n  Saved {len(combined)} total rows -> {OUT_PATH.relative_to(BASE)}")
    print("  Year breakdown:")
    for yr, n in year_counts.items():
        print(f"    {yr}: {n}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
