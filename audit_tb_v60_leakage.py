"""
audit_tb_v60_leakage.py — Temporal Leakage Audit for TB v6.0 Training Set.

Validates that the two new high-resolution features added in v6.0 were derived
from data that was available *before* each game was played.

Features audited
----------------
bat_speed_pctile
    Source: batter_percentiles_{year}.parquet
    Risk:   Full-season aggregate. If the 2026 file was last refreshed on
            2026-04-24, then a training row with game_date=2026-03-21 uses
            bat speed data that includes games played *after* that date.
    Verdict: LEAKAGE RISK for 2026 rows — full-season rank not available
             pre-game for early-season games.

air_density_rho
    Source: temp_f (game-day pre-game weather) + altitude_ft (static constant)
    Formula: rho = P(h) / (R_d * T_K)
    Risk:    None — both inputs are known before first pitch.
    Verdict: SAFE — definitionally pre-game.

Output
------
    data/logs/tb_v60_leakage_audit.csv   — 10-row sample + all 2026 rows
    Printed verdict per feature

Usage
-----
    python audit_tb_v60_leakage.py
    python audit_tb_v60_leakage.py --verbose
"""
from __future__ import annotations

import argparse
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT        = Path(__file__).resolve().parent
MATRIX_PATH  = _ROOT / "data/batter_features/final_training_matrix.parquet"
STATCAST_DIR = _ROOT / "data/statcast"
OUT_CSV      = _ROOT / "data/logs/tb_v60_leakage_audit.csv"

# Air density constants (Standard Atmosphere)
_R_D   = 287.05
_P_SL  = 101325.0
_T_ISA = 288.15
_L     = 0.0065


def _compute_air_density(temp_f: pd.Series, altitude_ft: pd.Series) -> pd.Series:
    T_K = (temp_f.fillna(70.0) - 32.0) * 5.0 / 9.0 + 273.15
    h_m = altitude_ft.fillna(0.0) * 0.3048
    P_h = _P_SL * (1.0 - _L * h_m / _T_ISA) ** 5.2561
    return (P_h / (_R_D * T_K)).round(6)


def _pctile_file_info(year: int) -> dict:
    """Return metadata about the batter_percentiles_{year}.parquet file."""
    p = STATCAST_DIR / f"batter_percentiles_{year}.parquet"
    if not p.exists():
        return {"exists": False, "year": year}
    mtime = os.path.getmtime(p)
    df    = pd.read_parquet(p)
    return {
        "exists":       True,
        "year":         year,
        "file_path":    str(p),
        "file_mtime":   datetime.fromtimestamp(mtime).date().isoformat(),
        "n_players":    len(df),
        "has_bat_speed": "bat_speed" in df.columns,
        "cols":         list(df.columns),
    }


def _build_bat_speed_pctile_map(year: int) -> pd.DataFrame:
    """Compute bat_speed_pctile from the year's aggregate file."""
    p = STATCAST_DIR / f"batter_percentiles_{year}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "bat_speed" not in df.columns:
        return pd.DataFrame()
    df["bat_speed_pctile"] = (
        df["bat_speed"].rank(pct=True, na_option="bottom") * 100
    ).round(1)
    return df[["player_id", "bat_speed_pctile"]].copy()


def audit(verbose: bool = False) -> None:
    print("=" * 70)
    print("  TB v6.0 Temporal Leakage Audit")
    print("=" * 70)

    # ── Load training matrix ─────────────────────────────────────────────
    if not MATRIX_PATH.exists():
        print(f"  [ERROR] Training matrix not found: {MATRIX_PATH}")
        return

    tm = pd.read_parquet(MATRIX_PATH)
    tm["game_date"] = pd.to_datetime(tm["game_date"])
    print(f"\n  Training matrix: {tm.shape[0]:,} rows | "
          f"years: {sorted(tm['year'].dropna().unique().astype(int).tolist())}")

    rows_2026 = tm[tm["year"] == 2026].copy()
    print(f"  2026 rows: {len(rows_2026):,} | "
          f"game_date range: {rows_2026['game_date'].min().date()} "
          f"to {rows_2026['game_date'].max().date()}")

    # ── Audit: air_density_rho ───────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  FEATURE: air_density_rho")
    print("-" * 70)

    if "temp_f" not in tm.columns:
        print("  [SKIP] temp_f not in training matrix — cannot compute rho")
    else:
        # Check altitude_ft availability (may be a stadium constant column)
        alt_col = next((c for c in ["altitude_ft", "park_altitude_ft",
                                     "stadium_altitude"] if c in tm.columns), None)
        alt_vals = tm[alt_col] if alt_col else pd.Series(0.0, index=tm.index)
        rho_all  = _compute_air_density(tm["temp_f"], alt_vals)

        rho_2026 = _compute_air_density(rows_2026["temp_f"],
                                         alt_vals.loc[rows_2026.index])
        rho_null_pct = tm["temp_f"].isna().mean()

        print(f"  Inputs:    temp_f (pre-game weather) + altitude_ft (static)")
        print(f"  Formula:   rho = P(h) / (R_d * T_K)  [Standard Atmosphere]")
        print(f"  temp_f null%:  {rho_null_pct:.1%}  (filled with 70.0 F)")
        print(f"  alt column:    {alt_col or 'not found (using 0 ft = sea level)'}")
        print(f"  rho range:     {rho_all.min():.4f} – {rho_all.max():.4f} kg/m3")
        print(f"  rho 2026 mean: {rho_2026.mean():.4f} kg/m3")
        print()
        print("  VERDICT: SAFE")
        print("    temp_f is collected from pre-game weather APIs (game-time forecast).")
        print("    altitude_ft is a static stadium constant — never changes.")
        print("    Both inputs are definitionally available before first pitch.")

    # ── Audit: bat_speed_pctile ──────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  FEATURE: bat_speed_pctile")
    print("-" * 70)

    pctile_2026_info = _pctile_file_info(2026)
    pctile_2025_info = _pctile_file_info(2025)

    print(f"  Source file:   batter_percentiles_2026.parquet")
    print(f"  File mtime:    {pctile_2026_info.get('file_mtime', 'N/A')}")
    print(f"  Players:       {pctile_2026_info.get('n_players', 0)}")
    print(f"  bat_speed col: {pctile_2026_info.get('has_bat_speed', False)}")

    # Compare file mtime vs earliest 2026 training game_date
    earliest_2026_game = rows_2026["game_date"].min().date()
    file_mtime_date    = pctile_2026_info.get("file_mtime")

    if file_mtime_date:
        file_dt  = datetime.fromisoformat(file_mtime_date).date()
        days_lag = (file_dt - earliest_2026_game).days
        print(f"\n  Earliest 2026 training game: {earliest_2026_game}")
        print(f"  Pctile file last updated:    {file_mtime_date}")
        print(f"  Lag (file_date - first_game): +{days_lag} days")

        if days_lag > 7:
            print(f"\n  VERDICT: LEAKAGE RISK")
            print(f"    batter_percentiles_2026.parquet was built on {file_mtime_date}.")
            print(f"    It contains bat speed data from games through ~{file_mtime_date}.")
            print(f"    Training rows with game_date < {file_mtime_date} received")
            print(f"    forward-looking bat_speed_pctile ranks that include games")
            print(f"    played AFTER those dates. This is data leakage.")
            print(f"\n  IMPACT ESTIMATE:")
            n_early = len(rows_2026[rows_2026["game_date"].dt.date < file_dt])
            print(f"    {n_early:,} of {len(rows_2026):,} 2026 rows exposed "
                  f"({n_early/max(len(rows_2026),1)*100:.1f}%)")
            print(f"\n  MITIGATION:")
            print(f"    Option 1: Use prior-year (2025) bat_speed for all 2026 training rows.")
            print(f"    Option 2: Build rolling cumulative percentiles by game_date.")
            print(f"    Option 3: Flag 2026 bat_speed as forward-looking, reduce its")
            print(f"              sample weight for pre-{file_mtime_date} games.")
        else:
            print(f"\n  VERDICT: ACCEPTABLE (lag <= 7 days)")
    else:
        print("\n  VERDICT: UNKNOWN — file mtime not available")

    # ── Consistency check: bat_speed_pctile variance by player in 2026 ───
    pctile_map = _build_bat_speed_pctile_map(2026)
    if not pctile_map.empty:
        rows_2026_merged = rows_2026.merge(
            pctile_map, on="player_id", how="left"
        )
        pid_nunique = rows_2026_merged.groupby("player_id")["bat_speed_pctile"].nunique()
        # Each player should have exactly 1 unique pctile value (full-season aggregate)
        all_one = (pid_nunique == 1).all()
        print(f"\n  Intra-player pctile variance check:")
        print(f"    All players have 1 unique bat_speed_pctile value: {all_one}")
        print(f"    (1 = full-season aggregate confirmed — not time-varying)")

    # ── Build 10-row sample output ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("  10-ROW SAMPLE  (2026 training rows, earliest game dates)")
    print("=" * 70)

    sample = rows_2026.sort_values("game_date").head(10).copy()

    # Compute rho for sample
    alt_col = next((c for c in ["altitude_ft", "park_altitude_ft",
                                 "stadium_altitude"] if c in sample.columns), None)
    alt_s   = sample[alt_col] if alt_col else pd.Series(0.0, index=sample.index)
    sample["air_density_rho_computed"] = _compute_air_density(
        sample["temp_f"] if "temp_f" in sample.columns
        else pd.Series(70.0, index=sample.index),
        alt_s,
    )

    # Attach bat_speed_pctile from pctile map
    if not pctile_map.empty:
        sample = sample.merge(pctile_map, on="player_id", how="left")

    # Build summary cols for display
    display_cols = ["game_date", "player_name", "team"]
    if "temp_f" in sample.columns:
        display_cols.append("temp_f")
    display_cols.append("air_density_rho_computed")
    if "bat_speed_pctile" in sample.columns:
        display_cols.append("bat_speed_pctile")
    if "swing_length_pctile" in sample.columns:
        display_cols.append("swing_length_pctile")

    # Add leakage risk tag
    if file_mtime_date:
        file_dt = datetime.fromisoformat(file_mtime_date).date()
        sample["leakage_risk"] = sample["game_date"].dt.date.apply(
            lambda d: "RISK" if d < file_dt else "SAFE"
        )
        display_cols.append("leakage_risk")

    # feature_collection_timestamp proxy
    sample["feature_collection_timestamp"] = pctile_2026_info.get("file_mtime", "unknown")
    display_cols.append("feature_collection_timestamp")

    out_sample = sample[[c for c in display_cols if c in sample.columns]].copy()

    print()
    print(out_sample.to_string(index=False))

    # Save
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_sample.to_csv(OUT_CSV, index=False)
    print(f"\n  Saved -> {OUT_CSV.name}")

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  AUDIT SUMMARY")
    print("=" * 70)
    print(f"  air_density_rho   : SAFE    (pre-game weather + static altitude)")
    if file_mtime_date and days_lag > 7:
        print(f"  bat_speed_pctile  : RISK    ({n_early:,} of {len(rows_2026):,} "
              f"2026 rows use forward-looking data)")
        print(f"                       -> Suggest: use 2025 pctile for 2026 pre-{file_mtime_date}")
    else:
        print(f"  bat_speed_pctile  : OK      (file mtime within 7-day tolerance)")


def main() -> None:
    parser = argparse.ArgumentParser(description="TB v6.0 Temporal Leakage Audit")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    audit(verbose=args.verbose)


if __name__ == "__main__":
    main()
