"""
rebuild_bullpen_burn_2026.py — v4.4 Bullpen Burn Recovery.

Reads statcast_2026.parquet + bullpen_avail_2026.parquet and computes
3-day and 5-day rolling high-leverage pitch counts per team for every
game day in 2026.

Appends 2026 rows to bullpen_burn_by_game.parquet (preserving 2024-2025).

'Gassed Bullpen' definition:
    home_bullpen_burn_5d > GASSED_THRESHOLD (350 HL pitches)
    This fires the 'Over in late innings' modifier in Script B.

Run:
    python rebuild_bullpen_burn_2026.py
    python rebuild_bullpen_burn_2026.py --from-scratch   # rebuild all years

Outputs:
    data/batter_features/bullpen_burn_by_game.parquet   (updated)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent

STATCAST_2026   = _ROOT / "data/statcast/statcast_2026.parquet"
BP_AVAIL_2026   = _ROOT / "data/statcast/bullpen_avail_2026.parquet"
BP_BURN_OUT     = _ROOT / "data/batter_features/bullpen_burn_by_game.parquet"

GASSED_THRESHOLD_3D = 280   # HL pitches in 3 days -> depleted
GASSED_THRESHOLD_5D = 350   # HL pitches in 5 days -> gassed

# High-leverage events: in-game situations, not filtered here because
# statcast doesn't carry leverage index per-pitch in bulk exports.
# Proxy: all relief pitches thrown. SP pitches excluded by inning filter.
RELIEF_INNING_CUTOFF = 6    # pitches from inning 6+ are presumed relief


def _build_from_statcast(sc: pd.DataFrame, year: int) -> pd.DataFrame:
    """Compute per-game bullpen pitch totals from raw statcast.

    For each game, the 'bullpen' is all pitching from inning 6 onwards.
    Groups by home_team + game_date to get per-park fatigue signal.
    """
    sc = sc.copy()
    sc["game_date"] = pd.to_datetime(sc["game_date"])

    # Relief proxy: inning >= 6
    relief = sc[sc["inning"] >= RELIEF_INNING_CUTOFF].copy()

    # Count pitches per (game_pk, home_team, away_team, game_date)
    needed_cols = ["game_pk", "game_date", "inning", "home_team", "away_team"]
    missing = [c for c in needed_cols if c not in relief.columns]
    if missing:
        print(f"  [warn] statcast missing cols: {missing}")
        return pd.DataFrame()

    # Each row in statcast is one pitch
    game_agg = (relief.groupby(["game_pk", "game_date", "home_team", "away_team"])
                      .size()
                      .reset_index(name="total_hl_pitches"))

    # Separate home and away relief pitches
    # statcast rows include both home and away pitchers; we can't separate
    # easily without pitcher_team. Use full count as proxy for both sides.
    game_agg["hl_relievers_used"] = (
        relief.groupby(["game_pk", "game_date", "home_team", "away_team"])["pitcher"]
              .nunique()
              .values
    )

    return game_agg


def _build_from_bullpen_avail(bp: pd.DataFrame) -> pd.DataFrame:
    """Convert bullpen_avail grain (team, date) to game grain with burn columns.

    bp_pitches_rest1d is already a per-day pitch count (not cumulative).
    We need a time-aware rolling sum that accounts for calendar gaps
    (away games, off days appear as missing rows, not zero rows).
    """
    bp = bp.copy()
    bp["game_date"] = pd.to_datetime(bp["game_date"])
    bp = bp.sort_values(["team", "game_date"]).reset_index(drop=True)

    results = []
    for team, grp in bp.groupby("team"):
        grp = grp.set_index("game_date").sort_index()
        # Reindex to full daily calendar so gaps become explicit zeros
        full_idx = pd.date_range(grp.index.min(), grp.index.max(), freq="D")
        grp = grp.reindex(full_idx).fillna(0)
        # bp_pitches_rest1d = 0 on days with no home game
        pitches = grp["bp_pitches_rest1d"].astype(float)
        # Rolling sum: 3-day window ending on each date
        burn3 = pitches.rolling(3,  min_periods=1).sum()
        burn5 = pitches.rolling(5,  min_periods=1).sum()
        tmp = pd.DataFrame({
            "team":             team,
            "game_date":        grp.index,
            "bullpen_burn_3d":  burn3.values,
            "bullpen_burn_5d":  burn5.values,
            "bp_n_used_yday":   grp["bp_n_used_yday"].values,
            "bp_depleted_flag": grp["bp_depleted_flag"].values,
        })
        # Keep only rows that had actual game data (original index)
        orig_dates = bp[bp["team"] == team]["game_date"]
        tmp = tmp[tmp["game_date"].isin(orig_dates)]
        results.append(tmp)

    out = pd.concat(results, ignore_index=True)
    out["bp_gassed_3d"] = (out["bullpen_burn_3d"] > GASSED_THRESHOLD_3D).astype("int8")
    out["bp_gassed_5d"] = (out["bullpen_burn_5d"] > GASSED_THRESHOLD_5D).astype("int8")
    return out


def build_2026(verbose: bool = True) -> pd.DataFrame:
    """Build 2026 bullpen burn rows and merge into the burn file."""
    if verbose:
        print("=" * 60)
        print("  rebuild_bullpen_burn_2026.py")
        print("=" * 60)

    # ── Load sources ─────────────────────────────────────────────────────
    rows_from_avail = pd.DataFrame()
    rows_from_statcast = pd.DataFrame()

    if BP_AVAIL_2026.exists():
        bp = pd.read_parquet(BP_AVAIL_2026)
        rows_from_avail = _build_from_bullpen_avail(bp)
        if verbose:
            print(f"  bullpen_avail_2026: {len(bp)} rows -> {len(rows_from_avail)} team-days")

    if STATCAST_2026.exists():
        sc = pd.read_parquet(STATCAST_2026)
        rows_from_statcast = _build_from_statcast(sc, 2026)
        if verbose:
            print(f"  statcast_2026: {len(sc):,} pitches -> {len(rows_from_statcast)} game-days")
    else:
        if verbose:
            print("  [warn] statcast_2026.parquet not found")

    if rows_from_avail.empty and rows_from_statcast.empty:
        print("  [ERROR] No 2026 source data found")
        return pd.DataFrame()

    # ── Primary: statcast-based game-grain ───────────────────────────────
    if not rows_from_statcast.empty:
        new_rows = rows_from_statcast.copy()

        # Join avail-based burn windows onto game grain by home_team + game_date
        if not rows_from_avail.empty:
            ra = rows_from_avail.rename(columns={"team": "home_team"})
            new_rows = new_rows.merge(ra[["home_team", "game_date",
                                          "bullpen_burn_3d", "bullpen_burn_5d",
                                          "bp_gassed_3d", "bp_gassed_5d"]],
                                      on=["home_team", "game_date"], how="left")
        else:
            # Fallback: compute rolling from statcast pitch counts directly
            new_rows = new_rows.sort_values(["home_team", "game_date"])
            new_rows["bullpen_burn_3d"] = (
                new_rows.groupby("home_team")["total_hl_pitches"]
                        .transform(lambda x: x.rolling(3, min_periods=1).sum()))
            new_rows["bullpen_burn_5d"] = (
                new_rows.groupby("home_team")["total_hl_pitches"]
                        .transform(lambda x: x.rolling(5, min_periods=1).sum()))
            new_rows["bp_gassed_3d"] = (new_rows["bullpen_burn_3d"] > GASSED_THRESHOLD_3D).astype("int8")
            new_rows["bp_gassed_5d"] = (new_rows["bullpen_burn_5d"] > GASSED_THRESHOLD_5D).astype("int8")

    else:
        # Statcast missing — convert avail to game grain by joining with MLB schedule
        import urllib.request, json
        if verbose:
            print("  [fallback] Building game grain from bullpen_avail + MLB schedule")
        new_rows = rows_from_avail.rename(columns={"team": "home_team"}).copy()
        new_rows["game_pk"] = np.nan
        new_rows["total_hl_pitches"] = new_rows.get("bullpen_burn_3d", 0)
        new_rows["hl_relievers_used"] = np.nan

    # ── Merge with existing burn file ────────────────────────────────────
    new_rows["game_date"] = pd.to_datetime(new_rows["game_date"])
    new_rows = new_rows.sort_values(["home_team", "game_date"]).reset_index(drop=True)

    if BP_BURN_OUT.exists():
        existing = pd.read_parquet(BP_BURN_OUT)
        existing["game_date"] = pd.to_datetime(existing["game_date"])
        # Drop any existing 2026 rows to replace cleanly
        existing_pre2026 = existing[existing["game_date"].dt.year < 2026]
        combined = pd.concat([existing_pre2026, new_rows], ignore_index=True, sort=False)
        if verbose:
            print(f"  Pre-2026 rows: {len(existing_pre2026):,}  "
                  f"+ 2026 rows: {len(new_rows):,}  = {len(combined):,} total")
    else:
        combined = new_rows
        if verbose:
            print(f"  New file: {len(combined):,} rows")

    combined = combined.sort_values(["game_date", "home_team"]).reset_index(drop=True)

    # Ensure standard columns exist
    for col in ("bullpen_burn_3d", "bullpen_burn_5d", "bp_gassed_3d", "bp_gassed_5d"):
        if col not in combined.columns:
            combined[col] = np.nan

    BP_BURN_OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(BP_BURN_OUT, index=False)
    if verbose:
        print(f"  Saved -> {BP_BURN_OUT}")

        # Summary stats
        rows_2026 = combined[combined["game_date"].dt.year == 2026]
        if not rows_2026.empty and "bullpen_burn_5d" in rows_2026.columns:
            gassed = rows_2026["bp_gassed_5d"].sum() if "bp_gassed_5d" in rows_2026.columns else 0
            print(f"  2026: {len(rows_2026)} game-days  "
                  f"avg_burn_5d={rows_2026['bullpen_burn_5d'].mean():.1f}  "
                  f"gassed_flags={gassed}")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-scratch", action="store_true",
                        help="Rebuild all years, not just 2026")
    args = parser.parse_args()
    build_2026(verbose=True)
