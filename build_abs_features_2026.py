"""
build_abs_features_2026.py
==========================
ABS regime fix: depreciate umpire/framing features, inject pitch-level
unbiased metrics (Edge%, Zone%, Whiff%, called-K vulnerability).

Background
----------
Starting in 2026 MLB adopted the Automated Ball-Strike (ABS) challenge
system, which mechanically eliminates umpire discretion on called ball/
strike decisions.  Two signal families that fed the F5 and NRFI models
are now dead:
  1. Umpire tendency features  (ump_* columns in feature_matrix)
  2. Catcher framing runs      (any column containing "framing" or
                                "catcher_frame")

This script:
  Part 1 – Zeros out those columns for every game_date >= 2026-03-01
            and writes abs_mask_report.txt.
  Part 2 – Engineers replacement features from raw pitch-level Statcast
            data (edge%, zone%, whiff%, called-K vulnerability) both
            full-game and first-inning.
  Part 3 – Assembles the features at game level (home_sp_* / away_sp_*
            plus differential columns) and saves abs_features_2026.parquet.
  Part 4 – Temporal-leakage guard: new features are ONLY populated for
            dates that appear in statcast_2026.parquet (2026-03-21 +).
            For 2023-2025 rows these columns remain NaN — do NOT back-
            fill with 2025-computed values under the 2026 methodology,
            because (a) the 2025 ABS-regime features would be counterfactual
            and (b) filling them would let the CV folds "see" a signal
            that did not exist during training.

Run:  python build_abs_features_2026.py

Outputs
-------
  data/statcast/abs_features_2026.parquet  – new game-level features
  data/statcast/abs_mask_report.txt        – depreciation summary
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATCAST_2026 = os.path.join(BASE_DIR, "data", "statcast", "statcast_2026.parquet")
FEATURE_MATRIX = os.path.join(BASE_DIR, "feature_matrix.parquet")
CATCHER_FRAMING_2026 = os.path.join(BASE_DIR, "data", "statcast", "catcher_framing_2026.parquet")
OUT_FEATURES = os.path.join(BASE_DIR, "data", "statcast", "abs_features_2026.parquet")
OUT_MASK_REPORT = os.path.join(BASE_DIR, "data", "statcast", "abs_mask_report.txt")

# ABS goes live 2026 spring training
ABS_CUTOFF = pd.Timestamp("2026-03-01")

# Umpire feature columns to zero out (explicit list)
UMP_COLS = [
    "ump_called_strike_above_avg",
    "ump_k_above_avg",
    "ump_bb_above_avg",
    "ump_rpg_above_avg",
    "ump_command_edge_home",
    "ump_command_edge_away",
    "ump_command_net_edge",
]

# Statcast zone definitions
EDGE_ZONES = {11, 12, 13, 14}   # shadow / border zones
IN_ZONES   = {1, 2, 3, 4, 5, 6, 7, 8, 9}  # inside strike zone


# ===========================================================================
# Part 1 — Hard Depreciation (ABS mask)
# ===========================================================================

def apply_abs_mask(feature_matrix_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Zero out umpire and framing columns for rows where game_date >= ABS_CUTOFF.

    Parameters
    ----------
    feature_matrix_df : pd.DataFrame
        The full feature matrix (any vintage — 2023 through present).

    Returns
    -------
    masked_df : pd.DataFrame
        Copy of input with deprecated columns set to 0.0 for ABS-era rows.
    report : dict
        { column_name: n_rows_zeroed } for every column that was touched.
    """
    df = feature_matrix_df.copy()

    # Ensure game_date is datetime
    df["game_date"] = pd.to_datetime(df["game_date"])
    abs_mask = df["game_date"] >= ABS_CUTOFF

    # Build complete list of columns to zero:
    # 1. Explicit ump columns that exist in the dataframe
    # 2. Any column whose name contains "framing" or "catcher_frame"
    framing_pattern_cols = [
        c for c in df.columns
        if "framing" in c.lower() or "catcher_frame" in c.lower()
    ]
    all_target_cols = list(dict.fromkeys(UMP_COLS + framing_pattern_cols))  # dedup, preserve order
    existing_cols   = [c for c in all_target_cols if c in df.columns]

    report = {}
    for col in existing_cols:
        n_affected = int(abs_mask.sum())
        df.loc[abs_mask, col] = 0.0
        report[col] = n_affected

    return df, report


# ===========================================================================
# Part 2 — New Features from Raw Pitch Data
# ===========================================================================

def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Division that returns NaN instead of inf/ZeroDivision."""
    return np.where(denominator > 0, numerator / denominator, np.nan)


def compute_pitcher_game_features(sc: pd.DataFrame,
                                   inning_filter: int | None = None,
                                   inning_max: int | None = None) -> pd.DataFrame:
    """
    Compute per-pitcher-per-game pitch features from raw Statcast data.

    Parameters
    ----------
    sc : pd.DataFrame
        Full statcast dataframe (statcast_2026.parquet).
    inning_filter : int or None
        If an integer, restrict computation to that inning (or inning range min) only.
        Pass None for full-game stats.
    inning_max : int or None
        If set along with inning_filter, filter to innings inning_filter..inning_max
        (inclusive). E.g. inning_filter=1, inning_max=5 for F5 window.

    Returns
    -------
    pd.DataFrame indexed by (pitcher, game_pk, game_date, home_team, away_team)
    with columns: edge_pct, zone_pct, called_k_rate, swinging_k_rate,
                  abs_vulnerability, whiff_pct
    """
    df = sc.copy()

    if inning_filter is not None:
        if inning_max is not None:
            df = df[(df["inning"] >= inning_filter) & (df["inning"] <= inning_max)]
        else:
            df = df[df["inning"] == inning_filter]
        if df.empty:
            return pd.DataFrame()

    # -----------------------------------------------------------------------
    # Flag each pitch type
    # -----------------------------------------------------------------------
    zone_vals = df["zone"].fillna(-1).astype(int)
    df["is_edge"]  = zone_vals.isin(EDGE_ZONES).astype(int)
    df["is_in_zone"] = zone_vals.isin(IN_ZONES).astype(int)

    desc = df["description"].fillna("").astype(str)

    # Whiff: swinging strikes over total swings
    # A "swing" = swinging_strike, swinging_strike_blocked, foul, foul_tip,
    #             foul_bunt, bunt_foul_tip, hit_into_play, missed_bunt
    df["is_swinging_strike"] = desc.str.contains(
        r"swinging_strike|missed_bunt|swinging_pitchout", regex=True
    ).astype(int)
    df["is_swing"] = desc.str.contains(
        r"swinging_strike|foul|hit_into_play|missed_bunt", regex=True
    ).astype(int)

    # Called K vulnerability: strikes==2 rows (the pitch that ends the PA)
    # Note: the 'strikes' column reflects the count BEFORE the pitch.
    two_strike = df["strikes"] == 2
    df["is_called_k"]   = (two_strike & desc.str.contains("called_strike", regex=False)).astype(int)
    df["is_swinging_k"] = (two_strike & desc.str.contains(
        r"swinging_strike|missed_bunt", regex=True
    )).astype(int)

    # -----------------------------------------------------------------------
    # Group by pitcher × game
    # -----------------------------------------------------------------------
    grp_cols = ["pitcher", "player_name", "game_pk", "game_date", "home_team", "away_team"]
    g = df.groupby(grp_cols)

    agg = g.agg(
        total_pitches    = ("is_edge",          "count"),
        edge_count       = ("is_edge",           "sum"),
        zone_count       = ("is_in_zone",        "sum"),
        swing_count      = ("is_swing",          "sum"),
        swinging_k_count = ("is_swinging_k",     "sum"),
        sw_strike_count  = ("is_swinging_strike", "sum"),
        called_k_count   = ("is_called_k",       "sum"),
    ).reset_index()

    # -----------------------------------------------------------------------
    # Derived rates
    # -----------------------------------------------------------------------
    agg["edge_pct"]       = _safe_div(agg["edge_count"],       agg["total_pitches"])
    agg["zone_pct"]       = _safe_div(agg["zone_count"],       agg["total_pitches"])
    agg["whiff_pct"]      = _safe_div(agg["sw_strike_count"],  agg["swing_count"])

    total_ks = agg["called_k_count"] + agg["swinging_k_count"]
    agg["called_k_rate"]   = _safe_div(agg["called_k_count"],   total_ks)
    agg["swinging_k_rate"] = _safe_div(agg["swinging_k_count"], total_ks)
    # abs_vulnerability: share of strikeouts that were called-K3 (now auto-corrected by ABS)
    agg["abs_vulnerability"] = agg["called_k_rate"]

    keep = [
        "pitcher", "player_name", "game_pk", "game_date", "home_team", "away_team",
        "total_pitches", "edge_pct", "zone_pct", "whiff_pct",
        "called_k_rate", "swinging_k_rate", "abs_vulnerability",
    ]
    return agg[keep]


# ===========================================================================
# Part 3 — Assemble Game-Level Feature File
# ===========================================================================

def identify_starters(sc: pd.DataFrame) -> pd.DataFrame:
    """
    Return a dataframe of (game_pk, home_team, away_team, game_date,
    home_sp_name, away_sp_name) by finding the first pitcher in inning 1
    for each side.

    Statcast convention:
      inning_topbot == 'Top' to away team is batting to home SP is pitching
      inning_topbot == 'Bot' to home team is batting to away SP is pitching
    """
    inn1 = sc[sc["inning"] == 1].copy()

    # Sort so first appearance = first pitcher of the inning
    inn1 = inn1.sort_values(["game_pk", "inning_topbot", "at_bat_number", "pitch_number"])

    home_sp = (
        inn1[inn1["inning_topbot"] == "Top"]
        .groupby(["game_pk", "home_team", "away_team", "game_date"])
        .agg(home_sp=("player_name", "first"), home_sp_id=("pitcher", "first"))
        .reset_index()
    )
    away_sp = (
        inn1[inn1["inning_topbot"] == "Bot"]
        .groupby(["game_pk", "home_team", "away_team", "game_date"])
        .agg(away_sp=("player_name", "first"), away_sp_id=("pitcher", "first"))
        .reset_index()
    )

    starters = home_sp.merge(
        away_sp[["game_pk", "away_sp", "away_sp_id"]],
        on="game_pk", how="outer"
    )
    return starters


def pivot_to_game_level(
    starters: pd.DataFrame,
    full_game_feats: pd.DataFrame,
    fi_feats: pd.DataFrame,
    f5_feats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Join pitcher-level stats to each game's home/away SP and compute
    differential columns.

    Parameters
    ----------
    starters        : output of identify_starters()
    full_game_feats : output of compute_pitcher_game_features(sc)
    fi_feats        : output of compute_pitcher_game_features(sc, inning_filter=1)
    f5_feats        : output of compute_pitcher_game_features(sc, inning_filter=1, inning_max=5)

    Returns
    -------
    Wide game-level dataframe.
    """
    # Rename full-game feature columns
    base_cols = ["edge_pct", "zone_pct", "whiff_pct", "abs_vulnerability",
                 "called_k_rate", "swinging_k_rate", "total_pitches"]

    fg = full_game_feats[["pitcher", "game_pk"] + base_cols].copy()
    fi = fi_feats[["pitcher", "game_pk"] + base_cols].copy()
    fi = fi.rename(columns={c: f"fi_{c}" for c in base_cols})

    # Combine full-game, first-inning, and F5-window stats
    pitcher_stats = fg.merge(fi, on=["pitcher", "game_pk"], how="left")
    if f5_feats is not None and not f5_feats.empty:
        f5 = f5_feats[["pitcher", "game_pk"] + base_cols].copy()
        f5 = f5.rename(columns={c: f"f5_{c}" for c in base_cols})
        pitcher_stats = pitcher_stats.merge(f5, on=["pitcher", "game_pk"], how="left")

    def _attach_sp(df: pd.DataFrame, sp_id_col: str, prefix: str) -> pd.DataFrame:
        renamed = pitcher_stats.rename(
            columns={c: f"{prefix}_{c}" for c in pitcher_stats.columns if c not in ["pitcher", "game_pk"]}
        )
        renamed = renamed.rename(columns={"pitcher": sp_id_col})
        return df.merge(renamed, on=["game_pk", sp_id_col], how="left")

    game = starters.copy()
    game = _attach_sp(game, "home_sp_id", "home_sp")
    game = _attach_sp(game, "away_sp_id", "away_sp")

    # -----------------------------------------------------------------------
    # Differential features  (home − away, positive = home SP more exposed)
    # -----------------------------------------------------------------------
    diff_map = {
        "sp_edge_pct_diff":              ("home_sp_edge_pct",              "away_sp_edge_pct"),
        "sp_whiff_pct_diff":             ("home_sp_whiff_pct",             "away_sp_whiff_pct"),
        "sp_abs_vulnerability_diff":     ("home_sp_abs_vulnerability",     "away_sp_abs_vulnerability"),
        "sp_zone_pct_diff":              ("home_sp_zone_pct",              "away_sp_zone_pct"),
        # F5-window (innings 1-5) differentials
        "sp_f5_whiff_pct_diff":          ("home_sp_f5_whiff_pct",          "away_sp_f5_whiff_pct"),
        "sp_f5_abs_vulnerability_diff":  ("home_sp_f5_abs_vulnerability",  "away_sp_f5_abs_vulnerability"),
    }
    for new_col, (hcol, acol) in diff_map.items():
        if hcol in game.columns and acol in game.columns:
            game[new_col] = game[hcol] - game[acol]

    # -----------------------------------------------------------------------
    # Clean up: keep only the columns we want in the output
    # -----------------------------------------------------------------------
    id_cols = ["game_date", "home_team", "away_team", "home_sp", "away_sp",
               "home_sp_id", "away_sp_id", "game_pk"]

    feature_cols = (
        [c for c in game.columns if c.startswith("home_sp_") and c not in ["home_sp", "home_sp_id"]] +
        [c for c in game.columns if c.startswith("away_sp_") and c not in ["away_sp", "away_sp_id"]] +
        list(diff_map.keys())
    )
    # Deduplicate while preserving order
    seen = set()
    ordered_feat_cols = []
    for c in feature_cols:
        if c not in seen:
            seen.add(c)
            ordered_feat_cols.append(c)

    final_cols = [c for c in id_cols if c in game.columns] + \
                 [c for c in ordered_feat_cols if c in game.columns]

    return game[final_cols].sort_values(["game_date", "home_team"]).reset_index(drop=True)


# ===========================================================================
# Part 4 — Temporal Leakage Guard
# ===========================================================================
# CONSTRAINT (do not remove this comment block):
#
#   abs_features_2026.parquet covers ONLY dates present in statcast_2026.parquet
#   (currently 2026-03-21 through the most recent pull date).
#
#   When merging abs_features_2026 into the full feature matrix for model
#   training, rows from 2023-2025 will have NaN for all abs_* columns.
#   This is INTENTIONAL.  Do NOT:
#     - Back-fill NaN rows with 2025-era computed values under 2026 methodology
#     - Interpolate / forward-fill across the 2025to2026 boundary
#     - Use these features in CV folds that include 2023-2025 data without
#       explicitly excluding them or treating them as "new feature, post-cutoff"
#
#   The correct CV strategy is either:
#     (a) Walk-forward splits where ABS features are available only for the
#         live 2026 fold, or
#     (b) Drop abs_* columns entirely for any fold that uses pre-2026 labels.
#
#   Rationale: the 2025 umpire framing signal was real and meaningful.
#   Replacing it retroactively with ABS-era features on 2025 games would
#   produce a distribution shift that inflates held-out performance and leads
#   to overconfident live bets in 2026.

def validate_feature_coverage(abs_features: pd.DataFrame) -> str:
    """
    Produce a text report of feature coverage by date range.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("ABS FEATURES 2026 — COVERAGE VALIDATION REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append("")

    # Date range
    lines.append(f"Date range:   {abs_features['game_date'].min().date()} "
                 f"to {abs_features['game_date'].max().date()}")
    lines.append(f"Total games:  {len(abs_features)}")
    lines.append("")

    # Coverage by month
    abs_features["_ym"] = abs_features["game_date"].dt.to_period("M")
    monthly = abs_features.groupby("_ym").size().rename("games")
    lines.append("Games per month:")
    for period, cnt in monthly.items():
        lines.append(f"  {period}: {cnt} games")
    lines.append("")

    # Feature non-null rates
    feat_cols = [c for c in abs_features.columns
                 if c not in ["game_date", "home_team", "away_team", "home_sp",
                               "away_sp", "home_sp_id", "away_sp_id", "game_pk", "_ym"]]
    lines.append("Feature coverage (% non-null):")
    for col in feat_cols:
        pct = abs_features[col].notna().mean() * 100
        lines.append(f"  {col:<45} {pct:5.1f}%")

    abs_features.drop(columns=["_ym"], inplace=True)
    return "\n".join(lines)


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 60)
    print("build_abs_features_2026.py")
    print(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading statcast_2026.parquet …", flush=True)
    sc = pd.read_parquet(STATCAST_2026)
    sc["game_date"] = pd.to_datetime(sc["game_date"])
    print(f"      {len(sc):,} pitches | "
          f"{sc['game_date'].min().date()} to {sc['game_date'].max().date()}")

    print("[2/6] Loading feature_matrix.parquet …", flush=True)
    fm = pd.read_parquet(FEATURE_MATRIX)
    fm["game_date"] = pd.to_datetime(fm["game_date"])
    print(f"      {len(fm):,} rows × {len(fm.columns)} columns | "
          f"{fm['game_date'].min().date()} to {fm['game_date'].max().date()}")

    # ------------------------------------------------------------------
    # Part 1: Apply ABS mask to feature matrix
    # ------------------------------------------------------------------
    print("\n[3/6] Applying ABS depreciation mask …", flush=True)
    fm_masked, mask_report = apply_abs_mask(fm)
    abs_era_rows = int((fm["game_date"] >= ABS_CUTOFF).sum())
    print(f"      ABS-era rows (>= {ABS_CUTOFF.date()}): {abs_era_rows}")
    print(f"      Columns zeroed: {len(mask_report)}")
    for col, n in mask_report.items():
        print(f"        {col}: {n} rows to 0.0")

    # Write mask report
    mask_lines = [
        "ABS MASK REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"ABS cutoff date: {ABS_CUTOFF.date()}",
        f"Total feature matrix rows: {len(fm)}",
        f"Rows affected (game_date >= cutoff): {abs_era_rows}",
        "",
        "Columns set to 0.0 for ABS-era rows:",
    ]
    for col, n in mask_report.items():
        mask_lines.append(f"  {col}: {n} rows zeroed")
    if not mask_report:
        mask_lines.append("  (none — columns not found in feature matrix)")
    mask_lines += [
        "",
        "These features are mechanically dead under the ABS challenge system:",
        "  - ump_* columns: umpire discretion on ball/strike calls is eliminated",
        "  - framing columns: catcher pitch framing no longer changes call outcomes",
        "",
        "Replacement features are in abs_features_2026.parquet.",
    ]
    os.makedirs(os.path.dirname(OUT_MASK_REPORT), exist_ok=True)
    with open(OUT_MASK_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(mask_lines))
    print(f"      Mask report written to {OUT_MASK_REPORT}")

    # ------------------------------------------------------------------
    # Part 2: Identify starters
    # ------------------------------------------------------------------
    print("\n[4/6] Identifying starting pitchers from Statcast …", flush=True)
    starters = identify_starters(sc)
    print(f"      {len(starters)} games with identified starters")

    # ------------------------------------------------------------------
    # Part 2: Compute pitcher-level features
    # ------------------------------------------------------------------
    print("[5/6] Computing pitch-level features …", flush=True)

    print("      Full-game features …", flush=True)
    full_game = compute_pitcher_game_features(sc, inning_filter=None)
    print(f"        {len(full_game)} pitcher-game rows")

    print("      First-inning features …", flush=True)
    fi = compute_pitcher_game_features(sc, inning_filter=1)
    print(f"        {len(fi)} pitcher-game rows (inning 1)")

    print("      F5-window features (innings 1-5) …", flush=True)
    f5w = compute_pitcher_game_features(sc, inning_filter=1, inning_max=5)
    print(f"        {len(f5w)} pitcher-game rows (innings 1-5)")

    # ------------------------------------------------------------------
    # Part 3: Pivot to game level
    # ------------------------------------------------------------------
    print("      Pivoting to game level …", flush=True)
    abs_features = pivot_to_game_level(starters, full_game, fi, f5_feats=f5w)
    print(f"        {len(abs_features)} game rows × {len(abs_features.columns)} columns")

    # ------------------------------------------------------------------
    # Part 4: Coverage validation
    # ------------------------------------------------------------------
    print("\n[6/6] Validating feature coverage …", flush=True)
    coverage_report = validate_feature_coverage(abs_features)
    print(coverage_report)

    # Embed coverage report into mask report file
    with open(OUT_MASK_REPORT, "a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write(coverage_report)

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(OUT_FEATURES), exist_ok=True)
    abs_features.to_parquet(OUT_FEATURES, index=False)
    print(f"\nSaved to {OUT_FEATURES}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    key_feats = [
        "home_sp_edge_pct", "home_sp_zone_pct",
        "home_sp_whiff_pct", "home_sp_abs_vulnerability",
        "away_sp_whiff_pct", "away_sp_abs_vulnerability",
        "sp_abs_vulnerability_diff",
        "home_sp_f5_whiff_pct", "away_sp_f5_whiff_pct",
        "home_sp_f5_abs_vulnerability", "away_sp_f5_abs_vulnerability",
        "sp_f5_whiff_pct_diff", "sp_f5_abs_vulnerability_diff",
    ]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Rows processed (pitches):      {len(sc):>10,}")
    print(f"  Games in output:               {len(abs_features):>10,}")
    print(f"  Features built:                {len(abs_features.columns):>10,}")
    print(f"  ABS-era feature matrix rows:   {abs_era_rows:>10,}")
    print(f"  Ump/framing columns zeroed:    {len(mask_report):>10,}")
    print(f"  Date range (new features):     "
          f"{abs_features['game_date'].min().date()} to "
          f"{abs_features['game_date'].max().date()}")
    print()
    print("  Average values of key new features:")
    for col in key_feats:
        if col in abs_features.columns:
            val = abs_features[col].mean()
            print(f"    {col:<42} {val:.4f}" if not np.isnan(val) else
                  f"    {col:<42} NaN")

    print()
    print("  NOTE: abs_* features are intentionally NaN for 2023-2025 rows.")
    print("        Do not back-fill across the ABS regime boundary.")
    print()
    print(f"Outputs:")
    print(f"  {OUT_FEATURES}")
    print(f"  {OUT_MASK_REPORT}")
    print("=" * 60)
    print(f"Done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
