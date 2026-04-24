"""
enrich_ml_features_v8.py — v8.0 Unified Feature Store Migration

Post-processes feature_matrix_enriched_v2.parquet to add 6 v7.0 Sentinel
physics / bio-proxy features that the ML model currently lacks:

  home_bullpen_burn_5d   — Home bullpen 5-day rolling fatigue index
  away_bullpen_burn_5d   — Away bullpen 5-day rolling fatigue index
  ump_k_synergy_home     — Ump K-rate tendency * home-team SP K pct
  ump_k_synergy_away     — Ump K-rate tendency * away-team SP K pct
  home_catcher_k_mult    — Home catcher framing quality -> K mult
  away_catcher_k_mult    — Away catcher framing quality -> K mult

Note: air_density_rho is already in ml_feature_cols_v2.json — not re-added.

Data sources:
  data/batter_features/bullpen_burn_by_game.parquet
    -> game_pk, home_team, away_team, bullpen_burn_5d
  data/statcast/ump_features_{year}.parquet
    -> game_pk, ump_k_above_avg
  data/statcast/catcher_framing_all.parquet
    -> team, season, framing_runs (-> normalised k_mult)

Output:
  feature_matrix_v8.parquet          (7610 x 262 cols)
  ml_feature_cols_v8.json            (ml_feature_cols_v2 + 6 new names)

Usage
-----
  python enrich_ml_features_v8.py
  python enrich_ml_features_v8.py --validate
  python enrich_ml_features_v8.py --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent

FM_IN    = _ROOT / "feature_matrix_enriched_v2.parquet"
FM_OUT   = _ROOT / "feature_matrix_v8.parquet"
FEAT_IN  = _ROOT / "models/ml_feature_cols_v2.json"
FEAT_OUT = _ROOT / "models/ml_feature_cols_v8.json"

BP_FILE   = _ROOT / "data/batter_features/bullpen_burn_by_game.parquet"
UMP_DIR   = _ROOT / "data/statcast"
CF_FILE   = _ROOT / "data/statcast/catcher_framing_all.parquet"

NEW_FEATURES = [
    "home_bullpen_burn_5d",
    "away_bullpen_burn_5d",
    "ump_k_synergy_home",
    "ump_k_synergy_away",
    "home_catcher_k_mult",
    "away_catcher_k_mult",
]

# Catcher framing z-score -> K multiplier scaling
CATCHER_K_SCALE = 0.03   # each 1-SD framing_run unit = 3% K adjustment


# ---------------------------------------------------------------------------
# Bullpen burn joins
# ---------------------------------------------------------------------------

def _build_team_burn_lookup(bp: pd.DataFrame) -> pd.DataFrame:
    """
    Build a (game_date, team) -> bullpen_burn_5d lookup.
    Each row in bp is from the HOME team perspective.  The same team
    appearing as away_team in a different game needs to be looked up
    from its most-recent HOME appearance on or before the game date.
    """
    bp = bp[["game_date", "game_pk", "home_team", "away_team",
             "bullpen_burn_5d"]].copy()
    bp["game_date"] = pd.to_datetime(bp["game_date"])
    # Home-side lookup: team was home on that game_date
    home_side = bp[["game_date", "home_team", "bullpen_burn_5d"]].rename(
        columns={"home_team": "team", "bullpen_burn_5d": "burn_5d"})
    return home_side.dropna(subset=["burn_5d"])


def join_bullpen_burn(fm: pd.DataFrame, bp: pd.DataFrame) -> pd.DataFrame:
    """
    Add home_bullpen_burn_5d and away_bullpen_burn_5d to fm.

    Strategy:
      home_bullpen_burn_5d: direct join on game_pk  (bp row IS the home team)
      away_bullpen_burn_5d: for the away team on game_date, find the most
          recent prior row where that team appeared as home_team (so their
          bullpen fatigue was tracked).
    """
    bp = bp.copy()
    bp["game_date"] = pd.to_datetime(bp["game_date"])
    fm = fm.copy()
    fm["game_date"] = pd.to_datetime(fm["game_date"])

    # --- home_bullpen_burn_5d (direct game_pk join) ---
    bp_home = bp[["game_pk", "bullpen_burn_5d"]].rename(
        columns={"bullpen_burn_5d": "home_bullpen_burn_5d"})
    fm = fm.merge(bp_home, on="game_pk", how="left")

    # --- away_bullpen_burn_5d (lookback join) ---
    team_burn = _build_team_burn_lookup(bp)  # (game_date, team, burn_5d)
    # We need: for each (fm.game_date, fm.away_team), find the last entry
    # in team_burn where team=away_team and date <= fm.game_date
    # Efficient approach: merge-asof per team

    away_burns: list[pd.Series] = []
    fm_sorted = fm[["game_pk", "game_date", "away_team"]].sort_values("game_date")
    team_burn_sorted = team_burn.sort_values("game_date")

    results: dict[int, float] = {}
    for team, group in team_burn_sorted.groupby("team"):
        mask = fm_sorted["away_team"] == team
        if not mask.any():
            continue
        fm_team = fm_sorted[mask].copy()
        merged = pd.merge_asof(
            fm_team.sort_values("game_date"),
            group[["game_date", "burn_5d"]].sort_values("game_date"),
            on="game_date",
            direction="backward",
        )
        for _, row in merged.iterrows():
            results[row["game_pk"]] = row.get("burn_5d", np.nan)

    fm["away_bullpen_burn_5d"] = fm["game_pk"].map(results)

    return fm


# ---------------------------------------------------------------------------
# Umpire K synergy joins
# ---------------------------------------------------------------------------

def join_ump_k_synergy(fm: pd.DataFrame) -> pd.DataFrame:
    """
    Add ump_k_synergy_home and ump_k_synergy_away to fm.

    Both columns are computed directly from columns already present in the
    feature matrix (ump_k_above_avg, home/away_sp_k_pct_10d) — no second
    merge is needed.

    ump_k_synergy_home = ump_k_above_avg * home_sp_k_pct_10d
    ump_k_synergy_away = ump_k_above_avg * away_sp_k_pct_10d

    Interpretation: positive value = high-K ump paired with high-K SP
    (reinforcing K environment).  Negative = K-suppressing ump + low-K SP.
    """
    if "ump_k_above_avg" not in fm.columns:
        fm["ump_k_synergy_home"] = 0.0
        fm["ump_k_synergy_away"] = 0.0
        return fm

    k_above = fm["ump_k_above_avg"].fillna(0.0)

    home_sp_k = fm["home_sp_k_pct_10d"].fillna(0.22) \
        if "home_sp_k_pct_10d" in fm.columns else pd.Series(0.22, index=fm.index)
    away_sp_k = fm["away_sp_k_pct_10d"].fillna(0.22) \
        if "away_sp_k_pct_10d" in fm.columns else pd.Series(0.22, index=fm.index)

    fm["ump_k_synergy_home"] = (k_above * home_sp_k).round(5)
    fm["ump_k_synergy_away"] = (k_above * away_sp_k).round(5)
    return fm


# ---------------------------------------------------------------------------
# Catcher framing -> K multiplier
# ---------------------------------------------------------------------------

def _compute_catcher_k_mult(cf: pd.DataFrame) -> pd.DataFrame:
    """
    Convert framing_runs (seasonal) to a K multiplier per team-season.

    Normalisation: z-score within each season, then:
        k_mult = 1.0 + z * CATCHER_K_SCALE
    Clipped to [0.85, 1.15] to prevent extreme leverage.
    """
    cf = cf[["team", "season", "framing_runs"]].copy()
    cf["z"] = cf.groupby("season")["framing_runs"].transform(
        lambda s: (s - s.mean()) / max(s.std(ddof=1), 1e-6)
    )
    cf["catcher_k_mult"] = (1.0 + cf["z"] * CATCHER_K_SCALE).clip(0.85, 1.15).round(5)
    return cf[["team", "season", "catcher_k_mult"]]


def join_catcher_k_mult(fm: pd.DataFrame) -> pd.DataFrame:
    """Add home_catcher_k_mult and away_catcher_k_mult to fm."""
    if not CF_FILE.exists():
        fm["home_catcher_k_mult"] = 1.0
        fm["away_catcher_k_mult"] = 1.0
        return fm

    cf = pd.read_parquet(CF_FILE)
    mults = _compute_catcher_k_mult(cf)

    # Home team
    fm = fm.merge(
        mults.rename(columns={"team": "home_team",
                               "catcher_k_mult": "home_catcher_k_mult"}),
        on=["home_team", "season"], how="left",
    )
    # Away team
    fm = fm.merge(
        mults.rename(columns={"team": "away_team",
                               "catcher_k_mult": "away_catcher_k_mult"}),
        on=["away_team", "season"], how="left",
    )

    fm["home_catcher_k_mult"] = fm["home_catcher_k_mult"].fillna(1.0)
    fm["away_catcher_k_mult"] = fm["away_catcher_k_mult"].fillna(1.0)
    return fm


# ---------------------------------------------------------------------------
# Feature list update
# ---------------------------------------------------------------------------

def update_feature_cols(new_features: list[str]) -> list[str]:
    """Append new features to ml_feature_cols_v2.json -> ml_feature_cols_v8.json."""
    if FEAT_IN.exists():
        existing = json.loads(FEAT_IN.read_text())
    else:
        existing = []

    combined = existing + [f for f in new_features if f not in existing]
    FEAT_OUT.write_text(json.dumps(combined, indent=2))
    return combined


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def validate(fm: pd.DataFrame) -> None:
    print("\n  [v8.0 Validation]")
    for col in NEW_FEATURES:
        if col not in fm.columns:
            print(f"  MISSING: {col}")
            continue
        n_null  = fm[col].isna().sum()
        pct_fill = (1 - n_null / len(fm)) * 100
        print(f"  {col:30s}  fill={pct_fill:5.1f}%  "
              f"mean={fm[col].mean():.4f}  "
              f"min={fm[col].min():.4f}  max={fm[col].max():.4f}")

    # Sanity: burn values should be in reasonable range (raw HL pitch count, 5d sum)
    for col in ["home_bullpen_burn_5d", "away_bullpen_burn_5d"]:
        if col in fm.columns:
            outliers = (fm[col] > 700).sum()
            if outliers:
                print(f"  WARNING: {outliers} rows with {col} > 700 (check units)")

    # Sanity: k_mult should be in [0.85, 1.15]
    for col in ["home_catcher_k_mult", "away_catcher_k_mult"]:
        if col in fm.columns:
            bad = ((fm[col] < 0.84) | (fm[col] > 1.16)).sum()
            if bad:
                print(f"  WARNING: {bad} rows out of clip range for {col}")

    print(f"\n  Shape: {fm.shape[0]} rows x {fm.shape[1]} cols")
    print(f"  Output: {FM_OUT.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def enrich(dry_run: bool = False) -> pd.DataFrame:
    print(f"[enrich_ml_features_v8] Loading {FM_IN.name}...")
    fm = pd.read_parquet(FM_IN)
    print(f"  Input shape: {fm.shape}")

    # Check which features already exist
    already = [f for f in NEW_FEATURES if f in fm.columns]
    if already:
        print(f"  Already present (skipping): {already}")
    todo = [f for f in NEW_FEATURES if f not in fm.columns]
    if not todo:
        print("  All v8.0 features already present. Nothing to do.")
        return fm

    # --- 1. Bullpen burn ---
    print("  Joining bullpen_burn_5d...")
    bp = pd.read_parquet(BP_FILE)
    fm = join_bullpen_burn(fm, bp)
    home_fill = fm["home_bullpen_burn_5d"].notna().mean() * 100
    away_fill = fm["away_bullpen_burn_5d"].notna().mean() * 100
    print(f"    home fill: {home_fill:.1f}%  away fill: {away_fill:.1f}%")

    # --- 2. Ump K synergy ---
    print("  Joining ump_k_synergy...")
    fm = join_ump_k_synergy(fm)
    ump_fill = fm["ump_k_synergy_home"].notna().mean() * 100
    print(f"    ump k synergy fill: {ump_fill:.1f}%")

    # --- 3. Catcher K multiplier ---
    print("  Joining catcher_k_mult...")
    fm = join_catcher_k_mult(fm)
    ck_fill = fm["home_catcher_k_mult"].notna().mean() * 100
    print(f"    catcher k mult fill: {ck_fill:.1f}%")

    # --- Feature col list ---
    updated_cols = update_feature_cols(NEW_FEATURES)
    print(f"  Feature list: {len(updated_cols)} cols -> {FEAT_OUT.name}")

    if not dry_run:
        fm.to_parquet(FM_OUT, index=False)
        print(f"  Saved -> {FM_OUT.name}")
    else:
        print("  [dry-run] Output NOT saved.")

    return fm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="v8.0 Unified Feature Store Migration")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Run enrichment without writing output file")
    parser.add_argument("--validate", action="store_true",
                        help="Print per-column fill rates and stats after enrichment")
    args = parser.parse_args()

    fm = enrich(dry_run=args.dry_run)

    if args.validate or args.dry_run:
        validate(fm)


if __name__ == "__main__":
    main()
