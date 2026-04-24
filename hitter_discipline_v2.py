"""
hitter_discipline_v2.py — Hitter Check-Swing / ABS Challenge IQ Model (v2.0).

In the 2026 ABS (Automated Ball-Strike) Challenge system, batters can contest
borderline automated calls. A hitter who understands the strike zone and controls
their swing on borderline pitches benefits from more deep counts and better outcomes.

Since Statcast 2026 has no direct 'check_swing' event, we proxy discipline via:
  1. bat_speed            — lower bat speed = more controlled / shorter swings
  2. swing_length         — shorter swing length = more disciplined contact
  3. whiff_percent        — lower whiff = better swing decisions
  4. chase_percent        — lower chase = better out-of-zone discipline
  5. bb_percent           — higher walk rate = better zone recognition
  6. squared_up_rate      — higher = better contact precision on swings taken
  7. k_percent            — lower K% = survives deep counts

These 7 signals are combined into a `check_swing_iq` composite score via
z-score normalization and direction-adjusted sum.

Output: per-hitter `check_swing_iq`, percentile rank, `elite_discipline` flag
(top 25%), and `k_reduction` applied to opposing SP's K probability.

The -0.04 K reduction applies when an elite hitter faces a K-heavy SP
(>= 25th pctile K rate), reflecting their ability to foul off borderline
pitches and turn potential K's into walks/hits via ABS challenges.

Data sources
------------
  data/statcast/batter_percentiles_{year}.parquet  — bat_speed, swing_length,
    whiff_percent, chase_percent, bb_percent, squared_up_rate, k_percent
  (batter_splits are NOT used — splits require team context)

Output
------
  data/statcast/hitter_discipline_2026.parquet     — full per-batter scores
  data/statcast/hitter_discipline_by_team_2026.parquet  — team-level averages

Usage
-----
  python hitter_discipline_v2.py              # build 2026
  python hitter_discipline_v2.py --year 2025  # historical
  python hitter_discipline_v2.py --force      # re-compute even if cached
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT        = Path(__file__).resolve().parent
STATCAST_DIR = _ROOT / "data/statcast"
OUTPUT_DIR   = _ROOT / "data/statcast"

# Discipline signal definitions: (column, direction, weight)
# direction: +1 means higher is better discipline; -1 means lower is better
_SIGNALS: list[tuple[str, int, float]] = [
    ("bat_speed",        -1, 1.0),   # lower = more controlled
    ("swing_length",     -1, 1.2),   # lower = more disciplined swing (higher weight)
    ("whiff_percent",    -1, 1.2),   # lower = better decisions
    ("chase_percent",    -1, 1.5),   # lower = best zone discipline signal (highest weight)
    ("bb_percent",       +1, 1.0),   # higher = better zone recognition
    ("squared_up_rate",  +1, 0.8),   # higher = better contact on swings taken
    ("k_percent",        -1, 1.0),   # lower = survives deep counts
]

ELITE_PERCENTILE    = 75.0   # top quartile = elite discipline
K_REDUCTION_ELITE   = -0.04  # K probability reduction applied to opposing SP
K_REDUCTION_GOOD    = -0.02  # 50–75th percentile hitters get half reduction
MIN_PA              = 50     # minimum PA qualifier


# ---------------------------------------------------------------------------
# Load raw batter data
# ---------------------------------------------------------------------------

def _load_batter_data(year: int) -> pd.DataFrame:
    """Load batter percentiles parquet for the given year."""
    p = STATCAST_DIR / f"batter_percentiles_{year}.parquet"
    if not p.exists():
        print(f"  [discipline] batter_percentiles_{year}.parquet not found")
        return pd.DataFrame()

    df = pd.read_parquet(p)
    print(f"  [discipline] loaded batter_percentiles_{year}.parquet: {len(df)} batters")

    # Coerce all signal columns to numeric
    for col, _, _ in _SIGNALS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Compute discipline IQ
# ---------------------------------------------------------------------------

def compute_discipline_iq(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Compute per-batter check_swing_iq from available signals.

    Returns DataFrame with: player_id, player_name, team, season,
      check_swing_iq, iq_percentile, elite_discipline, k_reduction,
      signals_used (count)
    """
    df = df.copy()
    df["season"] = year

    z_cols = []
    for col, direction, weight in _SIGNALS:
        if col not in df.columns:
            continue

        valid = df[col].notna()
        if valid.sum() < 10:
            continue

        mu  = df.loc[valid, col].mean()
        sig = df.loc[valid, col].std()
        if sig < 1e-9:
            continue

        z_col = f"_z_{col}"
        df[z_col] = ((df[col] - mu) / sig) * direction * weight
        df[z_col] = df[z_col].fillna(0.0)
        z_cols.append(z_col)

    if not z_cols:
        print("  [discipline] no valid signal columns — returning empty frame")
        return pd.DataFrame()

    df["check_swing_iq"] = df[z_cols].sum(axis=1).round(4)
    df["signals_used"]   = len(z_cols)

    # Percentile rank
    df["iq_percentile"] = (
        df["check_swing_iq"].rank(pct=True, na_option="bottom") * 100
    ).round(1)

    # Elite gate
    df["elite_discipline"] = df["iq_percentile"] >= ELITE_PERCENTILE
    df["good_discipline"]  = (df["iq_percentile"] >= 50.0) & (~df["elite_discipline"])

    df["k_reduction"] = np.where(
        df["elite_discipline"], K_REDUCTION_ELITE,
        np.where(df["good_discipline"], K_REDUCTION_GOOD, 0.0)
    )

    # Clean up temp z-columns
    df = df.drop(columns=z_cols, errors="ignore")

    keep = ["player_id", "player_name", "season",
            "check_swing_iq", "iq_percentile",
            "elite_discipline", "good_discipline", "k_reduction",
            "signals_used",
            # Carry through raw signals for auditability
            "bat_speed", "swing_length", "whiff_percent",
            "chase_percent", "bb_percent", "squared_up_rate", "k_percent"]

    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Team-level aggregation
# ---------------------------------------------------------------------------

def aggregate_by_team(discipline_df: pd.DataFrame,
                       splits_year: int) -> pd.DataFrame:
    """
    Aggregate discipline IQ by team.

    Joins team from batter_splits (player -> team) since batter_percentiles
    doesn't include team column.  Falls back to player-level mean if no join.
    """
    splits_path = _ROOT / "data/statcast" / f"batter_splits_{splits_year}.parquet"
    if not splits_path.exists():
        # Try adjacent years
        for yr_try in (splits_year, splits_year - 1, splits_year + 1):
            p = _ROOT / "data/statcast" / f"batter_splits_{yr_try}.parquet"
            if p.exists():
                splits_path = p
                break

    df = discipline_df.copy()
    if splits_path.exists():
        splits = pd.read_parquet(splits_path)
        # Look for team identifier columns
        team_col = next((c for c in splits.columns if "team" in c.lower()), None)
        pid_col  = next((c for c in splits.columns if "player_id" in c.lower()), None)

        if team_col and pid_col:
            team_map = (splits[[pid_col, team_col]]
                        .drop_duplicates(subset=[pid_col])
                        .rename(columns={pid_col: "player_id", team_col: "team"}))
            df = df.merge(
                team_map[["player_id", "team"]],
                on="player_id", how="left"
            )

    if "team" not in df.columns:
        print("  [discipline] no team column available — skipping team aggregation")
        return pd.DataFrame()

    df = df.dropna(subset=["team"])
    if df.empty:
        return pd.DataFrame()

    grp = (df.groupby("team")
             .agg(
                 avg_discipline_iq=("check_swing_iq", "mean"),
                 avg_k_reduction=("k_reduction", "mean"),
                 n_elite=("elite_discipline", "sum"),
                 n_players=("player_id", "count"),
             )
             .reset_index())
    grp["season"] = splits_year
    grp["avg_discipline_iq"] = grp["avg_discipline_iq"].round(4)
    grp["avg_k_reduction"]   = grp["avg_k_reduction"].round(4)
    return grp


# ---------------------------------------------------------------------------
# Build & save
# ---------------------------------------------------------------------------

def build(year: int = 2026, force: bool = False) -> pd.DataFrame:
    """Full pipeline: load -> compute IQ -> save."""
    out_path      = OUTPUT_DIR / f"hitter_discipline_{year}.parquet"
    out_team_path = OUTPUT_DIR / f"hitter_discipline_by_team_{year}.parquet"

    if out_path.exists() and not force:
        print(f"  [discipline] {out_path.name} exists — use --force to rebuild")
        return pd.read_parquet(out_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = _load_batter_data(year)
    if raw.empty:
        print(f"  [discipline] no data for {year}")
        return pd.DataFrame()

    iq   = compute_discipline_iq(raw, year)
    team = aggregate_by_team(iq, year)

    iq.to_parquet(out_path, index=False)
    if not team.empty:
        team.to_parquet(out_team_path, index=False)

    n_elite = int(iq["elite_discipline"].sum()) if not iq.empty else 0
    print(f"  [discipline] {len(iq)} batters | {n_elite} elite "
          f"(iq >= {ELITE_PERCENTILE}th pctile) | "
          f"signals used: {iq['signals_used'].iloc[0] if not iq.empty else 0}/7")
    print(f"  [discipline] saved -> {out_path.name}")
    if not team.empty:
        print(f"  [discipline] saved -> {out_team_path.name}")

    return iq


# ---------------------------------------------------------------------------
# Lookup helpers (called by data_orchestrator.py)
# ---------------------------------------------------------------------------

_DISC_CACHE: dict[int, pd.DataFrame] = {}


def load_discipline(year: int = 2026) -> pd.DataFrame:
    """Load hitter discipline parquet (lazy, cached in-process)."""
    if year not in _DISC_CACHE:
        p = OUTPUT_DIR / f"hitter_discipline_{year}.parquet"
        _DISC_CACHE[year] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    return _DISC_CACHE[year]


def get_hitter_k_reduction(player_id: int | float,
                            year: int = 2026) -> float:
    """Return k_reduction for a given player_id. 0.0 if unknown."""
    df = load_discipline(year)
    if df.empty:
        return 0.0
    match = df[df["player_id"] == int(player_id)]
    if match.empty:
        return 0.0
    return float(match.iloc[0]["k_reduction"])


def get_lineup_avg_k_reduction(player_ids: list[int],
                                year: int = 2026) -> float:
    """Return the mean k_reduction across a lineup's player IDs."""
    reductions = [get_hitter_k_reduction(pid, year) for pid in player_ids if pid]
    return float(np.mean(reductions)) if reductions else 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Hitter Discipline / ABS Challenge IQ (v2.0)")
    parser.add_argument("--year",  type=int, default=date.today().year)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    iq = build(year=args.year, force=args.force)
    if iq.empty:
        print("No data — ensure batter_percentiles parquet is available.")
        return

    print(f"\n  Top 15 most disciplined batters ({args.year}):")
    top = iq.nlargest(15, "check_swing_iq")
    for _, r in top.iterrows():
        elite_tag = " [ELITE]" if r["elite_discipline"] else (
                    "  [GOOD]"  if r.get("good_discipline", False) else "")
        print(f"    {str(r.get('player_name','')):<28}  "
              f"IQ={r['check_swing_iq']:+.3f}  "
              f"pctile={r['iq_percentile']:5.1f}  "
              f"K_red={r['k_reduction']:+.2f}{elite_tag}")

    print(f"\n  Signal coverage: {iq['signals_used'].iloc[0]}/7 signals available")
    print(f"  K-reduction distribution:")
    print(f"    Elite  ({K_REDUCTION_ELITE:+.2f}): {int(iq['elite_discipline'].sum()):>4} batters")
    print(f"    Good   ({K_REDUCTION_GOOD:+.2f}): "
          f"{int(iq.get('good_discipline', pd.Series(False)).sum()):>4} batters")
    print(f"    None   (  0.00): "
          f"{int((iq['k_reduction'] == 0).sum()):>4} batters")


if __name__ == "__main__":
    main()
