"""
build_catcher_iq.py — v5.0 Catcher Challenge IQ Builder.

In 2026's ABS (Automated Ball-Strike) Challenge system, catchers can contest
borderline automated calls.  The league-wide overturn rate is ~56%.  Catchers
who challenge strategically reclaim strikes at a meaningfully higher rate than
the baseline, boosting their pitcher's effective K probability.

This script computes a per-catcher `reclaimed_strike_rate` and assigns a
K-anchor multiplier used by fetch_live_odds.py and data_orchestrator.py.

Methodology
-----------
Primary source: Baseball Savant catcher-framing leaderboard (2026).
  framing_runs     → framing runs above average (proxy for net extra strikes)
  n_innings_caught → playing time normaliser

reclaimed_strike_rate = framing_runs / n_innings_caught   (higher = better)

Elite gate: iq_percentile >= 75th → k_multiplier = 1.03x on K-anchor.
The 56% baseline is encoded as the league mean; outliers above 70th percentile
represent genuinely elite ABS challengers.

Fallback hierarchy:
  1. catcher_framing_2026.parquet  (built by statcast_framing_pull.py)
  2. Baseball Savant leaderboard API (live pull)
  3. Prior-year framing (2025) with regression-to-mean adjustment

Output
------
  data/statcast/catcher_iq_2026.parquet
  data/statcast/catcher_iq_by_team_2026.parquet   (team-aggregated, for context join)

Usage
-----
  python build_catcher_iq.py              # build 2026
  python build_catcher_iq.py --year 2025  # historical backfill
  python build_catcher_iq.py --force      # re-pull even if cached
"""
from __future__ import annotations

import argparse
import io
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR   = _ROOT / "data/statcast"
FRAMING_DIR  = _ROOT / "data/statcast"

# ABS system constants (2026 season)
ABS_LEAGUE_OVERTURN_RATE = 0.56     # 56% league-wide challenge win rate
ELITE_PERCENTILE         = 75       # top quartile = elite challenger
K_MULTIPLIER_ELITE       = 1.03     # applied to K-anchor for elite catchers
K_MULTIPLIER_BASELINE    = 1.00     # non-elite baseline

# Minimum innings caught to be included (< this = small sample, excluded)
MIN_INNINGS = 50


# ---------------------------------------------------------------------------
# Source 1: load existing catcher_framing parquet
# ---------------------------------------------------------------------------

def _load_framing_parquet(year: int) -> pd.DataFrame:
    p = FRAMING_DIR / f"catcher_framing_{year}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    print(f"  [catcher_iq] loaded catcher_framing_{year}.parquet: {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Source 2: live Baseball Savant catcher-framing leaderboard
# ---------------------------------------------------------------------------

SAVANT_FRAMING_URL = (
    "https://baseballsavant.mlb.com/leaderboard/catcher-framing"
    "?type=all&teamId=&min={min_pa}&csv=true&season={year}"
)


def _fetch_savant_framing(year: int) -> pd.DataFrame:
    """Pull framing leaderboard CSV from Baseball Savant."""
    import requests

    url = SAVANT_FRAMING_URL.format(year=year, min_pa=MIN_INNINGS)
    try:
        r = requests.get(url, timeout=30,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        print(f"  [catcher_iq] Savant live pull: {len(df)} catchers ({year})")
        return df
    except Exception as exc:
        print(f"  [catcher_iq] Savant pull failed: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Column normalisation — Savant changes column names across seasons
# ---------------------------------------------------------------------------

_COL_MAP = {
    # Savant column → canonical
    "last_name, first_name": "player_name_raw",
    "player_name":           "player_name_raw",
    "name":                  "player_name_raw",
    "team_name":             "team_raw",
    "team":                  "team_raw",
    "n_called_pitches":      "n_pitches",
    "runs_extra_strikes":    "framing_runs",
    "r_framing":             "framing_runs",
    "framing_runs":          "framing_runs",
    "n_innings":             "n_innings_caught",
    "innings":               "n_innings_caught",
    "player_id":             "player_id",
}

# Abbreviation map reused from statcast_framing_pull.py
TEAM_ABB: dict[str, str] = {
    "Angels": "LAA", "Astros": "HOU", "Athletics": "ATH", "Blue Jays": "TOR",
    "Braves": "ATL", "Brewers": "MIL", "Cardinals": "STL", "Cubs": "CHC",
    "Diamondbacks": "AZ", "Dodgers": "LAD", "Giants": "SF", "Guardians": "CLE",
    "Mariners": "SEA", "Marlins": "MIA", "Mets": "NYM", "Nationals": "WSH",
    "Orioles": "BAL", "Padres": "SD", "Phillies": "PHI", "Pirates": "PIT",
    "Rangers": "TEX", "Rays": "TB", "Red Sox": "BOS", "Reds": "CIN",
    "Rockies": "COL", "Royals": "KC", "Tigers": "DET", "Twins": "MIN",
    "White Sox": "CWS", "Yankees": "NYY",
}


def _normalise_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names and clean types."""
    df = raw.copy()
    # Rename known column aliases
    rename = {k: v for k, v in _COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Normalise player name (Savant sometimes: "Last, First")
    if "player_name_raw" in df.columns:
        def _flip_name(n: str) -> str:
            n = str(n).strip()
            if "," in n:
                parts = [p.strip() for p in n.split(",", 1)]
                return f"{parts[1]} {parts[0]}"
            return n
        df["player_name"] = df["player_name_raw"].apply(_flip_name)
    elif "player_name" not in df.columns:
        df["player_name"] = "Unknown"

    # Normalise team abbreviation
    if "team_raw" in df.columns:
        df["team"] = df["team_raw"].apply(
            lambda t: TEAM_ABB.get(str(t).strip(), str(t).strip().upper()[:3]))
    elif "team" not in df.columns:
        df["team"] = "UNK"

    # Coerce numerics
    for col in ("framing_runs", "n_innings_caught", "player_id"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure framing_runs column exists
    if "framing_runs" not in df.columns:
        # Some years use different column — derive from raw columns if possible
        if "runs_extra_strikes" in df.columns:
            df["framing_runs"] = pd.to_numeric(df["runs_extra_strikes"], errors="coerce")
        else:
            df["framing_runs"] = np.nan

    if "n_innings_caught" not in df.columns:
        df["n_innings_caught"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_catcher_iq(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Given a normalised framing DataFrame, compute IQ metrics.

    Returns one row per catcher with:
      player_name, team, season, n_innings_caught, framing_runs,
      reclaimed_strike_rate, iq_percentile, elite_iq, k_multiplier
    """
    df = df.copy()
    df["season"] = year

    # Filter minimum sample
    if "n_innings_caught" in df.columns:
        df = df[df["n_innings_caught"].fillna(0) >= MIN_INNINGS].copy()

    # Reclaimed strike rate: framing_runs normalised per 9 innings
    # (framing_runs is already in run-value units; higher = better)
    if "n_innings_caught" in df.columns and "framing_runs" in df.columns:
        df["reclaimed_strike_rate"] = (
            df["framing_runs"] / df["n_innings_caught"].replace(0, np.nan) * 9.0
        ).round(4)
    else:
        df["reclaimed_strike_rate"] = np.nan

    # ABS-calibrated baseline: league mean ≈ ABS_LEAGUE_OVERTURN_RATE proxy
    # Scale so that the 56% overturn rate aligns with the median reclaimed rate
    med = df["reclaimed_strike_rate"].median()
    if pd.notna(med) and med != 0:
        # Express as fraction relative to median (1.0 = league average)
        df["iq_relative"] = (df["reclaimed_strike_rate"] / med).round(4)
    else:
        df["iq_relative"] = 1.0

    # Percentile rank (0–100)
    df["iq_percentile"] = (
        df["reclaimed_strike_rate"].rank(pct=True, na_option="bottom") * 100
    ).round(1)

    # Elite gate
    df["elite_iq"]     = df["iq_percentile"] >= ELITE_PERCENTILE
    df["k_multiplier"] = df["elite_iq"].map(
        {True: K_MULTIPLIER_ELITE, False: K_MULTIPLIER_BASELINE})

    keep = ["player_name", "team", "season",
            "n_innings_caught", "framing_runs",
            "reclaimed_strike_rate", "iq_relative",
            "iq_percentile", "elite_iq", "k_multiplier"]
    if "player_id" in df.columns:
        keep = ["player_id"] + keep

    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Team-level aggregation (used by data_orchestrator.py)
# ---------------------------------------------------------------------------

def aggregate_by_team(iq_df: pd.DataFrame) -> pd.DataFrame:
    """Return team-level mean k_multiplier (weighted by innings caught)."""
    if iq_df.empty:
        return pd.DataFrame(columns=["team", "season",
                                      "team_k_multiplier", "n_elite_catchers"])
    g = iq_df.groupby(["team", "season"])
    rows = []
    for (team, season), grp in g:
        total_ip = grp["n_innings_caught"].sum()
        if total_ip > 0:
            w_mult = np.average(
                grp["k_multiplier"].fillna(K_MULTIPLIER_BASELINE),
                weights=grp["n_innings_caught"].fillna(1)
            )
        else:
            w_mult = K_MULTIPLIER_BASELINE
        rows.append({
            "team":               team,
            "season":             season,
            "team_k_multiplier":  round(float(w_mult), 4),
            "n_elite_catchers":   int(grp["elite_iq"].sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Build & save
# ---------------------------------------------------------------------------

def build(year: int = 2026, force: bool = False) -> pd.DataFrame:
    """Full pipeline: load → normalise → compute IQ → save."""
    out_path      = OUTPUT_DIR / f"catcher_iq_{year}.parquet"
    out_team_path = OUTPUT_DIR / f"catcher_iq_by_team_{year}.parquet"

    if out_path.exists() and not force:
        print(f"  [catcher_iq] {out_path.name} exists — use --force to rebuild")
        return pd.read_parquet(out_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Source selection ---
    raw = _load_framing_parquet(year)

    if raw.empty:
        print(f"  [catcher_iq] no parquet for {year} — trying live Savant pull …")
        raw = _fetch_savant_framing(year)

    # Fallback: prior year with regression-to-mean (0.7 weight)
    if raw.empty and year > 2023:
        print(f"  [catcher_iq] falling back to {year - 1} data (0.7 regression)")
        raw = _load_framing_parquet(year - 1)
        if raw.empty:
            raw = _fetch_savant_framing(year - 1)
        if not raw.empty and "framing_runs" in raw.columns:
            raw = raw.copy()
            raw["framing_runs"] = raw["framing_runs"] * 0.7   # regression to mean

    if raw.empty:
        print("  [catcher_iq] no source data available — returning empty frame")
        return pd.DataFrame()

    norm = _normalise_df(raw)
    iq   = compute_catcher_iq(norm, year)
    team = aggregate_by_team(iq)

    iq.to_parquet(out_path, index=False)
    team.to_parquet(out_team_path, index=False)

    n_elite = int(iq["elite_iq"].sum())
    print(f"  [catcher_iq] {len(iq)} catchers | {n_elite} elite (IQ >= {ELITE_PERCENTILE}th pctile)")
    print(f"  [catcher_iq] saved → {out_path.name}")
    print(f"  [catcher_iq] saved → {out_team_path.name}")

    return iq


# ---------------------------------------------------------------------------
# Quick-lookup helper (used by data_orchestrator.py)
# ---------------------------------------------------------------------------

_IQ_CACHE: dict[int, pd.DataFrame] = {}


def load_team_iq(year: int = 2026) -> pd.DataFrame:
    """Load team-level catcher IQ (lazy, cached in-process)."""
    if year not in _IQ_CACHE:
        p = OUTPUT_DIR / f"catcher_iq_by_team_{year}.parquet"
        if p.exists():
            _IQ_CACHE[year] = pd.read_parquet(p)
        else:
            _IQ_CACHE[year] = pd.DataFrame()
    return _IQ_CACHE[year]


def get_team_k_multiplier(team: str, year: int = 2026) -> float:
    """Return catcher IQ K-multiplier for a team. Defaults to 1.0 if unknown."""
    df = load_team_iq(year)
    if df.empty:
        return K_MULTIPLIER_BASELINE
    match = df[(df["team"].str.upper() == team.upper()) & (df["season"] == year)]
    if match.empty:
        return K_MULTIPLIER_BASELINE
    return float(match.iloc[0]["team_k_multiplier"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Catcher Challenge IQ feature (v5.0 ABS system)")
    parser.add_argument("--year",  type=int, default=date.today().year)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    iq = build(year=args.year, force=args.force)
    if iq.empty:
        print("No data — run statcast_framing_pull.py first.")
        return

    print(f"\n  Top 10 catchers by reclaimed_strike_rate ({args.year}):")
    top = iq.nlargest(10, "reclaimed_strike_rate")
    for _, r in top.iterrows():
        elite_tag = " ★ ELITE" if r["elite_iq"] else ""
        print(f"    {r['player_name']:28s}  {r['team']:4s}  "
              f"RSR={r['reclaimed_strike_rate']:+.3f}  "
              f"pctile={r['iq_percentile']:5.1f}  "
              f"mult={r['k_multiplier']:.2f}x{elite_tag}")


if __name__ == "__main__":
    main()
