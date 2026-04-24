"""
fetch_prop_market_signals.py — PrizePicks / Retail Sharp Steam Capture (v1.0).

Reads per-day k_props parquet files (FanDuel, DraftKings, BetMGM) and:

  1. Computes a retail consensus line per pitcher (median across books).
  2. Derives a PrizePicks proxy line (retail floor to nearest 0.5 step).
  3. Calculates market_delta_k = PrizePicks_line - retail_consensus.
  4. Detects cross-book steam: line_range = max_line - min_line.
  5. Infers sharp lean: the book with the lowest K line is implying sharp
     money on the OVER (they moved down to limit exposure).
  6. Computes juice-implied probability per pitcher (over/under average).

market_delta_k interpretation
------------------------------
  Negative (-0.5):  PrizePicks is offering a lower line → easier OVER
  Zero (0.0):       Aligned with retail consensus
  Positive (+0.5):  Rare — PrizePicks pricing aggressively vs retail

Steam detection
---------------
  line_range >= 0.5  → books disagree; sharp money has moved one side
  over_juice_lean > 0.05 → aggregate market leaning toward the OVER
                            (books pricing OVER as less likely than 50%)

Output columns (per pitcher per game)
--------------------------------------
  game_date, home_team, away_team, pitcher_name
  retail_consensus_line   — median across all retail books
  prizepicks_proxy_line   — floor(retail * 2) / 2  (nearest 0.5 step down)
  market_delta_k          — prizepicks_proxy - retail_consensus
  line_range              — max - min across books (0 = consensus, >=0.5 = steam)
  sharp_book              — book with the lowest line (suspected sharp side)
  sharp_line              — that book's line
  soft_line               — highest line across books
  over_juice_avg          — average market-implied probability for the OVER
  under_juice_avg         — average market-implied probability for the UNDER
  juice_lean              — over_juice_avg - 0.5 (positive = public on over)
  n_books                 — number of books quoting this pitcher
  steam_flag              — 1 if line_range >= 0.5 OR juice_lean > 0.06

Outputs
-------
  data/statcast/prop_market_signals_{date}.parquet
  data/statcast/prop_market_signals_{date}.csv

Usage
-----
  python fetch_prop_market_signals.py
  python fetch_prop_market_signals.py --date 2026-04-24
  python fetch_prop_market_signals.py --date 2026-04-24 --verbose
"""
from __future__ import annotations

import argparse
import glob
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT       = Path(__file__).resolve().parent
KPROPS_DIR  = _ROOT / "data/statcast"
OUTPUT_DIR  = _ROOT / "data/statcast"

# American odds to implied probability
def _american_to_prob(american: float) -> float | None:
    try:
        a = float(american)
        if np.isnan(a) or a == 0:
            return None
        if a > 0:
            return 100.0 / (100.0 + a)
        return abs(a) / (abs(a) + 100.0)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Load k_props for a given date
# ---------------------------------------------------------------------------

def load_kprops(date_str: str) -> pd.DataFrame:
    """
    Load k_props_{date}.parquet; fall back to latest available file.
    Returns empty DataFrame if none found.
    """
    tag  = date_str.replace("-", "_")
    path = KPROPS_DIR / f"k_props_{tag}.parquet"

    if not path.exists():
        files = sorted(glob.glob(str(KPROPS_DIR / "k_props_*.parquet")))
        if not files:
            print(f"  [steam] No k_props files found in {KPROPS_DIR}")
            return pd.DataFrame()
        path = Path(files[-1])
        print(f"  [steam] k_props for {date_str} not found; using {path.name}")

    df = pd.read_parquet(path)
    # Normalise book name
    df["book"] = df["book"].str.lower().str.strip()
    return df


# ---------------------------------------------------------------------------
# PrizePicks proxy line
# ---------------------------------------------------------------------------

def _prizepicks_proxy(retail_median: float) -> float:
    """
    PrizePicks typically sets K lines at 0.5 increments and tends to price
    1-2 half-steps below the retail consensus to attract action.

    Conservative proxy: round retail_median DOWN to the nearest 0.5 step.
    This mimics the PrizePicks 'player power play' line behaviour.
    """
    return np.floor(retail_median * 2.0) / 2.0


# ---------------------------------------------------------------------------
# Per-pitcher signal computation
# ---------------------------------------------------------------------------

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a k_props DataFrame (one row per pitcher × book), compute one
    row per pitcher per game containing all market signals.
    """
    if df.empty:
        return pd.DataFrame()

    records = []
    group_keys = ["game_date", "home_team", "away_team", "pitcher_name"]
    # Some files may not have event_id; group on available keys
    avail_keys = [k for k in group_keys if k in df.columns]

    for grp_vals, grp in df.groupby(avail_keys, sort=False):
        grp_dict = dict(zip(avail_keys, grp_vals
                            if isinstance(grp_vals, tuple) else (grp_vals,)))

        lines = grp["line"].dropna().astype(float)
        if lines.empty:
            continue

        # Retail consensus (median across books)
        retail_med = float(lines.median())
        retail_mean = float(lines.mean())
        min_line   = float(lines.min())
        max_line   = float(lines.max())
        n_books    = int(lines.count())

        # PrizePicks proxy
        pp_proxy   = _prizepicks_proxy(retail_med)
        delta_k    = round(pp_proxy - retail_med, 2)   # typically -0.5 or 0

        # Line range (steam proxy)
        line_range = round(max_line - min_line, 1)

        # Sharp / soft book (lowest line = sharp money on OVER)
        line_by_book = grp.dropna(subset=["line"]).set_index("book")["line"].astype(float)
        if not line_by_book.empty:
            sharp_book = line_by_book.idxmin()
            soft_book  = line_by_book.idxmax()
            sharp_line = float(line_by_book.min())
            soft_line  = float(line_by_book.max())
        else:
            sharp_book = soft_book = None
            sharp_line = soft_line = retail_med

        # Juice / implied probability
        over_probs  = grp["over_odds"].apply(
            lambda v: _american_to_prob(v)).dropna().values
        under_probs = grp["under_odds"].apply(
            lambda v: _american_to_prob(v)).dropna().values

        over_avg   = float(np.mean(over_probs))  if len(over_probs)  > 0 else 0.5
        under_avg  = float(np.mean(under_probs)) if len(under_probs) > 0 else 0.5
        juice_lean = round(over_avg - 0.5, 4)   # positive = public on over

        # Steam flag: books disagree on line OR heavy over juice
        steam_flag = int(line_range >= 0.5 or juice_lean > 0.06)

        records.append({
            **grp_dict,
            "retail_consensus_line":  round(retail_med, 2),
            "retail_mean_line":       round(retail_mean, 2),
            "prizepicks_proxy_line":  pp_proxy,
            "market_delta_k":         delta_k,
            "line_range":             line_range,
            "sharp_book":             sharp_book,
            "sharp_line":             round(sharp_line, 1),
            "soft_book":              soft_book,
            "soft_line":              round(soft_line, 1),
            "over_juice_avg":         round(over_avg, 4),
            "under_juice_avg":        round(under_avg, 4),
            "juice_lean":             juice_lean,
            "n_books":                n_books,
            "steam_flag":             steam_flag,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Join with SGP K-line data (enrich SGP score context)
# ---------------------------------------------------------------------------

def enrich_sgp_with_steam(signals: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """
    Join market steam signals onto the sgp_live_edge CSV so that the
    k-prop-specific leg can be evaluated against market steam.

    Returns enriched SGP DataFrame (market_delta_k, steam_flag per row).
    """
    sgp_path = _ROOT / f"data/sgp/sgp_live_edge_{date_str.replace('-','_')}.csv"
    if not sgp_path.exists() or signals.empty:
        return pd.DataFrame()

    sgp = pd.read_csv(sgp_path)
    if "home_sp" not in sgp.columns or "home_team" not in sgp.columns:
        return sgp

    # Match on home_team + pitcher name in signals
    sig = signals[["home_team", "pitcher_name",
                   "market_delta_k", "steam_flag",
                   "retail_consensus_line", "prizepicks_proxy_line",
                   "juice_lean", "line_range"]].copy()
    sig = sig.rename(columns={"pitcher_name": "_sp"})

    # Join via home_sp (SGP script uses home_sp field)
    enriched = sgp.merge(
        sig.rename(columns={"_sp": "home_sp"}),
        on=["home_team", "home_sp"],
        how="left",
        suffixes=("", "_steam"),
    )
    return enriched


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

def print_report(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("  PROP MARKET STEAM SIGNALS")
    print("=" * 70)
    if df.empty:
        print("  No signals computed.")
        return

    steam = df[df["steam_flag"] == 1]
    print(f"  Total pitchers: {len(df)} | Steam flags: {len(steam)}")
    print()

    if not steam.empty:
        print(f"  {'Pitcher':24s}  {'Line':>6s}  {'PP':>5s}  {'Delta':>6s}  "
              f"{'Range':>6s}  {'Lean':>7s}  {'SharpBook':12s}")
        print("  " + "-" * 68)
        for _, r in steam.sort_values("line_range", ascending=False).iterrows():
            print(f"  {r['pitcher_name']:24s}  "
                  f"{r['retail_consensus_line']:>6.1f}  "
                  f"{r['prizepicks_proxy_line']:>5.1f}  "
                  f"{r['market_delta_k']:>+6.2f}  "
                  f"{r['line_range']:>6.1f}  "
                  f"{r['juice_lean']:>+7.4f}  "
                  f"{str(r['sharp_book'] or ''):12s}")

    if len(df[df["steam_flag"] == 0]) > 0:
        print(f"\n  Non-steam pitchers: {len(df[df['steam_flag']==0])} "
              f"(median delta={df['market_delta_k'].median():+.2f})")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_signals(df: pd.DataFrame, date_str: str) -> None:
    if df.empty:
        return
    tag     = date_str.replace("-", "_")
    out_pq  = OUTPUT_DIR / f"prop_market_signals_{tag}.parquet"
    out_csv = OUTPUT_DIR / f"prop_market_signals_{tag}.csv"
    df.to_parquet(out_pq,  index=False)
    df.to_csv(out_csv, index=False)
    print(f"\n  Saved -> {out_pq.name}")
    print(f"  Saved -> {out_csv.name}")


# ---------------------------------------------------------------------------
# Public API (importable)
# ---------------------------------------------------------------------------

def load_signals(date_str: str) -> pd.DataFrame:
    """Load pre-computed prop market signals for a given date."""
    tag  = date_str.replace("-", "_")
    path = OUTPUT_DIR / f"prop_market_signals_{tag}.parquet"
    if not path.exists():
        # Fall back to CSV
        path_csv = OUTPUT_DIR / f"prop_market_signals_{tag}.csv"
        if path_csv.exists():
            return pd.read_csv(path_csv)
        return pd.DataFrame()
    return pd.read_parquet(path)


def get_pitcher_steam(pitcher_name: str, date_str: str) -> dict | None:
    """
    Return market steam signals for a single pitcher on a given date.
    Used by SGP scorer and dashboard.
    """
    df = load_signals(date_str)
    if df.empty:
        return None
    mask = df["pitcher_name"].str.lower() == pitcher_name.strip().lower()
    if not mask.any():
        return None
    row = df[mask].iloc[0]
    return {
        "retail_line":    row.get("retail_consensus_line"),
        "prizepicks_line": row.get("prizepicks_proxy_line"),
        "market_delta_k": row.get("market_delta_k"),
        "steam_flag":     int(row.get("steam_flag", 0)),
        "juice_lean":     row.get("juice_lean"),
        "line_range":     row.get("line_range"),
        "sharp_book":     row.get("sharp_book"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(date_str: str, verbose: bool = False) -> pd.DataFrame:
    print(f"[steam_signals] date={date_str}")
    kprops  = load_kprops(date_str)
    signals = compute_signals(kprops)

    if verbose or not signals.empty:
        print_report(signals)

    save_signals(signals, date_str)

    # Attempt SGP enrichment (best-effort)
    enriched_sgp = enrich_sgp_with_steam(signals, date_str)
    if not enriched_sgp.empty:
        tag    = date_str.replace("-", "_")
        out_sg = _ROOT / f"data/sgp/sgp_live_edge_steam_{tag}.csv"
        enriched_sgp.to_csv(out_sg, index=False)
        print(f"  Saved -> {out_sg.name}")

    n_steam = int(signals["steam_flag"].sum()) if not signals.empty else 0
    print(f"\n  Summary: {len(signals)} pitchers | {n_steam} steam flags")

    return signals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prop market steam capture (v1.0)")
    parser.add_argument("--date",    default=date.today().isoformat())
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(date_str=args.date, verbose=args.verbose)


if __name__ == "__main__":
    main()
