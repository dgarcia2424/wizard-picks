"""
fetch_challenger_projections.py
================================
Download external K-prop challenger model projections from FanGraphs and save
to data/statcast/.

Systems pulled:
  Projection systems (direct K% signal):
    - THE BAT X  (batx)   — Statcast-integrated regression, most reactive
    - ATC        (atc)    — Smart aggregate of ZiPS/Steamer/THE BAT
    - ZiPS       (zips)   — Nearest-neighbor cohort / aging-curve model
    - Steamer    (steamer) — Additional aggregate baseline

  Pitch-quality leaderboards (converted to implied K% via linear model):
    - Stuff+      — Physical pitch quality independent of results
    - Pitching+   — Stuff+ + Location+ + count context composite

Usage:
  python fetch_challenger_projections.py             # current year
  python fetch_challenger_projections.py --year 2025
  python fetch_challenger_projections.py --systems batx atc
"""

import argparse
import re
import sys
import time
import unicodedata
import warnings
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

DATA_DIR = Path("./data/statcast")

# FanGraphs API base endpoints
_FG_PROJ_URL   = "https://www.fangraphs.com/api/projections"
_FG_LEADER_URL = "https://www.fangraphs.com/api/leaders/major-league/data"

PROJECTION_SYSTEMS = {
    "thebatx": "THE BAT X",   # FG uses 'thebatx' not 'batx'
    "atc":     "ATC",
    "zips":    "ZiPS",
    "steamer": "Steamer",
}

# FG leaderboard type codes
_FG_STANDARD_TYPE = 1    # ERA, K/9, WHIP, etc.
_FG_STUFF_TYPE    = 36   # Stuff+, Pitching+, Location+

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.fangraphs.com/leaders.aspx",
}

_AVG_BF_PER_IP = 4.35   # league-average batters faced per inning


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _norm_name(name: str) -> str:
    """Normalize pitcher name: remove accents, strip, uppercase."""
    if not isinstance(name, str):
        return ""
    name = name.strip()
    if "," in name:
        last, first = [p.strip() for p in name.split(",", 1)]
        name = f"{first} {last}"
    name = name.upper()
    return "".join(
        c for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    )


def _k_pct_from_projection(row: pd.Series) -> float | None:
    """Extract K% from a FanGraphs projection row using available columns."""
    # Try explicit K% columns first
    for col in ["K%", "k_pct", "kpct", "K_pct"]:
        if col in row.index and pd.notna(row[col]):
            val = float(row[col])
            return val / 100.0 if val > 1.0 else val

    # Fall back to K/9 -> K%: K9 = K/(IP/9) -> K/BF = (K9/9) / (BF/IP)
    k9 = None
    for col in ["K/9", "k9", "K9", "SO9"]:
        if col in row.index and pd.notna(row[col]):
            k9 = float(row[col])
            break
    if k9 is not None:
        return (k9 / 9.0) / _AVG_BF_PER_IP

    # Fall back to SO + IP totals
    so_val = ip_val = None
    for col in ["SO", "so", "K", "strikeOuts"]:
        if col in row.index and pd.notna(row[col]):
            so_val = float(row[col])
            break
    for col in ["IP", "ip", "inningsPitched"]:
        if col in row.index and pd.notna(row[col]):
            ip_val = float(row[col])
            break
    if so_val is not None and ip_val is not None and ip_val > 0:
        return so_val / (ip_val * _AVG_BF_PER_IP)

    return None


def _get_player_id(row: pd.Series) -> str | None:
    """Extract FG or MLBAM player ID from projection row."""
    for col in ["MLBAMID", "mlbamid", "playerid", "PlayerID", "player_id"]:
        if col in row.index and pd.notna(row[col]):
            val = str(row[col]).strip()
            if not val:
                continue
            try:
                return str(int(float(val)))   # numeric IDs
            except ValueError:
                return val                     # FG string IDs like 'sa3020707'
    return None


# ---------------------------------------------------------------------------
# FETCHERS
# ---------------------------------------------------------------------------

def fetch_projection(system_key: str, year: int = 2026) -> pd.DataFrame:
    """
    Fetch a FanGraphs pitcher projection CSV for one system.

    Returns a tidy DataFrame with columns:
      name_key, fg_name, team, k_pct, fg_playerid, mlbam_id
    """
    params = {
        "type":    system_key,
        "stats":   "pit",
        "pos":     "all",
        "team":    "0",
        "players": "0",
        "lg":      "all",
    }
    label = PROJECTION_SYSTEMS.get(system_key, system_key.upper())
    print(f"  Fetching {label} projections from FanGraphs ...", end=" ", flush=True)

    try:
        resp = requests.get(_FG_PROJ_URL, params=params, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"FAILED ({e})")
        return pd.DataFrame()

    if not data:
        print("EMPTY response")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    print(f"OK ({len(df)} pitchers, cols: {list(df.columns[:10])}...)")

    # Find name column
    name_col = None
    for c in ["PlayerName", "Name", "name", "playerName", "FullName", "ShortName"]:
        if c in df.columns:
            name_col = c
            break
    if name_col is None:
        print(f"    [WARN] No name column found in {label} data")
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        k_pct = _k_pct_from_projection(row)
        if k_pct is None or k_pct <= 0:
            continue
        name_raw = str(row[name_col])
        team_raw = ""
        for tc in ["Team", "team", "Tm"]:
            if tc in row.index and pd.notna(row[tc]):
                team_raw = str(row[tc])
                break

        # ERA / FIP for run-prevention challenger eval
        era = None
        for ec in ["ERA", "era"]:
            if ec in row.index and pd.notna(row[ec]):
                era = round(float(row[ec]), 4)
                break
        fip = None
        for fc in ["FIP", "fip", "xFIP", "xfip"]:
            if fc in row.index and pd.notna(row[fc]):
                fip = round(float(row[fc]), 4)
                break

        # MLBAM ID — prefer xMLBAMID (present in newer FG exports)
        mlbam = None
        for ic in ["xMLBAMID", "MLBAMID", "mlbamid"]:
            if ic in row.index and pd.notna(row[ic]):
                try:
                    mlbam = str(int(float(row[ic])))
                except (ValueError, TypeError):
                    pass
                break

        rows.append({
            "name_key":    _norm_name(name_raw),
            "fg_name":     name_raw,
            "team":        team_raw,
            "k_pct":       round(k_pct, 5),
            "era":         era,
            "fip":         fip,
            "fg_playerid": str(row["playerid"]) if "playerid" in row.index and pd.notna(row["playerid"]) else None,
            "mlbam_id":    mlbam or _get_player_id(row),
            "system":      system_key,
        })

    result = pd.DataFrame(rows)
    print(f"    -> {len(result)} pitchers with valid K%")
    return result


def fetch_stuff_plus(year: int = 2026) -> pd.DataFrame:
    """
    Fetch Stuff+, Pitching+, and Location+ from FanGraphs leaderboard.

    Returns DataFrame with columns:
      name_key, fg_name, stuff_plus, pitching_plus, location_plus, k_pct_implied
    """
    print("  Fetching Stuff+/Pitching+ leaderboard from FanGraphs ...", end=" ", flush=True)

    params = {
        "pos":       "all",
        "stats":     "pit",
        "lg":        "all",
        "qual":      "0",
        "season":    str(year),
        "season1":   str(year),
        "startdate": "",
        "enddate":   "",
        "month":     "0",
        "hand":      "",
        "team":      "0",
        "pageitems": "10000",
        "pagenum":   "1",
        "ind":       "0",
        "rost":      "0",
        "players":   "0",
        "type":      str(_FG_STUFF_TYPE),
    }

    try:
        resp = requests.get(_FG_LEADER_URL, params=params, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        print(f"FAILED ({e})")
        return pd.DataFrame()

    # FG leaderboard wraps data in {"data": [...], "count": n}
    data = payload.get("data", payload) if isinstance(payload, dict) else payload
    if not data:
        print("EMPTY response")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    print(f"OK ({len(df)} rows, cols: {list(df.columns[:15])}...)")

    # FanGraphs leaderboard API uses sp_stuff=Stuff+ and sp_pitching=Pitching+
    # (confirmed via inspection of type=36 response)
    stuff_col    = "sp_stuff"    if "sp_stuff"    in df.columns else None
    pitching_col = "sp_pitching" if "sp_pitching" in df.columns else None
    location_col = None   # Location+ not separately exposed in this endpoint

    # Fallback: look for anything with "stuff" in name (avoids pb_stuff which is a bat-tracking metric)
    if stuff_col is None:
        for c in df.columns:
            if "stuff" in c.lower() and "pb" not in c.lower():
                stuff_col = c
                break

    name_col = next((c for c in df.columns if c.lower() in ["playername", "name", "shortname", "player"]), None)
    if name_col is None:
        print("    [WARN] No name column found in Stuff+ leaderboard")
        return pd.DataFrame()

    available = [c for c in [stuff_col, pitching_col] if c]
    if not available:
        print("    [WARN] Stuff+/Pitching+ columns not found — check FG leaderboard type")
        return pd.DataFrame()

    print(f"    Found quality cols: {available}")

    # Linear model: K% ≈ league_avg + (Stuff+ - 100) × sensitivity
    # Derived from historical FG/Statcast data: +10 Stuff+ ≈ +1.8% K rate
    _LEAGUE_K_PCT    = 0.222   # ~22% league-average K%
    _STUFF_SENS      = 0.0018  # per Stuff+ point above/below 100
    _PITCHING_SENS   = 0.0012  # Pitching+ is more correlated with ERA, less with K%

    _strip_html = re.compile(r"<[^>]+>")

    rows = []
    for _, row in df.iterrows():
        name_raw = str(row[name_col]) if pd.notna(row[name_col]) else ""
        # Strip HTML tags if present (some FG endpoints embed anchor tags)
        name_raw = _strip_html.sub("", name_raw).strip()
        if not name_raw:
            continue

        s_plus = float(row[stuff_col])    if stuff_col    and pd.notna(row.get(stuff_col))    else None
        p_plus = float(row[pitching_col]) if pitching_col and pd.notna(row.get(pitching_col)) else None

        # Sanity check: Stuff+ should be in the range [50, 200]; reject outliers
        if s_plus is not None and (s_plus < 40 or s_plus > 250):
            s_plus = None
        if p_plus is not None and (p_plus < 40 or p_plus > 250):
            p_plus = None

        # Implied K% from Stuff+ (primary signal for Ks)
        # Linear model: league avg K% ≈ 22%, each +10 Stuff+ ≈ +1.8% K rate
        # Pitching+ is more correlated with ERA than K%; use 70/30 blend
        if s_plus is not None:
            k_pct_stuff = _LEAGUE_K_PCT + (s_plus - 100.0) * _STUFF_SENS
            if p_plus is not None:
                k_pct_pitching = _LEAGUE_K_PCT + (p_plus - 100.0) * _PITCHING_SENS
                k_pct_implied = 0.70 * k_pct_stuff + 0.30 * k_pct_pitching
            else:
                k_pct_implied = k_pct_stuff
        elif p_plus is not None:
            k_pct_implied = _LEAGUE_K_PCT + (p_plus - 100.0) * _PITCHING_SENS
        else:
            continue

        k_pct_implied = max(0.05, min(0.50, k_pct_implied))

        fg_id = None
        for id_col in ["xMLBAMID", "playerid", "PlayerID", "MLBAMID"]:
            if id_col in row.index and pd.notna(row.get(id_col)):
                try:
                    fg_id = str(int(float(row[id_col])))
                except (ValueError, TypeError):
                    fg_id = str(row[id_col])
                break

        rows.append({
            "name_key":       _norm_name(name_raw),
            "fg_name":        name_raw,
            "stuff_plus":     round(s_plus, 1) if s_plus is not None else None,
            "pitching_plus":  round(p_plus, 1) if p_plus is not None else None,
            "k_pct_implied":  round(k_pct_implied, 5),
            "fg_playerid":    fg_id,
        })

    result = pd.DataFrame(rows)
    print(f"    -> {len(result)} pitchers with Stuff+ data")
    return result


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def refresh_stuff_plus(year: int = 2026) -> bool:
    """
    Lightweight daily refresh — only fetches Stuff+/Pitching+ leaderboard.

    Called by run_today.py at the start of each daily run.
    Returns True if the file was written successfully.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_stuff_plus(year)
    if df.empty:
        return False
    out = DATA_DIR / f"fg_stuff_plus_{year}.parquet"
    df.to_parquet(out, index=False)
    return True


def main():
    parser = argparse.ArgumentParser(description="Fetch FanGraphs challenger projections")
    parser.add_argument("--year",       type=int, default=2026)
    parser.add_argument("--systems",    nargs="+",
                        default=list(PROJECTION_SYSTEMS.keys()),
                        help="Projection systems to fetch")
    parser.add_argument("--no-stuff",   action="store_true",
                        help="Skip Stuff+/Pitching+ leaderboard fetch")
    parser.add_argument("--stuff-only", action="store_true",
                        help="Only refresh Stuff+/Pitching+ (fast daily mode, skips projections)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.stuff_only:
        print(f"\n  [Stuff+ daily refresh — {args.year}]")
        ok = refresh_stuff_plus(args.year)
        print(f"  {'OK' if ok else 'FAILED'}")
        return

    print(f"\n{'='*60}")
    print(f"  Fetching challenger projections — {args.year}")
    print(f"{'='*60}\n")

    # --- Projection systems ---
    all_proj = []
    for sys_key in args.systems:
        df = fetch_projection(sys_key, args.year)
        if not df.empty:
            out = DATA_DIR / f"fg_proj_{sys_key}_{args.year}.parquet"
            df.to_parquet(out, index=False)
            print(f"    Saved -> {out}")
            all_proj.append(df)
        time.sleep(1.0)   # polite rate-limit

    if all_proj:
        combined = pd.concat(all_proj, ignore_index=True)
        out = DATA_DIR / f"fg_proj_all_{args.year}.parquet"
        combined.to_parquet(out, index=False)
        print(f"\n  Combined saved -> {out} ({len(combined)} rows)")

    # --- Stuff+ / Pitching+ leaderboard ---
    if not args.no_stuff:
        print()
        stuff_df = fetch_stuff_plus(args.year)
        if not stuff_df.empty:
            out = DATA_DIR / f"fg_stuff_plus_{args.year}.parquet"
            stuff_df.to_parquet(out, index=False)
            print(f"    Saved -> {out}")

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
