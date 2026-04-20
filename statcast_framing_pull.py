"""
statcast_framing_pull.py
========================
Pull official catcher framing data from Baseball Savant's leaderboard API.

Replaces the shadow-zone proxy in enrich_feature_matrix_v2.py with the
official framing run values Statcast publishes (Extra Strikes × Run Value).

Source:
  https://baseballsavant.mlb.com/leaderboard/catcher-framing
  API endpoint: /statcast_search/csv?...&type=details  (too slow for bulk)
  Leaderboard endpoint: /api/leaderboard/sprint_speed  (wrong)
  Actual endpoint used: /leaderboard/catcher-framing CSV download

Fallback:
  If Savant is unavailable, falls back to pybaseball.statcast_catcher_framing()
  (requires pybaseball>=2.2.0).

Outputs (one file per year):
  data/statcast/catcher_framing_{year}.parquet
  Columns: team, season, framing_runs, n_innings, n_catchers

The enrich_feature_matrix_v2.py script checks for these files and uses them
in preference to the shadow-zone proxy when available.

Usage:
  python statcast_framing_pull.py                        # 2023-2026
  python statcast_framing_pull.py --years 2023 2024      # specific years
  python statcast_framing_pull.py --year 2025 --force    # re-pull even if cached
"""

import argparse
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

OUTPUT_DIR = Path("data/statcast")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_YEARS = [2023, 2024, 2025, 2026]

# Minimum innings caught to include a catcher
MIN_INNINGS = 50

TEAM_NAME_MAP = {
    "Angels": "LAA", "Astros": "HOU", "Athletics": "ATH", "Blue Jays": "TOR",
    "Braves": "ATL", "Brewers": "MIL", "Cardinals": "STL", "Cubs": "CHC",
    "Diamondbacks": "AZ", "Dodgers": "LAD", "Giants": "SF", "Guardians": "CLE",
    "Mariners": "SEA", "Marlins": "MIA", "Mets": "NYM", "Nationals": "WSH",
    "Orioles": "BAL", "Padres": "SD", "Phillies": "PHI", "Pirates": "PIT",
    "Rangers": "TEX", "Rays": "TB", "Red Sox": "BOS", "Reds": "CIN",
    "Rockies": "COL", "Royals": "KC", "Tigers": "DET", "Twins": "MIN",
    "White Sox": "CWS", "Yankees": "NYY",
    # Full names
    "Los Angeles Angels": "LAA", "Houston Astros": "HOU", "Oakland Athletics": "ATH",
    "Sacramento Athletics": "ATH", "Toronto Blue Jays": "TOR", "Atlanta Braves": "ATL",
    "Milwaukee Brewers": "MIL", "St. Louis Cardinals": "STL", "Chicago Cubs": "CHC",
    "Arizona Diamondbacks": "AZ", "Los Angeles Dodgers": "LAD", "San Francisco Giants": "SF",
    "Cleveland Guardians": "CLE", "Seattle Mariners": "SEA", "Miami Marlins": "MIA",
    "New York Mets": "NYM", "Washington Nationals": "WSH", "Baltimore Orioles": "BAL",
    "San Diego Padres": "SD", "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "Texas Rangers": "TEX", "Tampa Bay Rays": "TB", "Boston Red Sox": "BOS",
    "Cincinnati Reds": "CIN", "Colorado Rockies": "COL", "Kansas City Royals": "KC",
    "Detroit Tigers": "DET", "Minnesota Twins": "MIN", "Chicago White Sox": "CWS",
    "New York Yankees": "NYY",
    # Abbreviation pass-through
    "LAA": "LAA", "HOU": "HOU", "ATH": "ATH", "OAK": "ATH", "TOR": "TOR",
    "ATL": "ATL", "MIL": "MIL", "STL": "STL", "CHC": "CHC", "ARI": "AZ",
    "AZ": "AZ", "LAD": "LAD", "SF": "SF", "CLE": "CLE", "SEA": "SEA",
    "MIA": "MIA", "NYM": "NYM", "WSH": "WSH", "BAL": "BAL", "SD": "SD",
    "PHI": "PHI", "PIT": "PIT", "TEX": "TEX", "TB": "TB", "BOS": "BOS",
    "CIN": "CIN", "COL": "COL", "KC": "KC", "DET": "DET", "MIN": "MIN",
    "CWS": "CWS", "NYY": "NYY",
}


def _norm_team(raw) -> str | None:
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    return TEAM_NAME_MAP.get(s) or TEAM_NAME_MAP.get(s.title())


# ---------------------------------------------------------------------------
# Savant leaderboard pull
# ---------------------------------------------------------------------------

def _pull_savant_framing(year: int) -> pd.DataFrame | None:
    """
    Pull catcher framing leaderboard from Baseball Savant.

    Endpoint returns CSV with columns including:
      last_name, first_name, player_id, team_id, team, innings,
      runs_extra_strikes, strikes_above_average, ...
    """
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/catcher-framing"
        f"?year={year}&team=&min={MIN_INNINGS}&type=catcher&sort=4&sortDir=desc"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (research/data pipeline)",
        "Accept": "text/html,application/xhtml+xml",
    }

    # The leaderboard page returns HTML — we need the JSON/CSV data endpoint
    # Baseball Savant also exposes a statcast search CSV endpoint for framing
    csv_url = (
        f"https://baseballsavant.mlb.com/leaderboard/catcher-framing"
        f"?year={year}&team=&min={MIN_INNINGS}&type=catcher&csv=true"
    )

    try:
        print(f"  Fetching Savant framing {year} ...", end="", flush=True)
        resp = requests.get(csv_url, headers=headers, timeout=30)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "csv" in content_type or "text" in content_type or resp.text.startswith("last_name"):
            df = pd.read_csv(io.StringIO(resp.text))
            print(f" {len(df)} catchers")
            return df
        else:
            print(f" [non-CSV response, len={len(resp.text)}]")
            return None
    except Exception as e:
        print(f" ERROR: {e}")
        return None


def _pull_pybaseball_framing(year: int) -> pd.DataFrame | None:
    """Fallback: use pybaseball.statcast_catcher_framing()."""
    try:
        import pybaseball as pyb
        print(f"  pybaseball framing {year} ...", end="", flush=True)
        df = pyb.statcast_catcher_framing(year, min_inn=MIN_INNINGS)
        if df is not None and len(df) > 0:
            print(f" {len(df)} catchers")
            return df
        print(" empty")
        return None
    except ImportError:
        print("  pybaseball not installed — pip install pybaseball")
        return None
    except Exception as e:
        print(f"  pybaseball ERROR: {e}")
        return None


def _build_catcher_team_map(year: int) -> dict[int, str]:
    """
    Map catcher player IDs → team abbrev for a given year
    using the local statcast parquet (fielder_2 = catcher ID).
    Falls back to MLB Stats API for any unresolved IDs.
    """
    sc_path = OUTPUT_DIR / f"statcast_{year}.parquet"
    team_map: dict[int, str] = {}

    if sc_path.exists():
        try:
            sc = pd.read_parquet(
                sc_path, engine="pyarrow",
                columns=["fielder_2", "home_team", "away_team", "inning_topbot"]
            )
            sc["fielder_2"] = pd.to_numeric(sc["fielder_2"], errors="coerce")
            sc = sc.dropna(subset=["fielder_2"])
            sc["fielder_2"] = sc["fielder_2"].astype(int)

            # Fielding team: Top = home pitching/catching; Bot = away pitching/catching
            sc["catcher_team"] = np.where(
                sc["inning_topbot"] == "Top", sc["home_team"], sc["away_team"])

            # Plurality team per catcher
            for cid, grp in sc.groupby("fielder_2"):
                mode_team = grp["catcher_team"].mode()
                if len(mode_team) > 0 and pd.notna(mode_team.iloc[0]):
                    team_map[int(cid)] = str(mode_team.iloc[0])
        except Exception as e:
            print(f"    [WARN] statcast team map failed: {e}")

    return team_map


def _parse_framing_df(raw: pd.DataFrame, year: int) -> pd.DataFrame | None:
    """
    Normalize raw framing DataFrame (from either Savant CSV or pybaseball)
    into the standard schema:
      player_id, player_name, team, season, framing_runs, n_pitches
    """
    if raw is None or len(raw) == 0:
        return None

    df = raw.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    print(f"    Columns: {list(df.columns[:12])}")

    # --- Locate framing runs column ---
    # Savant catcher-framing CSV: 'rv_tot' (run value total)
    # pybaseball: 'framing_runs' or 'runs_above_average'
    # Other known aliases: 'runs_extra_strikes', 'strike_rate_runs'
    framing_col = None
    for candidate in ["rv_tot", "runs_extra_strikes", "framing_runs",
                       "runs_above_average", "strike_rate_runs", "total_runs"]:
        if candidate in df.columns:
            framing_col = candidate
            break

    if framing_col is None:
        print(f"    [WARN] No framing runs column found in {list(df.columns)}")
        return None

    # --- Locate pitch count column ---
    pitches_col = None
    for candidate in ["pitches", "innings", "inn", "innings_caught", "total_innings"]:
        if candidate in df.columns:
            pitches_col = candidate
            break

    # --- Locate player_id ---
    id_col = None
    for candidate in ["id", "player_id", "catcher_id"]:
        if candidate in df.columns:
            id_col = candidate
            break

    # --- Locate team column (optional — Savant endpoint omits it) ---
    team_col = None
    for candidate in ["team_name_alt", "team", "team_abbrev"]:
        if candidate in df.columns:
            team_col = candidate
            break

    # --- Build normalized frame ---
    out = pd.DataFrame()
    out["player_id"]    = pd.to_numeric(df[id_col],      errors="coerce") if id_col     else np.nan
    out["framing_runs"] = pd.to_numeric(df[framing_col], errors="coerce")
    out["n_pitches"]    = pd.to_numeric(df[pitches_col], errors="coerce") if pitches_col else np.nan
    out["season"]       = year

    # Name
    if "last_name" in df.columns and "first_name" in df.columns:
        out["player_name"] = df["last_name"].str.strip() + ", " + df["first_name"].str.strip()
    elif "name" in df.columns:
        out["player_name"] = df["name"].astype(str).str.strip()
    else:
        out["player_name"] = "Unknown"

    # Team — use inline column if present, else look up from statcast
    if team_col:
        out["team"] = df[team_col].astype(str).apply(_norm_team)
    else:
        # Build map from local statcast data
        catcher_team_map = _build_catcher_team_map(year)
        if catcher_team_map:
            out["team"] = out["player_id"].map(
                lambda pid: catcher_team_map.get(int(pid)) if pd.notna(pid) else None
            )
            mapped = out["team"].notna().sum()
            print(f"    Team lookup via statcast: {mapped}/{len(out)} catchers mapped")
        else:
            out["team"] = None
            print("    [WARN] No statcast data for team mapping — team will be null")

    out = out.dropna(subset=["framing_runs"])
    print(f"    Parsed: {len(out)} catcher-rows, framing range "
          f"[{out['framing_runs'].min():.1f}, {out['framing_runs'].max():.1f}]")
    return out


def _aggregate_to_team(catcher_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Aggregate individual catcher framing runs to team level.
    Uses innings-weighted sum (catchers share time; team total = sum).
    """
    if catcher_df is None or len(catcher_df) == 0:
        return pd.DataFrame()

    valid = catcher_df.dropna(subset=["team", "framing_runs"]).copy()
    if len(valid) == 0:
        print(f"    [WARN] No valid team mappings for {year}")
        return pd.DataFrame()

    agg_dict = {"framing_runs": ("framing_runs", "sum")}
    if "n_innings" in valid.columns and valid["n_innings"].notna().any():
        agg_dict["n_innings"] = ("n_innings", "sum")
    elif "n_pitches" in valid.columns and valid["n_pitches"].notna().any():
        agg_dict["n_pitches"] = ("n_pitches", "sum")
    if "player_id" in valid.columns:
        agg_dict["n_catchers"] = ("player_id", "nunique")

    team_agg = valid.groupby("team").agg(**agg_dict).reset_index()
    team_agg["season"] = year
    return team_agg


# ---------------------------------------------------------------------------
# Per-year orchestration
# ---------------------------------------------------------------------------

def pull_year(year: int, force: bool = False) -> pd.DataFrame:
    """
    Pull, parse, and save catcher framing for one year.
    Returns team-level DataFrame.
    """
    out_path = OUTPUT_DIR / f"catcher_framing_{year}.parquet"

    if out_path.exists() and not force:
        df = pd.read_parquet(out_path)
        print(f"  {year}: cached ({len(df)} teams) → {out_path.name}")
        return df

    # Try Savant CSV first, fall back to pybaseball
    raw = _pull_savant_framing(year)
    if raw is None or len(raw) == 0:
        print(f"  Savant failed for {year} — trying pybaseball ...")
        raw = _pull_pybaseball_framing(year)

    if raw is None or len(raw) == 0:
        print(f"  [ERROR] Could not retrieve framing data for {year}")
        return pd.DataFrame()

    parsed  = _parse_framing_df(raw, year)
    team_df = _aggregate_to_team(parsed, year)

    if len(team_df) == 0:
        print(f"  [ERROR] No team-level data produced for {year}")
        return pd.DataFrame()

    team_df.to_parquet(out_path, engine="pyarrow", index=False)
    print(f"  {year}: {len(team_df)} teams | "
          f"framing range [{team_df['framing_runs'].min():.1f}, "
          f"{team_df['framing_runs'].max():.1f}] -> {out_path.name}")
    return team_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pull official catcher framing data from Baseball Savant"
    )
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--year",  type=int, default=None,
                        help="Single year override")
    parser.add_argument("--force", action="store_true",
                        help="Re-pull even if cached parquet exists")
    args = parser.parse_args()

    years = [args.year] if args.year else args.years

    print("=" * 60)
    print("  statcast_framing_pull.py — Official Catcher Framing")
    print(f"  Years: {years}  |  Min innings: {MIN_INNINGS}")
    print("=" * 60)

    all_frames = []
    for year in years:
        result = pull_year(year, force=args.force)
        if len(result) > 0:
            all_frames.append(result)
        time.sleep(1.5)   # be polite to Savant

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        out_all = OUTPUT_DIR / "catcher_framing_all.parquet"
        combined.to_parquet(out_all, engine="pyarrow", index=False)
        print(f"\n  Combined → {out_all}  ({len(combined)} team-season rows)")

        print("\n  Sample:")
        show_cols = [c for c in ["team","season","framing_runs","n_pitches","n_innings","n_catchers"]
                     if c in combined.columns]
        print(combined.sort_values("framing_runs", ascending=False)[show_cols]
                      .head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("  Next: run enrich_feature_matrix_v2.py to use official framing")
    print("=" * 60)


if __name__ == "__main__":
    main()
