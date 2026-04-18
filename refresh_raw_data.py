"""
refresh_raw_data.py
===================
Auto-refreshes all six raw CSV files in data/raw/ that were previously
downloaded manually from FanGraphs and Baseball Savant.

Sources used (all free, no API key required):
  fangraphs_batters.csv   ← Baseball Savant batter custom leaderboard
                             (wOBA / xwOBA / K% / BB%) + wRC+ proxy
  fangraphs_pitchers.csv  ← Savant pitcher expected stats + MLB Stats API
                             + computed K/9, BB/9, FIP, xFIP
  savant_batters.csv      ← pybaseball.statcast_batter_expected_stats()
  savant_pitchers.csv     ← Baseball Savant pitcher custom leaderboard
  fangraphs_team_vs_lhp   ← MLB Stats API team batting splits vs LHP
  fangraphs_team_vs_rhp   ← MLB Stats API team batting splits vs RHP

Usage:
  python refresh_raw_data.py              # refresh current year
  python refresh_raw_data.py --year 2025  # backfill a specific year
  python refresh_raw_data.py --all        # refresh 2024 + 2025 + 2026
  python refresh_raw_data.py --target fg_batters  # single file only
"""

import argparse
import io
import time
import warnings
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

try:
    import pybaseball
    pybaseball.cache.enable()
except ImportError:
    pybaseball = None  # savant_batters fallback to direct API

RAW_DIR = Path("data/raw")

# League-average constants used for derived metrics
LG_WOBA       = 0.320
LG_FIP_CONST  = 3.10   # 2026 approximate

MLB_TEAMS = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
    "Athletics": "ATH",
}

SAVANT_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _savant_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, headers=SAVANT_HEADERS, timeout=40)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def _parse_name(combined: str) -> str:
    """'Last, First' → 'First Last'"""
    if "," in str(combined):
        parts = str(combined).split(",", 1)
        return f"{parts[1].strip()} {parts[0].strip()}"
    return str(combined).strip()


def _wrc_proxy(woba_series: pd.Series) -> pd.Series:
    """wRC+ proxy from wOBA: normalized to league average (100)."""
    return ((woba_series / LG_WOBA) * 100).round(1)


def _compute_fip(k, bb, hr, ip, hbp=0) -> float:
    """Standard FIP from counting stats."""
    if ip and ip > 0:
        return round((13 * hr + 3 * (bb + hbp) - 2 * k) / ip + LG_FIP_CONST, 2)
    return float("nan")


def _mlb_stats_api_pitchers(year: int) -> pd.DataFrame:
    """Fetch pitcher counting stats (ERA, IP, K, BB, HR) from MLB Stats API."""
    url = "https://statsapi.mlb.com/api/v1/stats"
    params = {
        "stats": "season", "season": str(year), "group": "pitching",
        "gameType": "R", "limit": "2000", "sportId": "1",
        "fields": "stats,splits,stat,era,inningsPitched,strikeOuts,baseOnBalls,homeRuns,battersFaced,player,fullName,id",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    rows = []
    for item in r.json().get("stats", []):
        for split in item.get("splits", []):
            stat = split.get("stat", {})
            player = split.get("player", {})
            rows.append({
                "player_id": player.get("id"),
                "Name":      player.get("fullName", ""),
                "ERA":       stat.get("era"),
                "IP":        stat.get("inningsPitched"),
                "K":         stat.get("strikeOuts", 0),
                "BB":        stat.get("baseOnBalls", 0),
                "HR":        stat.get("homeRuns", 0),
                "BF":        stat.get("battersFaced", 0),
            })
    df = pd.DataFrame(rows)
    df["IP"]  = pd.to_numeric(df["IP"],  errors="coerce")
    df["ERA"] = pd.to_numeric(df["ERA"], errors="coerce")
    return df


def _mlb_stats_api_team_splits(year: int, side_code: str) -> pd.DataFrame:
    """Fetch per-team batting splits vs LHP ('vl') or RHP ('vr')."""
    url = "https://statsapi.mlb.com/api/v1/stats"
    params = {
        "stats": "statSplits", "season": str(year), "group": "hitting",
        "gameType": "R", "sitCodes": side_code, "playerPool": "All",
        "limit": "100", "sportId": "1",
        "fields": "stats,splits,stat,avg,obp,slg,ops,plateAppearances,strikeOuts,baseOnBalls,team,name",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    rows = []
    for item in r.json().get("stats", []):
        for split in item.get("splits", []):
            stat  = split.get("stat", {})
            tname = split.get("team", {}).get("name", "")
            rows.append({
                "Name": tname,
                "Tm":   MLB_TEAMS.get(tname, ""),
                "Season": year,
                "PA":   stat.get("plateAppearances", 0),
                "AVG":  stat.get("avg", ".000"),
                "OBP":  stat.get("obp", ".000"),
                "SLG":  stat.get("slg", ".000"),
                "OPS":  stat.get("ops", ".000"),
                "K":    stat.get("strikeOuts", 0),
                "BB":   stat.get("baseOnBalls", 0),
            })
    df = pd.DataFrame(rows)
    for col in ["PA", "K", "BB"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in ["AVG", "OBP", "SLG", "OPS"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["wOBA"]  = ((df["OBP"] * 1.8 + df["SLG"]) / (1.8 + 1.0)).clip(0.200, 0.450)
    df["wRC+"]  = _wrc_proxy(df["wOBA"]).clip(50, 160)
    df["K%"]    = (df["K"]  / df["PA"].replace(0, 1)).round(3).clip(0.05, 0.45)
    df["BB%"]   = (df["BB"] / df["PA"].replace(0, 1)).round(3).clip(0.02, 0.25)
    df = df[(df["Tm"] != "") & (df["PA"] >= 20)].copy()
    return df[["Season", "Name", "Tm", "PA", "AVG", "OBP", "SLG", "OPS", "wOBA", "wRC+", "K%", "BB%"]]


# ---------------------------------------------------------------------------
# Refresh functions — one per file
# ---------------------------------------------------------------------------

def refresh_savant_batters(years: list[int]) -> None:
    """savant_batters.csv — batter expected stats (est_ba, est_woba, etc.)."""
    if pybaseball is None:
        print("  [savant_batters] pybaseball not available — skipping")
        return
    frames = []
    for yr in years:
        try:
            df = pybaseball.statcast_batter_expected_stats(yr)
            frames.append(df)
            print(f"  [savant_batters] {yr}: {len(df)} rows")
            time.sleep(1)
        except Exception as e:
            print(f"  [savant_batters] {yr} FAILED: {e}")
    if not frames:
        return
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["player_id", "year"])
    out.to_csv(RAW_DIR / "savant_batters.csv", index=False)
    print(f"  [savant_batters] saved {len(out)} rows → savant_batters.csv")


def refresh_savant_pitchers(years: list[int]) -> None:
    """savant_pitchers.csv — pitcher custom leaderboard (k%, exit_velo, whiff%, etc.)."""
    sv_cols = (
        "k_percent,bb_percent,batting_avg,slg_percent,on_base_percent,"
        "woba,xwoba,exit_velocity_avg,launch_angle_avg,sweet_spot_percent,"
        "barrel_batted_rate,hard_hit_percent,whiff_percent,swing_percent"
    )
    frames = []
    for yr in years:
        try:
            url = (
                f"https://baseballsavant.mlb.com/leaderboard/custom"
                f"?year={yr}&type=pitcher&filter=&min=5"
                f"&selections={sv_cols}&chart=false&x=k_percent&z=pid&csv=true"
            )
            df = _savant_csv(url)
            df["year"] = yr
            frames.append(df)
            print(f"  [savant_pitchers] {yr}: {len(df)} rows")
            time.sleep(1)
        except Exception as e:
            print(f"  [savant_pitchers] {yr} FAILED: {e}")
    if not frames:
        return
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["player_id", "year"])
    out.to_csv(RAW_DIR / "savant_pitchers.csv", index=False)
    print(f"  [savant_pitchers] saved {len(out)} rows → savant_pitchers.csv")


def refresh_fg_batters(years: list[int]) -> None:
    """fangraphs_batters.csv — wOBA/xwOBA/K%/BB% from Savant + wRC+ proxy."""
    sv_cols = "pa,k_percent,bb_percent,batting_avg,slg_percent,on_base_percent,woba,xwoba"
    frames = []
    for yr in years:
        try:
            url = (
                f"https://baseballsavant.mlb.com/leaderboard/custom"
                f"?year={yr}&type=batter&filter=&min=25"
                f"&selections={sv_cols}&chart=false&x=woba&z=pid&csv=true"
            )
            df = _savant_csv(url)
            # Build columns expected by build_lineup_quality.py and score_models.py
            df["Name"]  = df["last_name, first_name"].apply(_parse_name)
            df["year"]  = yr
            df["PA"]    = pd.to_numeric(df.get("pa", 25), errors="coerce").fillna(25).astype(int)
            df["wRC+"]  = _wrc_proxy(pd.to_numeric(df["woba"], errors="coerce"))
            df["K%"]    = pd.to_numeric(df["k_percent"],  errors="coerce") / 100
            df["BB%"]   = pd.to_numeric(df["bb_percent"], errors="coerce") / 100
            df["wOBA"]  = pd.to_numeric(df["woba"],  errors="coerce")
            df["xwOBA"] = pd.to_numeric(df["xwoba"], errors="coerce")
            frames.append(df)
            print(f"  [fg_batters] {yr}: {len(df)} rows")
            time.sleep(1)
        except Exception as e:
            print(f"  [fg_batters] {yr} FAILED: {e}")
    if not frames:
        return
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["player_id", "year"])
    out.to_csv(RAW_DIR / "fangraphs_batters.csv", index=False)
    print(f"  [fg_batters] saved {len(out)} rows → fangraphs_batters.csv")
    print(f"  [fg_batters] NOTE: wRC+ is computed proxy (wOBA/0.320*100), not native FanGraphs value")


def refresh_fg_pitchers(years: list[int]) -> None:
    """fangraphs_pitchers.csv — Savant expected stats + MLB Stats API + computed FIP/K9/BB9."""
    frames = []
    for yr in years:
        try:
            # --- Part 1: Savant pitcher expected stats (xERA, wOBA) ---
            sv_exp = pybaseball.statcast_pitcher_expected_stats(yr) if pybaseball else pd.DataFrame()
            if sv_exp.empty:
                # Fallback to direct API
                url = (
                    f"https://baseballsavant.mlb.com/leaderboard/custom"
                    f"?year={yr}&type=pitcher&filter=&min=5"
                    f"&selections=era,woba,xwoba&chart=false&x=era&z=pid&csv=true"
                )
                sv_exp = _savant_csv(url)
            sv_exp = sv_exp[["player_id", "era", "xera"]].copy()
            sv_exp.columns = ["player_id", "ERA_sv", "xERA"]
            sv_exp["player_id"] = pd.to_numeric(sv_exp["player_id"], errors="coerce")
            time.sleep(1)

            # --- Part 2: MLB Stats API (ERA, IP, K, BB, HR for FIP) ---
            mlb = _mlb_stats_api_pitchers(yr)
            mlb["player_id"] = pd.to_numeric(mlb["player_id"], errors="coerce")
            time.sleep(1)

            # --- Merge ---
            merged = mlb.merge(sv_exp, on="player_id", how="left")
            merged["ERA"] = merged["ERA"].fillna(merged["ERA_sv"])
            merged.drop(columns=["ERA_sv"], inplace=True, errors="ignore")

            # --- Derive FG-compatible columns ---
            merged["IP"]   = pd.to_numeric(merged["IP"], errors="coerce")
            merged["K"]    = pd.to_numeric(merged["K"],  errors="coerce").fillna(0)
            merged["BB"]   = pd.to_numeric(merged["BB"], errors="coerce").fillna(0)
            merged["HR"]   = pd.to_numeric(merged["HR"], errors="coerce").fillna(0)
            merged["BF"]   = pd.to_numeric(merged["BF"], errors="coerce").fillna(1)

            ip_safe = merged["IP"].replace(0, float("nan"))
            merged["K/9"]  = (merged["K"]  / ip_safe * 9).round(2)
            merged["BB/9"] = (merged["BB"] / ip_safe * 9).round(2)
            merged["HR/9"] = (merged["HR"] / ip_safe * 9).round(2)
            merged["FIP"]  = (
                (13 * merged["HR"] + 3 * merged["BB"] - 2 * merged["K"]) / ip_safe + LG_FIP_CONST
            ).round(2)
            # xFIP proxy: FIP but with normalized HR rate (10% of FB, ~35% GB rate)
            fb_est = merged["BF"] * 0.35   # rough FB count estimate
            norm_hr = fb_est * 0.10
            merged["xFIP"] = (
                (13 * norm_hr + 3 * merged["BB"] - 2 * merged["K"]) / ip_safe + LG_FIP_CONST
            ).round(2)
            # LOB% proxy: Tango approximation (3K + BB)/(3K + BB + HR + H_est)
            # Not reliable without H, so set to league average 72%
            merged["LOB%"]  = 0.72
            merged["year"]  = yr

            keep = ["Name", "player_id", "year", "ERA", "xERA", "FIP", "xFIP",
                    "LOB%", "IP", "K/9", "BB/9", "HR/9"]
            merged = merged[[c for c in keep if c in merged.columns]]
            frames.append(merged)
            print(f"  [fg_pitchers] {yr}: {len(merged)} rows")
        except Exception as e:
            print(f"  [fg_pitchers] {yr} FAILED: {e}")

    if not frames:
        return
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["player_id", "year"])
    out.to_csv(RAW_DIR / "fangraphs_pitchers.csv", index=False)
    print(f"  [fg_pitchers] saved {len(out)} rows → fangraphs_pitchers.csv")
    print(f"  [fg_pitchers] NOTE: xFIP=computed proxy; LOB%=72% placeholder")


def refresh_team_splits(years: list[int]) -> None:
    """fangraphs_team_vs_lhp/rhp.csv — MLB Stats API team batting splits."""
    for side_code, fname in [("vl", "fangraphs_team_vs_lhp.csv"), ("vr", "fangraphs_team_vs_rhp.csv")]:
        frames = []
        for yr in years:
            try:
                df = _mlb_stats_api_team_splits(yr, side_code)
                frames.append(df)
                print(f"  [team_splits {side_code}] {yr}: {len(df)} teams")
                time.sleep(0.5)
            except Exception as e:
                print(f"  [team_splits {side_code}] {yr} FAILED: {e}")
        if not frames:
            continue
        out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Tm", "Season"])
        out.to_csv(RAW_DIR / fname, index=False)
        print(f"  [team_splits] saved {len(out)} rows → {fname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

TARGETS = {
    "fg_batters":    refresh_fg_batters,
    "fg_pitchers":   refresh_fg_pitchers,
    "savant_batters": refresh_savant_batters,
    "savant_pitchers": refresh_savant_pitchers,
    "team_splits":   refresh_team_splits,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh raw data CSVs from Savant + MLB Stats API")
    parser.add_argument("--year",   type=int, help="Single year to refresh (default: current season)")
    parser.add_argument("--all",    action="store_true", help="Refresh 2024+2025+2026")
    parser.add_argument("--target", choices=list(TARGETS.keys()), help="Refresh one file only")
    args = parser.parse_args()

    if args.all:
        years = [2024, 2025, 2026]
    elif args.year:
        years = [args.year]
    else:
        from datetime import date
        years = [date.today().year]

    print(f"\n  Refreshing raw data for year(s): {years}\n")

    if args.target:
        TARGETS[args.target](years)
    else:
        refresh_fg_batters(years)
        refresh_fg_pitchers(years)
        refresh_savant_batters(years)
        refresh_savant_pitchers(years)
        refresh_team_splits(years)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
