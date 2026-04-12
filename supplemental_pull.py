"""
supplemental_pull.py
────────────────────
Supplemental MLB data pull for the prediction model pipeline.
Covers FanGraphs batting/pitching stats, Statcast leaderboards, team schedules,
division standings, MLB Stats API game schedule, pitcher handedness, and park factors.

All output files are written to ./statcast_data/ alongside statcast_pull.py output.

Install dependencies:
    pip install pybaseball pandas pyarrow requests beautifulsoup4

Usage:
    python supplemental_pull.py

Expected runtime: 20-40 minutes first run, ~2 min with cache
"""

import warnings
warnings.filterwarnings('ignore')

import time
import json
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

import pandas as pd
import pybaseball
from pybaseball import (
    statcast_batter_expected_stats,
    statcast_pitcher_expected_stats,
    statcast_batter_exitvelo_barrels,
    statcast_pitcher_exitvelo_barrels,
    statcast_batter_percentile_ranks,
    statcast_pitcher_percentile_ranks,
    statcast_sprint_speed,
    statcast_pitcher_pitch_arsenal,
    statcast_pitcher_spin_dir_comp,  # renamed from statcast_pitcher_active_spin in newer pybaseball
)

# ── CONFIG ───────────────────────────────────────────────────────────────────
YEARS = [2023, 2024, 2025, 2026]
OUTPUT_DIR = Path("./statcast_data")

TEAMS = [
    'ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CWS', 'CIN', 'CLE', 'COL', 'DET',
    'HOU', 'KC',  'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'OAK',
    'PHI', 'PIT', 'SD',  'SEA', 'SF',  'STL', 'TB',  'TEX', 'TOR', 'WSH',
]

# Division labels — standings() returns list of DataFrames ordered:
# AL East (0), AL Central (1), AL West (2), NL East (3), NL Central (4), NL West (5)
DIVISION_LABELS = [
    'AL East', 'AL Central', 'AL West',
    'NL East', 'NL Central', 'NL West',
]
# ─────────────────────────────────────────────────────────────────────────────


def _print_shape(label: str, df: pd.DataFrame, date_col: str = None) -> None:
    """Print shape and optional date range after saving."""
    msg = f"         shape: {df.shape}"
    if date_col and date_col in df.columns:
        try:
            dmin = pd.to_datetime(df[date_col]).min()
            dmax = pd.to_datetime(df[date_col]).max()
            msg += f"  |  dates: {dmin.date()} → {dmax.date()}"
        except Exception:
            pass
    print(msg)


def _save(df: pd.DataFrame, path: Path, label: str, date_col: str = None) -> None:
    """Write DataFrame to parquet and print summary."""
    df.to_parquet(path, engine='pyarrow', index=False)
    print(f"    Saved → {path}")
    _print_shape(label, df, date_col)


def _patch_pybaseball_session() -> None:
    """
    FanGraphs returns 403 to pybaseball's default requests session because it
    lacks a browser User-Agent. Patch the session pybaseball uses internally
    so all requests look like a normal Chrome browser.
    """
    import requests
    from pybaseball import cache as pb_cache

    browser_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.fangraphs.com/",
    }

    # pybaseball.datahelpers.postprocessing uses requests.get directly;
    # patch the module-level session if it exists, otherwise patch requests.Session
    try:
        import pybaseball.datahelpers.postprocessing as pp
        if hasattr(pp, "session"):
            pp.session.headers.update(browser_headers)
    except Exception:
        pass

    # Also patch the global requests default session as a broad fallback
    _orig_get = requests.get
    def _patched_get(url, **kwargs):
        headers = kwargs.pop("headers", {})
        headers = {**browser_headers, **headers}
        return _orig_get(url, headers=headers, **kwargs)
    requests.get = _patched_get
    print("  FanGraphs session patched with browser User-Agent")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Enable pybaseball cache — critical for large pulls
    pybaseball.cache.enable()
    print("Cache enabled\n")

    # Patch requests to avoid FanGraphs 403
    _patch_pybaseball_session()

    # ─── BATTING + PITCHING (MLB Stats API) ──────────────────────────────────
    # FanGraphs blocks server-side requests (403). Baseball Reference pybaseball
    # wrapper is broken (list index out of range — page structure changed).
    # MLB Stats API is the same free API used for schedules/lineups — no auth,
    # no scraping, complete player stat coverage.
    # Output filenames kept as batting_fg_*/pitching_fg_* for downstream compat.
    print("=" * 65)
    print("SECTION: Batting + Pitching Stats (MLB Stats API)")
    print("=" * 65)

    import requests as _requests

    def _mlb_stats_api(group: str, year: int) -> pd.DataFrame:
        """
        Fetch all player season stats from the MLB Stats API.
        group = 'hitting' or 'pitching'
        Returns a flat DataFrame with one row per player.
        """
        url = "https://statsapi.mlb.com/api/v1/stats"
        params = {
            "stats": "season",
            "group": group,
            "season": year,
            "playerPool": "All",
            "limit": 5000,
            "hydrate": "person,team",
        }
        resp = _requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        rows = []
        for stat_block in data.get("stats", []):
            for split in stat_block.get("splits", []):
                row = split.get("stat", {}).copy()
                person = split.get("person", {})
                row["player_id"]   = person.get("id")
                row["Name"]        = person.get("fullName", "")
                row["position"]    = split.get("position", {}).get("abbreviation", "")
                team               = split.get("team", {})
                row["Team"]        = team.get("abbreviation", team.get("name", ""))
                row["team_id"]     = team.get("id")
                row["season"]      = year
                rows.append(row)

        return pd.DataFrame(rows)

    for year in YEARS:
        # ── Batting (qual≈50 PA via post-filter) ─────────────────────────────
        out_path = OUTPUT_DIR / f"batting_fg_{year}.parquet"
        if not out_path.exists():
            try:
                df = _mlb_stats_api("hitting", year)
                if df is None or df.empty:
                    print(f"[{year}] WARNING: No batting data returned")
                else:
                    # Minimum plate appearances filter (equiv. to FanGraphs qual=50)
                    pa_col = next((c for c in df.columns
                                   if c.lower() in ("plateappearances", "pa")), None)
                    if pa_col:
                        df = df[pd.to_numeric(df[pa_col], errors="coerce").fillna(0) >= 50].copy()
                    _save(df, out_path, "batting_fg")
                    print(f"[{year}] batting_fg (MLB API) done — {len(df)} rows")
            except Exception as e:
                print(f"[{year}] ERROR batting_fg: {e}")
        else:
            print(f"[{year}] batting_fg already exists (skipping)")

        time.sleep(1)  # be polite to the API

        # ── Pitching starters (≥1 GS early season; ≥50 IP once available) ──────
        out_path = OUTPUT_DIR / f"pitching_fg_{year}.parquet"
        if not out_path.exists():
            try:
                df = _mlb_stats_api("pitching", year)
                if df is None or df.empty:
                    print(f"[{year}] WARNING: No pitching data returned")
                else:
                    ip_col = next((c for c in df.columns
                                   if c.lower() in ("inningspitched", "ip")), None)
                    gs_col = next((c for c in df.columns
                                   if c.lower() in ("gamesstarted", "gs")), None)
                    if ip_col:
                        ip_series = pd.to_numeric(df[ip_col], errors="coerce").fillna(0)
                        # Use 50 IP threshold if enough pitchers qualify; otherwise fall
                        # back to ≥1 GS (covers early season where no one has 50 IP yet)
                        df_50 = df[ip_series >= 50].copy()
                        if len(df_50) >= 30:
                            df_qual = df_50
                            qualifier = "≥50 IP"
                        elif gs_col is not None:
                            gs_series = pd.to_numeric(df[gs_col], errors="coerce").fillna(0)
                            df_qual = df[gs_series >= 1].copy()
                            qualifier = "≥1 GS (early season)"
                        else:
                            df_qual = df.copy()
                            qualifier = "all (no IP/GS col)"
                    elif gs_col is not None:
                        gs_series = pd.to_numeric(df[gs_col], errors="coerce").fillna(0)
                        df_qual = df[gs_series >= 1].copy()
                        qualifier = "≥1 GS"
                    else:
                        df_qual = df.copy()
                        qualifier = "all"
                    _save(df_qual, out_path, "pitching_fg")
                    print(f"[{year}] pitching_fg ({qualifier}) done — {len(df_qual)} rows")
            except Exception as e:
                print(f"[{year}] ERROR pitching_fg: {e}")
        else:
            print(f"[{year}] pitching_fg already exists (skipping)")

        # ── Pitching full + bullpen (all pitchers) ────────────────────────────
        full_path    = OUTPUT_DIR / f"pitching_fg_full_{year}.parquet"
        bullpen_path = OUTPUT_DIR / f"bullpen_fg_{year}.parquet"
        if not full_path.exists():
            try:
                df_full = _mlb_stats_api("pitching", year)
                if df_full is None or df_full.empty:
                    print(f"[{year}] WARNING: No pitching data (full) returned")
                else:
                    _save(df_full, full_path, "pitching_fg_full")
                    # Reliever = fewer than 50% of appearances are starts
                    gs_col = next((c for c in df_full.columns
                                   if c.lower() in ("gamesstarted", "gs")), None)
                    g_col  = next((c for c in df_full.columns
                                   if c.lower() in ("gamesplayed", "gamespitched", "g")), None)
                    if gs_col and g_col:
                        gs = pd.to_numeric(df_full[gs_col], errors="coerce").fillna(0)
                        g  = pd.to_numeric(df_full[g_col],  errors="coerce").fillna(1)
                        df_bp = df_full[gs < g * 0.5].copy()
                    else:
                        df_bp = df_full.copy()
                    _save(df_bp, bullpen_path, "bullpen_fg")
                    print(f"[{year}] pitching_fg_full ({len(df_full)}) + bullpen_fg ({len(df_bp)}) done")
            except Exception as e:
                print(f"[{year}] ERROR pitching_fg_full: {e}")
        else:
            print(f"[{year}] pitching_fg_full already exists (skipping)")

        time.sleep(1)

    # ─── STATCAST BATTER EXPECTED STATS ──────────────────────────────────────
    print()
    print("=" * 65)
    print("SECTION: Statcast Batter Expected Stats (xBA, xSLG, xwOBA)")
    print("=" * 65)
    for year in YEARS:
        out_path = OUTPUT_DIR / f"batter_xstats_{year}.parquet"
        if out_path.exists():
            print(f"[{year}] Already exists → {out_path} (skipping)")
            continue
        try:
            df = statcast_batter_expected_stats(year)
            if df is None or df.empty:
                print(f"[{year}] WARNING: No batter xstats returned.")
                continue
            _save(df, out_path, "batter_xstats")
            print(f"[{year}] Done.")
        except Exception as e:
            print(f"[{year}] ERROR pulling batter_xstats: {e}")

    # ─── STATCAST PITCHER EXPECTED STATS ─────────────────────────────────────
    print()
    print("=" * 65)
    print("SECTION: Statcast Pitcher Expected Stats")
    print("=" * 65)
    for year in YEARS:
        out_path = OUTPUT_DIR / f"pitcher_xstats_{year}.parquet"
        if out_path.exists():
            print(f"[{year}] Already exists → {out_path} (skipping)")
            continue
        try:
            df = statcast_pitcher_expected_stats(year)
            if df is None or df.empty:
                print(f"[{year}] WARNING: No pitcher xstats returned.")
                continue
            _save(df, out_path, "pitcher_xstats")
            print(f"[{year}] Done.")
        except Exception as e:
            print(f"[{year}] ERROR pulling pitcher_xstats: {e}")

    # ─── STATCAST BATTER EXIT VELO / BARRELS ─────────────────────────────────
    print()
    print("=" * 65)
    print("SECTION: Statcast Batter Exit Velo & Barrels")
    print("=" * 65)
    for year in YEARS:
        out_path = OUTPUT_DIR / f"batter_exitvelo_{year}.parquet"
        if out_path.exists():
            print(f"[{year}] Already exists → {out_path} (skipping)")
            continue
        try:
            df = statcast_batter_exitvelo_barrels(year)
            if df is None or df.empty:
                print(f"[{year}] WARNING: No batter exitvelo data returned.")
                continue
            _save(df, out_path, "batter_exitvelo")
            print(f"[{year}] Done.")
        except Exception as e:
            print(f"[{year}] ERROR pulling batter_exitvelo: {e}")

    # ─── STATCAST PITCHER EXIT VELO / BARRELS ────────────────────────────────
    print()
    print("=" * 65)
    print("SECTION: Statcast Pitcher Exit Velo & Barrels Allowed")
    print("=" * 65)
    for year in YEARS:
        out_path = OUTPUT_DIR / f"pitcher_exitvelo_{year}.parquet"
        if out_path.exists():
            print(f"[{year}] Already exists → {out_path} (skipping)")
            continue
        try:
            df = statcast_pitcher_exitvelo_barrels(year)
            if df is None or df.empty:
                print(f"[{year}] WARNING: No pitcher exitvelo data returned.")
                continue
            _save(df, out_path, "pitcher_exitvelo")
            print(f"[{year}] Done.")
        except Exception as e:
            print(f"[{year}] ERROR pulling pitcher_exitvelo: {e}")

    # ─── STATCAST BATTER PERCENTILE RANKS ────────────────────────────────────
    print()
    print("=" * 65)
    print("SECTION: Statcast Batter Percentile Ranks")
    print("=" * 65)
    for year in YEARS:
        out_path = OUTPUT_DIR / f"batter_percentiles_{year}.parquet"
        if out_path.exists():
            print(f"[{year}] Already exists → {out_path} (skipping)")
            continue
        try:
            df = statcast_batter_percentile_ranks(year)
            if df is None or df.empty:
                print(f"[{year}] WARNING: No batter percentile data returned.")
                continue
            _save(df, out_path, "batter_percentiles")
            print(f"[{year}] Done.")
        except Exception as e:
            print(f"[{year}] ERROR pulling batter_percentiles: {e}")

    # ─── STATCAST PITCHER PERCENTILE RANKS ───────────────────────────────────
    print()
    print("=" * 65)
    print("SECTION: Statcast Pitcher Percentile Ranks")
    print("=" * 65)
    for year in YEARS:
        out_path = OUTPUT_DIR / f"pitcher_percentiles_{year}.parquet"
        if out_path.exists():
            print(f"[{year}] Already exists → {out_path} (skipping)")
            continue
        try:
            df = statcast_pitcher_percentile_ranks(year)
            if df is None or df.empty:
                print(f"[{year}] WARNING: No pitcher percentile data returned.")
                continue
            _save(df, out_path, "pitcher_percentiles")
            print(f"[{year}] Done.")
        except Exception as e:
            print(f"[{year}] ERROR pulling pitcher_percentiles: {e}")

    # ─── STATCAST SPRINT SPEED ────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("SECTION: Statcast Sprint Speed")
    print("=" * 65)
    for year in YEARS:
        out_path = OUTPUT_DIR / f"sprint_speed_{year}.parquet"
        if out_path.exists():
            print(f"[{year}] Already exists → {out_path} (skipping)")
            continue
        try:
            df = statcast_sprint_speed(year)
            if df is None or df.empty:
                print(f"[{year}] WARNING: No sprint speed data returned.")
                continue
            _save(df, out_path, "sprint_speed")
            print(f"[{year}] Done.")
        except Exception as e:
            print(f"[{year}] ERROR pulling sprint_speed: {e}")

    # ─── STATCAST PITCH ARSENAL ───────────────────────────────────────────────
    print()
    print("=" * 65)
    print("SECTION: Statcast Pitcher Pitch Arsenal (minP=100)")
    print("=" * 65)
    for year in YEARS:
        out_path = OUTPUT_DIR / f"pitch_arsenal_{year}.parquet"
        if out_path.exists():
            print(f"[{year}] Already exists → {out_path} (skipping)")
            continue
        try:
            df = statcast_pitcher_pitch_arsenal(year, minP=100)
            if df is None or df.empty:
                print(f"[{year}] WARNING: No pitch arsenal data returned.")
                continue
            _save(df, out_path, "pitch_arsenal")
            print(f"[{year}] Done.")
        except Exception as e:
            print(f"[{year}] ERROR pulling pitch_arsenal: {e}")

    # ─── STATCAST PITCHER ACTIVE SPIN ─────────────────────────────────────────
    print()
    print("=" * 65)
    print("SECTION: Statcast Pitcher Active Spin (minP=100)")
    print("=" * 65)
    for year in YEARS:
        out_path = OUTPUT_DIR / f"pitcher_active_spin_{year}.parquet"
        if out_path.exists():
            print(f"[{year}] Already exists → {out_path} (skipping)")
            continue
        try:
            df = statcast_pitcher_spin_dir_comp(year, minP=100)
            if df is None or df.empty:
                print(f"[{year}] WARNING: No active spin data returned.")
                continue
            _save(df, out_path, "pitcher_active_spin")
            print(f"[{year}] Done.")
        except Exception as e:
            print(f"[{year}] ERROR pulling pitcher_active_spin: {e}")

    # ─── LHP/RHP BATTING SPLITS ───────────────────────────────────────────────
    print()
    print("=" * 65)
    print("SECTION: Batter Splits vs LHP / RHP")
    print("=" * 65)
    # TODO: pybaseball's batting_stats() and batting_stats_bref() do NOT natively
    # support pitcher-handedness splits (vs_lhp / vs_rhp) through a 'split' parameter.
    # The FanGraphs splits leaderboard (splits.fangraphs.com) is not exposed via
    # a stable pybaseball API as of the current package version.
    #
    # Options to implement in the future:
    #   1. Use the FanGraphs API directly:
    #      GET https://www.fangraphs.com/api/leaders/splits/...
    #      with split type "vp" (vs pitcher handedness) — requires scraping/reverse-eng.
    #   2. Derive from raw Statcast data (statcast_2023.parquet etc.) by filtering
    #      on 'p_throws' == 'L' or 'R' and aggregating per batter.
    #   3. Use Baseball Reference splits pages via requests + BeautifulSoup.
    #
    # For now, splits are derived from Statcast parquets in the feature-engineering step.
    print("  NOTE: pybaseball does not support LHP/RHP splits natively.")
    print("  Skipping batter_splits_vs_lhp_{year} and batter_splits_vs_rhp_{year}.")
    print("  See TODO comment in script for alternative approaches.")

    # ─── TEAM SCHEDULES & RECORDS (MLB Stats API) ────────────────────────────
    # pybaseball's schedule_and_record() hits Baseball Reference which blocks
    # all server-side requests. Use the MLB Stats API instead — same free API,
    # complete game-by-game results with scores and W/L.
    print()
    print("=" * 65)
    print("SECTION: Team Schedules & Records (MLB Stats API)")
    print("=" * 65)
    import requests as _requests

    for year in YEARS:
        combined_path = OUTPUT_DIR / f"schedule_all_{year}.parquet"
        if combined_path.exists():
            print(f"[{year}] schedule_all already exists (skipping)")
            continue

        try:
            url = (
                f"https://statsapi.mlb.com/api/v1/schedule"
                f"?sportId=1&season={year}&gameType=R"
                f"&hydrate=linescore,decisions,team&limit=2500"
            )
            resp = _requests.get(url, timeout=60)
            resp.raise_for_status()
            raw = resp.json()

            rows = []
            for date_entry in raw.get('dates', []):
                game_date = date_entry.get('date', '')
                for game in date_entry.get('games', []):
                    status = game.get('status', {}).get('abstractGameState', '')
                    teams_node = game.get('teams', {})
                    home = teams_node.get('home', {})
                    away = teams_node.get('away', {})
                    linescore = game.get('linescore', {})
                    home_score = linescore.get('teams', {}).get('home', {}).get('runs')
                    away_score = linescore.get('teams', {}).get('away', {}).get('runs')
                    home_abbrv = home.get('team', {}).get('abbreviation', '')
                    away_abbrv = away.get('team', {}).get('abbreviation', '')

                    base = {
                        'gamePk':     game.get('gamePk'),
                        'game_date':  game_date,
                        'status':     status,
                        'home_team':  home_abbrv,
                        'away_team':  away_abbrv,
                        'home_score': home_score,
                        'away_score': away_score,
                        'season':     year,
                    }

                    # One row per team perspective (home + away)
                    if home_score is not None and away_score is not None:
                        home_win = int(home_score) > int(away_score)
                        for is_home, team_abbrv, opp_abbrv, rs, ra in [
                            (True,  home_abbrv, away_abbrv, home_score, away_score),
                            (False, away_abbrv, home_abbrv, away_score, home_score),
                        ]:
                            rows.append({
                                **base,
                                'team':     team_abbrv,
                                'opponent': opp_abbrv,
                                'home_away': 'Home' if is_home else 'Away',
                                'R':  rs,
                                'RA': ra,
                                'WL': 'W' if (is_home == home_win) else 'L',
                            })
                    else:
                        # Future/postponed game — add once per team, no score
                        for is_home, team_abbrv, opp_abbrv in [
                            (True,  home_abbrv, away_abbrv),
                            (False, away_abbrv, home_abbrv),
                        ]:
                            rows.append({
                                **base,
                                'team':     team_abbrv,
                                'opponent': opp_abbrv,
                                'home_away': 'Home' if is_home else 'Away',
                                'R': None, 'RA': None, 'WL': None,
                            })

            if not rows:
                print(f"[{year}] WARNING: No games returned from API.")
                continue

            df_all = pd.DataFrame(rows)
            _save(df_all, combined_path, "schedule_all", date_col='game_date')
            n_games = df_all['gamePk'].nunique()
            print(f"[{year}] schedule_all done — {n_games} games, {len(df_all)} team-rows")

            # Also save per-team files for any downstream code that expects them
            for team in df_all['team'].dropna().unique():
                team_path = OUTPUT_DIR / f"schedule_{team}_{year}.parquet"
                if not team_path.exists():
                    df_t = df_all[df_all['team'] == team].copy()
                    df_t.to_parquet(team_path, index=False)

        except Exception as e:
            print(f"[{year}] ERROR pulling schedule: {e}")

        time.sleep(1)

    # ─── STANDINGS (MLB Stats API) ────────────────────────────────────────────
    # pybaseball's standings() hits Baseball Reference — now blocked.
    # MLB Stats API has a complete standings endpoint, free and no auth.
    print()
    print("=" * 65)
    print("SECTION: Division Standings (MLB Stats API)")
    print("=" * 65)
    for year in YEARS:
        out_path = OUTPUT_DIR / f"standings_{year}.parquet"
        if out_path.exists():
            print(f"[{year}] Already exists → {out_path} (skipping)")
            continue
        try:
            url = (
                f"https://statsapi.mlb.com/api/v1/standings"
                f"?leagueId=103,104&season={year}&standingsTypes=regularSeason"
                f"&hydrate=team,division,league"
            )
            resp = _requests.get(url, timeout=30)
            resp.raise_for_status()
            raw = resp.json()

            rows = []
            for record in raw.get('records', []):
                division = record.get('division', {})
                div_name = division.get('name', '')
                league   = record.get('league', {}).get('name', '')
                for tr in record.get('teamRecords', []):
                    team = tr.get('team', {})
                    rows.append({
                        'team_id':    team.get('id'),
                        'team_name':  team.get('name'),
                        'team_abbr':  team.get('abbreviation', ''),
                        'division':   div_name,
                        'league':     league,
                        'W':          tr.get('wins'),
                        'L':          tr.get('losses'),
                        'pct':        tr.get('winningPercentage'),
                        'GB':         tr.get('gamesBack'),
                        'season':     year,
                    })

            if not rows:
                print(f"[{year}] WARNING: No standings data returned.")
                continue

            df_stand = pd.DataFrame(rows)
            _save(df_stand, out_path, "standings")
            print(f"[{year}] standings done — {len(df_stand)} teams")

        except Exception as e:
            print(f"[{year}] ERROR pulling standings: {e}")

        time.sleep(1)

    # ─── MLB STATS API GAME SCHEDULE ─────────────────────────────────────────
    print()
    print("=" * 65)
    print("SECTION: MLB Stats API Game Schedule (urllib)")
    print("=" * 65)
    for year in YEARS:
        out_path = OUTPUT_DIR / f"mlb_schedule_{year}.parquet"
        if out_path.exists():
            print(f"[{year}] Already exists → {out_path} (skipping)")
            continue
        url = (
            f"https://statsapi.mlb.com/api/v1/schedule"
            f"?sportId=1&season={year}&gameType=R"
        )
        try:
            with urlopen(url, timeout=30) as resp:
                raw = json.loads(resp.read().decode('utf-8'))

            rows = []
            for date_entry in raw.get('dates', []):
                game_date = date_entry.get('date', '')
                for game in date_entry.get('games', []):
                    teams_node = game.get('teams', {})
                    home = teams_node.get('home', {}).get('team', {})
                    away = teams_node.get('away', {}).get('team', {})
                    venue = game.get('venue', {})
                    status = game.get('status', {}).get('detailedState', '')
                    rows.append({
                        'gamePk':          game.get('gamePk'),
                        'gameDate':        game_date,
                        'home_team_name':  home.get('name'),
                        'home_team_id':    home.get('id'),
                        'away_team_name':  away.get('name'),
                        'away_team_id':    away.get('id'),
                        'venue_id':        venue.get('id'),
                        'venue_name':      venue.get('name'),
                        'status':          status,
                    })

            if not rows:
                print(f"[{year}] WARNING: No games found in API response.")
                continue

            df_sched = pd.DataFrame(rows)
            _save(df_sched, out_path, "mlb_schedule", date_col='gameDate')
            print(f"[{year}] Done.")
        except URLError as e:
            print(f"[{year}] NETWORK ERROR pulling MLB schedule: {e}")
        except Exception as e:
            print(f"[{year}] ERROR pulling MLB schedule: {e}")

    # ─── PITCHER HANDEDNESS ───────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("SECTION: Pitcher Handedness from MLB Stats API")
    print("=" * 65)
    hand_path = OUTPUT_DIR / "pitcher_handedness.parquet"
    if hand_path.exists():
        print(f"pitcher_handedness.parquet already exists (skipping)")
    else:
        # Collect unique pitcher MLBAM IDs from available statcast parquets
        pitcher_ids = set()
        statcast_files = list(OUTPUT_DIR.glob("statcast_*.parquet"))
        if not statcast_files:
            print(
                "WARNING: No statcast_*.parquet files found in output dir. "
                "Run statcast_pull.py first to enable pitcher handedness lookup. "
                "Skipping pitcher_handedness.parquet."
            )
        else:
            print(f"  Loading pitcher IDs from {len(statcast_files)} statcast file(s)...")
            for sc_file in statcast_files:
                try:
                    df_sc = pd.read_parquet(sc_file, columns=['pitcher'])
                    pitcher_ids.update(df_sc['pitcher'].dropna().astype(int).unique().tolist())
                except Exception as e:
                    print(f"  WARNING: Could not read {sc_file.name}: {e}")

            if not pitcher_ids:
                print("  WARNING: No pitcher IDs found in statcast files. Skipping.")
            else:
                print(f"  Found {len(pitcher_ids):,} unique pitcher IDs. Querying MLB API...")
                # MLB Stats API has a limit on personIds per request — chunk at 500
                pitcher_id_list = sorted(pitcher_ids)
                chunk_size = 500
                hand_rows = []

                for chunk_start in range(0, len(pitcher_id_list), chunk_size):
                    chunk = pitcher_id_list[chunk_start: chunk_start + chunk_size]
                    ids_str = ','.join(str(x) for x in chunk)
                    url = (
                        f"https://statsapi.mlb.com/api/v1/people"
                        f"?personIds={ids_str}&fields=people,id,pitchHand"
                    )
                    try:
                        with urlopen(url, timeout=30) as resp:
                            data = json.loads(resp.read().decode('utf-8'))
                        for person in data.get('people', []):
                            hand_rows.append({
                                'mlbam_id':   person.get('id'),
                                'pitch_hand': person.get('pitchHand', {}).get('code'),
                            })
                        time.sleep(0.25)
                    except Exception as e:
                        print(f"  ERROR on chunk starting {chunk_start}: {e}")

                if hand_rows:
                    df_hand = pd.DataFrame(hand_rows).drop_duplicates('mlbam_id')
                    _save(df_hand, hand_path, "pitcher_handedness")
                    print("  pitcher_handedness.parquet done.")
                else:
                    print("  WARNING: No handedness data retrieved.")

    # ─── PARK FACTORS ─────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("SECTION: Park Factors (FanGraphs Guts page via requests + BeautifulSoup)")
    print("=" * 65)
    try:
        import requests
        from bs4 import BeautifulSoup
        _bs4_available = True
    except ImportError:
        _bs4_available = False
        print("WARNING: 'requests' or 'beautifulsoup4' not installed.")
        print("  Run: pip install requests beautifulsoup4")
        print("  Skipping all park_factors pulls.")

    if _bs4_available:
        # Fetch venue list from MLB Stats API (metadata only — no park factors)
        venues_url = "https://statsapi.mlb.com/api/v1/venues?sportId=1"
        try:
            with urlopen(venues_url, timeout=30) as resp:
                venues_data = json.loads(resp.read().decode('utf-8'))
            venue_rows = []
            for v in venues_data.get('venues', []):
                venue_rows.append({
                    'venue_id':   v.get('id'),
                    'venue_name': v.get('name'),
                })
            df_venues = pd.DataFrame(venue_rows)
            venues_path = OUTPUT_DIR / "mlb_venues.parquet"
            if not venues_path.exists():
                _save(df_venues, venues_path, "mlb_venues")
                print("  mlb_venues.parquet saved.")
            else:
                print("  mlb_venues.parquet already exists (skipping).")
        except Exception as e:
            print(f"  WARNING: Could not fetch venue metadata: {e}")

        # Actual park factors from FanGraphs Guts page
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            )
        }

        for year in YEARS:
            pf_path = OUTPUT_DIR / f"park_factors_{year}.parquet"
            if pf_path.exists():
                print(f"[{year}] Already exists → {pf_path} (skipping)")
                continue
            url = (
                f"https://www.fangraphs.com/guts.aspx"
                f"?type=pf&teamid=0&season={year}"
            )
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, 'html.parser')

                # FanGraphs Guts page uses a table with id 'GutsBoard'
                table = soup.find('table', {'id': 'GutsBoard'})
                if table is None:
                    # Fallback: try any table that contains 'Team' and 'PF' headers
                    for t in soup.find_all('table'):
                        headers_row = t.find('tr')
                        if headers_row and 'Team' in headers_row.get_text():
                            table = t
                            break

                if table is None:
                    print(f"[{year}] WARNING: Could not find park factors table on FanGraphs page.")
                    print(f"         FanGraphs may require JavaScript or have changed layout.")
                    print(f"         URL attempted: {url}")
                    continue

                rows = []
                header_cells = [th.get_text(strip=True) for th in table.find('tr').find_all(['th', 'td'])]
                for tr in table.find_all('tr')[1:]:
                    cells = [td.get_text(strip=True) for td in tr.find_all('td')]
                    if cells:
                        rows.append(dict(zip(header_cells, cells)))

                if not rows:
                    print(f"[{year}] WARNING: Park factors table was empty.")
                    continue

                df_pf = pd.DataFrame(rows)
                df_pf['season'] = year

                # Normalize expected column names where possible
                rename_map = {}
                for col in df_pf.columns:
                    clean = col.strip()
                    if clean in ('Basic', 'Basic (5yr)', '5yr'):
                        rename_map[col] = 'Basic_5yr'
                if rename_map:
                    df_pf = df_pf.rename(columns=rename_map)

                _save(df_pf, pf_path, "park_factors")
                print(f"[{year}] Done. Columns: {list(df_pf.columns)}")
                time.sleep(2)  # Polite crawl delay for FanGraphs

            except requests.HTTPError as e:
                print(f"[{year}] HTTP ERROR pulling park factors: {e}")
            except Exception as e:
                print(f"[{year}] ERROR pulling park factors: {e}")

    # ─── DONE ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("supplemental_pull.py complete.")
    print(f"All files saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 65)
    print()
    print("To load any file in a notebook or agent:")
    print("    import pandas as pd")
    print("    df = pd.read_parquet('./statcast_data/batting_fg_2024.parquet')")
    print()
    print("Key output files:")
    print("  batting_fg_{year}.parquet         — FG batting stats (qual=50)")
    print("  pitching_fg_{year}.parquet        — FG pitching stats (qual=50)")
    print("  pitching_fg_full_{year}.parquet   — FG pitching stats (qual=1, all)")
    print("  bullpen_fg_{year}.parquet         — RP filtered from full pitching")
    print("  batter_xstats_{year}.parquet      — Statcast batter xBA/xSLG/xwOBA")
    print("  pitcher_xstats_{year}.parquet     — Statcast pitcher expected stats")
    print("  batter_exitvelo_{year}.parquet    — Batter exit velo & barrels")
    print("  pitcher_exitvelo_{year}.parquet   — Pitcher exit velo allowed")
    print("  batter_percentiles_{year}.parquet — Statcast batter percentile ranks")
    print("  pitcher_percentiles_{year}.parquet— Statcast pitcher percentile ranks")
    print("  sprint_speed_{year}.parquet       — Sprint speed leaderboard")
    print("  pitch_arsenal_{year}.parquet      — Pitcher pitch arsenal")
    print("  pitcher_active_spin_{year}.parquet— Pitcher active spin rates")
    print("  schedule_{TEAM}_{year}.parquet    — Per-team schedule & record")
    print("  schedule_all_{year}.parquet       — All 30 teams combined")
    print("  standings_{year}.parquet          — Division standings w/ label col")
    print("  mlb_schedule_{year}.parquet       — MLB Stats API game schedule")
    print("  pitcher_handedness.parquet        — MLBAM pitcher throw hand (all-time)")
    print("  park_factors_{year}.parquet       — FanGraphs park factors")
    print("  mlb_venues.parquet                — MLB venue metadata")


if __name__ == "__main__":
    main()
