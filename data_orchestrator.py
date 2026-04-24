"""
data_orchestrator.py — v4.3 Master Data Pipeline.

Single-schedule entry point. Run once daily (recommended: 3:30 PM ET,
after lineups typically post). All steps are idempotent.

Steps:
    1. RECOVER  : Fill any actuals gap via MLB Stats API -> actuals_2026.parquet
    2. PULL     : Today's schedule (with probablePitcher hydration) + lineups
    3. ENRICH   : SP metrics from FM + live ADI (Open-Meteo) + ump synergy + bullpen burn
    4. RESIDUALS: Update model_residuals.csv for yesterday
    5. VALIDATE : Fail loud if any critical source is empty / malformed

v4.3 changes:
    - probablePitcher hydrated from MLB API in pull_today() — no more FM carry-forward
      for starter names (fixes systematic away_starter_name bleed bug)
    - Live ADI: Open-Meteo fetches real-time temp/humidity/pressure per park;
      air_density_rho computed via Buck equation
    - Ump synergy: HP ump joined from ump_features; 1.2x K multiplier for
      wide-zone ump paired with stuff+ starter
    - Bullpen 5d: bullpen_burn_5d added alongside existing 3d window

Outputs:
    data/statcast/actuals_2026.parquet        (updated)
    data/statcast/lineups_today_long.parquet  (refreshed)
    data/orchestrator/daily_context.parquet   (enriched game+SP slate)
    data/logs/model_residuals.csv             (updated)
    data/orchestrator/orchestrator_log.csv    (run history)
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

ACTUALS_FILE  = _ROOT / "data/statcast/actuals_2026.parquet"
LINEUP_LONG   = _ROOT / "data/statcast/lineups_today_long.parquet"
FEATURE_MTX   = _ROOT / "feature_matrix_enriched_v2.parquet"
STADIUM_META  = _ROOT / "config/stadium_metadata.json"
OUT_DIR       = _ROOT / "data/orchestrator"
CONTEXT_FILE  = OUT_DIR / "daily_context.parquet"
LOG_FILE      = OUT_DIR / "orchestrator_log.csv"

TEAMS_URL    = "https://statsapi.mlb.com/api/v1/teams?sportId=1"
SCHEDULE_URL = ("https://statsapi.mlb.com/api/v1/schedule?sportId=1"
                "&date={d}&hydrate=linescore,probablePitcher,officials")

# ── Stadium lat/lon for Open-Meteo weather fetch ─────────────────────────
PARK_COORDS: dict[str, tuple[float, float]] = {
    "ARI": (33.4455, -112.0667), "ATL": (33.8908, -84.4678),
    "BAL": (39.2838, -76.6216),  "BOS": (42.3467, -71.0972),
    "CHC": (41.9484, -87.6553),  "CWS": (41.8299, -87.6338),
    "CIN": (39.0979, -84.5082),  "CLE": (41.4962, -81.6852),
    "COL": (39.7559, -104.9942), "DET": (42.3390, -83.0485),
    "HOU": (29.7573, -95.3555),  "KC":  (39.0517, -94.4803),
    "LAA": (33.8003, -117.8827), "LAD": (34.0739, -118.2400),
    "MIA": (25.7781, -80.2197),  "MIL": (43.0280, -87.9712),
    "MIN": (44.9817, -93.2778),  "NYM": (40.7571, -73.8458),
    "NYY": (40.8296, -73.9262),  "ATH": (37.7516, -122.2005),
    "PHI": (39.9061, -75.1665),  "PIT": (40.4469, -80.0057),
    "SD":  (32.7076, -117.1570), "SF":  (37.7786, -122.3893),
    "SEA": (47.5914, -122.3325), "STL": (38.6226, -90.1928),
    "TB":  (27.7682, -82.6534),  "TEX": (32.7473, -97.0832),
    "TOR": (43.6414, -79.3894),  "WSH": (38.8730, -77.0074),
    "OAK": (37.7516, -122.2005),
}

# ── Shared team-id -> abbr map (loaded once) ────────────────────────────
_TEAM_MAP: dict[int, str] = {}


def _get_team_map() -> dict[int, str]:
    global _TEAM_MAP
    if _TEAM_MAP:
        return _TEAM_MAP
    with urllib.request.urlopen(TEAMS_URL, timeout=15) as r:
        _TEAM_MAP = {int(t["id"]): t.get("abbreviation", "")
                     for t in json.loads(r.read())["teams"]}
    return _TEAM_MAP


# ════════════════════════════════════════════════════════════════════════
# STEP 1 — RECOVER: actuals gap fill
# ════════════════════════════════════════════════════════════════════════
def _fetch_finals_for_date(d: date) -> list[dict]:
    tm = _get_team_map()
    url = SCHEDULE_URL.format(d=d.isoformat())
    with urllib.request.urlopen(url, timeout=15) as r:
        payload = json.loads(r.read())
    rows = []
    for block in payload.get("dates", []):
        for g in block.get("games", []):
            status = g.get("status", {}).get("detailedState", "")
            if status not in ("Final", "Game Over", "Completed Early"):
                continue
            h = g["teams"]["home"]
            a = g["teams"]["away"]
            rows.append({
                "game_pk": g.get("gamePk"),
                "game_date": d.isoformat(),
                "home_team": tm.get(int(h["team"]["id"]), ""),
                "away_team": tm.get(int(a["team"]["id"]), ""),
                "home_score_final": h.get("score"),
                "away_score_final": a.get("score"),
            })
    return [r for r in rows if r["home_score_final"] is not None]


def recover_actuals(today: date) -> int:
    """Fill any gap between last recorded actual and yesterday. Returns rows added."""
    yesterday = today - timedelta(days=1)
    if ACTUALS_FILE.exists():
        existing = pd.read_parquet(ACTUALS_FILE)
        have_dates = set(pd.to_datetime(existing["game_date"]).dt.strftime("%Y-%m-%d"))
    else:
        existing = pd.DataFrame()
        have_dates = set()

    # Walk backward up to 14 days to catch any gap
    season_start = date(today.year, 3, 20)
    fill_dates = []
    d = yesterday
    while d >= season_start:
        if d.isoformat() not in have_dates:
            fill_dates.append(d)
        d -= timedelta(days=1)
        if len(fill_dates) > 14:
            break

    if not fill_dates:
        print(f"  [recover] actuals up-to-date through {yesterday}")
        return 0

    print(f"  [recover] filling {len(fill_dates)} missing date(s): "
          f"{[d.isoformat() for d in sorted(fill_dates)]}")
    new_rows = []
    for d in fill_dates:
        try:
            rows = _fetch_finals_for_date(d)
            print(f"    {d}: {len(rows)} games")
            new_rows.extend(rows)
        except Exception as exc:
            print(f"    {d}: FETCH ERROR {exc}")

    if not new_rows:
        return 0

    new = pd.DataFrame(new_rows).dropna(subset=["home_score_final", "away_score_final"])
    new["home_score_final"] = new["home_score_final"].astype("int16")
    new["away_score_final"] = new["away_score_final"].astype("int16")

    if not existing.empty:
        existing = existing[~existing["game_pk"].isin(new["game_pk"])]
        combined = pd.concat([existing, new], ignore_index=True, sort=False)
    else:
        combined = new

    combined = combined.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    ACTUALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(ACTUALS_FILE, index=False)
    print(f"  [recover] actuals_2026.parquet -> {len(combined)} total rows (+{len(new)})")
    return len(new)


# ════════════════════════════════════════════════════════════════════════
# STEP 2 — PULL: today's schedule + lineups
# ════════════════════════════════════════════════════════════════════════
def pull_today(today: str) -> pd.DataFrame:
    """Fetch schedule with probable pitchers and officials from MLB API.

    Starters come directly from probablePitcher hydration — never from FM
    carry-forward, which caused away_starter_name to bleed between match-ups.
    """
    print(f"  [pull] lineup_pull.py --date {today}")
    try:
        result = subprocess.run(
            [sys.executable, str(_ROOT / "lineup_pull.py"), "--date", today],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"    [warn] lineup_pull exit {result.returncode}: {result.stderr[:200]}")
    except Exception as exc:
        print(f"    [warn] lineup_pull failed: {exc}")

    tm = _get_team_map()
    url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}"
           f"&hydrate=probablePitcher,officials")
    with urllib.request.urlopen(url, timeout=15) as r:
        payload = json.loads(r.read())
    rows = []
    for block in payload.get("dates", []):
        for g in block.get("games", []):
            ht = g["teams"]["home"]
            at = g["teams"]["away"]
            # Probable pitchers from API (fresh, game-specific)
            home_sp = ht.get("probablePitcher", {}).get("fullName", "")
            away_sp = at.get("probablePitcher", {}).get("fullName", "")
            home_sp_id = ht.get("probablePitcher", {}).get("id")
            away_sp_id = at.get("probablePitcher", {}).get("id")
            # HP umpire if assigned (pre-game)
            officials = g.get("officials", [])
            hp_ump = next((o for o in officials
                           if o.get("officialType") == "Home Plate"), {})
            rows.append({
                "game_pk":          g.get("gamePk"),
                "home_team":        tm.get(int(ht["team"]["id"]), ""),
                "away_team":        tm.get(int(at["team"]["id"]), ""),
                "status":           g.get("status", {}).get("detailedState", ""),
                "home_starter_name": home_sp.upper() if home_sp else "",
                "away_starter_name": away_sp.upper() if away_sp else "",
                "home_sp_mlbam_id":  home_sp_id,
                "away_sp_mlbam_id":  away_sp_id,
                "ump_hp_name":       hp_ump.get("official", {}).get("fullName", ""),
                "ump_hp_id":         hp_ump.get("official", {}).get("id"),
            })
    sched = pd.DataFrame(rows)
    n_prob = (sched["home_starter_name"] != "").sum()
    print(f"  [pull] schedule: {len(sched)} games, "
          f"{(sched['status']=='Scheduled').sum()} scheduled, "
          f"{n_prob} probable pitchers posted")
    return sched


# ════════════════════════════════════════════════════════════════════════
# STEP 3 — ENRICH: Sticky SGP metrics from feature_matrix
# ════════════════════════════════════════════════════════════════════════
# NOTE: home_starter_name / away_starter_name are intentionally NOT here.
# They are hydrated live from MLB probablePitcher API in pull_today() and
# must never be overwritten by FM carry-forward (that caused the Woo/ATH bug).
STICKY_SP_COLS = [
    "home_sp_k_pct", "home_sp_bb_pct", "home_sp_whiff_pctl",
    "home_sp_ff_velo", "home_sp_xwoba_against", "home_sp_gb_pct",
    "home_sp_k_pct_10d", "home_sp_k_bb_ratio",
    "away_sp_k_pct", "away_sp_bb_pct", "away_sp_whiff_pctl",
    "away_sp_ff_velo", "away_sp_xwoba_against", "away_sp_gb_pct",
    "away_sp_k_pct_10d", "away_sp_k_bb_ratio",
    # home_sp_whiff_pctl / away_sp_whiff_pctl already above — used for ump synergy
]
STICKY_GAME_COLS = [
    "close_total", "wind_mph", "wind_bearing",
    "home_park_factor",
    "home_bat_k_vs_rhp", "home_bat_k_vs_lhp",
    "away_bat_k_vs_rhp", "away_bat_k_vs_lhp",
    "home_bullpen_xfip", "away_bullpen_xfip",
    "elo_diff",
]


# ════════════════════════════════════════════════════════════════════════
# STEP 3a — Live ADI via Open-Meteo
# ════════════════════════════════════════════════════════════════════════
def _buck_air_density(temp_f: float, rh_pct: float, pressure_hpa: float) -> float:
    """Compute dry-air density (kg/m³) via Buck equation.

    Higher density = more atmospheric drag = pitcher-friendly (Under anchor).
    Lower density (hot, humid, high altitude) = ball carries = Over anchor.
    """
    temp_c = (temp_f - 32.0) * 5.0 / 9.0
    # Saturation vapor pressure (Buck 1981)
    e_sat = 6.1121 * np.exp((18.678 - temp_c / 234.5) * (temp_c / (257.14 + temp_c)))
    e_v = (rh_pct / 100.0) * e_sat          # actual vapor pressure (hPa)
    p_dry = pressure_hpa - e_v               # partial pressure of dry air
    R_d, R_v = 287.058, 461.495             # gas constants J/(kg·K)
    T_k = temp_c + 273.15
    return (p_dry * 100.0) / (R_d * T_k) + (e_v * 100.0) / (R_v * T_k)


def enrich_live_adi(context: pd.DataFrame, game_hour_local: int = 19) -> pd.DataFrame:
    """Fetch real-time weather from Open-Meteo for each home park.

    Populates: temp_f, relative_humidity, surface_pressure_hpa, air_density_rho.
    Falls back to FM carry-forward values if Open-Meteo is unreachable.
    """
    context = context.copy()
    for col in ("temp_f", "relative_humidity", "surface_pressure_hpa", "air_density_rho"):
        if col not in context.columns:
            context[col] = np.nan

    fetched: dict[str, dict] = {}
    for _, row in context.iterrows():
        team = str(row.get("home_team", ""))
        if team in fetched or team not in PARK_COORDS:
            continue
        lat, lon = PARK_COORDS[team]
        url = (f"https://api.open-meteo.com/v1/forecast"
               f"?latitude={lat}&longitude={lon}"
               f"&hourly=temperature_2m,relative_humidity_2m,surface_pressure"
               f"&wind_speed_unit=mph&temperature_unit=fahrenheit"
               f"&timezone=auto&forecast_days=1")
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                w = json.loads(r.read())
            idx = min(game_hour_local, len(w["hourly"]["time"]) - 1)
            t   = float(w["hourly"]["temperature_2m"][idx] or 72.0)
            rh  = float(w["hourly"]["relative_humidity_2m"][idx] or 50.0)
            p   = float(w["hourly"]["surface_pressure"][idx] or 1013.25)
            fetched[team] = {"temp_f": t, "relative_humidity": rh,
                             "surface_pressure_hpa": p,
                             "air_density_rho": _buck_air_density(t, rh, p)}
        except Exception as exc:
            print(f"  [adi] Open-Meteo failed for {team}: {exc}")

    for i, row in context.iterrows():
        team = str(row.get("home_team", ""))
        if team in fetched:
            for col, val in fetched[team].items():
                context.at[i, col] = val

    n = sum(1 for t in context["home_team"] if t in fetched)
    print(f"  [adi] Live weather fetched for {n}/{len(context)} games")
    return context


# ════════════════════════════════════════════════════════════════════════
# STEP 3b — Umpire Synergy join
# WIDE_ZONE_THRESHOLD: ump_k_above_avg > 0.3 → wide zone (more called strikes)
# STUFF_PLUS_THRESHOLD: whiff_pctl > 0.70 → elite swing-and-miss stuff
# Multiplier: 1.2x on K-over implied probability when both conditions met
# ════════════════════════════════════════════════════════════════════════
WIDE_ZONE_THRESHOLD  = 0.30   # ump_k_above_avg > this → wide zone
STUFF_PLUS_THRESHOLD = 0.70   # whiff_pctl > this → stuff+ starter

_UMP_FEATURES_CACHE: pd.DataFrame | None = None


def _load_ump_features() -> pd.DataFrame:
    global _UMP_FEATURES_CACHE
    if _UMP_FEATURES_CACHE is not None:
        return _UMP_FEATURES_CACHE
    frames = []
    for yr in (2023, 2024, 2025):
        p = _ROOT / f"data/statcast/ump_features_{yr}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame(columns=["ump_hp_id", "ump_k_above_avg",
                                     "ump_bb_above_avg", "ump_rpg_above_avg"])
    uf = pd.concat(frames, ignore_index=True)
    # Career average per ump
    _UMP_FEATURES_CACHE = (uf.dropna(subset=["ump_hp_id", "ump_k_above_avg"])
                             .groupby("ump_hp_id")[["ump_k_above_avg",
                                                    "ump_bb_above_avg",
                                                    "ump_rpg_above_avg"]]
                             .mean()
                             .reset_index())
    return _UMP_FEATURES_CACHE


def enrich_ump_synergy(context: pd.DataFrame) -> pd.DataFrame:
    """Join umpire K stats and compute ump_k_synergy_home / _away multipliers.

    ump_k_synergy_home = 1.2 if wide-zone ump AND home SP is stuff+, else 1.0
    ump_k_synergy_away = 1.2 if wide-zone ump AND away SP is stuff+, else 1.0

    Also saves today's ump assignments to umpire_assignments_2026.parquet.
    """
    context = context.copy().reset_index(drop=True)
    uf = _load_ump_features()

    for col in ("ump_k_above_avg", "ump_bb_above_avg", "ump_rpg_above_avg",
                "ump_k_synergy_home", "ump_k_synergy_away"):
        if col not in context.columns:
            context[col] = np.nan if "synergy" not in col else 1.0

    # Populate from today's ump assignments (pulled in pull_today via officials)
    if "ump_hp_id" in context.columns and not uf.empty:
        context["ump_hp_id"] = pd.to_numeric(context["ump_hp_id"], errors="coerce")
        uf["ump_hp_id"] = pd.to_numeric(uf["ump_hp_id"], errors="coerce")
        merged = context.merge(uf, on="ump_hp_id", how="left",
                               suffixes=("", "_uf")).reset_index(drop=True)
        for col in ("ump_k_above_avg", "ump_bb_above_avg", "ump_rpg_above_avg"):
            if f"{col}_uf" in merged.columns:
                context[col] = merged[f"{col}_uf"].values
            elif col in merged.columns:
                context[col] = merged[col].values

    # Save assignments for audit trail
    if "ump_hp_id" in context.columns:
        today_str = str(context.get("orchestrator_date", pd.Series([""])).iloc[0])
        ump_today = context[["game_pk", "ump_hp_id", "ump_hp_name"]].copy()
        ump_today["year"] = pd.Timestamp.now().year
        assign_path = _ROOT / "data/statcast/umpire_assignments_2026.parquet"
        if assign_path.exists():
            existing = pd.read_parquet(assign_path)
            ump_today = pd.concat(
                [existing[~existing["game_pk"].isin(ump_today["game_pk"])],
                 ump_today], ignore_index=True)
        assign_path.parent.mkdir(parents=True, exist_ok=True)
        ump_today.to_parquet(assign_path, index=False)

    # Synergy multipliers — use .values to avoid index alignment issues
    context = context.reset_index(drop=True)
    context["ump_k_synergy_home"] = 1.0
    context["ump_k_synergy_away"] = 1.0
    wide_zone = (context["ump_k_above_avg"].fillna(0).values > WIDE_ZONE_THRESHOLD)
    if "home_sp_whiff_pctl" in context.columns:
        stuff_home = (context["home_sp_whiff_pctl"].fillna(0).values > STUFF_PLUS_THRESHOLD)
        context.loc[wide_zone & stuff_home, "ump_k_synergy_home"] = 1.2
    if "away_sp_whiff_pctl" in context.columns:
        stuff_away = (context["away_sp_whiff_pctl"].fillna(0).values > STUFF_PLUS_THRESHOLD)
        context.loc[wide_zone & stuff_away, "ump_k_synergy_away"] = 1.2

    synergy_games = ((context["ump_k_synergy_home"] > 1.0) |
                     (context["ump_k_synergy_away"] > 1.0)).sum()
    print(f"  [ump]  K synergy 1.2x applied to {synergy_games} game(s); "
          f"{context['ump_k_above_avg'].notna().sum()} umps matched")
    return context


# ════════════════════════════════════════════════════════════════════════
# STEP 3c — Bullpen 5d fatigue window
# ════════════════════════════════════════════════════════════════════════
def enrich_bullpen_5d(context: pd.DataFrame, today: str) -> pd.DataFrame:
    """Add bullpen_burn_5d to context by extending the existing 3d window.

    Reads bullpen_burn_by_game.parquet (has bullpen_burn_3d) and sums
    the 5-day rolling pitch-count window per team.
    """
    bp_path = _ROOT / "data/batter_features/bullpen_burn_by_game.parquet"
    if not bp_path.exists():
        print("  [bp5d] bullpen_burn_by_game.parquet not found — skipping")
        context["bullpen_burn_5d"] = np.nan
        return context

    bp = pd.read_parquet(bp_path)
    bp["game_date"] = pd.to_datetime(bp["game_date"])
    today_dt = pd.Timestamp(today)
    cutoff_5d = today_dt - pd.Timedelta(days=5)

    # Sum hl_pitches over the last 5 days per team
    bp_window = bp[(bp["game_date"] >= cutoff_5d) & (bp["game_date"] < today_dt)]
    burn_5d = (bp_window.groupby("home_team")["total_hl_pitches"]
                        .sum()
                        .reset_index()
                        .rename(columns={"total_hl_pitches": "home_bullpen_burn_5d"}))
    burn_5d_away = burn_5d.rename(columns={
        "home_team": "away_team",
        "home_bullpen_burn_5d": "away_bullpen_burn_5d"
    })

    context = context.copy()
    context = context.merge(burn_5d,      on="home_team", how="left")
    context = context.merge(burn_5d_away, on="away_team", how="left")
    context["home_bullpen_burn_5d"] = context["home_bullpen_burn_5d"].fillna(0)
    context["away_bullpen_burn_5d"] = context["away_bullpen_burn_5d"].fillna(0)

    # Gassed flag: > 350 high-leverage pitches in 5 days
    GASSED_THRESHOLD = 350
    context["home_bp_gassed"] = (context["home_bullpen_burn_5d"] > GASSED_THRESHOLD).astype("int8")
    context["away_bp_gassed"]  = (context["away_bullpen_burn_5d"] > GASSED_THRESHOLD).astype("int8")

    gassed = context["home_bp_gassed"].sum() + context["away_bp_gassed"].sum()
    print(f"  [bp5d] bullpen_burn_5d joined; {gassed} gassed-bullpen flags")
    return context


def enrich_slate(schedule: pd.DataFrame, today: str) -> pd.DataFrame:
    """Join sticky SP + game metrics from FM, then layer in live enrichments."""
    context = schedule.copy()

    if not FEATURE_MTX.exists():
        print("  [enrich] feature_matrix not found; SP metrics will be null")
    else:
        fm = pd.read_parquet(FEATURE_MTX)
        fm["game_date"] = pd.to_datetime(fm["game_date"])
        fm_today = fm[fm["game_date"].dt.strftime("%Y-%m-%d") == today]

        # Only join SP performance stats (no starter names — those come from pull_today)
        want = [c for c in STICKY_SP_COLS + STICKY_GAME_COLS if c in fm.columns]

        if fm_today.empty:
            print(f"  [enrich] no FM rows for {today}; carrying forward SP stats per team")
            fm_recent = (fm.sort_values("game_date")
                           .groupby("home_team", as_index=False)
                           .last()[["home_team"] + want])
            # Away team SP stats: join by away_team using most-recent away rows
            away_want = [c for c in want if c.startswith("away_sp_")]
            home_want = [c for c in want if not c.startswith("away_sp_")]
            fm_home_stats = fm_recent[["home_team"] + home_want]
            fm_away_stats = (fm.sort_values("game_date")
                               .groupby("away_team", as_index=False)
                               .last()[["away_team"] + away_want]
                             if away_want and "away_team" in fm.columns
                             else pd.DataFrame())
            context = context.merge(fm_home_stats, on="home_team", how="left")
            if not fm_away_stats.empty:
                context = context.merge(fm_away_stats, on="away_team", how="left",
                                        suffixes=("", "_away_fm"))
        else:
            fm_slim = fm_today[["game_pk", "home_team"] + want].copy()
            fm_slim = fm_slim.drop_duplicates("home_team", keep="last")
            context = context.merge(fm_slim, on=["game_pk", "home_team"], how="left")
            # Fallback for unmatched games
            null_mask = context[want[0]].isna() if want else pd.Series(False, index=context.index)
            if null_mask.any():
                fm_by_team = fm_slim.drop(columns=["game_pk"], errors="ignore")
                fill = context.loc[null_mask, ["home_team"]].merge(
                    fm_by_team, on="home_team", how="left")
                for col in want:
                    if col in fill.columns:
                        context.loc[null_mask, col] = fill[col].values

        print(f"  [enrich] SP stats joined for {len(context)} games")

    # Preserve starter names from pull_today (do not overwrite with FM)
    # If FM carry-forward accidentally populated them, restore from schedule
    for col in ("home_starter_name", "away_starter_name"):
        if col in schedule.columns and col in context.columns:
            from_api = schedule[col]
            context[col] = np.where(from_api.fillna("") != "",
                                    from_api, context[col].fillna(""))

    # Live enrichments
    context = enrich_live_adi(context)
    context = enrich_ump_synergy(context)
    context = enrich_bullpen_5d(context, today)

    return context


# ════════════════════════════════════════════════════════════════════════
# STEP 4 — RESIDUALS: update thermostat log
# ════════════════════════════════════════════════════════════════════════
def update_residuals(today: date) -> int:
    yesterday = (today - timedelta(days=1)).isoformat()
    print(f"  [residuals] updating for {yesterday}")
    try:
        result = subprocess.run(
            [sys.executable, str(_ROOT / "update_actuals.py"), "--date", yesterday],
            capture_output=True, text=True, timeout=60
        )
        print(f"    {result.stdout.strip()}")
        if result.returncode != 0:
            print(f"    [warn] update_actuals exit {result.returncode}")
        return 0
    except Exception as exc:
        print(f"    [warn] update_actuals failed: {exc}")
        return 0


# ════════════════════════════════════════════════════════════════════════
# STEP 5 — VALIDATE
# ════════════════════════════════════════════════════════════════════════
def validate(schedule: pd.DataFrame, context: pd.DataFrame, today: str) -> list[str]:
    errors = []
    if len(schedule) == 0:
        errors.append("CRITICAL: MLB schedule returned 0 games")
    if len(context) == 0:
        errors.append("CRITICAL: enriched context has 0 rows")
    # Check SP velo populated
    if "home_sp_ff_velo" in context.columns:
        null_velo = context["home_sp_ff_velo"].isna().sum()
        if null_velo == len(context):
            errors.append("WARN: home_sp_ff_velo ALL null — FM carry-forward failed")
    lineup_file = _ROOT / f"data/statcast/lineups_{today}_long.parquet"
    if lineup_file.exists():
        lu = pd.read_parquet(lineup_file)
        if len(lu) == 0:
            errors.append(f"WARN: lineup file for {today} is empty (lineups not posted yet)")
    else:
        errors.append(f"WARN: no lineup file for {today} — lineup_pull may have failed")
    return errors


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════
def main():
    today = date.today()
    today_str = today.isoformat()
    print(f"\n{'='*64}")
    print(f" data_orchestrator.py  [{today_str}]")
    print(f"{'='*64}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_log = {"date": today_str, "status": "running"}

    # Step 1 — recover
    print("\n[1/5] RECOVER actuals ...")
    added = recover_actuals(today)
    run_log["actuals_added"] = added

    # Step 2 — pull
    print("\n[2/5] PULL schedule + lineups ...")
    schedule = pull_today(today_str)
    run_log["games_today"] = len(schedule)

    # Step 3 — enrich
    print("\n[3/5] ENRICH sticky SP metrics ...")
    context = enrich_slate(schedule, today_str)

    # Step 4 — residuals
    print("\n[4/5] RESIDUALS update ...")
    update_residuals(today)

    # Step 5 — validate
    print("\n[5/5] VALIDATE ...")
    errors = validate(schedule, context, today_str)
    if errors:
        for e in errors:
            print(f"  {'[ERROR]' if 'CRITICAL' in e else '[WARN]'} {e}")
        if any("CRITICAL" in e for e in errors):
            run_log["status"] = "failed"
            _append_log(run_log)
            raise SystemExit("CRITICAL validation failure — see above")
    else:
        print("  All checks passed.")
    run_log["status"] = "ok"
    run_log["warnings"] = len([e for e in errors if "WARN" in e])

    # Write enriched context
    context["orchestrator_date"] = today_str
    context.to_parquet(CONTEXT_FILE, index=False)
    print(f"\nWrote daily_context.parquet -> {CONTEXT_FILE} ({len(context)} rows)")

    _append_log(run_log)
    print(f"Run log -> {LOG_FILE}")
    print("\n=== DONE ===")


def _append_log(row: dict):
    df = pd.DataFrame([row])
    if LOG_FILE.exists():
        existing = pd.read_csv(LOG_FILE)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)


if __name__ == "__main__":
    main()
