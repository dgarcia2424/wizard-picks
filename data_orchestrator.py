"""
data_orchestrator.py — v4.0 Master Data Pipeline.

Single-schedule entry point. Run once daily (recommended: 3:30 PM ET,
after lineups typically post). All steps are idempotent.

Steps:
    1. RECOVER  : Fill any actuals gap via MLB Stats API -> actuals_2026.parquet
    2. PULL     : Today's schedule + lineups (lineup_pull.py) + weather stub
    3. ENRICH   : Sticky SGP metrics for each game SP (from feature_matrix)
    4. RESIDUALS: Update model_residuals.csv for yesterday
    5. VALIDATE : Fail loud if any critical source is empty / malformed

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
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={d}&hydrate=linescore"

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
    """Fetch schedule and run lineup_pull. Returns schedule DataFrame."""
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
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}"
    with urllib.request.urlopen(url, timeout=15) as r:
        payload = json.loads(r.read())
    rows = []
    for block in payload.get("dates", []):
        for g in block.get("games", []):
            rows.append({
                "game_pk": g.get("gamePk"),
                "home_team": tm.get(int(g["teams"]["home"]["team"]["id"]), ""),
                "away_team": tm.get(int(g["teams"]["away"]["team"]["id"]), ""),
                "status": g.get("status", {}).get("detailedState", ""),
            })
    sched = pd.DataFrame(rows)
    print(f"  [pull] schedule: {len(sched)} games, "
          f"{(sched['status']=='Scheduled').sum()} scheduled")
    return sched


# ════════════════════════════════════════════════════════════════════════
# STEP 3 — ENRICH: Sticky SGP metrics from feature_matrix
# ════════════════════════════════════════════════════════════════════════
STICKY_SP_COLS = [
    "home_sp_k_pct", "home_sp_bb_pct", "home_sp_whiff_pctl",
    "home_sp_ff_velo", "home_sp_xwoba_against", "home_sp_gb_pct",
    "home_sp_k_pct_10d", "home_sp_k_bb_ratio",
    "away_sp_k_pct", "away_sp_bb_pct", "away_sp_whiff_pctl",
    "away_sp_ff_velo", "away_sp_xwoba_against", "away_sp_gb_pct",
    "away_sp_k_pct_10d", "away_sp_k_bb_ratio",
    "home_starter_name", "away_starter_name",
]
STICKY_GAME_COLS = [
    "close_total", "wind_mph", "wind_bearing", "temp_f",
    "ump_k_above_avg", "home_park_factor",
    "home_bat_k_vs_rhp", "home_bat_k_vs_lhp",
    "away_bat_k_vs_rhp", "away_bat_k_vs_lhp",
    "home_bullpen_xfip", "away_bullpen_xfip",
    "elo_diff",
]

# Batter sticky metrics not in game-grain FM — these require the batter
# feature matrix and are joined at score time, not here.
# Flagged: Zone-Contact%, Hard-Hit%, Sprint Speed per batter not in FM.

STICKY_BATTER_MISSING = [
    "zone_contact_pct",  # not in feature_matrix_enriched_v2
    "hard_hit_pct",      # not in feature_matrix_enriched_v2
    "sprint_speed",      # not in feature_matrix_enriched_v2
]


def enrich_slate(schedule: pd.DataFrame, today: str) -> pd.DataFrame:
    """Join sticky SP + game metrics from the most recent FM rows."""
    context = schedule.copy()

    if not FEATURE_MTX.exists():
        print("  [enrich] feature_matrix not found; SP metrics will be null")
        return context

    fm = pd.read_parquet(FEATURE_MTX)
    fm["game_date"] = pd.to_datetime(fm["game_date"])
    fm_today = fm[fm["game_date"].dt.strftime("%Y-%m-%d") == today]

    want = [c for c in STICKY_SP_COLS + STICKY_GAME_COLS if c in fm.columns]

    if fm_today.empty:
        # Carry-forward: most recent per home_team for non-time-varying stats
        print(f"  [enrich] no FM rows for {today}; carrying forward most-recent per team")
        fm_recent = (fm.sort_values("game_date")
                       .groupby("home_team", as_index=False)
                       .last()[["home_team"] + want])
        context = context.merge(fm_recent, on="home_team", how="left")
    else:
        fm_slim = fm_today[["game_pk", "home_team"] + want].copy()
        fm_slim = fm_slim.drop_duplicates("home_team", keep="last")
        context = context.merge(fm_slim, on=["game_pk", "home_team"], how="left")
    # fallback: match by home_team when game_pk merge didn't cover all rows
    if want and "home_team" in context.columns:
        null_mask = context[want[0]].isna()
        if null_mask.any():
            fm_by_team = fm_slim.drop(columns=["game_pk"], errors="ignore")
            fill = context.loc[null_mask, ["home_team"]].merge(
                fm_by_team, on="home_team", how="left")
            for col in want:
                if col in fill.columns:
                    context.loc[null_mask, col] = fill[col].values

    # Umpire zone factor from FM
    if "ump_k_above_avg" not in context.columns:
        context["ump_k_above_avg"] = np.nan

    # Flag batter sticky metrics as not available
    for col in STICKY_BATTER_MISSING:
        context[col] = np.nan  # stub; requires batter-grain external source

    print(f"  [enrich] SP metrics joined for {len(context)} games")
    if STICKY_BATTER_MISSING:
        print(f"  [enrich] GAPS (not in local data): {STICKY_BATTER_MISSING}")
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
