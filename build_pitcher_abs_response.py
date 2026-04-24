"""
build_pitcher_abs_response.py — ABS Challenge Tilt Model (v1.0).

In the 2026 ABS (Automated Ball-Strike) Challenge system, borderline pitches
on the edge of the strike zone (zones 11-14 in the MLB Stats API convention)
can be challenged by hitters or catchers.

This script audits pitch sequences BEFORE and AFTER a challenge event to
classify each pitcher's psychological response:

  COMPOSED — zone% and velocity hold stable post-challenge
  TILTED   — zone% drops OR velocity drops meaningfully after a challenge

Features computed per pitcher (across all challenges seen in the audit window):
  delta_zone_percent_post_abs   — change in fraction of pitches in zones 1-9
  delta_velocity_post_abs       — change in mean velocity
  n_challenges_audited          — sample size
  tilt_score                    — composite score (lower = more tilted)
  tilt_label                    — "Composed" | "Tilted" | "Uncertain"

Game-day application (via get_pitcher_tilt_mult):
  If a "Tilted" pitcher loses an ABS challenge (detected live via
  live_drift_monitor.py hasReview events), the opponent team-total
  probability is boosted by TILT_BOOST_MULT for the next TILT_INNING_WINDOW
  innings.

Data sources
------------
  MLB Stats API game feeds for all dates in the audit window.
  Games are fetched from /api/v1.1/game/{gamePk}/feed/live (Final status).

Output
------
  data/statcast/pitcher_abs_response.parquet  — per-pitcher ABS tilt profile
  data/live/abs_tilt_active_{date}.json       — per-game active tilt warnings

Usage
-----
  python build_pitcher_abs_response.py              # build 2026 season to date
  python build_pitcher_abs_response.py --days 14    # last 14 days only
  python build_pitcher_abs_response.py --force      # re-build from scratch
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT       = Path(__file__).resolve().parent
OUTPUT_DIR  = _ROOT / "data/statcast"
LIVE_DIR    = _ROOT / "data/live"
CONTEXT_PATH = _ROOT / "data/orchestrator/daily_context.parquet"

MLB_SCHEDULE  = "https://statsapi.mlb.com/api/v1/schedule"
MLB_LIVE_FEED = "https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live"

SEASON_START  = "2026-03-20"
WINDOW_PITCHES = 10        # pitches before/after challenge to analyze
MIN_CHALLENGES = 3         # minimum challenges to classify (else "Uncertain")
TILT_BOOST_MULT  = 1.15    # boost to opponent team-total probability when tilted pitcher loses challenge
TILT_INNING_WINDOW = 2     # innings the boost stays active after a tilt event

# Zones 1-9 are inside the strike zone; 11-14 are borderline/outside
ZONE_IN_STRIKE   = set(range(1, 10))
ZONE_BORDERLINE  = {11, 12, 13, 14}   # most likely challenged zones

# Tilt classification thresholds
TILT_ZONE_THRESHOLD  = -0.05   # delta_zone_percent below this = zone avoidance
TILT_VELO_THRESHOLD  = -0.5    # delta_velocity below this = velocity loss
COMPOSE_ZONE_MIN     =  0.02   # must improve zone entry to be Composed


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None, timeout: int = 20) -> dict:
    import urllib.request, urllib.parse
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as exc:
        return {"_error": str(exc)}


# ---------------------------------------------------------------------------
# Game fetching
# ---------------------------------------------------------------------------

def get_final_game_pks(start_date: str, end_date: str) -> list[tuple[int, str]]:
    """Return (gamePk, date_str) tuples for all Final games in the range."""
    result = []
    cur = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    while cur <= end:
        data  = _get(MLB_SCHEDULE, {"sportId": "1", "date": cur.isoformat()})
        games = data.get("dates", [{}])[0].get("games", [])
        for g in games:
            state = g.get("status", {}).get("abstractGameState", "")
            if state == "Final":
                result.append((g["gamePk"], cur.isoformat()))
        cur += timedelta(days=1)
        time.sleep(0.2)
    return result


# ---------------------------------------------------------------------------
# Per-game challenge event extraction
# ---------------------------------------------------------------------------

def extract_challenge_windows(game_pk: int) -> list[dict]:
    """
    Extract pre/post challenge windows from a completed game feed.

    Returns list of dicts, one per challenge event:
      pitcher_id, pitcher_name, game_pk, inning,
      pre_pitches: [{velo, zone, in_zone},...],
      post_pitches: [{velo, zone, in_zone},...],
      challenge_zone, challenge_desc
    """
    data = _get(MLB_LIVE_FEED.format(gamePk=game_pk))
    if "_error" in data:
        return []

    plays    = (data.get("liveData", {})
                    .get("plays", {})
                    .get("allPlays", []))

    # Build flat list of all pitch events with context
    all_pitches: list[dict] = []
    for play in plays:
        inning      = play.get("about", {}).get("inning", 0)
        pitcher     = play.get("matchup", {}).get("pitcher", {})
        pitcher_id  = pitcher.get("id")
        pitcher_name = pitcher.get("fullName", "Unknown")
        for ev in play.get("playEvents", []):
            if not ev.get("isPitch"):
                continue
            pd_data = ev.get("pitchData", {})
            velo    = pd_data.get("startSpeed")
            zone    = pd_data.get("zone")
            all_pitches.append({
                "pitcher_id":   pitcher_id,
                "pitcher_name": pitcher_name,
                "inning":       inning,
                "velo":         float(velo) if velo else None,
                "zone":         zone,
                "in_zone":      zone in ZONE_IN_STRIKE if zone else False,
                "has_review":   bool(ev.get("details", {}).get("hasReview")),
                "description":  ev.get("details", {}).get("description", ""),
            })

    # For each has_review pitch, extract context window
    windows = []
    for idx, pitch in enumerate(all_pitches):
        if not pitch["has_review"]:
            continue

        pid    = pitch["pitcher_id"]
        # Only keep pitches by the same pitcher
        pre    = [p for p in all_pitches[max(0, idx-WINDOW_PITCHES):idx]
                  if p["pitcher_id"] == pid and p["velo"] is not None]
        post   = [p for p in all_pitches[idx+1:idx+1+WINDOW_PITCHES]
                  if p["pitcher_id"] == pid and p["velo"] is not None]

        if len(pre) < 3 or len(post) < 3:
            continue   # insufficient context

        windows.append({
            "game_pk":        game_pk,
            "pitcher_id":     pid,
            "pitcher_name":   pitch["pitcher_name"],
            "inning":         pitch["inning"],
            "challenge_zone": pitch["zone"],
            "challenge_desc": pitch["description"],
            "in_borderline":  pitch["zone"] in ZONE_BORDERLINE if pitch["zone"] else False,
            "pre_pitches":    pre,
            "post_pitches":   post,
        })

    return windows


# ---------------------------------------------------------------------------
# Compute per-challenge delta metrics
# ---------------------------------------------------------------------------

def compute_deltas(window: dict) -> dict:
    """Compute delta_zone_percent and delta_velocity for one challenge window."""
    pre  = window["pre_pitches"]
    post = window["post_pitches"]

    pre_zone_pct  = sum(p["in_zone"] for p in pre)  / len(pre)
    post_zone_pct = sum(p["in_zone"] for p in post) / len(post)

    pre_velo_vals  = [p["velo"] for p in pre  if p["velo"]]
    post_velo_vals = [p["velo"] for p in post if p["velo"]]

    pre_velo  = np.mean(pre_velo_vals)  if pre_velo_vals  else None
    post_velo = np.mean(post_velo_vals) if post_velo_vals else None

    return {
        "delta_zone_percent_post_abs": round(post_zone_pct - pre_zone_pct, 4),
        "delta_velocity_post_abs":     round(post_velo - pre_velo, 4)
                                       if pre_velo and post_velo else None,
        "pre_zone_pct":   round(pre_zone_pct, 4),
        "post_zone_pct":  round(post_zone_pct, 4),
        "pre_velo":       round(float(pre_velo),  2) if pre_velo  else None,
        "post_velo":      round(float(post_velo), 2) if post_velo else None,
    }


# ---------------------------------------------------------------------------
# Aggregate per-pitcher
# ---------------------------------------------------------------------------

def aggregate_pitcher_profiles(all_windows: list[dict]) -> pd.DataFrame:
    """
    Aggregate challenge delta metrics per pitcher.

    Returns one row per pitcher with tilt_label, tilt_score, and
    the mean/median delta features used to classify them.
    """
    if not all_windows:
        return pd.DataFrame()

    rows = []
    for w in all_windows:
        d = compute_deltas(w)
        rows.append({
            "pitcher_id":   w["pitcher_id"],
            "pitcher_name": w["pitcher_name"],
            **d,
            "challenge_zone":  w["challenge_zone"],
            "in_borderline":   w["in_borderline"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    grp = df.groupby(["pitcher_id", "pitcher_name"])

    summary_rows = []
    for (pid, pname), g in grp:
        n        = len(g)
        dz_mean  = g["delta_zone_percent_post_abs"].mean()
        dv_valid = g["delta_velocity_post_abs"].dropna()
        dv_mean  = dv_valid.mean() if len(dv_valid) >= 2 else None

        # Tilt score: lower = more tilted
        # Normalise zone delta [-1, +1] and velo delta [-3, +3]
        z_score = float(np.clip(dz_mean / 0.15, -1, 1))
        v_score = float(np.clip(dv_mean / 1.5, -1, 1)) if dv_mean is not None else 0.0
        tilt_score = round((z_score + v_score) / 2.0, 4)

        if n < MIN_CHALLENGES:
            label = "Uncertain"
        elif (dz_mean <= TILT_ZONE_THRESHOLD or
              (dv_mean is not None and dv_mean <= TILT_VELO_THRESHOLD)):
            label = "Tilted"
        elif dz_mean >= COMPOSE_ZONE_MIN:
            label = "Composed"
        else:
            label = "Uncertain"

        summary_rows.append({
            "pitcher_id":                 int(pid) if pid else None,
            "pitcher_name":               pname,
            "n_challenges_audited":       n,
            "delta_zone_percent_post_abs": round(float(dz_mean), 4),
            "delta_velocity_post_abs":    round(float(dv_mean), 4)
                                          if dv_mean is not None else None,
            "tilt_score":                 tilt_score,
            "tilt_label":                 label,
            "tilt_boost_mult":            TILT_BOOST_MULT if label == "Tilted" else 1.0,
        })

    return pd.DataFrame(summary_rows).sort_values(
        "tilt_score", ascending=True).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------

def build(days: int = 30, force: bool = False) -> pd.DataFrame:
    """Full audit: scan `days` of completed games, build pitcher profiles."""
    out_path = OUTPUT_DIR / "pitcher_abs_response.parquet"

    if out_path.exists() and not force:
        print(f"  [abs_tilt] {out_path.name} exists — use --force to rebuild")
        return pd.read_parquet(out_path)

    end   = date.today() - timedelta(days=1)
    start = max(date.fromisoformat(SEASON_START),
                end - timedelta(days=days - 1))
    print(f"  [abs_tilt] scanning {start} to {end} for ABS challenge events …")

    game_list = get_final_game_pks(start.isoformat(), end.isoformat())
    print(f"  [abs_tilt] {len(game_list)} completed games found")

    all_windows: list[dict] = []
    for i, (gk, gdate) in enumerate(game_list):
        if (i + 1) % 10 == 0:
            print(f"  [abs_tilt] processing game {i+1}/{len(game_list)} …")
        windows = extract_challenge_windows(gk)
        all_windows.extend(windows)
        time.sleep(0.25)   # be polite to free API

    print(f"  [abs_tilt] {len(all_windows)} challenge windows extracted")

    profiles = aggregate_pitcher_profiles(all_windows)
    if profiles.empty:
        print("  [abs_tilt] no profiles built — insufficient challenge data")
        return pd.DataFrame()

    n_tilted   = (profiles["tilt_label"] == "Tilted").sum()
    n_composed = (profiles["tilt_label"] == "Composed").sum()
    print(f"  [abs_tilt] {len(profiles)} pitchers profiled | "
          f"Tilted: {n_tilted} | Composed: {n_composed}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profiles.to_parquet(out_path, index=False)
    print(f"  [abs_tilt] saved -> {out_path.name}")

    return profiles


# ---------------------------------------------------------------------------
# Game-day active tilt warning (called from live_drift_monitor / dashboard)
# ---------------------------------------------------------------------------

_TILT_CACHE: pd.DataFrame | None = None


def load_profiles() -> pd.DataFrame:
    """Load pitcher_abs_response.parquet (lazy, cached)."""
    global _TILT_CACHE
    if _TILT_CACHE is None:
        p = OUTPUT_DIR / "pitcher_abs_response.parquet"
        _TILT_CACHE = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    return _TILT_CACHE


def get_pitcher_tilt_label(pitcher_id: int) -> str:
    """Return tilt_label for a pitcher_id. 'Uncertain' if not profiled."""
    df = load_profiles()
    if df.empty or "pitcher_id" not in df.columns:
        return "Uncertain"
    match = df[df["pitcher_id"] == pitcher_id]
    if match.empty:
        return "Uncertain"
    return str(match.iloc[0]["tilt_label"])


def get_pitcher_tilt_mult(pitcher_id: int) -> float:
    """Return tilt_boost_mult for a pitcher. 1.0 if not Tilted."""
    df = load_profiles()
    if df.empty or "pitcher_id" not in df.columns:
        return 1.0
    match = df[df["pitcher_id"] == pitcher_id]
    if match.empty:
        return 1.0
    return float(match.iloc[0].get("tilt_boost_mult", 1.0))


def emit_active_tilt_warnings(date_str: str,
                               has_review_events: list[dict]) -> list[dict]:
    """
    Given a list of live has_review events (from live_drift_monitor),
    emit active tilt warnings for Tilted pitchers who just lost a challenge.

    Returns list of warning dicts; also writes data/live/abs_tilt_active_{date}.json.
    """
    warnings: list[dict] = []
    for ev in has_review_events:
        pid   = ev.get("pitcher_id")
        label = get_pitcher_tilt_label(pid) if pid else "Uncertain"
        mult  = get_pitcher_tilt_mult(pid) if pid else 1.0

        if label == "Tilted":
            warnings.append({
                "alert_type":           "ABS_TILT",
                "game_pk":              ev.get("game_pk"),
                "pitcher_id":           pid,
                "pitcher_name":         ev.get("pitcher_name", ""),
                "tilt_label":           label,
                "inning":               ev.get("inning"),
                "challenge_zone":       ev.get("zone"),
                "opp_team_total_mult":  mult,
                "active_innings":       TILT_INNING_WINDOW,
                "boost_expires_inning": (ev.get("inning", 0) or 0) + TILT_INNING_WINDOW,
                "timestamp":            ev.get("timestamp", ""),
            })

    if warnings:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        out = LIVE_DIR / f"abs_tilt_active_{date_str}.json"
        existing = []
        if out.exists():
            try:
                existing = json.loads(out.read_text())
            except Exception:
                pass
        out.write_text(json.dumps(existing + warnings, indent=2))

    return warnings


def load_tilt_warnings(date_str: str) -> list[dict]:
    """Load active tilt warnings for today (used by dashboard)."""
    p = LIVE_DIR / f"abs_tilt_active_{date_str}.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ABS Challenge Tilt Model (v1.0)")
    parser.add_argument("--days",  type=int, default=30,
                        help="Days of history to audit (default: 30)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    profiles = build(days=args.days, force=args.force)
    if profiles.empty:
        return

    print("\n  Top 10 most tilted pitchers:")
    tilted = profiles[profiles["tilt_label"] == "Tilted"].head(10)
    for _, r in tilted.iterrows():
        print(f"    {str(r['pitcher_name']):<28}  "
              f"dZone={r['delta_zone_percent_post_abs']:+.3f}  "
              f"dVelo={r['delta_velocity_post_abs'] if r['delta_velocity_post_abs'] else 'n/a':}  "
              f"n={r['n_challenges_audited']}  "
              f"boost={r['tilt_boost_mult']:.2f}x")

    print("\n  Top 10 most composed pitchers:")
    composed = profiles[profiles["tilt_label"] == "Composed"].head(10)
    for _, r in composed.iterrows():
        print(f"    {str(r['pitcher_name']):<28}  "
              f"dZone={r['delta_zone_percent_post_abs']:+.3f}  "
              f"n={r['n_challenges_audited']}")


if __name__ == "__main__":
    main()
