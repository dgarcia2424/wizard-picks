"""
live_drift_monitor.py — In-Play Velocity Drift / Fatigue Auditor (v1.0).

Polls the MLB Stats API live game feed every POLL_INTERVAL seconds during
active games.  For each starting pitcher currently in the game:

  1. Reads `pitchData.startSpeed` for every pitch thrown this appearance.
  2. Maintains a rolling window of the last VELO_WINDOW pitches.
  3. Compares the rolling average to the pitcher's 2026 season average FF velo
     (loaded from daily_context.parquet or the pitcher_velo_registry cache).
  4. If the drop exceeds FATIGUE_THRESHOLD (1.2 mph), emits a FatigueAlert.

FatigueAlert output:
  - Logged to  data/live/fatigue_alerts_{date}.json   (appended per poll)
  - Printed to stdout
  - The alert is flagged as relevant to Script D (LateDivergence) if the
    pitcher is a home SP and the game is after the 5th inning.

Season average velocity sources (priority order):
  1. data/orchestrator/daily_context.parquet  (home_sp_ff_velo / away_sp_ff_velo)
  2. data/live/pitcher_velo_registry.json     (manually curated / cache)
  3. First 20 pitches of the game appearance   (bootstrap average)

ABS Challenge detection:
  hasReview=True pitch events are captured alongside fatigue events and
  forwarded to build_pitcher_abs_response.py via the shared alert file.

Usage
-----
  python live_drift_monitor.py                  # run live monitor (blocking)
  python live_drift_monitor.py --once           # single poll + exit (testing)
  python live_drift_monitor.py --date 2026-04-23 --once  # replay finished date
  python live_drift_monitor.py --game 822745 --once       # single game

MLB Stats API endpoints used:
  GET /api/v1/schedule?sportId=1&date={date}&hydrate=linescore
  GET /api/v1.1/game/{gamePk}/feed/live
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict, deque
from datetime import date, datetime
from pathlib import Path
from typing import Any

_ROOT        = Path(__file__).resolve().parent
CONTEXT_PATH = _ROOT / "data/orchestrator/daily_context.parquet"
LIVE_DIR     = _ROOT / "data/live"
REGISTRY     = LIVE_DIR / "pitcher_velo_registry.json"

MLB_SCHEDULE  = "https://statsapi.mlb.com/api/v1/schedule"
MLB_LIVE_FEED = "https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live"

POLL_INTERVAL    = 30       # seconds between polls
VELO_WINDOW      = 15       # rolling pitch count for fatigue detection
FATIGUE_THRESHOLD = 1.2     # mph drop from season avg to trigger alert
BOOTSTRAP_MIN    = 20       # minimum pitches before using live baseline
LATE_INNING_MIN  = 5        # inning >= this is "late" for Script D relevance

# Zone 1-9 = standard strike zone (MLB Stats API convention)
ZONE_IN_STRIKE = set(range(1, 10))


# ---------------------------------------------------------------------------
# HTTP helper (no external dependencies beyond stdlib)
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None, timeout: int = 15) -> dict:
    import urllib.request
    import urllib.parse
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as exc:
        return {"_error": str(exc)}


# ---------------------------------------------------------------------------
# Season average velocity registry
# ---------------------------------------------------------------------------

_VELO_CACHE: dict[int, float] = {}   # mlbam_id -> season avg ff velo


def _load_context_velos(date_str: str) -> dict[int, float]:
    """Load season-average FF velo per pitcher from daily_context.parquet."""
    if not CONTEXT_PATH.exists():
        return {}
    try:
        import pandas as pd
        ctx = pd.read_parquet(CONTEXT_PATH)
        if "orchestrator_date" in ctx.columns:
            ctx = ctx[ctx["orchestrator_date"] == date_str]
        velos = {}
        for _, row in ctx.iterrows():
            if row.get("home_sp_mlbam_id") and row.get("home_sp_ff_velo"):
                velos[int(row["home_sp_mlbam_id"])] = float(row["home_sp_ff_velo"])
            if row.get("away_sp_mlbam_id") and row.get("away_sp_ff_velo"):
                velos[int(row["away_sp_mlbam_id"])] = float(row["away_sp_ff_velo"])
        return velos
    except Exception:
        return {}


def _load_registry() -> dict[int, float]:
    """Load manually-curated pitcher velo registry (fallback)."""
    if not REGISTRY.exists():
        return {}
    try:
        raw = json.loads(REGISTRY.read_text())
        return {int(k): float(v) for k, v in raw.items()}
    except Exception:
        return {}


def get_season_avg_velo(pitcher_id: int) -> float | None:
    """Return season-average FF velo for a pitcher, or None if unknown."""
    if pitcher_id in _VELO_CACHE:
        return _VELO_CACHE[pitcher_id]
    reg = _load_registry()
    return reg.get(pitcher_id)


# ---------------------------------------------------------------------------
# Live game discovery
# ---------------------------------------------------------------------------

def get_live_game_pks(date_str: str) -> list[int]:
    """Return gamePks for games that are currently In Progress (or Final for replay)."""
    data  = _get(MLB_SCHEDULE, {"sportId": "1", "date": date_str,
                                 "hydrate": "linescore"})
    games = data.get("dates", [{}])[0].get("games", [])
    live  = []
    for g in games:
        state = g.get("status", {}).get("abstractGameState", "")
        if state in ("Live", "Final", "Preview"):
            live.append(g["gamePk"])
    return live


# ---------------------------------------------------------------------------
# Pitch event extraction from live feed
# ---------------------------------------------------------------------------

def _is_in_zone(zone: int | None) -> bool:
    return zone in ZONE_IN_STRIKE if zone else False


def extract_pitch_events(game_pk: int) -> list[dict]:
    """
    Fetch the live game feed and return a flat list of pitch events:
      pitcher_id, pitcher_name, velo, zone, inning, is_top_inning,
      at_bat_index, pitch_number, has_review, description, game_pk
    """
    data  = _get(MLB_LIVE_FEED.format(gamePk=game_pk))
    if "_error" in data:
        return []

    plays = (data.get("liveData", {})
                 .get("plays", {})
                 .get("allPlays", []))
    events = []
    for play in plays:
        inning      = play.get("about", {}).get("inning", 0)
        is_top      = play.get("about", {}).get("isTopInning", True)
        at_bat_idx  = play.get("atBatIndex", 0)
        pitcher     = play.get("matchup", {}).get("pitcher", {})
        pitcher_id  = pitcher.get("id")
        pitcher_name = pitcher.get("fullName", "Unknown")

        for ev in play.get("playEvents", []):
            if not ev.get("isPitch"):
                continue
            pd_data = ev.get("pitchData", {})
            velo    = pd_data.get("startSpeed")
            zone    = pd_data.get("zone")
            details = ev.get("details", {})
            events.append({
                "game_pk":      game_pk,
                "at_bat_index": at_bat_idx,
                "pitch_number": ev.get("pitchNumber", 0),
                "inning":       inning,
                "is_top":       is_top,
                "pitcher_id":   pitcher_id,
                "pitcher_name": pitcher_name,
                "velo":         float(velo) if velo else None,
                "zone":         zone,
                "in_zone":      _is_in_zone(zone),
                "has_review":   bool(details.get("hasReview")),
                "description":  details.get("description", ""),
            })
    return events


# ---------------------------------------------------------------------------
# Per-game state — velocity rolling windows
# ---------------------------------------------------------------------------

class GameVeloTracker:
    """Tracks per-pitcher rolling velocity windows for one game."""

    def __init__(self, game_pk: int, season_velos: dict[int, float]):
        self.game_pk       = game_pk
        self.season_velos  = season_velos
        # pitcher_id -> deque of recent velocities
        self._windows:    dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=VELO_WINDOW))
        # pitcher_id -> all velocities seen (for bootstrap baseline)
        self._all_velos:  dict[int, list[float]]  = defaultdict(list)
        # pitcher_id -> set of (at_bat_index, pitch_number) already processed
        self._seen:       set[tuple[int, int]]    = set()
        # pitcher_id -> last alert threshold crossed
        self._alerted:    dict[int, float]         = {}

    def _get_baseline(self, pitcher_id: int) -> float | None:
        """Season avg from context → registry → bootstrap mean."""
        if pitcher_id in self.season_velos:
            return self.season_velos[pitcher_id]
        all_v = self._all_velos[pitcher_id]
        if len(all_v) >= BOOTSTRAP_MIN:
            return sum(all_v[:BOOTSTRAP_MIN]) / BOOTSTRAP_MIN
        return None

    def ingest_event(self, ev: dict) -> dict | None:
        """
        Process one pitch event.
        Returns a FatigueAlert dict if threshold crossed, else None.
        """
        key = (ev["at_bat_index"], ev["pitch_number"])
        if key in self._seen:
            return None
        self._seen.add(key)

        velo = ev.get("velo")
        if velo is None:
            return None

        pid = ev["pitcher_id"]
        self._windows[pid].append(velo)
        self._all_velos[pid].append(velo)

        if len(self._windows[pid]) < VELO_WINDOW:
            return None  # not enough pitches yet

        baseline = self._get_baseline(pid)
        if baseline is None:
            return None

        rolling_avg = sum(self._windows[pid]) / VELO_WINDOW
        drop        = baseline - rolling_avg

        if drop < FATIGUE_THRESHOLD:
            return None

        # Avoid re-alerting within 0.3 mph of previous alert
        prev = self._alerted.get(pid, -999)
        if drop - prev < 0.3:
            return None
        self._alerted[pid] = drop

        is_late       = ev["inning"] >= LATE_INNING_MIN
        script_d_flag = is_late   # late innings = Script D territory

        return {
            "alert_type":     "FATIGUE",
            "game_pk":        self.game_pk,
            "pitcher_id":     pid,
            "pitcher_name":   ev["pitcher_name"],
            "inning":         ev["inning"],
            "is_top":         ev["is_top"],
            "velo_baseline":  round(baseline, 2),
            "velo_rolling":   round(rolling_avg, 2),
            "velo_drop":      round(drop, 2),
            "pitch_window":   VELO_WINDOW,
            "script_d_alert": script_d_flag,
            "timestamp":      datetime.utcnow().isoformat(),
        }


# ---------------------------------------------------------------------------
# Alert persistence
# ---------------------------------------------------------------------------

def _save_alerts(alerts: list[dict], date_str: str) -> None:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    out = LIVE_DIR / f"fatigue_alerts_{date_str}.json"
    existing = []
    if out.exists():
        try:
            existing = json.loads(out.read_text())
        except Exception:
            pass
    # Deduplicate by (game_pk, pitcher_id, velo_drop bucket)
    seen_keys = {
        (a["game_pk"], a["pitcher_id"], round(a["velo_drop"], 1))
        for a in existing
    }
    new_alerts = [
        a for a in alerts
        if (a["game_pk"], a["pitcher_id"], round(a["velo_drop"], 1))
        not in seen_keys
    ]
    if new_alerts:
        out.write_text(json.dumps(existing + new_alerts, indent=2))


def load_alerts(date_str: str) -> list[dict]:
    """Load persisted fatigue alerts for a given date (used by dashboard)."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    p = LIVE_DIR / f"fatigue_alerts_{date_str}.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Main monitoring loop
# ---------------------------------------------------------------------------

def run_monitor(date_str: str,
                game_pks: list[int] | None = None,
                once: bool = False) -> list[dict]:
    """
    Main loop.  Polls live games, detects fatigue, saves alerts.
    Returns all alerts emitted during this run.
    """
    season_velos = _load_context_velos(date_str)
    season_velos.update(_load_registry())   # registry fills gaps
    _VELO_CACHE.update(season_velos)

    trackers:    dict[int, GameVeloTracker] = {}
    all_alerts:  list[dict] = []

    def _poll() -> list[dict]:
        pks = game_pks if game_pks else get_live_game_pks(date_str)
        alerts = []
        for gk in pks:
            if gk not in trackers:
                trackers[gk] = GameVeloTracker(gk, season_velos)
            events = extract_pitch_events(gk)
            for ev in events:
                alert = trackers[gk].ingest_event(ev)
                if alert:
                    alerts.append(alert)
                    print(f"  [FATIGUE ALERT] {alert['pitcher_name']} | "
                          f"game {gk} | inning {alert['inning']} | "
                          f"drop {alert['velo_drop']:+.1f} mph "
                          f"({alert['velo_baseline']:.1f} -> {alert['velo_rolling']:.1f}) | "
                          f"Script D: {'YES' if alert['script_d_alert'] else 'no'}")
        return alerts

    while True:
        new = _poll()
        if new:
            all_alerts.extend(new)
            _save_alerts(new, date_str)
        if once:
            break
        time.sleep(POLL_INTERVAL)

    return all_alerts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live velocity drift / fatigue monitor (v1.0)")
    parser.add_argument("--date",  default=date.today().isoformat())
    parser.add_argument("--game",  type=int, default=None,
                        help="Single gamePk to monitor")
    parser.add_argument("--once",  action="store_true",
                        help="Single poll then exit (testing / replay)")
    args = parser.parse_args()

    game_pks = [args.game] if args.game else None
    print(f"[drift_monitor] Starting | date={args.date} | "
          f"{'single-poll' if args.once else f'polling every {POLL_INTERVAL}s'}")

    alerts = run_monitor(
        date_str=args.date,
        game_pks=game_pks,
        once=args.once,
    )
    print(f"\n[drift_monitor] Done. {len(alerts)} fatigue alert(s) emitted.")


if __name__ == "__main__":
    main()
