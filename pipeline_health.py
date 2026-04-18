"""
pipeline_health.py
==================
Checks the health/freshness of every data artifact the pipeline depends on.
Writes pipeline_status.json and optionally uploads to Supabase.

Called by the scheduler after each job; read by the dashboard.

Usage:
  python pipeline_health.py              # check and print status
  python pipeline_health.py --quiet      # no verbose output, just write JSON
  python pipeline_health.py --upload     # check + upload to Supabase
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# .env loader (same pattern as run_today.py / supabase_upload.py)
# ---------------------------------------------------------------------------

def _load_dotenv():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

_load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent

# ANSI colour codes
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"

# Artifact definitions: (name, path, max_age_hours, critical, description)
# Paths containing {date} are substituted with today's date.
ARTIFACT_DEFS = [
    ("lineups",          "data/statcast/lineups_today.parquet",            4,   True,  "Today's starting lineups"),
    ("lineups_long",     "data/statcast/lineups_today_long.parquet",       4,   False, "Batting order for lineup quality"),
    ("odds_current",     "data/statcast/odds_current_{date_us}.parquet",   4,   True,  "Today's ML/RL/total odds"),
    ("k_props",          "data/statcast/k_props_{date}.parquet",           6,   False, "Pitcher K prop lines"),
    ("pitcher_profiles", "data/statcast/pitcher_profiles_2026.parquet",    36,  True,  "Pitcher xwOBA/K%/IP profiles"),
    ("pitcher_10d",      "data/statcast/pitcher_10d_2026.parquet",         36,  False, "Pitcher trailing-10d K%/xwOBA"),
    ("team_stats",       "data/statcast/team_stats_2026.parquet",          36,  True,  "Team batting/bullpen stats (incl 10d)"),
    ("lineup_quality",   "data/statcast/lineup_quality_today.parquet",     4,   False, "Lineup wRC+ scores"),
    ("daily_card",       "daily_card.csv",                                 14,  True,  "Today's model predictions"),
    ("backtest",         "backtest_2026_results.csv",                      26,  False, "Season backtest tracker"),
    ("statcast_2026",    "data/statcast/statcast_2026.parquet",            50,  False, "2026 Statcast pitch data"),
    ("fangraphs",        "data/raw/fangraphs_batters.csv",                 168, False, "FanGraphs batter wRC+ data"),
]

# Next-scheduled-run times (ET, 24-hour) — consolidated 11am architecture
_SCHEDULE = {
    "run_all":  (11, 0),   # full pipeline: lineups → odds → picks → upload
}

ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_et() -> datetime.datetime:
    return datetime.datetime.now(tz=ET)


def _fmt_time(hour: int, minute: int) -> str:
    """Format hour/minute as 'H:MM ET' without zero-padding (cross-platform)."""
    return f"{hour}:{minute:02d} ET"


def _next_occurrence_et(hour: int, minute: int, now_et: datetime.datetime) -> str:
    """Return 'H:MM ET' string for the next occurrence of the given ET time."""
    target = now_et.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now_et >= target:
        target += datetime.timedelta(days=1)
    return _fmt_time(target.hour, target.minute)


def _next_30min_slot(now_et: datetime.datetime) -> str:
    """Return the next 30-minute boundary (on the hour or :30) after now."""
    minute = now_et.minute
    if minute < 30:
        next_minute = 30
        next_hour = now_et.hour
    else:
        next_minute = 0
        next_hour = now_et.hour + 1

    # Handle midnight rollover
    if next_hour >= 24:
        next_hour = 0
    target = now_et.replace(hour=next_hour, minute=next_minute, second=0, microsecond=0)
    if target <= now_et:
        target += datetime.timedelta(hours=1)
    return _fmt_time(target.hour, target.minute)


def _check_artifact(name: str, rel_path: str, max_age_hours: float,
                    critical: bool, description: str,
                    date_str: str) -> dict:
    """Return the status dict for a single artifact."""
    date_us   = date_str.replace("-", "_")   # 2026-04-12 → 2026_04_12
    path_str  = rel_path.replace("{date}", date_str).replace("{date_us}", date_us)
    full_path = BASE_DIR / path_str

    result = {
        "status":        "missing",
        "path":          path_str,
        "age_hours":     None,
        "critical":      critical,
        "description":   description,
        "last_modified": None,
    }

    try:
        if not full_path.exists():
            return result

        mtime = full_path.stat().st_mtime
        mtime_dt = datetime.datetime.fromtimestamp(mtime)
        age_hours = (datetime.datetime.now() - mtime_dt).total_seconds() / 3600.0

        result["age_hours"]     = round(age_hours, 2)
        result["last_modified"] = mtime_dt.strftime("%Y-%m-%dT%H:%M:%S")

        if age_hours < max_age_hours:
            result["status"] = "ok"
        else:
            result["status"] = "stale"

    except Exception as exc:
        # Treat any I/O error as missing
        result["status"] = "missing"
        result["error"]  = str(exc)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_health(date_str: str = None, verbose: bool = True) -> dict:
    """Run all artifact checks and return the full status dict."""
    if date_str is None:
        date_str = datetime.date.today().isoformat()

    now_local = datetime.datetime.now()
    now_et    = _now_et()

    artifacts = {}
    missing_critical = []
    missing_optional = []
    warnings_list    = []

    for (name, rel_path, max_age_hours, critical, description) in ARTIFACT_DEFS:
        info = _check_artifact(name, rel_path, max_age_hours, critical, description, date_str)
        artifacts[name] = info

        if info["status"] in ("missing", "stale"):
            if critical:
                missing_critical.append(name)
            else:
                missing_optional.append(name)

            # Human-readable warning message
            if info["status"] == "missing":
                if name == "k_props":
                    warnings_list.append(
                        "k_props: not yet available (props post ~2h before game time)"
                    )
                else:
                    warnings_list.append(f"{name}: file missing ({info['path']})")
            else:  # stale
                age = info["age_hours"]
                warnings_list.append(
                    f"{name}: stale ({age:.1f}h old, max {max_age_hours}h)"
                )

    # Overall status
    if missing_critical:
        overall = "critical"
    elif missing_optional:
        overall = "warning"
    else:
        overall = "ok"

    # picks_ready: daily_card exists and was written today (age < 14h)
    picks_ready = (
        artifacts["daily_card"]["status"] == "ok"
        and artifacts["daily_card"]["age_hours"] is not None
        and artifacts["daily_card"]["age_hours"] < 14
    )

    # Next scheduled runs
    next_scheduled_runs = {
        label: _next_occurrence_et(*t, now_et)
        for label, t in _SCHEDULE.items()
    }

    status = {
        "generated_at":          now_local.strftime("%Y-%m-%dT%H:%M:%S"),
        "date":                  date_str,
        "overall":               overall,
        "picks_ready":           picks_ready,
        "artifacts":             artifacts,
        "next_scheduled_runs":   next_scheduled_runs,
        "missing_critical":      missing_critical,
        "missing_optional":      missing_optional,
        "warnings":              warnings_list,
    }

    if verbose:
        _print_status(status)

    return status


def write_status(status: dict) -> None:
    """Write pipeline_status.json to the project root."""
    out_path = BASE_DIR / "pipeline_status.json"
    try:
        with open(out_path, "w") as f:
            json.dump(status, f, indent=2)
    except Exception as exc:
        print(f"[ERROR] Could not write pipeline_status.json: {exc}", file=sys.stderr)


def upload_status(status: dict) -> None:
    """Upload status to Supabase wizard_pipeline_health table (no-op if not configured)."""
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        return  # Not configured — silently skip

    try:
        from supabase import create_client
        client = create_client(url, key)

        row = {
            "date":             status["date"],
            "generated_at":     status["generated_at"],
            "overall":          status["overall"],
            "picks_ready":      status["picks_ready"],
            "missing_critical": json.dumps(status["missing_critical"]),
            "missing_optional": json.dumps(status["missing_optional"]),
            "warnings":         json.dumps(status["warnings"]),
            "artifacts_json":   json.dumps(status["artifacts"]),
        }

        # Upsert on date
        client.table("wizard_pipeline_health").upsert(row, on_conflict="date").execute()
        print("  [Supabase] wizard_pipeline_health updated.")

    except ImportError:
        print("[WARN] supabase package not installed — skipping upload.", file=sys.stderr)
    except Exception as exc:
        print(f"[ERROR] Supabase upload failed: {exc}", file=sys.stderr)


def get_status(date_str: str = None) -> dict:
    """Load pipeline_status.json if fresh (<5 min old), else re-run check_health."""
    status_path = BASE_DIR / "pipeline_status.json"
    try:
        if status_path.exists():
            mtime = status_path.stat().st_mtime
            age_seconds = (datetime.datetime.now().timestamp() - mtime)
            if age_seconds < 300:  # 5 minutes
                with open(status_path) as f:
                    return json.load(f)
    except Exception:
        pass  # Fall through to a fresh check

    status = check_health(date_str=date_str, verbose=False)
    write_status(status)
    return status


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _colour(text: str, status: str) -> str:
    if status == "ok":
        return f"{_GREEN}{text}{_RESET}"
    if status == "warning":
        return f"{_YELLOW}{text}{_RESET}"
    return f"{_RED}{text}{_RESET}"


def _status_icon(status: str) -> str:
    return {"ok": "OK   ", "stale": "STALE", "missing": "MISS "}.get(status, status.upper())


def _print_status(status: dict) -> None:
    overall  = status["overall"]
    gen_at   = status["generated_at"]
    date_str = status["date"]

    # Header
    overall_colour = {"ok": _GREEN, "warning": _YELLOW, "critical": _RED}.get(overall, _RESET)
    print()
    print(f"{_BOLD}Pipeline Health — {date_str}{_RESET}  (generated {gen_at})")
    print(f"Overall: {overall_colour}{_BOLD}{overall.upper()}{_RESET}   "
          f"Picks ready: {'yes' if status['picks_ready'] else 'no'}")
    print()

    # Artifact table
    col_w = 18
    print(f"  {'Artifact':<{col_w}}  {'Status':<7}  {'Age':>6}  {'Crit':<5}  Description")
    print(f"  {'-'*col_w}  {'-'*7}  {'-'*6}  {'-'*5}  {'-'*30}")

    for name, info in status["artifacts"].items():
        st   = info["status"]
        age  = f"{info['age_hours']:.1f}h" if info["age_hours"] is not None else "—"
        crit = "YES" if info["critical"] else "no"
        icon = _status_icon(st)

        colour = _GREEN if st == "ok" else (_YELLOW if not info["critical"] else _RED)
        print(f"  {name:<{col_w}}  {colour}{icon}{_RESET}  {age:>6}  {crit:<5}  {info['description']}")

    # Next runs
    print()
    print(f"{_BOLD}Next scheduled runs (ET):{_RESET}")
    for label, t in status["next_scheduled_runs"].items():
        print(f"  {label:<18}  {t}")

    # Warnings
    if status["warnings"]:
        print()
        print(f"{_YELLOW}{_BOLD}Warnings:{_RESET}")
        for w in status["warnings"]:
            print(f"  {_YELLOW}•{_RESET} {w}")

    print()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Check pipeline artifact health and write pipeline_status.json"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output; just write the JSON file",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="After checking, upload status to Supabase",
    )
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Override today's date (for testing)",
    )
    args = parser.parse_args()

    verbose = not args.quiet

    status = check_health(date_str=args.date, verbose=verbose)
    write_status(status)

    if args.upload:
        if verbose:
            print("Uploading to Supabase...")
        upload_status(status)

    # Exit code: 0 = ok or warning (picks still ready), 2 = critical (missing files)
    # "warning" exits 0 so the scheduler doesn't log a spurious FAILED entry for
    # stale-but-non-critical artifacts like statcast or fangraphs refreshes.
    exit_codes = {"ok": 0, "warning": 0, "critical": 2}
    sys.exit(exit_codes.get(status["overall"], 2))


if __name__ == "__main__":
    main()
