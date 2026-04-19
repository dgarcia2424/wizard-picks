"""
run_daily_scheduler.py
======================
Continuously running scheduler that orchestrates the daily MLB pipeline.

Usage:
    python run_daily_scheduler.py                       # start scheduler (runs forever)
    python run_daily_scheduler.py --run-now morning     # run morning_run immediately and exit
    python run_daily_scheduler.py --run-now picks       # run picks_run immediately and exit
    python run_daily_scheduler.py --run-now night       # run night_run immediately and exit
    python run_daily_scheduler.py --run-now k_props     # run k_props_retry immediately and exit
    python run_daily_scheduler.py --dry-run             # print schedule and exit

Schedule (all times ET):
    06:00  MORNING_RUN    — lineup_pull.py + odds_current_pull.py + pipeline_health.py
    08:30  PICKS_RUN      — build_pitcher_profile.py + build_team_stats_2026.py +
                            run_today.py --csv + supabase_upload.py + pipeline_health.py
    Every 30 min 10:00-18:30  K_PROPS_RETRY — odds_current_pull.py (if stale/missing)
    23:00  NIGHT_RUN      — backtest_2026.py + backtest_mc_2026.py --rebuild +
                            supabase_upload.py + pipeline_health.py
"""

import argparse
import logging
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------
try:
    import schedule
except ImportError:
    print("Missing dependency. Install: pip install schedule")
    sys.exit(1)

try:
    from zoneinfo import ZoneInfo
except ImportError:
    try:
        from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]
    except ImportError:
        print("Missing dependency. Install: pip install backports.zoneinfo")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/scheduler.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("wizard.scheduler")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ET = ZoneInfo("America/New_York")

# Window during which K props retries are allowed (ET)
K_PROPS_START_ET = (10, 0)   # 10:00 ET
K_PROPS_END_ET   = (18, 30)  # 18:30 ET

K_PROPS_MAX_AGE_HOURS = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log_header(title: str) -> None:
    """Emit a prominent section header to the log."""
    bar = "=" * 60
    log.info(bar)
    log.info("  %s", title)
    log.info(bar)


def run_step(script: str, label: str = "") -> tuple[int, float]:
    """
    Run a pipeline script as a subprocess, streaming its output to the log.

    Parameters
    ----------
    script:
        Space-separated command string, e.g. ``"run_today.py --csv"``.
        The Python interpreter is automatically prepended.
    label:
        Short tag used in log output.

    Returns
    -------
    (returncode, elapsed_seconds)
    """
    parts = script.split()
    tag = label or parts[0]
    log.info(">>> Starting step [%s]: %s", tag, " ".join(parts))
    start = time.time()
    try:
        proc = subprocess.Popen(
            [sys.executable, "-X", "utf8"] + parts,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        for line in proc.stdout:
            log.info("  [%s] %s", tag, line.rstrip())
        proc.wait()
        elapsed = time.time() - start
        if proc.returncode == 0:
            log.info("<<< [%s] finished OK (%.1fs)", tag, elapsed)
        else:
            log.error("<<< [%s] FAILED with returncode=%d (%.1fs)", tag, proc.returncode, elapsed)
        return proc.returncode, elapsed
    except Exception as exc:
        elapsed = time.time() - start
        log.exception("<<< [%s] raised an exception after %.1fs: %s", tag, elapsed, exc)
        return 1, elapsed


def _now_et() -> datetime:
    """Return the current moment as an ET-aware datetime."""
    return datetime.now(tz=ET)


def schedule_et(time_str: str, job_fn, label: str) -> None:
    """
    Register a daily job that fires at a fixed ET clock time.

    The ET time is converted to local wall-clock time before being handed
    to the ``schedule`` library so the job fires correctly regardless of
    the machine's timezone.
    """
    # Build a reference ET datetime (date is arbitrary; only H:M matters)
    et_naive = datetime.strptime(time_str, "%H:%M").replace(
        year=2026, month=1, day=1, tzinfo=ET
    )
    local_time = et_naive.astimezone().strftime("%H:%M")
    schedule.every().day.at(local_time).do(job_fn).tag(label)
    log.info("Scheduled %-14s at %s ET  (local: %s)", label, time_str, local_time)


# ---------------------------------------------------------------------------
# Pipeline jobs
# ---------------------------------------------------------------------------

def morning_run() -> None:
    """06:00 ET — pull lineups, odds, and health-check."""
    try:
        log_header("MORNING RUN — lineups + odds")
        run_step("lineup_pull.py", "lineups")
        run_step("odds_current_pull.py", "odds")
        run_step("pipeline_health.py --upload", "health")
    except Exception:
        log.exception("Unhandled exception in morning_run — scheduler will continue.")


def picks_run() -> None:
    """08:30 ET — build profiles, generate picks, upload."""
    try:
        log_header("PICKS RUN — profiles + picks + upload")
        run_step("build_pitcher_profile.py", "profiles")
        run_step("build_team_stats_2026.py", "team_stats")
        run_step("run_today.py --csv", "picks")
        run_step("supabase_upload.py", "upload")
        run_step("pipeline_health.py --upload", "health")
    except Exception:
        log.exception("Unhandled exception in picks_run — scheduler will continue.")


def k_props_retry() -> None:
    """
    Every 30 min, 10:00–18:30 ET — fetch K props if stale or missing.

    Conditions to run:
    1. Current ET time is within the 10:00–18:30 window.
    2. The K props parquet for today is missing OR older than 6 hours.
    """
    try:
        now_et = _now_et()
        h, m = now_et.hour, now_et.minute
        start_h, start_m = K_PROPS_START_ET
        end_h, end_m = K_PROPS_END_ET
        in_window = (h * 60 + m) >= (start_h * 60 + start_m) and \
                    (h * 60 + m) <= (end_h * 60 + end_m)

        if not in_window:
            log.debug("K props retry: outside 10:00-18:30 ET window -- skipping")
            return

        today = date.today().isoformat()
        kprops_path = Path("data/statcast") / f"k_props_{today}.parquet"

        if kprops_path.exists():
            age_h = (time.time() - kprops_path.stat().st_mtime) / 3600
            if age_h < K_PROPS_MAX_AGE_HOURS:
                log.info("K props fresh (%.1fh old) — skipping retry", age_h)
                return
            log.info("K props stale (%.1fh old) — re-fetching", age_h)
        else:
            log.info("K props file not found for %s — fetching", today)

        log_header("K PROPS RETRY")
        rc, _ = run_step("odds_current_pull.py", "k_props")
        if rc == 0 and kprops_path.exists():
            log.info("K props successfully fetched — uploading updated picks")
            run_step("supabase_upload.py", "upload")
            run_step("pipeline_health.py --upload", "health")
        elif rc != 0:
            log.warning("K props fetch failed (rc=%d) — will retry at next interval", rc)
        else:
            log.warning("K props script exited OK but parquet still absent — will retry")
    except Exception:
        log.exception("Unhandled exception in k_props_retry — scheduler will continue.")


def night_run() -> None:
    """23:00 ET — run backtests, update model history, upload."""
    try:
        log_header("NIGHT RUN — backtest + model history + upload")
        run_step("backtest_2026.py", "backtest")
        run_step("backtest_mc_2026.py --rebuild", "mc_backtest")
        run_step("supabase_upload.py", "upload")
        run_step("pipeline_health.py --upload", "health")
    except Exception:
        log.exception("Unhandled exception in night_run — scheduler will continue.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

RUN_NOW_MAP = {
    "morning": morning_run,
    "picks":   picks_run,
    "night":   night_run,
    "k_props": k_props_retry,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wizard MLB daily pipeline scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--run-now",
        metavar="JOB",
        choices=list(RUN_NOW_MAP),
        help="Run a specific job immediately and exit (choices: %(choices)s)",
    )
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the schedule and exit without running anything",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Immediate single-job mode ---
    if args.run_now:
        log.info("--run-now %s: executing immediately", args.run_now)
        RUN_NOW_MAP[args.run_now]()
        log.info("--run-now %s: done", args.run_now)
        return

    # --- Register all scheduled jobs ---
    log.info("=" * 60)
    log.info("  Wizard MLB Scheduler starting")
    log.info("  All times ET. Local offset applied automatically.")
    log.info("=" * 60)

    schedule_et("06:00", morning_run, "morning_run")
    schedule_et("08:30", picks_run,   "picks_run")
    schedule_et("23:00", night_run,   "night_run")

    # K props: every 30 minutes; the job itself enforces the 10:00–18:30 ET window
    schedule.every(30).minutes.do(k_props_retry).tag("k_props")
    log.info("Scheduled %-14s every 30 min (window guard: 10:00-18:30 ET)", "k_props")

    # --- Dry-run mode: just print and exit ---
    if args.dry_run:
        log.info("")
        log.info("Dry-run mode — registered jobs:")
        for job in schedule.get_jobs():
            log.info("  %s", job)
        log.info("Exiting (dry-run).")
        return

    # --- Normal scheduler loop ---
    log.info("")
    log.info("Registered jobs:")
    for job in schedule.get_jobs():
        log.info("  %s", job)
    log.info("Waiting for next scheduled run... (Ctrl+C to stop)")

    try:
        while True:
            try:
                schedule.run_pending()
            except Exception:
                # Catch any scheduler-level errors so the loop never dies
                log.exception("Unexpected error in schedule.run_pending() — continuing")
            time.sleep(30)
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received — scheduler shutting down.")


if __name__ == "__main__":
    main()
