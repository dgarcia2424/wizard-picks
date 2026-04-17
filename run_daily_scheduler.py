"""
run_daily_scheduler.py
======================
Three-shot daily scheduler for The Wizard Report MLB pipeline.

Fires three times per day at 11:00 AM, 2:00 PM, and 5:00 PM ET so that
late-starting games receive updated lineups and closing odds before first pitch.

Usage:
    python run_daily_scheduler.py              # start scheduler (runs forever)
    python run_daily_scheduler.py --run-now    # fire run_all immediately and exit
    python run_daily_scheduler.py --dry-run    # print schedule and exit

Schedule (all times ET):
    11:00  RUN_ALL     — full pipeline (lineups, pitcher profiles, team stats,
                         lineup quality, odds, predictions, upload, health)
    14:00  RUN_REFRESH — lineups, lineup quality, odds, predictions, upload, health
    17:00  RUN_REFRESH — lineups, lineup quality, odds, predictions, upload, health

The 2 PM and 5 PM refreshes skip pitcher profiles and team stats (no intraday
change) but re-run lineup quality after the fresh lineup pull so wRC+ scores
reflect any late starters.  Odds are re-pulled to capture closing-line movement.
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


def schedule_et(time_str: str, job_fn, label: str) -> None:
    """
    Register a daily job that fires at a fixed ET clock time.

    The ET time is converted to local wall-clock time before being handed
    to the ``schedule`` library so the job fires correctly regardless of
    the machine's timezone.
    """
    et_naive = datetime.strptime(time_str, "%H:%M").replace(
        year=2026, month=1, day=1, tzinfo=ET
    )
    local_time = et_naive.astimezone().strftime("%H:%M")
    schedule.every().day.at(local_time).do(job_fn).tag(label)
    log.info("Scheduled %-14s at %s ET  (local: %s)", label, time_str, local_time)


# ---------------------------------------------------------------------------
# Pipeline job
# ---------------------------------------------------------------------------

def run_all() -> None:
    """11:00 AM ET — full pipeline run."""
    try:
        log_header("DAILY RUN — 11:00 AM ET — full pipeline")
        tomorrow = (date.today() + __import__("datetime").timedelta(days=1)).isoformat()

        # ── Step 1: Lineups + probable starters ───────────────────────────
        run_step("lineup_pull.py --recent", "lineups_today")
        run_step(f"lineup_pull.py --date {tomorrow}", "lineups_tmrw")

        # ── Step 2: Pitcher profiles + team stats refresh ─────────────────
        run_step("build_pitcher_profile.py", "pitcher_profiles")
        run_step("build_team_stats_2026.py", "team_stats")

        # ── Step 2c: Lineup quality scores (wRC+ per team) ────────────────
        run_step("build_lineup_quality.py", "lineup_quality")

        # ── Step 3: Fresh dual-region odds pull ───────────────────────────
        rc, _ = run_step("odds_current_pull.py", "odds_pull")
        if rc != 0:
            log.error("Odds pull failed — predictions will degrade to retail-only fallback.")

        # ── Step 4: Generate today's card ─────────────────────────────────
        rc, _ = run_step("run_today.py --csv --email", "picks")
        if rc != 0:
            log.warning("run_today.py exited non-zero — check model_scores.csv for errors.")

        # ── Step 5: Upload results to Supabase ────────────────────────────
        run_step("supabase_upload.py", "upload")

        # ── Step 6: Pipeline health snapshot ─────────────────────────────
        run_step("pipeline_health.py --upload", "health")

    except Exception:
        log.exception("Unhandled exception in run_all — scheduler will continue.")


def run_refresh(label: str) -> None:
    """
    2:00 PM and 5:00 PM ET — lightweight intraday refresh.

    Skips pitcher profiles and team stats (no intraday updates).
    Re-pulls lineups (late starters confirm after 11 AM) and odds
    (lines tighten toward close), then regenerates the card and uploads.
    """
    try:
        log_header(f"REFRESH RUN — {label} ET — lineups + odds + picks")

        # ── Step 1: Lineups — capture any starters confirmed since 11 AM ──
        run_step("lineup_pull.py --recent", "lineups_today")

        # ── Step 1b: Lineup quality — refresh wRC+ scores with new lineups ─
        run_step("build_lineup_quality.py", "lineup_quality")

        # ── Step 2: Updated odds (closing lines sharper than morning) ──────
        rc, _ = run_step("odds_current_pull.py", "odds_pull")
        if rc != 0:
            log.error("Odds pull failed — predictions will degrade to retail-only fallback.")

        # ── Step 3: Regenerate card with refreshed data ───────────────────
        rc, _ = run_step("run_today.py --csv --email", "picks")
        if rc != 0:
            log.warning("run_today.py exited non-zero — check model_scores.csv for errors.")

        # ── Step 4: Upload updated picks to Supabase ──────────────────────
        run_step("supabase_upload.py", "upload")

        # ── Step 5: Health snapshot ───────────────────────────────────────
        run_step("pipeline_health.py --upload", "health")

    except Exception:
        log.exception("Unhandled exception in run_refresh — scheduler will continue.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wizard MLB daily pipeline scheduler — 11 AM / 2 PM / 5 PM ET",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--run-now",
        action="store_true",
        help="Fire run_all immediately and exit",
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
        log.info("--run-now: firing run_all immediately")
        run_all()
        log.info("--run-now: done")
        return

    # --- Register scheduled job ---
    log.info("=" * 60)
    log.info("  Wizard MLB Scheduler starting")
    log.info("  All times ET. Local offset applied automatically.")
    log.info("=" * 60)

    schedule_et("11:00", run_all,                          "run_all")
    schedule_et("14:00", lambda: run_refresh("2:00 PM"),   "run_refresh_2pm")
    schedule_et("17:00", lambda: run_refresh("5:00 PM"),   "run_refresh_5pm")

    # --- Dry-run mode ---
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
                log.exception("Unexpected error in schedule.run_pending() — continuing")
            time.sleep(30)
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received — scheduler shutting down.")


if __name__ == "__main__":
    main()
