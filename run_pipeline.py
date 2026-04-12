"""
run_pipeline.py — Runs all MLB data pipeline scripts in order.

Executes each script as a subprocess so a failure in one step doesn't
crash the others. Prints a summary at the end showing what passed/failed.

Usage:
    python run_pipeline.py               # full pipeline
    python run_pipeline.py --skip odds   # skip odds scripts
    python run_pipeline.py --only supplemental weather  # run specific steps
    python run_pipeline.py --from weather               # resume from a step
    python run_pipeline.py --dry-run     # print order without running
    python run_pipeline.py --daily       # lightweight daily card (lineups → picks → upload)

DAILY RUN ORDER (--daily or run manually):
  1. lineup_pull.py           ← today's starters from MLB Stats API
  2. odds_current_pull.py     ← today's ML/RL/total + K props from Odds API
  3. build_pitcher_profile.py ← refresh pitcher xwOBA/K%/IP profiles
  4. build_team_stats_2026.py ← refresh team batting/bullpen stats
  5. run_today.py --csv       ← MC + XGBoost predictions → daily_card.csv
  6. backtest_2026.py         ← append today's games to backtest tracker
  7. supabase_upload.py       ← push daily_card.csv + backtest to Supabase

FULL PIPELINE (weekly / season start):
  Same as above but also runs supplemental_pull.py (FanGraphs/Statcast)
  and weather_pull.py (historical weather for all games).
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ─── PIPELINE STEPS ──────────────────────────────────────────────────────────
# Each entry: (step_name, script, description, estimated_minutes, tags)
# Tags are used for --skip and --only filtering.

STEPS = [
    # ── Heavy / weekly pulls ──────────────────────────────────────────────────
    (
        "supplemental",
        "supplemental_pull.py",
        "FanGraphs stats, Statcast leaderboards, MLB schedule, park factors, pitcher handedness",
        30,
        ["data", "pybaseball", "weekly"],
    ),
    (
        "weather",
        "weather_pull.py",
        "Open-Meteo per-game weather (temp, wind, humidity) for all historical games",
        20,
        ["data", "weather", "weekly"],
    ),
    (
        "odds_historical",
        "odds_historical_pull.py",
        "OddsPortal historical lines 2023-2025 (Playwright scraper — run once)",
        120,
        ["odds", "historical", "one-time"],
    ),
    (
        "odds_combine",
        "odds_combine.py",
        "Merge all odds sources into odds_combined_{year}.parquet",
        2,
        ["odds", "weekly"],
    ),
    # ── Daily pulls ───────────────────────────────────────────────────────────
    (
        "lineups",
        "lineup_pull.py",
        "Today's confirmed lineups + probable pitchers from MLB Stats API",
        1,
        ["data", "daily"],
    ),
    (
        "odds_current",
        "odds_current_pull.py",
        "Today's ML/RL/total odds + pitcher K props via Odds API",
        2,
        ["odds", "daily"],
    ),
    # ── Model inputs (build from raw data) ────────────────────────────────────
    (
        "pitcher_profiles",
        "build_pitcher_profile.py",
        "Compute trailing xwOBA/K%/BB%/velocity/expected_IP per pitcher from 2026 Statcast",
        3,
        ["model", "daily"],
    ),
    (
        "team_stats",
        "build_team_stats_2026.py",
        "Compute team batting splits (xwOBA vs RHP/LHP), RS/G, bullpen ERA for 2026",
        2,
        ["model", "daily"],
    ),
    # ── Predictions + tracking ────────────────────────────────────────────────
    (
        "picks",
        "run_today.py --csv",
        "Generate today's MC + XGBoost picks and save daily_card.csv",
        2,
        ["daily", "picks"],
    ),
    (
        "backtest",
        "backtest_2026.py",
        "Append completed games to backtest tracker (append mode, no duplication)",
        1,
        ["daily", "tracking"],
    ),
    (
        "upload",
        "supabase_upload.py",
        "Push daily_card.csv, backtest, and model history to Supabase",
        1,
        ["daily", "upload"],
    ),
    # ── Validation ────────────────────────────────────────────────────────────
    (
        "catalog",
        "data_catalog.py",
        "Validate all parquets, print coverage grid, flag missing files",
        1,
        ["validate"],
    ),
]

# Steps that run in the lightweight daily card generation
DAILY_STEPS = {"lineups", "odds_current", "pitcher_profiles", "team_stats",
               "picks", "backtest", "upload"}

STEP_NAMES = [s[0] for s in STEPS]

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def print_header(text: str):
    print(f"\n{'='*65}")
    print(f"  {text}")
    print(f"{'='*65}")


def print_step_start(i: int, total: int, name: str, desc: str, est_min: int):
    print(f"\n[{i}/{total}] {name.upper()}")
    print(f"  {desc}")
    if est_min >= 30:
        print(f"  Est. runtime: ~{est_min} min (first run)")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"  {'-'*60}")


# ─── RUNNER ──────────────────────────────────────────────────────────────────

def run_step(script: str) -> tuple[int, float]:
    """
    Run script (with optional args) as subprocess, streaming output in real time.
    script may be "run_today.py --csv" — it is split on spaces.
    Returns (returncode, elapsed_seconds).
    """
    start = time.time()
    parts = script.split()   # e.g. ["run_today.py", "--csv"]
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
        print("  " + line, end="")
    proc.wait()
    return proc.returncode, time.time() - start


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run full MLB data pipeline")
    parser.add_argument(
        "--skip", nargs="+", metavar="STEP",
        help=f"Steps to skip. Choices: {STEP_NAMES}"
    )
    parser.add_argument(
        "--only", nargs="+", metavar="STEP",
        help="Run only these steps"
    )
    parser.add_argument(
        "--from", dest="from_step", metavar="STEP",
        help="Resume from this step (skip all before it)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print steps without running them"
    )
    parser.add_argument(
        "--no-catalog", action="store_true",
        help="Skip the final data_catalog validation step"
    )
    parser.add_argument(
        "--daily", action="store_true",
        help=(
            "Run only the lightweight daily card steps: "
            "lineups → odds_current → pitcher_profiles → team_stats → picks → backtest → upload"
        )
    )
    args = parser.parse_args()

    # ── Filter steps ─────────────────────────────────────────────────────────
    steps = list(STEPS)

    # --daily shortcut: run only daily-tagged steps, skip catalog
    if args.daily and not args.only:
        steps = [s for s in steps if s[0] in DAILY_STEPS]

    if args.from_step:
        if args.from_step not in STEP_NAMES:
            print(f"Unknown step: {args.from_step}. Choices: {STEP_NAMES}")
            sys.exit(1)
        idx = STEP_NAMES.index(args.from_step)
        steps = steps[idx:]

    if args.only:
        unknown = [s for s in args.only if s not in STEP_NAMES]
        if unknown:
            print(f"Unknown steps: {unknown}. Choices: {STEP_NAMES}")
            sys.exit(1)
        steps = [s for s in steps if s[0] in args.only]

    if args.skip:
        steps = [s for s in steps if s[0] not in args.skip]

    if args.no_catalog:
        steps = [s for s in steps if s[0] != "catalog"]

    if not steps:
        print("No steps selected.")
        sys.exit(0)

    # ── Check scripts exist ───────────────────────────────────────────────────
    missing = []
    for name, script, *_ in steps:
        script_file = script.split()[0]   # strip any CLI args (e.g. "run_today.py --csv")
        if not Path(script_file).exists():
            missing.append(script_file)
    if missing:
        print("Missing scripts:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print_header(f"DRY RUN — {len(steps)} steps")
        for i, (name, script, desc, est, tags) in enumerate(steps, 1):
            print(f"\n  [{i}] {name:20s}  ~{est:>4} min  {script}")
            print(f"       {desc}")
        print()
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    total_est = sum(s[3] for s in steps)
    print_header(
        f"MLB DATA PIPELINE  —  {len(steps)} steps  "
        f"(est. {total_est} min first run)"
    )
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = []  # (name, returncode, elapsed)
    pipeline_start = time.time()

    for i, (name, script, desc, est_min, tags) in enumerate(steps, 1):
        print_step_start(i, len(steps), name, desc, est_min)

        returncode, elapsed = run_step(script)
        status = "PASS" if returncode == 0 else f"FAIL (exit {returncode})"
        results.append((name, returncode, elapsed))

        print(f"\n  [{status}]  {name}  —  {fmt_duration(elapsed)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - pipeline_start
    passed  = sum(1 for _, rc, _ in results if rc == 0)
    failed  = len(results) - passed

    print_header(f"PIPELINE COMPLETE  —  {fmt_duration(total_elapsed)}")
    print(f"  {passed}/{len(results)} steps passed\n")

    col_w = max(len(n) for n, *_ in results)
    for name, rc, elapsed in results:
        status_str = "PASS" if rc == 0 else f"FAIL"
        bar        = "OK" if rc == 0 else "!!"
        print(f"  [{bar}]  {name:{col_w}}  {fmt_duration(elapsed):>8}  {status_str}")

    if failed:
        print(f"\n  {failed} step(s) failed — check output above for errors.")
        sys.exit(1)
    else:
        print(f"\n  All steps completed successfully.")


if __name__ == "__main__":
    main()
