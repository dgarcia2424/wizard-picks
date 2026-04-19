"""
data_watchdog.py
================
Continuous data-integrity monitor for the Wizard Report MLB pipeline.

Runs every 30 minutes via Windows Task Scheduler. On each pass it:
  1. Checks every data source for staleness, missing files, bad row counts,
     and out-of-range values.
  2. Auto-repairs any issue it can fix by running the appropriate script.
  3. Sends an email alert when it repairs something or finds something it
     cannot repair automatically.
  4. Writes a JSON status file (watchdog_status.json) after every pass.

Usage:
    python data_watchdog.py              # run one pass and exit
    python data_watchdog.py --dry-run    # check only, no repairs, no email
    python data_watchdog.py --report     # print last watchdog_status.json
"""

import argparse
import json
import logging
import os
import smtplib
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

# Pipeline runs 4 AM – 4 PM ET. Outside that window staleness thresholds are
# relaxed so overnight watchdog passes don't generate false-positive alerts.
_ET = ZoneInfo("America/New_York")
_PIPELINE_START_H = 4   # 4 AM ET
_PIPELINE_END_H   = 18  # 6 PM ET (2h buffer after last 4PM run)

def _in_pipeline_hours() -> bool:
    h = datetime.now(_ET).hour
    return _PIPELINE_START_H <= h < _PIPELINE_END_H

def _stale_limit(tight_hours: float, loose_hours: float = 14.0) -> float:
    """Return tight threshold during pipeline hours, loose threshold overnight."""
    return tight_hours if _in_pipeline_hours() else loose_hours

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/watchdog.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("wizard.watchdog")


def _load_dotenv():
    env = BASE_DIR / ".env"
    if not env.exists():
        return
    with open(env) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

_load_dotenv()

# ---------------------------------------------------------------------------
# Check result dataclass (plain dict for simplicity)
# ---------------------------------------------------------------------------
# Each check returns a dict:
#   name        str   — human label
#   status      str   — "ok" | "warn" | "fail"
#   message     str   — what was found
#   repaired    bool  — whether auto-repair ran
#   repair_cmd  str   — script that was run (or "")


def _ok(name, msg):
    return dict(name=name, status="ok",   message=msg, repaired=False, repair_cmd="")

def _warn(name, msg, repair_cmd=""):
    return dict(name=name, status="warn", message=msg, repaired=False, repair_cmd=repair_cmd)

def _fail(name, msg, repair_cmd=""):
    return dict(name=name, status="fail", message=msg, repaired=False, repair_cmd=repair_cmd)


# ---------------------------------------------------------------------------
# Repair helper
# ---------------------------------------------------------------------------

def _repair(result: dict, dry_run: bool) -> dict:
    """Run the repair_cmd for a non-ok result. Marks result['repaired']."""
    cmd = result.get("repair_cmd", "")
    if not cmd or dry_run:
        return result
    log.warning("  AUTO-REPAIR [%s]: running %s", result["name"], cmd)
    parts = cmd.split()
    try:
        proc = subprocess.run(
            [sys.executable, "-X", "utf8"] + parts,
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode == 0:
            result["repaired"] = True
            result["message"] += f" | REPAIRED via {cmd}"
            log.info("  AUTO-REPAIR [%s]: OK", result["name"])
        else:
            result["message"] += f" | REPAIR FAILED (rc={proc.returncode}): {proc.stderr[:200]}"
            log.error("  AUTO-REPAIR [%s]: FAILED rc=%d", result["name"], proc.returncode)
    except subprocess.TimeoutExpired:
        result["message"] += f" | REPAIR TIMED OUT after 300s"
        log.error("  AUTO-REPAIR [%s]: TIMED OUT", result["name"])
    except Exception as e:
        result["message"] += f" | REPAIR ERROR: {e}"
        log.error("  AUTO-REPAIR [%s]: EXCEPTION %s", result["name"], e)
    return result


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _file_age_hours(p: Path) -> float:
    return (time.time() - p.stat().st_mtime) / 3600


def check_staleness(name, path, max_age_hours, repair_cmd=""):
    """File must exist and be newer than max_age_hours."""
    p = BASE_DIR / path
    if not p.exists():
        return _fail(name, f"FILE MISSING: {path}", repair_cmd)
    age = _file_age_hours(p)
    if age > max_age_hours:
        return _warn(name, f"STALE: {age:.1f}h old (limit {max_age_hours}h) — {path}", repair_cmd)
    return _ok(name, f"fresh ({age:.1f}h old)")


def check_parquet_date(name, path, date_col, max_data_age_days, repair_cmd=""):
    """Parquet must exist and its max date must be within max_data_age_days."""
    p = BASE_DIR / path
    if not p.exists():
        return _fail(name, f"FILE MISSING: {path}", repair_cmd)
    try:
        df = pd.read_parquet(p, columns=[date_col])
        mx = pd.to_datetime(df[date_col]).max().date()
        age = (date.today() - mx).days
        if age > max_data_age_days:
            return _warn(name, f"DATA STALE: max_date={mx} ({age}d ago, limit {max_data_age_days}d)", repair_cmd)
        return _ok(name, f"max_date={mx} ({age}d ago)")
    except Exception as e:
        return _fail(name, f"READ ERROR: {e}", repair_cmd)


def check_row_count(name, path, min_rows, repair_cmd="", is_csv=False):
    """File must have at least min_rows rows."""
    p = BASE_DIR / path
    if not p.exists():
        return _fail(name, f"FILE MISSING: {path}", repair_cmd)
    try:
        df = pd.read_csv(p) if is_csv else pd.read_parquet(p)
        if len(df) < min_rows:
            return _warn(name, f"LOW ROW COUNT: {len(df)} rows (min {min_rows})", repair_cmd)
        return _ok(name, f"{len(df)} rows")
    except Exception as e:
        return _fail(name, f"READ ERROR: {e}", repair_cmd)


def check_columns(name, path, required_cols, repair_cmd="", is_csv=False):
    """File must contain all required columns."""
    p = BASE_DIR / path
    if not p.exists():
        return _fail(name, f"FILE MISSING: {path}", repair_cmd)
    try:
        df = pd.read_csv(p, nrows=1) if is_csv else pd.read_parquet(p)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return _fail(name, f"MISSING COLUMNS: {missing}", repair_cmd)
        return _ok(name, f"all {len(required_cols)} required columns present")
    except Exception as e:
        return _fail(name, f"READ ERROR: {e}", repair_cmd)


def check_value_range(name, path, col, lo, hi, pct_bad_threshold=0.05, repair_cmd="", is_csv=False):
    """Column values must fall within [lo, hi] for at least (1-pct_bad_threshold) of rows."""
    p = BASE_DIR / path
    if not p.exists():
        return _ok(name, "skipped (file absent)")   # column check will catch absence
    try:
        df = pd.read_csv(p) if is_csv else pd.read_parquet(p)
        if col not in df.columns:
            return _ok(name, f"skipped (col {col!r} absent)")
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            return _ok(name, f"{col}: no numeric values to check")
        bad = ((series < lo) | (series > hi)).mean()
        bad_vals = series[(series < lo) | (series > hi)]
        if bad > pct_bad_threshold:
            sample = bad_vals.head(3).tolist()
            return _fail(
                name,
                f"BAD VALUES in {col}: {bad:.1%} out of range [{lo},{hi}] — sample={sample}",
                repair_cmd,
            )
        return _ok(name, f"{col}: {series.min():.3f}–{series.max():.3f} (all in [{lo},{hi}])")
    except Exception as e:
        return _fail(name, f"READ ERROR: {e}", repair_cmd)


# ---------------------------------------------------------------------------
# Full check suite
# ---------------------------------------------------------------------------

def run_all_checks() -> list[dict]:
    today_str   = date.today().isoformat()
    today_us    = date.today().strftime("%Y_%m_%d")
    is_gameday  = True   # assume game day; staleness thresholds handle off-days

    results = []

    # ── STATCAST ────────────────────────────────────────────────────────────
    results.append(check_parquet_date(
        "statcast_2026", "data/statcast/statcast_2026.parquet",
        "game_date", max_data_age_days=2,
        repair_cmd="statcast_pull_2026.py",
    ))
    results.append(check_parquet_date(
        "actuals_2026", "data/statcast/actuals_2026.parquet",
        "game_date", max_data_age_days=2,
        repair_cmd="statcast_pull_2026.py",
    ))

    # ── LINEUPS ─────────────────────────────────────────────────────────────
    results.append(check_staleness(
        "lineups_today", "data/statcast/lineups_today.parquet",
        max_age_hours=_stale_limit(5),
        repair_cmd="lineup_pull.py --recent",
    ))
    results.append(check_row_count(
        "lineups_today_rows", "data/statcast/lineups_today.parquet",
        min_rows=5,
        repair_cmd="lineup_pull.py --recent",
    ))

    # ── ODDS ────────────────────────────────────────────────────────────────
    odds_path = f"data/statcast/odds_current_{today_us}.parquet"
    results.append(check_staleness(
        "odds_current", odds_path,
        max_age_hours=_stale_limit(5),
        repair_cmd="odds_current_pull.py",
    ))

    # ── K PROPS ─────────────────────────────────────────────────────────────
    kprop_path = f"data/statcast/k_props_{today_str}.parquet"
    results.append(check_staleness(
        "k_props", kprop_path,
        max_age_hours=_stale_limit(4),
        repair_cmd="prizepicks_pull.py",
    ))

    # ── WEATHER ─────────────────────────────────────────────────────────────
    results.append(check_parquet_date(
        "weather_2026", "data/statcast/weather_2026.parquet",
        "game_date", max_data_age_days=1,
        repair_cmd=f"weather_pull.py --date {today_str}",
    ))

    # ── PITCHER PROFILES ────────────────────────────────────────────────────
    results.append(check_staleness(
        "pitcher_profiles", "data/statcast/pitcher_profiles_2026.parquet",
        max_age_hours=_stale_limit(5, 25),
        repair_cmd="build_pitcher_profile.py",
    ))
    results.append(check_row_count(
        "pitcher_profiles_rows", "data/statcast/pitcher_profiles_2026.parquet",
        min_rows=100,
        repair_cmd="build_pitcher_profile.py",
    ))

    # ── TEAM STATS ──────────────────────────────────────────────────────────
    results.append(check_staleness(
        "team_stats", "data/statcast/team_stats_2026.parquet",
        max_age_hours=_stale_limit(5, 25),
        repair_cmd="build_team_stats_2026.py",
    ))
    results.append(check_row_count(
        "team_stats_rows", "data/statcast/team_stats_2026.parquet",
        min_rows=28,
        repair_cmd="build_team_stats_2026.py",
    ))

    # ── BULLPEN AVAILABILITY ─────────────────────────────────────────────────
    results.append(check_parquet_date(
        "bullpen_avail_2026", "data/statcast/bullpen_avail_2026.parquet",
        "game_date", max_data_age_days=2,
        repair_cmd="build_bullpen_avail.py",
    ))

    # ── UMP ASSIGNMENTS ──────────────────────────────────────────────────────
    results.append(check_staleness(
        "ump_assignments", "data/statcast/umpire_assignments_2026.parquet",
        max_age_hours=_stale_limit(5, 25),
        repair_cmd="ump_pull.py",
    ))

    # ── LINEUP QUALITY ───────────────────────────────────────────────────────
    results.append(check_staleness(
        "lineup_quality", "data/statcast/lineup_quality_today.parquet",
        max_age_hours=_stale_limit(5),
        repair_cmd="build_lineup_quality.py",
    ))

    # ── DAILY CARD ───────────────────────────────────────────────────────────
    results.append(check_staleness(
        "daily_card", "daily_card.csv",
        max_age_hours=_stale_limit(5),
        repair_cmd="run_today.py --csv",
    ))
    results.append(check_row_count(
        "daily_card_rows", "daily_card.csv",
        min_rows=5, is_csv=True,
        repair_cmd="run_today.py --csv",
    ))

    # ── RAW CSV — COLUMNS ────────────────────────────────────────────────────
    results.append(check_columns(
        "fg_batters_cols", "data/raw/fangraphs_batters.csv",
        ["PA", "K%", "BB%", "wOBA", "wRC+"], is_csv=True,
        repair_cmd="refresh_raw_data.py --target fg_batters",
    ))
    results.append(check_columns(
        "fg_pitchers_cols", "data/raw/fangraphs_pitchers.csv",
        ["K/9", "BB/9", "FIP", "ERA"], is_csv=True,
        repair_cmd="refresh_raw_data.py --target fg_pitchers",
    ))
    results.append(check_columns(
        "savant_batters_cols", "data/raw/savant_batters.csv",
        ["xwoba", "xba", "pa"], is_csv=True,
        repair_cmd="refresh_raw_data.py --target savant_batters",
    ))
    results.append(check_columns(
        "savant_pitchers_cols", "data/raw/savant_pitchers.csv",
        ["xwoba", "k_percent"], is_csv=True,
        repair_cmd="refresh_raw_data.py --target savant_pitchers",
    ))
    results.append(check_columns(
        "fg_vs_lhp_cols", "data/raw/fangraphs_team_vs_lhp.csv",
        ["wRC+"], is_csv=True,
        repair_cmd="refresh_raw_data.py --target team_splits",
    ))
    results.append(check_columns(
        "fg_vs_rhp_cols", "data/raw/fangraphs_team_vs_rhp.csv",
        ["wRC+"], is_csv=True,
        repair_cmd="refresh_raw_data.py --target team_splits",
    ))

    # ── RAW CSV — VALUE RANGES ────────────────────────────────────────────────
    results.append(check_value_range(
        "fg_batters_wrc", "data/raw/fangraphs_batters.csv",
        "wRC+", lo=20, hi=220, is_csv=True,
        repair_cmd="refresh_raw_data.py --target fg_batters",
    ))
    results.append(check_value_range(
        "fg_batters_kpct", "data/raw/fangraphs_batters.csv",
        "K%", lo=0.01, hi=0.60, is_csv=True,
        repair_cmd="refresh_raw_data.py --target fg_batters",
    ))
    results.append(check_value_range(
        "fg_batters_bbpct", "data/raw/fangraphs_batters.csv",
        "BB%", lo=0.01, hi=0.30, is_csv=True,
        repair_cmd="refresh_raw_data.py --target fg_batters",
    ))
    results.append(check_value_range(
        "fg_pitchers_era", "data/raw/fangraphs_pitchers.csv",
        "ERA", lo=0.5, hi=12.0, is_csv=True,
        repair_cmd="refresh_raw_data.py --target fg_pitchers",
    ))
    results.append(check_value_range(
        "fg_pitchers_fip", "data/raw/fangraphs_pitchers.csv",
        "FIP", lo=1.0, hi=10.0, is_csv=True,
        repair_cmd="refresh_raw_data.py --target fg_pitchers",
    ))
    results.append(check_value_range(
        "fg_pitchers_k9", "data/raw/fangraphs_pitchers.csv",
        "K/9", lo=2.0, hi=18.0, is_csv=True,
        repair_cmd="refresh_raw_data.py --target fg_pitchers",
    ))
    results.append(check_value_range(
        "savant_batters_xwoba", "data/raw/savant_batters.csv",
        "xwoba", lo=0.200, hi=0.500, is_csv=True,
        repair_cmd="refresh_raw_data.py --target savant_batters",
    ))
    results.append(check_value_range(
        "fg_vs_lhp_wrc", "data/raw/fangraphs_team_vs_lhp.csv",
        "wRC+", lo=40, hi=180, is_csv=True,
        repair_cmd="refresh_raw_data.py --target team_splits",
    ))
    results.append(check_value_range(
        "fg_vs_rhp_wrc", "data/raw/fangraphs_team_vs_rhp.csv",
        "wRC+", lo=40, hi=180, is_csv=True,
        repair_cmd="refresh_raw_data.py --target team_splits",
    ))

    # ── RAW CSV — STALENESS ──────────────────────────────────────────────────
    results.append(check_staleness(
        "fg_batters_age", "data/raw/fangraphs_batters.csv",
        max_age_hours=_stale_limit(5, 25),
        repair_cmd="refresh_raw_data.py --target fg_batters",
    ))
    results.append(check_staleness(
        "savant_batters_age", "data/raw/savant_batters.csv",
        max_age_hours=_stale_limit(5, 25),
        repair_cmd="refresh_raw_data.py --target savant_batters",
    ))

    # ── SUPPLEMENTAL ─────────────────────────────────────────────────────────
    results.append(check_staleness(
        "park_factors_2026", "data/statcast/park_factors_2026.parquet",
        max_age_hours=_stale_limit(5, 26),
        repair_cmd="supplemental_pull.py --force-year 2026",
    ))
    results.append(check_staleness(
        "standings_2026", "data/statcast/standings_2026.parquet",
        max_age_hours=_stale_limit(5, 26),
        repair_cmd="supplemental_pull.py --force-year 2026",
    ))

    return results


# ---------------------------------------------------------------------------
# Email alert
# ---------------------------------------------------------------------------

def _send_alert(issues: list[dict], repaired: list[dict], dry_run: bool) -> None:
    if dry_run:
        return
    gmail_from = os.getenv("GMAIL_FROM", "garcia.dan24@gmail.com")
    gmail_pass = os.getenv("GMAIL_APP_PASSWORD", "")
    recipients = os.getenv("EMAIL_RECIPIENTS", gmail_from).split(",")

    if not gmail_pass:
        log.warning("GMAIL_APP_PASSWORD not set — skipping alert email")
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M ET")
    subject = f"[Wizard Watchdog] {len(issues)} unrepaired issue(s) — {now}"

    lines = [f"Data watchdog pass: {now}", ""]
    if repaired:
        lines.append(f"AUTO-REPAIRED ({len(repaired)}):")
        for r in repaired:
            lines.append(f"  * {r['name']}: {r['message']}")
        lines.append("")
    still_broken = [i for i in issues if not i["repaired"]]
    if still_broken:
        lines.append(f"STILL BROKEN ({len(still_broken)}) — manual action needed:")
        for r in still_broken:
            lines.append(f"  * [{r['status'].upper()}] {r['name']}: {r['message']}")
    body = "\n".join(lines)

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = gmail_from
        msg["To"]      = ", ".join(recipients)
        msg.attach(MIMEText(body, "plain", "utf-8"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(gmail_from, gmail_pass)
            s.sendmail(gmail_from, recipients, msg.as_string())
        log.info("Alert email sent to %s", recipients)
    except Exception as e:
        log.error("Failed to send alert email: %s", e)


# ---------------------------------------------------------------------------
# Status file
# ---------------------------------------------------------------------------

def _write_status(results: list[dict], elapsed: float) -> None:
    counts = {s: sum(1 for r in results if r["status"] == s) for s in ("ok", "warn", "fail")}
    repaired = sum(1 for r in results if r["repaired"])
    status = {
        "last_run":    datetime.now().isoformat(timespec="seconds"),
        "elapsed_s":   round(elapsed, 1),
        "total_checks": len(results),
        "ok":          counts["ok"],
        "warn":        counts["warn"],
        "fail":        counts["fail"],
        "repaired":    repaired,
        "overall":     "ok" if counts["fail"] == 0 and counts["warn"] == 0 else
                       ("warn" if counts["fail"] == 0 else "fail"),
        "issues": [r for r in results if r["status"] != "ok"],
    }
    with open(BASE_DIR / "watchdog_status.json", "w") as f:
        json.dump(status, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Wizard data integrity watchdog")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check only — no repairs, no email")
    parser.add_argument("--report",  action="store_true",
                        help="Print last watchdog_status.json and exit")
    args = parser.parse_args()

    if args.report:
        p = BASE_DIR / "watchdog_status.json"
        if p.exists():
            print(p.read_text())
        else:
            print("No watchdog_status.json found — run without --report first.")
        return

    mode = "DRY-RUN" if args.dry_run else "LIVE"
    log.info("=" * 60)
    log.info("  Wizard Data Watchdog — %s — %s", mode, datetime.now().strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 60)

    t0 = time.time()
    results = run_all_checks()

    issues = [r for r in results if r["status"] != "ok"]

    # ── Auto-repair ─────────────────────────────────────────────────────────
    repaired = []
    for r in issues:
        if r.get("repair_cmd"):
            _repair(r, dry_run=args.dry_run)
            if r["repaired"]:
                repaired.append(r)

    elapsed = time.time() - t0

    # ── Summary log ─────────────────────────────────────────────────────────
    ok_count   = sum(1 for r in results if r["status"] == "ok")
    fail_count = sum(1 for r in results if r["status"] == "fail")
    warn_count = sum(1 for r in results if r["status"] == "warn")

    log.info("")
    log.info("RESULTS: %d ok | %d warn | %d fail | %d repaired | %.1fs",
             ok_count, warn_count, fail_count, len(repaired), elapsed)

    for r in issues:
        flag = "REPAIRED" if r["repaired"] else r["status"].upper()
        log.info("  [%s] %s: %s", flag, r["name"], r["message"])

    if not issues:
        log.info("  All checks passed.")

    # ── Write status ─────────────────────────────────────────────────────────
    _write_status(results, elapsed)

    # ── Alert email (only when unrepaired issues remain) ─────────────────────
    unrepaired = [r for r in issues if not r["repaired"]]
    if unrepaired:
        _send_alert(unrepaired, repaired, dry_run=args.dry_run)

    log.info("=" * 60)


if __name__ == "__main__":
    main()
