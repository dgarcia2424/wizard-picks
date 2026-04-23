"""
sunday_routine.py — Deterministic weekly maintenance.

Replaces the LLM-driven Agent 2 (Static Data Manager) and Agent 7
(Maintenance & Roadmap). No Anthropic calls. Runs in milliseconds.

Usage:
    python sunday_routine.py              # validate static data + status report
    python sunday_routine.py --roadmap    # status report only
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from config.settings import (
    FILES, STATIC_FILES, STALE_THRESHOLD_DAYS,
    YEAR_WEIGHTS, PIPELINE_DIR,
)

TODAY = date.today()

# ── Static-file validation specs ──────────────────────────────────────────────
# (file_key, required_cols, min_rows, size_floor_kb, size_ceiling_kb)
STATIC_SPECS: list[tuple[str, list[str], int, int | None, int | None]] = [
    ("savant_pitchers",       ["woba", "est_woba", "xera", "era"],  50, None, 200),
    ("savant_batters",        ["woba", "est_woba", "est_ba"],       100, 100, None),
    ("fangraphs_pitchers",    ["Name", "Team", "ERA", "xFIP", "FIP", "WAR"], 100, None, None),
    # SO% / PA added for the Monte Carlo K-prop engine: per-batter K-rate
    # drives Log5 matchup math against the pitcher's K-rate. File ships this
    # column labelled "K%"; "SO%" is the semantic synonym — we require K%.
    ("fangraphs_batters",     ["Name", "Team", "wOBA", "xwOBA", "K%", "PA"], 100, None, None),
    # K% / BB% on the team-vs-handedness tables give the 9-man lineup proxy
    # when confirmed batting orders are not yet published.
    ("fangraphs_team_vs_lhp", ["Tm", "wOBA", "K%", "BB%"],           30, None, None),
    ("fangraphs_team_vs_rhp", ["Tm", "wOBA", "K%", "BB%"],           30, None, None),
    ("fangraphs_bullpen",     ["Team", "ERA", "xFIP", "WAR"],        30, None, None),
]

# Team-level files must have exactly one row per MLB franchise. A missing
# or duplicated team silently poisons the home/away join in downstream
# scorers — enforce exact-30 after the column-presence check passes.
EXACT_30_KEYS = ("fangraphs_team_vs_lhp", "fangraphs_team_vs_rhp", "fangraphs_bullpen")

FANGRAPHS_BLEND_KEYS = ["fangraphs_pitchers", "fangraphs_batters"]
CATEGORICAL_COLS = {"Name", "Team", "Tm", "Season", "Age", "Pos", "Throws", "Bats"}


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2 LOGIC — Static data validation + FanGraphs blending
# ══════════════════════════════════════════════════════════════════════════════

def _validate_one(
    file_key: str,
    required_cols: list[str],
    min_rows: int,
    size_floor_kb: int | None,
    size_ceiling_kb: int | None,
) -> tuple[bool, str, int]:
    path = FILES[file_key]
    if not path.exists():
        return False, f"missing file: {path}", 0

    size_kb = path.stat().st_size / 1024
    if size_ceiling_kb is not None and size_kb > size_ceiling_kb:
        return False, (f"size {size_kb:.0f}KB > ceiling {size_ceiling_kb}KB "
                       f"(likely wrong file — check savant_pitchers vs savant_batters)"), 0
    if size_floor_kb is not None and size_kb < size_floor_kb:
        return False, (f"size {size_kb:.0f}KB < floor {size_floor_kb}KB "
                       f"(likely wrong file — check savant_pitchers vs savant_batters)"), 0

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return False, f"read failed: {type(e).__name__}: {e}", 0

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, f"missing required columns: {missing}", len(df)
    if len(df) < min_rows:
        return False, f"only {len(df)} rows (min {min_rows})", len(df)

    return True, f"{len(df)} rows, {size_kb:.0f}KB", len(df)


def _blend_fangraphs(file_key: str) -> str | None:
    """If per-year source files exist (e.g. fangraphs_pitchers_2026.csv),
    blend them with YEAR_WEIGHTS and overwrite the canonical CSV.
    Returns a status string, or None if per-year sources absent."""
    raw_dir = FILES[file_key].parent
    base    = FILES[file_key].stem

    frames: dict[int, pd.DataFrame] = {}
    for yr in YEAR_WEIGHTS:
        p = raw_dir / f"{base}_{yr}.csv"
        if p.exists():
            frames[yr] = pd.read_csv(p)

    if not frames:
        return None  # nothing to blend — canonical CSV is taken as-is

    years  = sorted(frames, reverse=True)
    newest = frames[years[0]]
    key    = "Name" if "Name" in newest.columns else newest.columns[0]

    keep_cats = [c for c in newest.columns if c in CATEGORICAL_COLS]
    num_cols  = [c for c in newest.columns
                 if c not in CATEGORICAL_COLS
                 and pd.api.types.is_numeric_dtype(newest[c])]

    blended = newest[[key] + keep_cats].copy()
    for col in num_cols:
        total_w = 0.0
        acc     = pd.Series(0.0, index=blended[key], dtype=float)
        for yr in years:
            if col not in frames[yr].columns:
                continue
            w = YEAR_WEIGHTS[yr]
            s = frames[yr].set_index(key)[col].astype(float)
            acc = acc.add(s.reindex(blended[key]).fillna(0.0) * w, fill_value=0.0)
            total_w += w
        blended[col] = (acc.values / total_w) if total_w > 0 else 0.0

    blended.to_csv(FILES[file_key], index=False)
    return f"blended {len(frames)} years → {len(blended)} rows ({','.join(map(str, years))})"


def run_static_validation() -> tuple[list[str], int]:
    lines: list[str] = ["── STATIC DATA VALIDATION ──"]
    failures = 0

    for file_key in FANGRAPHS_BLEND_KEYS:
        status = _blend_fangraphs(file_key)
        if status:
            lines.append(f"  [blend] {file_key}: {status}")

    for key, cols, min_rows, floor, ceil in STATIC_SPECS:
        ok, msg, nrows = _validate_one(key, cols, min_rows, floor, ceil)
        mark = "✅" if ok else "❌"
        lines.append(f"  {mark} {key:<24} {msg}")
        if not ok:
            failures += 1
            continue
        if key in EXACT_30_KEYS and nrows != 30:
            lines.append(f"  ❌ {key:<24} has {nrows} rows — must be exactly 30 (one per MLB team)")
            failures += 1

    return lines, failures


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 7 LOGIC — Performance alerts + roadmap status
# ══════════════════════════════════════════════════════════════════════════════

ROADMAP_NEXT    = "#3 Bullpen quality signal (FanGraphs bullpen ERA/xFIP by team, +2-3% acc, Low effort)"
ACCURACY_TARGET = "70–75% by mid-May 2026"
MILESTONE_PICKS = 200
ROLLING_WINDOW  = 30
ROLLING_FLOOR   = 0.57
STALE_ALERT_DAYS = 10


def _load_ledger() -> pd.DataFrame:
    p = FILES["historical_picks"]
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "result" in df.columns:
        df["result"] = df["result"].astype(str).str.upper()
    return df


def _ledger_stats(df: pd.DataFrame) -> dict:
    if df.empty or "result" not in df.columns:
        return {"total": 0, "wins": 0, "losses": 0, "pushes": 0,
                "win_pct": 0.0, "pl": 0.0, "roi_pct": 0.0,
                "rolling_win_pct": None, "by_market": {}}

    settled = df[df["result"].isin(["WIN", "LOSS", "PUSH"])].copy()
    wins    = int((settled["result"] == "WIN").sum())
    losses  = int((settled["result"] == "LOSS").sum())
    pushes  = int((settled["result"] == "PUSH").sum())
    decisive = wins + losses
    win_pct  = (wins / decisive) if decisive else 0.0

    pl = float(pd.to_numeric(settled.get("profit_loss"), errors="coerce").fillna(0).sum())
    stake_col = "dollar_stake" if "dollar_stake" in settled.columns else None
    risked = float(pd.to_numeric(settled[stake_col], errors="coerce").fillna(0).sum()) if stake_col else 0.0
    roi_pct = (pl / risked * 100.0) if risked else 0.0

    rolling = None
    if decisive >= ROLLING_WINDOW:
        last = settled[settled["result"].isin(["WIN", "LOSS"])].tail(ROLLING_WINDOW)
        rolling = (last["result"] == "WIN").mean()

    by_market: dict[str, dict] = {}
    if "model" in settled.columns:
        for model, sub in settled.groupby("model"):
            d = sub[sub["result"].isin(["WIN", "LOSS"])]
            n = len(d)
            w = int((d["result"] == "WIN").sum())
            by_market[str(model)] = {
                "bets":    int(len(sub)),
                "win_pct": (w / n) if n else 0.0,
            }

    return {"total": len(settled), "wins": wins, "losses": losses, "pushes": pushes,
            "win_pct": win_pct, "pl": pl, "roi_pct": roi_pct,
            "rolling_win_pct": rolling, "by_market": by_market}


def _stale_static_files() -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for key in STATIC_FILES:
        p = FILES.get(key)
        if not p or not p.exists():
            continue
        age_days = (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).days
        if age_days >= STALE_ALERT_DAYS:
            out.append((key, age_days))
    return out


def run_maintenance_report() -> list[str]:
    df    = _load_ledger()
    stats = _ledger_stats(df)

    alerts: list[str] = []
    if stats["total"] >= MILESTONE_PICKS:
        alerts.append(f"🎯 MILESTONE: Activate CLV flag + begin logistic regression "
                      f"(pick count = {stats['total']} ≥ {MILESTONE_PICKS})")
    rolling = stats["rolling_win_pct"]
    if rolling is not None and rolling < ROLLING_FLOOR:
        alerts.append(f"⚠️ PERFORMANCE ALERT: 30-pick rolling win rate {rolling*100:.1f}% "
                      f"< {ROLLING_FLOOR*100:.0f}% — review model parameters")
    for key, age in _stale_static_files():
        alerts.append(f"⚠️ STALE DATA: {key} is {age}d old (threshold {STALE_ALERT_DAYS}d)")

    lines: list[str] = [
        "",
        f"THE WIZARD REPORT — MAINTENANCE STATUS | {TODAY.strftime('%B %d, %Y')}",
        "=" * 60,
        f"ROADMAP    : Next = {ROADMAP_NEXT}",
        f"             Target = {ACCURACY_TARGET} | Picks to date = {stats['total']} "
        f"| Milestone ETA = {MILESTONE_PICKS} picks",
        f"PERFORMANCE: {stats['wins']}-{stats['losses']}-{stats['pushes']} "
        f"| Win% = {stats['win_pct']*100:.1f}% "
        f"| P/L = ${stats['pl']:+,.2f} "
        f"| ROI = {stats['roi_pct']:+.2f}%",
    ]
    if stats["by_market"]:
        parts = [f"{m}:{v['bets']}@{v['win_pct']*100:.0f}%" for m, v in stats["by_market"].items()]
        lines.append(f"             Per-model: {'  '.join(parts)}")
    if rolling is not None:
        lines.append(f"             Rolling {ROLLING_WINDOW}: {rolling*100:.1f}%")

    lines.append("")
    lines.append("ALERTS:")
    if alerts:
        for a in alerts:
            lines.append(f"  {a}")
    else:
        lines.append("  (none — system within tolerances)")

    lines.append("")
    lines.append(f"NEXT ACTION: Ship {ROADMAP_NEXT}")
    lines.append("=" * 60)
    return lines


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roadmap", action="store_true",
                    help="Skip static-data validation; print status report only.")
    args = ap.parse_args()

    exit_code = 0

    if not args.roadmap:
        lines, failures = run_static_validation()
        print("\n".join(lines))
        if failures:
            print(f"\n❌ {failures} static file(s) failed validation — fix before Monday pipeline run.")
            exit_code = 1
        else:
            print(f"\n✅ Static data current as of {TODAY.isoformat()}. "
                  f"Next update: {(TODAY + timedelta(days=7)).isoformat()}.")

    for line in run_maintenance_report():
        print(line)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
