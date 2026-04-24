"""
run_retrain_all.py — Full pipeline retrain + fit comparison.

Runs every training script in dependency order, captures AUC metrics,
and prints a before/after comparison table.

Usage
-----
  python run_retrain_all.py                  # all models
  python run_retrain_all.py --skip-current   # skip models already trained today
  python run_retrain_all.py --dry-run        # show plan only, no training
  python run_retrain_all.py --models ml,f5   # comma-separated subset

Models (in run order)
---------------------
  ml          train_ml_model.py --with-2026
  run_dist    train_run_dist_model.py --with-2026
  f5          train_f5_model.py --with-2026
  nrfi        train_nrfi_model.py --with-2026
  tb60        retrain_tb_v60.py
  k_over      train_k_over_v1.py
  script_a    train_script_a.py
  script_a2   train_script_a2.py
  script_b    train_script_b.py
  script_c    train_script_c.py
  rl_v1       train_rl_v1.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

_ROOT    = Path(__file__).resolve().parent
_MODELS  = _ROOT / "models"
_LOGS    = _ROOT / "data" / "logs"
_LOGS.mkdir(parents=True, exist_ok=True)

TODAY = date.today().isoformat()

# ---------------------------------------------------------------------------
# Training plan: (key, script, extra_args, auc_log_file)
# auc_log_file = path to existing structured metrics txt; None = parse stdout
# ---------------------------------------------------------------------------
PLAN = [
    {
        "key":       "ml",
        "label":     "ML v2 (Moneyline)",
        "script":    "train_ml_model.py",
        "args":      ["--with-2026"],
        "log_file":  None,
        "before_src": "v2_meta_auc",   # special: pull from models/v2_meta.json
    },
    {
        "key":       "run_dist",
        "label":     "Run Distribution",
        "script":    "train_run_dist_model.py",
        "args":      ["--with-2026"],
        "log_file":  None,
        "before_src": None,
    },
    {
        "key":       "f5",
        "label":     "F5 Total",
        "script":    "train_f5_model.py",
        "args":      ["--with-2026"],
        "log_file":  None,
        "before_src": None,
    },
    {
        "key":       "nrfi",
        "label":     "NRFI",
        "script":    "train_nrfi_model.py",
        "args":      ["--with-2026"],
        "log_file":  None,
        "before_src": None,
    },
    {
        "key":       "tb60",
        "label":     "TB v6.0",
        "script":    "retrain_tb_v60.py",
        "args":      [],
        "log_file":  _LOGS / "tb_v60_metrics.txt",
        "before_src": "log_file",
    },
    {
        "key":       "k_over",
        "label":     "K-Over v1",
        "script":    "train_k_over_v1.py",
        "args":      [],
        "log_file":  _LOGS / "k_over_v1_metrics.txt",
        "before_src": "log_file",
    },
    {
        "key":       "script_a",
        "label":     "Script A (Dominance)",
        "script":    "train_script_a.py",
        "args":      [],
        "log_file":  _LOGS / "script_a_v1_metrics.txt",
        "before_src": "log_file",
    },
    {
        "key":       "script_a2",
        "label":     "Script A2 (Dom F5)",
        "script":    "train_script_a2.py",
        "args":      [],
        "log_file":  _LOGS / "script_a2_v1_metrics.txt",
        "before_src": "log_file",
    },
    {
        "key":       "script_b",
        "label":     "Script B (Explosion)",
        "script":    "train_script_b.py",
        "args":      [],
        "log_file":  _LOGS / "script_b_v1_metrics.txt",
        "before_src": "log_file",
    },
    {
        "key":       "script_c",
        "label":     "Script C (Elite Duel)",
        "script":    "train_script_c.py",
        "args":      [],
        "log_file":  _LOGS / "script_c_v1_metrics.txt",
        "before_src": "log_file",
    },
    {
        "key":       "rl_v1",
        "label":     "RL v1 Stacker",
        "script":    "train_rl_v1.py",
        "args":      [],
        "log_file":  _LOGS / "rl_v1_metrics.txt",
        "before_src": "log_file",
    },
]


# ---------------------------------------------------------------------------
# AUC parsers
# ---------------------------------------------------------------------------

def _parse_auc_from_log_file(path: Path) -> float | None:
    """Read a structured metrics .txt file and extract the auc: line."""
    if not path or not path.exists():
        return None
    txt = path.read_text(errors="ignore")
    m = re.search(r"auc:\s*([\d.]+)", txt, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # fallback: AUC= pattern
    m = re.search(r"AUC[=:]\s*([\d.]+)", txt)
    if m:
        return float(m.group(1))
    return None


def _parse_auc_from_stdout(text: str) -> float | None:
    """
    Try several common patterns used across training scripts:
      Validation 2025  ...  AUC=0.5839
      AUC:  0.5721
      auc: 0.57538
      Stacker L2    AUC=0.5839
    """
    # Prefer val-2025 stacker line first
    patterns = [
        r"Stacker L2\s+AUC=([\d.]+)",
        r"Validation 2025.*?AUC=([\d.]+)",
        r"\bAUC[=:]\s*([\d.]+)",
        r"\bauc[=:]\s*([\d.]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return float(m.group(1))
    return None


def _before_auc(entry: dict) -> float | None:
    src = entry.get("before_src")
    if src == "log_file":
        return _parse_auc_from_log_file(entry.get("log_file"))
    if src == "v2_meta_auc":
        meta = _MODELS / "v2_meta.json"
        if meta.exists():
            d = json.loads(meta.read_text())
            return d.get("ml_auc_v2") or d.get("ml_auc_v1")
    return None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_script(entry: dict, dry_run: bool = False) -> tuple[bool, str, float]:
    """
    Returns (success, stdout_text, after_auc).
    Saves stdout to data/logs/retrain_{key}_{date}.log.
    """
    script = _ROOT / entry["script"]
    cmd    = [sys.executable, str(script)] + entry["args"]
    log_out = _LOGS / f"retrain_{entry['key']}_{TODAY}.log"

    print(f"\n  {'[DRY-RUN] ' if dry_run else ''}Running: {entry['script']} {' '.join(entry['args'])}")
    print(f"  Log: {log_out.name}")

    if dry_run:
        return True, "", None

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=1800,   # 30-min hard limit per model
            encoding="utf-8",
            errors="replace",
        )
        stdout = result.stdout + ("\n[STDERR]\n" + result.stderr if result.stderr.strip() else "")
        log_out.write_text(stdout, encoding="utf-8")

        elapsed = time.time() - t0
        success = result.returncode == 0
        if not success:
            print(f"  [ERROR] exit code {result.returncode} after {elapsed:.0f}s")
            print(f"  Last 5 lines: " + "\n    ".join((result.stderr or result.stdout).strip().splitlines()[-5:]))
        else:
            print(f"  [OK] {elapsed:.0f}s")

        after_auc = _parse_auc_from_stdout(stdout)

        # Also refresh from log file if it was written by the script itself
        if entry.get("log_file") and entry["log_file"].exists():
            parsed = _parse_auc_from_log_file(entry["log_file"])
            if parsed is not None:
                after_auc = parsed

        return success, stdout, after_auc

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {entry['script']} exceeded 30 minutes — killed")
        return False, "", None
    except Exception as exc:
        print(f"  [EXCEPTION] {exc}")
        return False, "", None


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _fmt(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "  N/A "


def print_comparison(results: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("  RETRAIN FIT COMPARISON")
    print("=" * 72)
    header = f"  {'Model':<26}  {'Before':>8}  {'After':>8}  {'Delta':>8}  Status"
    print(header)
    print("  " + "-" * 68)

    for r in results:
        before = r["before_auc"]
        after  = r["after_auc"]
        delta  = (after - before) if (before is not None and after is not None) else None
        delta_str = (f"{delta:+.4f}" if delta is not None else "    N/A ")
        tag = ""
        if delta is not None:
            tag = "UP" if delta > 0.001 else ("DOWN" if delta < -0.001 else "FLAT")
        status = r["status"]
        print(f"  {r['label']:<26}  {_fmt(before):>8}  {_fmt(after):>8}  {delta_str:>8}  {status} {tag}")

    print("=" * 72)


def save_comparison_csv(results: list[dict]) -> None:
    import csv
    out = _LOGS / f"retrain_comparison_{TODAY}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key","label","before_auc","after_auc","delta","status"])
        w.writeheader()
        for r in results:
            b, a = r["before_auc"], r["after_auc"]
            w.writerow({
                "key":        r["key"],
                "label":      r["label"],
                "before_auc": f"{b:.4f}" if b is not None else "",
                "after_auc":  f"{a:.4f}" if a is not None else "",
                "delta":      f"{a-b:+.4f}" if (a is not None and b is not None) else "",
                "status":     r["status"],
            })
    print(f"\n  Saved -> {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Full pipeline retrain + fit comparison")
    parser.add_argument("--dry-run",      action="store_true", help="Print plan only")
    parser.add_argument("--skip-current", action="store_true",
                        help="Skip models whose log file mtime is today")
    parser.add_argument("--models", type=str, default="",
                        help="Comma-separated keys to run (default: all)")
    args = parser.parse_args()

    selected_keys = {k.strip() for k in args.models.split(",") if k.strip()} if args.models else None

    plan = [e for e in PLAN if (selected_keys is None or e["key"] in selected_keys)]

    if args.skip_current:
        def _is_fresh(entry):
            lf = entry.get("log_file")
            if lf and lf.exists():
                mtime = datetime.fromtimestamp(lf.stat().st_mtime).date()
                return mtime == date.today()
            # also check retrain_{key}_{today}.log
            run_log = _LOGS / f"retrain_{entry['key']}_{TODAY}.log"
            return run_log.exists()
        plan = [e for e in plan if not _is_fresh(e)]

    print("=" * 72)
    print("  MLB Model Pipeline — Full Retrain")
    print(f"  Date: {TODAY}  |  Models: {len(plan)}")
    print("=" * 72)

    results = []
    for entry in plan:
        before_auc = _before_auc(entry)
        print(f"\n  [{entry['key']}] {entry['label']}")
        print(f"  Before AUC: {_fmt(before_auc)}")

        success, stdout, after_auc = run_script(entry, dry_run=args.dry_run)
        if not args.dry_run:
            print(f"  After  AUC: {_fmt(after_auc)}")

        results.append({
            "key":        entry["key"],
            "label":      entry["label"],
            "before_auc": before_auc,
            "after_auc":  after_auc,
            "status":     "OK" if success else "FAILED",
        })

    if not args.dry_run:
        print_comparison(results)
        save_comparison_csv(results)
    else:
        print("\n[DRY-RUN] Plan:")
        for e in plan:
            print(f"  {e['key']:<12}  {e['script']} {' '.join(e['args'])}")


if __name__ == "__main__":
    main()
