"""
retrain_all.py — Walk-forward retraining orchestrator.

Phase 1: Train on 2023+2024 -> validate on 2025. Copy models to models/v3/phase1/
Phase 2: Train on 2023+2024+2025 -> validate on 2026. Copy to models/v3/phase2/

Prints AUC comparison table at the end.

Usage:
    python retrain_all.py [--phase 1|2|both] [--backup] [--dry-run]
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASE   = Path(__file__).resolve().parent
MATRIX = "feature_matrix_enriched_v2.parquet"

# ── Model files each training script produces ───────────────────────────────

_RL_ML_FILES = [
    "xgb_rl.json", "xgb_ml_v2.json", "xgb_ml.json",
    "calibrator_rl.pkl", "calibrator_ml.pkl",
    "stacking_lr_rl.pkl", "stacking_lr_rl.npz",
    "lgbm_rl.pkl", "lgbm_ml.pkl",
    "cat_rl.pkl", "cat_ml.pkl",
    "feature_cols.json", "ml_feature_cols_v2.json", "ml_feature_cols.json",
    "xgb_rl_team.json", "feature_cols_team.json",
]
_F5_FILES = [
    "xgb_f5.json", "xgb_f5_calibrator.pkl",
    "stacking_lr_f5.pkl", "stacking_lr_f5.npz",
    "f5_feature_cols.json", "f5_feature_cols_v1.json",
    "f5_pois_home.json", "f5_pois_away.json",
]
_NRFI_FILES = [
    "xgb_nrfi.json", "xgb_nrfi_calibrator.pkl",
    "stacking_lr_nrfi.pkl", "stacking_lr_nrfi.npz",
    "nrfi_feature_cols.json",
    "xgb_pois_f1_home.json", "xgb_pois_f1_away.json",
]
_KOVER_FILES = [
    "k_over_v1.json",
]
_ML_FILES = [
    "xgb_ml.json", "xgb_ml_calibrator.pkl",
    "stacking_lr_ml.pkl", "stacking_lr_ml.npz",
    "ml_feature_cols.json",
    "team_ml_model.json", "team_ml_feat_cols.json",
]
_RUNDIST_FILES = [
    "dc_model_run_dist.pkl",
    "xgb_run_dist_lam_home.json", "xgb_run_dist_lam_away.json",
    "stacker_totals.pkl", "stacker_totals.npz",
    "stacker_rl.pkl", "stacker_rl.npz",
    "run_dist_feature_cols.json",
]
_RL_V1_FILES = [
    "rl_v1_stacker.json", "rl_v1_feature_cols.json",
]
_SCRIPT_A_FILES  = ["script_a_v1.json"]
_SCRIPT_A2_FILES = ["script_a2_v1.json"]
_SCRIPT_B_FILES  = ["script_b_v1.json"]
_SCRIPT_C_FILES  = ["script_c_v1.json"]
_TB35_FILES = ["tb_stacker_v35.json"]
_TB37_FILES = ["tb_stacker_v37.json"]

TB_MATRIX = "data/batter_features/final_training_matrix.parquet"

SCRIPTS: list[dict] = [
    {
        "name":         "RL+ML (xgboost)",
        "script":       "train_xgboost.py",
        "files":        _RL_ML_FILES,
        "phases":       [1, 2],
        "extra":        [],
        "phase2_extra": [],
    },
    {
        "name":         "F5",
        "script":       "train_f5_model.py",
        "files":        _F5_FILES,
        "phases":       [1, 2],
        "extra":        [],
        "phase2_extra": ["--with-2026"],
    },
    {
        "name":         "NRFI",
        "script":       "train_nrfi_model.py",
        "files":        _NRFI_FILES,
        "phases":       [1, 2],
        "extra":        [],
        "phase2_extra": ["--with-2026"],
    },
    {
        "name":         "K-Over",
        "script":       "train_k_over_v1.py",
        "files":        _KOVER_FILES,
        "phases":       [1, 2],
        "extra":        ["--matrix", str(BASE / MATRIX)],
        "no_matrix":    True,  # matrix passed via extra above
    },
    {
        "name":         "ML (standalone)",
        "script":       "train_ml_model.py",
        "files":        _ML_FILES,
        "phases":       [1, 2],
        "extra":        [],
        "phase2_extra": ["--with-2026"],
    },
    {
        "name":         "Run Distribution",
        "script":       "train_run_dist_model.py",
        "files":        _RUNDIST_FILES,
        "phases":       [1, 2],
        "extra":        [],
        "phase2_extra": ["--with-2026"],
    },
    {
        "name":         "RL v1 Stacker",
        "script":       "train_rl_v1.py",
        "files":        _RL_V1_FILES,
        "phases":       [1, 2],
        "extra":        [],
        "phase2_extra": ["--with-2026"],
        "no_matrix":    True,  # uses FM_PATH global, overridden via --matrix
    },
    {
        "name":         "Script A (SP K + Under)",
        "script":       "train_script_a.py",
        "files":        _SCRIPT_A_FILES,
        "phases":       [1, 2],
        "extra":        [],
        "phase2_extra": ["--with-2026"],
    },
    {
        "name":         "Script A2 (F5 K + F5 Under)",
        "script":       "train_script_a2.py",
        "files":        _SCRIPT_A2_FILES,
        "phases":       [1, 2],
        "extra":        [],
        "no_matrix":    True,
    },
    {
        "name":         "Script B (Offensive Explosion)",
        "script":       "train_script_b.py",
        "files":        _SCRIPT_B_FILES,
        "phases":       [1, 2],
        "extra":        [],
        "no_matrix":    True,
    },
    {
        "name":         "Script C (Elite Duel)",
        "script":       "train_script_c.py",
        "files":        _SCRIPT_C_FILES,
        "phases":       [1, 2],
        "extra":        [],
        "no_matrix":    True,
    },
    {
        "name":         "TB v3.5",
        "script":       "train_tb_v35.py",
        "files":        _TB35_FILES,
        "phases":       [1, 2],
        "extra":        [],
        "no_matrix":    True,  # uses final_training_matrix, not feature_matrix
    },
    {
        "name":         "TB v3.7",
        "script":       "train_tb_v37.py",
        "files":        _TB37_FILES,
        "phases":       [1, 2],
        "extra":        [],
        "no_matrix":    True,
    },
]

PHASE_VAL_YEAR = {1: 2025, 2: 2026}


# ── AUC parsing helpers ──────────────────────────────────────────────────────

def _extract_auc(stdout: str, script_name: str) -> float | None:
    """Pull the best/final AUC from a script's stdout."""
    # train_xgboost: "AUC-ROC     : 0.XXXX"
    m = re.findall(r"AUC-ROC\s*:\s*([\d.]+)", stdout)
    if m:
        return float(m[-1])

    # train_f5 / train_nrfi: "AUC=0.XXXX"
    m = re.findall(r"\bAUC=([\d.]+)", stdout)
    if m:
        return float(m[-1])

    # train_k_over: "  auc: 0.XXXX"
    m = re.findall(r"\bauc:\s*([\d.]+)", stdout)
    if m:
        return float(m[-1])

    return None


# ── Backup ───────────────────────────────────────────────────────────────────

def backup_models() -> Path:
    src  = BASE / "models"
    dest = BASE / f"models_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if src.exists():
        shutil.copytree(src, dest)
        print(f"  Backed up models/ -> {dest.name}")
    else:
        print("  models/ not found — skipping backup")
    return dest


# ── Copy model files after a training run ───────────────────────────────────

def copy_phase_models(phase_dir: Path, file_list: list[str]) -> None:
    phase_dir.mkdir(parents=True, exist_ok=True)
    src_dir = BASE / "models"
    copied = 0
    for fname in file_list:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, phase_dir / fname)
            copied += 1
    print(f"    Copied {copied}/{len(file_list)} model files -> {phase_dir.relative_to(BASE)}")


# ── Run one training script ──────────────────────────────────────────────────

def run_script(script: dict, val_year: int, dry_run: bool) -> tuple[bool, float | None]:
    """
    Run a training script with --val-year and --matrix.
    Returns (success, auc).
    """
    cmd = [sys.executable, str(BASE / script["script"]),
           "--val-year", str(val_year)]
    if not script.get("no_matrix"):
        cmd += ["--matrix", str(BASE / MATRIX)]
    cmd.extend(script["extra"])
    if val_year == PHASE_VAL_YEAR[2]:
        cmd.extend(script.get("phase2_extra", []))

    print(f"\n  > {' '.join(cmd[1:])}")  # skip python path for brevity

    if dry_run:
        print("    [dry-run] skipped")
        return True, None

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(BASE),
    )
    stdout = result.stdout + result.stderr

    # Stream to console in real time would require Popen; for now show tail
    lines = stdout.splitlines()
    for line in lines[-30:]:
        print(f"    | {line}")

    success = result.returncode == 0
    auc = _extract_auc(stdout, script["name"])
    return success, auc


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Walk-forward retraining orchestrator")
    parser.add_argument("--phase", choices=["1", "2", "both"], default="both",
                        help="Which phase(s) to run (default: both)")
    parser.add_argument("--backup", action="store_true",
                        help="Back up current models/ before starting")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    phases = [1, 2] if args.phase == "both" else [int(args.phase)]

    print("=" * 70)
    print("  retrain_all.py — walk-forward model retraining")
    print(f"  Phases: {phases}   Matrix: {MATRIX}")
    print("=" * 70)

    if args.backup:
        backup_models()

    results: dict[int, dict[str, tuple[bool, float | None]]] = {}

    for phase in phases:
        val_year  = PHASE_VAL_YEAR[phase]
        phase_dir = BASE / "models" / "v3" / f"phase{phase}"
        results[phase] = {}

        print(f"\n{'-'*70}")
        print(f"  PHASE {phase}: train <= {val_year-1}  ->  validate {val_year}")
        print(f"{'-'*70}")

        for script in SCRIPTS:
            if phase not in script["phases"]:
                print(f"\n  [{script['name']}] SKIPPED (no {val_year} labels)")
                results[phase][script["name"]] = (None, None)
                continue

            print(f"\n  [{script['name']}]")
            ok, auc = run_script(script, val_year, dry_run=args.dry_run)
            results[phase][script["name"]] = (ok, auc)

            if ok:
                copy_phase_models(phase_dir, script["files"])
                print(f"    AUC: {auc:.4f}" if auc is not None else "    AUC: (not parsed)")
            else:
                print(f"    !! FAILED (return code non-zero)")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")

    all_names = [s["name"] for s in SCRIPTS]
    header = f"  {'Signal':<18}  " + "  ".join(
        f"Ph{p} AUC (val={PHASE_VAL_YEAR[p]})" for p in phases)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name in all_names:
        row = f"  {name:<18}"
        for phase in phases:
            ok, auc = results[phase].get(name, (None, None))
            if ok is None:
                row += f"  {'SKIP':>20}"
            elif not ok:
                row += f"  {'FAILED':>20}"
            elif auc is None:
                row += f"  {'ok (no parse)':>20}"
            else:
                row += f"  {auc:>20.4f}"
        print(row)

    print(f"\n  Phase 1 models -> models/v3/phase1/")
    if 2 in phases:
        print(f"  Phase 2 models -> models/v3/phase2/")
    print(f"\n  To promote Phase 2 to production:")
    print(f"    cp models/v3/phase2/* models/")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
