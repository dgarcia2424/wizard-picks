"""
verify_f5_parity.py
===================
Parity harness for the pruned score_f5_today.py.

Workflow
--------
1. Load a legacy output CSV produced by the PRE-PRUNE score_f5_today.py
   for a given date.  Must contain at minimum:
     home_team, away_team, xgb_l1, stacker_l2
   (and ideally team_log_odds).

2. Import and invoke the refactored predict_games(date_str) from the
   CURRENT score_f5_today.py to generate a fresh DataFrame.

3. Inner-join legacy and new on (home_team, away_team).

4. numpy.testing.assert_allclose on xgb_l1 and stacker_l2 at atol=1e-6.
   (Note: predict_games rounds these columns to 3 decimals in the output,
    so any residual float drift is bounded below ±5e-4 by construction —
    an atol of 1e-6 is effectively a bit-equality check for this pipeline.)

5. Also diff team_log_odds (rounded to 3 decimals) when present.

6. Exits 0 on PASS with a clear stdout banner; exits 1 on FAIL and dumps
   the offending rows + the largest |delta| per column.

Usage
-----
    python verify_f5_parity.py --legacy legacy_f5_output.csv --date 2026-04-19

The --date must match the date the legacy CSV was generated for.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Keys used to join legacy vs. new.  No game_pk in the script output,
# so we fall back to the team pair (unique per date).
JOIN_KEYS = ["home_team", "away_team"]

# Columns we will assert equality on.  xgb_l1 = Platt-calibrated L1 prob,
# stacker_l2 = Bayesian L2 output.  team_log_odds is optional bonus.
NUMERIC_COLS = ["xgb_l1", "stacker_l2"]
OPTIONAL_COLS = ["team_log_odds"]

# predict_games() rounds xgb_l1, stacker_l2, team_log_odds to 3 decimals,
# so any value-level drift below 5e-4 is hidden by rounding.  Use a tight
# tolerance to make any deterministic drift trip the harness.
ATOL = 1e-6
RTOL = 0.0


def _load_legacy(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Legacy output CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in JOIN_KEYS + NUMERIC_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Legacy CSV missing required columns: {missing}\n"
            f"Columns present: {list(df.columns)}"
        )
    return df


def _invoke_new_pipeline(date_str: str) -> pd.DataFrame:
    """Import the refactored module and call predict_games(date_str)."""
    # Import fresh so any code change in score_f5_today.py is picked up.
    sys.path.insert(0, str(Path(__file__).parent))
    if "score_f5_today" in sys.modules:
        del sys.modules["score_f5_today"]
    import score_f5_today  # noqa: E402

    df = score_f5_today.predict_games(date_str)
    if df is None or len(df) == 0:
        raise RuntimeError("predict_games returned an empty DataFrame.")
    # Drop rows that the pipeline flagged as unscored.
    df = df.dropna(subset=["xgb_l1", "stacker_l2"]).copy()
    return df


def _join(legacy: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    legacy_k = legacy[JOIN_KEYS].apply(tuple, axis=1)
    new_k    = new[JOIN_KEYS].apply(tuple, axis=1)
    only_legacy = set(legacy_k) - set(new_k)
    only_new    = set(new_k) - set(legacy_k)
    if only_legacy:
        print(f"[WARN] {len(only_legacy)} rows present in legacy but not in new: "
              f"{sorted(only_legacy)[:5]}{' …' if len(only_legacy) > 5 else ''}")
    if only_new:
        print(f"[WARN] {len(only_new)} rows present in new but not in legacy: "
              f"{sorted(only_new)[:5]}{' …' if len(only_new) > 5 else ''}")

    merged = legacy.merge(
        new, on=JOIN_KEYS, how="inner", suffixes=("_legacy", "_new")
    )
    if len(merged) == 0:
        raise RuntimeError(
            "Inner join on (home_team, away_team) produced 0 rows. "
            "Is the --date argument aligned with the legacy CSV?"
        )
    return merged


def _assert_column(merged: pd.DataFrame, col: str) -> tuple[bool, float, pd.DataFrame]:
    a = merged[f"{col}_legacy"].to_numpy(dtype=float)
    b = merged[f"{col}_new"].to_numpy(dtype=float)
    nan_mask = np.isnan(a) | np.isnan(b)
    if nan_mask.any():
        print(f"[WARN] {col}: {int(nan_mask.sum())} row(s) contain NaN on one side — dropping")
        a = a[~nan_mask]
        b = b[~nan_mask]
        m = merged.loc[~nan_mask].reset_index(drop=True)
    else:
        m = merged

    delta = np.abs(a - b)
    max_delta = float(delta.max()) if len(delta) else 0.0
    ok = len(delta) > 0 and np.all(delta <= ATOL + RTOL * np.abs(b))

    if ok:
        return True, max_delta, pd.DataFrame()

    # Build diff report
    bad_idx = np.where(delta > ATOL + RTOL * np.abs(b))[0]
    diff = m.loc[bad_idx, JOIN_KEYS + [f"{col}_legacy", f"{col}_new"]].copy()
    diff[f"{col}_delta"] = delta[bad_idx]
    diff = diff.sort_values(f"{col}_delta", ascending=False).reset_index(drop=True)
    return False, max_delta, diff


def main() -> int:
    global ATOL  # noqa: PLW0603
    ap = argparse.ArgumentParser(description="F5 pipeline parity harness")
    ap.add_argument("--legacy", required=True, type=Path,
                    help="Path to legacy score_f5_today.py output CSV")
    ap.add_argument("--date", required=True,
                    help="Date the legacy CSV was generated for (YYYY-MM-DD)")
    ap.add_argument("--atol", type=float, default=ATOL,
                    help=f"Absolute tolerance (default {ATOL})")
    args = ap.parse_args()
    ATOL = args.atol

    print("=" * 72)
    print(f"F5 Parity Harness   legacy={args.legacy.name}   date={args.date}   atol={ATOL:g}")
    print("=" * 72)

    # 1. Legacy
    print(f"\n[1/4] Loading legacy CSV: {args.legacy}")
    legacy = _load_legacy(args.legacy)
    print(f"      rows={len(legacy)}  cols={list(legacy.columns)}")

    # 2. New pipeline
    print(f"\n[2/4] Invoking refactored predict_games('{args.date}') …")
    try:
        new = _invoke_new_pipeline(args.date)
    except Exception:
        traceback.print_exc()
        print("\nFAIL: refactored predict_games raised an exception.")
        return 1
    print(f"      rows={len(new)}  cols={list(new.columns)}")

    # 3. Join
    print("\n[3/4] Joining on (home_team, away_team) …")
    merged = _join(legacy, new)
    print(f"      joined rows={len(merged)}")

    # 4. Assert
    print(f"\n[4/4] Comparing numeric outputs (atol={ATOL:g}, rtol={RTOL:g}) …")
    any_fail = False
    report: list[tuple[str, float]] = []

    cols_to_check = list(NUMERIC_COLS)
    for opt in OPTIONAL_COLS:
        if f"{opt}_legacy" in merged.columns and f"{opt}_new" in merged.columns:
            cols_to_check.append(opt)

    for col in cols_to_check:
        if f"{col}_legacy" not in merged.columns or f"{col}_new" not in merged.columns:
            print(f"  [skip] {col}: not present in both frames")
            continue
        ok, max_delta, diff = _assert_column(merged, col)
        status = "PASS" if ok else "FAIL"
        report.append((col, max_delta))
        print(f"  [{status}] {col:<15s}  max |Δ| = {max_delta:.3e}")
        if not ok:
            any_fail = True
            print(f"         offending rows (top 10 by |Δ|):")
            print(diff.head(10).to_string(index=False))

    # Try a stricter numpy.testing.assert_allclose as a final belt-and-braces
    # check on the two required columns, so the CI log records the exact
    # exception payload numpy emits.
    if not any_fail:
        for col in NUMERIC_COLS:
            if f"{col}_legacy" in merged.columns and f"{col}_new" in merged.columns:
                a = merged[f"{col}_legacy"].to_numpy(dtype=float)
                b = merged[f"{col}_new"].to_numpy(dtype=float)
                mask = ~(np.isnan(a) | np.isnan(b))
                np.testing.assert_allclose(a[mask], b[mask], atol=ATOL, rtol=RTOL,
                                           err_msg=f"{col} parity assert_allclose failed")

    print("\n" + "=" * 72)
    if any_fail:
        print("FAIL: Mathematical parity broken. See offending rows above.")
        print("Summary of max |Δ| per column:")
        for col, d in report:
            print(f"  {col:<15s}  {d:.3e}")
        print("=" * 72)
        return 1

    print("PASS: Mathematical parity confirmed")
    for col, d in report:
        print(f"  {col:<15s}  max |Δ| = {d:.3e}  (atol={ATOL:g})")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
