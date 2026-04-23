"""
drift_monitor.py — Totals model drift watchdog.

Joins every archived model_scores_<date>.csv with actuals_2026.parquet,
isolates Totals/OVER picks (one row per game), and computes:
    bias = mean(actual_total - projected_total_adj)
    mae  = mean(|actual_total - projected_total_adj|)

If the rolling-10-game bias exceeds ±0.5 runs, emits a CRITICAL alert.
Runs stand-alone as the last step of the daily pipeline.

Usage:
    python drift_monitor.py                # rolling 10-game window
    python drift_monitor.py --window 20    # custom window
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_ROOT     = Path(__file__).resolve().parent
_ACTUALS  = _ROOT / "data" / "statcast" / "actuals_2026.parquet"
_ARCHIVE  = _ROOT / "model_scores_archive"
_LIVE     = _ROOT / "model_scores.csv"

DRIFT_BIAS_THRESHOLD = 0.5   # runs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("drift")


def _load_scored_totals() -> pd.DataFrame:
    """Pull one Totals/OVER row per (date, game) from archives + live file."""
    frames: list[pd.DataFrame] = []
    archive_dir = _ROOT / "model_scores_archive"
    if archive_dir.exists():
        for f in sorted(archive_dir.glob("model_scores_*.csv")):
            try:
                frames.append(pd.read_csv(f))
            except Exception as e:
                log.warning(f"skip {f.name}: {e}")
    if _LIVE.exists():
        try:
            frames.append(pd.read_csv(_LIVE))
        except Exception as e:
            log.warning(f"skip live {_LIVE.name}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)
    if "model" not in df.columns:
        return pd.DataFrame()
    df = df[(df["model"] == "Totals") &
            (df.get("pick_direction") == "OVER")].copy()
    if "projected_total_adj" not in df.columns:
        log.warning("projected_total_adj absent — older archives; drift will use subset.")
        df["projected_total_adj"] = None
    df["projected_total_adj"] = pd.to_numeric(df["projected_total_adj"], errors="coerce")
    df = df.dropna(subset=["projected_total_adj"])
    df = df.drop_duplicates(subset=["date", "game"], keep="last")
    return df[["date", "game", "projected_total_adj"]]


def _load_actuals() -> pd.DataFrame:
    if not _ACTUALS.exists():
        log.warning(f"actuals parquet missing: {_ACTUALS} — drift cannot be evaluated.")
        return pd.DataFrame()
    a = pd.read_parquet(_ACTUALS)
    need = {"home_score_final", "away_score_final"}
    if not need.issubset(a.columns):
        log.warning(f"actuals missing final-score cols; have {list(a.columns)[:8]}...")
        return pd.DataFrame()
    a["actual_total"] = (pd.to_numeric(a["home_score_final"], errors="coerce") +
                         pd.to_numeric(a["away_score_final"], errors="coerce"))
    a = a.dropna(subset=["actual_total"])
    # Build a joinable game label: "Away @ Home" using whatever team cols exist.
    home_col = "home_team" if "home_team" in a.columns else None
    away_col = "away_team" if "away_team" in a.columns else None
    if home_col and away_col:
        a["game"] = a[away_col].astype(str) + " @ " + a[home_col].astype(str)
    a["date"] = pd.to_datetime(a["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return a[["date", "game", "actual_total"]].dropna()


def compute_drift(window: int = 10) -> dict:
    preds = _load_scored_totals()
    acts  = _load_actuals()
    if preds.empty or acts.empty:
        log.info(f"[Drift] insufficient data: preds={len(preds)} actuals={len(acts)}")
        return {"status": "NO_DATA", "preds": len(preds), "actuals": len(acts)}

    merged = preds.merge(acts, on=["date", "game"], how="inner")
    merged = merged.sort_values("date").tail(window).copy()
    if merged.empty:
        log.info("[Drift] no overlapping rows between predictions and actuals.")
        return {"status": "NO_OVERLAP"}

    merged["bias"] = merged["actual_total"] - merged["projected_total_adj"]
    bias = float(merged["bias"].mean())
    mae  = float(merged["bias"].abs().mean())
    n    = len(merged)

    log.info(f"[Drift] last {n} Totals games | bias={bias:+.3f} | MAE={mae:.3f}")

    alert = False
    if abs(bias) > DRIFT_BIAS_THRESHOLD:
        alert = True
        direction = "UNDER-predicting" if bias > 0 else "OVER-predicting"
        log.critical(
            f"CRITICAL TOTALS DRIFT — bias {bias:+.3f} runs over last {n} games "
            f"(|bias| > {DRIFT_BIAS_THRESHOLD}). Model is {direction} scoring."
        )

    return {
        "status": "OK",
        "window":    n,
        "bias":      round(bias, 3),
        "mae":       round(mae, 3),
        "alert":     alert,
        "threshold": DRIFT_BIAS_THRESHOLD,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=10)
    args = ap.parse_args()
    result = compute_drift(window=args.window)
    print(result)


if __name__ == "__main__":
    main()
