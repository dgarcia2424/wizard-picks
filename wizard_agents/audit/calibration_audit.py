"""
calibration_audit.py — Weekly calibration + ballpark-bias audit.

Joins model_scores_archive.csv with actuals_2026.parquet on (date, home_team)
and reports:
  1. Global Totals bias (Pred - Actual) and MAE.
  2. Ballpark Bias Report: bias grouped by home_team (park proxy) — flags
     any park with |bias| > 0.5 runs as a candidate for prior adjustment.

Rationale: if v2 is 'hallucinating' wind/temp effects at specific stadiums
(e.g. Coors baked too high, or domes baked too low), the grouped bias will
surface it before the bankroll does.

Usage:  python -m wizard_agents.audit.calibration_audit
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_ROOT     = Path(__file__).resolve().parents[2]
_ARCHIVE  = _ROOT / "model_scores_archive.csv"
_ACTUALS  = _ROOT / "data" / "statcast" / "actuals_2026.parquet"
_BACKTEST = _ROOT / "backtest_2026_results.csv"  # historical overlay (has model_total + actual_total)

BIAS_FLAG_THRESHOLD = 0.5  # runs/game — parks above this get flagged

# game string in archive is "Away Team @ Home Team" (full names).
# actuals uses team abbreviations. Import the canonical map.
import sys
sys.path.insert(0, str(_ROOT / "wizard_agents"))
try:
    from tools.implementations import TEAM_NAME_TO_ABBR
except Exception:
    TEAM_NAME_TO_ABBR = {}


def _home_abbr_from_game(game_str: str) -> str | None:
    if not isinstance(game_str, str) or "@" not in game_str:
        return None
    home_full = game_str.split("@", 1)[1].strip()
    return TEAM_NAME_TO_ABBR.get(home_full)


def _load_totals_preds() -> pd.DataFrame:
    df = pd.read_csv(_ARCHIVE)
    df = df[df["model"] == "Totals"].copy()
    df = df.dropna(subset=["projected_total_adj"])
    # One prediction per (date, game) — take latest archive entry.
    df = df.drop_duplicates(subset=["date", "game"], keep="last")
    df["home_team_abbr"] = df["game"].map(_home_abbr_from_game)
    return df[["date", "game", "home_team_abbr", "projected_total_adj"]]


def _load_actuals() -> pd.DataFrame:
    a = pd.read_parquet(_ACTUALS)
    a["date"] = pd.to_datetime(a["game_date"]).dt.strftime("%Y-%m-%d")
    a["actual_total"] = a["home_score_final"] + a["away_score_final"]
    return a[["date", "home_team", "actual_total"]].rename(
        columns={"home_team": "home_team_abbr"}
    )


def _load_backtest_overlay() -> pd.DataFrame:
    """Historical Totals predictions from backtest_2026_results.csv — these
    already carry actual_total, so we use them directly for the audit as
    the archive only retains projected_total_adj from 2026-04-23 forward."""
    if not _BACKTEST.exists():
        return pd.DataFrame(columns=["date", "home_team_abbr", "error"])
    d = pd.read_csv(_BACKTEST)
    if not {"date", "game", "model_total", "actual_total"} <= set(d.columns):
        return pd.DataFrame(columns=["date", "home_team_abbr", "error"])
    d = d.dropna(subset=["model_total", "actual_total"]).copy()
    # backtest 'game' is already "AWAY @ HOME" abbreviation form.
    d["home_team_abbr"] = d["game"].str.split("@").str[-1].str.strip()
    d["error"] = d["model_total"] - d["actual_total"]
    return d[["date", "home_team_abbr", "error"]]


def run_audit() -> dict:
    preds = _load_totals_preds()
    acts  = _load_actuals()
    arch = preds.merge(acts, on=["date", "home_team_abbr"], how="inner")
    if not arch.empty:
        arch = arch.assign(error=arch["projected_total_adj"] - arch["actual_total"])
        arch = arch[["date", "home_team_abbr", "error"]]
        print(f"[archive] matched {len(arch)} rows from model_scores_archive")
    else:
        arch = pd.DataFrame(columns=["date", "home_team_abbr", "error"])
        print("[archive] 0 matches — projected_total_adj only populated from 2026-04-23; "
              "archive will grow forward.")

    overlay = _load_backtest_overlay()
    if not overlay.empty:
        print(f"[overlay] loaded {len(overlay)} rows from backtest_2026_results.csv "
              f"({overlay['date'].min()} → {overlay['date'].max()})")

    j = pd.concat([arch, overlay], ignore_index=True).dropna(subset=["home_team_abbr", "error"])
    if j.empty:
        print("[calibration_audit] no data — neither archive nor backtest provided matches.")
        return {"n": 0}

    g_bias = float(j["error"].mean())
    g_mae  = float(j["error"].abs().mean())
    g_rmse = float(np.sqrt((j["error"] ** 2).mean()))
    print(f"[global] n={len(j)}  Bias={g_bias:+.3f}  MAE={g_mae:.3f}  RMSE={g_rmse:.3f}")
    print(f"         date range: {j['date'].min()} → {j['date'].max()}")

    park = (j.groupby("home_team_abbr")
              .agg(n=("error", "size"),
                   bias=("error", "mean"),
                   mae=("error", lambda x: x.abs().mean()))
              .sort_values("bias", ascending=False))
    park["flag"] = park["bias"].abs() > BIAS_FLAG_THRESHOLD

    print("\n[ballpark-bias] Pred − Actual (runs/game), sorted by bias desc")
    print(park.round(3).to_string())

    flagged = park[park["flag"]]
    if not flagged.empty:
        print(f"\n⚠ {len(flagged)} park(s) with |bias| > {BIAS_FLAG_THRESHOLD}r — review priors:")
        for t, r in flagged.iterrows():
            direction = "over-pred" if r["bias"] > 0 else "under-pred"
            print(f"   {t}: {direction} by {abs(r['bias']):.2f}r over n={int(r['n'])}")
    else:
        print(f"\n✓ no parks exceed |bias| > {BIAS_FLAG_THRESHOLD}r threshold.")

    return {
        "n": int(len(j)),
        "global_bias": g_bias, "global_mae": g_mae, "global_rmse": g_rmse,
        "park_bias": park.to_dict("index"),
        "flagged_parks": flagged.index.tolist(),
    }


if __name__ == "__main__":
    run_audit()
