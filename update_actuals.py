"""
update_actuals.py — Persistent residual logger.

After drift_monitor.refresh_actuals() writes new box-score rows into
actuals_2026.parquet, this script joins them with archived totals predictions
(projected_total_adj) and appends per-game residuals to:

    data/logs/model_residuals.csv  [date, game_id, home_park_id, residual]

residual = projected_total_adj - actual_total

A positive residual means the model over-projected (we hit the Over too often
in theory but the game went Under). get_dynamic_bias() in score_run_dist_today
consumes this file to self-heal the Totals thermostat.

Idempotent: re-running the same date overwrites that date's rows only.

Usage:
    python update_actuals.py                 # default: yesterday
    python update_actuals.py --date 2026-04-22
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

_ROOT     = Path(__file__).resolve().parent
_ARCHIVE  = _ROOT / "model_scores_archive.csv"
_ACTUALS  = _ROOT / "data" / "statcast" / "actuals_2026.parquet"
_BACKTEST = _ROOT / "backtest_2026_results.csv"  # bootstrap source for history
_LOG_DIR  = _ROOT / "data" / "logs"
_RESID    = _LOG_DIR / "model_residuals.csv"

sys.path.insert(0, str(_ROOT / "wizard_agents"))
try:
    from tools.implementations import TEAM_NAME_TO_ABBR
except Exception:
    TEAM_NAME_TO_ABBR = {}


def _home_abbr_from_game(g: str) -> str | None:
    if not isinstance(g, str) or "@" not in g:
        return None
    tail = g.split("@", 1)[1].strip()
    # If already an abbreviation (e.g. "TEX"), pass through.
    if len(tail) <= 4 and tail.isupper():
        return tail
    return TEAM_NAME_TO_ABBR.get(tail)


def _load_archive_preds() -> pd.DataFrame:
    if not _ARCHIVE.exists():
        return pd.DataFrame(columns=["date", "game_id", "home_park_id", "projected_total"])
    a = pd.read_csv(_ARCHIVE)
    a = a[a["model"] == "Totals"].dropna(subset=["projected_total_adj"]).copy()
    a = a.drop_duplicates(subset=["date", "game"], keep="last")
    a["home_park_id"] = a["game"].map(_home_abbr_from_game)
    a["game_id"] = a["date"].astype(str) + "|" + a["game"].astype(str)
    a = a.rename(columns={"projected_total_adj": "projected_total"})
    return a[["date", "game_id", "home_park_id", "projected_total"]]


def _load_backtest_preds() -> pd.DataFrame:
    """Bootstrap residuals from backtest_2026_results (model_total + actual_total
    already joined). Provides history before model_scores_archive was populated."""
    if not _BACKTEST.exists():
        return pd.DataFrame()
    d = pd.read_csv(_BACKTEST)
    if not {"date", "game", "model_total", "actual_total"} <= set(d.columns):
        return pd.DataFrame()
    d = d.dropna(subset=["model_total", "actual_total"]).copy()
    d["home_park_id"] = d["game"].str.split("@").str[-1].str.strip()
    d["game_id"] = d["date"].astype(str) + "|" + d["game"].astype(str)
    d["residual"] = d["model_total"] - d["actual_total"]
    return d[["date", "game_id", "home_park_id", "residual"]]


def _load_actuals() -> pd.DataFrame:
    if not _ACTUALS.exists():
        print(f"  [ERROR] actuals source missing: {_ACTUALS}. "
              f"Upstream drift_monitor.refresh_actuals() never ran. "
              f"No residuals can be computed until this is populated.")
        return pd.DataFrame()
    a = pd.read_parquet(_ACTUALS)
    a["date"] = pd.to_datetime(a["game_date"]).dt.strftime("%Y-%m-%d")
    a["actual_total"] = a["home_score_final"] + a["away_score_final"]
    return a[["date", "home_team", "actual_total"]].rename(
        columns={"home_team": "home_park_id"}
    )


def build_residuals(target_date: str | None = None) -> pd.DataFrame:
    """Return the incremental residual rows for target_date (or yesterday)."""
    d = target_date or (date.today() - timedelta(days=1)).isoformat()
    preds = _load_archive_preds()
    acts  = _load_actuals()
    if preds.empty or acts.empty:
        return pd.DataFrame(columns=["date", "game_id", "home_park_id", "residual"])
    preds_d = preds[preds["date"] == d]
    if preds_d.empty:
        return pd.DataFrame(columns=["date", "game_id", "home_park_id", "residual"])
    j = preds_d.merge(acts, on=["date", "home_park_id"], how="inner")
    j["residual"] = j["projected_total"] - j["actual_total"]
    return j[["date", "game_id", "home_park_id", "residual"]]


def upsert(new_rows: pd.DataFrame) -> int:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    if new_rows.empty:
        return 0
    if _RESID.exists():
        existing = pd.read_csv(_RESID)
        existing = existing[~existing["game_id"].isin(new_rows["game_id"])]
        out = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    else:
        out = new_rows.copy()
    out = out.sort_values(["date", "game_id"]).reset_index(drop=True)
    out.to_csv(_RESID, index=False)
    return len(new_rows)


def bootstrap_from_backtest() -> int:
    """Seed the residuals log from backtest_2026 so get_dynamic_bias has
    history on day one. Safe to re-run — game_ids collide and upsert replaces."""
    bt = _load_backtest_preds()
    if bt.empty:
        return 0
    return upsert(bt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: yesterday)")
    ap.add_argument("--bootstrap", action="store_true",
                    help="Seed residuals log from backtest_2026_results.csv first.")
    args = ap.parse_args()

    if args.bootstrap:
        n = bootstrap_from_backtest()
        print(f"[update_actuals] bootstrapped {n} rows from backtest_2026_results.csv")

    new = build_residuals(args.date)
    n = upsert(new)
    print(f"[update_actuals] wrote {n} residual rows for {args.date or 'yesterday'} "
          f"-> {_RESID}")


if __name__ == "__main__":
    main()
