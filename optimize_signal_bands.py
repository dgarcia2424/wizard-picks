"""
optimize_signal_bands.py
========================
Grid-searches optimal green/yellow/red cutoffs for every model signal,
using the same ROI-maximization approach as optimize_locks.py.

Signals optimized
-----------------
  rl_edge       RL home/away +1.5 edge  (already in optimize_locks; kept here for unified output)
  ml_edge       Moneyline home/away edge
  ou_edge       Over/Under edge
  k_over_edge   K-prop over edge (home SP + away SP)
  nrfi_prob     NRFI model probability (vs -110 baseline)
  f5_win_prob   F5 home win model probability
  script_a      Script A corr_edge  (SP K>=5.5 AND Game Total <8.0)
  script_a2     Script A2 corr_edge (SP K>=5.5 AND F5 Total <4.5)
  script_b      Script B corr_edge  (SP K>=5.5 AND Game Total >8.5)
  script_c      Script C corr_edge  (Both SP 6+ IP AND Total <8.0)

Data sources
------------
  daily_cards/daily_card_2026-*.csv     — model signal columns
  data/statcast/actuals_2026.parquet    — game outcomes
  data/sgp/sgp_live_edge_2026-*.csv    — historical SGP script scores (grows over season)

Output
------
  signal_bands.json   — per-signal optimized thresholds, read by report/dashboard

Usage
-----
  python optimize_signal_bands.py            # optimize all signals
  python optimize_signal_bands.py --signal k_over_edge   # single signal
  python optimize_signal_bands.py --dry-run  # show data summary only
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT    = Path(__file__).resolve().parent
_CARDS   = _ROOT / "daily_cards"
_ACTUALS = _ROOT / "data" / "statcast" / "actuals_2026.parquet"
_SGP_DIR = _ROOT / "data" / "sgp"
_OUT     = _ROOT / "signal_bands.json"

# Minimum bets required before trusting optimization results
MIN_SAMPLE = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def american_to_decimal(odds: float) -> float:
    if odds >= 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def roi_from_bets(bets: pd.DataFrame) -> float:
    """bets must have columns: stake, odds, actual (0/1)."""
    resolved = bets.dropna(subset=["actual"])
    if len(resolved) == 0:
        return np.nan
    staked = resolved["stake"].sum()
    if staked == 0:
        return np.nan
    pl = sum(
        r["stake"] * american_to_decimal(r["odds"]) if r["actual"]
        else -r["stake"]
        for _, r in resolved.iterrows()
    )
    return pl / staked


def brier(y_true, y_prob) -> float:
    y, p = np.asarray(y_true, float), np.asarray(y_prob, float)
    mask = ~(np.isnan(y) | np.isnan(p))
    return float(np.mean((y[mask] - p[mask]) ** 2)) if mask.sum() > 0 else np.nan


def win_rate(y_true) -> float:
    arr = np.asarray(y_true, float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if len(arr) > 0 else np.nan


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cards() -> pd.DataFrame:
    """Load all 2026 daily cards into one DataFrame."""
    files = sorted(glob.glob(str(_CARDS / "daily_card_2026-*.csv")))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        d["_date"] = Path(f).stem.replace("daily_card_", "")
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def load_actuals() -> pd.DataFrame:
    if not _ACTUALS.exists():
        return pd.DataFrame()
    act = pd.read_parquet(_ACTUALS)
    act["game_date"] = pd.to_datetime(act["game_date"]).dt.strftime("%Y-%m-%d")
    return act


def load_sgp_history() -> pd.DataFrame:
    """
    Load all historical SGP live-edge files (dated filenames).
    Files named: sgp_live_edge_2026_04_24.csv  (note: underscore-separated date)
    or:          sgp_live_edge_2026-04-24.csv
    """
    frames = []
    for pattern in ["sgp_live_edge_2026_*.csv", "sgp_live_edge_2026-*.csv"]:
        for f in sorted(glob.glob(str(_SGP_DIR / pattern))):
            stem = Path(f).stem
            # skip steam file: sgp_live_edge_steam_2026_*
            if "steam" in stem:
                continue
            parts = stem.replace("-", "_").split("_")
            # expected: sgp_live_edge_2026_MM_DD
            year_idx = next((i for i, p in enumerate(parts) if p == "2026"), None)
            if year_idx is None or year_idx + 2 >= len(parts):
                continue
            try:
                game_date = f"2026-{parts[year_idx+1]:0>2}-{parts[year_idx+2]:0>2}"
            except Exception:
                continue
            try:
                d = pd.read_csv(f)
                d["game_date"] = game_date
                frames.append(d)
            except Exception:
                continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def merge_cards_actuals(cards: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Join daily cards with actuals on (home_team, _date / game_date)."""
    if cards.empty or actuals.empty:
        return cards
    act = actuals.rename(columns={"game_date": "_date"})
    keep = ["_date", "home_team", "away_team",
            "home_sp_k", "away_sp_k", "home_sp_ip", "away_sp_ip",
            "f1_nrfi", "f5_home_win", "f5_total",
            "home_score_final", "away_score_final", "home_covers_rl"]
    keep = [c for c in keep if c in act.columns]
    merged = cards.merge(act[keep], on=["_date", "home_team"], how="left", suffixes=("", "_act"))
    return merged


# ---------------------------------------------------------------------------
# Per-signal optimizers
# ---------------------------------------------------------------------------

def _grid_search_edge(rows: pd.DataFrame,
                      edge_col: str,
                      prob_col: str,
                      outcome_col: str,
                      odds_col: str | None,
                      default_odds: float = -110.0,
                      tier2_grid=None,
                      tier1_grid=None,
                      label: str = "") -> dict:
    """
    Generic grid search: find (tier2, tier1) edge thresholds that maximize ROI.
    Red  = edge < tier2
    Yellow = tier2 <= edge < tier1
    Green  = edge >= tier1
    """
    if tier2_grid is None:
        tier2_grid = [0.005, 0.010, 0.015, 0.020, 0.030]
    if tier1_grid is None:
        tier1_grid = [0.020, 0.030, 0.040, 0.050, 0.060, 0.080]

    keep = list(dict.fromkeys(c for c in [edge_col, prob_col, outcome_col, odds_col]
                              if c is not None and c in rows.columns))
    df = rows[keep].dropna(subset=[outcome_col, edge_col])
    df = df.reset_index(drop=True)
    df = df[pd.to_numeric(df[edge_col], errors="coerce") > -1.0].copy()

    n_total = len(df)
    if n_total < MIN_SAMPLE:
        return {
            "tier2": tier2_grid[0], "tier1": tier1_grid[-1],
            "n_games": n_total, "roi_best": None, "win_rate": None,
            "note": f"insufficient data (n={n_total})"
        }

    best_roi, best_t2, best_t1, best_n = -np.inf, tier2_grid[0], tier1_grid[-1], 0

    for t2, t1 in itertools.product(tier2_grid, tier1_grid):
        if t2 >= t1:
            continue
        sub = df[df[edge_col] >= t2].copy()
        if len(sub) < MIN_SAMPLE:
            continue
        odds_vals = (sub[odds_col].astype(float) if odds_col and odds_col in sub.columns
                     else pd.Series(default_odds, index=sub.index))
        stake = 10.0  # fixed stake for comparison
        sub = sub.assign(stake=stake, odds=odds_vals)
        r = roi_from_bets(sub.rename(columns={outcome_col: "actual"}))
        if r is not None and not np.isnan(r) and r > best_roi:
            best_roi, best_t2, best_t1, best_n = r, t2, t1, len(sub)

    wr = win_rate(df[df[edge_col] >= best_t2][outcome_col])
    note = f"n={n_total}" if n_total >= 150 else f"n={n_total} — directional only (target: 150+)"

    return {
        "tier2":    round(best_t2, 4),
        "tier1":    round(best_t1, 4),
        "n_games":  n_total,
        "roi_best": round(best_roi, 4) if best_roi > -np.inf else None,
        "win_rate": round(wr, 4) if not np.isnan(wr) else None,
        "note":     note,
    }


def _grid_search_prob(rows: pd.DataFrame,
                      prob_col: str,
                      outcome_col: str,
                      odds_col: str | None,
                      default_odds: float = -110.0,
                      prob_grid=None,
                      label: str = "") -> dict:
    """
    For probability-based signals (NRFI, F5): find the min-prob thresholds
    where betting above that threshold produces positive ROI.
    Maps naturally to edge = prob - implied_odds_prob.
    """
    if prob_grid is None:
        prob_grid = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65]

    df = rows[[c for c in [prob_col, outcome_col, odds_col]
               if c is not None and c in rows.columns]].dropna(subset=[outcome_col, prob_col])

    n_total = len(df)
    if n_total < MIN_SAMPLE:
        return {
            "tier2": prob_grid[0], "tier1": prob_grid[2],
            "n_games": n_total, "roi_best": None, "win_rate": None,
            "note": f"insufficient data (n={n_total})"
        }

    best_roi, best_t2, best_t1 = -np.inf, prob_grid[0], prob_grid[2]

    for t2, t1 in itertools.product(prob_grid, prob_grid):
        if t2 >= t1:
            continue
        sub = df[df[prob_col] >= t2].copy()
        if len(sub) < MIN_SAMPLE:
            continue
        odds_vals = (sub[odds_col].astype(float) if odds_col and odds_col in sub.columns
                     else pd.Series(default_odds, index=sub.index))
        sub = sub.assign(stake=10.0, odds=odds_vals)
        r = roi_from_bets(sub.rename(columns={outcome_col: "actual"}))
        if r is not None and not np.isnan(r) and r > best_roi:
            best_roi, best_t2, best_t1 = r, t2, t1

    wr = win_rate(df[df[prob_col] >= best_t2][outcome_col])
    note = f"n={n_total}" if n_total >= 150 else f"n={n_total} — directional only (target: 150+)"

    return {
        "tier2":    round(best_t2, 4),
        "tier1":    round(best_t1, 4),
        "n_games":  n_total,
        "roi_best": round(best_roi, 4) if best_roi > -np.inf else None,
        "win_rate": round(wr, 4) if not np.isnan(wr) else None,
        "note":     note,
    }


# ---------------------------------------------------------------------------
# Signal-specific phases
# ---------------------------------------------------------------------------

def optimize_rl_edge(df: pd.DataFrame) -> dict:
    """RL: blended model prob vs retail implied, outcome = home_covers_rl."""
    rows = df.copy()
    if "lock_p_model" not in rows.columns or "home_covers_rl" not in rows.columns:
        return {"note": "missing columns"}
    rows["rl_edge"] = rows["lock_p_model"].astype(float) - rows["lock_retail_implied"].astype(float)
    rows["rl_outcome"] = rows["home_covers_rl"].astype(float)
    return _grid_search_edge(rows, "rl_edge", "lock_p_model", "rl_outcome",
                             "rl_odds", default_odds=-115.0, label="rl_edge")


def optimize_ml_edge(df: pd.DataFrame) -> dict:
    rows = df.copy()
    if "ml_lock_p_model" not in rows.columns:
        return {"note": "missing columns"}
    rows["ml_edge"] = rows["ml_lock_p_model"].astype(float) - rows["ml_lock_retail_implied"].astype(float)
    rows["ml_outcome"] = pd.to_numeric(rows.get("actual_home_win", np.nan), errors="coerce")
    # derive from scores if not present
    if rows["ml_outcome"].isna().all() and "home_score_final" in rows.columns:
        rows["ml_outcome"] = (rows["home_score_final"] > rows["away_score_final"]).astype(float)
    return _grid_search_edge(rows, "ml_edge", "ml_lock_p_model", "ml_outcome",
                             "vegas_ml_home", default_odds=-110.0, label="ml_edge")


def optimize_ou_edge(df: pd.DataFrame) -> dict:
    rows = df.copy()
    if "ou_p_model" not in rows.columns:
        return {"note": "missing columns"}
    rows["ou_edge"] = rows["ou_p_model"].astype(float) - rows["ou_p_retail"].astype(float)
    # compute actual O/U outcome
    if "home_score_final" in rows.columns and "away_score_final" in rows.columns:
        actual_total = rows["home_score_final"].astype(float) + rows["away_score_final"].astype(float)
        posted = pd.to_numeric(rows["ou_posted_line"], errors="coerce")
        direction = rows.get("ou_direction", pd.Series("OVER", index=rows.index))
        rows["ou_outcome"] = np.where(
            direction == "OVER",
            (actual_total > posted).astype(float),
            (actual_total < posted).astype(float),
        )
        rows.loc[posted.isna() | actual_total.isna(), "ou_outcome"] = np.nan
    else:
        return {"note": "missing score actuals"}
    return _grid_search_edge(rows, "ou_edge", "ou_p_model", "ou_outcome",
                             None, default_odds=-110.0, label="ou_edge")


def optimize_k_over_edge(df: pd.DataFrame) -> dict:
    """K-Over: expand to one row per SP (home + away)."""
    rows_list = []
    for _, r in df.iterrows():
        for side in ("home", "away"):
            edge = pd.to_numeric(r.get(f"{side}_k_edge"), errors="coerce")
            prob = pd.to_numeric(r.get(f"{side}_k_model_over"), errors="coerce")
            line = pd.to_numeric(r.get(f"{side}_k_line"), errors="coerce")
            odds = pd.to_numeric(r.get(f"{side}_k_over_odds"), errors="coerce")
            actual_k = pd.to_numeric(r.get(f"{side}_sp_k"), errors="coerce")
            if pd.isna(edge) or pd.isna(line) or pd.isna(actual_k):
                continue
            outcome = float(actual_k > line) if not pd.isna(actual_k) else np.nan
            rows_list.append({"k_edge": edge, "k_prob": prob,
                               "k_outcome": outcome, "k_odds": odds if not pd.isna(odds) else -120.0})
    if not rows_list:
        return {"note": "no K-over rows"}
    sub = pd.DataFrame(rows_list)
    return _grid_search_edge(sub, "k_edge", "k_prob", "k_outcome", "k_odds",
                             default_odds=-120.0,
                             tier2_grid=[0.02, 0.04, 0.06, 0.08],
                             tier1_grid=[0.06, 0.08, 0.10, 0.12, 0.15],
                             label="k_over_edge")


def optimize_nrfi_prob(df: pd.DataFrame) -> dict:
    """
    Two-sided: bet NRFI when prob > threshold_high, bet YRFI when prob < threshold_low.
    We convert to edge = |prob - 0.5238| (implied at -110 juice = 52.38%).
    Grid-search the confidence threshold that maximises ROI.
    """
    rows = df.copy()
    if "mc_nrfi_prob" not in rows.columns or "f1_nrfi" not in rows.columns:
        return {"note": "missing columns"}
    rows["nrfi_prob"] = pd.to_numeric(rows["mc_nrfi_prob"], errors="coerce")
    rows["f1_nrfi"]   = pd.to_numeric(rows["f1_nrfi"], errors="coerce")
    rows = rows.dropna(subset=["nrfi_prob", "f1_nrfi"]).reset_index(drop=True)
    if len(rows) < MIN_SAMPLE:
        return {"tier2": 0.54, "tier1": 0.60, "n_games": len(rows),
                "roi_best": None, "win_rate": None, "note": f"n={len(rows)}"}

    JUICE_IMPLIED = 100 / (100 + 110)  # -110 → 52.38%
    rows["nrfi_edge"] = rows["nrfi_prob"].abs() - JUICE_IMPLIED  # confidence above juice

    # Build bet rows: always bet toward model's direction
    bet_rows = []
    for _, r in rows.iterrows():
        p = float(r["nrfi_prob"])
        if p >= 0.5:                   # model says NRFI
            bet_rows.append({"edge": p - JUICE_IMPLIED, "prob": p,
                              "outcome": r["f1_nrfi"], "odds": -110.0})
        else:                          # model says YRFI
            bet_rows.append({"edge": (1 - p) - JUICE_IMPLIED, "prob": 1 - p,
                              "outcome": 1 - r["f1_nrfi"], "odds": -110.0})
    bets = pd.DataFrame(bet_rows)

    tier2_grid = [0.00, 0.01, 0.02, 0.03, 0.05]
    tier1_grid = [0.03, 0.05, 0.07, 0.10, 0.13]
    best_roi, best_t2, best_t1 = -np.inf, tier2_grid[0], tier1_grid[1]

    for t2, t1 in itertools.product(tier2_grid, tier1_grid):
        if t2 >= t1:
            continue
        sub = bets[bets["edge"] >= t2].copy().rename(columns={"outcome": "actual"})
        sub["stake"] = 10.0
        if len(sub) < MIN_SAMPLE:
            continue
        r = roi_from_bets(sub)
        if r is not None and not np.isnan(r) and r > best_roi:
            best_roi, best_t2, best_t1 = r, t2, t1

    wr = win_rate(bets[bets["edge"] >= best_t2]["outcome"])
    note = f"n={len(rows)}" if len(rows) >= 150 else f"n={len(rows)} — directional only"
    return {
        "tier2": round(best_t2, 4), "tier1": round(best_t1, 4),
        "n_games": len(rows),
        "roi_best": round(best_roi, 4) if best_roi > -np.inf else None,
        "win_rate": round(wr, 4) if not np.isnan(wr) else None,
        "note": note,
    }


def optimize_f5_win_prob(df: pd.DataFrame) -> dict:
    rows = df.copy()
    prob_col = "f5_stacker_l2" if "f5_stacker_l2" in rows.columns else "mc_f5_home_win_prob"
    if prob_col not in rows.columns or "f5_home_win" not in rows.columns:
        return {"note": "missing columns"}
    rows["f5_prob"] = pd.to_numeric(rows[prob_col], errors="coerce")
    rows["f5_outcome"] = pd.to_numeric(rows["f5_home_win"], errors="coerce")
    return _grid_search_prob(rows, "f5_prob", "f5_outcome", None,
                             default_odds=-110.0, label="f5_win_prob")


def optimize_script(sgp_df: pd.DataFrame, actuals: pd.DataFrame,
                    script_name: str) -> dict:
    """
    SGP Scripts: join historical sgp_live_edge rows with actuals,
    compute whether both legs hit, grid-search corr_edge threshold.

    Script name must match the 'script' column in sgp_live_edge files:
      'A_Dominance', 'A2_DomF5', 'B_Explosion', 'C_EliteDuel'
    """
    if sgp_df.empty or actuals.empty:
        return {"note": "no historical SGP data yet — accumulates over season"}

    sub = sgp_df[sgp_df["script"] == script_name].copy() if "script" in sgp_df.columns else pd.DataFrame()
    if sub.empty:
        return {"note": f"no rows for script={script_name}"}

    # Join to actuals
    act = actuals.rename(columns={"game_date": "game_date"})
    merged = sub.merge(act, on=["game_date", "home_team"], how="left", suffixes=("", "_act"))
    merged = merged.reset_index(drop=True)

    # Compute joint outcome per script
    if script_name == "A_Dominance":
        k_line = pd.to_numeric(merged.get("home_k_line", 5.5), errors="coerce").fillna(5.5)
        total_line = 8.0
        merged["joint_hit"] = (
            (merged["home_sp_k"].astype(float) >= k_line) &
            (merged["home_score_final"].astype(float) + merged["away_score_final"].astype(float) < total_line)
        ).astype(float)
    elif script_name == "A2_DomF5":
        k_line = pd.to_numeric(merged.get("home_k_line", 5.5), errors="coerce").fillna(5.5)
        merged["joint_hit"] = (
            (merged["home_sp_k"].astype(float) >= k_line) &
            (merged["f5_total"].astype(float) < 4.5)
        ).astype(float)
    elif script_name == "B_Explosion":
        k_line = pd.to_numeric(merged.get("home_k_line", 5.5), errors="coerce").fillna(5.5)
        total_line = 8.5
        merged["joint_hit"] = (
            (merged["home_sp_k"].astype(float) >= k_line) &
            (merged["home_score_final"].astype(float) + merged["away_score_final"].astype(float) > total_line)
        ).astype(float)
    elif script_name == "C_EliteDuel":
        ip_min = 6.0
        total_line = 8.0
        merged["joint_hit"] = (
            (merged["home_sp_ip"].astype(float) >= ip_min) &
            (merged["away_sp_ip"].astype(float) >= ip_min) &
            (merged["home_score_final"].astype(float) + merged["away_score_final"].astype(float) < total_line)
        ).astype(float)
    else:
        return {"note": f"unknown script: {script_name}"}

    merged = merged.dropna(subset=["joint_hit"])
    if len(merged) < MIN_SAMPLE:
        return {
            "tier2": 0.010, "tier1": 0.030,
            "n_games": len(merged), "roi_best": None, "win_rate": None,
            "note": f"n={len(merged)} — accumulates over season"
        }

    edge_col = "sgp_edge" if "sgp_edge" in merged.columns else "corr_edge" if "corr_edge" in merged.columns else None
    if edge_col is None:
        return {"note": "no edge column in SGP data"}

    merged[edge_col] = pd.to_numeric(merged[edge_col], errors="coerce")
    return _grid_search_edge(
        merged, edge_col, edge_col, "joint_hit", None,
        default_odds=400.0,   # SGP typically pays ~+400 to +600
        tier2_grid=[0.005, 0.010, 0.015, 0.020],
        tier1_grid=[0.020, 0.030, 0.040, 0.050, 0.060],
        label=f"script_{script_name}",
    )


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

ALL_SIGNALS = [
    "rl_edge", "ml_edge", "ou_edge",
    "k_over_edge", "nrfi_prob", "f5_win_prob",
    "script_a", "script_a2", "script_b", "script_c",
]

SCRIPT_NAMES = {
    "script_a":  "A_Dominance",
    "script_a2": "A2_DomF5",
    "script_b":  "B_Explosion",
    "script_c":  "C_EliteDuel",
}


def run_optimization(signals: list[str], dry_run: bool = False) -> dict:
    print("=" * 68)
    print("  Signal Band Optimizer")
    print(f"  Date: {date.today().isoformat()}")
    print("=" * 68)

    # Load data
    cards   = load_cards()
    actuals = load_actuals()
    sgp_history = load_sgp_history()

    print(f"\n  Daily cards: {len(cards)} rows across "
          f"{cards['_date'].nunique() if not cards.empty else 0} dates")
    print(f"  Actuals 2026: {len(actuals)} games")
    print(f"  SGP history: {len(sgp_history)} script rows across "
          f"{sgp_history['game_date'].nunique() if not sgp_history.empty else 0} dates")

    if dry_run:
        print("\n[DRY-RUN] Data loaded. Exiting.")
        return {}

    if cards.empty:
        print("\n  [ERROR] No daily cards found.")
        return {}

    # Merge cards + actuals
    df = merge_cards_actuals(cards, actuals)
    print(f"\n  Merged: {len(df)} rows | "
          f"{df['home_covers_rl'].notna().sum() if 'home_covers_rl' in df.columns else 0} "
          f"RL resolved | "
          f"{df['f1_nrfi'].notna().sum() if 'f1_nrfi' in df.columns else 0} NRFI resolved")

    results = {}
    today = date.today().isoformat()

    signal_fns = {
        "rl_edge":     lambda: optimize_rl_edge(df),
        "ml_edge":     lambda: optimize_ml_edge(df),
        "ou_edge":     lambda: optimize_ou_edge(df),
        "k_over_edge": lambda: optimize_k_over_edge(df),
        "nrfi_prob":   lambda: optimize_nrfi_prob(df),
        "f5_win_prob": lambda: optimize_f5_win_prob(df),
        "script_a":    lambda: optimize_script(sgp_history, actuals, "A_Dominance"),
        "script_a2":   lambda: optimize_script(sgp_history, actuals, "A2_DomF5"),
        "script_b":    lambda: optimize_script(sgp_history, actuals, "B_Explosion"),
        "script_c":    lambda: optimize_script(sgp_history, actuals, "C_EliteDuel"),
    }

    for sig in signals:
        fn = signal_fns.get(sig)
        if fn is None:
            print(f"\n  [{sig}] unknown signal — skipping")
            continue

        print(f"\n  [{sig}]", end=" ", flush=True)
        r = fn()
        r["last_optimized"] = today
        results[sig] = r

        if "note" in r and "tier2" not in r:
            print(r["note"])
        else:
            t2 = r.get("tier2", "?")
            t1 = r.get("tier1", "?")
            n  = r.get("n_games", "?")
            roi = r.get("roi_best")
            wr  = r.get("win_rate")
            roi_str = f"ROI={roi:+.1%}" if roi is not None else "ROI=N/A"
            wr_str  = f"WR={wr:.1%}"   if wr  is not None else "WR=N/A"
            print(f"red<{t2:.3f}  yellow<{t1:.3f}  green>={t1:.3f} | "
                  f"n={n}  {roi_str}  {wr_str}")

    return results


def save_bands(results: dict) -> None:
    # Load existing file so we can preserve signals not re-optimized this run
    existing = {}
    if _OUT.exists():
        try:
            existing = json.loads(_OUT.read_text())
        except Exception:
            pass
    existing.update(results)
    _OUT.write_text(json.dumps(existing, indent=2))
    print(f"\n  Saved -> {_OUT.name}")


def print_summary(results: dict) -> None:
    print("\n" + "=" * 68)
    print("  SIGNAL BAND SUMMARY")
    print("  (Red = below tier2 | Yellow = tier2 to tier1 | Green = above tier1)")
    print("=" * 68)
    print(f"  {'Signal':<18}  {'Red <':>8}  {'Yellow <':>10}  {'Green >=':>10}  {'n':>6}  Note")
    print("  " + "-" * 64)
    for sig, r in results.items():
        if "tier2" not in r:
            print(f"  {sig:<18}  {'N/A':>8}  {'N/A':>10}  {'N/A':>10}  {'?':>6}  {r.get('note','')}")
        else:
            t2 = r["tier2"]
            t1 = r["tier1"]
            n  = r.get("n_games", "?")
            note = r.get("note", "")
            print(f"  {sig:<18}  {t2:>8.3f}  {t1:>10.3f}  {t1:>10.3f}  {n:>6}  {note}")
    print("=" * 68)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize signal band thresholds")
    parser.add_argument("--signal", type=str, default="",
                        help="Comma-separated signal names (default: all)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    signals = (
        [s.strip() for s in args.signal.split(",") if s.strip()]
        if args.signal else ALL_SIGNALS
    )

    results = run_optimization(signals, dry_run=args.dry_run)

    if results and not args.dry_run:
        print_summary(results)
        if not args.no_save:
            save_bands(results)


if __name__ == "__main__":
    main()
