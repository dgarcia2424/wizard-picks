"""
optimize_signal_bands.py
========================
Grid-searches three confidence cutoffs for every model signal.

Three tiers
-----------
  watch  — flag for tracking; no bet placed
  small  — bet at 0.5x unit (half Kelly)
  full   — bet at 1.0x unit (full Kelly)

Objective: maximize weighted P&L using tiered staking across all resolved bets.

Signals optimized
-----------------
  rl_edge       RL home/away +1.5 edge
  ml_win_prob   Moneyline home win probability
  ou_edge       Over/Under edge
  k_over_edge   K-prop over edge (home SP + away SP)
  nrfi_prob     NRFI model probability (vs -110 baseline)
  f5_win_prob   F5 home win model probability
  script_a      Script A corr_edge  (SP K>=5.5 AND Game Total <8.0)
  script_a2     Script A2 corr_edge (SP K>=5.5 AND F5 Total <4.5)
  script_b      Script B corr_edge  (SP K>=5.5 AND Game Total >8.5)
  script_c      Script C corr_edge  (Both SP 6+ IP AND Total <8.0)

Output
------
  signal_bands.json   — per-signal three-tier thresholds

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

MIN_SAMPLE = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def american_to_decimal(odds: float) -> float:
    if odds >= 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def _weighted_roi(df: pd.DataFrame, edge_col: str, outcome_col: str,
                  odds_col: str | None, default_odds: float,
                  t_small: float, t_full: float) -> float:
    """
    Simulate tiered staking on rows with edge >= t_small.
      edge >= t_full  → stake = 1.0
      t_small <= edge < t_full → stake = 0.5
    Returns ROI on total staked (NaN if no resolved bets).
    """
    sub = df[df[edge_col] >= t_small].copy()
    if len(sub) < MIN_SAMPLE:
        return np.nan
    sub = sub.dropna(subset=[outcome_col])
    if len(sub) < MIN_SAMPLE:
        return np.nan

    sub["_stake"] = np.where(sub[edge_col] >= t_full, 1.0, 0.5)
    odds_vals = (sub[odds_col].astype(float)
                 if odds_col and odds_col in sub.columns
                 else pd.Series(default_odds, index=sub.index))
    total_staked = sub["_stake"].sum()
    if total_staked == 0:
        return np.nan
    pl = sum(
        row["_stake"] * american_to_decimal(float(odds_vals[idx])) if row[outcome_col]
        else -row["_stake"]
        for idx, row in sub.iterrows()
    )
    return pl / total_staked


def win_rate(y_true) -> float:
    arr = np.asarray(y_true, float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if len(arr) > 0 else np.nan


# ---------------------------------------------------------------------------
# Core three-threshold grid search
# ---------------------------------------------------------------------------

def _grid_search_three(df: pd.DataFrame,
                       edge_col: str,
                       outcome_col: str,
                       odds_col: str | None,
                       default_odds: float = -110.0,
                       watch_grid=None,
                       small_grid=None,
                       full_grid=None,
                       label: str = "") -> dict:
    """
    Find (t_watch, t_small, t_full) that maximise weighted-stake ROI.
      Below t_watch         → ignore
      t_watch <= e < t_small → monitor only
      t_small <= e < t_full  → 0.5x stake
      e >= t_full            → 1.0x stake
    """
    if watch_grid is None:
        watch_grid = [0.005, 0.010, 0.015, 0.020, 0.030]
    if small_grid is None:
        small_grid = [0.015, 0.020, 0.030, 0.040, 0.050]
    if full_grid is None:
        full_grid  = [0.040, 0.050, 0.060, 0.080, 0.100]

    df = df.copy()
    df[edge_col] = pd.to_numeric(df[edge_col], errors="coerce")
    df = df.dropna(subset=[outcome_col, edge_col])
    df = df[df[edge_col] > -1.0].reset_index(drop=True)

    n_total = len(df)
    if n_total < MIN_SAMPLE:
        return {
            "watch": watch_grid[0], "small": small_grid[0], "full": full_grid[-1],
            "n_games": n_total,
            "roi_small": None, "roi_full": None,
            "wr_small": None, "wr_full": None,
            "note": f"insufficient data (n={n_total})"
        }

    best_roi   = -np.inf
    best_combo = (watch_grid[0], small_grid[0], full_grid[-1])

    for t_w, t_s, t_f in itertools.product(watch_grid, small_grid, full_grid):
        if not (t_w < t_s < t_f):
            continue
        r = _weighted_roi(df, edge_col, outcome_col, odds_col, default_odds, t_s, t_f)
        if r is not None and not np.isnan(r) and r > best_roi:
            best_roi   = r
            best_combo = (t_w, t_s, t_f)

    t_watch, t_small, t_full = best_combo

    roi_s = _weighted_roi(df, edge_col, outcome_col, odds_col, default_odds, t_small, t_full)
    roi_f_only = _weighted_roi(df[df[edge_col] >= t_full].copy() if len(df[df[edge_col] >= t_full]) >= MIN_SAMPLE
                               else pd.DataFrame(columns=df.columns),
                               edge_col, outcome_col, odds_col, default_odds, t_full, t_full + 1)
    wr_s = win_rate(df.loc[df[edge_col] >= t_small, outcome_col])
    wr_f = win_rate(df.loc[df[edge_col] >= t_full,  outcome_col])

    note = f"n={n_total}" if n_total >= 150 else f"n={n_total} — directional only (target: 150+)"

    return {
        "watch":    round(t_watch, 4),
        "small":    round(t_small, 4),
        "full":     round(t_full, 4),
        "n_games":  n_total,
        "n_small":  int((df[edge_col] >= t_small).sum()),
        "n_full":   int((df[edge_col] >= t_full).sum()),
        "roi_small": round(roi_s,    4) if roi_s    is not None and not np.isnan(roi_s)    else None,
        "roi_full":  round(roi_f_only, 4) if roi_f_only is not None and not np.isnan(roi_f_only) else None,
        "wr_small":  round(wr_s, 4) if not np.isnan(wr_s) else None,
        "wr_full":   round(wr_f, 4) if not np.isnan(wr_f) else None,
        "note":     note,
    }


def _grid_search_prob_three(df: pd.DataFrame,
                            prob_col: str,
                            outcome_col: str,
                            odds_col: str | None,
                            default_odds: float = -110.0,
                            watch_grid=None,
                            small_grid=None,
                            full_grid=None,
                            label: str = "") -> dict:
    """Same as _grid_search_three but the threshold column is a raw probability."""
    if watch_grid is None:
        watch_grid = [0.50, 0.51, 0.52, 0.53]
    if small_grid is None:
        small_grid = [0.52, 0.54, 0.56, 0.58]
    if full_grid is None:
        full_grid  = [0.56, 0.58, 0.60, 0.62, 0.65]
    return _grid_search_three(df, prob_col, outcome_col, odds_col,
                              default_odds, watch_grid, small_grid, full_grid, label)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cards() -> pd.DataFrame:
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
    frames = []
    for pattern in ["sgp_live_edge_2026_*.csv", "sgp_live_edge_2026-*.csv"]:
        for f in sorted(glob.glob(str(_SGP_DIR / pattern))):
            stem = Path(f).stem
            if "steam" in stem:
                continue
            parts = stem.replace("-", "_").split("_")
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
    if cards.empty or actuals.empty:
        return cards
    act = actuals.rename(columns={"game_date": "_date"})
    keep = ["_date", "home_team", "away_team",
            "home_sp_k", "away_sp_k", "home_sp_ip", "away_sp_ip",
            "f1_nrfi", "f5_home_win", "f5_total",
            "home_score_final", "away_score_final", "home_covers_rl"]
    keep = [c for c in keep if c in act.columns]
    return cards.merge(act[keep], on=["_date", "home_team"], how="left", suffixes=("", "_act"))


# ---------------------------------------------------------------------------
# Per-signal optimizers
# ---------------------------------------------------------------------------

def optimize_rl_prob(df: pd.DataFrame) -> dict:
    rows = df.copy()
    if "blended_rl" not in rows.columns or "home_covers_rl" not in rows.columns:
        return {"note": "missing columns"}
    rows["rl_prob"]    = pd.to_numeric(rows["blended_rl"], errors="coerce")
    rows["rl_outcome"] = rows["home_covers_rl"].astype(float)
    return _grid_search_prob_three(
        rows, "rl_prob", "rl_outcome", "rl_odds",
        default_odds=-115.0,
        watch_grid=[0.50, 0.51, 0.52],
        small_grid=[0.52, 0.54, 0.56, 0.58],
        full_grid=[0.58, 0.60, 0.62, 0.65],
        label="rl_prob",
    )


def optimize_ml_win_prob(df: pd.DataFrame) -> dict:
    rows = df.copy()
    if "mc_home_win" not in rows.columns:
        return {"note": "missing columns"}
    rows["ml_prob"] = pd.to_numeric(rows["mc_home_win"], errors="coerce")
    rows["ml_outcome"] = pd.to_numeric(rows.get("actual_home_win", np.nan), errors="coerce")
    if rows["ml_outcome"].isna().all() and "home_score_final" in rows.columns:
        rows["ml_outcome"] = (
            rows["home_score_final"].astype(float) > rows["away_score_final"].astype(float)
        ).astype(float)
    return _grid_search_prob_three(
        rows, "ml_prob", "ml_outcome", "vegas_ml_home",
        default_odds=-110.0,
        label="ml_win_prob",
    )


def optimize_ou_prob(df: pd.DataFrame) -> dict:
    rows = df.copy()
    if "ou_p_model" not in rows.columns:
        return {"note": "missing columns"}
    rows["ou_prob"] = pd.to_numeric(rows["ou_p_model"], errors="coerce")
    if "home_score_final" in rows.columns and "away_score_final" in rows.columns:
        actual_total = rows["home_score_final"].astype(float) + rows["away_score_final"].astype(float)
        posted    = pd.to_numeric(rows["ou_posted_line"], errors="coerce")
        direction = rows.get("ou_direction", pd.Series("OVER", index=rows.index))
        rows["ou_outcome"] = np.where(
            direction == "OVER",
            (actual_total > posted).astype(float),
            (actual_total < posted).astype(float),
        )
        rows.loc[posted.isna() | actual_total.isna(), "ou_outcome"] = np.nan
    else:
        return {"note": "missing score actuals"}
    return _grid_search_prob_three(
        rows, "ou_prob", "ou_outcome", None,
        default_odds=-110.0,
        label="ou_prob",
    )


def optimize_k_over_prob(df: pd.DataFrame) -> dict:
    rows_list = []
    for _, r in df.iterrows():
        for side in ("home", "away"):
            prob     = pd.to_numeric(r.get(f"{side}_k_model_over"), errors="coerce")
            line     = pd.to_numeric(r.get(f"{side}_k_line"),       errors="coerce")
            odds     = pd.to_numeric(r.get(f"{side}_k_over_odds"),  errors="coerce")
            actual_k = pd.to_numeric(r.get(f"{side}_sp_k"),         errors="coerce")
            if pd.isna(prob) or pd.isna(line) or pd.isna(actual_k):
                continue
            rows_list.append({
                "k_prob":    prob,
                "k_outcome": float(actual_k > line),
                "k_odds":    odds if not pd.isna(odds) else -120.0,
            })
    if not rows_list:
        return {"note": "no K-over rows"}
    sub = pd.DataFrame(rows_list)
    return _grid_search_prob_three(
        sub, "k_prob", "k_outcome", "k_odds",
        default_odds=-120.0,
        label="k_over_prob",
    )


def optimize_nrfi_prob(df: pd.DataFrame) -> dict:
    rows = df.copy()
    if "mc_nrfi_prob" not in rows.columns or "f1_nrfi" not in rows.columns:
        return {"note": "missing columns"}
    rows["nrfi_prob"] = pd.to_numeric(rows["mc_nrfi_prob"], errors="coerce")
    rows["f1_nrfi"]   = pd.to_numeric(rows["f1_nrfi"], errors="coerce")
    rows = rows.dropna(subset=["nrfi_prob", "f1_nrfi"]).reset_index(drop=True)
    if len(rows) < MIN_SAMPLE:
        return {
            "watch": 0.02, "small": 0.05, "full": 0.10,
            "n_games": len(rows), "roi_small": None, "roi_full": None,
            "wr_small": None, "wr_full": None, "note": f"n={len(rows)}",
        }

    JUICE_IMPLIED = 100 / (100 + 110)
    bet_rows = []
    for _, r in rows.iterrows():
        p = float(r["nrfi_prob"])
        if p >= 0.5:
            bet_rows.append({"edge": p - JUICE_IMPLIED, "outcome": r["f1_nrfi"],    "odds": -110.0})
        else:
            bet_rows.append({"edge": (1-p) - JUICE_IMPLIED, "outcome": 1 - r["f1_nrfi"], "odds": -110.0})

    bets = pd.DataFrame(bet_rows)
    return _grid_search_three(
        bets, "edge", "outcome", "odds",
        default_odds=-110.0,
        watch_grid=[0.00, 0.005, 0.010],
        small_grid=[0.010, 0.020, 0.030, 0.050],
        full_grid= [0.050, 0.070, 0.100, 0.130],
        label="nrfi_prob",
    )


def optimize_f5_win_prob(df: pd.DataFrame) -> dict:
    rows = df.copy()
    prob_col = "f5_stacker_l2" if "f5_stacker_l2" in rows.columns else "mc_f5_home_win_prob"
    if prob_col not in rows.columns or "f5_home_win" not in rows.columns:
        return {"note": "missing columns"}
    rows["f5_prob"]    = pd.to_numeric(rows[prob_col], errors="coerce")
    rows["f5_outcome"] = pd.to_numeric(rows["f5_home_win"], errors="coerce")
    return _grid_search_prob_three(
        rows, "f5_prob", "f5_outcome", None,
        default_odds=-110.0,
        label="f5_win_prob",
    )


def optimize_script(sgp_df: pd.DataFrame, actuals: pd.DataFrame,
                    script_name: str) -> dict:
    if sgp_df.empty or actuals.empty:
        return {"note": "no historical SGP data yet — accumulates over season"}

    sub = sgp_df[sgp_df["script"] == script_name].copy() if "script" in sgp_df.columns else pd.DataFrame()
    if sub.empty:
        return {"note": f"no rows for script={script_name}"}

    act = actuals.rename(columns={"game_date": "game_date"})
    merged = sub.merge(act, on=["game_date", "home_team"], how="left", suffixes=("", "_act"))
    merged = merged.reset_index(drop=True)

    import re

    def _parse_line(legs_str: str, pattern: str) -> float | None:
        m = re.search(pattern, str(legs_str))
        return float(m.group(1)) if m else None

    game_total = merged["home_score_final"].astype(float) + merged["away_score_final"].astype(float)

    if script_name == "A_Dominance":
        k_thresh  = merged["legs"].apply(lambda l: _parse_line(l, r"SP_K>=(\d+\.?\d*)")).fillna(5.5)
        g_thresh  = merged["legs"].apply(lambda l: _parse_line(l, r"Game_Total<([\d.]+)")).fillna(8.0)
        merged["joint_hit"] = (
            (merged["home_sp_k"].astype(float) >= k_thresh) &
            (game_total < g_thresh)
        ).astype(float)
        merged.loc[g_thresh.isna(), "joint_hit"] = np.nan

    elif script_name == "A2_Dominance":
        k_thresh  = merged["legs"].apply(lambda l: _parse_line(l, r"SP_K_F5>=(\d+)")).fillna(4.0)
        f5_thresh = merged["legs"].apply(lambda l: _parse_line(l, r"F5_Under_([\d.]+)"))
        g_thresh  = merged["legs"].apply(lambda l: _parse_line(l, r"Game_Under_([\d.]+)"))
        merged["joint_hit"] = (
            (merged["home_sp_k"].astype(float) >= k_thresh) &
            (merged["f5_total"].astype(float)  <  f5_thresh.fillna(99)) &
            (game_total < g_thresh.fillna(99))
        ).astype(float)
        merged.loc[f5_thresh.isna() | g_thresh.isna(), "joint_hit"] = np.nan

    elif script_name == "B_Explosion":
        g_thresh = merged["legs"].apply(lambda l: _parse_line(l, r"Game_Over_([\d.]+)"))
        merged["joint_hit"] = (
            (merged["home_score_final"].astype(float) >= 5.0) &
            (game_total > g_thresh.fillna(0)) &
            (merged["home_score_final"].astype(float) > merged["away_score_final"].astype(float))
        ).astype(float)
        merged.loc[g_thresh.isna(), "joint_hit"] = np.nan

    elif script_name == "C_EliteDuel":
        g_thresh = merged["legs"].apply(lambda l: _parse_line(l, r"Game_Under_([\d.]+)"))
        k_thresh = merged["legs"].apply(lambda l: _parse_line(l, r"SP_K_F5>=(\d+)")).fillna(3.0)
        merged["joint_hit"] = (
            (game_total < g_thresh.fillna(99)) &
            (merged["home_sp_k"].astype(float) >= k_thresh) &
            (merged["away_sp_k"].astype(float) >= k_thresh) &
            (merged["home_covers_rl"].astype(float) == 0)
        ).astype(float)
        merged.loc[g_thresh.isna(), "joint_hit"] = np.nan

    elif script_name == "D_LateDivergence":
        f5_thresh = merged["legs"].apply(lambda l: _parse_line(l, r"F5_Under_([\d.]+)"))
        g_thresh  = merged["legs"].apply(lambda l: _parse_line(l, r"Game_Over_([\d.]+)"))
        merged["joint_hit"] = (
            (merged["f5_total"].astype(float) < f5_thresh.fillna(99)) &
            (game_total > g_thresh.fillna(0))
        ).astype(float)
        merged.loc[f5_thresh.isna() | g_thresh.isna(), "joint_hit"] = np.nan

    else:
        return {"note": f"unknown script: {script_name}"}

    merged = merged.dropna(subset=["joint_hit"])
    if len(merged) < MIN_SAMPLE:
        return {
            "watch": 0.10, "small": 0.30, "full": 0.75,
            "n_games": len(merged), "roi_small": None, "roi_full": None,
            "wr_small": None, "wr_full": None,
            "note": f"n={len(merged)} — accumulates over season",
        }

    edge_col = next((c for c in ["sgp_edge", "corr_edge"] if c in merged.columns), None)
    if edge_col is None:
        return {"note": "no edge column in SGP data"}
    merged[edge_col] = pd.to_numeric(merged[edge_col], errors="coerce")
    merged = merged.dropna(subset=["joint_hit", edge_col]).reset_index(drop=True)

    return _grid_search_three(
        merged, edge_col, "joint_hit", None,
        default_odds=400.0,
        watch_grid=[0.10, 0.20, 0.30],
        small_grid=[0.30, 0.50, 0.75],
        full_grid= [0.75, 1.00, 1.50, 2.00],
        label=f"script_{script_name}",
    )


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

ALL_SIGNALS = [
    "rl_prob", "ml_win_prob", "ou_prob",
    "k_over_prob", "nrfi_prob", "f5_win_prob",
    "script_a", "script_a2", "script_b", "script_c", "script_d",
]

SCRIPT_NAMES = {
    "script_a":  "A_Dominance",
    "script_a2": "A2_Dominance",
    "script_b":  "B_Explosion",
    "script_c":  "C_EliteDuel",
    "script_d":  "D_LateDivergence",
}


def run_optimization(signals: list[str], dry_run: bool = False) -> dict:
    print("=" * 72)
    print("  Signal Band Optimizer  —  Three-Tier Thresholds")
    print(f"  Date: {date.today().isoformat()}")
    print("=" * 72)

    cards       = load_cards()
    actuals     = load_actuals()
    sgp_history = load_sgp_history()

    print(f"\n  Daily cards  : {len(cards)} rows across "
          f"{cards['_date'].nunique() if not cards.empty else 0} dates")
    print(f"  Actuals 2026 : {len(actuals)} games")
    print(f"  SGP history  : {len(sgp_history)} script rows across "
          f"{sgp_history['game_date'].nunique() if not sgp_history.empty else 0} dates")

    if dry_run:
        print("\n[DRY-RUN] Data loaded. Exiting.")
        return {}

    if cards.empty:
        print("\n  [ERROR] No daily cards found.")
        return {}

    df = merge_cards_actuals(cards, actuals)
    print(f"\n  Merged: {len(df)} rows | "
          f"{df['home_covers_rl'].notna().sum() if 'home_covers_rl' in df.columns else 0} "
          f"RL resolved | "
          f"{df['f1_nrfi'].notna().sum() if 'f1_nrfi' in df.columns else 0} NRFI resolved")

    results = {}
    today = date.today().isoformat()

    signal_fns = {
        "rl_prob":     lambda: optimize_rl_prob(df),
        "ml_win_prob": lambda: optimize_ml_win_prob(df),
        "ou_prob":     lambda: optimize_ou_prob(df),
        "k_over_prob": lambda: optimize_k_over_prob(df),
        "nrfi_prob":   lambda: optimize_nrfi_prob(df),
        "f5_win_prob": lambda: optimize_f5_win_prob(df),
        "script_a":    lambda: optimize_script(sgp_history, actuals, "A_Dominance"),
        "script_a2":   lambda: optimize_script(sgp_history, actuals, "A2_Dominance"),
        "script_b":    lambda: optimize_script(sgp_history, actuals, "B_Explosion"),
        "script_c":    lambda: optimize_script(sgp_history, actuals, "C_EliteDuel"),
        "script_d":    lambda: optimize_script(sgp_history, actuals, "D_LateDivergence"),
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

        if "watch" not in r:
            print(r.get("note", "error"))
        else:
            tw = r["watch"]
            ts = r["small"]
            tf = r["full"]
            ns = r.get("n_small", "?")
            nf = r.get("n_full",  "?")
            rs = r.get("roi_small")
            rf = r.get("roi_full")
            ws = r.get("wr_small")
            wf = r.get("wr_full")
            rs_str = f"{rs:+.1%}" if rs is not None else "N/A"
            rf_str = f"{rf:+.1%}" if rf is not None else "N/A"
            ws_str = f"{ws:.1%}"  if ws is not None else "N/A"
            wf_str = f"{wf:.1%}"  if wf is not None else "N/A"
            print(
                f"watch>={tw:.3f}  small>={ts:.3f}  full>={tf:.3f} | "
                f"n_small={ns}(ROI={rs_str} WR={ws_str})  "
                f"n_full={nf}(ROI={rf_str} WR={wf_str})"
            )

    return results


def save_bands(results: dict) -> None:
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
    print("\n" + "=" * 90)
    print("  SIGNAL BAND SUMMARY  —  Three Confidence Tiers")
    print("  Monitor = watch cutoff  |  Bet Small = small cutoff  |  Bet Full = full cutoff")
    print("=" * 90)
    hdr = f"  {'Signal':<18}  {'Monitor>=':>10}  {'BetSmall>=':>10}  {'BetFull>=':>10}  "
    hdr += f"{'n_sm':>5}  {'ROI_sm':>7}  {'n_fl':>5}  {'ROI_fl':>7}  Note"
    print(hdr)
    print("  " + "-" * 86)
    for sig, r in results.items():
        if "watch" not in r:
            print(f"  {sig:<18}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}  "
                  f"{'?':>5}  {'N/A':>7}  {'?':>5}  {'N/A':>7}  {r.get('note','')}")
        else:
            tw = r["watch"]
            ts = r["small"]
            tf = r["full"]
            ns = r.get("n_small", "?")
            nf = r.get("n_full",  "?")
            rs = r.get("roi_small")
            rf = r.get("roi_full")
            rs_str = f"{rs:+.1%}" if rs is not None else "N/A"
            rf_str = f"{rf:+.1%}" if rf is not None else "N/A"
            note   = r.get("note", "")
            print(f"  {sig:<18}  {tw:>10.3f}  {ts:>10.3f}  {tf:>10.3f}  "
                  f"{str(ns):>5}  {rs_str:>7}  {str(nf):>5}  {rf_str:>7}  {note}")
    print("=" * 90)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize three-tier signal thresholds")
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
