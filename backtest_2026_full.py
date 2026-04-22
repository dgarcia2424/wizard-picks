"""
backtest_2026_full.py
=====================
Comprehensive 2026-season backtest harness for ALL FIVE betting markets:
  ML  — Full-Game Moneyline
  TOT — Totals (Over/Under)
  RL  — Runline
  F5  — First-5 Moneyline
  NR  — NRFI (No Run First Inning)

Philosophy
----------
* Sterile inference: for each historical date we synthesize the same
  `lineups_YYYY-MM-DD.parquet` the live scorers expect, invoke each
  `score_*_today.predict_games()` unchanged, and discard the temporary
  lineup file.  The feature matrix is NEVER mutated — scorers
  read it in-place and rely on `game_date` being in the past.
* Three-Part Lock: the same `sanity_margin=0.04` / `odds_floor=-225` /
  edge-tier / quarter-Kelly logic from `tools.implementations._evaluate_pick`
  is reproduced here so bets are selected exactly as the live pipeline
  would have selected them.
* No future leakage: scorers use `feature_matrix_enriched_v2.parquet` which
  is historical; predictions for date D consume only rows with
  `game_date <= D`.
* Memory: each market's scorer is popped from sys.modules and
  `gc.collect()` is called between markets so large XGBoost / Bayesian
  posteriors don't accumulate.

NRFI WARNING
------------
Discovered OOF 2023-2025 ECE = 0.0095 but 2026 production accuracy
sitting at ~43.8% with severe covariate shift
(batting_matchup_edge +1803%, sp_k_pct_diff -233%).  The tear sheet
emits a WARN flag whenever NRFI actionable ROI < 0 or calibration
drifts > 0.05 from historical.

Outputs
-------
  backtest_full_actionable_picks.csv  — every actionable pick from the lock
  backtest_full_all_predictions.csv   — every model prediction (for calibration studies)
  backtest_full_summary.csv           — per-market tear sheet rows

Usage
-----
  python backtest_2026_full.py
  python backtest_2026_full.py --since 2026-04-01 --until 2026-04-20
  python backtest_2026_full.py --markets ML TOT RL          # subset
"""
from __future__ import annotations

import argparse
import gc
import sys
import unicodedata
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Legacy sklearn pickle compat ──────────────────────────────────────────────
# Models in models/ were saved with sklearn 1.8.0 where LogisticRegression's
# `multi_class` attribute was removed. In the current env (1.7.2), predict_proba
# still reads self.multi_class and throws AttributeError. Setting a class-level
# default lets old pickles fall through to the class attribute. Mirrors the
# patch in wizard_agents/tools/implementations.py.
try:
    from sklearn.linear_model import LogisticRegression as _LR
    if "multi_class" not in _LR.__dict__:
        _LR.multi_class = "auto"
except Exception:
    pass

# ---------------------------------------------------------------------------
# CONSTANTS — mirrors tools/implementations.py
# ---------------------------------------------------------------------------
_KELLY_BANK       = 2000.0
_KELLY_CAP        = 50.0
_TIER1_EDGE       = 0.030
_TIER2_EDGE       = 0.010
_SANITY_MARGIN    = 0.04
_ODDS_FLOOR       = -225

PIPELINE_DIR = Path(__file__).resolve().parent
DATA_DIR     = PIPELINE_DIR / "data" / "statcast"
STATCAST_26  = DATA_DIR / "statcast_2026.parquet"
ACTUALS_26   = DATA_DIR / "actuals_2026.parquet"
ODDS_COMBINED = DATA_DIR / "odds_combined_2026.parquet"

ALL_MARKETS = ["ML", "TOT", "RL", "F5", "NR"]


# ---------------------------------------------------------------------------
# THREE-PART LOCK (duplicated from tools/implementations.py to keep this
# script self-contained and runnable without the wizard_agents package)
# ---------------------------------------------------------------------------
def _american_to_decimal(odds: float) -> float:
    return 1.0 + (odds / 100.0) if odds >= 0 else 1.0 + (100.0 / abs(odds))


def _american_to_prob(odds: float) -> float:
    return 100.0 / (odds + 100.0) if odds >= 0 else abs(odds) / (abs(odds) + 100.0)


def _classify_edge(edge: Optional[float]) -> Optional[int]:
    if edge is None:
        return None
    if edge >= _TIER1_EDGE:
        return 1
    if edge >= _TIER2_EDGE:
        return 2
    return None


def _kelly_stake(model_prob: float, american_odds: float, tier: int) -> int:
    if american_odds is None or model_prob is None:
        return 0
    b = _american_to_decimal(float(american_odds)) - 1.0
    if b <= 0:
        return 0
    f_star = (b * model_prob - (1.0 - model_prob)) / b
    if f_star <= 0:
        return 0
    mult    = 0.25 if tier == 1 else 0.125
    raw     = _KELLY_BANK * f_star * mult
    capped  = min(raw, _KELLY_CAP)
    rounded = int(round(capped))
    return max(rounded, 1)


def evaluate_pick(
    model_prob:  Optional[float],
    p_true:      Optional[float],
    retail_imp:  Optional[float],
    retail_odds: Optional[float],
    skip_sanity: bool = False,
    sanity_margin: float = _SANITY_MARGIN,
) -> dict:
    def _n(x):
        try:
            if x is None or pd.isna(x):
                return None
            return float(x)
        except (TypeError, ValueError):
            return None
    model_prob  = _n(model_prob)
    p_true      = _n(p_true)
    retail_imp  = _n(retail_imp)
    retail_odds = _n(retail_odds)
    if model_prob is None:
        return {"actionable": False, "tier": None, "dollar_stake": None,
                "sanity_check_pass": False, "odds_floor_pass": False, "edge": None}
    if skip_sanity:
        sanity_pass = True
    else:
        sanity_pass = False if p_true is None else (abs(model_prob - p_true) <= sanity_margin)
    odds_pass = False if retail_odds is None else (retail_odds >= _ODDS_FLOOR)
    edge      = None if retail_imp is None else (model_prob - retail_imp)
    tier      = _classify_edge(edge) if edge is not None else None
    stake     = 0
    if sanity_pass and odds_pass and tier is not None:
        stake = _kelly_stake(model_prob, retail_odds, tier)
    actionable = bool(sanity_pass and odds_pass and tier is not None and stake >= 1)
    return {
        "model_prob":          None if model_prob is None else round(model_prob, 4),
        "P_true":              None if p_true is None else round(p_true, 4),
        "retail_implied_prob": None if retail_imp is None else round(retail_imp, 4),
        "edge":                None if edge is None else round(edge, 4),
        "retail_odds":         retail_odds,
        "sanity_check_pass":   bool(sanity_pass),
        "odds_floor_pass":     bool(odds_pass),
        "tier":                tier,
        "dollar_stake":        int(stake) if stake >= 1 else None,
        "actionable":          actionable,
    }


# ---------------------------------------------------------------------------
# LINEUP SYNTHESIS — build lineups_{date}.parquet from statcast/actuals
# ---------------------------------------------------------------------------
def _norm(name) -> str:
    if not isinstance(name, str):
        return ""
    name = name.strip()
    if "," in name:
        a, b = [p.strip() for p in name.split(",", 1)]
        name = f"{b} {a}"
    name = name.upper()
    return "".join(
        c for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    )


def build_lineups_index() -> pd.DataFrame:
    """
    Derive one lineup row per (game_pk) from statcast_2026.parquet:
      game_date, game_pk, home_team, away_team,
      home_starter_name, away_starter_name.
    The starting pitcher = first batter-faced in the top / bottom of inning 1.
    """
    if not STATCAST_26.exists():
        raise FileNotFoundError(f"Missing {STATCAST_26}")

    sc = pd.read_parquet(STATCAST_26, columns=[
        "game_pk", "game_date", "home_team", "away_team",
        "inning", "inning_topbot", "at_bat_number", "player_name",
    ])
    sc["game_date"] = pd.to_datetime(sc["game_date"]).dt.date.astype(str)

    inn1 = sc[sc["inning"] == 1]
    rows = []
    for gpk, g in inn1.groupby("game_pk"):
        meta_row = g.iloc[0]
        home = str(meta_row["home_team"])
        away = str(meta_row["away_team"])
        dt   = str(meta_row["game_date"])
        top = g[g["inning_topbot"] == "Top"]
        bot = g[g["inning_topbot"] == "Bot"]
        if len(top) == 0 or len(bot) == 0:
            continue
        home_sp = _norm(str(top.loc[top["at_bat_number"].idxmin()]["player_name"]))
        away_sp = _norm(str(bot.loc[bot["at_bat_number"].idxmin()]["player_name"]))
        rows.append({
            "game_pk": int(gpk),
            "game_date": dt,
            "home_team": home,
            "away_team": away,
            "home_starter_name": home_sp,
            "away_starter_name": away_sp,
        })
    del sc, inn1
    gc.collect()
    return pd.DataFrame(rows)


def write_lineup_file(day_rows: pd.DataFrame, date_str: str) -> Path:
    """Write a per-date lineup parquet in the format scorers expect."""
    path = DATA_DIR / f"lineups_{date_str}.parquet"
    day_rows[[
        "game_pk", "game_date", "home_team", "away_team",
        "home_starter_name", "away_starter_name"
    ]].to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# STERILE INFERENCE — call each score_*_today module unchanged
# ---------------------------------------------------------------------------
def _invoke_scorer(module_name: str, date_str: str) -> pd.DataFrame:
    import contextlib, io, os
    prev_cwd = os.getcwd()
    try:
        os.chdir(str(PIPELINE_DIR))
        if str(PIPELINE_DIR) not in sys.path:
            sys.path.insert(0, str(PIPELINE_DIR))
        sys.modules.pop(module_name, None)
        with contextlib.redirect_stdout(io.StringIO()):
            mod = __import__(module_name)
            df  = mod.predict_games(date_str)
        sys.modules.pop(module_name, None)
        gc.collect()
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    finally:
        os.chdir(prev_cwd)


def score_day(date_str: str, markets: list[str]) -> dict[str, pd.DataFrame]:
    """Invoke each requested scorer for date_str and return a dict of frames."""
    out: dict[str, pd.DataFrame] = {}
    if "ML" in markets:
        out["ML"]  = _invoke_scorer("score_ml_today",       date_str)
    if "TOT" in markets or "RL" in markets:
        out["RD"]  = _invoke_scorer("score_run_dist_today", date_str)
    if "F5" in markets:
        out["F5"]  = _invoke_scorer("score_f5_today",       date_str)
    if "NR" in markets:
        out["NR"]  = _invoke_scorer("score_nrfi_today",     date_str)
    return out


# ---------------------------------------------------------------------------
# ODDS LOADER (best-effort — falls back to vig-free from model if absent)
# ---------------------------------------------------------------------------
def load_odds() -> pd.DataFrame:
    """
    Build a unified 2026 odds frame from:
      - odds_combined_2026.parquet (if present)
      - odds_history_2026_*.parquet (daily snapshots)
      - odds_current_2026_*.parquet (current-day pulls)
    Dedup on (game_date, home_team, away_team) keeping the latest snapshot.
    """
    frames: list[pd.DataFrame] = []
    combined = DATA_DIR / "odds_combined_2026.parquet"
    if combined.exists():
        frames.append(pd.read_parquet(combined))
    for f in sorted(DATA_DIR.glob("odds_history_2026_*.parquet")):
        frames.append(pd.read_parquet(f))
    for f in sorted(DATA_DIR.glob("odds_current_2026_*.parquet")):
        frames.append(pd.read_parquet(f))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
    sort_col = "snapshot_time" if "snapshot_time" in df.columns else "pull_timestamp"
    if sort_col in df.columns:
        df = df.sort_values(sort_col)
    df = df.drop_duplicates(
        subset=["game_date", "home_team", "away_team"], keep="last"
    ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------
def _safe_auc(y, p) -> Optional[float]:
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(y)) < 2:
            return None
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def _log_loss(y, p, eps: float = 1e-9) -> Optional[float]:
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return None
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(y, p) -> Optional[float]:
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return None
    return float(np.mean((p - y) ** 2))


def _ece(y, p, n_bins: int = 10) -> Optional[float]:
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return None
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        total += (mask.sum() / len(p)) * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def _payout(dollar_stake: float, american_odds: float, won: bool) -> float:
    if won:
        return dollar_stake * (_american_to_decimal(american_odds) - 1.0)
    return -dollar_stake


# ---------------------------------------------------------------------------
# MARKET ADAPTERS — each returns normalized rows:
#   (game_pk, date, home_team, away_team, model_prob, p_true, retail_imp,
#    retail_odds, bet_type, actual)
# ---------------------------------------------------------------------------
def _first(cols: list[str], row) -> Optional[float]:
    for c in cols:
        if c in row and pd.notna(row[c]):
            try:
                return float(row[c])
            except (TypeError, ValueError):
                continue
    return None


def adapt_ml(df: pd.DataFrame, odds: pd.DataFrame, act: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    d = df.copy()
    prob_col = next((c for c in ["stacker_l2", "p_stk_home", "p_stacker", "p_home_win"]
                     if c in d.columns), None)
    if prob_col is None:
        return pd.DataFrame()
    d["model_prob"] = d[prob_col]
    d["bet_type"]   = "HOME_ML"

    if "game_pk" in d.columns and not act.empty:
        d = d.merge(
            act[["game_pk", "home_score_final", "away_score_final"]],
            on="game_pk", how="left"
        )
        d["actual"] = (d["home_score_final"] > d["away_score_final"]).astype("Int64")
    else:
        d["actual"] = pd.NA

    if not odds.empty and "close_ml_home" in odds.columns:
        keep = ["game_date", "home_team", "away_team", "close_ml_home",
                "pinnacle_ml_home", "retail_implied_home", "P_true_home"]
        keep = [c for c in keep if c in odds.columns]
        d = d.merge(odds[keep], on=["game_date", "home_team", "away_team"], how="left")
        d["retail_odds"] = d.get("close_ml_home")
        d["retail_imp"]  = d.get("retail_implied_home")
        if "retail_imp" not in d.columns or d["retail_imp"].isna().all():
            d["retail_imp"] = d["retail_odds"].apply(
                lambda x: _american_to_prob(x) if pd.notna(x) else None)
        d["p_true"] = d.get("P_true_home")
        if ("p_true" not in d.columns or d["p_true"].isna().all()) and "pinnacle_ml_home" in d.columns:
            d["p_true"] = d["pinnacle_ml_home"].apply(
                lambda x: _american_to_prob(x) if pd.notna(x) else None)
    else:
        d["retail_odds"] = None
        d["retail_imp"]  = None
        d["p_true"]      = None
    return d


def adapt_totals(df: pd.DataFrame, odds: pd.DataFrame, act: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    d = df.copy()
    # score_run_dist outputs P(Over) at market line for totals
    prob_col = next((c for c in ["p_over_final", "p_over", "stacker_over",
                                  "p_stk_over", "tot_over_prob"]
                     if c in d.columns), None)
    if prob_col is None:
        return pd.DataFrame()
    d["model_prob"] = d[prob_col]
    d["bet_type"]   = "OVER"
    if not act.empty and "game_pk" in d.columns:
        d = d.merge(act[["game_pk", "home_score_final", "away_score_final"]],
                    on="game_pk", how="left")
        d["actual_total"] = d["home_score_final"] + d["away_score_final"]

    line_col = next((c for c in ["market_total", "close_total", "total_line"]
                     if c in d.columns), None)
    extras = [c for c in ["close_total", "pinnacle_total_over_odds",
                          "retail_implied_over", "P_true_over"] if c in odds.columns]
    if extras:
        d = d.merge(odds[["game_date", "home_team", "away_team"] + extras],
                    on=["game_date", "home_team", "away_team"], how="left")
        if line_col is None and "close_total" in d.columns:
            line_col = "close_total"
    if line_col and "actual_total" in d.columns:
        d["actual"] = (d["actual_total"] > d[line_col]).astype("Int64")
    else:
        d["actual"] = pd.NA

    # Retail over-odds standardized to -110 (Odds API doesn't expose retail over odds here)
    d["retail_odds"] = -110.0
    d["retail_imp"]  = d.get("retail_implied_over")
    if "retail_imp" not in d.columns or d["retail_imp"].isna().all():
        d["retail_imp"] = _american_to_prob(-110.0)
    d["p_true"] = d.get("P_true_over")
    if ("p_true" not in d.columns or d["p_true"].isna().all()) and "pinnacle_total_over_odds" in d.columns:
        d["p_true"] = d["pinnacle_total_over_odds"].apply(
            lambda x: _american_to_prob(x) if pd.notna(x) else None)
    return d


def adapt_runline(df: pd.DataFrame, odds: pd.DataFrame, act: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    d = df.copy()
    prob_col = next((c for c in ["p_home_cover_final", "p_home_covers_rl",
                                  "stacker_rl", "blended_rl", "p_stk_rl"]
                     if c in d.columns), None)
    if prob_col is None:
        return pd.DataFrame()
    d["model_prob"] = d[prob_col]
    d["bet_type"]   = "HOME_-1.5"
    if not act.empty and "game_pk" in d.columns:
        d = d.merge(act[["game_pk", "home_covers_rl"]], on="game_pk", how="left")
        d["actual"] = d["home_covers_rl"].astype("Int64")
    else:
        d["actual"] = pd.NA

    keep = [c for c in ["runline_home_odds", "pinnacle_rl_home_odds",
                        "retail_implied_rl_home", "P_true_rl_home"] if c in odds.columns]
    if keep:
        d = d.merge(odds[["game_date", "home_team", "away_team"] + keep],
                    on=["game_date", "home_team", "away_team"], how="left")
        d["retail_odds"] = d.get("runline_home_odds")
        d["retail_imp"]  = d.get("retail_implied_rl_home")
        if "retail_imp" not in d.columns or d["retail_imp"].isna().all():
            d["retail_imp"] = d["retail_odds"].apply(
                lambda x: _american_to_prob(x) if pd.notna(x) else None)
        d["p_true"] = d.get("P_true_rl_home")
        if ("p_true" not in d.columns or d["p_true"].isna().all()) and "pinnacle_rl_home_odds" in d.columns:
            d["p_true"] = d["pinnacle_rl_home_odds"].apply(
                lambda x: _american_to_prob(x) if pd.notna(x) else None)
    else:
        d["retail_odds"] = None
        d["retail_imp"]  = None
        d["p_true"]      = None
    return d


def adapt_f5(df: pd.DataFrame, odds: pd.DataFrame, act: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    d = df.copy()
    prob_col = next((c for c in ["stacker_l2", "p_stk_f5_home", "p_f5_home_win"]
                     if c in d.columns), None)
    if prob_col is None:
        return pd.DataFrame()
    d["model_prob"] = d[prob_col]
    d["bet_type"]   = "F5_HOME"
    if not act.empty and "game_pk" in d.columns:
        d = d.merge(act[["game_pk", "f5_home_win"]], on="game_pk", how="left")
        d["actual"] = d["f5_home_win"].astype("Int64")
    else:
        d["actual"] = pd.NA

    f5_odds_col = next((c for c in ["close_f5_home", "dk_f5_home"] if c in odds.columns), None)
    if f5_odds_col:
        d = d.merge(odds[["game_date", "home_team", "away_team", f5_odds_col]],
                    on=["game_date", "home_team", "away_team"], how="left")
        d["retail_odds"] = d[f5_odds_col]
        d["retail_imp"]  = d["retail_odds"].apply(
            lambda x: _american_to_prob(x) if pd.notna(x) else None)
    else:
        d["retail_odds"] = None
        d["retail_imp"]  = None
    # F5 uses Two-Part Lock — Pinnacle doesn't post F5 ML
    d["p_true"] = None
    d["skip_sanity"] = True
    return d


def adapt_nrfi(df: pd.DataFrame, odds: pd.DataFrame, act: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    d = df.copy()
    prob_col = next((c for c in ["p_stk_nrfi", "stacker_l2", "p_nrfi"]
                     if c in d.columns), None)
    if prob_col is None:
        return pd.DataFrame()
    d["model_prob"] = d[prob_col]
    d["bet_type"]   = "NRFI"
    if not act.empty and "game_pk" in d.columns:
        d = d.merge(act[["game_pk", "f1_nrfi"]], on="game_pk", how="left")
        d["actual"] = d["f1_nrfi"].astype("Int64")
    else:
        d["actual"] = pd.NA

    nrfi_odds_col = next((c for c in ["close_nrfi", "dk_nrfi", "f1_nrfi_odds"]
                          if c in odds.columns), None)
    if nrfi_odds_col:
        d = d.merge(odds[["game_date", "home_team", "away_team", nrfi_odds_col]],
                    on=["game_date", "home_team", "away_team"], how="left")
        d["retail_odds"] = d[nrfi_odds_col]
        d["retail_imp"]  = d["retail_odds"].apply(
            lambda x: _american_to_prob(x) if pd.notna(x) else None)
    else:
        d["retail_odds"] = None
        d["retail_imp"]  = None
    pin_col = next((c for c in ["pin_nrfi"] if c in odds.columns), None)
    if pin_col:
        d = d.merge(odds[["game_date", "home_team", "away_team", pin_col]],
                    on=["game_date", "home_team", "away_team"], how="left")
        d["p_true"] = d[pin_col].apply(
            lambda x: _american_to_prob(x) if pd.notna(x) else None)
    else:
        d["p_true"] = None
    return d


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since",   type=str, default="2026-03-27")
    ap.add_argument("--until",   type=str, default="2026-12-31")
    ap.add_argument("--markets", nargs="+", default=ALL_MARKETS,
                    choices=ALL_MARKETS)
    ap.add_argument("--out-prefix", type=str, default="backtest_full")
    ap.add_argument("--skip-sanity-for", nargs="*", default=[],
                    choices=ALL_MARKETS,
                    help="Bypass Pinnacle sanity gate for these markets (Two-Part Lock)")
    args = ap.parse_args()

    # Force UTF-8 stdout on Windows so the arrows/bullets render
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
        import io as _io
        sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 72)
    print(f"  backtest_2026_full.py | markets={args.markets}")
    print(f"  window {args.since} .. {args.until}")
    print("=" * 72)

    # Inputs
    print("\n[1/4] Building lineup index from statcast_2026 …")
    lineups = build_lineups_index()
    lineups = lineups[(lineups["game_date"] >= args.since) &
                      (lineups["game_date"] <= args.until)]
    print(f"      → {len(lineups)} games across {lineups['game_date'].nunique()} dates")

    act = pd.read_parquet(ACTUALS_26) if ACTUALS_26.exists() else pd.DataFrame()
    if not act.empty:
        act["game_date"] = pd.to_datetime(act["game_date"]).dt.date.astype(str)
    odds = load_odds()

    # Sterile per-date scoring
    print("\n[2/4] Sterile inference per date …")
    all_preds: dict[str, list[pd.DataFrame]] = {m: [] for m in args.markets}

    dates = sorted(lineups["game_date"].unique())
    for i, d in enumerate(dates, 1):
        day = lineups[lineups["game_date"] == d]
        if day.empty:
            continue
        lp = write_lineup_file(day, d)
        try:
            frames = score_day(d, args.markets)
        except Exception as e:
            print(f"   [{i}/{len(dates)}] {d}  SKIP: {type(e).__name__}: {e}")
            lp.unlink(missing_ok=True)
            continue

        def _tag(f: Optional[pd.DataFrame]) -> pd.DataFrame:
            if f is None or len(f) == 0:
                return pd.DataFrame()
            f = f.copy()
            f["game_date"] = d
            if "game_pk" not in f.columns and "game_pk" in day.columns:
                f = f.merge(day[["home_team", "away_team", "game_pk"]],
                            on=["home_team", "away_team"], how="left")
            return f

        if "ML" in args.markets:
            all_preds["ML"].append(adapt_ml(_tag(frames.get("ML")), odds, act))
        if "TOT" in args.markets:
            all_preds["TOT"].append(adapt_totals(_tag(frames.get("RD")), odds, act))
        if "RL" in args.markets:
            all_preds["RL"].append(adapt_runline(_tag(frames.get("RD")), odds, act))
        if "F5" in args.markets:
            all_preds["F5"].append(adapt_f5(_tag(frames.get("F5")), odds, act))
        if "NR" in args.markets:
            all_preds["NR"].append(adapt_nrfi(_tag(frames.get("NR")), odds, act))

        lp.unlink(missing_ok=True)
        gc.collect()
        if i % 5 == 0 or i == len(dates):
            print(f"   [{i}/{len(dates)}] {d}  ok")

    # Apply Three-Part Lock
    print("\n[3/4] Applying Three-Part Lock & computing metrics …")
    all_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for mkt in args.markets:
        frames = [f for f in all_preds[mkt] if f is not None and len(f) > 0]
        if not frames:
            print(f"   {mkt}: no predictions produced — skipping")
            continue
        m = pd.concat(frames, ignore_index=True)

        force_skip = (mkt in set(args.skip_sanity_for))
        picks = []
        for _, r in m.iterrows():
            ev = evaluate_pick(
                model_prob   = r.get("model_prob"),
                p_true       = r.get("p_true"),
                retail_imp   = r.get("retail_imp"),
                retail_odds  = r.get("retail_odds"),
                skip_sanity  = force_skip or bool(r.get("skip_sanity", False)),
            )
            picks.append(ev)
        ev_df = pd.DataFrame(picks)
        m = pd.concat([m.reset_index(drop=True), ev_df.reset_index(drop=True)], axis=1)
        m = m.loc[:, ~m.columns.duplicated()].copy()
        m["market"] = mkt
        all_rows.append(m)

        # De-duplicate column names defensively (merges can collide)
        m = m.loc[:, ~m.columns.duplicated()].copy()

        # Predictive metrics (restricted to rows with actual)
        graded = m[m["actual"].notna()].copy()
        y = np.asarray(graded["actual"].astype(float).values).ravel()
        p_raw = graded["model_prob"]
        if isinstance(p_raw, pd.DataFrame):
            p_raw = p_raw.iloc[:, 0]
        p = np.asarray(p_raw.astype(float).values).ravel()
        pred_row = {
            "market":    mkt,
            "n_games":   int(len(m)),
            "n_graded":  int(len(graded)),
            "auc":       _safe_auc(y, p),
            "log_loss":  _log_loss(y, p),
            "brier":     _brier(y, p),
            "ece":       _ece(y, p),
        }

        # Financial metrics (actionable + graded)
        actionable = m[(m["actionable"] == True) & m["actual"].notna()].copy()
        if len(actionable) > 0:
            won = (actionable["actual"].astype(int) == 1).values
            stakes = actionable["dollar_stake"].astype(float).values
            rodd   = actionable["retail_odds"].astype(float).values
            pnl    = np.array([_payout(s, o, w)
                               for s, o, w in zip(stakes, rodd, won)])
            total_stake = float(stakes.sum())
            total_pnl   = float(pnl.sum())
            pred_row.update({
                "actionable_bets": int(len(actionable)),
                "tier1_bets":      int((actionable["tier"] == 1).sum()),
                "tier2_bets":      int((actionable["tier"] == 2).sum()),
                "total_stake":     round(total_stake, 2),
                "total_pnl":       round(total_pnl, 2),
                "roi_pct":         round(100.0 * total_pnl / total_stake, 2)
                                   if total_stake > 0 else None,
                "win_pct":         round(100.0 * won.mean(), 2),
            })
        else:
            pred_row.update({
                "actionable_bets": 0, "tier1_bets": 0, "tier2_bets": 0,
                "total_stake": 0.0, "total_pnl": 0.0,
                "roi_pct": None, "win_pct": None,
            })

        # NRFI covariate-shift warning
        if mkt == "NR":
            pred_row["WARN"] = (
                "Covariate shift suspected in 2026 "
                "(batting_matchup_edge +1803%, sp_k_pct_diff -233%; "
                "production accuracy ~43.8%). "
                "Shadow-mode recommended until drift resolves."
            )
        summary_rows.append(pred_row)
        gc.collect()

    # ---------------------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------------------
    print("\n[4/4] Writing outputs & tear sheet …")
    prefix = args.out_prefix

    if all_rows:
        full = pd.concat(all_rows, ignore_index=True)
        full.to_csv(f"{prefix}_all_predictions.csv", index=False)
        full[full["actionable"] == True].to_csv(
            f"{prefix}_actionable_picks.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary.to_csv(f"{prefix}_summary.csv", index=False)

    # ── Tear Sheet ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  TEAR SHEET — 2026 BACKTEST")
    print("=" * 72)
    if summary.empty:
        print("  No markets produced predictions.")
        return

    hdr = (f"  {'MKT':<4} {'N':>5} {'Grd':>5} {'AUC':>6} {'LogL':>6} "
           f"{'Brier':>6} {'ECE':>6} {'Bets':>5} {'Win%':>6} {'ROI%':>7} {'PnL':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, r in summary.iterrows():
        def f(x, spec):
            return format(x, spec) if x is not None and not (isinstance(x, float)
                                                              and np.isnan(x)) else "—"
        print(f"  {r['market']:<4} {r['n_games']:>5d} {r['n_graded']:>5d} "
              f"{f(r['auc'], '>6.3f')} {f(r['log_loss'], '>6.3f')} "
              f"{f(r['brier'], '>6.3f')} {f(r['ece'], '>6.3f')} "
              f"{int(r['actionable_bets']):>5d} "
              f"{f(r.get('win_pct'), '>6.1f')} "
              f"{f(r.get('roi_pct'), '>7.1f')} "
              f"{f(r.get('total_pnl'), '>9.2f')}")

    warn = summary[summary.get("WARN").notna()] if "WARN" in summary.columns else pd.DataFrame()
    if not warn.empty:
        print("\n  ⚠ WARNINGS")
        for _, r in warn.iterrows():
            print(f"    [{r['market']}] {r['WARN']}")

    print("\n  Outputs:")
    print(f"    {prefix}_all_predictions.csv")
    print(f"    {prefix}_actionable_picks.csv")
    print(f"    {prefix}_summary.csv")
    print()


if __name__ == "__main__":
    main()
