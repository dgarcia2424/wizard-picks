"""
analyze_signal_accuracy.py
==========================
For every model signal, find three probability cutoffs that maximize hit rate
(fraction of predictions that are correct).

Three displayed tiers per signal per window:
  ★ Optimal      — highest hit rate (highest confidence cutoff with n >= MIN_N)
  • Less optimal — intermediate confidence cutoff
  ✗ Stay away   — baseline (prob >= 50%, all predictions)

Three time windows:
  2025 OOF   — out-of-fold validation from eval_predictions.csv
  2026 YTD   — all 2026 daily cards joined with actuals
  Last 30 D  — rolling last-30-day subset of 2026

Signals
-------
  ML        mc_home_win / ml_cal          → actual_home_win
  Totals    ou_p_model (directional)      → direction vs posted line
  Runline   blended_rl / rl_stacked       → home_covers_rl
  F5        f5_stacker_l2                 → f5_home_win
  NRFI      mc_nrfi_prob (max confidence) → f1_nrfi
  K-over    home/away_k_model_over        → actual K vs K line

Usage
-----
  python analyze_signal_accuracy.py             # print table + save JSON
  python analyze_signal_accuracy.py --no-save   # print only
  python analyze_signal_accuracy.py --min-n 10  # lower sample threshold
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT        = Path(__file__).resolve().parent
_EVAL_PRED   = _ROOT / "eval_predictions.csv"
_FEAT_MTX    = _ROOT / "feature_matrix_enriched_v2.parquet"
_F5_VAL_25   = _ROOT / "f5_val_predictions_2025.csv"
_NRFI_VAL_25 = _ROOT / "nrfi_val_predictions_2025.csv"
_K_VAL_25    = _ROOT / "k_val_predictions_2025.csv"
_BACKTEST_26 = _ROOT / "backtest_full_all_predictions.csv"
_LIVE_26     = _ROOT / "live_predictions_2026.csv"
_OUT         = _ROOT / "signal_accuracy.json"

# Backtest market code -> signal name
_MARKET_MAP = {"ML": "ML", "RL": "Runline", "F5": "F5", "NR": "NRFI"}

# Probability cutoff grid to evaluate (round numbers so table is readable)
CUTOFF_GRID = [0.50, 0.52, 0.54, 0.55, 0.56, 0.58,
               0.60, 0.62, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

DEFAULT_MIN_N = 20


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_2026() -> pd.DataFrame:
    """Load backtest + live predictions for 2026. Returns long-format df with
    columns: market, model_prob, actual, game_date."""
    frames = []
    for path in (_BACKTEST_26, _LIVE_26):
        if path.exists():
            frames.append(pd.read_csv(path, usecols=["market", "model_prob", "actual", "game_date"],
                                      low_memory=False))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    df["model_prob"] = pd.to_numeric(df["model_prob"], errors="coerce")
    df["actual"]     = pd.to_numeric(df["actual"],     errors="coerce")
    return df.dropna(subset=["model_prob", "actual"])


def _load_k_over_2026() -> pd.DataFrame:
    """Build K-over 2026 rows from daily cards + actuals using 3.5 line (best signal).
    Returns df with columns: game_date, model_prob (max-conf), actual (hit)."""
    import glob
    cards_dir = _ROOT / "daily_cards"
    files = sorted(glob.glob(str(cards_dir / "daily_card_2026-*.csv")))
    if not files:
        return pd.DataFrame()
    cards = pd.concat([
        pd.read_csv(f, low_memory=False).assign(
            game_date=Path(f).stem.replace("daily_card_", ""))
        for f in files
    ], ignore_index=True)

    act_path = _ROOT / "data" / "statcast" / "actuals_2026.parquet"
    if not act_path.exists():
        return pd.DataFrame()
    act = pd.read_parquet(act_path)[["home_team", "game_date", "home_sp_k", "away_sp_k"]]
    act["game_date"] = pd.to_datetime(act["game_date"]).dt.strftime("%Y-%m-%d")
    merged = cards.merge(act, on=["home_team", "game_date"], how="left")

    K_LINE = 3.5
    rows = []
    for _, r in merged.iterrows():
        for side in ("home", "away"):
            actual_k = pd.to_numeric(r.get(f"{side}_sp_k"), errors="coerce")
            col = f"mc_{side}_sp_k_over_{str(K_LINE).replace('.','')}"
            prob = pd.to_numeric(r.get(col), errors="coerce")
            if pd.isna(actual_k) or pd.isna(prob):
                continue
            conf = prob if prob >= 0.5 else 1.0 - prob
            hit  = int(actual_k > K_LINE) if prob >= 0.5 else int(actual_k <= K_LINE)
            rows.append({"game_date": r["game_date"], "model_prob": conf, "actual": float(hit)})
    return pd.DataFrame(rows)




def _load_eval_2025() -> pd.DataFrame:
    """Load eval_predictions.csv enriched with feature-matrix close_total for Totals signal."""
    if not _EVAL_PRED.exists():
        return pd.DataFrame()
    df = pd.read_csv(_EVAL_PRED)
    df = df[df["year"] == 2025].copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")

    # Enrich with close_total + actual_game_total from feature matrix so Totals signal works
    if _FEAT_MTX.exists():
        try:
            fm = pd.read_parquet(_FEAT_MTX, columns=["home_team", "game_date",
                                                      "close_total", "actual_game_total"])
            fm = fm[fm["actual_game_total"].notna()].copy()
            fm["game_date"] = pd.to_datetime(fm["game_date"]).dt.strftime("%Y-%m-%d")
            fm = fm.drop_duplicates(["home_team", "game_date"])
            df = df.merge(fm[["home_team", "game_date", "close_total", "actual_game_total"]],
                          on=["home_team", "game_date"], how="left")
        except Exception:
            pass
    return df


def _load_f5_val_2025() -> pd.DataFrame:
    if not _F5_VAL_25.exists():
        return pd.DataFrame()
    df = pd.read_csv(_F5_VAL_25)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    return df


def _load_nrfi_val_2025() -> pd.DataFrame:
    if not _NRFI_VAL_25.exists():
        return pd.DataFrame()
    df = pd.read_csv(_NRFI_VAL_25)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    return df


def _load_k_val_2025() -> pd.DataFrame:
    if not _K_VAL_25.exists():
        return pd.DataFrame()
    return pd.read_csv(_K_VAL_25)


# ---------------------------------------------------------------------------
# Core accuracy functions
# ---------------------------------------------------------------------------

def _hit_rate_at_cutoff(prob: pd.Series, outcome: pd.Series, cutoff: float
                        ) -> tuple[int, float] | None:
    """Return (n, hit_rate) for rows with prob >= cutoff. None if n=0."""
    mask = (prob >= cutoff) & prob.notna() & outcome.notna()
    n = int(mask.sum())
    if n == 0:
        return None
    hr = float(outcome[mask].mean())
    return n, hr


def find_three_tiers(prob: pd.Series, outcome: pd.Series,
                     min_n: int = DEFAULT_MIN_N,
                     cutoff_grid=None) -> dict:
    """
    Find three cutoffs that illustrate high / medium / low conviction.

    Returns dict with keys: optimal, less_optimal, stay_away
    Each value: {"cutoff": float, "n": int, "hit_rate": float}
    """
    if cutoff_grid is None:
        cutoff_grid = CUTOFF_GRID

    prob    = pd.to_numeric(prob,    errors="coerce")
    outcome = pd.to_numeric(outcome, errors="coerce")

    # Compute hit rate at every cutoff with n >= min_n
    valid = []
    for c in cutoff_grid:
        r = _hit_rate_at_cutoff(prob, outcome, c)
        if r is not None and r[0] >= min_n:
            valid.append({"cutoff": c, "n": r[0], "hit_rate": r[1]})

    if not valid:
        # fall back to min_n=1 so we return something
        for c in cutoff_grid:
            r = _hit_rate_at_cutoff(prob, outcome, c)
            if r is not None and r[0] >= 1:
                valid.append({"cutoff": c, "n": r[0], "hit_rate": r[1]})
        if not valid:
            return {"note": "no data"}

    # Optimal = highest hit rate (prefer higher cutoff on ties)
    best   = max(valid, key=lambda x: (x["hit_rate"], x["cutoff"]))

    # Stay away = lowest cutoff (≥50% baseline)
    worst  = min(valid, key=lambda x: x["cutoff"])

    # Less optimal = cutoff between worst and optimal with highest intermediate hit rate
    middle_candidates = [v for v in valid
                         if worst["cutoff"] < v["cutoff"] < best["cutoff"]]
    if middle_candidates:
        middle = max(middle_candidates, key=lambda x: x["hit_rate"])
    elif best["cutoff"] != worst["cutoff"]:
        middle = best  # collapse to just two if no middle
    else:
        middle = best

    return {
        "optimal":      best,
        "less_optimal": middle,
        "stay_away":    worst,
    }


# ---------------------------------------------------------------------------
# Signal extractors — return (prob_series, outcome_series)
# All use MAX-CONFIDENCE framing: prob = model's confidence in predicted direction,
# outcome = 1 if that direction was correct.
# ---------------------------------------------------------------------------

def _max_conf(prob_home: pd.Series, outcome_home: pd.Series):
    """Convert home-centric prob/outcome to max-confidence framing."""
    p = pd.to_numeric(prob_home, errors="coerce")
    o = pd.to_numeric(outcome_home, errors="coerce")
    # confidence = max(p, 1-p); outcome flipped when model predicts away
    conf = p.where(p >= 0.5, 1.0 - p)
    hit  = o.where(p >= 0.5, 1.0 - o)
    return conf, hit


# ── 2025 OOF extractors (eval_predictions.csv + year-specific val files) ──

def _sig_ml_2025(df: pd.DataFrame):
    return _max_conf(df.get("ml_cal", pd.Series(dtype=float)),
                     df.get("actual_home_win", pd.Series(dtype=float)))

def _sig_rl_2025(df: pd.DataFrame):
    return _max_conf(df.get("rl_stacked", pd.Series(dtype=float)),
                     df.get("home_covers_rl", pd.Series(dtype=float)))

def _sig_totals_2025(df: pd.DataFrame):
    """Use tot_pred vs close_total to determine direction, actual vs close_total for outcome."""
    tot_pred  = pd.to_numeric(df.get("tot_pred"), errors="coerce")
    close     = pd.to_numeric(df.get("close_total"), errors="coerce")
    actual    = pd.to_numeric(df.get("actual_game_total"), errors="coerce")
    valid = tot_pred.notna() & close.notna() & actual.notna()
    if valid.sum() == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    pred_over = tot_pred > close
    margin    = (tot_pred - close).abs()   # model conviction = distance from line
    # Scale by the margin std so meaningful confidence only starts at ~1-2 run margins.
    # Using fixed scale=0.92 (observed margin std for 2025) keeps the threshold stable
    # across re-runs. At margin=1.5 -> ~56% conf; margin=2.0 -> ~66% conf.
    scale     = margin[valid].std() if valid.sum() > 1 else 0.92
    prob      = 0.5 + 0.5 * (1 - np.exp(-margin / scale))
    prob      = pd.Series(prob, index=df.index)
    prob[~valid] = np.nan
    actual_over = actual > close
    outcome = (pred_over == actual_over).astype(float)
    outcome[~valid] = np.nan
    return prob, outcome

def _sig_f5_2025(df_f5: pd.DataFrame):
    """df_f5 is f5_val_predictions_2025.csv."""
    return _max_conf(df_f5.get("stacker_f5_cover", pd.Series(dtype=float)),
                     df_f5.get("f5_home_win", pd.Series(dtype=float)))

def _sig_nrfi_2025(df_nrfi: pd.DataFrame):
    """df_nrfi is nrfi_val_predictions_2025.csv."""
    return _max_conf(df_nrfi.get("stacker_nrfi", pd.Series(dtype=float)),
                     df_nrfi.get("f1_nrfi", pd.Series(dtype=float)))

def _sig_k_over_2025(df_k: pd.DataFrame):
    """df_k is k_val_predictions_2025.csv."""
    prob    = pd.to_numeric(df_k.get("k_over_pred"),   errors="coerce")
    outcome = pd.to_numeric(df_k.get("k_over_actual"), errors="coerce")
    return prob, outcome


# ── 2026 / Last-30 extractors (daily cards + actuals) ──

def _sig_ml_2026(df: pd.DataFrame):
    return _max_conf(df.get("mc_home_win",    pd.Series(dtype=float)),
                     df.get("actual_home_win", pd.Series(dtype=float)))

def _sig_rl_2026(df: pd.DataFrame):
    return _max_conf(df.get("blended_rl",     pd.Series(dtype=float)),
                     df.get("home_covers_rl",  pd.Series(dtype=float)))

def _sig_totals_2026(df: pd.DataFrame):
    """ou_p_model is already directional confidence."""
    prob      = pd.to_numeric(df.get("ou_p_model"),     errors="coerce")
    direction = df.get("ou_direction", pd.Series(dtype=str))
    posted    = pd.to_numeric(df.get("ou_posted_line"), errors="coerce")
    actual    = (pd.to_numeric(df.get("home_score_final"), errors="coerce")
                 + pd.to_numeric(df.get("away_score_final"), errors="coerce"))
    outcome = np.where(
        direction == "OVER",
        (actual > posted).astype(float),
        (actual < posted).astype(float),
    )
    outcome = pd.Series(outcome, index=df.index, dtype=float)
    outcome[posted.isna() | actual.isna()] = np.nan
    return prob, outcome

def _sig_f5_2026(df: pd.DataFrame):
    # Prefer dedicated stacker; fill gaps with Monte Carlo prob
    stacker = pd.to_numeric(df.get("f5_stacker_l2", pd.Series(dtype=float)), errors="coerce")
    mc_prob = pd.to_numeric(df.get("mc_f5_home_win_prob", pd.Series(dtype=float)), errors="coerce")
    prob = stacker.combine_first(mc_prob)
    return _max_conf(prob, df.get("f5_home_win", pd.Series(dtype=float)))

def _sig_nrfi_2026(df: pd.DataFrame):
    return _max_conf(df.get("mc_nrfi_prob", pd.Series(dtype=float)),
                     df.get("f1_nrfi",      pd.Series(dtype=float)))

def _sig_k_over_2026(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Expand home+away K-over into one row each."""
    rows_prob, rows_out = [], []
    for _, r in df.iterrows():
        for side in ("home", "away"):
            p    = pd.to_numeric(r.get(f"{side}_k_model_over"), errors="coerce")
            line = pd.to_numeric(r.get(f"{side}_k_line"),       errors="coerce")
            k    = pd.to_numeric(r.get(f"{side}_sp_k"),         errors="coerce")
            if pd.isna(p) or pd.isna(line) or pd.isna(k):
                continue
            # max-confidence: if model predicts under, flip
            conf = float(p) if float(p) >= 0.5 else 1 - float(p)
            hit  = float(k > line) if float(p) >= 0.5 else float(k <= line)
            rows_prob.append(conf)
            rows_out.append(hit)
    return pd.Series(rows_prob, dtype=float), pd.Series(rows_out, dtype=float)


ALL_SIGNAL_ORDER = ["ML", "Runline", "F5", "NRFI", "K-over"]


def _compute_window_backtest(df: pd.DataFrame, min_n: int,
                             df_k: pd.DataFrame | None = None,
                             date_floor: str | None = None) -> dict:
    """Compute tiers for each market from long-format backtest df.
    ML uses max-confidence (home + away). Others use raw home-centric prob.
    K-over uses separate df_k (daily cards + actuals, 3.5 line)."""
    out = {}
    for code, name in _MARKET_MAP.items():
        sub = df[df["market"] == code].copy()
        if date_floor:
            sub = sub[sub["game_date"] >= date_floor]
        if sub.empty:
            out[name] = {"note": "no data"}
            continue
        if code == "ML":
            prob, outcome = _max_conf(sub["model_prob"], sub["actual"])
        else:
            prob, outcome = sub["model_prob"], sub["actual"]
        out[name] = find_three_tiers(prob, outcome, min_n=min_n)

    # K-over from daily cards
    k = df_k.copy() if df_k is not None and not df_k.empty else pd.DataFrame()
    if date_floor and not k.empty:
        k = k[k["game_date"] >= date_floor]
    if k.empty:
        out["K-over"] = {"note": "no data"}
    else:
        out["K-over"] = find_three_tiers(k["model_prob"], k["actual"], min_n=min_n)
    return out


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_tier(t: dict | None, label: str, symbol: str) -> str:  # noqa: E501
    if t is None:
        return ""
    n   = t.get("n", 0)
    hr  = t.get("hit_rate")
    cut = t.get("cutoff")
    if hr is None or cut is None:
        return ""
    return f"  {symbol} {label:<14} prob >= {cut*100:.0f}%  ->  {hr*100:.1f}%  ({n} games)"


def _fmt_oof(tier: dict) -> str:
    """Single best line for OOF column."""
    opt = tier.get("optimal", {})
    if not opt:
        return "  N/A"
    cut = opt.get("cutoff", 0)
    hr  = opt.get("hit_rate", 0)
    n   = opt.get("n", 0)
    return f"  prob >= {cut*100:.0f}%  ->  {hr*100:.1f}%  ({n} games)"


def _is_flat(r: dict) -> bool:
    """True when all three tiers share the same cutoff (no confidence gradient)."""
    cuts = {r.get(k, {}).get("cutoff") for k in ("optimal", "less_optimal", "stay_away")}
    return len(cuts) == 1 and None not in cuts


def _print_tiers(r: dict) -> None:
    if _is_flat(r):
        opt = r["optimal"]
        print(f"  [*] Single tier    prob >= {opt['cutoff']*100:.0f}%  ->  "
              f"{opt['hit_rate']*100:.1f}%  ({opt['n']} games)  "
              f"[no confidence gradient]")
    else:
        print(_fmt_tier(r.get("optimal"),      "Optimal",      "[*]"))
        print(_fmt_tier(r.get("less_optimal"), "Less optimal", "[o]"))
        print(_fmt_tier(r.get("stay_away"),    "Stay away",    "[x]"))


def print_table(results_2025: dict, results_2026: dict,
                results_30d: dict, all_signals: list[str]) -> None:
    print()
    print("=" * 90)
    print("  SIGNAL ACCURACY TABLE")
    print("  [*] Optimal = highest hit rate | [o] Less optimal | [x] Stay away = >=50% baseline")
    print(f"  Generated: {date.today().isoformat()}")
    print("=" * 90)

    for sig in all_signals:
        print(f"\n{'-'*90}")
        print(f"  {sig}")
        print(f"{'-'*90}")

        # 2025 OOF
        r25 = results_2025.get(sig, {})
        if r25 and "note" not in r25:
            print(f"  {'2025 OOF':<14}  {_fmt_oof(r25).strip()}")
        else:
            print(f"  {'2025 OOF':<14}  N/A (no 2025 predictions for this signal)")

        # 2026 YTD
        r26 = results_2026.get(sig, {})
        print(f"\n  2026 YTD")
        if r26 and "note" not in r26:
            _print_tiers(r26)
        else:
            print(f"    {r26.get('note','no data')}")

        # Last 30 days
        r30 = results_30d.get(sig, {})
        print(f"\n  Last 30 Days")
        if r30 and "note" not in r30:
            _print_tiers(r30)
        else:
            print(f"    {r30.get('note','no data')}")

    print()
    print("=" * 90)


def print_compact_table(results_2025: dict, results_2026: dict,
                        results_30d: dict, all_signals: list[str]) -> None:
    """Compact view matching the screenshot layout."""
    w_sig = 10
    w_col = 42

    header = (f"  {'MARKET':<{w_sig}}  {'2025 OOF':<{w_col}}  "
              f"{'2026 YTD':<{w_col}}  {'LAST 30 DAYS'}")
    print()
    print("=" * (len(header) + 4))
    print(header)
    print("=" * (len(header) + 4))

    for sig in all_signals:
        r25 = results_2025.get(sig, {})
        r26 = results_2026.get(sig, {})
        r30 = results_30d.get(sig,  {})

        def _cell(r: dict, tier: str) -> str:
            t = r.get(tier, {})
            if not t or "cutoff" not in t:
                return "N/A"
            cut = t["cutoff"]
            hr  = t["hit_rate"]
            n   = t["n"]
            return f"prob>={cut*100:.0f}%→{hr*100:.1f}% ({n})"

        def _oof_cell(r: dict) -> str:
            opt = r.get("optimal", {})
            if not opt or "cutoff" not in opt:
                return "N/A"
            return f"prob>={opt['cutoff']*100:.0f}% {opt['hit_rate']*100:.1f}% ({opt['n']})"

        oof  = _oof_cell(r25)
        o26  = _cell(r26, "optimal")
        m26  = _cell(r26, "less_optimal")
        s26  = _cell(r26, "stay_away")
        o30  = _cell(r30, "optimal")
        m30  = _cell(r30, "less_optimal")
        s30  = _cell(r30, "stay_away")

        print(f"\n  {sig:<{w_sig}}")
        print(f"    {'OOF:':<12}  {oof:<{w_col}}")
        print(f"    {'[*] Optimal:':<14}  {o26:<{w_col}}  {o30}")
        print(f"    {'[o] Mid:':<14}  {m26:<{w_col}}  {m30}")
        print(f"    {'[x] Base:':<14}  {s26:<{w_col}}  {s30}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Signal accuracy three-tier table")
    parser.add_argument("--min-n",   type=int, default=DEFAULT_MIN_N,
                        help=f"Minimum games per tier (default {DEFAULT_MIN_N})")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--compact", action="store_true",
                        help="Show compact table instead of detailed")
    args = parser.parse_args()
    min_n = args.min_n

    # Load data
    df_2025    = _load_eval_2025()
    df_f5_25   = _load_f5_val_2025()
    df_nrfi_25 = _load_nrfi_val_2025()
    df_k_25    = _load_k_val_2025()
    df_2026    = _load_2026()
    df_k_26    = _load_k_over_2026()

    cutoff_30d = (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")

    n_dates_ytd = df_2026["game_date"].nunique() if not df_2026.empty else 0
    print(f"\n  2025 OOF (ML/RL)        : {len(df_2025):,} games")
    print(f"  2025 OOF F5             : {len(df_f5_25):,} games")
    print(f"  2025 OOF NRFI           : {len(df_nrfi_25):,} games")
    print(f"  2025 OOF K-over         : {len(df_k_25):,} games")
    print(f"  2026 YTD                : {len(df_2026):,} rows ({n_dates_ytd} dates) "
          f"[{_BACKTEST_26.name} + {_LIVE_26.name}]")
    print(f"  2026 K-over (3.5 line)  : {len(df_k_26):,} SP-game rows ({df_k_26['game_date'].nunique() if not df_k_26.empty else 0} dates)")
    print(f"  Last 30 days since      : {cutoff_30d}")
    print(f"  Min-n cutoff            : {min_n}")

    # Build 2025 results — each signal uses its own source dataframe
    res_2025 = {}
    if not df_2025.empty:
        res_2025["ML"]      = find_three_tiers(*_sig_ml_2025(df_2025),      min_n=min_n)
        res_2025["Runline"] = find_three_tiers(*_sig_rl_2025(df_2025),      min_n=min_n)
    if not df_f5_25.empty:
        res_2025["F5"]      = find_three_tiers(*_sig_f5_2025(df_f5_25),     min_n=min_n)
    if not df_nrfi_25.empty:
        res_2025["NRFI"]    = find_three_tiers(*_sig_nrfi_2025(df_nrfi_25), min_n=min_n)
    if not df_k_25.empty:
        res_2025["K-over"]  = find_three_tiers(*_sig_k_over_2025(df_k_25),  min_n=min_n)

    res_2026 = _compute_window_backtest(df_2026, min_n, df_k=df_k_26)
    res_30d  = _compute_window_backtest(df_2026, min_n, df_k=df_k_26, date_floor=cutoff_30d)

    all_signals = ALL_SIGNAL_ORDER

    if args.compact:
        print_compact_table(res_2025, res_2026, res_30d, all_signals)
    else:
        print_table(res_2025, res_2026, res_30d, all_signals)

    if not args.no_save:
        out = {
            "generated": date.today().isoformat(),
            "min_n": min_n,
            "2025_oof": res_2025,
            "2026_ytd": res_2026,
            "last_30d": res_30d,
        }
        _OUT.write_text(json.dumps(out, indent=2, default=str))
        print(f"  Saved -> {_OUT.name}")


if __name__ == "__main__":
    main()
