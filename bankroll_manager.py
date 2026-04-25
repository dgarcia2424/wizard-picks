"""
bankroll_manager.py — v5.0 Fractional Kelly Bankroll Engine.

Computes recommended stakes for every PLAY flagged by the alpha report
using the Fractional Kelly Criterion:

    f*     = (b·p − q) / b          (full Kelly fraction)
    stake  = bankroll · f* · kelly_fraction

where:
    b               = decimal_odds − 1   (net profit per $1 wagered)
    p               = model probability
    q               = 1 − p
    kelly_fraction  = 0.5  (half Kelly — recommended; reduces ruin risk ~75%)

Negative f* (no edge or negative EV) → stake = $0, bet skipped.

Sources
-------
  model_scores.csv              — ML / Totals / Runline / F5 / NRFI straight bets
  data/sgp/sgp_live_edge_{date}.csv — SGP PLAY rows (action == "PLAY")

SGP odds derivation
-------------------
The book doesn't post a single SGP price we can directly read.  We back out
the effective payout from the book's joint probability:

    decimal_odds = 1 / p_book_sgp   →   b = (1 / p_book_sgp) − 1

This is conservative: the book's SGP price has extra juice baked in, so the
actual payout is usually slightly less favourable than this implies.

Output
------
  data/bankroll/kelly_stakes_{date}.csv   — one row per PLAY
  Console: stakes table sorted by recommended stake descending

Usage
-----
  python bankroll_manager.py --bankroll 1000
  python bankroll_manager.py --bankroll 2500 --fraction 0.25  # quarter Kelly
  python bankroll_manager.py --bankroll 1000 --sgp-only
  python bankroll_manager.py --bankroll 1000 --straight-only
  python bankroll_manager.py --bankroll 1000 --date 2026-04-24
"""
from __future__ import annotations

import argparse
import glob
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent

MODEL_SCORES    = _ROOT / "model_scores.csv"
SGP_DIR         = _ROOT / "data/sgp"
OUTPUT_DIR      = _ROOT / "data/bankroll"
CLV_AUDIT_PATH  = _ROOT / "clv_audit.csv"
MARKET_REPORT   = _ROOT / "market_beating_report.csv"

# Kelly fractions
KELLY_FULL    = 1.00
KELLY_HALF    = 0.50    # recommended default (half Kelly)
KELLY_QUARTER = 0.25    # conservative floor

# Hard stake caps (% of bankroll) — prevent runaway from high-edge outliers
MAX_STAKE_PCT_STRAIGHT = 0.10   # never risk > 10% per straight bet
MAX_STAKE_PCT_SGP      = 0.05   # never risk > 5% per SGP (higher variance)

# Minimum model edge to consider for staking (filters noise)
MIN_EDGE_FOR_KELLY = 0.02       # 2% edge floor

# CLV feedback loop constants
CLV_LOOKBACK_DAYS  = 14         # rolling window for CLV averaging
CLV_BOOST_MULT     = 1.20       # 1.2x Kelly stake when avg CLV > 0
CLV_MIN_SAMPLE     = 5          # minimum resolved bets to apply boost


# ---------------------------------------------------------------------------
# American odds ↔ decimal conversion
# ---------------------------------------------------------------------------

def american_to_decimal(american: float) -> Optional[float]:
    """Convert American moneyline odds to decimal odds."""
    try:
        american = float(american)
    except (TypeError, ValueError):
        return None
    if np.isnan(american) or american == 0:
        return None
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def implied_to_decimal(p: float) -> Optional[float]:
    """Convert implied probability to decimal odds (fair price, no juice)."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return None
    if p <= 0 or p >= 1:
        return None
    return 1.0 / p


# ---------------------------------------------------------------------------
# Kelly formula
# ---------------------------------------------------------------------------

def kelly_fraction(p: float, decimal_odds: float) -> float:
    """Return full Kelly stake fraction f* = (b·p − q) / b.

    Returns 0.0 if the bet has no positive EV (f* ≤ 0).
    """
    b = decimal_odds - 1.0
    if b <= 0 or p <= 0 or p >= 1:
        return 0.0
    q   = 1.0 - p
    f   = (b * p - q) / b
    return max(0.0, float(f))


def recommended_stake(p_model: float,
                      decimal_odds: float,
                      bankroll: float,
                      fraction: float = KELLY_HALF,
                      cap_pct: float = MAX_STAKE_PCT_STRAIGHT) -> dict:
    """Compute full / half / quarter Kelly stakes and the recommended stake.

    Returns dict with keys:
      kelly_full_pct, kelly_half_pct, kelly_quarter_pct,
      kelly_full_$, kelly_half_$, kelly_quarter_$,
      recommended_$, recommended_fraction_used
    """
    f_full = kelly_fraction(p_model, decimal_odds)

    def _stake(frac: float) -> float:
        raw = bankroll * f_full * frac
        capped = min(raw, bankroll * cap_pct)
        return round(float(capped), 2)

    rec_stake = _stake(fraction)

    return {
        "kelly_full_pct":        round(f_full * 100, 2),
        "kelly_half_pct":        round(f_full * KELLY_HALF * 100, 2),
        "kelly_quarter_pct":     round(f_full * KELLY_QUARTER * 100, 2),
        "kelly_full_$":          _stake(KELLY_FULL),
        "kelly_half_$":          _stake(KELLY_HALF),
        "kelly_quarter_$":       _stake(KELLY_QUARTER),
        "recommended_$":         rec_stake,
        "recommended_fraction":  fraction,
        "edge_required_to_bet":  round((1.0 / decimal_odds), 4) if decimal_odds > 0 else None,
    }


# ---------------------------------------------------------------------------
# Load straight bets from model_scores.csv
# ---------------------------------------------------------------------------

def load_straight_bets(date_str: str) -> pd.DataFrame:
    """Return actionable straight bets for date_str from model_scores.csv."""
    if not MODEL_SCORES.exists():
        return pd.DataFrame()

    df = pd.read_csv(MODEL_SCORES)
    df["date"] = df["date"].astype(str).str.strip()
    df = df[df["date"] == date_str].copy()
    df = df[df["actionable"].fillna(False).astype(bool)].copy()

    if df.empty:
        return df

    # Compute decimal odds from American
    df["decimal_odds"] = df["retail_american_odds"].apply(american_to_decimal)
    df["source"]       = "straight"
    df["bet_label"]    = (df["game"].astype(str) + " | " +
                          df["bet_type"].astype(str) + " " +
                          df["pick_direction"].astype(str))
    df = df.rename(columns={"model_prob": "p_model", "edge": "model_edge"})
    df["cap_pct"] = MAX_STAKE_PCT_STRAIGHT

    keep = ["date", "game", "bet_label", "source", "bet_type",
            "p_model", "model_edge", "decimal_odds",
            "P_true", "retail_american_odds", "tier", "cap_pct"]
    return df[[c for c in keep if c in df.columns]]


# ---------------------------------------------------------------------------
# Load SGP plays from sgp_live_edge_{date}.csv
# ---------------------------------------------------------------------------

def load_sgp_plays(date_str: str) -> pd.DataFrame:
    """Return SGP PLAY rows for date_str."""
    tag = date_str.replace("-", "_")
    p   = SGP_DIR / f"sgp_live_edge_{tag}.csv"

    if not p.exists():
        # Try latest available
        files = sorted(glob.glob(str(SGP_DIR / "sgp_live_edge_*.csv")))
        if not files:
            return pd.DataFrame()
        p = Path(files[-1])
        print(f"  [kelly] SGP file for {date_str} not found; using {p.name}")

    df = pd.read_csv(p)
    df = df[df["action"] == "PLAY"].copy()

    if df.empty:
        return df

    # Derive decimal odds from p_book_sgp (conservative: book's joint price)
    df["decimal_odds"] = df["p_book_sgp"].apply(
        lambda p_bk: implied_to_decimal(float(p_bk)) if pd.notna(p_bk) and float(p_bk) > 0 else None)

    df["source"]     = "sgp"
    df["p_model"]    = df["p_joint_copula"]
    df["model_edge"] = df["sgp_edge"]
    df["date"]       = date_str
    df["bet_label"]  = df["game"].astype(str) + " | " + df["script"].astype(str)
    df["cap_pct"]    = MAX_STAKE_PCT_SGP

    keep = ["date", "game", "bet_label", "source", "script",
            "p_model", "model_edge", "decimal_odds",
            "p_book_sgp", "p_joint_copula", "corr_lift", "legs", "cap_pct"]
    return df[[c for c in keep if c in df.columns]]


# ---------------------------------------------------------------------------
# CLV Feedback Loop — 14-day rolling edge by script / bet-type
# ---------------------------------------------------------------------------

def _load_sgp_clv_history(reference_date: str,
                           lookback: int = CLV_LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Scan sgp_live_edge_*.csv files from the last `lookback` days.
    Returns long frame: [script, sgp_edge, date] for PLAY-action rows.
    """
    ref_dt  = date.fromisoformat(reference_date)
    cutoff  = ref_dt - timedelta(days=lookback)
    rows    = []

    for fpath in sorted(glob.glob(str(SGP_DIR / "sgp_live_edge_*.csv"))):
        try:
            # Parse date from filename: sgp_live_edge_2026_04_20.csv
            stem = Path(fpath).stem.replace("sgp_live_edge_", "")
            file_date = date.fromisoformat(stem.replace("_", "-"))
        except ValueError:
            continue

        if file_date < cutoff or file_date >= ref_dt:
            continue

        try:
            df = pd.read_csv(fpath)
            df = df[df["action"] == "PLAY"].copy()
            if df.empty or "script" not in df.columns:
                continue
            df["_date"] = file_date.isoformat()
            rows.append(df[["script", "sgp_edge", "_date"]])
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["script", "sgp_edge", "_date"])
    return pd.concat(rows, ignore_index=True)


def _load_straight_clv_history(reference_date: str,
                                 lookback: int = CLV_LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Load clv_audit.csv and return resolved bets within the lookback window.
    Falls back to market_beating_report.csv if available.

    Returns long frame: [bet_type, edge_pct, date] for resolved rows.
    """
    ref_dt = date.fromisoformat(reference_date)
    cutoff = ref_dt - timedelta(days=lookback)

    # Prefer market_beating_report if it exists (has script-level CLV)
    if MARKET_REPORT.exists():
        try:
            rpt = pd.read_csv(MARKET_REPORT)
            if "script" in rpt.columns and "clv" in rpt.columns:
                rpt["_date"] = rpt.get("date", reference_date)
                rpt["_dt"]   = pd.to_datetime(rpt["_date"], errors="coerce").dt.date
                rpt = rpt[(rpt["_dt"] >= cutoff) & (rpt["_dt"] < ref_dt)]
                return rpt[["script", "clv", "_date"]].rename(
                    columns={"script": "bet_type", "clv": "edge_pct"})
        except Exception:
            pass

    if not CLV_AUDIT_PATH.exists():
        return pd.DataFrame(columns=["bet_type", "edge_pct", "_date"])

    df = pd.read_csv(CLV_AUDIT_PATH)
    df["_dt"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df[(df["_dt"] >= cutoff) & (df["_dt"] < ref_dt)]
    # Only count resolved (non-pending) bets
    df = df[df["outcome"].fillna("PENDING") != "PENDING"].copy()
    if df.empty:
        return pd.DataFrame(columns=["bet_type", "edge_pct", "_date"])

    df["bet_type"] = df["bet_type"].fillna("UNKNOWN")
    df["_date"]    = df["date"].astype(str)
    return df[["bet_type", "edge_pct", "_date"]]


def build_clv_multipliers(date_str: str,
                           lookback: int = CLV_LOOKBACK_DAYS) -> dict[str, float]:
    """
    Compute CLV confidence multipliers for each script / bet-type.

    Returns dict mapping script/bet_type → multiplier:
      CLV_BOOST_MULT (1.2x) if rolling avg edge > 0 with >= CLV_MIN_SAMPLE bets
      1.0 otherwise (no boost)
    """
    mults: dict[str, float] = {}

    # SGP scripts
    sgp_hist = _load_sgp_clv_history(date_str, lookback)
    if not sgp_hist.empty:
        for script, grp in sgp_hist.groupby("script"):
            if len(grp) >= CLV_MIN_SAMPLE and grp["sgp_edge"].mean() > 0:
                mults[str(script)] = CLV_BOOST_MULT

    # Straight bet types
    str_hist = _load_straight_clv_history(date_str, lookback)
    if not str_hist.empty:
        for bet_type, grp in str_hist.groupby("bet_type"):
            if len(grp) >= CLV_MIN_SAMPLE and grp["edge_pct"].mean() > 0:
                mults[str(bet_type)] = CLV_BOOST_MULT

    if mults:
        print(f"  [clv] boosted scripts/types: "
              f"{', '.join(f'{k}={v:.2f}x' for k, v in mults.items())}")
    else:
        print(f"  [clv] no scripts with sufficient positive CLV history "
              f"(lookback={lookback}d, min_sample={CLV_MIN_SAMPLE})")

    return mults


def apply_clv_boost(result: pd.DataFrame,
                    clv_mults: dict[str, float],
                    bankroll: float,
                    fraction: float) -> pd.DataFrame:
    """
    Apply CLV confidence multipliers to recommended stakes.

    Looks up the script (SGP) or bet_type (straight) for each row.
    Multiplied stake is still capped at the hard stake cap.
    Adds `clv_boosted` (bool) and `clv_multiplier` columns.
    """
    if result.empty or not clv_mults:
        result["clv_boosted"]    = False
        result["clv_multiplier"] = 1.0
        return result

    result = result.copy()
    result["clv_boosted"]    = False
    result["clv_multiplier"] = 1.0

    for idx, row in result.iterrows():
        # Determine lookup key: script for SGP, bet_type for straight
        key = str(row.get("script") or row.get("bet_label", ""))
        mult = clv_mults.get(key, 1.0)

        # Also try the base bet_type extracted from bet_label for straight bets
        if mult == 1.0 and row.get("source") == "straight":
            label = str(row.get("bet_label", ""))
            for bt_key in clv_mults:
                if bt_key in label:
                    mult = clv_mults[bt_key]
                    break

        if mult > 1.0:
            cap_pct = MAX_STAKE_PCT_SGP if row.get("source") == "sgp" \
                      else MAX_STAKE_PCT_STRAIGHT
            raw_new  = result.at[idx, "recommended_$"] * mult
            capped   = min(raw_new, bankroll * cap_pct)
            result.at[idx, "recommended_$"] = round(float(capped), 2)
            result.at[idx, "clv_boosted"]   = True
            result.at[idx, "clv_multiplier"] = mult

    return result


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

def compute_kelly_stakes(date_str: str,
                         bankroll: float,
                         fraction: float = KELLY_HALF,
                         include_straight: bool = True,
                         include_sgp: bool = True,
                         clv_boost: bool = False,
                         ) -> pd.DataFrame:
    """Build full Kelly stake table for all actionable plays on date_str."""
    frames = []
    if include_straight:
        frames.append(load_straight_bets(date_str))
    if include_sgp:
        frames.append(load_sgp_plays(date_str))

    all_plays = pd.concat([f for f in frames if not f.empty],
                           ignore_index=True) if frames else pd.DataFrame()
    if all_plays.empty:
        return pd.DataFrame()

    # Filter minimum edge
    all_plays = all_plays[
        all_plays["model_edge"].fillna(0) >= MIN_EDGE_FOR_KELLY
    ].copy()

    rows = []
    for _, play in all_plays.iterrows():
        p       = float(play.get("p_model") or 0)
        dec_odd = play.get("decimal_odds")
        cap     = float(play.get("cap_pct", MAX_STAKE_PCT_STRAIGHT))

        if not dec_odd or pd.isna(dec_odd) or dec_odd <= 1 or p <= 0 or p >= 1:
            stakes = {
                "kelly_full_pct": 0, "kelly_half_pct": 0, "kelly_quarter_pct": 0,
                "kelly_full_$": 0, "kelly_half_$": 0, "kelly_quarter_$": 0,
                "recommended_$": 0, "recommended_fraction": fraction,
                "edge_required_to_bet": None,
            }
        else:
            stakes = recommended_stake(p, float(dec_odd), bankroll,
                                       fraction=fraction, cap_pct=cap)

        row = {
            "date":           play.get("date", date_str),
            "game":           play.get("game", ""),
            "bet_label":      play.get("bet_label", ""),
            "source":         play.get("source", ""),
            "p_model":        round(float(p), 4) if p else None,
            "model_edge":     round(float(play.get("model_edge", 0) or 0), 4),
            "decimal_odds":   round(float(dec_odd), 3) if dec_odd else None,
            "bankroll":       bankroll,
            **stakes,
        }
        # Carry through any script/tier metadata
        for col in ("tier", "script", "legs", "corr_lift"):
            if col in play.index:
                row[col] = play[col]
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    # Sort by recommended stake descending, then edge
    result = result.sort_values(
        ["recommended_$", "model_edge"], ascending=[False, False]
    ).reset_index(drop=True)

    # Optional CLV feedback boost
    if clv_boost:
        clv_mults = build_clv_multipliers(date_str)
        result    = apply_clv_boost(result, clv_mults, bankroll, fraction)
    else:
        result["clv_boosted"]    = False
        result["clv_multiplier"] = 1.0

    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(df: pd.DataFrame, bankroll: float, fraction: float) -> None:
    print("=" * 75)
    print(f"  KELLY BANKROLL ENGINE  |  Bankroll: ${bankroll:,.2f}  "
          f"|  Fraction: {fraction:.0%} Kelly")
    print("=" * 75)

    if df.empty:
        print("  No actionable plays found.")
        return

    total_recommended = df["recommended_$"].sum()
    pct_deployed      = total_recommended / bankroll * 100

    print(f"\n  {'#':>3}  {'BET':<42} {'SRC':>5}  {'P_MDL':>6}  "
          f"{'EDGE':>6}  {'DEC_ODD':>8}  {'FULL_K%':>7}  {'REC_$':>7}")
    print("  " + "-" * 95)

    for i, r in df.iterrows():
        src_tag   = "SGP" if r.get("source") == "sgp" else "STR"
        label     = str(r.get("bet_label", ""))[:42]
        clv_flag  = " ★" if r.get("clv_boosted") else "  "
        print(f"  {i+1:>3}  {label:<42} {src_tag:>5}  "
              f"{r['p_model']:>6.3f}  {r['model_edge']:>+6.3f}  "
              f"{r['decimal_odds'] or 0:>8.3f}  "
              f"{r['kelly_full_pct']:>6.2f}%  "
              f"${r['recommended_$']:>7.2f}{clv_flag}")

    print(f"\n  Total recommended stake:  ${total_recommended:,.2f}  "
          f"({pct_deployed:.1f}% of bankroll)")
    print(f"  Plays: {len(df)} total  |  "
          f"SGP: {(df['source']=='sgp').sum()}  |  "
          f"Straight: {(df['source']=='straight').sum()}")

    # Kelly breakdown
    print(f"\n  Stake sensitivity (full slate):")
    print(f"    Full  Kelly (100%): ${df['kelly_full_$'].sum():,.2f}")
    print(f"    Half  Kelly  (50%): ${df['kelly_half_$'].sum():,.2f}")
    print(f"    Qtr   Kelly  (25%): ${df['kelly_quarter_$'].sum():,.2f}")

    # CLV boost summary
    if "clv_boosted" in df.columns and df["clv_boosted"].any():
        n_boosted  = df["clv_boosted"].sum()
        boost_adds = df.loc[df["clv_boosted"], "recommended_$"].sum()
        print(f"\n  CLV boost: {n_boosted} plays boosted ({CLV_BOOST_MULT:.0%})  "
              f"★ = CLV-boosted stake")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fractional Kelly bankroll engine (v5.0)")
    parser.add_argument("--bankroll",      type=float, required=True,
                        help="Current bankroll in dollars (e.g. 1000)")
    parser.add_argument("--fraction",      type=float, default=KELLY_HALF,
                        help=f"Kelly fraction (default: {KELLY_HALF} = half Kelly)")
    parser.add_argument("--date",          default=date.today().isoformat())
    parser.add_argument("--sgp-only",      action="store_true")
    parser.add_argument("--straight-only", action="store_true")
    parser.add_argument("--save",          action="store_true",
                        help="Save stakes to data/bankroll/kelly_stakes_{date}.csv")
    parser.add_argument("--clv-boost",     action="store_true",
                        help=f"Apply {int(CLV_BOOST_MULT*100)}pct CLV confidence boost for "
                             f"scripts with positive {CLV_LOOKBACK_DAYS}-day rolling edge")
    args = parser.parse_args()

    include_sgp      = not args.straight_only
    include_straight = not args.sgp_only

    df = compute_kelly_stakes(
        date_str=args.date,
        bankroll=args.bankroll,
        fraction=args.fraction,
        include_straight=include_straight,
        include_sgp=include_sgp,
        clv_boost=args.clv_boost,
    )

    print_report(df, args.bankroll, args.fraction)

    if args.save and not df.empty:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR / f"kelly_stakes_{args.date}.csv"
        df.to_csv(out, index=False)
        print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
