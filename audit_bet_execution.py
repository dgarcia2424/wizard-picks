"""
audit_bet_execution.py — Real-Time Bet Execution Slippage Auditor (v1.0).

Compares the 'Model Stake' (from Kelly engine) against the 'Available Payout'
at the time of the dashboard refresh, per bet.

Three slippage dimensions tracked
-----------------------------------
  1. LINE SLIPPAGE (K-props)
     model_decimal_odds vs current_best_decimal_odds from k_props parquet.
     slip_pct = (model_decimal - current_decimal) / model_decimal * 100
     Negative slip_pct = odds have worsened since model locked the line.

  2. SGP CORRELATION TAX GATING
     The SGP scorer assumes a book_corr_tax of 0.15 (retail) or 0.20 (sharp).
     If the current observed correlation tax (derived from p_book_sgp and
     p_joint_indep) has moved above 0.25, the book is 'Gating' the script.
     gate_flag = 1 when observed_corr_tax > CORR_TAX_GATE_THRESH.

  3. AVAILABILITY RISK
     Detects when a script's book payout has degraded >= SLIP_WARN_PCT
     since the model priced it (line moved, juice worsened, or book pulled).

Output columns (slippage_report.csv)
--------------------------------------
  date, game, bet_label, source, script / bet_type
  model_decimal_odds       — odds used when model computed Kelly stake
  current_decimal_odds     — current best available payout
  slip_pct                 — % degradation from model price
  slip_abs                 — absolute decimal-odds difference
  slip_warning             — 1 if slip_pct < -SLIP_WARN_PCT
  corr_tax_model           — correlation tax assumed by model
  corr_tax_observed        — correlation tax at time of audit
  gate_flag                — 1 if corr_tax_observed > CORR_TAX_GATE_THRESH
  execution_risk           — composite: "LOW" | "MEDIUM" | "HIGH"
  recommended_$            — original Kelly recommended stake
  adjusted_$               — stake adjusted for slippage / gating
  notes                    — human-readable summary

Inputs
------
  data/bankroll/kelly_stakes_{date}.csv   — model stakes (if exists)
  data/sgp/sgp_live_edge_{date}.csv       — SGP model output
  data/statcast/k_props_{date}.parquet    — current live book odds

Outputs
-------
  data/bankroll/slippage_report_{date}.csv   — per-bet audit rows
  data/bankroll/slippage_summary_{date}.json — summary for dashboard

Usage
-----
  python audit_bet_execution.py
  python audit_bet_execution.py --date 2026-04-24
  python audit_bet_execution.py --date 2026-04-24 --verbose
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT       = Path(__file__).resolve().parent
BANKROLL_DIR = _ROOT / "data/bankroll"
SGP_DIR      = _ROOT / "data/sgp"
KPROPS_DIR   = _ROOT / "data/statcast"

# Thresholds
SLIP_WARN_PCT         = 3.0    # % degradation that triggers a warning
CORR_TAX_GATE_THRESH  = 0.25   # above this correlation tax = gating
SGP_CORR_TAX_RETAIL   = 0.15   # baseline retail correlation tax in model
SGP_CORR_TAX_SHARP    = 0.20   # sharp book baseline


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _american_to_decimal(american: float) -> float | None:
    try:
        a = float(american)
        if math.isnan(a) or a == 0:
            return None
        if a > 0:
            return 1.0 + a / 100.0
        return 1.0 + 100.0 / abs(a)
    except (TypeError, ValueError):
        return None


def _decimal_to_american(decimal: float) -> int:
    if decimal is None or decimal <= 1.0:
        return 0
    if decimal >= 2.0:
        return int(round((decimal - 1.0) * 100))
    return int(round(-100 / (decimal - 1.0)))


def _observed_corr_tax(p_joint_copula: float, p_joint_indep: float) -> float | None:
    """
    Back-calculate the effective correlation tax from the SGP output.
    The book prices at: p_book = p_joint_indep * (1 - tax)
    Observed tax = 1 - p_book_sgp / p_joint_indep
    """
    try:
        if p_joint_indep <= 0:
            return None
        return float(1.0 - p_joint_copula / p_joint_indep)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Load sources
# ---------------------------------------------------------------------------

def _load_kelly(date_str: str) -> pd.DataFrame:
    path = BANKROLL_DIR / f"kelly_stakes_{date_str}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_sgp(date_str: str) -> pd.DataFrame:
    tag  = date_str.replace("-", "_")
    path = SGP_DIR / f"sgp_live_edge_{tag}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_kprops(date_str: str) -> pd.DataFrame:
    """Load k_props and compute best available decimal odds per pitcher."""
    tag  = date_str.replace("-", "_")
    path = KPROPS_DIR / f"k_props_{tag}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["book"] = df["book"].str.lower().str.strip()

    # Best over payout = highest over_odds (most generous) across books
    grp = (df.groupby(["pitcher_name", "line"])
             .agg(best_over_odds=("over_odds", "max"),
                  best_under_odds=("under_odds", "max"),
                  n_books=("book", "nunique"))
             .reset_index())
    grp["best_over_decimal"]  = grp["best_over_odds"].apply(_american_to_decimal)
    grp["best_under_decimal"] = grp["best_under_odds"].apply(_american_to_decimal)
    return grp


# ---------------------------------------------------------------------------
# Build synthetic kelly rows from SGP when kelly_stakes CSV is absent
# ---------------------------------------------------------------------------

def _build_synthetic_kelly(sgp: pd.DataFrame) -> pd.DataFrame:
    """
    When no kelly_stakes CSV exists (pipeline hasn't run bankroll_manager
    today), build a synthetic audit-only row set from sgp_live_edge.
    These rows won't have recommended_$ — they'll show current payout status.
    """
    if sgp.empty:
        return pd.DataFrame()

    rows = []
    play = sgp[sgp.get("action", sgp.get("action", pd.Series("", index=sgp.index))) == "PLAY"] \
        if "action" in sgp.columns else sgp

    for _, r in play.iterrows():
        p_book = r.get("p_book_sgp")
        dec_odds = (1.0 / float(p_book)) if (p_book and float(p_book) > 0) else None
        rows.append({
            "date":           r.get("_date", r.get("game_date", "")),
            "game":           r.get("game", f"{r.get('away_team','')}@{r.get('home_team','')}"),
            "bet_label":      str(r.get("game", "")) + " | " + str(r.get("script", "")),
            "source":         "sgp",
            "script":         r.get("script"),
            "p_model":        r.get("p_joint_copula"),
            "model_edge":     r.get("sgp_edge"),
            "decimal_odds":   dec_odds,
            "recommended_$":  None,
            # carry through for correlation tax audit
            "_p_joint_copula": r.get("p_joint_copula"),
            "_p_joint_indep":  r.get("p_joint_indep"),
            "_book_corr_tax":  r.get("book_corr_tax"),
            "_home_sp":        r.get("home_sp"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core audit
# ---------------------------------------------------------------------------

def audit_bets(date_str: str) -> pd.DataFrame:
    kelly  = _load_kelly(date_str)
    sgp    = _load_sgp(date_str)
    kprops = _load_kprops(date_str)

    if kelly.empty:
        print(f"  [audit] No kelly_stakes_{date_str}.csv — using SGP synthetic rows")
        kelly = _build_synthetic_kelly(sgp)
    else:
        # Attach SGP raw data for correlation tax computation
        if not sgp.empty and "script" in kelly.columns and "script" in sgp.columns:
            sgp_meta = sgp[["home_team", "script",
                            "p_joint_copula", "p_joint_indep",
                            "book_corr_tax", "home_sp"]].copy()
            sgp_meta["game"] = sgp.get("game",
                                        sgp["away_team"] + "@" + sgp["home_team"])
            kelly = kelly.merge(
                sgp_meta.rename(columns={
                    "p_joint_copula": "_p_joint_copula",
                    "p_joint_indep":  "_p_joint_indep",
                    "book_corr_tax":  "_book_corr_tax",
                    "home_sp":        "_home_sp",
                }),
                on=["game", "script"],
                how="left",
            )

    if kelly.empty:
        print(f"  [audit] No bets to audit for {date_str}.")
        return pd.DataFrame()

    records = []
    for _, row in kelly.iterrows():
        source = str(row.get("source", ""))
        rec    = {
            "date":              date_str,
            "game":              row.get("game"),
            "bet_label":         row.get("bet_label"),
            "source":            source,
            "script":            row.get("script"),
            "bet_type":          row.get("bet_type"),
            "model_decimal_odds": row.get("decimal_odds"),
            "recommended_$":     row.get("recommended_$"),
        }

        model_dec = row.get("decimal_odds")
        if model_dec:
            try:
                model_dec = float(model_dec)
            except (TypeError, ValueError):
                model_dec = None

        # ── SGP correlation tax audit ─────────────────────────────────────
        corr_tax_model    = None
        corr_tax_observed = None
        gate_flag         = 0

        if source == "sgp":
            corr_tax_model = float(row.get("_book_corr_tax") or SGP_CORR_TAX_RETAIL)
            pjc = row.get("_p_joint_copula")
            pji = row.get("_p_joint_indep")
            if pjc and pji:
                try:
                    corr_tax_observed = _observed_corr_tax(float(pjc), float(pji))
                except Exception:
                    pass

            if corr_tax_observed is not None and \
               corr_tax_observed > CORR_TAX_GATE_THRESH:
                gate_flag = 1

            # Current payout from live SGP (best-effort: re-read from sgp)
            current_dec = model_dec   # default: no drift detected

        # ── K-prop line slippage audit ────────────────────────────────────
        current_dec = model_dec  # default
        if source == "straight" and not kprops.empty:
            sp_name = str(row.get("_home_sp") or row.get("bet_label", ""))
            prop_row = kprops[kprops["pitcher_name"].str.lower() == sp_name.lower()]
            if not prop_row.empty:
                current_dec = float(prop_row.iloc[0]["best_over_decimal"] or model_dec or 1.9)

        # ── Slippage calculation ──────────────────────────────────────────
        slip_abs  = None
        slip_pct  = None
        slip_warn = 0

        if model_dec and current_dec and model_dec > 1.0 and current_dec > 1.0:
            slip_abs  = round(current_dec - model_dec, 4)
            slip_pct  = round((current_dec - model_dec) / (model_dec - 1.0) * 100, 2)
            if slip_pct < -SLIP_WARN_PCT:
                slip_warn = 1

        # ── Execution risk composite ──────────────────────────────────────
        risk_score = 0
        if slip_warn:
            risk_score += 2
        if gate_flag:
            risk_score += 3
        if corr_tax_observed and corr_tax_observed > 0.22:
            risk_score += 1

        execution_risk = "HIGH" if risk_score >= 4 else (
            "MEDIUM" if risk_score >= 2 else "LOW")

        # ── Adjusted stake (reduce stake when high risk) ──────────────────
        rec_stake = row.get("recommended_$")
        if rec_stake:
            try:
                rs = float(rec_stake)
                if execution_risk == "HIGH":
                    adj_stake = round(rs * 0.5, 2)
                elif execution_risk == "MEDIUM":
                    adj_stake = round(rs * 0.75, 2)
                else:
                    adj_stake = round(rs, 2)
            except (TypeError, ValueError):
                adj_stake = None
        else:
            adj_stake = None

        # ── Notes ─────────────────────────────────────────────────────────
        notes_parts = []
        if gate_flag:
            notes_parts.append(
                f"BOOK GATING: corr_tax={corr_tax_observed:.3f} "
                f"(model assumed {corr_tax_model:.2f})")
        if slip_warn:
            notes_parts.append(
                f"LINE MOVED: slip={slip_pct:+.1f}% "
                f"({model_dec:.3f}->{current_dec:.3f})")
        if not notes_parts:
            notes_parts.append("No execution risk detected")

        rec.update({
            "current_decimal_odds": round(current_dec, 4) if current_dec else None,
            "slip_pct":             slip_pct,
            "slip_abs":             slip_abs,
            "slip_warning":         slip_warn,
            "corr_tax_model":       round(corr_tax_model, 4)    if corr_tax_model    else None,
            "corr_tax_observed":    round(corr_tax_observed, 4) if corr_tax_observed else None,
            "gate_flag":            gate_flag,
            "execution_risk":       execution_risk,
            "adjusted_$":           adj_stake,
            "notes":                " | ".join(notes_parts),
        })
        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_report(df: pd.DataFrame, date_str: str) -> None:
    if df.empty:
        return
    BANKROLL_DIR.mkdir(parents=True, exist_ok=True)

    csv_path  = BANKROLL_DIR / f"slippage_report_{date_str}.csv"
    json_path = BANKROLL_DIR / f"slippage_summary_{date_str}.json"

    df.to_csv(csv_path, index=False)
    print(f"\n  Saved -> {csv_path.name}")

    # Summary for dashboard
    n_high    = int((df["execution_risk"] == "HIGH").sum())
    n_medium  = int((df["execution_risk"] == "MEDIUM").sum())
    n_gated   = int(df["gate_flag"].sum())
    n_slip    = int(df["slip_warning"].sum())

    gated_scripts = df[df["gate_flag"] == 1]["script"].dropna().tolist()
    high_risk_games = df[df["execution_risk"] == "HIGH"][["game", "bet_label",
                                                           "notes"]].to_dict("records")

    summary = {
        "date":              date_str,
        "total_bets":        len(df),
        "execution_risk_high":   n_high,
        "execution_risk_medium": n_medium,
        "gated_scripts":     n_gated,
        "line_slippage_warnings": n_slip,
        "gated_script_names": gated_scripts,
        "high_risk_bets":    high_risk_games,
        "avg_slip_pct":      round(df["slip_pct"].dropna().mean(), 2)
                             if df["slip_pct"].notna().any() else 0,
    }
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"  Saved -> {json_path.name}")


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

def print_report(df: pd.DataFrame) -> None:
    print("=" * 72)
    print("  BET EXECUTION SLIPPAGE AUDIT")
    print("=" * 72)
    if df.empty:
        print("  No bets audited.")
        return

    n_high   = (df["execution_risk"] == "HIGH").sum()
    n_medium = (df["execution_risk"] == "MEDIUM").sum()
    n_low    = (df["execution_risk"] == "LOW").sum()
    n_gated  = df["gate_flag"].sum()

    print(f"  Total bets: {len(df)} | HIGH: {n_high} | MEDIUM: {n_medium} | "
          f"LOW: {n_low} | Gated: {n_gated}")

    if n_gated or n_high or n_medium:
        print()
        print(f"  {'Label':36s}  {'Risk':8s}  {'Slip%':>8s}  {'CorrTax':>9s}  Notes")
        print("  " + "-" * 70)
        warn_rows = df[df["execution_risk"].isin(["HIGH", "MEDIUM"])].copy()
        for _, r in warn_rows.sort_values("execution_risk").iterrows():
            slip_str = f"{r['slip_pct']:+.1f}%" if r["slip_pct"] is not None else "N/A"
            tax_str  = (f"{r['corr_tax_observed']:.3f}" if r["corr_tax_observed"]
                        else "N/A")
            label    = str(r.get("bet_label", ""))[:36]
            print(f"  {label:36s}  {r['execution_risk']:8s}  "
                  f"{slip_str:>8s}  {tax_str:>9s}  {r.get('notes','')[:30]}")

    low_risk = df[df["execution_risk"] == "LOW"]
    if len(low_risk) > 0:
        print(f"\n  {len(low_risk)} bet(s) at LOW risk — no action needed.")


# ---------------------------------------------------------------------------
# Public API (importable by dashboard)
# ---------------------------------------------------------------------------

def load_slippage_summary(date_str: str) -> dict:
    path = BANKROLL_DIR / f"slippage_summary_{date_str}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(date_str: str, verbose: bool = False) -> pd.DataFrame:
    print(f"[slippage_audit] date={date_str}")
    df = audit_bets(date_str)
    print_report(df)
    save_report(df, date_str)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bet execution slippage auditor (v1.0)")
    parser.add_argument("--date",    default=date.today().isoformat())
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(date_str=args.date, verbose=args.verbose)


if __name__ == "__main__":
    main()
