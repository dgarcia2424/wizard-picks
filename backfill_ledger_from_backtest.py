"""
backfill_ledger_from_backtest.py
=================================
One-shot tool to populate historical_actionable_picks.csv from the 2026
backtest predictions so the rolling-accuracy tear sheet (7d/28d/YTD) has
real history to show instead of "no bets".

Approach (B, "Relaxed"):
  * Keep edge >= 1.0% and odds_floor_pass=True (same as live).
  * DROP the Pinnacle sanity check — historical odds store only carries
    Pinnacle on ~15% of games, which would leave the ledger empty.
  * Require retail_odds present + actual known so we can grade.
  * Flag each row with sanity_skipped=True so the provenance is visible.

Markets backfilled: ML, TOT, RL.  F5 and NRFI have zero historical odds
coverage, so they remain live-forward only.

Usage:
    python backfill_ledger_from_backtest.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT        = Path(__file__).resolve().parent
BACKTEST    = ROOT / "backtest_full_all_predictions.csv"
LEDGER      = ROOT / "historical_actionable_picks.csv"

_TIER1_EDGE = 0.030
_TIER2_EDGE = 0.010
_KELLY_BANK = 2000.0
_KELLY_CAP  = 50.0

# Mapping backtest market codes → live-ledger conventions.
_MODEL_LABEL = {"ML": "ML", "TOT": "Totals", "RL": "Runline"}

TEAM_ABBR_TO_NAME = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",       "CHC": "Chicago Cubs",    "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds",      "CLE": "Cleveland Guardians", "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",       "HOU": "Houston Astros",  "KC":  "Kansas City Royals",
    "LAA": "Los Angeles Angels",   "LAD": "Los Angeles Dodgers", "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",    "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees",     "ATH": "Oakland Athletics", "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies","PIT": "Pittsburgh Pirates", "SD":  "San Diego Padres",
    "SEA": "Seattle Mariners",     "SF":  "San Francisco Giants", "STL": "St. Louis Cardinals",
    "TB":  "Tampa Bay Rays",       "TEX": "Texas Rangers",   "TOR": "Toronto Blue Jays",
    "WSH": "Washington Nationals",
}


def _american_to_decimal(odds: float) -> float:
    if pd.isna(odds) or odds == 0:
        return float("nan")
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / abs(odds))


def _classify_tier(edge: float) -> int | None:
    if edge is None or pd.isna(edge):
        return None
    if edge >= _TIER1_EDGE:
        return 1
    if edge >= _TIER2_EDGE:
        return 2
    return None


def _kelly_stake(model_prob: float, american_odds: float, tier: int) -> float:
    dec = _american_to_decimal(american_odds)
    if not (dec > 1) or pd.isna(model_prob):
        return 0.0
    b = dec - 1.0
    p = float(model_prob)
    q = 1.0 - p
    f = (b * p - q) / b
    if f <= 0:
        return 0.0
    frac = 0.25 if tier == 1 else 0.125
    raw  = f * frac * _KELLY_BANK
    return round(min(raw, _KELLY_CAP), 2)


def _grade(actual: float, dec_odds: float, stake: float) -> tuple[str, float]:
    if pd.isna(actual) or pd.isna(dec_odds):
        return ("", float("nan"))
    a = int(actual)
    if a == 1:
        return ("WIN", round(stake * (dec_odds - 1.0), 2))
    if a == 0:
        return ("LOSS", round(-stake, 2))
    return ("PUSH", 0.0)


def _row_for_ml(r) -> dict:
    home_abbr = r["home_team"]
    away_abbr = r["away_team"]
    # bet_type is "HOME_ML" or "AWAY_ML"
    direction = "HOME" if str(r["bet_type"]).startswith("HOME") else "AWAY"
    return {
        "model":          "ML",
        "bet_type":       "Moneyline",
        "pick_direction": direction,
        "home_abbr":      home_abbr,
        "away_abbr":      away_abbr,
    }


def _row_for_tot(r) -> dict:
    line = r.get("total_line")
    line_s = f"{line:g}" if pd.notna(line) else "?"
    direction = "OVER" if str(r["bet_type"]).upper() == "OVER" else "UNDER"
    return {
        "model":          "Totals",
        "bet_type":       f"Total {line_s}",
        "pick_direction": direction,
        "home_abbr":      r["home_team"],
        "away_abbr":      r["away_team"],
    }


def _row_for_rl(r) -> dict:
    # bet_type in backtest is "HOME_-1.5"
    direction = "HOME" if str(r["bet_type"]).startswith("HOME") else "AWAY"
    return {
        "model":          "Runline",
        "bet_type":       "Runline -1.5",
        "pick_direction": direction,
        "home_abbr":      r["home_team"],
        "away_abbr":      r["away_team"],
    }


_BUILDERS = {"ML": _row_for_ml, "TOT": _row_for_tot, "RL": _row_for_rl}


def main() -> int:
    if not BACKTEST.exists():
        print(f"ERROR: {BACKTEST} not found. Run backtest_2026_full.py first.", file=sys.stderr)
        return 1

    df = pd.read_csv(BACKTEST)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: list[dict] = []

    for mkt, build in _BUILDERS.items():
        sub = df[df["market"] == mkt].copy()
        # Relaxed gate: drop sanity, keep edge + odds_floor + retail + actual.
        keep = (
            sub["retail_odds"].notna()
            & sub["actual"].notna()
            & (sub["edge"] >= _TIER2_EDGE)
            & (sub.get("odds_floor_pass") == True)
        )
        sub = sub[keep]
        if sub.empty:
            print(f"[{mkt}] no eligible rows")
            continue

        kept = 0
        for _, r in sub.iterrows():
            tier = _classify_tier(r["edge"])
            if tier is None:
                continue
            stake = _kelly_stake(r["model_prob"], r["retail_odds"], tier)
            if stake <= 0:
                continue
            dec   = _american_to_decimal(r["retail_odds"])
            res, pnl = _grade(r["actual"], dec, stake)

            base = build(r)
            away_nm = TEAM_ABBR_TO_NAME.get(base["away_abbr"], base["away_abbr"])
            home_nm = TEAM_ABBR_TO_NAME.get(base["home_abbr"], base["home_abbr"])
            rows.append({
                "date":                 r["game_date"],
                "game":                 f"{away_nm} @ {home_nm}",
                "home_abbr":            base["home_abbr"],
                "away_abbr":            base["away_abbr"],
                "model":                base["model"],
                "bet_type":             base["bet_type"],
                "pick_direction":       base["pick_direction"],
                "model_prob":           float(r["model_prob"]),
                "P_true":               float(r["P_true"])         if pd.notna(r.get("P_true"))         else None,
                "Retail_Implied_Prob":  float(r["retail_imp"])     if pd.notna(r.get("retail_imp"))     else None,
                "edge":                 float(r["edge"]),
                "retail_american_odds": float(r["retail_odds"]),
                "tier":                 tier,
                "dollar_stake":         stake,
                "result":               res,
                "profit_loss":          pnl,
                "graded_at":            now,
                "sanity_skipped":       True,
                "source":               "backfill_2026",
            })
            kept += 1
        print(f"[{mkt}] backfilled {kept} rows")

    if not rows:
        print("Nothing to write.")
        return 0

    new_df = pd.DataFrame(rows)

    # Merge with existing ledger — dedupe by (date, home_abbr, away_abbr, bet_type, pick_direction).
    if LEDGER.exists():
        old = pd.read_csv(LEDGER)
        key_cols = ["date", "home_abbr", "away_abbr", "bet_type", "pick_direction"]
        combined = pd.concat([old, new_df], ignore_index=True)
        before   = len(combined)
        combined = combined.drop_duplicates(subset=key_cols, keep="first")
        removed  = before - len(combined)
        if removed:
            print(f"[ledger] dropped {removed} duplicate rows (live picks win over backfill)")
    else:
        combined = new_df

    # Stable ordering: by date ascending.
    combined = combined.sort_values(["date", "home_abbr", "bet_type"]).reset_index(drop=True)
    combined.to_csv(LEDGER, index=False)

    wins   = (combined["result"] == "WIN").sum()
    losses = (combined["result"] == "LOSS").sum()
    pushes = (combined["result"] == "PUSH").sum()
    pnl    = pd.to_numeric(combined["profit_loss"], errors="coerce").sum()
    print(f"\nLedger now: {len(combined)} rows | WIN={wins} LOSS={losses} PUSH={pushes} | PnL=${pnl:.2f}")
    print(f"Written -> {LEDGER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
