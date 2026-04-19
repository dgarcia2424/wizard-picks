"""
mlb_execution_agent.py
======================
Standalone execution engine: Three-Part Lock gatekeeper + terminal payload.

Takes any pandas DataFrame that contains the columns documented in
``REQUIRED_COLS`` and ``OPTIONAL_COLS`` and returns a list of ``LockResult``
objects — one per market (RL / ML / O-U) that passed all three gates.

Design contract
---------------
All financial constants (bankroll, Kelly fractions, odds floor, edge tiers,
sanity threshold) are imported from run_today.py so there is a single source
of truth.  This module never re-defines them.

Typical call-sites:
    from mlb_execution_agent import score_card, print_payload

    # From run_today.py (results list → DataFrame conversion):
    df = pd.DataFrame(run_card(date_str))
    locks = score_card(df)
    print_payload(locks, date_str)

    # Standalone from any merged model-scores + odds DataFrame:
    python mlb_execution_agent.py --csv daily_card.csv

Column contract (DataFrame that goes into score_card)
------------------------------------------------------
Required for RL gate:
    game               str    "TB @ CWS"
    home_team          str
    away_team          str
    lock_p_model       float  stacking-calibrated RL home probability
    lock_p_true        float  Pinnacle de-vigged implied (NaN if not posted)
    lock_retail_implied float retail book de-vigged implied
    rl_odds            float  American odds on the retail RL home line

Required for ML gate:
    ml_lock_p_model       float
    ml_lock_p_true        float
    ml_lock_retail_implied float
    vegas_ml_home          float  American odds on retail ML home

Required for O/U gate:
    ou_p_model         float  MC over probability
    ou_p_true          float  Pinnacle over implied
    ou_p_retail        float  retail over implied
    ou_posted_line     float  e.g. 8.5
    ou_direction       str    "OVER" | "UNDER"
    close_total        float  (used for odds; absent = no odds_floor check on O/U)

Optional (enriches terminal payload):
    home_sp, away_sp            str   starter name
    home_sp_xwoba, away_sp_xwoba float
    home_sp_flag, away_sp_flag  str   VOLATILE / GAINER / NORMAL / UNKNOWN
    pinnacle_rl_home_odds       float Pinnacle RL odds (for PIN column)
    pinnacle_ml_home            float Pinnacle ML odds
    game_time_et                str   "1:05 pm ET"
    lineup_confirmed            bool
    retail_book_used            str   "DraftKings" etc.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import financial constants from run_today.py (single source of truth)
# ---------------------------------------------------------------------------
# We do a targeted import so this module can also be run standalone from
# any directory — if run_today.py is on the path it will be found.
try:
    from run_today import (
        SYNTHETIC_BANKROLL,
        MAX_BET_DOLLARS,
        ODDS_FLOOR,
        EDGE_TIER1,
        EDGE_TIER2,
        SANITY_THRESHOLD,
        SANITY_THRESHOLD_RETAIL,
    )
except ImportError:
    # Fallback definitions when used outside the project root
    SYNTHETIC_BANKROLL      = 2_000.0
    MAX_BET_DOLLARS         = 50.0
    ODDS_FLOOR              = -225.0
    EDGE_TIER1              = 0.030   # >= 3.0%  → Tier 1 strong
    EDGE_TIER2              = 0.010   # >= 1.0%  → Tier 2 medium
    SANITY_THRESHOLD        = 0.04
    SANITY_THRESHOLD_RETAIL = 0.08

# ---------------------------------------------------------------------------
# ANSI colour helpers (disabled on non-tty or Windows without FORCE_COLOR)
# ---------------------------------------------------------------------------
import os as _os
_USE_COLOR = sys.stdout.isatty() or _os.environ.get("FORCE_COLOR") == "1"

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def _green(t):  return _c(t, "92")
def _yellow(t): return _c(t, "93")
def _red(t):    return _c(t, "91")
def _cyan(t):   return _c(t, "96")
def _bold(t):   return _c(t, "1")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

MARKET_LABELS = {
    "RL":  "RL -1.5",
    "ML":  "ML WIN",
    "OVR": "OVER",
    "UND": "UNDER",
}

TIER_LABELS = {
    1: "TIER 1: STRONG EDGE",
    2: "TIER 2: MEDIUM EDGE",
}


@dataclass
class GateTrace:
    """Records which gate passed/failed and why — for diagnostics."""
    sanity_pass:      bool = False
    sanity_source:    str  = ""     # "pinnacle" | "retail_fallback" | "none"
    odds_floor_pass:  bool = False
    edge_pass:        bool = False
    edge:             float = float("nan")
    fail_reason:      str  = ""     # first gate that failed


@dataclass
class LockResult:
    """One locked pick that passed all three gates."""
    game:         str
    home_team:    str
    away_team:    str
    market:       str          # "RL" | "ML" | "OVR" | "UND"
    tier:         int          # 1 = strong  2 = medium
    p_model:      float
    p_true:       float | None
    p_retail:     float
    edge:         float        # p_model - p_retail
    pinnacle_odds: float | None  # American, for PIN column
    retail_odds:  float | None   # American, for [Book] column
    retail_book:  str          # e.g. "DraftKings"
    stake:        float        # Kelly-sized, capped, rounded
    kelly_f:      float        # raw fractional Kelly before scaling
    ou_line:      float | None  = None
    home_sp:      str          = ""
    away_sp:      str          = ""
    home_sp_xwoba: float | None = None
    away_sp_xwoba: float | None = None
    home_sp_flag: str          = ""
    away_sp_flag: str          = ""
    game_time:    str          = ""
    lineup_confirmed: bool     = False
    trace:        GateTrace    = field(default_factory=GateTrace)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _nan(val: Any) -> bool:
    """True when val is None / NaN / empty string / 'nan'."""
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    if isinstance(val, str) and val.lower() in ("nan", "none", ""):
        return True
    try:
        return np.isnan(float(val))
    except (TypeError, ValueError):
        return False


def _f(val: Any, default: float = float("nan")) -> float:
    """Safe float conversion."""
    if _nan(val):
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal (European) odds."""
    if odds >= 0:
        return odds / 100.0 + 1.0
    return 100.0 / abs(odds) + 1.0


def _american_to_implied(odds: float) -> float:
    """Convert American odds to raw implied probability (includes vig)."""
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _kelly(p_model: float, retail_odds: float | None) -> float:
    """
    Fractional Kelly: f* = (b*p - q) / b  where b = decimal_odds - 1.

    Returns the raw Kelly fraction (clipped to [0, 1]).
    Falls back to a rough breakeven-delta if odds are unavailable.
    """
    if not _nan(retail_odds):
        b = _american_to_decimal(_f(retail_odds)) - 1.0
        if b <= 0:
            return 0.0
        p, q = p_model, 1.0 - p_model
        return float(max(0.0, (b * p - q) / b))
    # No odds available — use margin above -110 breakeven as proxy
    return float(max(0.0, p_model - 0.5238))


def _stake(tier: int, kelly_f: float) -> float:
    """
    Dollar stake: fraction of Kelly * bankroll, capped and rounded.

        Tier 1 (strong): SYNTHETIC_BANKROLL * f* * 0.25
        Tier 2 (medium): SYNTHETIC_BANKROLL * f* * 0.125

    Final value is rounded to the nearest whole dollar and capped at MAX_BET.
    """
    fraction = 0.25 if tier == 1 else 0.125
    raw = fraction * kelly_f * SYNTHETIC_BANKROLL
    capped = min(raw, MAX_BET_DOLLARS)
    return float(round(capped))


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

def _run_gates(
    p_model: float,
    p_true: float | None,
    p_retail: float | None,
    retail_odds: float | None,
) -> tuple[int | None, GateTrace]:
    """
    Run the Three-Part Lock sequence.

    Returns (tier, GateTrace).  tier is None when any gate fails.

    Gate 1 — Sanity
        Pinnacle preferred (tight 4% threshold).
        Falls back to retail with 8% threshold when Pinnacle is absent.
        Fails conservatively when neither is available.

    Gate 2 — Odds floor
        retail_odds must be >= ODDS_FLOOR (-225).
        Passes by default when odds are absent (no data = no veto).

    Gate 3 — Edge classification
        edge = p_model - p_retail_implied
        Tier 1 if edge >= EDGE_TIER1 (3%)
        Tier 2 if edge >= EDGE_TIER2 (1%)
        Fail if edge < EDGE_TIER2
    """
    trace = GateTrace()

    # --- Gate 1: Sanity ---
    if not _nan(p_true):
        ok = abs(p_model - _f(p_true)) <= SANITY_THRESHOLD
        trace.sanity_source = "pinnacle"
    elif not _nan(p_retail):
        ok = abs(p_model - _f(p_retail)) <= SANITY_THRESHOLD_RETAIL
        trace.sanity_source = "retail_fallback"
    else:
        trace.fail_reason = "no_price_reference"
        return None, trace

    trace.sanity_pass = ok
    if not ok:
        trace.fail_reason = (
            f"sanity_fail|delta={abs(p_model - _f(p_true or p_retail)):.3f}"
            f">{SANITY_THRESHOLD:.2f}"
        )
        return None, trace

    # --- Gate 2: Odds floor ---
    if not _nan(retail_odds):
        floor_ok = _f(retail_odds) >= ODDS_FLOOR
    else:
        floor_ok = True   # absent → no veto

    trace.odds_floor_pass = floor_ok
    if not floor_ok:
        trace.fail_reason = f"odds_floor_fail|{_f(retail_odds):+.0f}<{ODDS_FLOOR:.0f}"
        return None, trace

    # --- Gate 3: Edge ---
    if _nan(p_retail):
        trace.fail_reason = "no_retail_implied"
        return None, trace

    edge = p_model - _f(p_retail)
    trace.edge = round(edge, 4)

    if edge < EDGE_TIER2:
        trace.fail_reason = f"edge_below_tier2|{edge:+.3f}<{EDGE_TIER2}"
        return None, trace

    trace.edge_pass = True
    tier = 1 if edge >= EDGE_TIER1 else 2
    return tier, trace


# ---------------------------------------------------------------------------
# Public API — score_card()
# ---------------------------------------------------------------------------

def score_card(df: pd.DataFrame) -> list[LockResult]:
    """
    Run the Three-Part Lock on every row of ``df`` across three markets
    (RL home, ML home, O/U), and return a sorted list of LockResults.

    Sorting: Tier 1 before Tier 2; within each tier, by edge descending.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the columns described in this module's docstring.
        Extra columns are silently ignored.
        NaN/None in any required column causes that market to fail its gate
        gracefully without raising an exception.

    Returns
    -------
    list[LockResult]   May be empty — no locks fired.
    """
    locks: list[LockResult] = []

    for _, row in df.iterrows():
        game      = str(row.get("game", ""))
        home      = str(row.get("home_team", ""))
        away      = str(row.get("away_team", ""))
        book      = str(row.get("retail_book_used", "Book"))

        _meta = dict(
            game=game, home_team=home, away_team=away,
            home_sp=str(row.get("home_sp", "")),
            away_sp=str(row.get("away_sp", "")),
            home_sp_xwoba=_f(row.get("home_sp_xwoba")),
            away_sp_xwoba=_f(row.get("away_sp_xwoba")),
            home_sp_flag=str(row.get("home_sp_flag", "")),
            away_sp_flag=str(row.get("away_sp_flag", "")),
            game_time=str(row.get("game_time_et", "")),
            lineup_confirmed=bool(row.get("lineup_confirmed", False)),
            retail_book=book,
        )

        # ── RL gate ────────────────────────────────────────────────────────
        pm_rl   = _f(row.get("lock_p_model"))
        pt_rl   = row.get("lock_p_true")
        pr_rl   = row.get("lock_retail_implied")
        odds_rl = row.get("rl_odds")
        pin_rl  = row.get("pinnacle_rl_home_odds")

        if not _nan(pm_rl):
            tier_rl, trace_rl = _run_gates(pm_rl, pt_rl, pr_rl, odds_rl)
            if tier_rl is not None:
                kf = _kelly(pm_rl, odds_rl)
                locks.append(LockResult(
                    market="RL",
                    tier=tier_rl,
                    p_model=pm_rl,
                    p_true=None if _nan(pt_rl) else _f(pt_rl),
                    p_retail=_f(pr_rl),
                    edge=trace_rl.edge,
                    pinnacle_odds=None if _nan(pin_rl) else _f(pin_rl),
                    retail_odds=None if _nan(odds_rl) else _f(odds_rl),
                    stake=_stake(tier_rl, kf),
                    kelly_f=round(kf, 4),
                    trace=trace_rl,
                    **_meta,
                ))

        # ── ML gate ────────────────────────────────────────────────────────
        pm_ml   = _f(row.get("ml_lock_p_model"))
        pt_ml   = row.get("ml_lock_p_true")
        pr_ml   = row.get("ml_lock_retail_implied")
        odds_ml = row.get("vegas_ml_home")
        pin_ml  = row.get("pinnacle_ml_home")

        if not _nan(pm_ml):
            tier_ml, trace_ml = _run_gates(pm_ml, pt_ml, pr_ml, odds_ml)
            if tier_ml is not None:
                kf = _kelly(pm_ml, odds_ml)
                locks.append(LockResult(
                    market="ML",
                    tier=tier_ml,
                    p_model=pm_ml,
                    p_true=None if _nan(pt_ml) else _f(pt_ml),
                    p_retail=_f(pr_ml),
                    edge=trace_ml.edge,
                    pinnacle_odds=None if _nan(pin_ml) else _f(pin_ml),
                    retail_odds=None if _nan(odds_ml) else _f(odds_ml),
                    stake=_stake(tier_ml, kf),
                    kelly_f=round(kf, 4),
                    trace=trace_ml,
                    **_meta,
                ))

        # ── O/U gate ───────────────────────────────────────────────────────
        ou_dir  = str(row.get("ou_direction", "OVER")).strip().upper()
        ou_line = _f(row.get("ou_posted_line"))
        pm_ou   = _f(row.get("ou_p_model"))     # always MC over prob
        pt_ou   = row.get("ou_p_true")           # Pinnacle over implied
        pr_ou   = row.get("ou_p_retail")         # retail over implied

        # Flip model/retail implied when direction is UNDER
        if ou_dir == "UNDER":
            pm_ou_gate = 1.0 - pm_ou if not _nan(pm_ou) else float("nan")
            pt_ou_gate = (1.0 - _f(pt_ou)) if not _nan(pt_ou) else None
            pr_ou_gate = (1.0 - _f(pr_ou)) if not _nan(pr_ou) else None
        else:
            pm_ou_gate = pm_ou
            pt_ou_gate = pt_ou
            pr_ou_gate = pr_ou

        # O/U has no retail American odds column — odds_floor gate passes by default
        if not _nan(pm_ou_gate):
            tier_ou, trace_ou = _run_gates(pm_ou_gate, pt_ou_gate, pr_ou_gate, None)
            if tier_ou is not None:
                kf = _kelly(pm_ou_gate, None)   # no American odds for O/U
                market_key = "OVR" if ou_dir == "OVER" else "UND"
                locks.append(LockResult(
                    market=market_key,
                    tier=tier_ou,
                    p_model=round(pm_ou_gate, 4),
                    p_true=None if _nan(pt_ou_gate) else _f(pt_ou_gate),
                    p_retail=_f(pr_ou_gate),
                    edge=trace_ou.edge,
                    pinnacle_odds=None,
                    retail_odds=None,
                    stake=_stake(tier_ou, kf),
                    kelly_f=round(kf, 4),
                    ou_line=None if _nan(ou_line) else ou_line,
                    trace=trace_ou,
                    **_meta,
                ))

    # Sort: Tier 1 first, then within tier by edge descending
    locks.sort(key=lambda r: (r.tier, -r.edge))
    return locks


# ---------------------------------------------------------------------------
# Terminal payload — print_payload()
# ---------------------------------------------------------------------------

def print_payload(
    locks: list[LockResult],
    date_str: str = "",
    all_rows: pd.DataFrame | None = None,
) -> None:
    """
    Print the terminal ledger to stdout.

    Locked picks are grouped by tier.  If ``all_rows`` is supplied, games
    with no lock are printed in a compact diagnostic section at the bottom
    so you can see every game's gate failure reason on one screen.

    Lock line format (exact):
        [Away] @ [Home] | [Market] | PIN: [Odds] | [Book]: [Odds] | Edge: [X]% | Bet: $[N]
    """
    import datetime as _dt

    # Header
    try:
        day_label = _dt.datetime.strptime(date_str, "%Y-%m-%d").strftime("%B %-d, %Y")
    except Exception:
        day_label = date_str or str(_dt.date.today())

    header = f"  --- THE WIZARD: MLB EDITION --- {day_label} ---"
    print()
    print(_bold("=" * len(header)))
    print(_bold(header))
    print(_bold("=" * len(header)))

    tier1 = [r for r in locks if r.tier == 1]
    tier2 = [r for r in locks if r.tier == 2]

    if not locks:
        print()
        print("  No picks passed the Three-Part Lock today.")
        print("  (Sanity <=4% | Odds >= -225 | Edge >= 1.0%)")
    else:
        if tier1:
            _print_tier_block(tier1)
        if tier2:
            _print_tier_block(tier2)

    # Compact all-games diagnostic (optional)
    if all_rows is not None and not all_rows.empty:
        locked_games = {r.game for r in locks}
        no_lock = all_rows[~all_rows["game"].astype(str).isin(locked_games)]
        if not no_lock.empty:
            print(f"\n  {_bold('--- ALL GAMES (no lock signal) ---')}")
            for _, row in no_lock.iterrows():
                _print_no_lock_row(row)

    # Footer
    print()
    print(f"  Lock gates: |P_model - P_pin| <= {SANITY_THRESHOLD:.0%}  |  "
          f"odds >= {ODDS_FLOOR:.0f}  |  edge >= {EDGE_TIER2:.0%}")
    print(f"  Kelly: T1 = quarter-Kelly  |  T2 = eighth-Kelly  |  cap = ${MAX_BET_DOLLARS:.0f}")
    print()


def _print_tier_block(group: list[LockResult]) -> None:
    tier  = group[0].tier
    label = TIER_LABELS.get(tier, f"TIER {tier}")
    n     = len(group)
    color = _green if tier == 1 else _yellow

    print()
    print(color(_bold(f"  --- {label} ({n} pick{'s' if n > 1 else ''}) ---")))
    print()

    for r in group:
        _print_lock_line(r)


def _print_lock_line(r: LockResult) -> None:
    """
    Print one locked pick in the specified format:

        [Away] @ [Home] | [Market/Pick] | PIN: [Odds] | [Book]: [Odds] | Edge: [X]% | Bet: $[N]

    Additional detail lines follow for pitcher context and gate trace.
    """
    color = _green if r.tier == 1 else _yellow

    # Build the pipe-delimited summary line
    away_at_home = f"{r.away_team} @ {r.home_team}"
    market_str   = _market_label(r)
    pin_str      = _odds_str(r.pinnacle_odds, prefix="PIN")
    book_str     = _odds_str(r.retail_odds,   prefix=r.retail_book or "Book")
    edge_str     = f"Edge: {r.edge * 100:+.2f}%"
    bet_str      = f"Bet: ${int(r.stake)}" if r.stake > 0 else "Bet: $0 (Kelly vetoed)"
    time_str     = f"  [{r.game_time}]" if r.game_time else ""
    proj_str     = "" if r.lineup_confirmed else " [PROJ]"

    pipe = " | "
    line = pipe.join([
        f"  {away_at_home}{proj_str}",
        market_str,
        pin_str,
        book_str,
        edge_str,
        color(_bold(bet_str)),
    ])
    print(line + time_str)

    # Pitcher quality context
    sp_h = r.home_sp or "TBD"
    sp_a = r.away_sp or "TBD"
    xwoba_h = f"  xwOBA={r.home_sp_xwoba:.3f}" if r.home_sp_xwoba and not np.isnan(r.home_sp_xwoba) else ""
    xwoba_a = f"  xwOBA={r.away_sp_xwoba:.3f}" if r.away_sp_xwoba and not np.isnan(r.away_sp_xwoba) else ""
    flag_h   = f" [{r.home_sp_flag}]" if r.home_sp_flag not in ("", "NORMAL", "UNKNOWN") else ""
    flag_a   = f" [{r.away_sp_flag}]" if r.away_sp_flag not in ("", "NORMAL", "UNKNOWN") else ""
    print(f"    SP: {sp_h}{xwoba_h}{flag_h}  vs  {sp_a}{xwoba_a}{flag_a}")

    # Gate trace
    p_true_str = f"{r.p_true:.3f}" if r.p_true is not None else "N/A"
    sanity_src = f" [{r.trace.sanity_source}]" if r.trace.sanity_source else ""
    print(f"    P_model={r.p_model:.3f}  P_pin={p_true_str}{sanity_src}  "
          f"P_retail={r.p_retail:.3f}  kelly_f={r.kelly_f:.4f}")
    print()


def _print_no_lock_row(row: pd.Series) -> None:
    """Compact one-line diagnostic for a game that did not lock."""
    game   = str(row.get("game", "?"))
    bl_rl  = _f(row.get("blended_rl"))
    ml_h   = _f(row.get("vegas_ml_home"))
    rl_o   = _f(row.get("rl_odds"))

    ml_str = f"ML {int(ml_h):+d}" if not _nan(ml_h) else "no ML"
    rl_str = f"RL {int(rl_o):+d}" if not _nan(rl_o) else ""
    bl_str = f"blend={bl_rl:.3f}" if not _nan(bl_rl) else "[TBD starters]"

    # Surface first gate failure
    fail = _first_fail(row)
    print(f"  {game:<26}  {bl_str}  {ml_str}  {rl_str}  (no lock: {fail})")


def _first_fail(row: pd.Series) -> str:
    """Return a short string naming the first gate that failed."""
    if _nan(row.get("lock_p_true")) and _nan(row.get("ml_lock_p_true")):
        return "no P_true"
    # Check RL sanity
    sp = row.get("lock_sanity_pass")
    if sp is not None and not bool(sp):
        delta = abs(_f(row.get("lock_p_model")) - _f(row.get("lock_p_true")))
        return f"sanity({delta:.3f}>{SANITY_THRESHOLD:.2f})"
    # Check odds floor
    fp = row.get("lock_odds_floor_pass")
    if fp is not None and not bool(fp):
        return f"odds_floor({_f(row.get('rl_odds')):+.0f})"
    # Edge
    ep = row.get("lock_edge_pass")
    if ep is not None and not bool(ep):
        edge = _f(row.get("lock_edge"))
        return f"edge({edge:+.1%})" if not _nan(edge) else "edge"
    return "edge"


def _market_label(r: LockResult) -> str:
    """Return the market string for the pipe-delimited line."""
    if r.market == "RL":
        # Lock always evaluates home direction; stake direction is implied by p_model
        direction = "HOME -1.5" if r.p_model >= 0.50 else "AWAY +1.5"
        return f"RL {direction}"
    if r.market == "ML":
        direction = "HOME WIN" if r.p_model >= 0.50 else "AWAY WIN"
        return f"ML {direction}"
    if r.market == "OVR":
        return f"OVER {r.ou_line}" if r.ou_line else "OVER"
    if r.market == "UND":
        return f"UNDER {r.ou_line}" if r.ou_line else "UNDER"
    return r.market


def _odds_str(american: float | None, prefix: str = "") -> str:
    """Format American odds as 'PREFIX: +105' or 'PREFIX: N/A'."""
    if american is None or np.isnan(american):
        return f"{prefix}: N/A" if prefix else "N/A"
    sign = "+" if american >= 0 else ""
    return f"{prefix}: {sign}{int(american)}" if prefix else f"{sign}{int(american)}"


# ---------------------------------------------------------------------------
# Diagnostic helpers (importable)
# ---------------------------------------------------------------------------

def gate_trace_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all gates for all rows and return a diagnostic DataFrame.

    Columns: game, market, tier, edge, sanity_pass, sanity_source,
             odds_floor_pass, edge_pass, fail_reason.

    Useful for understanding why locks didn't fire on a given slate.
    """
    rows = []
    for _, row in df.iterrows():
        game = str(row.get("game", ""))
        for mkt, pm_col, pt_col, pr_col, odds_col in [
            ("RL",  "lock_p_model",    "lock_p_true",    "lock_retail_implied",    "rl_odds"),
            ("ML",  "ml_lock_p_model", "ml_lock_p_true", "ml_lock_retail_implied", "vegas_ml_home"),
            ("OVR", "ou_p_model",      "ou_p_true",      "ou_p_retail",            None),
        ]:
            pm = _f(row.get(pm_col))
            if _nan(pm):
                continue
            tier, trace = _run_gates(pm, row.get(pt_col), row.get(pr_col),
                                     row.get(odds_col) if odds_col else None)
            rows.append({
                "game":            game,
                "market":          mkt,
                "tier":            tier,
                "edge":            round(trace.edge, 4) if not np.isnan(trace.edge) else None,
                "sanity_pass":     trace.sanity_pass,
                "sanity_source":   trace.sanity_source,
                "odds_floor_pass": trace.odds_floor_pass,
                "edge_pass":       trace.edge_pass,
                "fail_reason":     trace.fail_reason or "PASSED",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MLB Three-Part Lock + terminal payload on a daily card CSV"
    )
    parser.add_argument(
        "--csv", "-c",
        default="daily_card.csv",
        metavar="PATH",
        help="Path to daily_card.csv (default: daily_card.csv in cwd)",
    )
    parser.add_argument(
        "--date", "-d",
        default="",
        metavar="YYYY-MM-DD",
        help="Override date label in header",
    )
    parser.add_argument(
        "--trace", "-t",
        action="store_true",
        help="Also print full gate diagnostic table for all games",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Suppress the 'no lock' diagnostic section at the bottom",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path, low_memory=False)

    # Infer date from filename if not supplied
    date_str = args.date
    if not date_str:
        import re
        m = re.search(r"(\d{4}-\d{2}-\d{2})", csv_path.stem)
        date_str = m.group(1) if m else ""

    locks = score_card(df)
    all_df = None if args.no_context else df
    print_payload(locks, date_str=date_str, all_rows=all_df)

    if args.trace:
        print(_bold("\n  --- GATE DIAGNOSTIC TABLE ---\n"))
        trace_df = gate_trace_all(df)
        with pd.option_context("display.max_rows", 200, "display.width", 120):
            print(trace_df.to_string(index=False))
        print()

    sys.exit(0 if locks else 1)


if __name__ == "__main__":
    main()
