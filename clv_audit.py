"""
clv_audit.py
============
Closing Line Value (CLV) and P&L audit for the Wizard pipeline.

Scans every daily_card_{date}.csv and extracts each locked pick
(Three-Part Lock: RL / ML / O-U).  For each pick it records:

  p_model      -- model probability at pick time (11am ET)
  p_true       -- Pinnacle de-vigged implied prob at pick time  ← CLV anchor
  edge_pct     -- p_model - p_true, expressed as a percentage
  market_odds  -- American odds at which the bet is priced
  stake        -- Kelly-sized dollar stake
  outcome      -- WIN / LOSS / PUSH / PENDING
  profit_loss  -- dollars won or lost on the bet
  units_pl     -- profit_loss expressed in "1 unit = 1 dollar" terms

CLV note
--------
True CLV requires the *closing* Pinnacle line (5 min before first pitch).
The pipeline pulls odds at 11am ET, so p_true here is an early-afternoon
Pinnacle snapshot, not a true closing line.  This is the best available
proxy until a late-pull job is added.

Positive avg edge_pct = model consistently found spots where Pinnacle agreed
(sanity pass) but retail was softer.  Even without true closing lines, a
persistently positive mean edge_pct on *resolved* picks is meaningful signal.

Outputs
-------
  clv_audit.csv         one row per locked pick, updated in-place
  (stdout)              summary table by sport / bet_type

Usage
-----
  python clv_audit.py               # update + print summary
  python clv_audit.py --summary     # print existing CSV summary, no scan
  python clv_audit.py --date 2026-04-15   # refresh a single date only
  python clv_audit.py --reset       # wipe and rebuild from scratch
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Adjust these paths if your repo layout differs
MLB_DIR  = Path(__file__).parent
NBA_DIR  = MLB_DIR.parent / "nba_model_pipeline_dan"

AUDIT_PATH = MLB_DIR / "clv_audit.csv"

SYNTHETIC_BANKROLL = 2_000.0
MAX_BET            = 50.0

# Tier labels (integers in the CSV; displayed as strings in the summary)
TIER_LABELS = {1: "STRONG (T1)", 2: "MEDIUM (T2)"}

# ANSI colours (disabled on non-tty / Windows without FORCE_COLOR)
import os as _os
_USE_COLOR = sys.stdout.isatty() or _os.environ.get("FORCE_COLOR") == "1"
_GRN  = "\033[92m" if _USE_COLOR else ""
_YEL  = "\033[93m" if _USE_COLOR else ""
_RED  = "\033[91m" if _USE_COLOR else ""
_CYN  = "\033[96m" if _USE_COLOR else ""
_BOLD = "\033[1m"  if _USE_COLOR else ""
_RST  = "\033[0m"  if _USE_COLOR else ""

# ---------------------------------------------------------------------------
# Audit CSV schema
# ---------------------------------------------------------------------------
AUDIT_COLS = [
    "sport",           # MLB | NBA
    "date",            # YYYY-MM-DD
    "game",            # "TB @ CWS"
    "bet_type",        # RL_HOME | ML_HOME | OVER | UNDER
    "tier",            # 1 (strong) | 2 (medium)
    "p_model",         # model probability at pick time
    "p_true",          # Pinnacle implied at pick time (CLV anchor)
    "edge_pct",        # (p_model - p_true) * 100
    "market_odds",     # American odds (retail book)
    "stake",           # Kelly-sized dollar stake
    "ou_line",         # posted O/U line (O/U bets only, else NaN)
    "outcome",         # WIN | LOSS | PUSH | PENDING
    "profit_loss",     # dollars
    "units_pl",        # profit_loss / 1 (stake in dollars; 1 unit = $1)
    "notes",           # sanity_source, missing data flags, etc.
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _float(val) -> float | None:
    """Return float or None for NaN/missing."""
    try:
        v = float(val)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _american_profit(stake: float, american_odds: float) -> float:
    """Return dollar profit on a winning bet."""
    if american_odds >= 0:
        return stake * american_odds / 100.0
    else:
        return stake * 100.0 / abs(american_odds)


def _payout_from_implied(stake: float, implied_prob: float) -> float:
    """Estimate profit from implied probability when American odds unavailable."""
    if implied_prob <= 0 or implied_prob >= 1:
        return stake * (100 / 110)  # assume -110 juice
    decimal = 1.0 / implied_prob
    return stake * (decimal - 1.0)


# ---------------------------------------------------------------------------
# Outcome resolver -- MLB
# ---------------------------------------------------------------------------

def _build_score_lookup(mlb_dir: Path) -> dict[tuple[str, str], dict]:
    """
    Build a {(date_str, game_str): {home_score, away_score, home_covers_rl,
    actual_total}} lookup from all available sources.

    Sources (in priority order):
      1. backtest_2026_results.csv  -- curated, has home_covers_rl
      2. daily_card_{date}.csv files that have home_score populated
         (future: could add a results feed here)
    """
    lookup: dict[tuple[str, str], dict] = {}

    bt_path = mlb_dir / "backtest_2026_results.csv"
    if bt_path.exists():
        bt = pd.read_csv(bt_path)
        bt["date"] = bt["date"].astype(str)
        for _, row in bt.iterrows():
            hs = _float(row.get("home_score"))
            as_ = _float(row.get("away_score"))
            if hs is None or as_ is None:
                continue
            key = (row["date"], row["game"])
            lookup[key] = {
                "home_score":     hs,
                "away_score":     as_,
                "home_covers_rl": int(row.get("home_covers_rl", -1)),
                "actual_total":   hs + as_,
            }

    return lookup


def _resolve_outcome(bet_type: str, ou_line: float | None,
                     score_row: dict | None) -> tuple[str, float]:
    """
    Return (outcome_str, profit_loss_multiplier) given score data.

    profit_loss_multiplier:
        +1.0 → WIN (caller multiplies by stake payout)
        -1.0 → LOSS (caller returns -stake)
         0.0 → PUSH
    """
    if score_row is None:
        return "PENDING", 0.0

    hcrl  = score_row.get("home_covers_rl", -1)
    hm    = score_row.get("home_score", 0) - score_row.get("away_score", 0)
    total = score_row.get("actual_total", 0.0)

    if bet_type == "RL_HOME":
        if hcrl == 1:  return "WIN",  +1.0
        if hcrl == 0:  return "LOSS", -1.0
        return "PENDING", 0.0

    if bet_type == "RL_AWAY":
        # Away covers RL = home does NOT cover (margin <= 1)
        if hcrl == 0:  return "WIN",  +1.0
        if hcrl == 1:  return "LOSS", -1.0
        return "PENDING", 0.0

    if bet_type == "ML_HOME":
        if hm > 0:  return "WIN",  +1.0
        if hm < 0:  return "LOSS", -1.0
        return "PUSH", 0.0

    if bet_type == "ML_AWAY":
        if hm < 0:  return "WIN",  +1.0
        if hm > 0:  return "LOSS", -1.0
        return "PUSH", 0.0

    if bet_type in ("OVER", "UNDER") and ou_line is not None:
        if bet_type == "OVER":
            if total > ou_line:  return "WIN",  +1.0
            if total < ou_line:  return "LOSS", -1.0
            return "PUSH", 0.0
        else:
            if total < ou_line:  return "WIN",  +1.0
            if total > ou_line:  return "LOSS", -1.0
            return "PUSH", 0.0

    return "PENDING", 0.0


# ---------------------------------------------------------------------------
# MLB pick extractor
# ---------------------------------------------------------------------------

def _extract_mlb_picks(card: pd.DataFrame, date_str: str,
                       score_lookup: dict) -> list[dict]:
    """
    Extract locked picks from one MLB daily_card_*.csv dataframe.

    Produces one dict per lock signal (RL / ML / OU) per game.
    Games with no lock signals are skipped.
    """
    picks = []

    for _, row in card.iterrows():
        game = str(row.get("game", ""))
        if not game:
            continue

        score_row = score_lookup.get((date_str, game))

        # -- RL lock ----------------------------------------------------------
        rl_tier = _float(row.get("lock_tier"))
        if rl_tier is not None:
            p_model  = _float(row.get("lock_p_model"))
            p_true   = _float(row.get("lock_p_true"))
            stake    = _float(row.get("lock_dollars")) or 0.0
            edge_raw = _float(row.get("lock_edge"))
            odds     = _float(row.get("rl_odds"))
            retail_i = _float(row.get("lock_retail_implied"))

            # Determine direction -- lock always evaluates home RL
            # p_model > 0.5 → model likes home -1.5
            bet_type = "RL_HOME" if (p_model or 0.5) >= 0.5 else "RL_AWAY"

            edge_pct = (edge_raw * 100.0) if edge_raw is not None else (
                ((p_model or 0) - (p_true or 0)) * 100.0
            )

            outcome, mult = _resolve_outcome(bet_type, None, score_row)
            if outcome == "WIN":
                if odds is not None:
                    pl = _american_profit(stake, odds)
                elif retail_i is not None:
                    pl = _payout_from_implied(stake, retail_i)
                else:
                    pl = _american_profit(stake, -110)
            elif outcome == "LOSS":
                pl = -stake
            else:
                pl = 0.0

            picks.append({
                "sport":       "MLB",
                "date":        date_str,
                "game":        game,
                "bet_type":    bet_type,
                "tier":        int(rl_tier),
                "p_model":     round(p_model, 4) if p_model else None,
                "p_true":      round(p_true, 4) if p_true else None,
                "edge_pct":    round(edge_pct, 2),
                "market_odds": odds,
                "stake":       stake,
                "ou_line":     None,
                "outcome":     outcome,
                "profit_loss": round(pl, 2) if outcome != "PENDING" else None,
                "units_pl":    round(pl, 2) if outcome != "PENDING" else None,
                "notes":       f"sanity_source:{row.get('sanity_source','')}",
            })

        # -- ML lock ----------------------------------------------------------
        ml_tier = _float(row.get("ml_lock_tier"))
        if ml_tier is not None:
            p_model  = _float(row.get("ml_lock_p_model"))
            p_true   = _float(row.get("ml_lock_p_true"))
            stake    = _float(row.get("ml_lock_dollars")) or 0.0
            edge_raw = _float(row.get("ml_lock_edge"))
            odds     = _float(row.get("vegas_ml_home"))
            retail_i = _float(row.get("ml_lock_retail_implied"))

            bet_type = "ML_HOME" if (p_model or 0.5) >= 0.5 else "ML_AWAY"
            edge_pct = (edge_raw * 100.0) if edge_raw is not None else (
                ((p_model or 0) - (p_true or 0)) * 100.0
            )

            outcome, _ = _resolve_outcome(bet_type, None, score_row)
            if outcome == "WIN":
                if odds is not None:
                    pl = _american_profit(stake, odds)
                elif retail_i is not None:
                    pl = _payout_from_implied(stake, retail_i)
                else:
                    pl = _american_profit(stake, -110)
            elif outcome == "LOSS":
                pl = -stake
            else:
                pl = 0.0

            picks.append({
                "sport":       "MLB",
                "date":        date_str,
                "game":        game,
                "bet_type":    bet_type,
                "tier":        int(ml_tier),
                "p_model":     round(p_model, 4) if p_model else None,
                "p_true":      round(p_true, 4) if p_true else None,
                "edge_pct":    round(edge_pct, 2),
                "market_odds": odds,
                "stake":       stake,
                "ou_line":     None,
                "outcome":     outcome,
                "profit_loss": round(pl, 2) if outcome != "PENDING" else None,
                "units_pl":    round(pl, 2) if outcome != "PENDING" else None,
                "notes":       "",
            })

        # -- O/U lock ---------------------------------------------------------
        ou_tier = _float(row.get("ou_lock_tier"))
        if ou_tier is not None:
            direction = str(row.get("ou_direction", "OVER")).strip().upper()
            ou_line   = _float(row.get("ou_posted_line"))
            p_model   = _float(row.get("ou_p_model"))
            p_true    = _float(row.get("ou_p_true"))
            stake     = _float(row.get("ou_lock_dollars")) or 0.0
            edge_raw  = _float(row.get("ou_lock_edge"))
            retail_i  = _float(row.get("ou_p_retail"))

            edge_pct = (edge_raw * 100.0) if edge_raw is not None else (
                ((p_model or 0) - (p_true or 0)) * 100.0
            )
            bet_type = "OVER" if direction == "OVER" else "UNDER"

            outcome, _ = _resolve_outcome(bet_type, ou_line, score_row)
            if outcome == "WIN":
                pl = _payout_from_implied(stake, retail_i or 0.524)
            elif outcome == "LOSS":
                pl = -stake
            else:
                pl = 0.0

            picks.append({
                "sport":       "MLB",
                "date":        date_str,
                "game":        game,
                "bet_type":    bet_type,
                "tier":        int(ou_tier),
                "p_model":     round(p_model, 4) if p_model else None,
                "p_true":      round(p_true, 4) if p_true else None,
                "edge_pct":    round(edge_pct, 2),
                "market_odds": None,
                "stake":       stake,
                "ou_line":     ou_line,
                "outcome":     outcome,
                "profit_loss": round(pl, 2) if outcome != "PENDING" else None,
                "units_pl":    round(pl, 2) if outcome != "PENDING" else None,
                "notes":       f"line:{ou_line} dir:{direction}",
            })

    return picks


# ---------------------------------------------------------------------------
# NBA pick extractor  (stub -- wired in once NBA daily cards exist)
# ---------------------------------------------------------------------------

def _extract_nba_picks(nba_dir: Path, score_lookup: dict) -> list[dict]:
    """
    Extract locked NBA picks from wizard_agents data/model_scores.csv.

    Returns [] if the NBA pipeline has no data yet.
    Designed to mirror the MLB extractor so the summary table works
    identically for both sports.
    """
    scores_path = nba_dir / "data" / "model_scores.csv"
    if not scores_path.exists():
        return []

    try:
        df = pd.read_csv(scores_path)
    except Exception:
        return []

    picks = []
    # TODO: map NBA model_scores columns to the same pick schema once
    # the NBA Three-Part Lock is implemented.  Expected columns needed:
    #   game_date, matchup, bet_type, tier, p_model, p_true,
    #   edge_pct, market_odds, stake_dollars
    # For now, log any rows with a non-null tier as PENDING placeholders.
    for _, row in df.iterrows():
        tier = _float(row.get("tier") or row.get("lock_tier"))
        if tier is None:
            continue
        picks.append({
            "sport":       "NBA",
            "date":        str(row.get("game_date", "")),
            "game":        str(row.get("matchup", row.get("game", ""))),
            "bet_type":    str(row.get("bet_type", "ML")),
            "tier":        int(tier),
            "p_model":     _float(row.get("p_model")),
            "p_true":      _float(row.get("p_true")),
            "edge_pct":    _float(row.get("edge_pct")) or 0.0,
            "market_odds": _float(row.get("market_odds")),
            "stake":       _float(row.get("stake_dollars")) or 0.0,
            "ou_line":     _float(row.get("total_line")),
            "outcome":     "PENDING",
            "profit_loss": None,
            "units_pl":    None,
            "notes":       "nba",
        })
    return picks


# ---------------------------------------------------------------------------
# Core scan
# ---------------------------------------------------------------------------

def scan_mlb(mlb_dir: Path, date_filter: str | None = None) -> list[dict]:
    """Scan all MLB daily card CSVs; return list of pick dicts."""
    score_lookup = _build_score_lookup(mlb_dir)
    picks = []

    pattern = f"daily_card_{date_filter}.csv" if date_filter else "daily_card_????-??-??.csv"
    # Check both root dir and daily_cards/ subfolder
    search_dirs = [mlb_dir, mlb_dir / "daily_cards"]
    card_paths = sorted(p for d in search_dirs for p in d.glob(pattern) if d.exists())
    for card_path in card_paths:
        date_str = card_path.stem.replace("daily_card_", "")
        try:
            card = pd.read_csv(card_path, low_memory=False)
        except Exception as exc:
            print(f"  [WARN] Could not read {card_path.name}: {exc}")
            continue
        picks.extend(_extract_mlb_picks(card, date_str, score_lookup))

    return picks


def scan_nba(nba_dir: Path) -> list[dict]:
    if not nba_dir.exists():
        return []
    return _extract_nba_picks(nba_dir, {})


# ---------------------------------------------------------------------------
# Update audit CSV  (idempotent -- deduplicates on sport+date+game+bet_type)
# ---------------------------------------------------------------------------

def update_audit(new_picks: list[dict], audit_path: Path,
                 reset: bool = False) -> pd.DataFrame:
    """
    Merge new_picks into audit_path, deduplicating and updating outcomes.

    Dedup key: (sport, date, game, bet_type)
    On conflict, the new row wins so outcomes get refreshed automatically.
    """
    new_df = pd.DataFrame(new_picks, columns=AUDIT_COLS) if new_picks \
             else pd.DataFrame(columns=AUDIT_COLS)

    if not reset and audit_path.exists():
        existing = pd.read_csv(audit_path, low_memory=False)
        # Drop rows in existing that are now being refreshed
        key_cols = ["sport", "date", "game", "bet_type"]
        if not new_df.empty:
            new_keys = set(zip(*[new_df[c].astype(str) for c in key_cols]))
            mask = ~existing.apply(
                lambda r: tuple(str(r[c]) for c in key_cols) in new_keys,
                axis=1
            )
            existing = existing[mask]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.sort_values(["date", "sport", "game", "bet_type"],
                                    ignore_index=True)
    combined.to_csv(audit_path, index=False)
    return combined


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    """Print a formatted CLV / P&L summary table to stdout."""
    print()
    print(f"{_BOLD}{'='*70}{_RST}")
    print(f"{_BOLD}  WIZARD CLV AUDIT -- {datetime.date.today()}{_RST}")
    print(f"{_BOLD}{'='*70}{_RST}")

    if df.empty:
        print(f"\n  {_YEL}No locked picks recorded yet.{_RST}")
        print(f"  The Three-Part Lock (sanity + odds floor + >=1% edge) has not")
        print(f"  fired on any game since tracking began.  When it does, picks")
        print(f"  will appear here automatically after the next run.\n")
        print(f"{_BOLD}{'='*70}{_RST}\n")
        return

    resolved = df[df["outcome"].isin(["WIN", "LOSS", "PUSH"])]
    pending  = df[df["outcome"] == "PENDING"]

    def _section(label: str, sub: pd.DataFrame, sport_tag: str = "") -> None:
        if sub.empty:
            return
        n        = len(sub)
        wins     = (sub["outcome"] == "WIN").sum()
        losses   = (sub["outcome"] == "LOSS").sum()
        pushes   = (sub["outcome"] == "PUSH").sum()
        wr       = wins / max(wins + losses, 1)
        pl       = sub["profit_loss"].fillna(0).sum()
        staked   = sub["stake"].fillna(0).sum()
        roi      = pl / max(staked, 1) * 100
        avg_edge = sub["edge_pct"].mean()
        t1       = (sub["tier"] == 1).sum()
        t2       = (sub["tier"] == 2).sum()

        pl_color = _GRN if pl >= 0 else _RED
        wr_color = _GRN if wr >= 0.53 else (_YEL if wr >= 0.50 else _RED)

        print(f"\n{_BOLD}  {label}{_RST}")
        print(f"  {'-'*62}")
        n_pend = len(pending[pending["sport"] == sport_tag]) if sport_tag else len(pending)
        print(f"  {'Picks':18}  {n:>4}  "
              f"(T1-strong:{t1}  T2-medium:{t2}  pending:{n_pend})")
        print(f"  {'Record':18}  {wins}W-{losses}L-{pushes}P  "
              f"win%: {wr_color}{wr:.1%}{_RST}")
        print(f"  {'P&L (dollars)':18}  {pl_color}{pl:+.2f}{_RST}")
        print(f"  {'Staked':18}  ${staked:.2f}")
        print(f"  {'ROI':18}  {pl_color}{roi:+.1f}%{_RST}")
        print(f"  {'Avg edge at pick':18}  {avg_edge:+.2f}%  "
              f"(CLV proxy -- 11am Pinnacle, not true close)")

    # By sport
    for sport in sorted(df["sport"].unique()):
        sport_res = resolved[resolved["sport"] == sport]
        _section(f"{sport} -- Resolved picks", sport_res, sport)

    # By bet type (all sports combined)
    if len(resolved) > 0:
        print(f"\n{_BOLD}  Resolved -- by bet type{_RST}")
        print(f"  {'-'*62}")
        fmt = f"  {'Type':<10}  {'N':>4}  {'W':>4}  {'L':>4}  {'WR%':>6}  {'P&L':>8}  {'Edge%':>7}"
        print(fmt)
        print(f"  {'-'*62}")
        for bt in sorted(resolved["bet_type"].unique()):
            sub = resolved[resolved["bet_type"] == bt]
            w   = (sub["outcome"] == "WIN").sum()
            l   = (sub["outcome"] == "LOSS").sum()
            wr  = w / max(w + l, 1)
            pl  = sub["profit_loss"].fillna(0).sum()
            eg  = sub["edge_pct"].mean()
            pl_c = _GRN if pl >= 0 else _RED
            print(f"  {bt:<10}  {len(sub):>4}  {w:>4}  {l:>4}  "
                  f"{wr:>6.1%}  {pl_c}{pl:>+8.2f}{_RST}  {eg:>+6.2f}%")

    # Pending summary
    if len(pending) > 0:
        kelly_vetoed = (pending["stake"].fillna(0) == 0).sum()
        actionable   = len(pending) - kelly_vetoed
        print(f"\n{_BOLD}  Pending picks ({len(pending)} total | "
              f"{actionable} actionable | {kelly_vetoed} Kelly-vetoed $0){_RST}")
        print(f"  {'-'*62}")
        for _, r in pending.sort_values("date").iterrows():
            tier_str  = TIER_LABELS.get(int(r["tier"]), f"T{r['tier']}")
            edge_str  = f"{r['edge_pct']:+.2f}%" if pd.notna(r["edge_pct"]) else "n/a"
            stake_val = float(r["stake"]) if pd.notna(r["stake"]) else 0.0
            stake_str = f"${stake_val:.0f}"
            # Kelly veto: gates all passed, but odds too juiced vs edge to justify bet
            veto_note = "  [Kelly $0 -- price too juiced relative to edge]" \
                        if stake_val == 0.0 else ""
            print(f"  {r['date']}  {r['sport']}  {r['game']:<22}  "
                  f"{r['bet_type']:<10}  {tier_str:<14}  edge:{edge_str}  "
                  f"{stake_str}{veto_note}")

    print()
    print(f"{_BOLD}  Audit file: {AUDIT_PATH}{_RST}")
    print(f"{_BOLD}{'='*70}{_RST}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CLV + P&L audit for MLB and NBA Wizard picks"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print existing audit summary without scanning for new picks"
    )
    parser.add_argument(
        "--date", default=None, metavar="YYYY-MM-DD",
        help="Refresh a single date only (leave other dates unchanged)"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Wipe and rebuild clv_audit.csv from scratch"
    )
    parser.add_argument(
        "--no-nba", action="store_true",
        help="Skip NBA scan (faster if NBA pipeline not set up)"
    )
    args = parser.parse_args()

    # --summary: just read and print
    if args.summary:
        if AUDIT_PATH.exists():
            df = pd.read_csv(AUDIT_PATH, low_memory=False)
            print_summary(df)
        else:
            print(f"  No audit file found at {AUDIT_PATH}")
            print("  Run without --summary to generate it.")
        return

    # --reset
    if args.reset and AUDIT_PATH.exists():
        AUDIT_PATH.unlink()
        print(f"  Wiped {AUDIT_PATH}")

    # Scan picks
    print(f"Scanning MLB picks in {MLB_DIR} ...")
    mlb_picks = scan_mlb(MLB_DIR, date_filter=args.date)
    print(f"  {len(mlb_picks)} MLB locked pick(s) found")

    nba_picks = []
    if not args.no_nba:
        if NBA_DIR.exists():
            print(f"Scanning NBA picks in {NBA_DIR} ...")
            nba_picks = scan_nba(NBA_DIR)
            print(f"  {len(nba_picks)} NBA locked pick(s) found")
        else:
            print(f"  NBA pipeline not found at {NBA_DIR} -- skipping")

    all_picks = mlb_picks + nba_picks

    # Update audit file
    df = update_audit(all_picks, AUDIT_PATH, reset=False)
    print(f"  clv_audit.csv updated -- {len(df)} total rows")

    # Print summary
    print_summary(df)


if __name__ == "__main__":
    main()
