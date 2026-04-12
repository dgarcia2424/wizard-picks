"""
run_today.py
============
Fully automated daily prediction card.

Pulls today's starters, Vegas lines, and weather from files already
on disk (refreshed by the data pipeline), runs Monte Carlo + XGBoost
for every game, and prints a ranked bet card.

Usage:
  python run_today.py               # today's card
  python run_today.py --date 2026-04-12
  python run_today.py --min-edge 0.56   # only print games above threshold
  python run_today.py --csv             # also write daily_card.csv
"""

import argparse
import datetime
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Load .env file so GMAIL_APP_PASSWORD etc. are available
def _load_dotenv():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

_load_dotenv()

DATA_DIR = Path("./data/statcast")

# Bet signal thresholds
# Historical home -1.5 cover rate: ~35.7%.  Thresholds must clear break-even
# at -110 juice AND show meaningful deviation from the base rate.
BET_THRESHOLD_HIGH      = 0.58   # HOME -1.5 strong signal
BET_THRESHOLD_HIGH_LEAN = 0.54   # HOME -1.5 lean
BET_THRESHOLD_LOW       = 0.34   # AWAY +1.5 strong signal  (well below ~36% base)
BET_THRESHOLD_LOW_LEAN  = 0.40   # AWAY +1.5 lean


# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------

def load_lineups(date_str: str) -> pd.DataFrame:
    """Load today's starters from lineups_today.parquet."""
    path = DATA_DIR / "lineups_today.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_parquet(path, engine="pyarrow")
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)

    today = df[df["game_date"] == date_str]
    if len(today) == 0:
        # Fall back to most recent date in file
        latest = df["game_date"].max()
        print(f"  [WARN] No lineups for {date_str}, using most recent: {latest}")
        today = df[df["game_date"] == latest]

    return today[[
        "game_pk", "game_date", "home_team", "away_team",
        "home_starter_name", "away_starter_name",
        "home_lineup_confirmed", "away_lineup_confirmed",
    ]]


def load_odds(date_str: str) -> pd.DataFrame:
    """Load today's Vegas odds — tries dated file first, then most recent."""
    # Try exact date file
    dated = DATA_DIR / f"odds_current_{date_str}.parquet"
    if dated.exists():
        df = pd.read_parquet(dated, engine="pyarrow")
    else:
        # Fall back to any odds_current file, most recent
        candidates = sorted(DATA_DIR.glob("odds_current_*.parquet"), reverse=True)
        if not candidates:
            print("  [WARN] No odds_current file found — running without Vegas lines")
            return pd.DataFrame()
        df = pd.read_parquet(candidates[0], engine="pyarrow")
        print(f"  [WARN] Using odds file: {candidates[0].name}")

    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
    today = df[df["game_date"] == date_str]
    if len(today) == 0:
        today = df   # use all rows if date doesn't match

    for col in ["close_ml_home", "close_ml_away", "close_total",
                "runline_home_odds"]:
        if col in today.columns:
            today[col] = pd.to_numeric(today[col], errors="coerce")
    return today


def load_weather(date_str: str) -> pd.DataFrame:
    """Load weather for today's games."""
    path = DATA_DIR / "weather_2026.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path, engine="pyarrow")
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
    today = df[df["game_date"] == date_str]
    if len(today) == 0:
        return pd.DataFrame()
    return today[["home_team", "temp_f", "wind_mph", "humidity"]]


def ml_to_prob(ml):
    if pd.isna(ml) or ml is None:
        return np.nan
    ml = float(ml)
    return 100 / (ml + 100) if ml > 0 else abs(ml) / (abs(ml) + 100)


def _edge_vs_line(model_prob: float, market_odds: float) -> float:
    """Edge = model probability minus implied probability from market odds."""
    implied = ml_to_prob(market_odds)
    if np.isnan(implied):
        return np.nan
    return model_prob - implied


def _best_bet(
    mc_home_win: float,
    mc_home_covers_rl: float,
    mc_home_covers_25: float,
    mc_away_covers_25: float,
    blended_rl: float,
    vegas_ml_home,
    vegas_ml_away,
    rl_home_odds,
    alt_home_25_odds,
    alt_away_25_odds,
) -> dict:
    """
    Evaluate edge across all available lines and return the best bet.

    Lines evaluated (home-perspective):
      ML home   : model=mc_home_win     vs vegas_ml_home
      ML away   : model=(1-mc_home_win) vs vegas_ml_away
      RL -1.5 H : model=blended_rl      vs rl_home_odds (market spread: -1.5 H)
      RL +1.5 A : model=(1-blended_rl)  vs implied from rl_home_odds (flipped)
      RL -2.5 H : model=mc_home_covers_25 vs alt_home_25_odds
      RL +2.5 A : model=mc_away_covers_25  vs alt_away_25_odds

    Returns dict with keys: line, side, model_prob, market_odds, edge, tier
    tier: ** = strong (edge >= 0.08), * = lean (edge >= 0.04), "" = skip
    """
    candidates = []

    def add(line, model_prob, market_odds):
        if market_odds is None or pd.isna(market_odds):
            return
        edge = _edge_vs_line(model_prob, float(market_odds))
        if not np.isnan(edge):
            candidates.append({
                "line":         line,
                "model_prob":   round(model_prob, 3),
                "market_odds":  int(market_odds),
                "edge":         round(edge, 4),
            })

    # Moneyline candidates
    add(f"ML (home)",  mc_home_win,       vegas_ml_home)
    add(f"ML (away)",  1 - mc_home_win,   vegas_ml_away)

    # Run line candidates — standard (-1.5)
    add("RL -1.5 (home)", blended_rl,       rl_home_odds)
    # Away +1.5: need away RL odds — estimate from home RL odds if not available
    # At -1.5 the away is +1.5; if home RL is e.g. -130, away +1.5 is roughly +110
    if rl_home_odds is not None and not pd.isna(rl_home_odds):
        # Simple approximation: flip + 20 cents juice
        ho = float(rl_home_odds)
        away_rl_est = int(round((-ho + 20) if ho < 0 else (-ho - 20)))
        add("RL +1.5 (away)", 1 - blended_rl, away_rl_est)

    # Alternate run line candidates
    add("RL -2.5 (home)", mc_home_covers_25, alt_home_25_odds)
    add("RL +2.5 (away)", mc_away_covers_25, alt_away_25_odds)

    if not candidates:
        return {"line": "", "model_prob": blended_rl, "market_odds": None,
                "edge": 0.0, "tier": ""}

    # Pick highest edge candidate
    best = max(candidates, key=lambda c: c["edge"])

    if best["edge"] >= 0.08:
        tier = "**"
    elif best["edge"] >= 0.04:
        tier = "*"
    else:
        tier = ""

    best["tier"] = tier

    # Build a clean signal string like "HOME -1.5 **" or "ML (away) *"
    if tier:
        best["signal"] = f"{best['line']} {tier}"
    else:
        best["signal"] = ""

    return best


# ---------------------------------------------------------------------------
# RUN PREDICTIONS
# ---------------------------------------------------------------------------

def run_card(date_str: str, min_edge: float = 0.0) -> list[dict]:
    from monte_carlo_runline import predict_game, load_profiles

    # Load all data sources
    lineups = load_lineups(date_str)
    odds    = load_odds(date_str)
    weather = load_weather(date_str)
    profiles = load_profiles()

    if len(lineups) == 0:
        print("  No games found for today.")
        return []

    # Merge odds and weather onto lineups
    if not odds.empty:
        odds_cols = ["home_team", "away_team", "close_ml_home",
                     "close_ml_away", "close_total", "runline_home_odds"]
        # Include alternate line odds if present in the odds file
        for alt_col in ["alt_rl_home_25_odds", "alt_rl_away_25_odds",
                        "alt_rl_home_ml_odds", "alt_rl_away_ml_odds"]:
            if alt_col in odds.columns:
                odds_cols.append(alt_col)
        odds_merge = odds[odds_cols].drop_duplicates(subset=["home_team", "away_team"])
        lineups = lineups.merge(odds_merge, on=["home_team", "away_team"],
                                how="left")

    if not weather.empty:
        lineups = lineups.merge(weather, on="home_team", how="left")

    # Fill defaults
    if "temp_f" not in lineups.columns:
        lineups["temp_f"] = 72.0
    lineups["temp_f"] = lineups["temp_f"].fillna(72.0)

    results = []
    today_month = int(date_str.split("-")[1])

    print(f"\n  Running predictions for {len(lineups)} games on {date_str} ...\n")

    for _, game in lineups.iterrows():
        home = str(game["home_team"])
        away = str(game["away_team"])
        home_sp = str(game.get("home_starter_name", ""))
        away_sp = str(game.get("away_starter_name", ""))
        temp    = float(game.get("temp_f", 72.0))

        if not home_sp or home_sp == "nan" or not away_sp or away_sp == "nan":
            print(f"  {away} @ {home}: missing starter(s) — skipping")
            continue

        # Normalize name format: "Last, First" -> "FIRST LAST"
        def norm(name):
            name = str(name).strip()
            if "," in name:
                parts = [p.strip() for p in name.split(",", 1)]
                return f"{parts[1]} {parts[0]}".upper()
            return name.upper()

        home_sp_norm = norm(home_sp)
        away_sp_norm = norm(away_sp)

        vegas_ml   = game.get("close_ml_home")
        vegas_tot  = game.get("close_total")
        rl_odds    = game.get("runline_home_odds")

        try:
            res = predict_game(
                home_team=home, away_team=away,
                home_sp_name=home_sp_norm, away_sp_name=away_sp_norm,
                temp_f=temp, month=today_month,
                verbose=False,
            )
        except Exception as e:
            print(f"  {away} @ {home}: ERROR — {e}")
            continue

        blended_rl  = res.get("blended_home_covers_rl",
                               res.get("mc_home_covers_rl"))
        blended_tot = res.get("blended_expected_total",
                               res.get("mc_expected_total"))
        xgb_rl      = res.get("xgb_home_covers_rl", np.nan)
        mc_rl       = res["mc_home_covers_rl"]

        # Gather alternate line odds (only present if odds file has them)
        vegas_ml_away    = game.get("close_ml_away")
        alt_home_25_odds = game.get("alt_rl_home_25_odds")
        alt_away_25_odds = game.get("alt_rl_away_25_odds")

        # Best-bet selection across all available lines
        best = _best_bet(
            mc_home_win      = res.get("mc_home_win_prob", 0.5),
            mc_home_covers_rl= blended_rl,
            mc_home_covers_25= res.get("mc_home_covers_25", 0.0),
            mc_away_covers_25= res.get("mc_away_covers_25", 0.0),
            blended_rl       = blended_rl,
            vegas_ml_home    = vegas_ml,
            vegas_ml_away    = vegas_ml_away,
            rl_home_odds     = rl_odds,
            alt_home_25_odds = alt_home_25_odds,
            alt_away_25_odds = alt_away_25_odds,
        )

        signal = best.get("signal", "")

        # Also keep the legacy rl_signal for backwards compat (backtest etc.)
        if blended_rl >= BET_THRESHOLD_HIGH:
            rl_signal_legacy = "HOME -1.5 **"
        elif blended_rl >= BET_THRESHOLD_HIGH_LEAN:
            rl_signal_legacy = "HOME -1.5 *"
        elif blended_rl <= BET_THRESHOLD_LOW:
            rl_signal_legacy = "AWAY +1.5 **"
        elif blended_rl <= BET_THRESHOLD_LOW_LEAN:
            rl_signal_legacy = "AWAY +1.5 *"
        else:
            rl_signal_legacy = ""

        # Total signal
        total_signal = ""
        if not pd.isna(vegas_tot) and not pd.isna(blended_tot):
            diff = blended_tot - float(vegas_tot)
            if abs(diff) >= 0.75:
                total_signal = f"{'OVER' if diff > 0 else 'UNDER'} {vegas_tot}"

        results.append({
            "game":           f"{away} @ {home}",
            "home_team":      home,
            "away_team":      away,
            "home_sp":        home_sp_norm,
            "away_sp":        away_sp_norm,
            "home_sp_flag":   res.get("home_sp_velocity_flag", ""),
            "away_sp_flag":   res.get("away_sp_velocity_flag", ""),
            "home_sp_xwoba":  res.get("home_sp_mu"),
            "away_sp_xwoba":  res.get("away_sp_mu"),
            "temp_f":         temp,
            "mc_rl":          round(mc_rl, 3),
            "xgb_rl":         round(xgb_rl, 3) if not pd.isna(xgb_rl) else None,
            "blended_rl":     round(blended_rl, 3),
            "mc_home_win":    round(res.get("mc_home_win_prob", 0.5), 3),
            "mc_home_cvr25":  round(res.get("mc_home_covers_25", 0.0), 3),
            "mc_away_cvr25":  round(res.get("mc_away_covers_25", 0.0), 3),
            "mc_total":       round(res["mc_expected_total"], 2),
            "blended_total":  round(blended_tot, 2) if blended_tot else None,
            "vegas_ml_home":  vegas_ml,
            "vegas_ml_away":  vegas_ml_away,
            "vegas_total":    vegas_tot,
            "rl_odds":        rl_odds,
            "alt_home_25_odds": alt_home_25_odds,
            "alt_away_25_odds": alt_away_25_odds,
            "vegas_implied":  round(ml_to_prob(vegas_ml), 3) if not pd.isna(vegas_ml) else None,
            # Best bet recommendation
            "best_line":      best.get("line", ""),
            "best_model_prob":best.get("model_prob", blended_rl),
            "best_market_odds":best.get("market_odds"),
            "best_edge":      best.get("edge", 0.0),
            "best_tier":      best.get("tier", ""),
            # Legacy fields (used by backtest)
            "rl_signal":      rl_signal_legacy,
            "total_signal":   total_signal,
            "lineup_confirmed": bool(game.get("home_lineup_confirmed", False)
                                     and game.get("away_lineup_confirmed", False)),
        })

    return results


# ---------------------------------------------------------------------------
# FORMATTING
# ---------------------------------------------------------------------------

def print_card(results: list[dict], min_edge: float = 0.0) -> None:
    if not results:
        return

    # Sort: ** signals first, then *, then no signal; within each group by strength
    def sort_key(r):
        sig = r["rl_signal"]
        if "**" in sig:
            tier = 0
        elif "*" in sig:
            tier = 1
        else:
            tier = 2
        return (tier, -abs(r["blended_rl"] - 0.5))

    results = sorted(results, key=sort_key)

    print("=" * 72)
    print(f"  MLB RUN LINE CARD  —  {results[0]['game'].split()[2] if results else ''}")
    print("=" * 72)

    strong = [r for r in results if "**" in r["rl_signal"]]
    lean   = [r for r in results if "*" in r["rl_signal"] and "**" not in r["rl_signal"]]
    watch  = [r for r in results if not r["rl_signal"]]

    if strong:
        print(f"\n  ** STRONG SIGNALS ({len(strong)} games)\n")
        for r in strong:
            _print_game_row(r, highlight=True)

    if lean:
        print(f"\n  *  LEAN SIGNALS ({len(lean)} games)\n")
        for r in lean:
            _print_game_row(r, highlight=True)

    if watch:
        if min_edge > 0:
            return
        print(f"\n  —  NO SIGNAL ({len(watch)} games)\n")
        for r in watch:
            _print_game_row(r, highlight=False)

    print()
    print(f"  Thresholds: ** HOME >= {BET_THRESHOLD_HIGH:.2f} | "
          f"* HOME >= {BET_THRESHOLD_HIGH_LEAN:.2f} | "
          f"** AWAY <= {BET_THRESHOLD_LOW:.2f} | "
          f"* AWAY <= {BET_THRESHOLD_LOW_LEAN:.2f}")
    print(f"  Blend: 60% Monte Carlo + 40% XGBoost  "
          f"(base home -1.5 cover rate: ~35.7%)")
    print()


def _print_game_row(r: dict, highlight: bool) -> None:
    conf = "" if r["lineup_confirmed"] else " [PROJECTED]"
    flag_h = f" [{r['home_sp_flag']}]" if r["home_sp_flag"] not in ("NORMAL", "UNKNOWN", "") else ""
    flag_a = f" [{r['away_sp_flag']}]" if r["away_sp_flag"] not in ("NORMAL", "UNKNOWN", "") else ""

    print(f"  {r['game']}{conf}")
    print(f"    HOME SP: {r['home_sp']:<28} xwOBA={r['home_sp_xwoba']:.3f}{flag_h}")
    print(f"    AWAY SP: {r['away_sp']:<28} xwOBA={r['away_sp_xwoba']:.3f}{flag_a}")

    temp_str = f"{r['temp_f']:.0f}°F"
    has_vegas = r['vegas_ml_home'] is not None and not pd.isna(r['vegas_ml_home'])
    has_rl    = r['rl_odds'] is not None and not pd.isna(r['rl_odds'])
    vegas_str = (f"ML={int(r['vegas_ml_home']):+d}  total={r['vegas_total']}"
                 if has_vegas else "no Vegas line")
    rl_str  = f"RL_odds={int(r['rl_odds']):+d}" if has_rl else ""

    print(f"    {temp_str}  |  {vegas_str}  {rl_str}")
    print(f"    MC={r['mc_rl']:.3f}  XGB={r['xgb_rl'] or 'N/A'}  "
          f"BLEND={r['blended_rl']:.3f}  |  "
          f"total_model={r['blended_total'] or r['mc_total']:.1f}")

    if r["rl_signal"]:
        print(f"    >>> BET: {r['rl_signal']}   prob={r['blended_rl']:.3f}")
    if r["total_signal"]:
        print(f"    >>> TOTAL: {r['total_signal']}")
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def _edge_pct(blended_rl: float, signal: str) -> str:
    """Return edge vs -110 breakeven (52.4%) as a string like +18%."""
    breakeven = 100 / 210  # 52.38%
    if "HOME" in signal:
        edge = blended_rl - breakeven
    else:
        edge = (1 - blended_rl) - breakeven
    return f"{edge:+.0%}"


def _why(r: dict) -> str:
    """One-line reason for the bet."""
    parts = []
    home_x = r.get("home_sp_xwoba") or 0
    away_x = r.get("away_sp_xwoba") or 0
    signal = r["rl_signal"]

    # Flag the favored pitcher
    if "AWAY" in signal:
        sp_name = r["away_sp"].title()
        sp_x = away_x
        opp_name = r["home_sp"].title()
        opp_x = home_x
    else:
        sp_name = r["home_sp"].title()
        sp_x = home_x
        opp_name = r["away_sp"].title()
        opp_x = away_x

    if sp_x < 0.280:
        parts.append(f"{sp_name} elite (xwOBA {sp_x:.3f})")
    elif sp_x < 0.305:
        parts.append(f"{sp_name} strong (xwOBA {sp_x:.3f})")

    if opp_x > 0.360:
        parts.append(f"{opp_name} struggling (xwOBA {opp_x:.3f})")
    elif opp_x > 0.330:
        parts.append(f"{opp_name} below avg (xwOBA {opp_x:.3f})")

    # Velocity flags
    if "AWAY" in signal and r.get("home_sp_flag") in ("VOLATILE",):
        parts.append(f"{r['home_sp'].title()} velocity declining [VOLATILE]")
    if "HOME" in signal and r.get("away_sp_flag") in ("VOLATILE",):
        parts.append(f"{r['away_sp'].title()} velocity declining [VOLATILE]")
    if "AWAY" in signal and r.get("away_sp_flag") == "GAINER":
        parts.append(f"{r['away_sp'].title()} velo trending up [GAINER]")

    # Temperature
    temp = r.get("temp_f", 72)
    if temp and temp > 80:
        parts.append(f"warm weather ({temp:.0f}F)")
    elif temp and temp < 50:
        parts.append(f"cold weather ({temp:.0f}F, suppresses scoring)")

    return " | ".join(parts) if parts else "model edge on run differential"


def _build_email_body(results: list[dict], date_str: str) -> str:
    """Build a clean action-first email: BET -> Game -> Why -> Stats."""
    import datetime as dt
    try:
        day_label = dt.datetime.strptime(date_str, "%Y-%m-%d").strftime("%A, %B %-d")
    except Exception:
        day_label = date_str

    strong = [r for r in results if "**" in r["rl_signal"]]
    lean   = [r for r in results if "*"  in r["rl_signal"] and "**" not in r["rl_signal"]]
    skip   = [r for r in results if not r["rl_signal"]]

    play_count = len(strong) + len(lean)
    lines = []
    lines.append("THE WIZARD — MLB PICKS")
    lines.append(day_label)
    lines.append("=" * 48)

    if play_count == 0:
        lines.append("\nNo plays today — model sees no edge on the slate.")
        lines.append("=" * 48)
        return "\n".join(lines)

    lines.append(f"\nTODAY'S PLAYS  ({len(strong)} strong  |  {len(lean)} lean)")

    def fmt_play(r, n, units):
        signal   = r["rl_signal"]
        away, home = r["away_team"], r["home_team"]
        conf = "" if r["lineup_confirmed"] else " *"

        # Action line — what to bet
        if "AWAY" in signal:
            bet_team = away
            bet_line = f"{away} +1.5"
        else:
            bet_team = home
            bet_line = f"{home} -1.5"

        total_bets = []
        if r["total_signal"]:
            total_bets.append(r["total_signal"])

        edge = _edge_pct(r["blended_rl"], signal)
        conf_pct = int((1 - r["blended_rl"] if "AWAY" in signal else r["blended_rl"]) * 100)

        has_vegas = r["vegas_ml_home"] is not None and not pd.isna(r["vegas_ml_home"])
        vegas_str = f"O/U {r['vegas_total']}" if has_vegas and r["vegas_total"] else ""

        block = [
            f"",
            f"[{n}]  BET: {bet_line}  ({units} unit{'s' if units > 1 else ''}){conf}",
            f"     {away} @ {home}  {vegas_str}",
            f"     Model confidence: {conf_pct}%  |  Edge vs breakeven: {edge}",
            f"     Why: {_why(r)}",
        ]
        if total_bets:
            block.append(f"     Also: {' + '.join(total_bets)}")

        # Stats (secondary)
        block.append(f"     ---")
        home_flag = f" [{r['home_sp_flag']}]" if r["home_sp_flag"] not in ("NORMAL","UNKNOWN","") else ""
        away_flag = f" [{r['away_sp_flag']}]" if r["away_sp_flag"] not in ("NORMAL","UNKNOWN","") else ""
        block.append(f"     {home} SP: {r['home_sp'].title()}{home_flag}  xwOBA {r['home_sp_xwoba']:.3f}")
        block.append(f"     {away} SP: {r['away_sp'].title()}{away_flag}  xwOBA {r['away_sp_xwoba']:.3f}")

        return "\n".join(block)

    if strong:
        lines.append("\n" + "=" * 48)
        lines.append("  STRONG PLAYS (2 units each)")
        lines.append("=" * 48)
        for i, r in enumerate(strong, 1):
            lines.append(fmt_play(r, i, 2))

    if lean:
        lines.append("\n" + "-" * 48)
        lines.append("  LEAN PLAYS (1 unit each)")
        lines.append("-" * 48)
        offset = len(strong)
        for i, r in enumerate(lean, offset + 1):
            lines.append(fmt_play(r, i, 1))

    if skip:
        lines.append("\n" + "-" * 48)
        lines.append(f"  NO EDGE ({len(skip)} games skipped)")
        skip_games = "  " + ",  ".join(r["game"] for r in skip)
        lines.append(skip_games)

    lines.append("\n" + "=" * 48)
    lines.append("* lineup projected, not confirmed")
    lines.append("Breakeven at -110 juice: 52.4%")
    lines.append("=" * 48)

    return "\n".join(lines)


# legacy — keep for console output formatting
def _build_email_body_old(results: list[dict], date_str: str) -> str:
    lines = []
    lines.append(f"MLB Run Line Card — {date_str}")
    lines.append("=" * 60)

    strong = [r for r in results if "**" in r["rl_signal"]]
    lean   = [r for r in results if "*"  in r["rl_signal"] and "**" not in r["rl_signal"]]
    watch  = [r for r in results if not r["rl_signal"]]

    if strong:
        lines.append(f"\n** STRONG SIGNALS ({len(strong)} games)")
        lines.append("-" * 40)
        for r in strong:
            lines.append(fmt(r))
            lines.append("")

    if lean:
        lines.append(f"\n*  LEAN SIGNALS ({len(lean)} games)")
        lines.append("-" * 40)
        for r in lean:
            lines.append(fmt(r))
            lines.append("")

    if watch:
        lines.append(f"\n-- NO SIGNAL ({len(watch)} games)")
        lines.append("-" * 40)
        for r in watch:
            lines.append(fmt(r))
            lines.append("")

    lines.append(f"Thresholds: ** HOME>=0.58 | * HOME>=0.54 | ** AWAY<=0.34 | * AWAY<=0.40")
    lines.append(f"Blend: 60% Monte Carlo + 40% XGBoost")
    return "\n".join(lines)


def send_card_email(results: list[dict], date_str: str) -> None:
    """Send the daily card as a plain-text email via Gmail."""
    import os
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    gmail_from = os.getenv("GMAIL_FROM", "garcia.dan24@gmail.com")
    gmail_pass = os.getenv("GMAIL_APP_PASSWORD", "")
    recipients = os.getenv("EMAIL_RECIPIENTS", gmail_from).split(",")

    if not gmail_pass:
        print("  [WARN] GMAIL_APP_PASSWORD not set — skipping email")
        return

    strong_count = sum(1 for r in results if "**" in r["rl_signal"])
    lean_count   = sum(1 for r in results if "*"  in r["rl_signal"] and "**" not in r["rl_signal"])
    subject = f"MLB Picks {date_str} — {strong_count} strong, {lean_count} lean signals"

    body = _build_email_body(results, date_str)

    msg = MIMEMultipart()
    msg["From"]    = gmail_from
    msg["To"]      = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(gmail_from, gmail_pass)
            s.sendmail(gmail_from, recipients, msg.as_string())
        print(f"  Email sent -> {', '.join(recipients)}")
    except Exception as e:
        print(f"  [ERROR] Email failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run automated MLB daily prediction card")
    parser.add_argument("--date",      type=str,   default=None,
                        help="Date (YYYY-MM-DD, default: today)")
    parser.add_argument("--min-edge",  type=float, default=0.0,
                        help="Only show games above this RL probability threshold")
    parser.add_argument("--csv",       action="store_true",
                        help="Write results to daily_card.csv")
    parser.add_argument("--email",     action="store_true",
                        help="Send the card as an email via Gmail")
    args = parser.parse_args()

    date_str = args.date or str(datetime.date.today())

    print("=" * 72)
    print("  run_today.py — automated daily card")
    print(f"  Date: {date_str}")
    print("=" * 72)

    results = run_card(date_str, min_edge=args.min_edge)
    print_card(results, min_edge=args.min_edge)

    if args.csv and results:
        out = "daily_card.csv"
        pd.DataFrame(results).to_csv(out, index=False)
        print(f"  Saved -> {out}")

    if args.email and results:
        send_card_email(results, date_str)


if __name__ == "__main__":
    main()
