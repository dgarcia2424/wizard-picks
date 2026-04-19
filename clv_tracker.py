"""
clv_tracker.py
==============
True Closing Line Value (CLV) tracker for the Wizard Report MLB pipeline.

CLV measures whether our picks consistently get better odds than the closing
line. Positive CLV = we bet when the market undervalued our side; the market
later moved toward us. Long-run positive CLV is the strongest evidence of
a real edge, independent of short-run P&L noise.

How it works
------------
  "Open" odds  = Pinnacle P_true at 10 AM (when the morning card is made)
  "Closing" odds = Pinnacle P_true at the latest snapshot before game time
                   (typically the 4 PM odds pull)

  CLV per bet = closing_p_true - pick_p_true
    Positive → market moved away from our side → we got better than closing
    Negative → market moved toward our side → line moved against us

Data sources
------------
  daily_cards/daily_card_2026-*.csv   — pick time, direction, p_true at pick
  data/statcast/odds_history_*.parquet — all intraday snapshots with timestamp
  data/statcast/actuals_2026.parquet   — actual outcomes for resolved games

Output
------
  clv_tracker_results.csv  — one row per pick with CLV and outcome
  Console summary: average CLV by bet type, win rate at positive CLV, etc.

Usage
-----
  python clv_tracker.py              # print report
  python clv_tracker.py --update     # save results to CSV
  python clv_tracker.py --days 14    # look back N days (default: all)
"""

import argparse
import glob
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR  = Path("./data/statcast")
CARDS_DIR = Path("./daily_cards")

# Snapshot hour used as "closing" — last pull before most evening games
CLOSING_SNAPSHOT_HOUR_UTC = 20   # 4 PM ET = 20:00 UTC


# ---------------------------------------------------------------------------
# Load picks from daily cards
# ---------------------------------------------------------------------------

def load_picks(days: int | None = None) -> pd.DataFrame:
    """
    Extract all locked picks from daily card CSVs.
    Returns one row per pick direction (RL home, RL away, ML, O/U).
    """
    card_files = sorted(glob.glob(str(CARDS_DIR / "daily_card_2026-*.csv")))
    if not card_files:
        return pd.DataFrame()

    rows = []
    for f in card_files:
        card_date = Path(f).stem.replace("daily_card_", "")
        if days is not None:
            cutoff = date.today().toordinal() - days
            if datetime.strptime(card_date, "%Y-%m-%d").toordinal() < cutoff:
                continue

        df = pd.read_csv(f)

        for _, g in df.iterrows():
            home = g.get("home_team")
            away = g.get("away_team")

            # RL home -1.5
            if pd.notna(g.get("lock_tier")) and pd.notna(g.get("lock_p_true")):
                rows.append({
                    "date": card_date, "home_team": home, "away_team": away,
                    "bet_type": "RL_HOME", "direction": "home",
                    "pick_p_true": float(g["lock_p_true"]),
                    "pick_p_model": float(g.get("lock_p_model", np.nan)),
                    "pick_edge": float(g.get("lock_edge", np.nan)),
                    "tier": int(g["lock_tier"]),
                    "stake": float(g.get("lock_dollars", 0)),
                    "odds": float(g.get("rl_odds", np.nan)),
                })

            # RL away +1.5
            if pd.notna(g.get("away_lock_tier")) and pd.notna(g.get("away_lock_p_true")):
                rows.append({
                    "date": card_date, "home_team": home, "away_team": away,
                    "bet_type": "RL_AWAY", "direction": "away",
                    "pick_p_true": float(g["away_lock_p_true"]),
                    "pick_p_model": float(g.get("away_lock_p_model", np.nan)),
                    "pick_edge": float(g.get("away_lock_edge", np.nan)),
                    "tier": int(g["away_lock_tier"]),
                    "stake": float(g.get("away_lock_dollars", 0)),
                    "odds": float(-g["rl_odds"]) if pd.notna(g.get("rl_odds")) else np.nan,
                })

            # ML home
            if pd.notna(g.get("ml_lock_tier")) and pd.notna(g.get("ml_lock_p_true")):
                rows.append({
                    "date": card_date, "home_team": home, "away_team": away,
                    "bet_type": "ML_HOME", "direction": "home",
                    "pick_p_true": float(g["ml_lock_p_true"]),
                    "pick_p_model": float(g.get("ml_lock_p_model", np.nan)),
                    "pick_edge": float(g.get("ml_lock_edge", np.nan)),
                    "tier": int(g["ml_lock_tier"]),
                    "stake": float(g.get("ml_lock_dollars", 0)),
                    "odds": float(g.get("vegas_ml_home", np.nan)),
                })

            # O/U
            if pd.notna(g.get("ou_lock_tier")) and pd.notna(g.get("ou_p_true")):
                direction = g.get("ou_direction", "OVER")
                rows.append({
                    "date": card_date, "home_team": home, "away_team": away,
                    "bet_type": f"OU_{direction}", "direction": direction.lower(),
                    "pick_p_true": float(g["ou_p_true"]),
                    "pick_p_model": float(g.get("ou_p_model", np.nan)),
                    "pick_edge": float(g.get("ou_lock_edge", np.nan)),
                    "tier": int(g["ou_lock_tier"]),
                    "stake": float(g.get("ou_lock_dollars", 0)),
                    "odds": -110.0,
                })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Load closing odds from history snapshots
# ---------------------------------------------------------------------------

def load_closing_p_true(game_date: str) -> pd.DataFrame | None:
    """
    Load the closing-odds snapshot for a given date.
    Uses the latest snapshot at or before CLOSING_SNAPSHOT_HOUR_UTC (4 PM ET).
    Returns a DataFrame with home_team, away_team, close_p_true_home,
    close_p_true_rl_home, close_p_true_over.
    """
    hist_path = DATA_DIR / f"odds_history_{game_date.replace('-', '_')}.parquet"
    if not hist_path.exists():
        return None

    hist = pd.read_parquet(hist_path, engine="pyarrow")
    if hist.empty:
        return None

    hist["snapshot_time"] = pd.to_datetime(hist["snapshot_time"], utc=True)

    # Filter to snapshots at or before closing hour
    cutoff = pd.Timestamp(f"{game_date} {CLOSING_SNAPSHOT_HOUR_UTC:02d}:00:00", tz="UTC")
    before_close = hist[hist["snapshot_time"] <= cutoff]
    if before_close.empty:
        before_close = hist  # fallback: use any available snapshot

    # Take the latest snapshot per game
    latest = (
        before_close.sort_values("snapshot_time")
        .groupby(["home_team", "away_team"])
        .last()
        .reset_index()
    )

    cols = ["home_team", "away_team", "snapshot_time"]
    for col in ["P_true_home", "P_true_away", "P_true_rl_home", "P_true_rl_away",
                "P_true_over", "P_true_under"]:
        if col in latest.columns:
            cols.append(col)

    return latest[cols].rename(columns={
        "P_true_home":     "close_p_true_home",
        "P_true_away":     "close_p_true_away",
        "P_true_rl_home":  "close_p_true_rl_home",
        "P_true_rl_away":  "close_p_true_rl_away",
        "P_true_over":     "close_p_true_over",
        "P_true_under":    "close_p_true_under",
    })


# ---------------------------------------------------------------------------
# Load actuals
# ---------------------------------------------------------------------------

def load_actuals() -> pd.DataFrame:
    path = DATA_DIR / "actuals_2026.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path, engine="pyarrow",
                         columns=["game_date", "home_team", "away_team",
                                  "home_covers_rl", "home_score_final",
                                  "away_score_final"])
    df["date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    df["actual_total"] = df["home_score_final"] + df["away_score_final"]
    return df.drop(columns=["game_date"])


# ---------------------------------------------------------------------------
# Build CLV per pick
# ---------------------------------------------------------------------------

def compute_clv(picks: pd.DataFrame) -> pd.DataFrame:
    """Join picks with closing odds and actuals, compute CLV per pick."""
    actuals = load_actuals()
    results = []

    for _, pick in picks.iterrows():
        row = pick.to_dict()

        # Closing odds
        closing = load_closing_p_true(pick["date"])
        if closing is not None:
            match = closing[
                (closing["home_team"] == pick["home_team"]) &
                (closing["away_team"] == pick["away_team"])
            ]
            if not match.empty:
                m = match.iloc[0]
                bet = pick["bet_type"]
                if bet == "RL_HOME":
                    row["close_p_true"] = m.get("close_p_true_rl_home", np.nan)
                elif bet == "RL_AWAY":
                    row["close_p_true"] = m.get("close_p_true_rl_away", np.nan)
                elif bet == "ML_HOME":
                    row["close_p_true"] = m.get("close_p_true_home", np.nan)
                elif bet == "OU_OVER":
                    row["close_p_true"] = m.get("close_p_true_over", np.nan)
                elif bet == "OU_UNDER":
                    row["close_p_true"] = m.get("close_p_true_under", np.nan)
                row["snapshot_time"] = str(m.get("snapshot_time", ""))

        if "close_p_true" in row and pd.notna(row.get("close_p_true")):
            # CLV = closing - pick. Positive = we got better than closing.
            row["clv"] = float(row["close_p_true"]) - float(row["pick_p_true"])

        # Actual outcome
        if not actuals.empty:
            act = actuals[
                (actuals["home_team"] == pick["home_team"]) &
                (actuals["away_team"] == pick["away_team"]) &
                (actuals["date"] == pick["date"])
            ]
            if not act.empty:
                a = act.iloc[0]
                bet = pick["bet_type"]
                if bet == "RL_HOME":
                    row["won"] = float(a["home_covers_rl"]) if pd.notna(a["home_covers_rl"]) else np.nan
                elif bet == "RL_AWAY":
                    row["won"] = (1 - float(a["home_covers_rl"])) if pd.notna(a["home_covers_rl"]) else np.nan
                elif bet == "ML_HOME":
                    home_won = float(a["home_score_final"]) > float(a["away_score_final"]) \
                               if pd.notna(a["home_score_final"]) else np.nan
                    row["won"] = float(home_won) if home_won is not np.nan else np.nan
                elif bet in ("OU_OVER", "OU_UNDER"):
                    total = a.get("actual_total")
                    posted = pick.get("ou_line", np.nan)
                    if pd.notna(total):
                        row["won"] = 1.0 if (bet == "OU_OVER" and total > posted) \
                                     else (1.0 if (bet == "OU_UNDER" and total < posted) else 0.0)

        results.append(row)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(df: pd.DataFrame) -> None:
    today_str = date.today().isoformat()
    print("=" * 65)
    print(f"  CLV Tracker — {today_str}")
    print("=" * 65)

    if "clv" not in df.columns:
        df["clv"] = np.nan
    if "won" not in df.columns:
        df["won"] = np.nan
    has_clv = df.dropna(subset=["clv"])
    has_outcome = df.dropna(subset=["won"])

    print(f"\n  Total picks: {len(df)} | With CLV: {len(has_clv)} | Resolved: {len(has_outcome)}")

    if has_clv.empty:
        print("\n  No CLV data yet — history snapshots accumulate from today's odds pulls.")
        print("  Re-run tomorrow once the 4 PM snapshot exists for today's picks.")
        return

    print(f"\n  Overall CLV: {has_clv['clv'].mean():+.4f}  "
          f"({has_clv['clv'].mean()*100:+.2f} pp)")
    print(f"  Positive CLV bets: {(has_clv['clv'] > 0).sum()}/{len(has_clv)} "
          f"({(has_clv['clv'] > 0).mean():.1%})")

    print(f"\n  By bet type:")
    print(f"  {'Type':<12}  {'N':>4}  {'Avg CLV':>9}  {'% Positive':>11}  {'Win Rate':>9}")
    for bt, grp in has_clv.groupby("bet_type"):
        resolved = grp.dropna(subset=["won"])
        wr = f"{resolved['won'].mean():.1%}" if len(resolved) > 0 else "—"
        print(f"  {bt:<12}  {len(grp):>4}  {grp['clv'].mean():>+8.4f}  "
              f"{(grp['clv'] > 0).mean():>10.1%}  {wr:>9}")

    if len(has_outcome) >= 5:
        pos_clv = has_outcome[has_outcome.get("clv", pd.Series(dtype=float)) > 0] \
                  if "clv" in has_outcome.columns else pd.DataFrame()
        neg_clv = has_outcome[has_outcome.get("clv", pd.Series(dtype=float)) <= 0] \
                  if "clv" in has_outcome.columns else pd.DataFrame()
        print(f"\n  Win rate by CLV sign:")
        if len(pos_clv) > 0:
            print(f"    Positive CLV picks: {pos_clv['won'].mean():.1%} ({len(pos_clv)} bets)")
        if len(neg_clv) > 0:
            print(f"    Negative CLV picks: {neg_clv['won'].mean():.1%} ({len(neg_clv)} bets)")

    print(f"\n  By tier:")
    for tier, grp in has_clv.groupby("tier"):
        resolved = grp.dropna(subset=["won"])
        wr = f"{resolved['won'].mean():.1%}" if len(resolved) > 0 else "—"
        print(f"    Tier {tier}: avg CLV={grp['clv'].mean():+.4f}  "
              f"n={len(grp)}  win={wr}")

    if len(has_clv) < 30:
        print(f"\n  ⚠  Only {len(has_clv)} picks with CLV. Need 30+ for reliable signal.")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true",
                        help="Save results to clv_tracker_results.csv")
    parser.add_argument("--days", type=int, default=None,
                        help="Only include picks from last N days")
    args = parser.parse_args()

    picks = load_picks(days=args.days)
    if picks.empty:
        print("No locked picks found in daily cards.")
        return

    df = compute_clv(picks)
    report(df)

    if args.update:
        out = Path("clv_tracker_results.csv")
        df.to_csv(out, index=False)
        print(f"\n  Saved {len(df)} picks to {out}")


if __name__ == "__main__":
    main()
