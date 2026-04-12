"""
backtest_2026.py
================
Evaluate model accuracy on completed 2026 games.
Derives starters from statcast_2026.parquet, runs predictions,
and compares to actual results.

Runs in append mode by default — only processes games not already
in the tracker CSV, so it can be called daily without duplication.

Usage:
  python backtest_2026.py                   # append new games, print summary
  python backtest_2026.py --since 2026-04-01  # only process from this date
  python backtest_2026.py --rebuild         # wipe tracker and reprocess all
  python backtest_2026.py --summary         # print summary only (no new processing)
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR    = Path("./statcast_data")
TRACKER_CSV = Path("backtest_2026_results.csv")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    import unicodedata
    if not isinstance(name, str):
        return ""
    name = name.strip()
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        name = f"{parts[1]} {parts[0]}"
    name = name.upper()
    return "".join(
        c for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    )


# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------

def load_games_2026(since: str = None) -> pd.DataFrame:
    """Load completed 2026 games with scores and starters from statcast."""
    sc = pd.read_parquet(DATA_DIR / "statcast_2026.parquet", engine="pyarrow")
    sc["game_date"] = pd.to_datetime(sc["game_date"]).dt.date.astype(str)

    if since:
        sc = sc[sc["game_date"] >= since]
    if len(sc) == 0:
        return pd.DataFrame()

    # Final scores
    scores = sc.groupby(["game_pk", "game_date", "home_team", "away_team"]).agg(
        home_score=("post_home_score", "max"),
        away_score=("post_away_score", "max"),
    ).reset_index()

    # Starters from inning 1
    inn1 = sc[sc["inning"] == 1].copy()
    starters = {}
    for game_pk, g in inn1.groupby("game_pk"):
        row = {}
        top = g[g["inning_topbot"] == "Top"]
        if len(top) > 0:
            sp = top.loc[top["at_bat_number"].idxmin()]
            row["home_starter"] = _normalize_name(str(sp["player_name"]))
        bot = g[g["inning_topbot"] == "Bot"]
        if len(bot) > 0:
            sp = bot.loc[bot["at_bat_number"].idxmin()]
            row["away_starter"] = _normalize_name(str(sp["player_name"]))
        starters[game_pk] = row

    starter_df = pd.DataFrame.from_dict(starters, orient="index").reset_index()
    starter_df.rename(columns={"index": "game_pk"}, inplace=True)

    games = scores.merge(starter_df, on="game_pk", how="left")
    games["home_margin"]    = games["home_score"] - games["away_score"]
    games["home_covers_rl"] = (games["home_margin"] >= 2).astype(int)
    games["actual_total"]   = games["home_score"] + games["away_score"]
    return games


def load_odds_2026() -> pd.DataFrame:
    combined = DATA_DIR / "odds_combined_2026.parquet"
    candidates = sorted(DATA_DIR.glob("odds_current_*.parquet"), reverse=True)
    frames = []
    if combined.exists():
        frames.append(pd.read_parquet(combined, engine="pyarrow"))
    for f in candidates:
        frames.append(pd.read_parquet(f, engine="pyarrow"))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
    for col in ["close_ml_home", "close_ml_away", "close_total"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.drop_duplicates(subset=["game_date", "home_team", "away_team"])


# ---------------------------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------------------------

def run_predictions(games: pd.DataFrame) -> pd.DataFrame:
    from monte_carlo_runline import predict_game, load_profiles

    odds    = load_odds_2026()
    if not odds.empty:
        games = games.merge(
            odds[["game_date", "home_team", "away_team",
                  "close_ml_home", "close_ml_away", "close_total"]],
            on=["game_date", "home_team", "away_team"], how="left"
        )

    profiles = load_profiles()
    results  = []
    skipped  = 0

    for _, game in games.iterrows():
        home    = str(game["home_team"])
        away    = str(game["away_team"])
        home_sp = str(game.get("home_starter", ""))
        away_sp = str(game.get("away_starter", ""))
        month   = int(str(game["game_date"]).split("-")[1])

        if not home_sp or home_sp == "NAN" or not away_sp or away_sp == "NAN":
            skipped += 1
            continue

        try:
            res = predict_game(
                home_team=home, away_team=away,
                home_sp_name=home_sp, away_sp_name=away_sp,
                month=month, verbose=False,
            )
        except Exception:
            skipped += 1
            continue

        blended_rl  = res.get("blended_home_covers_rl", res.get("mc_home_covers_rl"))
        blended_tot = res.get("blended_expected_total", res.get("mc_expected_total"))
        xgb_rl      = res.get("xgb_home_covers_rl", np.nan)

        # Tiered signal
        if blended_rl >= 0.58:
            signal = "HOME -1.5 **"
        elif blended_rl >= 0.54:
            signal = "HOME -1.5 *"
        elif blended_rl <= 0.34:
            signal = "AWAY +1.5 **"
        elif blended_rl <= 0.40:
            signal = "AWAY +1.5 *"
        else:
            signal = ""

        actual_covers = int(game["home_covers_rl"])
        if signal in ("HOME -1.5 **", "HOME -1.5 *"):
            bet_win = actual_covers == 1
        elif signal in ("AWAY +1.5 **", "AWAY +1.5 *"):
            bet_win = actual_covers == 0
        else:
            bet_win = None

        vegas_tot = game.get("close_total")

        results.append({
            "date":           str(game["game_date"]),
            "game_pk":        int(game["game_pk"]),
            "game":           f"{away} @ {home}",
            "home_sp":        home_sp,
            "away_sp":        away_sp,
            "mc_rl":          round(res["mc_home_covers_rl"], 3),
            "xgb_rl":         round(xgb_rl, 3) if not pd.isna(xgb_rl) else None,
            "blended_rl":     round(blended_rl, 3),
            "model_total":    round(blended_tot, 1) if blended_tot else None,
            "vegas_total":    vegas_tot,
            "home_score":     int(game["home_score"]),
            "away_score":     int(game["away_score"]),
            "actual_total":   int(game["actual_total"]),
            "home_margin":    int(game["home_margin"]),
            "home_covers_rl": actual_covers,
            "signal":         signal,
            "bet_win":        bet_win,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("  No results yet.")
        return

    print("\n" + "=" * 70)
    print(f"  2026 SEASON TRACKER  ({df['date'].min()} to {df['date'].max()})")
    print("=" * 70)

    total = len(df)
    actual_rate = df["home_covers_rl"].mean()
    print(f"\n  Games tracked : {total}")
    print(f"  Home -1.5 cover rate: {actual_rate:.1%}  (historical avg 35.7%)")

    # Signal performance table
    print(f"\n  {'Signal':<17} {'Bets':>5} {'Wins':>5} {'Win%':>7} {'ROI':>7}")
    print(f"  {'-'*46}")

    total_bets = 0
    total_wins = 0

    for sig in ["HOME -1.5 **", "HOME -1.5 *", "AWAY +1.5 **", "AWAY +1.5 *"]:
        sub = df[df["signal"] == sig]
        if len(sub) == 0:
            continue
        n    = len(sub)
        wins = int(sub["bet_win"].sum())
        wr   = wins / n
        roi  = (wins * (100/110) - (n - wins)) / n
        total_bets += n
        total_wins += wins
        print(f"  {sig:<17} {n:>5} {wins:>5} {wr:>7.1%} {roi:>+7.1%}")

    if total_bets > 0:
        wr  = total_wins / total_bets
        roi = (total_wins * (100/110) - (total_bets - total_wins)) / total_bets
        print(f"  {'-'*46}")
        print(f"  {'ALL SIGNALS':<17} {total_bets:>5} {total_wins:>5} {wr:>7.1%} {roi:>+7.1%}")

    # Weekly breakdown
    df["week"] = pd.to_datetime(df["date"]).dt.to_period("W").astype(str)
    weeks = df.groupby("week")
    print(f"\n  --- Weekly Signal Performance ---")
    print(f"  {'Week':<22} {'Bets':>5} {'Win%':>7} {'ROI':>7}")
    print(f"  {'-'*42}")
    for week, wdf in weeks:
        sig_wdf = wdf[wdf["signal"] != ""]
        if len(sig_wdf) == 0:
            print(f"  {week:<22} {'—':>5}")
            continue
        n    = len(sig_wdf)
        wins = int(sig_wdf["bet_win"].sum())
        wr   = wins / n
        roi  = (wins * (100/110) - (n - wins)) / n
        print(f"  {week:<22} {n:>5} {wr:>7.1%} {roi:>+7.1%}")

    # Calibration
    print(f"\n  --- Calibration ---")
    bins   = [0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 1.0]
    labels = ["<.30", ".30-.35", ".35-.40", ".40-.45", ".45-.50", ".50-.55", ".55-.60", ">.60"]
    df["bucket"] = pd.cut(df["blended_rl"], bins=bins, labels=labels)
    cal = df.groupby("bucket", observed=True).agg(
        n=("home_covers_rl", "count"),
        actual=("home_covers_rl", "mean"),
        model=("blended_rl", "mean"),
    )
    print(f"  {'bucket':>10}  {'n':>5}  {'actual':>8}  {'model':>8}")
    print(f"  {'-'*38}")
    for bucket, row in cal.iterrows():
        flag = " <--" if row["n"] >= 10 and abs(row["actual"] - row["model"]) > 0.08 else ""
        print(f"  {bucket:>10}  {int(row['n']):>5}  {row['actual']:>8.3f}  {row['model']:>8.3f}{flag}")

    # Total model accuracy
    has_vegas = df["vegas_total"].notna() & df["model_total"].notna()
    if has_vegas.sum() >= 10:
        v = df[has_vegas].copy()
        v["vegas_total"] = pd.to_numeric(v["vegas_total"], errors="coerce")
        v = v.dropna(subset=["vegas_total"])
        mae = (v["model_total"] - v["vegas_total"]).abs().mean()
        w15 = ((v["model_total"] - v["vegas_total"]).abs() <= 1.5).mean()
        actual_mae = (v["actual_total"] - v["vegas_total"]).abs().mean()
        print(f"\n  --- Total Runs ---")
        print(f"  Model MAE vs Vegas : {mae:.2f} runs")
        print(f"  Actual MAE vs Vegas: {actual_mae:.2f} runs  (how hard the problem is)")
        print(f"  Model within 1.5   : {w15:.1%}")

    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Track 2026 model performance")
    parser.add_argument("--since",   type=str, default=None,
                        help="Only process games from this date (YYYY-MM-DD)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Wipe tracker CSV and reprocess all games")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary only, no new processing")
    args = parser.parse_args()

    print("=" * 70)
    print("  backtest_2026.py — 2026 season model tracker")
    print("=" * 70)

    # Load existing tracker
    if TRACKER_CSV.exists() and not args.rebuild:
        existing = pd.read_csv(TRACKER_CSV)
        existing["date"] = existing["date"].astype(str)
    else:
        existing = pd.DataFrame()
        if args.rebuild:
            print("  [REBUILD] wiping tracker CSV")

    if args.summary:
        print_summary(existing)
        return

    # Figure out which game_pks we already have
    already_done = set(existing["game_pk"].astype(int).tolist()) if not existing.empty else set()

    # Load all completed games from statcast
    since = args.since or "2026-03-28"
    print(f"\n  Loading completed games since {since} ...")
    all_games = load_games_2026(since=since)

    if len(all_games) == 0:
        print("  No games found in statcast_2026.parquet for that range.")
        print_summary(existing)
        return

    # Filter to new games only
    new_games = all_games[~all_games["game_pk"].isin(already_done)]
    print(f"  {len(all_games)} total games | {len(already_done)} already tracked | "
          f"{len(new_games)} new to process")

    if len(new_games) == 0:
        print("  Nothing new to process.")
        print_summary(existing)
        return

    print(f"  Running predictions for {len(new_games)} new games ...\n")
    new_results = run_predictions(new_games)

    # Append and save
    combined = pd.concat([existing, new_results], ignore_index=True) if not existing.empty else new_results
    combined = combined.sort_values(["date", "game_pk"]).reset_index(drop=True)
    combined.to_csv(TRACKER_CSV, index=False)
    print(f"  Saved -> {TRACKER_CSV}  ({len(combined)} total games)")

    print_summary(combined)


if __name__ == "__main__":
    main()
