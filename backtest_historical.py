"""
backtest_historical.py
======================
Run historical backtest for 2023, 2024, 2025 seasons using the same
pitcher profiles and team stats built by build_historical_profiles.py.

For each year:
  1. Load game results (backtest_games_{year}.csv or derive from schedule)
  2. Load odds (odds_combined_{year}.parquet)
  3. Load pitcher profiles + team stats
  4. For each game with both starters found, compute a blended RL signal
  5. Save backtest_{year}_results.csv
  6. Print summary stats

Usage:
  python backtest_historical.py
  python backtest_historical.py --years 2024 2025
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "statcast"
RAW_DIR  = BASE_DIR / "data" / "raw"

# Edge thresholds for signaling
EDGE_STRONG = 0.10   # "**"
EDGE_LEAN   = 0.05   # "*"


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def ml_to_prob(ml):
    """Convert American odds to implied probability."""
    if ml is None or (isinstance(ml, float) and np.isnan(ml)):
        return np.nan
    ml = float(ml)
    return 100.0 / (ml + 100.0) if ml > 0 else abs(ml) / (abs(ml) + 100.0)


def _normalize_name(name: str) -> str:
    """
    Normalize a pitcher name to uppercase 'FIRST LAST' format.
    Handles 'Last, First' (statcast) and 'FIRST LAST' (backtest) inputs.
    """
    name = str(name).strip().upper()
    # Remove accents / special characters for fuzzy matching
    replacements = {
        "Á": "A", "À": "A", "Â": "A", "Ä": "A", "Ã": "A",
        "É": "E", "È": "E", "Ê": "E", "Ë": "E",
        "Í": "I", "Ì": "I", "Î": "I", "Ï": "I",
        "Ó": "O", "Ò": "O", "Ô": "O", "Ö": "O", "Õ": "O",
        "Ú": "U", "Ù": "U", "Û": "U", "Ü": "U",
        "Ñ": "N", "Ç": "C",
        "á": "A", "à": "A", "â": "A", "ä": "A", "ã": "A",
        "é": "E", "è": "E", "ê": "E", "ë": "E",
        "í": "I", "ì": "I", "î": "I", "ï": "I",
        "ó": "O", "ò": "O", "ô": "O", "ö": "O", "õ": "O",
        "ú": "U", "ù": "U", "û": "U", "ü": "U",
        "ñ": "N", "ç": "C",
    }
    for src, dst in replacements.items():
        name = name.replace(src, dst)

    if "," in name:
        # "LAST, FIRST" → "FIRST LAST"
        parts = [p.strip() for p in name.split(",", 1)]
        return f"{parts[1]} {parts[0]}"
    return name


def _build_name_lookup(profiles: pd.DataFrame) -> dict:
    """
    Build a dict mapping normalized pitcher name → row (Series).
    Multiple name variations are stored so matching is robust.
    """
    lookup = {}
    for _, row in profiles.iterrows():
        # From pitcher_name ("Last, First") → "FIRST LAST"
        if pd.notna(row.get("pitcher_name")):
            norm = _normalize_name(row["pitcher_name"])
            lookup[norm] = row
        # Also store the raw pitcher_name_upper if available
        if pd.notna(row.get("pitcher_name_upper")):
            upper = str(row["pitcher_name_upper"]).strip()
            norm2 = _normalize_name(upper)
            lookup[norm2] = row
    return lookup


# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------

def load_games(year: int) -> pd.DataFrame:
    """Load game results for the year. Derive from schedule if CSV missing."""
    csv_path = RAW_DIR / f"backtest_games_{year}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        for col in ["actual_game_total", "actual_home_win"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Derive home/away scores from schedule if not present
        if "home_score" not in df.columns or "away_score" not in df.columns:
            sched = _load_schedule(year)
            if sched is not None:
                pk_col = "game_pk" if "game_pk" in df.columns else "gamePk"
                df = df.merge(
                    sched[["game_pk", "home_score", "away_score"]],
                    left_on=pk_col, right_on="game_pk", how="left"
                )
        return df

    # No CSV — derive from schedule + statcast starters
    print(f"  [INFO] backtest_games_{year}.csv not found — deriving from schedule + statcast")
    sched = _load_schedule(year)
    if sched is None:
        raise FileNotFoundError(f"No game data found for {year}")

    df = sched.copy()
    df["actual_game_total"] = df["home_score"] + df["away_score"]
    df["actual_home_win"]   = (df["home_score"] > df["away_score"]).astype(float)

    # Derive starters from statcast: pitcher who threw inning 1
    starters = _derive_starters_from_statcast(year)
    if starters is not None:
        df = df.merge(starters, on="game_pk", how="left")
        df["home_starter"] = df["home_starter"].fillna("")
        df["away_starter"] = df["away_starter"].fillna("")
        n_with = (df["home_starter"] != "").sum()
        print(f"  [INFO] Derived starters for {n_with}/{len(df)} games from statcast")
    else:
        df["home_starter"] = ""
        df["away_starter"] = ""

    return df


def _derive_starters_from_statcast(year: int) -> pd.DataFrame | None:
    """
    Derive starting pitchers from statcast by finding the pitcher
    who threw in inning 1 for each game.
    Returns DataFrame with columns: game_pk, home_starter, away_starter.
    """
    sc_path = DATA_DIR / f"statcast_{year}.parquet"
    if not sc_path.exists():
        return None

    try:
        import pyarrow.parquet as pq
        available = set(pq.read_schema(sc_path).names)
        load_cols = [c for c in ["game_pk", "inning", "inning_topbot",
                                  "pitcher", "player_name",
                                  "home_team", "away_team",
                                  "at_bat_number"] if c in available]
        sc = pd.read_parquet(sc_path, engine="pyarrow", columns=load_cols)

        # Only look at inning 1
        inn1 = sc[pd.to_numeric(sc.get("inning", pd.Series(dtype=float)),
                                errors="coerce") == 1].copy()

        if inn1.empty:
            return None

        results = []
        for game_pk, grp in inn1.groupby("game_pk"):
            # Home starter pitches in Top of inning 1 (away team bats Top)
            # Away starter pitches in Bot of inning 1
            if "inning_topbot" in grp.columns:
                top = grp[grp["inning_topbot"] == "Top"]
                bot = grp[grp["inning_topbot"] == "Bot"]
            else:
                continue

            # The starter is the first pitcher in each half-inning
            home_sp = ""
            away_sp = ""
            if len(top) > 0 and "player_name" in top.columns:
                first_top = top.sort_values("at_bat_number").iloc[0] if "at_bat_number" in top.columns else top.iloc[0]
                name = str(first_top.get("player_name", "") or "")
                if name and name != "nan":
                    home_sp = _normalize_name(name)
            if len(bot) > 0 and "player_name" in bot.columns:
                first_bot = bot.sort_values("at_bat_number").iloc[0] if "at_bat_number" in bot.columns else bot.iloc[0]
                name = str(first_bot.get("player_name", "") or "")
                if name and name != "nan":
                    away_sp = _normalize_name(name)

            results.append({
                "game_pk":      game_pk,
                "home_starter": home_sp,
                "away_starter": away_sp,
            })

        return pd.DataFrame(results) if results else None

    except Exception as e:
        print(f"  [WARN] Could not derive starters from statcast_{year}: {e}")
        return None


def _load_schedule(year: int) -> pd.DataFrame | None:
    """Load schedule parquet and return home-game rows with scores."""
    sched_path = DATA_DIR / f"schedule_all_{year}.parquet"
    if not sched_path.exists():
        return None
    sched = pd.read_parquet(sched_path, engine="pyarrow")
    # Keep one row per game (home perspective)
    if "home_away" in sched.columns:
        sched = sched[sched["home_away"] == "Home"].copy()
    else:
        # Drop duplicates by game_pk
        pk_col = "gamePk" if "gamePk" in sched.columns else "game_pk"
        sched = sched.drop_duplicates(subset=pk_col).copy()

    pk_col = "gamePk" if "gamePk" in sched.columns else "game_pk"
    sched = sched.rename(columns={pk_col: "game_pk"})
    sched["home_score"] = pd.to_numeric(sched.get("home_score"), errors="coerce")
    sched["away_score"] = pd.to_numeric(sched.get("away_score"), errors="coerce")
    sched["game_date"]  = pd.to_datetime(sched["game_date"], errors="coerce")
    sched = sched.dropna(subset=["home_score", "away_score", "game_date"])
    return sched[["game_pk", "game_date", "home_team", "away_team",
                  "home_score", "away_score"]]


def load_odds(year: int) -> pd.DataFrame:
    """Load historical odds."""
    path = DATA_DIR / f"odds_combined_{year}.parquet"
    if not path.exists():
        print(f"  [WARN] odds_combined_{year}.parquet not found")
        return pd.DataFrame()
    df = pd.read_parquet(path, engine="pyarrow")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    for col in ["close_ml_home", "close_ml_away", "close_total", "runline_home_odds"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_pitcher_profiles(year: int) -> dict:
    """Load pitcher profiles and return a name→row lookup dict."""
    path = DATA_DIR / f"pitcher_profiles_{year}.parquet"
    if not path.exists():
        print(f"  [WARN] pitcher_profiles_{year}.parquet not found — run build_historical_profiles.py")
        return {}
    profiles = pd.read_parquet(path, engine="pyarrow")
    return _build_name_lookup(profiles)


def load_team_stats(year: int) -> pd.DataFrame | None:
    """Load team stats indexed by batting_team."""
    path = DATA_DIR / f"team_stats_{year}.parquet"
    if not path.exists():
        print(f"  [WARN] team_stats_{year}.parquet not found")
        return None
    df = pd.read_parquet(path, engine="pyarrow")
    return df.set_index("batting_team")


# ---------------------------------------------------------------------------
# SIGNAL COMPUTATION
# ---------------------------------------------------------------------------

def compute_signal(
    home_sp_row: pd.Series,
    away_sp_row: pd.Series,
    close_ml_home: float,
    home_ts: pd.Series | None,
    away_ts: pd.Series | None,
) -> dict:
    """
    Compute a blended run-line probability and bet signal.

    Uses the simple formula described in the spec:
      home_advantage = 0.54
      pitcher_edge = (away_sp_xwoba - home_sp_xwoba) * 2.5
      blended_rl = clamp(home_advantage + pitcher_edge, 0.25, 0.75)

    Then compute edge vs Vegas implied probability.
    """
    home_xwoba = float(home_sp_row.get("xwoba_against", 0.318) or 0.318)
    away_xwoba = float(away_sp_row.get("xwoba_against", 0.318) or 0.318)

    # Pitcher edge: positive = home pitcher is better (allows lower xwoba)
    pitcher_edge = (away_xwoba - home_xwoba) * 2.5

    # Team batting adjustment (optional, small weight)
    bat_edge = 0.0
    if home_ts is not None and away_ts is not None:
        # Use batting xwoba vs opposing pitcher's handedness
        away_hand = str(away_sp_row.get("p_throws", "R"))
        home_hand = str(home_sp_row.get("p_throws", "R"))
        away_col = "bat_xwoba_vs_rhp" if away_hand == "R" else "bat_xwoba_vs_lhp"
        home_col = "bat_xwoba_vs_rhp" if home_hand == "R" else "bat_xwoba_vs_lhp"

        home_bat_xwoba = float(home_ts.get(away_col, 0.318) or 0.318)
        away_bat_xwoba = float(away_ts.get(home_col, 0.318) or 0.318)
        bat_edge = (home_bat_xwoba - away_bat_xwoba) * 0.5

    blended_rl = float(np.clip(0.54 + pitcher_edge + bat_edge, 0.25, 0.75))

    vegas_implied = ml_to_prob(close_ml_home)
    edge = (blended_rl - vegas_implied) if not np.isnan(vegas_implied) else np.nan

    # Determine signal
    if np.isnan(edge):
        signal = ""
    elif edge >= EDGE_STRONG:
        signal = "HOME -1.5 **"
    elif edge >= EDGE_LEAN:
        signal = "HOME -1.5 *"
    elif edge <= -EDGE_STRONG:
        signal = "AWAY +1.5 **"
    elif edge <= -EDGE_LEAN:
        signal = "AWAY +1.5 *"
    else:
        signal = ""

    return {
        "blended_rl":    round(blended_rl, 4),
        "home_sp_xwoba": round(home_xwoba, 4),
        "away_sp_xwoba": round(away_xwoba, 4),
        "vegas_implied": round(vegas_implied, 4) if not np.isnan(vegas_implied) else None,
        "edge":          round(edge, 4) if not np.isnan(edge) else None,
        "signal":        signal,
    }


# ---------------------------------------------------------------------------
# BACKTEST RUNNER
# ---------------------------------------------------------------------------

def run_year(year: int, verbose: bool = True) -> pd.DataFrame:
    """Run the backtest for one season year. Returns results DataFrame."""
    print("\n" + "="*60)
    print(f"  Backtest {year}")
    print("="*60)

    try:
        games = load_games(year)
    except Exception as e:
        print(f"  [ERROR] Could not load games for {year}: {e}")
        return pd.DataFrame()

    # Ensure home_score / away_score exist
    if "home_score" not in games.columns:
        # Try to get from schedule
        sched = _load_schedule(year)
        if sched is not None:
            pk_col = "game_pk" if "game_pk" in games.columns else "gamePk"
            games = games.merge(
                sched[["game_pk", "home_score", "away_score"]],
                left_on=pk_col, right_on="game_pk", how="left", suffixes=("", "_sched")
            )
        else:
            games["home_score"] = np.nan
            games["away_score"] = np.nan

    odds = load_odds(year)
    pitcher_lookup = load_pitcher_profiles(year)
    team_stats = load_team_stats(year)

    print(f"  Games: {len(games)} | Odds rows: {len(odds)} | "
          f"Pitchers in lookup: {len(pitcher_lookup)}")

    # Build odds lookup: (home_team, away_team, date) → odds row
    if not odds.empty:
        odds["_key"] = (
            odds["home_team"].astype(str) + "|" +
            odds["away_team"].astype(str) + "|" +
            odds["game_date"].dt.strftime("%Y-%m-%d")
        )
        odds_dict = odds.drop_duplicates("_key").set_index("_key").to_dict("index")
    else:
        odds_dict = {}

    results = []
    n_no_starter  = 0
    n_no_odds     = 0
    n_no_pitcher  = 0
    n_processed   = 0

    for _, game in games.iterrows():
        home = str(game.get("home_team", ""))
        away = str(game.get("away_team", ""))
        gdate = game.get("game_date")
        if pd.isna(gdate):
            continue

        date_str = pd.Timestamp(gdate).strftime("%Y-%m-%d")

        # Get starters (may be blank if derived from schedule)
        home_sp_raw = str(game.get("home_starter", "") or "")
        away_sp_raw = str(game.get("away_starter", "") or "")

        if not home_sp_raw.strip() or not away_sp_raw.strip() or \
           home_sp_raw == "nan" or away_sp_raw == "nan":
            n_no_starter += 1
            continue

        # Normalize starter names
        home_sp_norm = _normalize_name(home_sp_raw)
        away_sp_norm = _normalize_name(away_sp_raw)

        # Look up pitcher profiles
        home_sp_row = pitcher_lookup.get(home_sp_norm)
        away_sp_row = pitcher_lookup.get(away_sp_norm)

        if home_sp_row is None or away_sp_row is None:
            n_no_pitcher += 1
            if verbose and n_no_pitcher <= 10:
                miss = []
                if home_sp_row is None:
                    miss.append(f"home={home_sp_norm!r}")
                if away_sp_row is None:
                    miss.append(f"away={away_sp_norm!r}")
                print(f"    [MISS] {date_str} {away}@{home}: pitcher not found: {', '.join(miss)}")
            continue

        # Look up odds
        odds_key = f"{home}|{away}|{date_str}"
        odds_row = odds_dict.get(odds_key, {})
        close_ml_home = odds_row.get("close_ml_home")
        close_ml_away = odds_row.get("close_ml_away")
        close_total   = odds_row.get("close_total")
        if close_ml_home is None or (isinstance(close_ml_home, float) and np.isnan(close_ml_home)):
            n_no_odds += 1
            # Still process but no Vegas edge calc

        # Team stats
        home_ts = team_stats.loc[home] if (team_stats is not None and home in team_stats.index) else None
        away_ts = team_stats.loc[away] if (team_stats is not None and away in team_stats.index) else None

        # Compute signal
        sig_info = compute_signal(home_sp_row, away_sp_row, close_ml_home, home_ts, away_ts)
        signal   = sig_info["signal"]

        n_processed += 1

        # Determine actual outcomes
        home_score = pd.to_numeric(game.get("home_score"), errors="coerce")
        away_score = pd.to_numeric(game.get("away_score"), errors="coerce")

        if pd.isna(home_score) or pd.isna(away_score):
            # Try from backtest fields
            actual_home_win = pd.to_numeric(game.get("actual_home_win"), errors="coerce")
            actual_total    = pd.to_numeric(game.get("actual_game_total"), errors="coerce")
            home_score = np.nan
            away_score = np.nan
        else:
            actual_home_win = float(home_score > away_score)
            actual_total    = float(home_score + away_score)

        # Run line cover
        if not pd.isna(home_score) and not pd.isna(away_score):
            home_covers_rl = int((home_score - away_score) >= 2)
        else:
            home_covers_rl = np.nan

        # Bet win
        if signal.startswith("HOME") and not pd.isna(home_covers_rl):
            bet_win = int(home_covers_rl == 1)
        elif signal.startswith("AWAY") and not pd.isna(home_covers_rl):
            bet_win = int(home_covers_rl == 0)
        else:
            bet_win = np.nan

        results.append({
            "date":          date_str,
            "game":          f"{away} @ {home}",
            "home_team":     home,
            "away_team":     away,
            "home_sp":       home_sp_norm,
            "away_sp":       away_sp_norm,
            "home_sp_xwoba": sig_info["home_sp_xwoba"],
            "away_sp_xwoba": sig_info["away_sp_xwoba"],
            "blended_rl":    sig_info["blended_rl"],
            "signal":        signal,
            "home_covers_rl":home_covers_rl,
            "bet_win":       bet_win,
            "actual_total":  actual_total,
            "actual_home_win": actual_home_win,
            "vegas_ml_home": close_ml_home,
            "vegas_ml_away": close_ml_away,
            "vegas_total":   close_total,
            "home_score":    home_score,
            "away_score":    away_score,
            "edge":          sig_info["edge"],
            "vegas_implied": sig_info["vegas_implied"],
            "season":        year,
        })

    df = pd.DataFrame(results)

    print(f"\n  Processing summary:")
    print(f"    Total games:       {len(games)}")
    print(f"    Missing starters:  {n_no_starter}")
    print(f"    Pitcher not found: {n_no_pitcher}")
    print(f"    No odds:           {n_no_odds}")
    print(f"    Processed:         {n_processed}")

    if df.empty:
        print("  [WARN] No results generated for this year")
        return df

    # ── Summary ──────────────────────────────────────────────────────────────
    bets = df[(df["signal"] != "") & df["bet_win"].notna()].copy()
    wins    = int(bets["bet_win"].sum())
    losses  = len(bets) - wins
    win_rate = wins / len(bets) if len(bets) > 0 else np.nan
    # Simple ROI at -110 juice: each win = +0.909 units, each loss = -1 unit
    roi = (wins * 0.909 - losses) / len(bets) if len(bets) > 0 else np.nan

    print(f"\n  {year} BACKTEST SUMMARY")
    print("  " + "-"*40)
    print(f"  Total games analyzed:  {n_processed}")
    print(f"  Games with signal:     {len(bets)}")
    print(f"  Record:                {wins}-{losses}")
    print(f"  Win rate:              {win_rate:.1%}" if not np.isnan(win_rate) else "  Win rate: N/A")
    print(f"  ROI (at -110):         {roi:.1%}" if not np.isnan(roi) else "  ROI: N/A")

    # ── Save ────────────────────────────────────────────────────────────────
    out_path = BASE_DIR / f"backtest_{year}_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}  ({len(df)} rows, {len(bets)} bets)")

    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(years=None):
    years = years or [2023, 2024, 2025]
    print("=" * 60)
    print("  backtest_historical.py")
    print("=" * 60)

    all_results = {}
    for year in years:
        try:
            df = run_year(year, verbose=True)
            all_results[year] = df
        except Exception as e:
            print(f"\n  [ERROR] {year} backtest failed: {e}")
            import traceback
            traceback.print_exc()

    # ── Multi-year summary ────────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("  MULTI-YEAR SUMMARY")
    print("="*60)
    print(f"  {'Season':<8} {'Bets':>6} {'Record':>10} {'Win%':>8} {'ROI':>8}")
    print("  " + "-"*50)

    for year, df in all_results.items():
        if df.empty:
            print(f"  {year:<8} {'NO DATA':>6}")
            continue
        bets = df[(df["signal"] != "") & df["bet_win"].notna()]
        wins = int(bets["bet_win"].sum()) if len(bets) > 0 else 0
        losses = len(bets) - wins
        wr = wins / len(bets) if len(bets) > 0 else 0
        roi = (wins * 0.909 - losses) / len(bets) if len(bets) > 0 else 0
        print(f"  {year:<8} {len(bets):>6} {wins:>4}-{losses:<5} {wr:>8.1%} {roi:>8.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    args = parser.parse_args()
    main(years=args.years)
