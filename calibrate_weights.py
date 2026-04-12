"""
calibrate_weights.py - Optimize year-blend weights for pitchers AND batters.

Phase 1 - Pitcher weights (MF5i):
  Grid search year blends vs actual F5 home-win outcomes.
  No betting lines required.

Phase 2 - Batter weights (MFull / MF3i / MF1i):
  Only runs if backtest files include market_total_game column
  (added by build_lines_backtest.py).
  Grid searches batter year weights independently from pitcher weights,
  fixing pitcher weights at the Phase 1 optimum.

Usage:
    python calibrate_weights.py               # uses 2024 + 2025
    python calibrate_weights.py --year 2025   # single season
    python calibrate_weights.py --threshold 0.60
    python calibrate_weights.py --pitcher-only   # skip batter calibration
    python calibrate_weights.py --batter-only    # skip pitcher calibration
"""

import argparse
import io
import contextlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── PATHS ────────────────────────────────────────────────────────────────────

BACKTEST_FILES = {
    2024: "backtest_games_2024.csv",
    2025: "backtest_games_2025.csv",
}
SAVANT_FILE         = "savant_pitchers.csv"
FANGRAPHS_FILE      = "fangraphs_pitchers.csv"
SAVANT_BATTERS_FILE = "savant_batters.csv"
FG_BATTERS_FILE     = "fangraphs_batters.csv"

# ─── GRID SEARCH SPACE ────────────────────────────────────────────────────────

# Pitcher/batter current-year weight steps (5% increments)
CUR_YEAR_STEPS = [round(x, 2) for x in np.arange(0.10, 0.56, 0.05)]

DEFAULT_THRESHOLD = 0.60

# ─── IMPORT SCORING FUNCTIONS ─────────────────────────────────────────────────

sys.path.insert(0, ".")
import score_models as sm


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _suppress(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) with stdout suppressed."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


def _make_year_weights(blend_year, w_cur):
    remaining = 1.0 - w_cur
    w_prev = round(remaining * 0.786, 3)
    w_2yr  = round(remaining * 0.214, 3)
    total  = w_cur + w_prev + w_2yr
    return {
        blend_year:     round(w_cur  / total, 4),
        blend_year - 1: round(w_prev / total, 4),
        blend_year - 2: round(w_2yr  / total, 4),
    }


def blend_pitchers(blend_year, year_weights):
    sm.YEAR_WEIGHTS   = year_weights
    sm.PA_FULL_SAMPLE = 150
    savant_df = _suppress(sm.load_savant, SAVANT_FILE)
    fg_df     = _suppress(sm.load_fangraphs, FANGRAPHS_FILE)
    return _suppress(sm.merge_pitcher_data, savant_df, fg_df)


def blend_batters(blend_year, year_weights):
    sm.YEAR_WEIGHTS = year_weights
    sm.PA_FULL_BAT  = 300
    batter_df    = _suppress(sm.load_batters, SAVANT_BATTERS_FILE)
    fg_batter_df = _suppress(sm.load_fangraphs_batters, FG_BATTERS_FILE)
    return batter_df, fg_batter_df


LEAGUE_AVG_ROW = {
    "k_pct": 22.0, "bb_pct": 8.5, "hard_hit_pct": 38.0,
    "exit_velo": 87.5, "whiff_pct": 25.0, "xwoba": 0.315,
    "xera": 4.20, "stuff_plus": 100.0, "fastball_velo": 93.5,
    "k9": 8.5, "bb9": 3.3, "siera": 4.20, "fip": 4.20,
    "k_bb_pct": 10.0, "pa_current": 0, "pa_scale": 0.0,
}


def lookup_pitcher(pitcher_df, name):
    row = pitcher_df[pitcher_df["pitcher_name"] == str(name).upper().strip()]
    if row.empty:
        return {**LEAGUE_AVG_ROW, "pitcher_name": name, "_fallback": True}
    r = row.iloc[0].to_dict()
    r["_fallback"] = False
    return r


def accuracy(results):
    if not results:
        return 0.0
    correct = sum(1 for r in results if r["predicted_home_win"] == r["actual_home_win"])
    return correct / len(results)


def brier_score(results):
    if not results:
        return 1.0
    total = 0.0
    for r in results:
        outcome = 1.0 if r["actual_home_win"] else 0.0
        prob    = r["prob"] if r["predicted_home_win"] else 1.0 - r["prob"]
        total  += (prob - outcome) ** 2
    return total / len(results)


# ─── PHASE 1: PITCHER WEIGHTS via MF5i ───────────────────────────────────────

def score_season_mf5i(games_df, pitcher_df, threshold):
    results = []
    for _, game in games_df.iterrows():
        game = game.to_dict()
        home_sp = str(game.get("home_starter", "")).upper().strip()
        away_sp = str(game.get("away_starter", "")).upper().strip()
        if not home_sp or not away_sp or home_sp == "NAN" or away_sp == "NAN":
            continue

        home_row = lookup_pitcher(pitcher_df, home_sp)
        away_row = lookup_pitcher(pitcher_df, away_sp)
        if home_row.get("_fallback") and away_row.get("_fallback"):
            continue

        mf5    = sm.score_mf5i(home_row, away_row, game)
        prob   = mf5["probability"] / 100.0
        lean   = mf5["lean"]
        actual = game.get("actual_home_win")
        if actual is None or str(actual) == "nan":
            continue
        if prob < threshold:
            continue
        if lean.startswith("HOME"):
            pred = True
        elif lean.startswith("AWAY"):
            pred = False
        else:
            continue

        results.append({
            "predicted_home_win": pred,
            "actual_home_win":    bool(actual),
            "prob":               prob,
            "home_fallback":      home_row.get("_fallback", False),
            "away_fallback":      away_row.get("_fallback", False),
        })
    return results


def run_pitcher_grid(games_df, blend_year, threshold):
    print(f"\n  PHASE 1 - Pitcher weights (MF5i)")
    print(f"  {len(CUR_YEAR_STEPS)} combinations  |  threshold: {threshold:.0%}")
    print(f"\n  {'w_cur':>6}  {'w_prev':>6}  {'w_2yr':>6}  {'picks':>6}  {'acc':>7}  {'brier':>7}")
    print(f"  {'-'*55}")

    rows = []
    for w_cur in CUR_YEAR_STEPS:
        year_weights = _make_year_weights(blend_year, w_cur)
        pitcher_df   = blend_pitchers(blend_year, year_weights)
        results      = score_season_mf5i(games_df, pitcher_df, threshold)
        acc          = accuracy(results)
        brier        = brier_score(results)
        n            = len(results)
        remaining    = 1.0 - w_cur
        w_prev       = round(remaining * 0.786, 3)
        w_2yr        = round(remaining * 0.214, 3)

        rows.append({
            "w_cur": w_cur, "w_prev": w_prev, "w_2yr": w_2yr,
            "year_weights": year_weights, "n_picks": n,
            "accuracy": round(acc, 4), "brier": round(brier, 4),
        })
        marker = " <-- current" if abs(w_cur - 0.40) < 0.01 else ""
        print(f"  {w_cur:>6.0%}  {w_prev:>6.0%}  {w_2yr:>6.0%}  "
              f"{n:>6}  {acc:>7.1%}  {brier:>7.4f}{marker}")

    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)


# ─── PHASE 2: BATTER WEIGHTS via MFull / MF3i / MF1i ─────────────────────────

def score_season_model(games_df, pitcher_df, model_fn, market_col, threshold):
    """Generic scorer for any model that uses market_total_* and returns probability/lean."""
    results = []
    for _, game in games_df.iterrows():
        game = game.to_dict()
        home_sp = str(game.get("home_starter", "")).upper().strip()
        away_sp = str(game.get("away_starter", "")).upper().strip()
        if not home_sp or not away_sp or home_sp == "NAN" or away_sp == "NAN":
            continue

        # Require a market line for these models
        market_total = game.get(market_col)
        if market_total is None or str(market_total) == "nan":
            continue

        home_row = lookup_pitcher(pitcher_df, home_sp)
        away_row = lookup_pitcher(pitcher_df, away_sp)
        if home_row.get("_fallback") and away_row.get("_fallback"):
            continue

        result = _suppress(model_fn, home_row, away_row, game)
        prob   = result["probability"] / 100.0
        lean   = result["lean"]
        actual = game.get("actual_home_win")
        if actual is None or str(actual) == "nan":
            continue
        if prob < threshold:
            continue
        if lean.startswith("HOME"):
            pred = True
        elif lean.startswith("AWAY"):
            pred = False
        else:
            continue

        results.append({
            "predicted_home_win": pred,
            "actual_home_win":    bool(actual),
            "prob":               prob,
        })
    return results


def run_batter_grid(games_df, blend_year, pitcher_year_weights, threshold):
    """
    Fix pitcher weights at optimum. Grid search batter year weights only.
    Uses MFull (most batter-sensitive) as primary metric.
    Also scores MF3i and MF1i for reference.
    """
    # Check if lines are available
    has_lines = "market_total_game" in games_df.columns and games_df["market_total_game"].notna().sum() > 100
    if not has_lines:
        print(f"\n  PHASE 2 - Batter weights: SKIPPED (no market lines in backtest file)")
        print(f"  Run build_lines_backtest.py first, then re-run this script.")
        return None

    n_lines = games_df["market_total_game"].notna().sum()
    print(f"\n  PHASE 2 - Batter weights (MFull + MF3i + MF1i)")
    print(f"  {n_lines}/{len(games_df)} games have market lines  |  threshold: {threshold:.0%}")
    print(f"  Pitcher weights fixed at: {pitcher_year_weights}")

    # Fix pitcher blend at optimum
    sm.YEAR_WEIGHTS   = pitcher_year_weights
    sm.PA_FULL_SAMPLE = 150
    pitcher_df = blend_pitchers(blend_year, pitcher_year_weights)

    print(f"\n  {'w_cur':>6}  {'w_prev':>6}  {'w_2yr':>6}  {'MFull':>8}  {'MF3i':>8}  {'MF1i':>8}  {'picks':>6}")
    print(f"  {'-'*65}")

    rows = []
    for w_cur in CUR_YEAR_STEPS:
        bat_weights = _make_year_weights(blend_year, w_cur)

        # Re-blend batters only
        sm.YEAR_WEIGHTS = bat_weights
        sm.PA_FULL_BAT  = 300
        _suppress(sm.load_batters, SAVANT_BATTERS_FILE)
        _suppress(sm.load_fangraphs_batters, FG_BATTERS_FILE)

        # Score each model
        r_full = score_season_model(games_df, pitcher_df, sm.score_mfull,
                                    "market_total_f5", threshold)
        r_f3   = score_season_model(games_df, pitcher_df, sm.score_mf3i,
                                    "market_total_f3", threshold)
        r_f1   = score_season_model(games_df, pitcher_df, sm.score_mf1i,
                                    "market_total_f1", threshold)

        acc_full = accuracy(r_full)
        acc_f3   = accuracy(r_f3)
        acc_f1   = accuracy(r_f1)
        n_picks  = len(r_full)

        remaining = 1.0 - w_cur
        w_prev = round(remaining * 0.786, 3)
        w_2yr  = round(remaining * 0.214, 3)

        rows.append({
            "w_cur": w_cur, "w_prev": w_prev, "w_2yr": w_2yr,
            "bat_year_weights": bat_weights,
            "acc_mfull": round(acc_full, 4),
            "acc_mf3i":  round(acc_f3, 4),
            "acc_mf1i":  round(acc_f1, 4),
            "n_picks":   n_picks,
            "brier_mfull": round(brier_score(r_full), 4),
        })
        marker = " <-- current" if abs(w_cur - 0.40) < 0.01 else ""
        print(f"  {w_cur:>6.0%}  {w_prev:>6.0%}  {w_2yr:>6.0%}  "
              f"{acc_full:>8.1%}  {acc_f3:>8.1%}  {acc_f1:>8.1%}  "
              f"{n_picks:>6}{marker}")

    df = pd.DataFrame(rows).sort_values("acc_mfull", ascending=False)
    return df


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",         type=int,   help="Single backtest year")
    parser.add_argument("--threshold",    type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--pitcher-only", action="store_true")
    parser.add_argument("--batter-only",  action="store_true")
    args = parser.parse_args()

    years = [args.year] if args.year else [2024, 2025]
    threshold = args.threshold

    all_games = []
    for yr in years:
        path = BACKTEST_FILES.get(yr)
        if not path:
            continue
        try:
            df = pd.read_csv(path)
            df["_year"] = yr
            all_games.append(df)
            lines_col = "market_total_game"
            has = df[lines_col].notna().sum() if lines_col in df.columns else 0
            print(f"  Loaded {len(df)} games from {path}  ({has} with market lines)")
        except FileNotFoundError:
            print(f"  {path} not found - run build_backtest.py first")

    if not all_games:
        print("No backtest data found.")
        sys.exit(1)

    games_df  = pd.concat(all_games, ignore_index=True)
    blend_year = max(years)

    print(f"\n  Total: {len(games_df)} games  |  threshold: {threshold:.0%}")

    # ── Phase 1: Pitcher weights ──────────────────────────────────────────────
    pitcher_results = None
    best_pitcher_weights = _make_year_weights(blend_year, 0.40)  # April-baseline default

    if not args.batter_only:
        pitcher_results = run_pitcher_grid(games_df, blend_year, threshold)
        best_pitcher = pitcher_results.iloc[0]
        best_pitcher_weights = best_pitcher["year_weights"]

        cur_baseline = 0.40   # calibrated April baseline in score_models.py
        baseline = pitcher_results[pitcher_results["w_cur"].apply(lambda x: abs(x - cur_baseline) < 0.01)]
        baseline_acc = baseline.iloc[0]["accuracy"] if not baseline.empty else None

        print(f"\n{'='*60}")
        print(f"  PITCHER WEIGHTS - RESULTS")
        print(f"{'='*60}")
        print(f"  Best   : {best_pitcher['year_weights']}")
        print(f"  Acc    : {best_pitcher['accuracy']:.1%}  ({best_pitcher['n_picks']} picks)")
        print(f"  Brier  : {best_pitcher['brier']:.4f}")
        if baseline_acc:
            print(f"  vs {cur_baseline:.0%}  : {baseline_acc:.1%}  ({best_pitcher['accuracy'] - baseline_acc:+.1%})")

        print(f"\n  Top 5:")
        print(f"  {'w_cur':>6}  {'w_prev':>6}  {'w_2yr':>6}  {'picks':>6}  {'accuracy':>9}  {'brier':>7}")
        for _, row in pitcher_results.head(5).iterrows():
            print(f"  {row['w_cur']:>6.0%}  {row['w_prev']:>6.0%}  {row['w_2yr']:>6.0%}  "
                  f"{row['n_picks']:>6}  {row['accuracy']:>9.1%}  {row['brier']:>7.4f}")

        out_p = "calibration_pitcher_results.csv"
        pitcher_results.drop(columns=["year_weights"]).to_csv(out_p, index=False)
        print(f"\n  Saved -> {out_p}")

    # ── Phase 2: Batter weights ───────────────────────────────────────────────
    if not args.pitcher_only:
        batter_results = run_batter_grid(games_df, blend_year, best_pitcher_weights, threshold)

        if batter_results is not None and not batter_results.empty:
            best_bat = batter_results.iloc[0]

            print(f"\n{'='*60}")
            print(f"  BATTER WEIGHTS - RESULTS")
            print(f"{'='*60}")
            print(f"  Best   : {best_bat['bat_year_weights']}")
            print(f"  MFull  : {best_bat['acc_mfull']:.1%}  ({best_bat['n_picks']} picks)")
            print(f"  MF3i   : {best_bat['acc_mf3i']:.1%}")
            print(f"  MF1i   : {best_bat['acc_mf1i']:.1%}")

            print(f"\n  Top 5 (sorted by MFull accuracy):")
            print(f"  {'w_cur':>6}  {'w_prev':>6}  {'w_2yr':>6}  {'MFull':>8}  {'MF3i':>8}  {'MF1i':>8}")
            for _, row in batter_results.head(5).iterrows():
                print(f"  {row['w_cur']:>6.0%}  {row['w_prev']:>6.0%}  {row['w_2yr']:>6.0%}  "
                      f"{row['acc_mfull']:>8.1%}  {row['acc_mf3i']:>8.1%}  {row['acc_mf1i']:>8.1%}")

            out_b = "calibration_batter_results.csv"
            batter_results.drop(columns=["bat_year_weights"]).to_csv(out_b, index=False)
            print(f"\n  Saved -> {out_b}")

    # ── Recommended updates ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RECOMMENDED updates to score_models.py")
    print(f"{'='*60}")
    if pitcher_results is not None:
        yw = pitcher_results.iloc[0]["year_weights"]
        print(f"\n  Pitcher YEAR_WEIGHTS (get_season_year_weights April baseline):")
        for k in sorted(yw.keys(), reverse=True):
            print(f"    {k}: {yw[k]:.3f}")

    if not args.pitcher_only and batter_results is not None and not batter_results.empty:
        byw = batter_results.iloc[0]["bat_year_weights"]
        print(f"\n  Batter BAT_YEAR_WEIGHTS (new - April baseline):")
        for k in sorted(byw.keys(), reverse=True):
            print(f"    {k}: {byw[k]:.3f}")


if __name__ == "__main__":
    main()
