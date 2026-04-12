"""
calibrate_edge.py — Empirically fit edge_to_prob() and HOME_FIELD_BONUS
                    from historical game outcomes.

Uses the 5,000+ game backtest to answer:
  1. What is the actual win rate at each score-edge bucket?
  2. What sigmoid parameters best fit the empirical curve?
  3. Is HOME_FIELD_BONUS (currently 8.0 pts) correctly sized?
  4. Is NEUTRAL_EDGE_THRESHOLD (currently 20 pts) correctly placed?

Outputs:
  - calibration_edge_results.csv   (edge buckets vs actual win rates)
  - Recommended parameter updates for score_models.py

Usage:
    python calibrate_edge.py
    python calibrate_edge.py --no-home-bonus   # test without home field
"""

import argparse
import io
import contextlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
import score_models as sm

BACKTEST_FILES = {
    2024: "backtest_games_2024.csv",
    2025: "backtest_games_2025.csv",
}
SAVANT_FILE    = "savant_pitchers.csv"
FANGRAPHS_FILE = "fangraphs_pitchers.csv"

LEAGUE_AVG_ROW = {
    "k_pct": 22.0, "bb_pct": 8.5, "hard_hit_pct": 38.0,
    "exit_velo": 87.5, "whiff_pct": 25.0, "xwoba": 0.315,
    "xera": 4.20, "stuff_plus": 100.0, "fastball_velo": 93.5,
    "k9": 8.5, "bb9": 3.3, "siera": 4.20, "fip": 4.20,
    "k_bb_pct": 10.0, "pa_current": 0, "pa_scale": 0.0,
}


def _suppress(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


def lookup_pitcher(pitcher_df, name):
    row = pitcher_df[pitcher_df["pitcher_name"] == str(name).upper().strip()]
    if row.empty:
        return {**LEAGUE_AVG_ROW, "pitcher_name": name, "_fallback": True}
    r = row.iloc[0].to_dict()
    r["_fallback"] = False
    return r


def calc_pitcher_score(row):
    """Replicate score_models._calc_stuff_proxy() output."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return sm._calc_stuff_proxy(row)


def collect_edges(games_df, pitcher_df, include_home_bonus: bool = True):
    """
    Score every game and collect (raw_edge, actual_home_win) pairs.
    raw_edge = home_pitcher_score - away_pitcher_score
    (before home field bonus if include_home_bonus=False)
    """
    records = []
    for _, game in games_df.iterrows():
        game = game.to_dict()
        home_sp = str(game.get("home_starter", "")).upper().strip()
        away_sp = str(game.get("away_starter", "")).upper().strip()
        if not home_sp or not away_sp or home_sp in ("", "NAN") or away_sp in ("", "NAN"):
            continue

        home_row = lookup_pitcher(pitcher_df, home_sp)
        away_row = lookup_pitcher(pitcher_df, away_sp)

        # Skip double-fallback (no signal)
        if home_row.get("_fallback") and away_row.get("_fallback"):
            continue

        actual = game.get("actual_home_win")
        if actual is None or str(actual) == "nan":
            continue

        home_score = calc_pitcher_score(home_row)
        away_score = calc_pitcher_score(away_row)

        if include_home_bonus:
            home_score += sm.HOME_FIELD_BONUS

        edge = home_score - away_score

        records.append({
            "edge":            edge,
            "actual_home_win": bool(actual),
            "home_fallback":   home_row.get("_fallback", False),
            "away_fallback":   away_row.get("_fallback", False),
        })

    return pd.DataFrame(records)


# ─── SIGMOID FITTING ─────────────────────────────────────────────────────────

def sigmoid_model(edge_abs, base, scale, decay):
    """Model: base + scale * (1 - exp(-edge_abs / decay))"""
    return base + scale * (1 - np.exp(-edge_abs / decay))


def fit_sigmoid(edges, outcomes):
    """
    Fit sigmoid to (edge, outcome) pairs.
    edge = raw edge (can be negative for away-favored games).
    We convert to the model's perspective: positive edge = predicted win.

    Returns (base, scale, decay, r_squared)
    """
    # Only use games where the model has a clear lean
    mask  = np.abs(edges) > sm.NEUTRAL_EDGE_THRESHOLD
    e_abs = np.abs(edges[mask]) - sm.NEUTRAL_EDGE_THRESHOLD
    # Flip outcome for away-favored games so we measure "predicted team wins"
    wins  = np.where(edges[mask] > 0, outcomes[mask], 1.0 - outcomes[mask]).astype(float)

    if len(wins) < 50:
        return None

    try:
        p0 = [0.591, 0.40, 118.8]          # warm-start from last calibration
        # Bounds expanded — previous fit hit scale=0.40/decay=150 ceiling.
        # Decay lower bound raised to 20 to avoid degenerate instant-saturation fits.
        bounds = ([0.50, 0.05, 20], [0.80, 0.60, 300])
        popt, _ = curve_fit(sigmoid_model, e_abs, wins, p0=p0, bounds=bounds, maxfev=5000)
        y_pred   = sigmoid_model(e_abs, *popt)
        ss_res   = np.sum((wins - y_pred) ** 2)
        ss_tot   = np.sum((wins - wins.mean()) ** 2)
        r2       = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Require minimum R² — with <300 above-threshold games the fit is noise
        if r2 < 0.05:
            print(f"  Low R2={r2:.3f} -- fit unreliable; keeping current parameters")
            return None

        return (*popt, r2)
    except Exception as e:
        print(f"  Sigmoid fit failed: {e}")
        return None


# ─── EDGE BUCKET ANALYSIS ────────────────────────────────────────────────────

def bucket_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group games by edge bucket and compute actual win rates.
    Shows what probability the model should assign at each edge level.
    """
    bins   = [-200, -80, -50, -30, -20, -10, 0, 10, 20, 30, 50, 80, 200]
    labels = ["<-80", "-80:-50", "-50:-30", "-30:-20", "-20:-10",
              "-10:0", "0:10", "10:20", "20:30", "30:50", "50:80", ">80"]

    df2 = df.copy()
    df2["bucket"] = pd.cut(df2["edge"], bins=bins, labels=labels)

    rows = []
    for label in labels:
        bucket_df = df2[df2["bucket"] == label]
        n = len(bucket_df)
        if n == 0:
            continue
        home_win_rate = bucket_df["actual_home_win"].mean()
        edge_mean     = bucket_df["edge"].mean()
        rows.append({
            "edge_bucket":    label,
            "n_games":        n,
            "edge_mean":      round(edge_mean, 1),
            "home_win_rate":  round(home_win_rate, 3),
            "model_prob":     round(sm.edge_to_prob(max(0, abs(edge_mean) - sm.NEUTRAL_EDGE_THRESHOLD)) + sm.HOME_FIELD_PROB_BUMP if abs(edge_mean) > sm.NEUTRAL_EDGE_THRESHOLD else 0.52, 3),
        })

    return pd.DataFrame(rows)


# ─── HOME FIELD BONUS CALIBRATION ────────────────────────────────────────────

def calibrate_home_bonus(games_df, pitcher_df):
    """
    Find the HOME_FIELD_BONUS value that maximizes accuracy by testing
    different bonus sizes and measuring overall directional accuracy.
    """
    bonuses = [0, 4, 6, 8, 10, 12, 15, 20]
    print(f"\n  Home Field Bonus calibration:")
    print(f"  {'bonus':>6}  {'n_picks':>8}  {'accuracy':>9}")
    print(f"  {'-'*30}")

    original_bonus = sm.HOME_FIELD_BONUS   # capture before grid modifies it
    best_bonus     = original_bonus
    best_acc       = 0
    rows = []

    for bonus in bonuses:
        sm.HOME_FIELD_BONUS = bonus
        df_edges = collect_edges(games_df, pitcher_df, include_home_bonus=True)

        # Only games where edge exceeds threshold
        picks = df_edges[np.abs(df_edges["edge"]) > sm.NEUTRAL_EDGE_THRESHOLD].copy()
        if len(picks) < 100:
            continue

        picks["predicted_home_win"] = picks["edge"] > 0
        acc = (picks["predicted_home_win"] == picks["actual_home_win"]).mean()
        n   = len(picks)

        rows.append({"bonus": bonus, "n_picks": n, "accuracy": round(acc, 4)})
        marker = " <-- current" if bonus == original_bonus else ""
        print(f"  {bonus:>6.0f}  {n:>8}  {acc:>9.1%}{marker}")

        if acc > best_acc:
            best_acc   = acc
            best_bonus = bonus

    # Restore original
    sm.HOME_FIELD_BONUS = best_bonus
    return best_bonus, original_bonus, pd.DataFrame(rows)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, help="Single backtest year")
    args = parser.parse_args()

    years = [args.year] if args.year else [2024, 2025]

    # Load games
    all_games = []
    for yr in years:
        path = BACKTEST_FILES.get(yr)
        if not path:
            continue
        try:
            df = pd.read_csv(path)
            all_games.append(df)
            print(f"  Loaded {len(df)} games from {path}")
        except FileNotFoundError:
            print(f"  {path} not found -- run build_backtest.py first")

    if not all_games:
        sys.exit(1)

    games_df = pd.concat(all_games, ignore_index=True)

    # Load pitchers with calibrated April-baseline weights (40% current year)
    blend_year = max(years)
    year_weights = {blend_year: 0.40, blend_year-1: 0.472, blend_year-2: 0.128}
    sm.YEAR_WEIGHTS   = year_weights
    sm.PA_FULL_SAMPLE = 150
    savant_df  = _suppress(sm.load_savant, SAVANT_FILE)
    fg_df      = _suppress(sm.load_fangraphs, FANGRAPHS_FILE)
    pitcher_df = _suppress(sm.merge_pitcher_data, savant_df, fg_df)
    print(f"  Pitcher data: {len(pitcher_df)} pitchers loaded")

    # ── Edge collection ───────────────────────────────────────────────────────
    print(f"\n  Collecting edges for {len(games_df)} games ...")
    df_edges = collect_edges(games_df, pitcher_df, include_home_bonus=True)
    print(f"  {len(df_edges)} scoreable games  "
          f"({len(df_edges[np.abs(df_edges['edge']) > sm.NEUTRAL_EDGE_THRESHOLD])} above threshold)")

    # ── Bucket analysis ───────────────────────────────────────────────────────
    print(f"\n  Edge bucket analysis:")
    print(f"  (+ edge = home favored, current HOME_FIELD_BONUS = {sm.HOME_FIELD_BONUS})")
    buckets = bucket_analysis(df_edges)
    print(f"\n  {'bucket':>12}  {'n':>6}  {'edge_mean':>10}  {'actual_hw%':>10}  {'model_prob':>10}")
    print(f"  {'-'*55}")
    for _, r in buckets.iterrows():
        flag = ""
        if abs(r["home_win_rate"] - r["model_prob"]) > 0.05:
            flag = " <-- miscalibrated"
        print(f"  {r['edge_bucket']:>12}  {r['n_games']:>6}  {r['edge_mean']:>10.1f}  "
              f"{r['home_win_rate']:>10.1%}  {r['model_prob']:>10.1%}{flag}")

    # ── Sigmoid fitting ───────────────────────────────────────────────────────
    print(f"\n  Fitting sigmoid to empirical data ...")
    edges   = df_edges["edge"].values
    outcomes = df_edges["actual_home_win"].values.astype(float)
    fit = fit_sigmoid(edges, outcomes)

    if fit:
        base, scale, decay, r2 = fit
        print(f"\n  Fitted sigmoid:  base={base:.3f}  scale={scale:.3f}  decay={decay:.1f}  R2={r2:.3f}")
        print(f"  Current sigmoid: base=0.591  scale=0.400  decay=118.8  (calibrated 2026-04-11)")
        print(f"\n  Implied probabilities (fitted vs current):")
        print(f"  {'edge_above_thresh':>18}  {'fitted':>8}  {'current':>8}  {'delta':>7}")
        for e in [5, 10, 20, 30, 50, 80]:
            fitted_p  = sigmoid_model(e, base, scale, decay)
            current_p = sm.edge_to_prob(e) + sm.HOME_FIELD_PROB_BUMP
            print(f"  {e:>18}  {fitted_p:>8.1%}  {current_p:>8.1%}  {fitted_p-current_p:>+7.1%}")
    else:
        base, scale, decay, r2 = 0.591, 0.400, 118.8, None
        print("  Not enough data to fit sigmoid (need 50+ above-threshold games)")

    # ── Home field bonus calibration ──────────────────────────────────────────
    best_bonus, original_bonus, bonus_df = calibrate_home_bonus(games_df, pitcher_df)

    # ── Save results ──────────────────────────────────────────────────────────
    buckets.to_csv("calibration_edge_results.csv", index=False)
    print(f"\n  Bucket results saved -> calibration_edge_results.csv")

    # ── Recommendations ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RECOMMENDED updates to score_models.py")
    print(f"{'='*60}")

    if fit:
        print(f"\n  def edge_to_prob(edge_abs):")
        print(f"      base  = {base:.3f}")
        print(f"      return base + ({scale:.3f} * (1 - np.exp(-edge_abs / {decay:.1f})))")
        print(f"      # R2 = {r2:.3f}  (fitted from {len(df_edges)} games)")
    else:
        print(f"\n  edge_to_prob(): insufficient data -- keep current parameters")

    print(f"\n  HOME_FIELD_BONUS = {best_bonus:.1f}  (was {original_bonus:.1f})")

    best_bucket = buckets[buckets["edge_bucket"] == "20:30"]
    if not best_bucket.empty:
        actual_at_threshold = best_bucket.iloc[0]["home_win_rate"]
        print(f"\n  Note: games in 20-30 edge bucket win at {actual_at_threshold:.1%} actual rate")
        if actual_at_threshold < 0.57:
            print(f"  Consider raising NEUTRAL_EDGE_THRESHOLD (currently {sm.NEUTRAL_EDGE_THRESHOLD})")


if __name__ == "__main__":
    main()
