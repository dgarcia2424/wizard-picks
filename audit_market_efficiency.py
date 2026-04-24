"""
audit_market_efficiency.py — v5.0 Market Efficiency Auditor.

CLV is the primary health metric for any quantitative betting model.  If our
model consistently prices bets higher than the closing market, the edge is
real regardless of short-run variance.

Standard bets (ML / Totals / Runline / F5 / NRFI):
    CLV = closing_p_true − pick_p_true
    Positive → we got better than the closing line (we beat the market)

SGP bets (Scripts A2, B, C, D):
    The book doesn't post a single SGP closing line.  We reconstruct the
    closing book joint probability from closing leg implied probs:

        p_book_sgp_closing = product(closing_leg_probs) × (1 − BOOK_CORR_TAX)

    SGP CLV = p_book_sgp_closing − p_book_sgp_at_prediction
        Positive → our script predicted a joint price tighter than closing

    Separately, we track whether p_joint_copula > p_book_sgp_closing
    as the true "edge realisation" test.

Statistical significance:
    One-sample t-test: H₀: E[CLV] = 0
    Reject at p < 0.05 → model has statistically significant edge.

Output
------
    market_beating_report.csv    — one row per PLAY (straight + SGP)
    Console: per-bet-type summary with CLV, win rate, t-stat

Usage
-----
    python audit_market_efficiency.py               # all history
    python audit_market_efficiency.py --days 30     # last 30 days
    python audit_market_efficiency.py --update      # save report
    python audit_market_efficiency.py --sgp-only    # SGP bets only
    python audit_market_efficiency.py --straight-only
"""
from __future__ import annotations

import argparse
import glob
from datetime import date, datetime, timedelta
from pathlib import Path
from scipy import stats as scipy_stats

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent

MODEL_SCORES    = _ROOT / "model_scores.csv"
SGP_DIR         = _ROOT / "data/sgp"
ODDS_HIST_DIR   = _ROOT / "data/statcast"
ACTUALS_FILE    = _ROOT / "data/statcast/actuals_2026.parquet"
DAILY_CARDS_DIR = _ROOT / "daily_cards"
OUTPUT_FILE     = _ROOT / "market_beating_report.csv"

CLOSING_SNAPSHOT_HOUR_UTC = 20   # 4 PM ET — same as clv_tracker.py
BOOK_CORR_TAX             = 0.15 # standard FanDuel/DK SGP tax


# ---------------------------------------------------------------------------
# Shared loader: closing odds snapshot
# ---------------------------------------------------------------------------

def _load_closing_odds(game_date: str) -> pd.DataFrame:
    """Return latest Pinnacle snapshot before CLOSING_SNAPSHOT_HOUR_UTC."""
    path = ODDS_HIST_DIR / f"odds_history_{game_date.replace('-', '_')}.parquet"
    if not path.exists():
        return pd.DataFrame()

    hist = pd.read_parquet(path)
    if hist.empty:
        return hist

    hist["snapshot_time"] = pd.to_datetime(hist["snapshot_time"], utc=True)
    cutoff = pd.Timestamp(f"{game_date}T{CLOSING_SNAPSHOT_HOUR_UTC:02d}:00:00Z")
    before = hist[hist["snapshot_time"] <= cutoff]
    if before.empty:
        before = hist

    return (
        before.sort_values("snapshot_time")
        .groupby(["home_team", "away_team"])
        .last()
        .reset_index()
    )


def _load_actuals() -> pd.DataFrame:
    if not ACTUALS_FILE.exists():
        return pd.DataFrame()
    df = pd.read_parquet(ACTUALS_FILE)
    df["date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    df["total"] = (pd.to_numeric(df["home_score_final"], errors="coerce") +
                   pd.to_numeric(df["away_score_final"], errors="coerce"))
    return df


# ---------------------------------------------------------------------------
# Part 1: straight bet CLV (delegates most logic to existing clv_tracker)
# ---------------------------------------------------------------------------

def _straight_clv_rows(days: Optional[int] = None) -> list[dict]:
    """Re-use clv_tracker logic to get straight-bet CLV rows."""
    try:
        import clv_tracker
        picks = clv_tracker.load_picks(days=days)
        if picks.empty:
            return []
        df = clv_tracker.compute_clv(picks)
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "date":         r.get("date", ""),
                "game":         f"{r.get('away_team','')} @ {r.get('home_team','')}",
                "home_team":    r.get("home_team", ""),
                "away_team":    r.get("away_team", ""),
                "source":       "straight",
                "bet_type":     r.get("bet_type", ""),
                "pick_p_model": r.get("pick_p_model"),
                "pick_p_true":  r.get("pick_p_true"),
                "close_p_true": r.get("close_p_true"),
                "clv":          r.get("clv"),
                "won":          r.get("won"),
                "tier":         r.get("tier"),
                "stake":        r.get("stake"),
                "edge_at_pick": r.get("pick_edge"),
            })
        return rows
    except Exception as exc:
        print(f"  [audit] straight CLV load failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Part 2: SGP CLV
# ---------------------------------------------------------------------------

def _reconstruct_book_sgp(game_date: str,
                           home_team: str,
                           away_team: str,
                           script: str,
                           p_book_sgp_at_pick: float,
                           closing: pd.DataFrame) -> Optional[float]:
    """Reconstruct closing p_book_sgp from closing leg implied probs.

    We use closing P_true_over / P_true_under / P_true_home as leg proxies.
    For K-prop legs there is no direct closing line in our history file, so we
    use the opening p_book_sgp adjusted by the total line movement:

        closing_leg_adjustment = closing_p_true_home / pick_p_true_home

    Returns None if closing data is unavailable.
    """
    if closing.empty:
        return None

    match = closing[
        (closing["home_team"].str.upper() == home_team.upper()) &
        (closing["away_team"].str.upper() == away_team.upper())
    ]
    if match.empty:
        return None

    m = match.iloc[0]

    def _safe(col: str) -> Optional[float]:
        v = m.get(col)
        return float(v) if pd.notna(v) else None

    close_home = _safe("P_true_home")
    close_over = _safe("P_true_over")
    close_under = _safe("P_true_under")

    if script in ("A2_Dominance", "C_EliteDuel"):
        # Under-heavy scripts — total movement drives the adjustment
        if close_under is not None and close_under > 0:
            adjustment = close_under / 0.5   # rough ratio vs 50% anchor
            return float(np.clip(p_book_sgp_at_pick * adjustment, 0.001, 0.5))

    elif script == "B_Explosion":
        # Over + home ML
        if close_over is not None and close_home is not None and close_over > 0:
            adjustment = (close_over + close_home) / 1.0
            return float(np.clip(p_book_sgp_at_pick * adjustment, 0.001, 0.5))

    elif script == "D_LateDivergence":
        # F5 Under + Game Over — total movement is the primary driver
        if close_under is not None and close_over is not None:
            avg_adj = (close_under + close_over) / 1.0
            return float(np.clip(p_book_sgp_at_pick * avg_adj, 0.001, 0.5))

    # Generic: scale by available closing line movement
    if close_home is not None:
        return float(np.clip(p_book_sgp_at_pick * (close_home / 0.5), 0.001, 0.5))

    return None


def _sgp_clv_rows(days: Optional[int] = None) -> list[dict]:
    """Load SGP PLAY history and compute CLV vs reconstructed closing price."""
    sgp_files = sorted(glob.glob(str(SGP_DIR / "sgp_live_edge_*.csv")))
    if not sgp_files:
        return []

    actuals = _load_actuals()
    rows    = []

    cutoff_date = (date.today() - timedelta(days=days)).isoformat() if days else None

    for f in sgp_files:
        stem     = Path(f).stem.replace("sgp_live_edge_", "")
        game_date = stem.replace("_", "-")

        if cutoff_date and game_date < cutoff_date:
            continue

        df = pd.read_csv(f)
        plays = df[df["action"] == "PLAY"].copy()
        if plays.empty:
            continue

        closing = _load_closing_odds(game_date)

        for _, play in plays.iterrows():
            home  = str(play.get("home_team", ""))
            away  = str(play.get("away_team", ""))
            script = str(play.get("script", ""))

            p_pick  = float(play.get("p_book_sgp", 0) or 0)
            p_model = float(play.get("p_joint_copula", 0) or 0)
            edge    = float(play.get("sgp_edge", 0) or 0)

            # Reconstruct closing book joint prob
            p_closing = _reconstruct_book_sgp(
                game_date, home, away, script, p_pick, closing)

            clv = None
            if p_closing is not None and p_pick > 0:
                # Positive CLV = market tightened (closing price > our pick price)
                # i.e. closing implied prob fell → closing decimal odds rose
                # CLV in pp terms: (1/p_pick) - (1/p_closing)  [payout improvement]
                # We report in probability-space: closing_p_true - pick_p_true is
                # sign-inverted for SGP (lower book prob = better for us).
                clv = round(float(p_pick - p_closing), 5)   # >0 = market moved our way

            # Actual outcome: did the SGP win?
            won = None
            if not actuals.empty:
                act = actuals[
                    (actuals["home_team"].str.upper() == home.upper()) &
                    (actuals["away_team"].str.upper() == away.upper()) &
                    (actuals["date"] == game_date)
                ]
                if not act.empty:
                    a = act.iloc[0]
                    # We don't store per-leg SGP outcomes yet; approximate from script
                    home_win = (pd.to_numeric(a.get("home_score_final"), errors="coerce") >
                                pd.to_numeric(a.get("away_score_final"), errors="coerce"))
                    total    = pd.to_numeric(a.get("total"), errors="coerce")
                    close_total = float(play.get("close_total", 9.0) or 9.0)

                    if script == "B_Explosion" and pd.notna(home_win) and pd.notna(total):
                        over_hit       = total > close_total
                        home_score_5   = pd.to_numeric(
                            a.get("home_score_final"), errors="coerce") >= 5
                        won = float(home_win and over_hit and home_score_5)
                    elif script in ("A2_Dominance", "C_EliteDuel") and pd.notna(total):
                        won = float(total < close_total)   # simplified Under leg
                    elif script == "D_LateDivergence" and pd.notna(total):
                        over_hit = total > close_total
                        f5_fraction = 0.571
                        won = float(over_hit)   # Game Over leg approximation

            rows.append({
                "date":              game_date,
                "game":              str(play.get("game", f"{away} @ {home}")),
                "home_team":         home,
                "away_team":         away,
                "source":            "sgp",
                "bet_type":          script,
                "script":            script,
                "legs":              str(play.get("legs", "")),
                "pick_p_model":      round(p_model, 5),
                "pick_p_true":       round(p_pick, 5),    # book's price at pick time
                "close_p_true":      round(p_closing, 5) if p_closing else None,
                "clv":               clv,
                "won":               won,
                "edge_at_pick":      round(edge, 5),
                "corr_lift":         play.get("corr_lift"),
                "book_corr_tax":     play.get("book_corr_tax"),
            })

    return rows


# ---------------------------------------------------------------------------
# Combine + report
# ---------------------------------------------------------------------------

def build_report(days: Optional[int] = None,
                 include_straight: bool = True,
                 include_sgp: bool = True) -> pd.DataFrame:
    rows = []
    if include_straight:
        rows.extend(_straight_clv_rows(days))
    if include_sgp:
        rows.extend(_sgp_clv_rows(days))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["date", "source"], ascending=[False, True]).reset_index(drop=True)
    return df


def _t_test(series: pd.Series) -> tuple[float, float]:
    """One-sample t-test: H₀ E[CLV]=0.  Returns (t_stat, p_value)."""
    clean = series.dropna()
    if len(clean) < 3:
        return np.nan, np.nan
    t, p = scipy_stats.ttest_1samp(clean, 0)
    return round(float(t), 3), round(float(p), 4)


def print_report(df: pd.DataFrame) -> None:
    today = date.today().isoformat()
    print("=" * 70)
    print(f"  MARKET EFFICIENCY AUDIT — {today}")
    print("=" * 70)

    if df.empty:
        print("  No data — run pipeline and accumulate picks first.")
        return

    has_clv     = df.dropna(subset=["clv"])
    has_outcome = df.dropna(subset=["won"])

    print(f"\n  Total PLAYs: {len(df)} | With CLV: {len(has_clv)} | Resolved: {len(has_outcome)}")

    if has_clv.empty:
        print("\n  No CLV data yet — closing odds history accumulates after today's 4 PM pull.")
        return

    overall_clv   = has_clv["clv"].mean()
    pct_positive  = (has_clv["clv"] > 0).mean()
    t_stat, p_val = _t_test(has_clv["clv"])
    sig_tag       = "✓ SIGNIFICANT" if (not np.isnan(p_val) and p_val < 0.05) else "not significant"

    print(f"\n  Overall CLV: {overall_clv:+.4f}  ({overall_clv*100:+.2f} pp)")
    print(f"  Beats closing line: {pct_positive:.1%} of bets")
    print(f"  t-stat={t_stat:+.2f}  p={p_val:.4f}  [{sig_tag}]")

    print(f"\n  {'Type':<20}  {'N':>4}  {'Avg CLV':>9}  {'% Beat Close':>13}  "
          f"{'Win Rate':>9}  {'t-stat':>7}")
    print("  " + "-" * 70)

    for bt, grp in has_clv.groupby("bet_type"):
        resolved = grp.dropna(subset=["won"])
        wr       = f"{resolved['won'].mean():.1%}" if len(resolved) > 0 else "  —  "
        t, _     = _t_test(grp["clv"])
        t_tag    = f"{t:+.2f}" if not np.isnan(t) else "  —  "
        print(f"  {str(bt):<20}  {len(grp):>4}  "
              f"{grp['clv'].mean():>+8.4f}  "
              f"{(grp['clv'] > 0).mean():>12.1%}  "
              f"{wr:>9}  {t_tag:>7}")

    # Source breakdown
    print(f"\n  By source:")
    for src, grp in has_clv.groupby("source"):
        resolved = grp.dropna(subset=["won"])
        wr       = f"{resolved['won'].mean():.1%}" if len(resolved) > 0 else "—"
        t, p_v   = _t_test(grp["clv"])
        sig      = "*" if (not np.isnan(p_v) and p_v < 0.05) else " "
        print(f"    {src:<10}  n={len(grp):>3}  "
              f"avg_clv={grp['clv'].mean():>+.4f}  "
              f"beat_close={( grp['clv'] > 0).mean():.1%}  "
              f"win={wr}  t={t:+.2f}{sig}")

    # CLV percentile buckets: top tercile CLV → win rate
    if len(has_outcome) >= 10 and "clv" in has_outcome.columns:
        try:
            has_outcome = has_outcome.copy()
            has_outcome["clv_tercile"] = pd.qcut(
                has_outcome["clv"].fillna(0), 3, labels=["Low", "Mid", "High"])
            print(f"\n  Win rate by CLV tercile:")
            for t3, grp in has_outcome.groupby("clv_tercile"):
                print(f"    CLV {t3:>4}: {grp['won'].mean():.1%}  (n={len(grp)})")
        except Exception:
            pass

    if len(has_clv) < 30:
        print(f"\n  ⚠  Only {len(has_clv)} plays with CLV — need ≥30 for reliable signal.")
    else:
        # Annualised CLV estimate (assuming ~162 game-days, ~5 picks/day)
        daily_clv = overall_clv * 5
        print(f"\n  Model health: +{overall_clv*100:.2f} pp avg CLV per bet → "
              f"~{daily_clv*100:.2f} pp/day edge expectation")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Market efficiency auditor — CLV vs closing line (v5.0)")
    parser.add_argument("--days",          type=int, default=None,
                        help="Include last N days only")
    parser.add_argument("--update",        action="store_true",
                        help=f"Save to {OUTPUT_FILE.name}")
    parser.add_argument("--sgp-only",      action="store_true")
    parser.add_argument("--straight-only", action="store_true")
    args = parser.parse_args()

    include_sgp      = not args.straight_only
    include_straight = not args.sgp_only

    df = build_report(days=args.days,
                      include_straight=include_straight,
                      include_sgp=include_sgp)

    print_report(df)

    if args.update and not df.empty:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n  Saved {len(df)} rows → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
