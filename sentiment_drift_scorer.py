"""
sentiment_drift_scorer.py — Public Betting Sentiment / Trap Line Detector (v1.0).

Identifies 'Trap Games' where:
  - The retail public is heavily betting one side (> TRAP_PUBLIC_THRESH)
  - The model's projection contradicts the public's position

Trap scenarios detected:
  1. OVER TRAP:   public_pct_over > 70% AND model says Under
                  (P_true_under > P_true_over, or model_total < market_total)
  2. UNDER TRAP:  public_pct_over < 30% AND model says Over
  3. ML TRAP:     public_pct_home > 75% AND model says Away (P_true_away > P_true_home)
  4. SHARP FADE:  Pinnacle total meaningfully different from retail total
                  (sharp book disagreeing with the crowd)

Sharp vs Public divergence:
  The pipeline stores Pinnacle (sharp) and retail book totals separately.
  When Pinnacle line != retail line by >= SHARP_LINE_GAP, the sharp book
  is fading the public — a classic reverse-line-movement signal.

Data sources
------------
  data/statcast/odds_current_{date}.parquet   — public_pct_over, public_pct_home,
    close_total, P_true_over, P_true_under, P_true_home, P_true_away,
    pinnacle_total_line, retail_implied_over

Output
------
  trap_line_alerts.csv           — all trap games for today
  data/live/trap_alerts_{date}.json  — JSON version for Command Center

Usage
-----
  python sentiment_drift_scorer.py
  python sentiment_drift_scorer.py --date 2026-04-24
  python sentiment_drift_scorer.py --thresh-over 0.65   # lower threshold
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT       = Path(__file__).resolve().parent
ODDS_DIR    = _ROOT / "data/statcast"
LIVE_DIR    = _ROOT / "data/live"
OUTPUT_CSV  = _ROOT / "trap_line_alerts.csv"

# Thresholds
TRAP_PUBLIC_OVER_MAX  = 0.70   # public > 70% on over = potential trap
TRAP_PUBLIC_OVER_MIN  = 0.30   # public < 30% on over = potential trap (other direction)
TRAP_PUBLIC_ML_MAX    = 0.75   # public > 75% on home ML = potential trap
SHARP_LINE_GAP        = 0.5    # Pinnacle vs retail total diff (sharp fade signal)
MODEL_EDGE_MIN        = 0.02   # minimum model edge to flag as actionable trap


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_odds(date_str: str) -> pd.DataFrame:
    """Load odds_current parquet for date_str; fall back to latest available."""
    tag  = date_str.replace("-", "_")
    path = ODDS_DIR / f"odds_current_{tag}.parquet"

    if not path.exists():
        import glob
        files = sorted(glob.glob(str(ODDS_DIR / "odds_current_*.parquet")))
        if not files:
            return pd.DataFrame()
        path = Path(files[-1])
        print(f"  [sentiment] odds file for {date_str} not found; using {path.name}")

    df = pd.read_parquet(path)
    return df


# ---------------------------------------------------------------------------
# Trap detection
# ---------------------------------------------------------------------------

def _safe_float(val) -> float | None:
    try:
        v = float(val)
        return None if (np.isnan(v) or np.isinf(v)) else v
    except (TypeError, ValueError):
        return None


def detect_traps(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """
    Run all trap detection rules.
    Returns DataFrame with one row per trap game found.
    """
    if df.empty:
        return pd.DataFrame()

    records = []

    for _, row in df.iterrows():
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        game = f"{away}@{home}"

        pub_over  = _safe_float(row.get("public_pct_over"))
        pub_home  = _safe_float(row.get("public_pct_home"))
        close_tot = _safe_float(row.get("close_total"))
        pin_total = _safe_float(row.get("pinnacle_total_line"))
        p_true_over  = _safe_float(row.get("P_true_over"))
        p_true_under = _safe_float(row.get("P_true_under"))
        p_true_home  = _safe_float(row.get("P_true_home"))
        p_true_away  = _safe_float(row.get("P_true_away"))
        ret_imp_over = _safe_float(row.get("retail_implied_over"))

        traps_for_game: list[dict] = []

        # --- Trap 1: OVER TRAP ---
        if (pub_over is not None and pub_over > TRAP_PUBLIC_OVER_MAX
                and p_true_over is not None and p_true_under is not None):
            if p_true_under > p_true_over:
                model_edge = p_true_under - (1 - (ret_imp_over or 0.5))
                if abs(model_edge) >= MODEL_EDGE_MIN:
                    traps_for_game.append({
                        "trap_type":      "OVER_TRAP",
                        "direction":      "UNDER",
                        "public_pct_over": round(pub_over, 3),
                        "model_prob_side": round(p_true_under, 4),
                        "retail_implied": round(ret_imp_over, 4) if ret_imp_over else None,
                        "model_edge":     round(model_edge, 4),
                        "description": (
                            f"{round(pub_over*100,0):.0f}% public on OVER "
                            f"but model favors UNDER "
                            f"(P_under={p_true_under:.3f} vs P_over={p_true_over:.3f})"
                        ),
                    })

        # --- Trap 2: UNDER TRAP (reverse) ---
        if (pub_over is not None and pub_over < TRAP_PUBLIC_OVER_MIN
                and p_true_over is not None and p_true_under is not None):
            if p_true_over > p_true_under:
                model_edge = p_true_over - (1 - (ret_imp_over or 0.5))
                if abs(model_edge) >= MODEL_EDGE_MIN:
                    traps_for_game.append({
                        "trap_type":      "UNDER_TRAP",
                        "direction":      "OVER",
                        "public_pct_over": round(pub_over, 3),
                        "model_prob_side": round(p_true_over, 4),
                        "retail_implied": round(ret_imp_over, 4) if ret_imp_over else None,
                        "model_edge":     round(model_edge, 4),
                        "description": (
                            f"{round((1-pub_over)*100,0):.0f}% public on UNDER "
                            f"but model favors OVER "
                            f"(P_over={p_true_over:.3f} vs P_under={p_true_under:.3f})"
                        ),
                    })

        # --- Trap 3: ML TRAP (public on home, model on away) ---
        if (pub_home is not None and pub_home > TRAP_PUBLIC_ML_MAX
                and p_true_home is not None and p_true_away is not None):
            if p_true_away > p_true_home:
                model_edge = p_true_away - 0.5   # vs coin flip
                if abs(model_edge) >= MODEL_EDGE_MIN:
                    traps_for_game.append({
                        "trap_type":       "ML_TRAP",
                        "direction":       f"AWAY ({away})",
                        "public_pct_home": round(pub_home, 3),
                        "model_prob_side": round(p_true_away, 4),
                        "retail_implied":  None,
                        "model_edge":      round(model_edge, 4),
                        "description": (
                            f"{round(pub_home*100,0):.0f}% public on HOME {home} "
                            f"but model favors AWAY {away} "
                            f"(P_away={p_true_away:.3f} vs P_home={p_true_home:.3f})"
                        ),
                    })

        # --- Trap 4: SHARP FADE (Pinnacle disagrees with retail total) ---
        if (pin_total is not None and close_tot is not None):
            line_diff = pin_total - close_tot
            if abs(line_diff) >= SHARP_LINE_GAP:
                fade_dir = "UNDER" if line_diff < 0 else "OVER"
                # Pinnacle lower = sharp books favor under; higher = favor over
                traps_for_game.append({
                    "trap_type":    "SHARP_FADE",
                    "direction":    fade_dir,
                    "pinnacle_total": round(pin_total, 1),
                    "retail_total":  round(close_tot, 1),
                    "line_diff":    round(line_diff, 2),
                    "model_edge":   round(abs(line_diff) * 0.01, 4),  # rough proxy
                    "description": (
                        f"Pinnacle ({pin_total}) vs retail ({close_tot}) = "
                        f"{line_diff:+.1f} run diff. "
                        f"Sharp money on {fade_dir}. Retail public fading sharps."
                    ),
                })

        for t in traps_for_game:
            records.append({
                "date":      date_str,
                "game":      game,
                "home_team": home,
                "away_team": away,
                "close_total": close_tot,
                "pinnacle_total": pin_total,
                **t,
            })

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values("model_edge", ascending=False, key=abs
                                    ).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_report(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("  TRAP LINE ALERTS")
    print("=" * 70)
    if df.empty:
        print("  No trap games detected today.")
        return

    for _, r in df.iterrows():
        print(f"\n  [{r['trap_type']}]  {r['game']}")
        print(f"    Side:  {r.get('direction', 'N/A')}")
        print(f"    Edge:  {r.get('model_edge', 0):+.4f}")
        print(f"    Note:  {r.get('description', '')}")


def save_outputs(df: pd.DataFrame, date_str: str) -> None:
    """Save trap_line_alerts.csv and JSON for the dashboard."""
    if df.empty:
        return

    # CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved -> {OUTPUT_CSV.name}")

    # JSON for dashboard
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    import json
    def _safe(v):
        if v is None:
            return None
        try:
            import math
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
        except Exception:
            pass
        if hasattr(v, 'item'):
            return v.item()
        return v

    records = [{k: _safe(v) for k, v in row.items()}
               for row in df.to_dict("records")]
    out = LIVE_DIR / f"trap_alerts_{date_str}.json"
    out.write_text(json.dumps(records, indent=2))
    print(f"  Saved -> {out.name}")


def load_trap_alerts(date_str: str) -> list[dict]:
    """Load trap alerts for the dashboard."""
    import json
    p = LIVE_DIR / f"trap_alerts_{date_str}.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Wind vector multiplier lookup (uses updated stadium_metadata.json)
# ---------------------------------------------------------------------------

_STADIUM_CACHE: dict | None = None


def _load_stadium_meta() -> dict:
    global _STADIUM_CACHE
    if _STADIUM_CACHE is None:
        meta_path = Path(__file__).resolve().parent / "config/stadium_metadata.json"
        if meta_path.exists():
            import json
            _STADIUM_CACHE = json.loads(meta_path.read_text()).get("parks", {})
        else:
            _STADIUM_CACHE = {}
    return _STADIUM_CACHE


def get_wind_vector_mult(home_team: str, wind_bearing: float,
                          wind_direction: str = "auto") -> float:
    """
    Return the TB wind multiplier for a given park and wind condition.

    wind_direction: "out" | "in" | "cross" | "auto"
    If "auto", determines direction from wind_bearing vs CF azimuth.
    """
    parks = _load_stadium_meta()
    park  = parks.get(home_team.upper(), {})

    tier = park.get("wind_sensitivity_tier", "medium")
    if tier == "dome":
        return 0.0

    mult_out   = float(park.get("wind_mult_out",   1.5))
    mult_in    = float(park.get("wind_mult_in",    1.0))
    mult_cross = float(park.get("wind_mult_cross", 1.1))

    if wind_direction != "auto":
        d = wind_direction.lower()
        if d == "out":
            return mult_out
        if d == "in":
            return mult_in
        return mult_cross

    cf_az = float(park.get("cf_azimuth_deg", 45))
    # Angular difference between wind bearing and CF azimuth
    diff = abs((wind_bearing - cf_az + 180) % 360 - 180)

    if diff <= 45:
        return mult_out       # wind behind batter, toward CF = OUT
    elif diff >= 135:
        return mult_in        # wind from CF = IN
    else:
        return mult_cross     # crosswind


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def score(date_str: str) -> pd.DataFrame:
    odds = load_odds(date_str)
    if odds.empty:
        print(f"  [sentiment] no odds data for {date_str}")
        return pd.DataFrame()

    traps = detect_traps(odds, date_str)
    print_report(traps)
    save_outputs(traps, date_str)

    over_traps  = (traps["trap_type"] == "OVER_TRAP").sum()  if not traps.empty else 0
    ml_traps    = (traps["trap_type"] == "ML_TRAP").sum()    if not traps.empty else 0
    sharp_fades = (traps["trap_type"] == "SHARP_FADE").sum() if not traps.empty else 0
    print(f"\n  Summary: {len(traps)} alerts | "
          f"Over traps: {over_traps} | ML traps: {ml_traps} | Sharp fades: {sharp_fades}")

    return traps


def main() -> None:
    global TRAP_PUBLIC_OVER_MAX

    parser = argparse.ArgumentParser(
        description="Public sentiment / trap line detector (v1.0)")
    parser.add_argument("--date",          default=date.today().isoformat())
    parser.add_argument("--thresh-over",   type=float, default=TRAP_PUBLIC_OVER_MAX,
                        help="Public over pct threshold for OVER_TRAP (default 0.70)")
    args = parser.parse_args()

    TRAP_PUBLIC_OVER_MAX = args.thresh_over

    score(date_str=args.date)


if __name__ == "__main__":
    main()
