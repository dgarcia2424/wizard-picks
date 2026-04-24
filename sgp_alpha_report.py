"""
DEPRECATED (v4.4) — sgp_alpha_report.py is superseded by fetch_live_odds.py.
fetch_live_odds.py is the single rigorous SGP scorer:
  - Gaussian copula correlation engine
  - NO_PROP integrity gate (requires matched retail K-prop line)
  - F5_FRACTION = 0.571 (empirical)
  - ABS-adjusted ump synergy
Do not run this script directly. It remains for reference only.

sgp_alpha_report.py — SGP Correlation Auditor & Action Report.

Formula:
    SGP_Edge = P(Joint_Model) / P(Joint_Market) - 1

P(Joint_Market) = product of individual leg implied probabilities
                  × (1 - BOOK_CORR_TAX)

where BOOK_CORR_TAX = 0.15 (books apply ~15% haircut to SGP payout
relative to independent parlay math).

When SGP_Edge > 0: model believes the joint event is more probable than
the book is paying for — structural alpha from mis-priced correlation.

Outputs:
    data/sgp/sgp_alpha_report.csv
    data/sgp/sgp_alpha_report.json

Usage:
    python sgp_alpha_report.py [--date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

_ROOT = Path(__file__).resolve().parent

# ── Model paths ─────────────────────────────────────────────────────────────
MODEL_A  = _ROOT / "models/script_a_v1.json"      # original 2-leg
MODEL_A2 = _ROOT / "models/script_a2_v1.json"     # 3-leg F5
MODEL_B  = _ROOT / "models/script_b_v1.json"
MODEL_C  = _ROOT / "models/script_c_v1.json"

# ── Data paths ───────────────────────────────────────────────────────────────
CONTEXT_FILE   = _ROOT / "data/orchestrator/daily_context.parquet"
FEATURE_MTX    = _ROOT / "feature_matrix_enriched_v2.parquet"
ODDS_DIR       = _ROOT / "data/statcast"
STADIUM_META   = _ROOT / "config/stadium_metadata.json"
OUT_DIR        = _ROOT / "data/sgp"

# ── Constants ─────────────────────────────────────────────────────────────────
BOOK_CORR_TAX     = 0.15   # conservative; some books use 20%
MIN_EDGE          = 0.05   # minimum SGP_Edge to surface as a pick
F5_FRACTION       = 0.56   # F5 total ≈ 56% of game total (empirical average)
LEAGUE_AVG_FF_VELO = 94.19

# Script feature sets (must match training)
A2_FEATURES = [
    "home_sp_k_pct", "home_sp_bb_pct", "home_sp_whiff_pctl", "home_sp_ff_velo",
    "home_sp_k_pct_10d", "home_sp_k_bb_ratio", "home_sp_1st_k_pct",
    "home_sp_xwoba_against", "home_sp_gb_pct", "home_sp_avg_ip",
    "home_sp_fatigue_signal", "home_sp_days_rest",
    "away_bat_k_vs_rhp", "away_bat_k_vs_lhp",
    "away_lineup_xwoba_vs_rhp", "away_lineup_xwoba_vs_lhp",
    "ump_k_above_avg", "home_catcher_framing_runs",
    "home_park_factor", "temp_f", "air_density_rho", "roof_closed_flag", "wind_mph",
    "close_total", "mc_f5_expected_total", "elo_diff",
    "home_bullpen_vulnerability", "away_bullpen_vulnerability",
]

B_FEATURES = [
    "home_park_factor", "temp_f", "air_density_rho", "roof_closed_flag", "wind_mph",
    "home_lineup_wrc_plus", "home_lineup_xwoba_vs_rhp", "home_lineup_xwoba_vs_lhp",
    "home_team_barrel_pct_15g", "home_team_woba_15g", "home_team_xwoba_off_15g",
    "away_sp_k_pct", "away_sp_xwoba_against", "away_sp_gb_pct",
    "away_sp_ff_velo", "away_sp_whiff_pctl", "away_sp_bb_pct",
    "away_bullpen_vulnerability", "away_bp_fatigue_72h",
    "close_total", "elo_diff",
    "home_pyth_win_pct_15g", "home_rolling_rd_15g",
    "ump_rpg_above_avg", "home_catcher_framing_runs",
]

C_FEATURES = [
    "home_sp_k_pct", "home_sp_whiff_pctl", "home_sp_ff_velo",
    "home_sp_k_pct_10d", "home_sp_xwoba_against", "home_sp_gb_pct",
    "away_sp_k_pct", "away_sp_whiff_pctl", "away_sp_ff_velo",
    "away_sp_k_pct_10d", "away_sp_xwoba_against", "away_sp_gb_pct",
    "sp_k_pct_diff", "sp_xwoba_diff", "sp_velo_diff",
    "elo_diff", "batting_matchup_edge",
    "home_park_factor", "temp_f", "air_density_rho", "roof_closed_flag", "wind_mph",
    "ump_k_above_avg", "ump_called_strike_above_avg",
    "home_catcher_framing_runs", "away_catcher_framing_runs",
    "close_total",
    "home_bullpen_vulnerability", "away_bullpen_vulnerability",
    "away_bat_k_vs_rhp", "away_bat_k_vs_lhp",
    "home_bat_k_vs_rhp", "home_bat_k_vs_lhp",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _american_to_implied(odds: float) -> float:
    """American odds → implied probability (raw, not de-vigged)."""
    if pd.isna(odds):
        return float("nan")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def _load_models() -> dict[str, xgb.Booster]:
    models = {}
    for name, path in [("A2", MODEL_A2), ("B", MODEL_B), ("C", MODEL_C)]:
        if path.exists():
            b = xgb.Booster()
            b.load_model(str(path))
            models[name] = b
        else:
            print(f"  [warn] Model {name} not found at {path} — skipping")
    return models


def _safe(val, default):
    """Return val coerced to the type of default, or default if val is NA/None/empty."""
    if val is None:
        return default
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        pass
    if str(val) in ("", "nan", "<NA>", "None"):
        return default
    try:
        return type(default)(val)
    except (TypeError, ValueError):
        return default


def _load_today_context(date_str: str) -> pd.DataFrame:
    if not CONTEXT_FILE.exists():
        raise SystemExit(f"[ERROR] {CONTEXT_FILE} not found — run data_orchestrator.py first")
    ctx = pd.read_parquet(CONTEXT_FILE)
    ctx = ctx[ctx["orchestrator_date"] == date_str].copy()
    if ctx.empty:
        raise SystemExit(f"[ERROR] No games in context for {date_str}")
    return ctx


def _load_fm_latest() -> pd.DataFrame:
    fm = pd.read_parquet(FEATURE_MTX)
    fm["game_date"] = pd.to_datetime(fm["game_date"])
    return fm.sort_values("game_date").groupby("home_team", as_index=False).last()


def _load_odds(date_str: str) -> pd.DataFrame:
    tag = date_str.replace("-", "_")
    path = ODDS_DIR / f"odds_current_{tag}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _load_k_props(date_str: str) -> pd.DataFrame:
    path = ODDS_DIR / f"k_props_{date_str}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _load_stadium_meta() -> dict:
    if STADIUM_META.exists():
        return json.loads(STADIUM_META.read_text())
    return {}


def _pull_side_wind_vector(wind_bearing: float, wind_mph: float,
                            cf_azimuth: float, stand: str = "R") -> float:
    """Physical wind advantage vs pull-side fly balls."""
    PULL_OFFSET = 22.0
    if stand == "L":
        pull_az = cf_azimuth + PULL_OFFSET
    else:
        pull_az = cf_azimuth - PULL_OFFSET
    angle_diff = math.radians(wind_bearing - pull_az)
    return math.cos(angle_diff) * (wind_mph / 10.0)


def _build_game_row(ctx_row: pd.Series, fm_latest: pd.DataFrame,
                    odds: pd.DataFrame) -> pd.Series:
    """Merge context + FM fallback + odds for one game into a flat feature row."""
    row = ctx_row.copy()

    # FM fallback for missing sticky features
    fm_home = fm_latest[fm_latest["home_team"] == row.get("home_team")]
    if not fm_home.empty:
        fm_r = fm_home.iloc[0]
        for col in fm_r.index:
            if col not in row.index or pd.isna(row.get(col)):
                row[col] = fm_r[col]

    # Odds join
    if not odds.empty:
        od = odds[odds["home_team"] == row.get("home_team")]
        if not od.empty:
            od_r = od.iloc[0]
            for col in ("close_total", "close_ml_home", "close_ml_away",
                        "runline_home", "runline_home_odds",
                        "pinnacle_ml_home", "public_pct_home", "public_pct_over"):
                if col in od_r.index:
                    row[col] = od_r[col]

    return row


def _score_row(row: pd.Series, features: list[str], booster: xgb.Booster) -> float:
    """Score a single game row against a script model."""
    vals = []
    for f in features:
        v = row.get(f)
        try:
            vals.append(float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else float("nan"))
        except (TypeError, ValueError):
            vals.append(float("nan"))
    dm = xgb.DMatrix([vals], feature_names=features)
    return float(booster.predict(dm, iteration_range=(0, booster.best_iteration + 1))[0])


def _compute_physics_note(home_team: str, wind_mph: float, wind_bearing: float,
                           temp_f: float, stadium_meta: dict) -> str:
    parks = stadium_meta.get("parks", {})
    park = parks.get(home_team, {})
    cf = park.get("cf_azimuth_deg", 45)
    alt = park.get("altitude_ft", 500)
    roof = park.get("roof", "open")
    wind_v_r = _pull_side_wind_vector(wind_bearing, wind_mph, cf, "R")
    wind_v_l = _pull_side_wind_vector(wind_bearing, wind_mph, cf, "L")
    direction = "headwind" if wind_v_r < -0.3 else ("tailwind" if wind_v_r > 0.3 else "neutral")
    return (f"Park={park.get('park_name','?')} alt={alt}ft roof={roof} | "
            f"wind={wind_mph}mph bear={wind_bearing}deg -> "
            f"RHB pull={wind_v_r:+.2f} LHB pull={wind_v_l:+.2f} ({direction}) | "
            f"temp={temp_f}F")


def build_alpha_report(date_str: str) -> pd.DataFrame:
    print(f"\n{'='*64}")
    print(f"  SGP Alpha Report  [{date_str}]")
    print(f"{'='*64}")

    models   = _load_models()
    ctx      = _load_today_context(date_str)
    fm_latest = _load_fm_latest()
    odds     = _load_odds(date_str)
    k_props  = _load_k_props(date_str)
    meta     = _load_stadium_meta()

    print(f"  Games: {len(ctx)}  |  Models loaded: {list(models.keys())}")

    records = []
    for _, game in ctx.iterrows():
        row = _build_game_row(game, fm_latest, odds)
        home = row.get("home_team", "")
        away = row.get("away_team", "")
        game_label = f"{away} @ {home}"
        close_total = _safe(row.get("close_total"), 9.0)
        wind_mph    = _safe(row.get("wind_mph"), 0.0)
        wind_bear   = _safe(row.get("wind_bearing"), 180.0)
        temp_f      = _safe(row.get("temp_f"), 72.0)
        physics_note = _compute_physics_note(home, wind_mph, wind_bear, temp_f, meta)

        # ── Odds for each leg ──────────────────────────────────────────────
        ml_home_odds = _safe(row.get("close_ml_home"), -110.0)
        ml_away_odds = _safe(row.get("close_ml_away"), -110.0)
        p_home_win   = _american_to_implied(ml_home_odds)
        p_game_over  = 0.5   # no exact odds; use 50/50 as conservative
        p_game_under = 1.0 - p_game_over

        # F5 total: proportional from game total (0.56 × close_total)
        f5_est = close_total * F5_FRACTION
        p_f5_under = 0.52   # slight lean under (low-scoring early innings)

        # K props (FanDuel, prefer over line for dominant SP)
        # Join to home/away starter names
        home_starter = _safe(row.get("home_starter_name"), "")
        away_starter = _safe(row.get("away_starter_name"), "")
        p_home_k = 0.45   # default
        p_away_k = 0.45
        home_k_line = None
        away_k_line = None
        if not k_props.empty:
            for starter, is_home in [(home_starter, True), (away_starter, False)]:
                if not starter:
                    continue
                last = starter.split()[-1].lower() if starter else ""
                match = k_props[k_props["pitcher_name"].str.lower().str.contains(last, na=False)]
                # Prefer FanDuel
                fd = match[match["book"] == "fanduel"]
                if fd.empty:
                    fd = match
                if not fd.empty:
                    r = fd.iloc[0]
                    line = float(r.get("line") or 3.5)
                    ov   = float(r.get("over_odds") or -110)
                    p_k  = _american_to_implied(ov)
                    if is_home:
                        p_home_k = p_k
                        home_k_line = line
                    else:
                        p_away_k = p_k
                        away_k_line = line

        # ── Script A2 (Dominance — home SP) ───────────────────────────────
        if "A2" in models:
            p_a2 = _score_row(row, A2_FEATURES, models["A2"])
            # Indep = P(K_F5>=4) × P(F5_Under) × P(Game_Under)
            p_indep_a2 = p_home_k * p_f5_under * p_game_under
            p_market_a2 = p_indep_a2 * (1.0 - BOOK_CORR_TAX)
            edge_a2 = (p_a2 / p_market_a2 - 1.0) if p_market_a2 > 0 else 0.0
            records.append({
                "rank": 0,
                "game_label":     game_label,
                "script":         "A2 Dominance",
                "legs":           f"SP_F5_K>={home_k_line or '4'} | F5_Under_{f5_est:.1f} | Game_Under_{close_total}",
                "sp_name":        home_starter,
                "sp_side":        "home",
                "p_joint_model":  round(p_a2, 4),
                "p_joint_indep":  round(p_indep_a2, 4),
                "p_joint_market": round(p_market_a2, 4),
                "sgp_edge":       round(edge_a2, 4),
                "leg_probs":      f"K_over={p_home_k:.3f} F5_under={p_f5_under:.3f} game_under={p_game_under:.3f}",
                "close_total":    close_total,
                "f5_est":         round(f5_est, 2),
                "sp_k_line":      home_k_line,
                "physics_note":   physics_note,
                "action": "PLAY" if edge_a2 >= MIN_EDGE else "PASS",
            })

        # ── Script B (Explosion — home team) ──────────────────────────────
        if "B" in models:
            p_b = _score_row(row, B_FEATURES, models["B"])
            p_indep_b = 0.40 * p_game_over * p_home_win  # P(score>=5)≈0.40
            p_market_b = p_indep_b * (1.0 - BOOK_CORR_TAX)
            edge_b = (p_b / p_market_b - 1.0) if p_market_b > 0 else 0.0
            records.append({
                "rank": 0,
                "game_label":     game_label,
                "script":         "B Explosion",
                "legs":           f"Home_Score>=5 | Game_Over_{close_total} | Home_ML_Win",
                "sp_name":        away_starter,
                "sp_side":        "away",
                "p_joint_model":  round(p_b, 4),
                "p_joint_indep":  round(p_indep_b, 4),
                "p_joint_market": round(p_market_b, 4),
                "sgp_edge":       round(edge_b, 4),
                "leg_probs":      f"home_score_5+={0.40:.3f} game_over={p_game_over:.3f} ml_win={p_home_win:.3f}",
                "close_total":    close_total,
                "f5_est":         round(f5_est, 2),
                "sp_k_line":      away_k_line,
                "physics_note":   physics_note,
                "action": "PLAY" if edge_b >= MIN_EDGE else "PASS",
            })

        # ── Script C (Elite Duel — both SP) ───────────────────────────────
        if "C" in models:
            p_c = _score_row(row, C_FEATURES, models["C"])
            p_indep_c = p_game_under * (p_home_k * p_away_k) * 0.25  # P(close game)≈0.25
            p_market_c = p_indep_c * (1.0 - BOOK_CORR_TAX)
            edge_c = (p_c / p_market_c - 1.0) if p_market_c > 0 else 0.0
            records.append({
                "rank": 0,
                "game_label":     game_label,
                "script":         "C Elite Duel",
                "legs":           f"Game_Under_{close_total} | Both_SP_F5_K>=3 | Close_Game(+1.5)",
                "sp_name":        f"{home_starter} vs {away_starter}",
                "sp_side":        "both",
                "p_joint_model":  round(p_c, 4),
                "p_joint_indep":  round(p_indep_c, 4),
                "p_joint_market": round(p_market_c, 4),
                "sgp_edge":       round(edge_c, 4),
                "leg_probs":      f"game_under={p_game_under:.3f} home_k={p_home_k:.3f} away_k={p_away_k:.3f} close=0.250",
                "close_total":    close_total,
                "f5_est":         round(f5_est, 2),
                "sp_k_line":      f"{home_k_line}/{away_k_line}",
                "physics_note":   physics_note,
                "action": "PLAY" if edge_c >= MIN_EDGE else "PASS",
            })

    if not records:
        print("  No records generated.")
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("sgp_edge", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path  = OUT_DIR / "sgp_alpha_report.csv"
    json_path = OUT_DIR / "sgp_alpha_report.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    # ── Console report ────────────────────────────────────────────────────
    plays = df[df["action"] == "PLAY"]
    print(f"\n  PLAYS (edge >= {MIN_EDGE:.0%}): {len(plays)} / {len(df)} narratives")
    print(f"\n  {'RANK':<4} {'GAME':<18} {'SCRIPT':<14} {'P_MDL':<7} {'P_MKT':<7} {'EDGE':<8} {'ACTION'}")
    print(f"  {'-'*75}")
    for _, r in df.head(12).iterrows():
        print(f"  #{int(r['rank']):<3} {r['game_label']:<18} {r['script']:<14} "
              f"{r['p_joint_model']:.4f}  {r['p_joint_market']:.4f}  "
              f"{r['sgp_edge']:+.4f}  {r['action']}")

    # ── Detailed audit for top PLAY ────────────────────────────────────────
    top_plays = plays.head(1)
    if not top_plays.empty:
        r = top_plays.iloc[0]
        print(f"\n  TOP PLAY DETAIL:")
        print(f"    Game:          {r['game_label']}")
        print(f"    Script:        {r['script']}")
        print(f"    SP:            {r['sp_name']}")
        print(f"    Legs:          {r['legs']}")
        print(f"    Leg probs:     {r['leg_probs']}")
        print(f"    P(Model):      {r['p_joint_model']:.4f}")
        print(f"    P(Indep):      {r['p_joint_indep']:.4f}")
        print(f"    P(Market):     {r['p_joint_market']:.4f}  (after {BOOK_CORR_TAX:.0%} corr tax)")
        print(f"    SGP Edge:      {r['sgp_edge']:+.4f}  ({r['sgp_edge']*100:+.1f}%)")
        print(f"    Physics:       {r['physics_note']}")

    print(f"\n  Report -> {csv_path}")
    return df


def audit_game(home_team: str, away_starter: str, date_str: str) -> None:
    """
    Detailed audit for a specific game (Task 4 entry point).
    Prints full physics + model breakdown.
    """
    df = build_alpha_report(date_str)
    if df.empty:
        return

    mask = (df["game_label"].str.contains(home_team, case=False) |
            df["sp_name"].str.contains(away_starter.split()[-1], case=False, na=False))
    game_df = df[mask]

    if game_df.empty:
        print(f"\n  [warn] No rows found for {home_team} / {away_starter}")
        return

    print(f"\n{'='*64}")
    print(f"  DETAILED AUDIT: {away_starter} @ {home_team}  [{date_str}]")
    print(f"{'='*64}")
    for _, r in game_df.iterrows():
        print(f"\n  Script {r['script']}:")
        print(f"    Legs:         {r['legs']}")
        print(f"    P(Model):     {r['p_joint_model']:.4f}")
        print(f"    P(Indep):     {r['p_joint_indep']:.4f}")
        print(f"    P(Market):    {r['p_joint_market']:.4f}")
        print(f"    SGP Edge:     {r['sgp_edge']:+.4f} ({r['sgp_edge']*100:+.1f}%)")
        print(f"    Leg probs:    {r['leg_probs']}")
        print(f"    Physics:      {r['physics_note']}")
        print(f"    Action:       {r['action']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None)
    parser.add_argument("--audit-game", default=None,
                        help="HOME_TEAM,AWAY_STARTER e.g. ATL,Andrew Painter")
    args = parser.parse_args()

    import pandas as pd
    ctx = pd.read_parquet(CONTEXT_FILE) if CONTEXT_FILE.exists() else pd.DataFrame()
    date_str = args.date or (ctx["orchestrator_date"].iloc[0] if not ctx.empty else
                             __import__("datetime").date.today().isoformat())

    if args.audit_game:
        parts = args.audit_game.split(",", 1)
        audit_game(parts[0].strip(), parts[1].strip() if len(parts) > 1 else "", date_str)
    else:
        build_alpha_report(date_str)


if __name__ == "__main__":
    main()
