"""
generate_dashboard_json.py — Real-Time Command Center JSON (v5.0 / v10.0).

Aggregates all pipeline outputs into a single command_center.json per day.
One JSON object per game, containing:
  - game metadata (teams, starters, time, park)
  - atmospheric edge (ADI: air_density_rho, wind, temp, adi_note)
  - wind vector multiplier (park-specific, from stadium_metadata.json v2.0)
  - straight bet plays from model_scores.csv
  - SGP plays from sgp_live_edge_{date}.csv
  - Kelly stakes from kelly_stakes_{date}.csv (if saved)
  - in_play_alerts: fatigue alerts from live_drift_monitor.py
  - tilt_warnings:  ABS tilt events from build_pitcher_abs_response.py
  - trap_alerts:    sentiment trap lines from sentiment_drift_scorer.py
  - narrative summary (1-2 sentence English description of edge story)

v2.0 additions
--------------
  - in_play_alerts section per game (live fatigue / velocity drift)
  - tilt_warnings section per game (ABS psychological response)
  - trap_alerts summary section in header
  - park_wind_profile (sensitivity tier, multipliers from stadium_metadata v2)

v3.0 additions
--------------
  - unified_edge section per game (ML + Script + BP health + divergence flag)
  - divergence_plays section in summary header

v4.0 additions (v9.0 Market-Signal pass)
-----------------------------------------
  - market_steam section per game: steamed K-prop pitchers, PrizePicks deltas
  - execution_risk section per game: slippage %, corr-tax gating per bet
  - execution_risk_summary in header: gated scripts, high-risk bet count
  - market_steam_summary in header: total steamed pitchers across slate

v5.0 additions (v10.0 Sovereign pass)
--------------------------------------
  - rl_divergence section per game: Poisson RL vs XGBoost RL stacker comparison
  - Flag when BOTH models agree on Run Line edge > 5% in the same direction
  - rl_divergence_plays summary in header: list of high-conviction RL games
  - pipeline_version updated to "v10.0"

Output
------
  data/dashboard/command_center_{date}.json   — structured JSON per game
  data/dashboard/command_center.json          — symlink/copy of today's file

Usage
-----
  python generate_dashboard_json.py
  python generate_dashboard_json.py --date 2026-04-24
  python generate_dashboard_json.py --pretty   # indented JSON
  python generate_dashboard_json.py --no-live  # skip live alert loading
  python generate_dashboard_json.py --no-rl    # skip RL divergence scoring
"""
from __future__ import annotations

import argparse
import glob
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT           = Path(__file__).resolve().parent
MODEL_SCORES    = _ROOT / "model_scores.csv"
SGP_DIR         = _ROOT / "data/sgp"
BANKROLL_DIR    = _ROOT / "data/bankroll"
STATCAST_DIR    = _ROOT / "data/statcast"
CONTEXT_PATH    = _ROOT / "data/orchestrator/daily_context.parquet"
STADIUM_META    = _ROOT / "config/stadium_metadata.json"
LIVE_DIR        = _ROOT / "data/live"
OUTPUT_DIR      = _ROOT / "data/dashboard"

# ---------------------------------------------------------------------------
# Stadium metadata (v2.0) — wind sensitivity profiles
# ---------------------------------------------------------------------------

_PARK_META_CACHE: dict | None = None


def _load_park_meta() -> dict:
    global _PARK_META_CACHE
    if _PARK_META_CACHE is None:
        if STADIUM_META.exists():
            _PARK_META_CACHE = json.load(STADIUM_META.open()).get("parks", {})
        else:
            _PARK_META_CACHE = {}
    return _PARK_META_CACHE


def _park_wind_profile(home_team: str, wind_bearing: float | None) -> dict:
    """Return park wind profile dict for embedding in game_obj."""
    parks = _load_park_meta()
    park  = parks.get(home_team.upper(), {})
    if not park:
        return {}

    tier       = park.get("wind_sensitivity_tier", "medium")
    mult_out   = park.get("wind_mult_out",   1.5)
    mult_in    = park.get("wind_mult_in",    1.0)
    mult_cross = park.get("wind_mult_cross", 1.1)

    # Compute active multiplier based on current wind bearing
    active_mult = None
    active_dir  = None
    if wind_bearing is not None and tier != "dome":
        cf_az  = float(park.get("cf_azimuth_deg", 45))
        diff   = abs((wind_bearing - cf_az + 180) % 360 - 180)
        if diff <= 45:
            active_mult = mult_out
            active_dir  = "OUT"
        elif diff >= 135:
            active_mult = mult_in
            active_dir  = "IN"
        else:
            active_mult = mult_cross
            active_dir  = "CROSS"
    elif tier == "dome":
        active_mult = 0.0
        active_dir  = "N/A (dome)"

    return {
        "park_name":             park.get("park_name", ""),
        "wind_sensitivity_tier": tier,
        "wind_mult_out":         mult_out,
        "wind_mult_in":          mult_in,
        "wind_mult_cross":       mult_cross,
        "active_wind_mult":      active_mult,
        "active_wind_direction": active_dir,
        "wind_note":             park.get("wind_sensitivity_note", ""),
    }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_model_scores(date_str: str) -> pd.DataFrame:
    if not MODEL_SCORES.exists():
        return pd.DataFrame()
    df = pd.read_csv(MODEL_SCORES)
    df["date"] = df["date"].astype(str).str.strip()
    return df[df["date"] == date_str].copy()


def _load_sgp(date_str: str) -> pd.DataFrame:
    tag = date_str.replace("-", "_")
    p   = SGP_DIR / f"sgp_live_edge_{tag}.csv"
    if not p.exists():
        files = sorted(glob.glob(str(SGP_DIR / "sgp_live_edge_*.csv")))
        if not files:
            return pd.DataFrame()
        p = Path(files[-1])
    df = pd.read_csv(p)
    df["_date"] = date_str
    return df


def _load_kelly(date_str: str) -> pd.DataFrame:
    p = BANKROLL_DIR / f"kelly_stakes_{date_str}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _load_live_alerts(date_str: str) -> list[dict]:
    """Load fatigue alerts from live_drift_monitor."""
    p = LIVE_DIR / f"fatigue_alerts_{date_str}.json"
    if not p.exists():
        return []
    try:
        return json.load(p.open())
    except Exception:
        return []


def _load_tilt_warnings(date_str: str) -> list[dict]:
    """Load ABS tilt warnings from build_pitcher_abs_response."""
    p = LIVE_DIR / f"abs_tilt_active_{date_str}.json"
    if not p.exists():
        return []
    try:
        return json.load(p.open())
    except Exception:
        return []


def _load_trap_alerts(date_str: str) -> list[dict]:
    """Load sentiment trap alerts from sentiment_drift_scorer."""
    p = LIVE_DIR / f"trap_alerts_{date_str}.json"
    if not p.exists():
        return []
    try:
        return json.load(p.open())
    except Exception:
        return []


def _load_context(date_str: str) -> pd.DataFrame:
    if not CONTEXT_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(CONTEXT_PATH)
    if "orchestrator_date" in df.columns:
        df = df[df["orchestrator_date"] == date_str].copy()
    return df


def _load_market_signals(date_str: str) -> pd.DataFrame:
    """Load prop market steam signals from fetch_prop_market_signals.py."""
    tag = date_str.replace("-", "_")
    p = STATCAST_DIR / f"prop_market_signals_{tag}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    p_csv = STATCAST_DIR / f"prop_market_signals_{tag}.csv"
    if p_csv.exists():
        return pd.read_csv(p_csv)
    return pd.DataFrame()


def _load_slippage_report(date_str: str) -> pd.DataFrame:
    """Load per-bet slippage rows from audit_bet_execution.py."""
    p = BANKROLL_DIR / f"slippage_report_{date_str}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _load_slippage_summary(date_str: str) -> dict:
    """Load aggregate slippage summary JSON from audit_bet_execution.py."""
    p = BANKROLL_DIR / f"slippage_summary_{date_str}.json"
    if not p.exists():
        return {}
    try:
        return json.load(p.open())
    except Exception:
        return {}


def _load_rl_predictions(date_str: str) -> pd.DataFrame:
    """
    Load Poisson + XGBoost RL stacker predictions for today.

    Attempts (in order):
      1. Cached parquet from a prior score_run_dist + score_rl_v1 run.
      2. On-demand via train_rl_v1.score_today() (requires model artifact).

    Returns DataFrame with columns:
        home_team, away_team, p_home_cover_final (Poisson stacker),
        p_home_cover_v1 (XGBoost v1), rl_signal_flag.
    Returns empty DataFrame on failure.
    """
    # 1. Check for saved parquet
    cache_p = OUTPUT_DIR / f"rl_predictions_{date_str}.parquet"
    if cache_p.exists():
        return pd.read_parquet(cache_p)

    # 2. On-demand scoring from v1 stacker
    try:
        import sys
        sys.path.insert(0, str(_ROOT))
        from train_rl_v1 import score_today as _rl_score
        rl_v1 = _rl_score(date_str)
        if rl_v1.empty:
            return pd.DataFrame()

        # Try attaching Poisson p_home_cover_final from run_dist scorer
        try:
            from score_run_dist_today import predict_games as _poisson_score
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                poisson_df = _poisson_score(date_str)
            if not poisson_df.empty and "p_home_cover_final" in poisson_df.columns:
                poi_slim = poisson_df[["home_team", "p_home_cover_final",
                                       "p_cover_xgb"]].copy()
                poi_slim["home_team"] = poi_slim["home_team"].str.upper()
                rl_v1["home_team"]    = rl_v1["home_team"].str.upper()
                rl_v1 = rl_v1.merge(poi_slim, on="home_team", how="left")
        except Exception as e:
            print(f"  [rl_div] Poisson attach failed: {e}")

        # Cache for re-use within same session
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        rl_v1.to_parquet(cache_p, index=False)
        return rl_v1
    except Exception as e:
        print(f"  [rl_div] RL predictions unavailable: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Game key normalisation
# ---------------------------------------------------------------------------

def _alert_matches_game(alert: dict, home: str, away: str) -> bool:
    """Heuristic: check if an alert dict belongs to this game by pitcher name context."""
    name = str(alert.get("pitcher_name", "")).upper()
    return False  # gamePk match is preferred; name-based fallback not implemented


def _normalise_game(game: str) -> str:
    """Normalise game string to 'AWAY@HOME' upper-case."""
    return str(game).strip().upper()


def _extract_teams(game: str) -> tuple[str, str]:
    """Extract (away, home) from 'AWAY @ HOME' or 'AWAY@HOME' etc."""
    g = _normalise_game(game).replace(" ", "")
    if "@" in g:
        parts = g.split("@", 1)
        return parts[0].strip(), parts[1].strip()
    return "", g


# ---------------------------------------------------------------------------
# Narrative builder
# ---------------------------------------------------------------------------

def _build_narrative(home: str, away: str,
                     ctx: dict | None,
                     straight_plays: list[dict],
                     sgp_plays: list[dict],
                     in_play_alerts: list[dict] | None = None,
                     tilt_warnings: list[dict] | None = None) -> str:
    """Generate 1-2 sentence English narrative describing the edge story."""
    parts = []

    # Atmospheric context
    if ctx:
        rho   = ctx.get("air_density_rho", 1.18)
        wind  = ctx.get("wind_mph", 0)
        temp  = ctx.get("temp_f", 70)
        if rho is not None and isinstance(rho, (int, float)) and rho < 1.10:
            parts.append(f"Thin air (rho={rho:.3f} kg/m3) at {temp:.0f}F with "
                         f"{wind:.0f} mph wind favors power hitters.")
        elif rho is not None and isinstance(rho, (int, float)) and rho > 1.20:
            parts.append(f"Dense air (rho={rho:.3f} kg/m3) suppresses fly balls "
                         f"({temp:.0f}F, {wind:.0f} mph).")

        bp_h = ctx.get("home_bullpen_burn_5d", 0) or 0
        bp_a = ctx.get("away_bullpen_burn_5d", 0) or 0
        if bp_h > 400 or bp_a > 400:
            tired_team = home if bp_h >= bp_a else away
            parts.append(f"{tired_team} bullpen is gassed ({max(bp_h, bp_a):.0f} pitches/5d).")

    # In-play fatigue alerts (v2.0)
    if in_play_alerts:
        for alert in in_play_alerts[:1]:
            parts.append(f"LIVE: {alert.get('pitcher_name', '')} showing fatigue "
                         f"(velo drop {alert.get('velo_drop', 0):.1f} mph, "
                         f"inn {alert.get('inning', '?')}).")

    # ABS tilt warnings (v2.0)
    if tilt_warnings:
        for w in tilt_warnings[:1]:
            mult = w.get("opp_team_total_mult", 1.0)
            inn  = w.get("active_innings", 2)
            parts.append(f"TILT: {w.get('pitcher_name', '')} is Tilted post-ABS challenge -- "
                         f"{mult:.2f}x team-total boost, {inn} inn active.")

    # Edge summary
    play_count = len(straight_plays) + len(sgp_plays)
    if play_count == 0 and not parts:
        parts.append("No actionable plays today.")
    else:
        edge_descs = []
        for p in straight_plays[:2]:
            edge_descs.append(f"{p.get('bet_type', '')} {p.get('pick_direction', '')} "
                              f"(edge={p.get('edge', 0):+.3f})")
        for p in sgp_plays[:1]:
            edge_descs.append(f"SGP {p.get('script', '')} (edge={p.get('sgp_edge', 0):+.3f})")
        if edge_descs:
            parts.append("Plays: " + "; ".join(edge_descs) + ".")

    return " ".join(parts) if parts else f"{away}@{home} -- no notable context."


# ---------------------------------------------------------------------------
# Build per-game JSON objects
# ---------------------------------------------------------------------------

def _safe(val: Any) -> Any:
    """Convert numpy types and NaN to JSON-safe Python types."""
    if val is None:
        return None
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return round(float(val), 6)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val


def _row_to_dict(row: pd.Series, cols: list[str]) -> dict:
    return {c: _safe(row.get(c)) for c in cols if c in row.index}


# ---------------------------------------------------------------------------
# v3.0 Unified Edge builder
# ---------------------------------------------------------------------------

_BP_BURN_THRESH = 360.0   # 90th-pctile historical high-leverage pitch count (5d)
_ML_BULLISH_MIN = 0.55    # adj_home_win_prob threshold to be "bullish on home"
_ML_BEARISH_MAX = 0.45    # below this = "bearish on home (bullish on away)"


def _build_unified_edge(
    home: str,
    away: str,
    straight_plays: list[dict],
    sgp_plays: list[dict],
    ctx_row: dict | None,
    game_fatigue: list[dict],
    game_tilts: list[dict],
) -> dict:
    """
    Synthesise ML edge, script probability, bullpen health, and RL coverage
    into a single 'unified_edge' object for the Command Center.

    Returns a dict that is serialisable to JSON.
    """
    # ── ML win probability ───────────────────────────────────────────────
    ml_home_win_prob: float | None = None
    ml_edge_best: float | None = None
    for p in straight_plays:
        if str(p.get("bet_type", "")).upper() in ("ML", "MONEYLINE"):
            prob = _safe(p.get("model_prob"))
            edge = _safe(p.get("edge"))
            if prob is not None:
                if ml_home_win_prob is None or (edge or 0) > (ml_edge_best or 0):
                    ml_home_win_prob = float(prob)
                    ml_edge_best = float(edge) if edge is not None else None

    # ── Best script probability ──────────────────────────────────────────
    best_script_prob: float | None = None
    best_script_name: str | None = None
    for s in sgp_plays:
        prob = _safe(s.get("p_joint_copula"))
        if prob is not None:
            if best_script_prob is None or float(prob) > best_script_prob:
                best_script_prob = float(prob)
                best_script_name = s.get("script")

    # ── Bullpen health ───────────────────────────────────────────────────
    h_burn: float | None = None
    a_burn: float | None = None
    if ctx_row:
        h_burn = _safe(ctx_row.get("home_bullpen_burn_5d"))
        a_burn = _safe(ctx_row.get("away_bullpen_burn_5d"))

    h_gassed = h_burn is not None and h_burn > _BP_BURN_THRESH
    a_gassed = a_burn is not None and a_burn > _BP_BURN_THRESH
    home_bp_health = "gassed" if h_gassed else ("ok" if h_burn is not None else "unknown")
    away_bp_health = "gassed" if a_gassed else ("ok" if a_burn is not None else "unknown")

    # ── Divergence play detection ────────────────────────────────────────
    # Definition: ML model is bullish on a side BUT that side's bullpen is
    # burned beyond the 90th-pctile fatigue threshold — the win expectation
    # may erode in late innings.
    divergence = False
    divergence_reason: str | None = None
    if ml_home_win_prob is not None:
        if ml_home_win_prob > _ML_BULLISH_MIN and h_gassed:
            divergence = True
            divergence_reason = (
                f"ML bullish home ({ml_home_win_prob:.1%}) "
                f"but home BP gassed (burn={h_burn:.0f})"
            )
        elif ml_home_win_prob < _ML_BEARISH_MAX and a_gassed:
            divergence = True
            divergence_reason = (
                f"ML bearish home / bullish away ({1-ml_home_win_prob:.1%}) "
                f"but away BP gassed (burn={a_burn:.0f})"
            )

    # Also flag when a live fatigue alert exists for the projected winner's SP
    if not divergence and game_fatigue:
        script_d_late = any(a.get("script_d_alert") for a in game_fatigue)
        if script_d_late and ml_home_win_prob is not None and ml_home_win_prob > _ML_BULLISH_MIN:
            divergence = True
            divergence_reason = "ML bullish home but live Script-D fatigue alert active"

    return {
        "ml_home_win_prob":  round(ml_home_win_prob, 4) if ml_home_win_prob is not None else None,
        "ml_edge_best":      round(ml_edge_best, 4)     if ml_edge_best is not None else None,
        "best_script_prob":  round(best_script_prob, 4) if best_script_prob is not None else None,
        "best_script_name":  best_script_name,
        "home_bullpen_burn": h_burn,
        "away_bullpen_burn": a_burn,
        "home_bp_health":    home_bp_health,
        "away_bp_health":    away_bp_health,
        "divergence_play":   divergence,
        "divergence_reason": divergence_reason,
    }


# ---------------------------------------------------------------------------
# v5.0 RL Divergence builder  (v10.0 Sovereign pass)
# ---------------------------------------------------------------------------

_RL_EDGE_MIN  = 0.05   # both models must show > 5% edge above fair 50%
_RL_AGREE_DIR = True   # edges must be in the same direction (same sign)


def _build_rl_divergence(rl_df: pd.DataFrame, home: str) -> dict:
    """
    Return Poisson-Stacker RL divergence object for a single game.

    A 'rl_divergence' flag fires when:
      1. Poisson RL scorer AND XGBoost v1 stacker BOTH show edge > 5%.
      2. Both models agree on direction (home cover vs away cover).

    This consensus signals the Poisson fair-price is reinforced by physical
    environment variables (wind, bullpen, Script-C duel alignment) rather than
    contradicted — making the RL signal high-conviction.
    """
    empty = {
        "p_cover_poisson":     None,
        "p_cover_v1":          None,
        "p_cover_final":       None,
        "rl_edge_poisson":     None,
        "rl_edge_v1":          None,
        "rl_signal_flag":      0,
        "rl_divergence":       False,
        "rl_divergence_side":  None,
        "rl_divergence_note":  None,
    }
    if rl_df.empty:
        return empty

    mask = rl_df["home_team"].str.upper() == home.upper()
    if not mask.any():
        return empty

    row = rl_df[mask].iloc[0]

    p_poi   = _safe(row.get("p_cover_poisson"))
    p_v1    = _safe(row.get("p_home_cover_v1"))
    p_final = _safe(row.get("p_home_cover_final"))  # Poisson stacker final

    # Use p_final if available; otherwise p_poi as the Poisson reference
    p_poisson_ref = p_final if p_final is not None else p_poi

    edge_poi = round(float(p_poisson_ref) - 0.50, 4) if p_poisson_ref is not None else None
    edge_v1  = round(float(p_v1) - 0.50, 4)          if p_v1 is not None else None

    # Divergence gate: both edges must exceed threshold AND agree on direction
    diverge = False
    div_side = None
    div_note = None
    if edge_poi is not None and edge_v1 is not None:
        both_strong = abs(edge_poi) > _RL_EDGE_MIN and abs(edge_v1) > _RL_EDGE_MIN
        same_dir    = (edge_poi * edge_v1) > 0   # same sign
        if both_strong and same_dir:
            diverge  = True
            div_side = "HOME_COVER" if edge_poi > 0 else "AWAY_COVER"
            div_note = (
                f"Poisson edge={edge_poi:+.3f} & v1_stacker edge={edge_v1:+.3f} "
                f"agree on {div_side} — physical env confirms Poisson prior"
            )

    return {
        "p_cover_poisson":     round(float(p_poisson_ref), 4) if p_poisson_ref is not None else None,
        "p_cover_v1":          round(float(p_v1), 4)         if p_v1 is not None else None,
        "p_cover_final":       round(float(p_final), 4)      if p_final is not None else None,
        "rl_edge_poisson":     edge_poi,
        "rl_edge_v1":          edge_v1,
        "rl_signal_flag":      int(row.get("rl_signal_flag", 0)),
        "rl_divergence":       diverge,
        "rl_divergence_side":  div_side,
        "rl_divergence_note":  div_note,
    }


# ---------------------------------------------------------------------------
# v4.0 Market Steam section builder
# ---------------------------------------------------------------------------

def _build_market_steam(signals: pd.DataFrame, home: str, away: str) -> dict:
    """
    Return K-prop steam signals for pitchers in this game.
    Matches on home_team OR away_team (case-insensitive).
    """
    empty = {"pitchers": [], "steam_count": 0, "game_has_steam": False}
    if signals.empty:
        return empty

    ht_col = signals["home_team"].str.upper() if "home_team" in signals.columns else pd.Series(dtype=str)
    at_col = signals["away_team"].str.upper() if "away_team" in signals.columns else pd.Series(dtype=str)
    mask   = (ht_col == home) | (at_col == away)
    if not mask.any():
        return empty

    pitchers = []
    for _, r in signals[mask].iterrows():
        pitchers.append({
            "pitcher_name":    r.get("pitcher_name"),
            "retail_line":     _safe(r.get("retail_consensus_line")),
            "prizepicks_line": _safe(r.get("prizepicks_proxy_line")),
            "market_delta_k":  _safe(r.get("market_delta_k")),
            "line_range":      _safe(r.get("line_range")),
            "steam_flag":      int(r.get("steam_flag", 0)),
            "juice_lean":      _safe(r.get("juice_lean")),
            "sharp_book":      r.get("sharp_book"),
        })

    steam_count = sum(1 for p in pitchers if p["steam_flag"])
    return {
        "pitchers":       pitchers,
        "steam_count":    steam_count,
        "game_has_steam": steam_count > 0,
    }


# ---------------------------------------------------------------------------
# v4.0 Execution Risk section builder
# ---------------------------------------------------------------------------

def _build_execution_risk(slip_report: pd.DataFrame, home: str) -> dict:
    """
    Return per-bet slippage / gating info for bets on this game.
    Matches on home_team (case-insensitive).
    """
    empty = {"bets": [], "high_count": 0, "medium_count": 0, "gated_count": 0}
    if slip_report.empty or "home_team" not in slip_report.columns:
        return empty

    mask = slip_report["home_team"].str.upper() == home
    if not mask.any():
        return empty

    bets = []
    for _, r in slip_report[mask].iterrows():
        bets.append({
            "bet_label":        r.get("bet_label"),
            "execution_risk":   r.get("execution_risk"),
            "slip_pct":         _safe(r.get("slip_pct")),
            "corr_tax_observed": _safe(r.get("corr_tax_observed")),
            "gate_flag":        bool(r.get("gate_flag", False)),
            "adjusted_$":       _safe(r.get("adjusted_$")),
        })

    return {
        "bets":         bets,
        "high_count":   sum(1 for b in bets if b["execution_risk"] == "HIGH"),
        "medium_count": sum(1 for b in bets if b["execution_risk"] == "MEDIUM"),
        "gated_count":  sum(1 for b in bets if b["gate_flag"]),
    }


def build_game_objects(date_str: str, include_live: bool = True,
                        include_rl: bool = True) -> list[dict]:
    scores       = _load_model_scores(date_str)
    sgp          = _load_sgp(date_str)
    kelly        = _load_kelly(date_str)
    context      = _load_context(date_str)
    mkt_signals  = _load_market_signals(date_str)           # v4.0
    slip_report  = _load_slippage_report(date_str)          # v4.0
    rl_preds     = _load_rl_predictions(date_str) \
                   if include_rl else pd.DataFrame()        # v5.0

    # v2.0 live intelligence
    all_fatigue = _load_live_alerts(date_str)   if include_live else []
    all_tilts   = _load_tilt_warnings(date_str) if include_live else []

    # Collect all games from any available source
    all_games: set[str] = set()
    for df, col in [(scores, "game"), (sgp, "game")]:
        if not df.empty and col in df.columns:
            all_games.update(df[col].dropna().apply(_normalise_game).tolist())

    if not all_games and not context.empty:
        # Fall back to context game keys
        for _, r in context.iterrows():
            all_games.add(_normalise_game(f"{r['away_team']}@{r['home_team']}"))

    game_objects = []
    for game_key in sorted(all_games):
        away, home = _extract_teams(game_key)

        # --- Context row ---
        ctx_row = None
        if not context.empty:
            mask = (context["home_team"].str.upper() == home) | \
                   (context["away_team"].str.upper() == away)
            if mask.any():
                ctx_row = context[mask].iloc[0].to_dict()
                ctx_row = {k: _safe(v) for k, v in ctx_row.items()}

        # --- Straight plays ---
        straight_plays_raw = []
        if not scores.empty:
            mask = scores["game"].apply(_normalise_game) == game_key
            for _, r in scores[mask].iterrows():
                play = _row_to_dict(r, [
                    "model", "bet_type", "pick_direction", "model_prob",
                    "P_true", "edge", "retail_american_odds", "tier",
                    "actionable", "signal_flags", "projected_total_adj",
                    "k_lambda", "sim_mean_k",
                ])
                straight_plays_raw.append(play)

        # Filter to actionable only for the main list
        straight_actionable = [p for p in straight_plays_raw if p.get("actionable")]
        straight_all        = straight_plays_raw

        # --- SGP plays ---
        sgp_plays_raw = []
        if not sgp.empty:
            mask = sgp["game"].apply(_normalise_game) == game_key
            for _, r in sgp[mask].iterrows():
                play = _row_to_dict(r, [
                    "script", "legs", "home_sp", "away_sp",
                    "p_joint_copula", "p_book_sgp", "sgp_edge", "corr_lift",
                    "synergy_home", "synergy_away", "action", "adi_note",
                    "home_bp_gassed", "away_bp_gassed",
                    "home_burn_5d", "away_burn_5d",
                ])
                sgp_plays_raw.append(play)

        sgp_actionable = [p for p in sgp_plays_raw if p.get("action") == "PLAY"]
        sgp_all        = sgp_plays_raw

        # --- Kelly stakes ---
        kelly_straight: list[dict] = []
        kelly_sgp:      list[dict] = []
        if not kelly.empty:
            mask = kelly["game"].apply(_normalise_game) == game_key
            for _, r in kelly[mask].iterrows():
                kd = _row_to_dict(r, [
                    "bet_label", "source", "p_model", "model_edge",
                    "decimal_odds", "bankroll",
                    "kelly_full_pct", "kelly_half_pct", "kelly_quarter_pct",
                    "kelly_full_$", "kelly_half_$", "kelly_quarter_$",
                    "recommended_$",
                ])
                if r.get("source") == "sgp":
                    kelly_sgp.append(kd)
                else:
                    kelly_straight.append(kd)

        # --- ADI section ---
        adi: dict = {}
        wind_bearing_val: float | None = None
        if ctx_row:
            for key in ("air_density_rho", "wind_mph", "wind_bearing",
                        "temp_f", "relative_humidity", "surface_pressure_hpa"):
                if key in ctx_row:
                    adi[key] = ctx_row[key]
            wind_bearing_val = ctx_row.get("wind_bearing")
            # Pull adi_note from first SGP play for this game if available
            for p in sgp_all:
                if p.get("adi_note"):
                    adi["note"] = p["adi_note"]
                    break

        # --- Park wind profile (v2.0) ---
        wind_profile = _park_wind_profile(home, wind_bearing_val)

        # --- In-play alerts for this game (v2.0) ---
        game_fatigue = [a for a in all_fatigue
                        if a.get("game_pk") and str(a.get("game_pk")) in str(game_key)
                        or _alert_matches_game(a, home, away)]
        game_tilts   = [w for w in all_tilts
                        if w.get("game_pk") and str(w.get("game_pk")) in str(game_key)
                        or _alert_matches_game(w, home, away)]

        # --- Narrative (v2.0 with live alerts) ---
        narrative = _build_narrative(home, away, ctx_row,
                                      straight_actionable, sgp_actionable,
                                      in_play_alerts=game_fatigue,
                                      tilt_warnings=game_tilts)

        # --- Unified Edge (v3.0) ---
        unified_edge = _build_unified_edge(
            home, away,
            straight_plays=straight_all,
            sgp_plays=sgp_all,
            ctx_row=ctx_row,
            game_fatigue=game_fatigue,
            game_tilts=game_tilts,
        )

        # --- Market Steam (v4.0) ---
        market_steam = _build_market_steam(mkt_signals, home, away)

        # --- Execution Risk (v4.0) ---
        execution_risk = _build_execution_risk(slip_report, home)

        # --- RL Divergence (v5.0) ---
        rl_divergence = _build_rl_divergence(rl_preds, home)

        game_obj = {
            "game":              game_key,
            "date":              date_str,
            "home_team":         home,
            "away_team":         away,
            "home_starter":      _safe(ctx_row.get("home_starter_name")) if ctx_row else None,
            "away_starter":      _safe(ctx_row.get("away_starter_name")) if ctx_row else None,
            "close_total":       _safe(ctx_row.get("close_total")) if ctx_row else None,
            "narrative":         narrative,
            "adi":               adi,
            "park_wind_profile": wind_profile,
            "straight_plays": {
                "actionable":    straight_actionable,
                "all":           straight_all,
                "count":         len(straight_actionable),
            },
            "sgp_plays": {
                "actionable":    sgp_actionable,
                "all":           sgp_all,
                "count":         len(sgp_actionable),
            },
            "kelly_stakes": {
                "straight":      kelly_straight,
                "sgp":           kelly_sgp,
                "total_recommended_$": _safe(
                    sum(k.get("recommended_$") or 0 for k in kelly_straight + kelly_sgp)
                ),
            },
            "in_play_alerts": {
                "fatigue_alerts": game_fatigue,
                "tilt_warnings":  game_tilts,
                "script_d_relevant": any(a.get("script_d_alert") for a in game_fatigue),
                "count": len(game_fatigue) + len(game_tilts),
            },
            "unified_edge":        unified_edge,
            "divergence_play":     unified_edge["divergence_play"],
            "market_steam":        market_steam,    # v4.0
            "execution_risk":      execution_risk,  # v4.0
            "rl_divergence":       rl_divergence,   # v5.0
            "rl_divergence_play":  rl_divergence["rl_divergence"],  # v5.0
            "context":             ctx_row or {},
        }
        game_objects.append(game_obj)

    return game_objects


# ---------------------------------------------------------------------------
# Summary header
# ---------------------------------------------------------------------------

def build_summary(game_objects: list[dict], date_str: str,
                   trap_alerts: list[dict] | None = None,
                   slip_summary: dict | None = None) -> dict:
    total_straight   = sum(g["straight_plays"]["count"] for g in game_objects)
    total_sgp        = sum(g["sgp_plays"]["count"] for g in game_objects)
    total_kelly      = sum(
        (g["kelly_stakes"]["total_recommended_$"] or 0) for g in game_objects
    )
    total_fatigue    = sum(g["in_play_alerts"]["count"] for g in game_objects)
    script_d_live    = sum(1 for g in game_objects
                           if g["in_play_alerts"]["script_d_relevant"])
    divergence_count = sum(1 for g in game_objects if g.get("divergence_play"))

    # Collect divergence game details for the header
    divergence_games = [
        {
            "game": g["game"],
            "reason": g.get("unified_edge", {}).get("divergence_reason"),
            "ml_home_win_prob": g.get("unified_edge", {}).get("ml_home_win_prob"),
        }
        for g in game_objects if g.get("divergence_play")
    ]

    # v4.0: Market steam summary across entire slate
    steam_pitchers_total = sum(
        g.get("market_steam", {}).get("steam_count", 0) for g in game_objects
    )
    steam_games = [
        {
            "game":    g["game"],
            "pitchers": [
                p["pitcher_name"] for p in g.get("market_steam", {}).get("pitchers", [])
                if p.get("steam_flag")
            ],
        }
        for g in game_objects if g.get("market_steam", {}).get("game_has_steam")
    ]

    # v4.0: Execution risk summary across entire slate
    high_risk_total  = sum(g.get("execution_risk", {}).get("high_count", 0) for g in game_objects)
    medium_risk_total= sum(g.get("execution_risk", {}).get("medium_count", 0) for g in game_objects)
    gated_total      = sum(g.get("execution_risk", {}).get("gated_count", 0) for g in game_objects)
    gated_scripts    = [
        b["bet_label"]
        for g in game_objects
        for b in g.get("execution_risk", {}).get("bets", [])
        if b.get("gate_flag")
    ]

    return {
        "date":                      date_str,
        "games":                     len(game_objects),
        "total_straight_plays":      total_straight,
        "total_sgp_plays":           total_sgp,
        "total_kelly_stake_$":       round(total_kelly, 2),
        "live_alerts_total":         total_fatigue,
        "script_d_live_games":       script_d_live,
        "trap_alerts":               trap_alerts or [],
        "trap_alert_count":          len(trap_alerts) if trap_alerts else 0,
        "divergence_plays":          divergence_games,
        "divergence_play_count":     divergence_count,
        # v4.0 market steam
        "market_steam_summary": {
            "steamed_pitchers_total": steam_pitchers_total,
            "steamed_games":          steam_games,
        },
        # v4.0 execution risk
        "execution_risk_summary": {
            "high_risk_bets":   high_risk_total,
            "medium_risk_bets": medium_risk_total,
            "gated_scripts":    gated_scripts,
            "gated_count":      gated_total,
            "audit":            slip_summary or {},
        },
        # v5.0 RL divergence summary
        "rl_divergence_summary": {
            "divergence_plays":  [
                {
                    "game":           g["game"],
                    "side":           g["rl_divergence"]["rl_divergence_side"],
                    "edge_poisson":   g["rl_divergence"]["rl_edge_poisson"],
                    "edge_v1":        g["rl_divergence"]["rl_edge_v1"],
                    "note":           g["rl_divergence"]["rl_divergence_note"],
                }
                for g in game_objects if g.get("rl_divergence_play")
            ],
            "divergence_count":  sum(1 for g in game_objects
                                     if g.get("rl_divergence_play")),
        },
        "generated_at":              pd.Timestamp.now().isoformat(),
        "pipeline_version":          "v10.0",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(date_str: str, pretty: bool = False,
              include_live: bool = True, include_rl: bool = True) -> dict:
    print(f"[dashboard v5/v10] Building command center for {date_str} ...")

    trap_alerts   = _load_trap_alerts(date_str)   if include_live else []
    slip_summary  = _load_slippage_summary(date_str)
    game_objects  = build_game_objects(date_str, include_live=include_live,
                                        include_rl=include_rl)
    summary       = build_summary(game_objects, date_str,
                                  trap_alerts=trap_alerts,
                                  slip_summary=slip_summary)

    payload = {
        "summary": summary,
        "games":   game_objects,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    indent = 2 if pretty else None

    dated_path  = OUTPUT_DIR / f"command_center_{date_str}.json"
    latest_path = OUTPUT_DIR / "command_center.json"

    with dated_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)

    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)

    print(f"  games: {summary['games']} | "
          f"straight: {summary['total_straight_plays']} | "
          f"SGP: {summary['total_sgp_plays']} | "
          f"Kelly: ${summary['total_kelly_stake_$']:,.2f}")
    print(f"  live alerts: {summary['live_alerts_total']} | "
          f"Script D live: {summary['script_d_live_games']} | "
          f"trap alerts: {summary['trap_alert_count']} | "
          f"divergence plays: {summary['divergence_play_count']}")
    steam_ct  = summary["market_steam_summary"]["steamed_pitchers_total"]
    risk_h    = summary["execution_risk_summary"]["high_risk_bets"]
    risk_m    = summary["execution_risk_summary"]["medium_risk_bets"]
    gated     = summary["execution_risk_summary"]["gated_count"]
    rl_div_ct = summary["rl_divergence_summary"]["divergence_count"]
    print(f"  steam flags: {steam_ct} | "
          f"exec risk HIGH: {risk_h} | MEDIUM: {risk_m} | gated: {gated}")
    print(f"  RL divergence plays: {rl_div_ct} "
          f"(Poisson+Stacker edge > {_RL_EDGE_MIN*100:.0f}% agree)")
    print(f"  Saved -> {dated_path.name}")
    print(f"  Saved -> {latest_path.name}")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Command Center JSON (v5.0 / v10.0)")
    parser.add_argument("--date",     default=date.today().isoformat())
    parser.add_argument("--pretty",   action="store_true",
                        help="Indent output JSON (larger file)")
    parser.add_argument("--no-live",  dest="include_live", action="store_false",
                        default=True,
                        help="Skip loading live alerts (fatigue/tilt/trap)")
    parser.add_argument("--no-rl",    dest="include_rl",   action="store_false",
                        default=True,
                        help="Skip RL divergence scoring (faster, no model needed)")
    args = parser.parse_args()

    generate(date_str=args.date, pretty=args.pretty,
             include_live=args.include_live, include_rl=args.include_rl)


if __name__ == "__main__":
    main()
