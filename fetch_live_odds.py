"""
fetch_live_odds.py — v4.3 SGP Edge Auditor (Correlation-Tax Aware).

Implements:
    SGP_Edge = P(Joint_Model) / P(Book_SGP) - 1

where:
    P(Joint_Model)  = Gaussian copula joint prob (from correlation_matrix.py)
    P(Book_SGP)     = product(leg_implied_probs) x (1 - BOOK_CORR_TAX)
    BOOK_CORR_TAX   = ~0.15 (FanDuel/DK standard) or ~0.20 (sharper books)

The structural edge: books assume low r between legs and apply a flat ~15%
haircut. When the true r is high (e.g. SP K + Game Under = +0.20), the copula
joint is meaningfully above the book's joint, creating exploitable alpha.

Usage:
    python fetch_live_odds.py                      # today's full slate
    python fetch_live_odds.py --date 2026-04-24    # specific date
    python fetch_live_odds.py --min-edge 0.10      # higher threshold

Outputs:
    data/sgp/sgp_live_edge_{date}.csv
    data/sgp/sgp_live_edge_{date}.json
"""
from __future__ import annotations

import argparse
import json
import os
import glob
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent

ODDS_DIR      = _ROOT / "data/statcast"
KPROPS_DIR    = _ROOT / "data/statcast"
CONTEXT_FILE  = _ROOT / "data/orchestrator/daily_context.parquet"
CORR_MATRIX   = _ROOT / "data/corr/correlation_matrix.json"
MODEL_A2      = _ROOT / "models/script_a2_v1.json"
MODEL_B       = _ROOT / "models/script_b_v1.json"
MODEL_C       = _ROOT / "models/script_c_v1.json"
OUT_DIR       = _ROOT / "data/sgp"

BOOK_CORR_TAX_STANDARD = 0.15   # DraftKings / FanDuel
BOOK_CORR_TAX_SHARP    = 0.20   # Pinnacle
MIN_EDGE_DEFAULT       = 0.05
F5_FRACTION            = 0.571  # empirical from correlation_matrix (was 0.56)

# Script D gate thresholds (v4.5)
BP_BURN_5D_75TH   = 280   # approx 75th pctile of home_bullpen_burn_5d (2026)
STUFF_ELITE_PCTILE = 0.70  # whiff_pctl > this = elite SP (Stuff+ proxy)


# ── American odds helpers ─────────────────────────────────────────────────────
def _implied(odds: float) -> float:
    """American odds -> raw (overround-inclusive) implied probability."""
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return 0.5
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _fair(odds: float, other_odds: float) -> float:
    """Devig two-way market to get fair probability for the first leg."""
    p1 = _implied(odds)
    p2 = _implied(other_odds)
    total = p1 + p2
    return p1 / total if total > 0 else 0.5


# ── Correlation matrix loader ─────────────────────────────────────────────────
_CORR_CACHE: dict | None = None


def _load_corr() -> dict:
    global _CORR_CACHE
    if _CORR_CACHE is None:
        if not CORR_MATRIX.exists():
            print(f"  [corr] {CORR_MATRIX} not found — run correlation_matrix.py --build")
            _CORR_CACHE = {}
        else:
            with open(CORR_MATRIX) as f:
                _CORR_CACHE = json.load(f).get("pairs", {})
    return _CORR_CACHE


def _rho(pair: str) -> float:
    return _load_corr().get(pair, {}).get("rho", 0.0)


# ── Gaussian copula (inline, no external dependency on correlation_matrix.py) ─
def _copula_joint_2way(p_a: float, p_b: float, r: float) -> float:
    """P(A AND B) via bivariate Gaussian copula with correlation r."""
    from scipy import stats
    eps = 1e-6
    p_a = float(np.clip(p_a, eps, 1 - eps))
    p_b = float(np.clip(p_b, eps, 1 - eps))
    z_a = float(stats.norm.ppf(p_a))
    z_b = float(stats.norm.ppf(p_b))
    r   = float(np.clip(r, -0.9999, 0.9999))
    cov = [[1.0, r], [r, 1.0]]
    return float(stats.multivariate_normal.cdf([z_a, z_b], mean=[0, 0], cov=cov))


def _copula_joint_3way(p_a: float, p_b: float, p_c: float,
                        r_ab: float, r_ac: float, r_bc: float) -> float:
    """P(A AND B AND C) via trivariate Gaussian copula (50k MC)."""
    from scipy import stats
    eps = 1e-6
    p_a, p_b, p_c = (float(np.clip(p, eps, 1 - eps)) for p in (p_a, p_b, p_c))
    z_a = float(stats.norm.ppf(p_a))
    z_b = float(stats.norm.ppf(p_b))
    z_c = float(stats.norm.ppf(p_c))
    r_ab = float(np.clip(r_ab, -0.9999, 0.9999))
    r_ac = float(np.clip(r_ac, -0.9999, 0.9999))
    r_bc = float(np.clip(r_bc, -0.9999, 0.9999))
    cov = np.array([[1.0, r_ab, r_ac],
                    [r_ab, 1.0, r_bc],
                    [r_ac, r_bc, 1.0]])
    try:
        samples = stats.multivariate_normal.rvs(
            mean=[0, 0, 0], cov=cov, size=50_000, random_state=42)
        return float(np.mean(
            (samples[:, 0] <= z_a) & (samples[:, 1] <= z_b) & (samples[:, 2] <= z_c)))
    except Exception:
        return p_a * p_b * p_c


# ── Data loaders ──────────────────────────────────────────────────────────────
def _load_odds(date_str: str) -> pd.DataFrame:
    tag = date_str.replace("-", "_")
    p = ODDS_DIR / f"odds_current_{tag}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    # Try latest
    files = sorted(glob.glob(str(ODDS_DIR / "odds_current_*.parquet")))
    if files:
        print(f"  [odds] {p.name} not found; using {Path(files[-1]).name}")
        return pd.read_parquet(files[-1])
    return pd.DataFrame()


def _load_kprops(date_str: str) -> pd.DataFrame:
    p = KPROPS_DIR / f"k_props_{date_str}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    files = sorted(glob.glob(str(KPROPS_DIR / "k_props_*.parquet")))
    if files:
        return pd.read_parquet(files[-1])
    return pd.DataFrame()


def _load_context(date_str: str) -> pd.DataFrame:
    if not CONTEXT_FILE.exists():
        return pd.DataFrame()
    ctx = pd.read_parquet(CONTEXT_FILE)
    if "orchestrator_date" in ctx.columns:
        return ctx[ctx["orchestrator_date"] == date_str].copy()
    return ctx.copy()


def _load_models() -> dict:
    try:
        import xgboost as xgb
    except ImportError:
        return {}
    models = {}
    for name, path in (("A2", MODEL_A2), ("B", MODEL_B), ("C", MODEL_C)):
        if path.exists():
            b = xgb.Booster()
            b.load_model(str(path))
            models[name] = b
    return models


# ── K-prop lookup ─────────────────────────────────────────────────────────────
_NO_PROP = (None, None)   # sentinel: no retail line found


def _find_k_prop(kprops: pd.DataFrame, starter_name: str
                 ) -> tuple[float | None, float | None]:
    """Return (p_k_over, line) for the given starter from K-props.

    Returns (None, None) — NOT a default — if the starter has no retail line.
    Callers MUST check for None before using the probability; any script that
    relies on a K-prop leg must gate on prop_matched=True.
    """
    if kprops.empty or not starter_name:
        return _NO_PROP
    last = starter_name.strip().split()[-1].lower()
    mask = kprops["pitcher_name"].str.lower().str.contains(last, na=False)
    matches = kprops[mask]
    if matches.empty:
        return _NO_PROP
    row = matches.iloc[0]
    line       = float(row.get("line", 4.5) or 4.5)
    over_odds  = row.get("over_odds")
    under_odds = row.get("under_odds")
    try:
        if over_odds is not None and not pd.isna(over_odds) and \
           under_odds is not None and not pd.isna(under_odds):
            p = _fair(over_odds, under_odds)
        elif over_odds is not None and not pd.isna(over_odds):
            p = _implied(over_odds)
        else:
            return _NO_PROP
    except (TypeError, ValueError):
        return _NO_PROP
    return float(np.clip(p, 0.05, 0.95)), line


# ── Single-game SGP scorer ────────────────────────────────────────────────────
def score_game_sgp(
    game_row: pd.Series,
    ctx_row: pd.Series | None,
    kprops: pd.DataFrame,
    models: dict,
    book_corr_tax: float = BOOK_CORR_TAX_STANDARD,
    verbose: bool = False,
) -> list[dict]:
    """Score all three scripts for one game. Returns list of result dicts."""

    home = str(game_row.get("home_team", ""))
    away = str(game_row.get("away_team", ""))

    # ── Market leg probabilities ──────────────────────────────────────────
    # Moneyline
    p_home_win  = float(game_row.get("P_true_home") or
                        game_row.get("retail_implied_home") or 0.5)
    p_away_win  = 1.0 - p_home_win

    # Total (devigged if possible)
    p_true_over  = float(game_row.get("P_true_over")  or
                         game_row.get("retail_implied_over") or 0.5)
    p_true_under = float(game_row.get("P_true_under") or
                         game_row.get("retail_implied_under") or 0.5)
    # Re-normalize if doesn't sum to 1
    tot = p_true_over + p_true_under
    p_true_over  = p_true_over  / tot if tot > 0 else 0.5
    p_true_under = p_true_under / tot if tot > 0 else 0.5

    close_total = float(game_row.get("close_total") or 9.0)

    # RL +1.5 away (close-game proxy): use rl_away implied
    def _safe_odds(val):
        if val is None: return None
        try:
            if pd.isna(val): return None
        except (TypeError, ValueError): pass
        return val
    rl_away_odds = _safe_odds(game_row.get("pinnacle_rl_away_odds")) or \
                   _safe_odds(game_row.get("runline_home_odds"))
    p_close_game = _implied(rl_away_odds) if rl_away_odds is not None else 0.30

    # Starter names from context (accurate after v4.3 fix) or fallback
    home_sp = ""
    away_sp = ""
    if ctx_row is not None:
        home_sp = str(ctx_row.get("home_starter_name") or "")
        away_sp = str(ctx_row.get("away_starter_name") or "")

    # K props — None means no retail line exists (MUST gate on this)
    p_home_k_raw, home_k_line = _find_k_prop(kprops, home_sp)
    p_away_k_raw, away_k_line = _find_k_prop(kprops, away_sp)
    home_k_matched = p_home_k_raw is not None
    away_k_matched = p_away_k_raw is not None

    # Ump synergy multipliers (from context)
    synergy_home = float(ctx_row.get("ump_k_synergy_home") or 1.0) if ctx_row is not None else 1.0
    synergy_away = float(ctx_row.get("ump_k_synergy_away") or 1.0) if ctx_row is not None else 1.0

    # Apply synergy only when we have a real prop — never on a default
    p_home_k = float(np.clip((p_home_k_raw or 0.45) * synergy_home, 0.05, 0.95))
    p_away_k = float(np.clip((p_away_k_raw or 0.45) * synergy_away, 0.05, 0.95))

    # F5 under (proportional from game total)
    p_f5_under = float(np.clip(p_true_under * 1.04, 0.05, 0.95))

    # ADI context note
    adi = float(ctx_row.get("air_density_rho") or 1.20) if ctx_row is not None else 1.20
    adi_note = f"ADI={adi:.3f} ({'dense/drag->under' if adi > 1.18 else 'light/carry->over'})"

    # Bullpen gassed flags (from context)
    home_bp_gassed = int(ctx_row.get("home_bp_gassed") or 0) if ctx_row is not None else 0
    away_bp_gassed  = int(ctx_row.get("away_bp_gassed")  or 0) if ctx_row is not None else 0

    results = []

    # ── Script A2: SP_K_F5>=4 | F5_Under | Game_Under ────────────────────
    # INTEGRITY GATE: home SP must have a matched retail K-prop line
    r_ku  = _rho("sp_k_vs_team_total")
    r_kf5 = _rho("sp_k_vs_team_total")
    r_f5g = _rho("f5_total_vs_game_total") or 0.74

    if home_k_matched:
        p_joint_a2_copula = _copula_joint_3way(
            p_home_k, p_f5_under, p_true_under, r_ku, r_kf5, r_f5g)
        p_indep_a2 = p_home_k * p_f5_under * p_true_under
        p_book_a2  = p_indep_a2 * (1.0 - book_corr_tax)
        edge_a2    = (p_joint_a2_copula / p_book_a2 - 1.0) if p_book_a2 > 0 else 0.0
        a2_action  = "PLAY" if edge_a2 >= MIN_EDGE_DEFAULT else "pass"
    else:
        # No retail line — cannot compute a real edge; force pass
        p_joint_a2_copula = p_indep_a2 = p_book_a2 = edge_a2 = 0.0
        a2_action = "NO_PROP"

    results.append({
        "game":             f"{away} @ {home}",
        "home_team":        home,
        "away_team":        away,
        "script":           "A2_Dominance",
        "legs":             f"Home_SP_K_F5>=4({home_sp}) | F5_Under_{close_total * F5_FRACTION:.1f} | Game_Under_{close_total}",
        "home_sp":          home_sp,
        "away_sp":          away_sp,
        "close_total":      close_total,
        "home_k_line":      home_k_line,
        "home_k_matched":   home_k_matched,
        "p_leg_k":          round(p_home_k, 4) if home_k_matched else None,
        "p_leg_f5_under":   round(p_f5_under, 4),
        "p_leg_game_under": round(p_true_under, 4),
        "p_joint_copula":   round(p_joint_a2_copula, 4),
        "p_joint_indep":    round(p_indep_a2, 4),
        "p_book_sgp":       round(p_book_a2, 4),
        "sgp_edge":         round(edge_a2, 4),
        "corr_lift":        round(p_joint_a2_copula / p_indep_a2, 3) if p_indep_a2 > 0 else None,
        "synergy_home":     round(synergy_home, 2),
        "home_bp_gassed":   home_bp_gassed,
        "adi_note":         adi_note,
        "book_corr_tax":    book_corr_tax,
        "action":           a2_action,
    })

    # ── Script B: Home_Score>=5 | Game_Over | Home_ML_Win ────────────────
    # Script B uses game total + ML (no pitcher K-prop needed) — always scoreable
    r_score_total = _rho("game_total_vs_home_score")
    p_home_score5 = float(np.clip(p_home_win * 0.80, 0.05, 0.95))

    p_joint_b_copula = _copula_joint_3way(
        p_home_score5, p_true_over, p_home_win,
        r_score_total, r_score_total, r_score_total)
    p_indep_b  = p_home_score5 * p_true_over * p_home_win
    p_book_b   = p_indep_b * (1.0 - book_corr_tax)
    edge_b     = (p_joint_b_copula / p_book_b - 1.0) if p_book_b > 0 else 0.0
    # Boost Script B when away bullpen is gassed (late-innings run scoring elevated)
    b_note = " [away_BP_GASSED]" if away_bp_gassed else ""

    results.append({
        "game":             f"{away} @ {home}",
        "home_team":        home,
        "away_team":        away,
        "script":           "B_Explosion",
        "legs":             f"Home_Score>=5 | Game_Over_{close_total} | {home}_ML_Win{b_note}",
        "home_sp":          home_sp,
        "away_sp":          away_sp,
        "close_total":      close_total,
        "home_k_matched":   True,   # B doesn't use K-prop
        "p_leg_home_score5":  round(p_home_score5, 4),
        "p_leg_game_over":    round(p_true_over, 4),
        "p_leg_home_win":     round(p_home_win, 4),
        "p_joint_copula":     round(p_joint_b_copula, 4),
        "p_joint_indep":      round(p_indep_b, 4),
        "p_book_sgp":         round(p_book_b, 4),
        "sgp_edge":           round(edge_b, 4),
        "corr_lift":          round(p_joint_b_copula / p_indep_b, 3) if p_indep_b > 0 else None,
        "synergy_home":       round(synergy_home, 2),
        "away_bp_gassed":     away_bp_gassed,
        "adi_note":           adi_note,
        "book_corr_tax":      book_corr_tax,
        "action":             "PLAY" if edge_b >= MIN_EDGE_DEFAULT else "pass",
    })

    # ── Script C: Game_Under | Both_SP_K>=3 | Close_Game(+1.5) ──────────
    # INTEGRITY GATE: BOTH starters must have matched retail K-prop lines
    r_both_k_close = _rho("both_sp_k_vs_close_game")
    r_k_under      = _rho("sp_k_vs_team_total")

    if home_k_matched and away_k_matched:
        p_both_k = _copula_joint_2way(p_home_k, p_away_k, r_k_under)
        p_joint_c_copula = _copula_joint_3way(
            p_true_under, p_both_k, p_close_game,
            r_k_under, r_both_k_close, r_both_k_close)
        p_indep_c  = p_true_under * p_both_k * p_close_game
        p_book_c   = p_indep_c * (1.0 - book_corr_tax)
        edge_c     = (p_joint_c_copula / p_book_c - 1.0) if p_book_c > 0 else 0.0
        c_action   = "PLAY" if edge_c >= MIN_EDGE_DEFAULT else "pass"
    else:
        # One or both starters missing retail line
        missing = []
        if not home_k_matched: missing.append(home_sp or "home_SP")
        if not away_k_matched: missing.append(away_sp or "away_SP")
        p_joint_c_copula = p_indep_c = p_book_c = edge_c = 0.0
        p_both_k = 0.0
        c_action = f"NO_PROP:{','.join(missing)}"

    results.append({
        "game":             f"{away} @ {home}",
        "home_team":        home,
        "away_team":        away,
        "script":           "C_EliteDuel",
        "legs":             f"Game_Under_{close_total} | Both_SP_K_F5>=3({home_sp}/{away_sp}) | {away}_RL+1.5",
        "home_sp":          home_sp,
        "away_sp":          away_sp,
        "close_total":      close_total,
        "home_k_line":      home_k_line,
        "away_k_line":      away_k_line,
        "home_k_matched":   home_k_matched,
        "away_k_matched":   away_k_matched,
        "p_leg_game_under": round(p_true_under, 4),
        "p_leg_both_sp_k":  round(p_both_k, 4) if (home_k_matched and away_k_matched) else None,
        "p_leg_close_game": round(p_close_game, 4),
        "p_joint_copula":   round(p_joint_c_copula, 4),
        "p_joint_indep":    round(p_indep_c, 4),
        "p_book_sgp":       round(p_book_c, 4),
        "sgp_edge":         round(edge_c, 4),
        "corr_lift":        round(p_joint_c_copula / p_indep_c, 3) if p_indep_c > 0 else None,
        "synergy_home":     round(synergy_home, 2),
        "synergy_away":     round(synergy_away, 2),
        "home_bp_gassed":   home_bp_gassed,
        "adi_note":         adi_note,
        "book_corr_tax":    book_corr_tax,
        "action":           c_action,
    })

    # ── Script D: F5_Under | Game_Over (Late Inning Divergence) ────────────
    # GATE: (home OR away bullpen_burn_5d > 75th pctile) AND elite SP
    # Thesis: dominant SP suppresses F5 total; gassed bullpen surrenders
    # late runs → Full Game Over fires after the F5 result is set.
    # Books price F5_Under and Game_Over as negatively correlated (~r=-0.74).
    # In the gassed-bullpen + elite-SP universe, the relationship flips toward
    # mildly positive (r≈+0.10), creating structural alpha.
    home_burn_5d = float(ctx_row.get("home_bullpen_burn_5d") or 0) if ctx_row is not None else 0.0
    away_burn_5d = float(ctx_row.get("away_bullpen_burn_5d") or 0) if ctx_row is not None else 0.0
    home_whiff   = float(ctx_row.get("home_sp_whiff_pctl")   or 0) if ctx_row is not None else 0.0
    away_whiff   = float(ctx_row.get("away_sp_whiff_pctl")   or 0) if ctx_row is not None else 0.0

    bullpen_gassed_d = (home_burn_5d > BP_BURN_5D_75TH) or (away_burn_5d > BP_BURN_5D_75TH)
    elite_sp_d       = (home_whiff > STUFF_ELITE_PCTILE) or (away_whiff > STUFF_ELITE_PCTILE)
    script_d_gate    = bullpen_gassed_d and elite_sp_d

    if script_d_gate:
        r_f5u_go = _rho("f5_under_vs_game_over_conditional") or 0.10
        p_joint_d_copula = _copula_joint_2way(p_f5_under, p_true_over, r_f5u_go)
        p_indep_d = p_f5_under * p_true_over
        p_book_d  = p_indep_d * (1.0 - book_corr_tax)
        edge_d    = (p_joint_d_copula / p_book_d - 1.0) if p_book_d > 0 else 0.0
        d_action  = "PLAY" if edge_d >= MIN_EDGE_DEFAULT else "pass"
        d_note    = (f"bp5d_home={home_burn_5d:.0f} away={away_burn_5d:.0f} | "
                     f"whiff_home={home_whiff:.2f} away={away_whiff:.2f}")
    else:
        p_joint_d_copula = p_indep_d = p_book_d = edge_d = 0.0
        d_action = "GATE"
        d_note   = ("gate_not_met: "
                    + ("bullpen_ok" if not bullpen_gassed_d else "")
                    + ("sp_not_elite" if not elite_sp_d else ""))

    results.append({
        "game":              f"{away} @ {home}",
        "home_team":         home,
        "away_team":         away,
        "script":            "D_LateDivergence",
        "legs":              f"F5_Under_{close_total * F5_FRACTION:.1f} | Game_Over_{close_total}",
        "home_sp":           home_sp,
        "away_sp":           away_sp,
        "close_total":       close_total,
        "home_k_matched":    True,
        "script_d_gate":     script_d_gate,
        "home_burn_5d":      round(home_burn_5d, 1),
        "away_burn_5d":      round(away_burn_5d, 1),
        "p_leg_f5_under":    round(p_f5_under, 4),
        "p_leg_game_over":   round(p_true_over, 4),
        "p_joint_copula":    round(p_joint_d_copula, 4),
        "p_joint_indep":     round(p_indep_d, 4),
        "p_book_sgp":        round(p_book_d, 4),
        "sgp_edge":          round(edge_d, 4),
        "corr_lift":         round(p_joint_d_copula / p_indep_d, 3) if p_indep_d > 0 else None,
        "synergy_home":      round(synergy_home, 2),
        "home_bp_gassed":    home_bp_gassed,
        "away_bp_gassed":    away_bp_gassed,
        "adi_note":          adi_note,
        "book_corr_tax":     book_corr_tax,
        "d_note":            d_note,
        "action":            d_action,
    })

    return results


# ── Full slate scorer ─────────────────────────────────────────────────────────
def build_live_edge_report(
    date_str: str,
    min_edge: float = MIN_EDGE_DEFAULT,
    verbose: bool = True,
) -> pd.DataFrame:
    """Score today's full slate. Returns DataFrame sorted by SGP Edge."""

    if verbose:
        print("=" * 64)
        print(f"  SGP Live Edge Report  [{date_str}]")
        print("=" * 64)

    odds    = _load_odds(date_str)
    kprops  = _load_kprops(date_str)
    context = _load_context(date_str)
    models  = _load_models()
    corr    = _load_corr()

    if odds.empty:
        print("  [ERROR] No odds data — run odds_current_pull.py first")
        return pd.DataFrame()

    n_models = len(models)
    n_corr   = len(corr)
    if verbose:
        print(f"  Games: {len(odds)}  Models: {n_models}  Corr pairs: {n_corr}")
        if not kprops.empty:
            print(f"  K-props: {len(kprops)} rows ({date_str})")
        else:
            print(f"  K-props: none for {date_str} — using default p=0.45")

    all_records = []
    for _, game_row in odds.iterrows():
        home = str(game_row.get("home_team", ""))
        ctx_row = None
        if not context.empty and "home_team" in context.columns:
            match = context[context["home_team"] == home]
            if not match.empty:
                ctx_row = match.iloc[0]

        records = score_game_sgp(game_row, ctx_row, kprops, models, verbose=False)
        all_records.extend(records)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df = df.sort_values("sgp_edge", ascending=False).reset_index(drop=True)
    plays = df[df["sgp_edge"] >= min_edge]

    if verbose:
        print(f"\n  PLAYS (edge >= {min_edge:.0%}): {len(plays)} / {len(df)}\n")
        header = f"  {'#':>3}  {'GAME':<22} {'SCRIPT':<16} {'P_COPULA':>8} {'P_BOOK':>8} {'EDGE':>8}  {'K_SYNERGY':>9}  ACTION"
        print(header)
        print("  " + "-" * 85)
        for rank, (_, row) in enumerate(plays.iterrows(), 1):
            syn = f"{row.get('synergy_home', 1.0):.1f}x"
            print(f"  #{rank:<3}  {row['game']:<22} {row['script']:<16} "
                  f"{row['p_joint_copula']:>8.4f} {row['p_book_sgp']:>8.4f} "
                  f"{row['sgp_edge']:>+8.4f}  {syn:>9}  {row['action']}")

        if not plays.empty:
            top = plays.iloc[0]
            print(f"\n  TOP PLAY:")
            print(f"    Game:       {top['game']}")
            print(f"    Script:     {top['script']}")
            print(f"    Legs:       {top.get('legs', '')}")
            print(f"    SP:         {top.get('home_sp','')} (home) vs {top.get('away_sp','')} (away)")
            print(f"    P(Copula):  {top['p_joint_copula']:.4f}  (corr-lift: {top.get('corr_lift',1.0):.3f}x)")
            print(f"    P(Indep):   {top['p_joint_indep']:.4f}")
            print(f"    P(Book):    {top['p_book_sgp']:.4f}  (after {top['book_corr_tax']:.0%} corr tax)")
            print(f"    SGP Edge:   {top['sgp_edge']:+.4f}  ({top['sgp_edge']:.1%})")
            print(f"    ADI:        {top.get('adi_note','')}")
            print(f"    Ump Synergy:{top.get('synergy_home', 1.0):.1f}x (home SP)")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = date_str.replace("-", "_")
    csv_path  = OUT_DIR / f"sgp_live_edge_{tag}.csv"
    json_path = OUT_DIR / f"sgp_live_edge_{tag}.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    if verbose:
        print(f"\n  Report -> {csv_path}")

    return df


# ── Detailed game audit ───────────────────────────────────────────────────────
def audit_game(home_team: str, away_starter: str, date_str: str) -> None:
    """Deep audit: all three scripts for one specific game + ump synergy flag."""
    print("=" * 64)
    print(f"  DETAILED SGP AUDIT: {away_starter} @ {home_team}  [{date_str}]")
    print("=" * 64)

    odds    = _load_odds(date_str)
    kprops  = _load_kprops(date_str)
    context = _load_context(date_str)
    corr    = _load_corr()
    models  = _load_models()

    game_row = None
    if not odds.empty:
        match = odds[odds["home_team"].str.upper() == home_team.upper()]
        if not match.empty:
            game_row = match.iloc[0]

    if game_row is None:
        print(f"  [ERROR] No odds found for home_team={home_team}")
        return

    ctx_row = None
    if not context.empty:
        match = context[context["home_team"].str.upper() == home_team.upper()]
        if not match.empty:
            ctx_row = match.iloc[0]

    print(f"\n  Correlation Matrix (r-values in use):")
    for pair, meta in corr.items():
        print(f"    {pair}: r={meta['rho']:+.4f}  n={meta.get('n_obs','?')}")

    records = score_game_sgp(game_row, ctx_row, kprops, models, verbose=True)

    print(f"\n  Script Results:")
    for rec in records:
        print(f"\n    [{rec['script']}]")
        print(f"      Legs:       {rec['legs']}")
        print(f"      P(Copula):  {rec['p_joint_copula']:.4f}  "
              f"Corr-lift={rec.get('corr_lift',1.0):.3f}x vs independence")
        print(f"      P(Book):    {rec['p_book_sgp']:.4f}")
        print(f"      SGP Edge:   {rec['sgp_edge']:+.4f}  ({rec['sgp_edge']:.1%})")
        print(f"      Ump Synergy:{rec.get('synergy_home',1.0):.1f}x (home K mult)")
        print(f"      ADI:        {rec.get('adi_note','n/a')}")
        print(f"      Action:     {rec['action']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGP Live Edge Report")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE_DEFAULT)
    parser.add_argument("--audit-game", metavar="HOME,AWAY_SP",
                        help="Deep audit: 'ATL,Andrew Painter'")
    parser.add_argument("--sharp", action="store_true",
                        help="Use sharp book corr tax (20%% vs 15%%)")
    args = parser.parse_args()

    date_str = args.date
    tax = BOOK_CORR_TAX_SHARP if args.sharp else BOOK_CORR_TAX_STANDARD

    if args.audit_game:
        parts = args.audit_game.split(",", 1)
        if len(parts) == 2:
            audit_game(parts[0].strip(), parts[1].strip(), date_str)
        else:
            print("--audit-game expects 'HOME_TEAM,AWAY_STARTER'")
    else:
        build_live_edge_report(date_str, min_edge=args.min_edge)
