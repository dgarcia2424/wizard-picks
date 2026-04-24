"""
p_complete_game.py — v4.5 Lineup Persistence & Substitution Risk.

Computes P(4+ PA) for each batter in today's lineup using:
    - batting_order_slot: primary PA driver
    - defensive_utility (DRS): high DRS = frequent defensive-sub target in blowouts
    - projected_spread: blowout potential raises sub risk for deep-lineup hitters

Batters with P(4+ PA) < 0.85 are flagged 'high_sub_risk' and excluded from
'Over TB' SGP legs (their over probability is over-stated without a 4th PA).

Usage:
    python p_complete_game.py                         # today
    python p_complete_game.py --date 2026-04-24       # specific date
    from p_complete_game import build_persistence      # import in pipeline

Outputs:
    data/batter_features/lineup_persistence_{date}.parquet
"""
from __future__ import annotations

import argparse
import glob
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent

LINEUPS_DIR   = _ROOT / "data/statcast"
SCORES_FILE   = _ROOT / "model_scores.csv"
DRS_FILE      = _ROOT / "data/raw/fangraphs_batters.csv"
OUT_DIR       = _ROOT / "data/batter_features"

HIGH_SUB_RISK_THRESHOLD = 0.85   # P(4+ PA) below this = high sub risk

# Empirical P(4+ PA) by batting order slot.
# PA per game follows a tight near-deterministic distribution (9 fixed innings),
# NOT Poisson — variance is much lower than Poisson would predict.
# Source: MLB 2022-2025 game logs, all 9-inning games.
_P_4PLUS_BY_SLOT = {1: 0.92, 2: 0.90, 3: 0.88, 4: 0.86, 5: 0.84,
                    6: 0.82, 7: 0.79, 8: 0.76, 9: 0.72}


def _p_4plus_pa(slot: int) -> float:
    """Empirical P(batter gets 4+ PA) for batting order slot 1-9."""
    return _P_4PLUS_BY_SLOT.get(int(slot), 0.80)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_today_lineups(date_str: str) -> pd.DataFrame:
    """Load today's long-format lineup file."""
    p = LINEUPS_DIR / f"lineups_{date_str}_long.parquet"
    if p.exists():
        return pd.read_parquet(p)
    # Fallback: latest available
    files = sorted(glob.glob(str(LINEUPS_DIR / "lineups_*_long.parquet")))
    if files:
        return pd.read_parquet(files[-1])
    return pd.DataFrame()


def _load_drs() -> pd.DataFrame:
    """DRS (Defensive Runs Saved) from FanGraphs batters CSV.
    Returns: [player_name_upper, drs]
    """
    if not DRS_FILE.exists():
        return pd.DataFrame(columns=["player_name_upper", "drs"])
    try:
        fg = pd.read_csv(DRS_FILE, encoding="utf-8-sig")
        fg.columns = fg.columns.str.strip()
        name_col = next((c for c in fg.columns
                         if c.lower() in ("name", "playername", "player")), None)
        drs_col  = next((c for c in fg.columns
                         if c.upper() in ("DRS", "DEF")), None)
        if name_col is None or drs_col is None:
            return pd.DataFrame(columns=["player_name_upper", "drs"])
        fg["player_name_upper"] = fg[name_col].str.upper().str.strip()
        fg["drs"] = pd.to_numeric(fg[drs_col], errors="coerce").fillna(0.0)
        return fg[["player_name_upper", "drs"]].drop_duplicates("player_name_upper")
    except Exception as exc:
        print(f"  [persist] DRS load failed: {exc}")
        return pd.DataFrame(columns=["player_name_upper", "drs"])


def _load_spreads() -> pd.DataFrame:
    """Load today's projected spread from model_scores.csv or odds parquet.
    Returns: [home_team, projected_spread]  (absolute value of win prob gap * 20)
    """
    if not SCORES_FILE.exists():
        return pd.DataFrame(columns=["home_team", "projected_spread"])
    try:
        df = pd.read_csv(SCORES_FILE)
        if "projected_spread" in df.columns and "home_team" in df.columns:
            return (df[["home_team", "projected_spread"]]
                    .dropna(subset=["projected_spread"])
                    .drop_duplicates("home_team"))
        # Derive spread proxy from blend probability
        if "blend_prob" in df.columns and "game" in df.columns:
            df["home_team"] = df["game"].str.split("@").str[-1].str.strip()
            df["projected_spread"] = ((df["blend_prob"].fillna(0.5) - 0.5).abs() * 20)
            return df[["home_team", "projected_spread"]].drop_duplicates("home_team")
    except Exception as exc:
        print(f"  [persist] spread load failed: {exc}")
    return pd.DataFrame(columns=["home_team", "projected_spread"])


# ---------------------------------------------------------------------------
# P(4+ PA) computation
# ---------------------------------------------------------------------------

def build_persistence(date_str: str, verbose: bool = True) -> pd.DataFrame:
    """Compute P(4+ PA) and sub-risk flag for every batter in today's lineup.

    P(4+ PA) model:
        base_p  = _p_4plus_pa(_BASE_EXP_PA[slot])
        spread_penalty = min(0.08, projected_spread / 100) * blowout_factor
            where blowout_factor = 1.0 for slots 1-5, 1.5 for slots 6-9
        drs_penalty = 0.02 if drs > 10 and spread > 3 else 0.0
        final_p = base_p - spread_penalty - drs_penalty
    """
    if verbose:
        print("=" * 60)
        print(f"  p_complete_game.py  [{date_str}]")
        print("=" * 60)

    lineups = _load_today_lineups(date_str)
    if lineups.empty:
        print("  [ERROR] No lineup data — run lineup_pull.py first")
        return pd.DataFrame()

    drs    = _load_drs()
    spread = _load_spreads()

    if verbose:
        print(f"  Lineup rows: {len(lineups)}  DRS entries: {len(drs)}  Spreads: {len(spread)}")

    df = lineups.copy()

    # Normalize slot column
    if "batting_order" in df.columns:
        df["slot"] = pd.to_numeric(df["batting_order"], errors="coerce").clip(1, 9).fillna(5)
    else:
        df["slot"] = 5

    # Base P(4+ PA) from slot (empirical)
    df["base_p4pa"] = df["slot"].map(lambda s: _p_4plus_pa(int(s)))

    # Join spread by home_team
    home_col = next((c for c in df.columns if c.lower() in ("home_team", "home")), None)
    if home_col and not spread.empty:
        df = df.merge(spread.rename(columns={"home_team": home_col}),
                      on=home_col, how="left")
    if "projected_spread" not in df.columns:
        df["projected_spread"] = 0.0
    df["projected_spread"] = df["projected_spread"].fillna(0.0)

    # Join DRS by player name
    if "player_name" in df.columns and not drs.empty:
        df["_name_upper"] = df["player_name"].str.upper().str.strip()
        df = df.merge(drs, left_on="_name_upper", right_on="player_name_upper", how="left")
        df = df.drop(columns=["_name_upper", "player_name_upper"], errors="ignore")
    if "drs" not in df.columns:
        df["drs"] = 0.0
    df["drs"] = df["drs"].fillna(0.0)

    # Spread penalty: deeper-order batters more exposed in blowouts
    blowout_factor = np.where(df["slot"] >= 6, 1.5, 1.0)
    spread_penalty = (df["projected_spread"].clip(upper=10) / 100.0) * blowout_factor

    # DRS penalty: high-DRS players in blowouts are subbed early for defence
    drs_penalty = np.where(
        (df["drs"] > 10) & (df["projected_spread"] > 3), 0.02, 0.0)

    df["p_4plus_pa"]    = (df["base_p4pa"] - spread_penalty - drs_penalty).clip(lower=0.0, upper=1.0)
    df["high_sub_risk"] = (df["p_4plus_pa"] < HIGH_SUB_RISK_THRESHOLD).astype("int8")

    if verbose:
        n_high = df["high_sub_risk"].sum()
        avg_p  = df["p_4plus_pa"].mean()
        print(f"  P(4+ PA) avg: {avg_p:.3f}  high_sub_risk: {n_high}/{len(df)} batters")
        if n_high > 0:
            risky = df[df["high_sub_risk"] == 1][["player_name", "slot", "p_4plus_pa", "drs"]].head(10)
            print()
            print(risky.to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"lineup_persistence_{date_str}.parquet"
    df.to_parquet(out_path, index=False)
    if verbose:
        print(f"\n  Saved -> {out_path}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lineup Persistence & Sub Risk")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--threshold", type=float, default=HIGH_SUB_RISK_THRESHOLD,
                        help="P(4+ PA) below this = high sub risk")
    args = parser.parse_args()

    if args.threshold != HIGH_SUB_RISK_THRESHOLD:
        HIGH_SUB_RISK_THRESHOLD = args.threshold

    result = build_persistence(args.date, verbose=True)
    if not result.empty:
        print()
        print(result[["player_name", "slot", "p_4plus_pa", "high_sub_risk",
                       "drs", "projected_spread"]]
              .sort_values("p_4plus_pa")
              .head(20)
              .to_string(index=False))
