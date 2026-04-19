"""
build_batter_splits.py
======================
Derive per-batter platoon split stats (vs RHP / vs LHP) from Statcast
pitch-level data using Bayesian shrinkage toward the league average.

Output: data/statcast/batter_splits_{year}.parquet
Columns:
  player_id, year,
  pa_vs_rhp, xwoba_vs_rhp, k_pct_vs_rhp, bb_pct_vs_rhp,
  pa_vs_lhp, xwoba_vs_lhp, k_pct_vs_lhp, bb_pct_vs_lhp,
  platoon_diff   (xwoba_vs_rhp - xwoba_vs_lhp, positive = RHP advantage)

Bayesian shrinkage
------------------
  shrunk = (n * raw + PA_PRIOR * league_avg) / (n + PA_PRIOR)
  PA_PRIOR = 150 — roughly one full season vs one handedness.
  Batters with <PA_PRIOR PAs vs a given hand regress heavily to the mean.

Usage
-----
  python build_batter_splits.py                    # 2023–2026
  python build_batter_splits.py --years 2026       # current year only
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("./data/statcast")
YEARS    = [2023, 2024, 2025, 2026]
PA_PRIOR = 150   # Bayesian shrinkage strength in PA


def _safe_mean(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce").dropna()
    return float(v.mean()) if len(v) > 0 else np.nan


def _shrink(raw: float, n: int, league_avg: float, prior: int = PA_PRIOR) -> float:
    """Bayesian shrinkage toward league average."""
    return (n * raw + prior * league_avg) / (n + prior)


def build_year(year: int, verbose: bool = True) -> pd.DataFrame:
    sc_path = DATA_DIR / f"statcast_{year}.parquet"
    if not sc_path.exists():
        if verbose:
            print(f"  {year}: statcast file not found — skip")
        return pd.DataFrame()

    cols = ["batter", "p_throws", "woba_denom",
            "estimated_woba_using_speedangle", "events"]
    df = pd.read_parquet(sc_path, engine="pyarrow", columns=cols)

    # PA-level rows only (woba_denom > 0)
    df["woba_denom"] = pd.to_numeric(df["woba_denom"], errors="coerce")
    pa = df[df["woba_denom"] > 0].copy()
    pa["batter"] = pd.to_numeric(pa["batter"], errors="coerce")
    pa = pa.dropna(subset=["batter"])
    pa["batter"] = pa["batter"].astype(int)

    pa["xwoba"] = pd.to_numeric(pa["estimated_woba_using_speedangle"], errors="coerce")
    pa["is_k"]  = (pa["events"] == "strikeout").astype(float)
    pa["is_bb"] = (pa["events"] == "walk").astype(float)

    # League averages
    lg_xwoba = _safe_mean(pa["xwoba"])
    lg_k     = float(pa["is_k"].mean())
    lg_bb    = float(pa["is_bb"].mean())

    rows = []
    for hand in ["R", "L"]:
        sub = pa[pa["p_throws"] == hand]
        grp = sub.groupby("batter").agg(
            pa      = ("xwoba",  "count"),
            xwoba   = ("xwoba",  "mean"),
            k_pct   = ("is_k",   "mean"),
            bb_pct  = ("is_bb",  "mean"),
        ).reset_index()

        # Apply Bayesian shrinkage
        grp["xwoba_shrunk"]  = grp.apply(
            lambda r: _shrink(r["xwoba"], r["pa"], lg_xwoba), axis=1)
        grp["k_pct_shrunk"]  = grp.apply(
            lambda r: _shrink(r["k_pct"], r["pa"], lg_k),     axis=1)
        grp["bb_pct_shrunk"] = grp.apply(
            lambda r: _shrink(r["bb_pct"], r["pa"], lg_bb),   axis=1)

        suffix = "rhp" if hand == "R" else "lhp"
        grp = grp.rename(columns={
            "pa":           f"pa_vs_{suffix}",
            "xwoba_shrunk": f"xwoba_vs_{suffix}",
            "k_pct_shrunk": f"k_pct_vs_{suffix}",
            "bb_pct_shrunk":f"bb_pct_vs_{suffix}",
        })[["batter", f"pa_vs_{suffix}", f"xwoba_vs_{suffix}",
             f"k_pct_vs_{suffix}", f"bb_pct_vs_{suffix}"]]
        rows.append(grp)

    out = rows[0].merge(rows[1], on="batter", how="outer")
    out["year"] = year

    # Fill missing hand with league average (batter never faced that hand)
    for suffix, lg in [("rhp", lg_xwoba), ("lhp", lg_xwoba)]:
        out[f"xwoba_vs_{suffix}"] = out[f"xwoba_vs_{suffix}"].fillna(lg)
        out[f"k_pct_vs_{suffix}"] = out[f"k_pct_vs_{suffix}"].fillna(lg_k)
        out[f"bb_pct_vs_{suffix}"] = out[f"bb_pct_vs_{suffix}"].fillna(lg_bb)
        out[f"pa_vs_{suffix}"] = out[f"pa_vs_{suffix}"].fillna(0)

    out["platoon_diff"] = out["xwoba_vs_rhp"] - out["xwoba_vs_lhp"]
    out = out.rename(columns={"batter": "player_id"})

    out_path = DATA_DIR / f"batter_splits_{year}.parquet"
    out.to_parquet(out_path, engine="pyarrow", index=False)

    if verbose:
        n = len(out)
        avg_pa_r = out["pa_vs_rhp"].mean()
        avg_pa_l = out["pa_vs_lhp"].mean()
        platoon_std = out["platoon_diff"].std()
        print(f"  {year}: {n} batters | "
              f"avg PA vs RHP={avg_pa_r:.0f}, vs LHP={avg_pa_l:.0f} | "
              f"platoon_diff std={platoon_std:.4f} → {out_path.name}")

    return out


def main():
    parser = argparse.ArgumentParser(description="Build per-batter platoon splits")
    parser.add_argument("--years", nargs="+", type=int, default=YEARS)
    args = parser.parse_args()

    print("=" * 60)
    print("  build_batter_splits.py — Per-Batter Platoon Splits")
    print("=" * 60)
    for yr in args.years:
        build_year(yr)
    print("\n  Done.")


if __name__ == "__main__":
    main()
