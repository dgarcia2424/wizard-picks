"""
backfill_processor.py — 2024–2025 historical alpha feature backfill.

Generates, per game_pk, the four alpha features the v2 stackers need:
    - wind_vector_out          (CF-axis wind projection, dome → 0.0)
    - thermal_aging            (team-avg SP age * temp_f)
    - bullpen_xfip_diff        (home − away, derived from bullpen_fg ERA)
    - days_since_opening_day   (T-0 aligned across seasons)

Output: data/raw/historical_alpha_matrix_24_25.parquet
Row count must match the union of backtest_games_{2024,2025}.csv exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.feature_engineering import (
    PARK_CF_AZIMUTH,
    SEASON_OPENING_DAY,
    compute_wind_vector_out,
)

_ROOT = Path(__file__).resolve().parent
_RAW  = _ROOT / "data" / "raw"
_SC   = _ROOT / "data" / "statcast"
_OUT  = _RAW / "historical_alpha_matrix_24_25.parquet"

# Fixed-roof / retractable-closed parks. wind_vector_out forced to 0.0.
DOME_STADIUMS = {"TB", "MIA", "MIL", "ARI", "AZ", "TOR", "SEA", "TEX", "HOU"}

LEAGUE_AVG_SP_AGE = 28.0  # fallback when team-year age missing


def _load_backtest(year: int) -> pd.DataFrame:
    p = _RAW / f"backtest_games_{year}.csv"
    df = pd.read_csv(p)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df["season"]    = year
    return df


def _load_weather(year: int) -> pd.DataFrame:
    p = _SC / f"weather_{year}.parquet"
    w = pd.read_parquet(p)
    w["game_date"] = pd.to_datetime(w["game_date"]).dt.date
    # home_team in weather uses abbr convention; backtest uses same. Normalize.
    w["home_team"] = w["home_team"].astype(str).str.upper().str.strip()
    return w[["game_date", "home_team", "temp_f", "wind_mph", "wind_bearing"]]


def _team_bullpen_era(year: int) -> dict[str, float]:
    """Team → weighted mean bullpen ERA (proxy for xFIP)."""
    p = _SC / f"bullpen_fg_{year}.parquet"
    df = pd.read_parquet(p)
    rp = df[(df["position"] == "P") &
            (df["gamesStarted"].fillna(0) == 0) &
            (df["gamesPitched"].fillna(0) > 0) &
            (df["inningsPitched"].notna())].copy()
    rp["era"] = pd.to_numeric(rp["era"], errors="coerce")
    rp["ip"]  = pd.to_numeric(rp["inningsPitched"], errors="coerce")
    rp = rp.dropna(subset=["era", "ip"])
    rp["wera"] = rp["era"] * rp["ip"]
    agg = rp.groupby("Team").agg(wera=("wera", "sum"), ip=("ip", "sum"))
    agg["bullpen_era"] = agg["wera"] / agg["ip"]
    return agg["bullpen_era"].to_dict()


def _team_sp_age(year: int) -> dict[str, float]:
    """Team → mean age of pitchers with gamesStarted >= 3."""
    p = _SC / f"bullpen_fg_{year}.parquet"
    df = pd.read_parquet(p)
    sp = df[(df["position"] == "P") &
            (df["gamesStarted"].fillna(0) >= 3) &
            (df["age"].notna())]
    if sp.empty:
        return {}
    return sp.groupby("Team")["age"].mean().to_dict()


def _wind_vector_for_row(row) -> float | None:
    home = str(row["home_team"]).upper().strip()
    if home in DOME_STADIUMS:
        return 0.0
    return compute_wind_vector_out(
        wind_mph=row.get("wind_mph"),
        wind_bearing_deg=row.get("wind_bearing"),
        home_team=home,
    )


def _days_since_opening(row) -> int | None:
    od = SEASON_OPENING_DAY.get(int(row["season"]))
    if od is None:
        return None
    gd = row["game_date"]
    return int((gd - od).days)


def build_year(year: int) -> pd.DataFrame:
    bt = _load_backtest(year)
    wx = _load_weather(year)

    bullpen = _team_bullpen_era(year)
    sp_age  = _team_sp_age(year)

    m = bt.merge(wx, on=["game_date", "home_team"], how="left")
    if len(m) != len(bt):
        raise RuntimeError(
            f"[backfill] weather merge changed row count {len(bt)} → {len(m)} for {year}"
        )

    m["wind_vector_out"] = m.apply(_wind_vector_for_row, axis=1)

    # Roof hygiene — any dome row to 0.0 regardless of weather data present.
    dome_mask = m["home_team"].astype(str).str.upper().isin(DOME_STADIUMS)
    m.loc[dome_mask, "wind_vector_out"] = 0.0

    m["home_bullpen_era"] = m["home_team"].map(bullpen)
    m["away_bullpen_era"] = m["away_team"].map(bullpen)
    m["bullpen_xfip_diff"] = m["home_bullpen_era"] - m["away_bullpen_era"]

    m["home_sp_age"] = m["home_team"].map(sp_age).fillna(LEAGUE_AVG_SP_AGE)
    m["away_sp_age"] = m["away_team"].map(sp_age).fillna(LEAGUE_AVG_SP_AGE)
    avg_sp_age = (m["home_sp_age"] + m["away_sp_age"]) / 2.0
    m["thermal_aging"] = avg_sp_age * m["temp_f"]

    m["days_since_opening_day"] = m.apply(_days_since_opening, axis=1)

    keep = [
        "game_pk", "game_date", "season", "home_team", "away_team",
        "temp_f", "wind_mph", "wind_bearing",
        "wind_vector_out", "thermal_aging",
        "home_bullpen_era", "away_bullpen_era", "bullpen_xfip_diff",
        "home_sp_age", "away_sp_age",
        "days_since_opening_day",
    ]
    return m[keep]


def main():
    frames = [build_year(2024), build_year(2025)]
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(_OUT, index=False)

    # Coverage report
    total = len(out)
    wv_cov = out["wind_vector_out"].notna().mean()
    bp_cov = out["bullpen_xfip_diff"].notna().mean()
    ta_cov = out["thermal_aging"].notna().mean()
    dome   = int(out["home_team"].astype(str).str.upper().isin(DOME_STADIUMS).sum())
    print(f"[backfill] wrote {total:,} rows → {_OUT.name}")
    print(f"  wind_vector_out coverage: {wv_cov:.1%} | dome rows (forced 0): {dome:,}")
    print(f"  bullpen_xfip_diff cov   : {bp_cov:.1%}")
    print(f"  thermal_aging cov       : {ta_cov:.1%}")
    print(f"  days_since_opening range: {out['days_since_opening_day'].min()} "
          f"→ {out['days_since_opening_day'].max()}")


if __name__ == "__main__":
    main()
