"""
score_props_today.py — Pitching K-Prop Engine.

E[K] = K% * TBF_est, where:
  K%       ≈ (K/9) / 38.5   (league-avg BF/9 proxy)
  TBF_est  = 23 base (typical SP faces ~23 batters in 5.5 IP)
             × 1.03 if wind_vector_out < -5.0 OR temp_f < 50°F
               (wind-in / cold suppresses offense → pitchers stay in longer)

Actionable when E[K] > (k_line + 0.5).

Sources:
  data/raw/fangraphs_pitchers.csv       K/9 per pitcher
  data/statcast/lineups_<date>.parquet  confirmed SPs + team abbrs
  data/statcast/k_props_<date>.parquet  market K lines per SP
  games.csv                             wind_vector_out, temp_f (per home team)

Output: list[dict] rows keyed for model_scores.csv schema, with
        model="PROP_K", bet_type=f"K {line} {pitcher}".

Entrypoint score_props(date_str) returns a DataFrame ready to concat onto
model_scores.csv. CLI runs stand-alone.
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from datetime import date
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent
_PIPE = _ROOT  # repo root == pipeline dir
_DATA = _PIPE / "data"
_RAW  = _DATA / "raw"
_SC   = _DATA / "statcast"

LEAGUE_BF_PER_9   = 38.5   # league-avg batters faced per 9 IP
TBF_BASE          = 23.0   # typical starter batters faced
PHYSICS_MULT      = 1.03
WIND_IN_THRESHOLD = -5.0   # wind_vector_out (mph, CF-axis)
COLD_THRESHOLD    = 50.0   # °F
ACTIONABLE_EDGE   = 0.5    # E[K] must exceed k_line by this much


def _strip_accents(s: str) -> str:
    if not isinstance(s, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).strip().lower()


def _load_pitcher_k9() -> dict[str, float]:
    """Name (accent-normalized, lower) → K/9."""
    path = _RAW / "fangraphs_pitchers.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "Name" not in df.columns or "K/9" not in df.columns:
        return {}
    df["_key"] = df["Name"].map(_strip_accents)
    df = df.drop_duplicates("_key", keep="first")
    return dict(zip(df["_key"], pd.to_numeric(df["K/9"], errors="coerce")))


def _load_k_lines(date_str: str) -> pd.DataFrame:
    """Median line + best over_odds per pitcher across books."""
    path = _SC / f"k_props_{date_str}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["pitcher_key", "k_line", "over_odds", "under_odds"])
    df = pd.read_parquet(path)
    df["pitcher_key"] = df["pitcher_name"].map(_strip_accents)
    agg = (df.groupby("pitcher_key", as_index=False)
             .agg(pitcher_name=("pitcher_name", "first"),
                  k_line=("line", "median"),
                  over_odds=("over_odds", "max"),     # best price for the bettor
                  under_odds=("under_odds", "max")))
    return agg


def _load_lineups(date_str: str) -> pd.DataFrame:
    path = _SC / f"lineups_{date_str}.parquet"
    if not path.exists():
        path = _SC / "lineups_today.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _load_weather_by_team() -> dict[str, tuple[float | None, float | None]]:
    """home_team (abbr + full name) → (wind_vector_out, temp_f)."""
    games_path = _PIPE / "games.csv"
    if not games_path.exists():
        return {}
    df = pd.read_csv(games_path)
    if "home_team" not in df.columns:
        return {}
    # games.csv carries full names; key both full + abbr so lineups (abbr) match.
    sys.path.insert(0, str(_PIPE / "wizard_agents"))
    try:
        from tools.implementations import TEAM_NAME_TO_ABBR
    except Exception:
        TEAM_NAME_TO_ABBR = {}
    out: dict[str, tuple] = {}
    for _, r in df.iterrows():
        h   = r.get("home_team")
        wvo = r.get("wind_vector_out")
        tmp = r.get("temp_f")
        wvo = float(wvo) if pd.notna(wvo) else None
        tmp = float(tmp) if pd.notna(tmp) else None
        out[h] = (wvo, tmp)
        abbr = TEAM_NAME_TO_ABBR.get(h)
        if abbr:
            out[abbr] = (wvo, tmp)
    return out


def score_props(date_str: str) -> pd.DataFrame:
    k9_map   = _load_pitcher_k9()
    k_lines  = _load_k_lines(date_str)
    lineups  = _load_lineups(date_str)
    wx_by_t  = _load_weather_by_team()

    if lineups.empty or k_lines.empty:
        print(f"[PropScorer] lineups={len(lineups)} k_lines={len(k_lines)} — nothing to score.")
        return pd.DataFrame()

    k_map = dict(zip(k_lines["pitcher_key"], k_lines.to_dict("records")))

    rows: list[dict] = []
    for _, g in lineups.iterrows():
        home_abbr = g.get("home_team")
        away_abbr = g.get("away_team")
        wvo, tmp = wx_by_t.get(home_abbr, (None, None))

        physics = (wvo is not None and wvo < WIND_IN_THRESHOLD) or \
                  (tmp is not None and tmp < COLD_THRESHOLD)
        tbf_est = TBF_BASE * (PHYSICS_MULT if physics else 1.0)

        for side, sp_col, team in (
            ("home", "home_starter_name", home_abbr),
            ("away", "away_starter_name", away_abbr),
        ):
            sp_name = g.get(sp_col)
            if not sp_name or pd.isna(sp_name):
                continue
            key = _strip_accents(sp_name)
            k9 = k9_map.get(key)
            mkt = k_map.get(key)
            if k9 is None or mkt is None or pd.isna(k9) or pd.isna(mkt["k_line"]):
                continue

            k_pct = float(k9) / LEAGUE_BF_PER_9
            e_k   = k_pct * tbf_est
            k_line = float(mkt["k_line"])
            edge_k = e_k - k_line
            actionable = bool(edge_k > ACTIONABLE_EDGE)

            rows.append({
                "date":  date_str,
                "game":  f"{away_abbr} @ {home_abbr}",
                "model": "PROP_K",
                "bet_type": f"K {k_line} {sp_name}",
                "pick_direction": "OVER",
                "model_prob":  round(e_k, 3),
                "P_true":      None,
                "Retail_Implied_Prob": None,
                "edge":        round(edge_k, 3),
                "retail_american_odds": mkt.get("over_odds"),
                "sanity_check_pass": actionable,
                "odds_floor_pass":   True,
                "tier":  1 if actionable else None,
                "dollar_stake":      None,
                "actionable":        actionable,
                # context cols (blank for non-Totals picks, preserved here for traceability)
                "home_bullpen_xfip": None,
                "away_bullpen_xfip": None,
                "bullpen_xfip_diff": None,
                "signal_flags": "WEATHER_K_ADJ" if physics else "",
            })

    df = pd.DataFrame(rows)
    n_act = int(df["actionable"].sum()) if not df.empty else 0
    print(f"[PropScorer] scored {len(df)} K-props | actionable={n_act} "
          f"(edge > +{ACTIONABLE_EDGE}K, physics_mult on wvo<{WIND_IN_THRESHOLD} or temp<{COLD_THRESHOLD}°F)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    args = ap.parse_args()
    date_str = args.date or date.today().isoformat()
    df = score_props(date_str)
    if not df.empty:
        print(df[["game", "bet_type", "model_prob", "edge", "actionable"]].to_string(index=False))


if __name__ == "__main__":
    main()
