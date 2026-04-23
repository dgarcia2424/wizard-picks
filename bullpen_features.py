"""
bullpen_features.py — Attach FanGraphs bullpen quality to today's games.

Adds columns:
    home_bullpen_era, home_bullpen_xfip, home_bullpen_war,
    away_bullpen_era, away_bullpen_xfip, away_bullpen_war,
    bullpen_xfip_diff   (home xFIP - away xFIP; lower home = home edge)

Missing teams → 0.0 (neutral), never NaN — safe for models that don't
tolerate null inputs and for downstream arithmetic.

Consumed by score_ml_today.py and score_run_dist_today.py. The current
production feature_cols.json lists don't include these columns yet, so
the XGB stacks ignore them at inference; they live on the output frame
for the renderer, the auto-reconciler, and the next retrain cycle.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Resolve the bullpen CSV via the canonical FILES registry rather than
# hard-coding a path — keeps this module location-agnostic.
_ROOT = Path(__file__).resolve().parent
_WIZ  = _ROOT / "wizard_agents"
if str(_WIZ) not in sys.path:
    sys.path.insert(0, str(_WIZ))

from config.settings import FILES  # noqa: E402

# Odds-API full team name → FanGraphs 3-letter abbreviation.
# Matches the convention already established in tools/implementations.py
# TEAM_NAME_TO_ABBR — kept local so this module has no cross-pipeline
# imports beyond FILES.
TEAM_NAME_TO_ABBR: dict[str, str] = {
    "Arizona Diamondbacks":  "ARI", "Atlanta Braves":       "ATL",
    "Baltimore Orioles":     "BAL", "Boston Red Sox":       "BOS",
    "Chicago Cubs":          "CHC", "Chicago White Sox":    "CWS",
    "Cincinnati Reds":       "CIN", "Cleveland Guardians":  "CLE",
    "Colorado Rockies":      "COL", "Detroit Tigers":       "DET",
    "Houston Astros":        "HOU", "Kansas City Royals":   "KC",
    "Los Angeles Angels":    "LAA", "Los Angeles Dodgers":  "LAD",
    "Miami Marlins":         "MIA", "Milwaukee Brewers":    "MIL",
    "Minnesota Twins":       "MIN", "New York Mets":        "NYM",
    "New York Yankees":      "NYY", "Oakland Athletics":    "OAK",
    "Athletics":             "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates":   "PIT",
    "San Diego Padres":      "SD",  "San Francisco Giants": "SF",
    "Seattle Mariners":      "SEA", "St. Louis Cardinals":  "STL",
    "Tampa Bay Rays":        "TB",  "Texas Rangers":        "TEX",
    "Toronto Blue Jays":     "TOR", "Washington Nationals": "WSH",
}


def _normalize(team: str) -> str:
    """Accept either full name ('Texas Rangers') or abbr ('TEX')."""
    t = str(team).strip()
    return TEAM_NAME_TO_ABBR.get(t, t.upper())


def _load_bullpen() -> pd.DataFrame:
    """Load fangraphs_bullpen.csv indexed by abbreviation.

    The `Team` column may arrive as either a full name or a FanGraphs
    abbreviation; both are normalized to abbr. Returns empty frame with
    the right schema if the file is missing (sunday_routine will flag it).
    """
    path = FILES["fangraphs_bullpen"]
    cols = ["team_abbr", "bullpen_era", "bullpen_xfip", "bullpen_war"]
    if not path.exists():
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(path)
    df["team_abbr"] = df["Team"].map(_normalize)
    return (
        df[["team_abbr", "ERA", "xFIP", "WAR"]]
        .rename(columns={"ERA": "bullpen_era", "xFIP": "bullpen_xfip", "WAR": "bullpen_war"})
        .drop_duplicates("team_abbr")
    )


def append_bullpen_features(df: pd.DataFrame) -> pd.DataFrame:
    """Join bullpen ERA/xFIP/WAR onto home+away, add bullpen_xfip_diff.

    Non-destructive: returns a copy. `home_team` / `away_team` may be
    full names or abbrs. Missing teams → 0.0 (neutral).
    """
    bp = _load_bullpen()
    out = df.copy()

    out["_home_abbr"] = out["home_team"].map(_normalize)
    out["_away_abbr"] = out["away_team"].map(_normalize)

    home = bp.add_prefix("home_").rename(columns={"home_team_abbr": "_home_abbr"})
    away = bp.add_prefix("away_").rename(columns={"away_team_abbr": "_away_abbr"})

    out = out.merge(home, on="_home_abbr", how="left")
    out = out.merge(away, on="_away_abbr", how="left")

    new_cols = [
        "home_bullpen_era", "home_bullpen_xfip", "home_bullpen_war",
        "away_bullpen_era", "away_bullpen_xfip", "away_bullpen_war",
    ]
    for c in new_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        else:
            out[c] = 0.0

    out["bullpen_xfip_diff"] = out["home_bullpen_xfip"] - out["away_bullpen_xfip"]

    return out.drop(columns=["_home_abbr", "_away_abbr"])
