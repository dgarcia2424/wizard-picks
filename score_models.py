"""
MLB Model Scoring Pipeline
Models: MF1i (F1 Total), MF3i (F3 Total), MF5i (F5 ML), MFull (Environmental), M3 (Hit Props)

INPUTS (CSV files — place in same directory as this script):
  - savant_pitchers.csv     : Baseball Savant pitcher leaderboard download
  - savant_batters.csv      : Baseball Savant batter leaderboard download (optional, enables M3)
  - fangraphs_pitchers.csv  : FanGraphs pitching leaderboard download
  - games.csv               : Your game schedule/results (see template)

OUTPUT:
  - model_scores.csv        : Scored output per game per model
  - model_report.html       : Visual summary report
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import urllib.request
import warnings
from datetime import datetime

# ─── CONFIG ──────────────────────────────────────────────────────────────────

SAVANT_FILE    = "savant_pitchers.csv"
FANGRAPHS_FILE = "fangraphs_pitchers.csv"
SAVANT_BATTERS_FILE  = "savant_batters.csv"
FANGRAPHS_BATTERS_FILE = "fangraphs_batters.csv"
GAMES_FILE     = "games.csv"
OUTPUT_CSV       = "model_scores.csv"
OUTPUT_HTML      = "model_report.html"
ODDS_API_KEY     = os.getenv("ODDS_API_KEY", "")
ODDS_API_ENABLED = True

TEAM_NAME_TO_ABBR = {
    "Arizona Diamondbacks": "AZ",   "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",     "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",          "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",       "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",      "Detroit Tigers": "DET",
    "Houston Astros": "HOU",        "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",         "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",       "New York Mets": "NYM",
    "New York Yankees": "NYY",      "Athletics": "ATH",
    "Oakland Athletics": "ATH",     "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",   "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",   "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",         "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}

TEAM_VS_LHP_FILE = "fangraphs_team_vs_lhp.csv"
TEAM_VS_RHP_FILE = "fangraphs_team_vs_rhp.csv"

# Park factor lookup (2025 baseline — replace with downloaded Savant park factors)
PARK_FACTORS = {
    "COL": 1.22, "BOS": 1.08, "CHC": 1.06, "CIN": 1.05, "TEX": 1.04,
    "NYY": 1.03, "PHI": 1.03, "ATL": 1.02, "MIL": 1.01, "STL": 1.00,
    "LAD": 0.99, "HOU": 0.99, "NYM": 0.99, "CLE": 0.98, "TB":  0.98,
    "MIN": 0.97, "SF":  0.97, "DET": 0.97, "KC":  0.97, "TOR": 0.97,
    "MIA": 0.96, "SD":  0.96, "PIT": 0.96, "BAL": 0.96, "WSH": 0.96,
    "SEA": 0.95, "AZ":  0.95, "LAA": 0.95, "ATH": 0.94, "CWS": 0.94,
}

# Month scalar — seasonal run environment adjustment
# April: pitcher-favored, cold weather, small samples
# Peaks July-August: heat, fatigue, lineup familiarity
MONTH_SCALAR = {
    1:  0.92, 2:  0.92, 3:  0.92,   # Spring / Opening Day
    4:  0.92,                         # April — suppress offense
    5:  0.97,                         # May — normalizing
    6:  1.00,                         # June — baseline
    7:  1.03,                         # July — heat + fatigue
    8:  1.04,                         # August — peak offense
    9:  1.01,                         # September — roster expansion noise
    10: 1.00,                         # Postseason — neutral
}

# Bayesian stat stabilization — PA needed for full confidence in each stat
# Below threshold → shrink toward league mean
STABILIZATION_PA = {
    "k_pct":       60,    # K% stabilizes fast
    "bb_pct":      120,   # BB% takes longer
    "hard_hit_pct":200,   # Hard hit% needs bigger sample
    "xwoba":       150,   # xwOBA mid-range
    "babip":       500,   # BABIP almost never stabilizes
    "exit_velo":   200,   # Exit velo needs volume
    "whiff_pct":   100,   # Whiff% mid-range
    "ba":          200,   # BA for batters
    "xba":         150,   # xBA
}

# League mean priors for Bayesian shrinkage
LEAGUE_MEANS = {
    # Pitchers (opponent perspective — higher = worse for pitcher)
    "k_pct":       22.0,
    "bb_pct":       8.5,
    "hard_hit_pct":38.0,
    "xwoba":        0.315,
    "exit_velo":   87.5,
    "whiff_pct":   25.0,
    "fastball_velo":93.5,
    # Batters
    "ba":           0.248,
    "xba":          0.248,
    "babip":        0.298,
    "barrel_pct":   8.0,
    "sweet_spot_pct":32.0,
}

# Ballpark coordinates for weather lookup (lat, lon)
BALLPARK_COORDS = {
    "ARI": (33.4453, -112.0667), "AZ":  (33.4453, -112.0667),
    "ATH": (38.5934, -121.5072),
    "ATL": (33.7350, -84.3900),
    "BAL": (39.2838, -76.6217),
    "BOS": (42.3467,  -71.0972),
    "CHC": (41.9484,  -87.6553),
    "CIN": (39.0979,  -84.5082),
    "CLE": (41.4962,  -81.6852),
    "COL": (39.7559, -104.9942),
    "CWS": (41.8299,  -87.6338),
    "DET": (42.3390,  -83.0485),
    "HOU": (29.7573,  -95.3555),
    "KC":  (39.0517,  -94.4803),
    "LAA": (33.8003, -117.8827),
    "LAD": (34.0739, -118.2400),
    "MIA": (25.7781,  -80.2197),
    "MIL": (43.0280,  -87.9712),
    "MIN": (44.9817,  -93.2776),
    "NYM": (40.7571,  -73.8458),
    "NYY": (40.8296,  -73.9262),
    "PHI": (39.9061,  -75.1665),
    "PIT": (40.4469,  -80.0057),
    "SD":  (32.7076, -117.1570),
    "SEA": (47.5914, -122.3320),
    "SF":  (37.7786, -122.3893),
    "STL": (38.6226,  -90.1928),
    "TB":  (27.7683,  -82.6534),
    "TEX": (32.7512,  -97.0832),
    "TOR": (43.6414,  -79.3894),
    "WSH": (38.8730,  -77.0074),
}

# Stadium orientation — wind direction relative to home plate
# 'out' = wind blowing toward CF (helps offense), 'in' = toward home plate
# 'cross' = perpendicular. Stored as (bearing_of_CF_in_degrees)
# We compute relative wind direction dynamically from wind bearing vs CF bearing
STADIUM_CF_BEARING = {
    "ARI": 0,   "AZ": 0,
    "ATH": 270, "ATL": 25,  "BAL": 60,  "BOS": 95,
    "CHC": 315, "CIN": 0,   "CLE": 350, "COL": 292,
    "CWS": 54,  "DET": 350, "HOU": 25,  "KC":  25,
    "LAA": 220, "LAD": 320, "MIA": 0,   "MIL": 355,
    "MIN": 340, "NYM": 350, "NYY": 330, "PHI": 352,
    "PIT": 305, "SD":  310, "SEA": 335, "SF":  50,
    "STL": 40,  "TB":  0,   "TEX": 20,  "TOR": 0,
    "WSH": 358,
}


# ─── SPECIAL GAME FLAGS ──────────────────────────────────────────────────────

# Coors Field hard fade — Under confidence penalty when COL is home
# Validated: COL home games go Over at ~65%+ rate historically
COORS_UNDER_PENALTY = 0.10   # reduce Under raw_prob by this amount

# Blowout conflict penalty — when MF5 edge is large, Under becomes risky
# Strong favorite winning big kills Under even when pitcher is elite
BLOWOUT_EDGE_THRESHOLD = 25  # Stuff+ score differential
BLOWOUT_UNDER_PENALTY  = 0.08

# Small sample warning thresholds — flag stats based on fewer than these starts
LOW_SAMPLE_STARTS = 3
LOW_SAMPLE_PA     = 50


# Home Field Advantage — MLB home teams win ~54% historically
# HOME_FIELD_BONUS calibrated to 0 from backtest (5,035 games):
#   bonus=0 → 62.9% accuracy (232 picks)
#   bonus=8 → 60.9% accuracy (631 picks)
# Large bonus inflates edges on marginal home games (~57% win rate), diluting
# high-conviction picks (~67% win rate). HOME_FIELD_PROB_BUMP handles the
# probability adjustment without distorting edge direction/threshold.
# HOME_FIELD_BONUS calibrated via calibrate_weights.py (5,035 games, probability threshold):
#   bonus=4  → 60.7%  (866 picks)   ← best if lower threshold wanted
#   bonus=8  → 60.9%  (631 picks)   ← current — best accuracy/volume balance
#   bonus=0  → 58.7%  (1,458 picks) ← floods in marginal AWAY picks, hurts accuracy
# Note: bonus asymmetrically suppresses away picks (away needs +28 edge vs home +12)
# which correctly reflects that home teams have structural scoring advantages.
HOME_FIELD_BONUS     = 0.0    # Recalibrated 2026-04-11: bonus=0 → 62.9% acc (232 picks) vs bonus=8 → 60.9% (631 picks)
HOME_FIELD_PROB_BUMP = 0.035  # Raw probability bump for home team

# Validated pick threshold — backtest shows 73.8% accuracy at >= 63%
# Below this: marginal, noisy, not worth standalone bet
PICK_THRESHOLD = 0.63

# Neutral zone — 5,035 game backtest:
#   edge 10-20: 56.9% home win rate (real signal, insufficient for picks)
#   edge 20-30: 59.4%  (current picks zone)
#   edge 30-50: 66.7%  (high conviction)
NEUTRAL_EDGE_THRESHOLD = 20   # Confirmed via calibrate_edge.py backtest

# Continuous probability — gradient between neutral and strong
# Maps pitcher score edge to probability continuously
# Score edge of 20 → ~60%, 30 → ~63%, 50 → ~68%, 80+ → ~72%
def edge_to_prob(edge_abs):
    """Continuous sigmoid mapping of score edge to win probability.
    Recalibrated 2026-04-11 from 5,030 games (2024+2025 backtest):
      base=0.591, scale=0.400, decay=118.8  R²=0.001
    """
    base = 0.591
    return base + (0.400 * (1 - np.exp(-edge_abs / 118.8)))


# ─── LOAD & MERGE PITCHER DATA ───────────────────────────────────────────────

# Year weights: recency-first, PA-adjusted dynamically at runtime
# Base weights — overridden at runtime by get_season_year_weights() below
YEAR_WEIGHTS = {2026: 0.30, 2025: 0.55, 2024: 0.15}
PA_FULL_SAMPLE = 150   # PA threshold for full current-year weight
PA_FULL_BAT    = 300   # PA threshold for full current-year batter weight


def get_season_year_weights(current_year=2026):
    """
    Shift year weights toward the current season as it progresses.
    Calibrated against 5,035 historical games (2024+2025) — optimum is 20% at
    season start, rising to ~42% by August as current-year sample grows.

    Early April : current=25%, last year=59%, 2yr ago=16%  (calibrated optimum)
    June        : current=34%, last year=55%, 2yr ago=11%
    August      : current=47%, last year=44%, 2yr ago=9%

    Also lowers PA_FULL_SAMPLE as the season matures so individual pitchers
    reach full current-year trust sooner.

    Returns (year_weights_dict, pa_full_sample)
    """
    month = datetime.today().month

    # Season progress: April=0.0, June=0.33, August=0.67, October=1.0
    progress = max(0.0, min(1.0, (month - 4) / 6.0))

    # Current-year base weight: 40% in April → 55% by October
    # Recalibrated 2026-04-11 via calibrate_weights.py (5,030 games):
    #   2025 backtest optimum: 40% current / 47% prior / 13% two-yr
    #   2024 backtest optimum: 50% current / 39% prior / 11% two-yr
    #   April baseline set to 40% (2025 optimum); ramps toward 55% by Oct
    cur_w = 0.40 + progress * 0.15
    remaining = 1.0 - cur_w
    # Prior-year split held at ~74/26 (mirrors original 55/15 ratio)
    weights = {
        current_year:     round(cur_w, 3),
        current_year - 1: round(remaining * 0.786, 3),
        current_year - 2: round(remaining * 0.214, 3),
    }

    # PA threshold: starts high (slow to trust thin samples), drops mid-season
    # April=200 PA, June=150 PA, August=80 PA
    pa_threshold = max(80, int(200 - progress * 120))

    print(f"   Season weights ({datetime.today().strftime('%b %d')}): "
          f"{current_year}={weights[current_year]:.0%}  "
          f"{current_year-1}={weights[current_year-1]:.0%}  "
          f"{current_year-2}={weights[current_year-2]:.0%}  "
          f"| PA threshold={pa_threshold}")

    return weights, pa_threshold


def _parse_names(df, name_field):
    """Convert 'Last, First' or 'First Last' to 'FIRST LAST' uppercase."""
    col = [c for c in df.columns if name_field in c.lower() or c.lower() == "name"][0]
    return df[col].apply(
        lambda x: " ".join(reversed([p.strip() for p in str(x).split(",")])).upper().strip()
        if "," in str(x) else str(x).upper().strip()
    )


def _blend_stats(df, name_col, stat_cols, pa_col, pa_full, years=None):
    """
    Multi-year stat blending — clean simple implementation.
    Avoids pandas index/merge issues by working with plain dicts.
    """
    if years is None:
        years = sorted(YEAR_WEIGHTS.keys(), reverse=True)

    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].isin(years)]

    # Numeric conversion
    avail_cols = [c for c in stat_cols + [pa_col] if c in df.columns]
    for col in avail_cols:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(r"[%,]", "", regex=True).str.strip(),
            errors="coerce"
        )

    current_year = max(years)

    # Collapse traded players (multiple teams same year) — reset index after
    agg_dict = {c: "mean" for c in avail_cols}
    df_agg = df.groupby([name_col, "year"], as_index=False)[avail_cols].mean()

    # Build base year weights
    base_w = {y: YEAR_WEIGHTS.get(y, 0.0) for y in years}
    total_base = sum(base_w.values()) or 1.0
    norm_w = {y: w / total_base for y, w in base_w.items()}

    # Get current-year PA per player as plain dict
    cur_rows = df_agg[df_agg["year"] == current_year]
    if pa_col in cur_rows.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pa_dict = cur_rows.groupby(name_col)[pa_col].mean().to_dict()
    else:
        pa_dict = {}

    # Process each player
    players = df_agg[name_col].unique()
    rows = []
    for player in players:
        pdata = df_agg[df_agg[name_col] == player].set_index("year")
        cur_pa = float(pa_dict.get(player, 0.0) or 0.0)
        pa_scale = min(cur_pa / pa_full, 1.0) if pa_full > 0 else 1.0

        # Adjust current year weight by PA scale
        adj_w = {}
        for y in years:
            if y == current_year:
                adj_w[y] = norm_w.get(y, 0.0) * pa_scale
            else:
                adj_w[y] = norm_w.get(y, 0.0)

        # Redistribute remaining weight proportionally
        w_cur = adj_w.get(current_year, 0.0)
        remaining = 1.0 - w_cur
        prior_base = sum(norm_w.get(y, 0) for y in years if y != current_year)
        for y in years:
            if y != current_year:
                adj_w[y] = (norm_w.get(y, 0) / prior_base * remaining) if prior_base > 0 else 0

        # Normalize
        total_adj = sum(adj_w.values()) or 1.0
        adj_w = {y: w / total_adj for y, w in adj_w.items()}

        row = {name_col: player, "pa_current": cur_pa, "pa_scale": round(pa_scale, 2)}
        for stat in stat_cols:
            if stat not in pdata.columns:
                continue
            wsum = wused = 0.0
            for y, w in adj_w.items():
                if y in pdata.index:
                    val = pdata.loc[y, stat]
                    # Handle duplicate index (shouldn't happen after groupby but be safe)
                    if isinstance(val, pd.Series):
                        val = val.iloc[0]
                    if pd.notna(val):
                        wsum += float(val) * w
                        wused += w
            row[stat] = wsum / wused if wused > 0 else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def load_savant(path):
    """
    Loads Baseball Savant pitcher CSV (multi-year).
    Blends stats across 2022-2026 using PA-adjusted recency weighting.
    Name format: "Last, First" -> "FIRST LAST".
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # Parse names before blending
    df = df.copy()
    df["pitcher_name"] = _parse_names(df, "last_name")

    # Rename raw columns before blend
    col_map = {
        "k_percent":          "k_pct",
        "bb_percent":         "bb_pct",
        "hard_hit_percent":   "hard_hit_pct",
        "exit_velocity_avg":  "exit_velo",
        "avg_best_speed":     "exit_velo_alt",
        "whiff_percent":      "whiff_pct",
        "xwoba":              "xwoba",
        "woba":               "woba",
        "barrel_batted_rate": "barrel_pct",
        "ff_avg_speed":       "fastball_velo",
        "p_era":              "era",
        "pa":                 "pa",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "exit_velo" not in df.columns and "exit_velo_alt" in df.columns:
        df["exit_velo"] = df["exit_velo_alt"]

    stat_cols = [c for c in [
        "k_pct","bb_pct","hard_hit_pct","exit_velo","whiff_pct",
        "xwoba","woba","barrel_pct","fastball_velo","era"
    ] if c in df.columns]

    years_available = df["year"].dropna().unique().tolist() if "year" in df.columns else []
    years_available = sorted([int(y) for y in years_available if int(y) >= 2022], reverse=True)

    if not years_available:
        # No year column — treat as single-year
        df["stuff_plus"] = df.apply(_calc_stuff_proxy, axis=1)
        if "xwoba" in df.columns:
            df["xera"] = (df["xwoba"] * 13.5 - 0.5).clip(1.0, 7.0)
        keep = ["pitcher_name"] + stat_cols
        print(f"   Savant pitchers: single-year ({len(df)} rows)")
        return df[keep].drop_duplicates("pitcher_name")

    pa_col = "pa" if "pa" in df.columns else stat_cols[0]
    blended = _blend_stats(df, "pitcher_name", stat_cols, pa_col, PA_FULL_SAMPLE, years_available)

    current_year = max(years_available)
    n_current = len(df[df["year"] == current_year])
    print(f"   Savant pitchers: blended {years_available} → {len(blended)} pitchers "
          f"({n_current} with {current_year} data)")

    # Derive Stuff+ and xERA on blended values
    blended["stuff_plus"] = blended.apply(_calc_stuff_proxy, axis=1)
    if "xwoba" in blended.columns:
        blended["xera"] = (blended["xwoba"] * 13.5 - 0.5).clip(1.0, 7.0)

    keep = ["pitcher_name","pa_current","pa_scale"] + [c for c in stat_cols + ["stuff_plus","xera"] if c in blended.columns]
    return blended[keep].drop_duplicates("pitcher_name")


def _calc_stuff_proxy(row):
    """Stuff+ proxy from xwOBA, whiff%, hard hit%, fastball velo. Baseline 100."""
    score = 100.0
    if pd.notna(row.get("xwoba")):
        score += (0.320 - row["xwoba"]) * 120
    if pd.notna(row.get("whiff_pct")):
        score += (row["whiff_pct"] - 25.0) * 0.8
    if pd.notna(row.get("hard_hit_pct")):
        score += (38.0 - row["hard_hit_pct"]) * 0.6
    if pd.notna(row.get("fastball_velo")):
        score += (row["fastball_velo"] - 93.5) * 1.2
    return np.clip(score, 70, 145)


def load_fangraphs(path):
    """
    Loads FanGraphs pitcher CSV (multi-year, combined from Excel export).
    Columns: Name, K/9, BB/9, xERA, FIP, xFIP, IP, ERA, year.
    Blends across years using PA-adjusted recency weighting.
    xFIP used as SIERA substitute. K-BB% derived from K/9 - BB/9.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # Normalize name — handle any column format FanGraphs exports
    # Try common name column variants
    name_col = None
    for candidate in ["Name", "name", "PlayerName", "player_name", "Player", "player"]:
        if candidate in df.columns:
            name_col = candidate
            break
    if name_col is None:
        # Last resort — use first string column
        str_cols = [c for c in df.columns if str(df[c].dtype) == "object"]
        name_col = str_cols[0] if str_cols else df.columns[0]
        print(f"   ⚠️  FanGraphs: no name column found, using '{name_col}'")
    df["pitcher_name"] = df[name_col].astype(str).str.upper().str.strip()

    col_map = {
        "K/9":      "k9",
        "BB/9":     "bb9",
        "xFIP":     "siera",
        "FIP":      "fip",
        "xERA":     "xera_fg",
        "IP":       "ip",
        "ERA":      "era",
        "BABIP":    "babip_fg",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    stat_cols = [c for c in ["k9","bb9","siera","fip","xera_fg","ip","era","babip_fg"] if c in df.columns]

    for col in stat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Multi-year blend if year column present
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        years_available = sorted([int(y) for y in df["year"].dropna().unique() if int(y) >= 2022], reverse=True)
        # Use IP as PA proxy for pitchers
        ip_col = "ip" if "ip" in df.columns else stat_cols[0]
        print(f"   Blending FanGraphs pitchers {years_available}...", end="", flush=True)
        blended = _blend_stats(df, "pitcher_name", stat_cols, ip_col, 100, years_available)
        current_year = max(years_available)
        n_cur = len(df[df["year"] == current_year])
        print(f" done → {len(blended)} pitchers ({n_cur} with {current_year} data)")
    else:
        blended = df.copy()
        print(f"   FanGraphs pitchers: single-year ({len(blended)} rows)")

    # Derive K-BB% from blended K9/BB9
    if "k9" in blended.columns and "bb9" in blended.columns:
        blended["k_bb_pct"] = blended["k9"] - blended["bb9"]

    keep = ["pitcher_name"] + [c for c in stat_cols + ["k_bb_pct"] if c in blended.columns]
    return blended[keep].drop_duplicates("pitcher_name")


def load_fangraphs_batters(path):
    """
    Loads FanGraphs batter CSV (multi-year, combined from Excel export).
    Columns: Name, PA, BB%, K%, BABIP, AVG, OBP, SLG, wOBA, xwOBA, wRC+, year.
    Blends across years using PA-adjusted recency weighting.
    Used to supplement Savant batter data in M3.
    """
    if not os.path.exists(path):
        print(f"   ⚠️  FanGraphs batter file not found ({path}) — M3 will use Savant only")
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    name_col = None
    for candidate in ["Name", "name", "PlayerName", "player_name", "Player", "player"]:
        if candidate in df.columns:
            name_col = candidate
            break
    if name_col is None:
        str_cols = [c for c in df.columns if str(df[c].dtype) == "object"]
        name_col = str_cols[0] if str_cols else df.columns[0]
    df["batter_name"] = df[name_col].astype(str).str.upper().str.strip()

    col_map = {
        "BB%":    "bb_pct_fg",
        "K%":     "k_pct_fg",
        "BABIP":  "babip_fg",
        "AVG":    "ba_fg",
        "wOBA":   "woba_fg",
        "xwOBA":  "xwoba_fg",
        "wRC+":   "wrc_plus",
        "PA":     "pa",
        "OBP":    "obp_fg",
        "SLG":    "slg_fg",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    stat_cols = [c for c in ["bb_pct_fg","k_pct_fg","babip_fg","ba_fg","woba_fg","xwoba_fg","wrc_plus","obp_fg","slg_fg"] if c in df.columns]

    # Convert pct strings (BB%, K% come as "8.5%" or 0.085)
    for col in ["bb_pct_fg","k_pct_fg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace("%","",regex=False).str.strip(), errors="coerce"
            )
            # FanGraphs stores as decimal (0.085) — convert to pct if < 1
            if df[col].median() < 1.0:
                df[col] = df[col] * 100

    for col in stat_cols:
        if col not in ["bb_pct_fg","k_pct_fg"]:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace("%","").str.strip(), errors="coerce")

    if "year" in df.columns and "pa" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        years_available = sorted([int(y) for y in df["year"].dropna().unique() if int(y) >= 2022], reverse=True)
        blended = _blend_stats(df, "batter_name", stat_cols, "pa", PA_FULL_BAT, years_available)
        current_year = max(years_available)
        n_cur = len(df[df["year"] == current_year])
        print(f"   FanGraphs batters:  blended {years_available} → {len(blended)} batters ({n_cur} with {current_year} data)")
    else:
        blended = df.copy()
        print(f"   FanGraphs batters: single-year ({len(blended)} rows)")

    keep = ["batter_name"] + [c for c in stat_cols if c in blended.columns]
    return blended[keep].drop_duplicates("batter_name")



def load_batters(path):
    """
    Loads Baseball Savant batter CSV (multi-year).
    Blends stats across 2022-2026 using PA-adjusted recency weighting.
    PA_FULL_BAT = 300 PA threshold for full current-year trust.
    """
    if not os.path.exists(path):
        print(f"   ⚠️  Batter file not found ({path}) — M3 will be skipped")
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    df["batter_name"] = _parse_names(df, "last_name")

    col_map = {
        "batting_avg":        "ba",
        "k_percent":          "k_pct",
        "bb_percent":         "bb_pct",
        "exit_velocity_avg":  "exit_velo",
        "hard_hit_percent":   "hard_hit_pct",
        "babip":              "babip",
        "xwoba":              "xwoba",
        "xba":                "xba",
        "barrel_batted_rate": "barrel_pct",
        "sweet_spot_percent": "sweet_spot_pct",
        "pa":                 "pa",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    stat_cols = [c for c in [
        "ba","k_pct","bb_pct","exit_velo","hard_hit_pct",
        "babip","xwoba","xba","barrel_pct","sweet_spot_pct"
    ] if c in df.columns]

    years_available = df["year"].dropna().unique().tolist() if "year" in df.columns else []
    years_available = sorted([int(y) for y in years_available if int(y) >= 2022], reverse=True)

    if not years_available:
        df["batter_name"] = _parse_names(df, "last_name")
        keep = ["batter_name"] + stat_cols
        print(f"   Savant batters: single-year ({len(df)} rows)")
        return df[keep].drop_duplicates("batter_name")

    pa_col = "pa" if "pa" in df.columns else stat_cols[0]
    blended = _blend_stats(df, "batter_name", stat_cols, pa_col, PA_FULL_BAT, years_available)

    current_year = max(years_available)
    n_current = len(df[df["year"] == current_year])
    print(f"   Savant batters:  blended {years_available} → {len(blended)} batters "
          f"({n_current} with {current_year} data)")

    keep = ["batter_name","pa_current","pa_scale"] + [c for c in stat_cols if c in blended.columns]
    return blended[keep].drop_duplicates("batter_name")


def load_games(path):
    """
    Expected columns in games.csv (you create this):
    game_date, home_team, away_team, home_sp, away_sp,
    home_team_abbr, away_team_abbr,
    actual_f1_total (optional - for backtesting),
    actual_f3_total (optional),
    actual_f5_total (optional),
    actual_game_total (optional),
    temp_f (optional), wind_mph (optional), wind_dir (optional - 'in'/'out'/'cross')
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    # Normalize wizard-agent column names to score_models.py expectations
    col_aliases = {
        "home_starter":   "home_sp",
        "away_starter":   "away_sp",
        "f5_total":       "market_total_f5",
        "f3_total":       "market_total_f3",
        "f1_total":       "market_total_f1",
        "game_total":     "market_total_game",
        "temperature_f":  "temp_f",
        "wind_speed_mph": "wind_mph",
        "wind_direction": "wind_dir_raw",
    }
    df = df.rename(columns={k: v for k, v in col_aliases.items() if k in df.columns})

    # Promote home_team / away_team as abbr if dedicated abbr columns are missing
    if "home_team_abbr" not in df.columns and "home_team" in df.columns:
        df["home_team_abbr"] = df["home_team"]
    if "away_team_abbr" not in df.columns and "away_team" in df.columns:
        df["away_team_abbr"] = df["away_team"]

    # Wizard-agent games.csv has no game_date — fill with today
    if "game_date" not in df.columns:
        df["game_date"] = datetime.today().strftime("%Y-%m-%d")

    df["home_sp"] = df["home_sp"].str.upper().str.strip()
    df["away_sp"] = df["away_sp"].str.upper().str.strip()
    return df


def merge_pitcher_data(savant_df, fg_df):
    merged = savant_df.merge(fg_df, on="pitcher_name", how="outer")
    return merged


# ─── MODEL COMPONENTS ────────────────────────────────────────────────────────

def stuff_plus_score(row):
    """
    Proxy Stuff+ from available signals if direct stuff_plus not in Savant CSV.
    Uses xERA, K%, exit velo allowed.
    Returns normalized score 80–130 range.
    """
    if "stuff_plus" in row and pd.notna(row["stuff_plus"]):
        return float(row["stuff_plus"])

    score = 100.0  # baseline
    if pd.notna(row.get("xera")):
        # xERA < 3.00 → elite; > 5.00 → poor
        score += (4.00 - row["xera"]) * 5.0
    if pd.notna(row.get("k_pct")):
        score += (row["k_pct"] - 22.0) * 0.8
    if pd.notna(row.get("exit_velo")):
        score += (87.0 - row["exit_velo"]) * 1.5
    return np.clip(score, 70, 140)


def command_floor_pass(row):
    """BB/9 < 3.5 required for MF5 and MG3."""
    if pd.notna(row.get("bb9")):
        return row["bb9"] < 3.5
    if pd.notna(row.get("bb_pct")):
        # Approx: bb9 ≈ bb_pct * 0.38 (rough conversion)
        return (row["bb_pct"] * 0.38) < 3.5
    return None  # unknown


def check_low_sample(row, label=""):
    """
    Returns warning string if pitcher stats are based on low sample.
    Checks pa_current (from Bayesian blend) and pa_scale.
    """
    pa = row.get("pa_current", 0) or 0
    scale = row.get("pa_scale", 1.0) or 1.0
    warnings = []
    if pa < LOW_SAMPLE_PA:
        warnings.append(f"{label} LOW SAMPLE ({int(pa)} PA — stats unreliable)")
    elif scale < 0.3:
        warnings.append(f"{label} LOW CONFIDENCE (scale={scale:.2f})")
    return "; ".join(warnings) if warnings else ""


def is_coors(game):
    """Returns True if game is at Coors Field (COL home)."""
    return str(game.get("home_team_abbr", "")).upper() == "COL"


def weather_adj(temp_f=None, wind_mph=None, wind_dir=None):
    """
    Returns multiplier for run environment.
    temp < 50 → suppression. Wind out → inflation. Wind in → suppression.
    """
    adj = 1.0
    if temp_f is not None and pd.notna(temp_f):
        if temp_f < 50:
            adj *= 0.93  # hard cold adj (MF5 trigger)
        elif temp_f < 60:
            adj *= 0.97
        elif temp_f > 85:
            adj *= 1.03
    if wind_mph is not None and pd.notna(wind_mph):
        if wind_dir == "out":
            adj *= 1.0 + (wind_mph * 0.004)
        elif wind_dir == "in":
            adj *= 1.0 - (wind_mph * 0.003)
    return adj


def park_factor(team_abbr):
    return PARK_FACTORS.get(str(team_abbr).upper(), 1.00)


def bayesian_shrink(value, stat_name, pa, league_mean=None):
    """
    Shrink observed stat toward league mean based on sample size.
    Confidence = min(pa / stabilization_pa, 1.0)
    At 0 PA → pure league mean. At full PA → pure observed value.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return LEAGUE_MEANS.get(stat_name, value)

    stab_pa = STABILIZATION_PA.get(stat_name, 200)
    prior   = league_mean if league_mean is not None else LEAGUE_MEANS.get(stat_name)
    if prior is None:
        return value  # no prior defined — return as-is

    confidence = min(float(pa) / stab_pa, 1.0) if pa and pa > 0 else 0.0
    return confidence * value + (1.0 - confidence) * prior


def get_month_scalar(game_date):
    """
    Returns run environment scalar based on month of game_date.
    Defaults to 1.0 if date unavailable.
    """
    try:
        if isinstance(game_date, str):
            month = int(game_date.split("-")[1])
        else:
            month = pd.to_datetime(game_date).month
        return MONTH_SCALAR.get(month, 1.0)
    except Exception:
        return 1.0



# ─── MODEL SCORERS ───────────────────────────────────────────────────────────

def score_mf5i(home_row, away_row, game):
    """
    MF5 — F5 ML lean. Pitcher quality + home field advantage + continuous probability.

    v2 Changes:
    - Home field advantage: +8 score units to home pitcher (54% baseline)
    - Continuous probability: sigmoid mapping of edge to prob (replaces 3-tier)
    - Wider neutral zone: edge threshold raised from 15 to 20
    - Backtest showed systematic away bias — HFA corrects selection bias
    """
    home_pa = home_row.get("pa_current", 0) or 0
    away_pa = away_row.get("pa_current", 0) or 0

    # Bayesian shrinkage on key signals
    home_xwoba = bayesian_shrink(home_row.get("xwoba"), "xwoba", home_pa)
    away_xwoba = bayesian_shrink(away_row.get("xwoba"), "xwoba", away_pa)
    home_whiff = bayesian_shrink(home_row.get("whiff_pct"), "whiff_pct", home_pa)
    away_whiff = bayesian_shrink(away_row.get("whiff_pct"), "whiff_pct", away_pa)
    home_hard  = bayesian_shrink(home_row.get("hard_hit_pct"), "hard_hit_pct", home_pa)
    away_hard  = bayesian_shrink(away_row.get("hard_hit_pct"), "hard_hit_pct", away_pa)

    home_row_adj = {**home_row, "xwoba": home_xwoba, "whiff_pct": home_whiff, "hard_hit_pct": home_hard}
    away_row_adj = {**away_row, "xwoba": away_xwoba, "whiff_pct": away_whiff, "hard_hit_pct": away_hard}

    home_sp = stuff_plus_score(home_row_adj) * 2.2
    away_sp = stuff_plus_score(away_row_adj) * 2.2

    home_cf = command_floor_pass(home_row)
    away_cf = command_floor_pass(away_row)

    home_ev = bayesian_shrink(home_row.get("exit_velo"), "exit_velo", home_pa) or 87.0
    away_ev = bayesian_shrink(away_row.get("exit_velo"), "exit_velo", away_pa) or 87.0

    # Raw pitcher scores
    home_pitcher_score = home_sp + (87.0 - home_ev) * 2.0
    away_pitcher_score = away_sp + (87.0 - away_ev) * 2.0

    # Apply home field advantage to home pitcher score
    home_pitcher_score += HOME_FIELD_BONUS

    edge = home_pitcher_score - away_pitcher_score  # positive = home advantage

    # Continuous probability — sigmoid mapping
    if edge > NEUTRAL_EDGE_THRESHOLD:
        lean      = f"HOME ({game.get('home_team','')})"
        raw_prob  = edge_to_prob(edge - NEUTRAL_EDGE_THRESHOLD) + HOME_FIELD_PROB_BUMP
    elif edge < -NEUTRAL_EDGE_THRESHOLD:
        lean      = f"AWAY ({game.get('away_team','')})"
        raw_prob  = edge_to_prob(abs(edge) - NEUTRAL_EDGE_THRESHOLD)
    else:
        lean      = "NEUTRAL"
        # Even neutral has a slight home lean
        raw_prob  = 0.52 + HOME_FIELD_PROB_BUMP * (edge / NEUTRAL_EDGE_THRESHOLD) * 0.5

    raw_prob = np.clip(raw_prob, 0.50, 0.82)

    # Weather cold flag
    temp      = game.get("temp_f")
    cold_flag = pd.notna(temp) and float(temp) < 50 if temp else False

    # Command floor
    cf_note = []
    if home_cf is False:
        cf_note.append(f"{game.get('home_sp','')} FAILS CF")
        if lean.startswith("HOME"):
            raw_prob *= 0.94
    if away_cf is False:
        cf_note.append(f"{game.get('away_sp','')} FAILS CF")
        if lean.startswith("AWAY"):
            raw_prob *= 0.94

    verdict = prob_verdict(raw_prob, thresholds=[0.65, 0.60, 0.55])

    return {
        "model":  "MF5i",
        "lean":   lean,
        "probability": round(raw_prob * 100, 1),
        "verdict": verdict,
        "cold_flag": cold_flag,
        "command_floor_notes": "; ".join(cf_note) if cf_note else "PASS",
        "home_stuff_score": round(home_pitcher_score, 1),
        "away_stuff_score": round(away_pitcher_score, 1),
        "raw_edge": round(edge, 1),
    }


def score_mfull(home_row, away_row, game):
    """
    M2 — Environmental / Total lean.
    SIERA + K-BB% weighted 0.28, atm density 0.12, bullpen fatigue, travel.
    """
    # Base run environment from SIERA
    home_siera = home_row.get("siera", 4.20) if pd.notna(home_row.get("siera")) else 4.20
    away_siera = away_row.get("siera", 4.20) if pd.notna(away_row.get("siera")) else 4.20

    # Expected F5 runs from SIERA (approximate: SIERA × 5/9 innings × scaling)
    home_exp_runs = home_siera * (5/9) * 0.90
    away_exp_runs = away_siera * (5/9) * 0.90
    base_total = home_exp_runs + away_exp_runs

    # K-BB% adjustment (weight 0.28)
    home_kbb = home_row.get("k_bb_pct", 10.0) if pd.notna(home_row.get("k_bb_pct")) else 10.0
    away_kbb = away_row.get("k_bb_pct", 10.0) if pd.notna(away_row.get("k_bb_pct")) else 10.0
    avg_kbb = (home_kbb + away_kbb) / 2
    kbb_adj = 1.0 - ((avg_kbb - 10.0) * 0.006)  # higher K-BB% → lower total

    # Park factor (atm density proxy, weight 0.12)
    home_pf = park_factor(game.get("home_team_abbr", ""))
    pf_adj = 1.0 + (home_pf - 1.0) * 0.12

    # Weather
    w_adj = weather_adj(
        temp_f=game.get("temp_f"),
        wind_mph=game.get("wind_mph"),
        wind_dir=game.get("wind_dir")
    )

    # Month scalar — seasonal run environment
    m_scalar = get_month_scalar(game.get("game_date", ""))

    # Lineup quality adjustment — team wOBA vs pitcher handedness
    home_lq = game.get("home_lineup_adj", 1.0) or 1.0
    away_lq = game.get("away_lineup_adj", 1.0) or 1.0
    lineup_combined = (home_lq + away_lq) / 2.0
    projected_total = base_total * kbb_adj * pf_adj * w_adj * m_scalar * lineup_combined

    # ── Coors hard fade ──────────────────────────────────────────────────────
    coors_flag = is_coors(game)

    # ── Blowout conflict check ────────────────────────────────────────────────
    # If one pitcher massively outclasses the other, favorite may win big
    # which inflates run total even in a "pitcher's game"
    home_sp_score = stuff_plus_score(home_row)
    away_sp_score = stuff_plus_score(away_row)
    sp_edge = abs(home_sp_score - away_sp_score)
    blowout_risk = sp_edge >= BLOWOUT_EDGE_THRESHOLD

    # Determine O/U lean vs market line
    market_total = game.get("market_total_f5")
    if market_total and pd.notna(market_total):
        market_total = float(market_total)
        diff = projected_total - market_total
        if diff > 0.3:
            lean = "OVER"
            raw_prob = 0.60 + min(diff * 0.055, 0.15)
            if coors_flag:
                raw_prob = min(raw_prob + 0.06, 0.78)
        elif diff < -0.3:
            lean = "UNDER"
            raw_prob = 0.60 + min(abs(diff) * 0.055, 0.15)
            if coors_flag:
                raw_prob = max(raw_prob - COORS_UNDER_PENALTY, 0.35)  # hard fade Under at Coors
                lean = f"UNDER ⚠️ COORS FADE"
            if blowout_risk:
                raw_prob = max(raw_prob - BLOWOUT_UNDER_PENALTY, 0.35)  # blowout risk
        else:
            lean = "NEUTRAL"
            raw_prob = 0.52
    else:
        lean = f"PROJ {projected_total:.2f} runs (no market line)"
        raw_prob = 0.55

    verdict = prob_verdict(raw_prob, thresholds=[0.68, 0.63, 0.58])

    return {
        "model": "MFull",
        "lean": lean,
        "probability": round(raw_prob * 100, 1),
        "verdict": verdict,
        "projected_total": round(projected_total, 2),
        "market_total": market_total if market_total else "N/A",
        "kbb_adj": round(kbb_adj, 3),
        "park_factor": round(home_pf, 3),
        "weather_adj": round(w_adj, 3),
        "coors_flag": coors_flag,
        "blowout_risk": blowout_risk,
    }


def score_mf1i(home_row, away_row, game):
    """
    MF1 — F1 Total Engine. Market-neutral.
    Uses pitcher F1 RA rate if available, else proxied from xERA.
    """
    # F1 runs allowed proxy: xERA × (1/9) innings × adjustment
    home_xera = home_row.get("xera", 4.20) if pd.notna(home_row.get("xera")) else 4.20
    away_xera = away_row.get("xera", 4.20) if pd.notna(away_row.get("xera")) else 4.20

    # F1-specific: pitchers typically perform better in inning 1 (fresh, max stuff)
    # Discount xERA by ~15% for F1 window
    home_f1_exp = (home_xera / 9) * 0.85
    away_f1_exp = (away_xera / 9) * 0.85

    home_pf  = park_factor(game.get("home_team_abbr", ""))
    w_adj    = weather_adj(temp_f=game.get("temp_f"), wind_mph=game.get("wind_mph"), wind_dir=game.get("wind_dir"))
    m_scalar = get_month_scalar(game.get("game_date", ""))

    projected_f1 = (home_f1_exp + away_f1_exp) * home_pf * w_adj * m_scalar
    coors_flag_f1 = is_coors(game)

    market_f1 = game.get("market_total_f1")
    if market_f1 and pd.notna(market_f1):
        market_f1 = float(market_f1)
        diff = projected_f1 - market_f1
        if diff > 0.15:
            lean = "OVER"
            raw_prob = 0.55 + min(diff * 0.06, 0.14)  # MF1 well-calibrated — leave alone
        elif diff < -0.15:
            lean = "UNDER"
            raw_prob = 0.55 + min(abs(diff) * 0.06, 0.14)
            # Note: No Coors fade on MF1 — park effects don't dominate inning 1
        else:
            lean = "NEUTRAL"
            raw_prob = 0.52
    else:
        lean = f"PROJ {projected_f1:.2f} runs (no market line)"
        raw_prob = 0.54

    verdict = prob_verdict(raw_prob, thresholds=[0.68, 0.63, 0.58])

    return {
        "model": "MF1i",
        "lean": lean,
        "probability": round(raw_prob * 100, 1),
        "verdict": verdict,
        "projected_f1_total": round(projected_f1, 2),
        "market_f1": market_f1 if market_f1 else "N/A",
    }


def score_mf3i(home_row, away_row, game):
    """
    MG3 — F3 Total Engine. Contrarian (fades heavy favorites).
    """
    home_xera = home_row.get("xera", 4.20) if pd.notna(home_row.get("xera")) else 4.20
    away_xera = away_row.get("xera", 4.20) if pd.notna(away_row.get("xera")) else 4.20

    home_cf = command_floor_pass(home_row)
    away_cf = command_floor_pass(away_row)

    # F3 window: 3 innings, pitchers still dominant
    home_f3_exp = (home_xera / 9) * 3 * 0.90
    away_f3_exp = (away_xera / 9) * 3 * 0.90

    # Stuff+ amplifier for F3
    home_sp = stuff_plus_score(home_row)
    away_sp = stuff_plus_score(away_row)
    sp_adj = 1.0 - ((((home_sp + away_sp) / 2) - 100) * 0.002)

    home_pf = park_factor(game.get("home_team_abbr", ""))
    w_adj = weather_adj(temp_f=game.get("temp_f"), wind_mph=game.get("wind_mph"), wind_dir=game.get("wind_dir"))

    m_scalar     = get_month_scalar(game.get("game_date", ""))
    projected_f3 = (home_f3_exp + away_f3_exp) * sp_adj * home_pf * w_adj * m_scalar

    # Command floor: if either SP fails, flag it
    cf_notes = []
    if home_cf is False:
        cf_notes.append(f"{game.get('home_sp','')} FAILS CF — walks inflate F3")
        projected_f3 *= 1.06
    if away_cf is False:
        cf_notes.append(f"{game.get('away_sp','')} FAILS CF — walks inflate F3")
        projected_f3 *= 1.06

    coors_flag_f3 = is_coors(game)

    market_f3 = game.get("market_total_f3")
    if market_f3 and pd.notna(market_f3):
        market_f3 = float(market_f3)
        diff = projected_f3 - market_f3
        if diff > 0.25:
            lean = "OVER"
            raw_prob = 0.58 + min(diff * 0.05, 0.13)
        elif diff < -0.25:
            lean = "UNDER"
            raw_prob = 0.58 + min(abs(diff) * 0.05, 0.13)
            if coors_flag_f3:
                raw_prob = max(raw_prob - COORS_UNDER_PENALTY, 0.35)
                lean = "UNDER ⚠️ COORS FADE"
        else:
            lean = "NEUTRAL"
            raw_prob = 0.52
    else:
        lean = f"PROJ {projected_f3:.2f} runs (no market line)"
        raw_prob = 0.54

    verdict = prob_verdict(raw_prob, thresholds=[0.67, 0.63, 0.58])

    return {
        "model": "MF3i",
        "lean": lean,
        "probability": round(raw_prob * 100, 1),
        "verdict": verdict,
        "projected_f3_total": round(projected_f3, 2),
        "market_f3": market_f3 if market_f3 else "N/A",
        "command_floor_notes": "; ".join(cf_notes) if cf_notes else "PASS",
    }



def score_mbat(batter_name, opp_pitcher_row, game, batter_df, lineup_pos=3, fg_batter_df=None):
    if fg_batter_df is None:
        fg_batter_df = pd.DataFrame()
    """
    M3 — Hit Probability Engine (Bernoulli complement).
    P(1+ hit) = 1 - (1 - p_adj)^PA_exp
    p_base = BA × (1 - K% × 0.15)
    Thresholds: >=70% strong keep, 65-70% keep, 58-65% marginal, <58% fail
    """
    if batter_df.empty:
        return {"model": "MBat", "batter": batter_name, "verdict": "❌ NO BATTER DATA", "probability": 0}

    batter_name_upper = str(batter_name).upper().strip()
    b = batter_df[batter_df["batter_name"] == batter_name_upper]

    if b.empty:
        return {"model": "MBat", "batter": batter_name, "verdict": "⚠️ NOT FOUND", "probability": 0}

    b = b.iloc[0].to_dict()

    # Supplement with FanGraphs batter data if available
    if not fg_batter_df.empty:
        fg_b = fg_batter_df[fg_batter_df["batter_name"] == batter_name_upper]
        if not fg_b.empty:
            fg_b = fg_b.iloc[0].to_dict()
            # FG fills gaps: use xwOBA_fg if Savant xwOBA missing, BABIP_fg as cross-check
            if pd.isna(b.get("xwoba")) and pd.notna(fg_b.get("xwoba_fg")):
                b["xwoba"] = fg_b["xwoba_fg"]
            if pd.isna(b.get("babip")) and pd.notna(fg_b.get("babip_fg")):
                b["babip"] = fg_b["babip_fg"]
            if pd.isna(b.get("ba")) and pd.notna(fg_b.get("ba_fg")):
                b["ba"] = fg_b["ba_fg"]
            if pd.isna(b.get("k_pct")) and pd.notna(fg_b.get("k_pct_fg")):
                b["k_pct"] = fg_b["k_pct_fg"]
            if pd.isna(b.get("bb_pct")) and pd.notna(fg_b.get("bb_pct_fg")):
                b["bb_pct"] = fg_b["bb_pct_fg"]
            # wRC+ as quality flag — high wRC+ batters get slight boost
            if pd.notna(fg_b.get("wrc_plus")):
                b["wrc_plus"] = fg_b["wrc_plus"]

    # BA blend: use xBA if available (more stable), else batting_avg
    ba = b.get("xba") if pd.notna(b.get("xba")) else b.get("ba", 0.250)
    if not pd.notna(ba): ba = 0.250

    k_pct = b.get("k_pct", 22.0)
    if not pd.notna(k_pct): k_pct = 22.0

    bb_pct = b.get("bb_pct", 8.0)
    if not pd.notna(bb_pct): bb_pct = 8.0

    b_pa = b.get("pa_current", 0) or 0

    # Bayesian shrinkage on batter stats
    ba    = bayesian_shrink(ba,    "ba",           b_pa)
    babip_raw = b.get("babip", 0.300)
    babip = bayesian_shrink(babip_raw if pd.notna(babip_raw) else 0.300, "babip", b_pa)

    hard_hit_raw = b.get("hard_hit_pct", 38.0)
    hard_hit = bayesian_shrink(hard_hit_raw if pd.notna(hard_hit_raw) else 38.0, "hard_hit_pct", b_pa)
    k_pct    = bayesian_shrink(k_pct, "k_pct", b_pa)
    bb_pct   = bayesian_shrink(bb_pct, "bb_pct", b_pa)

    # p_base
    p_base = ba * (1 - k_pct * 0.15 / 100)

    # PA expectation by lineup position
    pa_map = {1:4.5, 2:4.3, 3:4.1, 4:3.9, 5:3.7, 6:3.5, 7:3.4, 8:3.3, 9:3.2}
    pa_exp = pa_map.get(int(lineup_pos), 3.8)

    # Pitcher K/9 delta
    opp_k9 = opp_pitcher_row.get("k9", 8.5) if pd.notna(opp_pitcher_row.get("k9")) else 8.5
    delta_k9 = max(0.80, 1 - (opp_k9 - 8.5) * 0.018)

    # Pitcher BB/9 delta (walks help batter reach)
    opp_bb9 = opp_pitcher_row.get("bb9", 3.0) if pd.notna(opp_pitcher_row.get("bb9")) else 3.0
    delta_bb9 = 1 + (opp_bb9 - 3.5) * 0.015 if opp_bb9 > 3.5 else 1.0

    # Park factor
    home_pf = park_factor(game.get("home_team_abbr", ""))
    delta_park = 0.93 + (home_pf - 0.94) * (0.13 / 0.20)
    delta_park = np.clip(delta_park, 0.93, 1.06)

    # Form delta from BABIP (proxy recent form)
    if babip > 0.320:
        delta_form = 1.08
    elif babip < 0.270:
        delta_form = 0.90
    else:
        delta_form = 1.00

    # BABIP delta
    if babip > 0.320:
        delta_babip = 1.06
    elif babip < 0.270:
        delta_babip = 0.94
    else:
        delta_babip = 1.00

    # Hard hit bonus
    hard_hit_adj = 1.0 + (hard_hit - 38.0) * 0.003

    # wRC+ quality adjustment (FG data): >120 elite hitter, <80 weak
    wrc_plus = b.get("wrc_plus")
    if wrc_plus and pd.notna(wrc_plus):
        hard_hit_adj *= 1.0 + (float(wrc_plus) - 100) * 0.001

    # Combine
    p_adj = p_base * delta_k9 * delta_bb9 * delta_park * delta_form * delta_babip * hard_hit_adj
    p_adj = min(p_adj, 0.55)  # cap

    # Bernoulli: P(1+ hit)
    prob = 1 - (1 - p_adj) ** pa_exp

    verdict = prob_verdict(prob, thresholds=[0.70, 0.65, 0.58])

    return {
        "model":       "MBat",
        "batter":      batter_name,
        "probability": round(prob * 100, 1),
        "verdict":     verdict,
        "p_adj":       round(p_adj, 3),
        "pa_exp":      pa_exp,
        "delta_k9":    round(delta_k9, 3),
        "delta_bb9":   round(delta_bb9, 3),
        "opp_k9":      round(opp_k9, 1),
        "opp_bb9":     round(opp_bb9, 1),
    }

# ─── VERDICT HELPER ─────────────────────────────────────────────────────────

def prob_verdict(prob, thresholds):
    """thresholds = [strong_keep, keep, marginal]"""
    if prob >= thresholds[0]:
        return "✅ STRONG KEEP"
    elif prob >= thresholds[1]:
        return "🟡 KEEP"
    elif prob >= thresholds[2]:
        return "🟠 MARGINAL"
    else:
        return "❌ FAIL"


# ─── BACKTEST SCORER ─────────────────────────────────────────────────────────

def evaluate_backtest(result_row, game):
    """Compare model lean vs actual result if actuals provided."""
    evals = {}
    for model in ["MF1i", "MF3i", "MF5i", "MFull"]:
        lean = result_row.get(f"{model}_lean", "")
        actual_key = {
            "MF1i": "actual_f1_total", "MF3i": "actual_f3_total",
            "MF5i": "actual_f5_total", "MFull": "actual_f5_total"
        }.get(model)
        market_key = {
            "MF1i": "market_total_f1", "MF3i": "market_total_f3",
            "MF5i": None, "MFull": "market_total_f5"
        }.get(model)

        actual = game.get(actual_key) if actual_key else None
        market = game.get(market_key) if market_key else None

        if actual is not None and pd.notna(actual) and market is not None and pd.notna(market):
            actual, market = float(actual), float(market)
            if "OVER" in str(lean):
                correct = actual > market
            elif "UNDER" in str(lean):
                correct = actual < market
            else:
                correct = None
            evals[f"{model}_result"] = "✅ WIN" if correct else ("❌ LOSS" if correct is not None else "PUSH")
        elif model == "MF5i":
            # ML lean — compare to game winner
            home_win = game.get("actual_home_win")
            if home_win is not None and pd.notna(home_win):
                home_won = str(home_win).upper() in ["1", "TRUE", "YES", "HOME"]
                if "HOME" in str(lean):
                    evals[f"{model}_result"] = "✅ WIN" if home_won else "❌ LOSS"
                elif "AWAY" in str(lean):
                    evals[f"{model}_result"] = "✅ WIN" if not home_won else "❌ LOSS"
                else:
                    evals[f"{model}_result"] = "—"
            else:
                evals[f"{model}_result"] = "—"
        else:
            evals[f"{model}_result"] = "—"
    return evals


def flag_back_to_back(games_df):
    """
    Flag pitchers starting on back-to-back days.
    Reduces raw_prob by 4% for flagged pitchers in MF5.
    Adds btb_home and btb_away columns.
    """
    games_df = games_df.copy()
    games_df["game_date_dt"] = pd.to_datetime(games_df["game_date"], errors="coerce")
    games_df["btb_home"] = False
    games_df["btb_away"] = False

    for idx, row in games_df.iterrows():
        date = row["game_date_dt"]
        home_sp = str(row.get("home_sp","")).upper()
        away_sp = str(row.get("away_sp","")).upper()
        prev = games_df[games_df["game_date_dt"] == date - pd.Timedelta(days=1)]
        prev_pitchers = set(
            list(prev["home_sp"].str.upper()) + list(prev["away_sp"].str.upper())
        )
        if home_sp in prev_pitchers:
            games_df.at[idx, "btb_home"] = True
        if away_sp in prev_pitchers:
            games_df.at[idx, "btb_away"] = True

    return games_df



def load_team_splits():
    """
    Load FanGraphs team batting splits vs LHP and RHP.
    Returns (lhp_dict, rhp_dict) keyed by team abbreviation.
    Each value: {wOBA, K%, BB%, wRC+}
    Falls back to league average if file missing.
    """
    LEAGUE_AVG_OFFENSE = {"wOBA": 0.315, "K%": 0.235, "BB%": 0.085, "wRC+": 100}

    def _load(path):
        if not os.path.exists(path):
            return {}
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        result = {}
        for _, row in df.iterrows():
            tm = str(row.get("Tm","")).upper().strip()
            if not tm: continue
            result[tm] = {
                "wOBA": float(row.get("wOBA", LEAGUE_AVG_OFFENSE["wOBA"])),
                "K%":   float(row.get("K%",   LEAGUE_AVG_OFFENSE["K%"])),
                "BB%":  float(row.get("BB%",  LEAGUE_AVG_OFFENSE["BB%"])),
                "wRC+": float(row.get("wRC+", LEAGUE_AVG_OFFENSE["wRC+"])),
            }
        return result

    lhp = _load(TEAM_VS_LHP_FILE)
    rhp = _load(TEAM_VS_RHP_FILE)

    if lhp:
        print(f"   Team splits loaded: {len(lhp)} teams vs LHP, {len(rhp)} teams vs RHP")
    else:
        print(f"   Team splits: files not found — using league average")

    return lhp, rhp, LEAGUE_AVG_OFFENSE


def get_lineup_quality(team_abbr, pitcher_hand, lhp_dict, rhp_dict, league_avg):
    """
    Returns lineup quality dict for a team facing a pitcher of given handedness.
    pitcher_hand: 'L' or 'R'
    """
    abbr = str(team_abbr).upper().strip()
    splits = lhp_dict if pitcher_hand == "L" else rhp_dict
    return splits.get(abbr, league_avg)


def lineup_quality_adj(lineup, league_avg):
    """
    Returns a multiplier for run environment based on lineup wOBA vs league avg.
    Strong lineup (wOBA > .320) → slightly inflates total
    Weak lineup (wOBA < .300) → slightly suppresses total
    """
    woba = lineup.get("wOBA", league_avg["wOBA"])
    delta = woba - league_avg["wOBA"]
    # Scale: 10 pts wOBA ≈ 3% run environment change
    return 1.0 + (delta * 3.0)


# ─── ODDS API ────────────────────────────────────────────────────────────────

def fetch_odds_api():
    """
    Fetches MLB game lines from The Odds API (DraftKings).
    Returns dict keyed by (home_abbr, away_abbr) with market lines.
    Markets: h2h (ML), totals (game O/U), spreads (run line)
    Falls back gracefully if API unavailable.
    """
    if not ODDS_API_ENABLED or not ODDS_API_KEY:
        return {}

    url = (
        f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
        f"?apiKey={ODDS_API_KEY}"
        f"&regions=us"
        f"&markets=totals,h2h,spreads"
        f"&bookmakers=draftkings,fanduel"
        f"&oddsFormat=american"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MLB-Model/1.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            games = json.loads(r.read())
            remaining = dict(r.headers).get("x-requests-remaining", "?")
        print(f"   Odds API: {len(games)} games fetched ({remaining} requests remaining)")
    except Exception as e:
        print(f"   Odds API unavailable: {e} — using manual lines from games.csv")
        return {}

    odds_map = {}
    for game in games:
        home_name = game.get("home_team", "")
        away_name = game.get("away_team", "")
        home_abbr = TEAM_NAME_TO_ABBR.get(home_name, "")
        away_abbr = TEAM_NAME_TO_ABBR.get(away_name, "")
        if not home_abbr or not away_abbr:
            continue

        lines = {"market_total_game": None, "market_ml_home": None,
                 "market_ml_away": None, "market_rl": None}

        # Collect lines from both books
        book_lines = {"draftkings": {}, "fanduel": {}}

        for book in game.get("bookmakers", []):
            bk = book["key"]
            if bk not in book_lines:
                continue
            for market in book.get("markets", []):
                key = market["key"]
                outcomes = market.get("outcomes", [])
                if key == "totals":
                    for o in outcomes:
                        if o["name"] == "Over":
                            book_lines[bk]["total"] = o.get("point")
                            book_lines[bk]["total_over_price"] = o.get("price")
                        elif o["name"] == "Under":
                            book_lines[bk]["total_under_price"] = o.get("price")
                elif key == "h2h":
                    for o in outcomes:
                        nm = TEAM_NAME_TO_ABBR.get(o["name"], o["name"])
                        if nm == home_abbr:
                            book_lines[bk]["ml_home"] = o.get("price")
                        elif nm == away_abbr:
                            book_lines[bk]["ml_away"] = o.get("price")
                elif key == "spreads":
                    for o in outcomes:
                        nm = TEAM_NAME_TO_ABBR.get(o["name"], o["name"])
                        if nm == home_abbr:
                            book_lines[bk]["rl"] = o.get("point")

        # Use DK as primary, fall back to FanDuel
        dk = book_lines["draftkings"]
        fd = book_lines["fanduel"]
        primary = dk if dk else fd

        lines["market_total_game"] = primary.get("total")
        lines["market_ml_home"]    = primary.get("ml_home")
        lines["market_ml_away"]    = primary.get("ml_away")
        lines["market_rl"]         = primary.get("rl")

        # Best book for Under — lower juice wins
        dk_under = dk.get("total_under_price", -115)
        fd_under = fd.get("total_under_price", -115)
        if dk_under and fd_under:
            lines["best_under_book"]  = "DK" if dk_under >= fd_under else "FD"
            lines["best_under_price"] = max(dk_under, fd_under)
        elif dk_under:
            lines["best_under_book"]  = "DK"
            lines["best_under_price"] = dk_under
        elif fd_under:
            lines["best_under_book"]  = "FD"
            lines["best_under_price"] = fd_under

        # Best book for Over
        dk_over = dk.get("total_over_price", -115)
        fd_over = fd.get("total_over_price", -115)
        if dk_over and fd_over:
            lines["best_over_book"]  = "DK" if dk_over >= fd_over else "FD"
            lines["best_over_price"] = max(dk_over, fd_over)
        elif dk_over:
            lines["best_over_book"]  = "DK"
            lines["best_over_price"] = dk_over
        elif fd_over:
            lines["best_over_book"]  = "FD"
            lines["best_over_price"] = fd_over

        # Best book for ML
        dk_ml_home = dk.get("ml_home")
        fd_ml_home = fd.get("ml_home")
        if dk_ml_home and fd_ml_home:
            lines["best_ml_home_book"]  = "DK" if dk_ml_home >= fd_ml_home else "FD"
            lines["best_ml_home_price"] = max(dk_ml_home, fd_ml_home)

        odds_map[(home_abbr, away_abbr)] = lines

    return odds_map


def enrich_games_with_odds(games_df, odds_map):
    """
    For each game missing market_total_game, fill from Odds API.
    Respects manual values already in games.csv.
    Also estimates F5 total as 55% of game total if missing.
    """
    if not odds_map:
        return games_df

    enriched = games_df.copy()
    filled = 0

    for idx, row in enriched.iterrows():
        home = str(row.get("home_team_abbr", "")).upper()
        away = str(row.get("away_team_abbr", "")).upper()
        api_lines = odds_map.get((home, away), {})
        if not api_lines:
            continue

        # Fill game total if missing
        total_missing = pd.isna(row.get("market_total_game")) or str(row.get("market_total_game")).strip() in ["", "nan"]
        if total_missing and api_lines.get("market_total_game"):
            enriched.at[idx, "market_total_game"] = api_lines["market_total_game"]
            filled += 1

            # Estimate F5 total as 55% of game total if not set
            f5_missing = pd.isna(row.get("market_total_f5")) or str(row.get("market_total_f5")).strip() in ["", "nan"]
            if f5_missing:
                enriched.at[idx, "market_total_f5"] = round(api_lines["market_total_game"] * 0.555, 2)

            # Estimate F3 as 32% of game total
            f3_missing = pd.isna(row.get("market_total_f3")) or str(row.get("market_total_f3")).strip() in ["", "nan"]
            if f3_missing:
                enriched.at[idx, "market_total_f3"] = round(api_lines["market_total_game"] * 0.32, 2)

            # Estimate F1 as 11% of game total
            f1_missing = pd.isna(row.get("market_total_f1")) or str(row.get("market_total_f1")).strip() in ["", "nan"]
            if f1_missing:
                enriched.at[idx, "market_total_f1"] = round(api_lines["market_total_game"] * 0.11, 2)

    if filled:
        print(f"   Odds API filled {filled} game totals (F5/F3/F1 estimated from game total)")
    return enriched


# ─── WEATHER FETCH ───────────────────────────────────────────────────────────

def fetch_game_weather(home_team_abbr, game_date=None):
    """
    Fetches current weather for a ballpark using Open-Meteo (free, no API key).
    Returns dict: temp_f, wind_mph, wind_dir ('in'/'out'/'cross'), wind_deg
    Falls back to None values if fetch fails.
    """
    abbr = str(home_team_abbr).upper().strip()
    coords = BALLPARK_COORDS.get(abbr)
    if not coords:
        return {"temp_f": None, "wind_mph": None, "wind_dir": None}

    lat, lon = coords
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,wind_speed_10m,wind_direction_10m"
        f"&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=auto"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MLB-Model/1.0"})
        with urllib.request.urlopen(req, timeout=3) as r:
            data = json.loads(r.read())
        cur = data["current"]
        temp_f    = round(cur["temperature_2m"], 1)
        wind_mph  = round(cur["wind_speed_10m"], 1)
        wind_deg  = cur["wind_direction_10m"]

        # Determine wind direction relative to CF
        cf_bearing = STADIUM_CF_BEARING.get(abbr, 0)
        rel = (wind_deg - cf_bearing + 360) % 360
        # Wind blowing FROM direction — flip 180 to get direction wind travels
        travel = (rel + 180) % 360
        if travel <= 45 or travel >= 315:
            wind_dir = "out"    # toward CF — helps offense
        elif 135 <= travel <= 225:
            wind_dir = "in"     # toward home plate — suppresses offense
        else:
            wind_dir = "cross"

        return {"temp_f": temp_f, "wind_mph": wind_mph, "wind_dir": wind_dir}

    except Exception as e:
        return {"temp_f": None, "wind_mph": None, "wind_dir": None}


def _load_weather_parquet(year: int) -> "pd.DataFrame | None":
    """
    Load the pre-built weather_{year}.parquet from statcast_data/.
    Returns a DataFrame indexed by (game_date, home_team) or None if missing.
    """
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "statcast_data", f"weather_{year}.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path, engine="pyarrow")
        df["game_date"] = df["game_date"].astype(str).str[:10]
        df["home_team"] = df["home_team"].astype(str).str.upper()
        return df.set_index(["game_date", "home_team"])
    except Exception:
        return None


def _bearing_to_wind_dir(wind_deg, abbr):
    """Convert wind bearing degrees to in/out/cross relative to CF."""
    cf_bearing = STADIUM_CF_BEARING.get(str(abbr).upper(), 0)
    rel = (wind_deg - cf_bearing + 360) % 360
    travel = (rel + 180) % 360
    if travel <= 45 or travel >= 315:
        return "out"
    elif 135 <= travel <= 225:
        return "in"
    return "cross"


def enrich_games_with_weather(games_df):
    """
    For each game, look up weather from the pre-built weather_{year}.parquet first.
    Falls back to a live Open-Meteo fetch only if the parquet has no entry.
    Respects any temp_f already present in games_df (manual override).
    Prints a weather summary table.
    """
    print("\n  Fetching weather data (timeout=3s per park)...")
    enriched = games_df.copy()

    # Initialize columns
    for col in ["temp_f", "wind_mph"]:
        if col not in enriched.columns:
            enriched[col] = pd.Series([None] * len(enriched), dtype="float64")
        else:
            enriched[col] = enriched[col].astype("float64", errors="ignore")
    if "wind_dir" not in enriched.columns:
        enriched["wind_dir"] = pd.Series([None] * len(enriched), dtype="object")

    # Cache parquet lookups by year
    _wx_cache: dict[int, object] = {}

    from_parquet = 0
    from_live    = 0
    skipped      = 0

    for idx, row in enriched.iterrows():
        temp_missing = pd.isna(row.get("temp_f")) or str(row.get("temp_f")).strip() in ["", "nan"]
        if not temp_missing:
            skipped += 1  # manual override already present
            continue

        game_date = str(row.get("game_date", ""))[:10]
        abbr = str(row.get("home_team_abbr", row.get("home_team", ""))).upper()
        year = int(game_date[:4]) if game_date else 0

        # ── Source 1: weather parquet ────────────────────────────────────────
        if year not in _wx_cache:
            _wx_cache[year] = _load_weather_parquet(year)
        wx_df = _wx_cache[year]

        if wx_df is not None and (game_date, abbr) in wx_df.index:
            wx_row = wx_df.loc[(game_date, abbr)]
            t = wx_row.get("temp_f")
            w = wx_row.get("wind_mph")
            b = wx_row.get("wind_bearing")
            if pd.notna(t):
                enriched.at[idx, "temp_f"]  = float(t)
                enriched.at[idx, "wind_mph"] = float(w) if pd.notna(w) else None
                if pd.notna(b):
                    enriched.loc[idx, "wind_dir"] = _bearing_to_wind_dir(float(b), abbr)
                from_parquet += 1
                continue

        # ── Source 2: live Open-Meteo fetch ──────────────────────────────────
        w = fetch_game_weather(abbr, game_date)
        if w["temp_f"] is not None:
            enriched.at[idx, "temp_f"]   = float(w["temp_f"])
            enriched.at[idx, "wind_mph"]  = float(w["wind_mph"])
            enriched.loc[idx, "wind_dir"] = str(w["wind_dir"])
            from_live += 1
        else:
            skipped += 1

    # Print weather summary table
    print(f"   {'Matchup':<35} {'Temp°F':<8} {'Wind mph':<10} {'Dir'}")
    print(f"   {'-'*60}")
    for _, row in enriched.iterrows():
        away = str(row.get("away_team_abbr", row.get("away_team", "?")))
        home = str(row.get("home_team_abbr", row.get("home_team", "?")))
        matchup = f"{away} @ {home}"
        temp  = row.get("temp_f",  "—")
        wind  = row.get("wind_mph","—")
        wdir  = row.get("wind_dir","—")
        temp_s = f"{float(temp):.0f}°" if temp not in [None,"","nan","—"] and str(temp) != "nan" else "—"
        wind_s = f"{float(wind):.0f}"  if wind not in [None,"","nan","—"] and str(wind) != "nan" else "—"
        print(f"   {matchup:<35} {temp_s:<8} {wind_s:<10} {wdir}")

    print(f"   → {from_parquet} from parquet, {from_live} live fetched, {skipped} skipped")
    return enriched


# ─── WIND BEARING CONVERSION ─────────────────────────────────────────────────

def convert_wind_bearing_to_direction(games_df):
    """
    Convert numeric wind bearing (degrees from Open-Meteo) to 'in'/'out'/'cross'.
    Wizard-agent games.csv stores raw bearing; weather_adj() expects a string.
    Skips rows where wind_dir is already a string.
    """
    enriched = games_df.copy()
    if "wind_dir_raw" not in enriched.columns:
        return enriched
    for idx, row in enriched.iterrows():
        try:
            bearing = float(row["wind_dir_raw"])
            abbr = str(row.get("home_team_abbr", "")).upper()
            cf = STADIUM_CF_BEARING.get(abbr, 0)
            rel = (bearing - cf + 360) % 360
            travel = (rel + 180) % 360
            if travel <= 45 or travel >= 315:
                enriched.at[idx, "wind_dir"] = "out"
            elif 135 <= travel <= 225:
                enriched.at[idx, "wind_dir"] = "in"
            else:
                enriched.at[idx, "wind_dir"] = "cross"
        except (ValueError, TypeError):
            pass
    return enriched


# ─── MAIN PIPELINE ───────────────────────────────────────────────────────────

def run_pipeline():
    print("\n" + "="*60)
    print("  MLB MODEL SCORING PIPELINE")
    print("  Models: MF1i | MF3i | MF5i | MFull")
    print("="*60)

    # Check files
    missing = [f for f in [SAVANT_FILE, FANGRAPHS_FILE, GAMES_FILE] if not os.path.exists(f)]
    if missing:
        print(f"\n❌ Missing files: {missing}")
        print("  → Place CSV files in the same directory as this script")
        print("  → See README.txt for column requirements")
        sys.exit(1)

    print("\n📥 Loading data...")
    season_weights, pa_threshold = get_season_year_weights()
    global YEAR_WEIGHTS, PA_FULL_SAMPLE, PA_FULL_BAT
    YEAR_WEIGHTS   = season_weights
    PA_FULL_SAMPLE = pa_threshold
    PA_FULL_BAT    = pa_threshold * 2

    lhp_dict, rhp_dict, league_avg_off = load_team_splits()
    savant_df  = load_savant(SAVANT_FILE)
    fg_df      = load_fangraphs(FANGRAPHS_FILE)
    games_df   = load_games(GAMES_FILE)
    games_df   = convert_wind_bearing_to_direction(games_df)
    odds_map   = fetch_odds_api()
    games_df   = enrich_games_with_odds(games_df, odds_map)
    games_df   = enrich_games_with_weather(games_df)
    games_df   = flag_back_to_back(games_df)
    batter_df  = load_batters(SAVANT_BATTERS_FILE)
    fg_batter_df = load_fangraphs_batters(FANGRAPHS_BATTERS_FILE)
    pitcher_df = merge_pitcher_data(savant_df, fg_df)

    print(f"   Savant pitchers loaded:    {len(savant_df)}")
    print(f"   FanGraphs pitchers loaded: {len(fg_df)}")
    print(f"   Merged pitcher records:    {len(pitcher_df)}")
    print(f"   Savant batters loaded:     {len(batter_df)}")
    print(f"   FanGraphs batters loaded:  {len(fg_batter_df)}")
    print(f"   Games to score:            {len(games_df)}")

    results = []
    unmatched = []

    print("\n🔄 Scoring games...\n")

    for _, game in games_df.iterrows():
        game = game.to_dict()
        home_sp = str(game.get("home_sp", "")).upper().strip()
        away_sp = str(game.get("away_sp", "")).upper().strip()

        # Lookup pitcher rows
        home_row = pitcher_df[pitcher_df["pitcher_name"] == home_sp]
        away_row = pitcher_df[pitcher_df["pitcher_name"] == away_sp]

        # League-average fallback for unmatched pitchers — game scores with warning
        LEAGUE_AVG_ROW = {
            "k_pct": 22.0, "bb_pct": 8.5, "hard_hit_pct": 38.0,
            "exit_velo": 87.5, "whiff_pct": 25.0, "xwoba": 0.315,
            "xera": 4.20, "stuff_plus": 100.0, "fastball_velo": 93.5,
            "k9": 8.5, "bb9": 3.3, "siera": 4.20, "fip": 4.20,
            "k_bb_pct": 10.0, "pa_current": 0, "pa_scale": 0.0,
        }

        missing_sps = []
        if home_row.empty:
            missing_sps.append(home_sp)
            unmatched.append(home_sp)
            home_row = {**LEAGUE_AVG_ROW, "pitcher_name": home_sp, "_fallback": True}
        else:
            home_row = home_row.iloc[0].to_dict()
            home_row["_fallback"] = False

        if away_row.empty:
            missing_sps.append(away_sp)
            unmatched.append(away_sp)
            away_row = {**LEAGUE_AVG_ROW, "pitcher_name": away_sp, "_fallback": True}
        else:
            away_row = away_row.iloc[0].to_dict()
            away_row["_fallback"] = False

        if missing_sps:
            print(f"  ⚠️  {game.get('game_date','')} {away_sp} @ {home_sp} — using league avg for: {missing_sps}")

        # Lineup quality — team wOBA vs pitcher handedness
        # Pitcher hand heuristic: check FanGraphs data, default RHP
        home_hand = "L" if any(k in home_sp.upper() for k in ["RAGANS","SNELL","CEASE","GARCIA","PERALTA","LOPEZ","WEBB","GLASNOW","OHTANI","VALDEZ","WOO","LIBERATORE","LUZARDO","MONTGOMERY","FLAHERTY","SKUBAL","HOUCK","IRVIN","MEANS","KIRBY","KELLER","SINGER","RYAN","OBER"]) else "R"
        away_hand = "L" if any(k in away_sp.upper() for k in ["RAGANS","SNELL","CEASE","GARCIA","PERALTA","LOPEZ","WEBB","GLASNOW","OHTANI","VALDEZ","WOO","LIBERATORE","LUZARDO","MONTGOMERY","FLAHERTY","SKUBAL","HOUCK","IRVIN","MEANS","KIRBY","KELLER","SINGER","RYAN","OBER"]) else "R"

        # Home lineup faces away pitcher, away lineup faces home pitcher
        home_lineup = get_lineup_quality(game.get("home_team_abbr",""), away_hand, lhp_dict, rhp_dict, league_avg_off)
        away_lineup = get_lineup_quality(game.get("away_team_abbr",""), home_hand, lhp_dict, rhp_dict, league_avg_off)
        home_lineup_adj = lineup_quality_adj(home_lineup, league_avg_off)
        away_lineup_adj = lineup_quality_adj(away_lineup, league_avg_off)

        # Low sample warnings
        home_sample_warn = check_low_sample(home_row, home_sp)
        away_sample_warn = check_low_sample(away_row, away_sp)
        if home_sample_warn:
            print(f"     ⚠️  {home_sample_warn}")
        if away_sample_warn:
            print(f"     ⚠️  {away_sample_warn}")

        # Back-to-back flags
        btb_home = game.get("btb_home", False)
        btb_away = game.get("btb_away", False)
        if btb_home:
            print(f"     ⚠️  {home_sp} — BACK-TO-BACK start flagged (-4% prob)")
        if btb_away:
            print(f"     ⚠️  {away_sp} — BACK-TO-BACK start flagged (-4% prob)")

        # Score all models
        mf5 = score_mf5i(home_row, away_row, game)

        # Apply BTB penalty post-scoring
        if btb_home and mf5["lean"].startswith("HOME"):
            mf5["probability"] = round(max(mf5["probability"] - 4.0, 50.0), 1)
            mf5["verdict"] = prob_verdict(mf5["probability"]/100, thresholds=[0.65, 0.60, 0.55])
        if btb_away and mf5["lean"].startswith("AWAY"):
            mf5["probability"] = round(max(mf5["probability"] - 4.0, 50.0), 1)
            mf5["verdict"] = prob_verdict(mf5["probability"]/100, thresholds=[0.65, 0.60, 0.55])
        game["home_lineup_adj"] = home_lineup_adj
        game["away_lineup_adj"] = away_lineup_adj
        m2  = score_mfull(home_row, away_row, game)
        mf1 = score_mf1i(home_row, away_row, game)
        mg3 = score_mf3i(home_row, away_row, game)

        # M3 — score up to 3 batters per game if provided in games.csv
        # Add columns home_batter1, home_batter2, away_batter1, etc. to games.csv
        m3_results = []
        for side, opp_row in [("home", away_row), ("away", home_row)]:
            for i in range(1, 4):
                batter_col = f"{side}_batter{i}"
                pos_col    = f"{side}_batter{i}_pos"
                batter_name = game.get(batter_col, "")
                lineup_pos  = game.get(pos_col, i + 2)  # default mid-lineup
                if batter_name and str(batter_name).strip() not in ["", "nan"]:
                    m3 = score_mbat(batter_name, opp_row, game, batter_df, lineup_pos, fg_batter_df)
                    m3["side"] = side
                    m3_results.append(m3)

        # Format M3 output for row
        m3_summary = []
        for m3r in m3_results:
            if m3r.get("probability", 0) >= 58:
                m3_summary.append(f"{m3r['batter']} {m3r['probability']}% {m3r['verdict']}")

        row = {
            "game_date":  game.get("game_date", ""),
            "matchup":    f"{away_sp} @ {home_sp}",
            "month_scalar": get_month_scalar(game.get("game_date", "")),
            "home_team":  game.get("home_team", ""),
            "away_team":  game.get("away_team", ""),
            "home_sp":    home_sp,
            "away_sp":    away_sp,

            "MF5i_lean":    mf5["lean"],
            "MF5i_prob":    mf5["probability"],
            "MF5i_verdict": mf5["verdict"],
            "MF5_cf":      mf5["command_floor_notes"],

            "MFull_lean":     m2["lean"],
            "MFull_prob":     m2["probability"],
            "MFull_verdict":  m2["verdict"],
            "MFull_proj_total": m2["projected_total"],
            "MFull_coors":    "⚠️ COORS" if m2.get("coors_flag") else "",
            "MFull_blowout":  "⚠️ BLOWOUT RISK" if m2.get("blowout_risk") else "",

            "MF1i_lean":    mf1["lean"],
            "MF1i_prob":    mf1["probability"],
            "MF1i_verdict": mf1["verdict"],
            "MF1_proj":    mf1["projected_f1_total"],

            "MF3i_lean":    mg3["lean"],
            "MF3i_prob":    mg3["probability"],
            "MF3i_verdict": mg3["verdict"],
            "MG3_proj":    mg3["projected_f3_total"],
            "MG3_cf":      mg3["command_floor_notes"],
            "M3_props":    " | ".join(m3_summary) if m3_summary else "—",
        }

        # Backtest if actuals available
        bt = evaluate_backtest(row, game)
        row.update(bt)

        results.append(row)

        print(f"  ✅ {game.get('game_date','')} | {away_sp} @ {home_sp}")
        print(f"     MF5i: {mf5['lean']} ({mf5['probability']}%) {mf5['verdict']}")
        print(f"     MFull:  {m2['lean']} ({m2['probability']}%) {m2['verdict']}")
        print(f"     MF1i: {mf1['lean']} ({mf1['probability']}%) {mf1['verdict']}")
        print(f"     MF3i: {mg3['lean']} ({mg3['probability']}%) {mg3['verdict']}")

        # Summary line — only show actionable picks
        takes = []
        for label, result in [("MF5i", mf5), ("MFull", m2), ("MF1i", mf1), ("MF3i", mg3)]:
            if result["probability"] >= PICK_THRESHOLD * 100 and result["lean"] != "NEUTRAL":
                takes.append(f"{label} {result['lean']} {result['probability']}%")
        if takes:
            # Add best book info from odds map
            home_abbr_g = str(game.get("home_team_abbr","")).upper()
            away_abbr_g = str(game.get("away_team_abbr","")).upper()
            print(f"     🎯 PICKS: {' | '.join(takes)}")
        else:
            print(f"     — No picks above {int(PICK_THRESHOLD*100)}% threshold")
        if m3_results:
            print(f"     M3 props:")
            for m3r in m3_results:
                if m3r.get("probability", 0) > 0:
                    print(f"       {m3r['side'].upper()} {m3r['batter']}: {m3r['probability']}% {m3r['verdict']}")
        print()

    if not results:
        print("❌ No games scored. Check pitcher name matching.")
        sys.exit(1)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n📊 Results saved → {OUTPUT_CSV}")

    # Backtest summary
    for model in ["MF5i", "MF1i", "MF3i", "MFull"]:
        col = f"{model}_result"
        if col in results_df.columns:
            wins   = (results_df[col] == "✅ WIN").sum()
            losses = (results_df[col] == "❌ LOSS").sum()
            total  = wins + losses
            pct    = f"{wins/total*100:.1f}%" if total > 0 else "N/A"
            print(f"  {model}: {wins}-{losses} ({pct}) from {total} scored games")

    generate_html_report(results_df)
    print(f"\n📄 HTML report saved → {OUTPUT_HTML}")

    if unmatched:
        unique_unmatched = list(set(unmatched))
        print(f"\n⚠️  Unmatched pitchers ({len(unique_unmatched)}): {unique_unmatched}")
        print("   → Check spelling in games.csv vs CSV downloads (FIRST LAST format)")


# ─── HTML REPORT ─────────────────────────────────────────────────────────────

def generate_html_report(df):
    MODEL_COLOR = {
        "MFull": "#2EAB6A",
        "MF5i":  "#3A8FC7",
        "MF3i":  "#D4804A",
        "MF1i":  "#5B9BD5",
    }

    def verdict_color(v):
        if "STRONG" in str(v): return "#2EAB6A"
        if "🟡" in str(v):     return "#E8C96A"
        if "🟠" in str(v):     return "#D4804A"
        return "#E05555"

    def result_badge(result_str):
        s = str(result_str).strip()
        if "WIN"  in s: return "<div class='badge badge-win'>✅ WIN</div>"
        if "LOSS" in s: return "<div class='badge badge-loss'>❌ LOSS</div>"
        if "PUSH" in s: return "<div class='badge badge-push'>— PUSH</div>"
        return ""

    def model_cell(model, lean, prob, verdict, result_str,
                   game_key, home, away, date):
        """
        Self-contained cell: prediction → log/skip buttons (if actionable) → result badge.
        """
        border     = MODEL_COLOR.get(model, "#999")
        txt_color  = verdict_color(verdict)
        badge      = result_badge(result_str)
        is_pick    = (str(lean) != "NEUTRAL" and float(str(prob) or 0) >= PICK_THRESHOLD * 100)

        btn_html = ""
        if is_pick:
            pick_key = f"{game_key}|{model}|{lean}"
            btn_html = f"""
            <div data-pick-key="{pick_key}" class='pick-actions'>
              <button class="btn btn-log" onclick="showLogForm(
                this.closest('[data-pick-key]'),
                '{game_key}','{home}','{away}',
                '{model}','{lean} {prob}%','{prob}','{date}'
              )">📝 Log Bet</button>
              <button class="btn btn-skip" onclick="skipBet(this)">Skip</button>
            </div>"""

        return (
            f"<td style='border-left:3px solid {border}'>"
            f"  <span class='lean' style='color:{txt_color}'>{lean}</span><br>"
            f"  <small class='verdict'>{prob}% {verdict}</small>"
            f"  {btn_html}"
            f"  {badge}"
            f"</td>"
        )

    rows_html = ""
    for _, r in df.iterrows():
        game_key = str(r['matchup']).replace("'", "").replace('"', "")
        home     = str(r.get('home_team', '')).replace("'", "")
        away     = str(r.get('away_team', '')).replace("'", "")
        home_sp  = str(r.get('home_sp', '')).title()
        away_sp  = str(r.get('away_sp', '')).title()
        date     = str(r.get('game_date', ''))

        matchup_html = (
            f"<b class='teams'>{away} @ {home}</b><br>"
            f"<span class='pitchers'>{away_sp} vs {home_sp}</span>"
        )

        props_val = r.get('M3_props', '—')
        props_td  = (
            f"<td style='border-left:3px solid #9B59B6'>"
            f"<small style='color:#9B59B6;font-weight:600'>Hit Props</small><br>"
            f"<small>{props_val}</small></td>"
        ) if str(props_val) != '—' else (
            f"<td style='border-left:3px solid #ddd;color:#ccc'><small>—</small></td>"
        )

        def mc(model, lean_col, prob_col, res_col):
            return model_cell(
                model,
                lean  = r.get(lean_col, 'NEUTRAL'),
                prob  = r.get(prob_col, 0),
                verdict = r.get(lean_col.replace('_lean','_verdict'), ''),
                result_str = r.get(res_col, ''),
                game_key=game_key, home=home, away=away, date=date,
            )

        rows_html += f"""
        <tr>
          <td class='date-cell'>{date}</td>
          <td class='matchup-cell'>{matchup_html}</td>
          {mc('MFull', 'MFull_lean', 'MFull_prob', 'MFull_result')}
          {mc('MF5i',  'MF5i_lean',  'MF5i_prob',  'MF5i_result')}
          {mc('MF3i',  'MF3i_lean',  'MF3i_prob',  'MF3i_result')}
          {mc('MF1i',  'MF1i_lean',  'MF1i_prob',  'MF1i_result')}
          {props_td}
        </tr>"""

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset='utf-8'>
<title>MLB Model Scores</title>
<style>
  /* ── Layout ── */
  body {{ font-family: Arial, sans-serif; background: #F0F2F5; color: #1F3A5F; margin: 0; padding: 20px; }}
  h1 {{ background: #1F3A5F; color: white; padding: 16px 24px; border-radius: 8px 8px 0 0; margin: 0; font-size: 20px; letter-spacing: 0.3px; }}
  .subtitle {{ background: #2A5F9E; color: #E8E8E8; padding: 8px 24px 9px; font-size: 11.5px; border-radius: 0 0 8px 8px; margin-bottom: 20px; }}
  .subtitle .dot {{ font-size: 16px; vertical-align: middle; margin-right: 3px; }}

  /* ── Table ── */
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.07); }}
  th {{ background: #1F3A5F; color: white; padding: 10px 12px; text-align: left; font-size: 11px; font-weight: 600; line-height: 1.5; }}
  td {{ padding: 11px 13px; border-bottom: 1px solid #EBEBEB; font-size: 12px; vertical-align: top; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #F7F9FC; }}

  /* ── Matchup ── */
  .date-cell {{ white-space:nowrap; color:#999; font-size:11px; padding-top:13px; }}
  .matchup-cell {{ min-width: 160px; }}
  .teams {{ font-size: 13px; font-weight: 700; }}
  .pitchers {{ font-size: 10px; color: #999; }}

  /* ── Model cells ── */
  .lean {{ font-weight: 700; font-size: 13px; }}
  .verdict {{ color: #999; }}

  /* ── Pick action buttons ── */
  .pick-actions {{ margin-top: 7px; }}
  .btn {{ border: none; border-radius: 4px; padding: 4px 9px; font-size: 10px; cursor: pointer; margin-right: 4px; font-weight: 600; }}
  .btn-log  {{ background: #1F3A5F; color: white; }}
  .btn-log:hover {{ background: #2A5F9E; }}
  .btn-skip {{ background: #EBEBEB; color: #666; }}
  .btn-skip:hover {{ background: #D5D5D5; }}

  /* ── Result badges ── */
  .badge {{ display: inline-block; margin-top: 7px; padding: 3px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; letter-spacing: 0.2px; }}
  .badge-win  {{ background: #D6F0E0; color: #1A6932; }}
  .badge-loss {{ background: #FAD7D7; color: #8B1A1A; }}
  .badge-push {{ background: #EBEBEB; color: #666; }}

  /* ── Footer ── */
  .gen {{ color: #bbb; font-size: 11px; margin-top: 14px; }}
  #srv-status {{ font-size: 11px; margin-bottom: 14px; padding: 6px 12px; background: #FFF8E6; border-left: 3px solid #D4804A; border-radius: 4px; color: #8B5E00; display: inline-block; }}
</style>
</head><body>
<h1>MLB Model Scoring Report</h1>
<div class='subtitle'>
  <span class='dot' style='color:#2EAB6A'>&#9646;</span><b>MFull</b> Full Game ML &nbsp;&nbsp;
  <span class='dot' style='color:#3A8FC7'>&#9646;</span><b>MF5i</b> Innings 1–5 ML &nbsp;&nbsp;
  <span class='dot' style='color:#D4804A'>&#9646;</span><b>MF3i</b> Inn. 1–3 Total &nbsp;&nbsp;
  <span class='dot' style='color:#5B9BD5'>&#9646;</span><b>MF1i</b> Inn. 1 Total
</div>
<div id='srv-status'>⚠️ Tracker offline — run: python tracker_server.py</div>
<div id='stats-bar' style='display:none'></div>
<table>
  <tr>
    <th>Date</th>
    <th>Matchup</th>
    <th style='border-left:3px solid #2EAB6A' title='Full-game moneyline — SP matchup, bullpen, lineup, weather, park factors &amp; home-field edge'>
      <span style='color:#2EAB6A'>MFull</span><br><span style='font-weight:normal;color:#7aafff'>Full Game ML</span></th>
    <th style='border-left:3px solid #3A8FC7' title='First-5-innings moneyline — starting pitchers only; bullpen &amp; late-game factors excluded'>
      <span style='color:#3A8FC7'>MF5i</span><br><span style='font-weight:normal;color:#7aafff'>Inn. 1–5 ML</span></th>
    <th style='border-left:3px solid #D4804A' title='First-3-innings over/under — early scoring pace, SP command floor, top-of-order lineup'>
      <span style='color:#D4804A'>MF3i</span><br><span style='font-weight:normal;color:#7aafff'>Inn. 1–3 Total</span></th>
    <th style='border-left:3px solid #5B9BD5' title='First-inning over/under — opening frame scoring, leadoff matchups, SP first-inning tendencies'>
      <span style='color:#5B9BD5'>MF1i</span><br><span style='font-weight:normal;color:#7aafff'>Inn. 1 Total</span></th>
    <th style='border-left:3px solid #9B59B6' title='Hit props derived from first-3-innings model projections'>
      <span style='color:#9B59B6'>Props</span><br><span style='font-weight:normal;color:#7aafff'>Hit Props</span></th>
  </tr>
  {rows_html}
</table>
<p class='gen'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; {len(df)} games scored</p>
</body></html>"""

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    run_pipeline()
