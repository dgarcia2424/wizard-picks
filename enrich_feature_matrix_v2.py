# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding="utf-8", errors="replace")
"""
enrich_feature_matrix_v2.py
Adds 8 new feature groups to feature_matrix_enriched.parquet.
Output: feature_matrix_enriched_v2.parquet
"""

import os
import re
import unicodedata
import warnings
import numpy as np
import pandas as pd
from glob import glob
from math import floor

warnings.filterwarnings("ignore")

BASE = r"C:\Users\garci\OneDrive\Documents\Claude\mlb_model_pipeline_dan"
STATCAST = os.path.join(BASE, "data", "statcast")

# -
# Helpers
# -

def strip_accents(s):
    """Remove accents / diacritics from a string."""
    if not isinstance(s, str):
        return s
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

def norm_name(s):
    """Normalize a name to uppercase ASCII (no accents, extra spaces)."""
    if not isinstance(s, str) or not s:
        return ""
    return re.sub(r"\s+", " ", strip_accents(s).upper().strip())

def sc_to_fm_name(sc_name):
    """Convert 'Last, First' (statcast) to 'FIRST LAST' (FM) format."""
    if not isinstance(sc_name, str) or "," not in sc_name:
        return norm_name(sc_name)
    parts = sc_name.split(",", 1)
    last = parts[0].strip()
    first = parts[1].strip()
    return norm_name(f"{first} {last}")

def fg_to_fm_name(fg_name):
    """FanGraphs name is 'First Last'; FM is 'FIRST LAST' (already normalized)."""
    return norm_name(fg_name)

def ip_to_decimal(ip):
    """Convert FanGraphs IP format (6.2 = 6⅔ inn) to decimal innings."""
    ip = pd.to_numeric(ip, errors="coerce")
    full = np.floor(ip)
    frac = ip % 1
    return full + frac * (10 / 3)

def fill_nan_group_median(df, cols, group_col="season"):
    """Fill NaN values in cols with the median of the group_col group."""
    for col in cols:
        if col in df.columns:
            medians = df.groupby(group_col)[col].transform("median")
            df[col] = df[col].fillna(medians)
            # still-null (whole group is null) -> overall median
            overall = df[col].median()
            df[col] = df[col].fillna(overall)
    return df


# -
# Load feature matrix
# -
print("=" * 70)
print("Loading feature_matrix_enriched.parquet ...")
fm = pd.read_parquet(os.path.join(BASE, "feature_matrix_enriched.parquet"))
print(f"  Input shape: {fm.shape}")

# Ensure game_date is datetime and sorted
fm["game_date"] = pd.to_datetime(fm["game_date"])
fm = fm.sort_values(["game_date", "game_pk"]).reset_index(drop=True)

new_cols_all = []   # track every new column added


# -
# Group 1 -- Lineup wRC+ (per-game, from lineup_quality files)
# -
print("\n" + "=" * 70)
print("Group 1 -- Lineup wRC+ ...")

# Per-date files (daily format: lineup_quality_YYYY-MM-DD.parquet)
lq_files = sorted(glob(os.path.join(STATCAST, "lineup_quality_????-??-??.parquet")))
# Annual backfill files (lineup_quality_YYYY.parquet from backfill_lineup_quality.py)
lq_annual_files = sorted(glob(os.path.join(STATCAST, "lineup_quality_????.parquet")))
print(f"  Found {len(lq_files)} daily + {len(lq_annual_files)} annual lineup_quality files")

lq_frames = []
for f in lq_annual_files + lq_files:   # annual first (lower priority), daily overrides
    try:
        df = pd.read_parquet(f)
        lq_frames.append(df)
    except Exception as e:
        print(f"  WARN: could not read {os.path.basename(f)}: {e}")

if lq_frames:
    lq_all = pd.concat(lq_frames, ignore_index=True)
    lq_all["game_pk"] = lq_all["game_pk"].astype("int64")

    # For each game_pk+team get the latest row (in case duplicates)
    lq_all = lq_all.sort_values("game_date").groupby(["game_pk", "team"]).last().reset_index()

    # Separate home / away by joining to FM on game_pk
    gm_teams = fm[["game_pk", "home_team", "away_team"]].drop_duplicates()

    lq_home = lq_all.merge(
        gm_teams, left_on=["game_pk", "team"], right_on=["game_pk", "home_team"], how="inner"
    )[["game_pk", "team", "lineup_wrc_plus",
       "lineup_xwoba_vs_rhp" if "lineup_xwoba_vs_rhp" in lq_all.columns else None,
       "lineup_xwoba_vs_lhp" if "lineup_xwoba_vs_lhp" in lq_all.columns else None]
    ]
    lq_home = lq_home.drop(columns=[c for c in lq_home.columns if c is None], errors="ignore")

    # Build home cols
    home_cols = {"lineup_wrc_plus": "home_lineup_wrc_plus"}
    if "lineup_xwoba_vs_rhp" in lq_all.columns:
        home_cols["lineup_xwoba_vs_rhp"] = "home_lineup_xwoba_vs_rhp"
    if "lineup_xwoba_vs_lhp" in lq_all.columns:
        home_cols["lineup_xwoba_vs_lhp"] = "home_lineup_xwoba_vs_lhp"

    lq_home_final = (
        lq_all.merge(gm_teams[["game_pk", "home_team"]],
                     left_on=["game_pk", "team"], right_on=["game_pk", "home_team"], how="inner")
              [["game_pk"] + list(home_cols.keys())]
              .rename(columns=home_cols)
    )

    away_cols = {"lineup_wrc_plus": "away_lineup_wrc_plus"}
    if "lineup_xwoba_vs_rhp" in lq_all.columns:
        away_cols["lineup_xwoba_vs_rhp"] = "away_lineup_xwoba_vs_rhp"
    if "lineup_xwoba_vs_lhp" in lq_all.columns:
        away_cols["lineup_xwoba_vs_lhp"] = "away_lineup_xwoba_vs_lhp"

    lq_away_final = (
        lq_all.merge(gm_teams[["game_pk", "away_team"]],
                     left_on=["game_pk", "team"], right_on=["game_pk", "away_team"], how="inner")
              [["game_pk"] + list(away_cols.keys())]
              .rename(columns=away_cols)
    )

    fm = fm.merge(lq_home_final, on="game_pk", how="left")
    fm = fm.merge(lq_away_final, on="game_pk", how="left")

    # Compute diff
    if "home_lineup_wrc_plus" in fm.columns and "away_lineup_wrc_plus" in fm.columns:
        fm["lineup_wrc_plus_diff"] = fm["home_lineup_wrc_plus"] - fm["away_lineup_wrc_plus"]

    g1_cols = list(home_cols.values()) + list(away_cols.values()) + ["lineup_wrc_plus_diff"]
    g1_cols = [c for c in g1_cols if c in fm.columns]

    # Fall back to season median per team for missing
    for base_col in home_cols.values():
        if base_col in fm.columns:
            med = fm.groupby("season")[base_col].transform("median")
            fm[base_col] = fm[base_col].fillna(med)
    for base_col in away_cols.values():
        if base_col in fm.columns:
            med = fm.groupby("season")[base_col].transform("median")
            fm[base_col] = fm[base_col].fillna(med)
    if "lineup_wrc_plus_diff" in fm.columns:
        fm["lineup_wrc_plus_diff"] = fm["home_lineup_wrc_plus"] - fm["away_lineup_wrc_plus"]

    print(f"  Added {len(g1_cols)} columns: {g1_cols}")
    new_cols_all.extend(g1_cols)
else:
    print("  WARN: No lineup_quality files found, skipping Group 1")


# -
# Group 2 -- SP FIP and LOB% (computed from statcast pitch-level data)
# Note: pitching_fg_full files have empty Name/player_id columns so we compute
# FIP/LOB% directly from statcast events.  Prior-year stats used for leakage prevention.
# -
print("\n" + "=" * 70)
print("Group 2 -- SP FIP and LOB% (from statcast) ...")

def get_prior_fg_year(game_year):
    """Return the prior year to use for SP stats (prevents look-ahead)."""
    if game_year <= 2023:
        return 2023
    return game_year - 1

FIP_SC_COLS = [
    "pitcher", "player_name", "game_year",
    "events", "woba_denom",
    "post_bat_score", "bat_score",
]

# Event mappings
OUT_EVENTS = {
    "field_out": 1, "strikeout": 1, "force_out": 1,
    "grounded_into_double_play": 2, "double_play": 2, "triple_play": 3,
    "strikeout_double_play": 2, "fielders_choice_out": 1,
    "sac_fly": 1, "sac_bunt": 1, "sac_fly_double_play": 2,
}
HIT_EVENTS = {"single", "double", "triple", "home_run"}

fg_fip_lookup = {}   # {(norm_name, stat_year): {'fip': x, 'lob_pct': x}}

for yr in [2023, 2024, 2025]:
    sc_path = os.path.join(STATCAST, f"statcast_{yr}.parquet")
    if not os.path.exists(sc_path):
        print(f"  WARN: statcast_{yr}.parquet not found, skipping FIP year {yr}")
        continue
    print(f"  Computing FIP/LOB from statcast_{yr} ...", end="", flush=True)
    sc_fip = pd.read_parquet(sc_path, columns=FIP_SC_COLS)
    # Only last pitch of each PA
    pa = sc_fip[sc_fip["woba_denom"] >= 1].copy()

    pa["is_k"] = pa["events"].isin(["strikeout", "strikeout_double_play"]).astype(int)
    pa["is_bb"] = (pa["events"] == "walk").astype(int)
    pa["is_hbp"] = (pa["events"] == "hit_by_pitch").astype(int)
    pa["is_hr"] = (pa["events"] == "home_run").astype(int)
    pa["is_hit"] = pa["events"].isin(HIT_EVENTS).astype(int)
    pa["outs_on_play"] = pa["events"].map(OUT_EVENTS).fillna(0)
    # Runs scored on this PA = increase in batting team's score
    pa["runs_on_play"] = (pa["post_bat_score"] - pa["bat_score"]).clip(lower=0)

    p_stats = pa.groupby(["pitcher", "player_name"]).agg(
        K=("is_k", "sum"),
        BB=("is_bb", "sum"),
        HBP=("is_hbp", "sum"),
        HR=("is_hr", "sum"),
        H=("is_hit", "sum"),
        R=("runs_on_play", "sum"),
        outs=("outs_on_play", "sum"),
    ).reset_index()

    p_stats["IP_dec"] = p_stats["outs"] / 3.0
    p_stats = p_stats[p_stats["IP_dec"] >= 5]  # filter extreme noise

    # FIP = (13*HR + 3*(BB+HBP) - 2*K) / IP + 3.15
    p_stats["fip"] = (
        (13 * p_stats["HR"] + 3 * (p_stats["BB"] + p_stats["HBP"]) - 2 * p_stats["K"])
        / p_stats["IP_dec"].replace(0, np.nan)
        + 3.15
    )

    # LOB% = (H + BB + HBP - R) / (H + BB + HBP - 1.4*HR)
    lobdenom = p_stats["H"] + p_stats["BB"] + p_stats["HBP"] - 1.4 * p_stats["HR"]
    lobnumer = p_stats["H"] + p_stats["BB"] + p_stats["HBP"] - p_stats["R"]
    p_stats["lob_pct"] = (lobnumer / lobdenom.replace(0, np.nan)).clip(0, 1)

    # Normalize name to FM format
    p_stats["norm_name"] = p_stats["player_name"].apply(sc_to_fm_name)

    for _, row in p_stats.iterrows():
        nm = row["norm_name"]
        if nm:
            fg_fip_lookup[(nm, yr)] = {"fip": row["fip"], "lob_pct": row["lob_pct"]}

    print(f" {len(p_stats)} pitchers in lookup")

print(f"  Total FIP lookup entries: {len(fg_fip_lookup)}")

# Assign to FM using prior-year stats
def get_sp_fip_stats(sp_name, game_year):
    nm = norm_name(sp_name)
    prior_yr = get_prior_fg_year(game_year)
    stats = fg_fip_lookup.get((nm, prior_yr), {})
    return stats.get("fip", np.nan), stats.get("lob_pct", np.nan)

fip_results = fm.apply(
    lambda r: pd.Series(get_sp_fip_stats(r["home_starter_name"], r["season"])), axis=1
)
fm["home_sp_fip"] = fip_results[0]
fm["home_sp_lob_pct"] = fip_results[1]

fip_results_a = fm.apply(
    lambda r: pd.Series(get_sp_fip_stats(r["away_starter_name"], r["season"])), axis=1
)
fm["away_sp_fip"] = fip_results_a[0]
fm["away_sp_lob_pct"] = fip_results_a[1]

fm["sp_fip_diff"] = fm["home_sp_fip"] - fm["away_sp_fip"]

g2_cols = ["home_sp_fip", "away_sp_fip", "sp_fip_diff", "home_sp_lob_pct", "away_sp_lob_pct"]

hit_rate_h = fm["home_sp_fip"].notna().mean()
hit_rate_a = fm["away_sp_fip"].notna().mean()
print(f"  FIP match rate (before fill): home={hit_rate_h:.1%}, away={hit_rate_a:.1%}")

fm = fill_nan_group_median(fm, g2_cols)
print(f"  Added: {g2_cols}")
new_cols_all.extend(g2_cols)


# -
# Group 3 -- SP Barrel% Against (from pitcher_exitvelo)
# -
print("\n" + "=" * 70)
print("Group 3 -- SP Barrel% Against ...")

ev_barrel_lookup = {}  # {(norm_name, yr): barrel_pct}

for yr in [2023, 2024, 2025]:
    ev_path = os.path.join(STATCAST, f"pitcher_exitvelo_{yr}.parquet")
    if not os.path.exists(ev_path):
        print(f"  WARN: pitcher_exitvelo_{yr}.parquet not found")
        continue
    ev = pd.read_parquet(ev_path)
    print(f"  pitcher_exitvelo_{yr}: {ev.shape}, cols: {list(ev.columns)}")

    # Name column: 'last_name, first_name'
    name_col = "last_name, first_name"
    if name_col not in ev.columns:
        print(f"  WARN: no name column in {yr} exitvelo")
        continue

    ev["norm_name"] = ev[name_col].apply(sc_to_fm_name)

    # Barrel% = barrels / attempts
    if "attempts" in ev.columns:
        ev["barrel_pct"] = ev["barrels"] / ev["attempts"].replace(0, np.nan)
    elif "pa" in ev.columns:
        ev["barrel_pct"] = ev["barrels"] / ev["pa"].replace(0, np.nan)
    else:
        # Use brl_percent / 100 if available
        if "brl_percent" in ev.columns:
            ev["barrel_pct"] = ev["brl_percent"] / 100.0
        else:
            print(f"  WARN: no attempts/pa column in {yr} exitvelo")
            continue

    for _, row in ev.iterrows():
        nm = row["norm_name"]
        if nm:
            ev_barrel_lookup[(nm, yr)] = row["barrel_pct"]

    print(f"  Year {yr}: {len(ev)} pitchers in exitvelo lookup")

def get_sp_barrel(sp_name, game_year):
    nm = norm_name(sp_name)
    prior_yr = get_prior_fg_year(game_year)
    return ev_barrel_lookup.get((nm, prior_yr), np.nan)

fm["home_sp_barrel_pct_against"] = fm.apply(
    lambda r: get_sp_barrel(r["home_starter_name"], r["season"]), axis=1
)
fm["away_sp_barrel_pct_against"] = fm.apply(
    lambda r: get_sp_barrel(r["away_starter_name"], r["season"]), axis=1
)
fm["sp_barrel_pct_against_diff"] = fm["home_sp_barrel_pct_against"] - fm["away_sp_barrel_pct_against"]

g3_cols = ["home_sp_barrel_pct_against", "away_sp_barrel_pct_against", "sp_barrel_pct_against_diff"]
fm = fill_nan_group_median(fm, g3_cols)

print(f"  Barrel match rate: home={fm['home_sp_barrel_pct_against'].notna().mean():.1%}")
print(f"  Added: {g3_cols}")
new_cols_all.extend(g3_cols)


# -
# Helper: build team score history from statcast + FM + actuals
# -
print("\n" + "=" * 70)
print("Building team score history for Groups 4 + 7 ...")

# Collect final scores per game from statcast (all years) + actuals
score_frames = []

for yr in [2023, 2024, 2025, 2026]:
    sc_path = os.path.join(STATCAST, f"statcast_{yr}.parquet")
    if os.path.exists(sc_path):
        tmp = pd.read_parquet(sc_path,
                              columns=["game_pk", "game_date", "home_team", "away_team",
                                       "post_home_score", "post_away_score"])
        final = (tmp.sort_values("game_date")
                    .groupby("game_pk")
                    .agg(game_date=("game_date", "first"),
                         home_team=("home_team", "first"),
                         away_team=("away_team", "first"),
                         home_score=("post_home_score", "max"),
                         away_score=("post_away_score", "max"))
                    .reset_index())
        score_frames.append(final)

# Also pull from FM itself (has home_score/away_score for most games)
fm_scores = fm[["game_pk", "game_date", "home_team", "away_team", "home_score", "away_score"]].dropna(subset=["home_score"])
score_frames.append(fm_scores)

# Also try actuals_2026
act_path = os.path.join(STATCAST, "actuals_2026.parquet")
if os.path.exists(act_path):
    act = pd.read_parquet(act_path, columns=["game_pk", "game_date", "home_team", "away_team",
                                              "home_score_final", "away_score_final"])
    act = act.rename(columns={"home_score_final": "home_score", "away_score_final": "away_score"})
    score_frames.append(act)

scores_all = pd.concat(score_frames, ignore_index=True)
scores_all = scores_all.dropna(subset=["home_score", "away_score"])
scores_all["game_date"] = pd.to_datetime(scores_all["game_date"])
# Deduplicate: keep one row per game_pk
scores_all = scores_all.sort_values("game_date").groupby("game_pk").first().reset_index()

print(f"  Total games with scores: {len(scores_all)}")

# Expand to per-team rows for rolling
team_game_rows = []
for _, r in scores_all.iterrows():
    team_game_rows.append({
        "game_pk": r["game_pk"],
        "game_date": r["game_date"],
        "team": r["home_team"],
        "rs": r["home_score"],
        "ra": r["away_score"],
        "is_home": True,
    })
    team_game_rows.append({
        "game_pk": r["game_pk"],
        "game_date": r["game_date"],
        "team": r["away_team"],
        "rs": r["away_score"],
        "ra": r["home_score"],
        "is_home": False,
    })

team_games = pd.DataFrame(team_game_rows).sort_values(["team", "game_date", "game_pk"]).reset_index(drop=True)
print(f"  Per-team game rows: {len(team_games)}")


# -
# Group 4 -- Rolling Run Differential + Pythagorean Win%
# -
print("\n" + "=" * 70)
print("Group 4 -- Rolling Run Differential + Pythagorean Win% ...")

W = 15

def pyth_wp(rs_sum, ra_sum, exp=1.83):
    """Pythagorean Win%."""
    if ra_sum == 0 and rs_sum == 0:
        return 0.5
    denom = rs_sum ** exp + ra_sum ** exp
    if denom == 0:
        return 0.5
    return (rs_sum ** exp) / denom

tg = team_games.copy()
tg = tg.sort_values(["team", "game_date", "game_pk"]).reset_index(drop=True)

# Rolling metrics per team (shift 1)
tg["rd"] = tg["rs"] - tg["ra"]
tg["rolling_rd_15g"] = (
    tg.groupby("team")["rd"]
      .transform(lambda x: x.shift(1).rolling(W, min_periods=3).mean())
)
tg["rolling_rs_15g"] = (
    tg.groupby("team")["rs"]
      .transform(lambda x: x.shift(1).rolling(W, min_periods=3).sum())
)
tg["rolling_ra_15g"] = (
    tg.groupby("team")["ra"]
      .transform(lambda x: x.shift(1).rolling(W, min_periods=3).sum())
)
tg["pyth_wp_15g"] = tg.apply(
    lambda r: pyth_wp(r["rolling_rs_15g"], r["rolling_ra_15g"])
    if pd.notna(r["rolling_rs_15g"]) else np.nan,
    axis=1
)

# Create lookup: (team, game_pk) -> metrics
rd_lookup = tg.set_index(["team", "game_pk"])[["rolling_rd_15g", "pyth_wp_15g"]].to_dict("index")

for side in ["home", "away"]:
    fm[f"{side}_rolling_rd_15g"] = fm.apply(
        lambda r: rd_lookup.get((r[f"{side}_team"], r["game_pk"]), {}).get("rolling_rd_15g", np.nan), axis=1
    )
    fm[f"{side}_pyth_win_pct_15g"] = fm.apply(
        lambda r: rd_lookup.get((r[f"{side}_team"], r["game_pk"]), {}).get("pyth_wp_15g", np.nan), axis=1
    )

fm["rolling_rd_diff"] = fm["home_rolling_rd_15g"] - fm["away_rolling_rd_15g"]
fm["pyth_win_pct_diff"] = fm["home_pyth_win_pct_15g"] - fm["away_pyth_win_pct_15g"]

g4_cols = ["home_rolling_rd_15g", "away_rolling_rd_15g", "rolling_rd_diff",
           "home_pyth_win_pct_15g", "away_pyth_win_pct_15g", "pyth_win_pct_diff"]
fm = fill_nan_group_median(fm, g4_cols)
print(f"  Added: {g4_cols}")
new_cols_all.extend(g4_cols)


# -
# Group 5 -- Team Rolling Offensive Metrics (from statcast)
# -
print("\n" + "=" * 70)
print("Group 5 -- Team Rolling Offensive Metrics ...")

SC_OFF_COLS = [
    "game_pk", "game_date", "home_team", "away_team",
    "inning_topbot", "woba_value", "woba_denom",
    "estimated_woba_using_speedangle", "launch_speed_angle", "type",
]

off_metrics_frames = []  # list of (team, game_pk, game_date, woba_num, woba_den, xwoba_sum, xwoba_n, barrel_n, pa_n)

for yr in [2023, 2024, 2025, 2026]:
    sc_path = os.path.join(STATCAST, f"statcast_{yr}.parquet")
    if not os.path.exists(sc_path):
        print(f"  WARN: statcast_{yr}.parquet not found, skipping")
        continue
    print(f"  Loading statcast_{yr} ...", end="", flush=True)
    sc = pd.read_parquet(sc_path, columns=SC_OFF_COLS)
    print(f" {len(sc):,} rows")

    # Keep only last pitch of each PA (woba_denom >= 1)
    pa_pitches = sc[sc["woba_denom"] >= 1].copy()

    # Assign batting team
    pa_pitches["batting_team"] = np.where(
        pa_pitches["inning_topbot"] == "Top",
        pa_pitches["away_team"],
        pa_pitches["home_team"],
    )

    # Per-game per-team offense aggregation
    def agg_off(g):
        return pd.Series({
            "woba_num": g["woba_value"].sum(skipna=True),
            "woba_den": g["woba_denom"].sum(skipna=True),
            "xwoba_sum": g["estimated_woba_using_speedangle"].sum(skipna=True),
            "xwoba_n": g["estimated_woba_using_speedangle"].notna().sum(),
            "barrel_n": (g["launch_speed_angle"] == 6).sum(),
            "pa_n": len(g),
        })

    game_off = (
        pa_pitches.groupby(["game_pk", "batting_team", "game_date"])
                  .apply(agg_off)
                  .reset_index()
    )
    off_metrics_frames.append(game_off)

if off_metrics_frames:
    off_all = pd.concat(off_metrics_frames, ignore_index=True)
    off_all["game_date"] = pd.to_datetime(off_all["game_date"])
    off_all = off_all.sort_values(["batting_team", "game_date", "game_pk"]).reset_index(drop=True)

    # Rolling 15g offense metrics, shift(1)
    def roll_mean(x, w=W):
        return x.shift(1).rolling(w, min_periods=3).mean()
    def roll_sum(x, w=W):
        return x.shift(1).rolling(w, min_periods=3).sum()

    grp = off_all.groupby("batting_team")
    off_all["roll_woba_num"] = grp["woba_num"].transform(lambda x: roll_sum(x))
    off_all["roll_woba_den"] = grp["woba_den"].transform(lambda x: roll_sum(x))
    off_all["roll_xwoba_sum"] = grp["xwoba_sum"].transform(lambda x: roll_sum(x))
    off_all["roll_xwoba_n"] = grp["xwoba_n"].transform(lambda x: roll_sum(x))
    off_all["roll_barrel_n"] = grp["barrel_n"].transform(lambda x: roll_sum(x))
    off_all["roll_pa_n"] = grp["pa_n"].transform(lambda x: roll_sum(x))

    off_all["team_woba_15g"] = off_all["roll_woba_num"] / off_all["roll_woba_den"].replace(0, np.nan)
    off_all["team_xwoba_off_15g"] = off_all["roll_xwoba_sum"] / off_all["roll_xwoba_n"].replace(0, np.nan)
    off_all["team_barrel_pct_15g"] = off_all["roll_barrel_n"] / off_all["roll_pa_n"].replace(0, np.nan)

    off_lookup = off_all.set_index(["batting_team", "game_pk"])[
        ["team_woba_15g", "team_xwoba_off_15g", "team_barrel_pct_15g"]
    ].to_dict("index")

    for side in ["home", "away"]:
        fm[f"{side}_team_woba_15g"] = fm.apply(
            lambda r: off_lookup.get((r[f"{side}_team"], r["game_pk"]), {}).get("team_woba_15g", np.nan), axis=1
        )
        fm[f"{side}_team_xwoba_off_15g"] = fm.apply(
            lambda r: off_lookup.get((r[f"{side}_team"], r["game_pk"]), {}).get("team_xwoba_off_15g", np.nan), axis=1
        )
        fm[f"{side}_team_barrel_pct_15g"] = fm.apply(
            lambda r: off_lookup.get((r[f"{side}_team"], r["game_pk"]), {}).get("team_barrel_pct_15g", np.nan), axis=1
        )

    fm["team_woba_diff_15g"] = fm["home_team_woba_15g"] - fm["away_team_woba_15g"]
    fm["team_xwoba_off_diff_15g"] = fm["home_team_xwoba_off_15g"] - fm["away_team_xwoba_off_15g"]
    fm["team_barrel_pct_diff_15g"] = fm["home_team_barrel_pct_15g"] - fm["away_team_barrel_pct_15g"]

    g5_cols = [
        "home_team_woba_15g", "away_team_woba_15g", "team_woba_diff_15g",
        "home_team_xwoba_off_15g", "away_team_xwoba_off_15g", "team_xwoba_off_diff_15g",
        "home_team_barrel_pct_15g", "away_team_barrel_pct_15g", "team_barrel_pct_diff_15g",
    ]
    fm = fill_nan_group_median(fm, g5_cols)
    print(f"  Added: {g5_cols}")
    new_cols_all.extend(g5_cols)
else:
    print("  WARN: No statcast data for Group 5")


# -
# Group 6 -- DER (Defensive Efficiency Rate)
# -
print("\n" + "=" * 70)
print("Group 6 -- DER ...")

BIP_OUTS = {
    "field_out": 1,
    "grounded_into_double_play": 2,
    "force_out": 1,
    "fielders_choice_out": 1,
    "double_play": 2,
    "triple_play": 3,
    "sac_fly": 1,
    "sac_bunt": 1,
}
BIP_OUT_EVENTS = set(BIP_OUTS.keys())
# Events that are outs (for DER)
NON_HR_EVENTS = None  # we'll exclude HR from BIP

DER_COLS = ["game_pk", "game_date", "home_team", "away_team",
            "inning_topbot", "events", "type", "woba_denom"]

der_frames = []

for yr in [2023, 2024, 2025, 2026]:
    sc_path = os.path.join(STATCAST, f"statcast_{yr}.parquet")
    if not os.path.exists(sc_path):
        continue
    print(f"  Loading statcast_{yr} for DER ...", end="", flush=True)
    sc = pd.read_parquet(sc_path, columns=DER_COLS)
    print(f" {len(sc):,} rows")

    # Last pitch of PA
    pa = sc[sc["woba_denom"] >= 1].copy()

    # BIP = type == 'X' and not HR
    bip = pa[(pa["type"] == "X") & (pa["events"] != "home_run")].copy()

    # Fielding team = team that is pitching
    bip["fielding_team"] = np.where(
        bip["inning_topbot"] == "Top",
        bip["home_team"],    # home team pitches when away bats (Top)
        bip["away_team"],    # away team pitches when home bats (Bot)
    )

    bip["bip_outs"] = bip["events"].map(BIP_OUTS).fillna(0)
    bip["is_bip_out"] = (bip["bip_outs"] > 0).astype(int)

    # Per game per fielding team
    game_der = (
        bip.groupby(["game_pk", "fielding_team", "game_date"])
           .agg(bip_count=("is_bip_out", "count"),
                bip_outs=("is_bip_out", "sum"))
           .reset_index()
    )
    game_der["der"] = game_der["bip_outs"] / game_der["bip_count"].replace(0, np.nan)
    der_frames.append(game_der)

if der_frames:
    der_all = pd.concat(der_frames, ignore_index=True)
    der_all["game_date"] = pd.to_datetime(der_all["game_date"])
    der_all = der_all.sort_values(["fielding_team", "game_date", "game_pk"]).reset_index(drop=True)

    der_all["team_der_15g"] = (
        der_all.groupby("fielding_team")["der"]
               .transform(lambda x: x.shift(1).rolling(W, min_periods=3).mean())
    )

    der_lookup = der_all.set_index(["fielding_team", "game_pk"])["team_der_15g"].to_dict()

    for side in ["home", "away"]:
        fm[f"{side}_team_der_15g"] = fm.apply(
            lambda r: der_lookup.get((r[f"{side}_team"], r["game_pk"]), np.nan), axis=1
        )

    fm["team_der_diff_15g"] = fm["home_team_der_15g"] - fm["away_team_der_15g"]

    g6_cols = ["home_team_der_15g", "away_team_der_15g", "team_der_diff_15g"]
    fm = fill_nan_group_median(fm, g6_cols)
    print(f"  Added: {g6_cols}")
    new_cols_all.extend(g6_cols)
else:
    print("  WARN: No statcast data for Group 6")


# -
# Group 7 -- Elo Ratings
# -
print("\n" + "=" * 70)
print("Group 7 -- Elo Ratings ...")

K = 20
ELO_INIT = 1500

# Use scores_all (built above for Group 4)
elo_games = scores_all[["game_pk", "game_date", "home_team", "away_team", "home_score", "away_score"]].copy()
elo_games = elo_games.dropna(subset=["home_score", "away_score"])
elo_games = elo_games.sort_values(["game_date", "game_pk"]).reset_index(drop=True)

elo_ratings = {}  # team -> current Elo

def get_elo(team):
    return elo_ratings.get(team, ELO_INIT)

pre_game_elo = []  # (game_pk, home_elo, away_elo)

for _, row in elo_games.iterrows():
    ht = row["home_team"]
    at = row["away_team"]
    he = get_elo(ht)
    ae = get_elo(at)

    # Record pre-game Elo
    pre_game_elo.append({"game_pk": row["game_pk"], "home_elo": he, "away_elo": ae})

    # Update only if scores available and not a tie
    hs = row["home_score"]
    as_ = row["away_score"]
    if pd.isna(hs) or pd.isna(as_) or hs == as_:
        continue

    expected_home = 1 / (1 + 10 ** ((ae - he) / 400))
    home_win = 1 if hs > as_ else 0
    delta = K * (home_win - expected_home)
    elo_ratings[ht] = he + delta
    elo_ratings[at] = ae - delta

elo_df = pd.DataFrame(pre_game_elo)
elo_df["game_pk"] = elo_df["game_pk"].astype("int64")

fm = fm.merge(elo_df.rename(columns={"home_elo": "home_elo", "away_elo": "away_elo"}),
              on="game_pk", how="left")
fm["elo_diff"] = fm["home_elo"] - fm["away_elo"]

g7_cols = ["home_elo", "away_elo", "elo_diff"]
fm = fill_nan_group_median(fm, g7_cols)
print(f"  Elo coverage: {fm['home_elo'].notna().mean():.1%}")
print(f"  Added: {g7_cols}")
new_cols_all.extend(g7_cols)


# -
# Group 8 -- Catcher Framing proxy (from statcast)
# -
print("\n" + "=" * 70)
print("Group 8 -- Catcher Framing proxy ...")

FRAME_COLS = [
    "game_pk", "game_date", "home_team", "away_team",
    "type", "description", "plate_x", "plate_z", "sz_top", "sz_bot",
    "fielder_2", "inning_topbot",
]

framing_frames = []

for yr in [2023, 2024, 2025, 2026]:
    sc_path = os.path.join(STATCAST, f"statcast_{yr}.parquet")
    if not os.path.exists(sc_path):
        continue
    print(f"  Loading statcast_{yr} for framing ...", end="", flush=True)

    # Check columns available
    pq_meta = pd.read_parquet(sc_path, columns=["type"]).columns
    sc_full_cols = pd.read_parquet(sc_path, columns=FRAME_COLS[:3]).columns  # test read

    try:
        sc = pd.read_parquet(sc_path, columns=FRAME_COLS)
    except Exception as e:
        print(f" ERROR: {e}")
        continue

    print(f" {len(sc):,} rows")

    if "description" not in sc.columns or "plate_x" not in sc.columns:
        print(f"  WARN: Missing framing columns in {yr}, skipping")
        continue

    # Called pitches only
    called = sc[sc["description"].isin(["called_strike", "ball"])].copy()
    called = called.dropna(subset=["plate_x", "plate_z", "sz_top", "sz_bot", "fielder_2"])

    # Shadow zone: |plate_x| between 0.7 and 1.1, plate_z near strike zone border
    called["in_shadow"] = (
        (called["plate_x"].abs().between(0.7, 1.1)) |
        (called["plate_z"].between(called["sz_bot"] - 0.15, called["sz_bot"] + 0.15)) |
        (called["plate_z"].between(called["sz_top"] - 0.15, called["sz_top"] + 0.15))
    )

    called["is_called_strike"] = (called["description"] == "called_strike").astype(int)

    # Shadow zone pitches only
    shadow = called[called["in_shadow"]].copy()

    if len(shadow) == 0:
        print(f"  WARN: No shadow zone pitches in {yr}")
        continue

    # Overall called strike rate in shadow zone (league expected)
    league_csw_rate = shadow["is_called_strike"].mean()

    # Per-catcher framing in shadow zone
    catcher_stats = (
        shadow.groupby("fielder_2")
              .agg(shadow_pitches=("is_called_strike", "count"),
                   shadow_strikes=("is_called_strike", "sum"))
              .reset_index()
    )
    catcher_stats["actual_csw_rate"] = catcher_stats["shadow_strikes"] / catcher_stats["shadow_pitches"]
    catcher_stats["framing_runs_proxy"] = (
        (catcher_stats["actual_csw_rate"] - league_csw_rate)
        * catcher_stats["shadow_pitches"]
        / 1000  # per 1000 pitches scale
    )
    catcher_stats["season"] = yr

    # Map catchers to teams via fielder_2 -> team, season
    # Get the team each catcher played for (plurality)
    catcher_team = (
        called[called["inning_topbot"] == "Bot"]  # fielding team = home when Bot
        .groupby("fielder_2")["home_team"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else None)
        .reset_index()
        .rename(columns={"home_team": "team_bot"})
    )
    catcher_team2 = (
        called[called["inning_topbot"] == "Top"]
        .groupby("fielder_2")["away_team"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else None)
        .reset_index()
        .rename(columns={"away_team": "team_top"})
    )
    catcher_teams = catcher_team.merge(catcher_team2, on="fielder_2", how="outer")

    catcher_stats = catcher_stats.merge(catcher_teams, on="fielder_2", how="left")

    # Determine dominant team
    catcher_stats["team"] = catcher_stats["team_bot"].fillna(catcher_stats["team_top"])

    # Aggregate framing runs per team per season
    team_framing = (
        catcher_stats.groupby(["team", "season"])["framing_runs_proxy"]
                     .sum()
                     .reset_index()
    )
    framing_frames.append(team_framing)

# ── Prefer official Savant framing over proxy when available ─────────────────
official_framing_path = os.path.join(STATCAST, "catcher_framing_all.parquet")
_use_official = False

if os.path.exists(official_framing_path):
    try:
        off = pd.read_parquet(official_framing_path)
        off["season"] = off["season"].astype(int)
        official_framing_lookup = {}
        for _, row in off.iterrows():
            if pd.notna(row.get("team")) and pd.notna(row.get("framing_runs")):
                official_framing_lookup[(row["team"], int(row["season"]))] = float(row["framing_runs"])
        if official_framing_lookup:
            _use_official = True
            print(f"  Using OFFICIAL Savant framing: {len(official_framing_lookup)} team-season entries")
    except Exception as _e:
        print(f"  [WARN] Could not load official framing: {_e} — falling back to proxy")

if _use_official:
    def get_framing(team, game_year):
        prior_yr = get_prior_fg_year(game_year)
        return official_framing_lookup.get((team, prior_yr), np.nan)

    fm["home_catcher_framing_runs"] = fm.apply(
        lambda r: get_framing(r["home_team"], r["season"]), axis=1)
    fm["away_catcher_framing_runs"] = fm.apply(
        lambda r: get_framing(r["away_team"], r["season"]), axis=1)
    fm["catcher_framing_diff"] = fm["home_catcher_framing_runs"] - fm["away_catcher_framing_runs"]

    g8_cols = ["home_catcher_framing_runs", "away_catcher_framing_runs", "catcher_framing_diff"]
    fm = fill_nan_group_median(fm, g8_cols)
    print(f"  Official framing coverage: {fm['home_catcher_framing_runs'].notna().mean():.1%}")
    print(f"  Added: {g8_cols}")

elif framing_frames:
    framing_all = pd.concat(framing_frames, ignore_index=True)
    framing_all["season"] = framing_all["season"].astype(int)

    # Use prior year for framing (same leakage prevention)
    framing_lookup = {}  # (team, yr) -> framing_runs
    for _, row in framing_all.iterrows():
        framing_lookup[(row["team"], int(row["season"]))] = row["framing_runs_proxy"]

    def get_framing(team, game_year):
        prior_yr = get_prior_fg_year(game_year)
        return framing_lookup.get((team, prior_yr), np.nan)

    fm["home_catcher_framing_runs"] = fm.apply(
        lambda r: get_framing(r["home_team"], r["season"]), axis=1
    )
    fm["away_catcher_framing_runs"] = fm.apply(
        lambda r: get_framing(r["away_team"], r["season"]), axis=1
    )
    fm["catcher_framing_diff"] = fm["home_catcher_framing_runs"] - fm["away_catcher_framing_runs"]

    g8_cols = ["home_catcher_framing_runs", "away_catcher_framing_runs", "catcher_framing_diff"]
    fm = fill_nan_group_median(fm, g8_cols)
    print(f"  Proxy framing coverage: {fm['home_catcher_framing_runs'].notna().mean():.1%}")
    print(f"  Added: {g8_cols}")
    new_cols_all.extend(g8_cols)
else:
    print("  WARN: No framing data computed, skipping Group 8")


# -
# Merge any columns present in feature_matrix.parquet but not yet in fm
# (Phase 2 / QW features added after enriched.parquet was last rebuilt)
# -
base_fm_path = os.path.join(BASE, "feature_matrix.parquet")
if os.path.exists(base_fm_path):
    base_fm = pd.read_parquet(base_fm_path)
    missing_cols = [c for c in base_fm.columns if c not in fm.columns]
    if missing_cols:
        print(f"\n  Merging {len(missing_cols)} pass-through cols from feature_matrix.parquet ...")
        fm = fm.merge(base_fm[["game_pk"] + missing_cols], on="game_pk", how="left")
        print(f"  fm shape after merge: {fm.shape}")

# -
# Save output
# -
print("\n" + "=" * 70)
print("Saving feature_matrix_enriched_v2.parquet ...")
out_path = os.path.join(BASE, "feature_matrix_enriched_v2.parquet")
fm.to_parquet(out_path, index=False)

print(f"\n{'='*70}")
print(f"DONE.")
print(f"  Output shape: {fm.shape}  (input was 7610 x 155)")
print(f"  New columns added ({len(new_cols_all)} total):")
for col in new_cols_all:
    null_pct = fm[col].isna().mean() * 100 if col in fm.columns else 999
    print(f"    {col:45s}  null%={null_pct:.1f}")
print(f"\nSaved to: {out_path}")
