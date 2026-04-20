"""
enrich_feature_matrix.py
========================
Adds weather, bullpen availability, SP rest/durability, and 1st-inning split
features to feature_matrix_with_2026.parquet.

New features (19 total):
  Group 1 — Weather (3):       temp_f, wind_mph, wind_bearing
  Group 2 — Bullpen avail (4): home/away _bp_depleted_flag, _bp_pitches_rest1d
  Group 3 — SP days of rest (2): home/away _sp_days_rest
  Group 4 — SP avg IP (2):     home/away _sp_avg_ip
  Group 5 — 1st inning splits (8):
      home/away_1st_inn_run_rate
      home/away_sp_1st_k_pct, _bb_pct, _xwoba
      sp_1st_k_pct_diff, sp_1st_xwoba_diff

Output: feature_matrix_enriched.parquet
"""

import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR   = Path(".")
DATA_DIR   = BASE_DIR / "data" / "statcast"
INPUT_FILE  = BASE_DIR / "feature_matrix_with_2026.parquet"
OUTPUT_FILE = BASE_DIR / "feature_matrix_enriched.parquet"
YEARS = [2023, 2024, 2025, 2026]

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _null_rate(df, cols):
    """Print null rate for each column in cols."""
    for c in cols:
        if c in df.columns:
            rate = df[c].isna().mean()
            print(f"    {c:<45s} null={rate:.1%}  ({df[c].isna().sum()} / {len(df)})")
        else:
            print(f"    {c:<45s} MISSING from dataframe")


def statcast_name_to_upper(n):
    """Convert 'Last, First' (statcast format) to 'FIRST LAST' uppercase."""
    if pd.isna(n) or not isinstance(n, str):
        return ""
    n = n.strip()
    if "," in n:
        parts = n.split(",", 1)
        return (parts[1].strip() + " " + parts[0].strip()).upper()
    return n.upper()


# ---------------------------------------------------------------------------
# LOAD FEATURE MATRIX
# ---------------------------------------------------------------------------
print("=" * 70)
print("Loading feature matrix ...")
fm = pd.read_parquet(INPUT_FILE)
fm["game_date"] = pd.to_datetime(fm["game_date"])
fm["year"] = fm["game_date"].dt.year
original_shape = fm.shape
print(f"  Input shape: {original_shape}")


# ============================================================
# GROUP 1 — WEATHER
# ============================================================
print("\n" + "=" * 70)
print("Group 1: Weather ...")

weather_parts = []
for yr in YEARS:
    p = DATA_DIR / f"weather_{yr}.parquet"
    if p.exists():
        w = pd.read_parquet(p, columns=["game_date", "home_team", "temp_f", "wind_mph", "wind_bearing"])
        w["game_date"] = pd.to_datetime(w["game_date"])
        weather_parts.append(w)
    else:
        print(f"  WARNING: {p} not found, skipping year {yr}")

weather = pd.concat(weather_parts, ignore_index=True).drop_duplicates(["game_date", "home_team"])
print(f"  Weather rows: {len(weather)}")

fm = fm.merge(weather, on=["game_date", "home_team"], how="left")
fm["temp_f"]       = fm["temp_f"].fillna(72.0)
fm["wind_mph"]     = fm["wind_mph"].fillna(7.0)
fm["wind_bearing"] = fm["wind_bearing"].fillna(180.0)

G1_COLS = ["temp_f", "wind_mph", "wind_bearing"]
print("  Null rates after fill:")
_null_rate(fm, G1_COLS)


# ============================================================
# GROUP 2 — BULLPEN AVAILABILITY
# ============================================================
print("\n" + "=" * 70)
print("Group 2: Bullpen Availability ...")

bp_parts = []
for yr in YEARS:
    p = DATA_DIR / f"bullpen_avail_{yr}.parquet"
    if p.exists():
        b = pd.read_parquet(p, columns=["team", "game_date", "bp_pitches_rest1d", "bp_depleted_flag"])
        b["game_date"] = pd.to_datetime(b["game_date"])
        bp_parts.append(b)
    else:
        print(f"  WARNING: {p} not found, skipping year {yr}")

bp = pd.concat(bp_parts, ignore_index=True).drop_duplicates(["team", "game_date"])
print(f"  Bullpen avail rows: {len(bp)}")

# Home join
bp_home = bp.rename(columns={
    "bp_depleted_flag":   "home_bp_depleted_flag",
    "bp_pitches_rest1d":  "home_bp_pitches_rest1d",
    "team": "home_team",
})
fm = fm.merge(bp_home, on=["game_date", "home_team"], how="left")

# Away join
bp_away = bp.rename(columns={
    "bp_depleted_flag":   "away_bp_depleted_flag",
    "bp_pitches_rest1d":  "away_bp_pitches_rest1d",
    "team": "away_team",
})
fm = fm.merge(bp_away, on=["game_date", "away_team"], how="left")

for col in ["home_bp_depleted_flag", "away_bp_depleted_flag"]:
    fm[col] = fm[col].fillna(0).astype(int)
for col in ["home_bp_pitches_rest1d", "away_bp_pitches_rest1d"]:
    fm[col] = fm[col].fillna(0)

G2_COLS = ["home_bp_depleted_flag", "home_bp_pitches_rest1d",
           "away_bp_depleted_flag", "away_bp_pitches_rest1d"]
print("  Null rates after fill:")
_null_rate(fm, G2_COLS)


# ============================================================
# GROUP 3 — SP DAYS OF REST
# ============================================================
print("\n" + "=" * 70)
print("Group 3: SP Days of Rest ...")

home_app = fm[["game_date", "home_starter_name"]].rename(
    columns={"home_starter_name": "pitcher"})
away_app = fm[["game_date", "away_starter_name"]].rename(
    columns={"away_starter_name": "pitcher"})

all_app = (pd.concat([home_app, away_app])
           .dropna(subset=["pitcher"])
           .drop_duplicates()
           .sort_values(["pitcher", "game_date"]))

all_app["prev_start"] = all_app.groupby("pitcher")["game_date"].shift(1)
all_app["days_rest"]  = (
    (all_app["game_date"] - all_app["prev_start"]).dt.days
    .clip(3, 15)
)

# Home SP rest
home_rest = (all_app.rename(columns={"pitcher": "home_starter_name",
                                      "days_rest": "home_sp_days_rest"})
             [["game_date", "home_starter_name", "home_sp_days_rest"]])
fm = fm.merge(home_rest, on=["game_date", "home_starter_name"], how="left")

# Away SP rest
away_rest = (all_app.rename(columns={"pitcher": "away_starter_name",
                                      "days_rest": "away_sp_days_rest"})
             [["game_date", "away_starter_name", "away_sp_days_rest"]])
fm = fm.merge(away_rest, on=["game_date", "away_starter_name"], how="left")

fm["home_sp_days_rest"] = fm["home_sp_days_rest"].fillna(5)
fm["away_sp_days_rest"] = fm["away_sp_days_rest"].fillna(5)

G3_COLS = ["home_sp_days_rest", "away_sp_days_rest"]
print("  Null rates after fill:")
_null_rate(fm, G3_COLS)


# ============================================================
# GROUP 4 — SP AVG IP PER START
# ============================================================
print("\n" + "=" * 70)
print("Group 4: SP Average IP per Start ...")
print("  Building pitcher IP lookup from statcast (outs-per-game method) ...")

# Build pitcher IP lookup from statcast per game, then aggregate per (pitcher, year)
# We load only necessary columns; filter to inning==1 appearances to identify SPs
ip_parts = []
for yr in YEARS:
    p = DATA_DIR / f"statcast_{yr}.parquet"
    if not p.exists():
        print(f"  WARNING: {p} not found, skipping year {yr}")
        continue
    sc = pd.read_parquet(p, columns=["pitcher", "player_name", "game_pk",
                                      "game_date", "game_year", "inning",
                                      "outs_when_up"])
    sc["game_date"] = pd.to_datetime(sc["game_date"])

    # SP identification: pitched in inning 1
    sp_games = (sc[sc["inning"] == 1][["pitcher", "game_pk"]]
                .drop_duplicates()
                .assign(is_sp=True))

    # Last pitch per pitcher per game → approx IP
    last_pitch = (sc.sort_values(["pitcher", "game_pk", "inning", "outs_when_up"])
                   .groupby(["pitcher", "game_pk"])
                   .last()
                   .reset_index())
    last_pitch["ip_approx"] = ((last_pitch["inning"] - 1) * 3
                                + last_pitch["outs_when_up"]) / 3.0

    # Keep only SP starts
    sp_ip = last_pitch.merge(sp_games, on=["pitcher", "game_pk"], how="inner")

    # Aggregate per pitcher per year
    agg = (sp_ip.groupby(["pitcher", "player_name", "game_year"])
               .agg(avg_ip_per_start=("ip_approx", "mean"),
                    n_starts=("ip_approx", "count"))
               .reset_index())
    agg.columns = ["pitcher", "player_name", "season", "avg_ip_per_start", "n_starts"]
    ip_parts.append(agg)

all_ip = pd.concat(ip_parts, ignore_index=True)

# Filter to real SPs (>=3 starts), clip IP
all_ip = all_ip[all_ip["n_starts"] >= 3].copy()
all_ip["avg_ip_per_start"] = all_ip["avg_ip_per_start"].clip(2.0, 9.0)

# Normalize name: statcast 'Last, First' -> 'FIRST LAST' uppercase
all_ip["name_upper"] = all_ip["player_name"].apply(statcast_name_to_upper)

# Build lookup: name_upper + season -> avg_ip_per_start
ip_lookup = (all_ip.drop_duplicates(["name_upper", "season"])
             .set_index(["name_upper", "season"])["avg_ip_per_start"])

print(f"  IP lookup entries: {len(ip_lookup)}")

# Normalize FM starter names
fm["home_name_upper"] = fm["home_starter_name"].str.upper().str.strip()
fm["away_name_upper"] = fm["away_starter_name"].str.upper().str.strip()

def lookup_avg_ip(name_upper, year, ip_lookup):
    """Look up avg_ip for a pitcher: try current year, fall back to prior year."""
    if pd.isna(name_upper) or name_upper == "":
        return np.nan
    key_cur  = (name_upper, year)
    key_prev = (name_upper, year - 1)
    if key_cur in ip_lookup.index:
        return ip_lookup[key_cur]
    if key_prev in ip_lookup.index:
        return ip_lookup[key_prev]
    return np.nan

fm["home_sp_avg_ip"] = [
    lookup_avg_ip(n, y, ip_lookup)
    for n, y in zip(fm["home_name_upper"], fm["year"])
]
fm["away_sp_avg_ip"] = [
    lookup_avg_ip(n, y, ip_lookup)
    for n, y in zip(fm["away_name_upper"], fm["year"])
]

# Check match rate before filling nulls
home_matched = fm["home_sp_avg_ip"].notna().mean()
away_matched = fm["away_sp_avg_ip"].notna().mean()
print(f"  Home SP name match rate: {home_matched:.1%}")
print(f"  Away SP name match rate: {away_matched:.1%}")
if home_matched < 0.80 or away_matched < 0.80:
    print("  WARNING: < 80% match rate! Investigate name normalization.")
    unmatched = fm[fm["home_sp_avg_ip"].isna()]["home_starter_name"].dropna().unique()
    print(f"  Sample unmatched home starters: {list(unmatched[:10])}")

fm["home_sp_avg_ip"] = fm["home_sp_avg_ip"].fillna(5.1)
fm["away_sp_avg_ip"] = fm["away_sp_avg_ip"].fillna(5.1)

# Drop temp name columns
fm.drop(columns=["home_name_upper", "away_name_upper"], inplace=True, errors="ignore")

G4_COLS = ["home_sp_avg_ip", "away_sp_avg_ip"]
print("  Null rates after fill:")
_null_rate(fm, G4_COLS)


# ============================================================
# GROUP 5 — FIRST INNING SPLITS FROM STATCAST
# ============================================================
print("\n" + "=" * 70)
print("Group 5: First Inning Splits ...")
print("  Loading statcast inning-1 data (year by year, memory efficient) ...")

inn1_parts = []
for yr in YEARS:
    p = DATA_DIR / f"statcast_{yr}.parquet"
    if not p.exists():
        print(f"  WARNING: {p} not found, skipping year {yr}")
        continue
    sc = pd.read_parquet(p, columns=[
        "game_pk", "game_date", "pitcher", "inning", "inning_topbot",
        "events", "estimated_woba_using_speedangle",
        "home_team", "away_team", "post_home_score", "post_away_score",
    ])
    sc = sc[sc["inning"] == 1].copy()
    sc["game_date"] = pd.to_datetime(sc["game_date"])
    inn1_parts.append(sc)
    print(f"    Year {yr}: {len(sc)} inning-1 rows")

inn1 = pd.concat(inn1_parts, ignore_index=True)
print(f"  Total inning-1 rows: {len(inn1)}")

# ---- 5a: Per-team 1st inning run rate ----------------------------------------
print("\n  5a: Per-team 1st inning run rate ...")

# Away team: score in top of 1st?
away_1st = (inn1[inn1["inning_topbot"] == "Top"]
            .groupby(["game_pk", "away_team", "game_date"])["post_away_score"]
            .max()
            .reset_index())
away_1st["scored_1st"] = (away_1st["post_away_score"] > 0).astype(int)

# Home team: score in bot of 1st?
home_1st = (inn1[inn1["inning_topbot"] == "Bot"]
            .groupby(["game_pk", "home_team", "game_date"])["post_home_score"]
            .max()
            .reset_index())
home_1st["scored_1st"] = (home_1st["post_home_score"] > 0).astype(int)

LEAGUE_AVG_1ST_RATE = 0.28

def rolling_1st_rate(df, team_col):
    parts = []
    for team, grp in df.groupby(team_col):
        g = grp.sort_values("game_date").reset_index(drop=True)
        rate = (g["scored_1st"]
                .rolling(30, min_periods=10)
                .mean()
                .shift(1)
                .fillna(LEAGUE_AVG_1ST_RATE))
        parts.append(pd.DataFrame({
            "game_pk":  g["game_pk"],
            team_col:   team,
            "rate":     rate,
        }))
    return pd.concat(parts, ignore_index=True)

home_rate_df = rolling_1st_rate(home_1st, "home_team")
home_rate_df = home_rate_df.rename(columns={"rate": "home_1st_inn_run_rate"})
fm = fm.merge(home_rate_df, on=["game_pk", "home_team"], how="left")
fm["home_1st_inn_run_rate"] = fm["home_1st_inn_run_rate"].fillna(LEAGUE_AVG_1ST_RATE)

away_rate_df = rolling_1st_rate(away_1st, "away_team")
away_rate_df = away_rate_df.rename(columns={"rate": "away_1st_inn_run_rate"})
fm = fm.merge(away_rate_df, on=["game_pk", "away_team"], how="left")
fm["away_1st_inn_run_rate"] = fm["away_1st_inn_run_rate"].fillna(LEAGUE_AVG_1ST_RATE)

# ---- 5b: Per-pitcher season-to-date 1st inning stats -------------------------
print("  5b: Per-pitcher season-to-date 1st inning stats ...")

def build_sp_1st_stats(inn1, topbot, outcome_cols, prefix):
    """
    topbot: 'Top' for home SP (away team batting), 'Bot' for away SP.
    prefix: 'home_sp' or 'away_sp'.
    Returns df with game_pk + stats columns (one row per game_pk).
    """
    pa = inn1[(inn1["inning_topbot"] == topbot) & (inn1["events"].notna())].copy()

    game_sp = pa.groupby(["game_pk", "pitcher", "game_date"]).agg(
        n_pa      = ("events", "count"),
        k         = ("events", lambda x: (x == "strikeout").sum()),
        bb        = ("events", lambda x: x.isin(["walk", "intent_walk"]).sum()),
        xwoba_sum = ("estimated_woba_using_speedangle", "sum"),
        xwoba_n   = ("estimated_woba_using_speedangle", "count"),
    ).reset_index()
    game_sp["year"] = game_sp["game_date"].dt.year

    # If multiple pitchers appeared in inning 1, keep only the one with most PAs
    # (that's the true starting pitcher; the other was a mid-inning replacement)
    game_sp = (game_sp.sort_values("n_pa", ascending=False)
               .drop_duplicates("game_pk", keep="first")
               .copy())

    parts = []
    for (pid, yr), grp in game_sp.groupby(["pitcher", "year"]):
        g = grp.sort_values("game_date").reset_index(drop=True)
        cum_pa   = g["n_pa"].cumsum().shift(1)
        cum_k    = g["k"].cumsum().shift(1)
        cum_bb   = g["bb"].cumsum().shift(1)
        cum_xw_s = g["xwoba_sum"].cumsum().shift(1)
        cum_xw_n = g["xwoba_n"].cumsum().shift(1)
        sub = pd.DataFrame({
            "game_pk":                  g["game_pk"],
            f"{prefix}_1st_k_pct":     cum_k  / cum_pa,
            f"{prefix}_1st_bb_pct":    cum_bb / cum_pa,
            f"{prefix}_1st_xwoba":     cum_xw_s / cum_xw_n.replace(0, np.nan),
        })
        parts.append(sub)

    result = pd.concat(parts, ignore_index=True)
    # Fill first-game-of-season NaNs with league averages
    result[f"{prefix}_1st_k_pct"]  = result[f"{prefix}_1st_k_pct"].fillna(0.231)
    result[f"{prefix}_1st_bb_pct"] = result[f"{prefix}_1st_bb_pct"].fillna(0.085)
    result[f"{prefix}_1st_xwoba"]  = result[f"{prefix}_1st_xwoba"].fillna(0.330)
    return result


# Home SP stats (Top of 1st: home SP pitching to away batters)
home_sp_1st = build_sp_1st_stats(inn1, "Top", None, "home_sp")
fm = fm.merge(home_sp_1st, on="game_pk", how="left")
for col in ["home_sp_1st_k_pct", "home_sp_1st_bb_pct", "home_sp_1st_xwoba"]:
    fm[col] = fm[col].fillna(fm[col].median() if fm[col].notna().any() else 0.231)

# Away SP stats (Bot of 1st: away SP pitching to home batters)
away_sp_1st = build_sp_1st_stats(inn1, "Bot", None, "away_sp")
fm = fm.merge(away_sp_1st, on="game_pk", how="left")
for col in ["away_sp_1st_k_pct", "away_sp_1st_bb_pct", "away_sp_1st_xwoba"]:
    fm[col] = fm[col].fillna(fm[col].median() if fm[col].notna().any() else 0.231)

# Diff features
fm["sp_1st_k_pct_diff"]  = fm["home_sp_1st_k_pct"]  - fm["away_sp_1st_k_pct"]
fm["sp_1st_xwoba_diff"]  = fm["home_sp_1st_xwoba"]   - fm["away_sp_1st_xwoba"]

G5_COLS = [
    "home_1st_inn_run_rate", "away_1st_inn_run_rate",
    "home_sp_1st_k_pct", "home_sp_1st_bb_pct", "home_sp_1st_xwoba",
    "away_sp_1st_k_pct", "away_sp_1st_bb_pct", "away_sp_1st_xwoba",
    "sp_1st_k_pct_diff", "sp_1st_xwoba_diff",
]
print("  Null rates after fill:")
_null_rate(fm, G5_COLS)


# ============================================================
# SAVE
# ============================================================
print("\n" + "=" * 70)
print("Saving enriched feature matrix ...")

fm.to_parquet(OUTPUT_FILE, index=False)

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
ALL_NEW_COLS = G1_COLS + G2_COLS + G3_COLS + G4_COLS + G5_COLS

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Original shape : {original_shape}")
print(f"  Enriched shape : {fm.shape}")
print(f"  New features   : {len(ALL_NEW_COLS)}")
print(f"  Saved to       : {OUTPUT_FILE}")

print("\n  All new feature null rates (final, post-fill):")
_null_rate(fm, ALL_NEW_COLS)

print("\n  Sample rows (new features only):")
sample_cols = ["game_date", "home_team", "away_team"] + ALL_NEW_COLS[:10]
print(fm[sample_cols].dropna().head(5).to_string(index=False))

print("\nDone.")
