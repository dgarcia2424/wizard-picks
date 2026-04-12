"""
build_pitcher_profile.py
========================
Builds per-pitcher early-season profiles used by the XGBoost model and
Monte Carlo run-line simulator.

For each pitcher currently active in 2026, outputs one row containing:

VELOCITY SIGNALS (stabilize in 1 start)
  ff_velo_2025          season-avg 4-seam velocity (2025 pitch_arsenal)
  ff_velo_career_april  avg April velocity across 2023-2024
  ff_velo_2026_april    current 2026 season velocity (statcast_2026)
  velocity_delta_vs_2025        2026 april - 2025 full season
  velocity_delta_vs_career_april  2026 april - career april avg
  velocity_flag         VOLATILE / GAINER / NORMAL

CAREER APRIL DISTRIBUTION (mean + std -> Monte Carlo sigma)
  april_k_pct_mean / std
  april_xwoba_mean / std
  april_gb_pct_mean / std
  april_exit_velo_mean / std
  april_starts           number of April starts in career (2023-2024)

NEW PITCH SIGNAL (decaying advantage)
  new_pitch_flag         True if added pitch with >8% usage between prior years
  new_pitch_type         e.g. 'ST' (sweeper)
  new_pitch_usage_delta  fractional usage gain

STUFF PROXY (stabilizes ~100 pitches)
  xrv_per_pitch_2025     delta_pitcher_run_exp per pitch (2025 season)
  xrv_per_pitch_2026     same for 2026 so far
  fb_spin_2025           4-seam spin rate (pitcher_percentiles)
  stuff_proxy            composite: fb_velo + spin + whiff

BUY-LOW SIGNAL
  lob_pct_2025           left-on-base% (fangraphs_pitchers.csv)
  xfip_2025              xFIP (fangraphs_pitchers.csv)
  era_minus_xfip         ERA - xFIP (positive = due for regression)

MONTE CARLO VARIANCE
  monte_carlo_sigma      career_april_xwoba_std * volatility_multiplier
                         (VOLATILE -> x1.5, GAINER -> x0.85, NORMAL -> x1.0)

Outputs
-------
  statcast_data/pitcher_profiles_2026.parquet
  pitcher_profiles_2026.csv   (human-readable copy)

Usage
-----
  python build_pitcher_profile.py
  python build_pitcher_profile.py --velocity-threshold 1.5
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("./data/statcast")
VELOCITY_THRESHOLD   = 1.5   # mph delta that triggers VOLATILE / GAINER flag
NEW_PITCH_THRESHOLD  = 0.08  # usage fraction gain to count as "new primary pitch"
MIN_APRIL_STARTS     = 2     # minimum historical April starts for reliable sigma
MIN_PITCHES_GAME     = 40    # min pitches in a game to count as a real start

# Pitch type groups
FASTBALL_TYPES   = {"FF", "SI", "FC"}
OFFSPEED_TYPES   = {"CH", "FS", "SC"}
BREAKING_TYPES   = {"SL", "CU", "KC", "SV", "ST", "CS"}

# Stadium elevations (feet) for air-density calc in feature matrix
STADIUM_ELEVATION = {
    "COL": 5200, "AZ": 1082, "TEX": 551, "HOU": 43, "ATL": 1050,
    "STL": 465,  "KC": 740,  "MIN": 840, "CIN": 550, "MIL": 635,
    "CHC": 595,  "CLE": 653, "DET": 600, "PIT": 730, "PHI": 20,
    "NYY": 55,   "NYM": 33,  "BOS": 19,  "BAL": 53,  "WSH": 25,
    "TOR": 249,  "TB": 28,   "MIA": 8,   "SF": 52,   "LAD": 512,
    "LAA": 160,  "SD": 17,   "SEA": 56,  "ATH": 25,  "CWS": 20,
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _to_num(series):
    return pd.to_numeric(series, errors="coerce")


def _load_statcast(year: int, cols: list[str]) -> pd.DataFrame:
    path = OUTPUT_DIR / f"statcast_{year}.parquet"
    df = pd.read_parquet(path, engine="pyarrow", columns=cols)
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _pitcher_name_map(df: pd.DataFrame) -> dict:
    """Return {pitcher_id: player_name} from a statcast dataframe."""
    return (
        df[["pitcher", "player_name"]]
        .dropna(subset=["player_name"])
        .drop_duplicates("pitcher")
        .set_index("pitcher")["player_name"]
        .to_dict()
    )


# ---------------------------------------------------------------------------
# SECTION 1 — VELOCITY PROFILES
# ---------------------------------------------------------------------------

def build_velocity_profiles(verbose: bool = True) -> pd.DataFrame:
    """
    Returns one row per pitcher_id with columns:
      pitcher, p_throws,
      ff_velo_2025, ff_velo_career_april, ff_velo_2026_april,
      velocity_delta_vs_2025, velocity_delta_vs_career_april,
      velocity_flag
    """
    if verbose:
        print("  [1/6] Velocity profiles ...")

    cols = ["pitcher", "player_name", "pitch_type", "release_speed",
            "game_date", "p_throws"]

    # --- 2025 full-season baseline from pitch_arsenal (season avg) ----------
    ars25 = pd.read_parquet(OUTPUT_DIR / "pitch_arsenal_2025.parquet",
                            engine="pyarrow")
    ars25["pitcher"] = _to_num(ars25["pitcher"])
    ars25 = ars25[["pitcher", "ff_avg_speed"]].rename(
        columns={"ff_avg_speed": "ff_velo_2025"})

    # --- career April baseline (2023 + 2024 April FF velos) -----------------
    april_frames = []
    for yr in [2023, 2024]:
        df = _load_statcast(yr, cols)
        df = df[(df["game_date"].dt.month == 4) & (df["pitch_type"] == "FF")]
        df["release_speed"] = _to_num(df["release_speed"])
        g = df.groupby("pitcher")["release_speed"].agg(
            mean="mean", n="count").reset_index()
        g["year"] = yr
        april_frames.append(g)

    career_apr = (
        pd.concat(april_frames)
        .groupby("pitcher")
        .apply(
            lambda x: pd.Series({
                "ff_velo_career_april": np.average(
                    x["mean"].dropna(),
                    weights=x.loc[x["mean"].notna(), "n"]),
                "career_april_ff_n": x["n"].sum(),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    # --- 2026 April current velo (statcast_2026) ----------------------------
    df26 = _load_statcast(2026, cols)
    df26 = df26[df26["pitch_type"] == "FF"]
    df26["release_speed"] = _to_num(df26["release_speed"])
    df26["p_throws"] = df26["p_throws"].astype(str)

    # pitcher handedness lookup
    hand_map = (df26[["pitcher", "p_throws"]].dropna()
                .drop_duplicates("pitcher")
                .set_index("pitcher")["p_throws"].to_dict())

    apr26 = df26.groupby("pitcher")["release_speed"].agg(
        ff_velo_2026_april="mean", n_ff_2026="count").reset_index()

    # require at least 10 FF pitches in 2026 to use velocity
    apr26 = apr26[apr26["n_ff_2026"] >= 10]

    # --- Merge all velocity data --------------------------------------------
    vel = (apr26
           .merge(ars25, on="pitcher", how="left")
           .merge(career_apr, on="pitcher", how="left"))

    vel["p_throws"] = vel["pitcher"].map(hand_map)
    vel["velocity_delta_vs_2025"] = (
        vel["ff_velo_2026_april"] - vel["ff_velo_2025"])
    vel["velocity_delta_vs_career_april"] = (
        vel["ff_velo_2026_april"] - vel["ff_velo_career_april"])

    def flag(delta):
        if pd.isna(delta):
            return "UNKNOWN"
        if delta < -VELOCITY_THRESHOLD:
            return "VOLATILE"
        if delta > VELOCITY_THRESHOLD:
            return "GAINER"
        return "NORMAL"

    vel["velocity_flag"] = vel["velocity_delta_vs_2025"].apply(flag)

    if verbose:
        n_vol = (vel["velocity_flag"] == "VOLATILE").sum()
        n_gain = (vel["velocity_flag"] == "GAINER").sum()
        print(f"      {len(vel)} pitchers | VOLATILE: {n_vol} | "
              f"GAINER: {n_gain} | NORMAL: {len(vel)-n_vol-n_gain}")

    return vel[[
        "pitcher", "p_throws",
        "ff_velo_2025", "ff_velo_career_april", "ff_velo_2026_april",
        "n_ff_2026", "career_april_ff_n",
        "velocity_delta_vs_2025", "velocity_delta_vs_career_april",
        "velocity_flag",
    ]]


# ---------------------------------------------------------------------------
# SECTION 2 — CAREER MONTHLY DISTRIBUTIONS (mean + sigma per stat, all months)
# ---------------------------------------------------------------------------

MONTH_LABELS = {3: "mar", 4: "apr", 5: "may", 6: "jun",
                7: "jul", 8: "aug", 9: "sep"}
STAT_COLS = ["k_pct", "bb_pct", "xwoba", "gb_pct", "exit_velo", "xrv_per_pitch"]


def _game_level_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pitch-level statcast rows to one row per pitcher x game."""
    df = df.copy()
    df["xwoba"]     = _to_num(df["estimated_woba_using_speedangle"])
    df["exit_velo"] = _to_num(df["launch_speed"])
    df["xrv"]       = _to_num(df["delta_pitcher_run_exp"])
    df["k"]         = (df["events"] == "strikeout").astype(float)
    df["bb"]        = (df["events"] == "walk").astype(float)
    df["is_pa"]     = df["events"].notna().astype(float)
    df["is_gb"]     = (df["bb_type"] == "ground_ball").astype(float)
    df["is_bip"]    = df["bb_type"].notna().astype(float)

    g = (df.groupby(["pitcher", "game_pk", "month"]).agg(
        n_pitches   = ("xrv",      "count"),
        pa          = ("is_pa",    "sum"),
        k           = ("k",        "sum"),
        bb          = ("bb",       "sum"),
        gb          = ("is_gb",    "sum"),
        bip         = ("is_bip",   "sum"),
        xwoba       = ("xwoba",    "mean"),
        exit_velo   = ("exit_velo","mean"),
        xrv_total   = ("xrv",      "sum"),
    ).reset_index())

    g = g[g["n_pitches"] >= MIN_PITCHES_GAME]
    g["k_pct"]       = g["k"]         / g["pa"].clip(lower=1)
    g["bb_pct"]      = g["bb"]        / g["pa"].clip(lower=1)
    g["gb_pct"]      = g["gb"]        / g["bip"].clip(lower=1)
    g["xrv_per_pitch"] = g["xrv_total"] / g["n_pitches"].clip(lower=1)
    return g


def build_monthly_distributions(verbose: bool = True) -> pd.DataFrame:
    """
    For each pitcher, compute per-game stats for every month (Mar-Sep)
    across 2023, 2024, and 2025.  Returns one wide row per pitcher with:

      {mon}_k_pct_mean / std
      {mon}_bb_pct_mean / std
      {mon}_xwoba_mean / std
      {mon}_gb_pct_mean / std
      {mon}_exit_velo_mean / std
      {mon}_xrv_mean / std
      {mon}_starts          number of historical starts in that month

    League-average sigma is used when a pitcher has < MIN_APRIL_STARTS
    in a given month.
    """
    if verbose:
        print("  [2/6] Career monthly distributions (Mar-Sep) ...")

    cols = ["pitcher", "game_pk", "game_date", "pitch_type", "events",
            "bb_type", "estimated_woba_using_speedangle",
            "launch_speed", "delta_pitcher_run_exp"]

    # Load all three historical years together
    frames = []
    for yr in [2023, 2024, 2025]:
        df = _load_statcast(yr, cols)
        df["month"] = df["game_date"].dt.month
        df = df[df["month"].between(3, 9)]
        frames.append(df)

    all_raw = pd.concat(frames, ignore_index=True)
    all_games = _game_level_stats(all_raw)

    # League-level sigma per month (fallback for thin samples)
    league_sigma: dict[int, dict] = {}
    for m in MONTH_LABELS:
        mg = all_games[all_games["month"] == m]
        league_sigma[m] = {s: mg[s].std() for s in STAT_COLS}

    # Per-pitcher, per-month aggregation
    def month_stats(grp):
        n = len(grp)
        row: dict = {"starts": n}
        for s in STAT_COLS:
            row[f"{s}_mean"] = grp[s].mean()
            row[f"{s}_std"]  = grp[s].std() if n > 1 else np.nan
        return pd.Series(row)

    monthly = (all_games.groupby(["pitcher", "month"])
               .apply(month_stats, include_groups=False)
               .reset_index())

    # Pivot to wide: one row per pitcher
    wide_frames = []
    for m, label in MONTH_LABELS.items():
        mdf = monthly[monthly["month"] == m].copy()
        rename = {"starts": f"{label}_starts"}
        for s in STAT_COLS:
            rename[f"{s}_mean"] = f"{label}_{s}_mean"
            rename[f"{s}_std"]  = f"{label}_{s}_std"
        mdf = mdf.rename(columns=rename).drop(columns="month")
        wide_frames.append(mdf.set_index("pitcher"))

    dist = pd.concat(wide_frames, axis=1).reset_index()

    # Fill thin-sample sigmas with league average for that month
    for m, label in MONTH_LABELS.items():
        for s in STAT_COLS:
            std_col   = f"{label}_{s}_std"
            start_col = f"{label}_starts"
            if std_col not in dist.columns:
                continue
            thin = dist[start_col].fillna(0) < MIN_APRIL_STARTS
            dist.loc[thin, std_col] = league_sigma[m][s]

    # Store April league sigma for Monte Carlo fallback
    dist["league_sigma_xwoba"] = league_sigma[4]["xwoba"]

    if verbose:
        for m, label in MONTH_LABELS.items():
            col = f"{label}_starts"
            n_pitchers = dist[col].notna().sum() if col in dist.columns else 0
            n_qual = (dist[col].fillna(0) >= MIN_APRIL_STARTS).sum() \
                if col in dist.columns else 0
            sigma = league_sigma[m]["xwoba"]
            print(f"      {label.upper()}: {n_pitchers} pitchers | "
                  f"{n_qual} with {MIN_APRIL_STARTS}+ starts | "
                  f"league xwOBA sigma={sigma:.4f}")

    return dist


# ---------------------------------------------------------------------------
# SECTION 2b — TRAILING 2026 STATS (actual season-to-date performance)
# ---------------------------------------------------------------------------

def build_trailing_2026(verbose: bool = True) -> pd.DataFrame:
    """
    Compute each pitcher's actual 2026 stats through the most recent game.
    Returns per-pitcher trailing season stats + a trust weight that rises
    from 0 (0 starts) to 1.0 (~8 starts) — used to blend career monthly
    profile with actual 2026 performance.

    trust_weight = min(n_starts_2026 / 8, 1.0)

    At 0-2 starts: trust ~0-25%  -> lean on career profile
    At 4 starts  : trust ~50%    -> K-BB% becoming meaningful
    At 8+ starts : trust ~100%   -> 2026 stats drive the estimate
    """
    if verbose:
        print("  [2b] Trailing 2026 season stats ...")

    cols = ["pitcher", "player_name", "game_pk", "game_date", "pitch_type",
            "events", "bb_type", "estimated_woba_using_speedangle",
            "launch_speed", "delta_pitcher_run_exp", "p_throws"]

    df = _load_statcast(2026, cols)
    df["month"] = df["game_date"].dt.month

    games = _game_level_stats(df)

    # Aggregate across all 2026 starts
    def trailing_stats(grp):
        n = len(grp)
        return pd.Series({
            "trailing_n_starts":     n,
            "trailing_k_pct":        grp["k_pct"].mean(),
            "trailing_bb_pct":       grp["bb_pct"].mean(),
            "trailing_xwoba":        grp["xwoba"].mean(),
            "trailing_gb_pct":       grp["gb_pct"].mean(),
            "trailing_exit_velo":    grp["exit_velo"].mean(),
            "trailing_xrv_per_pitch":grp["xrv_per_pitch"].mean(),
            # Per-start std — early season volatility in actual 2026 data
            "trailing_xwoba_std":    grp["xwoba"].std() if n > 1 else np.nan,
            "trailing_k_pct_std":    grp["k_pct"].std() if n > 1 else np.nan,
        })

    trail = (games.groupby("pitcher")
             .apply(trailing_stats, include_groups=False)
             .reset_index())

    # Trust weight: ramps from 0 to 1 over 8 starts
    # K-BB% stabilizes ~3-4 starts, xwOBA ~6-8 starts
    trail["trailing_trust_kbb"]   = (trail["trailing_n_starts"] / 4).clip(upper=1.0)
    trail["trailing_trust_xwoba"] = (trail["trailing_n_starts"] / 8).clip(upper=1.0)

    if verbose:
        print(f"      {len(trail)} pitchers with 2026 starts | "
              f"avg starts: {trail['trailing_n_starts'].mean():.1f}")
        print(f"      trust_kbb  (0-1): {trail['trailing_trust_kbb'].mean():.2f} avg")
        print(f"      trust_xwoba(0-1): {trail['trailing_trust_xwoba'].mean():.2f} avg")

    return trail


# ---------------------------------------------------------------------------
# SECTION 3 — NEW PITCH DETECTION
# ---------------------------------------------------------------------------

def build_new_pitch_flags(verbose: bool = True) -> pd.DataFrame:
    """
    Compare pitch usage % between consecutive years.
    Flag pitchers who added or significantly increased a pitch type.

    primary comparison: 2024 -> 2025 (for 2026 season predictions)
    secondary:          2025 -> 2026 (anything new this spring)
    """
    if verbose:
        print("  [3/6] New pitch detection ...")

    def pitch_mix(year: int) -> pd.DataFrame:
        df = _load_statcast(year, ["pitcher", "pitch_type"])
        df = df[df["pitch_type"].notna()]
        counts = (df.groupby(["pitcher", "pitch_type"])
                  .size().reset_index(name="n"))
        totals = counts.groupby("pitcher")["n"].sum()
        counts["pct"] = counts["n"] / counts["pitcher"].map(totals)
        return counts[["pitcher", "pitch_type", "pct"]]

    mix24 = pitch_mix(2024)
    mix25 = pitch_mix(2025)
    mix26 = pitch_mix(2026)

    def find_new_pitches(old_mix: pd.DataFrame,
                         new_mix: pd.DataFrame,
                         label: str) -> pd.DataFrame:
        old_wide = old_mix.pivot(
            index="pitcher", columns="pitch_type", values="pct").fillna(0)
        new_wide = new_mix.pivot(
            index="pitcher", columns="pitch_type", values="pct").fillna(0)

        common = old_wide.index.intersection(new_wide.index)
        all_pt = old_wide.columns.union(new_wide.columns)
        old_w  = old_wide.reindex(index=common, columns=all_pt, fill_value=0)
        new_w  = new_wide.reindex(index=common, columns=all_pt, fill_value=0)
        delta  = new_w - old_w

        rows = []
        for pid in common:
            d = delta.loc[pid]
            best_delta = d.max()
            best_type  = d.idxmax()
            if best_delta >= NEW_PITCH_THRESHOLD:
                rows.append({
                    "pitcher":            pid,
                    f"new_pitch_flag_{label}":   True,
                    f"new_pitch_type_{label}":   best_type,
                    f"new_pitch_delta_{label}":  round(best_delta, 4),
                })
            else:
                rows.append({
                    "pitcher":            pid,
                    f"new_pitch_flag_{label}":   False,
                    f"new_pitch_type_{label}":   None,
                    f"new_pitch_delta_{label}":  0.0,
                })
        return pd.DataFrame(rows)

    flags_25 = find_new_pitches(mix24, mix25, "2025")  # added in 2025
    flags_26 = find_new_pitches(mix25, mix26, "2026")  # added in 2026

    flags = flags_25.merge(flags_26, on="pitcher", how="outer")

    # Summary flag: new pitch in either transition
    flags["new_pitch_flag"] = (
        flags["new_pitch_flag_2025"].fillna(False) |
        flags["new_pitch_flag_2026"].fillna(False)
    )
    # Use most recent new pitch as primary
    flags["new_pitch_type"] = np.where(
        flags["new_pitch_flag_2026"].fillna(False),
        flags["new_pitch_type_2026"],
        flags["new_pitch_type_2025"],
    )
    flags["new_pitch_usage_delta"] = np.where(
        flags["new_pitch_flag_2026"].fillna(False),
        flags["new_pitch_delta_2026"],
        flags["new_pitch_delta_2025"],
    )

    if verbose:
        n_new = flags["new_pitch_flag"].sum()
        print(f"      {len(flags)} pitchers compared | "
              f"new pitch flag: {n_new} ({n_new/len(flags):.0%})")

    return flags[["pitcher", "new_pitch_flag", "new_pitch_type",
                  "new_pitch_usage_delta",
                  "new_pitch_flag_2025", "new_pitch_type_2025",
                  "new_pitch_flag_2026", "new_pitch_type_2026"]]


# ---------------------------------------------------------------------------
# SECTION 4 — STUFF PROXY (xRV + spin)
# ---------------------------------------------------------------------------

def build_stuff_proxy(verbose: bool = True) -> pd.DataFrame:
    """
    xRV per pitch from delta_pitcher_run_exp (2025 full season + 2026 so far)
    fb_spin and whiff% from pitcher_percentiles.
    Composite Stuff proxy = normalized combination.
    """
    if verbose:
        print("  [4/6] Stuff proxy (xRV + spin + whiff) ...")

    cols = ["pitcher", "pitch_type", "delta_pitcher_run_exp",
            "release_spin_rate", "game_date"]

    def xrv_season(year: int, label: str) -> pd.DataFrame:
        df = _load_statcast(year, cols)
        df["xrv"] = _to_num(df["delta_pitcher_run_exp"])
        df["spin"] = _to_num(df["release_spin_rate"])
        df = df[df["xrv"].notna()]
        g = df.groupby("pitcher").agg(
            xrv_total=("xrv", "sum"),
            n_pitches=("pitch_type", "count"),
            ff_spin=("spin", lambda x: x[
                df.loc[x.index, "pitch_type"] == "FF"].mean()
                if (df.loc[x.index, "pitch_type"] == "FF").any() else np.nan),
        ).reset_index()
        g[f"xrv_per_pitch_{label}"] = g["xrv_total"] / g["n_pitches"].clip(lower=1)
        g[f"n_pitches_{label}"] = g["n_pitches"]
        return g[["pitcher", f"xrv_per_pitch_{label}", f"n_pitches_{label}"]]

    xrv25 = xrv_season(2025, "2025")
    xrv26 = xrv_season(2026, "2026")

    # Spin rate and whiff from pitcher_percentiles_2025
    perc = pd.read_parquet(
        OUTPUT_DIR / "pitcher_percentiles_2025.parquet", engine="pyarrow")
    perc = perc[["player_id", "fb_velocity", "fb_spin",
                 "whiff_percent", "xera"]].rename(
        columns={"player_id": "pitcher"})
    perc["pitcher"] = _to_num(perc["pitcher"])
    perc["fb_spin"] = _to_num(perc["fb_spin"])
    perc["fb_velocity"] = _to_num(perc["fb_velocity"])
    perc["whiff_percent"] = _to_num(perc["whiff_percent"])
    perc["xera"] = _to_num(perc["xera"])

    stuff = (xrv25
             .merge(xrv26, on="pitcher", how="left")
             .merge(perc, on="pitcher", how="left"))

    # Composite stuff proxy: high spin + high velo + high whiff = high stuff
    # Each term normalized to ~0-100 scale
    stuff["stuff_proxy"] = (
        ((stuff["fb_velocity"].clip(88, 100) - 88) / 12 * 40) +  # 0-40 pts
        ((stuff["fb_spin"].clip(1800, 3000) - 1800) / 1200 * 30) +  # 0-30 pts
        ((stuff["whiff_percent"].clip(15, 40) - 15) / 25 * 30)   # 0-30 pts
    ).clip(0, 100)

    if verbose:
        print(f"      {len(stuff)} pitchers | "
              f"stuff proxy range: {stuff['stuff_proxy'].min():.0f} - "
              f"{stuff['stuff_proxy'].max():.0f}")

    return stuff[["pitcher", "xrv_per_pitch_2025", "n_pitches_2025",
                  "xrv_per_pitch_2026", "n_pitches_2026",
                  "fb_velocity", "fb_spin", "whiff_percent",
                  "xera", "stuff_proxy"]]


# ---------------------------------------------------------------------------
# SECTION 5 — AGE + ARM ANGLE (age-adjusted sigma, matchup context)
# ---------------------------------------------------------------------------

AGE_BUCKET_MAP = {
    (0,  25): "young",        # still developing, high variance but not alarming
    (26, 29): "prime",        # peak performance window
    (30, 32): "experienced",  # slight decline typical, watch velocity
    (33, 99): "late_career",  # velocity loss is a meaningful red flag
}


def _age_bucket(age: float) -> str:
    if pd.isna(age):
        return "unknown"
    age = int(age)
    for (lo, hi), label in AGE_BUCKET_MAP.items():
        if lo <= age <= hi:
            return label
    return "unknown"


def build_age_arm_signals(verbose: bool = True) -> pd.DataFrame:
    """
    Pull pitcher age and arm angle from 2026 statcast (and 2025 for year-over-
    year arm angle change, which can flag injury compensation).

    Returns one row per pitcher with:
      age_pit           -- median reported age in 2026 statcast
      age_bucket        -- 'young' / 'prime' / 'experienced' / 'late_career'
      arm_angle_2026    -- median arm angle from 2026 statcast
      arm_angle_2025    -- median arm angle from 2025 (for comparison)
      arm_angle_delta   -- 2026 - 2025 (|>5 deg| may indicate adjustment or injury)
      arm_angle_style   -- 'submarine' (<20 deg), 'low' (20-40), 'standard' (40-65),
                           'over_top' (>65)
    """
    if verbose:
        print("  [5/7] Age + arm angle signals ...")

    cols_age  = ["pitcher", "age_pit", "arm_angle"]
    cols_2025 = ["pitcher", "arm_angle"]

    df26 = _load_statcast(2026, cols_age)
    df26["age_pit"]    = _to_num(df26["age_pit"])
    df26["arm_angle"]  = _to_num(df26["arm_angle"])

    df25 = _load_statcast(2025, cols_2025)
    df25["arm_angle"] = _to_num(df25["arm_angle"])

    # Per-pitcher medians (robust to outlier pitches)
    age26 = (df26.groupby("pitcher")
             .agg(age_pit=("age_pit", "median"),
                  arm_angle_2026=("arm_angle", "median"))
             .reset_index())

    ang25 = (df25.groupby("pitcher")
             .agg(arm_angle_2025=("arm_angle", "median"))
             .reset_index())

    sig = age26.merge(ang25, on="pitcher", how="left")

    sig["age_pit"]       = sig["age_pit"].round(0).astype("Int64")
    sig["age_bucket"]    = sig["age_pit"].apply(_age_bucket)
    sig["arm_angle_delta"] = sig["arm_angle_2026"] - sig["arm_angle_2025"]

    def _arm_style(angle):
        if pd.isna(angle):
            return "unknown"
        if angle < 20:
            return "submarine"
        if angle < 40:
            return "low"
        if angle < 65:
            return "standard"
        return "over_top"

    sig["arm_angle_style"] = sig["arm_angle_2026"].apply(_arm_style)

    if verbose:
        if sig["age_pit"].notna().any():
            print(f"      {len(sig)} pitchers | "
                  f"median age: {sig['age_pit'].median():.0f} | "
                  f"late-career (33+): {(sig['age_bucket']=='late_career').sum()}")
        if sig["arm_angle_2026"].notna().any():
            style_counts = sig["arm_angle_style"].value_counts()
            print(f"      arm angle: "
                  + " | ".join(f"{k}: {v}" for k, v in style_counts.items()))

    return sig[[
        "pitcher", "age_pit", "age_bucket",
        "arm_angle_2026", "arm_angle_2025", "arm_angle_delta", "arm_angle_style",
    ]]


# ---------------------------------------------------------------------------
# SECTION 6 — BUY-LOW SIGNALS (LOB%, xFIP from FanGraphs CSV)
# ---------------------------------------------------------------------------

def build_buyllow_signals(verbose: bool = True) -> pd.DataFrame:
    """
    LOB% and xFIP from fangraphs_pitchers.csv (most recent year available).
    High ERA - xFIP = due for positive regression (buy low).
    """
    if verbose:
        print("  [6/7] Buy-low signals (LOB%, xFIP) ...")

    fg = pd.read_csv("data/raw/fangraphs_pitchers.csv", encoding="utf-8-sig")
    fg.columns = fg.columns.str.strip()

    # Normalize pitcher name
    name_col = next((c for c in fg.columns
                     if c.lower() in ("name", "playername", "player")), None)
    if name_col:
        fg["pitcher_name_fg"] = fg[name_col].str.upper().str.strip()

    # Use most recent year per pitcher
    if "year" in fg.columns:
        fg["year"] = pd.to_numeric(fg["year"], errors="coerce")
        fg = fg.sort_values("year", ascending=False)
        fg = fg.drop_duplicates("pitcher_name_fg")

    # Column mapping
    col_map = {
        "LOB%": "lob_pct", "xFIP": "xfip", "FIP": "fip",
        "ERA":  "era_fg",  "xERA": "xera_fg", "IP": "ip",
        "K/9":  "k9",      "BB/9": "bb9",
    }
    fg = fg.rename(columns={k: v for k, v in col_map.items() if k in fg.columns})

    for col in ["lob_pct", "xfip", "fip", "era_fg", "xera_fg", "ip", "k9", "bb9"]:
        if col in fg.columns:
            if col == "lob_pct":
                # FanGraphs stores as string "72.3%" or decimal
                fg[col] = (fg[col].astype(str)
                           .str.replace("%", "", regex=False)
                           .pipe(lambda s: pd.to_numeric(s, errors="coerce")))
                if fg[col].median() < 1:  # decimal form
                    fg[col] *= 100
            else:
                fg[col] = pd.to_numeric(fg[col], errors="coerce")

    # ERA - xFIP: positive = pitching worse than skill suggests (buy low)
    if "era_fg" in fg.columns and "xfip" in fg.columns:
        fg["era_minus_xfip"] = fg["era_fg"] - fg["xfip"]

    # LOB% sustainability: <68% or >80% is unusual
    if "lob_pct" in fg.columns:
        fg["lob_unsustainable"] = (
            (fg["lob_pct"] < 68) | (fg["lob_pct"] > 80)
        )
        fg["lob_buy_low"] = fg["lob_pct"] < 68  # stranding fewer = due for luck

    keep = ["pitcher_name_fg"] + [
        c for c in ["lob_pct", "xfip", "fip", "era_fg", "xera_fg",
                    "ip", "k9", "bb9", "era_minus_xfip",
                    "lob_unsustainable", "lob_buy_low"]
        if c in fg.columns
    ]

    if verbose:
        print(f"      {len(fg)} pitchers loaded | "
              f"buy-low candidates (LOB%<68): "
              f"{fg['lob_buy_low'].sum() if 'lob_buy_low' in fg.columns else 'N/A'}")

    return fg[keep]


# ---------------------------------------------------------------------------
# SECTION 7 — ASSEMBLE FINAL PROFILE
# ---------------------------------------------------------------------------

def assemble_profiles(vel, dist, trail, flags, stuff, age_arm, buyllow,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Merge all sections on pitcher_id. Derive Monte Carlo sigma (age-adjusted).
    """
    if verbose:
        print("  [7/7] Assembling final profiles ...")

    # Name map from 2026 statcast
    df26 = _load_statcast(2026, ["pitcher", "player_name", "p_throws"])
    name_map = _pitcher_name_map(df26)

    # Start from velocity (defines 2026 active pitchers)
    df = vel.copy()
    df["pitcher_name"] = df["pitcher"].map(name_map)

    # Merge career monthly distributions
    df = df.merge(dist, on="pitcher", how="left")

    # Merge 2026 trailing stats
    df = df.merge(trail, on="pitcher", how="left")

    # Fill league_sigma for pitchers not in dist at all
    league_sigma_xwoba = dist["league_sigma_xwoba"].iloc[0]
    df["league_sigma_xwoba"] = df["league_sigma_xwoba"].fillna(league_sigma_xwoba)
    # Fill any remaining per-month sigma nulls with league average
    for label in MONTH_LABELS.values():
        for s in STAT_COLS:
            col = f"{label}_{s}_std"
            if col in df.columns:
                df[col] = df[col].fillna(league_sigma_xwoba)

    # Merge new pitch flags
    df = df.merge(flags, on="pitcher", how="left")

    # Merge stuff proxy
    df = df.merge(stuff, on="pitcher", how="left")

    # Merge age + arm angle
    df = df.merge(age_arm, on="pitcher", how="left")

    # Merge FanGraphs signals on name
    # FanGraphs stores "FIRST LAST" after our normalization; statcast is "Last, First"
    # Normalize statcast names: "Doe, John" -> "JOHN DOE"
    def _normalize_name(name):
        if not isinstance(name, str):
            return ""
        name = name.strip()
        if "," in name:
            parts = [p.strip() for p in name.split(",", 1)]
            name = f"{parts[1]} {parts[0]}"
        return name.upper()

    df["pitcher_name_upper"] = df["pitcher_name"].apply(_normalize_name)
    buyllow_named = buyllow.copy()
    df = df.merge(
        buyllow_named.rename(columns={"pitcher_name_fg": "pitcher_name_upper"}),
        on="pitcher_name_upper", how="left")

    # ---- Monte Carlo sigma (month-aware) -----------------------------------
    # Use current calendar month's career sigma as base.
    # If no history for current month, fall back to April sigma, then league.
    # Multiplier: VOLATILE -> x1.5, GAINER -> x0.85, NORMAL -> x1.0
    current_month_label = MONTH_LABELS.get(pd.Timestamp.today().month, "apr")
    cur_std_col  = f"{current_month_label}_xwoba_std"
    apr_std_col  = "apr_xwoba_std"

    # Age-adjusted volatility multiplier
    # Base: VOLATILE=1.5, GAINER=0.85, NORMAL=1.0
    # Adjustments by age bucket:
    #   late_career (33+) + VOLATILE -> 2.0x  (velocity loss is a red flag)
    #   young (<26) + VOLATILE      -> 1.2x  (may just be a slow ramp-up)
    #   late_career + GAINER        -> 0.90x (unsustainable uplift, slight caution)
    base_sigma_map = {"VOLATILE": 1.5, "GAINER": 0.85, "NORMAL": 1.0, "UNKNOWN": 1.2}
    df["volatility_multiplier"] = df["velocity_flag"].map(base_sigma_map).fillna(1.0)

    # Apply age-bucket adjustments
    late_career_volatile = (
        (df["velocity_flag"] == "VOLATILE") & (df["age_bucket"] == "late_career")
    )
    young_volatile = (
        (df["velocity_flag"] == "VOLATILE") & (df["age_bucket"] == "young")
    )
    late_career_gainer = (
        (df["velocity_flag"] == "GAINER") & (df["age_bucket"] == "late_career")
    )
    df.loc[late_career_volatile, "volatility_multiplier"] = 2.0
    df.loc[young_volatile,       "volatility_multiplier"] = 1.2
    df.loc[late_career_gainer,   "volatility_multiplier"] = 0.90

    base_sigma = (
        df.get(cur_std_col, pd.Series(np.nan, index=df.index))
          .fillna(df.get(apr_std_col, pd.Series(np.nan, index=df.index)))
          .fillna(df["league_sigma_xwoba"])
    )
    df["monte_carlo_sigma"] = base_sigma * df["volatility_multiplier"]
    df["monte_carlo_sigma_month"] = current_month_label

    # ---- Blended current estimate (career monthly + 2026 trailing) ---------
    # For each key stat, produce a blended estimate the feature matrix can use
    for stat, trust_col in [("k_pct",   "trailing_trust_kbb"),
                             ("bb_pct",  "trailing_trust_kbb"),
                             ("xwoba",   "trailing_trust_xwoba"),
                             ("gb_pct",  "trailing_trust_xwoba")]:
        career_col   = f"{current_month_label}_{stat}_mean"
        trailing_col = f"trailing_{stat}"
        trust        = df.get(trust_col, pd.Series(0.0, index=df.index)).fillna(0)
        career_val   = df.get(career_col, pd.Series(np.nan, index=df.index))
        trail_val    = df.get(trailing_col, pd.Series(np.nan, index=df.index))

        # Blend: trust * trailing + (1-trust) * career
        df[f"blended_{stat}"] = (
            trust * trail_val.fillna(career_val) +
            (1 - trust) * career_val.fillna(trail_val)
        )

    # ---- New pitch K% multiplier (decaying advantage) ---------------------
    # +7.5% K% boost in the first ~30 starts after adoption
    df["new_pitch_k_boost"] = np.where(
        df["new_pitch_flag"].fillna(False), 0.075, 0.0)

    # ---- Summary flags for easy downstream use ----------------------------
    df["is_volatile"]   = df["velocity_flag"] == "VOLATILE"
    df["is_gainer"]     = df["velocity_flag"] == "GAINER"
    df["has_new_pitch"] = df["new_pitch_flag"].fillna(False)
    df["is_buy_low"]    = (
        df.get("lob_buy_low", pd.Series(False, index=df.index)).fillna(False)
    )

    if verbose:
        n_total = len(df)
        apr_col = "apr_starts"
        n_hist  = (df[apr_col].fillna(0) >= MIN_APRIL_STARTS).sum() \
            if apr_col in df.columns else 0
        print(f"      {n_total} pitchers in final profile")
        print(f"      {n_hist} have {MIN_APRIL_STARTS}+ April starts (reliable sigma)")
        print(f"      Monte Carlo sigma uses: {current_month_label.upper()} distribution")
        print(f"      VOLATILE: {df['is_volatile'].sum()} | "
              f"GAINER: {df['is_gainer'].sum()} | "
              f"new pitch: {df['has_new_pitch'].sum()} | "
              f"buy-low: {df['is_buy_low'].sum()}")
        if "age_bucket" in df.columns:
            lc_vol = (late_career_volatile).sum()
            print(f"      Age-adjusted sigma: late-career VOLATILE=2.0x ({lc_vol} pitchers) | "
                  f"young VOLATILE=1.2x ({young_volatile.sum()}) | "
                  f"late-career GAINER=0.90x ({late_career_gainer.sum()})")

    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build 2026 pitcher profiles for XGBoost + Monte Carlo")
    parser.add_argument("--velocity-threshold", type=float,
                        default=VELOCITY_THRESHOLD,
                        help="mph delta to trigger VOLATILE/GAINER flag")
    args = parser.parse_args()

    if args.velocity_threshold != VELOCITY_THRESHOLD:
        globals()["VELOCITY_THRESHOLD"] = args.velocity_threshold

    print("=" * 60)
    print("  build_pitcher_profile.py")
    print("=" * 60)

    vel     = build_velocity_profiles()
    dist    = build_monthly_distributions()
    trail   = build_trailing_2026()
    flags   = build_new_pitch_flags()
    stuff   = build_stuff_proxy()
    age_arm = build_age_arm_signals()
    buyllow = build_buyllow_signals()
    profiles = assemble_profiles(vel, dist, trail, flags, stuff, age_arm, buyllow)

    # ---- Save outputs ------------------------------------------------------
    out_parquet = OUTPUT_DIR / "pitcher_profiles_2026.parquet"
    out_csv     = "pitcher_profiles_2026.csv"

    profiles.to_parquet(out_parquet, engine="pyarrow", index=False)
    profiles.to_csv(out_csv, index=False)

    print()
    print(f"  Saved -> {out_parquet}  ({len(profiles)} rows x {len(profiles.columns)} cols)")
    print(f"  Saved -> {out_csv}")

    # ---- Summary report ----------------------------------------------------
    print()
    print("=" * 60)
    print("  VOLATILE pitchers (velocity down >1.5 mph vs 2025):")
    print("=" * 60)
    vol_cols = ["pitcher_name", "p_throws", "ff_velo_2025", "ff_velo_2026_april",
                "velocity_delta_vs_2025", "monte_carlo_sigma", "age_pit", "age_bucket",
                "volatility_multiplier"]
    vol_cols = [c for c in vol_cols if c in profiles.columns]
    vol = profiles[profiles["is_volatile"]].sort_values(
        "velocity_delta_vs_2025")[vol_cols]
    for _, r in vol.head(10).iterrows():
        age_str = (f"  age={r['age_pit']} ({r['age_bucket']})"
                   if "age_pit" in r.index and pd.notna(r.get("age_pit")) else "")
        mult_str = (f"  mult={r['volatility_multiplier']:.2f}x"
                    if "volatility_multiplier" in r.index else "")
        print(f"  {str(r['pitcher_name']):<28} {r['p_throws']}  "
              f"{r['ff_velo_2025']:.1f} -> {r['ff_velo_2026_april']:.1f} mph  "
              f"({r['velocity_delta_vs_2025']:+.1f})  "
              f"sigma={r['monte_carlo_sigma']:.4f}{mult_str}{age_str}")

    print()
    print("=" * 60)
    print("  GAINER pitchers (velocity up >1.5 mph vs 2025):")
    print("=" * 60)
    gain = profiles[profiles["is_gainer"]].sort_values(
        "velocity_delta_vs_2025", ascending=False)[
        ["pitcher_name", "p_throws", "ff_velo_2025", "ff_velo_2026_april",
         "velocity_delta_vs_2025", "xfip" if "xfip" in profiles.columns else "xera"]]
    for _, r in gain.head(10).iterrows():
        xfip_col = "xfip" if "xfip" in profiles.columns else "xera"
        print(f"  {str(r['pitcher_name']):<28} {r['p_throws']}  "
              f"{r['ff_velo_2025']:.1f} -> {r['ff_velo_2026_april']:.1f} mph  "
              f"({r['velocity_delta_vs_2025']:+.1f})  "
              f"xFIP/xERA={r[xfip_col]:.2f}" if pd.notna(r[xfip_col]) else
              f"  {str(r['pitcher_name']):<28} {r['p_throws']}  "
              f"{r['ff_velo_2025']:.1f} -> {r['ff_velo_2026_april']:.1f} mph  "
              f"({r['velocity_delta_vs_2025']:+.1f})")

    print()
    print("=" * 60)
    print("  NEW PITCH pitchers (added/increased pitch type >8%):")
    print("=" * 60)
    new_p_cols = ["pitcher_name", "new_pitch_type", "new_pitch_usage_delta",
                  "apr_k_pct_mean", "age_pit", "age_bucket"]
    new_p_cols = [c for c in new_p_cols if c in profiles.columns]
    new_p = profiles[profiles["has_new_pitch"]].sort_values(
        "new_pitch_usage_delta", ascending=False)[new_p_cols].head(10)
    for _, r in new_p.iterrows():
        k_str = (f"  april K%={r['apr_k_pct_mean']:.1%}"
                 if "apr_k_pct_mean" in r.index and pd.notna(r.get("apr_k_pct_mean"))
                 else "")
        age_str = (f"  age={r['age_pit']} ({r['age_bucket']})"
                   if "age_pit" in r.index and pd.notna(r.get("age_pit"))
                   else "")
        print(f"  {str(r['pitcher_name']):<28} +{r['new_pitch_type']}  "
              f"usage +{r['new_pitch_usage_delta']:.0%}{k_str}{age_str}")


if __name__ == "__main__":
    main()
