"""
build_feature_matrix.py
=======================
Build one row per historical game for XGBoost training and validation.

Architecture note (post-refactor)
----------------------------------
Park factors, temperature, wind direction, and Vegas lines are EXCLUDED from
this feature matrix.  The Monte Carlo engine already handles ballpark physics
and scoring environment.  Encoding them again here would double-count the
market signal and overfit on the 3-season sample.

Instead, the Monte Carlo output (expected_runs) is accepted as a single
continuous input at inference time, and XGBoost learns the residuals on top
of it.  During training on historical games the mc_expected_runs column is
left as NaN; XGBoost handles missing values natively (default left-branch
routing) so the model gracefully degrades to pitcher/batting features alone
when the MC value is absent.

Recency weighting — EWMA(halflife=15 games)
-------------------------------------------
Static season averages are replaced with an Exponentially Weighted Moving
Average (EWMA) over a 15-game half-life per pitcher / team.  For each game
the feature value reflects only the pitchers/teams performance UP TO (but
not including) that game, preventing any look-ahead leakage.

Years:
  train  -- 2023 + 2024
  val    -- 2025
  held   -- 2026 (never touched here)

Feature groups:
  SP EWMA     pitcher EWMA stats (k_pct, bb_pct, xwOBA, gb_pct, xrv, velo, arm …)
  SP diffs    home - away pitcher quality gaps
  Team batting team EWMA xwOBA vs LHP / RHP per game
  Matchup     home bat vs away SP handedness cross feature
  Calendar    month, year, day-of-week
  MC residual mc_expected_runs (NaN during training; populated at inference)

Labels:
  actual_home_win   binary
  home_margin       home_score - away_score
  home_covers_rl    home_margin >= 2
  total_runs        home_score + away_score

Outputs:
  feature_matrix.parquet
  feature_matrix.csv

Usage:
  python build_feature_matrix.py
  python build_feature_matrix.py --years 2024 2025
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── GPU DataFrame backend: cuDF with transparent pandas fallback ────────────
# When RAPIDS cuDF is present (RTX 5080 / CUDA env) all groupby EWMA and
# rolling-window transforms execute on the GPU.  On machines without RAPIDS
# every cudf.* call silently falls through to the equivalent pandas operation.
try:
    import cudf
    import cupy as _cp_bfm
    _CUDF = _cp_bfm.cuda.is_available()
    if not _CUDF:
        import pandas as cudf          # no device → CPU fallback
except ImportError:
    import pandas as cudf              # RAPIDS not installed → CPU fallback
    _CUDF = False

# ── cuML backend: GPU clustering + scaling with sklearn fallback ─────────────
# cuml.cluster.KMeans and cuml.preprocessing.StandardScaler run entirely on
# the RTX 5080 when RAPIDS is present.  Both have sklearn-compatible APIs.
try:
    from cuml.preprocessing import StandardScaler as _ArchScaler
    from cuml.cluster import KMeans as _ArchKMeans
    import cupy as _cp_arch
    _CUML_ARCH = _CUDF   # only use cuML if GPU is available
except ImportError:
    from sklearn.preprocessing import StandardScaler as _ArchScaler
    from sklearn.cluster import KMeans as _ArchKMeans
    _cp_arch   = None
    _CUML_ARCH = False

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = Path("./data/statcast")

BACKTEST_FILES = {
    2024: "data/raw/backtest_games_2024.csv",
    2025: "data/raw/backtest_games_2025.csv",
}

# ---------------------------------------------------------------------------
# Park run factors  (multi-year averages, 1.00 = league neutral)
# Source: Fangraphs park factors (2023-2025 average, 3-year rolling).
# COL is the dominant outlier; most parks cluster 0.96-1.05.
# Used as a direct feature — gives the model explicit ballpark run context
# that the MC engine already encodes but the XGBoost layer never sees.
# ---------------------------------------------------------------------------
STADIUM_ELEVATION = {   # feet above sea level
    "COL": 5200, "AZ": 1082, "TEX": 551, "HOU": 43, "ATL": 1050,
    "STL": 465,  "KC": 740,  "MIN": 840, "CIN": 550, "MIL": 635,
    "CHC": 595,  "CLE": 653, "DET": 600, "PIT": 730, "PHI": 20,
    "NYY": 55,   "NYM": 33,  "BOS": 19,  "BAL": 53,  "WSH": 25,
    "TOR": 249,  "TB": 28,   "MIA": 8,   "SF": 52,   "LAD": 512,
    "LAA": 160,  "SD": 17,   "SEA": 56,  "ATH": 25,  "CWS": 20,
}

PARK_FACTORS = {
    "COL": 1.281, "BOS": 1.078, "CIN": 1.062, "TEX": 1.050,
    "PHI": 1.040, "BAL": 1.038, "ATL": 1.034, "MIL": 1.029,
    "DET": 1.024, "CHC": 1.023, "HOU": 1.018, "STL": 1.014,
    "PIT": 1.012, "TB":  1.010, "KC":  1.005, "TOR": 1.000,
    "NYY": 0.998, "LAA": 0.996, "MIN": 0.995, "AZ":  0.993,
    "ARI": 0.993, "WSH": 0.990, "WAS": 0.990, "CLE": 0.988,
    "NYM": 0.984, "MIA": 0.978, "CWS": 0.975, "SF":  0.972,
    "LAD": 0.970, "SEA": 0.968, "OAK": 0.962, "ATH": 0.962,
    "SD":  0.960,
}

# ---------------------------------------------------------------------------
# Team home timezone offsets from Eastern Time
# ---------------------------------------------------------------------------
# Used to compute circadian disadvantage for away teams.
# Negative = behind ET (body clock thinks it's earlier than game time).
# During both EDT (summer) and EST (winter) the *relative* offset is stable.
#   ET  teams: offset  0   (NYY, BOS, TOR, DET, CLE, BAL, PHI, PIT, WSH,
#                           MIA, ATL, TB, CIN)
#   CT  teams: offset -1   (CHC, CWS, MIL, MIN, STL, KC, HOU, TEX)
#   MT  teams: offset -2   (COL, AZ)
#   PT  teams: offset -3   (LAD, LAA, SEA, SF, SD, ATH)
TEAM_ET_OFFSET: dict[str, int] = {
    # Eastern Time (offset = 0)
    "NYY": 0, "NYM": 0, "BOS": 0, "TOR": 0, "DET": 0,
    "CLE": 0, "BAL": 0, "PHI": 0, "PIT": 0, "WSH": 0, "WAS": 0,
    "MIA": 0, "ATL": 0, "TB":  0, "CIN": 0,
    # Central Time (offset = -1)
    "CHC": -1, "CWS": -1, "MIL": -1, "MIN": -1,
    "STL": -1, "KC":  -1, "HOU": -1, "TEX": -1,
    # Mountain Time (offset = -2)
    "COL": -2, "AZ": -2, "ARI": -2,
    # Pacific Time (offset = -3)
    "LAD": -3, "LAA": -3, "SEA": -3, "SF": -3,
    "SD":  -3, "ATH": -3, "OAK": -3,
}

# ---------------------------------------------------------------------------
# Decay / trailing-window constants  (Roadmap #7 + #2)
# ---------------------------------------------------------------------------
# Time-based EWMA halflife (calendar days).  Unlike the old game-count halflife,
# calendar-based decay naturally penalises IL stints and long gaps — a start
# from 60 days ago gets far less weight than one from 5 days ago, regardless of
# how many games were played in between.
#
# SP:  30 days ≈ 5-6 start halflife (starters pitch every 5-6 days)
# BAT: 21 days ≈ 21 game halflife   (teams play daily)
EWMA_HALFLIFE_SP_DAYS  = 30
EWMA_HALFLIFE_BAT_DAYS = 21

# Short trailing window for recency signal (Roadmap #2).
# 10 calendar days = last 1-2 starts for SP, last ~9 games for team batting.
# Captures hot/cold streaks that the slow EWMA smooths out.
TRAILING_DAYS = 10

# Minimum plate appearances for a pitcher appearance to count toward the EWMA.
# Spring training pitches have woba_denom=0 (not official PAs) so n_pa=0 even
# when strikeouts are recorded, creating k_pct values like 9.0 that corrupt
# the rolling mean.  Regular-season starts produce 15-27 PA; spring outings
# are almost always under 10.  A floor of 10 cleanly separates them.
MIN_PA_PER_START = 10

# ---------------------------------------------------------------------------
# Team archetype clustering constants  (Feature Space Stratification v5.1)
# ---------------------------------------------------------------------------
# Stylistic features only — outcome metrics (runs, W/L, R/G, OPS) are
# deliberately excluded to prevent target leakage into the cluster IDs.
# All features listed here are team EWMA stats already computed with a
# .shift(1) temporal boundary: each game's snapshot reflects performance
# strictly prior to that game's first pitch.
#
# Offensive proxy: K-rate and BB-rate (handedness-split EWMA).
#   bat_gb_pct / bat_fb_pct are not in the current batting EWMA schema.
#   To add them, extend build_team_batting_ewma() and append them below.
#
# Bullpen proxy: strikeout and walk rates per 9 innings.
#   bp_gb_pct is not in the current bullpen schema.
#   To add it, extend build_bullpen_stats() and append it below.
#
# Column-existence checks in append_team_archetypes() are graceful:
# any listed feature absent from the assembled matrix is simply skipped.
_ARCH_OFF_FEATS: list[str] = [
    "bat_k_vs_rhp",  "bat_k_vs_lhp",   # contact-avoidance style
    "bat_bb_vs_rhp", "bat_bb_vs_lhp",   # patience / walk tendency
    "bat_gb_pct",                        # groundball tendency (batted ball profile)
    "bat_fb_pct",                        # fly-ball tendency (power vs. contact)
]
_ARCH_BP_FEATS: list[str] = [
    "bp_k9",     # swing-and-miss / power arm profile
    "bp_bb9",    # control vs. walk-prone profile
    "bp_gb_pct", # groundball-inducing vs. fly-ball bullpen
]
N_ARCH_CLUSTERS_OFF: int = 4   # free-swingers | balanced | patient | contact
N_ARCH_CLUSTERS_BP:  int = 4   # power/K | balanced | groundball | HR-prone


def _load_statcast(year: int, cols: list) -> pd.DataFrame:
    path = DATA_DIR / f"statcast_{year}.parquet"
    df   = pd.read_parquet(path, engine="pyarrow", columns=cols)
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _to_num(series):
    return pd.to_numeric(series, errors="coerce")


def _time_ewm_transform(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    col: str,
    halflife_days: int,
) -> pd.Series:
    """
    Time-based EWMA per group with leakage prevention via shift(1).

    GPU path  (RAPIDS cuDF available):
      Converts the DataFrame to a cuDF frame, then uses cuDF's native
      groupby().ewm(halflife, times).mean() followed by groupby().shift(1)
      — both execute as fused CUDA kernels on the RTX 5080.

    CPU path  (pandas fallback):
      Original per-group loop over groups, unchanged for correctness.

    Unlike game-count EWM, this weights each observation by actual elapsed
    calendar time.  A stat from 60 days ago receives far less weight than one
    from 5 days ago, naturally penalising IL gaps without a separate flag.

    Returns a Series aligned to ``df.index``.
    """
    hl = pd.Timedelta(days=halflife_days)

    if _CUDF:
        # ── GPU path ────────────────────────────────────────────────────
        try:
            # Work on a minimal slice to reduce GPU memory transfer cost
            tmp = df[[group_col, date_col, col]].copy()
            tmp["_orig_idx"] = np.arange(len(tmp))

            # cuDF requires datetime column for time-based ewm 'times'
            tmp[date_col] = pd.to_datetime(tmp[date_col])

            cdf = cudf.from_pandas(tmp.reset_index(drop=True))
            cdf = cdf.sort_values([group_col, date_col])

            # Time-based EWMA: cuDF supports halflife as Timedelta string
            hl_ns = int(hl.total_seconds() * 1e9)   # nanoseconds for cuDF
            cdf["_ewma"] = (
                cdf.groupby(group_col, sort=False)[col]
                .ewm(halflife=hl_ns, times=cdf[date_col])
                .mean()
            )
            # Leakage prevention: shift(1) within each group
            cdf["_ewma_shifted"] = (
                cdf.groupby(group_col, sort=False)["_ewma"]
                .shift(1)
            )

            result_pdf = cdf[["_orig_idx", "_ewma_shifted"]].to_pandas()
            out = pd.Series(np.nan, index=df.index, dtype=float)
            orig_index = df.index.values
            result_pdf = result_pdf.sort_values("_orig_idx")
            out.iloc[result_pdf["_orig_idx"].values] = (
                result_pdf["_ewma_shifted"].values.astype(float)
            )
            return out
        except Exception as _cudf_err:
            # Fall through to CPU path on any cuDF error (e.g. unsupported dtype)
            warnings.warn(f"cuDF EWMA failed ({_cudf_err}), falling back to pandas.")

    # ── CPU / pandas fallback path ───────────────────────────────────────
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, grp in df.groupby(group_col, sort=False):
        grp  = grp.sort_values(date_col)
        ewma = (
            grp[col]
            .ewm(halflife=hl, times=grp[date_col])
            .mean()
            .shift(1)
        )
        out.loc[ewma.index] = ewma.values
    return out


def _trailing_nd_stat(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    col: str,
    days: int = 10,
) -> pd.Series:
    """
    For each game, compute the mean of ``col`` over the preceding ``days``
    calendar days, excluding the current game (no look-ahead leakage).

    GPU path  (RAPIDS cuDF available):
      Uses a cuDF rolling window over calendar days with a preceding-row
      shift for leakage prevention, executing as a CUDA kernel.

    CPU path  (pandas fallback):
      Original O(n²) nested loop over groups — unchanged for correctness.

    Games with no prior data in the window return NaN.
    Returns a Series aligned to ``df.index``.
    """
    if _CUDF:
        try:
            tmp = df[[group_col, date_col, col]].copy()
            tmp["_orig_idx"] = np.arange(len(tmp))
            tmp[date_col] = pd.to_datetime(tmp[date_col])

            cdf = cudf.from_pandas(tmp.reset_index(drop=True))
            cdf = cdf.sort_values([group_col, date_col])

            window_str = f"{days}D"
            cdf["_roll"] = (
                cdf.groupby(group_col, sort=False)[col]
                .rolling(window=window_str, on=date_col, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            # Shift(1) inside group to exclude the current game
            cdf["_roll_shifted"] = (
                cdf.groupby(group_col, sort=False)["_roll"]
                .shift(1)
            )

            result_pdf = cdf[["_orig_idx", "_roll_shifted"]].to_pandas()
            out = pd.Series(np.nan, index=df.index, dtype=float)
            result_pdf = result_pdf.sort_values("_orig_idx")
            out.iloc[result_pdf["_orig_idx"].values] = (
                result_pdf["_roll_shifted"].values.astype(float)
            )
            return out
        except Exception as _cudf_err:
            warnings.warn(f"cuDF rolling failed ({_cudf_err}), falling back to pandas.")

    # ── CPU / pandas fallback path ───────────────────────────────────────
    out = pd.Series(np.nan, index=df.index, dtype=float)
    td  = np.timedelta64(days, "D")

    for _, grp in df.groupby(group_col, sort=False):
        grp   = grp.sort_values(date_col).reset_index()   # keep orig index
        dates = grp[date_col].values
        vals  = grp[col].values.astype(float)
        idx   = grp["index"].values

        for i in range(1, len(grp)):
            cutoff = dates[i] - td
            mask   = (dates[:i] >= cutoff) & ~np.isnan(vals[:i])
            if mask.any():
                out.loc[idx[i]] = vals[:i][mask].mean()   # loc: label-based
    return out


def _normalize_name(name: str) -> str:
    """'Glasnow, Tyler' or 'TYLER GLASNOW' -> 'TYLER GLASNOW'"""
    if not isinstance(name, str):
        return ""
    name = name.strip()
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        name  = f"{parts[1]} {parts[0]}"
    return name.upper()


# Columns that must NOT be winsorized — flags, IDs, categoricals, labels
_WINSORIZE_SKIP = frozenset({
    "game_pk", "game_pk_num", "game_date", "season", "year", "split",
    "home_team", "away_team", "home_starter_name", "away_starter_name",
    "home_score", "away_score", "home_margin", "total_runs",
    "actual_home_win", "home_covers_rl", "away_covers_rl",
    "game_month", "game_day_of_week",
    "home_sp_p_throws_R", "away_sp_p_throws_R",
    "home_sp_il_return_flag", "away_sp_il_return_flag",
    "home_sp_starts_since_il", "away_sp_starts_since_il",
    "home_off_cluster", "away_off_cluster",
    "home_bp_cluster", "away_bp_cluster",
    "is_day_game", "mc_expected_runs",
    "ump_k_above_avg", "ump_bb_above_avg", "ump_rpg_above_avg",
})

def _winsorize_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Clip continuous feature columns at the 1st / 99th percentile computed
    from training rows (split == 'train') only, then applied to all rows.

    Winsorizing is useful for GBDTs because extreme outlier values can waste
    tree splits that will never generalize (e.g. one pitcher with xrv=9.0
    from a spring-training outing that slipped through).

    Percentile anchors are computed on train rows to prevent val leakage.
    For ratio/interaction columns the same anchors apply.
    """
    df = df.copy()

    # Only clip float64 / float32 columns not in the skip set
    float_cols = [
        c for c in df.columns
        if c not in _WINSORIZE_SKIP
        and pd.api.types.is_float_dtype(df[c])
    ]

    # Compute percentile bounds from training data
    train_mask = df.get("split", pd.Series("train", index=df.index)) == "train"
    train_df   = df.loc[train_mask, float_cols] if train_mask.any() else df[float_cols]

    clipped = 0
    for col in float_cols:
        lo = float(np.nanpercentile(train_df[col].dropna(), 1))
        hi = float(np.nanpercentile(train_df[col].dropna(), 99))
        if lo >= hi:          # degenerate range (constant column) — skip
            continue
        before = df[col].isna().sum()
        df[col] = df[col].clip(lower=lo, upper=hi)
        clipped += 1

    if verbose:
        print(f"      Winsorized {clipped} continuous columns at [1st, 99th] pct "
              f"(train-anchored)")
    return df


# ---------------------------------------------------------------------------
# SECTION 1 — GAME LIST (starters + game_pk + scores)
# ---------------------------------------------------------------------------

def build_game_list(years: list[int], verbose: bool = True) -> pd.DataFrame:
    """
    For each year, build a game-level dataframe with:
      game_date, game_pk, home_team, away_team, season
      home_starter_name, away_starter_name (FIRST LAST uppercase)
      home_score, away_score
    """
    if verbose:
        print("  [1/4] Building game list ...")

    frames = []
    for year in years:
        if year in BACKTEST_FILES:
            df = pd.read_csv(BACKTEST_FILES[year])
            df["game_date"] = pd.to_datetime(df["game_date"])
            df["season"]    = year
            df = df.rename(columns={
                "home_starter": "home_starter_name",
                "away_starter": "away_starter_name",
            })
            if verbose:
                print(f"      {year}: {len(df)} games from backtest CSV")
        else:
            df = _derive_game_list_from_statcast(year, verbose=verbose)

        sched_path = DATA_DIR / f"schedule_all_{year}.parquet"
        try:
            sched = pd.read_parquet(sched_path, engine="pyarrow")
            sched["gamePk"] = _to_num(sched["gamePk"])
            sched = sched[sched["home_away"] == "Home"][
                ["gamePk", "home_score", "away_score"]
            ].rename(columns={"gamePk": "game_pk"})
            sched = sched.drop_duplicates("game_pk")
            df["game_pk"] = _to_num(df["game_pk"])
            df = df.merge(sched, on="game_pk", how="left")
        except FileNotFoundError:
            if verbose:
                print(f"      {year}: schedule_all not found — scores will be NaN")

        frames.append(df)

    games = pd.concat(frames, ignore_index=True)
    games["home_score"] = _to_num(games.get("home_score", pd.Series(dtype=float)))
    games["away_score"] = _to_num(games.get("away_score", pd.Series(dtype=float)))

    if verbose:
        n_scored = (games["home_score"].notna() & games["away_score"].notna()).sum()
        print(f"      Total: {len(games)} games | {n_scored} with scores")

    return games


def _derive_game_list_from_statcast(year: int, verbose: bool = True) -> pd.DataFrame:
    cols = ["game_pk", "game_date", "home_team", "away_team",
            "pitcher", "player_name", "inning", "inning_topbot", "at_bat_number"]
    df = _load_statcast(year, cols)
    df = df[df["inning"] == 1].copy()
    df["at_bat_number"] = _to_num(df["at_bat_number"])

    results = []
    for game_pk, g in df.groupby("game_pk"):
        row = {"game_pk": game_pk}
        meta = g.iloc[0]
        row["game_date"] = meta["game_date"]
        row["home_team"] = meta["home_team"]
        row["away_team"] = meta["away_team"]

        top = g[g["inning_topbot"] == "Top"]
        if len(top) > 0:
            sp = top.loc[top["at_bat_number"].idxmin()]
            row["home_starter_name"] = _normalize_name(str(sp["player_name"]))

        bot = g[g["inning_topbot"] == "Bot"]
        if len(bot) > 0:
            sp = bot.loc[bot["at_bat_number"].idxmin()]
            row["away_starter_name"] = _normalize_name(str(sp["player_name"]))

        results.append(row)

    games = pd.DataFrame(results)
    games["season"] = year
    games = games.dropna(subset=["home_starter_name", "away_starter_name"])

    if verbose:
        print(f"      {year}: {len(games)} games derived from statcast")

    return games


# ---------------------------------------------------------------------------
# SECTION 2 — PER-PITCHER EWMA STATS  (replaces static season averages)
# ---------------------------------------------------------------------------

def build_pitcher_ewma_stats(years: list[int], verbose: bool = True) -> pd.DataFrame:
    """
    Compute per-pitcher EWMA stats at each game start using halflife=15 games.

    For each pitcher, game-level PA stats are sorted by game_date.
    The EWMA is computed and then shifted forward by 1 game so that the
    feature value for game N reflects only games 1 … N-1 (no leakage).

    Returns one row per (pitcher_name_normalized, game_pk, year) with
    EWMA-smoothed feature columns ready for a join onto the game list.
    """
    if verbose:
        print(f"  [2/4] Building pitcher stats "
              f"(EWMA halflife={EWMA_HALFLIFE_SP_DAYS}d, trailing={TRAILING_DAYS}d) ...")

    sc_cols = [
        "pitcher", "player_name", "game_pk", "game_date",
        "p_throws", "age_pit", "arm_angle",
        "events", "bb_type",
        "estimated_woba_using_speedangle",
        "delta_pitcher_run_exp",
        "woba_denom",
        "release_speed", "inning",
    ]

    all_game_frames = []

    for year in years:
        df = _load_statcast(year, sc_cols)
        df["game_date"]   = pd.to_datetime(df["game_date"])
        df["xwoba"]       = _to_num(df["estimated_woba_using_speedangle"])
        df["xrv"]         = _to_num(df["delta_pitcher_run_exp"])
        df["age_pit"]     = _to_num(df["age_pit"])
        df["arm_angle"]   = _to_num(df["arm_angle"])
        df["is_pa"]       = _to_num(df["woba_denom"]) > 0
        df["k"]           = (df["events"] == "strikeout").astype(float)
        df["bb"]          = (df["events"] == "walk").astype(float)
        df["is_gb"]       = (df["bb_type"] == "ground_ball").astype(float)
        df["is_bip"]      = df["bb_type"].notna().astype(float)
        df["h_on_bip"]    = df["events"].isin(
            ["single", "double", "triple"]).astype(float)
        df["rs"]          = _to_num(df["release_speed"]) if "release_speed" in df.columns else np.nan
        df["inn"]         = _to_num(df["inning"])        if "inning"         in df.columns else np.nan

        # Aggregate pitch-level data up to one row per (pitcher, game)
        game_agg = df.groupby(["pitcher", "game_pk", "game_date"]).agg(
            n_pa          = ("is_pa",        "sum"),
            k             = ("k",            "sum"),
            bb            = ("bb",           "sum"),
            gb            = ("is_gb",        "sum"),
            bip           = ("is_bip",       "sum"),
            h_on_bip      = ("h_on_bip",     "sum"),
            xwoba_against = ("xwoba",        "mean"),
            xrv_total     = ("xrv",          "sum"),
            n_pitches     = ("xrv",          "count"),
            age_pit       = ("age_pit",      "median"),
            arm_angle     = ("arm_angle",    "median"),
            p_throws      = ("p_throws",     lambda x: x.mode().iat[0]
                                             if len(x) > 0 else "R"),
            player_name   = ("player_name",  lambda x: x.dropna().iloc[0]
                                             if x.notna().any() else ""),
        ).reset_index()

        game_agg["k_pct"]         = game_agg["k"]   / game_agg["n_pa"].clip(lower=1)
        game_agg["bb_pct"]        = game_agg["bb"]  / game_agg["n_pa"].clip(lower=1)
        game_agg["gb_pct"]        = game_agg["gb"]  / game_agg["bip"].clip(lower=1)
        game_agg["xrv_per_pitch"] = game_agg["xrv_total"] / game_agg["n_pitches"].clip(lower=1)
        game_agg["k_minus_bb"]    = game_agg["k_pct"] - game_agg["bb_pct"]
        game_agg["p_throws_R"]    = (game_agg["p_throws"] == "R").astype(int)
        game_agg["pitcher_name_normalized"] = game_agg["player_name"].apply(_normalize_name)
        game_agg["year"] = year

        # BABIP per game (regression-to-mean luck indicator; NaN if < 3 BIP)
        game_agg["babip_game"] = np.where(
            game_agg["bip"] >= 3,
            game_agg["h_on_bip"] / game_agg["bip"].clip(lower=1),
            np.nan,
        )

        # Velocity decay: mean velo in early innings (<=2) vs late innings (>=5)
        # Negative decay = pitcher losing velo as game progresses (fatigue proxy)
        if df["rs"].notna().any():
            _early = (
                df[df["inn"] <= 2].groupby(["pitcher", "game_pk"])["rs"]
                .mean().rename("_velo_early")
            )
            _late = (
                df[df["inn"] >= 5].groupby(["pitcher", "game_pk"])["rs"]
                .mean().rename("_velo_late")
            )
            _vd = (
                pd.concat([_early, _late], axis=1)
                .reset_index()
                .assign(velo_decay=lambda x: x["_velo_early"] - x["_velo_late"])
                [["pitcher", "game_pk", "velo_decay"]]
            )
            game_agg = game_agg.merge(_vd, on=["pitcher", "game_pk"], how="left")
        else:
            game_agg["velo_decay"] = np.nan

        # ── Spring training filter ────────────────────────────────────────
        # Spring training pitches have woba_denom=0 (not official PAs) but
        # strikeout events are still recorded, producing extreme k_pct values
        # (e.g. n_pa=0, n_k=9 → k_pct=9.0) that poison the EWMA for real-
        # season starts.  Drop any per-game appearance with fewer than
        # MIN_PA_PER_START plate appearances before feeding the EWMA.
        # Regular-season starts typically produce 15-27 PA; spring outings
        # are almost always < 10 (and often 0).
        n_before = len(game_agg)
        game_agg = game_agg[game_agg["n_pa"] >= MIN_PA_PER_START].copy()
        n_dropped = n_before - len(game_agg)
        if verbose and n_dropped:
            print(f"      {year}: dropped {n_dropped} spring-training/short "
                  f"appearances (n_pa < {MIN_PA_PER_START})")

        # --- Merge fastball velo from pitch_arsenal for this year --------
        ars_path = DATA_DIR / f"pitch_arsenal_{year}.parquet"
        try:
            ars = pd.read_parquet(ars_path, engine="pyarrow",
                                  columns=["pitcher", "ff_avg_speed"])
            ars["pitcher"]  = _to_num(ars["pitcher"])
            ars = ars.rename(columns={"ff_avg_speed": "ff_velo"})
            game_agg = game_agg.merge(ars, on="pitcher", how="left")
        except FileNotFoundError:
            game_agg["ff_velo"] = np.nan

        # --- Percentile ranks for this year (season-level, not per-game) -
        perc_path = DATA_DIR / f"pitcher_percentiles_{year}.parquet"
        try:
            perc = pd.read_parquet(
                perc_path, engine="pyarrow",
                columns=["player_id", "whiff_percent", "fb_spin",
                         "fb_velocity", "xera"])
            perc = perc.rename(columns={
                "player_id":     "pitcher",
                "whiff_percent": "whiff_pctl",
                "fb_spin":       "fb_spin_pctl",
                "fb_velocity":   "fb_velo_pctl",
                "xera":          "xera_pctl",
            })
            perc["pitcher"] = _to_num(perc["pitcher"])
            for c in ["whiff_pctl", "fb_spin_pctl", "fb_velo_pctl", "xera_pctl"]:
                perc[c] = _to_num(perc[c]) / 100.0
            game_agg = game_agg.merge(perc, on="pitcher", how="left")
        except FileNotFoundError:
            for c in ["whiff_pctl", "fb_spin_pctl", "fb_velo_pctl", "xera_pctl"]:
                game_agg[c] = np.nan

        # --- Arsenal stats: usage-weighted run value + primary pitch quality -
        # Source: pitcher_arsenal_stats_{year}.parquet (Phase 2 pull)
        # arsenal_weighted_rv  — usage-weighted run value/100 across all pitch types
        # primary_whiff_pct    — whiff% on pitcher's most-thrown pitch
        # primary_putaway_pct  — put-away% on pitcher's most-thrown pitch
        ars_stats_path = DATA_DIR / f"pitcher_arsenal_stats_{year}.parquet"
        _ars_new_cols = ["arsenal_weighted_rv", "primary_whiff_pct", "primary_putaway_pct"]
        try:
            ars_s = pd.read_parquet(ars_stats_path, engine="pyarrow")
            ars_s.columns = [c.lower().replace(" ", "_") for c in ars_s.columns]
            pid_c   = next((c for c in ars_s.columns if c == "pitcher"), None)
            use_c   = next((c for c in ars_s.columns if "pitch_percent" in c or ("usage" in c and "pct" in c)), None)
            rv_c    = next((c for c in ars_s.columns if "run_value_per_100" in c or "run_value_per100" in c), None)
            whiff_c = next((c for c in ars_s.columns if "whiff" in c), None)
            put_c   = next((c for c in ars_s.columns if "put_away" in c or "putaway" in c), None)
            if pid_c and use_c and rv_c:
                ars_s[rv_c]  = _to_num(ars_s[rv_c])
                ars_s[use_c] = _to_num(ars_s[use_c])

                def _ars_agg(g):
                    w     = g[use_c].fillna(0)
                    total = w.sum()
                    rv    = (g[rv_c] * w).sum() / total if total > 0 else np.nan
                    if total > 0 and w.max() > 0:
                        prim  = g.loc[w.idxmax()]
                        whiff = _to_num(pd.Series([prim.get(whiff_c, np.nan)]))[0] if whiff_c else np.nan
                        put   = _to_num(pd.Series([prim.get(put_c,   np.nan)]))[0] if put_c   else np.nan
                    else:
                        whiff = put = np.nan
                    return pd.Series({"arsenal_weighted_rv":  rv,
                                      "primary_whiff_pct":   whiff,
                                      "primary_putaway_pct": put})

                ars_agg = ars_s.groupby(pid_c).apply(_ars_agg, include_groups=False).reset_index()
                ars_agg = ars_agg.rename(columns={pid_c: "pitcher"})
                ars_agg["pitcher"] = _to_num(ars_agg["pitcher"])
                game_agg = game_agg.merge(ars_agg, on="pitcher", how="left")
            else:
                for c in _ars_new_cols:
                    game_agg[c] = np.nan
        except FileNotFoundError:
            for c in _ars_new_cols:
                game_agg[c] = np.nan

        # --- Pitcher run value: swing/take decision quality (seasonal) ------
        # swing_rv_per100  — run value per 100 pitches on batter swings
        # take_rv_per100   — run value per 100 pitches on batter takes
        rv_path = DATA_DIR / f"pitcher_run_value_{year}.parquet"
        _rv_new_cols = ["swing_rv_per100", "take_rv_per100"]
        try:
            prv = pd.read_parquet(rv_path, engine="pyarrow")
            prv.columns = [c.lower().replace(" ", "_") for c in prv.columns]
            pid_c    = next((c for c in prv.columns if c == "pitcher"), None)
            swing_c  = next((c for c in prv.columns if "swing" in c and ("run_value" in c or "rv" in c)), None)
            take_c   = next((c for c in prv.columns if "take"  in c and ("run_value" in c or "rv" in c)), None)
            if pid_c and swing_c and take_c:
                prv_sub = prv[[pid_c, swing_c, take_c]].rename(columns={
                    pid_c:   "pitcher",
                    swing_c: "swing_rv_per100",
                    take_c:  "take_rv_per100",
                })
                prv_sub["pitcher"]         = _to_num(prv_sub["pitcher"])
                prv_sub["swing_rv_per100"] = _to_num(prv_sub["swing_rv_per100"])
                prv_sub["take_rv_per100"]  = _to_num(prv_sub["take_rv_per100"])
                game_agg = game_agg.merge(prv_sub, on="pitcher", how="left")
            else:
                for c in _rv_new_cols:
                    game_agg[c] = np.nan
        except FileNotFoundError:
            for c in _rv_new_cols:
                game_agg[c] = np.nan

        # --- Pitch movement: FF horizontal/vertical break (seasonal) --------
        # ff_h_break_inch  — 4-seam fastball arm-side horizontal break (inches)
        # ff_v_break_inch  — 4-seam fastball induced vertical break (inches)
        mov_path = DATA_DIR / f"pitcher_pitch_movement_{year}.parquet"
        _mov_new_cols = ["ff_h_break_inch", "ff_v_break_inch"]
        try:
            mov = pd.read_parquet(mov_path, engine="pyarrow")
            mov.columns = [c.lower().replace(" ", "_") for c in mov.columns]
            pid_c = next((c for c in mov.columns if c == "pitcher"), None)
            pt_c  = next((c for c in mov.columns if "pitch_type" in c), None)
            hb_c  = next((c for c in mov.columns if "x_break" in c or "h_break" in c or "pfx_x" in c), None)
            vb_c  = next((c for c in mov.columns if "z_break_ind" in c or "induced" in c
                          or ("pfx_z" in c) or ("z_break" in c and "spin" not in c)), None)
            if pid_c and pt_c and hb_c and vb_c:
                ff_rows = mov[mov[pt_c].isin(["FF", "FA"])][[pid_c, hb_c, vb_c]].copy()
                ff_rows[hb_c] = _to_num(ff_rows[hb_c])
                ff_rows[vb_c] = _to_num(ff_rows[vb_c])
                ff_agg = (ff_rows.groupby(pid_c)[[hb_c, vb_c]].mean()
                          .rename(columns={hb_c: "ff_h_break_inch", vb_c: "ff_v_break_inch"})
                          .reset_index()
                          .rename(columns={pid_c: "pitcher"}))
                ff_agg["pitcher"] = _to_num(ff_agg["pitcher"])
                game_agg = game_agg.merge(ff_agg, on="pitcher", how="left")
            else:
                for c in _mov_new_cols:
                    game_agg[c] = np.nan
        except FileNotFoundError:
            for c in _mov_new_cols:
                game_agg[c] = np.nan

        # Append AFTER all enrichment merges so ff_velo and percentile cols are included
        all_game_frames.append(game_agg)

    if not all_game_frames:
        return pd.DataFrame()

    all_games = pd.concat(all_game_frames, ignore_index=True)
    all_games  = all_games.sort_values(["pitcher", "game_date"]).reset_index(drop=True)

    # ── Time-based EWMA (halflife=EWMA_HALFLIFE_SP_DAYS calendar days) ────────
    # shift(1) by position ensures the feature at game N uses only games 1…N-1.
    # Time-based decay naturally handles IL gaps — no separate special-casing.
    ewma_targets = ["k_pct", "bb_pct", "xwoba_against", "gb_pct",
                    "xrv_per_pitch", "k_minus_bb",
                    "babip_game", "velo_decay"]

    # Save raw per-game values before overwriting with EWMA.
    # These are used below for the trailing 10-day short-window stats.
    trailing_raw_cols = ["k_pct", "bb_pct", "xwoba_against"]
    for col in trailing_raw_cols:
        if col in all_games.columns:
            all_games[f"_raw_{col}"] = all_games[col]

    all_games = all_games.sort_values(["pitcher", "game_date"]).reset_index(drop=True)
    for col in ewma_targets:
        if col in all_games.columns:
            all_games[col] = _time_ewm_transform(
                all_games, "pitcher", "game_date", col, EWMA_HALFLIFE_SP_DAYS
            )

    # Drop the very first start per pitcher (shift gives NaN — no prior history)
    all_games = all_games.dropna(subset=["k_pct"]).reset_index(drop=True)

    # ── Trailing 10-day stats (Roadmap #2) ────────────────────────────────────
    # For each start, compute the average of raw stats over the preceding
    # TRAILING_DAYS calendar days (≈ last 1-2 starts for a typical SP rotation).
    # Captures hot/cold streaks that the slow EWMA smooths over.
    for col in trailing_raw_cols:
        raw_col = f"_raw_{col}"
        if raw_col in all_games.columns:
            # Plain names — prefix is added by get_pitcher_features() later
            new_col = col.replace("xwoba_against", "xwoba") + "_10d"
            all_games[new_col] = _trailing_nd_stat(
                all_games, "pitcher", "game_date", raw_col, TRAILING_DAYS
            )
    # Drop temporary raw columns
    all_games = all_games.drop(
        columns=[f"_raw_{c}" for c in trailing_raw_cols if f"_raw_{c}" in all_games.columns]
    )

    # ── IL return flag ────────────────────────────────────────────────────
    # A gap > 14 days before a start is a proxy for an IL stint.
    # The next 3 starts after the gap are flagged as "IL return" starts.
    # This is known at game time — no leakage.
    all_games = all_games.sort_values(["pitcher", "game_date"]).reset_index(drop=True)
    all_games["prev_date"] = all_games.groupby("pitcher")["game_date"].shift(1)
    all_games["days_gap"]  = (all_games["game_date"] - all_games["prev_date"]).dt.days

    def _il_return_counts(group: pd.DataFrame) -> pd.Series:
        """
        For each row, return how many starts into an IL return the pitcher is.
        0 = not returning from IL.  1/2/3 = first/second/third start back.
        Resets to 0 after 3 starts.
        """
        counts = []
        active = 0
        for gap in group["days_gap"]:
            if pd.notna(gap) and gap > 14:
                active = 1
            elif active > 0:
                active += 1
                if active > 3:
                    active = 0
            counts.append(active)
        return pd.Series(counts, index=group.index)

    all_games["starts_since_il"] = (
        all_games.groupby("pitcher", group_keys=False)
        .apply(_il_return_counts)
    )
    all_games["il_return_flag"] = (all_games["starts_since_il"] > 0).astype(int)
    all_games = all_games.drop(columns=["prev_date", "days_gap"])

    if verbose:
        n_pit = all_games["pitcher_name_normalized"].nunique()
        n_il  = all_games["il_return_flag"].sum()
        print(f"      {len(all_games)} pitcher-game rows | {n_pit} unique pitchers "
              f"| {n_il} IL-return starts flagged")

    return all_games


# ---------------------------------------------------------------------------
# SECTION 3 — TEAM BATTING EWMA SPLITS  (replaces static season averages)
# ---------------------------------------------------------------------------

def build_team_batting_ewma(years: list[int], verbose: bool = True) -> pd.DataFrame:
    """
    Per-team batting xwOBA vs LHP / RHP, computed as EWMA(halflife=15 games).

    Returns one row per (batting_team, game_date, year) with columns:
      bat_xwoba_vs_rhp_ewma, bat_xwoba_vs_lhp_ewma
      bat_k_vs_rhp_ewma,     bat_k_vs_lhp_ewma
      bat_bb_vs_rhp_ewma,    bat_bb_vs_lhp_ewma
    """
    if verbose:
        print(f"  [3/4] Building team batting stats "
              f"(EWMA halflife={EWMA_HALFLIFE_BAT_DAYS}d, trailing={TRAILING_DAYS}d) ...")

    sc_cols = ["game_pk", "game_date", "home_team", "away_team",
               "inning_topbot", "p_throws",
               "estimated_woba_using_speedangle", "woba_denom",
               "events", "bb_type"]

    frames     = []
    bip_frames = []   # separate accumulator for GB/FB (no handedness split)

    for year in years:
        df = _load_statcast(year, sc_cols)
        df["game_date"] = pd.to_datetime(df["game_date"])
        df["xwoba"]     = _to_num(df["estimated_woba_using_speedangle"])
        df["is_pa"]     = _to_num(df["woba_denom"]) > 0
        df["k"]         = (df["events"] == "strikeout").astype(float)
        df["bb"]        = (df["events"] == "walk").astype(float)

        # Identify batting team per pitch
        df["batting_team"] = np.where(
            df["inning_topbot"] == "Top", df["away_team"], df["home_team"])

        pa_df = df[df["is_pa"]].copy()

        # Per-game per-team per-handedness aggregation
        game_splits = pa_df.groupby(
            ["batting_team", "game_date", "game_pk", "p_throws"]
        ).agg(
            xwoba  = ("xwoba", "mean"),
            k_pct  = ("k",     "mean"),
            bb_pct = ("bb",    "mean"),
        ).reset_index()

        # Pivot to wide: one row per (batting_team, game_date, game_pk)
        rhp = (game_splits[game_splits["p_throws"] == "R"]
               .rename(columns={"xwoba": "xwoba_vs_rhp",
                                "k_pct": "k_vs_rhp",
                                "bb_pct": "bb_vs_rhp"})
               .drop(columns="p_throws"))
        lhp = (game_splits[game_splits["p_throws"] == "L"]
               .rename(columns={"xwoba": "xwoba_vs_lhp",
                                "k_pct": "k_vs_lhp",
                                "bb_pct": "bb_vs_lhp"})
               .drop(columns="p_throws"))

        wide = rhp.merge(lhp, on=["batting_team", "game_date", "game_pk"], how="outer")
        wide = wide.sort_values(["batting_team", "game_date"]).reset_index(drop=True)
        wide["year"] = year
        frames.append(wide)

        # ── GB / FB rates (no handedness split — team batted-ball profile) ──
        # Computed only on balls in play (bb_type present); K, BB, HBP skipped.
        pa_df["is_gb"]  = (pa_df["bb_type"] == "ground_ball").astype(float)
        pa_df["is_fb"]  = (pa_df["bb_type"] == "fly_ball").astype(float)
        pa_df["is_bip"] = pa_df["bb_type"].notna().astype(float)

        bip_agg = pa_df.groupby(
            ["batting_team", "game_date", "game_pk"]
        ).agg(
            gb_cnt  = ("is_gb",  "sum"),
            fb_cnt  = ("is_fb",  "sum"),
            bip_cnt = ("is_bip", "sum"),
        ).reset_index()
        bip_agg["gb_pct"] = bip_agg["gb_cnt"] / bip_agg["bip_cnt"].clip(lower=1)
        bip_agg["fb_pct"] = bip_agg["fb_cnt"] / bip_agg["bip_cnt"].clip(lower=1)
        # Zero out rows with zero BIP (all-K/BB game — set to NaN so EWMA skips them)
        bip_agg.loc[bip_agg["bip_cnt"] == 0, ["gb_pct", "fb_pct"]] = np.nan
        bip_agg["year"] = year
        bip_frames.append(bip_agg[["batting_team", "game_date", "game_pk",
                                    "gb_pct", "fb_pct", "year"]])

    if not frames:
        return pd.DataFrame()

    all_bat = pd.concat(frames, ignore_index=True)
    all_bat = all_bat.sort_values(["batting_team", "game_date"]).reset_index(drop=True)

    # ── GB / FB EWMA (computed on all-year concatenated table for stable decay) ──
    if bip_frames:
        all_bip = pd.concat(bip_frames, ignore_index=True)
        all_bip = all_bip.sort_values(["batting_team", "game_date"]).reset_index(drop=True)
        for col in ["gb_pct", "fb_pct"]:
            all_bip[f"bat_{col}_ewma"] = _time_ewm_transform(
                all_bip, "batting_team", "game_date", col, EWMA_HALFLIFE_BAT_DAYS
            )
        all_bip = all_bip.rename(columns={
            "bat_gb_pct_ewma": "bat_gb_pct",
            "bat_fb_pct_ewma": "bat_fb_pct",
        })
        all_bat = all_bat.merge(
            all_bip[["batting_team", "game_pk", "bat_gb_pct", "bat_fb_pct"]],
            on=["batting_team", "game_pk"],
            how="left",
        )

    stat_cols = ["xwoba_vs_rhp", "k_vs_rhp", "bb_vs_rhp",
                 "xwoba_vs_lhp", "k_vs_lhp", "bb_vs_lhp"]

    # Save raw per-game values for trailing window stats (Roadmap #2)
    bat_trailing_raw = ["xwoba_vs_rhp", "xwoba_vs_lhp"]
    for col in bat_trailing_raw:
        if col in all_bat.columns:
            all_bat[f"_raw_{col}"] = all_bat[col]

    # Time-based EWMA (halflife=EWMA_HALFLIFE_BAT_DAYS calendar days)
    all_bat = all_bat.sort_values(["batting_team", "game_date"]).reset_index(drop=True)
    for col in stat_cols:
        if col in all_bat.columns:
            all_bat[f"bat_{col}_ewma"] = _time_ewm_transform(
                all_bat, "batting_team", "game_date", col, EWMA_HALFLIFE_BAT_DAYS
            )

    # Trailing 10-day batting xwOBA splits
    for col in bat_trailing_raw:
        raw_col = f"_raw_{col}"
        if raw_col in all_bat.columns:
            all_bat[f"bat_{col}_10d"] = _trailing_nd_stat(
                all_bat, "batting_team", "game_date", raw_col, TRAILING_DAYS
            )
    all_bat = all_bat.drop(
        columns=[f"_raw_{c}" for c in bat_trailing_raw if f"_raw_{c}" in all_bat.columns]
    )

    # Keep EWMA columns + trailing 10d columns + explicit GB/FB cols
    # (bat_gb_pct / bat_fb_pct are already renamed post-EWMA and don't end in _ewma)
    id_cols      = ["batting_team", "game_date", "game_pk", "year"]
    ewma_cols    = [c for c in all_bat.columns if c.endswith("_ewma")]
    trail_10d    = [c for c in all_bat.columns if c.endswith("_10d")]
    batball_cols = [c for c in ("bat_gb_pct", "bat_fb_pct") if c in all_bat.columns]
    all_bat      = all_bat[id_cols + ewma_cols + trail_10d + batball_cols].copy()

    # Rename to standard pipeline names
    rename = {
        "bat_xwoba_vs_rhp_ewma": "bat_xwoba_vs_rhp",
        "bat_k_vs_rhp_ewma":     "bat_k_vs_rhp",
        "bat_bb_vs_rhp_ewma":    "bat_bb_vs_rhp",
        "bat_xwoba_vs_lhp_ewma": "bat_xwoba_vs_lhp",
        "bat_k_vs_lhp_ewma":     "bat_k_vs_lhp",
        "bat_bb_vs_lhp_ewma":    "bat_bb_vs_lhp",
    }
    all_bat = all_bat.rename(columns=rename)

    # Drop rows where we had no prior game history (first game per team)
    all_bat = all_bat.dropna(subset=["bat_xwoba_vs_rhp", "bat_xwoba_vs_lhp"],
                             how="all")

    if verbose:
        n_teams = all_bat["batting_team"].nunique()
        print(f"      {len(all_bat)} team-game rows | {n_teams} unique teams")

    return all_bat


# ---------------------------------------------------------------------------
# SECTION 4 — BULLPEN QUALITY  (season-level, IP-weighted per team)
# ---------------------------------------------------------------------------

def build_bullpen_stats(years: list[int], verbose: bool = True) -> pd.DataFrame:
    """
    Build season-level bullpen quality per team per year.

    Uses IP-weighted averages over relievers with >= 5 IP in the season.
    Returns one row per (team, season) with columns:
      bp_era, bp_k9, bp_bb9, bp_hr9, bp_whip

    These are joined onto the game list by (home_team/away_team, season).
    Season-level values are intentionally NOT EWMA'd — bullpen composition
    changes slowly and individual-game-level bullpen data is unavailable
    in the current dataset.
    """
    if verbose:
        print("  [3.5/4] Building bullpen stats (IP-weighted season averages) ...")

    frames = []
    for year in years:
        path = DATA_DIR / f"bullpen_fg_{year}.parquet"
        try:
            df = pd.read_parquet(path, engine="pyarrow")
        except FileNotFoundError:
            if verbose:
                print(f"      {year}: bullpen_fg not found — skipping")
            continue

        df["ip"]       = pd.to_numeric(df["inningsPitched"],   errors="coerce")
        df["era_num"]  = pd.to_numeric(df["era"],              errors="coerce")
        df["k9"]       = pd.to_numeric(df["strikeoutsPer9Inn"], errors="coerce")
        df["bb9"]      = pd.to_numeric(df["walksPer9Inn"],      errors="coerce")
        df["hr9"]      = pd.to_numeric(df["homeRunsPer9"],      errors="coerce")
        df["whip_num"] = pd.to_numeric(df["whip"],              errors="coerce")
        # GB% proxy: groundOuts / (groundOuts + airOuts)
        # groundOutsToAirouts = groundOuts/airOuts; recover the fraction from it
        df["go"]       = pd.to_numeric(df["groundOuts"], errors="coerce")
        df["ao"]       = pd.to_numeric(df["airOuts"],    errors="coerce")
        df["gb_frac"]  = df["go"] / (df["go"] + df["ao"]).clip(lower=1)

        # Relievers only (no starts), minimum 5 IP sample
        rel = df[(df["gamesStarted"] == 0) & (df["ip"] >= 5)].copy()

        def _wavg(g: pd.DataFrame) -> pd.Series:
            w = g["ip"].fillna(0)
            total = w.sum()
            if total == 0:
                return pd.Series({"bp_era": np.nan, "bp_k9": np.nan,
                                  "bp_bb9": np.nan, "bp_hr9": np.nan,
                                  "bp_whip": np.nan, "bp_gb_pct": np.nan})
            return pd.Series({
                "bp_era":    (g["era_num"]  * w).sum() / total,
                "bp_k9":     (g["k9"]       * w).sum() / total,
                "bp_bb9":    (g["bb9"]      * w).sum() / total,
                "bp_hr9":    (g["hr9"]      * w).sum() / total,
                "bp_whip":   (g["whip_num"] * w).sum() / total,
                "bp_gb_pct": (g["gb_frac"].fillna(0.44) * w).sum() / total,
            })

        team_bp = rel.groupby("Team").apply(_wavg, include_groups=False).reset_index()
        team_bp["season"] = year
        team_bp = team_bp.rename(columns={"Team": "team"})
        frames.append(team_bp)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    if verbose:
        n = len(out)
        print(f"      {n} team-season bullpen rows ({n // max(len(years), 1)} teams/yr avg)")
    return out


# ---------------------------------------------------------------------------
# SECTION 4b — TEAM BAT TRACKING  (Phase 2 — 2023+)
# ---------------------------------------------------------------------------

def build_team_bat_tracking(years: list[int], verbose: bool = True) -> pd.DataFrame:
    """
    Team-level bat tracking metrics aggregated from per-batter seasonal leaderboard.
    Available 2023+; returns empty DataFrame (NaN fill) for earlier years.

    Columns returned:
      bat_speed_weighted   — usage-weighted mean bat speed (mph)
      blast_rate           — blast rate (squared-up + hard-hit combo %)
      squared_up_pct       — % of contact attempts that are squared-up
      bat_swing_rv         — team mean batter run value on swings (discipline)
      bat_take_rv          — team mean batter run value on takes (patience)

    Returns one row per (team abbreviation, season).
    """
    if verbose:
        print("  [3.7/4] Building team bat tracking + batter run values (2023+) ...")

    frames = []
    for year in years:
        row_parts = {}

        # ── Bat tracking (bat speed, blast, squared-up) ───────────────────
        bt_path = DATA_DIR / f"batter_bat_tracking_{year}.parquet"
        if bt_path.exists():
            try:
                bt = pd.read_parquet(bt_path, engine="pyarrow")
                bt.columns = [c.lower().replace(" ", "_") for c in bt.columns]
                team_c  = next((c for c in bt.columns if c in ("team", "team_name", "team_abbreviation")), None)
                spd_c   = next((c for c in bt.columns if "bat_speed" in c or "batspeed" in c), None)
                blast_c = next((c for c in bt.columns if "blast" in c), None)
                sq_c    = next((c for c in bt.columns if "squared_up" in c), None)
                n_c     = next((c for c in bt.columns if c in ("n_bip_pa", "attempts", "swings", "n")), None)

                if team_c and spd_c:
                    bt[spd_c] = _to_num(bt[spd_c])
                    w_vals    = _to_num(bt[n_c]) if n_c else pd.Series(np.ones(len(bt)))

                    def _bt_wavg(g):
                        w     = w_vals.loc[g.index].fillna(1)
                        total = w.sum()
                        if total == 0:
                            return pd.Series(dtype=float)
                        out = {"bat_speed_weighted": (_to_num(g[spd_c]) * w).sum() / total}
                        if blast_c:
                            out["blast_rate"]      = (_to_num(g[blast_c]).fillna(0) * w).sum() / total
                        if sq_c:
                            out["squared_up_pct"]  = (_to_num(g[sq_c]).fillna(0)   * w).sum() / total
                        return pd.Series(out)

                    bt_agg = bt.groupby(team_c).apply(_bt_wavg, include_groups=False).reset_index()
                    bt_agg = bt_agg.rename(columns={team_c: "team"})
                    row_parts["bat_track"] = bt_agg
            except Exception as e:
                if verbose:
                    print(f"      [{year}] bat_tracking error: {e}")

        # ── Batter run values (swing/take quality) ────────────────────────
        brv_path = DATA_DIR / f"batter_run_value_{year}.parquet"
        if brv_path.exists():
            try:
                brv = pd.read_parquet(brv_path, engine="pyarrow")
                brv.columns = [c.lower().replace(" ", "_") for c in brv.columns]
                team_c  = next((c for c in brv.columns if c in ("team", "team_name", "team_abbreviation")), None)
                pid_c   = next((c for c in brv.columns if c in ("batter", "player_id")), None)
                swing_c = next((c for c in brv.columns if "swing" in c and ("run_value" in c or "rv" in c)), None)
                take_c  = next((c for c in brv.columns if "take"  in c and ("run_value" in c or "rv" in c)), None)

                if team_c and swing_c and take_c:
                    brv[swing_c] = _to_num(brv[swing_c])
                    brv[take_c]  = _to_num(brv[take_c])
                    brv_agg = (brv.groupby(team_c)[[swing_c, take_c]].mean()
                               .rename(columns={swing_c: "bat_swing_rv", take_c: "bat_take_rv"})
                               .reset_index()
                               .rename(columns={team_c: "team"}))
                    row_parts["bat_rv"] = brv_agg
            except Exception as e:
                if verbose:
                    print(f"      [{year}] batter_run_value error: {e}")

        # ── Merge both parts ──────────────────────────────────────────────
        if row_parts:
            merged = None
            for df_part in row_parts.values():
                merged = df_part if merged is None else merged.merge(df_part, on="team", how="outer")
            merged["season"] = year
            frames.append(merged)

    if not frames:
        if verbose:
            print("      No bat tracking / batter run value files found.")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    if verbose:
        n_rows = len(out)
        n_yrs  = out["season"].nunique()
        print(f"      {n_rows} team-season rows | {n_yrs} years | "
              f"cols: {[c for c in out.columns if c not in ('team','season')]}")
    return out


# ---------------------------------------------------------------------------
# SECTION 4c — TEAM DEFENSE  (Phase 2 — OAA + FRV)
# ---------------------------------------------------------------------------

def build_team_defense(years: list[int], verbose: bool = True) -> pd.DataFrame:
    """
    Team-level Outs Above Average (OAA) and Fielding Run Value (FRV)
    summed across all fielding positions per season.

    Columns returned:
      team_oaa   — total OAA for all fielders on the roster
      team_frv   — total Fielding Run Value for all fielders on the roster

    Returns one row per (team abbreviation, season).
    """
    if verbose:
        print("  [3.8/4] Building team defense (OAA + FRV) ...")

    frames = []
    for year in years:
        row_parts = {}

        # ── OAA ───────────────────────────────────────────────────────────
        oaa_path = DATA_DIR / f"oaa_{year}.parquet"
        if oaa_path.exists():
            try:
                oaa = pd.read_parquet(oaa_path, engine="pyarrow")
                oaa.columns = [c.lower().replace(" ", "_") for c in oaa.columns]
                team_c = next((c for c in oaa.columns
                               if c in ("team", "team_name", "team_abbreviation")), None)
                oaa_c  = next((c for c in oaa.columns
                               if "outs_above_average" in c or c == "oaa"), None)
                if team_c and oaa_c:
                    oaa[oaa_c] = _to_num(oaa[oaa_c])
                    t_oaa = (oaa.groupby(team_c)[oaa_c].sum()
                             .reset_index()
                             .rename(columns={team_c: "team", oaa_c: "team_oaa"}))
                    row_parts["oaa"] = t_oaa
            except Exception as e:
                if verbose:
                    print(f"      [{year}] OAA error: {e}")

        # ── FRV ───────────────────────────────────────────────────────────
        frv_path = DATA_DIR / f"fielding_run_value_{year}.parquet"
        if frv_path.exists():
            try:
                frv = pd.read_parquet(frv_path, engine="pyarrow")
                frv.columns = [c.lower().replace(" ", "_") for c in frv.columns]
                team_c = next((c for c in frv.columns
                               if c in ("team", "team_name", "team_abbreviation")), None)
                frv_c  = next((c for c in frv.columns
                               if "fielding_run_value" in c or c == "frv"
                               or ("run_value" in c and "swing" not in c and "take" not in c)), None)
                if team_c and frv_c:
                    frv[frv_c] = _to_num(frv[frv_c])
                    t_frv = (frv.groupby(team_c)[frv_c].sum()
                             .reset_index()
                             .rename(columns={team_c: "team", frv_c: "team_frv"}))
                    row_parts["frv"] = t_frv
            except Exception as e:
                if verbose:
                    print(f"      [{year}] FRV error: {e}")

        if row_parts:
            merged = None
            for df_part in row_parts.values():
                merged = df_part if merged is None else merged.merge(df_part, on="team", how="outer")
            merged["season"] = year
            frames.append(merged)

    if not frames:
        if verbose:
            print("      No OAA / FRV files found.")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    if verbose:
        n_rows = len(out)
        print(f"      {n_rows} team-season rows | "
              f"cols: {[c for c in out.columns if c not in ('team','season')]}")
    return out


# ---------------------------------------------------------------------------
# SECTION 4d — BULLPEN FATIGUE  (QW4 — rolling 72-hour relief pitch count)
# ---------------------------------------------------------------------------

def build_bullpen_fatigue(years: list[int], verbose: bool = True) -> pd.DataFrame:
    """
    Per-team rolling count of relief pitcher pitches thrown in the prior 72 hours.
    High count = fatigued bullpen entering the game.

    Relief identification: any pitcher who does NOT appear in inning 1 for their
    team in a given game is a reliever.  This correctly handles openers (who
    appear in inning 1) and bulk pitchers (innings 2+).

    The 72-hour window uses closed='left' on the DatetimeIndex so the current
    game's pitches are NEVER counted in their own fatigue score (no leakage).

    Returns one row per (team, game_pk) with column bp_pitch_count_72h.
    """
    if verbose:
        print("  [3.9/4] Building bullpen fatigue (rolling 72h relief pitches) ...")

    sc_cols = ["pitcher", "game_pk", "game_date",
               "home_team", "away_team", "inning", "inning_topbot"]

    frames = []
    for year in years:
        try:
            df = _load_statcast(year, sc_cols)
        except Exception as e:
            if verbose:
                print(f"      {year}: statcast load failed — {e}")
            continue

        df["game_date"]     = pd.to_datetime(df["game_date"])
        df["inning"]        = _to_num(df["inning"])
        df["pitcher"]       = _to_num(df["pitcher"])
        df["inning_topbot"] = df["inning_topbot"].fillna("Top")

        # Pitching team: Top half = home team pitching; Bottom = away pitching
        df["pitching_team"] = np.where(
            df["inning_topbot"] == "Top", df["home_team"], df["away_team"]
        )

        # Starters = any pitcher appearing in inning 1 for their team in that game.
        # Handles true starters and openers; bulk pitchers (inn >= 2) are relievers.
        inning1 = (
            df[df["inning"] == 1][["game_pk", "pitching_team", "pitcher"]]
            .drop_duplicates()
            .assign(is_starter=1)
        )
        df = df.merge(inning1, on=["game_pk", "pitching_team", "pitcher"], how="left")
        df["is_reliever"] = df["is_starter"].isna().astype(int)

        # Relief pitch counts per (team, game, date)
        rel = (
            df[df["is_reliever"] == 1]
            .groupby(["pitching_team", "game_pk", "game_date"])
            .size()
            .reset_index(name="rel_pitches")
        )
        frames.append(rel)

    if not frames:
        if verbose:
            print("      No statcast data found for bullpen fatigue.")
        return pd.DataFrame()

    all_rel = pd.concat(frames, ignore_index=True)
    all_rel  = all_rel.sort_values(["pitching_team", "game_date"]).reset_index(drop=True)

    # Rolling 72h sum per team — closed='left' excludes the current game date.
    # Explicit for-loop avoids pandas 3.x groupby.apply key-drop behaviour.
    result_parts = []
    for team_name, grp in all_rel.groupby("pitching_team"):
        grp = grp.sort_values("game_date").set_index("game_date")
        grp["bp_pitch_count_72h"] = (
            grp["rel_pitches"]
            .rolling("3D", min_periods=0, closed="left")
            .sum()
        )
        grp = grp.reset_index()[["game_pk", "bp_pitch_count_72h"]]
        grp["team"] = team_name
        result_parts.append(grp)

    result = (
        pd.concat(result_parts, ignore_index=True)
        [["team", "game_pk", "bp_pitch_count_72h"]]
    )

    if verbose:
        n_rows   = len(result)
        n_teams  = result["team"].nunique()
        med_72h  = result["bp_pitch_count_72h"].median()
        print(f"      {n_rows} team-game rows | {n_teams} teams | "
              f"median relief pitches in prior 72h: {med_72h:.0f}")

    return result


# ---------------------------------------------------------------------------
# SECTION 4E — PITCHER GAME STATE
#   sp_pitches_last_start  : pitch count from pitcher's most recent start
#   sp_velo_diff_l14       : 14-day rolling mean FF velo minus season baseline
#   is_opener_flag         : 1 if max pitches in last 3 appearances < 40
# ---------------------------------------------------------------------------

def build_pitcher_game_state(years: list[int], verbose: bool = True) -> pd.DataFrame:
    """
    Per-pitcher, per-game state features derived from pitch-level statcast.

    All features are computed from *prior* games only (shift/leakage-safe).

    Returns one row per (pitcher_id, game_pk) with columns:
        sp_pitches_last_start   int    — pitch count in most recent prior start
        sp_velo_diff_l14        float  — (14d rolling avg FF velo) − season_baseline_velo
        is_opener_flag          int8   — 1 if max pitches across last 3 prior games < 40
    """
    if verbose:
        print("  [3.95/4] Building pitcher game-state features ...")

    sc_cols = ["pitcher", "game_pk", "game_date", "inning",
               "release_speed", "pitch_type", "home_team", "away_team",
               "inning_topbot", "pitcher_days_since_prev_game"]

    frames = []
    for year in years:
        try:
            df = _load_statcast(year, sc_cols)
        except Exception as e:
            if verbose:
                print(f"      {year}: load failed — {e}")
            continue

        df["game_date"] = pd.to_datetime(df["game_date"])
        df["inning"]    = _to_num(df["inning"])
        df["pitcher"]   = _to_num(df["pitcher"])

        # ── Pitch count per pitcher per game ───────────────────────────────
        pitch_counts = (
            df.groupby(["game_pk", "game_date", "pitcher"])
            .size()
            .reset_index(name="n_pitches")
        )

        # ── FF velocity per pitcher per game (mean of four-seam pitches) ───
        ff_velo = (
            df[df["pitch_type"].isin(["FF", "FA"])]
            .groupby(["game_pk", "game_date", "pitcher"])["release_speed"]
            .mean()
            .reset_index(name="ff_velo_game")
        )

        game_stats = pitch_counts.merge(ff_velo, on=["game_pk", "game_date", "pitcher"], how="left")
        game_stats = game_stats.sort_values(["pitcher", "game_date", "game_pk"])

        # ── Season baseline velo (expanding mean up to but not including current game) ─
        game_stats["ff_velo_baseline"] = (
            game_stats.groupby(["pitcher"])["ff_velo_game"]
            .transform(lambda s: s.shift(1).expanding().mean())
        )

        # ── sp_pitches_last_start: pitch count in immediately prior appearance ─
        game_stats["sp_pitches_last_start"] = (
            game_stats.groupby("pitcher")["n_pitches"].shift(1)
        )

        # ── is_opener_flag: max pitches across last 3 prior appearances < 40 ─
        max_last3 = (
            game_stats.groupby("pitcher")["n_pitches"]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).max())
        )
        game_stats["is_opener_flag"] = (max_last3 < 40).astype("Int8")
        # Null out flag if no prior data (debut game)
        game_stats.loc[game_stats["sp_pitches_last_start"].isna(), "is_opener_flag"] = pd.NA

        # ── sp_velo_diff_l14: 14-day rolling mean minus season baseline ────
        # Use game_count-based rolling with a 14-day cutoff filter to avoid
        # duplicate DatetimeIndex issues from doubleheader games.
        def _velo_l14(grp: pd.DataFrame) -> pd.Series:
            grp = grp.sort_values("game_date").copy()
            vals = []
            for i, (_, row) in enumerate(grp.iterrows()):
                cutoff = row["game_date"] - pd.Timedelta("14D")
                prior  = grp.iloc[:i]
                prior  = prior[prior["game_date"] >= cutoff]["ff_velo_game"].dropna()
                vals.append(prior.mean() if len(prior) >= 3 else float("nan"))
            return pd.Series(vals, index=grp.index)

        game_stats["ff_velo_l14"] = (
            game_stats.groupby("pitcher", group_keys=False)
            .apply(_velo_l14)
        )
        game_stats["sp_velo_diff_l14"] = (
            game_stats["ff_velo_l14"] - game_stats["ff_velo_baseline"]
        )

        # ── Tag each pitcher-game with their pitching team ────────────────
        # Top of inning = home team is pitching (away batters).
        # Bot of inning = away team is pitching (home batters).
        team_map = (
            df[["game_pk", "pitcher", "home_team", "away_team", "inning_topbot"]]
            .drop_duplicates(["game_pk", "pitcher"])
        )
        team_map["pitching_team"] = np.where(
            team_map["inning_topbot"] == "Top",
            team_map["home_team"],
            team_map["away_team"],
        )
        team_map["pitching_role"] = np.where(
            team_map["inning_topbot"] == "Top", "home", "away"
        )
        game_stats = game_stats.merge(
            team_map[["game_pk", "pitcher", "pitching_team", "pitching_role"]],
            on=["game_pk", "pitcher"], how="left"
        )

        frames.append(game_stats[["game_pk", "pitcher", "pitching_team",
                                   "pitching_role", "sp_pitches_last_start",
                                   "sp_velo_diff_l14", "is_opener_flag"]])

    if not frames:
        return pd.DataFrame(columns=["game_pk", "pitcher", "pitching_team",
                                      "pitching_role", "sp_pitches_last_start",
                                      "sp_velo_diff_l14", "is_opener_flag"])

    result = pd.concat(frames, ignore_index=True)
    result["sp_pitches_last_start"] = pd.to_numeric(result["sp_pitches_last_start"],
                                                     errors="coerce")
    # Keep only one row per (game_pk, pitching_role) — use the pitcher with most pitches
    # if multiple pitchers are tagged to the same role (edge case: doubleheaders keyed wrong)
    result = (result.sort_values("sp_pitches_last_start", ascending=False)
              .drop_duplicates(["game_pk", "pitching_role"]))
    if verbose:
        n_ok = result["sp_pitches_last_start"].notna().sum()
        print(f"      {len(result)} pitcher-game rows | "
              f"{n_ok} with pitch count ({100*n_ok/len(result):.1f}%)")
    return result


# ---------------------------------------------------------------------------
# SECTION 4F — ENVIRONMENTAL FEATURES
#   air_density_rho  : kg/m³ from temp + humidity + elevation
#   roof_closed_flag : 1 for domed/retractable-closed stadiums
# ---------------------------------------------------------------------------

# Retractable-roof teams + their default closure behaviour.
# "always" = fixed dome (MIL, HOU, TB, TOR, MIA, SEA, MIN).
# "weather" = open by default, closed when temp < 50°F or precip > 0.
_ROOF_TEAMS: dict[str, str] = {
    "MIL": "always",   # American Family Field — retractable, closed ~75% of games
    "HOU": "always",   # Minute Maid Park — retractable
    "TB":  "always",   # Tropicana Field — fixed dome
    "TOR": "always",   # Rogers Centre — retractable
    "MIA": "always",   # loanDepot park — retractable
    "SEA": "always",   # T-Mobile Park — retractable
    "MIN": "always",   # Target Field — open air (not retractable), keep as open
    "AZ":  "weather",  # Chase Field — retractable, opens in good weather
    "TEX": "weather",  # Globe Life Field — retractable
    "ATL": "weather",  # Truist Park — open air (keep as open)
}
# MIN and ATL are open-air; listed for reference, effectively 0.
_OPEN_AIR_OVERRIDE = {"MIN", "ATL"}

# Molecular weight of dry air (kg/mol), universal gas constant (J/mol·K)
_M_AIR = 0.0289644
_R_GAS = 8.31446


def _compute_air_density(temp_f: float, humidity_pct: float, elevation_m: float) -> float:
    """
    Air density (kg/m³) via the Ideal Gas Law with humidity correction.

    Station pressure estimated from elevation via the barometric formula
    (ISA standard atmosphere, accurate to ±1% below 3000 m):
        P_station = 101325 × (1 − 2.2557e-5 × h)^5.2559

    Humidity correction uses the partial pressure of water vapor (Buck eq):
        P_sat = 611.21 × exp((18.678 − T_c/234.5) × T_c / (257.14 + T_c))
        P_v   = (RH/100) × P_sat
        ρ = (P_d × M_d + P_v × M_v) / (R × T_K)
    """
    if any(v != v for v in [temp_f, humidity_pct, elevation_m]):  # NaN check
        return float("nan")
    temp_c  = (temp_f - 32) * 5 / 9
    temp_k  = temp_c + 273.15
    # Station pressure (Pa) from elevation
    p_sta   = 101325 * (1 - 2.2557e-5 * elevation_m) ** 5.2559
    # Saturation vapour pressure (Buck equation, Pa)
    p_sat   = 611.21 * np.exp((18.678 - temp_c / 234.5) * temp_c / (257.14 + temp_c))
    p_v     = (humidity_pct / 100.0) * p_sat   # partial pressure of vapour
    p_d     = p_sta - p_v                       # partial pressure of dry air
    # Density: ρ = (P_d·M_d + P_v·M_v) / (R·T)
    m_v     = 0.018016                          # kg/mol water vapour
    rho     = (p_d * _M_AIR + p_v * m_v) / (_R_GAS * temp_k)
    return round(rho, 5)


def build_air_density_features(
    years: list[int],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Join weather parquets to game list and compute air_density_rho and
    roof_closed_flag for every (home_team, game_date) pair.

    Edge cases
    ----------
    - Missing weather  → NaN for density, flag stays at dome default.
    - Dome stadiums    → density computed from indoor conditions (72°F / 50% RH).
    - Debuts / new teams → elevation defaults to 100 m (sea-level-ish).

    Returns DataFrame with columns:
        home_team, game_date, air_density_rho, roof_closed_flag
    """
    if verbose:
        print("  [3.97/4] Building air density + roof flag ...")

    # Stadium elevation in metres (convert feet → metres; dict defined at module top)
    elev_m: dict[str, float] = {t: ft * 0.3048 for t, ft in STADIUM_ELEVATION.items()}

    frames = []
    for year in years:
        path = Path(f"data/statcast/weather_{year}.parquet")
        if not path.exists():
            if verbose:
                print(f"      {year}: no weather file, skipping")
            continue
        wx = pd.read_parquet(path)
        wx["game_date"] = pd.to_datetime(wx["game_date"])
        if "humidity" not in wx.columns:
            wx["humidity"] = 50.0  # conservative fallback
        if "precip_mm" not in wx.columns:
            wx["precip_mm"] = 0.0

        rows = []
        for _, r in wx.iterrows():
            team  = r["home_team"]
            elev  = elev_m.get(team, 30.0)        # ~100 ft default
            roof  = _ROOF_TEAMS.get(team, "open")

            # Roof status
            if team in _OPEN_AIR_OVERRIDE:
                closed = 0
            elif roof == "always":
                closed = 1
            elif roof == "weather":
                # Retractable: close when cold or raining
                closed = int(r.get("temp_f", 72) < 50 or r.get("precip_mm", 0) > 0.5)
            else:
                closed = 0

            # For domed games use standard indoor conditions
            if closed:
                eff_temp = 72.0
                eff_hum  = 50.0
            else:
                eff_temp = float(r.get("temp_f", 72))
                eff_hum  = float(r.get("humidity", 50))

            rho = _compute_air_density(eff_temp, eff_hum, elev)

            # Directional wind + dew point for env-aware features
            from models.feature_engineering import compute_wind_vector_out
            wvo = compute_wind_vector_out(
                r.get("wind_mph"), r.get("wind_bearing"), team, roof_closed=closed
            )
            dp  = r.get("dew_point_f") if "dew_point_f" in wx.columns else None

            rows.append({
                "home_team":       team,
                "game_date":       r["game_date"],
                "air_density_rho": rho,
                "roof_closed_flag": closed,
                "wind_vector_out":  wvo,
                "dew_point_f":      dp,
                "temp_f":           eff_temp if not closed else None,
            })
        frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame(columns=["home_team", "game_date",
                                      "air_density_rho", "roof_closed_flag",
                                      "wind_vector_out", "dew_point_f", "temp_f"])
    result = pd.concat(frames, ignore_index=True)
    if verbose:
        n = result["air_density_rho"].notna().sum()
        mean_rho = result["air_density_rho"].mean()
        print(f"      {len(result)} game-rows | air_density mean={mean_rho:.4f} kg/m³ | "
              f"roofs closed: {result['roof_closed_flag'].sum()}")
    return result


# ---------------------------------------------------------------------------
# SECTION 4G — BULLPEN TOP-3 FATIGUE  (quality-weighted, 3-day window)
#   bullpen_top3_fatigue_3d : total pitches by team's 3 most-used relievers
#                             in the prior 72 hours — focuses on high-leverage
#                             arms rather than the full pen.
# ---------------------------------------------------------------------------

def build_bullpen_top3_fatigue(years: list[int], verbose: bool = True) -> pd.DataFrame:
    """
    Per-team count of pitches thrown in the prior 3 days (72 h) by the team's
    three most frequently used relievers in the trailing 14 days.

    Rationale: the existing bp_fatigue_72h sums ALL relief pitches and can be
    diluted by mop-up arms.  This metric tracks only the elite/high-leverage
    relievers who actually affect game outcomes.

    Leakage prevention: current game pitches are excluded via closed='left'.

    Returns one row per (team, game_pk) with column bullpen_top3_fatigue_3d.
    """
    if verbose:
        print("  [3.98/4] Building bullpen top-3 fatigue (3-day, quality arms) ...")

    sc_cols = ["pitcher", "game_pk", "game_date", "inning",
               "home_team", "away_team", "inning_topbot"]

    frames = []
    for year in years:
        try:
            raw = _load_statcast(year, sc_cols)
        except Exception as e:
            if verbose:
                print(f"      {year}: load failed — {e}")
            continue

        raw["game_date"]     = pd.to_datetime(raw["game_date"])
        raw["inning"]        = _to_num(raw["inning"])
        raw["pitcher"]       = _to_num(raw["pitcher"])
        raw["inning_topbot"] = raw["inning_topbot"].fillna("Top")
        raw["pitching_team"] = np.where(
            raw["inning_topbot"] == "Top", raw["home_team"], raw["away_team"]
        )

        # Starters = pitched inning 1 in that game
        starters = (
            raw[raw["inning"] == 1][["game_pk", "pitching_team", "pitcher"]]
            .drop_duplicates()
        )
        raw = raw.merge(
            starters.assign(is_starter=1),
            on=["game_pk", "pitching_team", "pitcher"], how="left"
        )
        relievers = raw[raw["is_starter"].isna()].copy()

        # Pitch count per reliever per game
        rp_game = (
            relievers.groupby(["game_pk", "game_date", "pitching_team", "pitcher"])
            .size()
            .reset_index(name="pitches")
        )
        rp_game = rp_game.sort_values("game_date")

        team_frames = []
        for team, grp in rp_game.groupby("pitching_team"):
            grp = grp.sort_values(["game_date", "game_pk"]).copy()

            # Per game: find top-3 relievers by 14-day usage, sum their 3-day pitches
            result_rows = []
            for idx, row in grp.drop_duplicates("game_pk").iterrows():
                gdate = row["game_date"]
                gp    = row["game_pk"]
                win14_start = gdate - pd.Timedelta("14D")
                win3_start  = gdate - pd.Timedelta("3D")

                # Prior games only (strict less-than = current game excluded)
                prior14 = grp[(grp["game_date"] >= win14_start) & (grp["game_date"] < gdate)]
                prior3  = grp[(grp["game_date"] >= win3_start)  & (grp["game_date"] < gdate)]

                # Top-3 by total pitches in prior 14 days
                top3 = (
                    prior14.groupby("pitcher")["pitches"].sum()
                    .nlargest(3).index.tolist()
                )
                top3_pitches = int(
                    prior3[prior3["pitcher"].isin(top3)]["pitches"].sum()
                )
                result_rows.append({
                    "team": team, "game_pk": gp,
                    "bullpen_top3_fatigue_3d": top3_pitches,
                })
            if result_rows:
                team_frames.append(pd.DataFrame(result_rows))

        if team_frames:
            frames.append(pd.concat(team_frames, ignore_index=True))

    if not frames:
        return pd.DataFrame(columns=["team", "game_pk", "bullpen_top3_fatigue_3d"])
    result = pd.concat(frames, ignore_index=True)
    if verbose:
        med = result["bullpen_top3_fatigue_3d"].median()
        print(f"      {len(result)} team-game rows | median top-3 pitches (3d): {med:.0f}")
    return result


# ---------------------------------------------------------------------------
# SECTION 4H — UMPIRE CALLED-STRIKE RATE
#   ump_called_strike_above_avg : HP umpire's called-strike rate vs league avg
#                                 on non-swing pitches (balls + called strikes)
# ---------------------------------------------------------------------------

def build_ump_called_strike_rate(years: list[int], verbose: bool = True) -> pd.DataFrame:
    """
    Per-game HP umpire called-strike rate above league average.

    Methodology
    -----------
    - Filter to non-swing pitches: description in {'called_strike', 'ball',
      'blocked_ball', 'pitchout'} (excludes swings, fouls, HBP).
    - CS_rate = called_strikes / non_swing_pitches per umpire per game.
    - Rolling EWMA (halflife=30 days) per umpire, shift(1) for leakage safety.
    - Subtract the league-day mean for that game date = above-avg metric.

    Edge cases: debut umpires → NaN (filled with 0 = league average).

    Uses the statcast `umpire` column (HP umpire ID, present on every pitch).

    Returns one row per game_pk with columns:
        game_pk, ump_called_strike_above_avg
    """
    if verbose:
        print("  [3.99/4] Building umpire called-strike rate ...")

    sc_cols = ["game_pk", "game_date", "description", "umpire",
               "home_team", "away_team"]
    NON_SWING = {"called_strike", "ball", "blocked_ball", "pitchout"}

    frames = []
    for year in years:
        # ── Load umpire assignments (game_pk → ump_hp_id) ─────────────────
        ump_path = DATA_DIR / f"umpire_assignments_{year}.parquet"
        if not ump_path.exists():
            if verbose:
                print(f"      {year}: no umpire_assignments file, skipping")
            continue
        assignments = pd.read_parquet(ump_path)[["game_pk", "ump_hp_id"]] \
                        .drop_duplicates("game_pk")

        try:
            df = _load_statcast(year, sc_cols)
        except Exception as e:
            if verbose:
                print(f"      {year}: statcast load failed — {e}")
            continue

        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df[df["description"].isin(NON_SWING)].copy()
        if df.empty:
            continue

        df["is_cs"] = (df["description"] == "called_strike").astype(int)

        # Per game called-strike rate (aggregate across all pitchers)
        game_cs = (
            df.groupby(["game_pk", "game_date"])
            .agg(n_called=("is_cs", "count"), n_cs=("is_cs", "sum"))
            .reset_index()
        )
        game_cs["cs_rate"] = game_cs["n_cs"] / game_cs["n_called"].clip(lower=1)

        # Join umpire ID onto game stats
        ump_game = game_cs.merge(assignments[["game_pk", "ump_hp_id"]], on="game_pk", how="inner")
        ump_game = ump_game.sort_values(["ump_hp_id", "game_date", "game_pk"])

        # EWMA per umpire with shift(1) — prior games only
        ump_game["cs_rate_ewma"] = (
            ump_game.groupby("ump_hp_id")["cs_rate"]
            .transform(lambda s: s.shift(1).ewm(span=30, ignore_na=True).mean())
        )

        # League daily mean for normalisation (using prior-game EWMA values)
        daily_mean = (
            ump_game.groupby("game_date")["cs_rate_ewma"]
            .mean()
            .rename("league_cs_rate")
        )
        ump_game = ump_game.join(daily_mean, on="game_date")
        ump_game["ump_called_strike_above_avg"] = (
            ump_game["cs_rate_ewma"] - ump_game["league_cs_rate"]
        ).fillna(0)   # debut ump = league average

        frames.append(ump_game[["game_pk", "ump_called_strike_above_avg"]])

    if not frames:
        return pd.DataFrame(columns=["game_pk", "ump_called_strike_above_avg"])
    result = pd.concat(frames, ignore_index=True).drop_duplicates("game_pk")
    if verbose:
        print(f"      {len(result)} games | "
              f"mean above_avg={result['ump_called_strike_above_avg'].mean():.4f}")
    return result


# ---------------------------------------------------------------------------
# SECTION 4I — LINEUP HANDEDNESS MATCHUP
# ---------------------------------------------------------------------------

def build_lineup_handedness(years: list[int], verbose: bool = True) -> pd.DataFrame:
    """
    For each game, count how many starting batters (inning 1 appearances)
    have a platoon advantage against the opposing SP:
      - batter `stand` is opposite of opposing SP `p_throws`  (L vs R, R vs L)
      - switch hitters (S) always count as advantaged

    Returns one row per game_pk with:
        game_pk,
        home_lineup_opp_hand_count  (home batters with platoon edge vs away SP)
        away_lineup_opp_hand_count  (away batters with platoon edge vs home SP)
        lineup_opp_hand_diff        (home - away)
    """
    if verbose:
        print("  [3.995/4] Building lineup handedness matchup ...")

    sc_cols = ["game_pk", "game_date", "inning", "inning_topbot",
               "batter", "stand", "pitcher", "p_throws",
               "home_team", "away_team"]

    frames = []
    for year in years:
        try:
            df = _load_statcast(year, sc_cols)
        except Exception as e:
            if verbose:
                print(f"      {year}: statcast load failed — {e}")
            continue

        if df.empty:
            continue

        df["inning"] = pd.to_numeric(df["inning"], errors="coerce")

        # Starting lineup = inning 1 only, first appearance per batter per game
        inn1 = df[df["inning"] == 1].copy()
        if inn1.empty:
            continue

        # First appearance of each batter in each game
        inn1 = inn1.sort_values("game_pk")
        first_app = inn1.drop_duplicates(subset=["game_pk", "batter"])

        # Identify SP for each side: pitcher with most pitches in inning 1, top/bot
        # top inning = home team pitching against away batters
        # bot inning = away team pitching against home batters
        home_sp = (
            first_app[first_app["inning_topbot"] == "Top"]
            .groupby(["game_pk", "pitcher"])["p_throws"]
            .first()
            .reset_index()
            .rename(columns={"pitcher": "away_sp_id", "p_throws": "away_sp_throws"})
        )
        # Take first pitcher (starter) per game
        home_sp = home_sp.groupby("game_pk").first().reset_index()

        away_sp = (
            first_app[first_app["inning_topbot"] == "Bot"]
            .groupby(["game_pk", "pitcher"])["p_throws"]
            .first()
            .reset_index()
            .rename(columns={"pitcher": "home_sp_id", "p_throws": "home_sp_throws"})
        )
        away_sp = away_sp.groupby("game_pk").first().reset_index()

        # Get batters per side
        home_batters = (
            first_app[first_app["inning_topbot"] == "Bot"]
            [["game_pk", "batter", "stand"]]
            .drop_duplicates()
        )
        away_batters = (
            first_app[first_app["inning_topbot"] == "Top"]
            [["game_pk", "batter", "stand"]]
            .drop_duplicates()
        )

        def count_platoon_edge(batters_df: pd.DataFrame,
                               sp_df: pd.DataFrame,
                               sp_throws_col: str) -> pd.DataFrame:
            merged = batters_df.merge(sp_df[["game_pk", sp_throws_col]],
                                      on="game_pk", how="left")
            # Switch hitters always get platoon edge; opposite hand = edge
            merged["platoon_adv"] = (
                (merged["stand"] == "S") |
                ((merged["stand"] == "L") & (merged[sp_throws_col] == "R")) |
                ((merged["stand"] == "R") & (merged[sp_throws_col] == "L"))
            ).astype(int)
            return (
                merged.groupby("game_pk")["platoon_adv"]
                .sum()
                .reset_index()
            )

        home_counts = count_platoon_edge(
            home_batters, away_sp, "home_sp_throws"
        ).rename(columns={"platoon_adv": "home_lineup_opp_hand_count"})

        away_counts = count_platoon_edge(
            away_batters, home_sp, "away_sp_throws"
        ).rename(columns={"platoon_adv": "away_lineup_opp_hand_count"})

        game_result = home_counts.merge(away_counts, on="game_pk", how="outer")
        game_result["lineup_opp_hand_diff"] = (
            game_result["home_lineup_opp_hand_count"]
            - game_result["away_lineup_opp_hand_count"]
        )
        frames.append(game_result)

    if not frames:
        return pd.DataFrame(columns=["game_pk", "home_lineup_opp_hand_count",
                                     "away_lineup_opp_hand_count",
                                     "lineup_opp_hand_diff"])

    result = pd.concat(frames, ignore_index=True).drop_duplicates("game_pk")
    if verbose:
        n_ok = result["home_lineup_opp_hand_count"].notna().sum()
        mean_diff = result["lineup_opp_hand_diff"].mean()
        print(f"      {n_ok} games | mean lineup_opp_hand_diff={mean_diff:.2f}")
    return result


# ---------------------------------------------------------------------------
# SECTION 4J — SP K-PROP MARKET SIGNAL
# ---------------------------------------------------------------------------

def _normalize_pitcher_name(name: str) -> str:
    """Lowercase, strip accents/punctuation for fuzzy matching."""
    import unicodedata
    name = str(name).lower().strip()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.replace(".", "").replace("-", " ").replace("'", "")
    return " ".join(name.split())


def build_kprop_features(years: list[int], verbose: bool = True) -> pd.DataFrame:
    """
    Load daily `k_props_{date}.parquet` files and produce per-game SP K-line features.

    K-prop lines are DraftKings/market consensus SP strikeout over/unders.
    Only available from 2026-04-12 onwards; prior years return empty (NaN join).

    Returns one row per (game_pk, game_date, home_team, away_team) with:
        home_sp_k_line        — market K total for home SP (NaN if absent)
        away_sp_k_line        — market K total for away SP
        sp_k_line_diff        — home minus away (positive = home SP higher K line)
        home_sp_k_line_implied — vig-removed over probability for home SP
        away_sp_k_line_implied — vig-removed over probability for away SP
    """
    if verbose:
        print("  [3.996/4] Building SP K-prop market signal ...")

    def _vig_remove(over_odds: float, under_odds: float) -> float:
        """Convert American odds to vig-removed over probability."""
        def _to_prob(o: float) -> float:
            if o >= 0:
                return 100 / (o + 100)
            return (-o) / (-o + 100)
        p_over  = _to_prob(over_odds)
        p_under = _to_prob(under_odds)
        total   = p_over + p_under
        return p_over / total if total > 0 else 0.5

    all_frames = []
    for year in years:
        files = sorted(DATA_DIR.glob(f"k_props_{year}-*.parquet"))
        if not files:
            continue
        daily = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        daily["game_date"] = pd.to_datetime(daily["game_date"])
        daily["pitcher_norm"] = daily["pitcher_name"].apply(_normalize_pitcher_name)

        # Vig-removed over probability
        daily["k_over_prob"] = daily.apply(
            lambda r: _vig_remove(r["over_odds"], r["under_odds"]), axis=1
        )

        # Deduplicate multiple books: median line + mean implied prob per pitcher-game
        agg = (
            daily.groupby(["game_date", "home_team", "away_team", "pitcher_norm"])
            .agg(sp_k_line=("line", "median"),
                 sp_k_over_prob=("k_over_prob", "mean"))
            .reset_index()
        )
        all_frames.append(agg)

    if not all_frames:
        return pd.DataFrame(columns=["game_date", "home_team", "away_team",
                                     "home_sp_k_line", "away_sp_k_line",
                                     "sp_k_line_diff",
                                     "home_sp_k_line_implied",
                                     "away_sp_k_line_implied"])

    props = pd.concat(all_frames, ignore_index=True)

    # We need to match pitcher_norm against home/away starter names in game list.
    # This join happens in assemble_matrix where we have starter names.
    # Return the raw props; assemble_matrix does the merge.
    if verbose:
        n = len(props)
        n_games = props.groupby(["game_date", "home_team", "away_team"]).ngroups
        print(f"      {n} pitcher-game K-prop rows | {n_games} game-dates")
    return props


# ---------------------------------------------------------------------------
# SECTION 4B — ELO RATINGS
# ---------------------------------------------------------------------------

ELO_BASE      = 1500   # new-team / regression-target rating
ELO_K         = 20     # K-factor (MLB calibrated: 1 win ≈ ±10-15 Elo pts)
ELO_HFA       = 35     # home-field advantage in Elo points
ELO_REGRESS   = 0.50   # fraction of season-end deviation retained into next season


def compute_elo_ratings(
    df: pd.DataFrame,
    k: float = ELO_K,
    base_elo: float = ELO_BASE,
    hfa: float = ELO_HFA,
    season_regression: float = ELO_REGRESS,
) -> tuple[pd.DataFrame, dict]:
    """
    Compute rolling pre-game Elo ratings from the full game DataFrame.

    Uses standard Elo update with:
      - Home-field advantage of `hfa` Elo points added to expected score
      - 50% regression toward 1500 at each season boundary
      - shift-by-row leakage prevention (ratings updated AFTER recording)

    Returns
    -------
    elo_df : DataFrame with columns [game_pk, home_elo, away_elo]
    ratings : dict of {team: current_elo} after all processed games
    """
    df = df.sort_values(["game_date", "game_pk"]).copy()
    ratings: dict[str, float] = {}
    current_season: int | None = None
    records: list[dict] = []

    for _, row in df.iterrows():
        home   = row["home_team"]
        away   = row["away_team"]
        season = int(row["year"])

        # Season boundary: regress all ratings toward the mean
        if season != current_season:
            if current_season is not None:
                for team in list(ratings):
                    ratings[team] = base_elo + season_regression * (ratings[team] - base_elo)
            current_season = season

        home_r = ratings.get(home, base_elo)
        away_r = ratings.get(away, base_elo)

        records.append({"game_pk": row["game_pk"], "home_elo": home_r, "away_elo": away_r})

        # Update only when game result is known
        h_score = row.get("home_score")
        a_score = row.get("away_score")
        if pd.notna(h_score) and pd.notna(a_score):
            home_win = 1.0 if float(h_score) > float(a_score) else 0.0
            home_exp = 1.0 / (1.0 + 10.0 ** ((away_r - home_r - hfa) / 400.0))
            ratings[home] = home_r + k * (home_win - home_exp)
            ratings[away] = away_r + k * ((1.0 - home_win) - (1.0 - home_exp))

    return pd.DataFrame(records), ratings


# ---------------------------------------------------------------------------
# SECTION 5 — ASSEMBLE FEATURE MATRIX
# ---------------------------------------------------------------------------

def assemble_matrix(
    games:              pd.DataFrame,
    pitchers:           pd.DataFrame,
    batting:            pd.DataFrame,
    bullpen:            pd.DataFrame,
    bat_tracking:       pd.DataFrame | None = None,
    defense:            pd.DataFrame | None = None,
    bp_fatigue:         pd.DataFrame | None = None,
    pitcher_game_state: pd.DataFrame | None = None,
    air_density:        pd.DataFrame | None = None,
    bp_top3_fatigue:    pd.DataFrame | None = None,
    ump_cs_rate:        pd.DataFrame | None = None,
    lineup_handedness:  pd.DataFrame | None = None,
    kprop_features:     pd.DataFrame | None = None,
    verbose:            bool = True,
) -> pd.DataFrame:
    """
    Merge all feature tables onto the game-level dataframe.

    Park factors, weather, and Vegas lines are intentionally excluded:
    the Monte Carlo engine already encodes the scoring environment.
    mc_expected_runs is added as a NaN placeholder; it will be populated
    at inference time before XGBoost scoring.

    Phase 2 additions (graceful NaN if tables not provided):
      bat_tracking       — team bat speed, blast rate, batter run values
      defense            — team OAA, Fielding Run Value
      pitcher_game_state — sp_pitches_last_start, sp_velo_diff_l14, is_opener_flag
      air_density        — air_density_rho, roof_closed_flag
      bp_top3_fatigue    — bullpen_top3_fatigue_3d
      ump_cs_rate        — ump_called_strike_above_avg
      lineup_handedness  — home/away_lineup_opp_hand_count, lineup_opp_hand_diff
    """
    if verbose:
        print("  [4/4] Assembling feature matrix ...")

    df = games.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["year"]      = df["season"].astype(int)

    # ── Series game number ────────────────────────────────────────────────
    # 1 = first game of a series, 2/3/4 = subsequent games.
    # A new series starts when the same team pair hasn't played in > 5 days.
    df = df.sort_values("game_date").reset_index(drop=True)
    df["_team_pair"]       = [tuple(sorted([h, a]))
                               for h, a in zip(df["home_team"], df["away_team"])]
    df["_prev_pair_date"]  = df.groupby("_team_pair")["game_date"].shift(1)
    df["_days_since_pair"] = (df["game_date"] - df["_prev_pair_date"]).dt.days
    df["_new_series"]      = (
        df["_days_since_pair"].isna() | (df["_days_since_pair"] > 5)
    ).astype(int)
    df["_series_id"]       = df.groupby("_team_pair")["_new_series"].cumsum()
    df["series_game_number"] = (
        df.groupby(["_team_pair", "_series_id"]).cumcount() + 1
    )
    df = df.drop(columns=["_team_pair", "_prev_pair_date",
                           "_days_since_pair", "_new_series", "_series_id"])

    # Calendar features
    df["game_month"]       = df["game_date"].dt.month
    df["game_day_of_week"] = df["game_date"].dt.dayofweek   # 0=Mon

    # ── Pitcher EWMA lookup: (name, game_pk) → feature row ───────────────
    # Prefer game_pk join (exact); fall back to (name, year, nearest date)
    pit_lookup = pitchers.set_index(["pitcher_name_normalized", "game_pk"])

    def get_pitcher_features(name: str, game_pk, prefix: str) -> dict:
        feature_cols = [
            "k_pct", "bb_pct", "xwoba_against", "gb_pct", "xrv_per_pitch",
            "k_minus_bb", "ff_velo", "age_pit", "arm_angle", "p_throws_R",
            "whiff_pctl", "fb_spin_pctl", "fb_velo_pctl", "xera_pctl",
            "il_return_flag", "starts_since_il",
            # Trailing 10-day recency stats
            "k_pct_10d", "bb_pct_10d", "xwoba_10d",
            # Quick Win: BABIP luck + velocity decay trend
            "babip_game", "velo_decay",
            # NOTE: Phase 2 arsenal/movement/swing-rv features omitted — data files
            # (pitcher_arsenal_stats, pitcher_run_value, pitcher_pitch_movement)
            # are not yet populated; they are 100% NaN and add no signal.
        ]
        try:
            row = pit_lookup.loc[(name, game_pk)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
        except KeyError:
            row = pd.Series(dtype=float)
        return {f"{prefix}_{c}": row.get(c, np.nan) for c in feature_cols}

    home_feats = df.apply(
        lambda r: get_pitcher_features(
            r["home_starter_name"], r["game_pk"], "home_sp"),
        axis=1,
    ).apply(pd.Series)
    away_feats = df.apply(
        lambda r: get_pitcher_features(
            r["away_starter_name"], r["game_pk"], "away_sp"),
        axis=1,
    ).apply(pd.Series)

    df = pd.concat([df, home_feats, away_feats], axis=1)

    # SP differential features (positive = home advantage)
    df["sp_k_pct_diff"]    = df["home_sp_k_pct"]         - df["away_sp_k_pct"]
    df["sp_xwoba_diff"]    = df["away_sp_xwoba_against"]  - df["home_sp_xwoba_against"]
    df["sp_xrv_diff"]      = df["home_sp_xrv_per_pitch"]  - df["away_sp_xrv_per_pitch"]
    df["sp_velo_diff"]     = df["home_sp_ff_velo"]        - df["away_sp_ff_velo"]
    df["sp_age_diff"]      = df["home_sp_age_pit"]        - df["away_sp_age_pit"]
    df["sp_kminusbb_diff"] = df["home_sp_k_minus_bb"]     - df["away_sp_k_minus_bb"]

    # Trailing 10-day SP diffs — recency signal
    df["sp_k_pct_10d_diff"]  = (df["home_sp_k_pct_10d"]
                                - df["away_sp_k_pct_10d"])
    df["sp_xwoba_10d_diff"]  = (df["away_sp_xwoba_10d"]
                                - df["home_sp_xwoba_10d"])   # flipped: lower home xwOBA = better
    df["sp_bb_pct_10d_diff"] = (df["home_sp_bb_pct_10d"]
                                - df["away_sp_bb_pct_10d"])

    # Quick Win: form/talent ratio (recent xwOBA vs season-EWMA baseline)
    # > 1 = pitcher allowing more xwOBA recently than usual (regression candidate)
    _eps = 0.01
    df["home_sp_form_talent_ratio"] = (
        df["home_sp_xwoba_10d"] / (df["home_sp_xwoba_against"] + _eps)
    )
    df["away_sp_form_talent_ratio"] = (
        df["away_sp_xwoba_10d"] / (df["away_sp_xwoba_against"] + _eps)
    )
    df["sp_form_talent_ratio_diff"] = (
        df["home_sp_form_talent_ratio"] - df["away_sp_form_talent_ratio"]
    )

    # Quick Win: BABIP luck gap (EWMA BABIP vs .295 league avg pitcher BABIP)
    # Positive = pitching through bad luck on BIP (regression-toward-mean signal)
    if "home_sp_babip_game" in df.columns:
        _babip_avg = 0.295
        df["home_sp_babip_luck"] = df["home_sp_babip_game"] - _babip_avg
        df["away_sp_babip_luck"] = df["away_sp_babip_game"] - _babip_avg
        df["sp_babip_luck_diff"] = df["home_sp_babip_luck"] - df["away_sp_babip_luck"]

    # Quick Win: velocity decay diff (positive = home SP loses less velo late)
    if "home_sp_velo_decay" in df.columns:
        df["sp_velo_decay_diff"] = df["home_sp_velo_decay"] - df["away_sp_velo_decay"]

    # ── Team batting EWMA lookup: join via (team, game_pk) ───────────────
    bat_lookup = batting.set_index(["batting_team", "game_pk"])

    bat_stat_cols = ["bat_xwoba_vs_rhp", "bat_xwoba_vs_lhp",
                     "bat_k_vs_rhp",     "bat_k_vs_lhp",
                     "bat_bb_vs_rhp",    "bat_bb_vs_lhp",
                     # Batted-ball profile (for team archetype clustering)
                     "bat_gb_pct",       "bat_fb_pct",
                     # Trailing 10-day batting xwOBA (Roadmap #2)
                     "bat_xwoba_vs_rhp_10d", "bat_xwoba_vs_lhp_10d"]

    def get_batting(team: str, game_pk, prefix: str) -> dict:
        try:
            row = bat_lookup.loc[(team, game_pk)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
        except KeyError:
            row = pd.Series(dtype=float)
        return {f"{prefix}_{c}": row.get(c, np.nan) for c in bat_stat_cols}

    home_bat = df.apply(
        lambda r: get_batting(r["home_team"], r["game_pk"], "home"), axis=1
    ).apply(pd.Series)
    away_bat = df.apply(
        lambda r: get_batting(r["away_team"], r["game_pk"], "away"), axis=1
    ).apply(pd.Series)

    df = pd.concat([df, home_bat, away_bat], axis=1)

    # Matchup-specific: home bats vs away SP handedness
    # Columns from get_batting(prefix="home") are named: home_bat_xwoba_vs_rhp, etc.
    df["home_bat_vs_away_sp"] = np.where(
        df["away_sp_p_throws_R"] == 1,
        df["home_bat_xwoba_vs_rhp"],
        df["home_bat_xwoba_vs_lhp"],
    )
    df["away_bat_vs_home_sp"] = np.where(
        df["home_sp_p_throws_R"] == 1,
        df["away_bat_xwoba_vs_rhp"],
        df["away_bat_xwoba_vs_lhp"],
    )
    df["batting_matchup_edge"] = df["home_bat_vs_away_sp"] - df["away_bat_vs_home_sp"]

    # Trailing 10-day matchup xwOBA — same handedness logic with recent form
    df["home_bat_vs_away_sp_10d"] = np.where(
        df["away_sp_p_throws_R"] == 1,
        df["home_bat_xwoba_vs_rhp_10d"],
        df["home_bat_xwoba_vs_lhp_10d"],
    )
    df["away_bat_vs_home_sp_10d"] = np.where(
        df["home_sp_p_throws_R"] == 1,
        df["away_bat_xwoba_vs_rhp_10d"],
        df["away_bat_xwoba_vs_lhp_10d"],
    )
    df["batting_matchup_edge_10d"] = (df["home_bat_vs_away_sp_10d"]
                                      - df["away_bat_vs_home_sp_10d"])

    # ── Rolling home/away win% (last 10 home or away games per team) ──────
    # Captures venue-specific form: how a team performs at home vs on the road
    # recently, independent of overall record.  shift(1) prevents leakage.
    if "home_score" in df.columns and "away_score" in df.columns:
        _df_srt = (df[["game_pk", "game_date", "home_team", "away_team",
                        "home_score", "away_score"]]
                   .sort_values("game_date").copy())

        _home_agg = _df_srt.assign(
            team=_df_srt["home_team"],
            win=(_df_srt["home_score"] > _df_srt["away_score"]).astype(float),
        )
        _home_agg.loc[_home_agg["home_score"].isna(), "win"] = np.nan
        _home_agg = _home_agg.sort_values(["team", "game_date"])
        _home_agg["home_wp10"] = (
            _home_agg.groupby("team")["win"]
            .transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
        )

        _away_agg = _df_srt.assign(
            team=_df_srt["away_team"],
            win=(_df_srt["away_score"] > _df_srt["home_score"]).astype(float),
        )
        _away_agg.loc[_away_agg["home_score"].isna(), "win"] = np.nan
        _away_agg = _away_agg.sort_values(["team", "game_date"])
        _away_agg["away_wp10"] = (
            _away_agg.groupby("team")["win"]
            .transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
        )

        df = (df
              .merge(
                  _home_agg[["team", "game_pk", "home_wp10"]].rename(
                      columns={"team": "home_team", "home_wp10": "home_team_home_wp10"}),
                  on=["home_team", "game_pk"], how="left")
              .merge(
                  _away_agg[["team", "game_pk", "away_wp10"]].rename(
                      columns={"team": "away_team", "away_wp10": "away_team_away_wp10"}),
                  on=["away_team", "game_pk"], how="left"))
        df["win_pct_venue_edge"] = (
            df["home_team_home_wp10"] - df["away_team_away_wp10"]
        )

    # ── Elo ratings ───────────────────────────────────────────────────────
    # Strength-of-schedule-adjusted team quality that updates every game.
    # Addresses mid-season calibration drift: a .600-pace team in August is
    # meaningfully different from a .400-pace team, which wp10 can't distinguish
    # once their current-series form coincidentally matches.
    if "home_score" in df.columns and "year" in df.columns:
        _elo_input = df[["game_pk", "game_date", "home_team", "away_team",
                          "year", "home_score", "away_score"]].copy()
        _elo_df, _elo_final = compute_elo_ratings(_elo_input)
        df = df.merge(_elo_df, on="game_pk", how="left")
        df["elo_diff"] = df["home_elo"] - df["away_elo"]

        # Persist final state so inference can load current ratings without
        # re-running the full game history each day.
        _elo_state_path = Path("data/statcast/elo_state.parquet")
        _elo_state_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"team": t, "elo": r} for t, r in _elo_final.items()]
        ).to_parquet(_elo_state_path, index=False)
        if verbose:
            print(f"      Elo: saved {len(_elo_final)} team ratings -> {_elo_state_path}")

    # ── Bullpen quality (season-level, IP-weighted) ───────────────────────
    if not bullpen.empty:
        bp_lookup = bullpen.set_index(["team", "season"])
        bp_cols   = ["bp_era", "bp_k9", "bp_bb9", "bp_hr9", "bp_whip", "bp_gb_pct"]

        def get_bullpen(team: str, season: int, prefix: str) -> dict:
            try:
                row = bp_lookup.loc[(team, season)]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
            except KeyError:
                row = pd.Series(dtype=float)
            return {f"{prefix}_{c}": row.get(c, np.nan) for c in bp_cols}

        home_bp = df.apply(
            lambda r: get_bullpen(r["home_team"], r["year"], "home"), axis=1
        ).apply(pd.Series)
        away_bp = df.apply(
            lambda r: get_bullpen(r["away_team"], r["year"], "away"), axis=1
        ).apply(pd.Series)

        df = pd.concat([df, home_bp, away_bp], axis=1)

        # Differential features (negative bp_era_diff = home bullpen advantage)
        df["bp_era_diff"]  = df["home_bp_era"]  - df["away_bp_era"]
        df["bp_k9_diff"]   = df["home_bp_k9"]   - df["away_bp_k9"]
        df["bp_whip_diff"] = df["home_bp_whip"]  - df["away_bp_whip"]

    # ── QW4: Bullpen fatigue — rolling 72h relief pitch count ─────────────
    # Positive bp_fatigue_diff = away bullpen threw more pitches recently
    # = home team faces fresher arms = home advantage on expected run scoring.
    if bp_fatigue is not None and not bp_fatigue.empty:
        fat_lookup = bp_fatigue.set_index(["team", "game_pk"])

        def _get_fatigue(team: str, game_pk) -> float:
            try:
                row = fat_lookup.loc[(team, game_pk)]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                return float(row.get("bp_pitch_count_72h", np.nan))
            except KeyError:
                return np.nan

        df["home_bp_fatigue_72h"] = df.apply(
            lambda r: _get_fatigue(r["home_team"], r["game_pk"]), axis=1
        )
        df["away_bp_fatigue_72h"] = df.apply(
            lambda r: _get_fatigue(r["away_team"], r["game_pk"]), axis=1
        )
        df["bp_fatigue_diff"] = (
            df["away_bp_fatigue_72h"] - df["home_bp_fatigue_72h"]
        )
        n_fat = df["home_bp_fatigue_72h"].notna().sum()
        if verbose:
            print(f"      Bullpen fatigue joined: {n_fat}/{len(df)} rows have data")
    else:
        if verbose:
            print("      Bullpen fatigue: no data provided — columns omitted")

    # ── Pitcher game-state features ───────────────────────────────────────
    # sp_pitches_last_start, sp_velo_diff_l14, is_opener_flag
    # Joined via (game_pk, pitching_team) — works for both home and away SPs
    # without needing a pitcher_id column in the game list.
    if pitcher_game_state is not None and not pitcher_game_state.empty:
        state_cols = ["sp_pitches_last_start", "sp_velo_diff_l14", "is_opener_flag"]

        home_pgs = (pitcher_game_state[pitcher_game_state["pitching_role"] == "home"]
                    [["game_pk", "pitching_team"] + state_cols]
                    .rename(columns={c: f"home_{c}" for c in state_cols}
                            | {"pitching_team": "home_team"}))
        away_pgs = (pitcher_game_state[pitcher_game_state["pitching_role"] == "away"]
                    [["game_pk", "pitching_team"] + state_cols]
                    .rename(columns={c: f"away_{c}" for c in state_cols}
                            | {"pitching_team": "away_team"}))

        df = df.merge(home_pgs, on=["game_pk", "home_team"], how="left")
        df = df.merge(away_pgs, on=["game_pk", "away_team"], how="left")

        if verbose:
            n_home = df["home_sp_pitches_last_start"].notna().sum()
            n_away = df["away_sp_pitches_last_start"].notna().sum()
            print(f"      Pitcher game-state: home={n_home}/{len(df)}, "
                  f"away={n_away}/{len(df)}")

    # ── Air density + roof flag + wind vector + dew point ────────────────
    if air_density is not None and not air_density.empty:
        ad = air_density.copy()
        ad["game_date"] = pd.to_datetime(ad["game_date"])
        env_cols = ["home_team", "game_date", "air_density_rho", "roof_closed_flag"]
        for extra in ("wind_vector_out", "dew_point_f", "temp_f"):
            if extra in ad.columns:
                env_cols.append(extra)
        df = df.merge(ad[env_cols], on=["home_team", "game_date"], how="left")
        if verbose:
            n_rho = df["air_density_rho"].notna().sum()
            n_roof = df["roof_closed_flag"].notna().sum()
            extras = [c for c in ("wind_vector_out", "dew_point_f", "temp_f") if c in df.columns]
            extras_s = " | ".join(f"{c}: {df[c].notna().sum()}" for c in extras)
            print(f"      Air density: {n_rho}/{len(df)} rows | "
                  f"roof_flag: {n_roof}/{len(df)} rows" +
                  (f" | {extras_s}" if extras_s else ""))

    # ── Bullpen top-3 quality-arm fatigue ────────────────────────────────
    if bp_top3_fatigue is not None and not bp_top3_fatigue.empty:
        home_t3 = bp_top3_fatigue.rename(columns={
            "team": "home_team",
            "bullpen_top3_fatigue_3d": "home_bp_top3_fatigue_3d"
        })
        away_t3 = bp_top3_fatigue.rename(columns={
            "team": "away_team",
            "bullpen_top3_fatigue_3d": "away_bp_top3_fatigue_3d"
        })
        df = df.merge(home_t3[["home_team","game_pk","home_bp_top3_fatigue_3d"]],
                      on=["home_team","game_pk"], how="left")
        df = df.merge(away_t3[["away_team","game_pk","away_bp_top3_fatigue_3d"]],
                      on=["away_team","game_pk"], how="left")
        df["bp_top3_fatigue_diff"] = (
            df.get("away_bp_top3_fatigue_3d", np.nan)
            - df.get("home_bp_top3_fatigue_3d", np.nan)
        )
        if verbose:
            n_ok = df["home_bp_top3_fatigue_3d"].notna().sum()
            print(f"      Bullpen top-3 fatigue: {n_ok}/{len(df)} home rows have data")

    # ── Umpire called-strike rate ─────────────────────────────────────────
    if ump_cs_rate is not None and not ump_cs_rate.empty:
        df = df.merge(ump_cs_rate[["game_pk", "ump_called_strike_above_avg"]],
                      on="game_pk", how="left")
        df["ump_called_strike_above_avg"] = (
            df["ump_called_strike_above_avg"].fillna(0)   # debut ump = league average
        )
        if verbose:
            n_ok = (df["ump_called_strike_above_avg"] != 0).sum()
            print(f"      Ump called-strike: {n_ok}/{len(df)} non-zero rows")

    # ── Lineup handedness matchup ─────────────────────────────────────────
    if lineup_handedness is not None and not lineup_handedness.empty:
        lh_cols = ["game_pk", "home_lineup_opp_hand_count",
                   "away_lineup_opp_hand_count", "lineup_opp_hand_diff"]
        df = df.merge(lineup_handedness[lh_cols], on="game_pk", how="left")
        if verbose:
            n_ok = df["home_lineup_opp_hand_count"].notna().sum()
            mean_diff = df["lineup_opp_hand_diff"].mean()
            print(f"      Lineup handedness: {n_ok}/{len(df)} rows | "
                  f"mean_opp_hand_diff={mean_diff:.2f}")

    # ── SP K-prop market signal (DraftKings consensus) ────────────────────
    if kprop_features is not None and not kprop_features.empty:
        kp = kprop_features.copy()
        kp["game_date"] = pd.to_datetime(kp["game_date"])

        # Normalize starter names from games for matching
        df["_home_sp_norm"] = df["home_starter_name"].apply(_normalize_pitcher_name)
        df["_away_sp_norm"] = df["away_starter_name"].apply(_normalize_pitcher_name)

        # Home SP join: pitcher_norm == home_sp_norm
        kp_home = kp.rename(columns={
            "pitcher_norm": "_home_sp_norm",
            "sp_k_line": "home_sp_k_line",
            "sp_k_over_prob": "home_sp_k_line_implied",
        })[["game_date", "home_team", "away_team", "_home_sp_norm",
            "home_sp_k_line", "home_sp_k_line_implied"]]
        df = df.merge(kp_home,
                      on=["game_date", "home_team", "away_team", "_home_sp_norm"],
                      how="left")

        # Away SP join
        kp_away = kp.rename(columns={
            "pitcher_norm": "_away_sp_norm",
            "sp_k_line": "away_sp_k_line",
            "sp_k_over_prob": "away_sp_k_line_implied",
        })[["game_date", "home_team", "away_team", "_away_sp_norm",
            "away_sp_k_line", "away_sp_k_line_implied"]]
        df = df.merge(kp_away,
                      on=["game_date", "home_team", "away_team", "_away_sp_norm"],
                      how="left")

        df["sp_k_line_diff"] = df["home_sp_k_line"] - df["away_sp_k_line"]
        df = df.drop(columns=["_home_sp_norm", "_away_sp_norm"])

        if verbose:
            n_home = df["home_sp_k_line"].notna().sum()
            n_away = df["away_sp_k_line"].notna().sum()
            print(f"      K-prop market signal: home={n_home}, away={n_away} of {len(df)} rows")
    else:
        for c in ("home_sp_k_line", "away_sp_k_line", "sp_k_line_diff",
                  "home_sp_k_line_implied", "away_sp_k_line_implied"):
            df[c] = np.nan

    # ── Phase 2: Team bat tracking + batter run values ────────────────────
    # Seasonal aggregate (team, season) joined like bullpen data.
    # bat_speed_weighted, blast_rate, squared_up_pct  (2023+; NaN earlier)
    # bat_swing_rv, bat_take_rv  (batter swing/take discipline)
    if bat_tracking is not None and not bat_tracking.empty:
        bt_lookup = bat_tracking.set_index(["team", "season"])
        bt_cols   = [c for c in bat_tracking.columns if c not in ("team", "season")]

        def get_bat_tracking(team: str, season: int, prefix: str) -> dict:
            try:
                row = bt_lookup.loc[(team, season)]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
            except KeyError:
                row = pd.Series(dtype=float)
            return {f"{prefix}_{c}": row.get(c, np.nan) for c in bt_cols}

        home_bt = df.apply(
            lambda r: get_bat_tracking(r["home_team"], r["year"], "home"), axis=1
        ).apply(pd.Series)
        away_bt = df.apply(
            lambda r: get_bat_tracking(r["away_team"], r["year"], "away"), axis=1
        ).apply(pd.Series)
        df = pd.concat([df, home_bt, away_bt], axis=1)
        for col in bt_cols:
            h_col, a_col = f"home_{col}", f"away_{col}"
            if h_col in df.columns and a_col in df.columns:
                df[f"{col}_diff"] = df[h_col] - df[a_col]
        if verbose:
            sample_col = f"home_{bt_cols[0]}" if bt_cols else None
            n_bt = df[sample_col].notna().sum() if sample_col and sample_col in df.columns else 0
            print(f"      Bat tracking joined: {n_bt}/{len(df)} rows have data")
    else:
        if verbose:
            print("      Bat tracking: no data provided — columns omitted")

    # ── Phase 2: Team defense (OAA + FRV) ────────────────────────────────
    if defense is not None and not defense.empty:
        def_lookup = defense.set_index(["team", "season"])
        def_cols   = [c for c in defense.columns if c not in ("team", "season")]

        def get_defense(team: str, season: int, prefix: str) -> dict:
            try:
                row = def_lookup.loc[(team, season)]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
            except KeyError:
                row = pd.Series(dtype=float)
            return {f"{prefix}_{c}": row.get(c, np.nan) for c in def_cols}

        home_def = df.apply(
            lambda r: get_defense(r["home_team"], r["year"], "home"), axis=1
        ).apply(pd.Series)
        away_def = df.apply(
            lambda r: get_defense(r["away_team"], r["year"], "away"), axis=1
        ).apply(pd.Series)
        df = pd.concat([df, home_def, away_def], axis=1)
        for col in def_cols:
            h_col, a_col = f"home_{col}", f"away_{col}"
            if h_col in df.columns and a_col in df.columns:
                df[f"{col}_diff"] = df[h_col] - df[a_col]
        if verbose:
            sample_col = f"home_{def_cols[0]}" if def_cols else None
            n_def = df[sample_col].notna().sum() if sample_col and sample_col in df.columns else 0
            print(f"      Defense features joined: {n_def}/{len(df)} rows have data")
    else:
        if verbose:
            print("      Defense: no data provided — columns omitted")

    # ── Team archetype clustering ─────────────────────────────────────────
    # Now includes bat_gb_pct, bat_fb_pct (statcast bb_type), bp_gb_pct
    # (groundOuts/airOuts from FanGraphs reliever data), giving the KMeans
    # clusters genuine orthogonal signal (batted-ball tendencies) beyond the
    # K% and BB% that XGBoost already reads directly.
    df = append_team_archetypes(df, train_years=[2023, 2024, 2025],
                                verbose=verbose)

    # ── Park factors ──────────────────────────────────────────────────────
    # Static 3-year average park run factor keyed on home_team abbreviation.
    # 1.00 = league-neutral; COL = 1.28 (Coors), SD = 0.96 (Petco).
    # Gives XGBoost direct ballpark run-environment signal without double-
    # counting the MC engine's physics model.
    league_avg_park = sum(PARK_FACTORS.values()) / len(PARK_FACTORS)
    df["home_park_factor"] = (
        df["home_team"].map(PARK_FACTORS).fillna(league_avg_park)
    )

    # ── Umpire tendency features ───────────────────────────────────────────
    # HP umpire K%/BB%/runs-per-game tendency vs league avg (EWMA, shift=1).
    # Built by build_ump_stats.py from umpire_assignments_{year}.parquet.
    # Gracefully no-ops if files are missing (NaN fill).
    years_in_df = df["year"].dropna().astype(int).unique().tolist()
    ump_frames = []
    for yr in years_in_df:
        ump_path = DATA_DIR / f"ump_features_{yr}.parquet"
        if ump_path.exists():
            uf = pd.read_parquet(ump_path, engine="pyarrow")
            uf["game_pk"] = pd.to_numeric(uf["game_pk"], errors="coerce")
            ump_frames.append(uf)
    if ump_frames:
        ump_all = pd.concat(ump_frames, ignore_index=True)
        ump_all = ump_all.drop_duplicates(subset=["game_pk"], keep="last")
        ump_cols = ["game_pk", "ump_k_above_avg", "ump_bb_above_avg", "ump_rpg_above_avg"]
        ump_cols = [c for c in ump_cols if c in ump_all.columns]
        df["game_pk_num"] = pd.to_numeric(df.get("game_pk", pd.Series(dtype=float)), errors="coerce")
        df = df.merge(ump_all[ump_cols].rename(columns={"game_pk": "game_pk_num"}),
                      on="game_pk_num", how="left")
        df = df.drop(columns=["game_pk_num"])
        n_ump = df["ump_k_above_avg"].notna().sum() if "ump_k_above_avg" in df.columns else 0
        if verbose:
            print(f"      Ump features joined: {n_ump}/{len(df)} rows ({100*n_ump/len(df):.1f}%)")
    else:
        if verbose:
            print("      Ump features: no ump_features_*.parquet found — columns omitted")

    # ── Market lines (The Odds API historical) ────────────────────────────
    # Closing O/U total + moneyline implied probabilities from Pinnacle/DK/FD.
    # Built by pull_odds_history_api.py.  Gracefully no-ops if files missing.
    odds_frames = []
    for yr in years_in_df:
        op = DATA_DIR / f"odds_api_hist_{yr}.parquet"
        if op.exists():
            of = pd.read_parquet(op, engine="pyarrow")
            of["game_date"] = pd.to_datetime(of["game_date"])
            odds_frames.append(of)
        else:
            # Fallback: consolidate daily odds_history_{yr}_*.parquet snapshots
            # produced by pull_odds_history_api.py when the annual file is not yet built.
            daily_paths = sorted(DATA_DIR.glob(f"odds_history_{yr}_*.parquet"))
            if daily_paths:
                _daily = []
                for dp in daily_paths:
                    try:
                        _df = pd.read_parquet(dp, engine="pyarrow")
                        _df["game_date"] = pd.to_datetime(_df["game_date"])
                        _daily.append(_df)
                    except Exception:
                        pass
                if _daily:
                    _combined = pd.concat(_daily, ignore_index=True)
                    # Daily files use P_true_home/P_true_away; rename to match schema
                    _combined = _combined.rename(columns={
                        "P_true_home": "true_home_prob",
                        "P_true_away": "true_away_prob",
                    })
                    # Keep latest snapshot per game before the outer dedup pass
                    if "snapshot_time" in _combined.columns:
                        _combined = _combined.sort_values("snapshot_time")
                    odds_frames.append(_combined)
                    if verbose:
                        print(f"      Market lines fallback ({yr}): "
                              f"{len(daily_paths)} daily files → {len(_combined)} rows "
                              f"(run pull_odds_history_api.py --year {yr} to build annual file)")
    if odds_frames:
        odds_all = pd.concat(odds_frames, ignore_index=True)
        odds_all = odds_all.drop_duplicates(subset=["game_date", "home_team", "away_team"], keep="last")
        odds_cols = ["game_date", "home_team", "away_team",
                     "close_total", "close_ml_home", "close_ml_away",
                     "true_home_prob", "true_away_prob", "game_hour_et"]
        odds_cols = [c for c in odds_cols if c in odds_all.columns]
        df = df.merge(odds_all[odds_cols], on=["game_date", "home_team", "away_team"], how="left")
        n_tot = df["close_total"].notna().sum() if "close_total" in df.columns else 0
        if verbose:
            print(f"      Market lines joined: {n_tot}/{len(df)} rows ({100*n_tot/len(df):.1f}%) with totals")
    else:
        if verbose:
            print("      Market lines: no odds_api_hist_*.parquet found — columns omitted")

    # ── Circadian / game-time features ────────────────────────────────────
    # Research shows west-coast teams playing early ET starts (e.g. noon)
    # face meaningful circadian disruption (body clock says 9am).
    # Features:
    #   game_hour_et           — raw ET start hour (from commence_time in odds cache)
    #   is_day_game            — 1 if start < 5pm ET (affects bullpen rest, crowd)
    #   away_game_local_hour   — what time the away team's body clock reads
    #   circadian_edge         — home advantage from timezone mismatch
    if "game_hour_et" in df.columns:
        _away_offset = df["away_team"].map(TEAM_ET_OFFSET).fillna(0)
        _home_offset = df["home_team"].map(TEAM_ET_OFFSET).fillna(0)
        df["is_day_game"]          = (df["game_hour_et"] < 17).astype("Int8")
        df["away_game_local_hour"] = df["game_hour_et"] + _away_offset
        df["home_game_local_hour"] = df["game_hour_et"] + _home_offset
        # circadian_edge: positive = home team is more "awake" relative to away
        df["circadian_edge"]       = df["home_game_local_hour"] - df["away_game_local_hour"]
        # Null out derived cols where game_hour_et is missing
        for _col in ["is_day_game", "away_game_local_hour",
                     "home_game_local_hour", "circadian_edge"]:
            df.loc[df["game_hour_et"].isna(), _col] = pd.NA
        n_hour = df["game_hour_et"].notna().sum()
        if verbose:
            print(f"      Circadian features: {n_hour}/{len(df)} rows have game_hour_et "
                  f"({100*n_hour/len(df):.1f}%)")

    # =========================================================================
    # PHASE 2 — ENGINEERED INTERACTIONS & RATIO FEATURES
    # =========================================================================
    # GBDTs learn interactions implicitly through tree depth, but explicit
    # encoding is worthwhile when:
    #   (a) domain logic is genuinely multiplicative (trees approximate A×B
    #       as a staircase, requiring many splits and overfitting risk), or
    #   (b) cross-table interactions that the model can't discover without
    #       them being in the same column.
    # With ~4,500 labeled games, each explicit interaction saves ~3-5 splits
    # of data budget, meaningfully improving out-of-sample generalisation.
    # -------------------------------------------------------------------------

    # ── TIER 1A: K/BB ratio (command quality compressed into one number) ──
    # K% and BB% are already in the matrix; K/BB as a ratio captures the
    # non-linear "elite command" region more efficiently than the raw pair.
    _eps = 0.01   # prevent division by zero
    df["home_sp_k_bb_ratio"] = df["home_sp_k_pct"] / (df["home_sp_bb_pct"] + _eps)
    df["away_sp_k_bb_ratio"] = df["away_sp_k_pct"] / (df["away_sp_bb_pct"] + _eps)
    df["sp_k_bb_ratio_diff"] = df["home_sp_k_bb_ratio"] - df["away_sp_k_bb_ratio"]

    # Trailing 10-day K/BB ratio (recent form)
    df["home_sp_k_bb_ratio_10d"] = df["home_sp_k_pct_10d"] / (df["home_sp_bb_pct_10d"] + _eps)
    df["away_sp_k_bb_ratio_10d"] = df["away_sp_k_pct_10d"] / (df["away_sp_bb_pct_10d"] + _eps)
    df["sp_k_bb_ratio_10d_diff"] = df["home_sp_k_bb_ratio_10d"] - df["away_sp_k_bb_ratio_10d"]

    # ── TIER 1B: Pitcher quality × opposing lineup quality interaction ─────
    # The model already sees pitcher xwoba_against and lineup xwoba separately.
    # Their product captures the compounding effect: an elite pitcher facing a
    # weak lineup is a much stronger signal than either feature alone.
    df["home_sp_vs_lineup_quality"] = (
        df["home_sp_xwoba_against"] * df["away_bat_vs_home_sp"]
    )
    df["away_sp_vs_lineup_quality"] = (
        df["away_sp_xwoba_against"] * df["home_bat_vs_away_sp"]
    )
    df["sp_lineup_quality_diff"] = (
        df["home_sp_vs_lineup_quality"] - df["away_sp_vs_lineup_quality"]
    )

    # 10-day version — recent form × recent opponent quality
    if "home_sp_xwoba_10d" in df.columns and "home_bat_vs_away_sp_10d" in df.columns:
        df["home_sp_vs_lineup_10d"] = (
            df["home_sp_xwoba_10d"] * df["away_bat_vs_home_sp_10d"]
        )
        df["away_sp_vs_lineup_10d"] = (
            df["away_sp_xwoba_10d"] * df["home_bat_vs_away_sp_10d"]
        )
        df["sp_lineup_quality_10d_diff"] = (
            df["home_sp_vs_lineup_10d"] - df["away_sp_vs_lineup_10d"]
        )

    # ── TIER 1C: Fatigue × velocity signal ────────────────────────────────
    # IL return flag indicates first 1-3 starts after a long absence — pitchers
    # are often stronger early but can also be rusty. Multiplying by velo
    # gives the model a fused "is this a weakened start?" signal.
    _il_penalty = 0.015   # 1.5% velo discount per IL return
    df["home_sp_fatigue_signal"] = (
        df["home_sp_ff_velo"] * (1.0 - _il_penalty * df["home_sp_il_return_flag"].fillna(0))
    )
    df["away_sp_fatigue_signal"] = (
        df["away_sp_ff_velo"] * (1.0 - _il_penalty * df["away_sp_il_return_flag"].fillna(0))
    )
    df["sp_fatigue_diff"] = df["home_sp_fatigue_signal"] - df["away_sp_fatigue_signal"]

    # ── TIER 1D: Umpire × command interaction ─────────────────────────────
    # A tight umpire (large strike zone) amplifies the value of a command
    # pitcher (high K-BB%). Captures the asymmetric umpire effect between
    # the two starters when one has a significant command edge.
    if "ump_k_above_avg" in df.columns:
        df["ump_command_edge_home"] = (
            df["ump_k_above_avg"] * df["home_sp_k_minus_bb"]
        )
        df["ump_command_edge_away"] = (
            df["ump_k_above_avg"] * df["away_sp_k_minus_bb"]
        )
        df["ump_command_net_edge"] = (
            df["ump_command_edge_home"] - df["ump_command_edge_away"]
        )

    # ── TIER 1E: Arsenal put-away quality ratio ────────────────────────────
    # Ratio of weighted arsenal run value to primary whiff%. Separates pitchers
    # who get whiffs on their best pitch (dominant) from those whose run value
    # comes from contact management (crafty). Different predictive profile.
    if "home_sp_arsenal_weighted_rv" in df.columns and "home_sp_primary_whiff_pct" in df.columns:
        df["home_sp_arsenal_quality_ratio"] = (
            df["home_sp_arsenal_weighted_rv"] / (df["home_sp_primary_whiff_pct"] + _eps)
        )
        df["away_sp_arsenal_quality_ratio"] = (
            df["away_sp_arsenal_weighted_rv"] / (df["away_sp_primary_whiff_pct"] + _eps)
        )
        df["sp_arsenal_quality_ratio_diff"] = (
            df["home_sp_arsenal_quality_ratio"] - df["away_sp_arsenal_quality_ratio"]
        )

    # ── TIER 2A: Swing/take discipline mismatch ───────────────────────────
    # A pitcher who generates high run value on takes (commands the zone)
    # facing a patient lineup (high bat_take_rv) is a specific matchup tension
    # the model would need many splits to discover independently.
    if "home_sp_take_rv_per100" in df.columns and "away_bat_take_rv" in df.columns:
        df["home_take_mismatch"] = (
            df["home_sp_take_rv_per100"] * df["away_bat_take_rv"]
        )
        df["away_take_mismatch"] = (
            df["away_sp_take_rv_per100"] * df["home_bat_take_rv"]
        )
        df["take_mismatch_diff"] = df["home_take_mismatch"] - df["away_take_mismatch"]

    if "home_sp_swing_rv_per100" in df.columns and "away_bat_swing_rv" in df.columns:
        df["home_swing_mismatch"] = (
            df["home_sp_swing_rv_per100"] * df["away_bat_swing_rv"]
        )
        df["away_swing_mismatch"] = (
            df["away_sp_swing_rv_per100"] * df["home_bat_swing_rv"]
        )
        df["swing_mismatch_diff"] = df["home_swing_mismatch"] - df["away_swing_mismatch"]

    # ── TIER 2B: Put-away pitch × opposing bat speed ───────────────────────
    # A pitcher's put-away pitch is more effective against lineups with slower
    # bat speed. Encodes the physics: late-breaking pitches need reaction time
    # that bat speed determines. Key for K-prop and strikeout total models.
    if "home_sp_primary_putaway_pct" in df.columns and "away_bat_speed_weighted" in df.columns:
        df["home_putaway_vs_bat_speed"] = (
            df["home_sp_primary_putaway_pct"] * (1.0 / (df["away_bat_speed_weighted"] + 1.0))
        )
        df["away_putaway_vs_bat_speed"] = (
            df["away_sp_primary_putaway_pct"] * (1.0 / (df["home_bat_speed_weighted"] + 1.0))
        )
        df["putaway_bat_speed_diff"] = (
            df["home_putaway_vs_bat_speed"] - df["away_putaway_vs_bat_speed"]
        )

    # ── TIER 2C: Ground-ball pitcher × infield defense quality ────────────
    # A high-GB pitcher is worth more behind a good infield (high OAA).
    # The interaction captures that GB% alone understates value for pitchers
    # with elite defense behind them, and overstates it with poor defense.
    if "home_team_oaa" in df.columns:
        df["home_gb_defense_synergy"] = (
            df["home_sp_gb_pct"] * df["home_team_oaa"].clip(lower=0)
        )
        df["away_gb_defense_synergy"] = (
            df["away_sp_gb_pct"] * df["away_team_oaa"].clip(lower=0)
        )
        df["gb_defense_synergy_diff"] = (
            df["home_gb_defense_synergy"] - df["away_gb_defense_synergy"]
        )

    # ── TIER 2D: Blast rate × arsenal run value (power vs. stuff matchup) ─
    # A powerful lineup (high blast_rate) facing a dominant arsenal (high
    # arsenal_weighted_rv) is a key over/under signal. Encoding the product
    # compresses the "can they square it up?" question into one feature.
    if "home_bat_blast_rate" in df.columns and "away_sp_arsenal_weighted_rv" in df.columns:
        df["home_power_vs_stuff"] = (
            df["home_bat_blast_rate"] * df["away_sp_arsenal_weighted_rv"]
        )
        df["away_power_vs_stuff"] = (
            df["away_bat_blast_rate"] * df["home_sp_arsenal_weighted_rv"]
        )
        df["power_vs_stuff_diff"] = df["home_power_vs_stuff"] - df["away_power_vs_stuff"]

    # ── TIER 2E: Bullpen vulnerability × game state (late leverage) ───────
    # High-ERA bullpen × team run production variance: volatile teams facing
    # a bad bullpen have higher upside late. Useful for full-game totals.
    if "home_bp_era" in df.columns:
        df["home_bullpen_vulnerability"] = (
            df["home_bp_era"] * df["away_bat_xwoba_vs_rhp"]
        )
        df["away_bullpen_vulnerability"] = (
            df["away_bp_era"] * df["home_bat_xwoba_vs_rhp"]
        )
        df["bullpen_vulnerability_diff"] = (
            df["home_bullpen_vulnerability"] - df["away_bullpen_vulnerability"]
        )

    # ── TIER 2F: Circadian edge × aging pitcher ────────────────────────────
    # Older pitchers (age > 32) are more affected by time-zone travel fatigue.
    # Encodes the non-linear age × circadian interaction from the feature list.
    if "circadian_edge" in df.columns and "home_sp_age_pit" in df.columns:
        _age_scale = 32.0   # reference age; above this, circadian impact grows
        df["home_sp_circadian_age_adj"] = (
            df["circadian_edge"] * np.maximum(df["home_sp_age_pit"] - _age_scale, 0)
        )
        df["away_sp_circadian_age_adj"] = (
            -df["circadian_edge"] * np.maximum(df["away_sp_age_pit"] - _age_scale, 0)
        )
        df["circadian_age_net"] = (
            df["home_sp_circadian_age_adj"] - df["away_sp_circadian_age_adj"]
        )

    # ── WINSORIZE all continuous features at [1st, 99th] percentile ───────
    # Must happen AFTER train/val split column exists (used to anchor bounds).
    # The split column is added below in the labels section; to avoid a
    # circular dependency we add a temporary indicator here.
    _has_split = "split" in df.columns
    if not _has_split:
        df["split"] = np.where(df["year"].isin([2023, 2024]), "train", "val")
    df = _winsorize_features(df, verbose=verbose)
    if not _has_split:
        df = df.drop(columns=["split"])

    # ── Monte Carlo residual placeholder ──────────────────────────────────
    # Populated at inference time by run_today.py before XGBoost scoring.
    # XGBoost routes NaN rows via the learned default branch (no crash).
    df["mc_expected_runs"] = np.nan

    # ── Labels ────────────────────────────────────────────────────────────
    df["home_score"]     = _to_num(df.get("home_score", pd.Series(dtype=float)))
    df["away_score"]     = _to_num(df.get("away_score", pd.Series(dtype=float)))
    df["home_margin"]    = df["home_score"] - df["away_score"]
    df["home_covers_rl"] = (df["home_margin"] >= 2).astype("Int8")
    df["away_covers_rl"] = (df["home_margin"] <= 1).astype("Int8")
    df["total_runs"]     = df["home_score"] + df["away_score"]
    df["actual_home_win"] = (df["home_margin"] > 0).astype("Int8")

    df.loc[df["home_margin"].isna(),
           ["home_covers_rl", "away_covers_rl", "actual_home_win"]] = pd.NA

    # ── Drop spring training / pre-season games ───────────────────────────
    # Spring training games have no odds, no labels, and inflate null-rate
    # stats.  Keep only games that are either (a) labeled (score known) OR
    # (b) in the current / future season where we can't have labels yet
    # but the game_date is on or after April 1 (regular-season start).
    before_filter = len(df)
    regular_season_start = pd.to_datetime(
        df["year"].astype(str) + "-04-01"
    )
    keep_mask = df["home_covers_rl"].notna() | (df["game_date"] >= regular_season_start)
    df = df[keep_mask].reset_index(drop=True)
    dropped = before_filter - len(df)
    if verbose and dropped:
        print(f"      Dropped {dropped} spring-training / pre-season rows "
              f"(unlabeled before April 1)")

    # ── Train / val split ─────────────────────────────────────────────────
    df["split"] = np.where(df["year"].isin([2023, 2024]), "train", "val")

    if verbose:
        n_train   = (df["split"] == "train").sum()
        n_val     = (df["split"] == "val").sum()
        n_labeled = df["home_covers_rl"].notna().sum()
        rl_rate   = float(df["home_covers_rl"].mean()) if n_labeled > 0 else float("nan")
        print(f"      {len(df)} total games | train: {n_train} | val: {n_val}")
        print(f"      {n_labeled} labeled | home covers RL rate: {rl_rate:.3f}")

    return df


# ---------------------------------------------------------------------------
# SECTION 6 — TEAM ARCHETYPE CLUSTERING  (Feature Space Stratification v5.1)
# ---------------------------------------------------------------------------

def append_team_archetypes(
    df: pd.DataFrame,
    train_years: list[int] | None = None,
    n_off: int = N_ARCH_CLUSTERS_OFF,
    n_bp:  int = N_ARCH_CLUSTERS_BP,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Cluster teams into offensive and bullpen archetypes and append 4 new
    cluster-ID columns to the feature matrix:

        home_off_cluster  away_off_cluster
        home_bp_cluster   away_bp_cluster

    Values are Int8 integers in [0, n_clusters).  A value of -1 indicates
    that no style features were available for that row.

    ── Future-Leak Prevention ──────────────────────────────────────────────
    Two independent guards ensure zero future information contaminates the
    cluster assignments:

    Guard 1 — EWMA temporal boundary (upstream):
        Every batting and bullpen feature feeding the clusterer was computed
        with a .shift(1) applied per-team before the game-level join.  Each
        game's style snapshot therefore reflects ONLY performance prior to
        that game.  This guard is enforced in build_team_batting_ewma() and
        build_bullpen_stats(); it is not re-implemented here.

    Guard 2 — Full-corpus centroid fitting (this function):
        StandardScaler and KMeans are fit on all available years
        (default: 2023 + 2024 + 2025) so that centroid geometry is stable
        and representative across the full timeline.  Style features
        (K%, BB/9, etc.) contain no outcome information, so including 2025
        rows in centroid fitting is not target leakage — it is analogous to
        fitting a scaler on the full corpus before a train/val split.
        The alternative (train-only centroids) causes centroid drift: a team
        archetype that barely existed in 2023+2024 (e.g. bullpen C3) gets
        mis-assigned in 2025 val, degrading feature quality.

    ── GPU Execution (RTX 5080) ────────────────────────────────────────────
    When RAPIDS cuML is available (_CUML_ARCH=True):
      - Feature arrays are moved to GPU via cupy.asarray().
      - cuml.preprocessing.StandardScaler fits and transforms on-device.
      - cuml.cluster.KMeans fits and predicts on-device.
      - Cluster label arrays are returned to CPU via .get() for the join.
    Falls back to sklearn transparently on CPU when RAPIDS is absent.
    """
    if train_years is None:
        train_years = [2023, 2024]

    df = df.copy()
    year_arr        = df["year"].fillna(0).astype(int).values
    train_mask_bool = np.isin(year_arr, train_years)   # shape (n_games,)

    def _run_clustering(
        feat_bases: list[str],
        n_clusters: int,
        label: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit KMeans on train-year appearances; predict on all appearances.

        Strategy: stack home and away rows into a single long table of shape
        (2 * n_games, n_feats).  The first n_games rows are home appearances;
        the last n_games rows are away appearances.  This lets a single
        KMeans model learn the full range of team styles from both sides of
        each game, then cleanly separates home vs. away assignments by index.

        Returns (home_labels, away_labels) — int8 numpy arrays of len(df),
        each value in [0, n_clusters) or -1 if all features were missing.
        """
        null_arr = np.full(len(df), -1, dtype="int8")

        # ── Resolve which features are actually in the matrix ─────────────
        home_prefixed = [f"home_{c}" for c in feat_bases if f"home_{c}" in df.columns]
        away_prefixed = [f"away_{c}" for c in feat_bases if f"away_{c}" in df.columns]

        # Require matching home/away columns — drop any unpaired feature
        avail_bases = [
            c for c in feat_bases
            if f"home_{c}" in df.columns and f"away_{c}" in df.columns
        ]
        if not avail_bases:
            if verbose:
                print(f"      [ARCH] {label}: no matching home+away features — "
                      f"assigning -1")
            return null_arr, null_arr

        home_cols = [f"home_{c}" for c in avail_bases]
        away_cols = [f"away_{c}" for c in avail_bases]
        n_feats   = len(avail_bases)

        # ── Build long array: [home_rows ; away_rows] ─────────────────────
        # shape: (2 * n_games, n_feats)  dtype: float32
        home_X = df[home_cols].values.astype("float32")
        away_X = df[away_cols].values.astype("float32")
        X_long = np.concatenate([home_X, away_X], axis=0)

        # Replicate the train mask across both halves
        train_long = np.concatenate([train_mask_bool, train_mask_bool])

        # ── Impute NaN with column medians computed on train rows only ────
        # Median imputation is applied before scaling; medians are computed
        # from the training split to prevent val-set contamination.
        train_X_for_median = X_long[train_long]
        col_medians = np.nanmedian(train_X_for_median, axis=0)
        for j in range(n_feats):
            nan_idx = np.isnan(X_long[:, j])
            if nan_idx.any():
                X_long[nan_idx, j] = col_medians[j]

        # ── GPU path: cuML StandardScaler + KMeans ────────────────────────
        if _CUML_ARCH and _cp_arch is not None:
            X_gpu        = _cp_arch.asarray(X_long)          # host → device
            train_idx    = _cp_arch.asarray(train_long)
            X_train_gpu  = X_gpu[train_idx]

            scaler = _ArchScaler()
            scaler.fit(X_train_gpu)
            X_scaled = scaler.transform(X_gpu)               # full (2n, d)

            try:
                km = _ArchKMeans(n_clusters=n_clusters, random_state=42,
                                 max_iter=300, n_init=10)
            except TypeError:
                # Older cuML versions don't expose n_init
                km = _ArchKMeans(n_clusters=n_clusters, random_state=42,
                                 max_iter=300)

            km.fit(X_scaled[train_idx])
            labels_np = km.predict(X_scaled).get().astype("int8")  # device → host
            backend   = f"cuML (GPU)  n_feats={n_feats}"

        # ── CPU fallback: sklearn StandardScaler + KMeans ─────────────────
        else:
            X_train = X_long[train_long]

            scaler   = _ArchScaler()
            scaler.fit(X_train)
            X_scaled = scaler.transform(X_long)

            km = _ArchKMeans(n_clusters=n_clusters, random_state=42,
                             max_iter=300, n_init=10)
            km.fit(X_scaled[train_long])
            labels_np = km.predict(X_scaled).astype("int8")
            backend   = f"sklearn (CPU)  n_feats={n_feats}"

        # ── Split long labels back into home / away ────────────────────────
        n_games     = len(df)
        home_labels = labels_np[:n_games]
        away_labels = labels_np[n_games:]

        if verbose:
            vc   = pd.Series(labels_np).value_counts().sort_index()
            dist = "  ".join(f"C{i}:{cnt}" for i, cnt in vc.items())
            print(f"      [ARCH] {label}: k={n_clusters} | {backend} | {dist}")

        return home_labels, away_labels

    # ── Offensive archetypes ──────────────────────────────────────────────
    h_off, a_off = _run_clustering(_ARCH_OFF_FEATS, n_off, "Offensive")
    df["home_off_cluster"] = pd.array(h_off, dtype="Int8")
    df["away_off_cluster"] = pd.array(a_off, dtype="Int8")

    # ── Bullpen / pitching archetypes ─────────────────────────────────────
    h_bp, a_bp = _run_clustering(_ARCH_BP_FEATS, n_bp, "Bullpen")
    df["home_bp_cluster"] = pd.array(h_bp, dtype="Int8")
    df["away_bp_cluster"] = pd.array(a_bp, dtype="Int8")

    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build EWMA feature matrix for XGBoost (no park/weather/Vegas)")
    parser.add_argument("--years", type=int, nargs="+",
                        default=[2023, 2024, 2025],
                        help="Years to include (2023=train, 2024=train, 2025=val)")
    parser.add_argument("--out", type=str, default="feature_matrix",
                        help="Output filename stem (no extension)")
    args = parser.parse_args()

    years = args.years

    print("=" * 60)
    print("  build_feature_matrix.py  (time-EWMA + bullpen + trailing-10d)")
    print(f"  Years: {years}  |  SP halflife: {EWMA_HALFLIFE_SP_DAYS}d  "
          f"|  Bat halflife: {EWMA_HALFLIFE_BAT_DAYS}d  "
          f"|  Trailing: {TRAILING_DAYS}d")
    print("=" * 60)

    games             = build_game_list(years)
    pitchers          = build_pitcher_ewma_stats(years)
    batting           = build_team_batting_ewma(years)
    bullpen           = build_bullpen_stats(years)
    bat_tracking      = build_team_bat_tracking(years)
    defense           = build_team_defense(years)
    bp_fatigue        = build_bullpen_fatigue(years)
    pitcher_game_state = build_pitcher_game_state(years)
    air_density       = build_air_density_features(years)
    bp_top3_fatigue   = build_bullpen_top3_fatigue(years)
    ump_cs_rate       = build_ump_called_strike_rate(years)
    lineup_handedness = build_lineup_handedness(years)
    kprop_features    = build_kprop_features(years)

    print()
    matrix = assemble_matrix(games, pitchers, batting, bullpen,
                             bat_tracking=bat_tracking, defense=defense,
                             bp_fatigue=bp_fatigue,
                             pitcher_game_state=pitcher_game_state,
                             air_density=air_density,
                             bp_top3_fatigue=bp_top3_fatigue,
                             ump_cs_rate=ump_cs_rate,
                             lineup_handedness=lineup_handedness,
                             kprop_features=kprop_features)

    # ── Environmental calendar + interactions (v2 feature set) ──────────
    # These columns are additive: the currently-trained cat_total.pkl /
    # lgbm_total.pkl / dc_model_run_dist.pkl use run_dist_feature_cols.json
    # (unchanged) and never see them. They land in the feature matrix for
    # the next training cycle, which will read run_dist_feature_cols_next.json.
    from models.feature_engineering import (
        compute_days_since_opening_day,
        compute_league_rpg_rolling_7d,
        load_league_rpg_history,
        create_interaction_features,
    )

    matrix["days_since_opening_day"] = matrix["game_date"].apply(
        compute_days_since_opening_day
    )

    league_hist = load_league_rpg_history()
    unique_dates = sorted(matrix["game_date"].dropna().unique())
    rpg_map = {
        d: compute_league_rpg_rolling_7d(pd.Timestamp(d).date(), league_hist)
        for d in unique_dates
    }
    matrix["league_rpg_rolling_7d"] = matrix["game_date"].map(rpg_map)

    matrix = create_interaction_features(matrix)
    print(f"  Env features added: days_since_opening_day, league_rpg_rolling_7d, "
          f"aero_impact, sp_environment_vulnerability[±diff], sp_thermal_aging[±diff]")

    out_parquet = f"{args.out}.parquet"
    out_csv     = f"{args.out}.csv"
    matrix.to_parquet(out_parquet, engine="pyarrow", index=False)
    matrix.to_csv(out_csv, index=False)

    print()
    print(f"  Saved -> {out_parquet}  ({len(matrix)} rows x {len(matrix.columns)} cols)")
    print(f"  Saved -> {out_csv}")

    print()
    print("=" * 60)
    print("  Feature matrix summary")
    print("=" * 60)

    non_feat = {
        "game_date", "game_pk", "home_team", "away_team", "season",
        "home_starter_name", "away_starter_name",
        "home_score", "away_score", "actual_home_win",
        "home_margin", "home_covers_rl", "away_covers_rl",
        "total_runs", "split", "year",
    }
    feature_cols = [c for c in matrix.columns if c not in non_feat]
    print(f"  Feature columns: {len(feature_cols)}")

    null_pct = (matrix[feature_cols].isna().mean() * 100).sort_values(ascending=False)
    print("\n  Top-10 null % features:")
    for col, pct in null_pct.head(10).items():
        marker = "  (MC placeholder — NaN during training)" \
                 if col == "mc_expected_runs" else ""
        print(f"    {col:<45} {pct:.1f}%{marker}")

    print()
    for split_name in ["train", "val"]:
        sdf = matrix[matrix["split"] == split_name]
        n   = len(sdf)
        rl  = float(sdf["home_covers_rl"].mean()) if n > 0 else float("nan")
        tot = float(sdf["total_runs"].mean())     if n > 0 else float("nan")
        print(f"  {split_name.upper()}: {n} games | "
              f"home_covers_rl={rl:.3f} | avg_total={tot:.2f}")


if __name__ == "__main__":
    main()
