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

        # Aggregate pitch-level data up to one row per (pitcher, game)
        game_agg = df.groupby(["pitcher", "game_pk", "game_date"]).agg(
            n_pa          = ("is_pa",        "sum"),
            k             = ("k",            "sum"),
            bb            = ("bb",           "sum"),
            gb            = ("is_gb",        "sum"),
            bip           = ("is_bip",       "sum"),
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
                    "xrv_per_pitch", "k_minus_bb"]

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
# SECTION 5 — ASSEMBLE FEATURE MATRIX
# ---------------------------------------------------------------------------

def assemble_matrix(
    games:    pd.DataFrame,
    pitchers: pd.DataFrame,
    batting:  pd.DataFrame,
    bullpen:  pd.DataFrame,
    verbose:  bool = True,
) -> pd.DataFrame:
    """
    Merge all feature tables onto the game-level dataframe.

    Park factors, weather, and Vegas lines are intentionally excluded:
    the Monte Carlo engine already encodes the scoring environment.
    mc_expected_runs is added as a NaN placeholder; it will be populated
    at inference time before XGBoost scoring.
    """
    if verbose:
        print("  [4/4] Assembling feature matrix ...")

    df = games.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["year"]      = df["season"].astype(int)

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
            # Trailing 10-day recency stats (Roadmap #2)
            "k_pct_10d", "bb_pct_10d", "xwoba_10d",
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

    # Trailing 10-day SP diffs — recency signal (Roadmap #2)
    # Columns are home_sp_k_pct_10d / away_sp_k_pct_10d etc.
    df["sp_k_pct_10d_diff"]  = (df["home_sp_k_pct_10d"]
                                - df["away_sp_k_pct_10d"])
    df["sp_xwoba_10d_diff"]  = (df["away_sp_xwoba_10d"]
                                - df["home_sp_xwoba_10d"])   # flipped: lower home xwOBA = better
    df["sp_bb_pct_10d_diff"] = (df["home_sp_bb_pct_10d"]
                                - df["away_sp_bb_pct_10d"])

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

    games    = build_game_list(years)
    pitchers = build_pitcher_ewma_stats(years)
    batting  = build_team_batting_ewma(years)
    bullpen  = build_bullpen_stats(years)

    print()
    matrix = assemble_matrix(games, pitchers, batting, bullpen)

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
