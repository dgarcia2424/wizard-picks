"""
build_feature_matrix.py
=======================
Build one row per historical game for XGBoost training and validation.

Years:
  train  -- 2023 + 2024   (backtest CSV for 2024; statcast-derived for 2023)
  val    -- 2025           (backtest CSV)

Feature groups:
  SP features       per-pitcher K%, BB%, xwOBA, GB%, xRV/pitch, velo, age, arm
  SP differentials  home - away pitcher quality gaps
  Team batting       xwOBA vs RHP / LHP per team (from statcast p_throws split)
  Park factors       run factor, HR factor, GB factor
  Weather            temperature, wind, humidity
  Vegas lines        closing moneyline, opening total (implies market belief)
  Calendar           month, year, day-of-week

Labels:
  actual_home_win   binary
  home_margin       home_score - away_score (numeric)
  home_covers_rl    home_margin >= 2  (home -1.5 run line covers)
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

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = Path("./data/statcast")

BACKTEST_FILES = {
    2024: "data/raw/backtest_games_2024.csv",
    2025: "data/raw/backtest_games_2025.csv",
}

# Park factor team name -> MLB abbreviation mapping
PARK_NAME_TO_ABBREV = {
    "Angels": "LAA", "Astros": "HOU", "Athletics": "ATH", "Blue Jays": "TOR",
    "Braves": "ATL", "Brewers": "MIL", "Cardinals": "STL", "Cubs": "CHC",
    "Diamondbacks": "AZ", "Dodgers": "LAD", "Giants": "SF", "Guardians": "CLE",
    "Indians": "CLE",  # old name
    "Mariners": "SEA", "Marlins": "MIA", "Mets": "NYM", "Nationals": "WSH",
    "Orioles": "BAL", "Padres": "SD", "Phillies": "PHI", "Pirates": "PIT",
    "Rangers": "TEX", "Rays": "TB", "Red Sox": "BOS", "Reds": "CIN",
    "Rockies": "COL", "Royals": "KC", "Tigers": "DET", "Twins": "MIN",
    "White Sox": "CWS", "Yankees": "NYY", "A's": "ATH", "Mets": "NYM",
}

# ML -> implied probability conversion
def ml_to_prob(ml):
    """American moneyline to implied win probability (no vig removed)."""
    if pd.isna(ml):
        return np.nan
    ml = float(ml)
    if ml > 0:
        return 100 / (ml + 100)
    else:
        return abs(ml) / (abs(ml) + 100)


# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------

def _load_statcast(year: int, cols: list) -> pd.DataFrame:
    path = DATA_DIR / f"statcast_{year}.parquet"
    df = pd.read_parquet(path, engine="pyarrow", columns=cols)
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _to_num(series):
    return pd.to_numeric(series, errors="coerce")


def _normalize_name(name: str) -> str:
    """'Glasnow, Tyler' or 'TYLER GLASNOW' -> 'TYLER GLASNOW'"""
    if not isinstance(name, str):
        return ""
    name = name.strip()
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        name = f"{parts[1]} {parts[0]}"
    return name.upper()


# ---------------------------------------------------------------------------
# SECTION 1 — GAME LIST (starters + game_pk + scores)
# ---------------------------------------------------------------------------

def build_game_list(years: list[int], verbose: bool = True) -> pd.DataFrame:
    """
    For each year, build a game-level dataframe with:
      game_date, game_pk, home_team, away_team, season
      home_starter_name, away_starter_name (FIRST LAST format)
      home_score, away_score  (from schedule_all join)
    """
    if verbose:
        print("  [1/7] Building game list ...")

    frames = []
    for year in years:
        if year in BACKTEST_FILES:
            # Use existing backtest file (has confirmed starters)
            df = pd.read_csv(BACKTEST_FILES[year])
            df["game_date"] = pd.to_datetime(df["game_date"])
            df["season"]    = year
            df = df.rename(columns={
                "home_starter": "home_starter_name",
                "away_starter": "away_starter_name",
            })
            # Already FIRST LAST uppercase in backtest CSVs
            if verbose:
                print(f"      {year}: {len(df)} games from backtest CSV")
        else:
            # Derive from statcast + schedule
            df = _derive_game_list_from_statcast(year, verbose=verbose)

        # Join with schedule_all to get home_score, away_score
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
                print(f"      {year}: schedule_all not found, scores will be NaN")

        frames.append(df)

    games = pd.concat(frames, ignore_index=True)
    games["home_score"] = _to_num(games.get("home_score", pd.Series(dtype=float)))
    games["away_score"] = _to_num(games.get("away_score", pd.Series(dtype=float)))

    if verbose:
        n_scored = (games["home_score"].notna() & games["away_score"].notna()).sum()
        print(f"      Total: {len(games)} games | {n_scored} with scores")

    return games


def _derive_game_list_from_statcast(year: int, verbose: bool = True) -> pd.DataFrame:
    """
    Identify starting pitchers from statcast (first pitcher in inning 1 per team).
    - Top of 1st (inning_topbot='Top'): home team pitching -> home_starter
    - Bot of 1st (inning_topbot='Bot'): away team pitching -> away_starter
    """
    cols = ["game_pk", "game_date", "home_team", "away_team",
            "pitcher", "player_name", "inning", "inning_topbot", "at_bat_number"]
    df = _load_statcast(year, cols)
    df = df[df["inning"] == 1].copy()
    df["at_bat_number"] = _to_num(df["at_bat_number"])

    results = []
    for game_pk, g in df.groupby("game_pk"):
        row = {"game_pk": game_pk}
        # Home team info
        meta = g.iloc[0]
        row["game_date"]  = meta["game_date"]
        row["home_team"]  = meta["home_team"]
        row["away_team"]  = meta["away_team"]

        # Home SP: top of 1st (home pitcher facing away batters)
        top = g[g["inning_topbot"] == "Top"]
        if len(top) > 0:
            sp_row = top.loc[top["at_bat_number"].idxmin()]
            row["home_starter_name"] = _normalize_name(str(sp_row["player_name"]))

        # Away SP: bottom of 1st (away pitcher facing home batters)
        bot = g[g["inning_topbot"] == "Bot"]
        if len(bot) > 0:
            sp_row = bot.loc[bot["at_bat_number"].idxmin()]
            row["away_starter_name"] = _normalize_name(str(sp_row["player_name"]))

        results.append(row)

    games = pd.DataFrame(results)
    games["season"] = year
    games = games.dropna(subset=["home_starter_name", "away_starter_name"])

    if verbose:
        print(f"      {year}: {len(games)} games derived from statcast")

    return games


# ---------------------------------------------------------------------------
# SECTION 2 — PER-PITCHER SEASON STATS
# ---------------------------------------------------------------------------

def build_pitcher_season_stats(year: int, verbose: bool = True) -> pd.DataFrame:
    """
    Per-pitcher season aggregations from statcast + support files.
    Returns one row per pitcher (identified by pitcher_id) with:
      - k_pct, bb_pct, xwoba_against, gb_pct, xrv_per_pitch (from statcast)
      - ff_velo (from pitch_arsenal)
      - whiff_pctl, fb_spin_pctl, fb_velo_pctl, xera_pctl (from percentiles)
      - est_woba, era_minus_xera (from pitcher_xstats)
      - age_pit, arm_angle, p_throws (from statcast)
      - pitcher_name_normalized (FIRST LAST)
    """
    if verbose:
        print(f"    Building pitcher stats for {year} ...")

    # --- From statcast: K%, BB%, xwOBA, GB%, xRV/pitch, age, arm, handedness --
    sc_cols = ["pitcher", "player_name", "p_throws", "age_pit", "arm_angle",
               "events", "bb_type", "estimated_woba_using_speedangle",
               "delta_pitcher_run_exp", "woba_denom"]
    df = _load_statcast(year, sc_cols)

    df["xwoba"]   = _to_num(df["estimated_woba_using_speedangle"])
    df["xrv"]     = _to_num(df["delta_pitcher_run_exp"])
    df["age_pit"] = _to_num(df["age_pit"])
    df["arm_angle"] = _to_num(df["arm_angle"])
    df["is_pa"]   = _to_num(df["woba_denom"]) > 0
    df["k"]       = (df["events"] == "strikeout").astype(float)
    df["bb"]      = (df["events"] == "walk").astype(float)
    df["is_gb"]   = (df["bb_type"] == "ground_ball").astype(float)
    df["is_bip"]  = df["bb_type"].notna().astype(float)

    g = df.groupby("pitcher").agg(
        n_pa          = ("is_pa",    "sum"),
        k             = ("k",        "sum"),
        bb            = ("bb",       "sum"),
        gb            = ("is_gb",    "sum"),
        bip           = ("is_bip",   "sum"),
        xwoba_against = ("xwoba",    "mean"),
        xrv_total     = ("xrv",      "sum"),
        n_pitches     = ("xrv",      "count"),
        age_pit       = ("age_pit",  "median"),
        arm_angle     = ("arm_angle","median"),
        p_throws      = ("p_throws", lambda x: x.mode()[0] if len(x) > 0 else "R"),
        player_name   = ("player_name", lambda x: x.dropna().iloc[0]
                         if x.notna().any() else ""),
    ).reset_index()

    g["k_pct"]         = g["k"] / g["n_pa"].clip(lower=1)
    g["bb_pct"]        = g["bb"] / g["n_pa"].clip(lower=1)
    g["gb_pct"]        = g["gb"] / g["bip"].clip(lower=1)
    g["xrv_per_pitch"] = g["xrv_total"] / g["n_pitches"].clip(lower=1)
    g["k_minus_bb"]    = g["k_pct"] - g["bb_pct"]
    g["p_throws_R"]    = (g["p_throws"] == "R").astype(int)
    g["pitcher_name_normalized"] = g["player_name"].apply(_normalize_name)

    # --- Fastball velo from pitch_arsenal ------------------------------------
    ars_path = DATA_DIR / f"pitch_arsenal_{year}.parquet"
    try:
        ars = pd.read_parquet(ars_path, engine="pyarrow",
                              columns=["pitcher", "ff_avg_speed"])
        ars["pitcher"] = _to_num(ars["pitcher"])
        ars = ars.rename(columns={"ff_avg_speed": "ff_velo"})
        g = g.merge(ars, on="pitcher", how="left")
    except FileNotFoundError:
        g["ff_velo"] = np.nan

    # --- Percentile ranks from pitcher_percentiles ---------------------------
    perc_path = DATA_DIR / f"pitcher_percentiles_{year}.parquet"
    try:
        perc = pd.read_parquet(perc_path, engine="pyarrow",
                               columns=["player_id", "k_percent", "bb_percent",
                                        "whiff_percent", "fb_spin", "fb_velocity",
                                        "xera"])
        perc = perc.rename(columns={
            "player_id":     "pitcher",
            "k_percent":     "k_pct_pctl",
            "bb_percent":    "bb_pct_pctl",
            "whiff_percent": "whiff_pctl",
            "fb_spin":       "fb_spin_pctl",
            "fb_velocity":   "fb_velo_pctl",
            "xera":          "xera_pctl",
        })
        perc["pitcher"] = _to_num(perc["pitcher"])
        for c in ["k_pct_pctl", "bb_pct_pctl", "whiff_pctl",
                  "fb_spin_pctl", "fb_velo_pctl", "xera_pctl"]:
            perc[c] = _to_num(perc[c]) / 100.0   # normalize to 0-1
        g = g.merge(perc, on="pitcher", how="left")
    except FileNotFoundError:
        for c in ["k_pct_pctl", "bb_pct_pctl", "whiff_pctl",
                  "fb_spin_pctl", "fb_velo_pctl", "xera_pctl"]:
            g[c] = np.nan

    # --- ERA vs xERA from pitcher_xstats ------------------------------------
    xst_path = DATA_DIR / f"pitcher_xstats_{year}.parquet"
    try:
        xst = pd.read_parquet(xst_path, engine="pyarrow",
                              columns=["player_id", "est_woba",
                                       "era_minus_xera_diff"])
        xst = xst.rename(columns={
            "player_id":         "pitcher",
            "est_woba":          "xwoba_xstats",
            "era_minus_xera_diff": "era_minus_xera",
        })
        xst["pitcher"] = _to_num(xst["pitcher"])
        xst["xwoba_xstats"]  = _to_num(xst["xwoba_xstats"])
        xst["era_minus_xera"] = _to_num(xst["era_minus_xera"])
        g = g.merge(xst, on="pitcher", how="left")
    except FileNotFoundError:
        g["xwoba_xstats"]  = np.nan
        g["era_minus_xera"] = np.nan

    g["year"] = year
    if verbose:
        n_with_velo = g["ff_velo"].notna().sum()
        print(f"      {len(g)} pitchers | {n_with_velo} with ff_velo | "
              f"year={year}")
    return g


def build_all_pitcher_stats(years: list[int], verbose: bool = True) -> pd.DataFrame:
    frames = []
    for yr in years:
        frames.append(build_pitcher_season_stats(yr, verbose=verbose))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# SECTION 3 — TEAM BATTING SPLITS (vs LHP / vs RHP)
# ---------------------------------------------------------------------------

def build_team_batting_splits(year: int, verbose: bool = True) -> pd.DataFrame:
    """
    Per-team batting stats split by opposing pitcher handedness.
    inning_topbot='Top': away team batting, home pitcher pitching
    inning_topbot='Bot': home team batting, away pitcher pitching
    """
    cols = ["home_team", "away_team", "inning_topbot", "p_throws",
            "estimated_woba_using_speedangle", "woba_denom",
            "events", "bb_type"]
    df = _load_statcast(year, cols)

    df["xwoba"] = _to_num(df["estimated_woba_using_speedangle"])
    df["is_pa"] = _to_num(df["woba_denom"]) > 0
    df["k"]     = (df["events"] == "strikeout").astype(float)
    df["bb"]    = (df["events"] == "walk").astype(float)

    # Batting team: Top = away team bats, Bot = home team bats
    df["batting_team"] = np.where(
        df["inning_topbot"] == "Top", df["away_team"], df["home_team"])

    pa_df = df[df["is_pa"]].copy()

    g = pa_df.groupby(["batting_team", "p_throws"]).agg(
        n_pa   = ("is_pa",  "sum"),
        xwoba  = ("xwoba",  "mean"),
        k_pct  = ("k",      "mean"),
        bb_pct = ("bb",     "mean"),
    ).reset_index()

    # Pivot to wide (vs_RHP and vs_LHP columns per team)
    def hand_cols(hand, suffix):
        sub = g[g["p_throws"] == hand].rename(columns={
            "xwoba":  f"bat_xwoba_vs_{suffix}",
            "k_pct":  f"bat_k_vs_{suffix}",
            "bb_pct": f"bat_bb_vs_{suffix}",
            "n_pa":   f"bat_pa_vs_{suffix}",
        }).drop(columns="p_throws")
        return sub

    rhp = hand_cols("R", "rhp")
    lhp = hand_cols("L", "lhp")
    splits = rhp.merge(lhp, on="batting_team", how="outer")
    splits["year"] = year

    if verbose:
        n_teams = splits["batting_team"].nunique()
        print(f"      {year}: {n_teams} teams with batting splits")

    return splits


def build_all_batting_splits(years: list[int], verbose: bool = True) -> pd.DataFrame:
    frames = []
    for yr in years:
        frames.append(build_team_batting_splits(yr, verbose=verbose))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# SECTION 4 — PARK FACTORS
# ---------------------------------------------------------------------------

def build_park_factors(years: list[int], verbose: bool = True) -> pd.DataFrame:
    """Load park_factors_{year}.parquet and map team names to abbreviations."""
    frames = []
    for year in years:
        try:
            pf = pd.read_parquet(DATA_DIR / f"park_factors_{year}.parquet",
                                 engine="pyarrow")
            pf = pf.rename(columns={"Team": "team_full"})
            pf["home_team"] = pf["team_full"].map(PARK_NAME_TO_ABBREV)

            for col in ["Basic_5yr", "3yr", "1yr", "HR", "GB", "SO", "FIP"]:
                if col in pf.columns:
                    pf[col] = _to_num(pf[col])

            pf = pf.rename(columns={
                "Basic_5yr": "park_factor",
                "HR":        "park_hr",
                "GB":        "park_gb",
                "SO":        "park_so",
                "FIP":       "park_fip",
            })
            keep = [c for c in ["home_team", "park_factor", "park_hr",
                                 "park_gb", "park_so", "park_fip"]
                    if c in pf.columns]
            pf = pf[keep].copy()
            pf["year"] = year
            frames.append(pf)
        except FileNotFoundError:
            if verbose:
                print(f"      park_factors_{year}.parquet not found")

    if frames:
        df = pd.concat(frames, ignore_index=True)
        # Normalize to ratio vs 100 baseline
        for col in ["park_factor", "park_hr", "park_gb", "park_so", "park_fip"]:
            if col in df.columns:
                df[col] = df[col] / 100.0
        if verbose:
            print(f"      Park factors: {len(df)} team-years loaded")
        return df
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# SECTION 5 — WEATHER
# ---------------------------------------------------------------------------

def build_weather(years: list[int], verbose: bool = True) -> pd.DataFrame:
    frames = []
    for year in years:
        try:
            w = pd.read_parquet(DATA_DIR / f"weather_{year}.parquet",
                                engine="pyarrow")
            w["game_date"] = pd.to_datetime(w["game_date"])
            for col in ["temp_f", "wind_mph", "humidity"]:
                if col in w.columns:
                    w[col] = _to_num(w[col])
            w["year"] = year
            frames.append(w[["game_date", "home_team", "year",
                              "temp_f", "wind_mph", "humidity"]])
        except FileNotFoundError:
            pass

    if frames:
        df = pd.concat(frames, ignore_index=True)
        if verbose:
            print(f"      Weather: {len(df)} game-days loaded")
        return df
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# SECTION 6 — VEGAS LINES
# ---------------------------------------------------------------------------

def build_vegas_lines(years: list[int], verbose: bool = True) -> pd.DataFrame:
    frames = []
    for year in years:
        for fname in [f"odds_historical_{year}.parquet",
                      f"odds_combined_{year}.parquet"]:
            try:
                v = pd.read_parquet(DATA_DIR / fname, engine="pyarrow")
                v["game_date"] = pd.to_datetime(v["game_date"])
                for col in ["close_ml_home", "close_ml_away",
                             "open_total", "close_total"]:
                    if col in v.columns:
                        v[col] = _to_num(v[col])
                v["vegas_implied_home"] = v["close_ml_home"].apply(ml_to_prob)
                v["vegas_implied_away"] = v["close_ml_away"].apply(ml_to_prob)
                v["year"] = year
                keep = [c for c in ["game_date", "home_team", "away_team", "year",
                                     "close_ml_home", "close_ml_away",
                                     "open_total", "close_total",
                                     "vegas_implied_home", "vegas_implied_away"]
                        if c in v.columns]
                frames.append(v[keep])
                break  # only load first file found per year
            except FileNotFoundError:
                continue

    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df.dropna(subset=["close_ml_home"], how="all")
        if verbose:
            n_with_total = df["close_total"].notna().sum() if "close_total" in df.columns else 0
            print(f"      Vegas lines: {len(df)} games | "
                  f"{n_with_total} with closing total")
        return df
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# SECTION 7 — ASSEMBLE FEATURE MATRIX
# ---------------------------------------------------------------------------

def assemble_matrix(games: pd.DataFrame,
                    pitchers: pd.DataFrame,
                    batting: pd.DataFrame,
                    park_factors: pd.DataFrame,
                    weather: pd.DataFrame,
                    vegas: pd.DataFrame,
                    verbose: bool = True) -> pd.DataFrame:
    """
    Merge all feature tables onto the game-level dataframe.
    """
    if verbose:
        print("  [7/7] Assembling feature matrix ...")

    df = games.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["year"]      = df["season"].astype(int)

    # Calendar features
    df["game_month"]      = df["game_date"].dt.month
    df["game_day_of_week"] = df["game_date"].dt.dayofweek  # 0=Mon

    # Build pitcher lookup: {(FIRST LAST, year) -> stats row}
    # pitchers df has pitcher_name_normalized
    pitcher_lookup = pitchers.set_index(["pitcher_name_normalized", "year"])

    def get_pitcher_features(name: str, year: int, prefix: str) -> dict:
        """Return dict of prefixed pitcher features for one starter."""
        try:
            row = pitcher_lookup.loc[(name, year)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]  # multiple pitcher IDs with same name: take first
        except KeyError:
            row = pd.Series(dtype=float)

        feature_cols = [
            "k_pct", "bb_pct", "xwoba_against", "gb_pct", "xrv_per_pitch",
            "k_minus_bb", "ff_velo", "age_pit", "arm_angle", "p_throws_R",
            "k_pct_pctl", "bb_pct_pctl", "whiff_pctl", "fb_spin_pctl",
            "fb_velo_pctl", "xera_pctl", "era_minus_xera",
        ]
        return {f"{prefix}_{c}": row.get(c, np.nan) for c in feature_cols}

    # Apply pitcher feature lookup
    home_feats = df.apply(
        lambda r: get_pitcher_features(r["home_starter_name"], r["year"], "home_sp"),
        axis=1).apply(pd.Series)
    away_feats = df.apply(
        lambda r: get_pitcher_features(r["away_starter_name"], r["year"], "away_sp"),
        axis=1).apply(pd.Series)

    df = pd.concat([df, home_feats, away_feats], axis=1)

    # SP differential features (home advantage direction)
    df["sp_k_pct_diff"]    = df["home_sp_k_pct"]    - df["away_sp_k_pct"]
    df["sp_xwoba_diff"]    = df["away_sp_xwoba_against"] - df["home_sp_xwoba_against"]
    df["sp_xrv_diff"]      = df["home_sp_xrv_per_pitch"] - df["away_sp_xrv_per_pitch"]
    df["sp_velo_diff"]     = df["home_sp_ff_velo"]   - df["away_sp_ff_velo"]
    df["sp_age_diff"]      = df["home_sp_age_pit"]   - df["away_sp_age_pit"]
    df["sp_kminusbb_diff"] = df["home_sp_k_minus_bb"] - df["away_sp_k_minus_bb"]

    # Team batting splits: join home team and away team vs opposing SP's handedness
    bat_lookup = batting.set_index(["batting_team", "year"])

    def get_batting(team: str, year: int, prefix: str) -> dict:
        try:
            row = bat_lookup.loc[(team, year)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
        except KeyError:
            row = pd.Series(dtype=float)
        cols = ["bat_xwoba_vs_rhp", "bat_xwoba_vs_lhp",
                "bat_k_vs_rhp", "bat_k_vs_lhp",
                "bat_bb_vs_rhp", "bat_bb_vs_lhp"]
        return {f"{prefix}_{c}": row.get(c, np.nan) for c in cols}

    home_bat = df.apply(
        lambda r: get_batting(r["home_team"], r["year"], "home"), axis=1
    ).apply(pd.Series)
    away_bat = df.apply(
        lambda r: get_batting(r["away_team"], r["year"], "away"), axis=1
    ).apply(pd.Series)

    df = pd.concat([df, home_bat, away_bat], axis=1)

    # Matchup-specific batting: home team vs away SP's handedness
    # home_sp_p_throws_R=1 -> R-handed away SP -> use home bat_vs_rhp
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

    # Park factors
    if not park_factors.empty:
        pf_lookup = park_factors.set_index(["home_team", "year"])
        pf_cols = ["park_factor", "park_hr", "park_gb", "park_so", "park_fip"]
        pf_feats = df.apply(
            lambda r: {c: pf_lookup.loc[(r["home_team"], r["year"])].get(c, np.nan)
                       if (r["home_team"], r["year"]) in pf_lookup.index
                       else np.nan
                       for c in pf_cols},
            axis=1
        ).apply(pd.Series)
        df = pd.concat([df, pf_feats], axis=1)

    # Weather
    if not weather.empty:
        df = df.merge(
            weather.rename(columns={"year": "year_w"}),
            left_on=["game_date", "home_team"],
            right_on=["game_date", "home_team"],
            how="left",
        )
        df = df.drop(columns=["year_w"], errors="ignore")

    # Vegas lines
    if not vegas.empty:
        df = df.merge(
            vegas.rename(columns={"year": "year_v"}),
            left_on=["game_date", "home_team", "away_team"],
            right_on=["game_date", "home_team", "away_team"],
            how="left",
        )
        df = df.drop(columns=["year_v"], errors="ignore")

    # Labels
    df["home_score"]     = _to_num(df.get("home_score", pd.Series(dtype=float)))
    df["away_score"]     = _to_num(df.get("away_score", pd.Series(dtype=float)))
    df["home_margin"]    = df["home_score"] - df["away_score"]
    df["home_covers_rl"] = (df["home_margin"] >= 2).astype("Int8")
    df["away_covers_rl"] = (df["home_margin"] <= 1).astype("Int8")
    df["total_runs"]     = df["home_score"] + df["away_score"]

    # NaN-safe labels (drop games with missing scores)
    df.loc[df["home_margin"].isna(), ["home_covers_rl", "away_covers_rl"]] = pd.NA

    # Split column
    df["split"] = np.where(df["year"].isin([2023, 2024]), "train", "val")

    if verbose:
        n_train = (df["split"] == "train").sum()
        n_val   = (df["split"] == "val").sum()
        n_labeled = df["home_covers_rl"].notna().sum()
        rl_rate   = df["home_covers_rl"].mean() if n_labeled > 0 else float("nan")
        print(f"      {len(df)} total games | train: {n_train} | val: {n_val}")
        print(f"      {n_labeled} labeled | home covers RL rate: {rl_rate:.3f}")
        n_with_vegas = df["vegas_implied_home"].notna().sum() \
            if "vegas_implied_home" in df.columns else 0
        print(f"      {n_with_vegas} with Vegas implied probability")

    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build feature matrix for XGBoost run-line model")
    parser.add_argument("--years", type=int, nargs="+",
                        default=[2023, 2024, 2025],
                        help="Years to include (2023=train, 2024=train, 2025=val)")
    parser.add_argument("--out",   type=str,
                        default="feature_matrix",
                        help="Output filename stem (no extension)")
    args = parser.parse_args()

    years = args.years

    print("=" * 60)
    print("  build_feature_matrix.py")
    print(f"  Years: {years}")
    print("=" * 60)

    # Build each data layer
    games         = build_game_list(years)

    print("\n  [2-4/7] Pitcher season stats ...")
    pitchers      = build_all_pitcher_stats(years)

    print("\n  [5/7] Team batting splits ...")
    batting       = build_all_batting_splits(years)

    print("\n  [6a/7] Park factors ...")
    park_factors  = build_park_factors(years)

    print("  [6b/7] Weather ...")
    weather       = build_weather(years)

    print("  [6c/7] Vegas lines ...")
    vegas         = build_vegas_lines(years)

    # Assemble
    print()
    matrix = assemble_matrix(games, pitchers, batting,
                             park_factors, weather, vegas)

    # Save
    out_parquet = f"{args.out}.parquet"
    out_csv     = f"{args.out}.csv"
    matrix.to_parquet(out_parquet, engine="pyarrow", index=False)
    matrix.to_csv(out_csv, index=False)

    print()
    print(f"  Saved -> {out_parquet}  ({len(matrix)} rows x {len(matrix.columns)} cols)")
    print(f"  Saved -> {out_csv}")

    # Summary
    print()
    print("=" * 60)
    print("  Feature matrix summary")
    print("=" * 60)

    feature_cols = [c for c in matrix.columns if c not in
                    ["game_date", "game_pk", "home_team", "away_team", "season",
                     "home_starter_name", "away_starter_name", "home_score",
                     "away_score", "actual_home_win", "home_margin",
                     "home_covers_rl", "away_covers_rl", "total_runs", "split",
                     "year"]]
    print(f"  Feature columns: {len(feature_cols)}")

    null_pct = (matrix[feature_cols].isna().mean() * 100).sort_values(ascending=False)
    print("\n  Top-10 null % features (may need imputation):")
    for col, pct in null_pct.head(10).items():
        print(f"    {col:<40} {pct:.1f}%")

    print()
    train_df = matrix[matrix["split"] == "train"]
    val_df   = matrix[matrix["split"] == "val"]
    for split_name, sdf in [("train", train_df), ("val", val_df)]:
        n = len(sdf)
        rl = sdf["home_covers_rl"].mean() if n > 0 else float("nan")
        total_mean = sdf["total_runs"].mean() if n > 0 else float("nan")
        print(f"  {split_name.upper()}: {n} games | "
              f"home_covers_rl={rl:.3f} | "
              f"avg_total={total_mean:.2f}")


if __name__ == "__main__":
    main()
