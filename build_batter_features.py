"""
build_batter_features.py
------------------------
Assemble the per-batter-per-game feature matrix that feeds the Wizard v3.3
TB stacker. Grain: [date, player_id, game_id].

Joins (in order):
    1. Lineup slots (with handedness)     -> data/statcast/lineups_*_long.parquet
    2. TB labels (ground truth)           -> data/batter_labels/tb_by_game.parquet
    3. Batter Statcast xStats             -> data/statcast/batter_xstats_{yr}.parquet
                                             + batter_exitvelo_{yr}.parquet
    4. Grounded run environment (game)    -> model_scores.csv  (projected_total_adj)

This is the SCAFFOLD — it wires joins and emits audit stats so we can see
where rows are dropped before training. Missing upstream pieces are flagged
in the audit, not silently dropped.

Usage:
    python build_batter_features.py                      # all available dates
    python build_batter_features.py --years 2024 2025    # restrict
    python build_batter_features.py --dry-run            # report joins only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

STATCAST_DIR = Path("data/statcast")
LABEL_FILE   = Path("data/batter_labels/tb_by_game.parquet")
SCORES_FILE  = Path("model_scores.csv")
BACKTEST_FILES = [Path("backtest_2024_results.csv"),
                    Path("backtest_2025_results.csv")]
FEATURE_MATRIX = Path("feature_matrix_enriched_v2.parquet")
STADIUM_META   = Path("config/stadium_metadata.json")
OUT_DIR      = Path("data/batter_features")
OUT_FILE     = OUT_DIR / "batter_game_matrix.parquet"
FINAL_FILE   = OUT_DIR / "final_training_matrix.parquet"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_lineup_slots(years: list[int]) -> pd.DataFrame:
    """All available dated long-lineup parquets, filtered to requested years."""
    frames = []
    for f in sorted(STATCAST_DIR.glob("lineups_*_long.parquet")):
        name = f.stem  # lineups_2026-04-23_long
        date_str = name.split("_")[1]
        try:
            yr = int(date_str[:4])
        except ValueError:
            continue
        if yr not in years:
            continue
        try:
            frames.append(pd.read_parquet(f))
        except Exception as exc:
            print(f"  [skip] {f}: {exc}")
    if not frames:
        return pd.DataFrame(columns=["game_date", "game_pk", "player_id",
                                       "player_name", "team", "side",
                                       "batting_order", "position", "stand"])
    df = pd.concat(frames, ignore_index=True)
    # Today-file duplicates may exist; dedupe on (game_pk, player_id).
    df = df.drop_duplicates(subset=["game_pk", "player_id"], keep="last")
    if "stand" not in df.columns:
        df["stand"] = ""
    # Enrich from handedness cache (parquets may predate handedness pull).
    try:
        import json
        cache_path = Path("data/statcast/handedness_cache.json")
        if cache_path.exists():
            cache = json.loads(cache_path.read_text())
            mapped = df["player_id"].astype("Int64").astype(str).map(cache)
            cur = df["stand"].fillna("").astype(str)
            df["stand"] = cur.where(cur.isin(["L", "R", "S"]), mapped.fillna(""))
    except Exception as exc:
        print(f"  [warn] handedness cache enrich failed: {exc}")
    return df


def load_labels(years: list[int]) -> pd.DataFrame:
    if not LABEL_FILE.exists():
        raise SystemExit(f"Missing {LABEL_FILE}. Run build_tb_labels.py first.")
    lab = pd.read_parquet(LABEL_FILE)
    return lab[lab["year"].isin(years)].copy()


def load_batter_xstats(years: list[int]) -> pd.DataFrame:
    """Season-level batter xStats; joined as 'prior season / trailing' features.
    NOTE: these parquets are season-grain, not game-grain. We join by
    (year-1, player_id) so the feature row is leakage-safe.
    """
    frames = []
    for yr in years:
        xfile = STATCAST_DIR / f"batter_xstats_{yr}.parquet"
        efile = STATCAST_DIR / f"batter_exitvelo_{yr}.parquet"
        for f, tag in [(xfile, "xstats"), (efile, "exitvelo")]:
            if f.exists():
                try:
                    d = pd.read_parquet(f)
                    d["_src_year"] = yr
                    d["_src_tag"]  = tag
                    frames.append(d)
                except Exception as exc:
                    print(f"  [skip] {f}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


_TEAM_NAME_SUFFIX_TO_ABBR = {
    # Minimal mapping for model_scores.csv game strings ("Away Team @ Home Team").
    # Uses team nicknames (last word) -> pipeline abbr.
    "braves": "ATL", "diamondbacks": "AZ", "orioles": "BAL", "red sox": "BOS",
    "cubs": "CHC", "white sox": "CWS", "reds": "CIN", "guardians": "CLE",
    "rockies": "COL", "tigers": "DET", "astros": "HOU", "royals": "KC",
    "angels": "LAA", "dodgers": "LAD", "marlins": "MIA", "brewers": "MIL",
    "twins": "MIN", "nationals": "WSH", "mets": "NYM", "yankees": "NYY",
    "athletics": "ATH", "phillies": "PHI", "pirates": "PIT", "padres": "SD",
    "giants": "SF", "mariners": "SEA", "cardinals": "STL", "rays": "TB",
    "rangers": "TEX", "blue jays": "TOR",
}


def _matchup_to_home_abbr(game_str: str) -> str | None:
    if not isinstance(game_str, str) or "@" not in game_str:
        return None
    home_name = game_str.split("@", 1)[1].strip().lower()
    # Match by last word(s) against the nickname map.
    for nick, abbr in _TEAM_NAME_SUFFIX_TO_ABBR.items():
        if home_name.endswith(nick):
            return abbr
    return None


def load_grounded_env() -> pd.DataFrame:
    """Union of historical + current env signals.

    Join key is (date, home_park_id). No game_pk column required, because
    model_scores.csv does not carry one and backtest CSVs also lack it.

    Priority for 'projected_total_adj':
        1. model_scores.csv Totals rows (current/recent - authoritative)
        2. backtest_{2024,2025}_results.csv via 'vegas_total' proxy,
           when vegas_total is present and non-null.

    Returns cols: [date, home_park_id, projected_total_adj, env_source]
    """
    frames: list[pd.DataFrame] = []

    if SCORES_FILE.exists():
        df = pd.read_csv(SCORES_FILE)
        if "projected_total_adj" in df.columns:
            cur = df.dropna(subset=["projected_total_adj"]).copy()
            if not cur.empty:
                cur["home_park_id"] = cur["game"].map(_matchup_to_home_abbr)
                cur = cur.dropna(subset=["home_park_id"])
                cur["env_source"] = "model_scores"
                cur = (cur[["date", "home_park_id",
                            "projected_total_adj", "env_source"]]
                       .drop_duplicates(subset=["date", "home_park_id"],
                                          keep="last"))
                frames.append(cur)

    for f in BACKTEST_FILES:
        if not f.exists():
            continue
        d = pd.read_csv(f)
        if "vegas_total" not in d.columns or d["vegas_total"].isna().all():
            print(f"  [warn] {f.name} vegas_total all-NaN; skipping env proxy.")
            continue
        d = d.dropna(subset=["vegas_total"]).copy()
        d["projected_total_adj"] = d["vegas_total"].astype(float)
        d["env_source"]          = "vegas_proxy"
        d = d.rename(columns={"home_team": "home_park_id"})
        frames.append(d[["date", "home_park_id",
                          "projected_total_adj", "env_source"]])

    if not frames:
        print("  [warn] No env source resolved; projected_total_adj will be null.")
        return pd.DataFrame(columns=["date", "home_park_id",
                                       "projected_total_adj", "env_source"])
    env = pd.concat(frames, ignore_index=True)
    env["date"] = pd.to_datetime(env["date"])
    return env.drop_duplicates(subset=["date", "home_park_id"], keep="first")


def load_game_wind(years: list[int]) -> pd.DataFrame:
    """Per-game wind from feature_matrix_enriched_v2.parquet.
    Returns: [game_id (=game_pk), game_date, home_team, wind_mph, wind_bearing].
    """
    if not FEATURE_MATRIX.exists():
        print(f"  [warn] {FEATURE_MATRIX} not found; wind will be null.")
        return pd.DataFrame(columns=["game_id", "game_date", "home_team",
                                      "wind_mph", "wind_bearing"])
    cols = ["game_pk", "game_date", "home_team", "wind_mph", "wind_bearing",
            "temp_f", "home_sp_p_throws_R", "away_sp_p_throws_R",
            "sp_velo_decay_diff", "elo_diff",
            "home_sp_ff_velo", "away_sp_ff_velo"]
    df = pd.read_parquet(FEATURE_MATRIX, columns=cols)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df["game_date"].dt.year.isin(years)].copy()
    df = df.rename(columns={"game_pk": "game_id"})
    return df


def load_stadium_meta() -> dict[str, float]:
    """park_id -> cf_azimuth_deg map."""
    if not STADIUM_META.exists():
        print(f"  [warn] {STADIUM_META} not found; pull_side_wind_vector will be null.")
        return {}
    import json
    meta = json.loads(STADIUM_META.read_text(encoding="utf-8"))
    return {p: v["cf_azimuth_deg"] for p, v in meta.get("parks", {}).items()}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble(years: list[int], dry_run: bool = False) -> pd.DataFrame:
    print(f"Years: {years}")

    lineups = load_lineup_slots(years)
    print(f"  lineups rows:       {len(lineups):,}")
    labels  = load_labels(years)
    print(f"  labels rows:        {len(labels):,}")
    env     = load_grounded_env()
    print(f"  grounded env rows:  {len(env):,}")
    xstats  = load_batter_xstats(years)
    print(f"  batter xstats rows: {len(xstats):,}")
    wind    = load_game_wind(years)
    print(f"  game wind rows:     {len(wind):,}")
    cf_map  = load_stadium_meta()
    print(f"  stadium cf_azimuth entries: {len(cf_map)}")

    # 1) Start from labels (every batter-game with >=1 PA).
    base = labels.rename(columns={"date": "game_date"}).copy()
    base["game_date"] = pd.to_datetime(base["game_date"])

    # 2) Join lineup (slot, stand). Left join so we keep labeled rows even if
    #    lineup file is missing for that date.
    if not lineups.empty:
        lu = lineups.copy()
        lu["game_date"] = pd.to_datetime(lu["game_date"])
        lu = lu.rename(columns={"game_pk": "game_id"})
        keep = ["game_date", "game_id", "player_id",
                "batting_order", "position", "stand", "side", "team"]
        lu = lu[[c for c in keep if c in lu.columns]]
        base = base.merge(lu, on=["game_date", "game_id", "player_id"],
                           how="left")
    else:
        for c in ("batting_order", "position", "stand", "side", "team"):
            base[c] = pd.NA

    # 3) Grounded environment — keyed by (game_date, home_park_id).
    if not env.empty:
        base = base.merge(env, left_on=["game_date", "home_park_id"],
                           right_on=["date", "home_park_id"],
                           how="left")
        base = base.drop(columns=["date"], errors="ignore")
    else:
        base["projected_total_adj"] = pd.NA
        base["env_source"]          = pd.NA

    # 3b) Point-in-time grounded env (historical backfill from the thermostat).
    hist_env_path = Path("data/logs/historical_env_lookup.csv")
    if hist_env_path.exists():
        he = pd.read_csv(hist_env_path)
        he["date"] = pd.to_datetime(he["date"])
        he = he.rename(columns={"projected_total_adj": "hist_projected_total_adj",
                                  "bias_source": "hist_bias_source"})
        he = he[["date", "home_park_id", "hist_projected_total_adj",
                 "hist_bias_source", "bias_offset"]]
        base = base.merge(he, left_on=["game_date", "home_park_id"],
                           right_on=["date", "home_park_id"], how="left")
        base = base.drop(columns=["date"], errors="ignore")
        # Fill nulls from today's model_scores merge with the historical value.
        if "projected_total_adj" not in base.columns:
            base["projected_total_adj"] = pd.NA
        base["projected_total_adj"] = base["projected_total_adj"].fillna(
            base["hist_projected_total_adj"])
        if "env_source" not in base.columns:
            base["env_source"] = pd.NA
        base["env_source"] = base["env_source"].fillna(base["hist_bias_source"])
        print(f"  historical env rows: {len(he):,}")
    else:
        print(f"  [warn] {hist_env_path} not found; run generate_hist_env.py")

    # 4) Game wind + env pieces (temp_f, SP handedness, velo-decay diff, elo).
    if not wind.empty:
        w = wind.rename(columns={"home_team": "home_team_abbr"})
        w = w[["game_id", "wind_mph", "wind_bearing",
               "temp_f", "home_sp_p_throws_R", "away_sp_p_throws_R",
               "sp_velo_decay_diff", "elo_diff", "home_team_abbr",
               "home_sp_ff_velo", "away_sp_ff_velo"]]
        base = base.merge(w, on="game_id", how="left")
    else:
        for c in ("wind_mph", "wind_bearing", "temp_f",
                  "home_sp_p_throws_R", "away_sp_p_throws_R",
                  "sp_velo_decay_diff", "elo_diff", "home_team_abbr",
                  "home_sp_ff_velo", "away_sp_ff_velo"):
            base[c] = pd.NA

    base = _compute_pull_side_wind_vector(base, cf_map)
    base = _compute_velocity_decay_risk(base)
    base = _compute_lineup_fragility(base)
    base = _compute_exp_pa_heuristic(base)
    base = _join_bullpen_burn(base)
    base = _compute_thermal_expansion(base)
    base = _compute_rolling_xwoba_delta(base, years)

    # 4a) Batter Statcast percentiles (prior-season leakage-safe). Join (year-1, player_id).
    # Columns: hard_hit_percent, sprint_speed, chase_percent, bat_speed, squared_up_rate, whiff_percent
    pctile_frames = []
    for yr in range(2023, 2027):
        p = STATCAST_DIR / f"batter_percentiles_{yr}.parquet"
        if p.exists():
            pf = pd.read_parquet(p, engine="pyarrow")
            pf["_pctile_year"] = yr
            pctile_frames.append(pf)
    if pctile_frames:
        pctile = pd.concat(pctile_frames, ignore_index=True)
        id_col = next((c for c in ("player_id", "batter", "mlbam_id") if c in pctile.columns), None)
        if id_col:
            if id_col != "player_id":
                pctile = pctile.rename(columns={id_col: "player_id"})
            pctile["player_id"] = pd.to_numeric(pctile["player_id"], errors="coerce")
            sticky_cols = [c for c in ("hard_hit_percent", "sprint_speed", "chase_percent",
                                        "bat_speed", "squared_up_rate", "whiff_percent")
                           if c in pctile.columns]
            pctile = pctile[["player_id", "_pctile_year"] + sticky_cols].copy()
            pctile = pctile.drop_duplicates(subset=["player_id", "_pctile_year"], keep="last")
            base["_join_pctile_year"] = base["year"].astype("int16") - 1
            base["player_id"] = pd.to_numeric(base["player_id"], errors="coerce")
            base = base.merge(
                pctile.rename(columns={"_pctile_year": "_join_pctile_year"}),
                on=["player_id", "_join_pctile_year"], how="left", suffixes=("", "_pctile"))
            base = base.drop(columns=["_join_pctile_year"], errors="ignore")
            print(f"  [batter_pctile] joined {sticky_cols} | "
                  f"coverage: {base['sprint_speed'].notna().mean():.1%}" if "sprint_speed" in base.columns else "")
        else:
            print("  [warn] batter_percentiles: no player_id column found; skipped.")

    # 4) Batter xStats (prior-season leakage-safe). Join (year-1, player_id).
    if not xstats.empty:
        # xstats parquets may use 'player_id', 'batter', or 'last_name, first_name' —
        # normalize a key column.
        id_col = None
        for c in ("player_id", "batter", "mlbam_id", "MLBAMID"):
            if c in xstats.columns:
                id_col = c
                break
        if id_col is not None:
            x = xstats.copy()
            if id_col != "player_id":
                x = x.rename(columns={id_col: "player_id"})
            x["join_year"] = x["_src_year"].astype("int16")
            # Keep one row per (player_id, join_year) by source tag (xstats preferred).
            x = x.sort_values("_src_tag")  # exitvelo first alphabetically, xstats overrides
            x = x.drop_duplicates(subset=["player_id", "join_year"], keep="last")
            base["join_year"] = base["year"].astype("int16") - 1
            base = base.merge(x, on=["player_id", "join_year"], how="left",
                               suffixes=("", "_x"))
            base = base.drop(columns=["join_year"], errors="ignore")
        else:
            print("  [warn] batter xstats has no recognizable player id column; skipped.")

    if dry_run:
        return base

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.to_parquet(OUT_FILE, index=False)
    print(f"\nWrote {len(base):,} rows -> {OUT_FILE}")
    base.to_parquet(FINAL_FILE, index=False)
    print(f"Wrote {len(base):,} rows -> {FINAL_FILE}")
    return base


# ---------------------------------------------------------------------------
# Physics feature: pull-side wind vector
# ---------------------------------------------------------------------------

PULL_OFFSET_DEG = 22.0  # midway between CF and pull foul line (45deg arc / 2)


def _compute_pull_side_wind_vector(df: pd.DataFrame,
                                     cf_map: dict[str, float]) -> pd.DataFrame:
    """Compute cosine similarity between wind vector and batter's pull-zone.

    Convention:
      - wind_bearing in the feature matrix is the direction the wind is
        blowing TOWARD (met convention already normalized upstream).
      - CF azimuth is the bearing from home plate to CF.
      - LHB pulls to RCF -> pull_azimuth = CF + PULL_OFFSET_DEG
      - RHB pulls to LCF -> pull_azimuth = CF - PULL_OFFSET_DEG
      - Switch hitter -> take the average of both (=CF azimuth).
      - Feature = cos(wind_bearing - pull_azimuth) * (wind_mph / 10)
        Positive => wind helps push pulled fly balls; negative => blows them in.
    """
    import numpy as np

    if "wind_bearing" not in df.columns or "wind_mph" not in df.columns:
        df["pull_azimuth_deg"] = pd.NA
        df["pull_side_wind_vector"] = pd.NA
        return df

    cf = df["home_park_id"].map(cf_map)
    stand = df.get("stand", pd.Series(index=df.index, dtype=object)).fillna("")

    import numpy as np
    pull_az = pd.Series(np.nan, index=df.index, dtype="float64")
    pull_az = pull_az.mask(stand.eq("L"), cf + PULL_OFFSET_DEG)
    pull_az = pull_az.mask(stand.eq("R"), cf - PULL_OFFSET_DEG)
    pull_az = pull_az.mask(stand.eq("S"), cf)  # switch-hitter neutral
    pull_az = pull_az % 360.0

    wb = pd.to_numeric(df["wind_bearing"], errors="coerce")
    wm = pd.to_numeric(df["wind_mph"],     errors="coerce")

    theta = np.deg2rad((wb - pull_az).astype("float64"))
    vec = np.cos(theta) * (wm / 10.0)

    df["pull_azimuth_deg"]      = pull_az
    df["pull_side_wind_vector"] = vec
    return df


# ---------------------------------------------------------------------------
# v3.5 leak-free physics features
# ---------------------------------------------------------------------------

# Proxy for per-start fastball velocity std (mph). No per-SP rolling std is
# currently materialized; this is the league-average start-to-start ff_velo
# std. Using a constant keeps velocity_decay_risk strictly a function of temp
# + a pitcher-instability signal sourced from sp_velo_decay_diff.
_LEAGUE_SP_VELO_STD = 1.20              # legacy
_LEAGUE_AVG_FF_VELO_2026 = 94.19        # computed from feature_matrix_enriched_v2


def _compute_velocity_decay_risk(df: pd.DataFrame) -> pd.DataFrame:
    """v3.6: velocity_decay_risk = (temp_f - 72) * (SP_2026_STD_Velo - League_Avg).

    SP_2026_STD_Velo is the opposing-starter's season-to-date average FF
    velocity, sourced from feature_matrix_enriched_v2's home_sp_ff_velo /
    away_sp_ff_velo columns (those fields are maintained as rolling season
    aggregates by the upstream pipeline). Opposing-SP is selected per batter
    side: home batter -> away SP, away batter -> home SP.

    Sign convention: high-velo pitcher (>= league avg) on a hot day yields
    LARGE POSITIVE risk — velo doesn't decay, batter faces full heat. Low-velo
    pitcher on cold day yields positive too (temp negative, velo_delta
    negative). The feature magnitude is what matters; XGBoost learns sign.
    """
    import numpy as np
    temp = pd.to_numeric(df.get("temp_f"), errors="coerce")
    home_v = pd.to_numeric(df.get("home_sp_ff_velo"), errors="coerce")
    away_v = pd.to_numeric(df.get("away_sp_ff_velo"), errors="coerce")
    team = df.get("team", pd.Series(index=df.index, dtype=object)).fillna("")
    home = df.get("home_park_id", pd.Series(index=df.index, dtype=object)).fillna("")
    is_home = (team == home)
    opp_velo = pd.Series(np.where(is_home, away_v, home_v), index=df.index)
    velo_delta = opp_velo - _LEAGUE_AVG_FF_VELO_2026
    df["opp_sp_ff_velo"] = opp_velo
    df["velocity_decay_risk"] = (temp - 72.0) * velo_delta
    return df


def _compute_lineup_fragility(df: pd.DataFrame) -> pd.DataFrame:
    """P(sub) heuristic from platoon leverage + expected blowout magnitude.

    - platoon_same_hand: 1 if batter stand == opposing SP throws (LvL or RvR),
      0 otherwise. Switch-hitters (S) -> 0 (always neutral).
    - blowout_proxy: min(1, |elo_diff| / 200). Larger gap -> more late-game
      position-player churn -> higher fragility.
    - fragility = 0.35 * platoon_same_hand + 0.65 * blowout_proxy  (0..1).
    """
    import numpy as np
    stand = df.get("stand", pd.Series(index=df.index, dtype=object)).fillna("")
    team = df.get("team", pd.Series(index=df.index, dtype=object)).fillna("")
    home = df.get("home_park_id", pd.Series(index=df.index, dtype=object)).fillna("")
    home_sp_R = pd.to_numeric(df.get("home_sp_p_throws_R"), errors="coerce")
    away_sp_R = pd.to_numeric(df.get("away_sp_p_throws_R"), errors="coerce")
    elo = pd.to_numeric(df.get("elo_diff"), errors="coerce").fillna(0.0)

    is_home = (team == home)
    opp_sp_R = np.where(is_home, away_sp_R, home_sp_R)
    opp_sp_R = pd.Series(opp_sp_R, index=df.index)

    batter_R = stand.eq("R").astype("int8")
    batter_L = stand.eq("L").astype("int8")
    # Same-hand: (R batter vs R pitcher) or (L batter vs L pitcher).
    same_hand = ((batter_R & (opp_sp_R == 1)) |
                 (batter_L & (opp_sp_R == 0))).astype("float64")
    # If opp_sp_R is NaN, mark same_hand NaN (unknown).
    same_hand = same_hand.where(opp_sp_R.notna(), other=np.nan)

    blowout = (elo.abs() / 200.0).clip(upper=1.0)
    fragility = 0.35 * same_hand.fillna(0.5) + 0.65 * blowout
    df["platoon_same_hand"] = same_hand
    df["lineup_fragility"] = fragility
    return df


# Batting-order-slot -> expected PAs (pre-game heuristic).
_EXP_PA_MAP = {1: 4.6, 2: 4.4, 3: 4.2, 4: 4.0, 5: 3.8,
               6: 3.6, 7: 3.4, 8: 3.2, 9: 3.0}


def _compute_exp_pa_heuristic(df: pd.DataFrame) -> pd.DataFrame:
    bo = pd.to_numeric(df.get("batting_order"), errors="coerce")
    df["exp_pa_heuristic"] = bo.map(_EXP_PA_MAP).astype("float64")
    return df


_BULLPEN_BURN = Path("data/batter_features/bullpen_burn_by_game.parquet")


def _join_bullpen_burn(df: pd.DataFrame) -> pd.DataFrame:
    """Join 3-day rolling HL-reliever pitch count (pre-game safe) by
    (game_date, home_park_id). Game-level feature: same for all batters
    in the same game."""
    if not _BULLPEN_BURN.exists():
        print(f"  [warn] {_BULLPEN_BURN} missing; bullpen_burn_3d will be null")
        df["bullpen_burn_3d"] = float("nan")
        return df
    bb = pd.read_parquet(_BULLPEN_BURN,
                          columns=["game_date", "home_team", "bullpen_burn_3d"])
    bb["game_date"] = pd.to_datetime(bb["game_date"])
    bb = bb.rename(columns={"home_team": "home_park_id"})
    # Deduplicate: one row per (game_date, home_park_id) — keep max burn
    bb = bb.groupby(["game_date", "home_park_id"], as_index=False)["bullpen_burn_3d"].max()
    gd = pd.to_datetime(df["game_date"])
    park = df["home_park_id"]
    key_df = pd.DataFrame({"game_date": gd, "home_park_id": park})
    merged = key_df.merge(bb, on=["game_date", "home_park_id"], how="left")
    df["bullpen_burn_3d"] = merged["bullpen_burn_3d"].values
    return df


# Altitude per park (from stadium_metadata); preloaded at module level lazily.
_ALT_MAP: dict[str, float] = {}


def _load_alt_map() -> dict[str, float]:
    global _ALT_MAP
    if _ALT_MAP:
        return _ALT_MAP
    import json
    meta_path = Path("config/stadium_metadata.json")
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    _ALT_MAP = {p: v.get("altitude_ft", 0.0)
                for p, v in meta.get("parks", {}).items()}
    return _ALT_MAP


def _compute_thermal_expansion(df: pd.DataFrame) -> pd.DataFrame:
    """thermal_expansion_2.0 = air density interaction.

    Formula: (temp_f - 72) * altitude_factor
    altitude_factor = 1 + altitude_ft / 5280  (ratio above sea level)
    Humidity is flagged as a gap — not in any local data source.

    Physics: higher altitude + higher temp → lower air density → more carry.
    This is a multiplicative effect: Coors at 100F swings harder than Tropicana.
    """
    alt_map = _load_alt_map()
    import numpy as np
    temp = pd.to_numeric(df.get("temp_f"), errors="coerce")
    alt = df["home_park_id"].map(alt_map).fillna(0.0).astype("float64")
    alt_factor = 1.0 + alt / 5280.0
    df["altitude_ft"] = alt
    df["thermal_expansion"] = (temp - 72.0) * alt_factor
    return df


_XWOBA_REGRESSION_THRESHOLD = 0.100   # wOBA - xwOBA delta to flag regression candidate


def _compute_rolling_xwoba_delta(df: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Compute rolling 7-day xwOBA delta per batter.

    delta = actual_woba_7d - xwoba_7d
    regression_candidate = 1 if delta > 0.100 (primary anchor for UNDER TB props).

    Sources game-grain estimated_woba and woba_value from statcast per year.
    Degrades to null columns if statcast is unavailable.
    """
    import numpy as np

    df["rolling_7d_xwoba_delta"] = np.nan
    df["regression_candidate"]   = np.int8(0)

    needed = ["batter", "game_pk", "game_date",
              "estimated_woba_using_speedangle", "woba_value", "woba_denom"]
    frames = []
    for yr in years:
        sc_path = STATCAST_DIR / f"statcast_{yr}.parquet"
        if not sc_path.exists():
            continue
        try:
            sc = pd.read_parquet(sc_path, columns=needed)
        except Exception:
            try:
                sc = pd.read_parquet(sc_path)
                sc = sc[[c for c in needed if c in sc.columns]]
            except Exception:
                continue
        if "batter" not in sc.columns:
            continue
        sc["game_date"] = pd.to_datetime(sc["game_date"])
        frames.append(sc)

    if not frames:
        return df

    sc_all = pd.concat(frames, ignore_index=True)
    sc_all["xwoba_val"] = pd.to_numeric(
        sc_all.get("estimated_woba_using_speedangle"), errors="coerce")
    sc_all["woba_val"]  = pd.to_numeric(sc_all.get("woba_value"),  errors="coerce")
    sc_all["woba_denom"] = pd.to_numeric(sc_all.get("woba_denom"), errors="coerce")

    # Aggregate to per-batter per-game (PA-weighted means)
    pa = sc_all[sc_all["woba_denom"].fillna(0) > 0].copy()
    if pa.empty:
        return df

    game_grain = (pa.groupby(["batter", "game_date"])
                    .agg(xwoba_game=("xwoba_val", "mean"),
                         woba_game=("woba_val",   "mean"),
                         n_pa=("woba_denom",      "sum"))
                    .reset_index())
    game_grain = game_grain.sort_values(["batter", "game_date"])

    # 7-day rolling averages per batter
    rolling_rows = []
    for batter_id, grp in game_grain.groupby("batter"):
        grp = grp.set_index("game_date").sort_index()
        full_idx = pd.date_range(grp.index.min(), grp.index.max(), freq="D")
        grp = grp.reindex(full_idx)
        xw7 = grp["xwoba_game"].rolling("7D", min_periods=1).mean()
        w7  = grp["woba_game"].rolling("7D",  min_periods=1).mean()
        delta7 = w7 - xw7
        orig = game_grain[game_grain["batter"] == batter_id]["game_date"]
        for gd in orig:
            if gd in delta7.index:
                rolling_rows.append({
                    "player_id":              int(batter_id),
                    "game_date":              gd,
                    "rolling_7d_xwoba_delta": float(delta7.loc[gd])
                    if not pd.isna(delta7.loc[gd]) else np.nan,
                })

    if not rolling_rows:
        return df

    roll_df = pd.DataFrame(rolling_rows)
    roll_df["game_date"] = pd.to_datetime(roll_df["game_date"])

    # Join onto base matrix
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    before = len(df)
    df = df.merge(roll_df, on=["player_id", "game_date"], how="left")
    assert len(df) == before, "rolling xwOBA delta merge expanded rows"

    df["regression_candidate"] = (
        df["rolling_7d_xwoba_delta"].fillna(0) > _XWOBA_REGRESSION_THRESHOLD
    ).astype("int8")

    n_reg = df["regression_candidate"].sum()
    cov   = df["rolling_7d_xwoba_delta"].notna().mean()
    print(f"  [xwoba_delta] coverage={cov:.1%}  regression_candidates={n_reg:,}")
    return df


def audit(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("BATTER FEATURE MATRIX AUDIT")
    print("=" * 70)
    print(f"Rows:              {len(df):,}")
    if "stand" in df.columns:
        miss_stand = df["stand"].isna().sum() + (df["stand"] == "").sum()
        print(f"Missing 'stand':   {miss_stand:,} "
              f"({miss_stand/max(len(df),1)*100:.1f}%)")
    if "batting_order" in df.columns:
        miss_slot = df["batting_order"].isna().sum()
        print(f"Missing slot:      {miss_slot:,} "
              f"({miss_slot/max(len(df),1)*100:.1f}%)")
    if "projected_total_adj" in df.columns:
        miss_env = df["projected_total_adj"].isna().sum()
        print(f"Missing env prior: {miss_env:,} "
              f"({miss_env/max(len(df),1)*100:.1f}%)")
        if "env_source" in df.columns:
            print(f"  env_source breakdown:")
            print(df["env_source"].fillna("(missing)").value_counts()
                  .to_string().replace("\n", "\n    "))
    if "pull_side_wind_vector" in df.columns:
        psv = pd.to_numeric(df["pull_side_wind_vector"], errors="coerce")
        print(f"pull_side_wind_vector coverage: "
              f"non-null={psv.notna().sum():,}  "
              f"mean={psv.mean():.3f}  std={psv.std():.3f}  "
              f"min={psv.min():.2f}  max={psv.max():.2f}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)[:20]}"
          f"{' ...' if len(df.columns)>20 else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = assemble(args.years, dry_run=args.dry_run)
    audit(df)


if __name__ == "__main__":
    main()
