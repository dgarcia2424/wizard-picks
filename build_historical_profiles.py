"""
build_historical_profiles.py
=============================
Build pitcher profiles and team stats for 2023, 2024, 2025 seasons.
These parallel the 2026 files used at inference time.

Outputs (for each year in [2023, 2024, 2025]):
  data/statcast/pitcher_profiles_{year}.parquet
  data/statcast/team_stats_{year}.parquet

Usage:
  python build_historical_profiles.py
  python build_historical_profiles.py --years 2024 2025
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path("./data/statcast")

# League-average constants (matching build_team_stats_2026.py)
LEAGUE_XWOBA = 0.318
LEAGUE_RS_PER_GAME = 4.38
ERA_TO_XWOBA_INTERCEPT = 0.218
ERA_TO_XWOBA_SLOPE = 0.024

FASTBALL_TYPES = {"FF", "SI", "FC"}

TEAM_MAP = {
    "Angels": "LAA", "Astros": "HOU", "Athletics": "ATH", "Blue Jays": "TOR",
    "Braves": "ATL", "Brewers": "MIL", "Cardinals": "STL", "Cubs": "CHC",
    "Diamondbacks": "AZ", "Dodgers": "LAD", "Giants": "SF", "Guardians": "CLE",
    "Mariners": "SEA", "Marlins": "MIA", "Mets": "NYM", "Nationals": "WSH",
    "Orioles": "BAL", "Padres": "SD", "Phillies": "PHI", "Pirates": "PIT",
    "Rangers": "TEX", "Rays": "TB", "Red Sox": "BOS", "Reds": "CIN",
    "Rockies": "COL", "Royals": "KC", "Tigers": "DET", "Twins": "MIN",
    "White Sox": "CWS", "Yankees": "NYY",
    # short codes
    "LAA": "LAA", "HOU": "HOU", "ATH": "ATH", "OAK": "ATH",
    "TOR": "TOR", "ATL": "ATL", "MIL": "MIL", "STL": "STL",
    "CHC": "CHC", "ARI": "AZ", "AZ": "AZ", "LAD": "LAD",
    "SF": "SF", "CLE": "CLE", "SEA": "SEA", "MIA": "MIA",
    "NYM": "NYM", "WSH": "WSH", "BAL": "BAL", "SD": "SD",
    "PHI": "PHI", "PIT": "PIT", "TEX": "TEX", "TB": "TB",
    "BOS": "BOS", "CIN": "CIN", "COL": "COL", "KC": "KC",
    "DET": "DET", "MIN": "MIN", "CWS": "CWS", "NYY": "NYY",
}


def era_to_xwoba(era: float) -> float:
    """Convert bullpen ERA to xwOBA-against equivalent."""
    if pd.isna(era) or era <= 0:
        return LEAGUE_XWOBA
    return float(np.clip(ERA_TO_XWOBA_INTERCEPT + ERA_TO_XWOBA_SLOPE * era,
                         0.220, 0.420))


# ---------------------------------------------------------------------------
# PITCHER PROFILES
# ---------------------------------------------------------------------------

def build_pitcher_profiles(year: int, verbose: bool = True) -> pd.DataFrame:
    """
    Build pitcher_profiles_{year}.parquet from statcast + pitcher_percentiles.

    Each row: one pitcher with full-season aggregated stats.
    Columns mirror what the XGBoost feature matrix expects.
    """
    if verbose:
        print(f"\n  [pitcher_profiles_{year}] Loading statcast_{year}.parquet ...")

    sc_path = DATA_DIR / f"statcast_{year}.parquet"
    if not sc_path.exists():
        raise FileNotFoundError(f"{sc_path} not found")

    # Load only the columns we need to save memory
    needed_cols = [
        "pitcher", "player_name", "p_throws",
        "estimated_woba_using_speedangle", "woba_denom",
        "events", "bb_type",
        "delta_pitcher_run_exp",
        "release_speed", "pitch_type",
        "age_pit", "arm_angle",
        "game_date",
    ]
    # Only load columns that actually exist in the parquet
    import pyarrow.parquet as pq
    pq_schema = pq.read_schema(sc_path)
    available = set(pq_schema.names)
    load_cols = [c for c in needed_cols if c in available]

    sc = pd.read_parquet(sc_path, engine="pyarrow", columns=load_cols)
    sc["game_date"] = pd.to_datetime(sc["game_date"], errors="coerce")

    if verbose:
        print(f"    Rows: {len(sc):,}  |  Date range: "
              f"{sc['game_date'].min()} – {sc['game_date'].max()}")

    # ── Numeric coercions ───────────────────────────────────────────────────
    sc["xwoba"]    = pd.to_numeric(sc.get("estimated_woba_using_speedangle"), errors="coerce")
    sc["woba_den"] = pd.to_numeric(sc.get("woba_denom"), errors="coerce")
    sc["is_pa"]    = sc["woba_den"] > 0
    sc["delta_pre"]= pd.to_numeric(sc.get("delta_pitcher_run_exp"), errors="coerce")
    sc["rel_spd"]  = pd.to_numeric(sc.get("release_speed"), errors="coerce")
    sc["age_pit"]  = pd.to_numeric(sc.get("age_pit"), errors="coerce")
    sc["arm_angle"]= pd.to_numeric(sc.get("arm_angle"), errors="coerce")

    # ── Pitcher name map (pitcher_id → "Last, First") ───────────────────────
    name_map = (
        sc[["pitcher", "player_name"]]
        .dropna(subset=["player_name"])
        .drop_duplicates("pitcher")
        .set_index("pitcher")["player_name"]
        .to_dict()
    )

    # ── PA-level data ───────────────────────────────────────────────────────
    pa = sc[sc["is_pa"]].copy()

    # k / bb / xwoba / gb
    pa["is_k"]  = (pa["events"] == "strikeout").astype(float)
    pa["is_bb"] = (pa["events"] == "walk").astype(float)
    pa["is_bip"]= pa["bb_type"].notna().astype(float)
    pa["is_gb"] = (pa["bb_type"] == "ground_ball").astype(float)

    agg_pa = pa.groupby("pitcher").agg(
        n_pa       = ("is_pa",  "sum"),
        k_pct      = ("is_k",   "mean"),
        bb_pct     = ("is_bb",  "mean"),
        xwoba      = ("xwoba",  "mean"),
        gb_raw     = ("is_gb",  "sum"),
        bip_raw    = ("is_bip", "sum"),
    ).reset_index()

    agg_pa["gb_pct"] = agg_pa["gb_raw"] / agg_pa["bip_raw"].clip(lower=1)
    agg_pa["k_minus_bb"] = agg_pa["k_pct"] - agg_pa["bb_pct"]

    # xRV per pitch
    agg_xrv = sc.groupby("pitcher")["delta_pre"].mean().reset_index()
    agg_xrv.columns = ["pitcher", "xrv_per_pitch"]

    # Fastball velocity
    fb = sc[sc["pitch_type"].isin(FASTBALL_TYPES)].copy()
    agg_fb = fb.groupby("pitcher")["rel_spd"].mean().reset_index()
    agg_fb.columns = ["pitcher", "ff_velo"]

    # handedness (mode)
    hand = (
        sc[sc["p_throws"].notna()]
        .groupby("pitcher")["p_throws"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "R")
        .reset_index()
    )
    hand.columns = ["pitcher", "p_throws"]

    # age & arm_angle (mean per pitcher)
    age_aa = sc.groupby("pitcher").agg(
        age_pit   = ("age_pit",   "mean"),
        arm_angle = ("arm_angle", "mean"),
    ).reset_index()

    # ── Merge all pitch-level stats ─────────────────────────────────────────
    profiles = (
        agg_pa
        .merge(agg_xrv, on="pitcher", how="left")
        .merge(agg_fb,  on="pitcher", how="left")
        .merge(hand,    on="pitcher", how="left")
        .merge(age_aa,  on="pitcher", how="left")
    )

    # ── Join pitcher percentiles ────────────────────────────────────────────
    pctile_path = DATA_DIR / f"pitcher_percentiles_{year}.parquet"
    if pctile_path.exists():
        pctile = pd.read_parquet(pctile_path, engine="pyarrow")
        pctile = pctile.rename(columns={"player_id": "pitcher"})
        pctile["pitcher"] = pd.to_numeric(pctile["pitcher"], errors="coerce")
        # Keep only the columns we need
        keep_pctile = [c for c in ["pitcher", "fb_velocity", "fb_spin",
                                    "whiff_percent", "xera", "k_percent",
                                    "bb_percent"] if c in pctile.columns]
        pctile = pctile[keep_pctile].drop_duplicates("pitcher")
        profiles = profiles.merge(pctile, on="pitcher", how="left")
        if verbose:
            print(f"    Percentiles joined: {len(pctile)} pitchers")
    else:
        if verbose:
            print(f"    [WARN] pitcher_percentiles_{year}.parquet not found — skipping percentile join")
        for col in ["fb_velocity", "fb_spin", "whiff_percent", "xera"]:
            profiles[col] = np.nan

    # ── Add pitcher name ────────────────────────────────────────────────────
    profiles["pitcher_name"] = profiles["pitcher"].map(name_map)
    profiles["pitcher_name_upper"] = profiles["pitcher_name"].str.upper().fillna("")

    # ── Derived columns for XGBoost feature parity ─────────────────────────
    # The XGBoost model uses these column names via the feature matrix
    profiles = profiles.rename(columns={
        "xwoba":    "xwoba_against",   # pitcher's xwOBA against = quality metric
    })

    # era_minus_xera: available if we have xera from percentiles
    # Use k_percent from percentiles if available, otherwise computed k_pct
    if "k_percent" in profiles.columns:
        profiles["k_pct_pctl"] = profiles["k_percent"]
    if "bb_percent" in profiles.columns:
        profiles["bb_pct_pctl"] = profiles["bb_percent"]

    # era_minus_xera proxy (we don't have ERA directly, use nan)
    profiles["era_minus_xera"] = np.nan

    # ── Qualify: at least 50 PA ─────────────────────────────────────────────
    profiles = profiles[profiles["n_pa"] >= 50].copy()

    if verbose:
        print(f"    Qualified pitchers (n_pa >= 50): {len(profiles)}")

    # ── Save ────────────────────────────────────────────────────────────────
    out_path = DATA_DIR / f"pitcher_profiles_{year}.parquet"
    profiles.to_parquet(out_path, engine="pyarrow", index=False)
    if verbose:
        print(f"    Saved -> {out_path}  ({len(profiles)} pitchers x {len(profiles.columns)} cols)")

    return profiles


# ---------------------------------------------------------------------------
# TEAM STATS
# ---------------------------------------------------------------------------

def build_team_stats(year: int, verbose: bool = True) -> pd.DataFrame:
    """
    Build team_stats_{year}.parquet from batting_fg + bullpen_fg + schedule.
    Mirrors the column format of team_stats_2026.parquet.
    """
    if verbose:
        print(f"\n  [team_stats_{year}] Building team stats ...")

    # ── 1. Batting splits from statcast ─────────────────────────────────────
    sc_path = DATA_DIR / f"statcast_{year}.parquet"
    if not sc_path.exists():
        raise FileNotFoundError(f"{sc_path} not found")

    import pyarrow.parquet as pq
    pq_schema = pq.read_schema(sc_path)
    available = set(pq_schema.names)
    load_cols = [c for c in [
        "game_date", "home_team", "away_team",
        "inning_topbot", "p_throws",
        "estimated_woba_using_speedangle",
        "woba_denom", "events", "bb_type",
    ] if c in available]

    sc = pd.read_parquet(sc_path, engine="pyarrow", columns=load_cols)
    sc["xwoba"] = pd.to_numeric(sc.get("estimated_woba_using_speedangle"), errors="coerce")
    sc["is_pa"] = pd.to_numeric(sc.get("woba_denom"), errors="coerce") > 0
    sc["k"]     = (sc["events"] == "strikeout").astype(float)
    sc["bb"]    = (sc["events"] == "walk").astype(float)

    # batting_team: Top of inning = away team bats; Bot = home team bats
    if "inning_topbot" in sc.columns:
        sc["batting_team"] = np.where(
            sc["inning_topbot"] == "Top", sc["away_team"], sc["home_team"]
        )
    else:
        # Fallback: can't determine batting team without inning info
        sc["batting_team"] = sc["home_team"]

    pa = sc[sc["is_pa"]].copy()

    splits = pa.groupby(["batting_team", "p_throws"]).agg(
        n_pa   = ("is_pa",  "sum"),
        xwoba  = ("xwoba",  "mean"),
        k_pct  = ("k",      "mean"),
        bb_pct = ("bb",     "mean"),
    ).reset_index()

    def hand_cols(hand, suffix):
        sub = splits[splits["p_throws"] == hand].rename(columns={
            "xwoba":  f"bat_xwoba_vs_{suffix}",
            "k_pct":  f"bat_k_vs_{suffix}",
            "bb_pct": f"bat_bb_vs_{suffix}",
            "n_pa":   f"bat_pa_vs_{suffix}",
        }).drop(columns="p_throws")
        return sub

    rhp = hand_cols("R", "rhp")
    lhp = hand_cols("L", "lhp")
    bat = rhp.merge(lhp, on="batting_team", how="outer")

    if verbose:
        print(f"    Batting splits: {len(bat)} teams from statcast")

    # ── 2. Runs scored per game from schedule ────────────────────────────────
    sched_path = DATA_DIR / f"schedule_all_{year}.parquet"
    try:
        sched = pd.read_parquet(sched_path, engine="pyarrow")
        # Filter to home rows only (avoid double counting)
        if "home_away" in sched.columns:
            sched = sched[sched["home_away"] == "Home"]
        # Use gamePk or game_pk
        sched = sched[["home_team", "away_team", "home_score", "away_score"]].copy()
        sched["home_score"] = pd.to_numeric(sched["home_score"], errors="coerce")
        sched["away_score"] = pd.to_numeric(sched["away_score"], errors="coerce")
        sched = sched.dropna(subset=["home_score", "away_score"])

        home_rs = sched.groupby("home_team")["home_score"].agg(
            rs_total="sum", games="count").reset_index().rename(
            columns={"home_team": "batting_team"})
        away_rs = sched.groupby("away_team")["away_score"].agg(
            rs_total="sum", games="count").reset_index().rename(
            columns={"away_team": "batting_team"})
        rs = pd.concat([home_rs, away_rs]).groupby("batting_team").agg(
            rs_total=("rs_total", "sum"),
            games=("games", "sum"),
        ).reset_index()
        rs["team_rs_per_game"] = rs["rs_total"] / rs["games"].clip(lower=1)
        bat = bat.merge(rs[["batting_team", "team_rs_per_game"]], on="batting_team", how="left")
        if verbose:
            print(f"    RS/G: {len(rs)} teams | league avg: {rs['team_rs_per_game'].mean():.2f}")
    except FileNotFoundError:
        bat["team_rs_per_game"] = LEAGUE_RS_PER_GAME
        if verbose:
            print(f"    schedule_all_{year} not found — using league-avg RS/G")

    # ── 3. Bullpen ERA from bullpen_fg_{year} ────────────────────────────────
    bp_path = DATA_DIR / f"bullpen_fg_{year}.parquet"
    if bp_path.exists():
        bp = pd.read_parquet(bp_path, engine="pyarrow")
        bp["era"]  = pd.to_numeric(bp.get("era"), errors="coerce")
        bp["outs"] = pd.to_numeric(bp.get("outs"), errors="coerce")
        bp["Team"] = bp["Team"].map(TEAM_MAP).fillna(bp["Team"])

        bp_team = bp.groupby("Team").apply(
            lambda g: pd.Series({
                "bullpen_era": (
                    (g["era"] * g["outs"]).sum() / g["outs"].clip(lower=1).sum()
                    if g["outs"].sum() > 0 else np.nan
                )
            })
        ).reset_index().rename(columns={"Team": "batting_team"})
        bp_team["bullpen_xwoba"] = bp_team["bullpen_era"].apply(era_to_xwoba)

        bat = bat.merge(bp_team, on="batting_team", how="left")
        if verbose:
            print(f"    Bullpen ERA: {len(bp_team)} teams | "
                  f"mean ERA: {bp_team['bullpen_era'].mean():.2f}")
    else:
        bat["bullpen_era"]   = np.nan
        bat["bullpen_xwoba"] = LEAGUE_XWOBA
        if verbose:
            print(f"    bullpen_fg_{year} not found — using league-avg bullpen")

    # ── Fill missing values with league averages ────────────────────────────
    bat["bat_xwoba_vs_rhp"] = bat["bat_xwoba_vs_rhp"].fillna(LEAGUE_XWOBA)
    bat["bat_xwoba_vs_lhp"] = bat["bat_xwoba_vs_lhp"].fillna(LEAGUE_XWOBA)
    bat["bat_k_vs_rhp"]     = bat["bat_k_vs_rhp"].fillna(0.225)
    bat["bat_k_vs_lhp"]     = bat["bat_k_vs_lhp"].fillna(0.225)
    bat["bat_bb_vs_rhp"]    = bat["bat_bb_vs_rhp"].fillna(0.085)
    bat["bat_bb_vs_lhp"]    = bat["bat_bb_vs_lhp"].fillna(0.085)
    bat["team_rs_per_game"] = bat["team_rs_per_game"].fillna(LEAGUE_RS_PER_GAME)
    bat["bullpen_era"]       = bat["bullpen_era"].fillna(4.00)
    bat["bullpen_xwoba"]     = bat["bullpen_xwoba"].fillna(LEAGUE_XWOBA)

    # ── Save ────────────────────────────────────────────────────────────────
    out_path = DATA_DIR / f"team_stats_{year}.parquet"
    bat.to_parquet(out_path, engine="pyarrow", index=False)
    if verbose:
        print(f"    Saved -> {out_path}  ({len(bat)} teams x {len(bat.columns)} cols)")

    return bat


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(years=None):
    years = years or [2023, 2024, 2025]
    print("=" * 55)
    print("  build_historical_profiles.py")
    print("=" * 55)

    for year in years:
        print(f"\n{'='*55}")
        print(f"  Processing {year} ...")
        print(f"{'='*55}")
        try:
            build_pitcher_profiles(year, verbose=True)
        except Exception as e:
            print(f"  [ERROR] pitcher_profiles_{year}: {e}")
        try:
            build_team_stats(year, verbose=True)
        except Exception as e:
            print(f"  [ERROR] team_stats_{year}: {e}")

    print("\n  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    args = parser.parse_args()
    main(years=args.years)
