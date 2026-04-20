"""
build_team_stats_2026.py
========================
Compute 2026 season-to-date team batting & bullpen stats used at inference time.

Outputs:
  data/statcast/team_stats_2026.parquet    — one row per team (30 rows)
    batting_team               - MLB abbreviation
    bat_xwoba_vs_rhp/lhp       - season-to-date xwOBA vs RHP / LHP
    bat_xwoba_vs_rhp_10d/lhp_10d - trailing-10-calendar-day xwOBA
    bat_k_vs_rhp / bat_k_vs_lhp
    bat_bb_vs_rhp / bat_bb_vs_lhp
    team_rs_per_game           - runs scored per game (season-to-date)
    bullpen_era / bullpen_whip / bullpen_k9 / bullpen_xwoba

  data/statcast/pitcher_10d_2026.parquet  — one row per SP
    pitcher_name_upper         - normalised "FIRST LAST" key
    k_pct_10d / xwoba_10d      - trailing-10-calendar-day averages over SP starts

Run:  python build_team_stats_2026.py
      (called automatically by run_today.py if parquet is stale)
"""
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("./data/statcast")

# League-average xwOBA (2024 baseline)
LEAGUE_XWOBA = 0.318

# League-average runs per game (2024 baseline, both halves)
LEAGUE_RS_PER_GAME = 4.38

# ERA → xwOBA linear approximation
# ERA 3.50 ≈ xwOBA 0.295  |  ERA 4.50 ≈ xwOBA 0.325
ERA_TO_XWOBA_INTERCEPT = 0.218
ERA_TO_XWOBA_SLOPE     = 0.024


def era_to_xwoba(era: float) -> float:
    """Convert bullpen ERA to xwOBA-against equivalent."""
    if pd.isna(era) or era <= 0:
        return LEAGUE_XWOBA
    return float(np.clip(ERA_TO_XWOBA_INTERCEPT + ERA_TO_XWOBA_SLOPE * era,
                         0.220, 0.420))


def _normalize_pitcher_name(raw: str) -> str:
    """
    Convert 'Last, First' (possibly with accents) → 'FIRST LAST' uppercase.
    Matches the pitcher_name_upper key used in pitcher profiles.
    """
    name = str(raw).strip()
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        name = f"{parts[1]} {parts[0]}"
    name = name.upper()
    # Strip diacritics (é → E, ñ → N, etc.)
    return "".join(
        c for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    )


def build(verbose: bool = True) -> pd.DataFrame:
    # ── 1. Team batting splits from statcast ──────────────────────────────────
    sc_path = DATA_DIR / "statcast_2026.parquet"
    if not sc_path.exists():
        raise FileNotFoundError(f"{sc_path} not found — run supplemental_pull.py first")

    sc = pd.read_parquet(sc_path, engine="pyarrow",
                         columns=["game_date", "game_pk", "home_team", "away_team",
                                  "inning", "inning_topbot", "p_throws",
                                  "player_name", "pitcher",
                                  "estimated_woba_using_speedangle",
                                  "woba_denom", "events", "bb_type"])

    sc["game_date"] = pd.to_datetime(sc["game_date"])
    sc["xwoba"]    = pd.to_numeric(sc["estimated_woba_using_speedangle"], errors="coerce")
    sc["is_pa"]    = pd.to_numeric(sc["woba_denom"], errors="coerce") > 0
    sc["k"]        = (sc["events"] == "strikeout").astype(float)
    sc["bb"]       = (sc["events"] == "walk").astype(float)

    # batting_team: Top of inning = away team bats; Bot = home team bats
    sc["batting_team"] = np.where(
        sc["inning_topbot"] == "Top", sc["away_team"], sc["home_team"])

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
        print(f"  Batting splits: {len(bat)} teams computed from statcast")
        print(f"    Date range in statcast: {sc['game_date'].min().date()} – "
              f"{sc['game_date'].max().date()}")

    # ── 1b. Trailing-10-day batting splits ────────────────────────────────────
    latest_date = sc["game_date"].max()
    cutoff_10d  = latest_date - pd.Timedelta(days=10)
    sc_10d = sc[(sc["game_date"] > cutoff_10d) & (sc["game_date"] <= latest_date)]
    pa_10d = sc_10d[sc_10d["is_pa"]].copy()

    if len(pa_10d) > 0:
        splits_10d = pa_10d.groupby(["batting_team", "p_throws"]).agg(
            xwoba_10d = ("xwoba", "mean"),
        ).reset_index()

        def hand_cols_10d(hand, suffix):
            sub = splits_10d[splits_10d["p_throws"] == hand].rename(columns={
                "xwoba_10d": f"bat_xwoba_vs_{suffix}_10d",
            }).drop(columns="p_throws")
            return sub

        rhp_10d = hand_cols_10d("R", "rhp")
        lhp_10d = hand_cols_10d("L", "lhp")
        bat_10d = rhp_10d.merge(lhp_10d, on="batting_team", how="outer")
        bat = bat.merge(bat_10d, on="batting_team", how="left")
        if verbose:
            n_teams_10d = len(bat_10d)
            print(f"  Batting 10d splits: {n_teams_10d} teams "
                  f"({cutoff_10d.date()} – {latest_date.date()})")
    else:
        bat["bat_xwoba_vs_rhp_10d"] = np.nan
        bat["bat_xwoba_vs_lhp_10d"] = np.nan
        if verbose:
            print("  Batting 10d splits: no data in window")

    # ── 1c. Trailing-10-day SP stats (k_pct_10d, xwoba_10d per pitcher) ───────
    # Identify SPs: the pitcher who appears in inning 1 for each half-inning.
    sp_ids = (
        sc[sc["inning"] == 1]
        .drop_duplicates(subset=["game_pk", "inning_topbot"], keep="first")
        [["game_pk", "game_date", "inning_topbot", "pitcher", "player_name"]]
        .copy()
    )
    sp_ids["pitcher_name_upper"] = sp_ids["player_name"].apply(_normalize_pitcher_name)

    # Join SP identity back onto all pitches, keeping only their full outing
    sp_pitches = sc.merge(
        sp_ids[["game_pk", "inning_topbot", "pitcher"]].rename(
            columns={"pitcher": "sp_pitcher_id"}),
        on=["game_pk", "inning_topbot"],
        how="inner",
    )
    sp_pitches = sp_pitches[sp_pitches["pitcher"] == sp_pitches["sp_pitcher_id"]]

    # Aggregate to per-SP per-game level
    sp_games = (
        sp_pitches[sp_pitches["is_pa"]]
        .groupby(["pitcher", "game_pk", "game_date"]).agg(
            k_pct_game  = ("k",     "mean"),
            xwoba_game  = ("xwoba", "mean"),
            n_pa        = ("is_pa", "sum"),
        ).reset_index()
    )
    # Attach normalised name
    sp_games = sp_games.merge(
        sp_ids[["game_pk", "pitcher", "pitcher_name_upper"]].drop_duplicates(),
        on=["game_pk", "pitcher"],
        how="left",
    )

    # Trailing 10 days from latest date
    sp_games_10d = sp_games[
        (sp_games["game_date"] > cutoff_10d) &
        (sp_games["game_date"] <= latest_date) &
        sp_games["pitcher_name_upper"].notna()
    ]

    if len(sp_games_10d) > 0:
        pitcher_10d = (
            sp_games_10d.groupby("pitcher_name_upper").agg(
                k_pct_10d  = ("k_pct_game",  "mean"),
                xwoba_10d  = ("xwoba_game",  "mean"),
                n_starts   = ("game_pk",      "nunique"),
            ).reset_index()
        )
        out_sp = DATA_DIR / "pitcher_10d_2026.parquet"
        pitcher_10d.to_parquet(out_sp, engine="pyarrow", index=False)
        if verbose:
            print(f"  SP 10d stats: {len(pitcher_10d)} pitchers -> {out_sp}")
    else:
        pitcher_10d = pd.DataFrame(columns=["pitcher_name_upper", "k_pct_10d", "xwoba_10d"])
        if verbose:
            print("  SP 10d stats: no data in window")

    # ── 2. Team runs scored per game from statcast schedule ───────────────────
    sched_path = DATA_DIR / "schedule_all_2026.parquet"
    try:
        sched = pd.read_parquet(sched_path, engine="pyarrow")
        sched = sched[sched["home_away"] == "Home"][
            ["home_team", "away_team", "home_score", "away_score"]
        ].copy()
        sched["home_score"] = pd.to_numeric(sched["home_score"], errors="coerce")
        sched["away_score"] = pd.to_numeric(sched["away_score"], errors="coerce")
        sched = sched.dropna(subset=["home_score", "away_score"])

        # Runs scored per team per game
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
            print(f"  RS/G: {len(rs)} teams | league avg: {rs['team_rs_per_game'].mean():.2f}")
    except FileNotFoundError:
        bat["team_rs_per_game"] = LEAGUE_RS_PER_GAME
        if verbose:
            print("  schedule_all_2026 not found — using league-avg RS/G")

    # ── 3. Bullpen ERA from bullpen_fg_2026 ───────────────────────────────────
    bp_path = DATA_DIR / "bullpen_fg_2026.parquet"

    # MLB team abbreviation map (FanGraphs uses full names or slightly different abbrevs)
    TEAM_MAP = {
        "Angels": "LAA", "Astros": "HOU", "Athletics": "ATH", "Blue Jays": "TOR",
        "Braves": "ATL", "Brewers": "MIL", "Cardinals": "STL", "Cubs": "CHC",
        "Diamondbacks": "AZ", "Dodgers": "LAD", "Giants": "SF", "Guardians": "CLE",
        "Mariners": "SEA", "Marlins": "MIA", "Mets": "NYM", "Nationals": "WSH",
        "Orioles": "BAL", "Padres": "SD", "Phillies": "PHI", "Pirates": "PIT",
        "Rangers": "TEX", "Rays": "TB", "Red Sox": "BOS", "Reds": "CIN",
        "Rockies": "COL", "Royals": "KC", "Tigers": "DET", "Twins": "MIN",
        "White Sox": "CWS", "Yankees": "NYY",
        # short codes that may already be there
        "LAA": "LAA", "HOU": "HOU", "ATH": "ATH", "OAK": "ATH",
        "TOR": "TOR", "ATL": "ATL", "MIL": "MIL", "STL": "STL",
        "CHC": "CHC", "ARI": "AZ", "AZ": "AZ", "LAD": "LAD",
        "SF": "SF", "CLE": "CLE", "SEA": "SEA", "MIA": "MIA",
        "NYM": "NYM", "WSH": "WSH", "BAL": "BAL", "SD": "SD",
        "PHI": "PHI", "PIT": "PIT", "TEX": "TEX", "TB": "TB",
        "BOS": "BOS", "CIN": "CIN", "COL": "COL", "KC": "KC",
        "DET": "DET", "MIN": "MIN", "CWS": "CWS", "NYY": "NYY",
    }

    if bp_path.exists():
        bp = pd.read_parquet(bp_path, engine="pyarrow")
        bp["era"]  = pd.to_numeric(bp["era"],  errors="coerce")
        bp["whip"] = pd.to_numeric(bp["whip"], errors="coerce")
        bp["k9"]   = pd.to_numeric(bp["strikeoutsPer9Inn"], errors="coerce")
        bp["outs"] = pd.to_numeric(bp["outs"], errors="coerce")
        bp["Team"] = bp["Team"].map(TEAM_MAP).fillna(bp["Team"])

        # Weight by outs pitched → team-level bullpen ERA / WHIP / K9
        def _wavg(g, col):
            w     = g["outs"].clip(lower=0)
            total = w.sum()
            return (g[col] * w).sum() / total if total > 0 else np.nan

        bp_team = bp.groupby("Team").apply(
            lambda g: pd.Series({
                "bullpen_era":  _wavg(g, "era"),
                "bullpen_whip": _wavg(g, "whip"),
                "bullpen_k9":   _wavg(g, "k9"),
            })
        ).reset_index().rename(columns={"Team": "batting_team"})
        bp_team["bullpen_xwoba"] = bp_team["bullpen_era"].apply(era_to_xwoba)

        bat = bat.merge(bp_team, on="batting_team", how="left")
        if verbose:
            print(f"  Bullpen ERA/WHIP/K9: {len(bp_team)} teams | "
                  f"ERA: {bp_team['bullpen_era'].mean():.2f}  "
                  f"WHIP: {bp_team['bullpen_whip'].mean():.2f}  "
                  f"K9: {bp_team['bullpen_k9'].mean():.1f}")
    else:
        bat["bullpen_era"]   = np.nan
        bat["bullpen_whip"]  = np.nan
        bat["bullpen_k9"]    = np.nan
        bat["bullpen_xwoba"] = LEAGUE_XWOBA
        if verbose:
            print("  bullpen_fg_2026 not found — using league-avg bullpen")

    # Fill missing values with league averages
    bat["bat_xwoba_vs_rhp"]     = bat["bat_xwoba_vs_rhp"].fillna(LEAGUE_XWOBA)
    bat["bat_xwoba_vs_lhp"]     = bat["bat_xwoba_vs_lhp"].fillna(LEAGUE_XWOBA)
    bat["bat_xwoba_vs_rhp_10d"] = bat["bat_xwoba_vs_rhp_10d"].fillna(LEAGUE_XWOBA)
    bat["bat_xwoba_vs_lhp_10d"] = bat["bat_xwoba_vs_lhp_10d"].fillna(LEAGUE_XWOBA)
    bat["bat_k_vs_rhp"]     = bat["bat_k_vs_rhp"].fillna(0.225)
    bat["bat_k_vs_lhp"]     = bat["bat_k_vs_lhp"].fillna(0.225)
    bat["bat_bb_vs_rhp"]    = bat["bat_bb_vs_rhp"].fillna(0.085)
    bat["bat_bb_vs_lhp"]    = bat["bat_bb_vs_lhp"].fillna(0.085)
    bat["team_rs_per_game"] = bat["team_rs_per_game"].fillna(LEAGUE_RS_PER_GAME)
    bat["bullpen_era"]        = bat["bullpen_era"].fillna(4.00)
    bat["bullpen_whip"]       = bat["bullpen_whip"].fillna(1.28)
    bat["bullpen_k9"]         = bat["bullpen_k9"].fillna(8.5)
    bat["bullpen_xwoba"]      = bat["bullpen_xwoba"].fillna(LEAGUE_XWOBA)

    # ── 4. Rolling 15-game run differential + Pythagorean win% ───────────────
    # Used as features in the XGBoost full-game model (v2 feature set).
    # Computed from actuals_2026.parquet (game-by-game results).
    try:
        act_path = DATA_DIR / "actuals_2026.parquet"
        act = pd.read_parquet(act_path, engine="pyarrow",
                              columns=["game_date", "home_team", "away_team",
                                       "home_score_final", "away_score_final"])
        act["game_date"] = pd.to_datetime(act["game_date"])
        act["home_score"] = pd.to_numeric(act["home_score_final"], errors="coerce")
        act["away_score"] = pd.to_numeric(act["away_score_final"], errors="coerce")
        act = act.dropna(subset=["home_score", "away_score"])

        # Build per-team per-game frame: team, date, rs (scored), ra (allowed)
        home_g = act[["game_date", "home_team", "home_score", "away_score"]].rename(
            columns={"home_team": "team", "home_score": "rs", "away_score": "ra"})
        away_g = act[["game_date", "away_team", "away_score", "home_score"]].rename(
            columns={"away_team": "team", "away_score": "rs", "home_score": "ra"})
        games_long = pd.concat([home_g, away_g], ignore_index=True)
        games_long = games_long.sort_values(["team", "game_date"])

        rd_rows = []
        for team, grp in games_long.groupby("team"):
            grp = grp.set_index("game_date").sort_index()
            rs = grp["rs"]; ra = grp["ra"]
            # Rolling 15-game sums (trailing, no leakage — shift(1) excludes today)
            rs15 = rs.shift(1).rolling(15, min_periods=5).sum()
            ra15 = ra.shift(1).rolling(15, min_periods=5).sum()
            rd15 = (rs15 - ra15) / 15  # per-game average
            # Pythagorean win% = RS^1.83 / (RS^1.83 + RA^1.83)
            with np.errstate(divide="ignore", invalid="ignore"):
                rs_p = np.power(rs15.clip(lower=0.001), 1.83)
                ra_p = np.power(ra15.clip(lower=0.001), 1.83)
                pyth = rs_p / (rs_p + ra_p)
            rd_rows.append({"batting_team": team,
                            "rolling_rd_15g":      float(rd15.iloc[-1]) if not rd15.empty else 0.0,
                            "pyth_win_pct_15g":    float(pyth.iloc[-1]) if not pyth.empty else 0.5})

        rd_df = pd.DataFrame(rd_rows)
        bat = bat.merge(rd_df, on="batting_team", how="left")
        if verbose:
            present = bat["rolling_rd_15g"].notna().sum()
            print(f"  Rolling RD/Pyth: {present}/{len(bat)} teams computed from actuals")
    except Exception as _e:
        if verbose:
            print(f"  [WARN] rolling_rd/pyth computation failed: {_e} — using 0/0.5 defaults")

    bat["rolling_rd_15g"]   = bat.get("rolling_rd_15g",   pd.Series([0.0]   * len(bat))).fillna(0.0)
    bat["pyth_win_pct_15g"] = bat.get("pyth_win_pct_15g", pd.Series([0.5]   * len(bat))).fillna(0.5)

    out_path = DATA_DIR / "team_stats_2026.parquet"
    bat.to_parquet(out_path, engine="pyarrow", index=False)
    if verbose:
        print(f"\n  Saved -> {out_path}  ({len(bat)} teams x {len(bat.columns)} cols)")
        print("\n  Sample (top 5 by RS/G):")
        show_cols = ["batting_team", "bat_xwoba_vs_rhp", "bat_xwoba_vs_lhp",
                     "team_rs_per_game", "bullpen_era", "bullpen_xwoba"]
        show_cols = [c for c in show_cols if c in bat.columns]
        print(bat.sort_values("team_rs_per_game", ascending=False)
                 [show_cols].head(5).to_string(index=False))

    return bat


if __name__ == "__main__":
    print("=" * 55)
    print("  build_team_stats_2026.py")
    print("=" * 55)
    build(verbose=True)
