"""
build_team_stats_2026.py
========================
Compute 2026 season-to-date team batting & bullpen stats used at inference time.

Outputs  data/statcast/team_stats_2026.parquet  with one row per team (30 rows):

  batting_team               - MLB abbreviation
  bat_xwoba_vs_rhp           - team xwOBA vs right-handed pitchers
  bat_xwoba_vs_lhp           - team xwOBA vs left-handed pitchers
  bat_k_vs_rhp / bat_k_vs_lhp
  bat_bb_vs_rhp / bat_bb_vs_lhp
  team_rs_per_game           - runs scored per game (season-to-date)
  bullpen_era                - team bullpen ERA (season-to-date)
  bullpen_xwoba              - bullpen ERA converted to xwOBA equivalent

Run:  python build_team_stats_2026.py
      (called automatically by run_today.py if parquet is stale)
"""
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


def build(verbose: bool = True) -> pd.DataFrame:
    # ── 1. Team batting splits from statcast ──────────────────────────────────
    sc_path = DATA_DIR / "statcast_2026.parquet"
    if not sc_path.exists():
        raise FileNotFoundError(f"{sc_path} not found — run supplemental_pull.py first")

    sc = pd.read_parquet(sc_path, engine="pyarrow",
                         columns=["game_date", "home_team", "away_team",
                                  "inning_topbot", "p_throws",
                                  "estimated_woba_using_speedangle",
                                  "woba_denom", "events", "bb_type"])

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
        print(f"    Date range in statcast: {sc['game_date'].min()} – {sc['game_date'].max()}")

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
        bp["era"]  = pd.to_numeric(bp["era"], errors="coerce")
        bp["outs"] = pd.to_numeric(bp["outs"], errors="coerce")
        bp["Team"] = bp["Team"].map(TEAM_MAP).fillna(bp["Team"])

        # Weight by outs pitched → team bullpen ERA
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
            print(f"  Bullpen ERA: {len(bp_team)} teams | "
                  f"mean ERA: {bp_team['bullpen_era'].mean():.2f}")
    else:
        bat["bullpen_era"]   = np.nan
        bat["bullpen_xwoba"] = LEAGUE_XWOBA
        if verbose:
            print("  bullpen_fg_2026 not found — using league-avg bullpen")

    # Fill missing values with league averages
    bat["bat_xwoba_vs_rhp"] = bat["bat_xwoba_vs_rhp"].fillna(LEAGUE_XWOBA)
    bat["bat_xwoba_vs_lhp"] = bat["bat_xwoba_vs_lhp"].fillna(LEAGUE_XWOBA)
    bat["bat_k_vs_rhp"]     = bat["bat_k_vs_rhp"].fillna(0.225)
    bat["bat_k_vs_lhp"]     = bat["bat_k_vs_lhp"].fillna(0.225)
    bat["bat_bb_vs_rhp"]    = bat["bat_bb_vs_rhp"].fillna(0.085)
    bat["bat_bb_vs_lhp"]    = bat["bat_bb_vs_lhp"].fillna(0.085)
    bat["team_rs_per_game"] = bat["team_rs_per_game"].fillna(LEAGUE_RS_PER_GAME)
    bat["bullpen_era"]       = bat["bullpen_era"].fillna(4.00)
    bat["bullpen_xwoba"]     = bat["bullpen_xwoba"].fillna(LEAGUE_XWOBA)

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
