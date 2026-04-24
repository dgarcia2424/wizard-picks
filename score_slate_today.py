"""
score_slate_today.py — v3.6 unified daily TB-stacker scorer.

Pipeline:
    1. lineup_pull.py --date <today>        (starters + handedness)
    2. pull today's wind/weather per game   (feature_matrix if available,
                                              else neutral fallback with flag)
    3. grounded-env offset via get_dynamic_bias + -1.5r circuit breaker
    4. score with models/tb_stacker_v35.json
    5. write data/predictions/today_projections.csv

Output columns:
    Player | Game | Expected_TB | P_Over_1.5 | Edge
"""
from __future__ import annotations
import json
import subprocess
import sys
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_run_dist_today as srd

MODEL_PATH   = Path("models/tb_stacker_v35.json")
STADIUM_META = Path("config/stadium_metadata.json")
LINEUP_TODAY = Path("data/statcast/lineups_today_long.parquet")
FEATURE_MTX  = Path("feature_matrix_enriched_v2.parquet")
XSTATS_DIR   = Path("data/statcast")
OUT_FILE     = Path("data/predictions/today_projections.csv")

LEAGUE_BASELINE = 8.80
LEAGUE_AVG_FF_VELO = 94.19
PULL_OFFSET = 22.0
MARKET_LINE = 1.5
RETAIL_IMPLIED = 0.524

# Expected-TB calibration from 2024-2025 training matrix (by class of y).
# P(TB>1.5) -> E[TB] = p * mean_over + (1-p) * mean_under
MEAN_TB_OVER  = 3.353
MEAN_TB_UNDER = 0.361

EXP_PA_MAP = {1: 4.6, 2: 4.4, 3: 4.2, 4: 4.0, 5: 3.8,
              6: 3.6, 7: 3.4, 8: 3.2, 9: 3.0}


def fetch_schedule_from_mlb(d: str) -> pd.DataFrame:
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={d}"
    tm = ("https://statsapi.mlb.com/api/v1/teams?sportId=1")
    with urllib.request.urlopen(tm, timeout=15) as r:
        tmap = {int(t["id"]): t.get("abbreviation", "")
                for t in json.loads(r.read())["teams"]}
    with urllib.request.urlopen(url, timeout=15) as r:
        payload = json.loads(r.read())
    rows = []
    for block in payload.get("dates", []):
        for g in block.get("games", []):
            rows.append({
                "game_pk":   g.get("gamePk"),
                "home_team": tmap.get(int(g["teams"]["home"]["team"]["id"]), ""),
                "away_team": tmap.get(int(g["teams"]["away"]["team"]["id"]), ""),
            })
    return pd.DataFrame(rows)


# ─── step 1: lineups ────────────────────────────────────────────────────
def refresh_lineups(today: str) -> pd.DataFrame:
    print(f"[1/5] Refreshing lineups for {today} ...")
    try:
        subprocess.run([sys.executable, "lineup_pull.py", "--date", today],
                       check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        print(f"  [warn] lineup_pull exit != 0: {exc}; continuing with cache")
    if not LINEUP_TODAY.exists():
        raise SystemExit(f"No lineup file at {LINEUP_TODAY}; aborting.")
    df = pd.read_parquet(LINEUP_TODAY)
    print(f"  lineups today: {len(df)} rows, "
          f"{df['game_pk'].nunique()} games")
    return df


# ─── step 2: env per game ───────────────────────────────────────────────
def build_game_env(lineups: pd.DataFrame, today: str) -> pd.DataFrame:
    print("[2/5] Building per-game environment (wind/temp/SP velo) ...")
    # Lineup parquet has game_pk but not home_team/away_team; fetch schedule.
    sched = fetch_schedule_from_mlb(today)
    print(f"  schedule today: {len(sched)} games")
    games = sched.copy()
    # Try the enriched matrix first (has wind, temp, SP velos).
    env = pd.DataFrame()
    if FEATURE_MTX.exists():
        fm = pd.read_parquet(FEATURE_MTX,
                              columns=["game_pk", "game_date",
                                       "wind_mph", "wind_bearing", "temp_f",
                                       "home_sp_ff_velo", "away_sp_ff_velo",
                                       "home_sp_p_throws_R", "away_sp_p_throws_R",
                                       "elo_diff", "sp_velo_decay_diff"])
        fm["game_date"] = pd.to_datetime(fm["game_date"]).dt.strftime("%Y-%m-%d")
        env = fm[fm["game_date"] == today]
        if len(env):
            print(f"  env rows from feature_matrix: {len(env)}")
    if env.empty:
        print(f"  [warn] feature_matrix has no rows for {today}; "
              f"using neutral fallback (wind=0, temp=72F, SP velo=league avg)")
        env = games[["game_pk"]].copy()
        env["wind_mph"] = 0.0
        env["wind_bearing"] = 0.0
        env["temp_f"] = 72.0
        env["home_sp_ff_velo"] = LEAGUE_AVG_FF_VELO
        env["away_sp_ff_velo"] = LEAGUE_AVG_FF_VELO
        env["home_sp_p_throws_R"] = 1
        env["away_sp_p_throws_R"] = 1
        env["elo_diff"] = 0.0
        env["sp_velo_decay_diff"] = 0.0
    return games.merge(env, on="game_pk", how="left")


# ─── step 3: grounded env per park (dynamic bias + circuit breaker) ─────
def grounded_env(home_team: str) -> tuple[float, float, dict]:
    offset, n, meta = srd.get_dynamic_bias(home_team)
    # Circuit breaker is already applied inside _clamp_offset (|cap|=1.5)
    return offset, LEAGUE_BASELINE + offset, meta


# ─── step 4: 2025 xStats lookup ─────────────────────────────────────────
def load_xstats_2025() -> dict[int, dict]:
    files = sorted(XSTATS_DIR.glob("batter_xstats_2025*.parquet"))
    if not files:
        return {}
    df = pd.read_parquet(files[0])
    id_col = next((c for c in ("player_id", "batter", "mlbam_id", "MLBAMID")
                   if c in df.columns), None)
    if not id_col:
        return {}
    return {int(r[id_col]): r for _, r in df.iterrows()}


def pull_side_vec(stand: str, wm: float, wb: float, cf: float) -> float:
    if stand == "L":     pull = (cf + PULL_OFFSET) % 360.0
    elif stand == "R":   pull = (cf - PULL_OFFSET) % 360.0
    else:                pull = cf % 360.0
    theta = np.deg2rad(wb - pull)
    return float(np.cos(theta) * (wm / 10.0))


# ─── step 5: feature assembly + scoring ─────────────────────────────────
def build_feature_row(batter, game_env, offset, proj_adj, xstats_map, cf_map):
    pid    = int(batter["player_id"])
    stand  = (batter.get("stand") or "").upper()
    team   = batter["team"]
    home   = game_env["home_team"]
    is_home = team == home
    bo     = batter.get("batting_order") or 0
    bo     = int(bo) if pd.notna(bo) else 0

    wm = float(game_env.get("wind_mph") or 0)
    wb = float(game_env.get("wind_bearing") or 0)
    temp = float(game_env.get("temp_f") or 72.0)
    home_velo = float(game_env.get("home_sp_ff_velo") or LEAGUE_AVG_FF_VELO)
    away_velo = float(game_env.get("away_sp_ff_velo") or LEAGUE_AVG_FF_VELO)
    opp_velo = away_velo if is_home else home_velo
    elo = float(game_env.get("elo_diff") or 0)

    cf = cf_map.get(home, 0.0)
    psw = pull_side_vec(stand, wm, wb, cf)
    velo_decay_risk = (temp - 72.0) * (opp_velo - LEAGUE_AVG_FF_VELO)

    # lineup_fragility
    opp_R = float(game_env.get("away_sp_p_throws_R") if is_home
                  else game_env.get("home_sp_p_throws_R") or 1)
    if stand == "R" and opp_R == 1:   same = 1.0
    elif stand == "L" and opp_R == 0: same = 1.0
    elif stand == "S":                same = np.nan
    else:                              same = 0.0
    blowout = min(1.0, abs(elo) / 200.0)
    frag = 0.35 * (same if not np.isnan(same) else 0.5) + 0.65 * blowout

    xs = xstats_map.get(pid, {})
    return {
        "pull_side_wind_vector": psw,
        "projected_total_adj":   proj_adj,
        "bias_offset":           offset,
        "wind_mph":              wm,
        "wind_bearing":          wb,
        "temp_f":                temp,
        "velocity_decay_risk":   velo_decay_risk,
        "lineup_fragility":      frag,
        "platoon_same_hand":     same,
        "batting_order":         float(bo),
        "exp_pa_heuristic":      EXP_PA_MAP.get(bo, np.nan),
        "ba": xs.get("ba"), "est_ba": xs.get("est_ba"),
        "slg": xs.get("slg"), "est_slg": xs.get("est_slg"),
        "woba": xs.get("woba"), "est_woba": xs.get("est_woba"),
        "avg_hit_angle": xs.get("avg_hit_angle"),
        "anglesweetspotpercent": xs.get("anglesweetspotpercent"),
        "max_hit_speed": xs.get("max_hit_speed"),
        "avg_hit_speed": xs.get("avg_hit_speed"),
        "ev50": xs.get("ev50"), "fbld": xs.get("fbld"),
        "ev95percent": xs.get("ev95percent"),
        "brl_percent": xs.get("brl_percent"),
        "brl_pa": xs.get("brl_pa"),
        "stand_L": 1 if stand == "L" else 0,
        "stand_R": 1 if stand == "R" else 0,
        "stand_S": 1 if stand == "S" else 0,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    args = ap.parse_args()
    today = args.date or date.today().isoformat()
    print(f"=== score_slate_today.py  [{today}] ===")
    global LINEUP_TODAY
    if args.date:
        alt = Path(f"data/statcast/lineups_{args.date}_long.parquet")
        if alt.exists():
            LINEUP_TODAY = alt
            print(f"  using historical lineup parquet: {alt}")

    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    feat_names = booster.feature_names
    print(f"Model: {MODEL_PATH} ({len(feat_names)} features)")

    cf_map = {p: v["cf_azimuth_deg"]
              for p, v in json.loads(STADIUM_META.read_text(encoding="utf-8"))
                              .get("parks", {}).items()}

    lineups = refresh_lineups(today)
    games_env = build_game_env(lineups, today)

    print("[3/5] Computing grounded env (dynamic bias, -1.5r circuit breaker) ...")
    env_rows = []
    for _, g in games_env.iterrows():
        off, padj, meta = grounded_env(g["home_team"])
        env_rows.append({"game_pk": g["game_pk"], "bias_offset": off,
                         "projected_total_adj": padj,
                         "bias_source": meta.get("source"),
                         "park_blend": meta.get("park_blend"),
                         "clamped": bool(meta.get("clamped", False))})
    benv = pd.DataFrame(env_rows)
    clamped = int(benv["clamped"].sum())
    print(f"  parks scored: {len(benv)}  |  circuit-breaker clamps: {clamped}")

    print("[4/5] Loading 2025 xStats ...")
    xstats_map = load_xstats_2025()
    print(f"  xStats rows indexed: {len(xstats_map)}")

    print("[5/5] Building feature matrix + scoring ...")
    # Need game_pk -> env row map
    genv = games_env.merge(benv, on="game_pk")
    rows = []
    for _, bat in lineups.iterrows():
        if pd.isna(bat.get("player_id")):
            continue
        gk = bat["game_pk"]
        ge = genv[genv["game_pk"] == gk]
        if ge.empty:
            continue
        ge = ge.iloc[0]
        feat = build_feature_row(bat, ge, ge["bias_offset"],
                                  ge["projected_total_adj"],
                                  xstats_map, cf_map)
        feat["Player"] = bat.get("player_name", "")
        feat["Game"]   = f"{ge['away_team']} @ {ge['home_team']}"
        feat["_pid"]   = int(bat["player_id"])
        rows.append(feat)

    if not rows:
        raise SystemExit("No scorable batter rows assembled.")

    df = pd.DataFrame(rows)
    X = df[feat_names].astype(float)
    dm = xgb.DMatrix(X, feature_names=feat_names)
    p_over = booster.predict(dm)
    exp_tb = p_over * MEAN_TB_OVER + (1 - p_over) * MEAN_TB_UNDER
    out = pd.DataFrame({
        "Player":      df["Player"],
        "Game":        df["Game"],
        "Expected_TB": np.round(exp_tb, 3),
        "P_Over_1.5":  np.round(p_over, 4),
        "Edge":        np.round(p_over - RETAIL_IMPLIED, 4),
    }).sort_values("Edge", ascending=False).reset_index(drop=True)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)
    print(f"\nWrote {len(out)} batter projections -> {OUT_FILE}")
    print("\nTop 15 by edge:")
    print(out.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
