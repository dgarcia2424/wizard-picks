"""
v3.5 Shadow Audit — 2026-04-24 PHI @ ATL @ Truist (CF=15 deg).
Wind scenario: 7 mph SW (bearing=225 deg).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import score_run_dist_today as srd

TRUIST_CF_DEG = 15.0
PULL_OFFSET = 22.0
LEAGUE_BASELINE = 8.80
MARKET_LINE = 1.5

# Wind scenario (user-specified)
WIND_MPH = 7.0
WIND_BEARING = 225.0    # SW
GAME_TEMP_F = 72.0      # assume neutral — no live feed
SP_VELO_DECAY_DIFF_PROXY = 0.0   # no live SP decay signal; neutral
ELO_DIFF_PROXY = 0.0             # neutral matchup
HOME_SP_P_THROWS_R = 1           # assume RHP for both sides (typical)
AWAY_SP_P_THROWS_R = 1
LEAGUE_SP_VELO_STD = 1.20

import xgboost as xgb
booster = xgb.Booster()
booster.load_model("models/tb_stacker_v35.json")
feat_names = booster.feature_names
print("Model features:", feat_names)

# ── Grounded env (live PIT bias for ATL) ────────────────────────────────
offset, n_recent, meta = srd.get_dynamic_bias("ATL")
projected_total_adj = LEAGUE_BASELINE + offset
print(f"\nGrounded env ATL 2026-04-24:  bias={offset:+.3f}  "
      f"source={meta.get('source')} blend={meta.get('park_blend')} "
      f"park_n={meta.get('park_n',0)} n={n_recent}")
print(f"  projected_total_adj = {projected_total_adj:.3f}")

# ── 2025 xStats ─────────────────────────────────────────────────────────
XSTATS = Path("data/statcast")
xfiles = sorted(XSTATS.glob("batter_xstats_2025*.parquet"))
def xstats_row(pid: int) -> dict:
    for f in xfiles:
        df = pd.read_parquet(f)
        for ic in ("player_id", "batter", "mlbam_id", "MLBAMID"):
            if ic in df.columns:
                hit = df[df[ic] == pid]
                if len(hit):
                    return hit.iloc[0].to_dict()
                break
    return {}

def pull_side_vec(stand: str) -> float:
    if stand == "L":    pull = (TRUIST_CF_DEG + PULL_OFFSET) % 360.0
    elif stand == "R":  pull = (TRUIST_CF_DEG - PULL_OFFSET) % 360.0
    else:               pull = TRUIST_CF_DEG % 360.0
    theta = np.deg2rad(WIND_BEARING - pull)
    return float(np.cos(theta) * (WIND_MPH / 10.0))

def velocity_decay_risk() -> float:
    # mirror build_batter_features._compute_velocity_decay_risk
    scale = LEAGUE_SP_VELO_STD + 0.5 * abs(SP_VELO_DECAY_DIFF_PROXY)
    return (GAME_TEMP_F - 72.0) * scale

def lineup_fragility(stand: str, is_home: bool) -> tuple[float, float]:
    opp_R = AWAY_SP_P_THROWS_R if is_home else HOME_SP_P_THROWS_R
    if stand == "R" and opp_R == 1:   same = 1.0
    elif stand == "L" and opp_R == 0: same = 1.0
    elif stand == "S":                same = np.nan
    else:                              same = 0.0
    blowout = min(1.0, abs(ELO_DIFF_PROXY) / 200.0)
    frag = 0.35 * (same if not np.isnan(same) else 0.5) + 0.65 * blowout
    return same, frag

EXP_PA_MAP = {1:4.6,2:4.4,3:4.2,4:4.0,5:3.8,6:3.6,7:3.4,8:3.2,9:3.0}

players = [
    ("Matt Olson",     621566, "ATL", "L", True,  2),   # home, 2-hole
    ("Kyle Schwarber", 656941, "PHI", "L", False, 2),   # away, 2-hole
]

rows = []
for name, pid, team, stand, is_home, bo in players:
    xs = xstats_row(pid)
    psw = pull_side_vec(stand)
    vdr = velocity_decay_risk()
    same, frag = lineup_fragility(stand, is_home)
    row = {
        "pull_side_wind_vector": psw,
        "projected_total_adj":   projected_total_adj,
        "bias_offset":           offset,
        "wind_mph":              WIND_MPH,
        "wind_bearing":          WIND_BEARING,
        "temp_f":                GAME_TEMP_F,
        "velocity_decay_risk":   vdr,
        "lineup_fragility":      frag,
        "platoon_same_hand":     same if not np.isnan(same) else np.nan,
        "batting_order":         float(bo),
        "exp_pa_heuristic":      EXP_PA_MAP.get(bo),
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
        "stand_L": 1 if stand=="L" else 0,
        "stand_R": 1 if stand=="R" else 0,
        "stand_S": 1 if stand=="S" else 0,
    }
    row["_name"] = name
    rows.append(row)

df = pd.DataFrame(rows)
X = df[feat_names].astype(float)
dm = xgb.DMatrix(X, feature_names=feat_names)
p_over = booster.predict(dm)
# Expected-TB under a binary head: we only have P(TB>1.5). Report that directly
# as the "Grounded Pred TB" probability alongside the market implied.
retail_implied = 0.524

print(f"\n=== v3.5 SHADOW AUDIT  (wind = 7 mph SW / 225 deg, CF=15 deg) ===")
print(f"{'Player':<18}{'P(TB>1.5)':>12}{'Implied':>10}{'Edge':>10}"
      f"{'pull_vec':>10}{'velo_dec':>10}{'frag':>8}")
for (name, pid, team, stand, is_home, bo), p in zip(players, p_over):
    psw = df.loc[df['_name']==name, 'pull_side_wind_vector'].iloc[0]
    vdr = df.loc[df['_name']==name, 'velocity_decay_risk'].iloc[0]
    frag = df.loc[df['_name']==name, 'lineup_fragility'].iloc[0]
    edge = p - retail_implied
    print(f"{name:<18}{p:>12.3f}{retail_implied:>10.3f}{edge:>+10.3f}"
          f"{psw:>10.3f}{vdr:>10.3f}{frag:>8.3f}")

print("\nNotes:")
print("  - Wind: 7 mph @ 225 deg (SW). Pull azimuth for L hitter at Truist = 37 deg")
print("    -> cos(225-37)=cos(188)=-0.990, pull_vec=-0.693 (wind against pulled FBs).")
print(f"  - Temp 72F (neutral) -> velocity_decay_risk=0 by design.")
print(f"  - Elo diff 0 (neutral) -> blowout component of fragility=0.")
print(f"  - Env: projected_total_adj={projected_total_adj:.2f} (offset {offset:+.2f}).")
print("  - pa_count DROPPED from features; exp_pa_heuristic replaces it.")
