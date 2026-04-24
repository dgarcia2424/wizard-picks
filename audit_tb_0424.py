"""
Live audit — 2026-04-24 PHI @ ATL @ Truist Park.
Matt Olson (ATL, L, 621566)  vs.  Kyle Schwarber (PHI, L, 656941).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import score_run_dist_today as srd

TRUIST_CF_DEG = 15.0   # per today's brief
PULL_OFFSET = 22.0
LEAGUE_BASELINE = 8.80
MARKET_LINE = 1.5

import xgboost as xgb
booster = xgb.Booster()
booster.load_model("models/tb_stacker_v1.json")
feat_names = booster.feature_names
print("Model features:", feat_names)

# ── Grounded env (live) via point-in-time bias ──────────────────────────
offset, n_recent, meta = srd.get_dynamic_bias("ATL")
env_source = meta.get("source", "fallback")
park_blend = meta.get("park_blend", "fallback")
park_n = meta.get("park_n", 0)
projected_total_adj = LEAGUE_BASELINE + offset
print(f"\nGrounded env for ATL @ Truist 2026-04-24:")
print(f"  bias_offset        = {offset:+.3f}r  (source={env_source}, "
      f"blend={park_blend}, park_n={park_n}, n_recent={n_recent})")
print(f"  projected_total_adj= {projected_total_adj:.3f}  "
      f"(= {LEAGUE_BASELINE:.2f} + offset)")

# ── Prior-season xStats ─────────────────────────────────────────────────
XSTATS = Path("data/statcast")
xfiles = sorted(XSTATS.glob("batter_xstats_2025*.parquet"))
print(f"\n2025 xStats files: {[f.name for f in xfiles]}")

def load_xstats_row(pid: int) -> dict:
    for f in xfiles:
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        id_cols = [c for c in ("player_id", "batter", "mlbam_id", "MLBAMID") if c in df.columns]
        if not id_cols:
            continue
        ic = id_cols[0]
        hit = df[df[ic] == pid]
        if len(hit):
            return hit.iloc[0].to_dict()
    return {}

players = [
    ("Matt Olson",     621566, "ATL", "L"),
    ("Kyle Schwarber", 656941, "PHI", "L"),
]

# ── Wind: no live feed loaded; default to calm (0 mph). Flagged. ────────
WIND_MPH = 0.0
WIND_BEARING = 0.0
print(f"\n[warn] No live wind feed for 2026-04-24 @ Truist in local data; "
      f"using wind_mph=0 (calm) for audit.")

def pull_side_vec(stand: str, wm: float, wb: float, cf: float) -> float:
    if stand == "L":
        pull = (cf + PULL_OFFSET) % 360.0
    elif stand == "R":
        pull = (cf - PULL_OFFSET) % 360.0
    else:
        pull = cf % 360.0
    theta = np.deg2rad(wb - pull)
    return float(np.cos(theta) * (wm / 10.0))

rows = []
for name, pid, team, stand in players:
    xs = load_xstats_row(pid)
    psw = pull_side_vec(stand, WIND_MPH, WIND_BEARING, TRUIST_CF_DEG)
    row = {
        "pull_side_wind_vector": psw,
        "projected_total_adj":   projected_total_adj,
        "bias_offset":           offset,
        "wind_mph":              WIND_MPH,
        "wind_bearing":          WIND_BEARING,
        "batting_order":         2.0,    # typical slot, both 2-hole hitters
        "pa_count":              4.0,    # typical full-game exposure (HONESTY: leak)
        "ba":                    xs.get("ba"),
        "est_ba":                xs.get("est_ba"),
        "slg":                   xs.get("slg"),
        "est_slg":               xs.get("est_slg"),
        "woba":                  xs.get("woba"),
        "est_woba":              xs.get("est_woba"),
        "avg_hit_angle":         xs.get("avg_hit_angle"),
        "anglesweetspotpercent": xs.get("anglesweetspotpercent"),
        "max_hit_speed":         xs.get("max_hit_speed"),
        "avg_hit_speed":         xs.get("avg_hit_speed"),
        "ev50":                  xs.get("ev50"),
        "fbld":                  xs.get("fbld"),
        "ev95percent":           xs.get("ev95percent"),
        "brl_percent":           xs.get("brl_percent"),
        "brl_pa":                xs.get("brl_pa"),
        "stand_L": 1 if stand == "L" else 0,
        "stand_R": 1 if stand == "R" else 0,
        "stand_S": 1 if stand == "S" else 0,
    }
    row["_name"] = name
    row["_pid"] = pid
    rows.append(row)

df = pd.DataFrame(rows)
X = df[feat_names].astype(float)
dm = xgb.DMatrix(X, feature_names=feat_names)
p_over = booster.predict(dm)
# Expected TB given over/under is a calibration question; for edge report,
# use P(TB>1.5) vs retail implied (50/50 at a -110/-110 line = 0.524).
retail_implied = 0.524
print("\n=== TB > 1.5 AUDIT — 2026-04-24 @ Truist (CF=15 deg) ===")
print(f"{'Player':<18}{'P(TB>1.5)':>12}{'Implied':>10}{'Edge':>10}"
      f"{'pull_vec':>10}")
for (name, pid, team, stand), p, pv in zip(players, p_over,
                                             df["pull_side_wind_vector"]):
    edge = p - retail_implied
    print(f"{name:<18}{p:>12.3f}{retail_implied:>10.3f}{edge:>+10.3f}{pv:>10.3f}")

print("\nNOTES:")
print("  - Market line fixed at 1.5 TB; implied assumed -110/-110 (0.524).")
print("  - Wind set to 0 mph (no live feed loaded); pull_side_wind_vector=0 for both.")
print(f"  - Env: projected_total_adj={projected_total_adj:.2f} "
      f"(offset {offset:+.2f}, {env_source}/{park_blend}).")
print("  - pa_count set to 4 (typical); flagged as leakage in v1 — v2 must drop.")
