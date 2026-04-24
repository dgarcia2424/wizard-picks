"""
predict_home_plate_ump.py — v4.4 Umpire Rotation Predictor.

MLB umpires rotate clockwise within a 4-man crew across a series:
    Plate -> 3B -> 2B -> 1B -> Plate (next series)

To predict today's HP ump:
    Look at yesterday's game at the SAME park (same series).
    Yesterday's 1B umpire = today's HP umpire.

ABS Challenge System (2026):
    Batters/pitchers can challenge ball/strike calls via the automated
    ball-strike (ABS) system. If a pitcher has a high challenge overturn
    rate, the ump's zone calls are less reliable, and the K-synergy
    multiplier is reduced by 30% (from 1.20x to 1.14x).

    ABS overturn rate is proxied by the ump's called-strike divergence
    from expected zone (plate_x/plate_z vs sz_top/sz_bot) in statcast.
    High divergence = more likely to be overturned by ABS.

Usage:
    from predict_home_plate_ump import predict_hp_umps
    assignments = predict_hp_umps("2026-04-24")
    # Returns: {game_pk: {"ump_hp_name": ..., "ump_hp_id": ..., "abs_adjusted": bool}}

Run standalone to populate umpire_assignments_2026.parquet:
    python predict_home_plate_ump.py --date 2026-04-24
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent

UMP_FEATURES_DIR  = _ROOT / "data/statcast"
UMP_ASSIGN_2026   = _ROOT / "data/statcast/umpire_assignments_2026.parquet"
STATCAST_DIR      = _ROOT / "data/statcast"

# ABS thresholds
ABS_HIGH_DIVERGENCE = 0.08   # called-strike divergence > 8% of pitches = high overturn risk
ABS_SYNERGY_PENALTY = 0.30   # reduce synergy multiplier by 30% (1.20 -> 1.14)

TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams?sportId=1"
_TEAM_MAP: dict[int, str] = {}


def _get_team_map() -> dict[int, str]:
    global _TEAM_MAP
    if _TEAM_MAP:
        return _TEAM_MAP
    with urllib.request.urlopen(TEAMS_URL, timeout=15) as r:
        _TEAM_MAP = {int(t["id"]): t.get("abbreviation", "")
                     for t in json.loads(r.read())["teams"]}
    return _TEAM_MAP


def _fetch_officials_for_date(d: str) -> list[dict]:
    """Fetch full crew assignments (all 4 positions) for a given date."""
    url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1"
           f"&date={d}&hydrate=officials")
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            payload = json.loads(r.read())
    except Exception as exc:
        print(f"  [ump] API failed for {d}: {exc}")
        return []

    tm = _get_team_map()
    rows = []
    for block in payload.get("dates", []):
        for g in block.get("games", []):
            game_pk = g.get("gamePk")
            home_id = int(g["teams"]["home"]["team"]["id"])
            home_team = tm.get(home_id, "")
            officials = g.get("officials", [])
            if not officials:
                continue
            row = {"game_pk": game_pk, "game_date": d, "home_team": home_team}
            for o in officials:
                pos  = o.get("officialType", "")
                off  = o.get("official", {})
                name = off.get("fullName", "")
                uid  = off.get("id")
                pos_key = {
                    "Home Plate": "hp",
                    "First Base":  "1b",
                    "Second Base": "2b",
                    "Third Base":  "3b",
                }.get(pos)
                if pos_key:
                    row[f"ump_{pos_key}_name"] = name
                    row[f"ump_{pos_key}_id"]   = uid
            rows.append(row)
    return rows


def _build_abs_divergence_table() -> pd.DataFrame:
    """Proxy ABS challenge overturn risk from statcast called-strike divergence.

    A pitch is a 'contested call' when statcast plate_x/plate_z places it
    within 0.5 inches of the zone border (sz_top, sz_bot, +/-0.83 wide).
    The divergence rate = fraction of called strikes in the 'grey zone'
    that are outside the strict zone — these are most likely to be overturned.

    Returns DataFrame with columns: [ump_hp_id, abs_divergence_rate, abs_high_risk]
    grouped by the statcast umpire column (name string, not ID).
    """
    sc_path = STATCAST_DIR / "statcast_2026.parquet"
    if not sc_path.exists():
        return pd.DataFrame(columns=["ump_name", "abs_divergence_rate", "abs_high_risk"])

    needed = ["game_pk", "description", "plate_x", "plate_z",
              "sz_top", "sz_bot", "umpire"]
    sc = pd.read_parquet(sc_path, columns=[c for c in needed
                                            if c in pd.read_parquet(sc_path,
                                            columns=["game_pk"]).columns or True])
    avail = [c for c in needed if c in sc.columns]
    sc = sc[avail].copy()

    if "umpire" not in sc.columns or "description" not in sc.columns:
        return pd.DataFrame(columns=["ump_name", "abs_divergence_rate", "abs_high_risk"])

    # Filter to called strikes only
    cs = sc[sc["description"] == "called_strike"].copy()
    if cs.empty or not all(c in cs.columns for c in ["plate_x", "plate_z", "sz_top", "sz_bot"]):
        return pd.DataFrame(columns=["ump_name", "abs_divergence_rate", "abs_high_risk"])

    cs["plate_x"]  = pd.to_numeric(cs["plate_x"],  errors="coerce")
    cs["plate_z"]  = pd.to_numeric(cs["plate_z"],  errors="coerce")
    cs["sz_top"]   = pd.to_numeric(cs["sz_top"],   errors="coerce")
    cs["sz_bot"]   = pd.to_numeric(cs["sz_bot"],   errors="coerce")

    # Outside strict zone: |plate_x| > 0.83 OR plate_z outside [sz_bot, sz_top]
    cs["outside_zone"] = (
        (cs["plate_x"].abs() > 0.83) |
        (cs["plate_z"] > cs["sz_top"] + 0.05) |
        (cs["plate_z"] < cs["sz_bot"] - 0.05)
    ).astype(float)

    agg = (cs.groupby("umpire")
             .agg(total_cs=("outside_zone", "count"),
                  outside_cs=("outside_zone", "sum"))
             .reset_index())
    agg["abs_divergence_rate"] = agg["outside_cs"] / agg["total_cs"].clip(lower=1)
    agg["abs_high_risk"] = (agg["abs_divergence_rate"] > ABS_HIGH_DIVERGENCE).astype("int8")
    agg = agg.rename(columns={"umpire": "ump_name"})
    return agg[["ump_name", "abs_divergence_rate", "abs_high_risk"]]


def predict_hp_umps(target_date: str, verbose: bool = True) -> pd.DataFrame:
    """Predict today's HP umpires using yesterday's crew rotation.

    Rotation rule: yesterday's 1B ump = today's HP ump (for same crew/series).

    Also joins ABS divergence data and flags high-risk umps.

    Returns DataFrame with one row per game.
    """
    target_dt  = date.fromisoformat(target_date)
    yesterday  = (target_dt - timedelta(days=1)).isoformat()

    if verbose:
        print(f"  [ump] Fetching yesterday's crew assignments ({yesterday}) ...")

    # Step 1: Try to get today's assignments directly (post-game or same-day if posted)
    today_crews = _fetch_officials_for_date(target_date)
    if verbose:
        n_hp = sum(1 for c in today_crews if c.get("ump_hp_name"))
        print(f"  [ump] Today direct: {n_hp}/{len(today_crews)} games have HP assigned")

    # Step 2: Get yesterday's full crew (always available after games finish)
    yest_crews = _fetch_officials_for_date(yesterday)
    if verbose:
        print(f"  [ump] Yesterday: {len(yest_crews)} games with crew data")

    # Build yesterday's 1B-ump -> predicted HP lookup
    yest_by_hometeam: dict[str, dict] = {}
    for row in yest_crews:
        home = row.get("home_team", "")
        if home and row.get("ump_1b_name"):
            yest_by_hometeam[home] = {
                "predicted_hp_name": row["ump_1b_name"],
                "predicted_hp_id":   row.get("ump_1b_id"),
                "yesterday_crew":    {
                    "hp": row.get("ump_hp_name"), "1b": row.get("ump_1b_name"),
                    "2b": row.get("ump_2b_name"), "3b": row.get("ump_3b_name"),
                },
            }

    # Step 3: Load ABS divergence table
    abs_table = _build_abs_divergence_table()
    abs_lookup: dict[str, dict] = {}
    if not abs_table.empty:
        for _, r in abs_table.iterrows():
            abs_lookup[str(r["ump_name"]).lower()] = {
                "abs_divergence_rate": float(r["abs_divergence_rate"]),
                "abs_high_risk":       int(r["abs_high_risk"]),
            }

    # Step 4: Load ump career K stats
    uf_frames = []
    for yr in (2023, 2024, 2025):
        p = UMP_FEATURES_DIR / f"ump_features_{yr}.parquet"
        if p.exists():
            uf_frames.append(pd.read_parquet(p))
    uf_career = pd.DataFrame()
    if uf_frames:
        uf_all = pd.concat(uf_frames, ignore_index=True)
        uf_career = (uf_all.dropna(subset=["ump_hp_id", "ump_k_above_avg"])
                           .groupby("ump_hp_id")[["ump_k_above_avg",
                                                   "ump_bb_above_avg",
                                                   "ump_rpg_above_avg"]]
                           .mean().reset_index())

    # Step 5: Fetch today's schedule to know which games exist
    today_sched_url = (f"https://statsapi.mlb.com/api/v1/schedule"
                       f"?sportId=1&date={target_date}&hydrate=probablePitcher")
    try:
        with urllib.request.urlopen(today_sched_url, timeout=15) as r:
            sched = json.loads(r.read())
    except Exception:
        sched = {"dates": []}

    tm = _get_team_map()
    result_rows = []
    for block in sched.get("dates", []):
        for g in block.get("games", []):
            game_pk   = g.get("gamePk")
            home_id   = int(g["teams"]["home"]["team"]["id"])
            home_team = tm.get(home_id, "")

            # Direct assignment takes priority
            direct = next((c for c in today_crews if c.get("game_pk") == game_pk), None)
            if direct and direct.get("ump_hp_name"):
                hp_name = direct["ump_hp_name"]
                hp_id   = direct.get("ump_hp_id")
                source  = "direct"
            elif home_team in yest_by_hometeam:
                yest = yest_by_hometeam[home_team]
                hp_name = yest["predicted_hp_name"]
                hp_id   = yest["predicted_hp_id"]
                source  = "rotation_predicted"
            else:
                hp_name = None
                hp_id   = None
                source  = "unknown"

            # ABS risk: look up by name
            abs_info = abs_lookup.get(str(hp_name or "").lower(), {})
            abs_high = abs_info.get("abs_high_risk", 0)
            abs_rate = abs_info.get("abs_divergence_rate", 0.0)

            # K stats
            k_above_avg = np.nan
            if hp_id is not None and not uf_career.empty:
                hp_id_num = int(hp_id) if hp_id else None
                match = uf_career[uf_career["ump_hp_id"] == hp_id_num]
                if not match.empty:
                    k_above_avg = float(match.iloc[0]["ump_k_above_avg"])

            # Synergy multiplier with ABS penalty
            from data_orchestrator import WIDE_ZONE_THRESHOLD, STUFF_PLUS_THRESHOLD
            base_synergy = 1.20
            if abs_high:
                # ABS system reduces the value of a wide-zone ump
                effective_synergy = base_synergy * (1.0 - ABS_SYNERGY_PENALTY)
            else:
                effective_synergy = base_synergy

            result_rows.append({
                "game_pk":             game_pk,
                "game_date":           target_date,
                "home_team":           home_team,
                "ump_hp_name":         hp_name,
                "ump_hp_id":           hp_id,
                "ump_source":          source,
                "ump_k_above_avg":     k_above_avg,
                "abs_divergence_rate": abs_rate,
                "abs_high_risk":       abs_high,
                "synergy_max":         round(effective_synergy, 3),
            })

    result = pd.DataFrame(result_rows)

    if verbose:
        n_predicted = (result["ump_source"] == "rotation_predicted").sum()
        n_direct    = (result["ump_source"] == "direct").sum()
        n_abs_flag  = result["abs_high_risk"].sum() if "abs_high_risk" in result.columns else 0
        print(f"  [ump] {n_direct} direct + {n_predicted} rotation-predicted + "
              f"{(result['ump_source']=='unknown').sum()} unknown")
        print(f"  [ump] ABS high-risk umps: {n_abs_flag} "
              f"(synergy reduced {base_synergy:.2f}x -> {base_synergy*(1-ABS_SYNERGY_PENALTY):.2f}x)")
        matched_k = result["ump_k_above_avg"].notna().sum()
        print(f"  [ump] K stats matched: {matched_k}/{len(result)}")

    # Save to umpire_assignments_2026.parquet
    if not result.empty:
        if UMP_ASSIGN_2026.exists():
            existing = pd.read_parquet(UMP_ASSIGN_2026)
            existing = existing[~existing["game_pk"].isin(result["game_pk"])]
            combined = pd.concat([existing, result], ignore_index=True)
        else:
            combined = result
        UMP_ASSIGN_2026.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(UMP_ASSIGN_2026, index=False)
        if verbose:
            print(f"  [ump] Saved -> {UMP_ASSIGN_2026} ({len(combined)} total rows)")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--abs-only", action="store_true",
                        help="Only rebuild ABS divergence table, don't predict")
    args = parser.parse_args()

    if args.abs_only:
        tbl = _build_abs_divergence_table()
        if not tbl.empty:
            print(tbl.sort_values("abs_divergence_rate", ascending=False).head(15).to_string())
        else:
            print("No ABS data available")
    else:
        result = predict_hp_umps(args.date, verbose=True)
        if not result.empty:
            print()
            print(result[["home_team", "ump_hp_name", "ump_source",
                           "ump_k_above_avg", "abs_high_risk", "synergy_max"]]
                  .to_string(index=False))
