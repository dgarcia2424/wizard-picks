"""
recover_actuals_4_18_22.py — Fetches final scores from MLB Stats API schedule
for the 2026-04-18..2026-04-22 gap and writes/updates
data/statcast/actuals_2026.parquet.

This is the upstream fix for update_actuals.py: that script's _load_actuals()
was silently returning empty because the parquet had never been created.

Residuals for 4/18-4/22 cannot be computed retroactively — no per-date
projected_total_adj was persisted for those dates (daily prediction
snapshots dropped the column). This script ONLY populates actuals so the
residual pipeline can accrue from 4/23 forward.
"""
from __future__ import annotations
import urllib.request, json
from datetime import date, timedelta
from pathlib import Path
import pandas as pd

OUT = Path("data/statcast/actuals_2026.parquet")
START = date(2026, 4, 18)
END   = date(2026, 4, 22)

# MLB Stats API schedule endpoint
URL = ("https://statsapi.mlb.com/api/v1/schedule"
       "?sportId=1&date={d}&hydrate=linescore")
TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams?sportId=1"


def _load_team_map() -> dict[int, str]:
    with urllib.request.urlopen(TEAMS_URL, timeout=15) as r:
        payload = json.loads(r.read().decode())
    m = {}
    for t in payload.get("teams", []):
        m[int(t["id"])] = t.get("abbreviation") or t.get("teamCode", "").upper()
    return m


TEAM_MAP = _load_team_map()

# Statsapi uses 3-letter team abbr in teamCode; map to our park_id (usually
# same as home_team abbreviation already in actuals_2026 schema).

def fetch_day(d: date) -> list[dict]:
    with urllib.request.urlopen(URL.format(d=d.isoformat()), timeout=15) as r:
        payload = json.loads(r.read().decode())
    rows = []
    for block in payload.get("dates", []):
        for g in block.get("games", []):
            status = g.get("status", {}).get("detailedState", "")
            if status not in ("Final", "Game Over", "Completed Early"):
                continue
            ls = g.get("linescore", {})
            teams = g.get("teams", {})
            h = teams.get("home", {})
            a = teams.get("away", {})
            rows.append({
                "game_pk":          g.get("gamePk"),
                "game_date":        d.isoformat(),
                "home_team":        TEAM_MAP.get(int(h.get("team", {}).get("id", 0)), ""),
                "away_team":        TEAM_MAP.get(int(a.get("team", {}).get("id", 0)), ""),
                "home_score_final": h.get("score"),
                "away_score_final": a.get("score"),
                "status":           status,
            })
    return rows


def main():
    all_rows = []
    d = START
    while d <= END:
        try:
            rows = fetch_day(d)
            print(f"  {d.isoformat()}: {len(rows)} games")
            all_rows.extend(rows)
        except Exception as exc:
            print(f"  {d.isoformat()}: FETCH ERROR {exc}")
        d += timedelta(days=1)

    if not all_rows:
        print("No rows fetched; aborting.")
        return

    new = pd.DataFrame(all_rows)
    # Validate: drop rows where finals are null
    new = new.dropna(subset=["home_score_final", "away_score_final"])
    new["home_score_final"] = new["home_score_final"].astype("int16")
    new["away_score_final"] = new["away_score_final"].astype("int16")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        prior = pd.read_parquet(OUT)
        # Upsert by game_pk
        prior = prior[~prior["game_pk"].isin(new["game_pk"])]
        combined = pd.concat([prior, new], ignore_index=True, sort=False)
    else:
        combined = new
    combined = combined.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    combined.to_parquet(OUT, index=False)
    print(f"\nWrote {len(combined)} total rows -> {OUT}")
    print(f"  (this batch added/updated: {len(new)} rows)")
    print("\nSample:")
    print(combined.tail(10).to_string())


if __name__ == "__main__":
    main()
