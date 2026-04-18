"""
prizepicks_pull.py
==================
Pull today's MLB player prop lines from the PrizePicks semi-public API
and save to data/statcast/prizepicks_mlb_{date}.parquet.

Stat types captured:
  - Pitcher Strikeouts   → matches our K prop model
  - Hits                 → 0.5 line = P(1+ hit), matches hit prop model
  - Home Runs            → 0.5 demon line = P(HR), matches HR prop model
  - Total Bases          → future model

Odds types:
  standard = base line (~50/50)
  demon    = line pushed up, implied P(over) < 50% (popular unders)
  goblin   = line pushed down, implied P(over) > 50% (popular overs)

PrizePicks Power Play break-even per pick (assuming independence):
  2-pick  3x  → 57.7% per pick
  3-pick  5x  → 58.5% per pick
  4-pick 10x  → 56.2% per pick
  Practical threshold: model must exceed ~58% to have positive EV.

Usage:
    python prizepicks_pull.py
    python prizepicks_pull.py --date 2026-04-18
"""

import argparse
import datetime
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR   = Path("data/statcast")
LEAGUE_ID  = 2          # MLB on PrizePicks
PER_PAGE   = 250        # max rows per API call (pagination needed for full pull)
RATE_SLEEP = 2.0        # seconds between paginated requests

KEY_STATS = {
    "Pitcher Strikeouts",
    "Hits",
    "Home Runs",
    "Total Bases",
}

API_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":           "application/json",
    "Accept-Language":  "en-US,en;q=0.9",
    "Accept-Encoding":  "gzip, deflate, br",
    "Referer":          "https://app.prizepicks.com/",
    "Origin":           "https://app.prizepicks.com",
    "Connection":       "keep-alive",
    "Sec-Fetch-Dest":   "empty",
    "Sec-Fetch-Mode":   "cors",
    "Sec-Fetch-Site":   "same-site",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    """Lowercase, strip accents — for name matching."""
    n = unicodedata.normalize("NFD", str(name).strip())
    return "".join(c for c in n if unicodedata.category(c) != "Mn").lower()


def _fetch_page(page: int = 1) -> dict:
    url = (
        f"https://api.prizepicks.com/projections"
        f"?league_id={LEAGUE_ID}&per_page={PER_PAGE}&page={page}"
    )
    r = requests.get(url, headers=API_HEADERS, timeout=20)
    if r.status_code == 429:
        print("  [WARN] PrizePicks rate-limited — sleeping 15s")
        time.sleep(15)
        r = requests.get(url, headers=API_HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Main pull
# ---------------------------------------------------------------------------

def pull(date_str: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """
    Fetch all today's MLB PrizePicks projections, returning a DataFrame with:
    columns: date, player_name, player_name_norm, team, stat_type, line,
             odds_type, player_id, game_id, start_time
    """
    date_str = date_str or str(datetime.date.today())

    if verbose:
        print(f"\n  [PrizePicks] Pulling MLB props for {date_str} ...")

    # ── Paginate through all projections ─────────────────────────────────────
    all_projections = []
    all_included    = {}
    page = 1

    while True:
        raw = _fetch_page(page)
        batch = raw.get("data", [])
        all_projections.extend(batch)

        # Accumulate included (player/game/etc.) — deduplicate by id+type
        for item in raw.get("included", []):
            key = (item["type"], item["id"])
            all_included[key] = item

        # Check for next page
        meta = raw.get("meta", {})
        total_pages = meta.get("total_pages") or 1
        if verbose:
            print(f"  [PrizePicks] page {page}/{total_pages} "
                  f"({len(batch)} rows) ...", end="\r")
        if page >= total_pages:
            break
        page += 1
        time.sleep(RATE_SLEEP)

    if verbose:
        print(f"  [PrizePicks] fetched {len(all_projections)} total projections")

    # ── Build player lookup ───────────────────────────────────────────────────
    players: dict[str, dict] = {
        item["id"]: item["attributes"]
        for (typ, _), item in all_included.items()
        if typ == "new_player"
    }

    # ── Filter + parse ────────────────────────────────────────────────────────
    rows = []
    for proj in all_projections:
        attr = proj["attributes"]

        # Only today, pre-game, key stat types
        if not attr.get("today"):
            continue
        if attr.get("status") != "pre_game":
            continue
        if attr.get("stat_type") not in KEY_STATS:
            continue

        pid     = proj["relationships"]["new_player"]["data"]["id"]
        pl      = players.get(pid, {})
        name    = pl.get("name", "Unknown")
        team    = pl.get("team", "")
        game_id = proj["relationships"].get("game", {}).get("data", {}).get("id", "")

        rows.append({
            "date":             date_str,
            "player_id":        pid,
            "player_name":      name,
            "player_name_norm": _norm(name),
            "team":             team,
            "stat_type":        attr["stat_type"],
            "line":             float(attr["line_score"]),
            "odds_type":        attr.get("odds_type", "standard"),
            "game_id":          game_id,
            "start_time":       attr.get("start_time", ""),
            "projection_id":    proj["id"],
        })

    df = pd.DataFrame(rows)

    if df.empty:
        print("  [PrizePicks] No matching props found for today.")
        return df

    # ── Save ──────────────────────────────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"prizepicks_mlb_{date_str}.parquet"
    df.to_parquet(out, index=False)

    if verbose:
        counts = df.groupby(["stat_type", "odds_type"]).size().reset_index(name="n")
        print(f"  [PrizePicks] Saved {len(df)} props -> {out}")
        print()
        for _, row in counts.iterrows():
            print(f"    {row['stat_type']:22s}  {row['odds_type']:10s}  {row['n']}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pull PrizePicks MLB props")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    args = parser.parse_args()
    pull(args.date, verbose=True)


if __name__ == "__main__":
    main()
