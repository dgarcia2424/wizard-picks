"""
fetch_live_market_data.py — v5.0 Live Props + Totals Fetcher (The-Odds-API v4).

Replaces stubbed K-prop and totals fallbacks in fetch_live_odds.py with
real-time market data.  15-minute file cache prevents quota burn on
repeated intra-day calls.

API endpoints used:
  GET /v4/sports/baseball_mlb/events              — today's event IDs
  GET /v4/sports/baseball_mlb/events/{id}/odds   — player props per event
    markets: pitcher_strikeouts, batter_total_bases
  GET /v4/sports/baseball_mlb/odds/              — totals (already in
    odds_current_pull.py — this module supplements with finer props)

Outputs (saved by save_to_pipeline):
  data/statcast/k_props_{date}.parquet      — pitcher K over/under lines
  data/statcast/tb_props_{date}.parquet     — batter TB over/under lines
  odds_api_cache/props_{date}_{slot}.json   — raw 15-min cache shard

Usage:
  python fetch_live_market_data.py                    # today
  python fetch_live_market_data.py --date 2026-04-25
  python fetch_live_market_data.py --summary          # print counts only
  python fetch_live_market_data.py --force            # ignore cache
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parent

ODDS_API_KEY: Optional[str] = os.getenv("ODDS_API_KEY")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

CACHE_DIR   = _ROOT / "data/statcast/odds_api_cache"
OUTPUT_DIR  = _ROOT / "data/statcast"
QUOTA_LOG   = OUTPUT_DIR / "odds_api_quota.log"

CACHE_TTL_MINUTES = 15       # re-fetch if cache older than this
QUOTA_GUARD       = 30       # stop if remaining requests < this
FUZZY_THRESHOLD   = 0.72     # minimum SequenceMatcher ratio for name match
BOOKMAKERS_US     = "draftkings,fanduel,betmgm"
PROP_MARKETS      = "pitcher_strikeouts,batter_total_bases"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_slot(dt: datetime) -> str:
    """Round down to nearest 15-minute slot: '2026-04-24_1345'."""
    slot_min = (dt.minute // CACHE_TTL_MINUTES) * CACHE_TTL_MINUTES
    return dt.strftime(f"%Y-%m-%d_%H{slot_min:02d}")


def _cache_path(date_str: str, slot: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"props_{date_str}_{slot}.json"


def _load_cache(date_str: str) -> Optional[dict]:
    slot = _cache_slot(datetime.now(tz=timezone.utc))
    p = _cache_path(date_str, slot)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _save_cache(date_str: str, data: dict) -> None:
    slot = _cache_slot(datetime.now(tz=timezone.utc))
    p = _cache_path(date_str, slot)
    with open(p, "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Quota tracking
# ---------------------------------------------------------------------------

def _log_quota(remaining: Optional[str], used: Optional[str]) -> None:
    QUOTA_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).isoformat()
    with open(QUOTA_LOG, "a") as f:
        f.write(f"{ts}  remaining={remaining}  used={used}\n")


def _remaining_quota() -> int:
    """Read last logged remaining quota; returns large number if unknown."""
    if not QUOTA_LOG.exists():
        return 9999
    with open(QUOTA_LOG) as f:
        lines = [l for l in f.readlines() if "remaining=" in l]
    if not lines:
        return 9999
    last = lines[-1]
    try:
        val = last.split("remaining=")[1].split()[0]
        return int(val) if val.isdigit() else 9999
    except (IndexError, ValueError):
        return 9999


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get(url: str, params: dict) -> tuple[dict | list, dict]:
    """GET with requests; returns (body, response_headers).
    Raises RuntimeError on quota exhaustion or HTTP error."""
    import requests  # soft import — same as odds_current_pull.py

    if not ODDS_API_KEY:
        raise RuntimeError(
            "ODDS_API_KEY not set — add it to .env or set the environment variable")

    params = {**params, "apiKey": ODDS_API_KEY}
    r = requests.get(url, params=params, timeout=20)

    remaining = r.headers.get("x-requests-remaining")
    used      = r.headers.get("x-requests-used")
    _log_quota(remaining, used)

    if remaining and int(remaining) < QUOTA_GUARD:
        raise RuntimeError(
            f"Odds-API quota low ({remaining} remaining) — aborting to preserve budget")

    if r.status_code == 401:
        raise RuntimeError("Odds-API 401 Unauthorized — check ODDS_API_KEY")
    if r.status_code == 422:
        return [], r.headers   # event has no props yet — normal for early AM
    r.raise_for_status()
    return r.json(), r.headers


# ---------------------------------------------------------------------------
# Step 1: fetch today's MLB event IDs
# ---------------------------------------------------------------------------

def fetch_events(date_str: str) -> list[dict]:
    """Return list of {id, home_team, away_team, commence_time} for date_str."""
    url = f"{ODDS_API_BASE}/sports/baseball_mlb/events"
    body, _ = _get(url, {
        "dateFormat":    "iso",
        "commenceTimeFrom": f"{date_str}T00:00:00Z",
        "commenceTimeTo":   f"{date_str}T23:59:59Z",
    })
    if not isinstance(body, list):
        return []
    return [
        {
            "id":           e["id"],
            "home_team":    e.get("home_team", ""),
            "away_team":    e.get("away_team", ""),
            "commence_time": e.get("commence_time", ""),
        }
        for e in body
    ]


# ---------------------------------------------------------------------------
# Step 2: fetch player props for one event
# ---------------------------------------------------------------------------

def fetch_event_props(event_id: str) -> dict:
    """Return raw odds payload for one event (pitcher_strikeouts + batter_total_bases)."""
    url = f"{ODDS_API_BASE}/sports/baseball_mlb/events/{event_id}/odds"
    body, _ = _get(url, {
        "regions":    "us",
        "markets":    PROP_MARKETS,
        "bookmakers": BOOKMAKERS_US,
        "oddsFormat": "american",
        "dateFormat": "iso",
    })
    return body if isinstance(body, dict) else {}


# ---------------------------------------------------------------------------
# Fuzzy name matching
# ---------------------------------------------------------------------------

def fuzzy_match_name(query: str, candidates: list[str],
                     threshold: float = FUZZY_THRESHOLD) -> Optional[str]:
    """Return best-matching candidate name or None if below threshold.

    Normalises both strings to lowercase + strips punctuation before scoring.
    Prefers exact last-name match over fuzzy full-name ratio when both qualify.
    """
    if not query or not candidates:
        return None

    q_norm = query.lower().strip()
    q_last = q_norm.split()[-1] if q_norm.split() else q_norm

    best_ratio  = 0.0
    best_cand   = None

    for cand in candidates:
        c_norm = cand.lower().strip()
        c_last = c_norm.split()[-1] if c_norm.split() else c_norm

        # Exact last-name match wins immediately
        if q_last == c_last:
            return cand

        ratio = SequenceMatcher(None, q_norm, c_norm).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_cand  = cand

    return best_cand if best_ratio >= threshold else None


# ---------------------------------------------------------------------------
# Step 3: parse props payload → DataFrames
# ---------------------------------------------------------------------------

def _best_price(outcomes: list[dict], side: str) -> Optional[float]:
    """Return best (most favourable) American price for 'Over' or 'Under'."""
    prices = [o["price"] for o in outcomes
              if o.get("name", "").lower() == side.lower()
              and o.get("price") is not None]
    if not prices:
        return None
    # For Over: highest price; for Under: highest price (less juice)
    return max(prices)


def _parse_pitcher_k_rows(event_props: dict) -> list[dict]:
    """Extract pitcher_strikeouts rows from one event's bookmaker payload."""
    rows = {}   # keyed by (player_name, line) to dedup across books
    for book in event_props.get("bookmakers", []):
        for market in book.get("markets", []):
            if market.get("key") != "pitcher_strikeouts":
                continue
            # Group outcomes by player + line
            for o in market.get("outcomes", []):
                player = o.get("description", "")
                line   = o.get("point")
                if not player or line is None:
                    continue
                key = (player, float(line))
                if key not in rows:
                    rows[key] = {"pitcher_name": player, "line": float(line),
                                 "over_odds": None, "under_odds": None}
                side = o.get("name", "").lower()
                price = o.get("price")
                if side == "over" and price is not None:
                    prev = rows[key]["over_odds"]
                    rows[key]["over_odds"] = max(price, prev) if prev is not None else price
                elif side == "under" and price is not None:
                    prev = rows[key]["under_odds"]
                    rows[key]["under_odds"] = max(price, prev) if prev is not None else price
    return list(rows.values())


def _parse_batter_tb_rows(event_props: dict) -> list[dict]:
    """Extract batter_total_bases rows from one event's bookmaker payload."""
    rows = {}
    for book in event_props.get("bookmakers", []):
        for market in book.get("markets", []):
            if market.get("key") != "batter_total_bases":
                continue
            for o in market.get("outcomes", []):
                player = o.get("description", "")
                line   = o.get("point")
                if not player or line is None:
                    continue
                key = (player, float(line))
                if key not in rows:
                    rows[key] = {"batter_name": player, "line": float(line),
                                 "over_odds": None, "under_odds": None}
                side  = o.get("name", "").lower()
                price = o.get("price")
                if side == "over" and price is not None:
                    prev = rows[key]["over_odds"]
                    rows[key]["over_odds"] = max(price, prev) if prev is not None else price
                elif side == "under" and price is not None:
                    prev = rows[key]["under_odds"]
                    rows[key]["under_odds"] = max(price, prev) if prev is not None else price
    return list(rows.values())


# ---------------------------------------------------------------------------
# Main builder: all events → two DataFrames
# ---------------------------------------------------------------------------

def build_props_frame(date_str: str,
                      force: bool = False
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch or load cached props for date_str.

    Returns (k_props_df, tb_props_df).
      k_props_df  cols: pitcher_name, line, over_odds, under_odds, event_home, event_away
      tb_props_df cols: batter_name,  line, over_odds, under_odds, event_home, event_away
    """
    if not force:
        cached = _load_cache(date_str)
        if cached:
            print(f"  [live_market] loaded from 15-min cache ({date_str})")
            k  = pd.DataFrame(cached.get("k_props", []))
            tb = pd.DataFrame(cached.get("tb_props", []))
            return k, tb

    if not ODDS_API_KEY:
        print("  [live_market] ODDS_API_KEY not set — returning empty frames")
        return pd.DataFrame(), pd.DataFrame()

    if _remaining_quota() < QUOTA_GUARD:
        print(f"  [live_market] quota below {QUOTA_GUARD} — skipping live fetch")
        return pd.DataFrame(), pd.DataFrame()

    print(f"  [live_market] fetching events for {date_str} …")
    try:
        events = fetch_events(date_str)
    except Exception as exc:
        print(f"  [live_market] events fetch failed: {exc}")
        return pd.DataFrame(), pd.DataFrame()

    print(f"  [live_market] {len(events)} events found; fetching props …")

    all_k:  list[dict] = []
    all_tb: list[dict] = []

    for ev in events:
        try:
            props = fetch_event_props(ev["id"])
            if not props:
                continue
            for row in _parse_pitcher_k_rows(props):
                row["event_home"] = ev["home_team"]
                row["event_away"] = ev["away_team"]
                all_k.append(row)
            for row in _parse_batter_tb_rows(props):
                row["event_home"] = ev["home_team"]
                row["event_away"] = ev["away_team"]
                all_tb.append(row)
            time.sleep(0.05)   # gentle rate-limiting
        except Exception as exc:
            print(f"  [live_market] props failed for {ev['home_team']}: {exc}")

    k_df  = pd.DataFrame(all_k)  if all_k  else pd.DataFrame()
    tb_df = pd.DataFrame(all_tb) if all_tb else pd.DataFrame()

    # Persist cache
    _save_cache(date_str, {
        "k_props":  all_k,
        "tb_props": all_tb,
        "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
    })

    n_k  = len(k_df)
    n_tb = len(tb_df)
    print(f"  [live_market] K props: {n_k} lines | TB props: {n_tb} lines")
    return k_df, tb_df


# ---------------------------------------------------------------------------
# Pipeline persistence
# ---------------------------------------------------------------------------

def save_to_pipeline(date_str: str, force: bool = False) -> dict:
    """Build and save k_props and tb_props parquets for date_str.

    Returns summary dict with paths and counts.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    k_path  = OUTPUT_DIR / f"k_props_{date_str}.parquet"
    tb_path = OUTPUT_DIR / f"tb_props_{date_str}.parquet"

    k_df, tb_df = build_props_frame(date_str, force=force)

    if not k_df.empty:
        k_df.to_parquet(k_path, index=False)
    if not tb_df.empty:
        tb_df.to_parquet(tb_path, index=False)

    return {
        "k_props_path": str(k_path),
        "tb_props_path": str(tb_path),
        "n_k_props":  len(k_df),
        "n_tb_props": len(tb_df),
    }


# ---------------------------------------------------------------------------
# Fuzzy lookup helpers (called by fetch_live_odds.py and score_props_today.py)
# ---------------------------------------------------------------------------

def lookup_k_prop(k_df: pd.DataFrame, pitcher_name: str,
                  ) -> tuple[Optional[float], Optional[float]]:
    """Return (p_k_over, line) for a pitcher using fuzzy name matching.

    Mirrors the interface of _find_k_prop() in fetch_live_odds.py so it
    can be used as a drop-in replacement when live data is available.
    """
    from fetch_live_odds import _fair, _implied  # local import to avoid circular

    if k_df.empty or not pitcher_name:
        return None, None

    candidates = k_df["pitcher_name"].dropna().tolist()
    matched    = fuzzy_match_name(pitcher_name, candidates)
    if matched is None:
        return None, None

    row = k_df[k_df["pitcher_name"] == matched].iloc[0]
    line       = float(row.get("line") or 4.5)
    over_odds  = row.get("over_odds")
    under_odds = row.get("under_odds")

    try:
        if over_odds is not None and not pd.isna(over_odds) and \
           under_odds is not None and not pd.isna(under_odds):
            p = _fair(over_odds, under_odds)
        elif over_odds is not None and not pd.isna(over_odds):
            p = _implied(over_odds)
        else:
            return None, None
    except (TypeError, ValueError):
        return None, None

    import numpy as np
    return float(np.clip(p, 0.05, 0.95)), line


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch live props from The-Odds-API v4")
    parser.add_argument("--date",    default=date.today().isoformat())
    parser.add_argument("--force",   action="store_true", help="Ignore cache")
    parser.add_argument("--summary", action="store_true", help="Print counts only")
    args = parser.parse_args()

    result = save_to_pipeline(args.date, force=args.force)

    if not args.summary:
        k_df, tb_df = build_props_frame(args.date)
        if not k_df.empty:
            print(f"\n  Pitcher K lines ({len(k_df)} props):")
            for _, r in k_df.iterrows():
                print(f"    {r['pitcher_name']:30s}  O{r['line']:.1f}  "
                      f"over={r.get('over_odds','?'):>6}  under={r.get('under_odds','?'):>6}")
        if not tb_df.empty:
            print(f"\n  Batter TB lines ({len(tb_df)} props):")
            for _, r in tb_df.head(10).iterrows():
                print(f"    {r['batter_name']:30s}  O{r['line']:.1f}")

    print(f"\n  K props:  {result['n_k_props']}  → {result['k_props_path']}")
    print(f"  TB props: {result['n_tb_props']}  → {result['tb_props_path']}")


if __name__ == "__main__":
    main()
