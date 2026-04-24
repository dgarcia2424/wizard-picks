"""
backfill_handedness.py
----------------------
One-time batch fetch of batSide.code for every unique player_id that appears
in the TB label set. Persists to data/statcast/handedness_cache.json.

Run:
    python backfill_handedness.py
    python backfill_handedness.py --sleep 0.15          # faster (still polite)
    python backfill_handedness.py --source labels       # default
    python backfill_handedness.py --source lineups      # union of all lineup parquets
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from lineup_pull import (  # noqa: E402
    _load_handedness_cache, _save_handedness_cache, fetch_handedness,
)

LABEL_FILE  = Path("data/batter_labels/tb_by_game.parquet")
LINEUP_DIR  = Path("data/statcast")


def collect_ids(source: str) -> list[int]:
    if source == "labels":
        if not LABEL_FILE.exists():
            raise SystemExit(f"Missing {LABEL_FILE}. Run build_tb_labels.py first.")
        df = pd.read_parquet(LABEL_FILE, columns=["player_id"])
        return sorted({int(p) for p in df["player_id"].dropna().unique()})
    if source == "lineups":
        ids: set[int] = set()
        for f in sorted(LINEUP_DIR.glob("lineups_*_long.parquet")):
            try:
                d = pd.read_parquet(f, columns=["player_id"])
                ids.update(int(p) for p in d["player_id"].dropna().unique())
            except Exception as exc:
                print(f"  [skip] {f.name}: {exc}")
        return sorted(ids)
    raise SystemExit(f"Unknown source: {source}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["labels", "lineups"], default="labels")
    ap.add_argument("--sleep",  type=float, default=0.15)
    args = ap.parse_args()

    ids = collect_ids(args.source)
    cache = _load_handedness_cache()
    missing = [pid for pid in ids if str(pid) not in cache]
    print(f"Source: {args.source} | unique ids: {len(ids):,} | "
          f"already cached: {len(ids) - len(missing):,} | "
          f"to fetch: {len(missing):,}")

    if not missing:
        print("Nothing to do.")
        return

    start = time.time()
    for i, pid in enumerate(missing, 1):
        fetch_handedness(pid, cache=cache)
        if i % 100 == 0 or i == len(missing):
            elapsed = time.time() - start
            print(f"  [{i:>5d}/{len(missing)}] elapsed={elapsed:5.1f}s")
            _save_handedness_cache(cache)  # checkpoint every 100
        time.sleep(args.sleep)

    _save_handedness_cache(cache)

    # Summary of returned codes
    codes = {"L": 0, "R": 0, "S": 0, "": 0}
    for pid in ids:
        c = cache.get(str(pid), "")
        codes[c] = codes.get(c, 0) + 1
    print("\nHandedness distribution:")
    for k, v in codes.items():
        label = k if k else "(unknown)"
        print(f"  {label}: {v:,}")


if __name__ == "__main__":
    main()
