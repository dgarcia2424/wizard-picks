"""
backfill_history.py
====================
One-time script to populate wizard_daily_card with historical games
from backtest_2026_results.csv (March 28 – April 10).

Run once:  python backfill_history.py
"""
import os, sys, datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent

# ── reuse helpers from supabase_upload ──────────────────────────────────────
def _load_dotenv():
    env = BASE_DIR / ".env"
    if not env.exists():
        return
    with open(env) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

def _get_client():
    _load_dotenv()
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY not set")
    from supabase import create_client
    return create_client(url, key)

def _clean(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if v is None:
            out[k] = None
        elif isinstance(v, float) and (v != v or abs(v) == float("inf")):
            out[k] = None
        else:
            out[k] = v
    return out

# ── parse game string e.g. "NYY @ BOS" ──────────────────────────────────────
def _teams(game: str):
    parts = str(game).split("@")
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()   # away, home
    return "", ""

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    csv = BASE_DIR / "backtest_2026_results.csv"
    if not csv.exists():
        print(f"[ERROR] {csv} not found"); sys.exit(1)

    df = pd.read_csv(csv)
    print(f"Loaded {len(df)} rows from backtest CSV  ({df['date'].min()} – {df['date'].max()})")

    client = _get_client()

    # Group by date and upload each day's games
    total_ok = 0
    for date_str, grp in df.groupby("date"):
        # Skip dates that already have rows (don't overwrite today's live card)
        existing = client.table("wizard_daily_card").select("game_date").eq("game_date", date_str).execute()
        if existing.data:
            print(f"  [SKIP] {date_str} — already has {len(existing.data)} rows")
            continue

        day_ok = 0
        for _, row in grp.iterrows():
            away, home = _teams(row.get("game", ""))
            signal = str(row.get("signal") or "")
            # Map backtest columns → daily_card schema
            full_data = _clean({
                "game":           row.get("game", ""),
                "home_team":      home,
                "away_team":      away,
                "home_sp":        str(row.get("home_sp") or "").title(),
                "away_sp":        str(row.get("away_sp") or "").title(),
                "blended_rl":     row.get("blended_rl"),
                "mc_rl":          row.get("mc_rl"),
                "xgb_rl":         row.get("xgb_rl"),
                "mc_total":       row.get("model_total"),
                "blended_total":  row.get("model_total"),
                "vegas_total":    row.get("vegas_total"),
                "rl_signal":      signal,
                # Backtest outcome fields (shown in tracker, not on pick cards)
                "home_score":     row.get("home_score"),
                "away_score":     row.get("away_score"),
                "bet_win":        row.get("bet_win"),
                "home_covers_rl": row.get("home_covers_rl"),
            })
            record = _clean({
                "game_date":    date_str,
                "game":         row.get("game", ""),
                "home_team":    home,
                "away_team":    away,
                "rl_signal":    signal,
                "total_signal": "",
                "blended_rl":   row.get("blended_rl"),
                "mc_rl":        row.get("mc_rl"),
                "xgb_rl":       row.get("xgb_rl"),
                "mc_total":     row.get("model_total"),
                "blended_total":row.get("model_total"),
                "vegas_total":  row.get("vegas_total"),
                "vegas_ml_home":None,
                "home_sp":      str(row.get("home_sp") or "").title(),
                "away_sp":      str(row.get("away_sp") or "").title(),
                "home_sp_xwoba":None,
                "away_sp_xwoba":None,
                "home_sp_flag": "",
                "away_sp_flag": "",
                "temp_f":       None,
                "lineup_confirmed": False,
                "data":         full_data,
            })
            try:
                client.table("wizard_daily_card").insert(record).execute()
                day_ok += 1
            except Exception as e:
                print(f"    [ERROR] {record['game']}: {e}")

        print(f"  {date_str}: {day_ok}/{len(grp)} rows uploaded")
        total_ok += day_ok

    print(f"\nDone. {total_ok} total historical rows added to wizard_daily_card.")

if __name__ == "__main__":
    main()
