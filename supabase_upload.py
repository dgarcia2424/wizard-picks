"""
supabase_upload.py
==================
Pushes daily pipeline outputs to Supabase so the hosted dashboard can read them.

Uploads:
  daily_card.csv          → wizard_daily_card  (today's predictions)
  backtest_2026_results.csv → wizard_backtest  (season tracker)
  data/raw/bet_tracker.csv  → bet_tracker      (manually logged bets)

Run automatically by mlb_model_run.bat after run_today.py.
"""
import os
import sys
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("wizard.supabase")
logging.basicConfig(level=logging.INFO, format="%(message)s")

BASE_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

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
        raise RuntimeError("SUPABASE_URL or SUPABASE_KEY not set in .env")
    from supabase import create_client
    return create_client(url, key)


def _clean(row: dict) -> dict:
    """Make a dict JSON-safe (replace NaN/inf with None)."""
    out = {}
    for k, v in row.items():
        if v is None:
            out[k] = None
        elif isinstance(v, float) and (v != v or abs(v) == float("inf")):
            out[k] = None
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# UPLOAD FUNCTIONS
# ---------------------------------------------------------------------------

def upload_daily_card(csv_path: Path = None) -> int:
    """Upload today's run_today.py output to wizard_daily_card table."""
    csv_path = csv_path or BASE_DIR / "daily_card.csv"
    if not csv_path.exists():
        logger.warning(f"  [SKIP] {csv_path.name} not found")
        return 0

    df = pd.read_csv(csv_path)
    if df.empty:
        logger.warning("  [SKIP] daily_card.csv is empty")
        return 0

    client     = _get_client()
    game_date  = str(df["game"].iloc[0])  # use date from filename fallback
    # Try to get date from the data or use today
    import datetime
    game_date  = datetime.date.today().isoformat()

    # Clear today's rows
    client.table("wizard_daily_card").delete().eq("game_date", game_date).execute()

    success = 0
    for _, row in df.iterrows():
        # All data goes into the JSONB 'data' column — the app reads from there.
        # Top-level columns are only those that exist in the original table schema.
        # New fields (best_edge, mc_home_win, etc.) live inside 'data' only.
        full_data = _clean(row.to_dict())

        record = _clean({
            "game_date":    game_date,
            "game":         row.get("game", ""),
            "home_team":    row.get("home_team", ""),
            "away_team":    row.get("away_team", ""),
            "rl_signal":    row.get("rl_signal", ""),
            "total_signal": row.get("total_signal", ""),
            "blended_rl":   row.get("blended_rl"),
            "mc_rl":        row.get("mc_rl"),
            "xgb_rl":       row.get("xgb_rl"),
            "mc_total":     row.get("mc_total"),
            "blended_total":row.get("blended_total"),
            "vegas_total":  row.get("vegas_total"),
            "vegas_ml_home":row.get("vegas_ml_home"),
            "home_sp":      row.get("home_sp", ""),
            "away_sp":      row.get("away_sp", ""),
            "home_sp_xwoba":row.get("home_sp_xwoba"),
            "away_sp_xwoba":row.get("away_sp_xwoba"),
            "home_sp_flag": row.get("home_sp_flag", ""),
            "away_sp_flag": row.get("away_sp_flag", ""),
            "temp_f":       row.get("temp_f"),
            "lineup_confirmed": bool(row.get("lineup_confirmed", False)),
            "data":         full_data,   # all fields including new ones
        })
        try:
            client.table("wizard_daily_card").insert(record).execute()
            success += 1
        except Exception as e:
            logger.error(f"  [ERROR] {record['game']}: {e}")

    logger.info(f"  wizard_daily_card: {success}/{len(df)} rows uploaded for {game_date}")
    return success


def upload_backtest(csv_path: Path = None) -> int:
    """Upload season backtest results to wizard_backtest table."""
    csv_path = csv_path or BASE_DIR / "backtest_2026_results.csv"
    if not csv_path.exists():
        logger.warning(f"  [SKIP] {csv_path.name} not found")
        return 0

    df = pd.read_csv(csv_path)
    if df.empty:
        return 0

    client = _get_client()

    # Full replace — backtest file is the source of truth
    client.table("wizard_backtest").delete().neq("date", "1900-01-01").execute()

    success = 0
    for _, row in df.iterrows():
        record = _clean(row.to_dict())
        try:
            client.table("wizard_backtest").insert(record).execute()
            success += 1
        except Exception as e:
            logger.error(f"  [ERROR] backtest row {row.get('game', '')}: {e}")

    logger.info(f"  wizard_backtest: {success}/{len(df)} rows uploaded")
    return success


def upload_bet_tracker(csv_path: Path = None) -> int:
    """Upload bet tracker to Supabase bet_tracker table."""
    csv_path = csv_path or BASE_DIR / "data" / "raw" / "bet_tracker.csv"
    if not csv_path.exists():
        logger.warning(f"  [SKIP] bet_tracker.csv not found")
        return 0

    df = pd.read_csv(csv_path)
    if df.empty:
        return 0

    client = _get_client()
    client.table("bet_tracker").delete().neq("id", -1).execute()

    success = 0
    for _, row in df.iterrows():
        record = _clean(row.to_dict())
        try:
            client.table("bet_tracker").insert(record).execute()
            success += 1
        except Exception as e:
            logger.error(f"  [ERROR] bet_tracker row: {e}")

    logger.info(f"  bet_tracker: {success}/{len(df)} rows uploaded")
    return success


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 50)
    print("  supabase_upload.py")
    print("=" * 50)

    try:
        n1 = upload_daily_card()
        n2 = upload_backtest()
        n3 = upload_bet_tracker()
        total = n1 + n2 + n3
        print(f"\n  Done. {total} total rows uploaded to Supabase.")
        sys.exit(0)
    except RuntimeError as e:
        print(f"\n  [ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  [ERROR] Unexpected: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
