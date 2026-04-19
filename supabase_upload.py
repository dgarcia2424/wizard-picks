"""
supabase_upload.py
==================
Pushes daily pipeline outputs to Supabase so the hosted dashboard can read them.

Uploads:
  daily_card.csv               → wizard_daily_card   (today's predictions)
  backtest_2026_results.csv    → wizard_backtest     (season tracker)
  backtest_mc_2026_results.csv → wizard_model_history (enriched backtest w/ F5/NRFI/K actuals)
  pipeline_status.json         → wizard_pipeline_health (data freshness status)

Run automatically by mlb_model_run.bat after run_today.py.
"""
import hashlib
import json as _json
import os
import sys
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("wizard.supabase")
logging.basicConfig(level=logging.INFO, format="%(message)s")

BASE_DIR = Path(__file__).parent
_HASH_FILE = BASE_DIR / ".supabase_hashes.json"


def _load_hashes() -> dict:
    if _HASH_FILE.exists():
        try:
            return _json.loads(_HASH_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_hashes(hashes: dict) -> None:
    _HASH_FILE.write_text(_json.dumps(hashes, indent=2))


def _file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _unchanged(key: str, path: Path, hashes: dict) -> bool:
    current = _file_hash(path)
    if current and hashes.get(key) == current:
        logger.info(f"  [{key}] unchanged — skipping upload")
        return True
    return False


def _mark_uploaded(key: str, path: Path, hashes: dict) -> None:
    hashes[key] = _file_hash(path)
    _save_hashes(hashes)


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

def upload_daily_card(csv_path: Path = None, game_date: str = None) -> int:
    """Upload a daily card CSV to wizard_daily_card table.

    When csv_path is None, falls back to scanning for all dated
    daily_card_{date}.csv files and uploading each one via upload_all_daily_cards().

    game_date overrides the auto-detected date (useful for testing).
    """
    import datetime, re

    # If no path given, scan for all dated files
    if csv_path is None:
        return upload_all_daily_cards()

    if not csv_path.exists():
        logger.warning(f"  [SKIP] {csv_path.name} not found")
        return 0

    df = pd.read_csv(csv_path)
    if df.empty:
        logger.warning(f"  [SKIP] {csv_path.name} is empty")
        return 0

    # Infer date from filename (daily_card_YYYY-MM-DD.csv) or fall back to today
    if game_date is None:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", csv_path.name)
        game_date = m.group(1) if m else datetime.date.today().isoformat()

    client = _get_client()

    # Clear existing rows for this date before re-inserting
    client.table("wizard_daily_card").delete().eq("game_date", game_date).execute()

    success = 0
    for _, row in df.iterrows():
        # All data goes into the JSONB 'data' column — the app reads from there.
        # Top-level columns are only those that exist in the original table schema.
        # New fields (best_edge, mc_home_win, etc.) live inside 'data' only.
        full_data = _clean(row.to_dict())

        # Only include columns that exist in the wizard_daily_card schema.
        # All new fields (lineup_wrc, mc_nrfi_prob, F5, K props, etc.) are
        # stored in the 'data' JSONB column and read by the dashboard from there.
        # This avoids 400 errors when new model fields are added without ALTER TABLE.
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
            "data":         full_data,   # all fields — dashboard reads from here
        })
        try:
            client.table("wizard_daily_card").insert(record).execute()
            success += 1
        except Exception as e:
            logger.error(f"  [ERROR] {record['game']}: {e}")

    logger.info(f"  wizard_daily_card: {success}/{len(df)} rows uploaded for {game_date}")
    return success


def upload_all_daily_cards() -> int:
    """Upload today's (and tomorrow's, if available) daily card only.

    Historical cards are already in Supabase and don't change — re-uploading
    them every run wastes quota and time.
    """
    import datetime, re
    today = datetime.date.today().isoformat()
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()

    _cards_subdir = BASE_DIR / "daily_cards"
    total = 0

    for target_date in [today, tomorrow]:
        # Prefer dated file over legacy daily_card.csv
        candidates = [
            _cards_subdir / f"daily_card_{target_date}.csv",
            BASE_DIR / f"daily_card_{target_date}.csv",
        ]
        if target_date == today:
            candidates.append(BASE_DIR / "daily_card.csv")

        path = next((p for p in candidates if p.exists()), None)
        if path:
            total += upload_daily_card(csv_path=path, game_date=target_date)

    return total


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



def upload_model_history(csv_path: Path = None) -> int:
    """Upload enriched backtest (with F5/NRFI/K actuals) to wizard_model_history table.

    Schema expected in wizard_model_history:
      date, home_team, away_team, home_sp, away_sp,
      blended_rl, mc_rl,
      home_covers_rl (int 0/1), bet_win (int 0/1), signal,
      mc_f5_total, f5_total_actual,
      mc_nrfi_prob, f1_nrfi_actual,
      mc_home_sp_k_mean, home_sp_k_actual,
      mc_away_sp_k_mean, away_sp_k_actual

    Strategy: full replace — the CSV is the source of truth.
    """
    csv_path = csv_path or BASE_DIR / "backtest_mc_2026_results.csv"
    if not csv_path.exists():
        logger.warning(f"  [SKIP] {csv_path.name} not found")
        return 0

    df = pd.read_csv(csv_path)
    if df.empty:
        logger.warning("  [SKIP] backtest_mc_2026_results.csv is empty")
        return 0

    # Pre-convert boolean/object columns that Supabase expects as integers.
    # bet_win arrives as object dtype with Python True/False/NaN.
    # Convert at DataFrame level so row.get() never returns a bool.
    for _bool_col in ["bet_win", "home_covers_rl"]:
        if _bool_col in df.columns:
            df[_bool_col] = df[_bool_col].map(
                lambda x: None if (x is None or (isinstance(x, float) and x != x)) else int(x)
            )

    client = _get_client()

    # Full replace — delete all existing rows then re-insert
    client.table("wizard_model_history").delete().neq("date", "1900-01-01").execute()

    # CSV column name → Supabase column name mapping
    # (CSV was built by backtest_mc_2026.py; schema uses more explicit names)
    COL_MAP = {
        "game_date":    "date",
        "home_team":    "home_team",
        "away_team":    "away_team",
        "home_sp":      "home_sp",
        "away_sp":      "away_sp",
        "blended_rl":   "blended_rl",
        "mc_rl":        "mc_rl",
        "home_covers_rl": "home_covers_rl",
        "bet_win":      "bet_win",
        "signal":       "signal",
        # MC prediction columns (may be absent if MC not yet run)
        "mc_f5_total":      "mc_f5_total",
        "mc_nrfi_prob":     "mc_nrfi_prob",
        "mc_home_sp_k_mean":"mc_home_sp_k_mean",
        "mc_away_sp_k_mean":"mc_away_sp_k_mean",
        # Actual result columns — CSV uses shorter names
        "f5_total":     "f5_total_actual",
        "f1_nrfi":      "f1_nrfi_actual",
        "home_sp_k":    "home_sp_k_actual",
        "away_sp_k":    "away_sp_k_actual",
    }

    # Integer columns that may arrive as Python booleans from pandas
    BOOL_AS_INT = {"home_covers_rl", "bet_win", "f1_nrfi_actual"}

    success = 0
    for _, row in df.iterrows():
        # Build record using the column mapping (CSV name → Supabase name)
        raw = {}
        for csv_col, db_col in COL_MAP.items():
            val = row.get(csv_col)
            raw[db_col] = val

        # Convert bool/NaN → int (0/1/None) for integer columns
        for col in BOOL_AS_INT:
            if col in raw:
                v = raw[col]
                if v is None:
                    pass  # leave as None
                else:
                    try:
                        import math
                        raw[col] = None if (isinstance(v, float) and math.isnan(v)) else int(v)
                    except (ValueError, TypeError):
                        raw[col] = None

        record = _clean(raw)
        try:
            client.table("wizard_model_history").insert(record).execute()
            success += 1
        except Exception as e:
            logger.error(
                f"  [ERROR] model_history row "
                f"{row.get('date', '?')} {row.get('home_team', '?')}: {e}"
            )

    logger.info(f"  wizard_model_history: {success}/{len(df)} rows uploaded")
    return success


def upload_historical_backtest() -> int:
    """Upload multi-year historical backtest results to wizard_backtest_historical table.

    For each year in [2023, 2024, 2025], reads backtest_{year}_results.csv and
    upserts all rows (delete-then-insert per season) into the table.
    """
    years = [2023, 2024, 2025]
    total_uploaded = 0

    for year in years:
        csv_path = BASE_DIR / f"backtest_{year}_results.csv"
        if not csv_path.exists():
            logger.warning(f"  [SKIP] backtest_{year}_results.csv not found")
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            logger.warning(f"  [SKIP] backtest_{year}_results.csv is empty")
            continue

        # Only upload rows where a bet signal was generated (not all games)
        # This keeps the table small (~1,100 rows/year vs 2,400) and avoids
        # hitting Supabase's 1,000-row default page limit on reads.
        if "signal" in df.columns and "bet_win" in df.columns:
            df = df[df["signal"].notna() & (df["signal"] != "") & df["bet_win"].notna()]
        if df.empty:
            logger.warning(f"  [SKIP] {year}: no signal rows after filtering")
            continue

        df["season"] = year

        # Float columns that may have landed as object dtype
        for _col in ["home_sp_xwoba", "away_sp_xwoba", "blended_rl",
                     "actual_total", "vegas_ml_home", "vegas_ml_away",
                     "vegas_total", "home_score", "away_score",
                     "edge", "vegas_implied"]:
            if _col in df.columns:
                df[_col] = pd.to_numeric(df[_col], errors="coerce")

        # Integer columns: "True"/"False"/"0.0"/"1.0"/NaN → Python int or None
        # Must coerce per-row (not via .map) so numpy types become Python ints
        _INT_HIST_COLS = {"bet_win", "home_covers_rl", "actual_home_win"}

        client = _get_client()

        # Delete all rows for this season first (upsert pattern)
        try:
            client.table("wizard_backtest_historical").delete().eq("season", year).execute()
        except Exception as e:
            logger.warning(f"  [WARN] Could not delete {year} rows: {e}")

        success = 0
        for _, row in df.iterrows():
            raw = row.to_dict()
            # Force int columns to Python int (numpy float 0.0/1.0 and bool strings)
            for _c in _INT_HIST_COLS:
                if _c in raw:
                    v = raw[_c]
                    if v is None or (isinstance(v, float) and v != v):
                        raw[_c] = None
                    else:
                        try:
                            s = str(v).strip().lower()
                            raw[_c] = None if s in ("nan","none","") else (
                                1 if s in ("true","1","1.0") else
                                0 if s in ("false","0","0.0") else
                                int(float(s))
                            )
                        except (ValueError, TypeError):
                            raw[_c] = None
            record = _clean(raw)
            try:
                client.table("wizard_backtest_historical").insert(record).execute()
                success += 1
            except Exception as e:
                logger.error(f"  [ERROR] historical_backtest {year} row: {e}")

        logger.info(f"  wizard_backtest_historical ({year}): {success}/{len(df)} rows uploaded")
        total_uploaded += success

    return total_uploaded


def upload_pipeline_health(status: dict = None) -> int:
    """Upload pipeline health status to wizard_pipeline_health table.

    If status is None, reads from pipeline_status.json.
    Upserts on date (one row per day, updated throughout the day).
    """
    import json as _json

    if status is None:
        path = BASE_DIR / "pipeline_status.json"
        if not path.exists():
            logger.warning("  [SKIP] pipeline_status.json not found")
            return 0
        try:
            status = _json.loads(path.read_text())
        except Exception as e:
            logger.warning(f"  [SKIP] Could not read pipeline_status.json: {e}")
            return 0

    if not status:
        return 0

    client = _get_client()
    today  = status.get("date", str(__import__("datetime").date.today()))

    record = _clean({
        "date":            today,
        "generated_at":    status.get("generated_at", ""),
        "overall":         status.get("overall", "ok"),
        "picks_ready":     bool(status.get("picks_ready", False)),
        "missing_critical": _json.dumps(status.get("missing_critical", [])),
        "missing_optional": _json.dumps(status.get("missing_optional", [])),
        "warnings":         _json.dumps(status.get("warnings", [])),
        "artifacts_json":   _json.dumps(status.get("artifacts", {})),
    })

    try:
        # Upsert: delete today's row then insert fresh
        client.table("wizard_pipeline_health").delete().eq("date", today).execute()
        client.table("wizard_pipeline_health").insert(record).execute()
        logger.info(f"  wizard_pipeline_health: uploaded status={status.get('overall')} for {today}")
        return 1
    except Exception as e:
        logger.error(f"  [ERROR] pipeline_health upload: {e}")
        return 0


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 50)
    print("  supabase_upload.py")
    print("=" * 50)

    hashes = _load_hashes()

    # Hash-checked uploads: skip if source file hasn't changed since last upload
    _hashed = {
        "backtest":            BASE_DIR / "backtest_2026_results.csv",
        "model_history":       BASE_DIR / "backtest_mc_2026_results.csv",
    }

    # Each upload runs independently — one missing table won't stop the others
    results = {}
    for name, fn in [
        ("daily_card",          upload_all_daily_cards),   # always — odds/lineups change
        ("backtest",            upload_backtest),
        ("model_history",       upload_model_history),
        ("historical_backtest", upload_historical_backtest),
        ("pipeline_health",     upload_pipeline_health),   # always — reflects freshness
    ]:
        # historical_backtest: hash all three year files combined
        if name == "historical_backtest":
            combined = "".join(
                _file_hash(BASE_DIR / f"backtest_{y}_results.csv")
                for y in [2023, 2024, 2025]
            )
            if hashes.get("historical_backtest") == combined:
                logger.info("  [historical_backtest] unchanged — skipping upload")
                results[name] = -1
                continue
        elif name in _hashed and _unchanged(name, _hashed[name], hashes):
            results[name] = -1  # sentinel: skipped
            continue
        try:
            results[name] = fn()
            if name == "historical_backtest" and results[name] > 0:
                combined = "".join(
                    _file_hash(BASE_DIR / f"backtest_{y}_results.csv")
                    for y in [2023, 2024, 2025]
                )
                hashes["historical_backtest"] = combined
                _save_hashes(hashes)
            elif name in _hashed and results[name] > 0:
                _mark_uploaded(name, _hashed[name], hashes)
        except RuntimeError as e:
            logger.error(f"  [ERROR] {name}: {e}")
            results[name] = 0
        except Exception as e:
            logger.error(f"  [ERROR] {name} unexpected: {e}")
            results[name] = 0

    total = sum(n for n in results.values() if n > 0)
    print(f"\n  Done. {total} total rows uploaded.")
    for name, n in results.items():
        status = "SKIP" if n == -1 else ("OK" if n > 0 else "SKIP/FAIL")
        label = "no change" if n == -1 else f"{n} rows"
        print(f"    [{status:9s}] {name}: {label}")

    # Exit non-zero only if the critical uploads (daily_card, backtest) both failed (not skipped)
    if results["daily_card"] == 0 and results.get("backtest", -1) == 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
