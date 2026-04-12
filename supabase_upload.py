"""
supabase_upload.py

Uploads today's model_scores.csv to Supabase after Agent 3 runs.
Uses the secret key (server-side) so it bypasses RLS.
Stores all model data as JSONB to avoid schema cache issues.
"""
import os
import logging
import pandas as pd
from pathlib import Path
from supabase import create_client

logger = logging.getLogger("wizard.supabase")

NUMERIC_COLS = {
    "MF5i_prob", "MFull_prob", "MF1i_prob", "MF3i_prob",
    "MFull_proj_total", "MF1_proj", "MG3_proj",
}


def upload_picks(csv_path: Path) -> int:
    """
    Read model_scores.csv and upsert all rows into Supabase wizard_picks table.
    Stores all columns as a JSONB blob to avoid schema cache issues.
    Returns number of rows uploaded.
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SECRET_KEY")

    if not url or not key or key == "your-secret-key-here":
        logger.warning("Supabase credentials not set — skipping upload.")
        return 0

    if not csv_path.exists():
        logger.error(f"model_scores.csv not found at {csv_path}")
        return 0

    df = pd.read_csv(csv_path)
    df = df.where(pd.notnull(df), None)

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    run_date = str(df["game_date"].iloc[0])
    supabase = create_client(url, key)

    # Clear today's rows
    supabase.table("wizard_picks").delete().eq("game_date", run_date).execute()
    logger.info(f"[Supabase] 🗑️ Cleared existing rows for {run_date}.")

    # Insert each row: only game_date + matchup as real columns, everything else as JSONB
    success = 0
    for _, row in df.iterrows():
        # Convert NaN/inf to None so it's JSON-safe
        clean = {k: (None if (v != v or v is float('inf') or v == float('-inf')) else v)
                 for k, v in row.to_dict().items()}
        record = {
            "game_date": run_date,
            "matchup": row.get("matchup"),
            "data": clean,
        }
        try:
            supabase.table("wizard_picks").insert(record).execute()
            success += 1
        except Exception as e:
            logger.error(f"[Supabase] ❌ {record['matchup']} failed: {e}")

    logger.info(f"[Supabase] ✅ Uploaded {success}/{len(df)} rows to wizard_picks.")
    return success


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    env_file = Path(__file__).parent / "wizard_agents" / "env"
    if env_file.exists():
        load_dotenv(env_file)

    pipeline_dir = Path(os.environ.get("PIPELINE_DIR", Path(__file__).parent))
    csv_path = pipeline_dir / "model_scores.csv"

    n = upload_picks(csv_path)
    print(f"Uploaded {n} rows.")
    sys.exit(0 if n > 0 else 1)
