"""
Run this in your wizard_agents folder to see exactly where the pipeline
is looking for files and what's actually on disk.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PIPELINE_DIR = Path(os.getenv(
    "PIPELINE_DIR",
    r"C:\Users\garci\OneDrive\Documents\Claude\mlb_model_pipeline_dan"
))

FILES = {
    "savant_pitchers":       PIPELINE_DIR / "data/raw/savant_pitchers.csv",
    "savant_batters":        PIPELINE_DIR / "data/raw/savant_batters.csv",
    "fangraphs_pitchers":    PIPELINE_DIR / "data/raw/fangraphs_pitchers.csv",
    "fangraphs_batters":     PIPELINE_DIR / "data/raw/fangraphs_batters.csv",
    "fangraphs_team_vs_lhp": PIPELINE_DIR / "data/raw/fangraphs_team_vs_lhp.csv",
    "fangraphs_team_vs_rhp": PIPELINE_DIR / "data/raw/fangraphs_team_vs_rhp.csv",
}

print(f"\nPIPELINE_DIR from .env : {os.getenv('PIPELINE_DIR', 'NOT SET — using default')}")
print(f"Resolved PIPELINE_DIR  : {PIPELINE_DIR}")
print(f"PIPELINE_DIR exists    : {PIPELINE_DIR.exists()}")
print()

for key, path in FILES.items():
    exists = path.exists()
    size   = f"{path.stat().st_size // 1024}KB" if exists else "—"
    status = "✅" if exists else "❌ NOT FOUND"
    print(f"  {status}  {key:<30} {size:<8} {path}")

print()
print("Files actually in PIPELINE_DIR:")
if PIPELINE_DIR.exists():
    for f in sorted(PIPELINE_DIR.iterdir()):
        print(f"  {f.name}")
else:
    print("  ❌ Directory does not exist")
