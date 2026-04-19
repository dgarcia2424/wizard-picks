"""
config/settings.py
Central configuration for The Wizard Report pipeline.
All model parameters, thresholds, and file paths live here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Pipeline directory ────────────────────────────────────────────────────────
PIPELINE_DIR = Path(os.getenv(
    "PIPELINE_DIR",
    r"C:\Users\garci\OneDrive\Documents\Claude\mlb_model_pipeline_dan"
))

# ── File paths ────────────────────────────────────────────────────────────────
FILES = {
    "games":                 PIPELINE_DIR / "games.csv",
    "model_scores":          PIPELINE_DIR / "model_scores.csv",
    "model_report":          PIPELINE_DIR / "model_report.html",
    "bet_tracker":           PIPELINE_DIR / "data/raw/bet_tracker.csv",
    "fangraphs_pitchers":    PIPELINE_DIR / "data/raw/fangraphs_pitchers.csv",
    "fangraphs_batters":     PIPELINE_DIR / "data/raw/fangraphs_batters.csv",
    "fangraphs_team_vs_lhp": PIPELINE_DIR / "data/raw/fangraphs_team_vs_lhp.csv",
    "fangraphs_team_vs_rhp": PIPELINE_DIR / "data/raw/fangraphs_team_vs_rhp.csv",
    "savant_pitchers":       PIPELINE_DIR / "data/raw/savant_pitchers.csv",
    "savant_batters":        PIPELINE_DIR / "data/raw/savant_batters.csv",
}

STATIC_FILES = [
    "fangraphs_pitchers",
    "fangraphs_batters",
    "fangraphs_team_vs_lhp",
    "fangraphs_team_vs_rhp",
    "savant_pitchers",
    "savant_batters",
]

STALE_THRESHOLD_DAYS = 7

# ── API credentials ───────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ODDS_API_KEY      = os.getenv("ODDS_API_KEY", "")
GMAIL_FROM        = os.getenv("GMAIL_FROM", "garcia.dan24@gmail.com")
GMAIL_PASSWORD    = os.getenv("GMAIL_APP_PASSWORD", "")
EMAIL_RECIPIENTS  = os.getenv("EMAIL_RECIPIENTS", "garcia.dan24@gmail.com").split(",")

# ── Model parameters ──────────────────────────────────────────────────────────
PICK_THRESHOLD = 0.63

YEAR_WEIGHTS = {2026: 0.30, 2025: 0.55, 2024: 0.15}

PTS_SCALE          = 0.60
F5_ESTIMATE_RATIO  = 0.555
F3_ESTIMATE_RATIO  = 0.32
F1_ESTIMATE_RATIO  = 0.11

MODEL_PARAMS = {
    "MFull": {
        "bet_type":             "Full game Over/Under",
        "coors_under_penalty":  0.10,
        "home_field_bonus":     0.0,
        "apply_lineup_quality": True,    # Validated +7.3% accuracy
        "base_calibration":     None,    # Not locked
    },
    "MF5i": {
        "bet_type":             "First 5 innings moneyline",
        "coors_under_penalty":  0.0,
        "home_field_bonus":     8.0,     # Raises accuracy 45.5% → 66.7% — always apply
        "apply_lineup_quality": False,
        "base_calibration":     None,
    },
    "MF3i": {
        "bet_type":             "First 3 innings Over/Under",
        "coors_under_penalty":  0.10,
        "home_field_bonus":     0.0,
        "apply_lineup_quality": False,
        "base_calibration":     0.58,    # LOCKED — do NOT raise (0.64 caused regression)
        "calibration_step":     0.05,
        "calibration_floor":    0.13,
    },
    "MF1i": {
        "bet_type":             "First inning Over/Under",
        "coors_under_penalty":  0.0,     # Park effects don't dominate inning 1
        "home_field_bonus":     0.0,
        "apply_lineup_quality": False,
        "base_calibration":     None,
    },
    "MBat": {
        "bet_type":             "Batter hit prop",
        "method":               "bernoulli",
        "coors_under_penalty":  0.0,
        "home_field_bonus":     0.0,
        "apply_lineup_quality": False,
        "base_calibration":     None,
    },
}

UNIT_TIERS = [
    (0.72, 1.5),
    (0.67, 1.0),
    (0.63, 0.5),
]

def get_unit_size(prob: float) -> float:
    for threshold, units in UNIT_TIERS:
        if prob >= threshold:
            return units
    return 0.0

# ── Tracker server ────────────────────────────────────────────────────────────
TRACKER_PORT = int(os.getenv("TRACKER_SERVER_PORT", 5151))

# ── Claude model ──────────────────────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-20250514"
