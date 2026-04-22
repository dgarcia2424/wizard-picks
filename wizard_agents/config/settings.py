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
    "historical_picks":      PIPELINE_DIR / "historical_actionable_picks.csv",
    "actuals_2026":          PIPELINE_DIR / "data/statcast/actuals_2026.parquet",
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

# ── Odds API regional config ──────────────────────────────────────────────────
# Premium-tier dual-region ingestion:
#   US  — retail reference (DraftKings, FanDuel, BetMGM)
#   EU  — sharp benchmark (Pinnacle only)
# Pinnacle de-vigged lines become P_true. US de-vigged lines become Retail_Implied_Prob.
ODDS_API_US_BOOKMAKERS = ["draftkings", "fanduel", "betmgm"]
ODDS_API_EU_BOOKMAKERS = ["pinnacle"]

# ── Model parameters ──────────────────────────────────────────────────────────
YEAR_WEIGHTS = {2026: 0.30, 2025: 0.55, 2024: 0.15}

PTS_SCALE          = 0.60
F5_ESTIMATE_RATIO  = 0.555
F3_ESTIMATE_RATIO  = 0.32
F1_ESTIMATE_RATIO  = 0.11

# ── Three-Part Lock (execution gate) ─────────────────────────────────────────
# A game must pass ALL THREE conditions to be flagged actionable at 4:45 PM ET.
#
# 1. Sanity Check — model must closely agree with Pinnacle's sharp line.
#    Filters out games where late injury/news has moved the sharp market away
#    from the model without the model knowing.
SANITY_CHECK_TOLERANCE = 0.04        # abs(P_model - P_true) <= 0.04

# 2. Odds Floor — do not bet heavily-priced favorites.
#    American odds must be >= -225 (i.e., -225, -200, -150, +110, etc. all pass;
#    -230, -300 etc. fail).
ODDS_FLOOR_AMERICAN = -225

# 3. Two-Tier CLV Edge — edge against the retail book's de-vigged line.
TIER1_EDGE_THRESHOLD = 0.030         # Strong edge  : Edge >= 3.0%
TIER2_EDGE_THRESHOLD = 0.010         # Medium edge  : Edge >= 1.0% and < 3.0%

# ── Dynamic Kelly Staking ─────────────────────────────────────────────────────
# Fractional Kelly sizing against a synthetic bankroll.
# Final stake is always rounded to the nearest whole dollar and hard-capped.
SYNTHETIC_BANKROLL   = 2_000.0       # Synthetic unit base (not real cash at risk)
KELLY_FRACTION_TIER1 = 0.25          # Tier 1: Quarter-Kelly
KELLY_FRACTION_TIER2 = 0.125         # Tier 2: Eighth-Kelly
MAX_BET_DOLLARS      = 50.00         # Hard cap per wager


def american_to_decimal(american: int) -> float:
    """Convert American odds to decimal odds."""
    if american > 0:
        return (american / 100.0) + 1.0
    return (100.0 / abs(american)) + 1.0


def kelly_stake(
    p_model: float,
    retail_american: int,
    tier: int,
) -> int:
    """
    Calculate the final integer dollar stake using fractional Kelly sizing.

    Parameters
    ----------
    p_model         : Model win probability for the target side.
    retail_american : American odds at the lagging retail book for that side.
    tier            : 1 = Quarter-Kelly (Tier 1), 2 = Eighth-Kelly (Tier 2).

    Returns
    -------
    Integer dollar amount to wager, hard-capped at MAX_BET_DOLLARS.
    Returns 0 if the Kelly fraction is non-positive (no edge after de-vig).

    Formula
    -------
    b       = decimal_odds - 1   (net profit per $1 wagered)
    f_star  = (b * p - q) / b    (full Kelly fraction)
    raw     = SYNTHETIC_BANKROLL * f_star * kelly_multiplier
    final   = min(raw, MAX_BET_DOLLARS), rounded to nearest $1
    """
    b = american_to_decimal(retail_american) - 1.0
    p = p_model
    q = 1.0 - p_model

    if b <= 0:
        return 0

    f_star = (b * p - q) / b
    if f_star <= 0:
        return 0

    kelly_mult = KELLY_FRACTION_TIER1 if tier == 1 else KELLY_FRACTION_TIER2
    raw_stake  = SYNTHETIC_BANKROLL * (f_star * kelly_mult)
    final      = min(raw_stake, MAX_BET_DOLLARS)
    return int(round(final))


# ── Tracker server ────────────────────────────────────────────────────────────
TRACKER_PORT = int(os.getenv("TRACKER_SERVER_PORT", 5151))

# ── Claude model ──────────────────────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-20250514"
