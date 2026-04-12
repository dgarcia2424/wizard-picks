MLB MODEL SCORING PIPELINE — README
=====================================
Models: MF1 (F1 Total) | MG3 (F3 Total) | MF5 (F5 ML) | M2 (Environmental)

SETUP
-----
1. Install Python 3.9+
2. Run: pip install pandas numpy
3. Place this folder anywhere on your machine

STEP 1 — DOWNLOAD BASEBALL SAVANT DATA
---------------------------------------
URL: https://baseballsavant.mlb.com/leaderboard/custom?year=2026&type=pitcher
- Set: Year=2026, Type=Pitcher, Min PA=1
- Click "Download CSV" (top right)
- Save as: savant_pitchers.csv (same folder as score_models.py)

Key columns used: last_name, first_name, k_percent, bb_percent,
                  exit_velocity_avg, xera, stuff_plus (if available)

STEP 2 — DOWNLOAD FANGRAPHS DATA
----------------------------------
URL: https://fangraphs.com/leaders/major-league?pos=all&stats=pit&lg=all&qual=1&type=8&season=2026
- Click "Export Data" button (below the table)
- Save as: fangraphs_pitchers.csv (same folder)

Key columns used: Name, K/9, BB/9, SIERA, FIP, K-BB%

STEP 3 — CREATE YOUR GAMES FILE
---------------------------------
Copy games_TEMPLATE.csv → games.csv
Fill in one row per game:

  game_date         : YYYY-MM-DD format
  home_team         : Full team name (display only)
  away_team         : Full team name (display only)
  home_team_abbr    : 2-3 letter abbr (NYY, LAD, BOS, etc.)
  away_team_abbr    : 2-3 letter abbr
  home_sp           : Starting pitcher FIRST LAST (must match CSV exactly)
  away_sp           : Starting pitcher FIRST LAST
  temp_f            : Game-time temperature in Fahrenheit
  wind_mph          : Wind speed in MPH
  wind_dir          : 'in' / 'out' / 'cross' (relative to home plate)
  market_total_f1   : DraftKings F1 total line (e.g. 0.75)
  market_total_f3   : DraftKings F3 total line (e.g. 2.25)
  market_total_f5   : DraftKings F5 total line (e.g. 4.5)
  market_total_game : Full game total line (e.g. 8.5)

BACKTEST COLUMNS (fill in after games complete):
  actual_f1_total   : Actual runs scored in inning 1
  actual_f3_total   : Actual runs scored through 3 innings
  actual_f5_total   : Actual runs scored through 5 innings
  actual_game_total : Final combined run total
  actual_home_win   : HOME or AWAY (who won F5)

Leave blank if unknown / game not yet played.

STEP 4 — RUN THE PIPELINE
---------------------------
  cd /path/to/mlb_model_pipeline
  python score_models.py

OUTPUTS
--------
  model_scores.csv   — Full scored results, one row per game
  model_report.html  — Visual HTML report (open in browser)

PITCHER NAME MATCHING
----------------------
Pitcher names must match EXACTLY between games.csv and the CSV downloads.
Use FIRST LAST format (e.g. "Gerrit Cole", "Roki Sasaki").
The script uppercases all names — capitalization doesn't matter.

If a pitcher is unmatched, the script will report it and skip the game.
Common issues:
  - Accents stripped in one source (e.g. "Framber Valdez" vs "Framber Valdéz")
  - Nickname vs full name
  - Jr./Sr. suffixes

Fix by editing the name in games.csv to match the download exactly.

WEATHER (OPTIONAL)
-------------------
Leave temp_f, wind_mph, wind_dir blank if unavailable.
Model will run without weather adjustments.
Best source: weather.gov or weatherapi.com (free API, 1M calls/month)

UPDATING PARK FACTORS
----------------------
Open score_models.py → find PARK_FACTORS dict near top.
Update from: https://baseballsavant.mlb.com/leaderboard/statcast-park-factors
(Download CSV → use "index" column per team, normalized to 1.00)

QUESTIONS / ISSUES
-------------------
Check unmatched pitcher names first — that's the most common failure mode.
