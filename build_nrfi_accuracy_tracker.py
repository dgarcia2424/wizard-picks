"""
build_nrfi_accuracy_tracker.py

Daily accuracy tracker for NRFI model. Appends today's model_scores.csv to
a persistent archive, then evaluates against statcast actuals once games finish.

Run this after generate_ml_scores() (Agent 3) and before Agent 4/5.
Or run standalone to backfill/audit existing model_scores.csv.

Output:
  - model_scores_archive.csv — full prediction history, one row per pick per day
  - nrfi_accuracy_daily.csv — per-day summary (date, picks, hits, accuracy, etc.)

Data source: F1 NRFI derived from statcast for all years (2023-2026)
via inning==1 post-score max.
"""
import pandas as pd
from pathlib import Path
from datetime import date

ARCHIVE_PATH = Path("model_scores_archive.csv")
ACCURACY_PATH = Path("nrfi_accuracy_daily.csv")
SCORES_PATH = Path("model_scores.csv")


def append_to_archive():
    """Append today's model_scores.csv to the persistent archive."""
    if not SCORES_PATH.exists():
        print(f"[SKIP] {SCORES_PATH} not found.")
        return False
    
    today_scores = pd.read_csv(SCORES_PATH)
    if len(today_scores) == 0:
        print(f"[SKIP] {SCORES_PATH} is empty.")
        return False
    
    if ARCHIVE_PATH.exists():
        archive = pd.read_csv(ARCHIVE_PATH)
        # Avoid duplicates: drop any rows from today's date first
        if 'date' in archive.columns:
            today_str = today_scores['date'].iloc[0] if len(today_scores) > 0 else None
            if today_str:
                archive = archive[archive['date'] != today_str]
        archive = pd.concat([archive, today_scores], ignore_index=True)
    else:
        archive = today_scores.copy()
    
    archive.to_csv(ARCHIVE_PATH, index=False)
    print(f"[OK] Appended {len(today_scores)} rows to {ARCHIVE_PATH}")
    return True


def evaluate_accuracy():
    """Compute accuracy metrics by matching predictions with actuals."""
    if not ARCHIVE_PATH.exists():
        print(f"[SKIP] {ARCHIVE_PATH} not found. Run append_to_archive() first.")
        return

    archive = pd.read_csv(ARCHIVE_PATH)
    archive['date'] = pd.to_datetime(archive['date'])

    # Load F1 NRFI from statcast for all years (2023-2026)
    actuals_list = []
    for year in [2023, 2024, 2025, 2026]:
        statcast_path = Path(f"data/statcast/statcast_{year}.parquet")
        if not statcast_path.exists():
            continue

        sc = pd.read_parquet(statcast_path)
        # Group by game, filter inning==1, get max post-score
        f1_data = sc[sc['inning'] == 1].groupby('game_pk').agg({
            'game_date': 'first',
            'home_team': 'first',
            'away_team': 'first',
            'post_home_score': 'max',
            'post_away_score': 'max',
        }).reset_index()
        f1_data['f1_nrfi'] = ((f1_data['post_home_score'] == 0) & (f1_data['post_away_score'] == 0)).astype(int)
        actuals_list.append(f1_data[['game_date', 'home_team', 'away_team', 'f1_nrfi']])

    if not actuals_list:
        print(f"[SKIP] No statcast files found.")
        return

    actuals = pd.concat(actuals_list, ignore_index=True)
    actuals['game_date'] = pd.to_datetime(actuals['game_date'])
    
    # Parse game labels from archive
    def parse_game(game_label):
        parts = game_label.split(' @ ')
        if len(parts) == 2:
            return parts[1].strip(), parts[0].strip()
        return None, None
    
    archive['home_team_full'], archive['away_team_full'] = zip(*archive['game'].apply(parse_game))
    
    # Team abbreviation mapping (from tools/implementations.py)
    team_abbr_to_full = {
        'ARI': 'Arizona Diamondbacks', 'ATL': 'Atlanta Braves',
        'BAL': 'Baltimore Orioles', 'BOS': 'Boston Red Sox',
        'CHC': 'Chicago Cubs', 'CWS': 'Chicago White Sox',
        'CIN': 'Cincinnati Reds', 'CLE': 'Cleveland Guardians',
        'COL': 'Colorado Rockies', 'DET': 'Detroit Tigers',
        'HOU': 'Houston Astros', 'KC': 'Kansas City Royals',
        'LAA': 'Los Angeles Angels', 'LAD': 'Los Angeles Dodgers',
        'MIA': 'Miami Marlins', 'MIL': 'Milwaukee Brewers',
        'MIN': 'Minnesota Twins', 'NYM': 'New York Mets',
        'NYY': 'New York Yankees', 'OAK': 'Oakland Athletics',
        'PHI': 'Philadelphia Phillies', 'PIT': 'Pittsburgh Pirates',
        'SD': 'San Diego Padres', 'SF': 'San Francisco Giants',
        'SEA': 'Seattle Mariners', 'STL': 'St. Louis Cardinals',
        'TB': 'Tampa Bay Rays', 'TEX': 'Texas Rangers',
        'TOR': 'Toronto Blue Jays', 'WSH': 'Washington Nationals',
    }
    actuals['home_team_full'] = actuals['home_team'].map(team_abbr_to_full)
    actuals['away_team_full'] = actuals['away_team'].map(team_abbr_to_full)
    
    # Merge on date + teams
    merged = archive.merge(
        actuals[['game_date', 'home_team_full', 'away_team_full', 'f1_nrfi']],
        left_on=['date', 'home_team_full', 'away_team_full'],
        right_on=['game_date', 'home_team_full', 'away_team_full'],
        how='left'
    )
    
    # Evaluate correctness
    merged['predicted_nrfi'] = merged['bet_type'] == 'NRFI'
    merged['actual'] = merged['f1_nrfi'].astype('Int64')  # nullable int
    merged['correct'] = (
        ((merged['predicted_nrfi']) & (merged['actual'] == 1)) |
        ((~merged['predicted_nrfi']) & (merged['actual'] == 0))
    ).astype('Int64')
    
    # Per-day summary
    daily_summary = []
    for pred_date in merged['date'].dropna().unique():
        day_data = merged[merged['date'] == pred_date]
        day_with_result = day_data[day_data['actual'].notna()]
        
        if len(day_with_result) == 0:
            continue
        
        row = {
            'date': pred_date,
            'total_picks': len(day_data),
            'picks_with_result': len(day_with_result),
            'correct': day_with_result['correct'].sum(),
            'accuracy_pct': (day_with_result['correct'].mean() * 100) if len(day_with_result) > 0 else None,
            'actionable_picks': day_data['actionable'].sum(),
            'actionable_correct': day_with_result[day_with_result['actionable']]['correct'].sum(),
            'actionable_accuracy_pct': (day_with_result[day_with_result['actionable']]['correct'].mean() * 100) if len(day_with_result[day_with_result['actionable']]) > 0 else None,
        }
        daily_summary.append(row)
    
    if daily_summary:
        daily_df = pd.DataFrame(daily_summary)
        daily_df.to_csv(ACCURACY_PATH, index=False)
        print(f"[OK] Wrote {len(daily_df)} days of accuracy to {ACCURACY_PATH}")
        print()
        print(daily_df.to_string(index=False))
    else:
        print("[INFO] No completed games matched yet.")


if __name__ == "__main__":
    append_to_archive()
    evaluate_accuracy()
