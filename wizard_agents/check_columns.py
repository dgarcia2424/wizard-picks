import pandas as pd

BASE = r"C:\Users\garci\OneDrive\Documents\Claude\mlb_model_pipeline_dan"

files = [
    "savant_pitchers.csv",
    "savant_batters.csv",
    "fangraphs_pitchers.csv",
    "fangraphs_batters.csv",
    "fangraphs_team_vs_lhp.csv",
    "fangraphs_team_vs_rhp.csv",
]

for f in files:
    try:
        df = pd.read_csv(f"{BASE}\\{f}", nrows=2)
        size = __import__('os').path.getsize(f"{BASE}\\{f}") // 1024
        print(f"\n{f} ({size}KB, {len(df)} sample rows):")
        print(list(df.columns))
    except Exception as e:
        print(f"\n{f}: ERROR — {e}")
