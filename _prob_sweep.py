"""
Raw calibration by predicted-probability cutoff.
No edge / no odds / no gate. Just: when model_prob >= X, how often did it hit?
"""
import pandas as pd
import numpy as np

df = pd.read_csv('backtest_full_all_predictions.csv')

def sweep(sub, name, pos_label):
    sub = sub.copy()
    sub = sub[sub['actual'].notna() & sub['model_prob'].notna()]
    if sub.empty:
        print(f"{name}: no rows"); return
    thresholds = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75, 0.80]
    print(f"=== {name} ({pos_label} — 2026 sterile inference) ===")
    print(f"{'when model_prob >=':>20}  {'games':>6}  {'hit':>5}  {'hit%':>6}  {'avg_pred%':>10}  {'gap':>6}")
    for t in thresholds:
        s = sub[sub['model_prob'] >= t]
        n = len(s)
        if n == 0:
            continue
        hit = int((s['actual'] == 1).sum())
        hit_pct = 100 * hit / n
        pred_pct = 100 * s['model_prob'].mean()
        gap = hit_pct - pred_pct
        print(f"{t*100:>18.0f}%  {n:>6}  {hit:>5}  {hit_pct:>5.1f}%  {pred_pct:>9.1f}%  {gap:>+5.1f}")
    # Also show the full base rate for context
    base = 100 * (sub['actual'] == 1).mean()
    print(f"  base rate over all {len(sub)} games: {base:.1f}% hit")
    print()

for mkt, label, pos in [
    ("ML",  "ML — Full-Game Moneyline", "HOME wins"),
    ("TOT", "Totals",                    "OVER hits"),
    ("RL",  "Runline",                   "HOME covers -1.5"),
    ("F5",  "F5 Moneyline",              "HOME leads after 5"),
    ("NR",  "NRFI",                      "NRFI (no run in 1st)"),
]:
    sweep(df[df['market'] == mkt], label, pos)
