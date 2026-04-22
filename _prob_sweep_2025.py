"""
Raw calibration by predicted-probability cutoff — 2025 full season.
Sources: out-of-fold validation prediction files generated during model training.

- ML:   ml_val_predictions.csv     | prob=stacker_ml        truth=home_win
- F5:   f5_val_predictions.csv     | prob=stacker_f5_cover  truth=f5_home_cover
- NR:   nrfi_val_predictions.csv   | prob=stacker_nrfi      truth=f1_nrfi
- RL:   eval_predictions.csv (2025)| prob=rl_stacked        truth=home_covers_rl

Note: Totals is omitted — the 2025 val file only stored a regression
`tot_pred` (predicted runs), not a calibrated OVER probability.
"""
import pandas as pd

def sweep(df, prob_col, truth_col, name, pos_label):
    df = df[[prob_col, truth_col]].dropna().copy()
    if df.empty:
        print(f"{name}: no rows"); return
    thresholds = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75, 0.80]
    print(f"=== {name} ({pos_label} — 2025 OOF val predictions) ===")
    print(f"{'when model_prob >=':>20}  {'games':>6}  {'hit':>5}  {'hit%':>6}  {'avg_pred%':>10}  {'gap':>6}")
    for t in thresholds:
        s = df[df[prob_col] >= t]
        n = len(s)
        if n == 0:
            continue
        hit = int((s[truth_col] == 1).sum())
        hit_pct = 100 * hit / n
        pred_pct = 100 * s[prob_col].mean()
        gap = hit_pct - pred_pct
        print(f"{t*100:>18.0f}%  {n:>6}  {hit:>5}  {hit_pct:>5.1f}%  {pred_pct:>9.1f}%  {gap:>+5.1f}")
    base = 100 * (df[truth_col] == 1).mean()
    print(f"  base rate over all {len(df)} games: {base:.1f}% hit")
    print()


ml  = pd.read_csv('ml_val_predictions.csv')
f5  = pd.read_csv('f5_val_predictions.csv')
nr  = pd.read_csv('nrfi_val_predictions.csv')
rl  = pd.read_csv('eval_predictions.csv')
rl  = rl[rl['year'] == 2025]

sweep(ml, 'stacker_ml',        'home_win',        "ML — Full-Game Moneyline",   "HOME wins")
sweep(f5, 'stacker_f5_cover',  'f5_home_cover',   "F5 Moneyline",                "HOME covers F5 RL")
sweep(rl, 'rl_stacked',        'home_covers_rl',  "Runline",                     "HOME covers -1.5")
sweep(nr, 'stacker_nrfi',      'f1_nrfi',         "NRFI",                        "NRFI (no run 1st)")
