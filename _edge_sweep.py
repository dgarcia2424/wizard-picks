import pandas as pd, numpy as np

df = pd.read_csv('backtest_full_all_predictions.csv')

def am_to_dec(o):
    if pd.isna(o) or o == 0:
        return np.nan
    return 1 + (o/100 if o > 0 else 100/abs(o))

def sweep(sub, name):
    sub = sub.copy()
    sub = sub[sub['retail_odds'].notna() & sub['actual'].notna() & (sub.get('odds_floor_pass') == True)]
    if sub.empty:
        print(f'{name}: no eligible rows'); return
    sub['dec'] = sub['retail_odds'].apply(am_to_dec)
    sub['pnl_unit'] = np.where(sub['actual'] == 1, sub['dec'] - 1, -1.0)
    thresholds = [-0.10, -0.05, 0.0, 0.005, 0.01, 0.015, 0.02, 0.025,
                  0.03, 0.035, 0.04, 0.05, 0.06, 0.08, 0.10]
    print(f'=== {name} (flat 1-unit bets; gate: retail_odds + actual + odds_floor) ===')
    print(f'{"edge>=":>7}  {"bets":>5}  {"wins":>5}  {"win%":>6}  {"ROI%":>7}  {"PnL(u)":>8}')
    best = None
    for t in thresholds:
        s = sub[sub['edge'] >= t]
        n = len(s)
        if n == 0:
            continue
        w = int((s['actual'] == 1).sum())
        wp = 100 * w / n
        pl = s['pnl_unit'].sum()
        roi = 100 * pl / n
        if n >= 10 and (best is None or roi > best[0]):
            best = (roi, t, n, wp)
        print(f'{t*100:>6.1f}%  {n:>5}  {w:>5}  {wp:>5.1f}%  {roi:>6.1f}%  {pl:>8.2f}')
    if best:
        r, t, n, wp = best
        print(f'  -> peak ROI (n>=10): edge >= {t*100:.1f}%  |  n={n}  |  win%={wp:.1f}  |  ROI%={r:.1f}')
    print()

for mkt, label in [('ML','ML (Full-Game Moneyline)'),
                   ('TOT','Totals'),
                   ('RL','Runline'),
                   ('F5','F5 Moneyline'),
                   ('NR','NRFI')]:
    sweep(df[df['market'] == mkt], label)
