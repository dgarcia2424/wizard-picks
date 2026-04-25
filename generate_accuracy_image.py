"""
generate_accuracy_image.py
Dark-themed accuracy table: Optimal / Less Optimal / Stay Away x 2025 OOF / 2026 YTD / L30d
Output: accuracy_table.png
"""
from __future__ import annotations
import glob, json, re, warnings
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

_ROOT      = Path(__file__).resolve().parent
BANDS      = json.loads((_ROOT / 'signal_bands.json').read_text())
TODAY      = date.today()
L30_CUTOFF = (TODAY - timedelta(days=30)).strftime('%Y-%m-%d')
JUICE      = 100 / (100 + 110)

BG      = '#0d1221'
GREEN   = '#2ecc71'
YELLOW  = '#f0b429'
RED     = '#e05252'
WHITE   = '#dde8f0'
LGRAY   = '#7a8fa0'
DGRAY   = '#2a3a4a'
DIVIDER = '#1e2e3e'

SIGNALS = [
    ('rl_prob',     'RL',        '#4a9eff'),
    ('ml_win_prob', 'ML',        '#5590ff'),
    ('ou_prob',     'O/U',       '#9b59b6'),
    ('k_over_prob', 'K Over',    '#e67e22'),
    ('nrfi_prob',   'NRFI',      '#27ae60'),
    ('f5_win_prob', 'F5',        '#1abc9c'),
    ('script_a2',   'Script A2', '#e74c3c'),
    ('script_b',    'Script B',  '#c0392b'),
    ('script_c',    'Script C',  '#e74c3c'),
    ('script_d',    'Script D',  '#c0392b'),
]

FLOORS = {
    'rl_prob': 0.40, 'ml_win_prob': 0.50, 'ou_prob': 0.50,
    'k_over_prob': 0.50, 'nrfi_prob': 0.00, 'f5_win_prob': 0.50,
    'script_a2': 0.10, 'script_b': 0.10, 'script_c': 0.10, 'script_d': 0.10,
}

TIER_DEFS = [
    ('*', 'Optimal',      GREEN),
    ('o', 'Less optimal', YELLOW),
    ('x', 'Stay away',    RED),
]


def fmt_thresh(key, t):
    if 'script' in key:
        return f'corr>={t:.2f}'
    elif key == 'nrfi_prob':
        return f'edge>={t:.2f}'
    else:
        return f'p>={t:.0%}'


# ── Data loading ──────────────────────────────────────────────────────────────

def load_2026():
    cards = []
    for f in sorted(glob.glob(str(_ROOT / 'daily_cards/daily_card_2026-*.csv'))):
        d = pd.read_csv(f, low_memory=False)
        d['_date'] = Path(f).stem.replace('daily_card_', '')
        cards.append(d)
    if not cards:
        return pd.DataFrame()
    df = pd.concat(cards, ignore_index=True)
    act = pd.read_parquet(_ROOT / 'data/statcast/actuals_2026.parquet')
    act['_date'] = pd.to_datetime(act['game_date']).dt.strftime('%Y-%m-%d')
    keep = [c for c in ['_date', 'home_team', 'home_sp_k', 'away_sp_k', 'f1_nrfi',
                         'f5_home_win', 'f5_total', 'home_score_final',
                         'away_score_final', 'home_covers_rl'] if c in act.columns]
    return df.merge(act[keep], on=['_date', 'home_team'], how='left')


def load_sgp():
    frames = []
    for pattern in ['sgp_live_edge_2026_*.csv', 'sgp_live_edge_2026-*.csv']:
        for f in sorted(glob.glob(str(_ROOT / 'data/sgp' / pattern))):
            if 'steam' in Path(f).stem:
                continue
            stem = Path(f).stem.replace('-', '_').split('_')
            yi = next((i for i, p in enumerate(stem) if p == '2026'), None)
            if yi is None or yi + 2 >= len(stem):
                continue
            gdate = f'2026-{stem[yi+1]:0>2}-{stem[yi+2]:0>2}'
            try:
                d = pd.read_csv(f)
                d['game_date'] = gdate
                frames.append(d)
            except Exception:
                pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Row builders ──────────────────────────────────────────────────────────────

def _parse(legs, pat):
    m = re.search(pat, str(legs))
    return float(m.group(1)) if m else None


def rl_rows(df):
    r = df.dropna(subset=['blended_rl', 'home_covers_rl']).copy()
    r['prob']    = pd.to_numeric(r['blended_rl'], errors='coerce')
    r['outcome'] = r['home_covers_rl'].astype(float)
    return r.dropna(subset=['prob', 'outcome'])[['_date', 'prob', 'outcome']], 'prob'


def ml_rows(df):
    r = df.dropna(subset=['mc_home_win', 'home_score_final', 'away_score_final']).copy()
    r['prob'] = pd.to_numeric(r['mc_home_win'], errors='coerce')
    r['outcome'] = (r['home_score_final'].astype(float) > r['away_score_final'].astype(float)).astype(float)
    return r.dropna(subset=['prob', 'outcome'])[['_date', 'prob', 'outcome']], 'prob'


def ou_rows(df):
    r = df.dropna(subset=['ou_p_model', 'ou_posted_line',
                           'home_score_final', 'away_score_final']).copy()
    r['prob'] = pd.to_numeric(r['ou_p_model'], errors='coerce')
    gt = r['home_score_final'].astype(float) + r['away_score_final'].astype(float)
    posted = pd.to_numeric(r['ou_posted_line'], errors='coerce')
    dirn = r.get('ou_direction', pd.Series('OVER', index=r.index))
    r['outcome'] = np.where(dirn == 'OVER', (gt > posted).astype(float),
                            (gt < posted).astype(float))
    r.loc[posted.isna(), 'outcome'] = np.nan
    return r.dropna(subset=['prob', 'outcome'])[['_date', 'prob', 'outcome']], 'prob'


def k_rows(df):
    rows_list = []
    for _, row in df.iterrows():
        for side in ('home', 'away'):
            prob = pd.to_numeric(row.get(f'{side}_k_model_over'), errors='coerce')
            line = pd.to_numeric(row.get(f'{side}_k_line'),       errors='coerce')
            ak   = pd.to_numeric(row.get(f'{side}_sp_k'),         errors='coerce')
            if pd.isna(prob) or pd.isna(line) or pd.isna(ak):
                continue
            rows_list.append({'_date': row.get('_date', ''), 'prob': prob,
                               'outcome': float(ak > line)})
    if not rows_list:
        return pd.DataFrame(columns=['_date', 'prob', 'outcome']), 'prob'
    return pd.DataFrame(rows_list), 'prob'


def nrfi_rows(df):
    r = df.dropna(subset=['mc_nrfi_prob', 'f1_nrfi']).copy()
    r['prob'] = pd.to_numeric(r['mc_nrfi_prob'], errors='coerce')
    r['f1_nrfi'] = pd.to_numeric(r['f1_nrfi'], errors='coerce')
    r = r.dropna(subset=['prob', 'f1_nrfi'])
    r['edge'] = r['prob'].apply(lambda p: (p - JUICE) if p >= 0.5 else ((1 - p) - JUICE))
    r['outcome'] = r.apply(lambda x: x['f1_nrfi'] if x['prob'] >= 0.5 else 1 - x['f1_nrfi'], axis=1)
    return r[['_date', 'edge', 'outcome']], 'edge'


def f5_rows(df):
    col = 'f5_stacker_l2' if 'f5_stacker_l2' in df.columns else 'mc_f5_home_win_prob'
    if col not in df.columns:
        return pd.DataFrame(columns=['_date', 'prob', 'outcome']), 'prob'
    r = df.dropna(subset=[col, 'f5_home_win']).copy()
    r['prob'] = pd.to_numeric(r[col], errors='coerce')
    r['outcome'] = pd.to_numeric(r['f5_home_win'], errors='coerce')
    return r.dropna(subset=['prob', 'outcome'])[['_date', 'prob', 'outcome']], 'prob'


def script_rows(sgp, act, script_name):
    empty = pd.DataFrame(columns=['_date', 'sgp_edge', 'outcome'])
    if sgp.empty:
        return empty, 'sgp_edge'
    sub = sgp[sgp['script'] == script_name].copy() if 'script' in sgp.columns else pd.DataFrame()
    if sub.empty:
        return empty, 'sgp_edge'
    sub = sub.merge(act[['game_date', 'home_team', 'home_sp_k', 'away_sp_k',
                          'f5_total', 'home_score_final', 'away_score_final', 'home_covers_rl']],
                    on=['game_date', 'home_team'], how='left')
    sub = sub.dropna(subset=['home_score_final', 'away_score_final', 'f5_total',
                              'home_sp_k', 'away_sp_k', 'home_covers_rl'])
    if sub.empty:
        return empty, 'sgp_edge'
    gt = sub['home_score_final'].astype(float) + sub['away_score_final'].astype(float)

    def hit(row, game_total):
        legs = row['legs']
        if script_name == 'A2_Dominance':
            k  = _parse(legs, r'SP_K_F5>=(\d+)') or 4
            f5 = _parse(legs, r'F5_Under_([\d.]+)') or 99
            g  = _parse(legs, r'Game_Under_([\d.]+)') or 99
            return float(row['home_sp_k'] >= k and row['f5_total'] < f5 and game_total < g)
        elif script_name == 'B_Explosion':
            g = _parse(legs, r'Game_Over_([\d.]+)') or 0
            return float(row['home_score_final'] >= 5 and game_total > g
                         and row['home_score_final'] > row['away_score_final'])
        elif script_name == 'C_EliteDuel':
            g = _parse(legs, r'Game_Under_([\d.]+)') or 99
            k = _parse(legs, r'SP_K_F5>=(\d+)') or 3
            return float(game_total < g and row['home_sp_k'] >= k
                         and row['away_sp_k'] >= k and row['home_covers_rl'] == 0)
        elif script_name == 'D_LateDivergence':
            f5 = _parse(legs, r'F5_Under_([\d.]+)') or 99
            g  = _parse(legs, r'Game_Over_([\d.]+)') or 0
            return float(row['f5_total'] < f5 and game_total > g)
        return np.nan

    sub['outcome']  = [hit(row, gt.iloc[i]) for i, (_, row) in enumerate(sub.iterrows())]
    sub['sgp_edge'] = pd.to_numeric(sub['sgp_edge'], errors='coerce')
    return sub.rename(columns={'game_date': '_date'})[['_date', 'sgp_edge', 'outcome']], 'sgp_edge'


def get_rows_2026(key, df26, sgp, act):
    fn = {
        'rl_prob':     lambda: rl_rows(df26),
        'ml_win_prob': lambda: ml_rows(df26),
        'ou_prob':     lambda: ou_rows(df26),
        'k_over_prob': lambda: k_rows(df26),
        'nrfi_prob':   lambda: nrfi_rows(df26),
        'f5_win_prob': lambda: f5_rows(df26),
        'script_a2':   lambda: script_rows(sgp, act, 'A2_Dominance'),
        'script_b':    lambda: script_rows(sgp, act, 'B_Explosion'),
        'script_c':    lambda: script_rows(sgp, act, 'C_EliteDuel'),
        'script_d':    lambda: script_rows(sgp, act, 'D_LateDivergence'),
    }
    return fn[key]()


def get_rows_2025(key):
    ep_path = _ROOT / 'eval_predictions.csv'
    f5_path = _ROOT / 'f5_val_predictions.csv'
    # rl_stacked is raw probability (0.11-0.62), not an edge — not comparable to 2026 edge signal
    if key == 'ml_win_prob' and ep_path.exists():
        ep = pd.read_csv(ep_path)
        ep = ep[ep['_eval_mode'] == 'current'].copy()
        ep['prob'] = pd.to_numeric(ep['ml_raw'], errors='coerce')
        ep['outcome'] = pd.to_numeric(ep['actual_home_win'], errors='coerce')
        return ep[['prob', 'outcome']].dropna(), 'prob'
    if key == 'f5_win_prob' and f5_path.exists():
        f5 = pd.read_csv(f5_path)
        col = 'stacker_f5_cover' if 'stacker_f5_cover' in f5.columns else 'xgb_raw_f5_cover'
        out = 'f5_home_cover' if 'f5_home_cover' in f5.columns else 'f5_home_win'
        if col in f5.columns and out in f5.columns:
            f5 = f5.rename(columns={col: 'prob', out: 'outcome'})
            return f5[['prob', 'outcome']].dropna(), 'prob'
    return None, None


# ── Tier stats ────────────────────────────────────────────────────────────────

def tier_stats(rows, sc, oc, threshold):
    if rows is None or rows.empty:
        return None
    sub = rows.copy()
    sub[sc] = pd.to_numeric(sub[sc], errors='coerce')
    sub[oc] = pd.to_numeric(sub[oc], errors='coerce')
    sub = sub.dropna(subset=[sc, oc])
    sub = sub[sub[sc] >= threshold]
    if len(sub) == 0:
        return None
    return round(sub[oc].mean(), 4), int(round(sub[oc].sum())), len(sub)


# ── Rendering ─────────────────────────────────────────────────────────────────

def render(table_data):
    N = len(table_data)
    ROW_H   = 1.30
    HDR_H   = 1.00
    FOOT_H  = 0.65
    FIG_H   = HDR_H + N * ROW_H + FOOT_H
    FIG_W   = 22.0

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_facecolor(BG)
    ax.axis('off')

    # Column x anchors
    X_MKT   = 0.35   # market icon+label
    X_OOF   = 3.10   # 2025 OOF column
    X_STD   = 6.50   # 2026 YTD column
    X_L30   = 13.25  # Last 30d column

    # Within the STD/L30 cols, relative offsets
    dICON   = 0.00
    dLABEL  = 0.38
    dTHRESH = 2.20
    dARROW  = 3.55
    dACC    = 3.88
    dFRAC   = 5.10

    # Header row
    HDR_Y = FIG_H - 0.50
    hkw = dict(fontsize=11, color=LGRAY, fontweight='bold',
                fontfamily='monospace', va='center')
    ax.text(X_MKT,  HDR_Y, 'MARKET',       **hkw)
    ax.text(X_OOF,  HDR_Y, '2025 OOF',     **hkw)
    ax.text(X_STD,  HDR_Y, '2026 YTD',     **hkw)
    ax.text(X_L30,  HDR_Y, 'LAST 30 DAYS', **hkw)

    # Column dividers (vertical lines)
    for x in [X_OOF - 0.15, X_STD - 0.15, X_L30 - 0.15]:
        ax.axvline(x, color=DIVIDER, linewidth=1.0, ymin=FOOT_H / FIG_H, ymax=1.0)

    # Header underline
    ax.axhline(FIG_H - 0.72, color=DIVIDER, linewidth=1.2, xmin=0.01, xmax=0.99)

    for i, sig in enumerate(table_data):
        row_top    = FIG_H - HDR_H - i * ROW_H
        row_center = row_top - ROW_H / 2
        row_bot    = row_top - ROW_H

        # Alternating row shading
        if i % 2 == 1:
            rect = plt.Rectangle((0.05, row_bot + 0.04), FIG_W - 0.1, ROW_H - 0.04,
                                   facecolor='#111927', edgecolor='none', zorder=0)
            ax.add_patch(rect)

        # Signal badge (colored circle + abbreviation)
        cx, cy = X_MKT + 0.28, row_center
        circ = plt.Circle((cx, cy), 0.28, color=sig['color'], zorder=2)
        ax.add_patch(circ)
        abbr = sig['label'][:2].upper()
        ax.text(cx, cy, abbr, color='white', fontsize=8, fontweight='bold',
                ha='center', va='center', zorder=3, fontfamily='monospace')
        ax.text(X_MKT + 0.74, row_center, sig['label'],
                color=WHITE, fontsize=12, fontweight='bold', va='center')

        # 2025 OOF — single optimal-tier line
        oof   = sig['oof_2025']
        oof_t = sig['thresholds'][0]   # tier1
        if oof:
            acc, hits, n = oof
            ax.text(X_OOF, row_center + 0.26,
                    fmt_thresh(sig['key'], oof_t),
                    color=LGRAY, fontsize=9.5, va='center', fontfamily='monospace')
            ax.text(X_OOF, row_center - 0.05,
                    f'{acc:.1%}',
                    color=GREEN, fontsize=14, fontweight='bold',
                    va='center', fontfamily='monospace')
            ax.text(X_OOF + 1.15, row_center - 0.05,
                    f'({hits}/{n})',
                    color=LGRAY, fontsize=9.5, va='center', fontfamily='monospace')
        else:
            ax.text(X_OOF, row_center, 'n/a',
                    color=DGRAY, fontsize=11, va='center', fontfamily='monospace')

        # 2026 STD and Last 30d — three tier lines each
        for col_x, tier_list in [(X_STD, sig['tiers_2026']), (X_L30, sig['tiers_l30'])]:
            for j, (icon_ch, tier_lbl, t_color) in enumerate(TIER_DEFS):
                sub_y   = row_center + 0.36 - j * 0.36
                result  = tier_list[j]
                t_str   = fmt_thresh(sig['key'], sig['thresholds'][j])

                ax.text(col_x + dICON,   sub_y, icon_ch,   color=t_color,
                        fontsize=11, va='center', fontfamily='monospace', fontweight='bold')
                ax.text(col_x + dLABEL,  sub_y, tier_lbl,  color=t_color,
                        fontsize=10, va='center', fontweight='bold')
                ax.text(col_x + dTHRESH, sub_y, t_str,     color=LGRAY,
                        fontsize=9.5, va='center', fontfamily='monospace')

                if result:
                    acc, hits, n = result
                    ax.text(col_x + dARROW, sub_y, '->',
                            color=LGRAY, fontsize=10, va='center', fontfamily='monospace')
                    ax.text(col_x + dACC,   sub_y, f'{acc:.1%}',
                            color=t_color, fontsize=10.5, fontweight='bold',
                            va='center', fontfamily='monospace')
                    ax.text(col_x + dFRAC,  sub_y, f'({hits}/{n})',
                            color=LGRAY, fontsize=9.5, va='center', fontfamily='monospace')
                else:
                    ax.text(col_x + dARROW, sub_y, 'n/a',
                            color=DGRAY, fontsize=9.5, va='center', fontfamily='monospace')

        # Row divider
        if i < N - 1:
            ax.axhline(row_bot + 0.03, color=DIVIDER, linewidth=0.6,
                       xmin=0.01, xmax=0.99)

    # Footer
    ax.text(X_MKT, FOOT_H / 2,
            f'2025 OOF = full-season backtest (RL/ML/F5 only)  |  '
            f'2026 YTD = live picks  |  Last 30d = rolling window  |  '
            f'Tiers from signal_bands.json  |  {TODAY}',
            color=LGRAY, fontsize=7.5, va='center', fontfamily='monospace')

    out = _ROOT / 'accuracy_table.png'
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close()
    print(f'  Saved -> {out}')
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('Loading data...')
    df26 = load_2026()
    sgp  = load_sgp()
    act  = pd.read_parquet(_ROOT / 'data/statcast/actuals_2026.parquet')
    act['game_date'] = pd.to_datetime(act['game_date']).dt.strftime('%Y-%m-%d')

    table_data = []
    for key, label, color in SIGNALS:
        b      = BANDS.get(key, {})
        tier1  = b.get('tier1', 0.0)
        tier2  = b.get('tier2', 0.0)
        floor  = FLOORS[key]

        rows26, sc = get_rows_2026(key, df26, sgp, act)
        oc = 'outcome'

        r_l30 = rows26[rows26['_date'] >= L30_CUTOFF].copy() \
                if '_date' in rows26.columns and not rows26.empty else rows26.copy()

        def tiers(rows):
            return [
                tier_stats(rows, sc, oc, tier1),
                tier_stats(rows, sc, oc, tier2),
                tier_stats(rows, sc, oc, floor),
            ]

        rows25, sc25 = get_rows_2025(key)
        oof_2025 = tier_stats(rows25, sc25, 'outcome', tier1) if rows25 is not None else None

        table_data.append({
            'key':        key,
            'label':      label,
            'color':      color,
            'oof_2025':   oof_2025,
            'tiers_2026': tiers(rows26),
            'tiers_l30':  tiers(r_l30),
            'thresholds': [tier1, tier2, floor],
        })
        print(f'  [{key}] done')

    print('Rendering image...')
    render(table_data)


if __name__ == '__main__':
    main()
