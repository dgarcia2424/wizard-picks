"""Two-tier alert system: ELITE (very high accuracy) and STRONG (moderately high)."""
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from datetime import date, timedelta

LAST_7 = [(date(2026, 4, 24) - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]

# Manually calibrated from full 2026 backtest accuracy analysis
# ELITE  = very high accuracy tier
# STRONG = moderately high accuracy tier (fires more often)
# None   = signal excluded from that tier
TIERS_FIXED = {
    #               ELITE    STRONG
    "ML":          (0.85,    0.70),   # 91.7% / 80.8%  (max-conf, home+away)
    "ML-edge":     (0.07,    0.02),   # 80.0% / 78.6%  (home edge vs closing line)
    "Runline":     (None,    0.85),   # no elite; 62.5%
    "F5":          (None,    0.65),   # no elite; 60-65% bucket now valid after ABS/TTO refit
    "NRFI":        (None,    0.80),   # no elite; 63.9%
    "K-over 3.5":  (0.80,    0.75),   # 66.1% / 65.6%
    "K-over 4.5":  (None,    0.75),   # no elite; 57.4%
}

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
bt = pd.read_csv(ROOT / "backtest_full_all_predictions.csv")
lv = pd.read_csv(ROOT / "live_predictions_2026.csv")
combined = pd.concat([
    bt[["market","model_prob","actual","game_date","home_team","away_team","edge"]],
    lv[["market","model_prob","actual","game_date"]].assign(
        home_team="", away_team="", edge=float("nan")),
]).drop_duplicates(subset=["market","model_prob","game_date"])
combined["game_date"]  = pd.to_datetime(combined["game_date"]).dt.strftime("%Y-%m-%d")
combined["model_prob"] = pd.to_numeric(combined["model_prob"], errors="coerce")
combined["actual"]     = pd.to_numeric(combined["actual"],     errors="coerce")
combined["edge"]       = pd.to_numeric(combined["edge"],       errors="coerce")

cards = pd.concat([
    pd.read_csv(f, low_memory=False).assign(
        game_date=Path(f).stem.replace("daily_card_", ""))
    for f in sorted(glob.glob(str(ROOT / "daily_cards" / "daily_card_2026-*.csv")))
], ignore_index=True)
act = pd.read_parquet(ROOT / "data" / "statcast" / "actuals_2026.parquet")
act["game_date"] = pd.to_datetime(act["game_date"]).dt.strftime("%Y-%m-%d")
merged = cards.merge(act[["home_team","game_date","home_sp_k","away_sp_k"]],
                     on=["home_team","game_date"], how="left")

# ---------------------------------------------------------------------------
# Build signal dataframes
# ---------------------------------------------------------------------------
def _ml(df):
    r = df[df["market"]=="ML"].copy()
    r["conf"] = r["model_prob"].where(r["model_prob"] >= 0.5, 1 - r["model_prob"])
    r["hit"]  = r.apply(lambda x: x["actual"] if x["model_prob"] >= 0.5 else 1 - x["actual"], axis=1)
    r["dir"]  = r["model_prob"].apply(lambda p: "HOME" if p >= 0.5 else "AWAY")
    return r[["game_date","conf","hit","dir","home_team","away_team"]]

def _ml_edge(df):
    """Home edge vs closing line — positive edge only (model favors home vs market)."""
    r = df[(df["market"]=="ML") & df["edge"].notna() & (df["edge"] > 0)].copy()
    r["conf"] = r["edge"]
    r["hit"]  = r["actual"]   # actual=1 → home wins → edge bet wins
    r["dir"]  = "HOME"
    return r[["game_date","conf","hit","dir","home_team","away_team"]]

def _mkt(df, code, label):
    r = df[df["market"] == code].copy()
    r["conf"] = r["model_prob"]
    r["hit"]  = r["actual"]
    r["dir"]  = label
    return r[["game_date","conf","hit","dir","home_team","away_team"]]

def _kover(merged, line=3.5):
    col = str(line).replace(".", "")
    rows = []
    for _, r in merged.iterrows():
        for side in ("home", "away"):
            ak = pd.to_numeric(r.get(f"{side}_sp_k"), errors="coerce")
            p  = pd.to_numeric(r.get(f"mc_{side}_sp_k_over_{col}"), errors="coerce")
            sp = r.get(f"{side}_sp", "?")
            if pd.isna(ak) or pd.isna(p):
                continue
            conf = p if p >= 0.5 else 1 - p
            hit  = float(ak > line) if p >= 0.5 else float(ak <= line)
            dire = f"{sp} O{line}" if p >= 0.5 else f"{sp} U{line}"
            rows.append({"game_date": r["game_date"], "conf": conf, "hit": hit,
                         "dir": dire, "home_team": r["home_team"], "away_team": r["away_team"]})
    return pd.DataFrame(rows)

sig_dfs = {
    "ML":         _ml(combined),
    "ML-edge":    _ml_edge(combined),
    "Runline":    _mkt(combined, "RL", "HOME -1.5"),
    "F5":         _mkt(combined, "F5", "HOME F5"),
    "NRFI":       _mkt(combined, "NR", "NRFI"),
    "K-over 3.5": _kover(merged, 3.5),
    "K-over 4.5": _kover(merged, 4.5),
}

# ---------------------------------------------------------------------------
# Build tier dicts from fixed cutoffs + compute accuracy from full history
# ---------------------------------------------------------------------------
def _tier_stats(df, cut):
    if cut is None:
        return None
    prob = pd.to_numeric(df["conf"], errors="coerce")
    hit  = pd.to_numeric(df["hit"],  errors="coerce")
    mask = prob >= cut
    n    = int(mask.sum())
    acc  = float(hit[mask].mean()) if n else 0.0
    return {"cut": cut, "n": n, "acc": acc}

TIERS = {}
print("\nTWO-TIER CUTOFFS (full 2026 history)\n" + "=" * 55)
for sig, (cut_e, cut_s) in TIERS_FIXED.items():
    df     = sig_dfs[sig]
    elite  = _tier_stats(df, cut_e)
    strong = _tier_stats(df, cut_s)
    TIERS[sig] = {"elite": elite, "strong": strong}
    unit = "edge" if sig == "ML-edge" else "prob"
    print(f"\n{sig}:")
    if elite:
        print(f"  ELITE  {unit}>={elite['cut']:.2f}  ->  {elite['acc']:.1%} acc  ({elite['n']} games)")
    else:
        print(f"  ELITE   — excluded")
    if strong:
        print(f"  STRONG {unit}>={strong['cut']:.2f}  ->  {strong['acc']:.1%} acc  ({strong['n']} games)")
    else:
        print(f"  STRONG  — excluded")

# ---------------------------------------------------------------------------
# Last 7 days
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 70)
print("LAST 7 DAYS — ALERT FIRES BY DAY")
print("=" * 70)

totals = {"elite": {"n":0,"w":0,"l":0}, "strong": {"n":0,"w":0,"l":0}}

for day in LAST_7:
    elite_fires, strong_fires = [], []

    for sig, df in sig_dfs.items():
        d = df[df["game_date"] == day].copy()
        if d.empty:
            continue
        prob = pd.to_numeric(d["conf"], errors="coerce").reset_index(drop=True)
        hit  = pd.to_numeric(d["hit"],  errors="coerce").reset_index(drop=True)
        d = d.reset_index(drop=True)

        t_e = TIERS[sig]["elite"]
        t_s = TIERS[sig]["strong"]
        cut_e = t_e["cut"] if t_e else 999

        if t_e:
            for i in (prob >= t_e["cut"]).index[prob >= t_e["cut"]]:
                res = "WIN" if hit[i] == 1 else ("LOSS" if hit[i] == 0 else "?")
                elite_fires.append({"sig": sig, "dir": d.loc[i,"dir"], "prob": prob[i], "res": res})

        if t_s:
            mask = (prob >= t_s["cut"]) & (prob < cut_e)
            for i in mask.index[mask]:
                res = "WIN" if hit[i] == 1 else ("LOSS" if hit[i] == 0 else "?")
                strong_fires.append({"sig": sig, "dir": d.loc[i,"dir"], "prob": prob[i], "res": res})

    def _summary(fires, label):
        w = sum(1 for f in fires if f["res"]=="WIN")
        l = sum(1 for f in fires if f["res"]=="LOSS")
        print(f"  {label} ({len(fires):2d} alerts)  {w}W {l}L", end="")
        if fires:
            by_sig = {}
            for f in fires:
                by_sig.setdefault(f["sig"], []).append(f)
            parts = []
            for s, v in by_sig.items():
                sw = sum(1 for x in v if x["res"]=="WIN")
                sl = sum(1 for x in v if x["res"]=="LOSS")
                parts.append(f"{s}:{sw}W{sl}L")
            print(f"  [{' | '.join(parts)}]")
        else:
            print("  [none]")
        return w, l

    print(f"\n{day}")
    ew, el = _summary(elite_fires,  "ELITE ")
    sw, sl = _summary(strong_fires, "STRONG")
    totals["elite"]["n"]  += len(elite_fires)
    totals["elite"]["w"]  += ew
    totals["elite"]["l"]  += el
    totals["strong"]["n"] += len(strong_fires)
    totals["strong"]["w"] += sw
    totals["strong"]["l"] += sl

e = totals["elite"]
s = totals["strong"]
print(f"\n{'='*70}")
print(f"7-DAY TOTALS")
print(f"  ELITE  {e['n']} alerts  {e['w']}W {e['l']}L  "
      f"({e['w']/(e['w']+e['l']):.1%})" if e['w']+e['l'] else f"  ELITE  {e['n']} alerts")
print(f"  STRONG {s['n']} alerts  {s['w']}W {s['l']}L  "
      f"({s['w']/(s['w']+s['l']):.1%})" if s['w']+s['l'] else f"  STRONG {s['n']} alerts")

# ---------------------------------------------------------------------------
# TODAY'S SGP PICKS (no historical accuracy — informational only)
# ---------------------------------------------------------------------------
sgp_path = ROOT / "data" / "sgp" / "sgp_alpha_report.csv"
if sgp_path.exists():
    sgp = pd.read_csv(sgp_path)
    sgp["sgp_edge"] = pd.to_numeric(sgp["sgp_edge"], errors="coerce")
    today_str = date(2026, 4, 24).strftime("%Y-%m-%d")
    print(f"\n\n{'='*70}")
    print(f"TODAY'S SGP PICKS ({today_str})  [no historical baseline — informational]")
    print(f"{'='*70}")
    top = sgp[sgp["sgp_edge"] >= 2.0].sort_values("sgp_edge", ascending=False)
    if top.empty:
        print("  No SGP picks with edge >= 2.0x today.")
    else:
        for _, row in top.iterrows():
            legs_str = str(row.get("legs",""))[:80]
            print(f"  {row['script']:<20}  edge={row['sgp_edge']:.2f}x  "
                  f"p_model={row.get('p_joint_model',float('nan')):.3f}  "
                  f"p_mkt={row.get('p_joint_market',float('nan')):.3f}")
            print(f"    {legs_str}")
