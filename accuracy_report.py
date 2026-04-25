"""
accuracy_report.py — Signal accuracy across three windows: 2025, 2026 YTD, Last 30 days.
Shows Green-tier accuracy per window using thresholds from signal_bands.json.
Usage: python accuracy_report.py
"""
from __future__ import annotations
import glob, json, re
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT  = Path(__file__).resolve().parent
BANDS  = json.loads((_ROOT / "signal_bands.json").read_text())
JUICE  = 100 / (100 + 110)   # -110 implied = 52.38%
TODAY  = date.today()
L30_CUTOFF = (TODAY - timedelta(days=30)).strftime("%Y-%m-%d")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_merged_2026() -> pd.DataFrame:
    """All 2026 daily cards merged with actuals."""
    cards = []
    for f in sorted(glob.glob(str(_ROOT / "daily_cards/daily_card_2026-*.csv"))):
        d = pd.read_csv(f, low_memory=False)
        d["_date"] = Path(f).stem.replace("daily_card_", "")
        cards.append(d)
    if not cards:
        return pd.DataFrame()
    df = pd.concat(cards, ignore_index=True)

    act = pd.read_parquet(_ROOT / "data/statcast/actuals_2026.parquet")
    act["_date"] = pd.to_datetime(act["game_date"]).dt.strftime("%Y-%m-%d")
    keep = ["_date", "home_team", "home_sp_k", "away_sp_k",
            "home_sp_ip", "away_sp_ip", "f1_nrfi", "f5_home_win",
            "f5_total", "home_score_final", "away_score_final", "home_covers_rl"]
    keep = [c for c in keep if c in act.columns]
    return df.merge(act[keep], on=["_date", "home_team"], how="left")


def load_eval_2025() -> pd.DataFrame:
    """2025 backtest predictions (RL + ML)."""
    p = _ROOT / "eval_predictions.csv"
    if not p.exists():
        return pd.DataFrame()
    ep = pd.read_csv(p)
    return ep[ep["_eval_mode"] == "current"].copy()


def load_f5_val_2025() -> pd.DataFrame:
    """2025 F5 validation predictions."""
    p = _ROOT / "f5_val_predictions.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_sgp_history() -> pd.DataFrame:
    frames = []
    for pattern in ["sgp_live_edge_2026_*.csv", "sgp_live_edge_2026-*.csv"]:
        for f in sorted(glob.glob(str(_ROOT / "data/sgp" / pattern))):
            if "steam" in Path(f).stem:
                continue
            stem = Path(f).stem.replace("-", "_").split("_")
            yi   = next((i for i, p in enumerate(stem) if p == "2026"), None)
            if yi is None or yi + 2 >= len(stem):
                continue
            gdate = f"2026-{stem[yi+1]:0>2}-{stem[yi+2]:0>2}"
            try:
                d = pd.read_csv(f)
                d["game_date"] = gdate
                frames.append(d)
            except Exception:
                pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _acc(rows: pd.DataFrame, signal_col: str, outcome_col: str,
         threshold: float) -> tuple[float, int]:
    """Accuracy and n for rows where signal_col >= threshold."""
    sub = rows.dropna(subset=[signal_col, outcome_col]).copy()
    sub[signal_col]  = pd.to_numeric(sub[signal_col],  errors="coerce")
    sub[outcome_col] = pd.to_numeric(sub[outcome_col], errors="coerce")
    sub = sub.dropna(subset=[signal_col, outcome_col])
    sub = sub[sub[signal_col] >= threshold]
    if sub.empty:
        return np.nan, 0
    return sub[outcome_col].mean(), len(sub)


def _fmt(acc: float, n: int) -> str:
    if n == 0 or np.isnan(acc):
        return "  n/a      "
    return f"{acc:>6.1%} (n={n:<3})"


def _parse(legs: str, pat: str) -> float | None:
    m = re.search(pat, str(legs))
    return float(m.group(1)) if m else None


# ── Per-signal accuracy builders ──────────────────────────────────────────────

def _rl_rows_2026(df: pd.DataFrame) -> pd.DataFrame:
    rows = df.dropna(subset=["blended_rl", "home_covers_rl"]).copy()
    rows["prob"]    = pd.to_numeric(rows["blended_rl"], errors="coerce")
    rows["outcome"] = rows["home_covers_rl"].astype(float)
    return rows.dropna(subset=["prob", "outcome"])[["_date", "prob", "outcome"]]


def _ml_rows_2026(df: pd.DataFrame) -> pd.DataFrame:
    rows = df.dropna(subset=["mc_home_win", "home_score_final", "away_score_final"]).copy()
    rows["prob"]    = pd.to_numeric(rows["mc_home_win"], errors="coerce")
    rows["outcome"] = (rows["home_score_final"].astype(float) > rows["away_score_final"].astype(float)).astype(float)
    return rows.dropna(subset=["prob", "outcome"])[["_date", "prob", "outcome"]]


def _ou_rows_2026(df: pd.DataFrame) -> pd.DataFrame:
    rows = df.dropna(subset=["ou_p_model", "ou_posted_line",
                              "home_score_final", "away_score_final"]).copy()
    rows["prob"] = pd.to_numeric(rows["ou_p_model"], errors="coerce")
    gt     = rows["home_score_final"].astype(float) + rows["away_score_final"].astype(float)
    posted = pd.to_numeric(rows["ou_posted_line"], errors="coerce")
    dirn   = rows.get("ou_direction", pd.Series("OVER", index=rows.index))
    rows["outcome"] = np.where(dirn == "OVER",
                               (gt > posted).astype(float),
                               (gt < posted).astype(float))
    rows.loc[posted.isna(), "outcome"] = np.nan
    return rows.dropna(subset=["prob", "outcome"])[["_date", "prob", "outcome"]]


def _k_rows_2026(df: pd.DataFrame) -> pd.DataFrame:
    rows_list = []
    for _, r in df.iterrows():
        for side in ("home", "away"):
            prob = pd.to_numeric(r.get(f"{side}_k_model_over"), errors="coerce")
            line = pd.to_numeric(r.get(f"{side}_k_line"),       errors="coerce")
            ak   = pd.to_numeric(r.get(f"{side}_sp_k"),         errors="coerce")
            dt   = r.get("_date", "")
            if pd.isna(prob) or pd.isna(line) or pd.isna(ak):
                continue
            rows_list.append({"_date": dt, "prob": prob, "outcome": float(ak > line)})
    return pd.DataFrame(rows_list) if rows_list else pd.DataFrame(columns=["_date","prob","outcome"])


def _nrfi_rows_2026(df: pd.DataFrame) -> pd.DataFrame:
    rows = df.dropna(subset=["mc_nrfi_prob", "f1_nrfi"]).copy()
    rows["prob"]    = pd.to_numeric(rows["mc_nrfi_prob"], errors="coerce")
    rows["f1_nrfi"] = pd.to_numeric(rows["f1_nrfi"],     errors="coerce")
    rows = rows.dropna(subset=["prob", "f1_nrfi"])
    rows["edge"]    = rows["prob"].apply(lambda p: (p - JUICE) if p >= 0.5 else ((1 - p) - JUICE))
    rows["outcome"] = rows.apply(lambda r: r["f1_nrfi"] if r["prob"] >= 0.5 else 1 - r["f1_nrfi"], axis=1)
    return rows[["_date", "edge", "outcome"]]


def _f5_rows_2026(df: pd.DataFrame) -> pd.DataFrame:
    prob_col = "f5_stacker_l2" if "f5_stacker_l2" in df.columns else "mc_f5_home_win_prob"
    if prob_col not in df.columns:
        return pd.DataFrame(columns=["_date", "prob", "outcome"])
    rows = df.dropna(subset=[prob_col, "f5_home_win"]).copy()
    rows["prob"]    = pd.to_numeric(rows[prob_col],      errors="coerce")
    rows["outcome"] = pd.to_numeric(rows["f5_home_win"], errors="coerce")
    return rows.dropna(subset=["prob", "outcome"])[["_date", "prob", "outcome"]]


def _script_rows_2026(sgp: pd.DataFrame, act: pd.DataFrame, script_name: str) -> pd.DataFrame:
    if sgp.empty:
        return pd.DataFrame(columns=["game_date", "sgp_edge", "outcome"])
    sub = sgp[sgp["script"] == script_name].copy() if "script" in sgp.columns else pd.DataFrame()
    if sub.empty:
        return pd.DataFrame(columns=["game_date", "sgp_edge", "outcome"])
    sub = sub.merge(act[["game_date", "home_team", "home_sp_k", "away_sp_k",
                          "f5_total", "home_score_final", "away_score_final", "home_covers_rl"]],
                    on=["game_date", "home_team"], how="left")
    sub = sub.dropna(subset=["home_score_final", "away_score_final",
                              "f5_total", "home_sp_k", "away_sp_k", "home_covers_rl"])
    if sub.empty:
        return pd.DataFrame(columns=["game_date", "sgp_edge", "outcome"])
    gt = sub["home_score_final"].astype(float) + sub["away_score_final"].astype(float)

    def compute_hit(row, game_total):
        legs = row["legs"]
        if script_name == "A2_Dominance":
            k  = _parse(legs, r"SP_K_F5>=(\d+)") or 4
            f5 = _parse(legs, r"F5_Under_([\d.]+)") or 99
            g  = _parse(legs, r"Game_Under_([\d.]+)") or 99
            return float(row["home_sp_k"] >= k and row["f5_total"] < f5 and game_total < g)
        elif script_name == "B_Explosion":
            g = _parse(legs, r"Game_Over_([\d.]+)") or 0
            return float(row["home_score_final"] >= 5 and game_total > g
                         and row["home_score_final"] > row["away_score_final"])
        elif script_name == "C_EliteDuel":
            g = _parse(legs, r"Game_Under_([\d.]+)") or 99
            k = _parse(legs, r"SP_K_F5>=(\d+)") or 3
            return float(game_total < g and row["home_sp_k"] >= k
                         and row["away_sp_k"] >= k and row["home_covers_rl"] == 0)
        elif script_name == "D_LateDivergence":
            f5 = _parse(legs, r"F5_Under_([\d.]+)") or 99
            g  = _parse(legs, r"Game_Over_([\d.]+)") or 0
            return float(row["f5_total"] < f5 and game_total > g)
        return np.nan

    sub["outcome"]  = [compute_hit(row, gt.iloc[i]) for i, (_, row) in enumerate(sub.iterrows())]
    sub["sgp_edge"] = pd.to_numeric(sub["sgp_edge"], errors="coerce")
    return sub[["game_date", "sgp_edge", "outcome"]].rename(columns={"game_date": "_date"})


# ── Three-window accuracy per signal ─────────────────────────────────────────

def signal_accuracy(signal_key: str,
                    rows_2026: pd.DataFrame,
                    signal_col: str,
                    outcome_col: str,
                    rows_2025: pd.DataFrame | None = None,
                    signal_col_2025: str | None = None,
                    outcome_col_2025: str | None = None) -> tuple:
    """
    Returns (t1, acc_2025, n_2025, acc_2026, n_2026, acc_l30, n_l30).
    t1 = Green threshold from signal_bands.json.
    """
    b   = BANDS.get(signal_key, {})
    t1  = b.get("tier1", 0.0)

    # 2026 YTD
    acc_2026, n_2026 = _acc(rows_2026, signal_col, outcome_col, t1)

    # Last 30 days (subset of 2026)
    if "_date" in rows_2026.columns:
        r30 = rows_2026[rows_2026["_date"] >= L30_CUTOFF]
    else:
        r30 = rows_2026
    acc_l30, n_l30 = _acc(r30, signal_col, outcome_col, t1)

    # 2025 backtest
    acc_2025, n_2025 = np.nan, 0
    if rows_2025 is not None and not rows_2025.empty:
        sc = signal_col_2025 or signal_col
        oc = outcome_col_2025 or outcome_col
        acc_2025, n_2025 = _acc(rows_2025, sc, oc, t1)

    return t1, acc_2025, n_2025, acc_2026, n_2026, acc_l30, n_l30


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df26  = load_merged_2026()
    ep25  = load_eval_2025()
    f5_25 = load_f5_val_2025()
    sgp   = load_sgp_history()
    act   = pd.read_parquet(_ROOT / "data/statcast/actuals_2026.parquet")
    act["game_date"] = pd.to_datetime(act["game_date"]).dt.strftime("%Y-%m-%d")

    # Pre-build row sets
    rl26   = _rl_rows_2026(df26)   if not df26.empty  else pd.DataFrame()
    ml26   = _ml_rows_2026(df26)   if not df26.empty  else pd.DataFrame()
    ou26   = _ou_rows_2026(df26)   if not df26.empty  else pd.DataFrame()
    k26    = _k_rows_2026(df26)    if not df26.empty  else pd.DataFrame()
    nrfi26 = _nrfi_rows_2026(df26) if not df26.empty  else pd.DataFrame()
    f5_26  = _f5_rows_2026(df26)   if not df26.empty  else pd.DataFrame()
    sA2    = _script_rows_2026(sgp, act, "A2_Dominance")
    sB     = _script_rows_2026(sgp, act, "B_Explosion")
    sC     = _script_rows_2026(sgp, act, "C_EliteDuel")
    sD     = _script_rows_2026(sgp, act, "D_LateDivergence")

    # 2025 backtest row sets (RL + ML from eval_predictions; F5 from f5_val)
    ep25["_date"] = ep25["game_date"] if "game_date" in ep25.columns else ""
    rl25 = ep25.rename(columns={"rl_stacked": "edge", "home_covers_rl": "outcome"}
                       )[["_date", "edge", "outcome"]].dropna()
    ml25 = ep25.rename(columns={"ml_raw": "prob", "actual_home_win": "outcome"}
                       )[["_date", "prob", "outcome"]].dropna()

    f5_25_rows = pd.DataFrame()
    if not f5_25.empty:
        prob_col = ("stacker_f5_cover" if "stacker_f5_cover" in f5_25.columns
                    else "xgb_raw_f5_cover")
        out_col  = "f5_home_cover" if "f5_home_cover" in f5_25.columns else "f5_home_win"
        if prob_col in f5_25.columns and out_col in f5_25.columns:
            f5_25_rows = f5_25.rename(columns={prob_col: "prob", out_col: "outcome"}
                                      )[["prob", "outcome"]].dropna()
            f5_25_rows["_date"] = ""

    W = 14   # column width for acc cells

    print("=" * 96)
    print("  SIGNAL ACCURACY — GREEN TIER  (threshold from signal_bands.json)")
    print(f"  Last-30 cutoff: {L30_CUTOFF}   |   Today: {TODAY}")
    print("=" * 96)
    print(f"  {'Signal':<14}  {'Green>=':>7}  "
          f"{'-- 2025 --':^{W}}  {'-- 2026 YTD --':^{W}}  {'-- Last 30d --':^{W}}")
    print("  " + "-" * 90)

    def row(label, key, r26, sc, oc, r25=None, sc25=None, oc25=None):
        t1, a25, n25, a26, n26, a30, n30 = signal_accuracy(
            key, r26, sc, oc, r25, sc25, oc25)
        print(f"  {label:<14}  {t1:>7.3f}  "
              f"{_fmt(a25,n25):^{W}}  {_fmt(a26,n26):^{W}}  {_fmt(a30,n30):^{W}}")

    row("rl_prob",     "rl_prob",     rl26,   "prob", "outcome", rl25, "edge", "outcome")
    row("ml_win_prob", "ml_win_prob", ml26,   "prob", "outcome", ml25, "prob", "outcome")
    row("ou_prob",     "ou_prob",     ou26,   "prob", "outcome")
    row("k_over_prob", "k_over_prob", k26,    "prob", "outcome")
    row("nrfi",        "nrfi_prob",   nrfi26, "edge", "outcome")
    row("f5_win_prob", "f5_win_prob", f5_26,  "prob", "outcome", f5_25_rows, "prob", "outcome")
    row("script_a2",   "script_a2",   sA2,    "sgp_edge", "outcome")
    row("script_b",    "script_b",    sB,     "sgp_edge", "outcome")
    row("script_c",    "script_c",    sC,     "sgp_edge", "outcome")
    row("script_d",    "script_d",    sD,     "sgp_edge", "outcome")

    print("=" * 96)
    print("  2025 = full-season backtest  |  2026 YTD = live picks  |  Last 30d = rolling window")
    print("  Scripts accumulate daily — 2025 backtest not available for O/U, NRFI, K, Scripts.")
    print("=" * 96)


if __name__ == "__main__":
    main()
