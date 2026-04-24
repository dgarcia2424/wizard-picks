"""
train_rl_v1.py — Run Line Sovereign Stacker v1.0.

Multi-stage XGBoost model for full-game Run Line (+1.5 / -1.5) that learns
*when* the Poisson fair-price is broken by physical environment variables.

Architecture
------------
Stage 1 — Poisson RL fair price (per-game pre-model signal):
    Computed analytically from SP K-rate features in the feature matrix.
    lam_home ~ close_total/2 * f(away_sp_k_pct_10d)
    lam_away ~ close_total/2 * f(home_sp_k_pct_10d)
    p_cover_poisson = P(home_score - away_score > -1.5)  [Poisson convolution]

Stage 2 — XGBoost stacker features:
    1. p_cover_poisson          — Poisson RL fair price
    2. bullpen_burn_delta       — home_bullpen_vuln - away_bullpen_vuln (fatigue)
    3. p_script_c               — sigmoid(-0.5*(close_total-8.2))  [Duel proxy]
    4. wind_vector_out          — wind_mph * cos((bearing - cf_azimuth) * pi/180)
    5. close_total              — Vegas total (magnitude signal for blowout risk)
    6. home_sp_k_pct_10d        — Home SP 10-day K rate
    7. away_sp_k_pct_10d        — Away SP 10-day K rate
    8. poisson_script_disagree  — |p_cover_poisson - (1 - p_script_c)| [gap]

Label: home_covers_rl = 1 if home_score - away_score >= -1

Outputs
-------
    models/rl_v1_stacker.json        — XGBoost booster
    models/rl_v1_feature_cols.json   — ordered feature list
    data/logs/rl_v1_metrics.txt      — AUC / accuracy / threshold stats

Usage
-----
    python train_rl_v1.py
    python train_rl_v1.py --no-train     # feature-build + dry-run only
    python train_rl_v1.py --rounds 300
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit as sigmoid
from scipy.stats import poisson
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)

_ROOT       = Path(__file__).resolve().parent
FM_PATH     = _ROOT / "feature_matrix_enriched_v2.parquet"
STAD_META   = _ROOT / "config" / "stadium_metadata.json"
OUT_MODEL   = _ROOT / "models" / "rl_v1_stacker.json"
OUT_FEATS   = _ROOT / "models" / "rl_v1_feature_cols.json"
OUT_METRICS = _ROOT / "data" / "logs" / "rl_v1_metrics.txt"

# Poisson convolution truncation
K_TRUNC = 25
RUNLINE  = 1.5   # home must win by > 1.5 to NOT cover +1.5; home covers if margin > -1

# XGBoost hyper-parameters
XGB_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      "auc",
    "max_depth":        4,
    "learning_rate":    0.03,
    "subsample":        0.75,
    "colsample_bytree": 0.80,
    "min_child_weight": 8,
    "gamma":            0.10,
    "reg_lambda":       1.5,
    "seed":             42,
    "n_jobs":           -1,
    "tree_method":      "hist",
}
DEFAULT_ROUNDS  = 250
EARLY_STOP      = 30


# ---------------------------------------------------------------------------
# Stage 1 — Poisson RL fair price
# ---------------------------------------------------------------------------

def _p_cover_poisson_vec(lam_h: np.ndarray, lam_a: np.ndarray,
                          runline: float = RUNLINE) -> np.ndarray:
    """
    Vectorised P(home_score - away_score > -runline) via Poisson convolution.
    Home covers +runline when they win OR lose by fewer than runline runs.
    Home covers if: (home_runs - away_runs) >= ceil(-runline + 1) = 0  [+1.5]
    i.e., home_score >= away_score - 1
    """
    cutoff = int(math.ceil(-runline))   # -1 for runline=1.5
    out = np.zeros(len(lam_h))
    scores = np.arange(K_TRUNC + 1)
    for i, (lh, la) in enumerate(zip(lam_h, lam_a)):
        lh = max(float(lh), 0.1)
        la = max(float(la), 0.1)
        ph = poisson.pmf(scores, lh)
        pa = poisson.pmf(scores, la)
        # P(h - a > cutoff)  →  P(h > a + cutoff)
        p = 0.0
        for a_score in range(K_TRUNC + 1):
            h_min = a_score + cutoff + 1
            if h_min <= K_TRUNC:
                p += pa[a_score] * ph[h_min:].sum()
            elif h_min < 0:
                p += pa[a_score]
        out[i] = min(max(p, 0.0), 1.0)
    return out


def _build_poisson_lambdas(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate Poisson lambdas from feature-matrix SP K-rate data.
    lam = (close_total / 2) * (1 - sp_k_advantage)
    where sp_k_advantage = (sp_k_pct_10d - 0.22) * 3  [z-score-like scaling]
    """
    total = df["close_total"].fillna(8.8).clip(5.0, 15.0).values / 2.0

    hk = df["home_sp_k_pct_10d"].fillna(0.22).values
    ak = df["away_sp_k_pct_10d"].fillna(0.22).values

    # Home team scores AGAINST the away SP — high away SP K% → fewer home runs
    sp_adj_home = np.clip(1.0 - (ak - 0.22) * 3.0, 0.60, 1.40)
    # Away team scores AGAINST the home SP
    sp_adj_away = np.clip(1.0 - (hk - 0.22) * 3.0, 0.60, 1.40)

    lam_home = np.clip(total * sp_adj_home, 0.5, 12.0)
    lam_away = np.clip(total * sp_adj_away, 0.5, 12.0)
    return lam_home, lam_away


# ---------------------------------------------------------------------------
# Stadium CF azimuth lookup
# ---------------------------------------------------------------------------

_CF_AZIMUTH_CACHE: dict[str, float] | None = None


def _load_cf_azimuths() -> dict[str, float]:
    global _CF_AZIMUTH_CACHE
    if _CF_AZIMUTH_CACHE is not None:
        return _CF_AZIMUTH_CACHE
    if not STAD_META.exists():
        _CF_AZIMUTH_CACHE = {}
        return _CF_AZIMUTH_CACHE
    meta  = json.load(STAD_META.open())
    parks = meta.get("parks", {})
    _CF_AZIMUTH_CACHE = {
        k.upper(): float(v.get("cf_azimuth_deg", 45))
        for k, v in parks.items()
        if v.get("cf_azimuth_deg") is not None
    }
    return _CF_AZIMUTH_CACHE


def _wind_vector_out(wind_mph: pd.Series, wind_bearing: pd.Series,
                     home_team: pd.Series) -> np.ndarray:
    """
    Component of wind blowing OUT to CF: mph * cos(bearing - cf_azimuth).
    Positive = blowing out (more HR/runs), negative = blowing in.
    Falls back to wind_mph * cos(bearing * pi/180) when CF azimuth unknown.
    """
    cf_map  = _load_cf_azimuths()
    cf_az   = home_team.str.upper().map(cf_map).fillna(45.0).values
    bearing = wind_bearing.fillna(0.0).values
    mph     = wind_mph.fillna(0.0).values
    delta   = (bearing - cf_az) * math.pi / 180.0
    return mph * np.cos(delta)


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------

RL_V1_FEATURES = [
    "p_cover_poisson",        # Stage 1 Poisson fair price  [primary signal]
    "bullpen_burn_delta",     # home_bullpen_vuln - away_bullpen_vuln
    "p_script_c",             # Duel proxy  sigmoid(-0.5*(total-8.2))
    "wind_vector_out",        # park-adjusted wind component to CF
    "close_total",            # Vegas total line
    "home_sp_k_pct_10d",      # home SP 10-day K rate
    "away_sp_k_pct_10d",      # away SP 10-day K rate
    "poisson_script_disagree",# |p_cover_poisson - (1 - p_script_c)| [gap measure]
]


def build_rl_features(df: pd.DataFrame,
                      verbose: bool = False) -> pd.DataFrame:
    """Compute all RL v1 features from a game-level feature matrix row."""
    out = pd.DataFrame(index=df.index)

    # 1. Poisson RL fair price (vectorised)
    if verbose:
        print("  [rl_v1] Computing Poisson lambdas …")
    lam_h, lam_a = _build_poisson_lambdas(df)
    if verbose:
        print("  [rl_v1] Convolving Poisson distributions …")
    out["p_cover_poisson"] = _p_cover_poisson_vec(lam_h, lam_a)

    # 2. Bullpen fatigue delta
    h_vuln = df.get("home_bullpen_vulnerability",
                    pd.Series(0.0, index=df.index)).fillna(0.0)
    a_vuln = df.get("away_bullpen_vulnerability",
                    pd.Series(0.0, index=df.index)).fillna(0.0)
    out["bullpen_burn_delta"] = (h_vuln - a_vuln).values

    # 3. Script C Duel probability
    total = df["close_total"].fillna(8.8).values
    out["p_script_c"] = sigmoid(-0.5 * (total - 8.2))

    # 4. Stadium-specific wind vector
    out["wind_vector_out"] = _wind_vector_out(
        df["wind_mph"], df["wind_bearing"], df["home_team"]
    )

    # 5. Vegas total (raw)
    out["close_total"] = df["close_total"].fillna(8.8).values

    # 6-7. SP K rates
    out["home_sp_k_pct_10d"] = df["home_sp_k_pct_10d"].fillna(0.22).values
    out["away_sp_k_pct_10d"] = df["away_sp_k_pct_10d"].fillna(0.22).values

    # 8. Poisson vs Script C disagreement
    out["poisson_script_disagree"] = (
        out["p_cover_poisson"] - (1.0 - out["p_script_c"])
    ).abs()

    return out[RL_V1_FEATURES]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_dataset(fm: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Filter to labelled rows, build features, return (X, y)."""
    df = fm.dropna(subset=["home_covers_rl"]).copy()
    df = df.dropna(subset=["close_total", "home_sp_k_pct_10d", "away_sp_k_pct_10d"])
    print(f"  [rl_v1] Labelled rows: {len(df):,} "
          f"(cover rate={df['home_covers_rl'].mean():.3f})")

    X = build_rl_features(df, verbose=True)
    y = df["home_covers_rl"].astype(int)
    return X, y


def train(rounds: int = DEFAULT_ROUNDS) -> dict:
    print(f"[rl_v1] Loading feature matrix: {FM_PATH}")
    fm = pd.read_parquet(FM_PATH)
    fm["game_date"] = pd.to_datetime(fm["game_date"])

    X, y = build_dataset(fm)

    # Temporal split: last 20% of games as validation (date-ordered)
    dates = fm.loc[X.index, "game_date"].values
    sort_idx   = np.argsort(dates)
    n_val      = max(1, int(len(sort_idx) * 0.20))
    val_mask   = np.zeros(len(X), dtype=bool)
    val_mask[sort_idx[-n_val:]] = True

    X_tr, X_va = X.loc[~val_mask], X.loc[val_mask]
    y_tr, y_va = y.loc[~val_mask], y.loc[val_mask]
    print(f"  [rl_v1] Train: {len(X_tr):,} | Val: {len(X_va):,}")

    dtrain = xgb.DMatrix(X_tr.values, label=y_tr.values,
                         feature_names=RL_V1_FEATURES)
    dval   = xgb.DMatrix(X_va.values, label=y_va.values,
                         feature_names=RL_V1_FEATURES)

    print(f"  [rl_v1] Training {rounds} rounds (early_stop={EARLY_STOP}) …")
    evals_result: dict = {}
    model = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=50,
        evals_result=evals_result,
    )

    best_iter = model.best_iteration
    p_val     = model.predict(dval, iteration_range=(0, best_iter + 1))
    auc       = roc_auc_score(y_va.values, p_val)
    acc_50    = float(((p_val > 0.50).astype(int) == y_va.values).mean())
    cover_rate_va = float(y_va.mean())

    # Calibration check: mean predicted vs actual cover rate
    mean_pred = float(p_val.mean())

    print(f"\n  [rl_v1] RESULTS")
    print(f"  AUC        = {auc:.4f}")
    print(f"  Accuracy   = {acc_50:.4f}  (threshold=0.50)")
    print(f"  Cover rate = {cover_rate_va:.4f}  (actual val)")
    print(f"  Mean pred  = {mean_pred:.4f}  (calibration)")
    print(f"  Best iter  = {best_iter}")

    # Feature importance
    scores = model.get_score(importance_type="gain")
    top    = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:8]
    print("\n  Feature gain (top 8):")
    for fname, gain in top:
        print(f"    {fname:<30s} {gain:.2f}")

    # Save
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(OUT_MODEL))
    json.dump(RL_V1_FEATURES, OUT_FEATS.open("w"), indent=2)
    print(f"\n  Saved -> {OUT_MODEL.name}")
    print(f"  Saved -> {OUT_FEATS.name}")

    # Metrics log
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_METRICS.open("w") as f:
        f.write(f"RL v1 Stacker Metrics — {date.today().isoformat()}\n")
        f.write(f"AUC:         {auc:.4f}\n")
        f.write(f"Accuracy:    {acc_50:.4f}\n")
        f.write(f"Cover rate:  {cover_rate_va:.4f}\n")
        f.write(f"Mean pred:   {mean_pred:.4f}\n")
        f.write(f"Best iter:   {best_iter}\n")
        f.write(f"Train rows:  {len(X_tr)}\n")
        f.write(f"Val rows:    {len(X_va)}\n")
        f.write(f"\nFeature importance (gain):\n")
        for fname, gain in top:
            f.write(f"  {fname:<30s} {gain:.2f}\n")

    return {
        "auc": auc,
        "accuracy": acc_50,
        "best_iter": best_iter,
        "cover_rate_val": cover_rate_va,
        "mean_pred_val": mean_pred,
    }


# ---------------------------------------------------------------------------
# Public scoring API  (imported by generate_dashboard_json.py)
# ---------------------------------------------------------------------------

_MODEL_CACHE: xgb.Booster | None = None


def _load_model() -> xgb.Booster | None:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if not OUT_MODEL.exists():
        return None
    b = xgb.Booster()
    b.load_model(str(OUT_MODEL))
    _MODEL_CACHE = b
    return b


def score_today(date_str: str,
                lineup_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Score today's games with the v1 RL stacker.

    All games are batched into a single feature-computation pass — one
    Poisson convolution for the full slate, not one per game.

    Returns DataFrame with columns:
        home_team, away_team, home_sp, away_sp,
        p_home_cover_v1,   — XGBoost stacker output
        p_cover_poisson,   — Stage 1 Poisson prior
        rl_edge_v1,        — p_home_cover_v1 - 0.50
        rl_edge_poisson,   — p_cover_poisson - 0.50
        rl_signal_flag,    — 1 if both agree edge > 0.05
    """
    model = _load_model()
    if model is None:
        return pd.DataFrame()

    if lineup_df is None:
        try:
            import sys
            sys.path.insert(0, str(_ROOT))
            from score_ml_today import get_todays_games
            lineup_df = get_todays_games(date_str)
        except Exception as e:
            print(f"[rl_v1] lineup load failed: {e}")
            return pd.DataFrame()

    if lineup_df is None or lineup_df.empty:
        return pd.DataFrame()

    fm_path = _ROOT / "feature_matrix_enriched_v2.parquet"
    if not fm_path.exists():
        return pd.DataFrame()

    fm = pd.read_parquet(fm_path)
    fm["game_date"] = pd.to_datetime(fm["game_date"])

    # ── Batch: one feature row per game ──────────────────────────────────
    meta_rows:  list[dict]      = []
    base_rows:  list[pd.Series] = []

    for _, g in lineup_df.iterrows():
        home    = str(g.get("home_team", "")).upper()
        away    = str(g.get("away_team", "")).upper()
        home_sp = g.get("home_starter_name", "")
        away_sp = g.get("away_starter_name", "")

        mask = fm["home_team"].str.upper() == home
        sub  = fm[mask].sort_values("game_date", ascending=False).head(5)
        base = (sub.iloc[0] if not sub.empty else fm.iloc[-1]).copy()

        if "home_sp_k_pct_10d" in g.index:
            base["home_sp_k_pct_10d"] = g["home_sp_k_pct_10d"]
        if "away_sp_k_pct_10d" in g.index:
            base["away_sp_k_pct_10d"] = g["away_sp_k_pct_10d"]
        base["home_team"] = home

        meta_rows.append({"home_team": home, "away_team": away,
                           "home_sp": home_sp, "away_sp": away_sp})
        base_rows.append(base)

    if not base_rows:
        return pd.DataFrame()

    batch_df = pd.DataFrame(base_rows).reset_index(drop=True)
    X_batch  = build_rl_features(batch_df)   # single Poisson pass for all games

    dmat    = xgb.DMatrix(X_batch.values, feature_names=RL_V1_FEATURES)
    p_v1s   = model.predict(dmat)
    p_pois  = X_batch["p_cover_poisson"].values

    out_rows = []
    for i, meta in enumerate(meta_rows):
        p_v1    = float(p_v1s[i])
        p_poi   = float(p_pois[i])
        edge_v1 = round(p_v1 - 0.50, 4)
        edge_po = round(p_poi - 0.50, 4)
        flag    = int(abs(edge_v1) > 0.05 and abs(edge_po) > 0.05
                      and (edge_v1 * edge_po) > 0)
        out_rows.append({
            **meta,
            "p_home_cover_v1":  round(p_v1, 4),
            "p_cover_poisson":  round(p_poi, 4),
            "rl_edge_v1":       edge_v1,
            "rl_edge_poisson":  edge_po,
            "rl_signal_flag":   flag,
        })

    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL Sovereign Stacker v1.0")
    parser.add_argument("--rounds",   type=int,  default=DEFAULT_ROUNDS)
    parser.add_argument("--no-train", action="store_true",
                        help="Feature dry-run only — do not train or save model")
    args = parser.parse_args()

    if args.no_train:
        print("[rl_v1] Dry-run: building features only …")
        fm = pd.read_parquet(FM_PATH)
        X, y = build_dataset(fm)
        print(X.describe().to_string())
        print(f"\nNull rates:\n{X.isna().mean().round(4).to_string()}")
        return

    metrics = train(rounds=args.rounds)
    target_auc = 0.540
    verdict = "[TARGET MET]" if metrics["auc"] >= target_auc else f"[below {target_auc} target]"
    print(f"\n  Final AUC: {metrics['auc']:.4f}  {verdict}")


if __name__ == "__main__":
    main()
