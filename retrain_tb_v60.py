"""
retrain_tb_v60.py — v6.0 Total Bases fine-tune with Rule-Change Weighting.

Fine-tunes the v3.7 XGBoost booster with five new high-resolution features:
  air_density_rho        — computed from temp_f + altitude_ft (no external join)
  bullpen_burn_5d        — opponent bullpen fatigue over 5 days
  catcher_k_mult         — catcher framing IQ k-multiplier (via build_catcher_iq)
  bat_speed_pctile       — cross-sectional bat speed percentile (by year)
  swing_length_pctile    — cross-sectional swing length percentile (by year)

Rule-Change Weighting (2026 ABS system):
  2x sample weight for all 2026 rows
  Additional 1.5x (-> 3x total) for 2026 rows with bat_speed_pctile >= 75th pctile
  Rationale: disciplined/powerful hitters benefit disproportionately from ABS
  challenge opportunities in deep counts.

Fine-tunes by adding ~200 trees on top of the v3.7 checkpoint.
Target: AUC > 0.540 on 2026 hold-out.

Usage
-----
  python retrain_tb_v60.py
  python retrain_tb_v60.py --rounds 300 --no-fine-tune  # full retrain from scratch
  python retrain_tb_v60.py --dry-run                    # feature audit only
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

_ROOT       = Path(__file__).resolve().parent
MATRIX_PATH = _ROOT / "data/batter_features/final_training_matrix.parquet"
BURN_PATH   = _ROOT / "data/batter_features/bullpen_burn_by_game.parquet"
STATCAST_DIR = _ROOT / "data/statcast"
V37_MODEL   = _ROOT / "models/tb_stacker_v37.json"
OUT_MODEL   = _ROOT / "models/tb_stacker_v60.json"
OUT_TXT     = _ROOT / "data/logs/tb_v60_metrics.txt"

# --- Base v3.7 feature list (must match exactly) ---
NUMERIC_V37 = [
    "pull_side_wind_vector", "projected_total_adj", "bias_offset",
    "wind_mph", "wind_bearing", "temp_f",
    "velocity_decay_risk", "opp_sp_ff_velo",
    "lineup_fragility", "platoon_same_hand",
    "batting_order", "exp_pa_heuristic",
    "ba", "est_ba", "slg", "est_slg", "woba", "est_woba",
    "avg_hit_angle", "anglesweetspotpercent",
    "max_hit_speed", "avg_hit_speed", "ev50", "fbld",
    "ev95percent", "brl_percent", "brl_pa",
]

# --- New v6.0 features ---
NEW_V60 = [
    "air_density_rho",
    "bullpen_burn_5d",
    "catcher_k_mult",
    "bat_speed_pctile",
    "swing_length_pctile",
]

FEAT_COLS_V60 = NUMERIC_V37 + ["stand_L", "stand_R", "stand_S"] + NEW_V60

# Weight constants
W_BASE_2026        = 2.0    # all 2026 rows
W_HIGH_ABS_MULT    = 1.5    # applied on top of W_BASE_2026 for elite bat-speed (-> 3x)
BAT_SPEED_ELITE_PCT = 75.0  # percentile threshold for ABS weighting
FINE_TUNE_ROUNDS   = 200    # trees added on top of v3.7


# ---------------------------------------------------------------------------
# Air density from physics (no external join required)
# ---------------------------------------------------------------------------

# Standard atmosphere constants
_R_D   = 287.05   # J / (kg · K)
_P_SL  = 101325.0 # Pa at sea level
_T_ISA = 288.15   # K at sea level (15°C)
_L     = 0.0065   # K/m lapse rate


def _compute_air_density(temp_f: pd.Series, altitude_ft: pd.Series) -> pd.Series:
    """ρ = P(h) / (R_d · T_K) using standard atmosphere for pressure at altitude."""
    T_K   = (temp_f.fillna(70.0) - 32.0) * 5.0 / 9.0 + 273.15
    h_m   = altitude_ft.fillna(0.0) * 0.3048
    # Barometric formula: P(h) = P_SL * (1 - L*h/T_ISA)^5.2561
    P_h   = _P_SL * (1.0 - _L * h_m / _T_ISA) ** 5.2561
    rho   = P_h / (_R_D * T_K)
    return rho.round(6)


# ---------------------------------------------------------------------------
# Enrich: bullpen_burn_5d  (opponent's 5-day burn facing the batter)
# ---------------------------------------------------------------------------

def _enrich_bullpen_burn(df: pd.DataFrame) -> pd.DataFrame:
    """Join opponent bullpen_burn_5d from burn_by_game parquet."""
    if not BURN_PATH.exists():
        print("  [v60] bullpen_burn_by_game.parquet not found — using burn_3d proxy")
        df["bullpen_burn_5d"] = df.get("bullpen_burn_3d", pd.Series(np.nan, index=df.index))
        return df

    burn = pd.read_parquet(BURN_PATH)
    burn["game_date"] = pd.to_datetime(burn["game_date"])

    # Build a long frame: each game-date + team -> their 5d burn as *home* team
    home_burn = burn[["game_date", "home_team", "bullpen_burn_5d"]].copy()
    home_burn = home_burn.rename(columns={"home_team": "burn_team", "bullpen_burn_5d": "home_bp_burn_5d"})

    away_burn = burn[["game_date", "away_team", "bullpen_burn_5d"]].copy()
    away_burn = away_burn.rename(columns={"away_team": "burn_team", "bullpen_burn_5d": "away_bp_burn_5d"})

    # Opponent burn: if batter's team is home_team, the opponent is away -> use away_bullpen_burn_5d
    df = df.copy()
    df["_gd"] = pd.to_datetime(df["game_date"])
    df["_team"] = df["team"].str.strip().str.upper()

    # Merge opponent burn: join where batter's team == home_team (opponent = away)
    hm = (burn[["game_date", "home_team", "bullpen_burn_5d"]]
          .rename(columns={"home_team": "_team",
                            "bullpen_burn_5d": "_opp_burn_home"}))
    hm["game_date"] = pd.to_datetime(hm["game_date"])

    aw = (burn[["game_date", "away_team", "bullpen_burn_5d"]]
          .rename(columns={"away_team": "_team",
                            "bullpen_burn_5d": "_opp_burn_away"}))
    aw["game_date"] = pd.to_datetime(aw["game_date"])

    df = df.merge(hm, left_on=["_gd", "_team"], right_on=["game_date", "_team"],
                  how="left", suffixes=("", "_hm"))
    df = df.merge(aw, left_on=["_gd", "_team"], right_on=["game_date", "_team"],
                  how="left", suffixes=("", "_aw"))

    # Batter is home -> opponent is away; batter is away -> opponent is home
    df["bullpen_burn_5d"] = np.where(
        df["_team"] == df.get("home_team_abbr", df["_team"]),
        df["_opp_burn_away"].fillna(0),
        df["_opp_burn_home"].fillna(0),
    )

    # If we couldn't determine perspective, use whichever is non-null
    still_nan = df["bullpen_burn_5d"].isna()
    df.loc[still_nan, "bullpen_burn_5d"] = (
        df.loc[still_nan, "_opp_burn_home"].fillna(0) +
        df.loc[still_nan, "_opp_burn_away"].fillna(0)
    )

    drop_cols = [c for c in df.columns if c.startswith("_") or c.endswith("_hm") or c.endswith("_aw")]
    df = df.drop(columns=drop_cols, errors="ignore")

    fill_pct = df["bullpen_burn_5d"].isna().mean()
    if fill_pct > 0.05:
        print(f"  [v60] bullpen_burn_5d: {fill_pct:.1%} NaN after join — filling 0")
    df["bullpen_burn_5d"] = df["bullpen_burn_5d"].fillna(0)
    return df


# ---------------------------------------------------------------------------
# Enrich: bat_speed_pctile / swing_length_pctile  (cross-sectional by year)
# ---------------------------------------------------------------------------

def _load_pctile_for_year(year: int) -> pd.DataFrame:
    """Load batter_percentiles_{year}.parquet and compute cross-sectional pctile."""
    p = STATCAST_DIR / f"batter_percentiles_{year}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)

    for col in ("bat_speed", "swing_length"):
        if col in df.columns:
            df[f"{col}_pctile"] = (
                df[col].rank(pct=True, na_option="bottom") * 100
            ).round(1)
        else:
            df[f"{col}_pctile"] = np.nan

    keep = ["player_id", "bat_speed_pctile", "swing_length_pctile"]
    df["_pctile_year"] = year
    return df[[c for c in keep + ["_pctile_year"] if c in df.columns]]


def _enrich_bat_tracking(df: pd.DataFrame) -> pd.DataFrame:
    """Join bat_speed_pctile + swing_length_pctile per player per year."""
    pctile_frames = []
    for yr in df["year"].dropna().unique():
        pf = _load_pctile_for_year(int(yr))
        if not pf.empty:
            pf["year"] = int(yr)
            pctile_frames.append(pf)

    if not pctile_frames:
        print("  [v60] no batter_percentiles parquets found — using NaN for bat tracking")
        df["bat_speed_pctile"]    = np.nan
        df["swing_length_pctile"] = np.nan
        return df

    pctile_all = pd.concat(pctile_frames, ignore_index=True)
    pctile_all = pctile_all.drop_duplicates(subset=["player_id", "year"], keep="last")

    df = df.copy()
    df["_pid"] = pd.to_numeric(df["player_id"], errors="coerce")
    df["_yr"]  = df["year"].astype(int)

    # Drop extra columns that would clash with left-side df during merge
    pctile_join = (pctile_all
                   .rename(columns={"_pctile_year": "_yr_unused",
                                    "player_id": "_pid"})
                   .assign(_yr=lambda x: x["year"])
                   .drop(columns=["year", "_yr_unused"], errors="ignore")
                   [["_pid", "_yr", "bat_speed_pctile", "swing_length_pctile"]])

    df = df.merge(pctile_join, on=["_pid", "_yr"], how="left")
    df = df.drop(columns=["_pid", "_yr"], errors="ignore")

    for col in ("bat_speed_pctile", "swing_length_pctile"):
        fill_pct = df[col].isna().mean() if col in df.columns else 1.0
        if fill_pct > 0.10:
            print(f"  [v60] {col}: {fill_pct:.1%} NaN after join — filling 50th pctile")
        df[col] = df[col].fillna(50.0) if col in df.columns else 50.0

    return df


# ---------------------------------------------------------------------------
# Enrich: catcher_k_mult  (from catcher_framing_all.parquet -> build_catcher_iq)
# ---------------------------------------------------------------------------

def _enrich_catcher_iq(df: pd.DataFrame) -> pd.DataFrame:
    """Join catcher k_multiplier per team per year using build_catcher_iq."""
    try:
        from build_catcher_iq import compute_catcher_iq, _normalise_df, aggregate_by_team
    except ImportError:
        print("  [v60] build_catcher_iq not importable — catcher_k_mult = 1.0")
        df["catcher_k_mult"] = 1.0
        return df

    framing_all_path = STATCAST_DIR / "catcher_framing_all.parquet"
    if not framing_all_path.exists():
        print("  [v60] catcher_framing_all.parquet not found — catcher_k_mult = 1.0")
        df["catcher_k_mult"] = 1.0
        return df

    raw = pd.read_parquet(framing_all_path)
    team_iq_frames = []
    for yr in df["year"].dropna().unique():
        yr = int(yr)
        yr_raw = raw[raw.get("year", raw.get("season", pd.Series(yr))) == yr].copy() \
            if "year" in raw.columns or "season" in raw.columns else raw.copy()
        if yr_raw.empty:
            yr_raw = raw.copy()  # fallback: use all years
        try:
            norm     = _normalise_df(yr_raw)
            iq       = compute_catcher_iq(norm, yr)
            team_iq  = aggregate_by_team(iq)
            team_iq["_yr"] = yr
            team_iq_frames.append(team_iq)
        except Exception as e:
            print(f"  [v60] catcher IQ build failed for {yr}: {e}")

    if not team_iq_frames:
        df["catcher_k_mult"] = 1.0
        return df

    team_all = pd.concat(team_iq_frames, ignore_index=True)
    team_all["_yr"] = team_all["_yr"].astype(int)
    team_all["team"] = team_all["team"].str.upper()

    df = df.copy()
    df["_team_upper"] = df["team"].str.upper()
    df["_yr"]         = df["year"].astype(int)

    df = df.merge(
        team_all[["team", "_yr", "team_k_multiplier"]].rename(
            columns={"team": "_team_upper", "team_k_multiplier": "catcher_k_mult"}),
        on=["_team_upper", "_yr"],
        how="left",
    )
    df = df.drop(columns=["_team_upper", "_yr"], errors="ignore")
    df["catcher_k_mult"] = df["catcher_k_mult"].fillna(1.0)

    return df


# ---------------------------------------------------------------------------
# Full enrichment pipeline
# ---------------------------------------------------------------------------

def build_v60_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Add all 5 v6.0 features to the training matrix."""
    print("  [v60] computing air_density_rho …")
    df["air_density_rho"] = _compute_air_density(df["temp_f"], df["altitude_ft"])

    print("  [v60] enriching bullpen_burn_5d …")
    df = _enrich_bullpen_burn(df)

    print("  [v60] enriching bat tracking pctiles …")
    df = _enrich_bat_tracking(df)

    print("  [v60] enriching catcher IQ …")
    df = _enrich_catcher_iq(df)

    return df


# ---------------------------------------------------------------------------
# Sample weighting: Rule-Change 2026 ABS upsampling
# ---------------------------------------------------------------------------

def build_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """
    2026 rows get 2x weight (ABS rule change captures new signal).
    2026 rows with bat_speed_pctile >= 75th get an additional 1.5x -> 3x total.
    High bat-speed hitters are most impacted by ABS challenge opportunities.
    """
    weights = np.ones(len(df), dtype=float)

    mask_2026     = df["year"] == 2026
    mask_high_abs = mask_2026 & (df.get("bat_speed_pctile", 50.0) >= BAT_SPEED_ELITE_PCT)

    weights[mask_2026.values]     *= W_BASE_2026
    weights[mask_high_abs.values] *= W_HIGH_ABS_MULT

    n_2026      = mask_2026.sum()
    n_high_abs  = mask_high_abs.sum()
    print(f"  [v60] sample weights: {n_2026:,} 2026 rows (2x), "
          f"{n_high_abs:,} high-ABS rows (+1.5x -> 3x total)")
    return weights


# ---------------------------------------------------------------------------
# Train / fine-tune
# ---------------------------------------------------------------------------

def run_retrain(args: argparse.Namespace) -> dict:
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

    # 1. Load matrix
    print(f"\n[v60] Loading {MATRIX_PATH.name} …")
    df = pd.read_parquet(MATRIX_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)
    print(f"  rows: {len(df):,}  years: {sorted(df['year'].unique())}")

    # 2. Enrich with v6.0 features
    df = build_v60_matrix(df)

    # 3. Stand dummies
    for h in ("L", "R", "S"):
        df[f"stand_{h}"] = (df["stand"].fillna("") == h).astype("int8")

    # 4. Feature matrix + label
    for c in FEAT_COLS_V60:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    y = (df["total_bases"].astype(float) > 1.5).astype("int8")

    # Split: train on 2024+2025+first 75% of 2026; hold out last 25% of 2026.
    # This lets 2026 rows participate in training (getting the 2x/3x weights)
    # while preserving a recent holdout to measure rule-change lift.
    df_2026     = df[df["year"] == 2026].copy()
    n_hold      = max(1, int(len(df_2026) * 0.25))
    holdout_idx = set(df_2026.index[-n_hold:])
    is_valid    = df.index.isin(holdout_idx)

    X_tr, X_va  = df.loc[~is_valid, FEAT_COLS_V60], df.loc[is_valid, FEAT_COLS_V60]
    y_tr, y_va  = y[~is_valid], y[is_valid]

    # 5. Sample weights (training set — includes 2024, 2025, early 2026)
    sample_weights = build_sample_weights(df.loc[~is_valid])

    print(f"\n  Train: {len(X_tr):,} | Valid: {len(X_va):,}")
    print(f"  Pos rate train={y_tr.mean():.4f}  valid={y_va.mean():.4f}")

    if args.dry_run:
        print("\n  [dry-run] Feature audit complete — no training performed.")
        _print_feature_coverage(df, FEAT_COLS_V60)
        return {}

    # 6. DMatrix
    dtr = xgb.DMatrix(X_tr, label=y_tr, weight=sample_weights, feature_names=FEAT_COLS_V60)
    dva = xgb.DMatrix(X_va, label=y_va, feature_names=FEAT_COLS_V60)

    params = dict(
        objective="binary:logistic", eval_metric=["logloss", "auc"],
        tree_method="hist", max_depth=5, eta=0.04,
        subsample=0.85, colsample_bytree=0.80,
        min_child_weight=10, reg_lambda=1.2, seed=42,
    )

    if args.fine_tune and V37_MODEL.exists():
        print(f"\n  [v60] Fine-tuning from {V37_MODEL.name} (+{args.rounds} rounds) …")
        base_booster = xgb.Booster()
        base_booster.load_model(str(V37_MODEL))
        booster = xgb.train(
            params, dtr,
            num_boost_round=args.rounds,
            evals=[(dtr, "train"), (dva, "valid")],
            early_stopping_rounds=30,
            verbose_eval=25,
            xgb_model=base_booster,
        )
    else:
        print(f"\n  [v60] Full retrain from scratch ({args.rounds} rounds) …")
        booster = xgb.train(
            params, dtr,
            num_boost_round=args.rounds,
            evals=[(dtr, "train"), (dva, "valid")],
            early_stopping_rounds=40,
            verbose_eval=50,
        )

    # 7. Evaluate
    pred = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
    auc  = float(roc_auc_score(y_va, pred))
    ll   = float(log_loss(y_va, pred))
    bs   = float(brier_score_loss(y_va, pred))

    metrics = dict(
        rows_train=int(len(X_tr)), rows_valid=int(len(X_va)),
        auc=round(auc, 5), logloss=round(ll, 5), brier=round(bs, 5),
        best_iter=int(booster.best_iteration),
        base_rate_valid=round(float(y_va.mean()), 4),
    )

    print(f"\n{'='*55}")
    print(f"  TB v6.0 Validation Results")
    print(f"{'='*55}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    target_hit = "[TARGET MET]" if auc >= 0.540 else "[below 0.540 target]"
    print(f"\n  AUC = {auc:.4f}  {target_hit}")
    print(f"{'='*55}")

    # 8. Save model
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(OUT_MODEL))
    print(f"\n  Saved -> {OUT_MODEL}")

    # 9. Feature importance
    imp = booster.get_score(importance_type="gain")
    imp_df = (pd.DataFrame({"feature": list(imp), "gain": list(imp.values())})
              .sort_values("gain", ascending=False).reset_index(drop=True))
    imp_df["rank"] = imp_df.index + 1

    print("\n  Top 15 features by gain:")
    print(imp_df.head(15).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # New features ranking
    new_in_imp = imp_df[imp_df["feature"].isin(NEW_V60)]
    if not new_in_imp.empty:
        print("\n  v6.0 new feature ranks:")
        print(new_in_imp.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # 10. Save metrics log
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TXT.open("w") as f:
        f.write("TB Stacker v6.0 metrics\n")
        f.write(f"Fine-tuned from v3.7: {args.fine_tune}\n")
        f.write(f"AUC target (0.540): {target_hit}\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nFeature importance (gain):\n")
        f.write(imp_df.to_string(index=False))

    return metrics


# ---------------------------------------------------------------------------
# Feature coverage audit
# ---------------------------------------------------------------------------

def _print_feature_coverage(df: pd.DataFrame, feat_cols: list) -> None:
    print("\n  Feature coverage (null rates):")
    for c in feat_cols:
        if c in df.columns:
            null_pct = df[c].isna().mean() * 100
            flag = " [!]" if null_pct > 5 else ""
            print(f"    {c:<30} {null_pct:5.1f}% null{flag}")
        else:
            print(f"    {c:<30}  MISSING from matrix")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TB Stacker v6.0 retrain with Rule-Change Weighting")
    parser.add_argument("--rounds",      type=int,  default=FINE_TUNE_ROUNDS,
                        help=f"Boost rounds to add (default: {FINE_TUNE_ROUNDS})")
    parser.add_argument("--no-fine-tune", dest="fine_tune", action="store_false",
                        default=True,
                        help="Full retrain from scratch instead of fine-tuning v3.7")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Feature audit only — skip training")
    args = parser.parse_args()

    run_retrain(args)


if __name__ == "__main__":
    main()
