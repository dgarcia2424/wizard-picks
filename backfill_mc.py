"""
backfill_mc.py
==============
Retroactively populate MC simulation features in feature_matrix.parquet for
all historical training rows (2023-2025+) where the values are currently NaN.

At inference time run_today.py passes mc_expected_runs and mc_f5_* probabilities
from the live Monte Carlo engine into the XGBoost feature vector.  During
training these columns are NaN because the MC runs are done at prediction time,
not at matrix-build time.  This gap means XGBoost never learns the residual
between its own output and the physics-based MC estimate — a potentially large
signal ("Residual Learning" architecture).

Columns populated
-----------------
  Full-game (existing):
    mc_expected_runs          — E[total runs, 9 innings]

  F5 simulation (new — residual learning for F5 XGBoost):
    mc_f5_home_win_pct        — P(home wins F5 outright)
    mc_f5_away_win_pct        — P(away wins F5 outright)
    mc_f5_tie_pct             — P(tied after 5 innings)
    mc_f5_expected_total      — E[total F5 runs]
    mc_f5_home_cover_pct      — P(home wins OR ties) = home_win + tie
                                 This is the direct +0.5 run-line baseline.

Direction convention (matches monte_carlo_runline.py exactly)
--------------------------------------------------------------
  home_f5_lambda = away_sp_rpg × (5/9)   ← HOME team scores off AWAY SP
  away_f5_lambda = home_sp_rpg × (5/9)   ← AWAY team scores off HOME SP

  Feature matrix convention:
    home_sp_xwoba_against  → home SP's run-concession rate → AWAY team offense
    away_sp_xwoba_against  → away SP's run-concession rate → HOME team offense

How the backfill works
----------------------
Rather than re-running the full MC engine (which needs park/weather/lineup
data not always available for 2023-2025), we use the pitcher EWMA stats that
are ALREADY in the feature matrix (home_sp_xwoba_against, away_sp_xwoba_against,
etc.) — these are the same quality signals the MC engine uses, already
properly time-shifted to prevent leakage.

GPU-vectorised MC (v5.0)
------------------------
All N_GAMES are simulated simultaneously with a [N_GAMES × N_SIMS] tensor on
the RTX 5080 via CuPy.  With 50,000 sims × 7,600 games that is 380 M Poisson
draws in a single kernel launch rather than a Python for-loop.

The same Poisson-LogNormal copula used in the live engine is applied:
  • Overdispersion: V ~ LogNormal(0, σ_nb)  →  Var(X) ≈ 9.9  (vs Poisson 4.4)
  • Run correlation: Cholesky-correlated log-normal mixing with ρ_copula = 0.14
    → Corr(H,A) ≈ 0.07 (matches empirical run-pair correlation)
  • xwOBA → runs:  R_conceded ≈ XWOBA_INTERCEPT + XWOBA_SLOPE × pitcher_xwOBA

Output
------
  Overwrites feature_matrix.parquet with all MC columns populated.
  Also saves a diagnostic CSV: mc_backfill_diagnostics.csv

Usage
-----
  python backfill_mc.py
  python backfill_mc.py --sims 50000           # default
  python backfill_mc.py --matrix feature_matrix.parquet
  python backfill_mc.py --f5-only              # only F5 columns (skip full-game)
  python backfill_mc.py --force                # re-run even if columns exist
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── GPU backend: CuPy with transparent numpy fallback ──────────────────────
# Identical pattern to monte_carlo_runline.py — all cp.* calls work on CPU
# if CuPy is absent or no CUDA device is found.
try:
    import cupy as cp
    _GPU = cp.cuda.is_available()
    if _GPU:
        # Probe cuRAND — curand*.dll may be absent even when CUDA device exists
        try:
            _probe = cp.random.standard_normal(1)
            del _probe
        except (ImportError, Exception):
            _GPU = False
    if not _GPU:
        import numpy as cp          # no cuRAND or no device → CPU fallback
except ImportError:
    import numpy as cp              # CuPy not installed → CPU fallback
    _GPU = False

# ---------------------------------------------------------------------------
# MC constants (mirror monte_carlo_runline.py v5.0)
# ---------------------------------------------------------------------------
XWOBA_INTERCEPT  = -3.0
XWOBA_SLOPE      = 24.0
LEAGUE_XWOBA     = 0.318
INNINGS_SP       = 5.5
INNINGS_GAME     = 9.0
DEFAULT_BP_XWOBA = 0.310

# Poisson-LogNormal copula parameters (calibrated to 2022-2024 MLB data)
RUN_SIGMA_NB   : float = 0.50   # log-normal mixing SD  → effective NB r ≈ 4
                                 # Var(X) = μ + μ²(e^σ²−1) ≈ 9.9 at μ=4.4
RUN_RHO_COPULA : float = 0.14   # Gaussian copula ρ     → Corr(H,A) ≈ 0.07

# Stadium elevations (feet) — run multiplier via air density
STADIUM_ELEVATION = {
    "COL": 5200, "AZ":  1082, "ARI": 1082, "TEX": 551,  "HOU": 43,
    "ATL": 1050, "STL": 465,  "KC":  740,  "MIN": 840,  "CIN": 550,
    "MIL": 635,  "CHC": 595,  "CLE": 653,  "DET": 600,  "PIT": 730,
    "PHI": 20,   "NYY": 55,   "NYM": 33,   "BOS": 19,   "BAL": 53,
    "WSH": 25,   "WAS": 25,   "TOR": 249,  "TB":  28,   "MIA": 8,
    "SF":  52,   "LAD": 512,  "LAA": 160,  "SD":  17,   "SEA": 56,
    "ATH": 25,   "OAK": 25,   "CWS": 20,
}

N_SIMS_DEFAULT = 50_000   # scale to RTX 5080 — 380 M draws per batch (7.6k games)

# F5 innings parameters
INNINGS_F5     = 5.0
SP_F5_FRAC     = 0.92   # SP handles ~92% of F5 innings; 8% relief
BP_F5_FRAC     = 0.08

# New F5 column names written to the feature matrix
F5_SIM_COLS = [
    "mc_f5_home_win_pct",
    "mc_f5_away_win_pct",
    "mc_f5_tie_pct",
    "mc_f5_expected_total",
    "mc_f5_home_cover_pct",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def air_density_ratio(elev_ft: float) -> float:
    return 1.0 + (elev_ft / 5200.0) * 0.35


def xwoba_to_runs_per_9(xwoba: float) -> float:
    """Convert SP xwOBA-against to expected runs conceded per 9 innings."""
    return XWOBA_INTERCEPT + XWOBA_SLOPE * float(xwoba)


def whip_to_xwoba(whip) -> float:
    """Approximate bullpen xwOBA from WHIP (linear mapping)."""
    if pd.isna(whip):
        return DEFAULT_BP_XWOBA
    return float(np.clip(LEAGUE_XWOBA + (float(whip) - 1.25) * 0.08, 0.24, 0.42))


# ---------------------------------------------------------------------------
# GPU-vectorised batch simulation
# ---------------------------------------------------------------------------

def sim_batch(
    mu_h: np.ndarray,          # [N_GAMES] home-team expected runs (offense lambda)
    mu_a: np.ndarray,          # [N_GAMES] away-team expected runs (offense lambda)
    n_sims: int = N_SIMS_DEFAULT,
    sigma: float = RUN_SIGMA_NB,
    rho: float = RUN_RHO_COPULA,
) -> dict:
    """
    Simulate all games simultaneously on GPU with the Poisson-LogNormal copula.
    Returns a dict of [N_GAMES] arrays with full outcome distribution.

    Tensor shapes:
      z1, z2, eps_h, eps_a, V_h, V_a  →  [N_GAMES, N_SIMS]
      home_runs, away_runs             →  [N_GAMES, N_SIMS]

    Memory estimate (float32) at 7,600 games × 50,000 sims:
      6 tensors × 7600 × 50000 × 4 bytes ≈ 9.1 GB  (within RTX 5080 16 GB)
      If OOM: reduce --sims or process in batches.

    Direction convention:
      mu_h[i] = home team offense lambda = runs home team SCORES (vs away pitching)
      mu_a[i] = away team offense lambda = runs away team SCORES (vs home pitching)
    """
    N = len(mu_h)

    mu_h_gpu = cp.asarray(mu_h[:, None], dtype=cp.float32)   # [N, 1]
    mu_a_gpu = cp.asarray(mu_a[:, None], dtype=cp.float32)   # [N, 1]

    # Cholesky-correlated normals: eps_h = z1, eps_a = ρ·z1 + √(1-ρ²)·z2
    rho_sqrt = float(np.sqrt(max(0.0, 1.0 - rho ** 2)))
    z1 = cp.random.standard_normal((N, n_sims)).astype(cp.float32)
    z2 = cp.random.standard_normal((N, n_sims)).astype(cp.float32)
    eps_h = z1
    eps_a = cp.float32(rho) * z1 + cp.float32(rho_sqrt) * z2

    # Log-normal mixing: V ~ LogNormal(−σ²/2, σ²) → E[V] = 1
    half_s2 = cp.float32(0.5 * sigma * sigma)
    sig_f32  = cp.float32(sigma)
    V_h = cp.exp(sig_f32 * eps_h - half_s2)
    V_a = cp.exp(sig_f32 * eps_a - half_s2)

    # Compound Poisson draws: H|V_h ~ Poisson(μ_h · V_h)
    home_runs = cp.random.poisson(mu_h_gpu * V_h).astype(cp.float32)  # [N, N_SIMS]
    away_runs = cp.random.poisson(mu_a_gpu * V_a).astype(cp.float32)  # [N, N_SIMS]

    margin = home_runs - away_runs   # [N, N_SIMS]  positive = home leads

    def _np(arr):
        return cp.asnumpy(arr) if _GPU else np.asarray(arr)

    return {
        "expected_total":   _np((home_runs + away_runs).mean(axis=1)),   # [N]
        "home_win_pct":     _np((margin > 0).mean(axis=1)),              # [N]
        "away_win_pct":     _np((margin < 0).mean(axis=1)),              # [N]
        "tie_pct":          _np((margin == 0).mean(axis=1)),             # [N]
        "home_cover_pct":   _np((margin >= 0).mean(axis=1)),             # [N] +0.5
    }


def sim_batch_expected_runs(
    mu_h: np.ndarray,
    mu_a: np.ndarray,
    n_sims: int = N_SIMS_DEFAULT,
    sigma: float = RUN_SIGMA_NB,
    rho: float = RUN_RHO_COPULA,
) -> np.ndarray:
    """Backward-compatible wrapper — returns only expected_total."""
    return sim_batch(mu_h, mu_a, n_sims=n_sims, sigma=sigma, rho=rho)["expected_total"]


# ---------------------------------------------------------------------------
# Legacy single-game function (kept for CLI / unit-test compatibility)
# ---------------------------------------------------------------------------

def sim_game_expected_runs(
    home_sp_xwoba: float,
    away_sp_xwoba: float,
    home_bp_xwoba: float,
    away_bp_xwoba: float,
    home_team: str,
    n_sims: int = N_SIMS_DEFAULT,
    rng: np.random.Generator = None,   # ignored — GPU path used instead
) -> float:
    """Single-game wrapper that delegates to sim_batch_expected_runs."""
    elev     = STADIUM_ELEVATION.get(home_team, 0)
    alt_mult = air_density_ratio(elev)
    sp_frac  = INNINGS_SP / INNINGS_GAME
    bp_frac  = 1.0 - sp_frac

    home_sp_r9 = np.clip(xwoba_to_runs_per_9(home_sp_xwoba), 1.5, 10.0)
    away_sp_r9 = np.clip(xwoba_to_runs_per_9(away_sp_xwoba), 1.5, 10.0)
    home_bp_r9 = np.clip(xwoba_to_runs_per_9(home_bp_xwoba), 1.5,  8.0)
    away_bp_r9 = np.clip(xwoba_to_runs_per_9(away_bp_xwoba), 1.5,  8.0)

    mu_h = np.array([(home_sp_r9 * sp_frac + home_bp_r9 * bp_frac) * alt_mult])
    mu_a = np.array([(away_sp_r9 * sp_frac + away_bp_r9 * bp_frac) * alt_mult])

    return float(sim_batch_expected_runs(mu_h, mu_a, n_sims=n_sims)[0])


# ---------------------------------------------------------------------------
# Main backfill routine
# ---------------------------------------------------------------------------

def _build_lambdas(targets: pd.DataFrame) -> dict:
    """
    Derive per-game Poisson offense lambdas for both full-game and F5.

    Direction convention (matches monte_carlo_runline.py):
      home_sp_xwoba_against → HOME SP quality → AWAY team offense
      away_sp_xwoba_against → AWAY SP quality → HOME team offense

    So:
      mu_home_fullgame = away_sp_r9 * sp_frac + away_bp_r9 * bp_frac   (home scores)
      mu_away_fullgame = home_sp_r9 * sp_frac + home_bp_r9 * bp_frac   (away scores)

      mu_home_f5 = away_sp_r9 * SP_F5_FRAC * (5/9) + away_bp_r9 * BP_F5_FRAC * (5/9)
      mu_away_f5 = home_sp_r9 * SP_F5_FRAC * (5/9) + home_bp_r9 * BP_F5_FRAC * (5/9)
    """
    def _safe_xwoba(col):
        v = pd.to_numeric(targets.get(col, np.nan), errors="coerce").values
        return np.where(np.isnan(v), LEAGUE_XWOBA, v).astype(np.float64)

    def _whip_to_xwoba_arr(col):
        vals = pd.to_numeric(targets.get(col, np.nan), errors="coerce").values
        return np.array([whip_to_xwoba(w) for w in vals], dtype=np.float64)

    # xwOBA-against for each SP and bullpen
    home_sp_x = _safe_xwoba("home_sp_xwoba_against")   # HOME SP quality  → away offense
    away_sp_x = _safe_xwoba("away_sp_xwoba_against")   # AWAY SP quality  → home offense
    home_bp_x = _whip_to_xwoba_arr("home_bp_whip")     # HOME BP quality  → away offense
    away_bp_x = _whip_to_xwoba_arr("away_bp_whip")     # AWAY BP quality  → home offense

    # Altitude run multipliers per home team
    home_teams = targets.get("home_team", pd.Series(["NYY"] * len(targets))).values
    alt_mults  = np.array(
        [air_density_ratio(STADIUM_ELEVATION.get(str(t), 0)) for t in home_teams],
        dtype=np.float64,
    )

    # R/9 conversion (clipped to reasonable range)
    home_sp_r9 = np.clip(XWOBA_INTERCEPT + XWOBA_SLOPE * home_sp_x, 1.5, 10.0)
    away_sp_r9 = np.clip(XWOBA_INTERCEPT + XWOBA_SLOPE * away_sp_x, 1.5, 10.0)
    home_bp_r9 = np.clip(XWOBA_INTERCEPT + XWOBA_SLOPE * home_bp_x, 1.5,  8.0)
    away_bp_r9 = np.clip(XWOBA_INTERCEPT + XWOBA_SLOPE * away_bp_x, 1.5,  8.0)

    # Full-game lambdas (9 innings)
    sp_frac = INNINGS_SP  / INNINGS_GAME
    bp_frac = 1.0 - sp_frac

    mu_home_9 = (away_sp_r9 * sp_frac + away_bp_r9 * bp_frac) * alt_mults
    mu_away_9 = (home_sp_r9 * sp_frac + home_bp_r9 * bp_frac) * alt_mults

    # F5 lambdas (5 innings, mostly SP)
    f5_scale  = INNINGS_F5 / INNINGS_GAME   # 5/9 scaling
    mu_home_f5 = (away_sp_r9 * SP_F5_FRAC + away_bp_r9 * BP_F5_FRAC) * f5_scale * alt_mults
    mu_away_f5 = (home_sp_r9 * SP_F5_FRAC + home_bp_r9 * BP_F5_FRAC) * f5_scale * alt_mults

    # Floor: avoid degenerate near-zero lambdas
    mu_home_9  = np.maximum(mu_home_9,  0.10)
    mu_away_9  = np.maximum(mu_away_9,  0.10)
    mu_home_f5 = np.maximum(mu_home_f5, 0.05)
    mu_away_f5 = np.maximum(mu_away_f5, 0.05)

    return {
        "mu_home_9":  mu_home_9,
        "mu_away_9":  mu_away_9,
        "mu_home_f5": mu_home_f5,
        "mu_away_f5": mu_away_f5,
    }


def backfill(
    matrix_path: Path = Path("feature_matrix.parquet"),
    n_sims: int = N_SIMS_DEFAULT,
    f5_only: bool = False,
    force: bool = False,
    verbose: bool = True,
) -> None:
    """
    Populate mc_expected_runs AND F5 simulation columns in the feature matrix.
    Uses vectorised GPU batch launches — no Python for-loop over games.

    New columns added:
      mc_f5_home_win_pct    — P(home wins F5 outright)
      mc_f5_away_win_pct    — P(away wins F5 outright)
      mc_f5_tie_pct         — P(tied after 5)
      mc_f5_expected_total  — E[F5 runs home + away]
      mc_f5_home_cover_pct  — P(home wins OR ties) — direct +0.5 baseline
    """
    if not matrix_path.exists():
        print(f"  ERROR: {matrix_path} not found")
        return

    df      = pd.read_parquet(matrix_path, engine="pyarrow")
    n_total = len(df)

    # ── Ensure F5 columns exist ──────────────────────────────────────────
    for col in F5_SIM_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # ── Determine which rows need filling ───────────────────────────────
    if force:
        rows_to_fill = pd.Series(True, index=df.index)
    else:
        # Fill any row missing mc_expected_runs OR any F5 column
        missing_fullgame = df["mc_expected_runs"].isna()
        missing_f5       = df[F5_SIM_COLS].isna().any(axis=1)
        rows_to_fill     = missing_f5 | (missing_fullgame & ~f5_only)

    n_fill = rows_to_fill.sum()
    n_done = n_total - n_fill

    print(f"  Matrix   : {n_total} rows total")
    print(f"  Already  : {n_done} rows fully populated")
    print(f"  To fill  : {n_fill} rows")

    if n_fill == 0:
        print("  Nothing to backfill — all rows already populated.")
        return

    targets = df[rows_to_fill].copy()

    if verbose:
        device_str = "GPU (CuPy)" if _GPU else "CPU (NumPy fallback)"
        print(f"  Device   : {device_str}")
        total_draws = n_sims * len(targets)
        print(f"  Sims     : {n_sims:,} × {len(targets):,} games = {total_draws:,} Poisson draws")

    # ── Build offense lambdas ─────────────────────────────────────────────
    lam = _build_lambdas(targets)

    # ── Full-game simulation ──────────────────────────────────────────────
    if not f5_only:
        if verbose:
            print("  [1/2] Full-game simulation ...")
        fg_res = sim_batch(lam["mu_home_9"], lam["mu_away_9"], n_sims=n_sims)
        df.loc[rows_to_fill, "mc_expected_runs"] = fg_res["expected_total"]

    # ── F5 simulation ─────────────────────────────────────────────────────
    if verbose:
        label = "[1/1]" if f5_only else "[2/2]"
        print(f"  {label} F5 simulation ...")
    f5_res = sim_batch(lam["mu_home_f5"], lam["mu_away_f5"], n_sims=n_sims)

    df.loc[rows_to_fill, "mc_f5_home_win_pct"]   = f5_res["home_win_pct"].round(4)
    df.loc[rows_to_fill, "mc_f5_away_win_pct"]   = f5_res["away_win_pct"].round(4)
    df.loc[rows_to_fill, "mc_f5_tie_pct"]        = f5_res["tie_pct"].round(4)
    df.loc[rows_to_fill, "mc_f5_expected_total"] = f5_res["expected_total"].round(3)
    df.loc[rows_to_fill, "mc_f5_home_cover_pct"] = f5_res["home_cover_pct"].round(4)

    # ── Write back ────────────────────────────────────────────────────────
    df.to_parquet(matrix_path, engine="pyarrow", index=False)
    df.to_csv(str(matrix_path).replace(".parquet", ".csv"), index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    filled = df[rows_to_fill]
    print(f"\n  Backfill complete ({n_fill:,} games filled):")
    if not f5_only:
        mc_m = filled["mc_expected_runs"].mean()
        mc_s = filled["mc_expected_runs"].std()
        print(f"    mc_expected_runs      mean={mc_m:.2f}  std={mc_s:.2f}")
    print(f"    mc_f5_home_win_pct    mean={filled['mc_f5_home_win_pct'].mean():.3f}")
    print(f"    mc_f5_away_win_pct    mean={filled['mc_f5_away_win_pct'].mean():.3f}")
    print(f"    mc_f5_tie_pct         mean={filled['mc_f5_tie_pct'].mean():.3f}")
    print(f"    mc_f5_expected_total  mean={filled['mc_f5_expected_total'].mean():.2f}")
    print(f"    mc_f5_home_cover_pct  mean={filled['mc_f5_home_cover_pct'].mean():.3f}")

    still_nan_f5 = df[F5_SIM_COLS].isna().any(axis=1).sum()
    print(f"    Still NaN (F5): {still_nan_f5}")

    # Diagnostic CSV
    diag_cols = (
        ["game_date", "home_team", "away_team",
         "home_sp_xwoba_against", "away_sp_xwoba_against",
         "mc_expected_runs", "total_runs"]
        + F5_SIM_COLS
        + ["actual_f5_total" if "actual_f5_total" in df.columns else None]
    )
    diag_cols = [c for c in diag_cols if c and c in df.columns]
    df[rows_to_fill][diag_cols].to_csv("mc_backfill_diagnostics.csv", index=False)
    print(f"    Diagnostics -> mc_backfill_diagnostics.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill MC simulation features in feature_matrix.parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--matrix",   default="feature_matrix.parquet",
                        help="Feature matrix parquet path")
    parser.add_argument("--sims",     type=int, default=N_SIMS_DEFAULT,
                        help=f"MC simulations per game (default {N_SIMS_DEFAULT:,})")
    parser.add_argument("--f5-only",  action="store_true",
                        help="Only populate F5 columns (skip full-game mc_expected_runs)")
    parser.add_argument("--force",    action="store_true",
                        help="Re-run simulation even if columns already populated")
    args = parser.parse_args()

    print("=" * 60)
    print("  backfill_mc.py  — Historical MC Simulation (GPU v5.1)")
    print(f"  Backend : {'CuPy/RTX 5080' if _GPU else 'NumPy/CPU'}")
    print(f"  Sims    : {args.sims:,}")
    print(f"  Matrix  : {args.matrix}")
    print(f"  Mode    : {'F5 only' if args.f5_only else 'full-game + F5'}")
    print("=" * 60)

    backfill(
        Path(args.matrix),
        n_sims=args.sims,
        f5_only=args.f5_only,
        force=args.force,
    )

    print("\n  Done.")


if __name__ == "__main__":
    main()
