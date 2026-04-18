"""
backfill_mc.py
==============
Retroactively populate mc_expected_runs in feature_matrix.parquet for all
historical training rows (2023-2025) where the value is currently NaN.

At inference time run_today.py passes mc_expected_runs from the live Monte
Carlo engine into the XGBoost feature vector.  During training this column is
NaN because the MC runs are done at prediction time, not at matrix-build time.
This gap means XGBoost never learns the residual between its own output and
the physics-based MC estimate — a potentially large signal.

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
the RTX 5080 via CuPy.  With 50,000 sims × 2,000 games that is 100 M Poisson
draws in a single kernel launch rather than a Python for-loop.

The same Poisson-LogNormal copula used in the live engine is applied:
  • Overdispersion: V ~ LogNormal(0, σ_nb)  →  Var(X) ≈ 9.9  (vs Poisson 4.4)
  • Run correlation: Cholesky-correlated log-normal mixing with ρ_copula = 0.14
    → Corr(H,A) ≈ 0.07 (matches empirical run-pair correlation)
  • xwOBA → runs:  R_conceded ≈ XWOBA_INTERCEPT + XWOBA_SLOPE × pitcher_xwOBA

Output
------
  Overwrites feature_matrix.parquet with mc_expected_runs populated.
  Also saves a diagnostic CSV: mc_backfill_diagnostics.csv

Usage
-----
  python backfill_mc.py
  python backfill_mc.py --sims 50000     # default
  python backfill_mc.py --matrix feature_matrix.parquet
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

N_SIMS_DEFAULT = 50_000   # scale to RTX 5080 — 100 M draws per batch


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

def sim_batch_expected_runs(
    mu_h: np.ndarray,          # [N_GAMES] home-offense expected runs/game
    mu_a: np.ndarray,          # [N_GAMES] away-offense expected runs/game
    n_sims: int = N_SIMS_DEFAULT,
    sigma: float = RUN_SIGMA_NB,
    rho: float = RUN_RHO_COPULA,
) -> np.ndarray:               # returns [N_GAMES] expected total runs
    """
    Simulate all games simultaneously on the GPU with the Poisson-LogNormal
    copula model.

    Tensor shapes:
      z1, z2, eps_h, eps_a, V_h, V_a  →  [N_GAMES, N_SIMS]
      home_runs, away_runs             →  [N_GAMES, N_SIMS]
      expected_total                   →  [N_GAMES]

    Memory estimate (float32):
      6 × N_GAMES × N_SIMS × 4 bytes
      = 6 × 2000 × 50000 × 4 ≈ 2.4 GB  (well within RTX 5080's 16 GB)
    """
    N = len(mu_h)

    # Transfer offense lambdas to GPU — shape [N, 1] for broadcasting
    mu_h_gpu = cp.asarray(mu_h[:, None], dtype=cp.float32)   # [N, 1]
    mu_a_gpu = cp.asarray(mu_a[:, None], dtype=cp.float32)   # [N, 1]

    # ── Cholesky-correlated standard normals ─────────────────────────────
    # L = [[1, 0], [ρ, √(1-ρ²)]]  →  eps_h = z1,  eps_a = ρ·z1 + √(1-ρ²)·z2
    rho_sqrt = float(np.sqrt(max(0.0, 1.0 - rho ** 2)))
    z1 = cp.random.standard_normal((N, n_sims)).astype(cp.float32)
    z2 = cp.random.standard_normal((N, n_sims)).astype(cp.float32)
    eps_h = z1                                          # [N, N_SIMS]
    eps_a = cp.float32(rho) * z1 + cp.float32(rho_sqrt) * z2  # [N, N_SIMS]

    # ── Log-normal mixing: V ~ LogNormal(−σ²/2, σ²) so E[V] = 1 ─────────
    half_s2 = cp.float32(0.5 * sigma * sigma)
    sig_f32  = cp.float32(sigma)
    V_h = cp.exp(sig_f32 * eps_h - half_s2)            # [N, N_SIMS]
    V_a = cp.exp(sig_f32 * eps_a - half_s2)            # [N, N_SIMS]

    # ── Compound Poisson draws ────────────────────────────────────────────
    # H|V_h ~ Poisson(μ_h · V_h);  A|V_a ~ Poisson(μ_a · V_a)
    home_runs = cp.random.poisson(mu_h_gpu * V_h).astype(cp.float32)  # [N, N_SIMS]
    away_runs = cp.random.poisson(mu_a_gpu * V_a).astype(cp.float32)  # [N, N_SIMS]

    # ── Expected total runs per game ──────────────────────────────────────
    expected = (home_runs + away_runs).mean(axis=1)     # [N_GAMES]

    return cp.asnumpy(expected) if _GPU else np.asarray(expected)


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

def backfill(
    matrix_path: Path = Path("feature_matrix.parquet"),
    n_sims: int = N_SIMS_DEFAULT,
    verbose: bool = True,
) -> None:
    """
    Populate mc_expected_runs in the feature matrix for all training rows.
    Uses a single vectorised GPU launch — no Python for-loop over games.
    """
    if not matrix_path.exists():
        print(f"  ERROR: {matrix_path} not found")
        return

    df      = pd.read_parquet(matrix_path, engine="pyarrow")
    n_total = len(df)
    n_nan   = df["mc_expected_runs"].isna().sum()

    print(f"  Matrix: {n_total} rows | mc_expected_runs filled: {n_total - n_nan} | NaN: {n_nan}")

    if n_nan == 0:
        print("  Nothing to backfill — all rows already have mc_expected_runs.")
        return

    rows_to_fill = df["mc_expected_runs"].isna()
    targets      = df[rows_to_fill].copy()

    if verbose:
        device_str = f"GPU (RTX 5080 / CuPy)" if _GPU else "CPU (CuPy not available)"
        print(f"  Device : {device_str}")
        print(f"  Running {n_sims:,} sims × {len(targets):,} games "
              f"= {n_sims * len(targets):,} total Poisson draws ...")

    # ── Vectorise all inputs ──────────────────────────────────────────────

    def _safe_xwoba(col):
        v = pd.to_numeric(targets.get(col, np.nan), errors="coerce").values
        return np.where(np.isnan(v), LEAGUE_XWOBA, v).astype(np.float64)

    def _whip_col_to_xwoba(col):
        vals = pd.to_numeric(targets.get(col, np.nan), errors="coerce").values
        return np.array([whip_to_xwoba(w) for w in vals], dtype=np.float64)

    home_sp_x  = _safe_xwoba("home_sp_xwoba_against")
    away_sp_x  = _safe_xwoba("away_sp_xwoba_against")
    home_bp_x  = _whip_col_to_xwoba("home_bp_whip")
    away_bp_x  = _whip_col_to_xwoba("away_bp_whip")

    # Altitude multipliers for every home team
    home_teams = targets.get("home_team", pd.Series(["NYY"] * len(targets))).values
    alt_mults  = np.array(
        [air_density_ratio(STADIUM_ELEVATION.get(str(t), 0)) for t in home_teams],
        dtype=np.float64,
    )

    # Per-team blended offense lambdas [N_GAMES]
    sp_frac = INNINGS_SP / INNINGS_GAME
    bp_frac = 1.0 - sp_frac

    home_sp_r9 = np.clip(XWOBA_INTERCEPT + XWOBA_SLOPE * home_sp_x, 1.5, 10.0)
    away_sp_r9 = np.clip(XWOBA_INTERCEPT + XWOBA_SLOPE * away_sp_x, 1.5, 10.0)
    home_bp_r9 = np.clip(XWOBA_INTERCEPT + XWOBA_SLOPE * home_bp_x, 1.5,  8.0)
    away_bp_r9 = np.clip(XWOBA_INTERCEPT + XWOBA_SLOPE * away_bp_x, 1.5,  8.0)

    mu_h = (home_sp_r9 * sp_frac + home_bp_r9 * bp_frac) * alt_mults
    mu_a = (away_sp_r9 * sp_frac + away_bp_r9 * bp_frac) * alt_mults

    # ── Single GPU batch launch ───────────────────────────────────────────
    mc_expected = sim_batch_expected_runs(mu_h, mu_a, n_sims=n_sims)   # [N_GAMES]

    # ── Write back to DataFrame ───────────────────────────────────────────
    df.loc[rows_to_fill, "mc_expected_runs"] = mc_expected

    df.to_parquet(matrix_path, engine="pyarrow", index=False)
    df.to_csv(str(matrix_path).replace(".parquet", ".csv"), index=False)

    filled_now = df["mc_expected_runs"].notna().sum()
    still_nan  = df["mc_expected_runs"].isna().sum()
    mc_mean    = df.loc[rows_to_fill, "mc_expected_runs"].mean()
    mc_std     = df.loc[rows_to_fill, "mc_expected_runs"].std()

    print(f"\n  Backfill complete:")
    print(f"    Filled:    {len(mc_expected):,} games")
    print(f"    Still NaN: {still_nan} (should be 0 except 2026 held set)")
    print(f"    mc_expected_runs: mean={mc_mean:.2f}  std={mc_std:.2f}")

    # Diagnostic CSV
    diag = df[rows_to_fill][["game_date", "home_team", "away_team",
                              "home_sp_xwoba_against", "away_sp_xwoba_against",
                              "mc_expected_runs", "total_runs"]].copy()
    diag.to_csv("mc_backfill_diagnostics.csv", index=False)
    print(f"    Diagnostics -> mc_backfill_diagnostics.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill mc_expected_runs in feature_matrix.parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--matrix", default="feature_matrix.parquet")
    parser.add_argument("--sims",   type=int, default=N_SIMS_DEFAULT,
                        help=f"MC simulations per game (default {N_SIMS_DEFAULT:,})")
    args = parser.parse_args()

    print("=" * 60)
    print("  backfill_mc.py  — Historical MC Expected Runs (GPU v5.0)")
    print(f"  Backend : {'CuPy/RTX 5080' if _GPU else 'NumPy/CPU'}")
    print(f"  Sims    : {args.sims:,}")
    print("=" * 60)

    backfill(Path(args.matrix), n_sims=args.sims)

    print("\n  Done.")


if __name__ == "__main__":
    main()
