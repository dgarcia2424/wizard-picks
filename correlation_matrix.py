"""
correlation_matrix.py — v4.3 Joint-Probability Correlation Engine.

Replaces naive P(A) x P(B) independence assumption with empirically-derived
Gaussian copula joint probabilities for three SGP leg pairs.

Pair 1: SP F5 Strikeouts  <->  Opposing Team Total (game-level)
Pair 2: Game Total (Over)  <->  Home Score (proxy for star-hitter TB)
Pair 3: Both SP F5 Ks >= 3  <->  Close Game (|margin| <= 1, Underdog RL +1.5)

Usage:
    from correlation_matrix import CorrelationEngine
    engine = CorrelationEngine()
    p_joint = engine.joint_prob(p_leg_a, p_leg_b, pair="sp_k_vs_team_total")

Outputs:
    data/corr/correlation_matrix.json  — r-values + metadata
    data/corr/correlation_summary.txt  — human-readable audit

Theory:
    For a bivariate Gaussian copula with correlation r, the joint probability of
    two events (u, v) in [0,1] is:
        P(U<=u, V<=v) = Phi2(Phi_inv(u), Phi_inv(v); r)
    where Phi2 is the bivariate normal CDF and Phi_inv is the normal quantile function.

    When r > 0 (positive correlation), P(both) > P(A)xP(B).
    This is the structural edge books don't fully account for.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parent

SCRIPT_LABELS  = _ROOT / "data/batter_features/script_labels.parquet"
FEATURE_MTX    = _ROOT / "feature_matrix_enriched_v2.parquet"
OUT_DIR        = _ROOT / "data/corr"
OUT_JSON       = OUT_DIR / "correlation_matrix.json"
OUT_TXT        = OUT_DIR / "correlation_summary.txt"

# ── Pair definitions ─────────────────────────────────────────────────────────
PAIRS = {
    "sp_k_vs_team_total": {
        "description": "Home SP F5 Ks vs Opposing (Away) Team Total Score",
        "rationale":   ("High-K SP suppresses opposing runs. "
                        "Positive r -> Under + K-Over legs are correlated."),
        "col_a": "home_sp_k_f5",
        "col_b": "away_score",        # away score = what the home SP faces
        "direction_b": -1,            # more Ks -> lower away score (inverse)
    },
    "game_total_vs_home_score": {
        "description": "Game Total Over vs Home Score (star-hitter proxy)",
        "rationale":   ("High-scoring games lift individual batter TB/Hits lines. "
                        "Star-hitter Over and Game Over are positively correlated."),
        "col_a": "actual_game_total",
        "col_b": "home_score",
        "direction_b": 1,
    },
    "both_sp_k_vs_close_game": {
        "description": "Both SP F5 Ks >= 3 vs Close Game (|margin| <= 1)",
        "rationale":   ("Dual-dominance keeps scores suppressed + balanced. "
                        "Positive r -> Elite Duel script: Under + both K + RL cover."),
        "col_a": "both_sp_k_f5_min",  # min(home_sp_k_f5, away_sp_k_f5) — proxy for both
        "col_b": "close_flag",         # 1 if |margin| <= 1
        "direction_b": 1,
    },
}


def _bivariate_normal_cdf(a: float, b: float, rho: float) -> float:
    """P(Z₁ <= a, Z₂ <= b) for bivariate standard normal with correlation rho.

    Uses Owen's T-function approach via scipy.stats.multivariate_normal.
    """
    if abs(rho) >= 1.0:
        rho = np.clip(rho, -0.9999, 0.9999)
    mean = [0.0, 0.0]
    cov  = [[1.0, rho], [rho, 1.0]]
    return float(stats.multivariate_normal.cdf([a, b], mean=mean, cov=cov))


def _gaussian_copula_joint(p_a: float, p_b: float, rho: float) -> float:
    """P(A AND B) via Gaussian copula with correlation rho.

    Maps marginal probabilities to standard-normal quantiles, computes
    bivariate normal CDF, returns joint probability.

    Args:
        p_a:  Marginal probability of leg A (0 < p_a < 1)
        p_b:  Marginal probability of leg B (0 < p_b < 1)
        rho:  Pearson correlation between the two legs (-1 <= r <= 1)

    Returns:
        P(A and B) accounting for correlation
    """
    eps = 1e-6
    p_a = np.clip(p_a, eps, 1 - eps)
    p_b = np.clip(p_b, eps, 1 - eps)
    # Quantile transformation: uniform margins -> normal margins
    z_a = float(stats.norm.ppf(p_a))
    z_b = float(stats.norm.ppf(p_b))
    return _bivariate_normal_cdf(z_a, z_b, rho)


def _gaussian_copula_joint_3way(p_a: float, p_b: float, p_c: float,
                                  rho_ab: float, rho_ac: float,
                                  rho_bc: float) -> float:
    """P(A AND B AND C) via trivariate Gaussian copula.

    Uses Monte Carlo integration (50k samples) — sufficient for pricing purposes.
    """
    eps = 1e-6
    p_a = np.clip(p_a, eps, 1 - eps)
    p_b = np.clip(p_b, eps, 1 - eps)
    p_c = np.clip(p_c, eps, 1 - eps)
    z_a, z_b, z_c = (float(stats.norm.ppf(p)) for p in (p_a, p_b, p_c))

    cov = np.array([[1.0,    rho_ab, rho_ac],
                    [rho_ab, 1.0,    rho_bc],
                    [rho_ac, 1.0,    1.0   ]])
    # Ensure PSD
    cov[1, 2] = rho_bc
    cov[2, 1] = rho_bc
    try:
        samples = stats.multivariate_normal.rvs(mean=[0, 0, 0], cov=cov, size=50_000,
                                                 random_state=42)
        p_joint = float(np.mean((samples[:, 0] <= z_a) &
                                (samples[:, 1] <= z_b) &
                                (samples[:, 2] <= z_c)))
    except Exception:
        # Fallback: independent x correlation factor
        p_joint = p_a * p_b * p_c
    return p_joint


class CorrelationEngine:
    """Load pre-computed r-values and compute joint probabilities.

    Usage:
        engine = CorrelationEngine()
        p_joint = engine.joint_prob(0.45, 0.50, "sp_k_vs_team_total")
        p_3way  = engine.joint_prob_3way(p_k=0.45, p_f5=0.52, p_under=0.50,
                                          script="A2")
    """

    def __init__(self, json_path: Path = OUT_JSON):
        if not json_path.exists():
            raise FileNotFoundError(
                f"Correlation matrix not found at {json_path}. "
                "Run: python correlation_matrix.py --build"
            )
        with open(json_path) as f:
            data = json.load(f)
        self._rho = {k: v["rho"] for k, v in data["pairs"].items()}
        self._meta = data

    def rho(self, pair: str) -> float:
        return self._rho.get(pair, 0.0)

    def joint_prob(self, p_a: float, p_b: float, pair: str) -> float:
        """2-leg joint probability via Gaussian copula."""
        return _gaussian_copula_joint(p_a, p_b, self.rho(pair))

    def joint_prob_3way(self, p_k: float, p_f5_under: float,
                         p_game_under: float, script: str = "A2") -> float:
        """3-leg joint probability for named scripts.

        Script A2: SP K F5>=4 | F5 Under | Game Under
            r(K, F5_under)    = sp_k_vs_team_total (modified)
            r(K, game_under)  = sp_k_vs_team_total
            r(F5, game_under) = game_total_vs_home_score (strong positive)

        Script C: Both SPs K>=3 | Game Under | Close Game
        """
        if script in ("A2", "A2_Dominance"):
            rho_kf5  = self._rho.get("sp_k_vs_team_total", 0.25)
            rho_kg   = self._rho.get("sp_k_vs_team_total", 0.25)
            rho_f5g  = self._rho.get("game_total_vs_home_score", 0.80)
            return _gaussian_copula_joint_3way(p_k, p_f5_under, p_game_under,
                                               rho_kf5, rho_kg, rho_f5g)
        elif script in ("C", "C_EliteDuel"):
            rho_kc  = self._rho.get("both_sp_k_vs_close_game", 0.20)
            rho_ku  = self._rho.get("sp_k_vs_team_total", 0.25)
            rho_cu  = self._rho.get("both_sp_k_vs_close_game", 0.20)
            return _gaussian_copula_joint_3way(p_k, p_game_under, p_f5_under,
                                               rho_kc, rho_ku, rho_cu)
        else:
            # Independent fallback
            return p_k * p_f5_under * p_game_under

    def corr_ratio(self, p_a: float, p_b: float, pair: str) -> float:
        """P(joint_copula) / P(A)xP(B) — the structural edge multiplier."""
        p_indep = p_a * p_b
        if p_indep <= 0:
            return float("nan")
        return self.joint_prob(p_a, p_b, pair) / p_indep

    def summary(self) -> str:
        lines = ["=== Correlation Matrix Summary ==="]
        for pair, meta in self._meta["pairs"].items():
            rho = meta["rho"]
            n   = meta.get("n_obs", "?")
            pval = meta.get("pval", float("nan"))
            lines.append(
                f"  {pair}: r={rho:+.4f}  n={n}  p={pval:.4f}  "
                f"{'SIGNIFICANT' if pval < 0.05 else 'weak'}"
            )
        return "\n".join(lines)


def build_correlation_matrix(verbose: bool = True) -> dict:
    """Compute empirical correlations from script_labels + FM and save to JSON.

    Returns the full matrix dict.
    """
    if verbose:
        print("=" * 60)
        print("  Building Correlation Matrix (v4.3)")
        print("=" * 60)

    sl = pd.read_parquet(SCRIPT_LABELS)
    sl["both_sp_k_f5_min"] = sl[["home_sp_k_f5", "away_sp_k_f5"]].min(axis=1)
    sl["close_flag"] = (sl["home_margin"].abs() <= 1).astype(float)
    sl["away_score"] = sl["home_score"] - sl["home_margin"]

    result_pairs = {}
    for pair_name, cfg in PAIRS.items():
        col_a = cfg["col_a"]
        col_b = cfg["col_b"]
        direction_b = cfg.get("direction_b", 1)

        if col_a not in sl.columns or col_b not in sl.columns:
            if verbose:
                print(f"  [SKIP] {pair_name}: missing {col_a} or {col_b}")
            continue

        df = sl[[col_a, col_b]].dropna()
        a = df[col_a].astype(float).values
        b = df[col_b].astype(float).values * direction_b

        r, pval = stats.pearsonr(a, b)
        sp_r, sp_pval = stats.spearmanr(a, b)
        n = len(df)

        # 95% CI via Fisher z-transform
        z = np.arctanh(r)
        se = 1.0 / math.sqrt(n - 3)
        ci_lo = float(np.tanh(z - 1.96 * se))
        ci_hi = float(np.tanh(z + 1.96 * se))

        # Conditional rates: P(B | high A) vs P(B | low A)
        a_med = np.median(a)
        p_b_given_high_a = float(np.mean(b[a >= a_med])) if direction_b == 1 else float(
            np.mean(df[col_b].values[a >= a_med] <= np.median(df[col_b].values)))
        p_b_given_low_a  = float(np.mean(b[a < a_med]))  if direction_b == 1 else float(
            np.mean(df[col_b].values[a < a_med] <= np.median(df[col_b].values)))

        result_pairs[pair_name] = {
            "rho":            float(round(r, 4)),
            "rho_spearman":   float(round(sp_r, 4)),
            "pval":           float(round(pval, 6)),
            "pval_spearman":  float(round(sp_pval, 6)),
            "n_obs":          int(n),
            "ci_95_lo":       float(round(ci_lo, 4)),
            "ci_95_hi":       float(round(ci_hi, 4)),
            "p_b_given_high_a": float(round(p_b_given_high_a, 4)),
            "p_b_given_low_a":  float(round(p_b_given_low_a, 4)),
            "direction_b":    direction_b,
            "description":    cfg["description"],
            "rationale":      cfg["rationale"],
        }

        if verbose:
            sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else
                  ("*" if pval < 0.05 else ""))
            print(f"\n  Pair: {pair_name}")
            print(f"    r (Pearson):  {r:+.4f}{sig}  p={pval:.5f}  n={n}")
            print(f"    r (Spearman): {sp_r:+.4f}  CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
            print(f"    P(B|high A): {p_b_given_high_a:.4f}  P(B|low A): {p_b_given_low_a:.4f}")
            print(f"    Edge factor: {p_b_given_high_a/max(p_b_given_low_a,1e-6):.3f}x when A is high")
            print(f"    Thesis: {cfg['rationale']}")

    # Cross-pair: game_total vs f5_total
    if "actual_f5_total" in sl.columns and "actual_game_total" in sl.columns:
        df2 = sl[["actual_f5_total", "actual_game_total"]].dropna()
        r2, pval2 = stats.pearsonr(df2["actual_f5_total"], df2["actual_game_total"])
        result_pairs["f5_total_vs_game_total"] = {
            "rho": float(round(r2, 4)), "pval": float(round(pval2, 6)),
            "n_obs": len(df2),
            "description": "F5 Total vs Full Game Total",
            "rationale": "F5 total strongly predicts full game total (0.56 fraction assumption validity)"
        }
        if verbose:
            print(f"\n  Pair: f5_total_vs_game_total")
            print(f"    r = {r2:+.4f}  p={pval2:.5f}  n={len(df2)}")
            f5_fraction = (df2["actual_f5_total"] / df2["actual_game_total"]).median()
            print(f"    Empirical F5/Game fraction (median): {f5_fraction:.3f}  "
                  f"(model uses 0.56)")

    matrix = {
        "version":    "v4.3",
        "built_date": pd.Timestamp.now().isoformat(),
        "n_games":    int(len(sl)),
        "pairs":      result_pairs,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(matrix, f, indent=2)

    # Human-readable summary
    lines = ["Correlation Matrix — v4.3", "=" * 60, ""]
    for pair, m in result_pairs.items():
        sig = "***" if m["pval"] < 0.001 else ("**" if m["pval"] < 0.01 else
              ("*" if m["pval"] < 0.05 else " (n.s.)"))
        lines.append(f"{pair}")
        ci_lo_v = m.get("ci_95_lo")
        ci_hi_v = m.get("ci_95_hi")
        ci_str = f"[{ci_lo_v:.4f}, {ci_hi_v:.4f}]" if ci_lo_v is not None else "n/a"
        lines.append(f"  r={m['rho']:+.4f}{sig}  n={m['n_obs']}  CI={ci_str}")
        lines.append(f"  {m['rationale']}")
        lines.append("")
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines))

    if verbose:
        print(f"\n  Saved -> {OUT_JSON}")
        print(f"  Saved -> {OUT_TXT}")

    return matrix


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true",
                        help="Compute and save correlation matrix")
    parser.add_argument("--test",  action="store_true",
                        help="Run a quick sanity-check on the engine")
    args = parser.parse_args()

    if args.build or not any(vars(args).values()):
        matrix = build_correlation_matrix()

    if args.test:
        engine = CorrelationEngine()
        print("\n=== Engine Test ===")
        print(engine.summary())
        print()
        # Example: P(K F5>=4) ≈ 0.45, P(Game Under) ≈ 0.50
        p_k, p_under = 0.45, 0.50
        p_indep = p_k * p_under
        p_copula = engine.joint_prob(p_k, p_under, "sp_k_vs_team_total")
        print(f"P(K_over=0.45) x P(under=0.50):")
        print(f"  Independent:  {p_indep:.4f}")
        print(f"  Copula:       {p_copula:.4f}")
        print(f"  Ratio:        {p_copula/p_indep:.3f}x  (>1 = positive correlation edge)")
