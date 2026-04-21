"""
dixon_coles_mlb.py
==================
JAX-optimized Dixon-Coles run-distribution model for MLB.

Per-game intensities:
    lambda_h = exp(alpha_home + delta_away + H)
    lambda_a = exp(alpha_away + delta_home)

Joint PMF (Dixon-Coles 1997):
    P(H=x, A=y) = Pois(x; lambda_h) * Pois(y; lambda_a) * tau(x, y; lambda_h, lambda_a, rho)

where tau is 1 everywhere except the four low-score cells:
    tau(0,0) = 1 - lambda_h * lambda_a * rho
    tau(0,1) = 1 + lambda_h * rho
    tau(1,0) = 1 + lambda_a * rho
    tau(1,1) = 1 - rho

MLB doesn't cluster at 0-0 like soccer, so rho typically fits very close to
zero; including it costs nothing and makes the class reusable for other
low-scoring sports.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from jax.scipy.special import gammaln

_EPS = 1e-12
_RHO_MAX = 0.20


# ---------------------------------------------------------------------------
# Core JAX kernels
# ---------------------------------------------------------------------------

def _log_poisson_pmf(k: jnp.ndarray, lam: jnp.ndarray) -> jnp.ndarray:
    lam_safe = jnp.maximum(lam, _EPS)
    return k * jnp.log(lam_safe) - lam_safe - gammaln(k + 1.0)


def _log_dc_tau(
    x: jnp.ndarray,
    y: jnp.ndarray,
    lam_h: jnp.ndarray,
    lam_a: jnp.ndarray,
    rho: jnp.ndarray,
) -> jnp.ndarray:
    tau_00 = 1.0 - lam_h * lam_a * rho
    tau_01 = 1.0 + lam_h * rho
    tau_10 = 1.0 + lam_a * rho
    tau_11 = 1.0 - rho

    is00 = (x == 0) & (y == 0)
    is01 = (x == 0) & (y == 1)
    is10 = (x == 1) & (y == 0)
    is11 = (x == 1) & (y == 1)

    tau = jnp.where(is00, tau_00,
          jnp.where(is01, tau_01,
          jnp.where(is10, tau_10,
          jnp.where(is11, tau_11, 1.0))))
    return jnp.log(jnp.maximum(tau, _EPS))


def _lambdas(params, idx_h, idx_a):
    att_h = params["attack"][idx_h]
    def_a = params["defense"][idx_a]
    att_a = params["attack"][idx_a]
    def_h = params["defense"][idx_h]
    hfa   = params["hfa"]
    lam_h = jnp.exp(att_h + def_a + hfa)
    lam_a = jnp.exp(att_a + def_h)
    return lam_h, lam_a


def _neg_log_lik(params, idx_h, idx_a, h_runs, a_runs, ridge):
    lam_h, lam_a = _lambdas(params, idx_h, idx_a)
    rho = _RHO_MAX * jnp.tanh(params["rho_raw"])

    ll = (_log_poisson_pmf(h_runs, lam_h)
          + _log_poisson_pmf(a_runs, lam_a)
          + _log_dc_tau(h_runs, a_runs, lam_h, lam_a, rho))

    penalty = ridge * (jnp.sum(params["attack"] ** 2)
                       + jnp.sum(params["defense"] ** 2))
    return -jnp.sum(ll) + penalty


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class DixonColesMLB:
    """
    JAX-optimized Dixon-Coles run-distribution model for MLB.

    Parameters
    ----------
    max_runs : int
        Truncation bound for the scoreline matrix (inclusive -> size N+1).
    ridge : float
        L2 shrinkage on team attack/defense coefficients.
    n_iter : int
        Adam iterations for parameter optimization.
    lr : float
        Adam learning rate.
    """

    def __init__(self, max_runs: int = 20, ridge: float = 1e-3,
                 n_iter: int = 3000, lr: float = 0.05):
        self.max_runs   = int(max_runs)
        self.ridge      = float(ridge)
        self.n_iter     = int(n_iter)
        self.lr         = float(lr)

        self.teams_: Optional[np.ndarray] = None
        self.team_idx_: Optional[dict] = None
        self.params_: Optional[dict] = None
        self._fit_history_: list[float] = []

    # -----------------------------------------------------------------

    def fit(self, df: pd.DataFrame, verbose: bool = False) -> "DixonColesMLB":
        required = {"home_team", "away_team", "home_runs", "away_runs"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DixonColesMLB.fit: missing columns {missing}")
        df = df.dropna(subset=list(required)).copy()

        teams = np.unique(np.concatenate([
            df["home_team"].to_numpy(), df["away_team"].to_numpy()
        ]))
        team_idx = {t: i for i, t in enumerate(teams)}
        N = len(teams)

        idx_h  = jnp.asarray(df["home_team"].map(team_idx).to_numpy(), dtype=jnp.int32)
        idx_a  = jnp.asarray(df["away_team"].map(team_idx).to_numpy(), dtype=jnp.int32)
        h_runs = jnp.asarray(df["home_runs"].to_numpy(), dtype=jnp.float32)
        a_runs = jnp.asarray(df["away_runs"].to_numpy(), dtype=jnp.float32)

        mean_h = float(df["home_runs"].mean())
        mean_a = float(df["away_runs"].mean())
        hfa0   = np.log(max(mean_h / max(mean_a, _EPS), _EPS)) * 0.5

        params = {
            "attack":  jnp.zeros(N, dtype=jnp.float32),
            "defense": jnp.zeros(N, dtype=jnp.float32),
            "hfa":     jnp.asarray(hfa0, dtype=jnp.float32),
            "rho_raw": jnp.asarray(0.0,  dtype=jnp.float32),
        }

        loss_and_grad = jax.jit(jax.value_and_grad(_neg_log_lik))
        opt = optax.adam(self.lr)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = loss_and_grad(params, idx_h, idx_a, h_runs, a_runs, self.ridge)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            params["attack"]  = params["attack"]  - jnp.mean(params["attack"])
            params["defense"] = params["defense"] - jnp.mean(params["defense"])
            return params, opt_state, loss

        self._fit_history_ = []
        for i in range(self.n_iter):
            params, opt_state, loss = step(params, opt_state)
            self._fit_history_.append(float(loss))
            if verbose and (i % max(1, self.n_iter // 10) == 0 or i == self.n_iter - 1):
                rho_now = float(_RHO_MAX * np.tanh(float(params["rho_raw"])))
                print(f"  DC iter {i:5d}  nll={float(loss):12.2f}  "
                      f"hfa={float(params['hfa']):+.4f}  rho={rho_now:+.4f}")

        self.teams_    = teams
        self.team_idx_ = team_idx
        self.params_   = {k: np.asarray(v) for k, v in params.items()}
        return self

    # -----------------------------------------------------------------

    def _require_fit(self) -> None:
        if self.params_ is None:
            raise RuntimeError("Model not fit. Call fit(df) first.")

    def lambdas(self, home_team: str, away_team: str) -> tuple[float, float]:
        self._require_fit()
        p = self.params_
        if home_team in self.team_idx_ and away_team in self.team_idx_:
            ih = self.team_idx_[home_team]
            ia = self.team_idx_[away_team]
            lam_h = float(np.exp(p["attack"][ih] + p["defense"][ia] + p["hfa"]))
            lam_a = float(np.exp(p["attack"][ia] + p["defense"][ih]))
            return lam_h, lam_a
        lam_h = float(np.exp(p["hfa"]))
        lam_a = 1.0
        return lam_h, lam_a

    def rho(self) -> float:
        self._require_fit()
        return float(_RHO_MAX * np.tanh(float(self.params_["rho_raw"])))

    # -----------------------------------------------------------------

    def predict_match_matrix(
        self,
        home_team: str,
        away_team: str,
        size: Optional[int] = None,
    ) -> np.ndarray:
        self._require_fit()
        N = int(size if size is not None else self.max_runs)
        lam_h, lam_a = self.lambdas(home_team, away_team)
        rho = self.rho()

        xs = np.arange(N + 1)
        log_ph = xs * np.log(max(lam_h, _EPS)) - lam_h - _np_gammaln(xs + 1)
        log_pa = xs * np.log(max(lam_a, _EPS)) - lam_a - _np_gammaln(xs + 1)
        log_joint = log_ph[:, None] + log_pa[None, :]
        M = np.exp(log_joint - log_joint.max())
        M = M / M.sum()

        M[0, 0] *= max(1.0 - lam_h * lam_a * rho, _EPS)
        M[0, 1] *= max(1.0 + lam_h * rho,         _EPS)
        M[1, 0] *= max(1.0 + lam_a * rho,         _EPS)
        M[1, 1] *= max(1.0 - rho,                 _EPS)
        M = M / M.sum()
        return M

    # -----------------------------------------------------------------

    def get_runline_prob(self, home_team: str, away_team: str, line: float = -1.5) -> float:
        """P(home covers). Strict inequality (home - away) > line."""
        M = self.predict_match_matrix(home_team, away_team)
        N = M.shape[0]
        xs = np.arange(N)[:, None]
        ys = np.arange(N)[None, :]
        cover_mask = (xs - ys) > line
        return float(M[cover_mask].sum())

    def get_total_prob(self, home_team: str, away_team: str, line: float) -> float:
        """P(total > line). Strict inequality."""
        M = self.predict_match_matrix(home_team, away_team)
        N = M.shape[0]
        xs = np.arange(N)[:, None]
        ys = np.arange(N)[None, :]
        over_mask = (xs + ys) > line
        return float(M[over_mask].sum())

    # -----------------------------------------------------------------

    def team_ratings(self) -> pd.DataFrame:
        self._require_fit()
        return (pd.DataFrame({
                    "team":    self.teams_,
                    "attack":  self.params_["attack"],
                    "defense": self.params_["defense"],
                })
                .sort_values("attack", ascending=False)
                .reset_index(drop=True))


# ---------------------------------------------------------------------------

def _np_gammaln(x):
    from scipy.special import gammaln as _g
    return _g(x)
