"""
score_props_today.py — Pitching K-Prop Engine.

E[K] = K% * TBF_est, where:
  K%       ≈ (K/9) / 38.5   (league-avg BF/9 proxy)
  TBF_est  = 23 base (typical SP faces ~23 batters in 5.5 IP)
             × 1.03 if wind_vector_out < -5.0 OR temp_f < 50°F
               (wind-in / cold suppresses offense → pitchers stay in longer)

Actionable when E[K] > (k_line + 0.5).

Sources:
  data/raw/fangraphs_pitchers.csv       K/9 per pitcher
  data/statcast/lineups_<date>.parquet  confirmed SPs + team abbrs
  data/statcast/k_props_<date>.parquet  market K lines per SP
  games.csv                             wind_vector_out, temp_f (per home team)

Output: list[dict] rows keyed for model_scores.csv schema, with
        model="PROP_K", bet_type=f"K {line} {pitcher}".

Entrypoint score_props(date_str) returns a DataFrame ready to concat onto
model_scores.csv. CLI runs stand-alone.
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

_ROOT = Path(__file__).resolve().parent
_PIPE = _ROOT  # repo root == pipeline dir
_DATA = _PIPE / "data"
_RAW  = _DATA / "raw"
_SC   = _DATA / "statcast"

LEAGUE_BF_PER_9   = 38.5   # league-avg batters faced per 9 IP
TBF_BASE          = 23.0   # typical starter batters faced
PHYSICS_MULT      = 1.03
WIND_IN_THRESHOLD = -5.0   # wind_vector_out (mph, CF-axis)
COLD_THRESHOLD    = 50.0   # °F
ACTIONABLE_PROB_EDGE = 0.01  # prob_over − implied_prob must exceed this (1pp)
PRICE_GUARD_ODDS     = -250  # reject over_odds worse than this (e.g. -260, -300)
DEFAULT_PITCH_LIMIT  = 92    # fallback when no per-start history exists
EARLY_EXIT_BATTER_THRESHOLD = 18  # < 5 IP proxy (≈18 BF including baserunners)

# ── Monte Carlo simulator constants ──────────────────────────────────────────
N_TRIALS            = 5_000
MAX_PITCHES         = 95
MAX_RUNS_ALLOWED    = 5
PITCH_COST_K        = 4.8    # avg pitches on a strikeout PA
PITCH_COST_BB       = 5.5    # avg pitches on a walk PA
PITCH_COST_BIP      = 3.2    # avg pitches on a ball-in-play PA
LEAGUE_K_PCT        = 0.222  # MLB league average strikeout rate
LEAGUE_BB_PCT       = 0.086  # MLB league average walk rate
RUN_PER_BIP         = 0.14   # crude scoring proxy (expected runs contribution)


def get_matchup_k_rate(pitcher_rate: float,
                       batter_rate:  float,
                       league_avg:   float = LEAGUE_K_PCT) -> float:
    """Log5 matchup probability.

        P = (Pp * Pb / L) /
            (Pp * Pb / L + (1-Pp) * (1-Pb) / (1-L))

    Bill James' Log5 form — unbiased fusion of two opposing talent rates
    through the league baseline. Used here for K-rate and (same form) BB-rate.
    """
    if pitcher_rate is None or batter_rate is None:
        return league_avg
    pp = float(pitcher_rate)
    pb = float(batter_rate)
    la = float(league_avg)
    # Clip to avoid zero-division / saturating endpoints.
    pp = min(max(pp, 1e-4), 1 - 1e-4)
    pb = min(max(pb, 1e-4), 1 - 1e-4)
    la = min(max(la, 1e-4), 1 - 1e-4)
    num = (pp * pb) / la
    den = num + ((1 - pp) * (1 - pb)) / (1 - la)
    return num / den


def simulate_game(pitcher_stats: dict,
                  lineup_stats:  list[dict],
                  weather_env:   dict,
                  n_trials:      int = N_TRIALS,
                  rng:           np.random.Generator | None = None,
                  pitch_limit:   int = MAX_PITCHES) -> tuple[np.ndarray, float]:
    """Monte-Carlo strikeout distribution for one start.

    pitcher_stats : {"k_pct": float, "bb_pct": float}
    lineup_stats  : list of 9 dicts, each {"k_pct": float, "bb_pct": float}
    weather_env   : {"temp_f": float|None, "wind_vector_out": float|None}
                    Triggers a small endurance boost: pitchers throw longer
                    in cold / wind-in conditions.

    Exit conditions per trial: pitches >= pitch_limit or runs >= MAX_RUNS_ALLOWED.
    Returns (sim_Ks, failure_rate) where failure_rate is the fraction of trials
    that exited before the 5th inning (batters_faced < EARLY_EXIT_BATTER_THRESHOLD).
    """
    if rng is None:
        rng = np.random.default_rng()
    if not lineup_stats:
        return np.zeros(n_trials, dtype=np.int32), 0.0

    # Pre-compute per-batter Log5 K and BB probs — static across trials.
    k_probs  = np.array([get_matchup_k_rate(pitcher_stats.get("k_pct"),
                                            b.get("k_pct"),
                                            LEAGUE_K_PCT) for b in lineup_stats])
    bb_probs = np.array([get_matchup_k_rate(pitcher_stats.get("bb_pct"),
                                            b.get("bb_pct"),
                                            LEAGUE_BB_PCT) for b in lineup_stats])
    # Guard: K + BB must leave room for BIP.
    overflow = (k_probs + bb_probs) >= 0.95
    if overflow.any():
        scale = 0.95 / (k_probs[overflow] + bb_probs[overflow])
        k_probs[overflow]  *= scale
        bb_probs[overflow] *= scale
    bip_probs = 1.0 - k_probs - bb_probs

    # Endurance boost in cold / wind-in — pitcher absorbs ~5% more pitches.
    temp = weather_env.get("temp_f")
    wvo  = weather_env.get("wind_vector_out")
    endurance_bonus = 1.0
    if (temp is not None and temp < COLD_THRESHOLD) or \
       (wvo  is not None and wvo  < WIND_IN_THRESHOLD):
        endurance_bonus = 1.05
    pitch_cap = int(pitch_limit * endurance_bonus)

    sim_Ks    = np.empty(n_trials, dtype=np.int32)
    fail_mask = np.zeros(n_trials, dtype=bool)
    N = len(lineup_stats)

    for t in range(n_trials):
        pitches = 0
        runs    = 0.0
        Ks      = 0
        i       = 0
        while pitches < pitch_cap and runs < MAX_RUNS_ALLOWED:
            idx = i % N
            r   = rng.random()
            if r < k_probs[idx]:
                Ks       += 1
                pitches  += PITCH_COST_K
            elif r < k_probs[idx] + bb_probs[idx]:
                pitches  += PITCH_COST_BB
                # Walk contributes partial run expectation.
                runs     += RUN_PER_BIP * 0.6
            else:
                pitches  += PITCH_COST_BIP
                # BIP run contribution — Bernoulli variance, ~league-avg rate.
                if rng.random() < RUN_PER_BIP:
                    runs += 1.0
            i += 1
        sim_Ks[t] = Ks
        # Failure = pulled (runs or pitch-cap) before the 5th inning.
        if i < EARLY_EXIT_BATTER_THRESHOLD:
            fail_mask[t] = True

    failure_rate = float(fail_mask.mean())
    return sim_Ks, failure_rate


def _american_to_prob(odds) -> float | None:
    """Convert American odds to implied probability (vig-included)."""
    if odds is None or pd.isna(odds):
        return None
    o = float(odds)
    if o > 0:
        return 100.0 / (o + 100.0)
    return abs(o) / (abs(o) + 100.0)


def _strip_accents(s: str) -> str:
    if not isinstance(s, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).strip().lower()


def _load_pitcher_k9() -> dict[str, float]:
    """Name (accent-normalized, lower) → K/9."""
    path = _RAW / "fangraphs_pitchers.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "Name" not in df.columns or "K/9" not in df.columns:
        return {}
    df["_key"] = df["Name"].map(_strip_accents)
    df = df.drop_duplicates("_key", keep="first")
    return dict(zip(df["_key"], pd.to_numeric(df["K/9"], errors="coerce")))


def _load_pitcher_rates() -> dict[str, dict]:
    """Name key → {"k_pct", "bb_pct"} from savant_pitchers (real rates)."""
    path = _RAW / "savant_pitchers.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "last_name, first_name" not in df.columns:
        return {}

    def _flip(n):
        # "Sánchez, Cristopher" → "Cristopher Sanchez" (accent-stripped, lower).
        if not isinstance(n, str) or "," not in n:
            return _strip_accents(n)
        last, first = [p.strip() for p in n.split(",", 1)]
        return _strip_accents(f"{first} {last}")

    df["_key"] = df["last_name, first_name"].map(_flip)
    df = df.drop_duplicates("_key", keep="first")
    return {
        k: {"k_pct":  float(row["k_percent"])  / 100.0 if pd.notna(row.get("k_percent"))  else None,
            "bb_pct": float(row["bb_percent"]) / 100.0 if pd.notna(row.get("bb_percent")) else None}
        for k, row in zip(df["_key"], df.to_dict("records"))
    }


def _load_team_lineup_rates() -> dict[str, dict]:
    """Team abbr → {"k_pct", "bb_pct"} averaged across vs-LHP and vs-RHP
    tables. Used as the 9-man lineup proxy until batting_order populates.
    """
    out: dict[str, dict] = {}
    frames: list[pd.DataFrame] = []
    for fname in ("fangraphs_team_vs_lhp.csv", "fangraphs_team_vs_rhp.csv"):
        p = _RAW / fname
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return out
    df = pd.concat(frames, ignore_index=True)
    if {"Tm", "K%", "BB%"} - set(df.columns):
        return out
    # K% / BB% in these files are already fractional (0.095 form).
    agg = (df.groupby("Tm", as_index=False)
             .agg(k_pct=("K%", "mean"), bb_pct=("BB%", "mean")))
    for _, r in agg.iterrows():
        out[r["Tm"]] = {"k_pct": float(r["k_pct"]), "bb_pct": float(r["bb_pct"])}
    return out


def _load_pitch_limits() -> dict[str, int]:
    """Name key → max pitches across last 3 starts (statcast 2026).

    Uses raw pitch-level statcast to count pitches per (pitcher, game_pk),
    sorts by game_date desc, keeps last 3 per pitcher, returns the MAX.
    player_name in statcast is 'Last, First' — flip to match our accent-lower key.
    """
    path = _SC / "statcast_2026.parquet"
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path, columns=["game_pk", "game_date", "pitcher", "player_name"])
    except Exception:
        return {}
    if df.empty or "player_name" not in df.columns:
        return {}

    counts = (df.groupby(["pitcher", "player_name", "game_pk", "game_date"])
                .size().reset_index(name="n_pitches"))
    counts["game_date"] = pd.to_datetime(counts["game_date"], errors="coerce")
    counts = counts.sort_values(["pitcher", "game_date"], ascending=[True, False])
    # Keep last 3 starts per pitcher, then aggregate MAX.
    last3 = counts.groupby("pitcher", as_index=False).head(3)
    agg = last3.groupby(["pitcher", "player_name"], as_index=False)["n_pitches"].max()

    def _flip(n):
        if not isinstance(n, str) or "," not in n:
            return _strip_accents(n)
        last, first = [p.strip() for p in n.split(",", 1)]
        return _strip_accents(f"{first} {last}")

    agg["_key"] = agg["player_name"].map(_flip)
    agg = agg.drop_duplicates("_key", keep="first")
    return dict(zip(agg["_key"], agg["n_pitches"].astype(int)))


def get_pitch_limit(pitcher_key: str, limit_map: dict[str, int]) -> int:
    """Max pitches over last 3 starts; fall back to DEFAULT_PITCH_LIMIT."""
    v = limit_map.get(pitcher_key)
    return int(v) if v is not None and v > 0 else DEFAULT_PITCH_LIMIT


def _load_k_lines(date_str: str) -> pd.DataFrame:
    """Median line + best over_odds per pitcher across books."""
    path = _SC / f"k_props_{date_str}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["pitcher_key", "k_line", "over_odds", "under_odds"])
    df = pd.read_parquet(path)
    df["pitcher_key"] = df["pitcher_name"].map(_strip_accents)
    agg = (df.groupby("pitcher_key", as_index=False)
             .agg(pitcher_name=("pitcher_name", "first"),
                  k_line=("line", "median"),
                  over_odds=("over_odds", "max"),     # best price for the bettor
                  under_odds=("under_odds", "max")))
    return agg


def _load_lineups(date_str: str) -> pd.DataFrame:
    path = _SC / f"lineups_{date_str}.parquet"
    if not path.exists():
        path = _SC / "lineups_today.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _load_weather_by_team() -> dict[str, tuple[float | None, float | None]]:
    """home_team (abbr + full name) → (wind_vector_out, temp_f)."""
    games_path = _PIPE / "games.csv"
    if not games_path.exists():
        return {}
    df = pd.read_csv(games_path)
    if "home_team" not in df.columns:
        return {}
    # games.csv carries full names; key both full + abbr so lineups (abbr) match.
    sys.path.insert(0, str(_PIPE / "wizard_agents"))
    try:
        from tools.implementations import TEAM_NAME_TO_ABBR
    except Exception:
        TEAM_NAME_TO_ABBR = {}
    out: dict[str, tuple] = {}
    for _, r in df.iterrows():
        h   = r.get("home_team")
        wvo = r.get("wind_vector_out")
        tmp = r.get("temp_f")
        wvo = float(wvo) if pd.notna(wvo) else None
        tmp = float(tmp) if pd.notna(tmp) else None
        out[h] = (wvo, tmp)
        abbr = TEAM_NAME_TO_ABBR.get(h)
        if abbr:
            out[abbr] = (wvo, tmp)
    return out


def score_props(date_str: str) -> pd.DataFrame:
    k9_map      = _load_pitcher_k9()
    k_lines     = _load_k_lines(date_str)
    lineups     = _load_lineups(date_str)
    wx_by_t     = _load_weather_by_team()
    sp_rates    = _load_pitcher_rates()
    team_rates  = _load_team_lineup_rates()
    pitch_lims  = _load_pitch_limits()
    rng         = np.random.default_rng(seed=20260423)
    n_price_excluded = 0

    if lineups.empty or k_lines.empty:
        print(f"[PropScorer] lineups={len(lineups)} k_lines={len(k_lines)} — nothing to score.")
        return pd.DataFrame()

    k_map = dict(zip(k_lines["pitcher_key"], k_lines.to_dict("records")))

    rows: list[dict] = []
    for _, g in lineups.iterrows():
        home_abbr = g.get("home_team")
        away_abbr = g.get("away_team")
        wvo, tmp = wx_by_t.get(home_abbr, (None, None))

        physics = (wvo is not None and wvo < WIND_IN_THRESHOLD) or \
                  (tmp is not None and tmp < COLD_THRESHOLD)
        tbf_est = TBF_BASE * (PHYSICS_MULT if physics else 1.0)

        for side, sp_col, opp_abbr in (
            ("home", "home_starter_name", away_abbr),
            ("away", "away_starter_name", home_abbr),
        ):
            sp_name = g.get(sp_col)
            if not sp_name or pd.isna(sp_name):
                continue
            key = _strip_accents(sp_name)
            k9  = k9_map.get(key)
            mkt = k_map.get(key)
            if k9 is None or mkt is None or pd.isna(k9) or pd.isna(mkt["k_line"]):
                continue

            # Poisson baseline (kept for audit / skew comparison).
            poisson_k_pct = float(k9) / LEAGUE_BF_PER_9
            lam_poisson   = poisson_k_pct * tbf_est
            k_line        = float(mkt["k_line"])
            prob_poisson  = float(1.0 - poisson.cdf(int(k_line), lam_poisson))

            # ── Monte Carlo engine ────────────────────────────────────────────
            pitcher = sp_rates.get(key, {
                "k_pct":  poisson_k_pct,               # fallback from K/9 proxy
                "bb_pct": LEAGUE_BB_PCT,
            })
            lineup_stats_team = team_rates.get(opp_abbr)
            if lineup_stats_team is None:
                # No team table hit — use league averages as the 9 batters.
                lineup_stats_team = {"k_pct": LEAGUE_K_PCT, "bb_pct": LEAGUE_BB_PCT}
            lineup = [lineup_stats_team] * 9
            weather_env = {"temp_f": tmp, "wind_vector_out": wvo}

            over_odds    = mkt.get("over_odds")

            # Price Guard — refuse to price anything worse than PRICE_GUARD_ODDS.
            if over_odds is not None and not pd.isna(over_odds) \
               and float(over_odds) < 0 and float(over_odds) < PRICE_GUARD_ODDS:
                n_price_excluded += 1
                print(f"[PropScorer] PRICE_EXCLUDED {sp_name} K{k_line} @ {int(over_odds)} "
                      f"(worse than {PRICE_GUARD_ODDS})")
                continue

            pitch_limit = get_pitch_limit(key, pitch_lims)
            sim_Ks, failure_rate = simulate_game(
                pitcher, lineup, weather_env, rng=rng, pitch_limit=pitch_limit,
            )
            sim_mean = float(sim_Ks.mean())
            prob_sim = float((sim_Ks > k_line).mean())

            implied_prob = _american_to_prob(over_odds)
            prob_edge    = (prob_sim - implied_prob) if implied_prob is not None else None
            actionable   = bool(prob_edge is not None and prob_edge > ACTIONABLE_PROB_EDGE)

            rows.append({
                "date":  date_str,
                "game":  f"{away_abbr} @ {home_abbr}",
                "model": "PROP_K",
                "bet_type": f"K {k_line} {sp_name}",
                "pick_direction": "OVER",
                # model_prob = Monte-Carlo P(K > line) over N_TRIALS trials.
                "model_prob":  round(prob_sim, 4),
                "P_true":      None,
                "Retail_Implied_Prob": round(implied_prob, 4) if implied_prob is not None else None,
                "edge":        round(prob_edge, 4) if prob_edge is not None else None,
                "retail_american_odds": over_odds,
                "sanity_check_pass": actionable,
                "odds_floor_pass":   True,
                "tier":  1 if actionable else None,
                "dollar_stake":      None,
                "actionable":        actionable,
                "home_bullpen_xfip": None,
                "away_bullpen_xfip": None,
                "bullpen_xfip_diff": None,
                "signal_flags": "WEATHER_K_ADJ" if physics else "",
                # audit cols — Poisson baseline for drift/skew comparison
                "k_lambda":      round(lam_poisson, 3),   # Poisson E[K]
                "sim_mean_k":    round(sim_mean,    3),   # MC E[K]
                "prob_poisson":  round(prob_poisson, 4),  # Poisson P(K>line)
                "prob_mc":       round(prob_sim,    4),   # same as model_prob
                "pitch_limit":   int(pitch_limit),
                "failure_rate":  round(failure_rate, 4),
            })

    df = pd.DataFrame(rows)
    n_act = int(df["actionable"].sum()) if not df.empty else 0
    print(f"[PropScorer] MC-scored {len(df)} K-props | actionable={n_act} "
          f"| price_excluded={n_price_excluded} "
          f"(prob_edge > {ACTIONABLE_PROB_EDGE*100:.0f}pp | N_TRIALS={N_TRIALS} | "
          f"price_guard<{PRICE_GUARD_ODDS} | "
          f"physics_mult on wvo<{WIND_IN_THRESHOLD} or temp<{COLD_THRESHOLD}°F)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    args = ap.parse_args()
    date_str = args.date or date.today().isoformat()
    df = score_props(date_str)
    if not df.empty:
        print(df[["game", "bet_type", "model_prob", "edge", "actionable"]].to_string(index=False))


if __name__ == "__main__":
    main()
