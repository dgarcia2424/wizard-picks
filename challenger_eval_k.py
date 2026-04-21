"""
challenger_eval_k.py
=====================
Evaluate external challenger K-prop projection systems against our model on
completed 2026 games.

Challenger systems (loaded from fg_proj_*_2026.parquet):
  - THE BAT X  — Statcast-integrated regression (most reactive)
  - ATC        — Smart aggregate (lowest RMSE baseline)
  - ZiPS       — Cohort/aging-curve model (sticky, slow on breakouts)
  - Steamer    — Additional aggregate baseline

Pitch-quality challengers (loaded from fg_stuff_plus_2026.parquet):
  - Stuff+     — Physical pitch quality -> implied K%
  - Pitching+  — Stuff+ + Location+ composite -> implied K%

Our model:
  - daily_cards/ CSVs  -> mc_home_sp_k_mean / mc_away_sp_k_mean (pre-game μ)
  - pitcher_profiles_2026.parquet -> blended_k_pct (season-to-date EWMA)

Metrics reported per system:
  - MAE   — mean absolute error on raw K count (lower = better)
  - RMSE  — root-mean-square error on K count
  - Brier — Brier score at 4.5 K line (lower = better)
  - AUC   — ROC AUC at 4.5 line (over = 1, higher = better)
  - Acc   — accuracy of directional pick at 4.5 line
  - N     — number of games evaluated

Usage:
  python challenger_eval_k.py                   # all systems, full season
  python challenger_eval_k.py --since 2026-04-01
  python challenger_eval_k.py --line 5.5        # evaluate at a different line
  python challenger_eval_k.py --systems batx atc our_model
  python challenger_eval_k.py --save            # save results to CSV
"""

import argparse
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR    = Path("./data/statcast")
CARDS_DIR   = Path("./daily_cards")
RESULTS_CSV = Path("challenger_eval_k_results.csv")

# Must match monte_carlo_runline.py
KPROP_NB_R      = 20.0
KPROP_MEAN_CALIB = 0.899
_AVG_BF_PER_IP   = 4.35
N_SIMS           = 100_000

# Lines to evaluate
EVAL_LINES = [3.5, 4.5, 5.5, 6.5]
PRIMARY_LINE = 4.5


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _norm_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.strip()
    if "," in name:
        last, first = [p.strip() for p in name.split(",", 1)]
        name = f"{first} {last}"
    name = name.upper()
    return "".join(
        c for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    )


def _nb_over_prob(mu_k: float, line: float, n_sims: int = N_SIMS) -> float:
    """P(K > line) under Negative Binomial (Gamma-Poisson) with KPROP_MEAN_CALIB."""
    mu_k = max(float(mu_k) * KPROP_MEAN_CALIB, 0.01)
    scale = mu_k / KPROP_NB_R
    lam = np.random.gamma(KPROP_NB_R, scale, size=n_sims)
    k_sims = np.random.poisson(lam)
    return float(np.mean(k_sims > line))


def _brier(probs: np.ndarray, actuals: np.ndarray) -> float:
    return float(np.mean((probs - actuals) ** 2))


def _auc(probs: np.ndarray, actuals: np.ndarray) -> float:
    """Wilcoxon-Mann-Whitney AUC — no sklearn needed."""
    if len(np.unique(actuals)) < 2:
        return float("nan")
    pos = probs[actuals == 1]
    neg = probs[actuals == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Vectorised concordance count
    concordant = np.sum(pos[:, None] > neg[None, :])
    tied       = np.sum(pos[:, None] == neg[None, :])
    return float((concordant + 0.5 * tied) / (len(pos) * len(neg)))


# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------

def load_actuals(since: str | None = None) -> pd.DataFrame:
    """Load completed game actuals with starter K counts."""
    path = DATA_DIR / "actuals_2026.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Actuals not found: {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
    if since:
        df = df[df["game_date"] >= since]
    # Drop rows without K data
    df = df.dropna(subset=["home_sp_k", "away_sp_k", "home_sp_ip", "away_sp_ip"])
    return df.reset_index(drop=True)


def load_daily_cards() -> pd.DataFrame:
    """Load all daily cards to extract pitcher names and our model's K predictions."""
    frames = []
    for path in sorted(CARDS_DIR.glob("daily_card_2026-*.csv")):
        try:
            df = pd.read_csv(path)
            date_str = path.stem.replace("daily_card_", "")
            df["card_date"] = date_str
            frames.append(df)
        except Exception as e:
            print(f"  [WARN] Could not load {path.name}: {e}")
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined


def load_pitcher_profiles() -> pd.DataFrame:
    """Load 2026 pitcher profiles with blended_k_pct."""
    path = DATA_DIR / "pitcher_profiles_2026.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path, engine="pyarrow")
    df["name_key"] = df["pitcher_name"].apply(_norm_name)
    return df[["name_key", "pitcher_name", "blended_k_pct"]].dropna(subset=["blended_k_pct"])


def load_challenger_projections(year: int = 2026) -> dict[str, pd.DataFrame]:
    """
    Load all available challenger projection parquets.

    Returns dict: system_key -> DataFrame(name_key, k_pct, era, fip, ...)
    """
    systems = {}

    sys_map = {
        "thebatx": "THE BAT X",
        "atc":     "ATC",
        "zips":    "ZiPS",
        "steamer": "Steamer",
    }
    for key, label in sys_map.items():
        path = DATA_DIR / f"fg_proj_{key}_{year}.parquet"
        if path.exists():
            df = pd.read_parquet(path, engine="pyarrow")
            if "name_key" in df.columns and "k_pct" in df.columns:
                keep = ["name_key", "k_pct"]
                for c in ["era", "fip", "mlbam_id"]:
                    if c in df.columns:
                        keep.append(c)
                systems[label] = df[keep].dropna(subset=["k_pct"])
                print(f"  Loaded {label}: {len(systems[label])} pitchers")
        else:
            print(f"  [SKIP] {label} not found ({path.name}) — run fetch_challenger_projections.py")

    # Stuff+ / Pitching+
    stuff_path = DATA_DIR / f"fg_stuff_plus_{year}.parquet"
    if stuff_path.exists():
        df = pd.read_parquet(stuff_path, engine="pyarrow")
        if "k_pct_implied" in df.columns and "name_key" in df.columns:
            keep = ["name_key", "k_pct_implied", "stuff_plus", "pitching_plus"]
            if "fg_playerid" in df.columns:
                keep.append("fg_playerid")
            df = df[keep].dropna(subset=["k_pct_implied"])
            systems["Stuff+/Pitching+"] = df.rename(columns={"k_pct_implied": "k_pct"})
            print(f"  Loaded Stuff+/Pitching+: {len(systems['Stuff+/Pitching+'])} pitchers")
    else:
        print(f"  [SKIP] Stuff+/Pitching+ not found — run fetch_challenger_projections.py")

    return systems


# ---------------------------------------------------------------------------
# GAME ROWS BUILDER
# ---------------------------------------------------------------------------

def build_game_rows(
    actuals: pd.DataFrame,
    cards: pd.DataFrame,
    profiles: pd.DataFrame,
    challengers: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build one row per pitcher-game with actual Ks and all model K predictions.

    For each game:
      - home pitcher and away pitcher each get their own row
      - actual_k = home_sp_k / away_sp_k from actuals
      - actual_ip = home_sp_ip / away_sp_ip
      - our_model_k_mean = mc_home_sp_k_mean from daily card (pre-game, uncalibrated)
      - blended_k_pct = our model's season K% from pitcher profiles
      - {system}_k_pct = challenger K% for that pitcher
    """
    # Build card lookup: (date, home_team, away_team) -> row
    card_lookup: dict[tuple, pd.Series] = {}
    if not cards.empty:
        for _, row in cards.iterrows():
            key = (str(row.get("card_date", "")),
                   str(row.get("home_team", "")),
                   str(row.get("away_team", "")))
            card_lookup[key] = row

    # Build profile lookup: name_key -> blended_k_pct
    prof_lookup: dict[str, float] = {}
    if not profiles.empty:
        for _, row in profiles.iterrows():
            prof_lookup[row["name_key"]] = float(row["blended_k_pct"])

    # Build challenger lookups: system_label -> {name_key: k_pct}
    chal_lookups: dict[str, dict[str, float]] = {}
    for label, df in challengers.items():
        chal_lookups[label] = {
            str(r["name_key"]): float(r["k_pct"])
            for _, r in df.iterrows()
        }

    rows = []

    for _, game in actuals.iterrows():
        date_str  = str(game["game_date"])
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        card_key  = (date_str, home_team, away_team)
        card_row  = card_lookup.get(card_key)

        for side in ("home", "away"):
            actual_k  = float(game[f"{side}_sp_k"])
            actual_ip = float(game[f"{side}_sp_ip"])

            if actual_ip < 1.0:
                continue   # Skip extremely short outings (early injury exits)

            # Pitcher name from daily card
            sp_name = ""
            if card_row is not None:
                sp_name = _norm_name(str(card_row.get(f"{side}_sp", "")))

            # Our model's pre-game K mean from card (already applies KPROP_MEAN_CALIB)
            our_k_mean = None
            if card_row is not None:
                raw = card_row.get(f"mc_{side}_sp_k_mean")
                if pd.notna(raw):
                    our_k_mean = float(raw)

            # Our model's K% from pitcher profile (used when card not available)
            our_k_pct = prof_lookup.get(sp_name) if sp_name else None

            row: dict = {
                "game_date":   date_str,
                "game_pk":     int(game.get("game_pk", 0)),
                "home_team":   home_team,
                "away_team":   away_team,
                "side":        side,
                "pitcher":     sp_name,
                "actual_k":    actual_k,
                "actual_ip":   actual_ip,
                "our_k_mean":  our_k_mean,          # pre-game NB mean (calibrated)
                "our_k_pct":   our_k_pct,           # EWMA K% from profile
            }

            # Challenger K% lookups
            for label, lookup in chal_lookups.items():
                row[f"kpct_{label}"] = lookup.get(sp_name)

            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n  Built {len(df)} pitcher-game rows ({df['game_date'].min()} – {df['game_date'].max()})")
    return df


# ---------------------------------------------------------------------------
# K PREDICTION FROM K%
# ---------------------------------------------------------------------------

def k_pct_to_prob(k_pct: float, actual_ip: float, line: float) -> float:
    """
    Convert a season K% to P(K > line) for a specific game using actual IP.

    Expected K = K% × BF ≈ K% × IP × AVG_BF_PER_IP
    Then simulate with the same NB used by our model.
    """
    mu_k = k_pct * actual_ip * _AVG_BF_PER_IP
    return _nb_over_prob(mu_k, line)


def k_mean_to_prob(k_mean: float, line: float) -> float:
    """
    Convert a pre-game expected K mean (already calibrated) to P(K > line).

    NOTE: mc_*_sp_k_mean from the daily card already includes KPROP_MEAN_CALIB.
    Dividing back out and re-applying avoids double-calibration.
    """
    mu_k_raw = k_mean / KPROP_MEAN_CALIB   # undo the calibration stored in the card
    return _nb_over_prob(mu_k_raw, line)    # _nb_over_prob re-applies it


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def evaluate_system(
    df: pd.DataFrame,
    k_col: str,
    system_label: str,
    is_k_mean: bool = False,
    line: float = PRIMARY_LINE,
) -> dict:
    """
    Compute evaluation metrics for one system on all available game rows.

    k_col: column name of the K prediction (either k_pct or calibrated k_mean)
    is_k_mean: if True, treat the column as pre-calibrated K count mean (from our card)
    """
    valid = df[df[k_col].notna()].copy()
    n = len(valid)
    if n < 5:
        return {"system": system_label, "N": n, "note": "insufficient data"}

    # Compute P(over line) for each row
    probs = []
    for _, row in valid.iterrows():
        if is_k_mean:
            p = k_mean_to_prob(float(row[k_col]), line)
        else:
            p = k_pct_to_prob(float(row[k_col]), float(row["actual_ip"]), line)
        probs.append(p)

    probs = np.array(probs)
    actual_k = valid["actual_k"].values
    actual_over = (actual_k > line).astype(float)

    # K count prediction (expected Ks from the projection)
    if is_k_mean:
        k_pred = valid[k_col].values
    else:
        k_pred = valid[k_col].values * valid["actual_ip"].values * _AVG_BF_PER_IP * KPROP_MEAN_CALIB

    mae  = float(np.mean(np.abs(k_pred - actual_k)))
    rmse = float(np.sqrt(np.mean((k_pred - actual_k) ** 2)))

    brier = _brier(probs, actual_over)
    try:
        auc = _auc(probs, actual_over)
    except Exception:
        auc = float("nan")
    acc   = float(np.mean((probs > 0.5) == (actual_over == 1)))
    cover = float(actual_over.mean())

    return {
        "system":        system_label,
        "N":             n,
        "MAE":           round(mae, 3),
        "RMSE":          round(rmse, 3),
        f"Brier@{line}": round(brier, 4),
        f"AUC@{line}":   round(auc, 4) if not np.isnan(auc) else None,
        f"Acc@{line}":   round(acc, 3),
        "cover_rate":    round(cover, 3),
    }


# ---------------------------------------------------------------------------
# F5 ML EVALUATION
# ---------------------------------------------------------------------------

_LEAGUE_AVG_ERA  = 4.20   # 2026 season estimate for run-prevention calibration
_F5_INNINGS      = 5.0    # first-5-inning window

# Pitching+ sensitivity: Pitching+ 100 = league avg ERA
# implied_ERA = league_avg_ERA × (100 / Pitching+)
# Each +10 Pitching+ ≈ −0.38 ERA


def _era_to_f5_win_prob(home_era: float, away_era: float) -> float:
    """
    Pythagorean F5 win probability from SP ERA projections.

    Home team expected F5 runs ≈ away_SP_ERA × 5/9
    Away team expected F5 runs ≈ home_SP_ERA × 5/9
    P(home wins F5) = home_exp^2 / (home_exp^2 + away_exp^2)
    """
    home_exp = max(away_era * _F5_INNINGS / 9.0, 0.01)
    away_exp = max(home_era * _F5_INNINGS / 9.0, 0.01)
    return home_exp ** 2 / (home_exp ** 2 + away_exp ** 2)


def _pitching_plus_to_era(p_plus: float) -> float:
    """Linear Pitching+ → implied ERA. Pitching+ 100 = league avg."""
    return _LEAGUE_AVG_ERA * 100.0 / max(float(p_plus), 40.0)


def build_f5_game_rows(
    actuals: pd.DataFrame,
    cards: pd.DataFrame,
    challengers: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build one row per game for F5 ML evaluation.

    Columns:
      game_date, home_team, away_team, f5_home_win (actual 0/1)
      home_sp, away_sp (from daily card)
      our_mc_f5        — mc_f5_home_win_prob from card
      our_f5_stacker   — f5_stacker_l2 from card
      era_{system}     — Pythagorean P(home_f5_win) from ERA projection
      pitching_plus_p  — Pythagorean P(home_f5_win) from Pitching+
    """
    # Game-level actuals
    act = actuals[["game_pk", "game_date", "home_team", "away_team", "f5_home_win"]].copy()
    act = act.dropna(subset=["f5_home_win"])

    # Daily card lookup: (date, home, away) -> row
    card_lookup: dict[tuple, pd.Series] = {}
    if not cards.empty:
        for _, row in cards.iterrows():
            key = (str(row.get("card_date", "")),
                   str(row.get("home_team", "")),
                   str(row.get("away_team", "")))
            card_lookup[key] = row

    # ERA lookups per system: name_key -> era
    era_lookups: dict[str, dict[str, float]] = {}
    for label, df in challengers.items():
        if "era" in df.columns:
            era_lookups[label] = {
                str(r["name_key"]): float(r["era"])
                for _, r in df.iterrows()
                if pd.notna(r.get("era"))
            }

    # Pitching+ lookup: name_key -> pitching_plus
    pp_lookup: dict[str, float] = {}
    if "Stuff+/Pitching+" in challengers:
        sp_df = challengers["Stuff+/Pitching+"]
        if "pitching_plus" in sp_df.columns:
            pp_lookup = {
                str(r["name_key"]): float(r["pitching_plus"])
                for _, r in sp_df.iterrows()
                if pd.notna(r.get("pitching_plus"))
            }

    rows = []
    for _, game in act.iterrows():
        date_str  = str(game["game_date"])
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        card_key  = (date_str, home_team, away_team)
        card      = card_lookup.get(card_key)

        home_sp = away_sp = ""
        our_mc = our_stk = None
        if card is not None:
            home_sp  = _norm_name(str(card.get("home_sp", "")))
            away_sp  = _norm_name(str(card.get("away_sp", "")))
            our_mc   = card.get("mc_f5_home_win_prob")
            our_stk  = card.get("f5_stacker_l2")
            if pd.isna(our_mc):  our_mc  = None
            if pd.isna(our_stk): our_stk = None

        row: dict = {
            "game_date":    date_str,
            "game_pk":      int(game.get("game_pk", 0)),
            "home_team":    home_team,
            "away_team":    away_team,
            "home_sp":      home_sp,
            "away_sp":      away_sp,
            "f5_home_win":  int(game["f5_home_win"]),
            "our_mc_f5":    our_mc,
            "our_f5_stacker": our_stk,
        }

        # ERA-based Pythagorean for each projection system
        for label, era_lkp in era_lookups.items():
            home_era = era_lkp.get(home_sp)
            away_era = era_lkp.get(away_sp)
            if home_sp and away_sp and home_era is not None and away_era is not None:
                row[f"era_p_{label}"] = _era_to_f5_win_prob(home_era, away_era)
            else:
                row[f"era_p_{label}"] = None

        # Pitching+ Pythagorean
        home_pp = pp_lookup.get(home_sp)
        away_pp = pp_lookup.get(away_sp)
        if home_sp and away_sp and home_pp is not None and away_pp is not None:
            home_imp_era = _pitching_plus_to_era(home_pp)
            away_imp_era = _pitching_plus_to_era(away_pp)
            row["pitching_plus_p"] = _era_to_f5_win_prob(home_imp_era, away_imp_era)
        else:
            row["pitching_plus_p"] = None

        rows.append(row)

    df_out = pd.DataFrame(rows)
    card_games = df_out[df_out["our_mc_f5"].notna()]
    print(f"\n  F5 game rows: {len(df_out)} total, "
          f"{len(card_games)} with daily card predictions")
    return df_out


def evaluate_f5(game_df: pd.DataFrame, challengers: dict) -> None:
    """Evaluate F5 ML challenger systems and print comparison report."""

    actual = game_df["f5_home_win"].values.astype(float)

    def _metrics(prob_col: str, label: str) -> dict | None:
        valid = game_df[game_df[prob_col].notna()].copy()
        if len(valid) < 10:
            return None
        probs  = valid[prob_col].values.astype(float)
        actual_sub = valid["f5_home_win"].values.astype(float)
        brier  = _brier(probs, actual_sub)
        auc    = _auc(probs, actual_sub)
        acc    = float(np.mean((probs > 0.5) == (actual_sub == 1)))
        cover  = float(actual_sub.mean())
        return {
            "system":   label,
            "N":        len(valid),
            "Brier":    round(brier, 4),
            "AUC":      round(auc, 4) if not np.isnan(auc) else None,
            "Acc":      round(acc, 3),
            "cover":    round(cover, 3),
        }

    results = []

    # Our model signals
    for col, label in [("our_mc_f5", "Our Model (MC sim)"),
                        ("our_f5_stacker", "Our Model (F5 Stacker)")]:
        if col in game_df.columns:
            r = _metrics(col, label)
            if r:
                results.append(r)

    # Projection ERA challengers
    era_systems = [l for l in challengers if "era" in challengers[l].columns]
    for label in era_systems:
        col = f"era_p_{label}"
        if col in game_df.columns:
            r = _metrics(col, f"{label} (ERA)")
            if r:
                results.append(r)

    # Pitching+ challenger
    if "pitching_plus_p" in game_df.columns:
        r = _metrics("pitching_plus_p", "Pitching+ (implied ERA)")
        if r:
            results.append(r)

    if not results:
        print("\n  No F5 results — need daily cards with F5 predictions.")
        return

    results.sort(key=lambda r: r.get("Brier", 99))

    print(f"\n{'='*70}")
    print(f"  F5 ML CHALLENGER EVALUATION")
    print(f"{'='*70}")
    print(f"  {'System':<30} {'N':>5}  {'Brier':>7}  {'AUC':>7}  {'Acc':>6}  {'Cover':>6}")
    print("  " + "-" * 62)
    for r in results:
        auc = r.get("AUC")
        print(f"  {r['system']:<30} {r['N']:>5}  {r['Brier']:>7.4f}  "
              f"{auc if auc else 'N/A':>7}  {r['Acc']:>6.3f}  {r['cover']:>6.3f}")

    print(f"\n  Note: ERA challenger uses Pythagorean F5 win expectancy.")
    print(f"        Pitching+ converted to implied ERA (Pitching+ 100 = {_LEAGUE_AVG_ERA} ERA).")
    print(f"        Our model coverage limited to games with daily cards.")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# BLEND OPTIMIZER
# ---------------------------------------------------------------------------

def _precompute_probs(
    df: pd.DataFrame,
    col: str,
    line: float,
    is_k_mean: bool = False,
) -> np.ndarray:
    """Vectorised P(K > line) for every row in df using the given K% or K-mean column."""
    probs = np.full(len(df), np.nan)
    for i, (_, row) in enumerate(df.iterrows()):
        val = row.get(col)
        if pd.isna(val):
            continue
        if is_k_mean:
            probs[i] = k_mean_to_prob(float(val), line)
        else:
            probs[i] = k_pct_to_prob(float(val), float(row["actual_ip"]), line)
    return probs


def optimize_blend(
    df: pd.DataFrame,
    challengers: dict,
    line: float = PRIMARY_LINE,
    step: float = 0.05,
) -> None:
    """
    Grid-search optimal blend weights across challenger systems.

    Runs two analyses:
      1. 2-way: ATC vs Stuff+/Pitching+  (the two most complementary systems)
      2. 3-way: ATC + Stuff+ + Our Model  (full stack including our signal)

    Probabilities are pre-computed per system then linearly interpolated,
    which is fast and a very good approximation to re-simulating from blended K%.
    """
    print(f"\n{'='*80}")
    print(f"  BLEND OPTIMIZER — Line: {line}")
    print(f"{'='*80}")

    # ── Pre-compute per-system probabilities ─────────────────────────────────
    system_probs: dict[str, np.ndarray] = {}

    if "ATC" in challengers:
        system_probs["ATC"] = _precompute_probs(df, "kpct_ATC", line)
    if "Stuff+/Pitching+" in challengers:
        system_probs["Stuff+"] = _precompute_probs(df, "kpct_Stuff+/Pitching+", line)
    if "ZiPS" in challengers:
        system_probs["ZiPS"] = _precompute_probs(df, "kpct_ZiPS", line)
    if "our_k_mean" in df.columns:
        system_probs["Our Model"] = _precompute_probs(df, "our_k_mean", line, is_k_mean=True)

    if len(system_probs) < 2:
        print("  Need at least 2 systems with data to optimize blends.")
        return

    actual_over = (df["actual_k"].values > line).astype(float)

    def _blend_metrics(p_blend: np.ndarray, actual: np.ndarray) -> tuple[float, float, float]:
        """Brier, AUC, Acc — only on rows where p_blend is not NaN."""
        mask = ~np.isnan(p_blend)
        if mask.sum() < 10:
            return float("nan"), float("nan"), float("nan")
        p = p_blend[mask]
        a = actual[mask]
        brier = _brier(p, a)
        auc   = _auc(p, a)
        acc   = float(np.mean((p > 0.5) == (a == 1)))
        return brier, auc, acc

    weights = np.round(np.arange(0.0, 1.0 + step / 2, step), 2)

    # ── 2-way blend ──────────────────────────────────────────────────────────
    pairs = [
        ("ATC",       "Stuff+",    "ATC vs Stuff+/Pitching+"),
        ("ZiPS",      "Stuff+",    "ZiPS vs Stuff+/Pitching+"),
        ("ATC",       "Our Model", "ATC vs Our Model"),
        ("Stuff+",    "Our Model", "Stuff+ vs Our Model"),
    ]

    best_overall: dict = {}

    for sys_a, sys_b, label in pairs:
        pa = system_probs.get(sys_a)
        pb = system_probs.get(sys_b)
        if pa is None or pb is None:
            continue

        # Only rows where both are available
        joint_mask = ~np.isnan(pa) & ~np.isnan(pb)
        n_joint = int(joint_mask.sum())
        if n_joint < 10:
            continue

        print(f"\n  2-WAY: {label}  (N={n_joint})")
        print(f"  {'Weight_A':>10}  {'Weight_B':>10}  {'Brier':>8}  {'AUC':>7}  {'Acc':>7}")
        print("  " + "-" * 48)

        best_brier = float("inf")
        best_row   = None
        grid_rows  = []

        for alpha in weights:
            blended = np.where(joint_mask, alpha * pa + (1 - alpha) * pb, np.nan)
            brier, auc, acc = _blend_metrics(blended, actual_over)
            grid_rows.append((alpha, 1 - alpha, brier, auc, acc))
            if brier < best_brier:
                best_brier = brier
                best_row   = (alpha, 1 - alpha, brier, auc, acc)

        # Print every other row to keep output compact, always include endpoints + best
        best_alpha = best_row[0] if best_row else None
        printed = set()
        for i, (a, b, brier, auc, acc) in enumerate(grid_rows):
            show = (i % 4 == 0) or (a == best_alpha) or (a in [0.0, 1.0])
            if show and a not in printed:
                marker = " <-- best" if a == best_alpha else ""
                print(f"  {sys_a} {a:>7.0%}  {sys_b} {b:>7.0%}  {brier:>8.4f}  "
                      f"{auc:>7.4f}  {acc:>7.3f}{marker}")
                printed.add(a)

        if best_row:
            best_overall[label] = {
                "sys_a": sys_a, "sys_b": sys_b,
                "weight_a": best_row[0], "weight_b": best_row[1],
                "brier": best_row[2], "auc": best_row[3],
            }

    # ── 3-way blend: ATC + Stuff+ + Our Model ────────────────────────────────
    pa = system_probs.get("ATC")
    pb = system_probs.get("Stuff+")
    pc = system_probs.get("Our Model")

    if pa is not None and pb is not None and pc is not None:
        joint3 = ~np.isnan(pa) & ~np.isnan(pb) & ~np.isnan(pc)
        n3 = int(joint3.sum())

        if n3 >= 10:
            print(f"\n  3-WAY: ATC + Stuff+/Pitching+ + Our Model  (N={n3})")
            print(f"  {'ATC':>6}  {'Stuff+':>7}  {'OurMdl':>8}  {'Brier':>8}  {'AUC':>7}  {'Acc':>7}")
            print("  " + "-" * 56)

            best3_brier = float("inf")
            best3_row   = None

            for alpha in weights:
                for beta in weights:
                    gamma = round(1.0 - alpha - beta, 2)
                    if gamma < 0 or gamma > 1.0:
                        continue
                    blended = np.where(
                        joint3,
                        alpha * pa + beta * pb + gamma * pc,
                        np.nan
                    )
                    brier, auc, acc = _blend_metrics(blended, actual_over)
                    if not np.isnan(brier) and brier < best3_brier:
                        best3_brier = brier
                        best3_row   = (alpha, beta, gamma, brier, auc, acc)

            if best3_row:
                a, b, g, brier, auc, acc = best3_row
                print(f"  {a:>6.0%}  {b:>7.0%}  {g:>8.0%}  "
                      f"{brier:>8.4f}  {auc:>7.4f}  {acc:>7.3f}  <-- optimal 3-way")

                # Show pure baselines for comparison
                for name, p_arr in [("ATC only", pa), ("Stuff+ only", pb), ("Our Model only", pc)]:
                    brier_pure, auc_pure, acc_pure = _blend_metrics(
                        np.where(joint3, p_arr, np.nan), actual_over
                    )
                    print(f"  (vs {name:<16}: Brier={brier_pure:.4f}  AUC={auc_pure:.4f})")

    # ── Summary ───────────────────────────────────────────────────────────────
    if best_overall:
        print(f"\n  BEST 2-WAY BLENDS SUMMARY")
        print(f"  {'Pair':<32} {'Opt Weight A':>14}  {'Opt Weight B':>14}  {'Brier':>8}  {'AUC':>7}")
        print("  " + "-" * 78)
        for label, r in sorted(best_overall.items(), key=lambda x: x[1]["brier"]):
            print(f"  {label:<32} {r['sys_a']} {r['weight_a']:>5.0%}  "
                  f"{r['sys_b']} {r['weight_b']:>5.0%}  {r['brier']:>8.4f}  {r['auc']:>7.4f}")

    print(f"\n  Note: Blends use linear probability interpolation (fast approximation).")
    print(f"        Re-simulate from blended K% for exact results.")
    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

def print_report(results: list[dict], line: float) -> None:
    import math

    print(f"\n{'='*80}")
    print(f"  K-PROP CHALLENGER MODEL EVALUATION  — Line: {line}")
    print(f"{'='*80}")

    valid = [r for r in results if "MAE" in r]
    if not valid:
        print("  No results to display.")
        return

    # Sort by Brier ascending (lower is better)
    brier_col = f"Brier@{line}"
    valid.sort(key=lambda r: r.get(brier_col, 99))

    # Header
    hdr = (f"  {'System':<22} {'N':>5}  {'MAE':>6}  {'RMSE':>6}  "
           f"{'Brier':>7}  {'AUC':>6}  {'Acc':>6}  {'Cover':>6}")
    print(hdr)
    print("  " + "-" * 76)

    for r in valid:
        brier = r.get(brier_col)
        auc   = r.get(f"AUC@{line}")
        acc   = r.get(f"Acc@{line}")
        print(
            f"  {r['system']:<22} {r['N']:>5}  {r['MAE']:>6.3f}  {r['RMSE']:>6.3f}  "
            f"{brier:>7.4f}  {auc if auc else 'N/A':>6}  {acc if acc else 'N/A':>6}  "
            f"{r.get('cover_rate', '?'):>6}"
        )

    # Systems with insufficient data
    skipped = [r for r in results if "MAE" not in r]
    if skipped:
        print(f"\n  Skipped (insufficient data): {', '.join(r['system'] for r in skipped)}")

    print(f"\n  Note: Brier/AUC/Acc computed at {line} K line via Negative Binomial simulation")
    print(f"        MAE/RMSE on raw K count (lower = better for all metrics)")
    print(f"        Cover rate = actual fraction of games pitcher went over {line} Ks")
    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate K-prop challenger models")
    parser.add_argument("--since",   type=str, default=None,
                        help="Only include games on or after YYYY-MM-DD")
    parser.add_argument("--line",    type=float, default=PRIMARY_LINE,
                        help=f"Primary K line to evaluate at (default: {PRIMARY_LINE})")
    parser.add_argument("--year",    type=int, default=2026)
    parser.add_argument("--save",       action="store_true",
                        help="Save results to challenger_eval_k_results.csv")
    parser.add_argument("--systems",    nargs="+", default=None,
                        help="Filter to specific system labels")
    parser.add_argument("--no-blend",   action="store_true",
                        help="Skip blend optimization")
    parser.add_argument("--blend-step", type=float, default=0.05,
                        help="Grid search step size (default: 0.05)")
    parser.add_argument("--no-f5",      action="store_true",
                        help="Skip F5 ML evaluation")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  K-PROP CHALLENGER EVALUATION — {args.year}")
    print(f"{'='*60}\n")

    # --- Load data ---
    print("  Loading actuals ...")
    actuals = load_actuals(since=args.since)
    print(f"    {len(actuals)} completed games")

    print("  Loading daily cards ...")
    cards = load_daily_cards()
    print(f"    {len(cards)} game rows across {cards['card_date'].nunique() if not cards.empty else 0} dates")

    print("  Loading pitcher profiles ...")
    profiles = load_pitcher_profiles()
    print(f"    {len(profiles)} pitchers with blended_k_pct")

    print("  Loading challenger projections ...")
    challengers = load_challenger_projections(year=args.year)

    if not challengers and cards.empty:
        print("\n  [ERROR] No challenger data found. Run fetch_challenger_projections.py first.")
        return

    # --- Build game rows ---
    game_df = build_game_rows(actuals, cards, profiles, challengers)
    if game_df.empty:
        print("  No game rows built — check that actuals and daily cards overlap in date range.")
        return

    # --- Evaluate each system ---
    results = []

    # Our model: use pre-game K mean from daily card (most accurate — already uses all adjustments)
    if "our_k_mean" in game_df.columns and game_df["our_k_mean"].notna().sum() >= 5:
        results.append(evaluate_system(
            game_df, "our_k_mean", "Our Model (MC+XGB)", is_k_mean=True, line=args.line
        ))
    elif "our_k_pct" in game_df.columns:
        results.append(evaluate_system(
            game_df, "our_k_pct", "Our Model (profile K%)", line=args.line
        ))

    # Challenger systems
    for label in challengers:
        col = f"kpct_{label}"
        if col in game_df.columns:
            if args.systems and label not in args.systems:
                continue
            results.append(evaluate_system(game_df, col, label, line=args.line))

    # --- Print report ---
    print_report(results, line=args.line)

    # --- Multi-line summary ---
    if len(EVAL_LINES) > 1:
        print("  MULTI-LINE BRIER SUMMARY")
        print(f"  {'System':<22}", end="")
        for ln in EVAL_LINES:
            print(f"  {'Brier@'+str(ln):>10}", end="")
        print()
        print("  " + "-" * (22 + len(EVAL_LINES) * 12))

        for r_primary in [r for r in results if "MAE" in r]:
            sys_label = r_primary["system"]
            print(f"  {sys_label:<22}", end="")
            for ln in EVAL_LINES:
                if "our_k_mean" in game_df.columns and r_primary["system"].startswith("Our"):
                    res = evaluate_system(game_df, "our_k_mean", sys_label, is_k_mean=True, line=ln)
                else:
                    col = next((f"kpct_{l}" for l in challengers if l == sys_label), None)
                    res = evaluate_system(game_df, col, sys_label, line=ln) if col else {}
                b = res.get(f"Brier@{ln}", float("nan"))
                print(f"  {b:>10.4f}", end="")
            print()
        print()

    # --- F5 ML evaluation ---
    if not args.no_f5:
        print("\n  Building F5 game rows ...")
        f5_df = build_f5_game_rows(actuals, cards, challengers)
        evaluate_f5(f5_df, challengers)

    # --- Blend optimization ---
    if not args.no_blend:
        optimize_blend(game_df, challengers, line=args.line, step=args.blend_step)

    # --- Save ---
    if args.save and results:
        df_out = pd.DataFrame([r for r in results if "MAE" in r])
        df_out.to_csv(RESULTS_CSV, index=False)
        print(f"  Results saved -> {RESULTS_CSV}")

    # --- Coverage report ---
    print("  DATA COVERAGE (pitchers matched in each system)")
    print(f"  {'System':<22} {'Found':>8} {'Missing':>8} {'Pct':>6}")
    print("  " + "-" * 46)
    total = len(game_df)
    for label in challengers:
        col = f"kpct_{label}"
        if col in game_df.columns:
            found   = game_df[col].notna().sum()
            missing = total - found
            pct     = found / total * 100 if total > 0 else 0
            print(f"  {label:<22} {found:>8} {missing:>8} {pct:>5.1f}%")
    if "our_k_mean" in game_df.columns:
        found   = game_df["our_k_mean"].notna().sum()
        missing = total - found
        pct     = found / total * 100 if total > 0 else 0
        print(f"  {'Our Model (MC)':<22} {found:>8} {missing:>8} {pct:>5.1f}%")
    print()


if __name__ == "__main__":
    main()
