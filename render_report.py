"""
render_report.py
================
Deterministic HTML renderer for model_report.html.

Reads model_scores.csv and writes model_report.html with the new structure:
  A. Header + Stats Bar (live tracker)
  B. Actionable Bets — grouped Tier 1 / Tier 2 across ALL markets
  C. Alpha Markets — ML + Totals calibration showcase (Game | Pick |
                     Model Prob | Pinnacle Prob | Retail Edge)
  D. Result Entry placeholder (hydrated client-side from /pending_bets)
  E. Full picks table (collapsed by default)

No Votes / component columns. No MFull/MF5i/etc. legacy fields anywhere.

Usage:
  python render_report.py
  python render_report.py --scores model_scores.csv --out model_report.html
"""
from __future__ import annotations

import argparse
import html as _html
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

TRACKER_PORT = int(os.environ.get("TRACKER_PORT", "8765"))

# ── Auto-Reconciliation stats ────────────────────────────────────────────────
# compute_rolling_accuracy() lives in wizard_agents/tools/implementations.py.
# When this module is imported from the orchestrator, wizard_agents is already
# on sys.path. When run standalone via CLI, add it defensively so the rolling
# accuracy matrix still renders.
_WIZARD_DIR = Path(__file__).resolve().parent / "wizard_agents"
if _WIZARD_DIR.exists() and str(_WIZARD_DIR) not in sys.path:
    sys.path.insert(0, str(_WIZARD_DIR))

try:
    from tools.implementations import compute_rolling_accuracy  # type: ignore
except Exception:  # noqa: BLE001
    compute_rolling_accuracy = None  # type: ignore

MARKET_BADGE = {
    "ML":      ("#1e88e5", "ML"),       # blue
    "Totals":  ("#8e24aa", "TOT"),      # purple
    "Runline": ("#fb8c00", "RL"),       # orange
    "F5":      ("#00897b", "F5"),       # teal
    "NRFI":    ("#e53935", "NRFI"),     # red
}


def _fmt_prob(x) -> str:
    if pd.isna(x):
        return "—"
    return f"{float(x) * 100:.1f}%"


def _fmt_edge(x) -> str:
    if pd.isna(x):
        return "—"
    v = float(x) * 100
    sign = "+" if v >= 0 else "−"
    return f"{sign}{abs(v):.2f}%"


def _edge_class(x) -> str:
    if pd.isna(x):
        return "edge-zero"
    if x > 0.001:
        return "edge-pos"
    if x < -0.001:
        return "edge-neg"
    return "edge-zero"


def _fmt_odds(x) -> str:
    if pd.isna(x):
        return "—"
    v = int(x)
    return f"+{v}" if v > 0 else f"{v}"


def _fmt_stake(x) -> str:
    if pd.isna(x):
        return "—"
    return f"${int(x)}"


def _badge(market: str) -> str:
    color, label = MARKET_BADGE.get(market, ("#607d8b", market[:4]))
    return f'<span class="mkt-badge" style="background:{color}">{label}</span>'


def _pick_label(row) -> str:
    """Human-readable pick label for the Pick column.

    Shows the actual team name the model is backing (instead of just HOME/AWAY)
    so the full picks table reads like "Detroit Tigers" rather than "HOME".
    For Totals/NRFI, returns the OVER/UNDER line or NRFI label as before.
    """
    m    = row["model"]
    bet  = row.get("bet_type", "")
    side = str(row.get("pick_direction", "")).strip().upper()
    game = str(row.get("game", ""))

    if m == "NRFI":
        return _html.escape(str(bet))
    if m == "Totals":
        # e.g. "OVER Total 8.5"
        return f"{_html.escape(side)} {_html.escape(str(bet))}"

    # ML / Runline / F5 — show the team name being backed.
    if "@" in game:
        away_nm, home_nm = [p.strip() for p in game.split("@", 1)]
        team = home_nm if side == "HOME" else away_nm if side == "AWAY" else ""
        if team:
            # For Runline, append -1.5 / +1.5 so it's unambiguous.
            if m == "Runline":
                suffix = " -1.5" if side == "HOME" else " +1.5"
                return f"{_html.escape(team)}<span class='muted'>{_html.escape(suffix)}</span>"
            return _html.escape(team)
    return _html.escape(side)


# ─── Summary stats (static, derived from today's scores) ────────────────────

def _american_profit_multiplier(american) -> float:
    """Profit-per-$1 for a given American odds value. NaN → 0.0."""
    if pd.isna(american):
        return 0.0
    a = float(american)
    if a == 0:
        return 0.0
    return a / 100.0 if a > 0 else 100.0 / -a


def _calculate_summary_stats(df: pd.DataFrame) -> dict:
    """
    Header-tile stats derived purely from today's model_scores.csv.

    Returns:
        n_actionable : count of actionable rows
        total_risked : sum(dollar_stake) across actionable rows
        daily_ev     : sum over actionable rows of
                         (model_prob * potential_profit)
                         - ((1 - model_prob) * dollar_stake)
                       where potential_profit = dollar_stake * profit_multiplier,
                       and profit_multiplier is derived from retail_american_odds.
    """
    if "actionable" not in df.columns:
        return {"n_actionable": 0, "total_risked": 0.0, "daily_ev": 0.0}

    act = df[df["actionable"] == True].copy()
    total_risked = float(act["dollar_stake"].fillna(0).sum())

    daily_ev = 0.0
    for _, r in act.iterrows():
        stake = r.get("dollar_stake")
        prob  = r.get("model_prob")
        odds  = r.get("retail_american_odds")
        if pd.isna(stake) or pd.isna(prob) or pd.isna(odds) or float(stake) <= 0:
            continue
        stake = float(stake)
        prob  = float(prob)
        potential_profit = stake * _american_profit_multiplier(odds)
        daily_ev += (prob * potential_profit) - ((1.0 - prob) * stake)

    return {
        "n_actionable": int(len(act)),
        "total_risked": total_risked,
        "daily_ev":     daily_ev,
    }


def _render_summary_tiles(stats: dict) -> str:
    """Static tiles for Actionable / Total Risked / Daily EV — sits above the live stats bar."""
    ev = stats["daily_ev"]
    ev_cls   = "ev-pos" if ev >= 0 else "ev-neg"
    ev_sign  = "+" if ev >= 0 else "−"
    ev_abs   = abs(ev)
    return (
        "<div class='stat-tiles'>"
        "<div class='stat-tile'>"
        "<div class='l'>Actionable Picks</div>"
        f"<div class='v'>{stats['n_actionable']}</div>"
        "</div>"
        "<div class='stat-tile'>"
        "<div class='l'>Total Risked</div>"
        f"<div class='v'>${stats['total_risked']:,.0f}</div>"
        "</div>"
        "<div class='stat-tile'>"
        "<div class='l'>Daily EV</div>"
        f"<div class='v {ev_cls}'>{ev_sign}${ev_abs:,.2f}</div>"
        "</div>"
        "</div>"
    )


# ─── Actionable card grid ────────────────────────────────────────────────────

def _generate_cards_html(df: pd.DataFrame) -> str:
    """
    Responsive card-grid view of actionable picks — replaces the prior
    render_actionable() table. Tier 1 and Tier 2 share a single grid;
    tier is distinguished by left-border color (green for T1, gold for T2).
    """
    if "actionable" not in df.columns:
        return ("<section class='card hero'><h2>🎯 Actionable Bets</h2>"
                "<p class='muted'>model_scores.csv missing 'actionable' column.</p></section>")

    act = df[df["actionable"] == True].copy()
    if len(act) == 0:
        return ("<section class='card hero'><h2>🎯 Actionable Bets</h2>"
                "<p class='muted'>No actionable bets today.</p></section>")

    # Sort: Tier 1 first, then by descending edge within tier.
    act["_tier_sort"] = act["tier"].fillna(9).astype(float)
    act = act.sort_values(["_tier_sort", "edge"], ascending=[True, False]).drop(columns="_tier_sort")

    cards = []
    for _, r in act.iterrows():
        tier = None if pd.isna(r.get("tier")) else int(r["tier"])
        if tier == 1:
            tier_cls, tier_lbl = "bet-card tier-1-card", "TIER 1"
        elif tier == 2:
            tier_cls, tier_lbl = "bet-card tier-2-card", "TIER 2"
        else:
            tier_cls, tier_lbl = "bet-card", ""

        payload = {
            "date":                 r["date"],
            "game":                 r["game"],
            "model":                r["model"],
            "bet_type":             r["bet_type"],
            "pick_direction":       r["pick_direction"],
            "model_prob":           None if pd.isna(r["model_prob"]) else float(r["model_prob"]),
            "market_line":          r["bet_type"],
            "retail_american_odds": None if pd.isna(r["retail_american_odds"]) else int(r["retail_american_odds"]),
            "stake":                None if pd.isna(r["dollar_stake"]) else int(r["dollar_stake"]),
        }
        payload_json = _html.escape(json.dumps(payload), quote=True)

        cards.append(
            f"<article class='{tier_cls}'>"
              "<header class='bc-head'>"
                f"<span class='bc-match'>{_html.escape(str(r['game']))}</span>"
                f"<span class='bc-tier'>{tier_lbl}</span>"
              "</header>"
              "<div class='bc-sub'>"
                f"{_badge(r['model'])}"
                f"<span class='bc-market'>{_html.escape(str(r['bet_type']))}</span>"
                "<span class='bc-sep'>·</span>"
                f"<span class='bc-pick'>{_pick_label(r)}</span>"
              "</div>"
              "<div class='bc-grid'>"
                "<div class='bc-stat'>"
                  "<span class='bc-l'>Model</span>"
                  f"<span class='bc-v'>{_fmt_prob(r['model_prob'])}</span>"
                "</div>"
                "<div class='bc-stat'>"
                  "<span class='bc-l'>Pinnacle</span>"
                  f"<span class='bc-v subtle'>{_fmt_prob(r['P_true'])}</span>"
                "</div>"
                "<div class='bc-stat'>"
                  "<span class='bc-l'>Edge</span>"
                  f"<span class='bc-v {_edge_class(r['edge'])}'>{_fmt_edge(r['edge'])}</span>"
                "</div>"
                "<div class='bc-stat'>"
                  "<span class='bc-l'>Odds</span>"
                  f"<span class='bc-v'>{_fmt_odds(r['retail_american_odds'])}</span>"
                "</div>"
                "<div class='bc-stat'>"
                  "<span class='bc-l'>Stake</span>"
                  f"<span class='bc-v stake'>{_fmt_stake(r['dollar_stake'])}</span>"
                "</div>"
                "<div class='bc-stat bc-action'>"
                  f"<button class='btn-log' data-pick='{payload_json}'>Log Bet</button>"
                "</div>"
              "</div>"
            "</article>"
        )

    t1 = int((act["tier"] == 1).sum())
    t2 = int((act["tier"] == 2).sum())
    return (
        "<section class='card hero'>"
        f"<h2>🎯 Actionable Bets &middot; {len(act)} total "
        f"<span class='muted'>(Tier 1: {t1} · Tier 2: {t2})</span></h2>"
        f"<div class='card-grid'>{''.join(cards)}</div>"
        "</section>"
    )


# ─── Optimal probability cutoff per market (2025 OOF / 2026 / last-30d) ─────

_CUTOFF_THRESHOLDS = [0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75, 0.80]
_CUTOFF_MIN_N      = 20   # require at least this many games to call a cutoff

_CUTOFF_ROOT = Path(__file__).resolve().parent

# (market_label, label_shown, 2025_source_csv, 2025_prob_col, 2025_truth_col)
_CUTOFF_2025_SOURCES = [
    ("ML",      "ML",      "ml_val_predictions.csv",    "stacker_ml",       "home_win"),
    ("F5",      "F5",      "f5_val_predictions.csv",    "stacker_f5_cover", "f5_home_cover"),
    ("Runline", "Runline", "eval_predictions.csv",      "rl_stacked",       "home_covers_rl"),
    ("NRFI",    "NRFI",    "nrfi_val_predictions.csv",  "stacker_nrfi",     "f1_nrfi"),
]

# 2026 all-markets come from backtest_full_all_predictions.csv; key = market code in that file.
_CUTOFF_2026_MARKETS = {"ML": "ML", "Totals": "TOT", "Runline": "RL", "F5": "F5", "NRFI": "NR"}

# Cache computed cutoffs across renders (2025 numbers are historical & never change).
_CUTOFF_CACHE: dict[str, dict] = {}


def _sweep_cutoff(probs: pd.Series, truths: pd.Series) -> dict | None:
    """Find the probability cutoff with best hit rate subject to n >= _CUTOFF_MIN_N."""
    if probs is None or probs.empty or truths is None or truths.empty:
        return None
    pair = pd.DataFrame({"p": probs, "y": truths}).dropna()
    if pair.empty:
        return None
    best = None
    for t in _CUTOFF_THRESHOLDS:
        s = pair[pair["p"] >= t]
        n = len(s)
        if n < _CUTOFF_MIN_N:
            continue
        hits = int((s["y"] == 1).sum())
        win_pct = 100.0 * hits / n
        if best is None or win_pct > best["win_pct"]:
            best = {"cutoff": t, "n": n, "wins": hits, "win_pct": win_pct}
    return best


def _sweep_cutoff_tiers(probs: pd.Series, truths: pd.Series) -> dict | None:
    """Return three cutoff tiers for the sweep:
       * optimal  — highest hit-rate cutoff that meets min-n (take these)
       * mid      — middle hit-rate cutoff (less optimal / marginal edge)
       * avoid    — lowest hit-rate cutoff in the sweep (stay away)
    Tiers are the same cutoff when fewer than three valid rows exist."""
    if probs is None or probs.empty or truths is None or truths.empty:
        return None
    pair = pd.DataFrame({"p": probs, "y": truths}).dropna()
    if pair.empty:
        return None
    rows: list[dict] = []
    for t in _CUTOFF_THRESHOLDS:
        s = pair[pair["p"] >= t]
        n = len(s)
        if n < _CUTOFF_MIN_N:
            continue
        hits = int((s["y"] == 1).sum())
        rows.append({"cutoff": t, "n": n, "wins": hits, "win_pct": 100.0 * hits / n})
    if not rows:
        return None
    ranked = sorted(rows, key=lambda r: r["win_pct"], reverse=True)
    optimal = ranked[0]
    avoid   = ranked[-1]
    mid     = ranked[len(ranked) // 2]
    return {"optimal": optimal, "mid": mid, "avoid": avoid}


def _derive_2025_totals_cutoff() -> dict | None:
    """
    2025 val file stored a regression (`tot_pred` = predicted runs), not a
    calibrated OVER probability. We derive P(OVER) = 1 - Phi((close - tot_pred)/sigma)
    where sigma is the empirical std of residuals across the full season.

    Fitting sigma on the same data we evaluate introduces a tiny bias on a
    single scalar, which is acceptable for a cutoff sweep. Pushes (actual ==
    close_total) are dropped so the outcome is binary.
    """
    import math
    path = _CUTOFF_ROOT / "xgb_val_predictions.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=["close_total", "actual_game_total", "tot_pred"])
    except (ValueError, KeyError):
        return None
    df = df.dropna()
    if df.empty:
        return None
    resid = df["actual_game_total"] - df["tot_pred"]
    sigma = float(resid.std())
    if not (sigma and sigma > 0):
        return None
    # Standard-normal CDF via erf (no scipy dependency).
    def _phi(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    z = (df["close_total"] - df["tot_pred"]) / sigma
    df = df.copy()
    df["p_over"]     = 1.0 - z.map(_phi)
    df["over_truth"] = (df["actual_game_total"] > df["close_total"]).astype(int)
    # Drop pushes — binary outcome only.
    df = df[df["actual_game_total"] != df["close_total"]]
    if df.empty:
        return None
    return _sweep_cutoff(df["p_over"], df["over_truth"])


def _compute_cutoffs_2025() -> dict[str, dict | None]:
    if "2025" in _CUTOFF_CACHE:
        return _CUTOFF_CACHE["2025"]
    out: dict[str, dict | None] = {}
    for mkt, _label, fname, pcol, ycol in _CUTOFF_2025_SOURCES:
        path = _CUTOFF_ROOT / fname
        if not path.exists():
            out[mkt] = None
            continue
        try:
            df = pd.read_csv(path, usecols=[pcol, ycol])
        except (ValueError, KeyError):
            out[mkt] = None
            continue
        out[mkt] = _sweep_cutoff(df[pcol], df[ycol])
    # Totals 2025: derive OVER probability from regression residuals.
    out["Totals"] = _derive_2025_totals_cutoff()
    _CUTOFF_CACHE["2025"] = out
    return out


def _compute_cutoffs_2026(last_n_days: int | None = None) -> dict[str, dict | None]:
    """
    Pull from backtest_full_all_predictions.csv (the sterile 2026 backtest) and
    union any forward-working rows persisted to live_predictions_2026.csv by
    the daily pipeline. last_n_days=None = full 2026.
    """
    key = f"2026_{last_n_days or 'all'}"
    if key in _CUTOFF_CACHE:
        return _CUTOFF_CACHE[key]
    out: dict[str, dict | None] = {m: None for m in _CUTOFF_2026_MARKETS}

    frames: list[pd.DataFrame] = []
    for fname in ("backtest_full_all_predictions.csv", "live_predictions_2026.csv"):
        src = _CUTOFF_ROOT / fname
        if not src.exists():
            continue
        try:
            frames.append(pd.read_csv(src, usecols=["market", "model_prob", "actual", "game_date"]))
        except (ValueError, KeyError):
            continue
    if not frames:
        _CUTOFF_CACHE[key] = out
        return out
    df = pd.concat(frames, ignore_index=True)

    if last_n_days is not None and not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        cutoff_date    = df["game_date"].max() - pd.Timedelta(days=last_n_days)
        df = df[df["game_date"] >= cutoff_date]
    for mkt_label, mkt_code in _CUTOFF_2026_MARKETS.items():
        sub = df[df["market"] == mkt_code]
        out[mkt_label] = _sweep_cutoff(sub["model_prob"], sub["actual"])
    _CUTOFF_CACHE[key] = out
    return out


def _compute_cutoffs_2026_tiers(last_n_days: int | None = None) -> dict[str, dict | None]:
    """Same source as _compute_cutoffs_2026 but returns 3-tier info per market."""
    key = f"2026_tiers_{last_n_days or 'all'}"
    if key in _CUTOFF_CACHE:
        return _CUTOFF_CACHE[key]
    out: dict[str, dict | None] = {m: None for m in _CUTOFF_2026_MARKETS}

    frames: list[pd.DataFrame] = []
    for fname in ("backtest_full_all_predictions.csv", "live_predictions_2026.csv"):
        src = _CUTOFF_ROOT / fname
        if not src.exists():
            continue
        try:
            frames.append(pd.read_csv(src, usecols=["market", "model_prob", "actual", "game_date"]))
        except (ValueError, KeyError):
            continue
    if not frames:
        _CUTOFF_CACHE[key] = out
        return out
    df = pd.concat(frames, ignore_index=True)

    if last_n_days is not None and not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        cutoff_date    = df["game_date"].max() - pd.Timedelta(days=last_n_days)
        df = df[df["game_date"] >= cutoff_date]
    for mkt_label, mkt_code in _CUTOFF_2026_MARKETS.items():
        sub = df[df["market"] == mkt_code]
        out[mkt_label] = _sweep_cutoff_tiers(sub["model_prob"], sub["actual"])
    _CUTOFF_CACHE[key] = out
    return out


def _build_cutoff_html() -> str:
    markets_order = ["ML", "Totals", "Runline", "F5", "NRFI"]
    row_2025       = _compute_cutoffs_2025()
    row_2026_tiers = _compute_cutoffs_2026_tiers(last_n_days=None)
    row_l30_tiers  = _compute_cutoffs_2026_tiers(last_n_days=30)

    def _cell_single(d: dict | None) -> str:
        """2025 column — single best cutoff."""
        if not d:
            return "<td class='cutoff-cell'><span class='muted'>—</span></td>"
        return (
            "<td class='cutoff-cell'>"
            f"<div class='cutoff-hd'>prob ≥ <b>{int(d['cutoff']*100)}%</b></div>"
            f"<div class='winpct'><b>{d['win_pct']:.1f}%</b> "
            f"<span class='muted'>({d['wins']}/{d['n']})</span></div>"
            "</td>"
        )

    def _tier_row(label: str, cls: str, d: dict | None) -> str:
        if not d:
            return ""
        return (
            f"<div class='tier-row tier-{cls}'>"
            f"<span class='tier-label'>{label}</span> "
            f"prob ≥ <b>{int(d['cutoff']*100)}%</b> &rarr; "
            f"<b>{d['win_pct']:.1f}%</b> "
            f"<span class='muted'>({d['wins']}/{d['n']})</span>"
            "</div>"
        )

    def _cell_tiers(t: dict | None) -> str:
        if not t:
            return "<td class='cutoff-cell'><span class='muted'>—</span></td>"
        opt, mid, avd = t.get("optimal"), t.get("mid"), t.get("avoid")
        # Collapse rows that are identical (same cutoff) so we don't repeat.
        seen_cuts = set()
        def _take(d):
            if not d: return None
            c = d["cutoff"]
            if c in seen_cuts: return None
            seen_cuts.add(c)
            return d
        html = (
            _tier_row("★ Optimal",     "opt",  _take(opt)) +
            _tier_row("● Less optimal","mid",  _take(mid)) +
            _tier_row("✗ Stay away",   "avoid",_take(avd))
        )
        return f"<td class='cutoff-cell tiered'>{html}</td>"

    body_rows = []
    for mkt in markets_order:
        color, tag = MARKET_BADGE.get(mkt, ("#555", mkt))
        badge = (
            f"<span class='mkt-badge' style='background:{color}'>{tag}</span> "
            f"<span class='mkt-name'>{mkt}</span>"
        )
        body_rows.append(
            f"<tr><td>{badge}</td>"
            f"{_cell_single(row_2025.get(mkt))}"
            f"{_cell_tiers(row_2026_tiers.get(mkt))}"
            f"{_cell_tiers(row_l30_tiers.get(mkt))}"
            "</tr>"
        )

    return (
        "<section class='card'>"
        "<h2>🎯 Optimal Probability Cutoff — per Market</h2>"
        "<p class='muted'>For 2025 the single best cutoff is shown. For 2026 and Last 30 Days "
        f"we show three tiers (min n={_CUTOFF_MIN_N}): "
        "<span class='tier-opt'>★ Optimal</span> = highest hit rate, "
        "<span class='tier-mid'>● Less optimal</span> = middle tier, "
        "<span class='tier-avoid'>✗ Stay away</span> = lowest hit rate. "
        "Numbers under the win% show <i>hits / games</i>.</p>"
        "<table class='tbl acc-tbl'>"
        "<thead><tr><th>Market</th><th>2025 OOF</th><th>2026 STD</th><th>Last 30 Days</th></tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</section>"
    )


# ─── Rolling accuracy matrix ─────────────────────────────────────────────────

_ACCURACY_MARKETS = ["ML", "Totals", "Runline", "F5", "NRFI"]
_ACCURACY_WINDOWS = [
    ("last_7",   "Last 7 Days"),
    ("last_28",  "Last 28 Days"),
    ("ytd_2026", "YTD 2026"),
]


def _roi_class(roi_pct) -> str:
    """Color code: green >0, red <0, grey for 0/NaN."""
    try:
        v = float(roi_pct)
    except (TypeError, ValueError):
        return "edge-zero"
    if pd.isna(v) or v == 0:
        return "edge-zero"
    return "edge-pos" if v > 0 else "edge-neg-strong"


def _fetch_accuracy_stats() -> dict | None:
    """Return parsed windows dict from compute_rolling_accuracy, or None."""
    if compute_rolling_accuracy is None:
        return None
    try:
        raw = compute_rolling_accuracy()
        payload = json.loads(raw)
        if payload.get("status") != "OK":
            return None
        return payload.get("windows") or None
    except Exception:  # noqa: BLE001
        return None


def _build_accuracy_html(windows: dict | None) -> str:
    """HTML matrix: one row per market, one column per rolling window."""
    if not windows:
        return (
            "<section class='card'>"
            "<h2>📊 Rolling Accuracy</h2>"
            "<p class='muted'>Ledger is empty or rolling stats unavailable.</p>"
            "</section>"
        )

    hdr_cells = "".join(f"<th>{_html.escape(lbl)}</th>" for _, lbl in _ACCURACY_WINDOWS)
    head = f"<thead><tr><th>Market</th>{hdr_cells}</tr></thead>"

    body_rows = []
    for mkt in _ACCURACY_MARKETS:
        cells = [f"<td>{_badge(mkt)} <span class='mkt-name'>{mkt}</span></td>"]
        for wkey, _ in _ACCURACY_WINDOWS:
            win = windows.get(wkey) or {}
            mstats = (win.get("by_market") or {}).get(mkt) or {}
            bets    = int(mstats.get("bets", 0) or 0)
            win_pct = float(mstats.get("win_pct", 0.0) or 0.0)
            roi_pct = mstats.get("roi_pct", 0.0)
            try:
                roi_val = float(roi_pct)
            except (TypeError, ValueError):
                roi_val = float("nan")

            if bets == 0:
                cells.append("<td class='acc-cell muted'>— <div class='roi-sub'>no bets</div></td>")
                continue

            roi_txt = "—" if pd.isna(roi_val) else f"{roi_val:+.1f}%"
            roi_cls = _roi_class(roi_val)
            cells.append(
                "<td class='acc-cell'>"
                f"<div class='winpct'>{win_pct:.1f}% <span class='muted'>({bets} bets)</span></div>"
                f"<div class='roi-sub {roi_cls}'>ROI {roi_txt}</div>"
                "</td>"
            )
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    # Alpha SGP (Beta) — joint ML+Totals parlay in the validated sweet-spot band.
    sgp_cells = ["<td>🔗 <span class='mkt-name'>Alpha SGP (Beta)</span></td>"]
    for wkey, _ in _ACCURACY_WINDOWS:
        win = windows.get(wkey) or {}
        sg  = win.get("alpha_sgp") or {}
        bets    = int(sg.get("bets", 0) or 0)
        win_pct = float(sg.get("win_pct", 0.0) or 0.0)
        roi_pct = sg.get("roi_pct", 0.0)
        try:
            roi_val = float(roi_pct)
        except (TypeError, ValueError):
            roi_val = float("nan")
        if bets == 0:
            sgp_cells.append("<td class='acc-cell muted'>— <div class='roi-sub'>no bets</div></td>")
            continue
        roi_txt = "—" if pd.isna(roi_val) else f"{roi_val:+.1f}%"
        roi_cls = _roi_class(roi_val)
        sgp_cells.append(
            "<td class='acc-cell'>"
            f"<div class='winpct'>{win_pct:.1f}% <span class='muted'>({bets} bets)</span></div>"
            f"<div class='roi-sub {roi_cls}'>ROI {roi_txt}</div>"
            "</td>"
        )
    body_rows.append(f"<tr>{''.join(sgp_cells)}</tr>")

    # Overall footer row (across all markets, per window)
    foot_cells = ["<td><b>Overall</b></td>"]
    for wkey, _ in _ACCURACY_WINDOWS:
        win = windows.get(wkey) or {}
        ov  = win.get("overall") or {}
        bets    = int(ov.get("bets", 0) or 0)
        win_pct = float(ov.get("win_pct", 0.0) or 0.0)
        roi_pct = ov.get("roi_pct", 0.0)
        try:
            roi_val = float(roi_pct)
        except (TypeError, ValueError):
            roi_val = float("nan")
        if bets == 0:
            foot_cells.append("<td class='acc-cell muted'>—</td>")
            continue
        roi_txt = "—" if pd.isna(roi_val) else f"{roi_val:+.1f}%"
        roi_cls = _roi_class(roi_val)
        foot_cells.append(
            "<td class='acc-cell'>"
            f"<div class='winpct'><b>{win_pct:.1f}%</b> <span class='muted'>({bets})</span></div>"
            f"<div class='roi-sub {roi_cls}'>ROI {roi_txt}</div>"
            "</td>"
        )
    foot = f"<tfoot><tr class='acc-overall'>{''.join(foot_cells)}</tr></tfoot>"

    return (
        "<section class='card'>"
        "<h2>📊 Rolling Accuracy — Auto-Reconciled Ledger</h2>"
        "<p class='muted'>Win % &amp; bet count per window, with ROI underneath. "
        "Green = profitable · Red = losing · Grey = flat / no bets.</p>"
        f"<table class='tbl acc-tbl'>{head}<tbody>{''.join(body_rows)}</tbody>{foot}</table>"
        "</section>"
    )


# ─── Section renderers ────────────────────────────────────────────────────────

def render_actionable(df: pd.DataFrame) -> str:
    """Actionable Bets table split by Tier 1 / Tier 2."""
    act = df[df["actionable"] == True].copy()
    if len(act) == 0:
        return ("<section class='card'><h2>🎯 Actionable Bets</h2>"
                "<p class='muted'>No actionable bets today.</p></section>")

    act = act.sort_values("edge", ascending=False)

    def _rows(sub: pd.DataFrame) -> str:
        if len(sub) == 0:
            return "<tr><td colspan='9' class='muted'>None</td></tr>"
        parts = []
        for _, r in sub.iterrows():
            payload = {
                "date":            r["date"],
                "game":            r["game"],
                "model":           r["model"],
                "bet_type":        r["bet_type"],
                "pick_direction":  r["pick_direction"],
                "model_prob":      None if pd.isna(r["model_prob"]) else float(r["model_prob"]),
                "market_line":     r["bet_type"],
                "retail_american_odds": None if pd.isna(r["retail_american_odds"]) else int(r["retail_american_odds"]),
                "stake":           None if pd.isna(r["dollar_stake"]) else int(r["dollar_stake"]),
            }
            payload_json = _html.escape(json.dumps(payload), quote=True)
            parts.append(
                f"<tr class='act-row'>"
                f"<td>{_html.escape(str(r['game']))}</td>"
                f"<td>{_badge(r['model'])}</td>"
                f"<td>{_pick_label(r)}</td>"
                f"<td class='num'>{_fmt_prob(r['model_prob'])}</td>"
                f"<td class='num'>{_fmt_prob(r['P_true'])}</td>"
                f"<td class='num {_edge_class(r['edge'])}'><b>{_fmt_edge(r['edge'])}</b></td>"
                f"<td class='num'>{_fmt_odds(r['retail_american_odds'])}</td>"
                f"<td class='num stake'>{_fmt_stake(r['dollar_stake'])}</td>"
                f"<td><button class='btn-log' data-pick='{payload_json}'>Log Bet</button></td>"
                f"</tr>"
            )
        return "".join(parts)

    t1 = act[act["tier"] == 1]
    t2 = act[act["tier"] == 2]

    hdr = ("<thead><tr>"
           "<th>Game</th><th>Market</th><th>Pick</th>"
           "<th>Model Prob</th><th>Pinnacle Prob</th><th>Edge</th>"
           "<th>Retail Odds</th><th>Stake</th><th></th>"
           "</tr></thead>")

    return (
        "<section class='card hero'>"
        "<h2>🎯 Actionable Bets</h2>"
        f"<h3 class='tier-hdr tier-1'>🔥 Tier 1 Edge (≥3%) &middot; {len(t1)} bets</h3>"
        f"<table class='tbl act-tbl'>{hdr}<tbody>{_rows(t1)}</tbody></table>"
        f"<h3 class='tier-hdr tier-2'>◆ Tier 2 Edge (≥1%) &middot; {len(t2)} bets</h3>"
        f"<table class='tbl act-tbl'>{hdr}<tbody>{_rows(t2)}</tbody></table>"
        "</section>"
    )


def _tier_marker(prob, tiers: dict | None) -> str:
    """Return an HTML span marking which 2026 cutoff tier this prob clears:
       ★ (green) = clears optimal · ● (yellow) = clears mid · ✗ (red) = clears
       avoid-floor only · "" = below all cutoffs."""
    if not tiers:
        return ""
    try:
        p = float(prob)
    except (TypeError, ValueError):
        return ""
    opt = (tiers.get("optimal") or {}).get("cutoff")
    mid = (tiers.get("mid")     or {}).get("cutoff")
    avd = (tiers.get("avoid")   or {}).get("cutoff")
    if opt is not None and p >= float(opt):
        return " <span class='tier-mark tier-opt' title='Clears 2026 optimal cutoff'>★</span>"
    if mid is not None and p >= float(mid):
        return " <span class='tier-mark tier-mid' title='Clears 2026 less-optimal cutoff'>●</span>"
    if avd is not None and p >= float(avd):
        return " <span class='tier-mark tier-avoid' title='Only clears stay-away floor'>✗</span>"
    return ""


def render_alpha(df: pd.DataFrame) -> str:
    """Alpha Markets calibration showcase — ML + Totals only.

    Rows are sorted by model_prob (desc). Each pick gets a tier marker based
    on where its model_prob falls in the 2026 cutoff sweep:
      ★ = clears optimal · ● = clears mid · ✗ = only clears stay-away floor.
    """
    tiers_2026 = _compute_cutoffs_2026_tiers(last_n_days=None)

    def _sub(market: str, subtitle: str) -> str:
        sub = df[df["model"] == market].copy()
        if len(sub) == 0:
            return f"<h3>{market}</h3><p class='muted'>No rows.</p>"
        sub = sub.sort_values("model_prob", ascending=False)

        tinfo = tiers_2026.get(market)
        rows = []
        for _, r in sub.iterrows():
            mark = _tier_marker(r["model_prob"], tinfo)
            rows.append(
                f"<tr>"
                f"<td>{_html.escape(str(r['game']))}{mark}</td>"
                f"<td>{_pick_label(r)}</td>"
                f"<td class='num'>{_fmt_prob(r['model_prob'])}</td>"
                f"<td class='num'>{_fmt_prob(r['P_true'])}</td>"
                f"<td class='num {_edge_class(r['edge'])}'>{_fmt_edge(r['edge'])}</td>"
                f"</tr>"
            )
        cutoff_note = ""
        if tinfo:
            opt = tinfo.get("optimal") or {}
            mid = tinfo.get("mid")     or {}
            avd = tinfo.get("avoid")   or {}
            parts = []
            if opt: parts.append(f"<span class='tier-opt'>★ ≥{int(opt['cutoff']*100)}% → {opt['win_pct']:.1f}%</span>")
            if mid: parts.append(f"<span class='tier-mid'>● ≥{int(mid['cutoff']*100)}% → {mid['win_pct']:.1f}%</span>")
            if avd: parts.append(f"<span class='tier-avoid'>✗ ≥{int(avd['cutoff']*100)}% → {avd['win_pct']:.1f}%</span>")
            cutoff_note = f"<p class='muted' style='margin:6px 0 0'>2026 cutoffs: {' · '.join(parts)}</p>"
        return (
            f"<h3>Alpha Market · {subtitle}</h3>"
            "<table class='tbl alpha-tbl'><thead><tr>"
            "<th>Game</th><th>Pick</th><th>Model Prob</th>"
            "<th>Pinnacle Prob</th><th>Retail Edge</th>"
            "</tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>"
            + cutoff_note
        )

    return (
        "<section class='card'>"
        "<h2>📐 Alpha Markets — Calibration Showcase</h2>"
        "<p class='muted'>Best-calibrated markets vs. Pinnacle closing line "
        "(ML ECE 0.045, Totals ECE 0.080). Sorted by Model Prob. Tier markers: "
        "<span class='tier-opt'>★ optimal</span> · "
        "<span class='tier-mid'>● less optimal</span> · "
        "<span class='tier-avoid'>✗ stay away</span>.</p>"
        + _sub("ML", "Moneyline (ML)")
        + _sub("Totals", "Totals")
        + "</section>"
    )


def render_full_table(df: pd.DataFrame) -> str:
    order = ["ML", "Totals", "Runline", "F5", "NRFI"]
    tiers_2026 = _compute_cutoffs_2026_tiers(last_n_days=None)
    blocks = []
    for mkt in order:
        sub = df[df["model"] == mkt].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values(["actionable", "edge"], ascending=[False, False])
        tinfo = tiers_2026.get(mkt)
        rows = []
        for _, r in sub.iterrows():
            cls  = "act-row" if bool(r["actionable"]) else "na-row"
            mark = _tier_marker(r["model_prob"], tinfo)
            rows.append(
                f"<tr class='{cls}'>"
                f"<td>{_html.escape(str(r['game']))}{mark}</td>"
                f"<td>{_html.escape(str(r['bet_type']))}</td>"
                f"<td>{_pick_label(r)}</td>"
                f"<td class='num'>{_fmt_prob(r['model_prob'])}</td>"
                f"<td class='num'>{_fmt_prob(r['P_true'])}</td>"
                f"<td class='num {_edge_class(r['edge'])}'>{_fmt_edge(r['edge'])}</td>"
                f"<td class='num'>{_fmt_odds(r['retail_american_odds'])}</td>"
                f"<td class='num'>{'' if pd.isna(r['tier']) else int(r['tier'])}</td>"
                f"<td class='num'>{_fmt_stake(r['dollar_stake'])}</td>"
                f"</tr>"
            )
        blocks.append(
            f"<h3>{_badge(mkt)} {mkt} &middot; {len(sub)} rows</h3>"
            "<table class='tbl full-tbl'><thead><tr>"
            "<th>Game</th><th>Bet</th><th>Pick</th><th>Model Prob</th>"
            "<th>Pinnacle Prob</th><th>Edge</th><th>Retail Odds</th>"
            "<th>Tier</th><th>Stake</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )
    return (
        "<section class='card'>"
        "<details><summary><h2 style='display:inline'>📋 Full Picks Table</h2>"
        "<span class='muted'> &mdash; click to expand</span></summary>"
        + "".join(blocks) + "</details></section>"
    )


# ─── Assembly ─────────────────────────────────────────────────────────────────

CSS = """
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
     background:#0f1115;color:#ffffff;line-height:1.45}
body,.card,.tbl,.tbl td,.tbl th,.acc-tbl,.acc-tbl td,.acc-tbl th,
.stat,.stat-tile,.stat .v,.stat-tile .v,.stat .l,.stat-tile .l,
header h1,header .sub,.card h2,.card h3,p,div,span,td,th,li,a{color:#ffffff}
.container{max-width:1200px;margin:0 auto;padding:24px 16px}
header{margin-bottom:24px}
header h1{margin:0 0 6px;font-size:28px}
header .sub{color:#ffffff;font-size:13px}
.card{background:#1a1d24;border:1px solid #262a33;border-radius:12px;
      padding:20px;margin-bottom:18px}
.card h2{margin:0 0 12px;font-size:20px}
.card h3{margin:18px 0 8px;font-size:15px;color:#ffffff}
.hero{border:1px solid #2a5b3a;background:linear-gradient(180deg,#13241a,#1a1d24)}
.hero h2{color:#4ade80}
.muted{color:#ffffff;font-size:13px}
.tbl{width:100%;border-collapse:collapse;font-size:13px}
.tbl th,.tbl td{padding:8px 10px;text-align:left;border-bottom:1px solid #262a33}
.tbl th{background:#14161c;color:#ffffff;font-weight:600;text-transform:uppercase;
        font-size:11px;letter-spacing:.05em}
.tbl .num{text-align:right;font-variant-numeric:tabular-nums}
.tbl tr:hover{background:#1f232b}
.act-row{background:rgba(34,197,94,.05)}
.act-row td{border-bottom-color:rgba(34,197,94,.15)}
.na-row{opacity:.55}
.stake{color:#4ade80;font-weight:600}
.edge-pos{color:#4ade80}
.edge-neg{color:#ffffff}
.edge-neg-strong{color:#f87171}
.edge-zero{color:#ffffff}
.acc-tbl td,.acc-tbl th{vertical-align:top;color:#ffffff}
.acc-cell{color:#ffffff}
.acc-cell .winpct{font-size:14px;font-weight:700;color:#ffffff}
.acc-cell .winpct .muted{color:#ffffff;font-weight:400}
.acc-cell .roi-sub{font-size:11px;margin-top:2px;letter-spacing:.02em;color:#ffffff}
.acc-cell.muted{color:#ffffff}
.alpha-star{color:#fbbf24;font-weight:700;margin-left:4px}
.cutoff-cell.tiered{font-size:12px;line-height:1.5}
.cutoff-cell.tiered .tier-row{padding:3px 0;border-bottom:1px dashed rgba(255,255,255,.08)}
.cutoff-cell.tiered .tier-row:last-child{border-bottom:0}
.cutoff-cell.tiered .tier-label{font-weight:700;letter-spacing:.02em;margin-right:4px}
.tier-opt,.cutoff-cell.tiered .tier-opt .tier-label{color:#4ade80}
.tier-mid,.cutoff-cell.tiered .tier-mid .tier-label{color:#fbbf24}
.tier-avoid,.cutoff-cell.tiered .tier-avoid .tier-label{color:#f87171}
.tier-mark{font-weight:700;margin-left:4px;font-size:14px}
.cutoff-cell{vertical-align:top}
.cutoff-cell .cutoff-hd{font-size:11px;color:#ffffff;letter-spacing:.03em;margin-bottom:2px;font-weight:600}
.cutoff-cell .winpct{font-size:14px;font-weight:700;color:#e4e7eb}
.cutoff-cell .winpct .muted{color:#ffffff;font-weight:400}
.acc-overall td{background:#14161c;border-top:2px solid #262a33}
.mkt-name{margin-left:4px;color:#ffffff;font-size:12px}
.mkt-badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;
           font-weight:700;color:#fff;letter-spacing:.03em}
.tier-hdr{display:flex;align-items:center;gap:8px}
.tier-1{color:#f97316}.tier-2{color:#60a5fa}
.btn-log{background:#2563eb;color:#fff;border:0;padding:6px 12px;border-radius:6px;
         font-size:12px;cursor:pointer;font-weight:600}
.btn-log:hover{background:#1d4ed8}
.btn-log:disabled{background:#374151;cursor:not-allowed}
.stats-bar{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
           gap:10px;margin-bottom:18px}
.stat{background:#1a1d24;border:1px solid #262a33;border-radius:10px;padding:12px}
.stat .l{font-size:11px;color:#ffffff;text-transform:uppercase;letter-spacing:.05em}
.stat .v{font-size:20px;font-weight:700;margin-top:4px}
details summary{cursor:pointer;list-style:none}
details summary::-webkit-details-marker{display:none}

/* ── Summary stat tiles (static, derived from today's scores) ─────────────── */
.stat-tiles{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:14px}
.stat-tile{background:#1a1d24;border:1px solid #262a33;border-radius:10px;padding:14px 16px}
.stat-tile .l{font-size:11px;color:#ffffff;text-transform:uppercase;letter-spacing:.05em}
.stat-tile .v{font-size:22px;font-weight:700;margin-top:4px;font-variant-numeric:tabular-nums}
.stat-tile .v.ev-pos{color:#4ade80}
.stat-tile .v.ev-neg{color:#f87171}
@media (max-width:720px){
  .stat-tiles{grid-template-columns:1fr}
}

/* ── Responsive card grid for actionable picks ────────────────────────────── */
.card-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px;margin-top:6px}
.bet-card{background:#1a1d24;border:1px solid #262a33;border-left:4px solid #374151;
          border-radius:10px;padding:14px 16px;transition:transform .08s ease,border-color .08s ease}
.bet-card:hover{transform:translateY(-1px)}
.bet-card.tier-1-card{border-left-color:#4ade80}
.bet-card.tier-2-card{border-left-color:#f5b301}
.bc-head{display:flex;justify-content:space-between;align-items:baseline;gap:10px;margin-bottom:4px}
.bc-match{font-weight:600;font-size:14px;color:#e4e7eb}
.bc-tier{font-size:10px;letter-spacing:.08em;color:#ffffff;border:1px solid #262a33;
         border-radius:4px;padding:2px 6px;background:#14161c;white-space:nowrap}
.tier-1-card .bc-tier{color:#4ade80;border-color:#4ade80}
.tier-2-card .bc-tier{color:#f5b301;border-color:#f5b301}
.bc-sub{color:#ffffff;font-size:12px;margin-bottom:10px;display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.bc-sub .bc-sep{color:#374151}
.bc-sub .bc-market{color:#ffffff}
.bc-sub .bc-pick{color:#e4e7eb;font-weight:600}
.bc-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px 14px}
.bc-stat{display:flex;flex-direction:column;gap:2px}
.bc-l{font-size:10px;color:#ffffff;text-transform:uppercase;letter-spacing:.04em}
.bc-v{font-size:14px;font-weight:600;font-variant-numeric:tabular-nums;color:#e4e7eb}
.bc-v.subtle{color:#ffffff;font-weight:500}
.bc-v.stake{color:#4ade80}
.bc-action{grid-column:1 / -1;margin-top:4px}
.bc-action .btn-log{width:100%}
@media (max-width:480px){
  .bc-grid{grid-template-columns:repeat(2,1fr)}
  .bc-action{grid-column:1 / -1}
}
"""

JS_TEMPLATE = """
const TRACKER = 'http://localhost:__PORT__';

// --- Live stats bar ---------------------------------------------------------
async function loadStats(){
  try{
    const r = await fetch(TRACKER + '/stats', {cache:'no-store'});
    if(!r.ok) throw new Error('stats unreachable');
    const s = await r.json();
    const bar = document.getElementById('stats');
    const rec = `${s.wins||0}-${s.losses||0}-${s.pushes||0}`;
    bar.innerHTML =
      `<div class='stat'><div class='l'>Record</div><div class='v'>${rec}</div></div>`+
      `<div class='stat'><div class='l'>Win %</div><div class='v'>${(s.win_pct||0).toFixed(1)}%</div></div>`+
      `<div class='stat'><div class='l'>Units P/L</div><div class='v'>${(s.units_pl||0).toFixed(2)}u</div></div>`+
      `<div class='stat'><div class='l'>ROI</div><div class='v'>${(s.roi_pct||0).toFixed(1)}%</div></div>`;
  }catch(e){
    document.getElementById('stats').innerHTML =
      `<div class='stat'><div class='l'>Tracker</div><div class='v' style='font-size:13px'>offline — python tracker_server.py</div></div>`;
  }
}

// --- Log Bet ---------------------------------------------------------------
document.addEventListener('click', async (e)=>{
  const btn = e.target.closest('.btn-log');
  if(!btn) return;
  const data = JSON.parse(btn.dataset.pick);
  const stake = prompt('Stake ($):', data.stake ?? 50);
  if(stake===null) return;
  try{
    const res = await fetch(TRACKER + '/log_bet', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({...data, stake: parseFloat(stake)})
    });
    if(!res.ok) throw new Error(await res.text());
    btn.textContent='✓ Logged'; btn.disabled=true;
    loadStats();
  }catch(err){ alert('Log failed: '+err.message); }
});

// --- Pending bets / result entry -------------------------------------------
async function loadPending(){
  try{
    const r = await fetch(TRACKER + '/pending_bets', {cache:'no-store'});
    if(!r.ok) return;
    const bets = await r.json();
    const box = document.getElementById('pending');
    if(!bets.length){ box.innerHTML = "<p class='muted'>No pending bets.</p>"; return; }
    box.innerHTML = bets.map(b =>
      `<div class='stat' style='display:flex;justify-content:space-between;align-items:center'>`+
      `<div><b>${b.game}</b> — ${b.bet_type} (${b.model})</div>`+
      `<div>`+
      `<button class='btn-log' data-id='${b.id}' data-r='WIN'>WIN</button> `+
      `<button class='btn-log' data-id='${b.id}' data-r='LOSS' style='background:#b91c1c'>LOSS</button> `+
      `<button class='btn-log' data-id='${b.id}' data-r='PUSH' style='background:#ffffff'>PUSH</button>`+
      `</div></div>`
    ).join('');
    box.querySelectorAll('button').forEach(btn=>{
      btn.addEventListener('click', async ()=>{
        await fetch(TRACKER+'/log_result', {method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({id: btn.dataset.id, result: btn.dataset.r})});
        loadPending(); loadStats();
      });
    });
  }catch(e){}
}
loadStats(); loadPending();
"""


def render(df: pd.DataFrame) -> str:
    today = df["date"].iloc[0] if len(df) else datetime.now().strftime("%Y-%m-%d")
    try:
        title_date = datetime.strptime(str(today), "%Y-%m-%d").strftime("%B %d, %Y")
    except Exception:
        title_date = str(today)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    accuracy_windows = _fetch_accuracy_stats()
    accuracy_html    = _build_accuracy_html(accuracy_windows)

    summary_stats = _calculate_summary_stats(df)
    summary_html  = _render_summary_tiles(summary_stats)

    cutoff_html = _build_cutoff_html()

    body = (
        "<header>"
        f"<h1>⚾ The Wizard Report — {title_date}</h1>"
        f"<div class='sub'>Generated {ts}</div>"
        "</header>"
        + summary_html
        + "<div id='stats' class='stats-bar'></div>"
        # New order per user: overall accuracy first, then optimal cutoff table,
        # THEN the Alpha Markets picks list (calibration showcase), then
        # actionable bets detail, pending results, full table.
        + accuracy_html
        + cutoff_html
        + render_alpha(df)
        + _generate_cards_html(df)
        + "<section class='card'><h2>📝 Pending Results</h2>"
        "<div id='pending'><p class='muted'>Loading…</p></div></section>"
        + render_full_table(df)
    )

    js = JS_TEMPLATE.replace("__PORT__", str(TRACKER_PORT))
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>Wizard Report — {title_date}</title>"
        f"<style>{CSS}</style>"
        "</head><body><div class='container'>"
        + body +
        "</div>"
        f"<script>{js}</script>"
        "</body></html>"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="model_scores.csv")
    ap.add_argument("--out",    default="model_report.html")
    args = ap.parse_args()

    df = pd.read_csv(args.scores)
    html = render(df)
    Path(args.out).write_text(html, encoding="utf-8")

    act = df[df["actionable"] == True]
    t1  = (act["tier"] == 1).sum()
    t2  = (act["tier"] == 2).sum()
    print(f"[OK] Wrote {args.out}  ({len(html):,} bytes)")
    print(f"     Actionable: {len(act)}  (Tier 1: {t1} | Tier 2: {t2})")
    print(f"     Alpha rows: ML={sum(df['model']=='ML')}  Totals={sum(df['model']=='Totals')}")
    print(f"     Full table: {len(df)} rows across {df['model'].nunique()} markets")


if __name__ == "__main__":
    main()
