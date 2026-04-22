"""
tools/implementations.py

Pure Python tool functions. No framework inheritance needed.
Each function takes **kwargs matching the tool's input_schema and returns a JSON string.

These are called by tool_executor() in each agent module, which dispatches
by tool name. Claude decides WHEN to call them — you just implement WHAT they do.
"""
from __future__ import annotations

import json
import logging
import smtplib
from datetime import date, datetime, timedelta
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from config.settings import (
    FILES, STATIC_FILES, STALE_THRESHOLD_DAYS, PIPELINE_DIR,
    ODDS_API_KEY, F5_ESTIMATE_RATIO, F3_ESTIMATE_RATIO, F1_ESTIMATE_RATIO,
    ODDS_API_US_BOOKMAKERS, ODDS_API_EU_BOOKMAKERS,
    GMAIL_FROM, GMAIL_PASSWORD, EMAIL_RECIPIENTS,
)

logger = logging.getLogger("wizard.daily")

# ── Legacy sklearn pickle compat ──────────────────────────────────────────────
# The stacker/calibrator pickles in models/ were saved with sklearn 1.8.0 where
# LogisticRegression's `multi_class` attribute was removed. In the current env
# (sklearn 1.7.2), predict_proba still reads self.multi_class and throws
# AttributeError. Setting a class-level default lets old pickles fall through
# to the class attribute while fresh pickles continue to override via __dict__.
try:
    from sklearn.linear_model import LogisticRegression as _LR
    if "multi_class" not in _LR.__dict__:
        _LR.multi_class = "auto"
except Exception as _e:  # sklearn missing or unexpected error — don't block imports
    logging.getLogger("wizard.daily").warning(
        f"sklearn LR multi_class compat patch skipped: {type(_e).__name__}: {_e}"
    )

# ── Ballpark coordinates ──────────────────────────────────────────────────────
# Full name → abbreviation mapping (Odds API returns full names)
TEAM_NAME_TO_ABBR: dict[str, str] = {
    'Arizona Diamondbacks': 'ARI', 'Atlanta Braves': 'ATL',
    'Baltimore Orioles': 'BAL', 'Boston Red Sox': 'BOS',
    'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CWS',
    'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE',
    'Colorado Rockies': 'COL', 'Detroit Tigers': 'DET',
    'Houston Astros': 'HOU', 'Kansas City Royals': 'KC',
    'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD',
    'Miami Marlins': 'MIA', 'Milwaukee Brewers': 'MIL',
    'Minnesota Twins': 'MIN', 'New York Mets': 'NYM',
    'New York Yankees': 'NYY', 'Oakland Athletics': 'OAK',
    'Philadelphia Phillies': 'PHI', 'Pittsburgh Pirates': 'PIT',
    'San Diego Padres': 'SD', 'San Francisco Giants': 'SF',
    'Seattle Mariners': 'SEA', 'St. Louis Cardinals': 'STL',
    'Tampa Bay Rays': 'TB', 'Texas Rangers': 'TEX',
    'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSH',
    'Athletics': 'OAK',
}

BALLPARK_COORDS: dict[str, tuple[float, float]] = {
    "ARI": (33.4455, -112.0667), "ATL": (33.8908, -84.4678),
    "BAL": (39.2838, -76.6217),  "BOS": (42.3467, -71.0972),
    "CHC": (41.9484, -87.6553),  "CWS": (41.8300, -87.6339),
    "CIN": (39.0979, -84.5082),  "CLE": (41.4959, -81.6854),
    "COL": (39.7559, -104.9942), "DET": (42.3390, -83.0485),
    "HOU": (29.7573, -95.3555),  "KC":  (39.0517, -94.4803),
    "LAA": (33.8003, -117.8827), "LAD": (34.0739, -118.2400),
    "MIA": (25.7781, -80.2197),  "MIL": (43.0280, -87.9712),
    "MIN": (44.9817, -93.2776),  "NYM": (40.7571, -73.8458),
    "NYY": (40.8296, -73.9262),  "OAK": (37.7516, -122.2005),
    "PHI": (39.9061, -75.1665),  "PIT": (40.4469, -80.0057),
    "SD":  (32.7076, -117.1570), "SF":  (37.7786, -122.3893),
    "SEA": (47.5914, -122.3325), "STL": (38.6226, -90.1928),
    "TB":  (27.7683, -82.6534),  "TEX": (32.7473, -97.0822),
    "TOR": (43.6414, -79.3894),  "WSH": (38.8730, -77.0074),
}

BET_TRACKER_COLUMNS = [
    "id", "date", "game", "model", "bet_type", "model_prob",
    "market_line", "book", "units", "result", "actual_total",
    "profit_loss", "logged_at", "notes",
]


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — Data Ingestion Tools
# ══════════════════════════════════════════════════════════════════════════════

# ── Forward-working archival (Task 1/2 of the Hardened Production refactor) ──
# Daily snapshots so the rolling-accuracy tear sheet can grow forward without
# ever needing a retroactive Pinnacle backfill. We write:
#   * data/statcast/odds_snapshot_{date}.parquet  — full games list w/ F5+NRFI
#   * data/predictions/model_scores_{date}.csv    — today's raw stacker outputs
#   * live_predictions_2026.csv                   — rolling union graded vs actuals

_ARCHIVE_DIR_ODDS  = PIPELINE_DIR / "data" / "statcast"
_ARCHIVE_DIR_PREDS = PIPELINE_DIR / "data" / "predictions"
_ACTUALS_PARQUET   = PIPELINE_DIR / "data" / "statcast" / "actuals_2026.parquet"
_LIVE_LOG          = PIPELINE_DIR / "live_predictions_2026.csv"


def _archive_odds_snapshot(games: list, target_date: str) -> None:
    """Persist today's full odds snapshot (incl. F5/NRFI) as parquet.
    Silent on failure — archival is best-effort and never blocks ingest."""
    try:
        if not games:
            return
        _ARCHIVE_DIR_ODDS.mkdir(parents=True, exist_ok=True)
        out = _ARCHIVE_DIR_ODDS / f"odds_snapshot_{target_date}.parquet"
        pd.DataFrame(games).to_parquet(out, index=False)
        logger.info(f"[Archive] odds snapshot -> {out.name} ({len(games)} games)")
    except Exception as e:
        logger.warning(f"[Archive] odds snapshot failed (non-fatal): {e}")


def archive_model_scores(target_date: str = "") -> str:
    """Copy today's model_scores.csv into data/predictions/model_scores_{date}.csv.
    Idempotent: overwrites if rerun. Returns JSON status."""
    target = target_date or date.today().isoformat()
    try:
        src = PIPELINE_DIR / FILES["model_scores"]
        if not src.exists():
            return json.dumps({"status": "SKIP", "reason": "model_scores.csv missing"})
        _ARCHIVE_DIR_PREDS.mkdir(parents=True, exist_ok=True)
        dst = _ARCHIVE_DIR_PREDS / f"model_scores_{target}.csv"
        df = pd.read_csv(src)
        df["archive_date"] = target
        df.to_csv(dst, index=False)
        return json.dumps({"status": "OK", "rows": len(df), "file": str(dst.name)})
    except Exception as e:
        logger.warning(f"[Archive] model_scores archival failed: {e}")
        return json.dumps({"status": "ERROR", "error": str(e)})


def rebuild_live_predictions_log() -> str:
    """
    Union every archived model_scores_*.csv, join with actuals_2026.parquet,
    and write live_predictions_2026.csv with one row per (market × direction)
    carrying model_prob + realized `actual` (0/1).

    The resulting schema mirrors the columns `render_report._compute_cutoffs_2026`
    reads from backtest_full_all_predictions.csv: market, model_prob, actual,
    game_date. Rows still pending (no actual yet) are dropped.
    """
    try:
        if not _ARCHIVE_DIR_PREDS.exists():
            return json.dumps({"status": "SKIP", "reason": "no predictions archive"})
        files = sorted(_ARCHIVE_DIR_PREDS.glob("model_scores_*.csv"))
        if not files:
            return json.dumps({"status": "SKIP", "reason": "no archived scores"})

        frames = []
        for fp in files:
            try:
                frames.append(pd.read_csv(fp))
            except Exception as e:
                logger.warning(f"[LiveLog] skip {fp.name}: {e}")
        if not frames:
            return json.dumps({"status": "SKIP", "reason": "all archive files unreadable"})
        preds = pd.concat(frames, ignore_index=True)

        if not _ACTUALS_PARQUET.exists():
            return json.dumps({"status": "SKIP", "reason": "actuals_2026.parquet missing"})
        actuals = pd.read_parquet(_ACTUALS_PARQUET)

        # `game` string in model_scores is "Away @ Home" (full names). Actuals
        # store abbrev codes. Join on (date, home_abbr, away_abbr) via reverse
        # lookup so historical and live agree.
        def _split_game(s):
            if not isinstance(s, str) or "@" not in s:
                return (None, None)
            a, h = [x.strip() for x in s.split("@", 1)]
            return (TEAM_NAME_TO_ABBR.get(a), TEAM_NAME_TO_ABBR.get(h))

        preds[["away_abbr", "home_abbr"]] = preds["game"].apply(
            lambda s: pd.Series(_split_game(s))
        )
        preds["game_date"] = preds.get("date", preds.get("archive_date"))

        actuals = actuals.rename(columns={"home_team": "home_abbr", "away_team": "away_abbr"})
        actuals["game_date"] = pd.to_datetime(actuals["game_date"]).dt.strftime("%Y-%m-%d")

        merged = preds.merge(
            actuals[["game_date", "home_abbr", "away_abbr",
                     "home_score_final", "away_score_final",
                     "f5_home_win", "f1_nrfi", "home_covers_rl"]],
            on=["game_date", "home_abbr", "away_abbr"], how="left",
        )

        # Derive binary `actual` per (model, pick_direction).
        def _actual(r):
            m   = str(r.get("model", "")).strip()
            d   = str(r.get("pick_direction", "")).strip().upper()
            hs  = r.get("home_score_final")
            as_ = r.get("away_score_final")
            if pd.isna(hs) or pd.isna(as_):
                return pd.NA
            if m == "ML":
                home_win = 1 if hs > as_ else 0
                return home_win if d == "HOME" else (1 - home_win)
            if m == "Totals":
                tot = hs + as_
                line = r.get("total_line")
                if pd.isna(line):
                    # fall back: parse from bet_type "Total 8.5"
                    bt = str(r.get("bet_type", ""))
                    try:
                        line = float(bt.split()[-1])
                    except Exception:
                        return pd.NA
                if tot == line:
                    return pd.NA  # push
                over_hit = 1 if tot > line else 0
                return over_hit if d == "OVER" else (1 - over_hit)
            if m == "Runline":
                rl = r.get("home_covers_rl")
                if pd.isna(rl):
                    return pd.NA
                rl = int(rl)
                return rl if d == "HOME" else (1 - rl)
            if m == "F5":
                f5w = r.get("f5_home_win")
                if pd.isna(f5w):
                    return pd.NA
                f5w = int(f5w)
                return f5w if d == "HOME" else (1 - f5w)
            if m == "NRFI":
                nr = r.get("f1_nrfi")
                if pd.isna(nr):
                    return pd.NA
                return int(nr)  # NRFI pick is always the UNDER — no direction flip
            return pd.NA

        merged["actual"] = merged.apply(_actual, axis=1)

        _MODEL_TO_CODE = {"ML": "ML", "Totals": "TOT", "Runline": "RL", "F5": "F5", "NRFI": "NR"}
        merged["market"] = merged["model"].map(_MODEL_TO_CODE)

        keep_cols = ["game_date", "market", "model_prob", "actual",
                     "home_abbr", "away_abbr", "model", "pick_direction",
                     "retail_american_odds", "edge"]
        for c in keep_cols:
            if c not in merged.columns:
                merged[c] = pd.NA
        log = merged[keep_cols].dropna(subset=["actual", "model_prob", "market"]).copy()

        # Dedupe on (game_date, market, home_abbr, away_abbr, pick_direction) —
        # keep the latest archive_date if the same pick reappears.
        log = log.drop_duplicates(
            subset=["game_date", "market", "home_abbr", "away_abbr", "pick_direction"],
            keep="last",
        )
        log.to_csv(_LIVE_LOG, index=False)
        return json.dumps({
            "status": "OK",
            "rows_written": len(log),
            "file": str(_LIVE_LOG.name),
            "markets": {m: int((log["market"] == c).sum())
                        for m, c in _MODEL_TO_CODE.items()},
        })
    except Exception as e:
        logger.warning(f"[LiveLog] rebuild failed: {e}", exc_info=True)
        return json.dumps({"status": "ERROR", "error": str(e)})


def check_stale_files(threshold_days: int = STALE_THRESHOLD_DAYS) -> str:
    cutoff = datetime.now() - timedelta(days=threshold_days)
    results, stale, missing = {}, [], []

    for key in STATIC_FILES:
        path = FILES.get(key)
        if not path or not Path(path).exists():
            results[key] = {"status": "MISSING"}
            missing.append(key)
            continue
        mtime    = datetime.fromtimestamp(Path(path).stat().st_mtime)
        age_days = (datetime.now() - mtime).days
        if mtime < cutoff:
            results[key] = {"status": "STALE", "age_days": age_days,
                            "warning": f"⚠️ STALE: {key} last updated {age_days} days ago. Upload fresh data Sunday."}
            stale.append(key)
        else:
            results[key] = {"status": "OK", "age_days": age_days}

    return json.dumps({"stale_files": stale, "missing_files": missing, "details": results})


# ── Odds API helpers ──────────────────────────────────────────────────────────
# All market math lives in these helpers so fetch_odds_api stays declarative.
#   * De-vig uses the classic two-way normalization 1/dec / (1/dec_home + 1/dec_away)
#   * Strict line matching: runlines are filtered to ±1.5 only; totals use the modal
#     line across books (NOT max/min) and compare odds at that line only.

def _american_to_decimal_local(american) -> Optional[float]:
    if american is None:
        return None
    try:
        a = float(american)
    except (TypeError, ValueError):
        return None
    if a == 0:
        return None
    return (a / 100.0) + 1.0 if a > 0 else (100.0 / abs(a)) + 1.0


def _devig_two_way(odds_a, odds_b) -> tuple:
    """Two-way de-vig → (p_a, p_b). Returns (None, None) if either side missing."""
    dec_a = _american_to_decimal_local(odds_a)
    dec_b = _american_to_decimal_local(odds_b)
    if dec_a is None or dec_b is None:
        return None, None
    ia, ib = 1.0 / dec_a, 1.0 / dec_b
    s = ia + ib
    if s <= 0:
        return None, None
    return round(ia / s, 4), round(ib / s, 4)


def _collect_books(event: dict, allowed: set, market_key: str) -> list:
    """[(book, market_dict), ...] for event filtered to allowed books + market."""
    out = []
    for bm in event.get("bookmakers", []) or []:
        if bm.get("key") not in allowed:
            continue
        for mkt in bm.get("markets", []) or []:
            if mkt.get("key") == market_key:
                out.append((bm["key"], mkt))
    return out


def _extract_h2h(books: list, home_name: str, away_name: str) -> dict:
    """{book: (home_price, away_price)} from h2h outcomes."""
    res = {}
    for book, mkt in books:
        h = a = None
        for o in mkt.get("outcomes", []) or []:
            if o.get("name") == home_name:   h = o.get("price")
            elif o.get("name") == away_name: a = o.get("price")
        res[book] = (h, a)
    return res


def _extract_totals(books: list) -> dict:
    """{book: {point: (over_price, under_price)}} — preserves all posted lines."""
    out = {}
    for book, mkt in books:
        by_point: dict = {}
        for o in mkt.get("outcomes", []) or []:
            pt, side, price = o.get("point"), o.get("name", ""), o.get("price")
            if pt is None:
                continue
            pair = by_point.setdefault(float(pt), [None, None])
            if side == "Over":  pair[0] = price
            if side == "Under": pair[1] = price
        out[book] = {pt: tuple(v) for pt, v in by_point.items()}
    return out


def _extract_runline(books: list, home_name: str, away_name: str) -> dict:
    """{book: (home_-1.5_price, away_+1.5_price)}. STRICT ±1.5 filter — alt lines dropped."""
    res = {}
    for book, mkt in books:
        h15 = a15 = None
        for o in mkt.get("outcomes", []) or []:
            name, price, point = o.get("name"), o.get("price"), o.get("point")
            if point is None:
                continue
            try:
                pt = float(point)
            except (TypeError, ValueError):
                continue
            if name == home_name and abs(pt + 1.5) < 1e-6:
                h15 = price
            elif name == away_name and abs(pt - 1.5) < 1e-6:
                a15 = price
        res[book] = (h15, a15)
    return res


def _extract_nrfi(books: list) -> dict:
    """{book: (under_0.5_price, over_0.5_price)} — NRFI / YRFI at the 0.5 line only."""
    res = {}
    for book, mkt in books:
        u = o = None
        for oc in mkt.get("outcomes", []) or []:
            pt, side, price = oc.get("point"), oc.get("name", ""), oc.get("price")
            if pt is None:
                continue
            try:
                if abs(float(pt) - 0.5) > 1e-6:
                    continue
            except (TypeError, ValueError):
                continue
            if side == "Under": u = price
            elif side == "Over": o = price
        res[book] = (u, o)
    return res


def _best_nrfi(prices: dict) -> tuple:
    """Best NRFI (Under 0.5) American price across books, and paired YRFI from the same book."""
    best_book = None
    best_u = None
    for book, (u, _o) in prices.items():
        if u is None:
            continue
        if best_u is None or u > best_u:
            best_u = u
            best_book = book
    if best_book is None:
        return None, None
    paired_o = prices[best_book][1]
    return best_u, paired_o


def _best_two_way(prices: dict) -> tuple:
    """Best (home, away) American odds across books — highest price per side (bettor-favorable)."""
    bh = ba = None
    for _book, pair in prices.items():
        h, a = pair
        if h is not None and (bh is None or h > bh): bh = h
        if a is not None and (ba is None or a > ba): ba = a
    return bh, ba


def _modal_totals_line(totals: dict) -> Optional[float]:
    """Most common total line across books. Ties → lowest line (conservative)."""
    from collections import Counter
    c: Counter = Counter()
    for _book, by_pt in totals.items():
        for pt in by_pt:
            c[pt] += 1
    if not c:
        return None
    top = max(c.values())
    return min(pt for pt, n in c.items() if n == top)


def _best_totals_at_line(totals: dict, line: Optional[float]) -> tuple:
    """Best (over, under) American odds at the given line only. Never compares across lines."""
    if line is None:
        return None, None
    bo = bu = None
    for _book, by_pt in totals.items():
        pair = by_pt.get(line)
        if not pair:
            continue
        o, u = pair
        if o is not None and (bo is None or o > bo): bo = o
        if u is not None and (bu is None or u > bu): bu = u
    return bo, bu


def fetch_odds_api(game_date: str = "") -> str:
    """
    Dual-region odds ingestion (two-phase, because The Odds API only exposes F5
    markets through the per-event endpoint).

    Phase 1 — bulk /odds/ call per region for core markets:
        h2h, totals, spreads           on DK/FD/BetMGM  (US retail)
        h2h, totals, spreads           on Pinnacle      (EU sharp → P_true)

    Phase 2 — per-event /events/{id}/odds call per region for F5 markets:
        h2h_1st_5_innings, totals_1st_5_innings   on DK/FD/BetMGM
        h2h_1st_5_innings, totals_1st_5_innings   on Pinnacle

    Strict line matching:
      * Runline: ONLY the standard -1.5 / +1.5 spread is considered. Alt lines ignored.
      * Totals (full-game + F5): modal line across books; odds are compared only
        at that single line — never across different totals.

    HALT if the US phase-1 call returns no events. Pinnacle failure + any per-event
    F5 failure are non-fatal → those columns fall back to None.
    """
    target    = game_date or date.today().isoformat()
    base_odds = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
    base_evt  = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{eid}/odds"
    CORE_MKTS = "h2h,totals,spreads"
    F5_MKTS   = "h2h_1st_5_innings,totals_1st_5_innings,totals_1st_1_innings"

    def _bulk(region: str, books: list) -> tuple:
        try:
            r = requests.get(base_odds, timeout=20, params={
                "apiKey": ODDS_API_KEY, "regions": region, "markets": CORE_MKTS,
                "bookmakers": ",".join(books), "oddsFormat": "american", "dateFormat": "iso",
            })
            r.raise_for_status()
            return r.json(), None
        except Exception as e:
            return None, str(e)

    def _per_event(eid: str, region: str, books: list) -> Optional[dict]:
        try:
            r = requests.get(base_evt.format(eid=eid), timeout=15, params={
                "apiKey": ODDS_API_KEY, "regions": region, "markets": F5_MKTS,
                "bookmakers": ",".join(books), "oddsFormat": "american",
            })
            r.raise_for_status()
            return r.json()
        except Exception:
            return None   # Silent — F5 rows fall through as None.

    us_raw, us_err = _bulk("us", ODDS_API_US_BOOKMAKERS)
    eu_raw, eu_err = _bulk("eu", ODDS_API_EU_BOOKMAKERS)

    if not us_raw:
        return json.dumps({
            "status": "HALT",
            "error":  f"No US retail lines available (err={us_err}). Aborting. Try again after 10 AM.",
            "games":  [],
        })

    eu_by_id = {ev.get("id"): ev for ev in (eu_raw or [])}
    us_set   = set(ODDS_API_US_BOOKMAKERS)
    eu_set   = set(ODDS_API_EU_BOOKMAKERS)

    games = []
    for ev in us_raw:
        home, away = ev.get("home_team", ""), ev.get("away_team", "")
        eu_ev      = eu_by_id.get(ev.get("id"), {}) or {}

        eid = ev.get("id")

        # ── US retail core markets (from phase-1 bulk call) ─────────────────
        us_ml  = _extract_h2h(_collect_books(ev, us_set, "h2h"), home, away)
        us_tot = _extract_totals(_collect_books(ev, us_set, "totals"))
        us_rl  = _extract_runline(_collect_books(ev, us_set, "spreads"), home, away)

        # ── US retail F5 markets (phase-2 per-event call) ───────────────────
        us_f5_ev  = _per_event(eid, "us", ODDS_API_US_BOOKMAKERS) or {}
        us_f5_ml  = _extract_h2h(_collect_books(us_f5_ev, us_set, "h2h_1st_5_innings"), home, away)
        us_f5_tot = _extract_totals(_collect_books(us_f5_ev, us_set, "totals_1st_5_innings"))

        r_ml_h, r_ml_a             = _best_two_way(us_ml)
        r_imp_h, r_imp_a           = _devig_two_way(r_ml_h, r_ml_a)
        r_tot_line                 = _modal_totals_line(us_tot)
        r_over, r_under            = _best_totals_at_line(us_tot, r_tot_line)
        r_imp_o, r_imp_u           = _devig_two_way(r_over, r_under)
        r_rl_h, r_rl_a             = _best_two_way(us_rl)
        r_imp_rlh, r_imp_rla       = _devig_two_way(r_rl_h, r_rl_a)
        r_f5_h, r_f5_a             = _best_two_way(us_f5_ml)
        r_imp_f5h, r_imp_f5a       = _devig_two_way(r_f5_h, r_f5_a)
        r_f5_tot_line              = _modal_totals_line(us_f5_tot)
        r_f5_over, r_f5_under      = _best_totals_at_line(us_f5_tot, r_f5_tot_line)
        r_imp_f5o, r_imp_f5u       = _devig_two_way(r_f5_over, r_f5_under)

        # ── Retail NRFI (Under/Over 0.5 first-inning totals) ────────────────
        us_nrfi                    = _extract_nrfi(_collect_books(us_f5_ev, us_set, "totals_1st_1_innings"))
        r_nrfi_odds, r_yrfi_odds   = _best_nrfi(us_nrfi)
        r_imp_nrfi, _r_imp_yrfi    = _devig_two_way(r_nrfi_odds, r_yrfi_odds)

        # ── Pinnacle core markets (from phase-1 bulk call) ──────────────────
        eu_ml  = _extract_h2h(_collect_books(eu_ev, eu_set, "h2h"), home, away)
        eu_tot = _extract_totals(_collect_books(eu_ev, eu_set, "totals"))
        eu_rl  = _extract_runline(_collect_books(eu_ev, eu_set, "spreads"), home, away)

        # ── Pinnacle F5 markets (phase-2 per-event call) ────────────────────
        eu_f5_ev  = _per_event(eid, "eu", ODDS_API_EU_BOOKMAKERS) or {}
        eu_f5_ml  = _extract_h2h(_collect_books(eu_f5_ev, eu_set, "h2h_1st_5_innings"), home, away)
        eu_f5_tot = _extract_totals(_collect_books(eu_f5_ev, eu_set, "totals_1st_5_innings"))

        p_ml_h, p_ml_a             = _best_two_way(eu_ml)
        p_imp_h, p_imp_a           = _devig_two_way(p_ml_h, p_ml_a)
        p_tot_line                 = _modal_totals_line(eu_tot)
        p_over, p_under            = _best_totals_at_line(eu_tot, p_tot_line)
        p_imp_o, p_imp_u           = _devig_two_way(p_over, p_under)
        p_rl_h, p_rl_a             = _best_two_way(eu_rl)
        p_imp_rlh, p_imp_rla       = _devig_two_way(p_rl_h, p_rl_a)
        p_f5_h, p_f5_a             = _best_two_way(eu_f5_ml)
        p_imp_f5h, p_imp_f5a       = _devig_two_way(p_f5_h, p_f5_a)
        p_f5_tot_line              = _modal_totals_line(eu_f5_tot)
        p_f5_over, p_f5_under      = _best_totals_at_line(eu_f5_tot, p_f5_tot_line)
        p_imp_f5o, p_imp_f5u       = _devig_two_way(p_f5_over, p_f5_under)

        # ── Pinnacle NRFI ───────────────────────────────────────────────────
        eu_nrfi                    = _extract_nrfi(_collect_books(eu_f5_ev, eu_set, "totals_1st_1_innings"))
        p_nrfi_u, p_nrfi_o         = (None, None)
        # Pinnacle posts a single pair per event; take whichever book entry exists.
        for _bk, (_u, _o) in eu_nrfi.items():
            if _u is not None or _o is not None:
                p_nrfi_u, p_nrfi_o = _u, _o
                break
        p_imp_nrfi, _p_imp_yrfi    = _devig_two_way(p_nrfi_u, p_nrfi_o)

        # ── Legacy per-book fields (unchanged schema) ──────────────────────
        dk_pair = (us_ml.get("draftkings") or (None, None))
        fd_pair = (us_ml.get("fanduel")    or (None, None))
        dk_tot  = r_tot_line if r_tot_line in (us_tot.get("draftkings") or {}) else None
        fd_tot  = r_tot_line if r_tot_line in (us_tot.get("fanduel")    or {}) else None
        game_total = r_tot_line

        games.append({
            "game_id":    ev.get("id"),
            "home_team":  home,
            "away_team":  away,
            "commence":   ev.get("commence_time"),

            # ── Legacy fields (kept for backward compat) ──────────────────
            "game_total":      game_total,
            "dk_total":        dk_tot,
            "fd_total":        fd_tot,
            "dk_home_ml":      dk_pair[0],
            "dk_away_ml":      dk_pair[1],
            "fd_home_ml":      fd_pair[0],
            "fd_away_ml":      fd_pair[1],
            "f5_total":        round(game_total * F5_ESTIMATE_RATIO, 1) if game_total else None,
            "f3_total":        round(game_total * F3_ESTIMATE_RATIO, 1) if game_total else None,
            "f1_total":        round(game_total * F1_ESTIMATE_RATIO, 1) if game_total else None,
            "f5_estimated":    r_f5_tot_line is None,   # False when we actually fetched an F5 line
            "f3_estimated":    True,
            "f1_estimated":    True,
            "best_over_book":  "DK" if dk_tot is not None else ("FD" if fd_tot is not None else None),
            "best_under_book": "DK" if dk_tot is not None else ("FD" if fd_tot is not None else None),
            "is_coors":        "Colorado Rockies" in (home, away),

            # ── Retail full-game de-vigged ────────────────────────────────
            "retail_ml_home_odds":         r_ml_h,
            "retail_ml_away_odds":         r_ml_a,
            "Retail_Implied_Prob_home":    r_imp_h,
            "Retail_Implied_Prob_away":    r_imp_a,
            "retail_total_line":           r_tot_line,
            "retail_over_odds":            r_over,
            "retail_under_odds":           r_under,
            "Retail_Implied_Prob_over":    r_imp_o,
            "Retail_Implied_Prob_under":   r_imp_u,
            "retail_rl_home_odds":         r_rl_h,
            "retail_rl_away_odds":         r_rl_a,
            "Retail_Implied_Prob_rl_home": r_imp_rlh,
            "Retail_Implied_Prob_rl_away": r_imp_rla,

            # ── Retail F5 de-vigged ───────────────────────────────────────
            "retail_f5_ml_home_odds":       r_f5_h,
            "retail_f5_ml_away_odds":       r_f5_a,
            "Retail_Implied_Prob_f5_home":  r_imp_f5h,
            "Retail_Implied_Prob_f5_away":  r_imp_f5a,
            "retail_f5_total_line":         r_f5_tot_line,
            "retail_f5_over_odds":          r_f5_over,
            "retail_f5_under_odds":         r_f5_under,
            "Retail_Implied_Prob_f5_over":  r_imp_f5o,
            "Retail_Implied_Prob_f5_under": r_imp_f5u,

            # ── Pinnacle full-game de-vigged ──────────────────────────────
            "pinnacle_ml_home":        p_ml_h,
            "pinnacle_ml_away":        p_ml_a,
            "P_true_home":             p_imp_h,
            "P_true_away":             p_imp_a,
            "pinnacle_total_line":     p_tot_line,
            "pinnacle_over_odds":      p_over,
            "pinnacle_under_odds":     p_under,
            "P_true_over":             p_imp_o,
            "P_true_under":            p_imp_u,
            "pinnacle_rl_home_odds":   p_rl_h,
            "pinnacle_rl_away_odds":   p_rl_a,
            "P_true_rl_home":          p_imp_rlh,
            "P_true_rl_away":          p_imp_rla,

            # ── Pinnacle F5 de-vigged ─────────────────────────────────────
            "pinnacle_f5_ml_home":     p_f5_h,
            "pinnacle_f5_ml_away":     p_f5_a,
            "P_true_f5_home":          p_imp_f5h,
            "P_true_f5_away":          p_imp_f5a,
            "pinnacle_f5_total_line":  p_f5_tot_line,
            "P_true_f5_over":          p_imp_f5o,
            "P_true_f5_under":         p_imp_f5u,

            # ── NRFI (First-inning Under/Over 0.5) ────────────────────────
            "retail_nrfi_odds":            r_nrfi_odds,
            "retail_yrfi_odds":            r_yrfi_odds,
            "Retail_Implied_Prob_nrfi":    r_imp_nrfi,
            "pinnacle_nrfi_odds":          p_nrfi_u,
            "P_true_nrfi":                 p_imp_nrfi,
        })

    # Forward-working archival: persist the full games list (with F5+NRFI odds)
    # as a daily parquet so the rolling-accuracy tear sheet can grow forward.
    _archive_odds_snapshot(games, target)

    return json.dumps({
        "status":        "OK",
        "games_fetched": len(games),
        "us_error":      us_err,
        "eu_error":      eu_err,    # Non-fatal — Pinnacle columns fall back to None.
        "games":         games,
    })


def fetch_mlb_starters(game_date: str = "") -> str:
    target = game_date or date.today().isoformat()
    url    = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1, "date": target,
        "hydrate": "probablePitcher(note),team",
        "fields":  "dates,games,gamePk,teams,home,away,team,name,abbreviation,probablePitcher,fullName,id",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return json.dumps({"status": "WARNING", "error": str(e), "starters": {}})

    starters, excluded = {}, []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            gk       = str(game.get("gamePk"))
            home_d   = game.get("teams", {}).get("home", {})
            away_d   = game.get("teams", {}).get("away", {})
            home_abbr = home_d.get("team", {}).get("abbreviation", "")
            away_abbr = away_d.get("team", {}).get("abbreviation", "")
            hp        = home_d.get("probablePitcher", {})
            ap        = away_d.get("probablePitcher", {})

            if not hp or not ap:
                excluded.append(f"{away_abbr} @ {home_abbr} (TBD starter)")
                continue
            starters[gk] = {
                "home_team":    home_abbr, "away_team":    away_abbr,
                "home_starter": hp.get("fullName"), "home_pitcher_id": hp.get("id"),
                "away_starter": ap.get("fullName"), "away_pitcher_id": ap.get("id"),
            }

    return json.dumps({"status": "OK", "confirmed_games": len(starters),
                       "excluded_games": excluded, "starters": starters})


def fetch_weather(games_json: str) -> str:
    try:
        games = json.loads(games_json)
    except Exception as e:
        return json.dumps({"status": "WARNING", "error": f"Invalid games_json: {e}", "weather": {}})

    weather = {}
    for game in games:
        home    = game.get("home_team", "")
        game_id = game.get("game_id", home)
        home_key = TEAM_NAME_TO_ABBR.get(home, home)
        coords  = BALLPARK_COORDS.get(home_key)
        if not coords:
            weather[game_id] = {"status": "WARNING", "error": f"No coords for {home}"}
            continue
        lat, lon = coords
        try:
            resp = requests.get("https://api.open-meteo.com/v1/forecast", timeout=10, params={
                "latitude": lat, "longitude": lon,
                "current":  "temperature_2m,wind_speed_10m,wind_direction_10m,precipitation_probability,weather_code",
                "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "forecast_days": 1,
            })
            resp.raise_for_status()
            w = resp.json().get("current", {})
            weather[game_id] = {
                "temperature_f":             round(w.get("temperature_2m", 0)),
                "wind_speed_mph":            round(w.get("wind_speed_10m", 0)),
                "wind_direction":            w.get("wind_direction_10m"),
                "precipitation_probability": w.get("precipitation_probability"),
            }
        except Exception as e:
            weather[game_id] = {"status": "WARNING", "error": str(e)}

    return json.dumps({"status": "OK", "weather": weather})


# ══════════════════════════════════════════════════════════════════════════════
# SHARED — File I/O Tools
# ══════════════════════════════════════════════════════════════════════════════

def read_csv(file_key: str, max_rows: Optional[int] = None) -> str:
    file_key = file_key.replace(".csv", "").replace(".CSV", "")
    path = FILES.get(file_key)
    if not path or not Path(path).exists():
        return json.dumps({"status": "ERROR", "error": f"File not found: {file_key} → {path}"})
    try:
        # Cap at 500 rows by default to prevent context window overflow
        cap = max_rows if max_rows is not None else 500
        total_rows = sum(1 for _ in open(path)) - 1  # exclude header
        df = pd.read_csv(path, nrows=cap)
        result = {"status": "OK", "rows": len(df), "total_rows": total_rows,
                  "columns": list(df.columns), "records": df.to_dict(orient="records")}
        if total_rows > cap:
            result["warning"] = f"Showing {cap} of {total_rows} rows. Pass max_rows to get more."
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"status": "ERROR", "error": str(e)})


def read_csv_filtered(
    file_key: str,
    name_filter: Optional[str] = None,
    team_filter: Optional[str] = None,
    name_col: str = "Name",
    team_col: str = "Team",
) -> str:
    """Return only rows matching pitcher/batter names or team abbreviations.
    Cuts token payload by ~90% vs full read_csv on fangraphs/savant files."""
    file_key = file_key.replace(".csv", "").replace(".CSV", "")
    path = FILES.get(file_key)
    if not path or not Path(path).exists():
        return json.dumps({"status": "ERROR", "error": f"File not found: {file_key} → {path}"})
    try:
        df         = pd.read_csv(path)
        total_rows = len(df)
        names      = json.loads(name_filter) if name_filter else []
        teams      = json.loads(team_filter) if team_filter else []

        mask = pd.Series([False] * len(df), index=df.index)
        if names and name_col in df.columns:
            mask |= df[name_col].isin(names)
        if teams and team_col in df.columns:
            mask |= df[team_col].isin(teams)
        if not names and not teams:
            mask = pd.Series([True] * len(df), index=df.index)

        filtered = df[mask]
        return json.dumps({
            "status":     "OK",
            "rows":       len(filtered),
            "total_rows": total_rows,
            "filtered":   True,
            "columns":    list(filtered.columns),
            "records":    filtered.to_dict(orient="records"),
        })
    except Exception as e:
        return json.dumps({"status": "ERROR", "error": str(e)})


def write_csv(file_key: str, records_json: str, mode: str = "w") -> str:
    file_key = file_key.replace(".csv", "").replace(".CSV", "")
    if file_key == "bet_tracker":
        return json.dumps({"status": "REJECTED",
                           "error": "bet_tracker.csv is exclusively managed by the Bet Tracker Agent."})
    path = FILES.get(file_key)
    if not path:
        return json.dumps({"status": "ERROR", "error": f"Unknown file key: {file_key}"})
    try:
        records = json.loads(records_json)
        df      = pd.DataFrame(records)
        header  = (mode == "w") or not Path(path).exists()
        df.to_csv(path, mode=mode, index=False, header=header)
        return json.dumps({"status": "OK", "file": str(path), "rows_written": len(df)})
    except Exception as e:
        return json.dumps({"status": "ERROR", "error": str(e)})


def write_html(file_key: str, html_content: str) -> str:
    path = FILES.get(file_key)
    if not path:
        return json.dumps({"status": "ERROR", "error": f"Unknown file key: {file_key}"})
    try:
        Path(path).write_text(html_content, encoding="utf-8")
        return json.dumps({"status": "OK", "file": str(path), "bytes": len(html_content)})
    except Exception as e:
        return json.dumps({"status": "ERROR", "error": str(e)})


def validate_static_file(
    file_key: str,
    expected_columns: str,
    min_rows: int = 10,
    expected_size_kb: Optional[int] = None,
) -> str:
    # Normalize key — Claude sometimes passes savant_pitchers.csv instead of savant_pitchers
    file_key = file_key.replace(".csv", "").replace(".CSV", "")
    path = FILES.get(file_key)
    if not path or not Path(path).exists():
        return json.dumps({"status": "REJECTED", "error": f"File not found: {file_key}"})

    size_kb = Path(path).stat().st_size // 1024

    if size_kb == 0:
        return json.dumps({"status": "REJECTED", "error": f"File is empty (0KB): {file_key}. Re-export and re-upload."})

    # Size guard for savant files
    # Note: early season both savant files may be similar size — no size guard needed
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return json.dumps({"status": "REJECTED", "error": f"Cannot read CSV: {e}"})

    required = json.loads(expected_columns)
    missing  = [c for c in required if c not in df.columns]
    if missing:
        return json.dumps({"status": "REJECTED", "error": f"Missing columns: {missing}"})
    if len(df) < min_rows:
        return json.dumps({"status": "REJECTED",
                           "error": f"Only {len(df)} rows — expected ≥ {min_rows}."})

    return json.dumps({"status": "OK", "rows": len(df), "size_kb": size_kb,
                       "columns": list(df.columns)})


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — Scoring Engine (ML inference + Three-Part Lock)
# ══════════════════════════════════════════════════════════════════════════════
#
# generate_ml_scores() replaces the legacy heuristic scorers.  It:
#   1. Invokes score_ml_today.predict_games()        → full-game moneyline L2 prob
#   2. Invokes score_f5_today.predict_games()        → F5 home-cover L2 prob (informational)
#   3. Invokes score_run_dist_today.predict_games()  → totals + runline L2 probs
#   4. Joins model output with games.csv on (home_team, away_team)
#   5. Applies the Three-Part Lock (sanity / odds-floor / edge) per market
#   6. Computes Kelly dollar stake ($2,000 bank, cap $50, $1 floor)
#   7. Writes model_scores.csv and returns ONLY a compact actionable summary
#
# Design invariant: the heavy DataFrame work happens inside this tool; the LLM
# only ever sees a small JSON payload (counts + actionable pick list).

_KELLY_BANK       = 2000.0
_KELLY_CAP        = 50.0
_TIER1_EDGE       = 0.030
_TIER2_EDGE       = 0.010
_SANITY_THRESHOLD = 0.04
_ODDS_FLOOR       = -225


def _american_to_decimal(odds: float) -> float:
    return 1.0 + (odds / 100.0) if odds >= 0 else 1.0 + (100.0 / abs(odds))


def _kelly_stake(model_prob: float, american_odds: float, tier: int) -> int:
    """Quarter-Kelly (tier 1) / Eighth-Kelly (tier 2), $2000 bank, $50 cap, $1 floor."""
    if american_odds is None or model_prob is None:
        return 0
    b = _american_to_decimal(float(american_odds)) - 1.0
    if b <= 0:
        return 0
    f_star = (b * model_prob - (1.0 - model_prob)) / b
    if f_star <= 0:
        return 0
    mult    = 0.25 if tier == 1 else 0.125
    raw     = _KELLY_BANK * f_star * mult
    capped  = min(raw, _KELLY_CAP)
    rounded = int(round(capped))
    return max(rounded, 1)


def _classify_edge(edge: float) -> Optional[int]:
    if edge is None:
        return None
    if edge >= _TIER1_EDGE:
        return 1
    if edge >= _TIER2_EDGE:
        return 2
    return None


def _safe_float(v) -> Optional[float]:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _invoke_scorer(module_name: str, date_str: str) -> pd.DataFrame:
    """Import a score_*_today module fresh and run predict_games(date_str).
    Scoring scripts use cwd-relative paths, so we chdir into PIPELINE_DIR for the call."""
    import sys, os, io, contextlib
    prev_cwd = os.getcwd()
    prev_path = list(sys.path)
    try:
        os.chdir(str(PIPELINE_DIR))
        if str(PIPELINE_DIR) not in sys.path:
            sys.path.insert(0, str(PIPELINE_DIR))
        sys.modules.pop(module_name, None)
        with contextlib.redirect_stdout(io.StringIO()):
            mod = __import__(module_name)
            df  = mod.predict_games(date_str)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    finally:
        os.chdir(prev_cwd)
        sys.path[:] = prev_path


def _evaluate_pick(
    model_prob:   float,
    p_true:       Optional[float],
    retail_imp:   Optional[float],
    retail_odds:  Optional[float],
    skip_sanity:  bool = False,
    sanity_margin: float = 0.04,
) -> dict:
    """Apply the Three-Part Lock and compute tier + stake.

    When skip_sanity=True (F5 only — Pinnacle does not post F5 moneylines), the
    Pinnacle sanity gate is bypassed and actionability collapses to a Two-Part
    Lock: Odds Floor + Edge tier. `sanity_check_pass` is reported as True in
    that case so downstream audits can still see the bypass happened.

    sanity_margin overrides the default 0.04 window (|model - P_true| <= margin).
    Markets where the stacker's calibrated edge is genuinely wider than Pinnacle's
    anchor (e.g. NRFI — OOF ECE 0.0095) use a larger margin so real edge isn't
    crushed, while Pinnacle still acts as a backstop against late-scratch data
    corruption.
    """
    if skip_sanity:
        sanity_pass = True
    else:
        sanity_pass = False if p_true is None else (abs(model_prob - p_true) <= sanity_margin)

    odds_pass = False if retail_odds is None else (retail_odds >= _ODDS_FLOOR)
    edge      = None if retail_imp is None else (model_prob - retail_imp)
    tier      = _classify_edge(edge) if edge is not None else None
    stake     = 0
    if sanity_pass and odds_pass and tier is not None:
        stake = _kelly_stake(model_prob, retail_odds, tier)
    actionable = bool(sanity_pass and odds_pass and tier is not None and stake >= 1)
    return {
        "model_prob":           round(model_prob, 4),
        "P_true":               None if p_true is None else round(p_true, 4),
        "Retail_Implied_Prob":  None if retail_imp is None else round(retail_imp, 4),
        "edge":                 None if edge is None else round(edge, 4),
        "retail_american_odds": retail_odds,
        "sanity_check_pass":    bool(sanity_pass),
        "odds_floor_pass":      bool(odds_pass),
        "tier":                 tier,
        "dollar_stake":         int(stake) if stake >= 1 else None,
        "actionable":           actionable,
    }


def generate_ml_scores(game_date: str = "") -> str:
    """
    Run all three ML scoring scripts for game_date, join with games.csv,
    apply the Three-Part Lock + Kelly staking, write model_scores.csv,
    and return a compact actionable summary.

    Returns JSON with keys:
      status, games_scored, actionable, tier1, tier2,
      failed_sanity, failed_odds_floor, failed_edge,
      actionable_picks (list of ≤N compact dicts)
    """
    target = game_date or date.today().isoformat()

    # 1. Games + market context
    games_path = FILES.get("games")
    if not games_path or not Path(games_path).exists():
        return json.dumps({"status": "ERROR",
                           "error": "games.csv not found — Agent 1 must run first."})
    games_df = pd.read_csv(games_path)
    if len(games_df) == 0:
        return json.dumps({"status": "ERROR", "error": "games.csv is empty."})

    # 2. Invoke all three scorers (ML, F5, Run-Dist)
    try:
        ml_df   = _invoke_scorer("score_ml_today",       target)
        rd_df   = _invoke_scorer("score_run_dist_today", target)
        f5_df   = _invoke_scorer("score_f5_today",       target)
        nrfi_df = _invoke_scorer("score_nrfi_today",     target)
    except Exception as e:
        return json.dumps({"status": "ERROR",
                           "error": f"Scoring inference failed: {type(e).__name__}: {e}"})

    def _keyed(df: pd.DataFrame, cols: list[str]) -> dict:
        """Index a scoring frame by (home_team, away_team) for O(1) lookup."""
        out = {}
        if df is None or len(df) == 0:
            return out
        for _, r in df.iterrows():
            key = (str(r.get("home_team", "")).strip(),
                   str(r.get("away_team", "")).strip())
            out[key] = {c: _safe_float(r.get(c)) for c in cols}
        return out

    ml_by = _keyed(ml_df, ["stacker_l2"])                             # p_home_win
    rd_by = _keyed(rd_df, ["p_over_final", "p_home_cover_final",
                           "lam_home",     "lam_away",     "total_line"])
    f5_by = _keyed(f5_df, ["stacker_l2"])                             # p_f5_home_cover
    nrfi_by = _keyed(nrfi_df, ["p_stk_nrfi"])                          # p_no_run_first_inning

    # 3. Build pick rows — one per market per game
    picks: list[dict] = []
    counters = {"sanity_fail": 0, "odds_fail": 0, "edge_fail": 0,
                "tier1": 0, "tier2": 0, "actionable": 0}

    for _, g in games_df.iterrows():
        home       = str(g.get("home_team", "")).strip()
        away       = str(g.get("away_team", "")).strip()
        game_label = g.get("game_label") or f"{away} @ {home}"
        # Scorer dataframes are keyed on abbreviations (e.g. "CLE"); games.csv
        # carries Odds-API full names (e.g. "Cleveland Guardians"). Normalize.
        home_abbr  = TEAM_NAME_TO_ABBR.get(home, home)
        away_abbr  = TEAM_NAME_TO_ABBR.get(away, away)
        key        = (home_abbr, away_abbr)

        # ─── MODEL 1: ML (full-game moneyline) ──────────────────────────────
        ml_row = ml_by.get(key)
        if ml_row and ml_row.get("stacker_l2") is not None:
            p_home = float(ml_row["stacker_l2"])
            # Both directions evaluated — pick side with higher model prob.
            for side, prob, p_true_col, imp_col, odds_col in [
                ("HOME", p_home,      "P_true_home", "Retail_Implied_Prob_home", "retail_ml_home_odds"),
                ("AWAY", 1.0 - p_home, "P_true_away", "Retail_Implied_Prob_away", "retail_ml_away_odds"),
            ]:
                # ML — 2026 backtest: AUC 0.707, ECE 0.045, 28 bets @ +50% ROI
                # under Two-Part Lock. Widen sanity margin to 0.10 so real edge
                # vs. Pinnacle isn't crushed by the 0.04 default.
                ev = _evaluate_pick(prob,
                                    _safe_float(g.get(p_true_col)),
                                    _safe_float(g.get(imp_col)),
                                    _safe_float(g.get(odds_col)),
                                    sanity_margin=0.10)
                picks.append({"date": target, "game": game_label,
                              "model": "ML", "bet_type": "Moneyline",
                              "pick_direction": side, **ev})

        # ─── MODEL 2: Totals (Over / Under) ─────────────────────────────────
        rd_row = rd_by.get(key)
        if rd_row and rd_row.get("p_over_final") is not None:
            p_over = float(rd_row["p_over_final"])
            for side, prob, p_true_col, imp_col, odds_col in [
                ("OVER",  p_over,        "P_true_over",  "Retail_Implied_Prob_over",  "retail_over_odds"),
                ("UNDER", 1.0 - p_over,  "P_true_under", "Retail_Implied_Prob_under", "retail_under_odds"),
            ]:
                # Totals — 2026 backtest: AUC 0.685, ECE 0.080, 61 bets @ +38% ROI
                # under Two-Part Lock. Widen sanity margin to 0.08 (matches ECE).
                ev = _evaluate_pick(prob,
                                    _safe_float(g.get(p_true_col)),
                                    _safe_float(g.get(imp_col)),
                                    _safe_float(g.get(odds_col)),
                                    sanity_margin=0.08)
                picks.append({"date": target, "game": game_label,
                              "model": "Totals",
                              "bet_type": f"Total {_safe_float(g.get('retail_total_line')) or rd_row.get('total_line')}",
                              "pick_direction": side, **ev})

        # ─── MODEL 3: Runline (−1.5 / +1.5) ─────────────────────────────────
        # Retail odds come from Agent 1's dual-region fetch (DK/FD/BetMGM best price,
        # strict ±1.5 filter). Pinnacle gives us P_true. If retail RL is missing from
        # the API response, the pick falls through the lock as non-actionable.
        if rd_row and rd_row.get("p_home_cover_final") is not None:
            p_rl_home = float(rd_row["p_home_cover_final"])
            for side, prob, p_true_col, imp_col, odds_col in [
                ("HOME -1.5", p_rl_home,        "P_true_rl_home",
                 "Retail_Implied_Prob_rl_home", "retail_rl_home_odds"),
                ("AWAY +1.5", 1.0 - p_rl_home,  "P_true_rl_away",
                 "Retail_Implied_Prob_rl_away", "retail_rl_away_odds"),
            ]:
                # Runline — 2026 backtest: AUC 0.662 but ECE 0.284 (severe
                # miscalibration). Under Two-Part Lock RL lost money (44.7%
                # win rate, -3% ROI). Keep strict 0.04 sanity — Pinnacle is
                # actively protecting us here.
                ev = _evaluate_pick(prob,
                                    _safe_float(g.get(p_true_col)),
                                    _safe_float(g.get(imp_col)),
                                    _safe_float(g.get(odds_col)),
                                    sanity_margin=0.04)
                picks.append({"date": target, "game": game_label,
                              "model": "Runline", "bet_type": "Runline -1.5",
                              "pick_direction": side, **ev})

        # ─── MODEL 4: F5 Moneyline — TWO-PART LOCK (sanity bypassed) ────────
        # Pinnacle does NOT broadcast F5 moneylines through The Odds API, so
        # P_true_f5_* is structurally always None. We trust our F5 stacker's
        # calibration and drop the sanity gate for this market only — pick is
        # actionable on Odds Floor + Edge alone (retail F5 line must exist).
        #
        # NOTE: retail_f5_total_line / P_true_f5_over are ingested by Agent 1
        # for schema completeness, but we do NOT have an F5 totals model yet —
        # those columns are deliberately not consumed here.
        f5_row = f5_by.get(key)
        if f5_row and f5_row.get("stacker_l2") is not None:
            p_f5 = float(f5_row["stacker_l2"])
            for side, prob, p_true_col, imp_col, odds_col in [
                ("HOME", p_f5,       "P_true_f5_home",
                 "Retail_Implied_Prob_f5_home", "retail_f5_ml_home_odds"),
                ("AWAY", 1.0 - p_f5, "P_true_f5_away",
                 "Retail_Implied_Prob_f5_away", "retail_f5_ml_away_odds"),
            ]:
                ev = _evaluate_pick(prob,
                                    _safe_float(g.get(p_true_col)),
                                    _safe_float(g.get(imp_col)),
                                    _safe_float(g.get(odds_col)),
                                    skip_sanity=True)
                picks.append({"date": target, "game": game_label,
                              "model": "F5", "bet_type": "F5 Moneyline",
                              "pick_direction": side, **ev})

        # ─── MODEL 5: NRFI (No Run First Inning) — SHADOW MODE ─────────────────
        # Feature drift detected in 2026: pitcher/batter distributions radically
        # diverge from 2023-2025 training, causing model to hallucinate 85% probs.
        # Backtest accuracy dropped 76% → 44%, well below base rate.
        # SHADOW MODE: Revert to default sanity_margin=0.04 so all predictions
        # fail the Three-Part Lock and produce no actionable picks, while still
        # logging raw outputs to model_scores.csv for monitoring and retraining.
        # Once 2026 data stabilizes (May+), will retrain and re-enable.
        nrfi_row = nrfi_by.get(key)
        if nrfi_row and nrfi_row.get("p_stk_nrfi") is not None:
            p_nrfi  = float(nrfi_row["p_stk_nrfi"])
            p_true_nrfi     = _safe_float(g.get("P_true_nrfi"))
            retail_imp_nrfi = _safe_float(g.get("Retail_Implied_Prob_nrfi"))
            r_nrfi_odds_g   = _safe_float(g.get("retail_nrfi_odds"))
            r_yrfi_odds_g   = _safe_float(g.get("retail_yrfi_odds"))
            # NRFI / YRFI — 2026 backtest: AUC 0.528 (≈ noise), ECE 0.213.
            # Severe covariate shift (batting_matchup_edge +1803%,
            # sp_k_pct_diff -233%). Keep strict 0.04 sanity (Shadow Mode).
            ev_n = _evaluate_pick(p_nrfi, p_true_nrfi, retail_imp_nrfi,
                                  r_nrfi_odds_g, sanity_margin=0.04)
            picks.append({"date": target, "game": game_label,
                          "model": "NRFI", "bet_type": "NRFI",
                          "pick_direction": "NRFI", **ev_n})
            # YRFI side — flip all probabilities
            p_true_yrfi   = None if p_true_nrfi     is None else round(1.0 - p_true_nrfi, 4)
            retail_imp_y  = None if retail_imp_nrfi is None else round(1.0 - retail_imp_nrfi, 4)
            ev_y = _evaluate_pick(1.0 - p_nrfi, p_true_yrfi, retail_imp_y,
                                  r_yrfi_odds_g, sanity_margin=0.04)
            picks.append({"date": target, "game": game_label,
                          "model": "NRFI", "bet_type": "YRFI",
                          "pick_direction": "YRFI", **ev_y})

    # 4. Tally gate-failure breakdown + tier counts
    for p in picks:
        if p["actionable"]:
            counters["actionable"] += 1
            if p["tier"] == 1: counters["tier1"] += 1
            if p["tier"] == 2: counters["tier2"] += 1
            continue
        if not p["sanity_check_pass"]: counters["sanity_fail"] += 1
        if not p["odds_floor_pass"]:   counters["odds_fail"]   += 1
        if p["tier"] is None and p["edge"] is not None: counters["edge_fail"] += 1

    # 5. Persist full frame to model_scores.csv (all picks, gates + stakes)
    output_cols = ["date", "game", "model", "bet_type", "pick_direction",
                   "model_prob", "P_true", "Retail_Implied_Prob", "edge",
                   "retail_american_odds",
                   "sanity_check_pass", "odds_floor_pass",
                   "tier", "dollar_stake", "actionable"]
    out_df = pd.DataFrame(picks, columns=output_cols) if picks else pd.DataFrame(columns=output_cols)
    out_path = FILES["model_scores"]
    out_df.to_csv(out_path, index=False)

    # 5b. Append today's ACTIONABLE picks to the persistent ledger so the
    #     Auto-Reconciliation Engine can grade them once actuals arrive.
    #     Idempotent: we delete any existing rows for this date before appending,
    #     so re-running Agent 3 on the same date does not double-count bets.
    _append_to_ledger(target, games_df, [p for p in picks if p["actionable"]])

    # 6. Return compact actionable summary (LLM-safe payload)
    actionable_picks = [
        {k: p[k] for k in ("game", "model", "bet_type", "pick_direction",
                           "model_prob", "P_true", "Retail_Implied_Prob",
                           "edge", "retail_american_odds", "tier", "dollar_stake")}
        for p in picks if p["actionable"]
    ]
    actionable_picks.sort(key=lambda r: (r["tier"], -(r["edge"] or 0)))

    return json.dumps({
        "status":             "OK",
        "date":               target,
        "games_scored":       int(len(games_df)),
        "total_picks":        int(len(picks)),
        "actionable":         counters["actionable"],
        "tier1":              counters["tier1"],
        "tier2":              counters["tier2"],
        "failed_sanity":      counters["sanity_fail"],
        "failed_odds_floor":  counters["odds_fail"],
        "failed_edge":        counters["edge_fail"],
        "model_scores_path":  str(out_path),
        "actionable_picks":   actionable_picks,
    })


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 5 — Notification Tool
# ══════════════════════════════════════════════════════════════════════════════

def _render_report_pdf(html_path: Path, pdf_path: Path) -> bool:
    """Render an HTML file to PDF via playwright/chromium. Returns True on success."""
    logger.info(f"[PDF] rendering {html_path.name} → {pdf_path.name}")
    try:
        from playwright.sync_api import sync_playwright
        url = html_path.resolve().as_uri()
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url, wait_until="networkidle")
            page.pdf(
                path=str(pdf_path),
                format="Letter",
                print_background=True,
                margin={"top": "0.4in", "bottom": "0.4in", "left": "0.3in", "right": "0.3in"},
            )
            browser.close()
        size = pdf_path.stat().st_size if pdf_path.exists() else 0
        logger.info(f"[PDF] rendered {size:,} bytes")
        return pdf_path.exists() and size > 1024
    except Exception as e:
        logger.warning(f"[PDF] render failed: {type(e).__name__}: {e}", exc_info=True)
        return False


def send_email(subject: str, body: str, attach_report: bool = True) -> str:
    """
    Send the Wizard Picks email.

    BODY is the rendered HTML report itself (model_report.html), not the `body`
    argument — Agent 5's composed summary is ignored for delivery content. A
    PDF rendered from the same HTML is attached.

    The `body` parameter is preserved in the signature for schema compatibility
    and is appended as a short plaintext fallback for non-HTML mail clients.
    """
    report_path = Path(FILES["model_report"])
    pdf_path    = report_path.with_suffix(".pdf")

    # ── Build MIME envelope ────────────────────────────────────────────────
    msg           = MIMEMultipart("mixed")
    msg["From"]   = GMAIL_FROM
    msg["To"]     = ", ".join(EMAIL_RECIPIENTS)
    msg["Subject"]= subject

    # multipart/alternative: plaintext fallback + full HTML report as body
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(body or "The Wizard Report is attached (HTML body + PDF).", "plain"))
    if report_path.exists():
        html_content = report_path.read_text(encoding="utf-8")
        alt.attach(MIMEText(html_content, "html", _charset="utf-8"))
    msg.attach(alt)

    # ── PDF attachment ─────────────────────────────────────────────────────
    pdf_ok = False
    if attach_report and report_path.exists():
        pdf_ok = _render_report_pdf(report_path, pdf_path)
        if pdf_ok:
            with open(pdf_path, "rb") as f:
                part = MIMEApplication(f.read(), _subtype="pdf", Name="model_report.pdf")
            part["Content-Disposition"] = 'attachment; filename="model_report.pdf"'
            msg.attach(part)

    # ── SMTP send ──────────────────────────────────────────────────────────
    last_err = None
    for attempt in range(1, 3):
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
                s.login(GMAIL_FROM, GMAIL_PASSWORD)
                s.sendmail(GMAIL_FROM, EMAIL_RECIPIENTS, msg.as_string())
            return json.dumps({
                "status":     "OK",
                "recipients": EMAIL_RECIPIENTS,
                "subject":    subject,
                "attempt":    attempt,
                "body":       "html-report-embedded",
                "pdf":        "attached" if pdf_ok else "skipped",
            })
        except Exception as e:
            last_err = str(e)

    return json.dumps({"status": "WARNING",
                       "error": f"Email failed after 2 attempts: {last_err}. Report available locally."})


# ══════════════════════════════════════════════════════════════════════════════
# AUTO-RECONCILIATION ENGINE
# ──────────────────────────────────────────────────────────────────────────────
# Replaces manual bet-tracking (legacy Agent 6). Picks are appended to the
# master ledger by Agent 3, graded daily against actuals_2026.parquet by
# auto_grade_historical_picks(), and aggregated into 7d / 28d / YTD windows by
# compute_rolling_accuracy() for the HTML tear sheet.
# ══════════════════════════════════════════════════════════════════════════════

LEDGER_COLUMNS = [
    "date", "game", "home_abbr", "away_abbr",
    "model", "bet_type", "pick_direction",
    "model_prob", "P_true", "Retail_Implied_Prob", "edge",
    "retail_american_odds", "tier", "dollar_stake",
    "result", "profit_loss", "graded_at",
]


def _load_ledger() -> pd.DataFrame:
    p = Path(FILES["historical_picks"])
    str_cols = {"result": str, "graded_at": str, "game": str,
                "bet_type": str, "pick_direction": str, "model": str,
                "home_abbr": str, "away_abbr": str, "date": str}
    if not p.exists() or p.stat().st_size == 0:
        df = pd.DataFrame(columns=LEDGER_COLUMNS)
        # Ensure object dtype on string columns so WIN/LOSS writes don't fail.
        for c, _ in str_cols.items():
            df[c] = df[c].astype(object)
        return df
    df = pd.read_csv(p, dtype={c: object for c in str_cols})
    for c in LEDGER_COLUMNS:
        if c not in df.columns:
            df[c] = None
    # Coerce string-like columns to object so grader writes don't hit LossySetitem.
    for c in str_cols:
        df[c] = df[c].astype(object).where(df[c].notna(), "")
    return df[LEDGER_COLUMNS]


def _save_ledger(df: pd.DataFrame) -> None:
    df.to_csv(FILES["historical_picks"], index=False)


def _append_to_ledger(target_date: str, games_df: pd.DataFrame, actionable: list[dict]) -> None:
    """Append today's actionable picks to the master ledger, keyed by (date, game,
    model, bet_type, pick_direction). Existing rows for `target_date` are wiped
    first so re-runs don't duplicate."""
    if not actionable:
        return

    # Build game_label → (home_abbr, away_abbr) lookup so grader can join actuals.
    label_to_abbrs: dict = {}
    for _, g in games_df.iterrows():
        home = str(g.get("home_team", "")).strip()
        away = str(g.get("away_team", "")).strip()
        label = g.get("game_label") or f"{away} @ {home}"
        label_to_abbrs[label] = (
            TEAM_NAME_TO_ABBR.get(home, home),
            TEAM_NAME_TO_ABBR.get(away, away),
        )

    rows = []
    for p in actionable:
        home_abbr, away_abbr = label_to_abbrs.get(p["game"], (None, None))
        rows.append({
            "date":                 target_date,
            "game":                 p["game"],
            "home_abbr":            home_abbr,
            "away_abbr":            away_abbr,
            "model":                p["model"],
            "bet_type":             p["bet_type"],
            "pick_direction":       p["pick_direction"],
            "model_prob":           p["model_prob"],
            "P_true":               p["P_true"],
            "Retail_Implied_Prob":  p["Retail_Implied_Prob"],
            "edge":                 p["edge"],
            "retail_american_odds": p["retail_american_odds"],
            "tier":                 p["tier"],
            "dollar_stake":         p["dollar_stake"],
            "result":               "",
            "profit_loss":          "",
            "graded_at":            "",
        })

    ledger = _load_ledger()
    ledger = ledger[ledger["date"].astype(str) != str(target_date)]
    ledger = pd.concat([ledger, pd.DataFrame(rows, columns=LEDGER_COLUMNS)], ignore_index=True)
    _save_ledger(ledger)


def _pl_on_win(stake: float, american_odds: float) -> float:
    dec = _american_to_decimal(float(american_odds))
    return round(float(stake) * (dec - 1.0), 2)


def _grade_one(row: pd.Series, actuals: pd.Series) -> Optional[str]:
    """Return 'WIN' / 'LOSS' / 'PUSH' or None if the market cannot be graded
    from this actuals row (e.g. F5 SP didn't complete 5 IP)."""
    model  = str(row["model"])
    side   = str(row["pick_direction"])

    if model == "ML":
        home_won = actuals["home_score_final"] > actuals["away_score_final"]
        tied     = actuals["home_score_final"] == actuals["away_score_final"]
        if tied:
            return "PUSH"
        picked_home = side == "HOME"
        return "WIN" if (home_won == picked_home) else "LOSS"

    if model == "Totals":
        # bet_type looks like "Total 8.5" — extract the line.
        try:
            line = float(str(row["bet_type"]).split()[-1])
        except Exception:
            return None
        total = float(actuals["home_score_final"]) + float(actuals["away_score_final"])
        if total == line:
            return "PUSH"
        over_hit = total > line
        picked_over = side == "OVER"
        return "WIN" if (over_hit == picked_over) else "LOSS"

    if model == "Runline":
        home_covers = bool(actuals.get("home_covers_rl"))
        picked_home = side.startswith("HOME")
        return "WIN" if (home_covers == picked_home) else "LOSS"

    if model == "F5":
        fh = actuals.get("f5_home_runs")
        fa = actuals.get("f5_away_runs")
        if pd.isna(fh) or pd.isna(fa):
            return None
        if fh == fa:
            return "PUSH"
        home_won_f5 = fh > fa
        picked_home = side == "HOME"
        return "WIN" if (home_won_f5 == picked_home) else "LOSS"

    if model == "NRFI":
        nrfi_hit = bool(actuals.get("f1_nrfi"))
        bt = str(row["bet_type"]).upper()
        picked_nrfi = bt == "NRFI"
        return "WIN" if (nrfi_hit == picked_nrfi) else "LOSS"

    return None


def auto_grade_historical_picks() -> str:
    """Grade every ungraded row in historical_actionable_picks.csv against
    actuals_2026.parquet. WIN / LOSS / PUSH + profit_loss are written back."""
    ledger = _load_ledger()
    if ledger.empty:
        return json.dumps({"status": "OK", "graded": 0, "pending": 0, "note": "Ledger is empty."})

    actuals_path = Path(FILES["actuals_2026"])
    if not actuals_path.exists():
        return json.dumps({"status": "ERROR", "error": f"Actuals not found: {actuals_path}"})

    actuals = pd.read_parquet(actuals_path)
    actuals["game_date"] = actuals["game_date"].astype(str)
    # Index by (date, home_abbr, away_abbr) for O(1) lookup.
    act_idx = {
        (str(r["game_date"]), str(r["home_team"]), str(r["away_team"])): r
        for _, r in actuals.iterrows()
    }

    graded_now = 0
    ungradable = 0
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    for i, row in ledger.iterrows():
        if str(row.get("result", "")).upper() in ("WIN", "LOSS", "PUSH"):
            continue
        key = (str(row["date"]), str(row["home_abbr"]), str(row["away_abbr"]))
        act = act_idx.get(key)
        if act is None:
            ungradable += 1
            continue

        result = _grade_one(row, act)
        if result is None:
            ungradable += 1
            continue

        stake = float(row["dollar_stake"]) if pd.notna(row["dollar_stake"]) else 0.0
        odds  = float(row["retail_american_odds"]) if pd.notna(row["retail_american_odds"]) else 0.0
        if result == "WIN":
            pl = _pl_on_win(stake, odds)
        elif result == "LOSS":
            pl = -stake
        else:
            pl = 0.0

        ledger.at[i, "result"]      = result
        ledger.at[i, "profit_loss"] = pl
        ledger.at[i, "graded_at"]   = now_str
        graded_now += 1

    _save_ledger(ledger)

    graded_total = int((ledger["result"].isin(["WIN", "LOSS", "PUSH"])).sum())
    pending      = int(len(ledger) - graded_total)

    return json.dumps({
        "status":        "OK",
        "graded_now":    graded_now,
        "graded_total":  graded_total,
        "pending":       pending,
        "ungradable":    ungradable,  # matched no actuals row or market lacks data
    })


def _window_stats(df: pd.DataFrame) -> dict:
    """Aggregate a graded slice into Total Bets / Win % / ROI %, segmented by market."""
    graded = df[df["result"].isin(["WIN", "LOSS", "PUSH"])]

    def agg(sub: pd.DataFrame) -> dict:
        if sub.empty:
            return {"bets": 0, "wins": 0, "losses": 0, "pushes": 0,
                    "win_pct": 0.0, "roi_pct": 0.0, "pl": 0.0, "wagered": 0.0}
        w = int((sub["result"] == "WIN").sum())
        l = int((sub["result"] == "LOSS").sum())
        p = int((sub["result"] == "PUSH").sum())
        decided = w + l
        pl = float(pd.to_numeric(sub["profit_loss"], errors="coerce").fillna(0).sum())
        wagered = float(pd.to_numeric(sub["dollar_stake"], errors="coerce").fillna(0).sum())
        return {
            "bets":    int(len(sub)),
            "wins":    w, "losses": l, "pushes": p,
            "win_pct": round(100.0 * w / decided, 1) if decided > 0 else 0.0,
            "roi_pct": round(100.0 * pl / wagered, 1) if wagered > 0 else 0.0,
            "pl":      round(pl, 2),
            "wagered": round(wagered, 2),
        }

    by_market = {m: agg(graded[graded["model"] == m])
                 for m in ["ML", "Totals", "Runline", "F5", "NRFI"]}
    return {"overall": agg(graded), "by_market": by_market,
            "alpha_sgp": _alpha_sgp_stats(graded)}


# ── Alpha SGP (Beta): joint ML+Totals parlay in the validated sweet-spot band.
#    Sweet spot (from 2026 backtest + walk-forward test):
#      0.65 ≤ p_ml_pick ≤ 0.72  AND  p_tot_pick ≥ 0.62
#    One ticket per (date, game) pair; $1 unit stake; parlay pays dec_ml * dec_tot.
_SGP_ML_LO, _SGP_ML_HI = 0.65, 0.72
_SGP_TOT_MIN           = 0.62
_SGP_UNIT              = 1.0


def _alpha_sgp_stats(graded: pd.DataFrame) -> dict:
    empty = {"bets": 0, "wins": 0, "losses": 0, "pushes": 0,
             "win_pct": 0.0, "roi_pct": 0.0, "pl": 0.0, "wagered": 0.0}
    if graded.empty:
        return dict(empty)

    g = graded.copy()
    g["model_prob"] = pd.to_numeric(g["model_prob"], errors="coerce")
    g["retail_american_odds"] = pd.to_numeric(g["retail_american_odds"], errors="coerce")

    ml_legs  = g[g["model"] == "ML"]
    tot_legs = g[g["model"] == "Totals"]
    if ml_legs.empty or tot_legs.empty:
        return dict(empty)

    # Highest-conviction leg per (date, game) per market.
    ml_best  = (ml_legs.sort_values("model_prob", ascending=False)
                       .drop_duplicates(subset=["date", "game"], keep="first"))
    tot_best = (tot_legs.sort_values("model_prob", ascending=False)
                        .drop_duplicates(subset=["date", "game"], keep="first"))

    pairs = ml_best.merge(tot_best, on=["date", "game"], suffixes=("_ml", "_tot"))
    if pairs.empty:
        return dict(empty)

    mask = (pairs["model_prob_ml"].between(_SGP_ML_LO, _SGP_ML_HI, inclusive="both")
            & (pairs["model_prob_tot"] >= _SGP_TOT_MIN))
    qual = pairs[mask]
    if qual.empty:
        return dict(empty)

    wins = losses = pushes = 0
    pl_sum = 0.0
    wagered = 0.0
    for _, r in qual.iterrows():
        res_ml, res_tot = str(r["result_ml"]).upper(), str(r["result_tot"]).upper()
        dec_ml  = _american_to_decimal(float(r["retail_american_odds_ml"]))  if pd.notna(r["retail_american_odds_ml"])  else None
        dec_tot = _american_to_decimal(float(r["retail_american_odds_tot"])) if pd.notna(r["retail_american_odds_tot"]) else None
        if dec_ml is None or dec_tot is None:
            continue
        wagered += _SGP_UNIT
        if res_ml == "LOSS" or res_tot == "LOSS":
            losses += 1
            pl_sum -= _SGP_UNIT
        elif res_ml == "WIN" and res_tot == "WIN":
            wins += 1
            pl_sum += _SGP_UNIT * (dec_ml * dec_tot - 1.0)
        else:
            # Any PUSH without a LOSS → treat the joint as a PUSH (rare).
            pushes += 1

    decided = wins + losses
    bets = wins + losses + pushes
    return {
        "bets":    int(bets),
        "wins":    int(wins), "losses": int(losses), "pushes": int(pushes),
        "win_pct": round(100.0 * wins / decided, 1) if decided > 0 else 0.0,
        "roi_pct": round(100.0 * pl_sum / wagered, 1) if wagered > 0 else 0.0,
        "pl":      round(pl_sum, 2),
        "wagered": round(wagered, 2),
    }


def compute_rolling_accuracy() -> str:
    """Aggregate the graded ledger into 7-day / 28-day / season-to-date (2026)
    windows. Each window reports overall + per-market bets / win% / ROI%."""
    ledger = _load_ledger()
    if ledger.empty:
        empty = {"bets": 0, "wins": 0, "losses": 0, "pushes": 0,
                 "win_pct": 0.0, "roi_pct": 0.0, "pl": 0.0, "wagered": 0.0}
        markets = {m: dict(empty) for m in ["ML", "Totals", "Runline", "F5", "NRFI"]}
        blank = {"overall": dict(empty), "by_market": markets, "alpha_sgp": dict(empty)}
        return json.dumps({"status": "OK", "windows": {"last_7": blank, "last_28": blank, "ytd_2026": blank}})

    ledger["date_parsed"] = pd.to_datetime(ledger["date"], errors="coerce")
    today = pd.Timestamp(date.today())

    ytd_mask  = ledger["date_parsed"].dt.year == 2026
    d28_mask  = ledger["date_parsed"] >= (today - pd.Timedelta(days=28))
    d7_mask   = ledger["date_parsed"] >= (today - pd.Timedelta(days=7))

    return json.dumps({
        "status":  "OK",
        "as_of":   today.strftime("%Y-%m-%d"),
        "windows": {
            "last_7":    _window_stats(ledger[d7_mask]),
            "last_28":   _window_stats(ledger[d28_mask]),
            "ytd_2026":  _window_stats(ledger[ytd_mask]),
        },
    })


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 6 — Passive Stats Reader  (bet_tracker legacy helpers retained for
# read-only compatibility; manual append/log_result are deprecated and no
# longer exposed as tools to any agent).
# ══════════════════════════════════════════════════════════════════════════════

def _load_tracker() -> pd.DataFrame:
    p = Path(FILES["bet_tracker"])
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=BET_TRACKER_COLUMNS)
    return pd.read_csv(p)


def _save_tracker(df: pd.DataFrame) -> None:
    df.to_csv(FILES["bet_tracker"], index=False)


def append_bet(date: str, game: str, model: str, bet_type: str,
               model_prob: float, market_line: str, book: str,
               units: float, notes: str = "") -> str:
    df     = _load_tracker()
    new_id = int(df["id"].max()) + 1 if len(df) > 0 and "id" in df.columns else 1
    row    = {
        "id": new_id, "date": date, "game": game, "model": model,
        "bet_type": bet_type, "model_prob": model_prob, "market_line": market_line,
        "book": book, "units": units, "result": "PENDING",
        "actual_total": None, "profit_loss": None,
        "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "notes": notes,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save_tracker(df)
    return json.dumps({"status": "logged", "bet_id": new_id})


def log_result(bet_id: int, result: str, actual_total: Optional[float] = None) -> str:
    result = result.upper()
    if result not in ("WIN", "LOSS", "PUSH"):
        return json.dumps({"status": "REJECTED", "error": f"Invalid result: {result}."})

    df   = _load_tracker()
    mask = df["id"] == bet_id
    if not mask.any():
        return json.dumps({"status": "REJECTED", "error": f"No bet found with id={bet_id}."})

    existing = str(df.loc[mask, "result"].values[0])
    if existing != "PENDING":
        return json.dumps({"status": "REJECTED",
                           "error": f"❌ Result already recorded as {existing}. Cannot modify a finalized bet."})

    units = float(df.loc[mask, "units"].values[0])
    line  = str(df.loc[mask, "market_line"].values[0])

    if result == "PUSH":
        pl = 0.0
    elif result == "LOSS":
        pl = -units
    else:
        try:
            l = float(line)
            pl = round(units * (100 / abs(l)), 3) if l < 0 else round(units * (l / 100), 3)
        except Exception:
            pl = round(units * 0.909, 3)  # Default -110

    df.loc[mask, "result"]       = result
    df.loc[mask, "profit_loss"]  = pl
    df.loc[mask, "actual_total"] = actual_total
    _save_tracker(df)
    return json.dumps({"status": "result_recorded", "bet_id": bet_id,
                       "result": result, "profit_loss": pl})


def read_tracker_stats() -> str:
    df = _load_tracker()
    if df.empty:
        return json.dumps({"overall": {}, "by_model": {}})

    fin = df[df["result"].isin(["WIN", "LOSS", "PUSH"])]

    def stats(sub: pd.DataFrame) -> dict:
        w = (sub["result"] == "WIN").sum()
        l = (sub["result"] == "LOSS").sum()
        p = (sub["result"] == "PUSH").sum()
        t = w + l
        pl      = float(sub["profit_loss"].sum())
        wagered = float(sub["units"].sum())
        return {
            "record":   f"{w}-{l}-{p}",
            "win_rate": round(w / t, 3) if t > 0 else 0.0,
            "units_pl": round(pl, 2),
            "roi":      round(pl / wagered, 3) if wagered > 0 else 0.0,
            "pending":  int((df["result"] == "PENDING").sum()),
        }

    return json.dumps({
        "overall":  stats(fin),
        "by_model": {m: stats(fin[fin["model"] == m])
                     for m in ["ML", "Totals", "Runline", "F5", "NRFI"]},
    })
