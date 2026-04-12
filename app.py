"""
app.py — The Wizard MLB Picks Dashboard
One password at the door, everything open after.
Reads from Supabase when hosted, local files when run locally.
"""
import datetime
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Load .env for local dev ───────────────────────────────────────────────────
def _load_dotenv():
    env = Path(__file__).parent / ".env"
    if not env.exists():
        return
    with open(env) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())
_load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Wizard — MLB Picks",
    page_icon="🧙",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ─── Login ───────────────────────────────────────── */
.login-wrap  { max-width:380px; margin:80px auto; text-align:center; }
.login-title { font-size:2.4rem; font-weight:800; margin-bottom:4px; }
.login-sub   { color:#6b7280; margin-bottom:32px; }

/* ─── Cards ───────────────────────────────────────── */
.bet-card-strong {
    background: linear-gradient(135deg,#0a2016 0%,#13362a 100%);
    border: 2px solid #22c55e; border-radius:14px;
    padding:22px 26px; margin-bottom:20px;
}
.bet-card-lean {
    background: linear-gradient(135deg,#181600 0%,#2a2700 100%);
    border: 2px solid #eab308; border-radius:14px;
    padding:22px 26px; margin-bottom:20px;
}

/* Card header */
.card-head       { display:flex; align-items:flex-start; gap:14px; margin-bottom:12px; }
.card-num        { font-size:1rem; font-weight:700; color:#6b7280;
                   background:#1f2937; border-radius:6px; padding:2px 8px;
                   white-space:nowrap; margin-top:4px; }
.card-pick       { flex:1; }
.card-pick-line  { font-size:1.6rem; font-weight:800; color:#f9fafb; line-height:1.1; }
.card-pick-sub   { font-size:.88rem; color:#9ca3af; margin-top:2px; }
.badge-strong    { background:#16a34a; color:#fff; font-size:.72rem; font-weight:700;
                   padding:4px 12px; border-radius:99px; letter-spacing:.08em;
                   white-space:nowrap; margin-top:4px; display:inline-block; }
.badge-lean      { background:#ca8a04; color:#fff; font-size:.72rem; font-weight:700;
                   padding:4px 12px; border-radius:99px; letter-spacing:.08em;
                   white-space:nowrap; margin-top:4px; display:inline-block; }

/* Confidence section */
.conf-wrap   { margin:12px 0; }
.conf-detail { font-size:.9rem; color:#d1d5db; font-weight:500;
               background:rgba(255,255,255,.06); border-radius:6px;
               padding:7px 12px; display:inline-block; }

/* Score prediction */
.score-section  { background:rgba(255,255,255,.04); border-radius:8px;
                  padding:14px 18px; margin:12px 0; }
.score-label    { font-size:.72rem; color:#6b7280; text-transform:uppercase;
                  letter-spacing:.08em; margin-bottom:10px; }
.bp-row         { display:flex; align-items:center; gap:12px; margin:6px 0; }
.bp-team        { font-size:.82rem; font-weight:700; color:#9ca3af;
                  width:36px; text-align:right; }
.bp-avg         { font-size:1.05rem; font-weight:800; color:#f9fafb; width:28px; }
.bp-chart       { flex:1; height:18px; position:relative; max-width:160px; }
.bp-track       { height:4px; background:rgba(255,255,255,.1); border-radius:2px;
                  position:absolute; top:7px; width:100%; }
.bp-fill        { height:4px; border-radius:2px; position:absolute; top:0; }
.bp-fill-a      { background:#22c55e; }
.bp-fill-b      { background:#6b7280; }
.bp-dot         { width:12px; height:12px; border-radius:50%; background:#f9fafb;
                  position:absolute; top:-4px; transform:translateX(-50%);
                  border:2px solid #111827; }
.bp-range       { font-size:.78rem; color:#6b7280; white-space:nowrap; }
.score-divider  { border:none; border-top:1px solid rgba(255,255,255,.06);
                  margin:10px 0; }
.score-total    { font-size:.85rem; color:#9ca3af; }

/* Why */
.why-section { margin:12px 0; }
.why-label   { font-size:.72rem; color:#6b7280; text-transform:uppercase;
               letter-spacing:.08em; margin-bottom:5px; }
.why-item    { font-size:.92rem; color:#d1fae5; margin-bottom:3px; }
.why-item::before { content:"▶ "; font-size:.65rem; opacity:.7; }

/* Also play */
.also-play   { background:rgba(234,179,8,.08); border:1px solid rgba(234,179,8,.25);
               border-radius:6px; padding:8px 14px; margin:10px 0;
               font-size:.9rem; color:#fef08a; font-weight:600; }

/* Meta bar */
.meta-bar    { font-size:.8rem; color:#4b5563; margin-top:10px;
               border-top:1px solid #1f2937; padding-top:8px; }

/* Skip games */
.skip-card { background:#111827; border:1px solid #1f2937; border-radius:10px;
             padding:14px 18px; margin-bottom:10px; }
.skip-game { color:#6b7280; font-size:.88rem; }
.skip-game strong { color:#9ca3af; }

/* Section headers */
.section-hdr { font-size:.9rem; font-weight:700; color:#6b7280;
               letter-spacing:.12em; text-transform:uppercase;
               margin:24px 0 10px 0; border-bottom:1px solid #1f2937;
               padding-bottom:6px; }

/* Bigger tabs */
button[data-baseweb="tab"] { font-size:1rem !important; font-weight:600 !important; padding:10px 20px !important; }

/* Quick-pick chips */
.pick-chip { display:inline-block; background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.15);
             border-radius:99px; padding:5px 14px; margin:4px; font-size:.85rem; color:#f9fafb;
             font-weight:600; cursor:default; }
.pick-chip-strong { border-color:#22c55e; background:rgba(34,197,94,.12); color:#4ade80; }
.pick-chip-lean   { border-color:#eab308; background:rgba(234,179,8,.10); color:#fde047; }

/* Tracker date buttons */
div[data-testid="column"] button { font-size:.8rem !important; }

/* ─── Hover tooltips ──────────────────────────────────── */
.tip {
    border-bottom: 1px dotted #6b7280;
    cursor: help;
    position: relative;
    display: inline-block;
}
.tip::after {
    content: attr(data-tip);
    position: absolute;
    bottom: 135%;
    left: 50%;
    transform: translateX(-50%);
    background: #111827;
    color: #e5e7eb;
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 0.78rem;
    line-height: 1.5;
    max-width: 280px;
    width: max-content;
    white-space: normal;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.15s ease;
    z-index: 9999;
    border: 1px solid #374151;
    box-shadow: 0 4px 16px rgba(0,0,0,.5);
    text-align: left;
}
.tip:hover::after { opacity: 1; }
</style>
""", unsafe_allow_html=True)


# ── Password gate ─────────────────────────────────────────────────────────────
import hashlib

_PWD_HASH = "a00efb424a310a6b1e1621c32f5914fadffd6a7df22c7333524515487e372854"

def _check_pwd(entered: str) -> bool:
    return hashlib.sha256(entered.strip().encode()).hexdigest() == _PWD_HASH

def check_password() -> bool:
    if st.session_state.get("authenticated"):
        return True
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="login-title">🧙 The Wizard</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-sub">MLB Prediction Model</div>', unsafe_allow_html=True)
        pwd = st.text_input("Password", type="password", label_visibility="collapsed",
                            placeholder="Enter password")
        if st.button("Enter", use_container_width=True, type="primary"):
            if _check_pwd(pwd):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        st.markdown('</div>', unsafe_allow_html=True)
    return False

if not check_password():
    st.stop()


# ── Supabase client ───────────────────────────────────────────────────────────
@st.cache_resource
def _supabase():
    try:
        url = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL", ""))
        key = st.secrets.get("SUPABASE_KEY", os.environ.get("SUPABASE_KEY", ""))
    except Exception:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        return None
    from supabase import create_client
    return create_client(url, key)

USE_SUPABASE = _supabase() is not None


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_card(date_str: str) -> list[dict]:
    if USE_SUPABASE:
        client = _supabase()
        try:
            resp = client.table("wizard_daily_card").select("*").eq("game_date", date_str).execute()
        except Exception as e:
            st.error(f"Supabase error: {e}")
            return []
        if not resp.data:
            return []
        rows = []
        for row in resp.data:
            r = row.get("data", {}) or {}
            for col in ["game","home_team","away_team","rl_signal","total_signal",
                        "blended_rl","mc_rl","xgb_rl","mc_total","blended_total",
                        "vegas_total","vegas_ml_home","vegas_ml_away","home_sp","away_sp",
                        "home_sp_xwoba","away_sp_xwoba","home_sp_flag","away_sp_flag",
                        "temp_f","lineup_confirmed",
                        "best_line","best_tier","best_model_prob","best_market_odds","best_edge",
                        "mc_home_win","mc_home_cvr25","mc_away_cvr25",
                        "home_runs_mean","away_runs_mean",
                        "home_runs_lo","home_runs_hi","away_runs_lo","away_runs_hi"]:
                if col not in r:
                    r[col] = row.get(col)
            rows.append(r)
        return rows
    try:
        from run_today import run_card
        return run_card(date_str)
    except Exception as e:
        st.error(f"Could not load predictions: {e}")
        return []


@st.cache_data(ttl=300)
def load_backtest() -> pd.DataFrame:
    if USE_SUPABASE:
        client = _supabase()
        try:
            resp = client.table("wizard_backtest").select("*").execute()
            return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    path = Path(__file__).parent / "backtest_2026_results.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=60)
def load_tracker() -> pd.DataFrame:
    if USE_SUPABASE:
        client = _supabase()
        try:
            resp = client.table("bet_tracker").select("*").order("date", desc=True).execute()
            return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    path = Path(__file__).parent / "data" / "raw" / "bet_tracker.csv"
    return pd.read_csv(path).sort_values("date", ascending=False) if path.exists() else pd.DataFrame()


# ── Card helpers ──────────────────────────────────────────────────────────────

def _tip(term: str, definition: str) -> str:
    """Wrap a term in a hover tooltip span."""
    # Escape quotes in definition for HTML attribute safety
    defn = definition.replace('"', "&quot;")
    return f'<span class="tip" data-tip="{defn}">{term}</span>'

# Pre-built tooltip snippets for common terms
T_XWOBA  = _tip("xwOBA", "Expected Weighted On-Base Average — pitcher quality metric. Lower = better. Elite: <0.280 | Avg: ~0.318 | Poor: >0.350")
T_ML     = _tip("ML",    "Moneyline — bet on this team to win the game outright. No spread involved.")
T_RL     = _tip("RL",    "Run Line — baseball's point spread. Standard is ±1.5 runs.")
T_OU     = _tip("O/U",   "Over/Under — bet on whether total runs scored is above or below the listed number.")
T_EDGE   = _tip("edge",  "How much better our model's probability is vs. what the market odds imply. Higher = more value.")
T_IMPLIED= _tip("Market implied", "The win probability baked into the current betting odds.")
T_CONF   = _tip("Model confidence", "Probability our model assigns to this bet winning, based on 50,000 simulated games.")

def _safe_float(v, default=0.0):
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default

def _model_prob(r: dict) -> float:
    """Return probability from the perspective of the recommended bet (always > 0.5 if there's edge)."""
    if r.get("best_model_prob") is not None:
        return _safe_float(r["best_model_prob"])
    # Fallback: derive from blended_rl based on signal direction
    bl   = _safe_float(r.get("blended_rl"), 0.5)
    sig  = str(r.get("rl_signal") or "")
    return (1 - bl) if "AWAY" in sig else bl

def _edge_pct(r: dict) -> float:
    """Edge in percentage points vs market implied probability."""
    if r.get("best_edge") is not None:
        return _safe_float(r["best_edge"]) * 100
    # Estimate: model prob minus breakeven at -110
    return (_model_prob(r) - 0.5238) * 100

def _why_bullets(r: dict) -> list[str]:
    home_x = _safe_float(r.get("home_sp_xwoba"))
    away_x = _safe_float(r.get("away_sp_xwoba"))
    best_line = str(r.get("best_line") or r.get("rl_signal") or "")
    is_home_bet = "home" in best_line.lower() or (
        "ML" in best_line and home_x > 0 and home_x < away_x)

    fav_sp   = r.get("home_sp","").title() if is_home_bet else r.get("away_sp","").title()
    fav_x    = home_x if is_home_bet else away_x
    opp_sp   = r.get("away_sp","").title() if is_home_bet else r.get("home_sp","").title()
    opp_x    = away_x if is_home_bet else home_x

    bullets = []
    if fav_x and fav_x < 0.280:
        bullets.append(f"{fav_sp} is elite (xwOBA {fav_x:.3f})")
    elif fav_x and fav_x < 0.305:
        bullets.append(f"{fav_sp} is strong (xwOBA {fav_x:.3f})")
    elif fav_x and fav_x < 0.320:
        bullets.append(f"{fav_sp} is above average (xwOBA {fav_x:.3f})")

    if opp_x and opp_x > 0.360:
        bullets.append(f"{opp_sp} is struggling (xwOBA {opp_x:.3f})")
    elif opp_x and opp_x > 0.340:
        bullets.append(f"{opp_sp} is below average (xwOBA {opp_x:.3f})")

    # Velocity flags
    h_flag = r.get("home_sp_flag","")
    a_flag = r.get("away_sp_flag","")
    if not is_home_bet and h_flag == "VOLATILE":
        bullets.append(f"{r.get('home_sp','').title()} velocity declining")
    if is_home_bet and a_flag == "VOLATILE":
        bullets.append(f"{r.get('away_sp','').title()} velocity declining")
    if not is_home_bet and a_flag == "GAINER":
        bullets.append(f"{r.get('away_sp','').title()} velo trending up")

    temp = _safe_float(r.get("temp_f"), 72)
    if temp > 82:
        bullets.append(f"Hot weather ({temp:.0f}°F, ball carries)")
    elif temp < 48:
        bullets.append(f"Cold weather ({temp:.0f}°F, suppresses scoring)")

    line_l = best_line.lower()
    if "-2.5" in line_l:
        bullets.append("Dominant pitching matchup — expect comfortable margin")
    elif "ml" in line_l:
        bullets.append("Moneyline offers better value than run line")

    return bullets if bullets else ["Model edge on run differential"]


def _format_bet_label(r: dict) -> tuple[str, str]:
    """Return (bet_label, subtitle) e.g. ('TEX +1.5', 'Texas Rangers covering 1.5 runs vs. LAD')."""
    away, home = r.get("away_team",""), r.get("home_team","")
    line  = str(r.get("best_line") or "")
    odds  = r.get("best_market_odds")
    odds_str = f" ({int(odds):+d})" if odds is not None else ""

    line_l = line.lower()
    if "-2.5" in line_l and "home" in line_l:
        label = f"{home} -2.5{odds_str}"
        sub   = f"{away} @ {home}  ·  Home wins by 3+"
    elif "+2.5" in line_l and "away" in line_l:
        label = f"{away} +2.5{odds_str}"
        sub   = f"{away} @ {home}  ·  Away stays within 2 or wins"
    elif "-1.5" in line_l and "home" in line_l:
        label = f"{home} -1.5{odds_str}"
        sub   = f"{away} @ {home}  ·  Home wins by 2+"
    elif "+1.5" in line_l and "away" in line_l:
        label = f"{away} +1.5{odds_str}"
        sub   = f"{away} @ {home}  ·  Away stays within 1 or wins"
    elif "ml" in line_l and "home" in line_l:
        label = f"{home} ML{odds_str}"
        sub   = f"{away} @ {home}  ·  Home wins outright"
    elif "ml" in line_l and "away" in line_l:
        label = f"{away} ML{odds_str}"
        sub   = f"{away} @ {home}  ·  Away wins outright"
    else:
        # Legacy fallback
        sig = str(r.get("rl_signal",""))
        if "AWAY" in sig:
            label = f"{away} +1.5"
            sub   = f"{away} @ {home}  ·  Away stays within 1 or wins"
        elif "HOME" in sig:
            label = f"{home} -1.5"
            sub   = f"{away} @ {home}  ·  Home wins by 2+"
        else:
            label = r.get("game","")
            sub   = ""
    return label, sub


def render_card(r: dict, n: int, tier: str):
    away, home = r.get("away_team",""), r.get("home_team","")
    bet_label, bet_sub = _format_bet_label(r)
    model_prob = _model_prob(r)
    edge       = _edge_pct(r)
    conf_pct   = int(model_prob * 100)

    fill_cls  = "conf-fill-strong" if tier == "strong" else "conf-fill-lean"
    card_cls  = "bet-card-strong"  if tier == "strong" else "bet-card-lean"
    badge_cls = "badge-strong"     if tier == "strong" else "badge-lean"
    badge_txt = (f"★★ STRONG  ·  {conf_pct}% confidence" if tier == "strong"
                 else f"★ LEAN  ·  {conf_pct}% confidence")

    # Confidence / market comparison line
    market_odds    = r.get("best_market_odds")
    market_implied = r.get("best_market_implied")
    raw_model      = _safe_float(r.get("best_raw_model") or model_prob)
    deviation      = _safe_float(r.get("market_deviation"), 0.0)

    if market_odds is not None:
        mo = float(market_odds)
        impl = _safe_float(market_implied) if market_implied else (
            abs(mo)/(abs(mo)+100) if mo < 0 else 100/(mo+100))
        conf_detail = (
            f"<b style='color:#f9fafb'>Market odds: {int(mo):+d}</b>"
            f"&nbsp;({impl:.0%} implied)"
            f"&nbsp;·&nbsp;Model: {int(model_prob*100)}%"
            f"&nbsp;·&nbsp;Your {T_EDGE}: <b style='color:#4ade80'>{edge:+.1f}%</b>"
        )
    else:
        conf_detail = (
            f"Model: {int(model_prob*100)}%"
            f"&nbsp;·&nbsp;Breakeven at -110: 52.4%"
            f"&nbsp;·&nbsp;Your {T_EDGE}: <b style='color:#4ade80'>{edge:+.1f}%</b>"
        )

    # Warning when pitcher-only model diverges significantly from the market
    deviation_warning = ""
    if deviation >= 0.25:
        deviation_warning = (
            f'<div style="background:rgba(251,191,36,.12);border:1px solid rgba(251,191,36,.4);'
            f'border-radius:6px;padding:6px 12px;margin:8px 0;font-size:.82rem;color:#fcd34d;">'
            f'⚠️ <b>Model vs market gap: {deviation:.0%}</b> — pitcher matchup strongly favors '
            f'this side, but the market disagrees. Our model does not see lineup or bullpen strength. '
            f'Use judgment.</div>'
        )

    # Predicted score
    hm = r.get("home_runs_mean")
    am = r.get("away_runs_mean")
    hl = r.get("home_runs_lo"); hh = r.get("home_runs_hi")
    al = r.get("away_runs_lo"); ah = r.get("away_runs_hi")
    MAX_R = 12.0   # scale: 0–12 runs covers 99%+ of MLB games

    def _bp_row(team, avg, lo, hi, cls):
        """Generate one mini box-plot row."""
        if lo is None or hi is None:
            return (f'<div class="bp-row">'
                    f'<div class="bp-team">{team}</div>'
                    f'<div class="bp-avg">{avg:.1f}</div>'
                    f'</div>')
        box_left  = min(lo,  MAX_R) / MAX_R * 100
        box_width = max(0, min(hi, MAX_R) - min(lo, MAX_R)) / MAX_R * 100
        dot_left  = min(avg, MAX_R) / MAX_R * 100
        return (
            f'<div class="bp-row">'
            f'  <div class="bp-team">{team}</div>'
            f'  <div class="bp-avg">{avg:.1f}</div>'
            f'  <div class="bp-chart">'
            f'    <div class="bp-track">'
            f'      <div class="bp-fill {cls}" style="left:{box_left:.1f}%;width:{box_width:.1f}%"></div>'
            f'      <div class="bp-dot" style="left:{dot_left:.1f}%"></div>'
            f'    </div>'
            f'  </div>'
            f'  <div class="bp-range">{int(lo) if lo else "?"}–{int(hi) if hi else "?"} runs</div>'
            f'</div>'
        )

    if hm is not None and am is not None:
        # Determine which team is the bet (green bar) vs opponent (grey)
        is_home_bet_score = "home" in str(r.get("best_line") or "").lower()
        away_cls = "bp-fill-a" if not is_home_bet_score else "bp-fill-b"
        home_cls = "bp-fill-a" if is_home_bet_score  else "bp-fill-b"

        bands_html = (
            f'<div style="margin:8px 0">'
            + _bp_row(away, _safe_float(am), al, ah, away_cls)
            + _bp_row(home, _safe_float(hm), hl, hh, home_cls)
            + f'<div style="font-size:.7rem;color:#4b5563;margin-top:4px">'
            f'Bars show 25th–75th percentile of 50,000 simulations · '
            f'<span style="color:#22c55e">■</span> = recommended side</div>'
            + f'</div>'
        )
    else:
        bands_html = '<div style="font-size:.8rem;color:#4b5563">Re-run pipeline for score prediction</div>'

    bt = r.get("blended_total") or r.get("mc_total")
    vt = r.get("vegas_total")
    total_sig = str(r.get("total_signal") or "")

    if bt and vt:
        diff   = _safe_float(bt) - _safe_float(vt)
        ou_dir = "OVER" if diff > 0 else "UNDER"
        if total_sig:
            score_total = (f"Total: {_safe_float(bt):.1f} runs  ·  Market {T_OU}: {vt}"
                           f"  →  <b style='color:#fef08a'>Also play: {total_sig}</b>")
        else:
            score_total = (f"Total: {_safe_float(bt):.1f} runs  "
                           f"(Market {T_OU}: {vt}  ·  model leans {ou_dir})")
    elif bt:
        score_total = f"Expected total: {_safe_float(bt):.1f} runs"
    else:
        score_total = ""

    # Why bullets
    bullets_html = "".join(f'<div class="why-item">{b}</div>' for b in _why_bullets(r))

    # No separate "Also consider" block — it's now inline in the score section
    also_html = ""

    # Meta bar
    lineup_str = "Confirmed lineup" if r.get("lineup_confirmed") else "Projected lineup"
    temp_f = _safe_float(r.get("temp_f"), 72)
    ml_str = ""
    if r.get("vegas_ml_home") is not None:
        ml_str = (f"ML: {home} {int(r['vegas_ml_home']):+d}"
                  + (f" / {away} {int(r['vegas_ml_away']):+d}" if r.get("vegas_ml_away") else ""))

    # SP flags
    hf = f" [{r['home_sp_flag']}]" if r.get("home_sp_flag") not in ("NORMAL","UNKNOWN","","None",None) else ""
    af = f" [{r['away_sp_flag']}]" if r.get("away_sp_flag") not in ("NORMAL","UNKNOWN","","None",None) else ""
    sp_str = (f"{home} SP: {r.get('home_sp','').title()}{hf} {T_XWOBA} {_safe_float(r.get('home_sp_xwoba')):.3f}  "
              f"|  {away} SP: {r.get('away_sp','').title()}{af} {T_XWOBA} {_safe_float(r.get('away_sp_xwoba')):.3f}")

    score_total_html = f'<div class="score-total">{score_total}</div>' if score_total else ""

    st.markdown(f"""
<div class="{card_cls}">

  <div class="card-head">
    <div class="card-num">#{n}</div>
    <div class="card-pick">
      <div class="card-pick-line">{bet_label}</div>
      <div class="card-pick-sub">{bet_sub}</div>
    </div>
    <div><span class="{badge_cls}">{badge_txt}</span></div>
  </div>

  <div class="conf-wrap">
    <div class="conf-detail">{conf_detail}</div>
    {deviation_warning}
  </div>

  <div class="score-section">
    <div class="score-label">Predicted score</div>
    <div class="score-main">{score_main}</div>
    {bands_html}
    {score_total_html}
  </div>

  <div class="why-section">
    <div class="why-label">Why this bet</div>
    {bullets_html}
  </div>

  {also_html}

  <div class="meta-bar">
    {temp_f:.0f}°F &nbsp;·&nbsp; {lineup_str}
    {'&nbsp;·&nbsp; ' + ml_str if ml_str else ''}
    <br>{sp_str}
  </div>

</div>""", unsafe_allow_html=True)


# ── App header ────────────────────────────────────────────────────────────────
st.markdown("## 🧙 The Wizard — MLB Picks")
st.caption("Picks updated daily after 9 AM ET  ·  Sorted by strongest edge first")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_picks, tab_season, tab_log = st.tabs(["📋 Today's Picks", "📈 Season Tracker", "📓 Bet Log"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TODAY'S PICKS
# ══════════════════════════════════════════════════════════════════════════════
with tab_picks:
    today = datetime.date.today()
    if "pick_date" not in st.session_state:
        st.session_state.pick_date = today

    qc1, qc2, qc3, qc4, qc5 = st.columns([1, 1, 1, 2, 1])
    if qc1.button("Today",     use_container_width=True):
        st.session_state.pick_date = today
    if qc2.button("Yesterday", use_container_width=True):
        st.session_state.pick_date = today - datetime.timedelta(days=1)
    if qc3.button("–1 day",    use_container_width=True):
        st.session_state.pick_date = max(
            st.session_state.pick_date - datetime.timedelta(days=1),
            datetime.date(2026, 3, 28))
    with qc4:
        selected = st.date_input("Date", value=st.session_state.pick_date,
                                 min_value=datetime.date(2026, 3, 28),
                                 max_value=today, label_visibility="collapsed",
                                 key="pick_date_input")
        st.session_state.pick_date = selected
    if qc5.button("🔄",        use_container_width=True):
        st.cache_data.clear()

    date_str = st.session_state.pick_date.isoformat()
    st.caption(f"Showing picks for **{st.session_state.pick_date.strftime('%A, %B %d, %Y')}**")

    with st.expander("📖 How to read this — terms & definitions"):
        st.markdown("""
**Bet types**
- **ML (Moneyline)** — Bet on a team to win the game outright. No spread involved.
- **RL (Run Line)** — Baseball's version of a point spread. Standard is ±1.5 runs.
  - *Home -1.5* means the home team must win by 2 or more runs for you to win.
  - *Away +1.5* means the away team can lose by 1 or win, and you still win.
- **-2.5 / +2.5** — Alternate run line. Higher risk/reward than the standard ±1.5.
- **O/U (Over/Under)** — Bet on the combined total runs scored being over or under the number set by the book.

**Odds (American format)**
- **Negative odds** (e.g. -150): How much you must bet to win $100. -150 = bet $150 to win $100.
- **Positive odds** (e.g. +130): How much you win on a $100 bet. +130 = bet $100 to win $130.
- **Breakeven at -110** (standard juice): You need to win 52.4% of bets to break even.

**Model terms**
- **xwOBA** (Expected Weighted On-Base Average) — A pitcher quality metric. Lower is better for pitchers. Elite starters are below 0.280. League average is ~0.318. Above 0.350 is below average.
- **Model Confidence %** — The probability our model assigns to the recommended bet winning. 60% = model thinks this bet wins 6 out of 10 times.
- **Edge** — How much better our model probability is vs. what the market implies. +10% edge means the model sees a 10 percentage point advantage over the book's implied odds.
- **Market implied %** — The win probability baked into the current betting odds.
- **Predicted Score** — The model's expected final score based on 50,000 simulated games.
- **Typical Range** — The 25th to 75th percentile of simulated scores (middle 50% of outcomes).

**Confidence tiers**
- **★★ STRONG** — High confidence. Model sees significant edge vs. the market.
- **★ LEAN** — Moderate confidence. Smaller but still meaningful edge.
- **No signal** — Model sees no meaningful edge. These games are skipped.

**Lineup status**
- **Confirmed lineup** — Starting pitchers have been officially announced.
- **Projected lineup** — Starters are based on rotation projections, not yet confirmed.
""")

    with st.spinner("Loading picks..."):
        results = load_card(date_str)

    if not results:
        st.warning(f"No predictions for {selected.strftime('%B %d, %Y')} — "
                   "pipeline may not have run yet.")
        st.info("Run `python run_today.py --csv && python supabase_upload.py` "
                "to generate today's card.")
        st.stop()

    def _sort_key(r):
        rl_sig = str(r.get("rl_signal") or "")
        tier   = str(r.get("best_tier") or ("**" if "**" in rl_sig else
                                             "*"  if "*"  in rl_sig else ""))
        order  = 0 if tier == "**" else (1 if tier == "*" else 2)
        edge   = _safe_float(r.get("best_edge")) or abs(_safe_float(r.get("blended_rl"), 0.5) - 0.5)
        return (order, -edge)

    results = sorted(results, key=_sort_key)

    def _is_strong(r):
        rl_sig = str(r.get("rl_signal") or "")
        return str(r.get("best_tier") or "") == "**" or \
               ("**" in rl_sig and not r.get("best_tier"))
    def _is_lean(r):
        rl_sig = str(r.get("rl_signal") or "")
        return str(r.get("best_tier") or "") == "*" or \
               ("*" in rl_sig and "**" not in rl_sig and not r.get("best_tier"))

    strong = [r for r in results if _is_strong(r)]
    lean   = [r for r in results if _is_lean(r)]
    skip   = [r for r in results if not _is_strong(r) and not _is_lean(r)]

    # Summary bar
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Games Today",  len(results))
    c2.metric("Strong Plays", len(strong))
    c3.metric("Lean Plays",   len(lean))
    c4.metric("No Edge",      len(skip))

    # Quick-picks chip bar — all bets at a glance
    if strong or lean:
        chips_html = ""
        for r in strong + lean:
            label, _ = _format_bet_label(r)
            conf = int(_model_prob(r) * 100)
            cls = "pick-chip-strong" if _is_strong(r) else "pick-chip-lean"
            chips_html += f'<span class="pick-chip {cls}">{label} · {conf}%</span>'
        st.markdown(f'<div style="margin:8px 0 4px 0">{chips_html}</div>',
                    unsafe_allow_html=True)
    st.divider()

    n = 1
    if strong:
        st.markdown('<div class="section-hdr">Strong Plays</div>', unsafe_allow_html=True)
        for r in strong:
            render_card(r, n, "strong")
            n += 1

    if lean:
        st.markdown('<div class="section-hdr">Lean Plays</div>', unsafe_allow_html=True)
        for r in lean:
            render_card(r, n, "lean")
            n += 1

    if skip:
        with st.expander(f"No edge — {len(skip)} game{'s' if len(skip)!=1 else ''} skipped"):
            for r in skip:
                a, h = r.get("away_team",""), r.get("home_team","")
                vt   = r.get("vegas_total","")
                bl   = _safe_float(r.get("blended_rl"), 0.5)
                mw   = _safe_float(r.get("mc_home_win"), 0.5)
                tsig = r.get("total_signal","")
                tsig_html = f"  →  <b>{tsig}</b>" if tsig else ""
                st.markdown(
                    f'<div class="skip-card"><div class="skip-game">'
                    f'<strong>{a} @ {h}</strong>'
                    f'{f"  ·  O/U {vt}" if vt else ""}{tsig_html}'
                    f'<br><span style="font-size:.78rem">RL blend: {bl:.0%}  ·  Win%: {mw:.0%}  ·  '
                    f'No significant edge found</span>'
                    f'</div></div>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SEASON TRACKER
# ══════════════════════════════════════════════════════════════════════════════
with tab_season:
    bt = load_backtest()

    if bt.empty:
        st.info("Season tracker populates as games complete each day.")
    else:
        bt["date"] = pd.to_datetime(bt["date"], errors="coerce")
        bt = bt.dropna(subset=["date"])

        # ── Date range selector ───────────────────────────────────────────────
        st.markdown("**Date range**")
        dr_cols = st.columns(5)
        today_dt = datetime.date.today()

        # Use session state to track selected range
        if "tracker_range" not in st.session_state:
            st.session_state.tracker_range = "season"

        if dr_cols[0].button("Today",      use_container_width=True):
            st.session_state.tracker_range = "today"
        if dr_cols[1].button("This Week",  use_container_width=True):
            st.session_state.tracker_range = "week"
        if dr_cols[2].button("This Month", use_container_width=True):
            st.session_state.tracker_range = "month"
        if dr_cols[3].button("Full Season",use_container_width=True):
            st.session_state.tracker_range = "season"

        with dr_cols[4]:
            custom_range = st.date_input(
                "Custom", value=(bt["date"].min().date(), today_dt),
                key="custom_range", label_visibility="collapsed")

        rng = st.session_state.tracker_range
        if rng == "today":
            mask = bt["date"].dt.date == today_dt
            rng_label = "Today"
        elif rng == "week":
            week_start = today_dt - datetime.timedelta(days=today_dt.weekday())
            mask = bt["date"].dt.date >= week_start
            rng_label = "This Week"
        elif rng == "month":
            mask = (bt["date"].dt.year == today_dt.year) & \
                   (bt["date"].dt.month == today_dt.month)
            rng_label = "This Month"
        else:
            if isinstance(custom_range, (list, tuple)) and len(custom_range) == 2:
                mask = (bt["date"].dt.date >= custom_range[0]) & \
                       (bt["date"].dt.date <= custom_range[1])
                rng_label = f"{custom_range[0]} – {custom_range[1]}"
            else:
                mask = pd.Series([True] * len(bt), index=bt.index)
                rng_label = "Full Season"

        view = bt[mask].copy()
        st.divider()

        if view.empty:
            st.warning(f"No games in range: {rng_label}")
        else:
            # ── Only count games WITH a signal ────────────────────────────────
            if "signal" in view.columns:
                bet_view = view[view["signal"].notna() & (view["signal"] != "")]
            else:
                bet_view = pd.DataFrame()

            total_games = len(view)
            total_bets  = len(bet_view)

            st.caption(f"**{rng_label}**  ·  {total_games} games tracked  ·  "
                       f"{total_bets} generated a signal")

            if not bet_view.empty and "bet_win" in bet_view.columns:
                wins   = int(bet_view["bet_win"].sum())
                losses = total_bets - wins
                roi    = (wins * (100/110) - losses) / total_bets * 100

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Record",   f"{wins}–{losses}")
                m2.metric("Win Rate", f"{wins/total_bets:.1%}")
                m3.metric("ROI",      f"{roi:+.1f}%",
                          delta="profitable" if roi > 0 else "losing",
                          delta_color="normal" if roi > 0 else "inverse")
                m4.metric("Bets",     f"{total_bets}")
            else:
                st.info("No completed bet signals in this range yet.")

            st.divider()

            # ── Weekly breakdown (bet games only) ─────────────────────────────
            if not bet_view.empty and len(bet_view) > 0:
                bet_view["week"] = bet_view["date"].dt.to_period("W").astype(str)
                rows = []
                for week, wdf in bet_view.groupby("week"):
                    w = int(wdf["bet_win"].sum())
                    l = len(wdf) - w
                    r_roi = (w * (100/110) - l) / len(wdf) * 100
                    rows.append({"Week": week, "Bets": len(wdf),
                                 "Wins": w, "Losses": l,
                                 "Win%": f"{w/len(wdf):.1%}",
                                 "ROI":  f"{r_roi:+.1f}%"})
                if rows:
                    st.subheader("Weekly Breakdown  (bet signals only)")
                    st.dataframe(pd.DataFrame(rows), hide_index=True,
                                 use_container_width=True)

            # ── By signal type ─────────────────────────────────────────────────
            if "signal" in view.columns:
                sig_rows = []
                for s in ["AWAY +1.5 **","AWAY +1.5 *","HOME -1.5 **","HOME -1.5 *"]:
                    sub = view[view["signal"] == s]
                    if sub.empty:
                        continue
                    w = int(sub["bet_win"].sum())
                    l = len(sub) - w
                    r_roi = (w * (100/110) - l) / len(sub) * 100
                    sig_rows.append({"Signal": s, "Bets": len(sub),
                                     "Wins": w, "Losses": l,
                                     "Win%": f"{w/len(sub):.1%}",
                                     "ROI":  f"{r_roi:+.1f}%"})
                if sig_rows:
                    st.subheader("By Signal Type")
                    st.dataframe(pd.DataFrame(sig_rows), hide_index=True,
                                 use_container_width=True)

            # ── Home cover rate calibration ────────────────────────────────────
            if "blended_rl" in view.columns and "home_covers_rl" in view.columns:
                with st.expander("Model calibration (blended RL prob vs actual cover rate)"):
                    cal_data = view[["blended_rl","home_covers_rl"]].dropna()
                    if len(cal_data) >= 20:
                        cal_data["bucket"] = pd.cut(
                            cal_data["blended_rl"],
                            bins=[0,.30,.35,.40,.45,.50,.55,.60,1.0],
                            labels=["<.30",".30-.35",".35-.40",".40-.45",
                                    ".45-.50",".50-.55",".55-.60",">.60"])
                        cal = cal_data.groupby("bucket", observed=True).agg(
                            Games=("home_covers_rl","count"),
                            Actual=("home_covers_rl","mean"),
                            Model=("blended_rl","mean"),
                        ).reset_index()
                        cal["Actual"] = cal["Actual"].map("{:.1%}".format)
                        cal["Model"]  = cal["Model"].map("{:.1%}".format)
                        st.dataframe(cal, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BET LOG
# ══════════════════════════════════════════════════════════════════════════════
with tab_log:
    tracker = load_tracker()

    # ── Summary ───────────────────────────────────────────────────────────────
    if not tracker.empty and {"result","profit_loss"}.issubset(tracker.columns):
        completed = tracker[tracker["result"].isin(["WIN","LOSS","PUSH"])]
        if not completed.empty:
            wins    = (completed["result"] == "WIN").sum()
            losses  = (completed["result"] == "LOSS").sum()
            pushes  = (completed["result"] == "PUSH").sum()
            total_pl = pd.to_numeric(completed["profit_loss"], errors="coerce").sum()
            n_bets   = wins + losses + pushes
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Record",     f"{wins}–{losses}–{pushes}")
            t2.metric("Win %",      f"{wins/n_bets*100:.1f}%" if n_bets else "—")
            t3.metric("P&L (units)",f"{total_pl:+.2f}",
                      delta_color="normal" if total_pl >= 0 else "inverse")
            t4.metric("Total Bets", n_bets)
            st.divider()

    # ── Log a new bet ─────────────────────────────────────────────────────────
    with st.expander("➕ Log a bet", expanded=tracker.empty):
        log_date = st.date_input("Date", value=datetime.date.today(), key="log_date")

        # Load today's picks to pre-populate options
        picks_for_log = load_card(log_date.isoformat())
        pick_options  = ["-- Custom / Other --"]
        pick_map      = {}   # label → result dict
        for p in picks_for_log:
            label, _ = _format_bet_label(p)
            game      = p.get("game","")
            if label and game:
                display = f"{game}  →  {label}"
                pick_options.append(display)
                pick_map[display] = p

        selected_pick = st.selectbox(
            "Pick a recommendation from today (or choose Custom)",
            pick_options, key="log_pick")

        # Auto-fill from recommendation
        prefill = pick_map.get(selected_pick, {})
        pre_game    = prefill.get("game", "")
        pre_label, _= _format_bet_label(prefill) if prefill else ("", "")
        pre_odds    = prefill.get("best_market_odds") or -110
        pre_team    = ""
        if pre_label:
            parts = pre_label.split()
            pre_team = parts[0] if parts else ""

        st.caption("Fields below auto-fill from the recommendation. Edit anything before saving.")

        with st.form("log_bet_form", clear_on_submit=True):
            fa, fb = st.columns(2)
            bet_game = fa.text_input("Game", value=pre_game)
            bet_team = fb.text_input("Team / Side", value=pre_team,
                                     placeholder="e.g. TEX")

            fc, fd, fe = st.columns(3)
            bt_options = ["ML","RL +1.5","RL -1.5","RL +2.5","RL -2.5",
                          "OVER","UNDER","Parlay","Other"]
            # Guess type from pre_label
            pre_type = "ML"
            if "+1.5" in pre_label:   pre_type = "RL +1.5"
            elif "-1.5" in pre_label: pre_type = "RL -1.5"
            elif "+2.5" in pre_label: pre_type = "RL +2.5"
            elif "-2.5" in pre_label: pre_type = "RL -2.5"
            elif "OVER"  in pre_label: pre_type = "OVER"
            elif "UNDER" in pre_label: pre_type = "UNDER"

            bet_type   = fc.selectbox("Bet type", bt_options,
                                       index=bt_options.index(pre_type))
            bet_odds   = fd.number_input("Market odds", value=int(pre_odds), step=5)
            bet_result = fe.selectbox("Result", ["PENDING","WIN","LOSS","PUSH"])

            fg, fh = st.columns(2)
            is_parlay = fg.checkbox("Part of a parlay")
            parlay_id = fh.text_input("Parlay name", placeholder="e.g. Sunday 3-legger",
                                       disabled=not is_parlay)

            # Auto-calculate P&L when result is set and it's not a parlay
            auto_pl = 0.0
            if bet_result == "WIN":
                auto_pl = round(100/abs(bet_odds) if bet_odds < 0 else bet_odds/100, 2)
            elif bet_result == "LOSS":
                auto_pl = -1.0
            elif bet_result == "PUSH":
                auto_pl = 0.0

            fi, fj = st.columns(2)
            bet_pl    = fi.number_input("P&L (units)", value=float(auto_pl), step=0.1,
                                         help="Auto-calculated from odds. Edit if different.")
            notes     = fj.text_input("Notes (optional)")

            submitted = st.form_submit_button("✅ Log Bet", type="primary",
                                              use_container_width=True)

        if submitted:
            new_row = {
                "date":        str(log_date),
                "game":        bet_game,
                "bet_type":    bet_type,
                "team":        bet_team,
                "odds":        int(bet_odds),
                "units":       1.0,
                "is_parlay":   is_parlay,
                "parlay_id":   parlay_id if is_parlay else "",
                "result":      bet_result,
                "profit_loss": float(bet_pl),
                "notes":       notes,
            }
            if USE_SUPABASE:
                try:
                    _supabase().table("bet_tracker").insert(new_row).execute()
                    st.success("✅ Bet logged!")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Could not save: {e}")
            else:
                path = Path(__file__).parent / "data" / "raw" / "bet_tracker.csv"
                existing_local = pd.read_csv(path) if path.exists() else pd.DataFrame()
                updated = pd.concat([existing_local, pd.DataFrame([new_row])],
                                    ignore_index=True)
                updated.to_csv(path, index=False)
                st.success(f"✅ Saved to {path.name}")
                st.cache_data.clear()

    # ── Bet table ─────────────────────────────────────────────────────────────
    if tracker.empty:
        st.info("No bets logged yet. Use the form above to log your first bet.")
    else:
        # Parlay summary
        if "is_parlay" in tracker.columns:
            parlays = tracker[tracker["is_parlay"] == True]
            singles = tracker[tracker["is_parlay"] != True]
            if not parlays.empty:
                st.caption(f"**Singles:** {len(singles)}  ·  **Parlay legs:** {len(parlays)}")

        def _style_result(row):
            styles = [""] * len(row)
            if "result" in row.index:
                idx = row.index.get_loc("result")
                v = str(row["result"])
                styles[idx] = ("color:#4ade80;font-weight:bold" if v == "WIN"
                                else "color:#f87171;font-weight:bold" if v == "LOSS"
                                else "color:#facc15" if v == "PUSH" else "")
            return styles

        st.dataframe(tracker.style.apply(_style_result, axis=1),
                     hide_index=True, use_container_width=True)

        # Parlay breakdown
        if "is_parlay" in tracker.columns and "parlay_id" in tracker.columns:
            parlays_df = tracker[tracker["is_parlay"] == True]
            if not parlays_df.empty:
                with st.expander("Parlay breakdown"):
                    for pid, pdf in parlays_df.groupby("parlay_id"):
                        st.markdown(f"**{pid}**")
                        st.dataframe(pdf[["date","game","bet_type","team","odds","units",
                                         "result","profit_loss"]],
                                     hide_index=True, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Breakeven at -110 juice: 52.4%  ·  "
    "Model: 60% Monte Carlo + 40% XGBoost  ·  "
    "Base home -1.5 cover rate ~35.7%  ·  "
    "Score range = 25th–75th percentile of 50,000 simulations"
)
