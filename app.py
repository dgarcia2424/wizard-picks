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

/* Game matchup header */
.game-matchup   { font-size:1.55rem; font-weight:800; color:#f1f5f9;
                  letter-spacing:.01em; margin-bottom:2px; line-height:1.2; }
.game-matchup-sub { font-size:.82rem; color:#94a3b8; margin-bottom:10px; }
.bet-label-row  { font-size:.78rem; font-weight:700; color:#6b7280;
                  text-transform:uppercase; letter-spacing:.08em; margin-bottom:4px; }

/* Score prediction */
.score-section  { background:rgba(255,255,255,.04); border-radius:8px;
                  padding:14px 18px; margin:12px 0; }
.score-label    { font-size:.72rem; color:#94a3b8; text-transform:uppercase;
                  letter-spacing:.08em; margin-bottom:10px; }
.score-main     { font-size:1.15rem; font-weight:800; color:#f1f5f9;
                  margin-bottom:6px; }
.bp-row         { display:flex; align-items:center; gap:12px; margin:6px 0; }
.bp-team        { font-size:.84rem; font-weight:700; color:#cbd5e1;
                  width:36px; text-align:right; }
.bp-avg         { font-size:1.05rem; font-weight:800; color:#f1f5f9; width:28px; }
.bp-chart       { flex:1; height:18px; position:relative; max-width:160px; }
.bp-track       { height:4px; background:rgba(255,255,255,.15); border-radius:2px;
                  position:absolute; top:7px; width:100%; }
.bp-fill        { height:4px; border-radius:2px; position:absolute; top:0; }
.bp-fill-a      { background:#22c55e; }
.bp-fill-b      { background:#6b7280; }
.bp-dot         { width:12px; height:12px; border-radius:50%; background:#f1f5f9;
                  position:absolute; top:-4px; transform:translateX(-50%);
                  border:2px solid #111827; }
.bp-range       { font-size:.78rem; color:#94a3b8; white-space:nowrap; }
.score-divider  { border:none; border-top:1px solid rgba(255,255,255,.06);
                  margin:10px 0; }
.score-total    { font-size:.87rem; color:#cbd5e1; }
.ats-ci         { font-size:.84rem; color:#cbd5e1; margin-top:8px;
                  padding:6px 10px; background:rgba(255,255,255,.05);
                  border-radius:6px; border-left:3px solid #6366f1; }
.bet-action     { background:rgba(34,197,94,.10); border:1px solid rgba(34,197,94,.3);
                  border-radius:6px; padding:10px 16px; margin:10px 0;
                  font-size:.95rem; color:#d1fae5; font-weight:600; line-height:1.6; }

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
.meta-bar    { font-size:.8rem; color:#94a3b8; margin-top:10px;
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
            # Always pull game_date from the top-level row
            r["game_date"] = row.get("game_date", date_str)
            for col in ["game","home_team","away_team","rl_signal","total_signal",
                        "blended_rl","mc_rl","xgb_rl","mc_total","blended_total",
                        "vegas_total","vegas_ml_home","vegas_ml_away","home_sp","away_sp",
                        "home_sp_xwoba","away_sp_xwoba","home_sp_flag","away_sp_flag",
                        "temp_f","lineup_confirmed","game_time_et",
                        "best_line","best_tier","best_tier_capped","best_model_prob",
                        "best_market_odds","best_edge","best_market_implied","best_raw_model",
                        "market_deviation",
                        "mc_home_win","mc_home_cvr25","mc_away_cvr25",
                        "home_runs_mean","away_runs_mean",
                        "home_runs_lo","home_runs_hi","away_runs_lo","away_runs_hi",
                        "home_lineup_wrc","away_lineup_wrc",
                        "mc_nrfi_prob","mc_f5_total","mc_f5_home_win_prob",
                        "mc_f5_home_runs","mc_f5_away_runs",
                        "mc_home_sp_k_mean","mc_away_sp_k_mean",
                        "mc_home_sp_k_over_45","mc_away_sp_k_over_45",
                        "home_sp_expected_ip","away_sp_expected_ip",
                        "mc_p_home_scores_f1","mc_p_away_scores_f1"]:
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


@st.cache_data(ttl=300)
def load_historical_backtest() -> pd.DataFrame:
    """Load multi-year historical backtest from wizard_backtest_historical.

    Paginates in chunks of 1000 (Supabase default max) to get all rows.
    Only rows with a signal (bet rows) are stored in the table.
    """
    if USE_SUPABASE:
        client = _supabase()
        try:
            all_rows = []
            page_size = 1000
            offset = 0
            while True:
                resp = client.table("wizard_backtest_historical").select("*") \
                    .range(offset, offset + page_size - 1).execute()
                if not resp.data:
                    break
                all_rows.extend(resp.data)
                if len(resp.data) < page_size:
                    break
                offset += page_size
            return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    # Local fallback: concatenate all year CSVs (signal rows only)
    dfs = []
    for yr in [2023, 2024, 2025]:
        p = Path(__file__).parent / f"backtest_{yr}_results.csv"
        if p.exists():
            df = pd.read_csv(p)
            if "signal" in df.columns and "bet_win" in df.columns:
                df = df[df["signal"].notna() & (df["signal"] != "") & df["bet_win"].notna()]
            df["season"] = yr
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


@st.cache_data(ttl=300)
def load_model_history() -> pd.DataFrame:
    if USE_SUPABASE:
        client = _supabase()
        try:
            resp = client.table("wizard_model_history").select("*").order("date", desc=True).execute()
            return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    path = Path(__file__).parent / "backtest_mc_2026_results.csv"
    return pd.read_csv(path).sort_values("date", ascending=False) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=60)
def load_pipeline_health() -> dict:
    """Load pipeline health status from Supabase or local JSON."""
    import json as _json

    def _parse_row(row: dict) -> dict:
        """Ensure list fields are actual lists (Supabase stores them as JSON strings)."""
        for field in ["missing_critical", "missing_optional", "warnings"]:
            val = row.get(field, [])
            if isinstance(val, str):
                try:
                    row[field] = _json.loads(val)
                except Exception:
                    row[field] = []
        try:
            row["artifacts"] = _json.loads(row.get("artifacts_json", "{}"))
        except Exception:
            row["artifacts"] = {}
        if "next_scheduled_runs" not in row and "next_scheduled_runs" in row.get("artifacts", {}):
            row["next_scheduled_runs"] = row["artifacts"].get("next_scheduled_runs", {})
        return row

    if USE_SUPABASE:
        client = _supabase()
        try:
            today = datetime.date.today().isoformat()
            resp = client.table("wizard_pipeline_health").select("*").eq("date", today).execute()
            if resp.data:
                return _parse_row(resp.data[0])
        except Exception:
            pass
    # Fallback: local pipeline_status.json (only if <30 min old)
    path = Path(__file__).parent / "pipeline_status.json"
    if path.exists():
        try:
            age_min = (datetime.datetime.now().timestamp() - path.stat().st_mtime) / 60
            if age_min < 30:
                return _json.loads(path.read_text())
        except Exception:
            pass
    return {}


def render_health_banner(health: dict) -> None:
    """Render a compact pipeline health status banner in the picks tab."""
    if not health:
        return

    overall   = health.get("overall", "ok")
    picks_rdy = health.get("picks_ready", True)
    generated = health.get("generated_at", "")
    missing_c = health.get("missing_critical", [])
    missing_o = health.get("missing_optional", [])
    warnings  = health.get("warnings", [])
    next_runs = health.get("next_scheduled_runs", {})

    # Decide banner color/icon
    if overall == "critical":
        color  = "rgba(239,68,68,.12)"
        border = "rgba(239,68,68,.5)"
        icon   = "🔴"
        title  = "Pipeline issue — picks may be stale"
    elif overall == "warning":
        color  = "rgba(251,191,36,.10)"
        border = "rgba(251,191,36,.4)"
        icon   = "🟡"
        title  = "Some data not yet available"
    else:
        if not picks_rdy:
            color  = "rgba(251,191,36,.10)"
            border = "rgba(251,191,36,.4)"
            icon   = "🟡"
            title  = "Picks not yet generated today"
        else:
            color  = "rgba(34,197,94,.08)"
            border = "rgba(34,197,94,.3)"
            icon   = "🟢"
            title  = "All systems good"

    # Build detail lines
    details = []
    if missing_c:
        details.append(f"<b>Missing (critical):</b> {', '.join(missing_c)}")
    if missing_o:
        details.append(f"<b>Missing (optional):</b> {', '.join(missing_o)}")
    for w in warnings[:2]:   # cap at 2 warnings
        details.append(f"⚠ {w}")
    if next_runs:
        next_strs = []
        if "k_props_retry" in next_runs and missing_o and "k_props" in str(missing_o):
            next_strs.append(f"K props retry: {next_runs['k_props_retry']}")
        if "picks" in next_runs and not picks_rdy:
            next_strs.append(f"Picks: {next_runs['picks']}")
        if next_strs:
            details.append(f"🕐 Next update: {' · '.join(next_strs)}")
    if generated:
        try:
            gen_dt = datetime.datetime.fromisoformat(generated)
            ago_min = int((datetime.datetime.now() - gen_dt).total_seconds() / 60)
            details.append(f"Status checked {ago_min}m ago")
        except Exception:
            pass

    detail_html = "  &nbsp;·&nbsp;  ".join(details) if details else ""
    detail_block = f'<div style="font-size:.78rem;color:#9ca3af;margin-top:4px">{detail_html}</div>' if detail_html else ""

    # Only show banner if there's something to report (hide when all-green + picks ready)
    if overall == "ok" and picks_rdy and not details:
        return

    st.markdown(f"""
<div style="background:{color};border:1px solid {border};border-radius:8px;
            padding:10px 16px;margin-bottom:12px">
  <span style="font-weight:700;font-size:.9rem">{icon} {title}</span>
  {detail_block}
</div>""", unsafe_allow_html=True)


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


def _bet_description(r: dict) -> str:
    """Return a plain-English one-liner explaining exactly what to bet and why it wins."""
    away  = r.get("away_team", "")
    home  = r.get("home_team", "")
    line  = str(r.get("best_line") or "")
    odds  = r.get("best_market_odds")
    odds_str = f" ({int(odds):+d})" if odds is not None else ""
    line_l = line.lower()

    if "-2.5" in line_l and "home" in line_l:
        return (f"Bet <b>{home} -2.5{odds_str}</b> — back {home} to win by 3 or more runs. "
                f"You win if {home} wins by at least 3. You lose if {home} wins by 1–2 or {away} wins.")
    elif "+2.5" in line_l and "away" in line_l:
        return (f"Bet <b>{away} +2.5{odds_str}</b> — back {away} with a 2-run cushion. "
                f"You win if {away} wins outright, or loses by 1–2. You lose only if {home} wins by 3+.")
    elif "+2.5" in line_l and "home" in line_l:
        return (f"Bet <b>{home} +2.5{odds_str}</b> — back {home} with a 2-run cushion. "
                f"You win if {home} wins outright, or loses by 1–2. You lose only if {away} wins by 3+.")
    elif "-2.5" in line_l and "away" in line_l:
        return (f"Bet <b>{away} -2.5{odds_str}</b> — back {away} to win by 3 or more runs. "
                f"You win if {away} wins by at least 3. You lose if {away} wins by 1–2 or {home} wins.")
    elif "-1.5" in line_l and "home" in line_l:
        return (f"Bet <b>{home} -1.5{odds_str}</b> — back {home} to win by 2 or more runs. "
                f"You win if {home} wins by 2+. You lose if {home} wins by exactly 1, or {away} wins.")
    elif "+1.5" in line_l and "away" in line_l:
        return (f"Bet <b>{away} +1.5{odds_str}</b> — back {away} with a 1-run cushion. "
                f"You win if {away} wins outright or loses by exactly 1. You lose only if {home} wins by 2+.")
    elif "+1.5" in line_l and "home" in line_l:
        return (f"Bet <b>{home} +1.5{odds_str}</b> — back {home} with a 1-run cushion. "
                f"You win if {home} wins outright or loses by exactly 1. You lose only if {away} wins by 2+.")
    elif "-1.5" in line_l and "away" in line_l:
        return (f"Bet <b>{away} -1.5{odds_str}</b> — back {away} to win by 2 or more runs. "
                f"You win if {away} wins by 2+. You lose if {away} wins by exactly 1, or {home} wins.")
    elif "ml" in line_l and "home" in line_l:
        return (f"Bet <b>{home} ML{odds_str}</b> — back {home} to win the game outright. "
                f"No spread — {home} just needs to win. You lose if {away} wins by any margin.")
    elif "ml" in line_l and "away" in line_l:
        return (f"Bet <b>{away} ML{odds_str}</b> — back {away} to win the game outright. "
                f"No spread — {away} just needs to win. You lose if {home} wins by any margin.")
    else:
        # Fallback using legacy signal
        sig = str(r.get("rl_signal", ""))
        if "AWAY" in sig:
            return (f"Bet <b>{away} +1.5{odds_str}</b> — {away} wins or loses by exactly 1.")
        elif "HOME" in sig:
            return (f"Bet <b>{home} -1.5{odds_str}</b> — {home} must win by 2 or more.")
        return ""


def render_card(r: dict, n: int, tier: str):
    away, home = r.get("away_team",""), r.get("home_team","")
    bet_label, bet_sub = _format_bet_label(r)
    model_prob = _model_prob(r)
    edge       = _edge_pct(r)
    conf_pct   = int(model_prob * 100)

    fill_cls  = "conf-fill-strong" if tier == "strong" else "conf-fill-lean"
    card_cls  = "bet-card-strong"  if tier == "strong" else "bet-card-lean"
    badge_cls = "badge-strong"     if tier == "strong" else "badge-lean"

    # For ML bets on big underdogs (>+200), show edge instead of win probability.
    # "30% confidence" on a +750 underdog ML is misleading — the edge is what matters.
    best_line_str  = str(r.get("best_line") or "")
    best_odds_raw  = r.get("best_market_odds") or 0
    tier_capped    = bool(r.get("best_tier_capped") or False)
    is_ml_underdog = "ML" in best_line_str and (best_odds_raw or 0) > 200
    capped_note    = "  (odds capped)" if tier_capped else ""
    if is_ml_underdog:
        badge_txt = (f"★★ STRONG  ·  Edge: {edge:+.0f}%{capped_note}" if tier == "strong"
                     else f"★ LEAN  ·  Edge: {edge:+.0f}%{capped_note}")
    else:
        badge_txt = (f"★★ STRONG  ·  {conf_pct}% confidence{capped_note}" if tier == "strong"
                     else f"★ LEAN  ·  {conf_pct}% confidence{capped_note}")

    # Plain-English recommendation description
    bet_desc = _bet_description(r)

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
    if deviation >= 0.35:
        # Severe divergence — red-level warning, score bands suppressed
        deviation_warning = (
            f'<div style="background:rgba(239,68,68,.10);border:1px solid rgba(239,68,68,.40);'
            f'border-radius:6px;padding:8px 14px;margin:8px 0;font-size:.83rem;color:#fca5a5;">'
            f'🚨 <b>Large model vs market gap: {deviation:.0%}</b> — our pitcher-only model sees '
            f'a strong edge here, but Vegas disagrees by a wide margin. This often means the market '
            f'is pricing in factors we can\'t see (lineup, injury, bullpen, public money). '
            f'<b>Bet cautiously — half-unit max.</b></div>'
        )
    elif deviation >= 0.25:
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

    score_main = (f"{away} {_safe_float(am):.1f} — {home} {_safe_float(hm):.1f}"
                  if hm is not None and am is not None else "")

    # ── ATS confidence interval ──────────────────────────────────────────────
    # Derived from the 25th/75th score percentiles: spread = away_mean - home_mean
    # CI bounds: (away_lo - home_hi) to (away_hi - home_lo)
    ats_ci_html = ""
    if (hm is not None and am is not None and
            al is not None and ah is not None and
            hl is not None and hh is not None and
            deviation < 0.35):
        spread_mean = _safe_float(am) - _safe_float(hm)  # positive = away leads
        spread_lo   = _safe_float(al) - _safe_float(hh)  # away low - home high
        spread_hi   = _safe_float(ah) - _safe_float(hl)  # away high - home low
        # Format: "Model spread: TEX +0.5 (range: -1.8 to +2.8)"
        lead_team   = away if spread_mean >= 0 else home
        spread_abs  = abs(spread_mean)
        lo_fmt = f"{spread_lo:+.1f}"
        hi_fmt = f"{spread_hi:+.1f}"
        ats_ci_html = (
            f'<div class="ats-ci">'
            f'📐 <b>Model spread:</b> {lead_team} by {spread_abs:.1f} runs &nbsp;·&nbsp; '
            f'80% range: <b>{lo_fmt} to {hi_fmt}</b> (away minus home) &nbsp;·&nbsp; '
            f'Vegas RL: {away} {"+" if _safe_float(r.get("blended_rl"),0.5)<0.5 else "-"}1.5'
            f'</div>'
        )

    # ── Score bands ──────────────────────────────────────────────────────────
    if hm is not None and am is not None and deviation < 0.35:
        is_home_bet_score = "home" in str(r.get("best_line") or "").lower()
        away_cls = "bp-fill-a" if not is_home_bet_score else "bp-fill-b"
        home_cls = "bp-fill-a" if is_home_bet_score  else "bp-fill-b"
        bands_html = (
            f'<div style="margin:8px 0">'
            + _bp_row(away, _safe_float(am), al, ah, away_cls)
            + _bp_row(home, _safe_float(hm), hl, hh, home_cls)
            + f'<div style="font-size:.72rem;color:#64748b;margin-top:4px">'
            f'Bars = 25th–75th percentile · 50,000 simulations · '
            f'<span style="color:#22c55e">■</span> = recommended side</div>'
            + f'</div>'
        )
    elif deviation >= 0.35:
        bands_html = (
            '<div style="font-size:.82rem;color:#64748b;padding:6px 0">'
            '📊 Score bands hidden — model and market diverge too much for reliable score prediction.'
            '</div>'
        )
    else:
        bands_html = '<div style="font-size:.8rem;color:#64748b">Re-run pipeline for score prediction</div>'

    # ── O/U line ─────────────────────────────────────────────────────────────
    bt = r.get("blended_total") or r.get("mc_total")
    vt = r.get("vegas_total")
    total_sig = str(r.get("total_signal") or "")
    if bt and vt:
        diff   = _safe_float(bt) - _safe_float(vt)
        ou_dir = "OVER" if diff > 0 else "UNDER"
        if total_sig:
            score_total = (f"Model total: {_safe_float(bt):.1f} runs  ·  Vegas O/U: {vt}"
                           f"  →  <b style='color:#fef08a'>Also play: {total_sig}</b>")
        else:
            score_total = (f"Model total: {_safe_float(bt):.1f} runs  "
                           f"(Vegas O/U: {vt}  ·  model leans {ou_dir})")
    elif bt:
        score_total = f"Expected total: {_safe_float(bt):.1f} runs"
    else:
        score_total = ""

    score_total_html = f'<div class="score-total">{score_total}</div>' if score_total else ""

    # ── Props snapshot (F5, NRFI, K) ─────────────────────────────────────────
    nrfi_prob  = _safe_float(r.get("mc_nrfi_prob"), 0.0)
    f5_total   = _safe_float(r.get("mc_f5_total"))
    f5_hw      = _safe_float(r.get("mc_f5_home_win_prob"), 0.0)
    home_k     = _safe_float(r.get("mc_home_sp_k_mean"))
    away_k     = _safe_float(r.get("mc_away_sp_k_mean"))
    home_ip    = _safe_float(r.get("home_sp_expected_ip"), 5.5)
    away_ip    = _safe_float(r.get("away_sp_expected_ip"), 5.5)

    props_parts = []
    if nrfi_prob > 0:
        nrfi_color = "#4ade80" if nrfi_prob >= 0.65 else "#fde047" if nrfi_prob >= 0.55 else "#94a3b8"
        props_parts.append(f'<span style="color:{nrfi_color}">NRFI: {nrfi_prob:.0%}</span>')
    if f5_total > 0:
        props_parts.append(f'F5 total: {f5_total:.1f} runs')
    if f5_hw > 0:
        props_parts.append(f'F5 {home} win: {f5_hw:.0%}')
    if home_k > 0:
        props_parts.append(f'{home} SP Ks: {home_k:.1f} ({home_ip:.1f}ip)')
    if away_k > 0:
        props_parts.append(f'{away} SP Ks: {away_k:.1f} ({away_ip:.1f}ip)')

    props_html = ""
    if props_parts:
        props_inner = "  &nbsp;·&nbsp;  ".join(props_parts)
        props_html = f'<div style="font-size:.82rem;color:#94a3b8;margin-top:8px;padding:8px 0;border-top:1px solid rgba(255,255,255,.08)">{props_inner}</div>'

    # ── Why bullets ──────────────────────────────────────────────────────────
    bullets_html = "".join(f'<div class="why-item">{b}</div>' for b in _why_bullets(r))

    # ── Meta bar ─────────────────────────────────────────────────────────────
    lineup_str = "✅ Confirmed lineup" if r.get("lineup_confirmed") else "📋 Projected lineup"
    temp_f = _safe_float(r.get("temp_f"), 72)
    ml_str = ""
    if r.get("vegas_ml_home") is not None:
        ml_str = (f"ML: {home} {int(r['vegas_ml_home']):+d}"
                  + (f" / {away} {int(r['vegas_ml_away']):+d}" if r.get("vegas_ml_away") else ""))

    hf = f" [{r['home_sp_flag']}]" if r.get("home_sp_flag") not in ("NORMAL","UNKNOWN","","None",None) else ""
    af = f" [{r['away_sp_flag']}]" if r.get("away_sp_flag") not in ("NORMAL","UNKNOWN","","None",None) else ""
    sp_str = (f"{home} SP: {r.get('home_sp','').title()}{hf} xwOBA {_safe_float(r.get('home_sp_xwoba')):.3f}  "
              f"|  {away} SP: {r.get('away_sp','').title()}{af} xwOBA {_safe_float(r.get('away_sp_xwoba')):.3f}")

    # ── Game date + time label ────────────────────────────────────────────────
    game_date_val = r.get("game_date") or ""
    game_time_val = str(r.get("game_time_et") or "").strip()
    try:
        import datetime as _dt
        _d = _dt.date.fromisoformat(str(game_date_val)[:10])
        date_label = _d.strftime("%A, %B %d")
    except Exception:
        date_label = str(game_date_val)[:10] if game_date_val else ""
    if game_time_val:
        date_label = f"{date_label} · {game_time_val}"

    # ── Assemble card ────────────────────────────────────────────────────────
    st.markdown(f"""
<div class="{card_cls}">

  <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:4px">
    <div>
      <div style="font-size:.72rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-bottom:2px">#{n} &nbsp;·&nbsp; {date_label}</div>
      <div class="game-matchup">{away} @ {home}</div>
    </div>
    <div style="margin-top:4px"><span class="{badge_cls}">{badge_txt}</span></div>
  </div>

  <div style="margin-bottom:10px">
    <div style="font-size:.72rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px">Recommended bet</div>
    <div style="display:flex;align-items:center;gap:12px">
      <div style="font-size:1.25rem;font-weight:800;color:#4ade80">{bet_label}</div>
      <div style="font-size:.85rem;color:#94a3b8">{bet_sub.split('·')[-1].strip() if '·' in bet_sub else bet_sub}</div>
    </div>
  </div>

  <div class="conf-wrap">
    <div class="conf-detail">{conf_detail}</div>
    {deviation_warning}
  </div>

  <div class="score-section">
    <div class="score-label">Predicted score</div>
    <div class="score-main">{score_main}</div>
    {bands_html}
    {ats_ci_html}
    {score_total_html}
    {props_html}
  </div>

  <div class="bet-action">
    🎯 <b>The play:</b> {bet_desc if bet_desc else bet_label}
  </div>

  <div class="why-section">
    <div class="why-label">Why this bet</div>
    {bullets_html}
  </div>

  <div class="meta-bar">
    {temp_f:.0f}°F &nbsp;·&nbsp; {lineup_str}
    {'&nbsp;·&nbsp; ' + ml_str if ml_str else ''}
    <br>{sp_str}
  </div>

</div>""", unsafe_allow_html=True)


# ── App header ────────────────────────────────────────────────────────────────
st.markdown("## 🧙 The Wizard — MLB Picks")
st.caption("Picks updated daily after 8:30 AM ET  ·  Sorted by strongest edge first")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_picks, tab_season, tab_history, tab_performance, tab_raw = st.tabs([
    "📋 Today's Picks", "📈 Season Tracker",
    "🔬 Model History", "📊 Model Performance", "🗂️ Raw Data"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TODAY'S PICKS
# ══════════════════════════════════════════════════════════════════════════════
with tab_picks:
    today    = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)

    # ── Top controls ──────────────────────────────────────────────────────────
    ctrl_left, ctrl_mid, ctrl_right = st.columns([2, 3, 1])
    with ctrl_left:
        day_filter = st.radio(
            "Show games for",
            options=["Today", "Tomorrow", "Both"],
            index=0,
            horizontal=True,
            key="day_filter",
        )
    with ctrl_right:
        if st.button("🔄 Refresh", key="picks_refresh"):
            st.cache_data.clear()
            st.rerun()

    # Pipeline health banner
    _health = load_pipeline_health()
    render_health_banner(_health)

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

    # ── Load data for selected day(s) ─────────────────────────────────────────
    dates_to_show = []
    if day_filter in ("Today", "Both"):
        dates_to_show.append(today)
    if day_filter in ("Tomorrow", "Both"):
        dates_to_show.append(tomorrow)

    with st.spinner("Loading picks..."):
        cards_by_date = {}
        for d in dates_to_show:
            cards_by_date[d] = load_card(d.isoformat())

    all_results = [r for d in dates_to_show for r in cards_by_date[d]]

    if not all_results:
        st.warning(f"No predictions available — pipeline may not have run yet.")
        st.info("Run `python run_pipeline.py --daily` to generate picks, or start the scheduler with `python run_daily_scheduler.py`.")
        st.stop()

    def _sort_key(r):
        rl_sig = str(r.get("rl_signal") or "")
        tier   = str(r.get("best_tier") or ("**" if "**" in rl_sig else
                                             "*"  if "*"  in rl_sig else ""))
        order  = 0 if tier == "**" else (1 if tier == "*" else 2)
        edge   = _safe_float(r.get("best_edge")) or abs(_safe_float(r.get("blended_rl"), 0.5) - 0.5)
        return (order, -edge)

    def _is_strong(r):
        rl_sig = str(r.get("rl_signal") or "")
        return str(r.get("best_tier") or "") == "**" or \
               ("**" in rl_sig and not r.get("best_tier"))
    def _is_lean(r):
        rl_sig = str(r.get("rl_signal") or "")
        return str(r.get("best_tier") or "") == "*" or \
               ("*" in rl_sig and "**" not in rl_sig and not r.get("best_tier"))

    def _render_day_section(results: list, label: str, date_obj: datetime.date):
        """Render a full day's picks with section header, summary bar, and cards."""
        if not results:
            st.info(f"No predictions for {date_obj.strftime('%A, %B %d')} — run the pipeline to generate.")
            return

        results = sorted(results, key=_sort_key)
        strong  = [r for r in results if _is_strong(r)]
        lean    = [r for r in results if _is_lean(r)]
        skip    = [r for r in results if not _is_strong(r) and not _is_lean(r)]

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Games", len(results))
        c2.metric("Strong", len(strong))
        c3.metric("Lean",   len(lean))
        c4.metric("No Edge", len(skip))

        # Quick-picks chip bar
        if strong or lean:
            chips_html = ""
            for r in strong + lean:
                lbl, _ = _format_bet_label(r)
                conf   = int(_model_prob(r) * 100)
                cls    = "pick-chip-strong" if _is_strong(r) else "pick-chip-lean"
                chips_html += f'<span class="pick-chip {cls}">{lbl} · {conf}%</span>'
            st.markdown(f'<div style="margin:8px 0 4px 0">{chips_html}</div>',
                        unsafe_allow_html=True)

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
                    a, h   = r.get("away_team",""), r.get("home_team","")
                    vt     = r.get("vegas_total","")
                    bl     = _safe_float(r.get("blended_rl"), 0.5)
                    mw     = _safe_float(r.get("mc_home_win"), 0.5)
                    tsig   = r.get("total_signal","")
                    tsig_html = f"  →  <b>{tsig}</b>" if tsig else ""
                    st.markdown(
                        f'<div class="skip-card"><div class="skip-game">'
                        f'<strong>{a} @ {h}</strong>'
                        f'{f"  ·  O/U {vt}" if vt else ""}{tsig_html}'
                        f'<br><span style="font-size:.78rem">RL blend: {bl:.0%}  ·  '
                        f'Win%: {mw:.0%}  ·  No significant edge found</span>'
                        f'</div></div>',
                        unsafe_allow_html=True)

    # ── Render section(s) ─────────────────────────────────────────────────────
    if day_filter == "Both":
        # Today section
        st.markdown(
            f'<h3 style="margin:0 0 8px 0; color:#e2e8f0">📅 Today — '
            f'{today.strftime("%A, %B %d")}</h3>',
            unsafe_allow_html=True)
        _render_day_section(cards_by_date[today], "Today", today)

        st.divider()

        # Tomorrow section
        tomorrow_confirmed = any(
            r.get("lineup_confirmed") for r in cards_by_date.get(tomorrow, [])
        )
        lineup_note = "" if tomorrow_confirmed else " *(probable pitchers — lineups TBD)*"
        st.markdown(
            f'<h3 style="margin:16px 0 8px 0; color:#e2e8f0">📅 Tomorrow — '
            f'{tomorrow.strftime("%A, %B %d")}{lineup_note}</h3>',
            unsafe_allow_html=True)
        _render_day_section(cards_by_date.get(tomorrow, []), "Tomorrow", tomorrow)

    elif day_filter == "Tomorrow":
        tomorrow_confirmed = any(
            r.get("lineup_confirmed") for r in cards_by_date.get(tomorrow, [])
        )
        if not tomorrow_confirmed:
            st.caption("⚠️ Tomorrow's lineups not yet confirmed — showing probable pitchers.")
        _render_day_section(cards_by_date.get(tomorrow, []), "Tomorrow", tomorrow)

    else:  # Today
        _render_day_section(cards_by_date.get(today, []), "Today", today)


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

        if dr_cols[0].button("Today",       use_container_width=True, key="trk_today"):
            st.session_state.tracker_range = "today"
        if dr_cols[1].button("This Week",   use_container_width=True, key="trk_week"):
            st.session_state.tracker_range = "week"
        if dr_cols[2].button("This Month",  use_container_width=True, key="trk_month"):
            st.session_state.tracker_range = "month"
        if dr_cols[3].button("Full Season", use_container_width=True, key="trk_season"):
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
# TAB 3 — MODEL HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    mh = load_model_history()
    if mh.empty:
        st.info("Model history populates as the backtest runs. Check back after running backtest_mc_2026.py.")
    else:
        mh["date"] = pd.to_datetime(mh["date"], errors="coerce")
        mh = mh.dropna(subset=["date"])

        st.caption(f"Showing {len(mh)} games with completed MC model predictions")

        # Summary metrics
        col_a, col_b, col_c, col_d = st.columns(4)

        if "mc_nrfi_prob" in mh.columns and "f1_nrfi_actual" in mh.columns:
            nrfi_sub = mh[mh["mc_nrfi_prob"].notna() & mh["f1_nrfi_actual"].notna()]
            if len(nrfi_sub) > 0:
                nrfi_pred = (nrfi_sub["mc_nrfi_prob"] > 0.5).astype(int)
                nrfi_acc  = (nrfi_pred == nrfi_sub["f1_nrfi_actual"]).mean()
                col_a.metric("NRFI Accuracy", f"{nrfi_acc:.1%}", f"{len(nrfi_sub)} games")

        if "mc_f5_total" in mh.columns and "f5_total_actual" in mh.columns:
            f5_sub = mh[mh["mc_f5_total"].notna() & mh["f5_total_actual"].notna()]
            if len(f5_sub) > 0:
                f5_mae = (f5_sub["mc_f5_total"] - f5_sub["f5_total_actual"]).abs().mean()
                col_b.metric("F5 Total MAE", f"{f5_mae:.2f} runs", f"{len(f5_sub)} games")

        if "mc_home_sp_k_mean" in mh.columns and "home_sp_k_actual" in mh.columns:
            k_sub = mh[mh["mc_home_sp_k_mean"].notna() & mh["home_sp_k_actual"].notna()]
            if len(k_sub) > 0:
                k_mae = (k_sub["mc_home_sp_k_mean"] - k_sub["home_sp_k_actual"]).abs().mean()
                col_c.metric("SP K MAE", f"{k_mae:.2f} K/game", f"{len(k_sub)} games")

        if "bet_win" in mh.columns:
            bet_sub = mh[mh["bet_win"].notna()]
            if len(bet_sub) > 0:
                wr = bet_sub["bet_win"].mean()
                col_d.metric("RL Win Rate", f"{wr:.1%}", f"{len(bet_sub)} bets")

        st.divider()

        # Show recent games table
        display_cols = ["date","home_team","away_team","signal","bet_win",
                        "mc_nrfi_prob","f1_nrfi_actual","mc_f5_total","f5_total_actual",
                        "mc_home_sp_k_mean","home_sp_k_actual"]
        show_cols = [c for c in display_cols if c in mh.columns]

        if show_cols:
            display_df = mh[show_cols].copy()
            if "date" in display_df.columns:
                display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(display_df.head(50), hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_performance:
    import plotly.graph_objects as go
    import plotly.express as px

    # ── Signal filter — sits at top, affects EVERYTHING below ─────────────────
    if "perf_filter" not in st.session_state:
        st.session_state.perf_filter = "all"

    _fb1, _fb2, _fb3, _fb_spacer = st.columns([1, 1, 1, 3])
    if _fb1.button("All Signals",  use_container_width=True, key="pf_all"):
        st.session_state.perf_filter = "all"
    if _fb2.button("★★ Strong",    use_container_width=True, key="pf_strong"):
        st.session_state.perf_filter = "strong"
    if _fb3.button("★ Lean",       use_container_width=True, key="pf_lean"):
        st.session_state.perf_filter = "lean"

    _pf = st.session_state.perf_filter

    def _apply_signal_filter(df, pf):
        """Filter a dataframe's bet rows by signal tier."""
        base = df[df["signal"].notna() & (df["signal"] != "") & df["bet_win"].notna()]
        if pf == "strong":
            return base[base["signal"].str.contains(r"\*\*", na=False)]
        elif pf == "lean":
            return base[
                base["signal"].str.contains(r"\*", na=False) &
                ~base["signal"].str.contains(r"\*\*", na=False)
            ]
        return base

    _pf_label = {"all": "All signals", "strong": "★★ Strong only", "lean": "★ Lean only"}[_pf]
    st.caption(f"Filter: **{_pf_label}**")
    st.divider()

    # ── Multi-Season Overview ─────────────────────────────────────────────────
    st.subheader("📅 Multi-Season Overview")
    _hist_df = load_historical_backtest()
    _curr_df = load_backtest()

    _season_rows = []
    # Historical years (2023-2025)
    if not _hist_df.empty:
        for _yr, _grp in _hist_df.groupby("season"):
            _bets = _apply_signal_filter(_grp, _pf)
            if len(_bets) > 0:
                _w = int(_bets["bet_win"].sum())
                _l = len(_bets) - _w
                _wr = _w / len(_bets)
                _roi = (_w * 0.909 - _l) / len(_bets)
                _season_rows.append({
                    "Season": int(_yr),
                    "Bets": len(_bets),
                    "Record": f"{_w}-{_l}",
                    "Win %": f"{_wr:.1%}",
                    "ROI": f"{_roi:+.1%}",
                    "_win_rate": _wr,
                })
    # Current year 2026
    if not _curr_df.empty:
        _bets_26 = _apply_signal_filter(_curr_df, _pf)
        if len(_bets_26) > 0:
            _w26 = int(_bets_26["bet_win"].sum())
            _l26 = len(_bets_26) - _w26
            _wr26 = _w26 / len(_bets_26)
            _roi26 = (_w26 * 0.909 - _l26) / len(_bets_26)
            _season_rows.append({
                "Season": 2026,
                "Bets": len(_bets_26),
                "Record": f"{_w26}-{_l26}",
                "Win %": f"{_wr26:.1%}",
                "ROI": f"{_roi26:+.1%}",
                "_win_rate": _wr26,
            })

    if _season_rows:
        _season_rows = sorted(_season_rows, key=lambda x: x["Season"])

        # Plotly dark-mode color constants (defined here for use before _pl_layout)
        _DARK_BG_PRE  = "rgba(0,0,0,0)"
        _GREEN_PRE    = "#22c55e"
        _RED_PRE      = "#ef4444"
        _BLUE_PRE     = "#3b82f6"

        def _pl_layout_pre(fig, title="", height=280):
            fig.update_layout(
                title=title, height=height,
                paper_bgcolor=_DARK_BG_PRE, plot_bgcolor=_DARK_BG_PRE,
                font=dict(color="#e5e7eb", size=12),
                xaxis=dict(gridcolor="#374151", showgrid=True, zeroline=False),
                yaxis=dict(gridcolor="#374151", showgrid=True, zeroline=True,
                           zerolinecolor="#6b7280"),
                margin=dict(l=40, r=20, t=40 if title else 20, b=40),
                showlegend=False,
            )
            return fig

        _col1, _col2 = st.columns([2, 3])
        with _col1:
            _tbl_data = [{k: v for k, v in r.items() if not k.startswith("_")}
                         for r in _season_rows]
            st.dataframe(pd.DataFrame(_tbl_data), use_container_width=True, hide_index=True)

        with _col2:
            _fig_multi = go.Figure()
            _colors = [_GREEN_PRE if r["_win_rate"] > 0.525 else
                       (_RED_PRE if r["_win_rate"] < 0.475 else _BLUE_PRE)
                       for r in _season_rows]
            _fig_multi.add_trace(go.Bar(
                x=[str(r["Season"]) for r in _season_rows],
                y=[r["_win_rate"] for r in _season_rows],
                marker_color=_colors,
                text=[r["Win %"] for r in _season_rows],
                textposition="outside",
                textfont=dict(color="#ffffff", size=14, family="Arial Black"),
            ))
            _fig_multi.add_hline(y=0.5245, line_dash="dash",
                                  line_color="#f59e0b",
                                  annotation_text="Break-even (~52.4%)",
                                  annotation_position="top left")
            _fig_multi.update_yaxes(tickformat=".0%", range=[0.40, 0.80])
            _pl_layout_pre(_fig_multi, title="Win Rate by Season", height=260)
            st.plotly_chart(_fig_multi, use_container_width=True)
    else:
        st.info("No multi-season data available yet. Run backtest_historical.py to generate historical results.")

    st.markdown("---")

    # Load data — wizard_backtest has scores/totals; wizard_model_history has F1/F5/K
    _bt   = load_backtest()
    _mhst = load_model_history()

    _has_bt   = not _bt.empty
    _has_mh   = not _mhst.empty

    if not _has_bt and not _has_mh:
        st.info("2026 performance data will appear here once the backtest has been run.")
    else:
        # ── Pre-process wizard_backtest ────────────────────────────────────────
        if _has_bt:
            _bt = _bt.copy()
            _bt["date"] = pd.to_datetime(_bt["date"], errors="coerce")
            for _c in ["bet_win","home_covers_rl","home_score","away_score",
                       "model_total","actual_total","blended_rl"]:
                if _c in _bt.columns:
                    _bt[_c] = pd.to_numeric(_bt[_c], errors="coerce")
            _bt_bets = _bt[_bt["signal"].notna() & (_bt["signal"] != "") & _bt["bet_win"].notna()].copy()
            _bt_bets = _bt_bets.sort_values("date").reset_index(drop=True)

        # ── Pre-process wizard_model_history ──────────────────────────────────
        if _has_mh:
            _mhst = _mhst.copy()
            _date_col = "date" if "date" in _mhst.columns else "game_date"
            _mhst["date"] = pd.to_datetime(_mhst[_date_col], errors="coerce")
            for _c in ["f1_nrfi_actual","f5_total_actual","home_sp_k_actual","away_sp_k_actual",
                       "mc_nrfi_prob","mc_f5_total","home_covers_rl","bet_win","blended_rl"]:
                if _c in _mhst.columns:
                    _mhst[_c] = pd.to_numeric(_mhst[_c], errors="coerce")

        # ── Plotly helper ──────────────────────────────────────────────────────
        _DARK_BG  = "rgba(0,0,0,0)"
        _GREEN    = "#22c55e"
        _RED      = "#ef4444"
        _BLUE     = "#3b82f6"
        _AMBER    = "#f59e0b"
        _GRAY     = "#6b7280"

        def _pl_layout(fig, title="", height=280):
            fig.update_layout(
                title=title,
                height=height,
                paper_bgcolor=_DARK_BG,
                plot_bgcolor=_DARK_BG,
                font=dict(color="#e5e7eb", size=12),
                xaxis=dict(gridcolor="#374151", showgrid=True, zeroline=False),
                yaxis=dict(gridcolor="#374151", showgrid=True, zeroline=True,
                           zerolinecolor="#6b7280"),
                margin=dict(l=40, r=20, t=40 if title else 20, b=40),
                showlegend=False,
            )
            return fig

        # ══════════════════════════════════════════════════════════════════════
        # HEADER — overall summary
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("## 📊 2026 Season Performance Dashboard")
        st.caption("All completed games · Updated nightly")

        # Apply filter to create _bt_view — used for all metrics & charts below
        if _has_bt and not _bt_bets.empty:
            _bt_view = _apply_signal_filter(_bt_bets, _pf)
            _bt_view = _bt_view.copy()
        else:
            _bt_view = pd.DataFrame()

        if _has_bt and not _bt_view.empty:
            _w  = int(_bt_view["bet_win"].sum())
            _l  = len(_bt_view) - _w
            _roi = (_w * (100/110) - _l) / len(_bt_view) * 100
            _h1, _h2, _h3, _h4 = st.columns(4)
            _h1.metric("RL Record",  f"{_w}–{_l}")
            _h2.metric("Win Rate",   f"{_w/len(_bt_view):.1%}")
            _h3.metric("ROI (−110)", f"{_roi:+.1f}%",
                       delta="vs 52.4% break-even", delta_color="normal" if _roi > 0 else "inverse")
            _h4.metric("Total Bets", len(_bt_view))

        st.divider()

        # ── Sub-tabs ──────────────────────────────────────────────────────────
        _ptabs = st.tabs([
            "📉 Run Line",
            "🎯 Moneyline",
            "🔢 Totals",
            "⚡ NRFI / F1",
            "5️⃣  First 5",
            "⚾ Strikeouts",
            "🎓 Calibration",
        ])

        # ══════════════════════════════════════════════════════════════════════
        # SUB-TAB 1 — RUN LINE
        # ══════════════════════════════════════════════════════════════════════
        with _ptabs[0]:
            if not _has_bt or _bt_view.empty:
                st.info("No run-line bet data yet.")
            else:
                _rl = _bt_view.copy()
                _rl["unit_pl"]  = _rl["bet_win"].apply(lambda x: 100/110 if x == 1 else -1.0)
                _rl["cum_pl"]   = _rl["unit_pl"].cumsum()
                _rl["cum_bets"] = range(1, len(_rl) + 1)
                _rl["cum_wr"]   = _rl["bet_win"].cumsum() / _rl["cum_bets"]
                _rl["label"]    = _rl["date"].dt.strftime("%m/%d") + " · " + _rl.get("game", _rl.index.astype(str))

                _c1, _c2, _c3, _c4 = st.columns(4)
                _w = int(_rl["bet_win"].sum()); _l = len(_rl) - _w
                _roi_rl = (_w * (100/110) - _l) / len(_rl) * 100
                _c1.metric("Record", f"{_w}–{_l}")
                _c2.metric("Win Rate", f"{_w/len(_rl):.1%}")
                _c3.metric("ROI", f"{_roi_rl:+.1f}%")
                _c4.metric("Total Bets", len(_rl))

                # ── Cumulative P&L chart ───────────────────────────────────
                _fig_rl = go.Figure()
                _fig_rl.add_trace(go.Scatter(
                    x=list(range(1, len(_rl)+1)),
                    y=_rl["cum_pl"],
                    mode="lines",
                    line=dict(color=_GREEN, width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(34,197,94,0.12)",
                    hovertemplate="Bet #%{x}<br>Cum P&L: %{y:.2f}u<extra></extra>",
                    name="Cum P&L",
                ))
                _fig_rl.add_hline(y=0, line_color=_GRAY, line_width=1)
                _pl_layout(_fig_rl, title="Cumulative P&L (units, −110 juice)", height=280)
                _fig_rl.update_xaxes(title="Bet #")
                _fig_rl.update_yaxes(title="Units")
                st.plotly_chart(_fig_rl, use_container_width=True)

                # ── By signal type ─────────────────────────────────────────
                _sig_grp = _rl.groupby("signal").agg(
                    Bets=("bet_win","count"),
                    Wins=("bet_win","sum"),
                ).reset_index()
                _sig_grp["Win %"] = (_sig_grp["Wins"]/_sig_grp["Bets"]*100).map("{:.1f}%".format)
                _sig_grp["ROI"]   = ((_sig_grp["Wins"]*(100/110) - (_sig_grp["Bets"]-_sig_grp["Wins"])) / _sig_grp["Bets"] * 100).map("{:+.1f}%".format)

                st.markdown("**By Signal Type**")
                _fa, _fb = st.columns([2,3])
                _fa.dataframe(_sig_grp.rename(columns={"signal":"Signal"}),
                              hide_index=True, use_container_width=True)

                # ── Win rate over time (rolling 10) ────────────────────────
                if len(_rl) >= 10:
                    _rl["roll_wr"] = _rl["bet_win"].rolling(10, min_periods=5).mean()
                    _fig_roll = go.Figure()
                    _fig_roll.add_trace(go.Scatter(
                        x=list(range(1, len(_rl)+1)),
                        y=_rl["roll_wr"],
                        mode="lines",
                        line=dict(color=_BLUE, width=2),
                        hovertemplate="Bet #%{x}<br>Rolling 10 WR: %{y:.1%}<extra></extra>",
                    ))
                    _fig_roll.add_hline(y=0.524, line_dash="dash", line_color=_AMBER,
                                        annotation_text="Breakeven (52.4%)",
                                        annotation_position="bottom right")
                    _pl_layout(_fig_roll, title="Rolling 10-Bet Win Rate", height=220)
                    _fig_roll.update_yaxes(title="Win Rate", tickformat=".0%")
                    _fb.plotly_chart(_fig_roll, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════
        # SUB-TAB 2 — MONEYLINE
        # ══════════════════════════════════════════════════════════════════════
        with _ptabs[1]:
            if not _has_bt or _bt_view.empty or "home_score" not in _bt.columns:
                st.info("Score data needed for moneyline tracking.")
            else:
                _ml = _bt_view.copy()
                _ml = _ml[_ml["home_score"].notna() & _ml["away_score"].notna()]
                if _ml.empty:
                    st.info("No completed score data yet.")
                else:
                    _ml["is_home_signal"] = _ml["signal"].str.startswith("HOME")
                    _ml["home_won_ml"]    = (_ml["home_score"] > _ml["away_score"])
                    _ml["ml_correct"]     = (_ml["is_home_signal"] == _ml["home_won_ml"])

                    _ml_w = int(_ml["ml_correct"].sum())
                    _ml_l = len(_ml) - _ml_w
                    _ml_wr = _ml_w / len(_ml)

                    _mc1, _mc2, _mc3 = st.columns(3)
                    _mc1.metric("ML Record", f"{_ml_w}–{_ml_l}")
                    _mc2.metric("ML Win Rate", f"{_ml_wr:.1%}",
                                delta="vs 50% coin-flip",
                                delta_color="normal" if _ml_wr > 0.5 else "inverse")
                    _mc3.metric("Games", len(_ml))

                    st.caption("ML derived from RL signals — when model picks HOME -1.5, did home team win outright? When AWAY +1.5, did away team win outright?")

                    # ── By signal ─────────────────────────────────────────
                    _ml_sig = _ml.groupby("signal").agg(
                        Bets=("ml_correct","count"),
                        ML_Wins=("ml_correct","sum"),
                    ).reset_index()
                    _ml_sig["ML Win %"] = (_ml_sig["ML_Wins"]/_ml_sig["Bets"]*100).map("{:.1f}%".format)
                    st.markdown("**ML Win Rate by Signal**")
                    st.dataframe(_ml_sig.rename(columns={"signal":"Signal"}),
                                 hide_index=True, use_container_width=True)

                    # ── ML vs RL comparison bar ────────────────────────────
                    _comp_df = pd.DataFrame({
                        "Bet Type": ["Run Line (−1.5/+1.5)", "Moneyline (derived)"],
                        "Win Rate": [
                            _bt_view["bet_win"].mean() * 100,
                            _ml_wr * 100,
                        ],
                        "Color": [_GREEN, _BLUE],
                    })
                    _fig_ml = go.Figure(go.Bar(
                        x=_comp_df["Bet Type"],
                        y=_comp_df["Win Rate"],
                        marker_color=_comp_df["Color"],
                        text=[f"{v:.1f}%" for v in _comp_df["Win Rate"]],
                        textposition="outside",
                    ))
                    _fig_ml.add_hline(y=52.4, line_dash="dash", line_color=_AMBER,
                                      annotation_text="Breakeven")
                    _pl_layout(_fig_ml, title="RL vs ML Win Rate Comparison", height=300)
                    _fig_ml.update_yaxes(title="Win Rate (%)", range=[0, 90])
                    st.plotly_chart(_fig_ml, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════
        # SUB-TAB 3 — TOTALS
        # ══════════════════════════════════════════════════════════════════════
        with _ptabs[2]:
            if not _has_bt or "model_total" not in _bt.columns:
                st.info("Model total predictions not available yet.")
            else:
                _tot = _bt[_bt["model_total"].notna() & _bt["actual_total"].notna()].copy()
                if _tot.empty:
                    st.info("No completed game totals yet.")
                else:
                    _tot["error"] = _tot["model_total"] - _tot["actual_total"]
                    _mae   = float(_tot["error"].abs().mean())
                    _bias  = float(_tot["error"].mean())
                    _model_avg = float(_tot["model_total"].mean())
                    _actual_avg = float(_tot["actual_total"].mean())

                    _tc1, _tc2, _tc3, _tc4 = st.columns(4)
                    _tc1.metric("Model Avg Total", f"{_model_avg:.1f} runs")
                    _tc2.metric("Actual Avg Total", f"{_actual_avg:.1f} runs")
                    _tc3.metric("MAE", f"{_mae:.2f} runs")
                    _tc4.metric("Bias", f"{_bias:+.2f}",
                                help="Positive = model over-predicts scoring")

                    # ── Scatter: model_total vs actual_total ──────────────
                    _fig_sc = go.Figure()
                    _fig_sc.add_trace(go.Scatter(
                        x=_tot["model_total"],
                        y=_tot["actual_total"],
                        mode="markers",
                        marker=dict(color=_BLUE, opacity=0.6, size=6),
                        hovertemplate="Model: %{x:.1f}<br>Actual: %{y}<extra></extra>",
                    ))
                    # Perfect prediction line
                    _mn = min(_tot["model_total"].min(), _tot["actual_total"].min())
                    _mx = max(_tot["model_total"].max(), _tot["actual_total"].max())
                    _fig_sc.add_trace(go.Scatter(
                        x=[_mn, _mx], y=[_mn, _mx],
                        mode="lines",
                        line=dict(color=_AMBER, dash="dash", width=1.5),
                        name="Perfect",
                    ))
                    _pl_layout(_fig_sc, title="Model Predicted Total vs Actual Total", height=300)
                    _fig_sc.update_xaxes(title="Model Total (runs)")
                    _fig_sc.update_yaxes(title="Actual Total (runs)")
                    _fig_sc.update_layout(showlegend=True,
                                          legend=dict(font=dict(color="#e5e7eb")))
                    st.plotly_chart(_fig_sc, use_container_width=True)

                    # ── Distribution of actual totals ──────────────────────
                    _fig_hist = go.Figure()
                    _fig_hist.add_trace(go.Histogram(
                        x=_tot["actual_total"],
                        nbinsx=20,
                        marker_color=_BLUE,
                        opacity=0.75,
                        name="Actual",
                    ))
                    _fig_hist.add_vline(x=_actual_avg, line_dash="dash", line_color=_GREEN,
                                        annotation_text=f"Avg {_actual_avg:.1f}",
                                        annotation_position="top right")
                    _pl_layout(_fig_hist, title="Distribution of Game Totals (Actual Runs)", height=250)
                    _fig_hist.update_xaxes(title="Total Runs")
                    _fig_hist.update_yaxes(title="Games")
                    st.plotly_chart(_fig_hist, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════
        # SUB-TAB 4 — NRFI / F1
        # ══════════════════════════════════════════════════════════════════════
        with _ptabs[3]:
            _f1_src = _mhst if _has_mh and "f1_nrfi_actual" in _mhst.columns else (
                _bt if _has_bt and "home_covers_rl" in _bt.columns else None
            )
            # Use the source with f1_nrfi data
            _nrfi_col = None
            if _has_mh and "f1_nrfi_actual" in _mhst.columns:
                _f1df = _mhst[_mhst["f1_nrfi_actual"].notna()].copy()
                _nrfi_col = "f1_nrfi_actual"
            else:
                _f1df = pd.DataFrame()

            if _f1df.empty:
                st.info("First-inning NRFI data not yet available.")
            else:
                _nrfi_rate = float(_f1df[_nrfi_col].mean())
                _nrfi_n    = len(_f1df)

                _n1, _n2, _n3 = st.columns(3)
                _n1.metric("NRFI Rate", f"{_nrfi_rate:.1%}", help="Games where neither team scored in inning 1")
                _n2.metric("YRFI Rate", f"{1-_nrfi_rate:.1%}", help="Games where at least one team scored in inning 1")
                _n3.metric("Games Tracked", _nrfi_n)

                # ── NRFI rate over time ────────────────────────────────────
                _f1df = _f1df.sort_values("date")
                _f1df["cum_nrfi_rate"] = _f1df[_nrfi_col].cumsum() / (pd.Series(range(1, len(_f1df)+1), index=_f1df.index))
                _fig_nrfi = go.Figure()
                _fig_nrfi.add_trace(go.Scatter(
                    x=_f1df["date"],
                    y=_f1df["cum_nrfi_rate"],
                    mode="lines",
                    line=dict(color=_AMBER, width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(245,158,11,0.10)",
                    hovertemplate="%{x|%b %d}<br>Cum NRFI Rate: %{y:.1%}<extra></extra>",
                ))
                _fig_nrfi.add_hline(y=0.5, line_dash="dash", line_color=_GRAY,
                                    annotation_text="50%")
                _pl_layout(_fig_nrfi, title="Cumulative NRFI Rate Over Season", height=250)
                _fig_nrfi.update_yaxes(title="NRFI Rate", tickformat=".0%")
                st.plotly_chart(_fig_nrfi, use_container_width=True)

                # MC model predictions (when available)
                if _has_mh and "mc_nrfi_prob" in _mhst.columns:
                    _nrfi_mc = _mhst[_mhst["mc_nrfi_prob"].notna() & _mhst["f1_nrfi_actual"].notna()].copy()
                    if len(_nrfi_mc) >= 10:
                        _nrfi_mc["predicted"] = (_nrfi_mc["mc_nrfi_prob"] > 0.5).astype(int)
                        _nrfi_mc["correct"]   = (_nrfi_mc["predicted"] == _nrfi_mc["f1_nrfi_actual"]).astype(int)
                        _nrfi_mc["bucket"]    = pd.cut(
                            _nrfi_mc["mc_nrfi_prob"],
                            bins=[0, 0.40, 0.50, 0.60, 0.70, 1.0],
                            labels=["<40%","40–50%","50–60%","60–70%",">70%"]
                        )
                        _cal = _nrfi_mc.groupby("bucket", observed=True).agg(
                            Games=("correct","count"),
                            Model_Prob=("mc_nrfi_prob","mean"),
                            Actual_Rate=("f1_nrfi_actual","mean"),
                            Accuracy=("correct","mean"),
                        ).reset_index()
                        st.markdown("**NRFI Model Calibration** (when MC predictions available)")
                        _fig_nrfi_cal = go.Figure()
                        _fig_nrfi_cal.add_trace(go.Bar(
                            x=_cal["bucket"].astype(str),
                            y=_cal["Actual_Rate"],
                            marker_color=_AMBER,
                            name="Actual NRFI Rate",
                            text=[f"{v:.1%}" for v in _cal["Actual_Rate"]],
                            textposition="outside",
                        ))
                        _fig_nrfi_cal.add_trace(go.Scatter(
                            x=_cal["bucket"].astype(str),
                            y=_cal["Model_Prob"],
                            mode="lines+markers",
                            line=dict(color=_BLUE, width=2),
                            name="Model Probability",
                        ))
                        _pl_layout(_fig_nrfi_cal, title="NRFI Model Probability vs Actual Rate", height=300)
                        _fig_nrfi_cal.update_layout(showlegend=True,
                                                    legend=dict(font=dict(color="#e5e7eb")))
                        st.plotly_chart(_fig_nrfi_cal, use_container_width=True)
                    else:
                        st.caption("🔜 NRFI model predictions will populate once MC columns are included in the backtest.")

        # ══════════════════════════════════════════════════════════════════════
        # SUB-TAB 5 — FIRST 5 INNINGS
        # ══════════════════════════════════════════════════════════════════════
        with _ptabs[4]:
            _f5_available = _has_mh and "f5_total_actual" in _mhst.columns
            if not _f5_available:
                st.info("First-5 data not yet available.")
            else:
                _f5df = _mhst[_mhst["f5_total_actual"].notna()].copy()
                if _f5df.empty:
                    st.info("No F5 data yet.")
                else:
                    _f5_avg  = float(_f5df["f5_total_actual"].mean())
                    _f5_n    = len(_f5df)
                    _f5_over_5 = float((_f5df["f5_total_actual"] > 5).mean())

                    _f51, _f52, _f53 = st.columns(3)
                    _f51.metric("Avg F5 Total", f"{_f5_avg:.2f} runs")
                    _f52.metric("Over 5 Rate", f"{_f5_over_5:.1%}", help="% of F5 totals > 5 runs")
                    _f53.metric("Games Tracked", _f5_n)

                    # ── F5 total distribution ──────────────────────────────
                    _fig_f5 = go.Figure()
                    _fig_f5.add_trace(go.Histogram(
                        x=_f5df["f5_total_actual"],
                        nbinsx=15,
                        marker_color=_BLUE,
                        opacity=0.75,
                    ))
                    _fig_f5.add_vline(x=_f5_avg, line_dash="dash", line_color=_GREEN,
                                      annotation_text=f"Avg {_f5_avg:.2f}",
                                      annotation_position="top right")
                    _fig_f5.add_vline(x=5, line_dash="dot", line_color=_AMBER,
                                      annotation_text="O/U 5",
                                      annotation_position="top left")
                    _pl_layout(_fig_f5, title="First 5 Innings — Total Runs Distribution", height=280)
                    _fig_f5.update_xaxes(title="F5 Total Runs")
                    _fig_f5.update_yaxes(title="Games")
                    st.plotly_chart(_fig_f5, use_container_width=True)

                    # MC F5 prediction accuracy (when available)
                    if "mc_f5_total" in _mhst.columns:
                        _f5_mc = _mhst[_mhst["mc_f5_total"].notna() & _mhst["f5_total_actual"].notna()].copy()
                        if len(_f5_mc) >= 5:
                            _f5_mae  = float((_f5_mc["mc_f5_total"] - _f5_mc["f5_total_actual"]).abs().mean())
                            _f5_bias = float((_f5_mc["mc_f5_total"] - _f5_mc["f5_total_actual"]).mean())
                            _fa1, _fa2 = st.columns(2)
                            _fa1.metric("F5 Prediction MAE", f"{_f5_mae:.2f} runs")
                            _fa2.metric("F5 Prediction Bias", f"{_f5_bias:+.2f}")
                        else:
                            st.caption("🔜 F5 model predictions will populate once MC F5 columns are included in the backtest.")

        # ══════════════════════════════════════════════════════════════════════
        # SUB-TAB 6 — STRIKEOUTS
        # ══════════════════════════════════════════════════════════════════════
        with _ptabs[5]:
            _k_avail = _has_mh and "home_sp_k_actual" in _mhst.columns
            if not _k_avail:
                st.info("SP strikeout data not yet available.")
            else:
                _kdf = _mhst[_mhst["home_sp_k_actual"].notna() | _mhst["away_sp_k_actual"].notna()].copy()
                # Build long format for combined K distribution
                _k_home = _kdf[["date","home_sp","home_sp_k_actual"]].rename(
                    columns={"home_sp":"Pitcher","home_sp_k_actual":"K"})
                _k_home["Role"] = "Home SP"
                _k_away = _kdf[["date","away_sp","away_sp_k_actual"]].rename(
                    columns={"away_sp":"Pitcher","away_sp_k_actual":"K"})
                _k_away["Role"] = "Away SP"
                _k_long = pd.concat([_k_home, _k_away]).dropna(subset=["K"])

                _k_avg_h = float(_kdf["home_sp_k_actual"].mean()) if "home_sp_k_actual" in _kdf else 0
                _k_avg_a = float(_kdf["away_sp_k_actual"].mean()) if "away_sp_k_actual" in _kdf else 0
                _k_avg_all = float(_k_long["K"].mean())

                _k1, _k2, _k3 = st.columns(3)
                _k1.metric("Home SP Avg K", f"{_k_avg_h:.1f}")
                _k2.metric("Away SP Avg K", f"{_k_avg_a:.1f}")
                _k3.metric("Overall SP Avg K", f"{_k_avg_all:.1f}")

                # ── K distribution by home/away ────────────────────────────
                _fig_k = go.Figure()
                _fig_k.add_trace(go.Histogram(
                    x=_kdf["home_sp_k_actual"].dropna(),
                    nbinsx=15,
                    name="Home SP",
                    marker_color=_BLUE,
                    opacity=0.65,
                ))
                _fig_k.add_trace(go.Histogram(
                    x=_kdf["away_sp_k_actual"].dropna(),
                    nbinsx=15,
                    name="Away SP",
                    marker_color=_GREEN,
                    opacity=0.65,
                ))
                _fig_k.update_layout(barmode="overlay")
                _pl_layout(_fig_k, title="SP Strikeout Distribution — Home vs Away", height=280)
                _fig_k.update_xaxes(title="Strikeouts")
                _fig_k.update_yaxes(title="Starts")
                _fig_k.update_layout(showlegend=True, legend=dict(font=dict(color="#e5e7eb")))
                st.plotly_chart(_fig_k, use_container_width=True)

                # ── Top K performers ──────────────────────────────────────
                _k_by_sp = _k_long.groupby("Pitcher").agg(
                    Starts=("K","count"),
                    Avg_K=("K","mean"),
                    Max_K=("K","max"),
                    Total_K=("K","sum"),
                ).reset_index().sort_values("Avg_K", ascending=False)
                _k_by_sp = _k_by_sp[_k_by_sp["Starts"] >= 2].head(15)
                _k_by_sp["Avg_K"] = _k_by_sp["Avg_K"].map("{:.1f}".format)
                st.markdown("**Top K Performers** (min 2 starts)")
                st.dataframe(_k_by_sp, hide_index=True, use_container_width=True)

                # MC K accuracy (when available)
                if _has_mh and "mc_home_sp_k_mean" in _mhst.columns:
                    _k_mc = _mhst[_mhst["mc_home_sp_k_mean"].notna() & _mhst["home_sp_k_actual"].notna()].copy()
                    if len(_k_mc) >= 5:
                        _k_mae  = float((_k_mc["mc_home_sp_k_mean"] - _k_mc["home_sp_k_actual"]).abs().mean())
                        _k_bias = float((_k_mc["mc_home_sp_k_mean"] - _k_mc["home_sp_k_actual"]).mean())
                        _km1, _km2 = st.columns(2)
                        _km1.metric("K Prediction MAE", f"{_k_mae:.2f} Ks")
                        _km2.metric("K Prediction Bias", f"{_k_bias:+.2f} Ks",
                                    help="Positive = model over-predicts Ks")
                    else:
                        st.caption("🔜 K model predictions will show once MC K columns are included in the backtest.")

        # ══════════════════════════════════════════════════════════════════════
        # SUB-TAB 7 — MODEL CALIBRATION (label/train analysis)
        # ══════════════════════════════════════════════════════════════════════
        with _ptabs[6]:
            st.markdown("#### Model Calibration — How well do predicted probabilities match actual outcomes?")
            st.caption("A well-calibrated model has actual win rates that closely track predicted probabilities.")

            _src = _bt if (_has_bt and "blended_rl" in _bt.columns and "home_covers_rl" in _bt.columns) else None
            if _src is None:
                st.info("Calibration data not yet available.")
            else:
                _cal_df = _src[_src["blended_rl"].notna() & _src["home_covers_rl"].notna()].copy()
                _cal_df["home_covers_rl"] = pd.to_numeric(_cal_df["home_covers_rl"], errors="coerce")

                if len(_cal_df) >= 20:
                    _cal_df["bucket"] = pd.cut(
                        _cal_df["blended_rl"],
                        bins=[0, .35, .40, .45, .50, .55, .60, .65, 1.0],
                        labels=["<35%","35–40%","40–45%","45–50%",
                                "50–55%","55–60%","60–65%",">65%"]
                    )
                    _cal = _cal_df.groupby("bucket", observed=True).agg(
                        Games=("home_covers_rl","count"),
                        Model_Prob=("blended_rl","mean"),
                        Actual_Cover=("home_covers_rl","mean"),
                    ).reset_index()

                    # ── Calibration chart ──────────────────────────────────
                    _fig_cal = go.Figure()
                    # Perfect calibration diagonal
                    _fig_cal.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode="lines",
                        line=dict(color=_GRAY, dash="dash", width=1.5),
                        name="Perfect Calibration",
                    ))
                    _fig_cal.add_trace(go.Scatter(
                        x=_cal["Model_Prob"],
                        y=_cal["Actual_Cover"],
                        mode="markers+lines",
                        marker=dict(color=_GREEN, size=10,
                                    line=dict(color="white", width=1.5)),
                        line=dict(color=_GREEN, width=2),
                        customdata=_cal[["Games","bucket"]].values,
                        hovertemplate="Bucket: %{customdata[1]}<br>Model: %{x:.1%}<br>Actual: %{y:.1%}<br>Games: %{customdata[0]}<extra></extra>",
                        name="Model",
                    ))
                    _pl_layout(_fig_cal, title="RL Probability Calibration — Blended Model", height=340)
                    _fig_cal.update_xaxes(title="Model Predicted Probability", tickformat=".0%", range=[0.2, 0.8])
                    _fig_cal.update_yaxes(title="Actual Home Cover Rate", tickformat=".0%", range=[0.0, 1.0])
                    _fig_cal.update_layout(showlegend=True, legend=dict(font=dict(color="#e5e7eb")))
                    st.plotly_chart(_fig_cal, use_container_width=True)

                    # ── Calibration table ──────────────────────────────────
                    _cal_disp = _cal.copy()
                    _cal_disp["Model_Prob"]    = _cal_disp["Model_Prob"].map("{:.1%}".format)
                    _cal_disp["Actual_Cover"]  = _cal_disp["Actual_Cover"].map("{:.1%}".format)
                    _cal_disp = _cal_disp.rename(columns={
                        "bucket":"Prob Bucket","Games":"Games",
                        "Model_Prob":"Avg Model Prob","Actual_Cover":"Actual Cover Rate"
                    })
                    st.dataframe(_cal_disp, hide_index=True, use_container_width=True)

                    # ── MC vs XGB comparison ───────────────────────────────
                    if "mc_rl" in _src.columns and "xgb_rl" in _src.columns:
                        _comp2 = _cal_df.copy()
                        _mc_cal = _comp2.groupby("bucket", observed=True).agg(
                            MC_Prob=("mc_rl","mean"),
                            XGB_Prob=("xgb_rl","mean"),
                            Actual=("home_covers_rl","mean"),
                        ).reset_index()
                        st.divider()
                        st.markdown("**MC vs XGBoost — Individual Model Calibration**")
                        _fig_comp = go.Figure()
                        _fig_comp.add_trace(go.Scatter(
                            x=_mc_cal["bucket"].astype(str),
                            y=_mc_cal["Actual"],
                            mode="lines+markers",
                            line=dict(color=_GRAY, dash="dash"),
                            name="Actual Cover Rate",
                            marker=dict(color=_GRAY, size=6),
                        ))
                        _fig_comp.add_trace(go.Scatter(
                            x=_mc_cal["bucket"].astype(str),
                            y=_mc_cal["MC_Prob"],
                            mode="lines+markers",
                            line=dict(color=_BLUE, width=2),
                            name="Monte Carlo",
                            marker=dict(color=_BLUE, size=8),
                        ))
                        _fig_comp.add_trace(go.Scatter(
                            x=_mc_cal["bucket"].astype(str),
                            y=_mc_cal["XGB_Prob"],
                            mode="lines+markers",
                            line=dict(color=_GREEN, width=2),
                            name="XGBoost",
                            marker=dict(color=_GREEN, size=8),
                        ))
                        _pl_layout(_fig_comp, title="MC vs XGB Predicted Probability by Bucket", height=300)
                        _fig_comp.update_yaxes(title="Probability", tickformat=".0%")
                        _fig_comp.update_layout(showlegend=True, legend=dict(font=dict(color="#e5e7eb")))
                        st.plotly_chart(_fig_comp, use_container_width=True)
                else:
                    st.info(f"Need at least 20 games for calibration analysis (have {len(_cal_df)}).")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab_raw:
    import datetime as _rawdt

    if st.button("🔄 Refresh Data", key="raw_refresh"):
        st.cache_data.clear()
        st.rerun()

    _today    = _rawdt.date.today()
    _yd       = (_today - _rawdt.timedelta(days=1)).isoformat()
    _tod      = _today.isoformat()
    _tmrw     = (_today + _rawdt.timedelta(days=1)).isoformat()

    def _load_raw_for_date(date_str):
        """Load card predictions, supplemented by lineup file for any missing games."""
        rows = load_card(date_str)
        card_df = pd.DataFrame(rows) if rows else pd.DataFrame()
        if not card_df.empty:
            card_df["game_date"] = date_str

        # Also pull lineup file so we catch games the card skipped (missing starter)
        _lp = Path(__file__).parent / "data" / "statcast" / f"lineups_{date_str}.parquet"
        if not _lp.exists():
            _lp = Path(__file__).parent / "data" / "statcast" / "lineups_today.parquet"
        if _lp.exists():
            try:
                ldf = pd.read_parquet(_lp)
                ldf["game_date_col"] = pd.to_datetime(ldf["game_date"]).dt.date.astype(str)
                ldf = ldf[ldf["game_date_col"] == date_str]
                # Find games not already in the card
                card_games = set(card_df["game"].tolist()) if not card_df.empty and "game" in card_df.columns else set()
                extra_rows = []
                for _, lr in ldf.iterrows():
                    h = str(lr.get("home_team",""))
                    a = str(lr.get("away_team",""))
                    game_str = f"{a} @ {h}"
                    if game_str not in card_games:
                        extra_rows.append({
                            "game": game_str,
                            "game_date": date_str,
                            "home_team": h,
                            "away_team": a,
                            "home_sp": str(lr.get("home_starter_name","") or "TBD"),
                            "away_sp": str(lr.get("away_starter_name","") or "TBD"),
                            "game_time_et": str(lr.get("game_time_et","") or ""),
                            "lineup_confirmed": bool(lr.get("home_lineup_confirmed", False) and lr.get("away_lineup_confirmed", False)),
                        })
                if extra_rows:
                    extra_df = pd.DataFrame(extra_rows)
                    card_df = pd.concat([card_df, extra_df], ignore_index=True) if not card_df.empty else extra_df
            except Exception:
                pass

        return card_df

    _raw_frames = []
    for _d in [_yd, _tod, _tmrw]:
        _df = _load_raw_for_date(_d)
        if not _df.empty:
            _raw_frames.append(_df)

    if not _raw_frames:
        st.info("No prediction data found for yesterday, today, or tomorrow.")
    else:
        _raw = pd.concat(_raw_frames, ignore_index=True)

        # ── Determine who the model picks to win ─────────────────────────────
        def _model_pick(row):
            """Return the team the model favours and the confidence label."""
            home = str(row.get("home_team", ""))
            away = str(row.get("away_team", ""))
            bl   = row.get("blended_rl")
            mw   = row.get("mc_home_win")
            tier = str(row.get("best_tier") or "")
            # No prediction available (starter missing)
            if pd.isna(bl) and pd.isna(mw):
                return "—", ""
            bl = float(bl) if pd.notna(bl) else 0.5
            mw = float(mw) if pd.notna(mw) else 0.5
            # Use ML win prob as primary win indicator; RL blend for confidence
            if mw >= 0.55:
                pick = home
            elif mw <= 0.45:
                pick = away
            else:
                pick = away if bl < 0.46 else (home if bl > 0.54 else "—")
            conf = "★★" if tier == "**" else ("★" if tier == "*" else "")
            return pick, conf

        _raw[["Model Pick", "Conf"]] = _raw.apply(
            lambda r: pd.Series(_model_pick(r)), axis=1)

        # ── Build display table ───────────────────────────────────────────────
        def _fmt_odds(v):
            try:
                v = int(float(v))
                return f"+{v}" if v > 0 else str(v)
            except Exception:
                return "—"

        def _fmt_pct(v):
            try:
                return f"{float(v):.1%}"
            except Exception:
                return "—"

        def _fmt_f(v, decimals=2):
            try:
                return f"{float(v):.{decimals}f}"
            except Exception:
                return "—"

        _display_rows = []
        for _, r in _raw.sort_values(["game_date", "game_time_et"] if "game_time_et" in _raw.columns else ["game_date"]).iterrows():
            home = str(r.get("home_team", ""))
            away = str(r.get("away_team", ""))
            _display_rows.append({
                "Date":          r.get("game_date", ""),
                "Time (ET)":     r.get("game_time_et", "—") or "—",
                "Matchup":       f"{away} @ {home}",
                "Home SP":       str(r.get("home_sp", "—") or "—").title(),
                "Away SP":       str(r.get("away_sp", "—") or "—").title(),
                "Model Pick":    r.get("Model Pick", "—"),
                "Conf":          r.get("Conf", ""),
                "ML Win% (home)": _fmt_pct(r.get("mc_home_win")),
                "RL Blend":      _fmt_pct(r.get("blended_rl")),
                "Proj Total":    _fmt_f(r.get("blended_total") or r.get("mc_total"), 1),
                "Vegas Total":   r.get("vegas_total", "—") or "—",
                "ML Home":       _fmt_odds(r.get("vegas_ml_home")),
                "ML Away":       _fmt_odds(r.get("vegas_ml_away")),
                "Best Bet":      str(r.get("best_line", "") or ""),
                "Best Odds":     _fmt_odds(r.get("best_market_odds")),
                "Edge":          f"{float(r['best_edge'])*100:+.1f}%" if pd.notna(r.get("best_edge")) else "—",
                "Lineup":        "✅" if r.get("lineup_confirmed") else "⏳",
            })

        _disp_df = pd.DataFrame(_display_rows)

        # Summary counts
        st.caption("All games shown — including no-signal games. "
                   "Games with no Model Pick had missing starter data at prediction time.")
        _c1, _c2, _c3 = st.columns(3)
        _c1.metric("Yesterday", len(_raw[_raw["game_date"] == _yd]))
        _c2.metric("Today",     len(_raw[_raw["game_date"] == _tod]))
        _c3.metric("Tomorrow",  len(_raw[_raw["game_date"] == _tmrw]))

        st.divider()
        st.dataframe(
            _disp_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Date":           st.column_config.TextColumn("Date", width="small"),
                "Time (ET)":      st.column_config.TextColumn("Time", width="small"),
                "Matchup":        st.column_config.TextColumn("Matchup", width="medium"),
                "Home SP":        st.column_config.TextColumn("Home SP", width="medium"),
                "Away SP":        st.column_config.TextColumn("Away SP", width="medium"),
                "Model Pick":     st.column_config.TextColumn("Model Pick", width="small"),
                "Conf":           st.column_config.TextColumn("★", width="small"),
                "ML Win% (home)": st.column_config.TextColumn("Win% Home", width="small"),
                "RL Blend":       st.column_config.TextColumn("RL Blend", width="small"),
                "Proj Total":     st.column_config.TextColumn("Proj Total", width="small"),
                "Vegas Total":    st.column_config.TextColumn("O/U", width="small"),
                "ML Home":        st.column_config.TextColumn("ML Home", width="small"),
                "ML Away":        st.column_config.TextColumn("ML Away", width="small"),
                "Best Bet":       st.column_config.TextColumn("Best Bet", width="medium"),
                "Best Odds":      st.column_config.TextColumn("Odds", width="small"),
                "Edge":           st.column_config.TextColumn("Edge", width="small"),
                "Lineup":         st.column_config.TextColumn("Lineup", width="small"),
            },
        )

        # CSV download
        st.download_button(
            "⬇️ Download CSV",
            data=_disp_df.to_csv(index=False).encode("utf-8"),
            file_name=f"wizard_raw_{_tod}.csv",
            mime="text/csv",
        )


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Breakeven at -110 juice: 52.4%  ·  "
    "Model: 60% Monte Carlo + 40% XGBoost  ·  "
    "Base home -1.5 cover rate ~35.7%  ·  "
    "Score range = 25th–75th percentile of 50,000 simulations"
)
