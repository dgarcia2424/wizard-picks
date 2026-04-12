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
/* Login page */
.login-wrap { max-width: 380px; margin: 80px auto; text-align: center; }
.login-title { font-size: 2.4rem; font-weight: 800; margin-bottom: 4px; }
.login-sub   { color: #6b7280; margin-bottom: 32px; }

/* Cards */
.bet-card-strong {
    background: linear-gradient(135deg, #0f2f1a 0%, #1a4731 100%);
    border: 2px solid #22c55e;
    border-radius: 12px; padding: 20px 24px; margin-bottom: 16px;
}
.bet-card-lean {
    background: linear-gradient(135deg, #1c1f0a 0%, #2d3310 100%);
    border: 2px solid #eab308;
    border-radius: 12px; padding: 20px 24px; margin-bottom: 16px;
}
.bet-action        { font-size: 1.5rem; font-weight: 800; color: #f9fafb; letter-spacing:.03em; }
.bet-action-lean   { font-size: 1.3rem; font-weight: 700; color: #f9fafb; }
.bet-tag-strong    { display:inline-block; background:#16a34a; color:white; font-size:.75rem;
                     font-weight:700; padding:2px 10px; border-radius:99px; margin-left:10px;
                     vertical-align:middle; letter-spacing:.08em; }
.bet-tag-lean      { display:inline-block; background:#ca8a04; color:white; font-size:.75rem;
                     font-weight:700; padding:2px 10px; border-radius:99px; margin-left:10px;
                     vertical-align:middle; letter-spacing:.08em; }
.bet-meta  { color:#9ca3af; font-size:.9rem; margin-top:4px; }
.bet-why   { color:#d1fae5; font-size:.95rem; margin-top:10px; font-style:italic; }
.bet-also  { color:#fef08a; font-size:.9rem; margin-top:6px; font-weight:600; }
.bet-stats { color:#6b7280; font-size:.82rem; margin-top:10px;
             border-top:1px solid #374151; padding-top:8px; }
.skip-game { color:#6b7280; font-size:.88rem; }
.section-header { font-size:1.1rem; font-weight:700; color:#9ca3af;
                  letter-spacing:.1em; text-transform:uppercase; margin:24px 0 12px 0; }
</style>
""", unsafe_allow_html=True)


# ── Password gate ─────────────────────────────────────────────────────────────
def _site_password() -> str:
    # Try Streamlit secrets first (hosted), then env var (local)
    try:
        if hasattr(st, "secrets") and "SITE_PASSWORD" in st.secrets:
            return str(st.secrets["SITE_PASSWORD"]).strip()
    except Exception:
        pass
    return str(os.environ.get("SITE_PASSWORD", "")).strip()

def _debug_secrets() -> str:
    """Temporary: show what secrets are visible (key names only, not values)."""
    try:
        keys = list(st.secrets.keys())
        return f"Secrets loaded: {keys}"
    except Exception as e:
        return f"Secrets error: {e}"

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
            if pwd.strip() == _site_password():
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
                st.caption(_debug_secrets())  # temp: remove once login works
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
    """Load picks — Supabase if hosted, local run_card() if local."""
    if USE_SUPABASE:
        client = _supabase()
        resp = client.table("wizard_daily_card").select("*").eq("game_date", date_str).execute()
        if resp.data:
            rows = []
            for row in resp.data:
                r = row.get("data", {}) or {}
                # Ensure top-level fields are present
                for col in ["game","home_team","away_team","rl_signal","total_signal",
                            "blended_rl","mc_rl","xgb_rl","mc_total","blended_total",
                            "vegas_total","vegas_ml_home","home_sp","away_sp",
                            "home_sp_xwoba","away_sp_xwoba","home_sp_flag","away_sp_flag",
                            "temp_f","lineup_confirmed"]:
                    if col not in r:
                        r[col] = row.get(col)
                rows.append(r)
            return rows
    # Local fallback
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
        resp = client.table("wizard_backtest").select("*").execute()
        return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()
    path = Path(__file__).parent / "backtest_2026_results.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=60)
def load_tracker() -> pd.DataFrame:
    if USE_SUPABASE:
        client = _supabase()
        resp = client.table("bet_tracker").select("*").order("date", desc=True).execute()
        return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()
    path = Path(__file__).parent / "data" / "raw" / "bet_tracker.csv"
    return pd.read_csv(path).sort_values("date", ascending=False) if path.exists() else pd.DataFrame()


# ── Card helpers ──────────────────────────────────────────────────────────────

def _edge_pct(blended_rl: float, signal: str) -> float:
    breakeven = 100 / 210
    return ((1 - blended_rl if "AWAY" in signal else blended_rl) - breakeven) * 100

def _conf_pct(blended_rl: float, signal: str) -> int:
    return int((1 - blended_rl if "AWAY" in signal else blended_rl) * 100)

def _why(r: dict) -> str:
    parts = []
    home_x = float(r.get("home_sp_xwoba") or 0)
    away_x = float(r.get("away_sp_xwoba") or 0)
    signal = r.get("rl_signal", "")
    sp_name, sp_x    = (r.get("away_sp","").title(), away_x) if "AWAY" in signal else (r.get("home_sp","").title(), home_x)
    opp_name, opp_x  = (r.get("home_sp","").title(), home_x) if "AWAY" in signal else (r.get("away_sp","").title(), away_x)

    if sp_x and sp_x < 0.280:
        parts.append(f"{sp_name} is elite (xwOBA {sp_x:.3f})")
    elif sp_x and sp_x < 0.305:
        parts.append(f"{sp_name} is strong (xwOBA {sp_x:.3f})")
    if opp_x and opp_x > 0.360:
        parts.append(f"{opp_name} is struggling (xwOBA {opp_x:.3f})")
    elif opp_x and opp_x > 0.330:
        parts.append(f"{opp_name} is below average (xwOBA {opp_x:.3f})")
    if "AWAY" in signal and r.get("home_sp_flag") == "VOLATILE":
        parts.append(f"{r.get('home_sp','').title()} velocity declining")
    if "HOME" in signal and r.get("away_sp_flag") == "VOLATILE":
        parts.append(f"{r.get('away_sp','').title()} velocity declining")
    if "AWAY" in signal and r.get("away_sp_flag") == "GAINER":
        parts.append(f"{r.get('away_sp','').title()} velo trending up")
    temp = r.get("temp_f") or 72
    if float(temp) > 82:
        parts.append(f"hot weather ({float(temp):.0f}°F)")
    elif float(temp) < 48:
        parts.append(f"cold weather ({float(temp):.0f}°F, suppresses scoring)")
    return " · ".join(parts) if parts else "Model edge on run differential"


def render_card(r: dict, n: int, units: int, tier: str):
    signal   = r.get("rl_signal", "")
    away, home = r.get("away_team",""), r.get("home_team","")
    bet_line = f"{away} +1.5" if "AWAY" in signal else f"{home} -1.5"
    c        = _conf_pct(float(r.get("blended_rl", 0.5)), signal)
    e        = _edge_pct(float(r.get("blended_rl", 0.5)), signal)
    has_v    = r.get("vegas_ml_home") is not None
    ou_str   = f"O/U {r['vegas_total']}" if has_v and r.get("vegas_total") else ""
    conf_str = "" if r.get("lineup_confirmed") else \
               " <span style='color:#6b7280;font-size:0.8rem'>(projected)</span>"

    card_cls  = "bet-card-strong" if tier == "strong" else "bet-card-lean"
    act_cls   = "bet-action"      if tier == "strong" else "bet-action-lean"
    tag       = ('<span class="bet-tag-strong">STRONG · 2 units</span>' if tier == "strong"
                 else '<span class="bet-tag-lean">LEAN · 1 unit</span>')

    total_html = (f'<div class="bet-also">Also play: {r["total_signal"]}</div>'
                  if r.get("total_signal") else "")

    hf = f" <b>[{r['home_sp_flag']}]</b>" if r.get("home_sp_flag") not in ("NORMAL","UNKNOWN","","None",None) else ""
    af = f" <b>[{r['away_sp_flag']}]</b>" if r.get("away_sp_flag") not in ("NORMAL","UNKNOWN","","None",None) else ""
    xgb_str = f"{float(r['xgb_rl']):.0%}" if r.get("xgb_rl") else "N/A"
    bl = float(r.get("blended_rl", 0.5))
    mc = float(r.get("mc_rl", 0.5))

    st.markdown(f"""
<div class="{card_cls}">
  <div class="{act_cls}">[{n}] {bet_line} {tag}</div>
  <div class="bet-meta">{away} @ {home} &nbsp;·&nbsp; {ou_str} &nbsp;·&nbsp; {float(r.get('temp_f',72)):.0f}°F{conf_str}</div>
  <div class="bet-why">Why: {_why(r)}</div>
  {total_html}
  <div class="bet-stats">
    Confidence: {c}% &nbsp;|&nbsp; Edge vs breakeven: {e:+.1f}%<br>
    {home} SP: {r.get('home_sp','').title()}{hf} &nbsp;xwOBA {float(r.get('home_sp_xwoba') or 0):.3f}
    &nbsp;&nbsp;|&nbsp;&nbsp;
    {away} SP: {r.get('away_sp','').title()}{af} &nbsp;xwOBA {float(r.get('away_sp_xwoba') or 0):.3f}<br>
    <span style="color:#4b5563">MC: {mc:.0%} &nbsp;·&nbsp; XGB: {xgb_str} &nbsp;·&nbsp; Blend: {bl:.0%}</span>
  </div>
</div>""", unsafe_allow_html=True)


# ── App header ────────────────────────────────────────────────────────────────
st.markdown("## 🧙 The Wizard — MLB Picks")
st.caption("Daily card refreshes after 10 AM ET · Sorted by strongest edge first")

col_date, col_refresh = st.columns([3, 1])
with col_date:
    today    = datetime.date.today()
    selected = st.date_input("Date", value=today,
                             min_value=datetime.date(2026, 3, 28), max_value=today)
with col_refresh:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Refresh"):
        st.cache_data.clear()

date_str = selected.isoformat()

# ── Load & sort ───────────────────────────────────────────────────────────────
with st.spinner("Loading picks..."):
    results = load_card(date_str)

if not results:
    st.warning(f"No predictions for {selected.strftime('%B %d, %Y')} — pipeline may not have run yet.")
    st.stop()

def _sort(r):
    sig  = r.get("rl_signal", "")
    tier = 0 if "**" in sig else (1 if "*" in sig else 2)
    return (tier, -abs(float(r.get("blended_rl", 0.5)) - 0.5))

results = sorted(results, key=_sort)
strong  = [r for r in results if "**" in r.get("rl_signal","")]
lean    = [r for r in results if "*"  in r.get("rl_signal","") and "**" not in r.get("rl_signal","")]
skip    = [r for r in results if not r.get("rl_signal","")]

# ── Summary bar ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Games Today",  len(results))
c2.metric("Strong Plays", len(strong), delta="2 units each")
c3.metric("Lean Plays",   len(lean),   delta="1 unit each")
c4.metric("Skip",         len(skip))
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_picks, tab_season, tab_log = st.tabs(["Today's Picks", "Season Tracker", "Bet Log"])

with tab_picks:
    n = 1
    if strong:
        st.markdown('<div class="section-header">Strong Plays</div>', unsafe_allow_html=True)
        for r in strong:
            render_card(r, n, 2, "strong")
            n += 1
    if lean:
        st.markdown('<div class="section-header">Lean Plays</div>', unsafe_allow_html=True)
        for r in lean:
            render_card(r, n, 1, "lean")
            n += 1
    if skip:
        with st.expander(f"No edge — {len(skip)} games skipped"):
            for r in skip:
                away, home = r.get("away_team",""), r.get("home_team","")
                ou   = f" · O/U {r['vegas_total']}" if r.get("vegas_total") else ""
                tsig = f" → **{r['total_signal']}**" if r.get("total_signal") else ""
                bl   = float(r.get("blended_rl", 0.5))
                st.markdown(
                    f'<div class="skip-game">{away} @ {home}{ou}{tsig} &nbsp;·&nbsp; blend={bl:.0%}</div>',
                    unsafe_allow_html=True)

with tab_season:
    bt = load_backtest()
    if bt.empty:
        st.info("Season tracker populates as games complete each day.")
    else:
        sig = bt[bt.get("signal", pd.Series(dtype=str)) != ""] if "signal" in bt.columns else pd.DataFrame()
        if not sig.empty:
            wins   = int(sig["bet_win"].sum())
            n_bets = len(sig)
            losses = n_bets - wins
            roi    = (wins * (100/110) - losses) / n_bets * 100
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Record",   f"{wins}-{losses}")
            m2.metric("Win Rate", f"{wins/n_bets:.1%}")
            m3.metric("ROI",      f"{roi:+.1f}%")
            m4.metric("Games",    f"{len(bt)} tracked")
            st.divider()

        if "date" in bt.columns:
            bt["week"] = pd.to_datetime(bt["date"]).dt.to_period("W").astype(str)
            rows = []
            for week, wdf in bt.groupby("week"):
                s = wdf[wdf.get("signal", pd.Series(dtype=str)) != ""] if "signal" in wdf.columns else pd.DataFrame()
                if s.empty:
                    continue
                w = int(s["bet_win"].sum())
                l = len(s) - w
                r_roi = (w * (100/110) - l) / len(s) * 100
                rows.append({"Week": week, "Bets": len(s), "Wins": w,
                             "Win%": f"{w/len(s):.1%}", "ROI": f"{r_roi:+.1f}%"})
            if rows:
                st.subheader("Weekly Breakdown")
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        if "signal" in bt.columns:
            sig_rows = []
            for s in ["AWAY +1.5 **","AWAY +1.5 *","HOME -1.5 **","HOME -1.5 *"]:
                sub = bt[bt["signal"] == s]
                if sub.empty:
                    continue
                w = int(sub["bet_win"].sum())
                l = len(sub) - w
                r_roi = (w * (100/110) - l) / len(sub) * 100
                sig_rows.append({"Signal": s, "Bets": len(sub), "Wins": w,
                                  "Win%": f"{w/len(sub):.1%}", "ROI": f"{r_roi:+.1f}%"})
            if sig_rows:
                st.subheader("By Signal Type")
                st.dataframe(pd.DataFrame(sig_rows), hide_index=True, use_container_width=True)

with tab_log:
    tracker = load_tracker()
    if tracker.empty:
        st.info("No bets logged yet.")
    else:
        if {"result","profit_loss"}.issubset(tracker.columns):
            wins   = (tracker["result"] == "WIN").sum()
            losses = (tracker["result"] == "LOSS").sum()
            pushes = (tracker["result"] == "PUSH").sum()
            total_pl = pd.to_numeric(tracker["profit_loss"], errors="coerce").sum()
            n_bets   = wins + losses + pushes
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Record",    f"{wins}-{losses}-{pushes}")
            t2.metric("Win %",     f"{wins/n_bets*100:.1f}%" if n_bets else "—")
            t3.metric("P&L (units)", f"{total_pl:+.2f}")
            t4.metric("Total Bets", n_bets)
            st.divider()

        def _style(row):
            styles = [""] * len(row)
            if "result" in row.index:
                idx = row.index.get_loc("result")
                v = str(row["result"])
                styles[idx] = ("color:#4ade80;font-weight:bold" if v == "WIN"
                                else "color:#f87171;font-weight:bold" if v == "LOSS" else "")
            return styles
        st.dataframe(tracker.style.apply(_style, axis=1),
                     hide_index=True, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Breakeven at -110 juice: 52.4%  ·  Model: 60% Monte Carlo + 40% XGBoost  ·  Base home -1.5 cover rate ~35.7%")
