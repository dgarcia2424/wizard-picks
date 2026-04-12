"""
app.py — The Wizard MLB Picks Dashboard
Action-first design: BET -> Game -> Why -> Stats
"""
import datetime
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Load .env so local runs work without Streamlit secrets
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

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.bet-card-strong {
    background: linear-gradient(135deg, #0f2f1a 0%, #1a4731 100%);
    border: 2px solid #22c55e;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.bet-card-lean {
    background: linear-gradient(135deg, #1c1f0a 0%, #2d3310 100%);
    border: 2px solid #eab308;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.bet-card-skip {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    opacity: 0.7;
}
.bet-action {
    font-size: 1.5rem;
    font-weight: 800;
    color: #f9fafb;
    letter-spacing: 0.03em;
}
.bet-action-lean {
    font-size: 1.3rem;
    font-weight: 700;
    color: #f9fafb;
}
.bet-tag-strong {
    display: inline-block;
    background: #16a34a;
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 99px;
    margin-left: 10px;
    vertical-align: middle;
    letter-spacing: 0.08em;
}
.bet-tag-lean {
    display: inline-block;
    background: #ca8a04;
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 99px;
    margin-left: 10px;
    vertical-align: middle;
    letter-spacing: 0.08em;
}
.bet-meta {
    color: #9ca3af;
    font-size: 0.9rem;
    margin-top: 4px;
}
.bet-why {
    color: #d1fae5;
    font-size: 0.95rem;
    margin-top: 10px;
    font-style: italic;
}
.bet-also {
    color: #fef08a;
    font-size: 0.9rem;
    margin-top: 6px;
    font-weight: 600;
}
.bet-stats {
    color: #6b7280;
    font-size: 0.82rem;
    margin-top: 10px;
    border-top: 1px solid #374151;
    padding-top: 8px;
}
.confidence-bar-wrap {
    background: #374151;
    border-radius: 99px;
    height: 6px;
    width: 200px;
    display: inline-block;
    vertical-align: middle;
    margin: 0 8px;
}
.skip-game {
    color: #6b7280;
    font-size: 0.88rem;
}
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #9ca3af;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 24px 0 12px 0;
}
.record-win  { color: #4ade80; font-weight: 700; }
.record-loss { color: #f87171; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data" / "statcast"

@st.cache_data(ttl=300)
def load_card(date_str: str) -> list[dict]:
    """Load today's predictions by running run_today.py logic directly."""
    try:
        from run_today import run_card
        return run_card(date_str)
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return []

@st.cache_data(ttl=60)
def load_tracker() -> pd.DataFrame:
    path = Path(__file__).parent / "data" / "raw" / "bet_tracker.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df.sort_values("date", ascending=False) if "date" in df.columns else df

@st.cache_data(ttl=300)
def load_backtest() -> pd.DataFrame:
    path = Path(__file__).parent / "backtest_2026_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def edge_pct(blended_rl: float, signal: str) -> float:
    breakeven = 100 / 210
    if "HOME" in signal:
        return (blended_rl - breakeven) * 100
    return ((1 - blended_rl) - breakeven) * 100

def conf_pct(blended_rl: float, signal: str) -> int:
    if "AWAY" in signal:
        return int((1 - blended_rl) * 100)
    return int(blended_rl * 100)

def why_text(r: dict) -> str:
    parts = []
    home_x = r.get("home_sp_xwoba") or 0
    away_x = r.get("away_sp_xwoba") or 0
    signal = r.get("rl_signal", "")

    if "AWAY" in signal:
        sp_name, sp_x = r["away_sp"].title(), away_x
        opp_name, opp_x = r["home_sp"].title(), home_x
    else:
        sp_name, sp_x = r["home_sp"].title(), home_x
        opp_name, opp_x = r["away_sp"].title(), away_x

    if sp_x < 0.280:
        parts.append(f"{sp_name} is elite (xwOBA {sp_x:.3f})")
    elif sp_x < 0.305:
        parts.append(f"{sp_name} is strong (xwOBA {sp_x:.3f})")

    if opp_x > 0.360:
        parts.append(f"{opp_name} is struggling (xwOBA {opp_x:.3f})")
    elif opp_x > 0.330:
        parts.append(f"{opp_name} is below average (xwOBA {opp_x:.3f})")

    if "AWAY" in signal and r.get("home_sp_flag") == "VOLATILE":
        parts.append(f"{r['home_sp'].title()} showing velocity decline")
    if "HOME" in signal and r.get("away_sp_flag") == "VOLATILE":
        parts.append(f"{r['away_sp'].title()} showing velocity decline")
    if "AWAY" in signal and r.get("away_sp_flag") == "GAINER":
        parts.append(f"{r['away_sp'].title()} velocity trending up")

    temp = r.get("temp_f", 72)
    if temp and temp > 82:
        parts.append(f"hot weather ({temp:.0f}°F boosts scoring)")
    elif temp and temp < 48:
        parts.append(f"cold weather ({temp:.0f}°F suppresses scoring)")

    return " · ".join(parts) if parts else "Model edge on run differential"

def render_game_card(r: dict, n: int, units: int, tier: str) -> None:
    signal = r.get("rl_signal", "")
    away, home = r["away_team"], r["home_team"]
    conf_str = "" if r.get("lineup_confirmed") else " <span style='color:#6b7280;font-size:0.8rem'>(projected lineup)</span>"

    bet_line = f"{away} +1.5" if "AWAY" in signal else f"{home} -1.5"
    c = conf_pct(r["blended_rl"], signal)
    e = edge_pct(r["blended_rl"], signal)
    has_vegas = r.get("vegas_ml_home") is not None and not pd.isna(r.get("vegas_ml_home", float("nan")))
    ou_str = f"O/U {r['vegas_total']}" if has_vegas and r.get("vegas_total") else ""

    if tier == "strong":
        card_class = "bet-card-strong"
        action_class = "bet-action"
        tag = '<span class="bet-tag-strong">STRONG · 2 units</span>'
    else:
        card_class = "bet-card-lean"
        action_class = "bet-action-lean"
        tag = '<span class="bet-tag-lean">LEAN · 1 unit</span>'

    total_html = ""
    if r.get("total_signal"):
        total_html = f'<div class="bet-also">Also play: {r["total_signal"]}</div>'

    home_flag = f" <b>[{r['home_sp_flag']}]</b>" if r.get("home_sp_flag") not in ("NORMAL","UNKNOWN","","None",None) else ""
    away_flag = f" <b>[{r['away_sp_flag']}]</b>" if r.get("away_sp_flag") not in ("NORMAL","UNKNOWN","","None",None) else ""

    xgb_str = f"{r['xgb_rl']:.0%}" if r.get("xgb_rl") else "N/A"

    st.markdown(f"""
<div class="{card_class}">
  <div class="{action_class}">[{n}] {bet_line} {tag}</div>
  <div class="bet-meta">{away} @ {home} &nbsp;·&nbsp; {ou_str} &nbsp;·&nbsp; {r['temp_f']:.0f}°F{conf_str}</div>
  <div class="bet-why">Why: {why_text(r)}</div>
  {total_html}
  <div class="bet-stats">
    Confidence: {c}% &nbsp;|&nbsp; Edge vs breakeven: {e:+.1f}% &nbsp;|&nbsp; MC: {r['mc_rl']:.0%} &nbsp;·&nbsp; XGB: {xgb_str} &nbsp;·&nbsp; Blend: {r['blended_rl']:.0%}<br>
    {home} SP: {r['home_sp'].title()}{home_flag} &nbsp;xwOBA {r['home_sp_xwoba']:.3f} &nbsp;&nbsp;|&nbsp;&nbsp;
    {away} SP: {r['away_sp'].title()}{away_flag} &nbsp;xwOBA {r['away_sp_xwoba']:.3f}
  </div>
</div>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🧙 The Wizard — MLB Picks")
st.caption("Daily card auto-refreshes after 10 AM ET · Sorted strongest edge first")

col_date, col_refresh = st.columns([3, 1])
with col_date:
    today = datetime.date.today()
    selected = st.date_input("Date", value=today,
                              min_value=datetime.date(2026, 3, 28), max_value=today)
with col_refresh:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Refresh"):
        st.cache_data.clear()

date_str = selected.isoformat()

# ── Load card ─────────────────────────────────────────────────────────────────
with st.spinner("Running model..."):
    results = load_card(date_str)

if not results:
    st.warning(f"No predictions for {selected.strftime('%B %d, %Y')} — pipeline may not have run yet.")
    st.stop()

# Sort strongest first
def sort_key(r):
    sig = r["rl_signal"]
    tier = 0 if "**" in sig else (1 if "*" in sig else 2)
    return (tier, -abs(r["blended_rl"] - 0.5))

results = sorted(results, key=sort_key)
strong = [r for r in results if "**" in r["rl_signal"]]
lean   = [r for r in results if "*"  in r["rl_signal"] and "**" not in r["rl_signal"]]
skip   = [r for r in results if not r["rl_signal"]]

# ── Summary bar ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Games Today", len(results))
c2.metric("Strong Plays", len(strong), delta="2 units each")
c3.metric("Lean Plays", len(lean), delta="1 unit each")
c4.metric("Skip", len(skip))

st.divider()

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_picks, tab_season, tab_tracker = st.tabs(["Today's Picks", "Season Tracker", "Bet Log"])

with tab_picks:
    play_num = 1

    if strong:
        st.markdown('<div class="section-header">Strong Plays</div>', unsafe_allow_html=True)
        for r in strong:
            render_game_card(r, play_num, 2, "strong")
            play_num += 1

    if lean:
        st.markdown('<div class="section-header">Lean Plays</div>', unsafe_allow_html=True)
        for r in lean:
            render_game_card(r, play_num, 1, "lean")
            play_num += 1

    if skip:
        with st.expander(f"No edge — {len(skip)} games skipped"):
            for r in skip:
                away, home = r["away_team"], r["home_team"]
                has_v = r.get("vegas_total") and not pd.isna(r.get("vegas_total", float("nan")))
                ou = f" · O/U {r['vegas_total']}" if has_v else ""
                total_note = f" · model {r['blended_total'] or r['mc_total']:.1f}" if r.get('total_signal') else ""
                total_sig = f" → **{r['total_signal']}**" if r.get("total_signal") else ""
                st.markdown(
                    f'<div class="skip-game">{away} @ {home}{ou}{total_note}{total_sig} '
                    f'&nbsp;·&nbsp; blend={r["blended_rl"]:.0%}</div>',
                    unsafe_allow_html=True
                )

with tab_season:
    bt = load_backtest()
    if bt.empty:
        st.info("Season tracker will populate as games complete daily.")
    else:
        bt_sig = bt[bt["signal"] != ""].copy()
        if not bt_sig.empty:
            wins = int(bt_sig["bet_win"].sum())
            n    = len(bt_sig)
            losses = n - wins
            roi  = (wins * (100/110) - losses) / n * 100

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Record",   f"{wins}-{losses}")
            m2.metric("Win Rate", f"{wins/n:.1%}")
            m3.metric("ROI",      f"{roi:+.1f}%")
            m4.metric("Games",    f"{len(bt)} tracked")

            st.divider()

            # Weekly breakdown
            bt["week"] = pd.to_datetime(bt["date"]).dt.to_period("W").astype(str)
            weekly = []
            for week, wdf in bt.groupby("week"):
                s = wdf[wdf["signal"] != ""]
                if len(s) == 0:
                    continue
                w = int(s["bet_win"].sum())
                l = len(s) - w
                r_roi = (w * (100/110) - l) / len(s) * 100
                weekly.append({"Week": week, "Bets": len(s), "Wins": w,
                                "Win%": f"{w/len(s):.1%}", "ROI": f"{r_roi:+.1f}%"})
            if weekly:
                st.subheader("Weekly Breakdown")
                st.dataframe(pd.DataFrame(weekly), hide_index=True, use_container_width=True)

            # By signal type
            st.subheader("By Signal Type")
            sig_rows = []
            for sig in ["AWAY +1.5 **", "AWAY +1.5 *", "HOME -1.5 **", "HOME -1.5 *"]:
                s = bt[bt["signal"] == sig]
                if len(s) == 0:
                    continue
                w = int(s["bet_win"].sum())
                l = len(s) - w
                r_roi = (w * (100/110) - l) / len(s) * 100
                sig_rows.append({"Signal": sig, "Bets": len(s), "Wins": w,
                                  "Win%": f"{w/len(s):.1%}", "ROI": f"{r_roi:+.1f}%"})
            if sig_rows:
                st.dataframe(pd.DataFrame(sig_rows), hide_index=True, use_container_width=True)

with tab_tracker:
    tracker = load_tracker()
    if tracker.empty:
        st.info("No bets logged yet. Use the bet tracker to log plays.")
    else:
        if {"result", "profit_loss"}.issubset(tracker.columns):
            wins   = (tracker["result"] == "WIN").sum()
            losses = (tracker["result"] == "LOSS").sum()
            pushes = (tracker["result"] == "PUSH").sum()
            total_pl = pd.to_numeric(tracker["profit_loss"], errors="coerce").sum()
            total_bets = wins + losses + pushes
            win_pct = (wins / total_bets * 100) if total_bets > 0 else 0

            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Record", f"{wins}-{losses}-{pushes}")
            t2.metric("Win %", f"{win_pct:.1f}%")
            t3.metric("P&L (units)", f"{total_pl:+.2f}")
            t4.metric("Total Bets", total_bets)
            st.divider()

        def style_tracker(row):
            styles = [""] * len(row)
            if "result" in row.index:
                idx = row.index.get_loc("result")
                v = str(row["result"])
                if v == "WIN":
                    styles[idx] = "color: #4ade80; font-weight: bold"
                elif v == "LOSS":
                    styles[idx] = "color: #f87171; font-weight: bold"
            return styles

        st.dataframe(tracker.style.apply(style_tracker, axis=1),
                     hide_index=True, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Breakeven at -110 juice: 52.4%  ·  Model: 60% Monte Carlo + 40% XGBoost  ·  Base home -1.5 cover rate ~35.7%")
