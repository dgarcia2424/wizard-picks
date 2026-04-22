"""
orchestrator/daily_pipeline.py

Daily pipeline: Agent 1 → Agent 3 → Agent 4 → Agent 5
Runs at 10 AM after DK posts lines.

Agent 2 (Static Data Manager) runs independently on Sundays via main_weekly.py.
Agent 6 (Bet Tracker) runs as a separate server via tracker_server.py.
Agent 7 (Maintenance) runs on-demand via main_weekly.py or standalone.

CHECKPOINT SYSTEM:
  Each agent writes a checkpoint to checkpoints/YYYY-MM-DD_agentN.json on success.
  On re-run, completed agents are skipped automatically — saving tokens and avoiding
  rate limit cascades when re-running a partially failed pipeline.
  To force a full re-run: delete today's checkpoint files or pass force=True.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

# render_report.py lives at the project root, one level above wizard_agents/.
# Put that root on sys.path BEFORE the top-level `from render_report import render`
# so the deterministic renderer is a plain import — no importlib dance inside Step 4.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.definitions import AGENT3
from config import settings
from config.settings import FILES, PIPELINE_DIR
from orchestrator.agent_loop import run_agent, PipelineHaltError
from render_report import render
from tools.implementations import (
    archive_model_scores,
    auto_grade_historical_picks,
    fetch_odds_api,
    fetch_mlb_starters,
    fetch_weather,
    rebuild_live_predictions_log,
    write_csv,
)

# Agent 4 (Report Generation) has been replaced with a deterministic
# Python renderer — render_report.py in the pipeline root. Grading is
# refreshed via auto_grade_historical_picks() before each render so the
# rolling-accuracy matrix reflects the latest graded results.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("wizard_pipeline.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger("wizard.daily")

TODAY    = date.today().strftime("%B %d, %Y")
DATE_KEY = date.today().isoformat()   # e.g. "2026-04-11"

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _ckpt_path(agent_key: str) -> Path:
    return CHECKPOINT_DIR / f"{DATE_KEY}_{agent_key}.json"


def _load(agent_key: str) -> str | None:
    """Return saved output if today's checkpoint exists and succeeded, else None."""
    p = _ckpt_path(agent_key)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("date") == DATE_KEY and data.get("status") == "OK":
            logger.info(f"[Checkpoint] ✅ {agent_key} already completed today — skipping.")
            return data["output"]
    except Exception:
        pass
    return None


def _save(agent_key: str, output: str) -> None:
    """Persist agent output so it can be skipped on re-run."""
    _ckpt_path(agent_key).write_text(
        json.dumps({"date": DATE_KEY, "status": "OK", "output": output}, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"[Checkpoint] 💾 {agent_key} saved.")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_daily_pipeline(force: bool = False) -> dict[str, str]:
    """
    Execute the full daily pipeline sequentially.
    Agents that already completed today are skipped (checkpoint system).

    Args:
        force: If True, ignore checkpoints and re-run all agents.

    Returns:
        Dict of agent_name → output (or error message).
    Raises:
        PipelineHaltError if Agent 1 signals no lines available.
    """
    results: dict[str, str] = {}

    logger.info("=" * 60)
    logger.info(f"THE WIZARD REPORT — Daily Pipeline | {TODAY}")
    if force:
        logger.info("⚠️  force=True — ignoring all checkpoints")
    logger.info("=" * 60)

    # ── STEP 1: Data Ingestion (deterministic Python ETL) ─────────────────────
    # Agent 1's LLM-driven ingestion has been replaced by a direct Python
    # pipeline: fetch odds (HALT if zero games) → fetch starters → fetch
    # weather → write games.csv. No Claude call, no tool dispatch — just the
    # same underlying tool functions called in order. Downstream scorers read
    # starters/weather from their own parquet sources, so games.csv is the
    # odds API output as before.
    logger.info("▶ Starting Step 1: Data Ingestion (deterministic ETL)")
    agent1_output = None if force else _load("agent1")

    if agent1_output is None:
        try:
            # 1) Odds — HALT on zero games
            odds_raw = fetch_odds_api()
            odds     = json.loads(odds_raw)
            if odds.get("status") == "HALT":
                raise PipelineHaltError(
                    f"Odds API returned no games: {odds.get('error', 'unknown')}"
                )
            games = odds.get("games", []) or []
            if not games:
                raise PipelineHaltError("fetch_odds_api returned an empty games list.")
            logger.info(f"[Ingest] odds: {len(games)} games fetched "
                        f"(us_err={odds.get('us_error')}, eu_err={odds.get('eu_error')})")

            # 2) Starters
            starters_raw = fetch_mlb_starters()
            starters     = json.loads(starters_raw)
            logger.info(
                f"[Ingest] starters: confirmed={starters.get('confirmed_games', 0)}, "
                f"excluded={len(starters.get('excluded_games', []) or [])} "
                f"(status={starters.get('status')})"
            )

            # 3) Weather — pass minimal game shape the tool expects
            weather_in = json.dumps([
                {"home_team": g.get("home_team", ""),
                 "game_id":   g.get("game_id", g.get("home_team", ""))}
                for g in games
            ])
            weather_raw = fetch_weather(weather_in)
            weather     = json.loads(weather_raw).get("weather", {}) or {}

            # 4) Build human-readable "Enriched Weather" strings — logged for now.
            def _cardinal(deg):
                if deg is None:
                    return "—"
                try:
                    d = float(deg) % 360.0
                except Exception:
                    return "—"
                dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
                return dirs[int((d + 22.5) // 45) % 8]

            enriched_weather: list[str] = []
            for g in games:
                gid  = g.get("game_id", g.get("home_team", ""))
                home = g.get("home_team", "")
                w    = weather.get(gid, {}) or {}
                if "error" in w:
                    enriched_weather.append(f"{home}: weather unavailable ({w.get('error')})")
                    continue
                t   = w.get("temperature_f")
                ws  = w.get("wind_speed_mph")
                wd  = _cardinal(w.get("wind_direction"))
                pp  = w.get("precipitation_probability")
                pp_s = f", Precip {pp}%" if pp is not None else ""
                enriched_weather.append(
                    f"{home}: {t}°F, Wind {ws}mph {wd}{pp_s}"
                )
            for line in enriched_weather:
                logger.info(f"[Weather] {line}")

            # 5) Write games.csv — odds API output is the canonical shape.
            # TODO: join `enriched_weather` into a `weather_context` column on
            # games.csv once downstream consumers are ready to read it. For now
            # starter info lives in `lineups_{date}.parquet` and weather is
            # logged only — games.csv stays 1:1 with fetch_odds_api output.
            write_result = json.loads(write_csv(
                file_key="games",
                records_json=json.dumps(games),
                mode="w",
            ))
            if write_result.get("status") != "OK":
                raise RuntimeError(f"write_csv(games) failed: {write_result}")
            logger.info(f"[Ingest] games.csv written: {write_result.get('rows_written')} rows "
                        f"→ {write_result.get('file')}")

            agent1_output = json.dumps({
                "status":           "OK",
                "games_written":    write_result.get("rows_written"),
                "confirmed_starters": starters.get("confirmed_games", 0),
                "weather_reports":  sum(1 for w in weather.values() if "error" not in w),
                "us_error":         odds.get("us_error"),
                "eu_error":         odds.get("eu_error"),
                "enriched_weather": enriched_weather,
            }, ensure_ascii=False)
            _save("agent1", agent1_output)
            results["agent1"] = agent1_output
            logger.info(f"✅ Step 1 complete.\n{agent1_output[:500]}")

        except PipelineHaltError as e:
            logger.error(f"❌ PIPELINE HALTED by Step 1: {e}")
            results["agent1"] = f"HALT: {e}"
            results["pipeline_status"] = "HALTED"
            return results

        except Exception as e:
            logger.error(f"❌ Step 1 (Data Ingestion) error: {e}", exc_info=True)
            results["agent1"] = f"ERROR: {e}"
            results["pipeline_status"] = "FAILED"
            return results
    else:
        results["agent1"] = agent1_output

    # ── AGENT 3: Scoring Engine ───────────────────────────────────────────────
    agent3_output = None if force else _load("agent3")

    if agent3_output is None:
        # Step 1 is now deterministic Python (no LLM call) — no rate-limit wait needed.
        logger.info("▶ Starting Agent 3: Scoring Engine")
        try:
            agent3_output = run_agent(
                system_prompt=AGENT3.system_prompt,
                tools=AGENT3.tools,
                tool_executor=AGENT3.tool_executor,
                max_tokens=8096,
                user_message=(
                    f"Score today's slate ({TODAY}) by calling generate_ml_scores. "
                    "The tool runs the ML, Totals, Runline, and F5 stackers, applies the "
                    "Three-Part Lock, computes Kelly stakes, and writes model_scores.csv. "
                    "Emit the completion report from the returned JSON — do not re-score manually."
                ),
                context=f"Agent 1 (Data Ingestion) completed:\n{agent1_output}",
                agent_name=AGENT3.name,
            )
            _save("agent3", agent3_output)
            results["agent3"] = agent3_output
            logger.info(f"✅ Agent 3 complete.\n{agent3_output[:500]}")

            # Archive today's model_scores.csv so rolling cutoff stats can
            # grow forward without re-running the sterile backtest.
            archive_status = archive_model_scores(DATE_KEY)
            logger.info(f"[Archive] model_scores: {archive_status}")

        except Exception as e:
            logger.error(f"❌ Agent 3 error: {e}", exc_info=True)
            results["agent3"] = f"ERROR: {e}"
            # Don't halt — attempt report and email with whatever was scored
    else:
        results["agent3"] = agent3_output

    # ── STEP 4: Report Generation (deterministic Python renderer) ─────────────
    # Agent 4's LLM-driven HTML generation has been replaced by
    # render_report.py, which reads model_scores.csv and writes
    # model_report.html directly. This is ~1-2 seconds vs. ~90 seconds for
    # the Claude call and avoids the 16k-token output ceiling that truncated
    # the HTML. No rate-limit wait needed.
    agent4_output = None if force else _load("agent4")

    if agent4_output is None:
        logger.info("▶ Starting Step 4: Report Rendering (render_report.render)")
        try:
            scores_path = PIPELINE_DIR / FILES["model_scores"]
            report_path = PIPELINE_DIR / FILES["model_report"]

            if not scores_path.exists():
                raise FileNotFoundError(
                    f"model_scores.csv not found at {scores_path} — Agent 3 must run first"
                )

            # Refresh the master ledger's WIN/LOSS/PUSH grades BEFORE rendering
            # so the rolling-accuracy matrix inside render() reflects today's
            # latest graded results. Zero-arg, returns a JSON status string.
            grading_result = auto_grade_historical_picks()
            logger.info(f"[AutoGrade] {str(grading_result)[:300]}")

            # Rebuild the forward-working live predictions log (union of all
            # archived model_scores joined against actuals_2026.parquet). The
            # renderer's _compute_cutoffs_2026 union-reads this + the sterile
            # backtest so the Optimal Cutoff table updates each day.
            livelog_result = rebuild_live_predictions_log()
            logger.info(f"[LiveLog] {str(livelog_result)[:300]}")

            df = pd.read_csv(scores_path)
            html = render(df)
            report_path.write_text(html, encoding="utf-8")

            if not report_path.exists() or report_path.stat().st_size < 1024:
                raise RuntimeError(
                    f"Render wrote {report_path} but file is missing or truncated "
                    f"({report_path.stat().st_size if report_path.exists() else 0} bytes)"
                )

            act = df[df.get("actionable") == True]
            tier1 = int((act.get("tier") == 1).sum()) if "tier" in act.columns else 0
            tier2 = int((act.get("tier") == 2).sum()) if "tier" in act.columns else 0

            agent4_output = (
                f"Report rendered deterministically.\n"
                f"  path:            {report_path}\n"
                f"  bytes:           {report_path.stat().st_size:,}\n"
                f"  rows:            {len(df)}\n"
                f"  actionable:      {len(act)} (Tier 1: {tier1} | Tier 2: {tier2})"
            )
            _save("agent4", agent4_output)
            results["agent4"] = agent4_output
            logger.info(f"✅ Report rendered.\n{agent4_output}")

        except Exception as e:
            logger.error(f"❌ Report rendering error: {e}", exc_info=True)
            results["agent4"] = f"ERROR: {e}"
            # Don't halt — email stage can still fire off model_scores.csv
    else:
        results["agent4"] = agent4_output

    # ── STEP 5: Notification (deterministic Python sender) ────────────────────
    # Agent 5's LLM-driven email composition has been replaced by a direct
    # Python call to send_email(). The body is the full rendered HTML report
    # (model_report.html) embedded inline, plus a plaintext summary fallback
    # and a PDF attachment — all produced by send_email itself. No Claude call,
    # no inter-agent sleep, no token cost.
    logger.info("▶ Starting Step 5: Notification (deterministic send_email)")
    try:
        from tools.implementations import send_email

        scores_path = PIPELINE_DIR / FILES["model_scores"]
        df_all = pd.read_csv(scores_path)
        df_act = df_all[df_all.get("actionable") == True].copy()

        n_total = len(df_act)
        n_tier1 = int((df_act.get("tier") == 1).sum()) if "tier" in df_act.columns else 0
        n_tier2 = int((df_act.get("tier") == 2).sum()) if "tier" in df_act.columns else 0

        def _fmt_pick(r) -> str:
            game  = r.get("game", "")
            model = r.get("model", "")
            pick  = r.get("pick_direction", "")
            btype = r.get("bet_type", "")
            odds  = r.get("retail_american_odds", "")
            stake = r.get("dollar_stake", "")
            prob  = r.get("model_prob", None)
            prob_s = f"{prob*100:.1f}%" if isinstance(prob, (int, float)) else ""
            odds_s = f"{int(odds):+d}" if pd.notna(odds) else ""
            return (f"  • {game} — {model} {btype} {pick}  "
                    f"| prob {prob_s} | odds {odds_s} | stake ${stake}")

        lines: list[str] = [
            "=" * 60,
            f"WIZARD REPORT | {TODAY}",
            "=" * 60,
            f"Actionable edges: {n_total}  (Tier 1: {n_tier1}  |  Tier 2: {n_tier2})",
            "",
        ]
        if n_tier1:
            lines.append("--- TIER 1 (Strong Edge ≥ 3.0%) ---")
            for _, r in df_act[df_act["tier"] == 1].iterrows():
                lines.append(_fmt_pick(r))
            lines.append("")
        if n_tier2:
            lines.append("--- TIER 2 (Medium Edge 1.0–2.9%) ---")
            for _, r in df_act[df_act["tier"] == 2].iterrows():
                lines.append(_fmt_pick(r))
            lines.append("")
        lines.append("See attached HTML body and PDF for full detail.")
        body = "\n".join(lines)

        subject = f"WIZARD REPORT: {n_total} Actionable Edges — {TODAY}"
        send_result = send_email(subject=subject, body=body, attach_report=True)
        results["agent5"] = send_result
        logger.info(f"✅ Step 5 complete.\n{send_result[:400]}")

    except Exception as e:
        logger.error(f"❌ Step 5 (Notification) error: {e}", exc_info=True)
        results["agent5"] = f"ERROR: {e}"

    # ── ACCURACY TRACKING ─────────────────────────────────────────────────────
    # Archive today's predictions and evaluate against completed games.
    logger.info("▶ Archiving predictions and evaluating accuracy")
    try:
        import subprocess, sys
        from pathlib import Path as PathlibPath
        tracker_script = PathlibPath(settings.PIPELINE_DIR) / "build_nrfi_accuracy_tracker.py"
        if tracker_script.exists():
            subprocess.run([sys.executable, str(tracker_script)],
                          cwd=str(settings.PIPELINE_DIR),
                          capture_output=True, text=True)
            logger.info("✅ Accuracy tracking complete")
        else:
            logger.warning(f"⚠ Tracker script not found: {tracker_script}")
    except Exception as e:
        logger.warning(f"⚠ Accuracy tracking failed (non-fatal): {e}")

    results["pipeline_status"] = "COMPLETE"
    logger.info("=" * 60)
    logger.info(f"✅ Daily pipeline complete | {TODAY}")
    logger.info("=" * 60)
    return results
