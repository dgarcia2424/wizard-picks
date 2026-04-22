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

from agents.definitions import AGENT1, AGENT3, AGENT4, AGENT5
from orchestrator.agent_loop import run_agent, PipelineHaltError
from supabase_upload import upload_picks

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

    # ── AGENT 1: Data Ingestion ───────────────────────────────────────────────
    logger.info("▶ Starting Agent 1: Data Ingestion")

    agent1_output = None if force else _load("agent1")

    if agent1_output is None:
        try:
            agent1_output = run_agent(
                system_prompt=AGENT1.system_prompt,
                tools=AGENT1.tools,
                tool_executor=AGENT1.tool_executor,
                user_message=(
                    f"Run the full data ingestion workflow for today ({TODAY}). "
                    "Check stale files, fetch confirmed starters, fetch DK/FD lines, "
                    "fetch ballpark weather, and write games.csv. "
                    "HALT immediately if The Odds API returns zero lines."
                ),
                agent_name=AGENT1.name,
            )
            _save("agent1", agent1_output)
            results["agent1"] = agent1_output
            logger.info(f"✅ Agent 1 complete.\n{agent1_output[:500]}")

        except PipelineHaltError as e:
            logger.error(f"❌ PIPELINE HALTED by Agent 1: {e}")
            results["agent1"] = f"HALT: {e}"
            results["pipeline_status"] = "HALTED"
            return results

        except Exception as e:
            logger.error(f"❌ Agent 1 unexpected error: {e}", exc_info=True)
            results["agent1"] = f"ERROR: {e}"
            results["pipeline_status"] = "FAILED"
            return results
    else:
        results["agent1"] = agent1_output

    # ── AGENT 3: Scoring Engine ───────────────────────────────────────────────
    agent3_output = None if force else _load("agent3")

    if agent3_output is None:
        logger.info("Waiting 90s between agents to respect rate limits...")
        time.sleep(90)
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

            # Upload results to Supabase for the web dashboard
            try:
                from pathlib import Path as _Path
                import os as _os
                _pipeline_dir = _Path(_os.environ.get("PIPELINE_DIR", _Path(__file__).parent.parent.parent))
                upload_picks(_pipeline_dir / "model_scores.csv")
            except Exception as _e:
                logger.warning(f"[Supabase] Upload failed (non-fatal): {_e}")

        except Exception as e:
            logger.error(f"❌ Agent 3 error: {e}", exc_info=True)
            results["agent3"] = f"ERROR: {e}"
            # Don't halt — attempt report and email with whatever was scored
    else:
        results["agent3"] = agent3_output

    # ── AGENT 4: Report Generation ────────────────────────────────────────────
    agent4_output = None if force else _load("agent4")

    if agent4_output is None:
        logger.info("Waiting 90s between agents to respect rate limits...")
        time.sleep(90)
        logger.info("▶ Starting Agent 4: Report Generation")
        try:
            agent4_output = run_agent(
                system_prompt=AGENT4.system_prompt,
                tools=AGENT4.tools,
                tool_executor=AGENT4.tool_executor,
                max_tokens=8096,
                user_message=(
                    f"Generate today's model_report.html for {TODAY}. "
                    "Read model_scores.csv and current tracker stats. "
                    "Build the complete self-contained HTML report with picks table, "
                    "Log Bet buttons, WIN/LOSS/PUSH result entry, and live stats bar. "
                    "Write the file when complete."
                ),
                context=f"Agent 3 (Scoring Engine) completed:\n{results.get('agent3', 'No output')}",
                agent_name=AGENT4.name,
            )
            _save("agent4", agent4_output)
            results["agent4"] = agent4_output
            logger.info(f"✅ Agent 4 complete.\n{agent4_output[:300]}")

        except Exception as e:
            logger.error(f"❌ Agent 4 error: {e}", exc_info=True)
            results["agent4"] = f"ERROR: {e}"
    else:
        results["agent4"] = agent4_output

    # ── AGENT 5: Notification ─────────────────────────────────────────────────
    agent5_output = None if force else _load("agent5")

    if agent5_output is None:
        logger.info("Waiting 60s between agents to respect rate limits...")
        time.sleep(60)
        logger.info("▶ Starting Agent 5: Notification")
        try:
            agent5_output = run_agent(
                system_prompt=AGENT5.system_prompt,
                tools=AGENT5.tools,
                tool_executor=AGENT5.tool_executor,
                user_message=(
                    f"Generate today's picks summary from model_scores.csv and send the daily email. "
                    f"Subject: 'Wizard Picks — {TODAY}'. "
                    "Flag any picks at 72%+ as high conviction. Attach model_report.html."
                ),
                context=f"Agent 4 (Report Generation) completed:\n{results.get('agent4', 'No output')}",
                agent_name=AGENT5.name,
            )
            _save("agent5", agent5_output)
            results["agent5"] = agent5_output
            logger.info(f"✅ Agent 5 complete.\n{agent5_output[:300]}")

        except Exception as e:
            logger.error(f"❌ Agent 5 error: {e}", exc_info=True)
            results["agent5"] = f"ERROR: {e}"
    else:
        results["agent5"] = agent5_output

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
