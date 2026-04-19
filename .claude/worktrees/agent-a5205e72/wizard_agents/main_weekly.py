"""
main_weekly.py — Weekly maintenance entry point (run Sundays)
Also runs Agent 7 (Maintenance & Roadmap) on demand.

Usage:
  python main_weekly.py           → static data validation report + roadmap status
  python main_weekly.py --roadmap → roadmap status only (no file validation)
"""
import sys
import logging
from datetime import date

from agents.definitions import AGENT2, AGENT7
from orchestrator.agent_loop import run_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("wizard.weekly")

TODAY = date.today().strftime("%B %d, %Y")


def run_weekly_maintenance():
    logger.info("=" * 60)
    logger.info(f"THE WIZARD REPORT — Weekly Maintenance | {TODAY}")
    logger.info("=" * 60)

    # ── AGENT 2: Static Data Manager ─────────────────────────────────────────
    logger.info("▶ Agent 2: Static Data Manager")
    agent2_output = run_agent(
        system_prompt=AGENT2.system_prompt,
        tools=AGENT2.tools,
        tool_executor=AGENT2.tool_executor,
        user_message=(
            "Run the weekly static data validation for all pipeline CSVs. "
            "Validate savant_pitchers.csv (~52KB) and savant_batters.csv (~341KB). "
            "Validate fangraphs_pitchers.csv, fangraphs_batters.csv (confirm multi-year blending applied). "
            "Validate fangraphs_team_vs_lhp.csv and fangraphs_team_vs_rhp.csv (30 rows each). "
            "Report ✅/❌ per file with row counts and any warnings."
        ),
        agent_name=AGENT2.name,
    )
    print(f"\n{'='*60}\nSTATIC DATA REPORT\n{'='*60}\n{agent2_output}")

    # ── AGENT 7: Maintenance & Roadmap ────────────────────────────────────────
    logger.info("▶ Agent 7: Maintenance & Roadmap")
    agent7_output = run_agent(
        system_prompt=AGENT7.system_prompt,
        tools=AGENT7.tools,
        tool_executor=AGENT7.tool_executor,
        user_message=(
            f"Generate the weekly maintenance status report for {TODAY}. "
            "Check stale files, read current tracker stats, check pick count against 200-pick milestone, "
            "check 30-pick rolling win rate, and surface the next highest-priority roadmap item. "
            "Flag any triggered alert conditions."
        ),
        agent_name=AGENT7.name,
    )
    print(f"\n{'='*60}\nROADMAP & MAINTENANCE REPORT\n{'='*60}\n{agent7_output}")


def run_roadmap_only():
    logger.info("Running Agent 7 (Roadmap) only...")
    output = run_agent(
        system_prompt=AGENT7.system_prompt,
        tools=AGENT7.tools,
        tool_executor=AGENT7.tool_executor,
        user_message=(
            f"Generate a quick roadmap status for {TODAY}. "
            "Read tracker stats, check pick count vs 200-pick milestone, "
            "and recommend the single highest-priority next development task."
        ),
        agent_name=AGENT7.name,
    )
    print(f"\n{output}")


if __name__ == "__main__":
    if "--roadmap" in sys.argv:
        run_roadmap_only()
    else:
        run_weekly_maintenance()
