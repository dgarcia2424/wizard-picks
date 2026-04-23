"""
main_weekly.py — Weekly maintenance entry point (run Sundays)

Thin shim over sunday_routine.py, retained so existing cron/launchd jobs
that invoke `python main_weekly.py` continue to work after the LLM-driven
Agent 2 / Agent 7 were retired.

Usage:
  python main_weekly.py            → static data validation + roadmap status
  python main_weekly.py --roadmap  → roadmap status only (no file validation)
"""
import sys

from sunday_routine import main

if __name__ == "__main__":
    sys.exit(main())
