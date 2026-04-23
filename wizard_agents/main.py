"""
main.py — Daily pipeline entry point
Run at 10 AM: python main.py
"""
import sys

# Force UTF-8 on stdout/stderr so emoji in log lines (✅ ❌ ▶ 🎯 ⚠️) don't
# crash the Windows cp1252 console. Must run before the logger's StreamHandler
# is created in daily_pipeline. No-op on platforms that already default to utf-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

from orchestrator.daily_pipeline import run_daily_pipeline, PipelineHaltError


def main():
    try:
        results = run_daily_pipeline()
        status  = results.get("pipeline_status", "UNKNOWN")

        if status == "HALTED":
            print("\n❌ Pipeline halted — no lines available. Try again after 10 AM.")
            sys.exit(1)
        elif status == "FAILED":
            print("\n❌ Pipeline failed unexpectedly. Check wizard_pipeline.log.")
            sys.exit(1)
        else:
            print("\n✅ Pipeline complete. Check your email and open model_report.html.")
            sys.exit(0)

    except PipelineHaltError as e:
        print(f"\n❌ HALT: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
