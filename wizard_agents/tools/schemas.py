"""
tools/schemas.py

Tool schema dicts for each agent.
Claude reads these to know: what tools exist, what they do, what args they take.
Each agent only receives the schemas for its own tools — principle of least privilege.
"""

# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — Data Ingestion
# ══════════════════════════════════════════════════════════════════════════════

AGENT1_TOOLS = [
    {
        "name": "check_stale_files",
        "description": (
            "Check modification timestamps on all static data CSVs. "
            "Emits STALE warnings for files older than 7 days. Pipeline continues regardless — advisory only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "threshold_days": {
                    "type": "integer",
                    "description": "Days before a file is considered stale. Default: 7.",
                    "default": 7,
                }
            },
            "required": [],
        },
    },
    {
        "name": "fetch_odds_api",
        "description": (
            "Fetch DraftKings and FanDuel moneylines and totals for today's MLB slate from The Odds API. "
            "Returns game totals, F5/F3/F1 estimates, and best-book flags. "
            "Returns status=HALT if zero games returned — pipeline must stop."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "game_date": {
                    "type": "string",
                    "description": "ISO date YYYY-MM-DD. Leave empty for today.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "fetch_mlb_starters",
        "description": (
            "Fetch today's confirmed starting pitchers from the official MLB Stats API (free, no key). "
            "Excludes games with TBD starters."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "game_date": {
                    "type": "string",
                    "description": "ISO date YYYY-MM-DD. Leave empty for today.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "fetch_weather",
        "description": (
            "Fetch weather conditions for each MLB ballpark via Open-Meteo (free, no key). "
            "Returns temperature, wind speed, wind direction, precipitation probability per game."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "games_json": {
                    "type": "string",
                    "description": 'JSON array of game objects with "home_team" and "game_id" fields.',
                }
            },
            "required": ["games_json"],
        },
    },
    {
        "name": "write_csv",
        "description": (
            "Write a JSON array of records to a pipeline CSV file. "
            "Use file_key='games' to write games.csv. "
            "Never use file_key='bet_tracker'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_key":     {"type": "string", "description": "Pipeline file key, e.g. 'games'"},
                "records_json": {"type": "string", "description": "JSON array of row dicts"},
                "mode":         {"type": "string", "enum": ["w", "a"], "default": "w"},
            },
            "required": ["file_key", "records_json"],
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — Static Data Manager
# ══════════════════════════════════════════════════════════════════════════════

AGENT2_TOOLS = [
    {
        "name": "validate_static_file",
        "description": (
            "Validate a static CSV file for required columns, minimum row count, and correct file size. "
            "Enforces savant size guards: savant_pitchers ~52KB, savant_batters ~341KB. "
            "REJECTS corrupt, wrong-size, or schema-invalid files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_key":         {"type": "string", "description": "Pipeline file key"},
                "expected_columns": {"type": "string", "description": "JSON array of required column names"},
                "min_rows":         {"type": "integer", "default": 10},
                "expected_size_kb": {"type": "integer", "description": "Approximate expected size in KB"},
            },
            "required": ["file_key", "expected_columns"],
        },
    },
    {
        "name": "read_csv",
        "description": "Read a pipeline CSV file by key. Returns JSON with rows, columns, and records.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_key": {"type": "string"},
                "max_rows": {"type": "integer", "description": "Limit rows returned. Omit for all rows."},
            },
            "required": ["file_key"],
        },
    },
    {
        "name": "write_csv",
        "description": "Write validated and blended records to a pipeline CSV file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_key":     {"type": "string"},
                "records_json": {"type": "string", "description": "JSON array of row dicts"},
                "mode":         {"type": "string", "enum": ["w", "a"], "default": "w"},
            },
            "required": ["file_key", "records_json"],
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — Scoring Engine
# ══════════════════════════════════════════════════════════════════════════════

AGENT3_TOOLS = [
    {
        "name": "generate_ml_scores",
        "description": (
            "Run the full ML scoring pipeline for the given date. "
            "Invokes score_ml_today (full-game moneyline), score_run_dist_today "
            "(totals + runline), and score_f5_today (first 5 innings), joins with "
            "games.csv, applies the Three-Part Lock (sanity check vs Pinnacle, "
            "retail odds floor >= -225, edge >= 1.0%), computes Kelly dollar stakes "
            "(Tier 1 Quarter-Kelly / Tier 2 Eighth-Kelly, $2000 bank, $50 cap), "
            "writes model_scores.csv, and returns a compact JSON summary containing "
            "ONLY the actionable picks and gate-failure counts. "
            "This is the sole scoring tool — no external heuristic models are used."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "game_date": {
                    "type": "string",
                    "description": "ISO date YYYY-MM-DD. Leave empty for today.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "read_csv",
        "description": (
            "Read a pipeline CSV file. Use sparingly — generate_ml_scores already "
            "writes model_scores.csv and returns the actionable picks directly. "
            "Use this only to spot-check games.csv (≤15 rows) if something in the "
            "scoring summary looks off."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_key": {"type": "string"},
                "max_rows": {"type": "integer"},
            },
            "required": ["file_key"],
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 4 — Report Generation
# ══════════════════════════════════════════════════════════════════════════════

AGENT4_TOOLS = [
    {
        "name": "auto_grade_historical_picks",
        "description": (
            "Auto-Reconciliation Engine: grade every ungraded row in the master "
            "ledger (historical_actionable_picks.csv) against actuals_2026.parquet. "
            "Writes WIN/LOSS/PUSH + profit_loss back to the ledger. Call before "
            "compute_rolling_accuracy so the tear sheet reflects fresh results."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "compute_rolling_accuracy",
        "description": (
            "Aggregate the graded ledger into three rolling windows — last 7 days, "
            "last 28 days, season-to-date (2026) — each with overall and per-market "
            "(ML/Totals/Runline/F5/NRFI) Total Bets, Win %, and ROI %. Used to render "
            "the Rolling Accuracy Tear Sheet at the top of model_report.html."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_csv",
        "description": "Read model_scores.csv to build the HTML report.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_key": {"type": "string"},
                "max_rows": {"type": "integer"},
            },
            "required": ["file_key"],
        },
    },
    {
        "name": "read_tracker_stats",
        "description": "Compute live aggregate stats from bet_tracker.csv for the HTML stats bar.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "write_html",
        "description": (
            "Write the completed HTML string to model_report.html. "
            "The HTML must be fully self-contained (single file, no external dependencies). "
            "Use file_key='model_report'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_key":     {"type": "string", "description": "Use 'model_report'"},
                "html_content": {"type": "string", "description": "Complete HTML string"},
            },
            "required": ["file_key", "html_content"],
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 5 — Notification
# ══════════════════════════════════════════════════════════════════════════════

AGENT5_TOOLS = [
    {
        "name": "read_csv",
        "description": "Read model_scores.csv to generate the picks summary.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_key": {"type": "string"},
                "max_rows": {"type": "integer"},
            },
            "required": ["file_key"],
        },
    },
    {
        "name": "send_email",
        "description": (
            "Send the daily Wizard Picks email via Gmail SMTP. "
            "Attaches model_report.html automatically. Retries up to 2 times."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "subject":       {"type": "string", "description": "Email subject line"},
                "body":          {"type": "string", "description": "Picks summary for email body"},
                "attach_report": {"type": "boolean", "default": True},
            },
            "required": ["subject", "body"],
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 6 — Bet Tracker
# ══════════════════════════════════════════════════════════════════════════════

AGENT6_TOOLS = [
    {
        "name": "compute_rolling_accuracy",
        "description": (
            "Read-only: return rolling 7-day / 28-day / YTD accuracy windows from "
            "the auto-graded master ledger. Agent 6 is now a passive observer — "
            "manual bet logging has been deprecated in favor of the "
            "Auto-Reconciliation Engine."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 7 — Maintenance
# ══════════════════════════════════════════════════════════════════════════════

AGENT7_TOOLS = [
    {
        "name": "read_tracker_stats",
        "description": "Read aggregate performance stats from bet_tracker.csv for roadmap status report.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "check_stale_files",
        "description": "Check if static data files are overdue for weekly refresh.",
        "input_schema": {
            "type": "object",
            "properties": {
                "threshold_days": {"type": "integer", "default": 7}
            },
            "required": [],
        },
    },
]
