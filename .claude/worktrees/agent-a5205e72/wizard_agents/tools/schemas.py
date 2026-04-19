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
        "name": "read_csv",
        "description": "Read a pipeline CSV file (games, fangraphs_pitchers, savant_batters, etc.).",
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
        "name": "read_csv_filtered",
        "description": (
            "Read a pipeline CSV filtered to specific player names or team abbreviations. "
            "USE THIS instead of read_csv for fangraphs_pitchers, fangraphs_batters, "
            "fangraphs_team_vs_lhp, fangraphs_team_vs_rhp, savant_pitchers, and savant_batters. "
            "Pass today's starting pitcher names in name_filter and/or today's team "
            "abbreviations in team_filter to reduce token usage by ~90%. "
            "NEVER call plain read_csv on these large files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_key":    {"type": "string", "description": "Pipeline file key, e.g. 'fangraphs_pitchers'"},
                "name_filter": {"type": "string", "description": "JSON array of player names, e.g. '[\"Brandon Pfaadt\",\"Taijuan Walker\"]'"},
                "team_filter": {"type": "string", "description": "JSON array of team abbreviations, e.g. '[\"PHI\",\"ARI\",\"DET\"]'"},
                "name_col":    {"type": "string", "description": "Column to match names against. Default: 'Name'", "default": "Name"},
                "team_col":    {"type": "string", "description": "Column to match teams against. Default: 'Team'", "default": "Team"},
            },
            "required": ["file_key"],
        },
    },
    {
        "name": "write_csv",
        "description": (
            "Write model scoring results to model_scores.csv. "
            "One row per pick with: date, game, model, bet_type, pick_direction, model_prob, "
            "projected_total, market_line_dk, market_line_fd, best_book, recommended_units, actionable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_key":     {"type": "string"},
                "records_json": {"type": "string"},
                "mode":         {"type": "string", "enum": ["w", "a"], "default": "w"},
            },
            "required": ["file_key", "records_json"],
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 4 — Report Generation
# ══════════════════════════════════════════════════════════════════════════════

AGENT4_TOOLS = [
    {
        "name": "read_csv",
        "description": "Read model_scores.csv or bet_tracker.csv to build the HTML report.",
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
        "name": "append_bet",
        "description": (
            "Append a new bet to bet_tracker.csv with result=PENDING. "
            "This is the ONLY function that may write new rows to bet_tracker.csv."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date":        {"type": "string", "description": "YYYY-MM-DD"},
                "game":        {"type": "string", "description": "'AWAY @ HOME'"},
                "model":       {"type": "string", "enum": ["MFull", "MF5i", "MF3i", "MF1i", "MBat"]},
                "bet_type":    {"type": "string", "description": "e.g. 'Under 8.0'"},
                "model_prob":  {"type": "number", "description": "Probability as float, e.g. 69.5"},
                "market_line": {"type": "string", "description": "e.g. '-110', '+115'"},
                "book":        {"type": "string", "enum": ["FD", "DK", "BET365", "BETMGM"]},
                "units":       {"type": "number", "enum": [0.5, 1.0, 1.5]},
                "notes":       {"type": "string", "default": ""},
            },
            "required": ["date", "game", "model", "bet_type", "model_prob", "market_line", "book", "units"],
        },
    },
    {
        "name": "log_result",
        "description": (
            "Record WIN, LOSS, or PUSH for a logged bet. Calculates profit_loss. "
            "Freezes the row permanently — cannot be modified after this call. "
            "REJECTS if result is already recorded."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "bet_id":       {"type": "integer", "description": "The id from bet_tracker.csv"},
                "result":       {"type": "string", "enum": ["WIN", "LOSS", "PUSH"]},
                "actual_total": {"type": "number", "description": "Optional actual game total"},
            },
            "required": ["bet_id", "result"],
        },
    },
    {
        "name": "read_tracker_stats",
        "description": "Aggregate stats from bet_tracker.csv: overall and per-model records, win rates, P/L, ROI.",
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
