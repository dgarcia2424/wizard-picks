# The Wizard Report — Claude Agents Pipeline
**MLB Predictive Analytics | Dan Garcia | April 2026**
**Pure Anthropic SDK. No framework. No CrewAI.**

## Project Structure

```
wizard_agents/
├── main.py                        # Daily pipeline (10 AM)
├── main_weekly.py                 # Weekly maintenance + roadmap (Sundays)
├── tracker_server.py              # Agent 6 HTTP server (port 5151)
├── requirements.txt
├── .env.example
├── config/
│   └── settings.py                # All constants, parameters, file paths
├── tools/
│   ├── implementations.py         # Pure Python tool functions
│   └── schemas.py                 # JSON Schema dicts Claude reads per agent
├── agents/
│   └── definitions.py             # All 7 agents: system_prompt + tools + executor
└── orchestrator/
    ├── agent_loop.py              # THE LOOP: call → tool_use → execute → repeat
    └── daily_pipeline.py          # Agent 1 → 3 → 4 → 5 sequential chain
```

## How It Works

**No framework.** Each agent is three things:
1. `system_prompt` — the agent's identity, rules, and parameters
2. `tools` — JSON Schema dicts telling Claude what tools exist
3. `tool_executor` — Python dispatch function that runs the actual code

The `agent_loop.py` drives everything:
```
call Claude API
  → if stop_reason == "tool_use": execute tool(s), feed result back, repeat
  → if stop_reason == "end_turn": return final text
```

Context flows between agents as plain strings passed in the `context=` argument.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env — ANTHROPIC_API_KEY is required, others have defaults
```

## Usage

**Daily run (10 AM):**
```bash
python main.py
```

**Weekly maintenance (Sundays):**
```bash
python main_weekly.py
```

**Roadmap status only:**
```bash
python main_weekly.py --roadmap
```

**Start tracker server manually:**
```bash
python tracker_server.py
```

## Windows Task Scheduler

- Program: `C:\Users\garci\anaconda3\python.exe`
- Arguments: `main.py`
- Start in: `C:\Users\garci\Downloads\mlb_model_pipeline_dan\wizard_agents\`
- Trigger: Daily at 10:00 AM

## Agent Map

| Agent | File | Trigger | Writes to |
|-------|------|---------|-----------|
| 1 — Data Ingestion | definitions.py → AGENT1 | Daily 10 AM (auto) | games.csv |
| 2 — Static Data Manager | definitions.py → AGENT2 | Sunday manual | 6 static CSVs |
| 3 — Scoring Engine | definitions.py → AGENT3 | After Agent 1 | model_scores.csv |
| 4 — Report Generation | definitions.py → AGENT4 | After Agent 3 | model_report.html |
| 5 — Notification | definitions.py → AGENT5 | After Agent 4 | Email |
| 6 — Bet Tracker | tracker_server.py | On-demand (HTTP) | bet_tracker.csv ONLY |
| 7 — Maintenance | definitions.py → AGENT7 | Weekly/on-demand | Nothing |

## Key Design Decisions vs CrewAI

- **No BaseTool inheritance** — tools are plain Python functions
- **No Crew/Task objects** — orchestration is explicit Python in daily_pipeline.py
- **PipelineHaltError** — propagates HALT signals from Agent 1 cleanly
- **Context as strings** — prior agent output passed directly, no framework magic
- **Tool schemas per agent** — each agent only sees its own tools (least privilege)
- **Single dependency** — only `anthropic` SDK required for the agent loop
