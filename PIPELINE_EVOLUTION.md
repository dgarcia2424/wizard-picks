## The Wizard Report — Pipeline Evolution

**MLB Predictive Analytics | Dan Garcia | April 2026**

This document traces the architectural journey of the Wizard Report pipeline from its original multi-agent design to its current lean, mostly-deterministic form — and explains *why* each consolidation happened.

---

### Phase 1 — The Original Build: Seven Agents, Framework-Driven

The first version of the pipeline was structured around **seven distinct LLM-driven agents**, originally prototyped on top of a framework (CrewAI-style Crew/Task/BaseTool abstractions). The goal was to mirror a human analyst team, with each agent owning a role:

| Agent | Role |
|-------|------|
| 1 — Data Ingestion | Pull odds, starters, weather; assemble `games.csv` |
| 2 — Static Data Manager | Validate weekly FanGraphs / Savant uploads |
| 3 — Scoring Engine | Run stacker models, apply Three-Part Lock, write `model_scores.csv` |
| 4 — Report Generation | Render the HTML tear-sheet |
| 5 — Notification | Compose and send the daily email |
| 6 — Bet Tracker | Log bets and results |
| 7 — Maintenance / Roadmap | Weekly status + advisory alerts |

**Why this shape?** It was intuitive — each concern got a prompt, a toolset, and an executor. Context flowed between agents as plain strings. In theory, every stage could reason about its inputs, handle edge cases, and produce a narrative report.

---

### Phase 2 — Ditching the Framework

The first major change was abandoning CrewAI-style abstractions in favor of the **pure Anthropic SDK**.

**Why:**
- `BaseTool` inheritance, `Crew` objects, and `Task` orchestration were hiding what was really just: *call Claude → if `tool_use`, run the function, feed the result back → repeat*.
- Framework "magic" made debugging brittle (silent context loss, unclear retry semantics, tool-schema drift).
- A ~60-line `agent_loop.py` replaced hundreds of lines of framework glue.

Each agent became three plain things: a `system_prompt`, a list of JSON-schema tools, and a Python dispatch function. Orchestration became explicit Python in `daily_pipeline.py` — a visible DAG instead of a hidden state machine.

---

### Phase 3 — Collapsing Agents 1, 4, and 5 into Deterministic Python

Over time, three of the seven agents were pulled out of the LLM path entirely. The LLM `AgentDefinition`s were kept as **reference specifications**, but the runtime now calls their tool functions directly.

#### Agent 1 (Data Ingestion) → deterministic
**Why:** The work was never ambiguous. Hit the Odds API, hit MLB Stats API, hit Open-Meteo, write a CSV. Claude added latency and a nonzero chance of hallucinating a column or mis-mapping a starter, with zero upside over `fetch_odds_api() → fetch_mlb_starters() → fetch_weather() → write_csv()` called in order. HALT semantics (empty slate) were easier to enforce as a Python exception than as a prompt instruction.

#### Agent 4 (Report Generation) → `render_report.render()`
**Why:** Two reasons, one soft and one hard.
- *Soft:* the HTML layout (tear-sheet → actionable hero table → alpha markets → full picks) was fully specified. Every element — colored market badges, Tier 1 vs Tier 2 ordering, POST `/log_result` buttons — was deterministic. Letting Claude re-derive it daily was wasted tokens.
- *Hard:* `max_tokens` truncated the generated HTML at ~16k tokens on busy slates. The report would silently cut off mid-table. A deterministic renderer fixed this permanently.

#### Agent 5 (Notification) → `send_email()`
**Why:** The email is a rigid template (plain-text, terminal-style, no HTML/emoji). Subject lines follow a strict pattern. Per-pick blocks have fixed field order. There was no reasoning to do — only formatting — and a single misbehaving prompt could silently ship a malformed email to the distribution list.

---

### Phase 4 — Moving Agent 6 Out-of-Process

The Bet Tracker was originally an in-loop agent that would log bets via tool calls. It's now a standalone Flask-ish HTTP server (`tracker_server.py`, port 5151) with three endpoints: `POST /log_bet`, `POST /log_result`, `GET /stats`.

**Why:**
- Bet logging is driven by the HTML report's "Log Bet" / "Log Result" buttons — a browser event, not a pipeline stage.
- Coupling it to the daily agent chain meant the tracker was only "alive" during the 10 AM run. An always-on HTTP service fits the real usage pattern.
- The **Auto-Reconciliation Engine** (`auto_grade_historical_picks` + `actuals_2026.parquet`) now grades picks automatically against Statcast outcomes — eliminating most manual logging entirely. `AGENT6_TOOLS` was reduced to a read-only `compute_rolling_accuracy`.

---

### Phase 5 — Today: One LLM Agent in the Daily Hot Path

The current daily DAG:

```
Step 1 [Python]     fetch_odds → (HALT?) → starters → weather → write games.csv
        ↓
Agent 3 [Claude]    run_agent(AGENT3) → generate_ml_scores → archive snapshot
        ↓
Step 4 [Python]     auto_grade → rebuild_live_predictions → render_report
        ↓
Step 5 [Python]     send_email(subject, body, attach_report=True)
```

**Only Agent 3 (Scoring Engine) remains LLM-driven in daily production** — and even it is a near-single-tool-call agent (`generate_ml_scores` does the real work; Claude's job is to invoke it and emit the completion report).

**Agent 2** (static data) and **Agent 7** (roadmap) run weekly, not daily, and are genuinely discretionary tasks where LLM reasoning adds value: Agent 2 judges whether a FanGraphs upload is blended correctly (2026=30% / 2025=55% / 2024=15%) and whether size guards match the file signature; Agent 7 reads the tracker stats and the static roadmap and surfaces the next action.

---

### Why the Consolidation Worked

A pattern emerged across every agent we deprecated:

1. **The work was deterministic.** Fetch → transform → write. No judgment calls. No branching on ambiguous inputs.
2. **The output had a rigid schema.** Email templates, HTML layouts, CSV columns — all fully specified. Every LLM-generated variant was a chance to drift from the spec.
3. **Failures were silent.** Truncated HTML, malformed JSON, a missed column. Python raises; a prompt shrugs.
4. **Latency and tokens were pure cost.** No upside when the function is a straight pipe.

The agents we **kept** share the inverse profile:

- **Agent 3** — orchestrates multi-market scoring where a future extension (challenger models, new markets) benefits from prompt-level reasoning, and its tool already encapsulates the deterministic core.
- **Agent 2** — weekly, judgment-laden file validation with real "is this upload sane?" decisions.
- **Agent 7** — advisory reasoning over roadmap state, tracker trends, and stale-file alerts.

---

### Current State Summary

- **Harness:** Pure Anthropic SDK, ~60 lines of orchestration, no framework dependencies.
- **Daily LLM calls:** 1 (Agent 3).
- **Deterministic stages:** Ingestion, rendering, email, grading, archival.
- **Checkpoints:** `checkpoints/YYYY-MM-DD_*.json` provide per-agent idempotency; re-runs are free.
- **HALT semantics:** `{"status":"HALT"}` from any tool aborts the chain cleanly via `PipelineHaltError`.
- **Separation of concerns:** LLM agents reason; Python executes; HTTP server persists.

The trajectory was consistent throughout: **use the LLM where reasoning is real, and use Python everywhere else.** Every agent we retired was one where the LLM was doing a human impression of a function call.
