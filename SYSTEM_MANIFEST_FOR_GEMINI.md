# SYSTEM_MANIFEST_FOR_GEMINI

> Cold-start architectural context for The Wizard Report — an MLB predictive-analytics pipeline built on pure Anthropic SDK (no CrewAI/LangChain). Authored as a handoff to Gemini.

---

## 1. Functional Topology

Project root: `C:\Users\garci\OneDrive\Documents\Claude\mlb_model_pipeline_dan\`
Agent harness:  `./wizard_agents/` (the directory this manifest describes)

```
mlb_model_pipeline_dan/                ← pipeline root (ML inference code + data)
├── games.csv                          Daily odds+starters table (Agent 1 output)
├── model_scores.csv                   Daily picks w/ Three-Part Lock (Agent 3 output)
├── model_report.html                  Rendered tear-sheet (render_report.py)
├── historical_actionable_picks.csv    Master ledger (auto-graded each day)
├── live_predictions_2026.csv          Rolling union of archived scores vs actuals
├── data/
│   ├── raw/                           Manually-uploaded FanGraphs + Savant CSVs
│   ├── statcast/actuals_2026.parquet  Game outcomes for grading
│   ├── statcast/odds_snapshot_*.pq    Daily odds archival (incl. F5+NRFI)
│   └── predictions/model_scores_*.csv Daily raw stacker snapshot
├── score_ml_today.py                  Full-game ML stacker driver
├── score_run_dist_today.py            Totals + Runline driver
├── score_f5_today.py                  First-5-innings driver
├── score_nrfi_today.py                No-Run First-Inning driver
├── render_report.py                   Deterministic HTML renderer (replaces LLM Agent 4)
├── train_xgboost.py                   RL/ML XGBoost + Bayesian stacker training
├── train_f5_model.py / train_nrfi_model.py / train_run_dist_model.py
├── models/                            Pickled stackers + XGBoost boosters
├── tracker_server.py                  Port 5151 HTTP bet logger (Agent 6)
└── wizard_agents/                     ← THE AGENT HARNESS
    ├── main.py                        Daily pipeline entry (10 AM)
    ├── main_weekly.py                 Weekly Agent 2 + Agent 7 entry (Sundays)
    ├── config/settings.py             Constants, thresholds, Kelly math, file map
    ├── tools/
    │   ├── schemas.py                 Per-agent JSON-Schema tool defs (least privilege)
    │   └── implementations.py         Pure Python tool functions (~1800 LOC)
    ├── agents/definitions.py          7 AgentDefinition dataclasses (prompt+tools+exec)
    ├── orchestrator/
    │   ├── agent_loop.py              call→tool_use→execute→repeat (max 20 rounds)
    │   └── daily_pipeline.py          Sequential chain w/ per-agent JSON checkpoints
    └── checkpoints/YYYY-MM-DD_*.json  Idempotency barrier for re-runs
```

**Role of each agent file**

| Layer | File | Role |
|-------|------|------|
| Prompts | `agents/definitions.py` | Defines `AgentDefinition(name, system_prompt, tools, tool_executor)` for AGENT1..7 |
| Schemas | `tools/schemas.py` | AGENT{1,2,3,4,5,6,7}_TOOLS lists — Claude never sees tools outside its own list |
| Impl | `tools/implementations.py` | Deterministic Python; each returns a JSON string; HALT sentinel = `{"status":"HALT"}` |
| Loop | `orchestrator/agent_loop.py` | Raises `PipelineHaltError` on HALT; retries rate limits twice; 20-round cap |
| DAG | `orchestrator/daily_pipeline.py` | Linear DAG 1→3→4→5, with 1/4/5 now deterministic Python (no Claude) |
| Config | `config/settings.py` | `FILES{}`, Three-Part Lock thresholds, `kelly_stake()`, `CLAUDE_MODEL="claude-sonnet-4-20250514"` |

---

## 2. Agent Registry

### AGENT 1 — Data Ingestion Agent  *(LLM-driven path retained; production uses deterministic bypass in daily_pipeline.py)*

**System Prompt (verbatim):**
> You are the Data Ingestion Agent for The Wizard Report MLB analytics pipeline. Your sole responsibility is fetching all automated daily data inputs and assembling games.csv… (full block defines EXECUTION ORDER 1–4, REQUIRED COLUMNS — core/Retail FG/Retail F5/Retail NRFI/Pinnacle FG/Pinnacle F5/Pinnacle NRFI, STRICT LINE MATCHING, FAILURE HANDLING, final status report.)

**Tools (schemas.AGENT1_TOOLS):**
- `check_stale_files(threshold_days:int=7) → JSON` — advisory; never HALTs
- `fetch_odds_api(game_date:str="") → JSON` — dual-region (US retail + EU/Pinnacle); returns `{"status":"HALT"}` on empty slate
- `fetch_mlb_starters(game_date:str="") → JSON` — MLB Stats API, no key
- `fetch_weather(games_json:str) → JSON` — Open-Meteo per ballpark
- `write_csv(file_key:str, records_json:str, mode:"w"|"a") → JSON`

**Success criteria:** emits status report lines: `Games fetched: N | Starters confirmed: N | TBD: N | Retail lines: DK/FD/BetMGM | Pinnacle: N | games.csv written: ✅`.

---

### AGENT 2 — Static Data Manager  *(Sundays only, LLM-driven)*

**System Prompt:** FanGraphs Excel blend rules (2026=30%, 2025=55%, 2024=15%), per-file size guards (savant_pitchers ~52KB, savant_batters ~341KB — hard REJECT on inverted upload), required columns per file, never pass corrupt data downstream.

**Tools:** `validate_static_file`, `read_csv`, `write_csv`.

**Success criteria:** completion report with ✅/❌ per static CSV + row counts + "Next update: [next Sunday]".

---

### AGENT 3 — Scoring Engine  *(LLM-driven, single tool call)*

**System Prompt:** Exactly two steps — (1) call `generate_ml_scores(game_date)`; (2) emit completion report. Markets = ML, Totals, Runline, F5, NRFI. Three-Part Lock is enforced inside the tool. Never reference legacy labels (MF1i/MF3i/MFull/MF5i/MBat). No manual fallback path.

**Tools:**
- `generate_ml_scores(game_date:str="") → JSON` — runs `_invoke_scorer()` on `score_ml_today`, `score_run_dist_today`, `score_f5_today`, `score_nrfi_today`; joins with games.csv; applies Sanity/Odds-floor/Edge gates; computes Kelly stakes; writes `model_scores.csv`; returns actionable picks + gate-failure counters.
- `read_csv(file_key, max_rows?)` — spot-check only.

**Success criteria:** agent echoes returned JSON keys `{status, games_scored, actionable, tier1, tier2, failed_sanity, failed_odds_floor, failed_edge, actionable_picks[]}`.

---

### AGENT 4 — Report Generation  *(DEPRECATED — replaced by deterministic `render_report.render()` in production; LLM definition retained for emergency fallback)*

**System Prompt:** Auto-reconcile first (`auto_grade_historical_picks` → `compute_rolling_accuracy` → `read_csv("model_scores")`), then render self-contained HTML with: Header, Rolling Accuracy Tear Sheet (7/28/YTD × ML/Totals/Runline/F5/NRFI), optional live stats bar, **Actionable Bets** hero table (Tier 1 then Tier 2, colored market badges), **Alpha Markets** section (ML + Totals, incl. non-actionable, sorted by Retail Edge desc), Result Entry buttons (POST `/log_result`), collapsed Full Picks Table. Log Bet POST → `http://localhost:{TRACKER_PORT=5151}/log_bet`.

**Tools:** `auto_grade_historical_picks`, `compute_rolling_accuracy`, `read_csv`, `read_tracker_stats`, `write_html`.

**Success criteria:** file at `FILES["model_report"]` ≥ 1024 bytes; report-end summary contains Tier1/Tier2 counts + per-model row counts + file write status.

---

### AGENT 5 — Notification  *(DEPRECATED LLM path — production uses deterministic `send_email()` in daily_pipeline.py)*

**System Prompt:** Read model_scores, partition by tier, compose plain-text terminal-style email (no HTML/links/emoji). Subject templates: `WIZARD REPORT: N Actionable Edge[s] — Month DD, YYYY` (or `0 Actionable Edges` branch w/ gate-fail counts). Per-pick block format rigid: Model, Bet, Model Prob, Pinnacle, Retail, Edge, Stake. Email failure = warn, never block.

**Tools:** `read_csv`, `send_email(subject, body, attach_report=True)`.

**Success criteria:** `email delivered ✅/❌` line with recipient list.

---

### AGENT 6 — Bet Tracker Server  *(out-of-process, not in definitions.py)*

Standalone Flask-ish HTTP server (`../tracker_server.py`) on port 5151. Endpoints consumed by model_report.html: `POST /log_bet`, `POST /log_result`, `GET /stats`. Auto-Reconciliation Engine has deprecated manual logging; `AGENT6_TOOLS` now exposes only `compute_rolling_accuracy` as read-only.

---

### AGENT 7 — Maintenance & Roadmap  *(Advisory only; never writes)*

**System Prompt:** Static roadmap (✅ #1 lineup quality shipped; OPEN #2 10-day trailing, #3 bullpen quality ← NEXT, #4 IL return penalty, #5 logistic layer, #7 decay curves; SHELVED #6 CLV flag until 200 picks ≈ mid-June 2026). Trigger alerts: 200-pick milestone; 30-pick rolling WR < 57%; static files stale > 10 days.

**Tools:** `read_tracker_stats`, `check_stale_files`.

**Success criteria:** Weekly status report with ROADMAP / KNOWN ISSUES / PERFORMANCE / NEXT ACTION blocks.

---

## 3. Orchestration Logic

**Topology:** Sequential DAG (not a state machine, not async). `orchestrator/daily_pipeline.run_daily_pipeline(force=False)`:

```
Step 1 [Python]     fetch_odds_api → HALT? → fetch_mlb_starters → fetch_weather → write_csv(games)
          ↓         (Agent 1 LLM path replaced; tool funcs called directly)
Agent 3 [Claude]    run_agent(AGENT3) → generate_ml_scores → archive_model_scores
          ↓
Step 4 [Python]     auto_grade_historical_picks → rebuild_live_predictions_log → render_report.render → write HTML
          ↓
Step 5 [Python]     send_email(subject, body, attach_report=True)
          ↓
[Accuracy]          subprocess build_nrfi_accuracy_tracker.py (non-fatal)
```

**State sharing:** plain strings. Each agent's final text is stringified and passed to the next via `run_agent(context=prior_output, user_message=...)`. No global state. Return value is `dict[agent_key] → output_string | "ERROR: ..."` plus `pipeline_status ∈ {COMPLETE, HALTED, FAILED}`.

**Checkpoints (idempotency):** `checkpoints/{DATE_KEY}_{agent_key}.json` persists `{date, status:"OK", output}`. `_load()` skips re-execution if today's checkpoint is present and status=OK. `force=True` bypasses all checkpoints.

**Retries & loops:**
- `agent_loop.MAX_TOOL_ROUNDS = 20` hard cap per agent before `AgentLoopError`.
- `anthropic.RateLimitError` → `sleep(60)` twice, then fail.
- HALT propagation: any tool returning `{"status":"HALT"}` → `PipelineHaltError` → pipeline returns `{pipeline_status:"HALTED"}` without running downstream stages.
- Scoring/report/email failures are caught and recorded as `ERROR: …` but do NOT halt the chain (email can still fire a partial slate).

---

## 4. Communication Contracts

### 4.1 Tool call contract (Anthropic SDK native)
```json
// Claude emits
{"type":"tool_use","id":"toolu_...","name":"<tool>","input":{...}}
// Harness replies
{"type":"tool_result","tool_use_id":"toolu_...","content":"<JSON string>"}
```

### 4.2 Universal tool-result envelope
```json
{ "status": "OK" | "HALT" | "ERROR", "error": "<str?>", "...": "tool-specific payload" }
```

### 4.3 `fetch_odds_api` → games list row (abridged)
```json
{
  "game_id":"<str>", "home_team":"<str>", "away_team":"<str>",
  "home_starter":"<str|null>", "away_starter":"<str|null>",
  "game_total":7.5, "dk_total":7.5, "fd_total":7.5, "f5_total":3.5,
  "retail_ml_home_odds":-140, "Retail_Implied_Prob_home":0.565,
  "pinnacle_ml_home":-135, "P_true_home":0.572,
  "retail_f5_ml_home_odds":-118, "Retail_Implied_Prob_f5_home":0.541,
  "retail_nrfi_odds":-125, "retail_yrfi_odds":+105, "Retail_Implied_Prob_nrfi":0.555,
  "pinnacle_nrfi_odds":-130, "P_true_nrfi":0.562,
  "best_over_book":"DK","best_under_book":"FD","is_coors":false,
  "game_label":"AWAY @ HOME (Away_Last vs Home_Last)"
}
```

### 4.4 `generate_ml_scores` → pick row (in model_scores.csv + actionable_picks[])
```json
{
  "game":"<label>", "home_team":"<abbr>", "away_team":"<abbr>",
  "model":"ML|Totals|Runline|F5|NRFI",
  "bet_type":"<market desc | NRFI | YRFI>",
  "pick_direction":"home|away|OVER|UNDER|NRFI|YRFI",
  "model_prob":0.564, "P_true":0.552, "Retail_Implied_Prob":0.521,
  "edge":0.043, "retail_american_odds":-118,
  "tier":1, "actionable":true, "dollar_stake":27
}
```

### 4.5 `compute_rolling_accuracy` → tear-sheet payload
```json
{"windows":{
  "last_7":   {"overall":{"bets":N,"win_pct":X,"roi_pct":Y,"pl":Z},
               "by_market":{"ML":{...},"Totals":{...},"Runline":{...},"F5":{...},"NRFI":{...}}},
  "last_28":  {...},
  "ytd_2026": {...}
}}
```

### 4.6 Checkpoint file
```json
{"date":"2026-04-22","status":"OK","output":"<agent final text>"}
```

### 4.7 Tracker server endpoints
- `POST /log_bet   {game,model,bet_type,market_line,book,units,model_prob}` → 200 `{id}`
- `POST /log_result {bet_id,result:"WIN|LOSS|PUSH",actual_total?}` → 200
- `GET  /stats` → `{n_total,wins,losses,pushes,win_pct,units_pl,roi,per_model:{...}}`

---

## 5. Heuristic & Model Logic

**Inference path (inside `generate_ml_scores`)**
1. Load `games.csv` (odds + starters).
2. Invoke four scorers via `_invoke_scorer(module_name, date_str)` which imports each `score_*_today.py` dynamically and returns a pandas DataFrame:
   - `score_ml_today`      → column `stacker_l2` (P(home win))
   - `score_run_dist_today` → `p_over_final`, `p_home_cover_final`, `lam_home`, `lam_away`, `total_line`
   - `score_f5_today`      → `stacker_l2` (P(F5 home cover))
   - `score_nrfi_today`    → `p_stk_nrfi` (P(no run in 1st))
3. Key by `(home_abbr, away_abbr)`; merge against games row; build one candidate pick per (market × game).
4. Apply **Three-Part Lock** (config/settings.py):
   - **Sanity**: `abs(model_prob − P_true) ≤ SANITY_CHECK_TOLERANCE = 0.04`; fails if P_true null.
   - **Odds floor**: `retail_american_odds ≥ ODDS_FLOOR_AMERICAN = −225`.
   - **Edge tier**: Tier 1 if `edge ≥ 0.030`; Tier 2 if `0.010 ≤ edge < 0.030`; else non-actionable.
5. **Kelly staking** (`config.settings.kelly_stake`):
   - `b = decimal_odds − 1`; `f* = (b·p − q)/b`; `raw = 2000 · f* · {0.25|0.125}`; `stake = round(min(raw, 50))`.
   - Tier 1 = Quarter-Kelly; Tier 2 = Eighth-Kelly; $2000 synthetic bankroll; $50 hard cap.
6. Persist `model_scores.csv`; return compact JSON.

**Underlying models (outside harness, in pipeline root)**
- **Bayesian hierarchical stackers** (`models/stacking_lr_*.pkl`): Level-2 logistic-regression stackers over multiple Level-1 XGBoost boosters. Markets = ML, RL, F5 (Totals uses quantile regression on `total_runs`).
- **Dual-Poisson sidecar** (shipped for F5 2026-04-21): per-fold `count:poisson` XGB boosters for home/away runs → convolution → cover probability `pois_p_cover` → appended as stacker feature. Next project: port to RL (target home_runs − away_runs ≥ 2; see memory `project_rl_poisson_sidecar.md`).
- **PrizePicks K-line features** (matrix-side via `build_feature_matrix.py` SECTION 4J): `home_sp_k_line`, `away_sp_k_line`, `sp_k_line_diff`, implied probs. MC-side weight currently 0 (Stuff+ carries the signal at current N). See memory `project_prizepicks_kprop.md`.
- **Lineup quality** (FanGraphs team wOBA vs LHP/RHP) — roadmap item #1, validated +7.3% accuracy.
- **ECE calibration:** ML 0.045, Totals 0.080 — these are the two "Alpha Markets" shown uncensored in the HTML report.

**Auto-Reconciliation Engine**
- `auto_grade_historical_picks()` grades new rows in `historical_actionable_picks.csv` against `data/statcast/actuals_2026.parquet`, writing `result ∈ {WIN,LOSS,PUSH}` and `profit_loss = _pl_on_win(stake, odds)` or `−stake`.
- `compute_rolling_accuracy()` aggregates the graded ledger into 7d / 28d / YTD × per-market matrices.
- `rebuild_live_predictions_log()` unions all archived `data/predictions/model_scores_*.csv` joined to actuals into `live_predictions_2026.csv` for the renderer's Optimal Cutoff table.

---

## 6. Project Memory

**No `CLAUDE.md` exists in the repo.** Persistent memory lives under the user's `~/.claude/projects/.../memory/`:

**`project_email_filtering.md`** — Email body is an actionable-only slice (F5 win_prob>55%, Runs with |model−Vegas|≥0.5, ML/RL-tier game cards only, no parlays). Full detail flows as HTML+PDF attachments. Why: email = quick actionable summary; attachments = full detail.

**`project_prizepicks_kprop.md`** — K-prop market lines wired as matrix features (XGBoost learns weight); `_PP_K_WEIGHT` in `monte_carlo_runline.py` pinned at 0.0 because Stuff+ carries the K signal at current N. Don't retune until n≥200 post-Stuff+ starts.

**`project_rl_poisson_sidecar.md`** — Next project: port F5 dual-Poisson sidecar (merged 2026-04-21, PR dgarcia2424/wizard-picks#1, val Brier 0.2212→0.2207) into the RL stacker in `train_xgboost.py`. Targets home_score/away_score; RL cover = P(home−away ≥ 2); save `models/pois_home.json` / `models/pois_away.json`. Scope = RL only.

**Known failure modes / architectural debt** (from Agent 7 prompt + code comments):
- `tracker_dashboard.html` not built yet — open task.
- savant_{pitchers,batters} file confusion (52KB vs 341KB) — mitigated by `validate_static_file` size guard; downstream `score_models.py` verification TBD.
- Email delivery not fully validated end-to-end.
- Runline/F5 retail lines: ingested now, but historically non-actionable markets; F5 is informational; Runline is Pinnacle-gated.
- Legacy sklearn-1.8 pickles need `LogisticRegression.multi_class="auto"` monkey-patch in `tools/implementations.py` to run under sklearn 1.7.2.
- `max_tokens` historically truncated Agent 4 HTML at 16k — resolved by deterministic `render_report.render()` replacement.
- Agent 1's LLM path is superseded by deterministic Python in `daily_pipeline.py`; the LLM definition is retained as a spec source but not exercised in production.
- CLV flag (#6) and logistic layer (#5) gated on reaching 200 picks (ETA mid-June 2026); current sample (~28–33 picks) is statistically insufficient.
- Accuracy target: 70–75% by mid-May 2026 contingent on shipping roadmap #2 (10-day trailing) + #3 (bullpen quality).

---

## 7. Invariants Gemini must preserve

1. **Market vocabulary** is closed: `{ML, Totals, Runline, F5, NRFI}`. Never introduce legacy `MFull / MF5i / MF3i / MF1i / MBat`.
2. **HALT semantics**: `{"status":"HALT"}` from any tool aborts the pipeline immediately; downstream agents must not run.
3. **Least-privilege tools**: each agent only sees its own schema list; do not cross-wire.
4. **Checkpoints are the idempotency barrier** — re-runs are safe by default; `force=True` is the only bypass.
5. **Three-Part Lock** values live in `config/settings.py` (`SANITY_CHECK_TOLERANCE=0.04`, `ODDS_FLOOR_AMERICAN=-225`, `TIER1_EDGE_THRESHOLD=0.030`, `TIER2_EDGE_THRESHOLD=0.010`). Edits here shift the entire pick universe.
6. **Kelly constants**: bankroll=$2000, Tier1=0.25, Tier2=0.125, cap=$50, integer stakes only.
7. **Report/email are deterministic Python** in production; the LLM-driven Agent 4/Agent 5 definitions are reference specs, not the runtime path.
