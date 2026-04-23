"""
orchestrator/daily_pipeline.py

Daily pipeline: Ingest → Score → Render → Email
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

from config import settings
from config.settings import FILES, PIPELINE_DIR
from render_report import render

# Environmental feature engineering (wind vectors, league RPG, interactions)
from models.feature_engineering import (
    compute_wind_vector_out,
    compute_days_since_opening_day,
    compute_league_rpg_rolling_7d,
    load_league_rpg_history,
)


class PipelineHaltError(RuntimeError):
    """Raised by Step 1 when upstream data is missing (e.g. zero games from odds API)."""
from tools.implementations import (
    archive_model_scores,
    auto_grade_historical_picks,
    fetch_odds_api,
    fetch_mlb_starters,
    fetch_weather,
    generate_ml_scores,
    rebuild_live_predictions_log,
    write_csv,
    TEAM_NAME_TO_ABBR,
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
            # Environmental features computed deterministically in Step 1.
            # These are LOGGED per game and attached to agent1_output. Downstream
            # training / inference reads weather parquets + build_feature_matrix.py;
            # this block is the production-path parity check.
            league_hist = load_league_rpg_history()
            league_rpg_7d = compute_league_rpg_rolling_7d(DATE_KEY, league_hist)
            days_since_opener = compute_days_since_opening_day(DATE_KEY)
            logger.info(
                f"[EnvFeatures] league_rpg_rolling_7d={league_rpg_7d} "
                f"days_since_opening_day={days_since_opener}"
            )
            env_features: list[dict] = []
            for g in games:
                gid  = g.get("game_id", g.get("home_team", ""))
                home = g.get("home_team", "")
                w    = weather.get(gid, {}) or {}
                if "error" in w:
                    enriched_weather.append(f"{home}: weather unavailable ({w.get('error')})")
                    env_features.append({
                        "game_id": gid, "home_team": home,
                        "wind_vector_out": None, "dew_point_f": None,
                        "temp_f": None,
                        "days_since_opening_day": days_since_opener,
                        "league_rpg_rolling_7d": league_rpg_7d,
                    })
                    continue
                t   = w.get("temperature_f")
                ws  = w.get("wind_speed_mph")
                wb  = w.get("wind_direction")
                wd  = _cardinal(wb)
                pp  = w.get("precipitation_probability")
                dp  = w.get("dew_point_f")
                pp_s = f", Precip {pp}%" if pp is not None else ""
                dp_s = f", Dew {dp}°F" if dp is not None else ""
                home_abbr = TEAM_NAME_TO_ABBR.get(home, home)
                wvo = compute_wind_vector_out(ws, wb, home_abbr)
                wvo_s = f", WindOut {wvo:+.1f}" if wvo is not None else ""
                enriched_weather.append(
                    f"{home}: {t}°F, Wind {ws}mph {wd}{wvo_s}{dp_s}{pp_s}"
                )
                env_features.append({
                    "game_id": gid, "home_team": home,
                    "wind_vector_out": wvo,
                    "dew_point_f": dp,
                    "temp_f": t,
                    "days_since_opening_day": days_since_opener,
                    "league_rpg_rolling_7d": league_rpg_7d,
                })
            for line in enriched_weather:
                logger.info(f"[Weather] {line}")

            # 5) Merge Alpha Features (env physics) into each game row so they
            # land as columns in games.csv alongside the odds API payload.
            env_by_gid = {e["game_id"]: e for e in env_features}
            alpha_cols = ("wind_vector_out", "dew_point_f", "temp_f",
                          "days_since_opening_day", "league_rpg_rolling_7d")
            wvo_count = 0
            for g in games:
                env = env_by_gid.get(g.get("game_id", g.get("home_team", "")), {})
                for col in alpha_cols:
                    g[col] = env.get(col)
                if g.get("wind_vector_out") is not None:
                    wvo_count += 1
            logger.info(
                f"[AlphaFeatures] appended to games.csv: {alpha_cols} | "
                f"wind_vector_out populated for {wvo_count}/{len(games)} games | "
                f"league_rpg_rolling_7d={league_rpg_7d} | "
                f"days_since_opening_day={days_since_opener}"
            )

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
                "env_features":     env_features,
                "league_rpg_rolling_7d":   league_rpg_7d,
                "days_since_opening_day":  days_since_opener,
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

    # ── STEP 3: Scoring Engine (deterministic Python call) ────────────────────
    agent3_output = None if force else _load("agent3")

    if agent3_output is None:
        logger.info("▶ Starting Step 3: Scoring Engine (deterministic generate_ml_scores)")
        try:
            scoring_raw = generate_ml_scores(game_date=DATE_KEY)
            try:
                scoring_payload = json.loads(scoring_raw)
            except (TypeError, json.JSONDecodeError):
                scoring_payload = {"status": "ERROR", "error": "generate_ml_scores returned non-JSON"}

            if scoring_payload.get("status") == "ERROR":
                raise RuntimeError(f"generate_ml_scores failed: {scoring_payload.get('error')}")

            agent3_output = scoring_raw if isinstance(scoring_raw, str) else json.dumps(scoring_payload)
            _save("agent3", agent3_output)
            results["agent3"] = agent3_output

            summary = (
                f"games_scored={scoring_payload.get('games_scored')} | "
                f"actionable={scoring_payload.get('actionable')} "
                f"(T1={scoring_payload.get('tier1')} / T2={scoring_payload.get('tier2')}) | "
                f"fails: sanity={scoring_payload.get('failed_sanity')}, "
                f"odds={scoring_payload.get('failed_odds_floor')}, "
                f"edge={scoring_payload.get('failed_edge')}"
            )
            logger.info(f"✅ Step 3 complete. {summary}")

            # Archive today's model_scores.csv so rolling cutoff stats can
            # grow forward without re-running the sterile backtest.
            archive_status = archive_model_scores(DATE_KEY)
            logger.info(f"[Archive] model_scores: {archive_status}")

        except Exception as e:
            logger.error(f"❌ Step 3 error: {e}", exc_info=True)
            results["agent3"] = f"ERROR: {e}"
            # Don't halt — attempt report and email with whatever was scored
    else:
        results["agent3"] = agent3_output

    # ── STEP 6: Prop Scoring (Pitching K-Prop Engine) ────────────────────────
    # Appends PROP_K rows onto model_scores.csv BEFORE the renderer + email so
    # Step 4/5 pick them up. Runs even if Step 3 was checkpointed: props are
    # cheap and market K-lines refresh through the morning.
    logger.info("▶ Starting Step 6: Prop Scoring (PROP_K)")
    try:
        from score_props_today import score_props  # project root on sys.path
        props_df = score_props(DATE_KEY)
        scores_path = PIPELINE_DIR / FILES["model_scores"]
        if not props_df.empty and scores_path.exists():
            base_df = pd.read_csv(scores_path)
            # Strip any previous PROP_K rows for this date (idempotent re-run).
            if "model" in base_df.columns:
                base_df = base_df[base_df["model"] != "PROP_K"]
            combined = pd.concat([base_df, props_df], ignore_index=True, sort=False)
            combined.to_csv(scores_path, index=False)
            n_act = int(props_df["actionable"].sum())
            logger.info(f"✅ Step 6 complete. appended {len(props_df)} PROP_K rows "
                        f"(actionable={n_act}) → {scores_path.name}")
            results["agent6"] = f"OK: {len(props_df)} props appended, {n_act} actionable"
            # Invalidate agent4 checkpoint so the renderer re-runs against the
            # updated model_scores.csv (props were appended post-Step 3).
            try:
                _ckpt_path("agent4").unlink(missing_ok=True)
            except Exception:
                pass
        else:
            logger.info(f"Step 6 skipped — props_df rows={len(props_df)}, "
                        f"scores_path exists={scores_path.exists()}")
            results["agent6"] = "SKIPPED"
    except Exception as e:
        logger.error(f"❌ Step 6 (Prop Scoring) error: {e}", exc_info=True)
        results["agent6"] = f"ERROR: {e}"

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

        def _units_from_edge(edge) -> int:
            """1-2-3 Unit scale by edge size. 3U ≥ 10%, 2U ≥ 5%, 1U ≥ 1%, else 0."""
            if edge is None or pd.isna(edge):
                return 0
            e = float(edge)
            if e >= 0.10: return 3
            if e >= 0.05: return 2
            if e >= 0.01: return 1
            return 0

        # ── Signal Correlation Audit + Risk Scale-Back ────────────────────────
        # Group actionable picks by signal_flags. If a single signal carries
        # > 5U of aggregate exposure, every pick sharing that flag is scaled
        # down by 25% — prevents one miscalibrated driver (e.g. a bad wind
        # forecast) from tanking multiple correlated bets at once.
        SIGNAL_EXPOSURE_CAP = 5.0
        SCALE_BACK_FACTOR   = 0.75

        df_act["_base_units"] = df_act["edge"].map(_units_from_edge).astype(float)
        # Flags are single-token strings ("" when none). Empty flag → no group.
        df_act["_flag"] = df_act.get("signal_flags", "").fillna("").astype(str)
        exposure = (df_act[df_act["_flag"] != ""]
                    .groupby("_flag")["_base_units"].sum().to_dict())
        scaled_flags = {f: tot for f, tot in exposure.items() if tot > SIGNAL_EXPOSURE_CAP}

        def _final_units(row) -> float:
            u = row["_base_units"]
            if row["_flag"] in scaled_flags:
                return round(u * SCALE_BACK_FACTOR, 2)
            return u

        df_act["_units_final"] = df_act.apply(_final_units, axis=1)

        def _fmt_pick(r) -> str:
            game  = r.get("game", "")
            model = r.get("model", "")
            pick  = r.get("pick_direction", "")
            btype = r.get("bet_type", "")
            odds  = r.get("retail_american_odds", "")
            stake = r.get("dollar_stake", "")
            prob  = r.get("model_prob", None)
            edge  = r.get("edge", None)
            flags = r.get("signal_flags", "") or ""
            base_u  = int(r.get("_base_units", 0))
            final_u = r.get("_units_final", base_u)
            prob_s = f"{prob*100:.1f}%" if isinstance(prob, (int, float)) else ""
            odds_s = f"{int(odds):+d}" if pd.notna(odds) else ""
            edge_s = f"{edge*100:+.1f}%" if isinstance(edge, (int, float)) and not pd.isna(edge) else ""
            flag_s = f" [{flags}]" if flags else ""
            scaled_s = f" (scaled from {base_u}U)" if final_u != base_u else ""
            units_s  = f"{final_u}U" if isinstance(final_u, float) and final_u != int(final_u) else f"{int(final_u)}U"
            return (f"  • {game} — {model} {btype} {pick}  "
                    f"| prob {prob_s} | edge {edge_s} | odds {odds_s} "
                    f"| {units_s} (${stake}){flag_s}{scaled_s}")

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
        if exposure:
            lines.append("--- SIGNAL CORRELATION AUDIT ---")
            for flag, total in sorted(exposure.items(), key=lambda x: -x[1]):
                mark = "  ⚠ SCALED 25%" if flag in scaled_flags else ""
                lines.append(f"  • {flag}: {int(total) if total == int(total) else total}U total exposure{mark}")
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

    # ── DRIFT MONITOR (Totals bias watchdog) ──────────────────────────────────
    try:
        sys.path.insert(0, str(_PROJECT_ROOT))
        from drift_monitor import compute_drift
        drift = compute_drift(window=10)
        logger.info(f"[Drift] {drift}")
        results["drift"] = str(drift)
    except Exception as e:
        logger.warning(f"⚠ Drift monitor failed (non-fatal): {e}")

    results["pipeline_status"] = "COMPLETE"
    logger.info("=" * 60)
    logger.info(f"✅ Daily pipeline complete | {TODAY}")
    logger.info("=" * 60)
    return results
