"""
orchestrator/agent_loop.py

The agentic loop that replaces CrewAI's framework.
Each agent is: system_prompt + tools + this loop.

Flow:
  1. Call Claude with system prompt, tools, and messages
  2. If stop_reason == "tool_use" → execute each tool, feed results back, repeat
  3. If stop_reason == "end_turn"  → return final text
  4. On HALT signal in tool result  → raise PipelineHaltError immediately
"""
from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from config.settings import ANTHROPIC_API_KEY, CLAUDE_MODEL

logger = logging.getLogger("wizard.loop")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

MAX_TOOL_ROUNDS = 20  # Safety cap — prevents infinite loops
MAX_TOKENS_DEFAULT = 8096  # Raised from 4096 — agents need room for large tool outputs


class PipelineHaltError(Exception):
    """Raised when an agent signals a hard HALT (e.g. Odds API returns no lines)."""
    pass


class AgentLoopError(Exception):
    """Raised on unexpected loop failures."""
    pass


def run_agent(
    system_prompt: str,
    tools: list[dict],
    tool_executor: callable,
    user_message: str,
    context: str = "",
    agent_name: str = "Agent",
    max_tokens: int = 8096,
) -> str:
    """
    Run a single Claude agent to completion.

    Args:
        system_prompt:   The agent's identity, goals, and rules.
        tools:           List of tool schema dicts (name, description, input_schema).
        tool_executor:   Callable(tool_name, tool_input) → str (JSON result).
        user_message:    The task instruction for this agent.
        context:         Optional output from a prior agent to prepend as context.
        agent_name:      For logging only.
        max_tokens:      Max tokens per API call.

    Returns:
        The agent's final text output.

    Raises:
        PipelineHaltError: If any tool result contains status == "HALT".
        AgentLoopError:    On unexpected failures.
    """
    # Prepend prior agent context if provided
    full_message = f"{context}\n\n---\n{user_message}".strip() if context else user_message

    messages: list[dict] = [
        {"role": "user", "content": full_message}
    ]

    logger.info(f"[{agent_name}] Starting. Max tool rounds: {MAX_TOOL_ROUNDS}")

    for round_num in range(1, MAX_TOOL_ROUNDS + 1):
        logger.info(f"[{agent_name}] Round {round_num} — calling Claude API")

        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=max_tokens,
                system=system_prompt,
                tools=tools if tools else anthropic.NOT_GIVEN,
                messages=messages,
            )
        except anthropic.RateLimitError as e:
            # Rate limit — wait 60s and retry once before failing
            logger.warning(f"[{agent_name}] Rate limit hit — waiting 60s before retry...")
            import time; time.sleep(60)
            try:
                response = client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    tools=tools if tools else anthropic.NOT_GIVEN,
                    messages=messages,
                )
            except anthropic.RateLimitError as e2:
                logger.warning(f"[{agent_name}] Rate limit hit again — waiting 60s...")
                time.sleep(60)
                response = client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    tools=tools if tools else anthropic.NOT_GIVEN,
                    messages=messages,
                )
        except anthropic.APIError as e:
            raise AgentLoopError(f"[{agent_name}] Anthropic API error: {e}") from e

        logger.info(f"[{agent_name}] stop_reason={response.stop_reason}")

        # ── Terminal: agent finished ──────────────────────────────────────────
        if response.stop_reason in ("end_turn", "max_tokens"):
            final_text = _extract_text(response.content)
            if response.stop_reason == "max_tokens":
                logger.warning(f"[{agent_name}] Hit max_tokens — returning partial output. Consider increasing max_tokens.")
            logger.info(f"[{agent_name}] Completed. Output length: {len(final_text)} chars")
            return final_text

        # ── Tool use round ────────────────────────────────────────────────────
        if response.stop_reason == "tool_use":
            # Append assistant's response (with tool_use blocks) to history
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name  = block.name
                tool_input = block.input
                tool_id    = block.id

                logger.info(f"[{agent_name}] Tool call: {tool_name}({json.dumps(tool_input)[:200]})")

                try:
                    raw_result = tool_executor(tool_name, tool_input)
                except Exception as e:
                    raw_result = json.dumps({
                        "status": "ERROR",
                        "error": f"Tool execution failed: {e}"
                    })

                # ── HALT detection ────────────────────────────────────────────
                try:
                    parsed = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
                    if isinstance(parsed, dict) and parsed.get("status") == "HALT":
                        raise PipelineHaltError(
                            f"[{agent_name}] HALT signaled by {tool_name}: {parsed.get('error', raw_result)}"
                        )
                except (json.JSONDecodeError, TypeError):
                    pass  # Non-JSON result — not a HALT

                logger.info(f"[{agent_name}] Tool result: {str(raw_result)[:300]}")

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tool_id,
                    "content":     raw_result if isinstance(raw_result, str) else json.dumps(raw_result),
                })

            # Feed all tool results back as a user turn
            messages.append({"role": "user", "content": tool_results})
            continue

        # ── Unexpected stop reason ────────────────────────────────────────────
        raise AgentLoopError(
            f"[{agent_name}] Unexpected stop_reason: {response.stop_reason}"
        )

    raise AgentLoopError(
        f"[{agent_name}] Exceeded {MAX_TOOL_ROUNDS} tool rounds without completing. "
        "Check for tool errors or infinite loops."
    )


def _extract_text(content: list) -> str:
    """Extract all text blocks from a response content list."""
    parts = []
    for block in content:
        if hasattr(block, "type") and block.type == "text":
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block["text"])
    return "\n".join(parts).strip()
