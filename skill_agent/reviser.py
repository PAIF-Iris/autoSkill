"""
reviser.py — rewrites a failing tool using the LLM.

When a tool's combined health score (execution success rate + user sentiment)
drops below the degraded threshold, the agent can attempt a revision rather
than immediately retiring the tool.

Pipeline:
  1. Build a prompt from the tool's current code + degradation context
  2. Ask the LLM to produce an improved version (same JSON schema as tool_writer)
  3. Parse + AST-validate via _parse_written_tool (shared with tool_writer)
  4. Return a WrittenTool on success, or None if the LLM response is unusable

The caller (agent._attempt_revision) is responsible for:
  - Running validate_tool() on the returned WrittenTool before saving
  - Calling registry.update_tool() to snapshot the old version and save the new one
"""
from __future__ import annotations

import logging
from typing import Optional

from .tool_registry import Tool
from .tool_writer import WrittenTool, _parse_written_tool

logger = logging.getLogger(__name__)

REVISER_SYSTEM_PROMPT = """\
You are an expert Python engineer. A tool function has been flagged as degraded
due to a high failure rate or negative user feedback. You are given the original
code and asked to rewrite it to be more robust and correct.

Follow the same rules as the original writer:
1. Use ONLY the Python standard library.  No pip packages.
2. The function must be deterministic: same inputs always produce same output.
3. Return a JSON-serializable value: str, int, float, bool, list, dict, or None.
4. Validate inputs: raise ValueError with a clear human-readable message on bad input.
5. No I/O, no filesystem, no network, no subprocess calls inside the function.
6. Include a one-line docstring.
7. Include type hints on all parameters and the return type.

Respond ONLY with a single JSON object — no markdown, no extra text:
{
  "name":        "<same snake_case function name as the original>",
  "description": "<one sentence: what this function does and its key parameters>",
  "code":        "<complete revised Python function definition, \\n for newlines>"
}
"""


def revise_tool(tool: Tool, llm_client) -> Optional[WrittenTool]:
    """
    Ask the LLM to produce a revised (more robust) version of `tool`.

    Parameters
    ----------
    tool        : the degraded Tool to revise; its code and description are
                  included in the prompt so the LLM has full context
    llm_client  : any object with .complete(system, user, max_tokens) -> str

    Returns
    -------
    WrittenTool on success, or None if the response is unusable.
    The function name in the returned tool is guaranteed to match `tool.name`
    (enforced by _parse_written_tool + an extra name-consistency check here).
    """
    user_prompt = (
        f"Tool name: {tool.name}\n"
        f"Description: {tool.description}\n\n"
        f"Current code:\n{tool.code}\n\n"
        "This tool is degraded (high failure rate or negative feedback). "
        "Rewrite it to be more robust and correct, keeping the same function name."
    )

    try:
        raw = llm_client.complete(
            system=REVISER_SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=1500,
        )
    except Exception as exc:
        logger.error("LLM call failed in reviser for tool '%s': %s", tool.name, exc)
        return None

    written = _parse_written_tool(raw)
    if written is None:
        logger.warning("Reviser returned unusable response for tool '%s'.", tool.name)
        return None

    # Enforce name consistency — the revised function must keep the same name
    # so existing references (embeddings, registry entries) remain valid.
    if written.name != tool.name:
        logger.warning(
            "Reviser changed function name '%s' → '%s'; rejecting.",
            tool.name, written.name,
        )
        return None

    logger.info("Reviser produced a candidate for '%s'.", tool.name)
    return written
