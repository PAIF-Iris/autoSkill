"""
tool_writer.py — instructs the LLM to produce a validated Python function.

Design decisions:
  - We ask for JSON output (not raw Python) because JSON is easier to parse
    reliably than code wrapped in markdown fences.
  - The system prompt enforces four hard rules: stdlib-only, deterministic,
    JSON-serializable return, input validation.  These are prerequisites for
    the validator to work correctly.
  - We do a fast AST syntax-check here before handing off to the validator.
    This catches the most obvious LLM mistakes cheaply without running code.
  - The function name must be snake_case because we use it programmatically
    as the FAISS index key.
"""
from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


WRITER_SYSTEM_PROMPT = """\
You are an expert Python engineer writing minimal, correct utility functions.

Hard rules — violating ANY of these causes the function to be rejected:
1. Use ONLY the Python standard library.  No pip packages.
2. The function must be deterministic: same inputs always produce same output.
3. Return a JSON-serializable value: str, int, float, bool, list, dict, or None.
4. Validate inputs: raise ValueError with a clear human-readable message on bad input.
5. No I/O, no filesystem, no network, no subprocess calls inside the function.
6. Include a one-line docstring.
7. Include type hints on all parameters and the return type.

Respond ONLY with a single JSON object — no markdown, no extra text:
{
  "name":        "<snake_case function name>",
  "description": "<one sentence: what this function does and its key parameters>",
  "code":        "<complete Python function definition, \\n for newlines>"
}
"""


@dataclass
class WrittenTool:
    name: str
    description: str
    code: str


def write_tool(query: str, llm_client) -> Optional[WrittenTool]:
    """
    Ask the LLM to write a Python function that solves `query`.

    Returns a WrittenTool on success, or None if:
      - the LLM returned malformed JSON
      - the code has a syntax error
      - required fields are missing
    """
    user_prompt = (
        f"Write a Python function to solve the following task:\n\n"
        f"{query}\n\n"
        "Return ONLY the JSON object described in the instructions."
    )

    try:
        raw = llm_client.complete(
            system=WRITER_SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=1500,
        )
    except Exception as exc:
        logger.error("LLM call failed in tool writer: %s", exc)
        return None

    return _parse_written_tool(raw)


def _parse_written_tool(raw: str) -> Optional[WrittenTool]:
    """
    Parse and validate the raw LLM response into a WrittenTool.

    Shared by tool_writer.write_tool and reviser.revise_tool so the
    JSON-parse + AST-check logic lives in exactly one place.

    Returns None on any parse, validation, or syntax error.
    """
    cleaned = re.sub(r"```(?:json)?\s*|```", "", raw).strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning(
            "_parse_written_tool: non-JSON response (%s). First 300 chars: %s",
            exc, cleaned[:300],
        )
        return None

    name        = str(payload.get("name",        "")).strip()
    description = str(payload.get("description", "")).strip()
    code        = str(payload.get("code",        "")).strip()

    if not name or not description or not code:
        logger.warning(
            "_parse_written_tool: JSON missing fields. Got keys: %s", list(payload.keys())
        )
        return None

    if not name.isidentifier():
        logger.warning("_parse_written_tool: name '%s' is not a valid Python identifier.", name)
        return None

    try:
        ast.parse(code)
    except SyntaxError as exc:
        logger.warning("_parse_written_tool: syntactically invalid code: %s", exc)
        return None

    tree = ast.parse(code)
    defined_fns = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    if name not in defined_fns:
        logger.warning(
            "_parse_written_tool: function '%s' not found in code. Defined: %s",
            name, defined_fns,
        )
        return None

    return WrittenTool(name=name, description=description, code=code)
