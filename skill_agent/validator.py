"""
validator.py — validates a generated tool before saving it to the registry.

Pipeline:
  1. Ask LLM to generate 2–5 test cases as JSON
  2. Execute each test case via the sandbox (executor.py)
  3. Compare actual output to expected output
  4. A tool must pass ALL tests to be saved

Design decisions:
  - We require 100% pass rate (not "majority").  A tool with even one failing
    test case has unknown correctness — don't save it.
  - Numeric comparison uses math.isclose() with a small tolerance to handle
    floating-point rounding differences between the test author (LLM) and
    the runtime.
  - If the LLM cannot generate test cases (ambiguous function, bad output),
    we conservatively reject the tool.  Better to answer directly than to
    save an untested function.
  - We cap at 5 test cases: enough for meaningful coverage, cheap enough
    not to slow down the interactive experience.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .executor import execute_tool

logger = logging.getLogger(__name__)

MAX_TEST_CASES = 5


TESTER_SYSTEM_PROMPT = """\
You are a Python testing expert.

Given a function's name, description, and code, generate between 2 and 5 test cases
that together provide meaningful coverage: normal inputs, edge cases, and boundary values.

Rules:
- "kwargs" must be a dict whose keys exactly match the function's parameter names.
- "expected" must be the exact JSON-serializable return value.
- Do NOT include tests that depend on randomness, the current time, or external state.
- If the function raises ValueError for invalid input, do NOT include those as test cases
  (we test only the happy path and expected edge cases here).

Respond ONLY with a JSON array — no markdown, no extra text:
[
  {"kwargs": {"param": value, ...}, "expected": <return value>},
  ...
]
"""


@dataclass
class TestCase:
    kwargs: dict
    expected: Any


@dataclass
class ValidationResult:
    passed: bool
    total: int
    passed_count: int
    failures: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.passed_count}/{self.total} tests passed"


# ── Test-case generation ───────────────────────────────────────────────────────

def _generate_test_cases(
    fn_name: str,
    description: str,
    code: str,
    llm_client,
) -> Optional[List[TestCase]]:
    """
    Ask the LLM to produce test cases for `fn_name`.
    Returns None if the LLM response is unusable.
    """
    prompt = (
        f"Function name: {fn_name}\n"
        f"Description:   {description}\n\n"
        f"Code:\n{code}\n\n"
        "Generate test cases."
    )

    try:
        raw = llm_client.complete(
            system=TESTER_SYSTEM_PROMPT,
            user=prompt,
            max_tokens=800,
        )
    except Exception as exc:
        logger.error("LLM call failed during test generation: %s", exc)
        return None

    cleaned = re.sub(r"```(?:json)?\s*|```", "", raw).strip()

    try:
        raw_cases = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Test generator returned non-JSON (%s). First 300 chars: %s",
            exc, cleaned[:300],
        )
        return None

    if not isinstance(raw_cases, list) or len(raw_cases) == 0:
        logger.warning("Test generator returned empty or non-list: %r", raw_cases)
        return None

    cases: List[TestCase] = []
    for item in raw_cases[:MAX_TEST_CASES]:
        if (
            isinstance(item, dict)
            and "kwargs" in item
            and "expected" in item
            and isinstance(item["kwargs"], dict)
        ):
            cases.append(TestCase(kwargs=item["kwargs"], expected=item["expected"]))
        else:
            logger.debug("Skipping malformed test case: %r", item)

    return cases if cases else None


# ── Equality check ────────────────────────────────────────────────────────────

def _equal(actual: Any, expected: Any, rel_tol: float = 1e-6) -> bool:
    """
    Equality check with numeric tolerance.
    Handles int/float comparisons that differ only due to floating-point rounding.
    """
    if actual == expected:
        return True
    try:
        return math.isclose(float(actual), float(expected), rel_tol=rel_tol, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def validate_tool(
    fn_name: str,
    description: str,
    code: str,
    llm_client,
) -> ValidationResult:
    """
    Full validation pipeline for a generated tool.

    Steps:
      1. Generate test cases via LLM.
      2. Execute each test case in the subprocess sandbox.
      3. Assert actual == expected (with numeric tolerance).

    A tool PASSES only if every single test passes.
    """
    test_cases = _generate_test_cases(fn_name, description, code, llm_client)

    if not test_cases:
        return ValidationResult(
            passed=False,
            total=0,
            passed_count=0,
            failures=["Could not generate test cases — rejecting tool conservatively."],
        )

    failures: List[str] = []
    passed_count = 0

    for i, tc in enumerate(test_cases):
        result = execute_tool(code, fn_name, tc.kwargs)

        if not result.success:
            failures.append(
                f"Test {i+1}: execution error — {result.error!r} "
                f"(kwargs={tc.kwargs})"
            )
            continue

        if _equal(result.output, tc.expected):
            passed_count += 1
            logger.debug(
                "Test %d PASS: %s(**%s) = %r (%.1f ms)",
                i + 1, fn_name, tc.kwargs, result.output, result.latency_ms,
            )
        else:
            failures.append(
                f"Test {i+1}: expected {tc.expected!r}, "
                f"got {result.output!r} "
                f"(kwargs={tc.kwargs})"
            )
            logger.debug(
                "Test %d FAIL: expected %r, got %r", i + 1, tc.expected, result.output
            )

    all_passed = passed_count == len(test_cases)

    return ValidationResult(
        passed=all_passed,
        total=len(test_cases),
        passed_count=passed_count,
        failures=failures,
    )
