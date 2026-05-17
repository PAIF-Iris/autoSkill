"""
providers/mock_provider.py — Deterministic mock provider for tests and offline use.

No API key required. Dispatches on keywords in the system prompt to return
valid JSON responses for each stage of the agent pipeline.
"""
from __future__ import annotations

import json
from typing import Iterator

from .base import BaseProvider


class MockProvider(BaseProvider):
    """
    Deterministic fake provider for unit tests and CI.

    Mimics the JSON output expected by each pipeline stage (recognizer,
    tool_writer, validator, kwargs extractor, reviser) without any network call.
    """

    def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        # ── Recognizer ────────────────────────────────────────────────────────
        if "routing layer" in system:
            explain_words = ["explain", "what is", "what does", "describe",
                             "tell me about", "difference"]
            if any(kw in user.lower() for kw in explain_words):
                return json.dumps({"action": "answer", "reason": "Explanatory query."})
            calc_words = ["calculate", "convert", "compound interest on $",
                          "what is the", "how much"]
            if any(kw in user.lower() for kw in calc_words):
                return json.dumps({"action": "create_tool",
                                   "reason": "Deterministic calculation."})
            return json.dumps({"action": "answer", "reason": "Conversational query."})

        # ── Tool writer ───────────────────────────────────────────────────────
        if "Python engineer" in system:
            code = (
                "def calculate_compound_interest("
                "principal: float, rate: float, years: int) -> float:\n"
                "    \"\"\"Return principal grown by compound interest.\"\"\"\n"
                "    if principal < 0 or rate < 0 or years < 0:\n"
                "        raise ValueError('All inputs must be non-negative')\n"
                "    return round(principal * (1 + rate) ** years, 2)\n"
            )
            return json.dumps({
                "name": "calculate_compound_interest",
                "description": (
                    "Calculates compound interest given principal, "
                    "annual rate fraction, and years."
                ),
                "code": code,
            })

        # ── Test generator ────────────────────────────────────────────────────
        if "testing expert" in system:
            return json.dumps([
                {"kwargs": {"principal": 1000.0, "rate": 0.05, "years": 1},
                 "expected": 1050.0},
                {"kwargs": {"principal": 1000.0, "rate": 0.10, "years": 2},
                 "expected": 1210.0},
                {"kwargs": {"principal": 500.0,  "rate": 0.0,  "years": 10},
                 "expected": 500.0},
                {"kwargs": {"principal": 0.0,    "rate": 0.05, "years": 5},
                 "expected": 0.0},
            ])

        # ── kwargs extractor ──────────────────────────────────────────────────
        if "Extract the" in system and "function arguments" in system:
            return json.dumps({"principal": 5000.0, "rate": 0.042, "years": 7})

        # ── Reviser ───────────────────────────────────────────────────────────
        if "degraded" in system:
            code = (
                "def calculate_compound_interest("
                "principal: float, rate: float, years: int) -> float:\n"
                "    \"\"\"Return principal grown by compound interest (revised).\"\"\"\n"
                "    if not isinstance(principal, (int, float)) or principal < 0:\n"
                "        raise ValueError('principal must be non-negative')\n"
                "    if not isinstance(rate, (int, float)) or rate < 0:\n"
                "        raise ValueError('rate must be non-negative')\n"
                "    if not isinstance(years, int) or years < 0:\n"
                "        raise ValueError('years must be a non-negative integer')\n"
                "    return round(principal * (1 + rate) ** years, 2)\n"
            )
            return json.dumps({
                "name": "calculate_compound_interest",
                "description": (
                    "Calculates compound interest given principal, "
                    "annual rate fraction, and years (revised)."
                ),
                "code": code,
            })

        # ── Post-execution review ─────────────────────────────────────────────
        if "verifying whether" in system:
            return json.dumps({"appropriate": True, "reason": "Output matches query."})

        # ── Direct answer fallback ────────────────────────────────────────────
        return "I can help with that directly."

    def stream(self, system: str, user: str, max_tokens: int = 1000) -> Iterator[str]:
        yield self.complete(system, user, max_tokens)
