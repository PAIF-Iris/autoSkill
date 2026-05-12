"""
examples/sdk_basic.py — Basic SDK usage.

Shows the core learn-once / reuse-forever pattern.
No API key required (uses MockProvider).
For real results: python examples/sdk_basic.py --real
"""
from __future__ import annotations

import sys
import tempfile
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skill_agent import SkillAgent, AgentResult, AutoSkillConfig
from skill_agent.providers import MockProvider


def print_result(label: str, result: AgentResult) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    print(f"  answer       : {result.answer}")
    print(f"  action_taken : {result.action_taken}")
    print(f"  tool_name    : {result.tool_name}")
    print(f"  latency_ms   : {f'{result.latency_ms:.1f}' if result.latency_ms else 'N/A'}")
    if result.notes:
        for note in result.notes:
            print(f"  note         : {note}")


def main() -> None:
    use_real = "--real" in sys.argv

    # ── Build agent ───────────────────────────────────────────────────────────
    if use_real:
        # Reads ANTHROPIC_API_KEY from the environment
        cfg   = AutoSkillConfig(llm_provider="anthropic")
        agent = cfg.create_agent(db_path=tempfile.mktemp(suffix=".db"))
        print("Using Anthropic API")
    else:
        agent = SkillAgent(
            llm_client=MockProvider(),
            db_path=tempfile.mktemp(suffix=".db"),
        )
        print("Using MockProvider (no API key needed)")

    # ── Round 1: first encounter — agent writes and saves a tool ─────────────
    q1 = "What is the compound interest on $5000 at 4.2% annual rate over 7 years?"
    r1 = agent.run(q1)
    print_result("Round 1 — first encounter (tool should be created)", r1)

    # ── Round 2: same type — tool reused, no LLM call needed ─────────────────
    q2 = "Calculate compound interest: $10000 at 5% for 10 years"
    r2 = agent.run(q2)
    print_result("Round 2 — similar query (tool should be reused)", r2)

    # ── Round 3: conversational — answered directly, no tool ─────────────────
    q3 = "Can you explain what compound interest means?"
    r3 = agent.run(q3)
    print_result("Round 3 — conversational (direct LLM answer)", r3)

    # ── Feedback ──────────────────────────────────────────────────────────────
    if r2.tool_name:
        agent.feedback(r2, positive=True, comment="Correct and fast")
        print(f"\n  Feedback recorded for '{r2.tool_name}'")

    # ── Registry summary ──────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  Registry contents")
    print(f"{'─' * 60}")
    for t in agent.registry.list_tools():
        print(f"  [{t.status.upper():8}] {t.name}  "
              f"uses={t.usage_count}  rate={t.success_rate:.0%}")


if __name__ == "__main__":
    main()
