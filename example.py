"""
example.py — demonstrates the SkillAgent end-to-end.

Run with the mock client (no API key needed):
    python example.py

Run with the real Anthropic API:
    ANTHROPIC_API_KEY=sk-... python example.py --real
"""
from __future__ import annotations
import sys
import logging
import pprint

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-7s %(name)s — %(message)s",
)

from skill_agent import SkillAgent, AgentResult
from skill_agent.llm import MockClient, AnthropicClient


def print_result(result: AgentResult) -> None:
    print("\n" + "=" * 60)
    print(f"  Answer       : {result.answer}")
    print(f"  Action taken : {result.action_taken}")
    print(f"  Tool name    : {result.tool_name}")
    print(f"  Validation   : {result.validation_passed}")
    print(f"  Latency      : {f'{result.latency_ms:.1f} ms' if result.latency_ms else 'N/A'}")
    if result.notes:
        print("  Notes:")
        for note in result.notes:
            print(f"    • {note}")
    print("=" * 60 + "\n")


def main() -> None:
    use_real = "--real" in sys.argv
    client = AnthropicClient() if use_real else MockClient()
    mode = "Anthropic API" if use_real else "MockClient (no API key needed)"
    print(f"\n🤖  SkillAgent demo — using {mode}\n")

    # Use a fresh in-memory-adjacent DB for the demo
    agent = SkillAgent(llm_client=client, db_path="demo_skills.db")

    # ── Round 1: novel deterministic query → should CREATE a tool ─────────────
    print("── Round 1: First encounter with compound interest query ──")
    q1 = "What is the compound interest on $5000 at 4.2% annual rate over 7 years?"
    r1 = agent.run(q1)
    print_result(r1)

    # ── Round 2: same query type → should REUSE the saved tool ───────────────
    print("── Round 2: Same query type (should reuse tool) ──")
    q2 = "Calculate compound interest: $10000, 5% rate, 10 years"
    r2 = agent.run(q2)
    print_result(r2)

    # ── Round 3: conversational query → should ANSWER directly ───────────────
    print("── Round 3: Conversational query (should answer directly) ──")
    q3 = "Can you explain the difference between simple and compound interest?"
    r3 = agent.run(q3)
    print_result(r3)

    # ── Registry inspection ───────────────────────────────────────────────────
    print("── Registry contents ──")
    tools = agent.registry.list_tools()
    if tools:
        for t in tools:
            print(
                f"  [{t.status.upper():8}] {t.name!r:45} "
                f"uses={t.usage_count}  "
                f"rate={t.success_rate:.0%}"
            )
    else:
        print("  (empty)")
    print()


if __name__ == "__main__":
    main()
