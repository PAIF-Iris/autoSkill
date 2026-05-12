"""
examples/sdk_streaming.py — Real-time streaming with event callbacks.

Shows how to use on_event= to stream answer chunks to the terminal and
observe every routing decision the agent makes.

Requires ANTHROPIC_API_KEY (or swap in OllamaProvider for fully local use).
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skill_agent import SkillAgent, AgentEvent, EventType

# ── Event handler ─────────────────────────────────────────────────────────────

def on_event(event: AgentEvent) -> None:
    t = event.type

    if t == EventType.ROUTING_START:
        print("\n[routing]  analysing query…", flush=True)

    elif t == EventType.ROUTING_DONE:
        action = event.payload.get("action")
        reason = event.payload.get("reason", "")
        tool   = event.payload.get("tool_name")
        sim    = event.payload.get("similarity")
        parts  = [f"[routing]  → {action}"]
        if tool:
            parts.append(f"'{tool}'")
        if sim is not None:
            parts.append(f"(sim={sim:.2f})")
        print("  ".join(parts))
        if reason:
            print(f"           {reason}")

    elif t == EventType.TOOL_WRITING:
        attempt = event.payload.get("attempt", 1)
        suffix  = f" (attempt {attempt})" if attempt > 1 else ""
        print(f"[writing]  generating tool{suffix}…", end="\r", flush=True)

    elif t == EventType.TOOL_WRITTEN:
        if event.payload.get("success"):
            print(f"[writing]  tool '{event.payload['name']}' written      ")

    elif t == EventType.TOOL_VALIDATING:
        print(f"[validate] checking '{event.payload['name']}'…",
              end="\r", flush=True)

    elif t == EventType.TOOL_SAVED:
        print(f"[saved]    '{event.payload['name']}'  id={event.payload['tool_id']}  ")

    elif t == EventType.TOOL_EXECUTING:
        print(f"[exec]     running '{event.payload['name']}'…",
              end="\r", flush=True)

    elif t == EventType.TOOL_EXECUTED:
        mark    = "✓" if event.payload.get("success") else "✗"
        latency = event.payload.get("latency_ms", 0)
        print(f"[exec]  {mark} done in {latency:.0f} ms                    ")

    elif t == EventType.ANSWER_START:
        print("[answer]   ", end="", flush=True)

    elif t == EventType.ANSWER_CHUNK:
        sys.stdout.write(event.payload.get("chunk", ""))
        sys.stdout.flush()

    elif t == EventType.ANSWER_DONE:
        print()   # newline after streaming

    elif t == EventType.ERROR:
        print(f"[error]    {event.payload.get('stage')}: "
              f"{event.payload.get('message')}", file=sys.stderr)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Set ANTHROPIC_API_KEY to run this example.")
        print("For a local alternative, replace AnthropicProvider with OllamaProvider:")
        print("  from skill_agent.providers import OllamaProvider")
        print("  agent = SkillAgent(llm_client=OllamaProvider('llama3'), on_event=on_event)")
        sys.exit(1)

    agent = SkillAgent(
        llm="anthropic",
        db_path=tempfile.mktemp(suffix=".db"),
        on_event=on_event,
        tool_decision=None,   # auto-keep generated tools
    )

    queries = [
        "What is 12% compound interest on $5000 over 7 years?",
        "Calculate compound interest: $20000 at 3.5% for 15 years",
        "What is the difference between APR and APY?",
    ]

    for q in queries:
        print(f"\n{'═' * 60}")
        print(f"Query: {q}")
        print('═' * 60)
        result = agent.run(q)
        print(f"\n  action : {result.action_taken}")
        if result.tool_name:
            print(f"  tool   : {result.tool_name}")


if __name__ == "__main__":
    main()
