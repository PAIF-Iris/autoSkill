"""
events.py — typed event system for real-time agent observability.

Every significant step the agent takes emits an AgentEvent.  Callers opt-in
by passing on_event= to SkillAgent; when on_event is None the overhead is a
single pointer comparison per emit call (zero heap allocation).

Usage (SDK):
    def handler(event: AgentEvent) -> None:
        print(f"[{event.type.value}] {event.payload}")

    agent = SkillAgent(on_event=handler)
    agent.run("Calculate compound interest …")

Event catalogue — see EventType docstring for payload shapes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class EventType(str, Enum):
    """
    All event types emitted by SkillAgent.run() and related methods.

    Payload shapes
    ──────────────
    ROUTING_START       {}
    ROUTING_DONE        {action, reason, tool_name?, similarity?}

    TOOL_FOUND          {name, similarity, status}
    TOOL_EXECUTING      {name, kwargs}
    TOOL_EXECUTED       {name, success, latency_ms, output?, error?}
    TOOL_REVIEWED       {name, appropriate, reason}

    TOOL_WRITING        {query}
    TOOL_WRITTEN        {name?, description?, success}
    TOOL_VALIDATING     {name}
    TOOL_SAVED          {name, tool_id?}
    TOOL_DECISION       {name, code, validation_summary, passed}

    ANSWER_START        {query}
    ANSWER_CHUNK        {chunk}
    ANSWER_DONE         {answer}

    PERMISSION_REQUEST  {permissions: [{type, reason, required}]}
    MCP_REQUIRED        {dependencies: [{server_name, tool_name, reason}]}

    ERROR               {stage, message, exc_type?}
    """

    # ── Routing ───────────────────────────────────────────────────────────────
    ROUTING_START  = "routing_start"
    ROUTING_DONE   = "routing_done"

    # ── Tool-use path ─────────────────────────────────────────────────────────
    TOOL_FOUND     = "tool_found"
    TOOL_EXECUTING = "tool_executing"
    TOOL_EXECUTED  = "tool_executed"
    TOOL_REVIEWED  = "tool_reviewed"    # post-execution LLM review: {name, appropriate, reason}

    # ── Tool-create path ──────────────────────────────────────────────────────
    TOOL_WRITING   = "tool_writing"
    TOOL_WRITTEN   = "tool_written"
    TOOL_VALIDATING = "tool_validating"
    TOOL_SAVED     = "tool_saved"
    TOOL_DECISION  = "tool_decision"     # user must choose keep/revise/discard

    # ── Direct-answer path ────────────────────────────────────────────────────
    ANSWER_START   = "answer_start"
    ANSWER_CHUNK   = "answer_chunk"      # one per streamed token
    ANSWER_DONE    = "answer_done"

    # ── Security / permissions ────────────────────────────────────────────────
    PERMISSION_REQUEST = "permission_request"
    MCP_REQUIRED       = "mcp_required"

    # ── Errors ────────────────────────────────────────────────────────────────
    ERROR = "error"


@dataclass
class AgentEvent:
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)


# Convenience type alias for callers
EventHandler = Callable[[AgentEvent], None]
