"""
recognizer.py — decides how to handle a user query.

Decision tree:
  1. Embed query → search tool registry (vector similarity)
  2. similarity >= HIGH_THRESHOLD  → "use_tool"   (skip LLM call entirely)
  3. similarity >= MID_THRESHOLD   → LLM confirms whether the candidate tool applies
  4. No good match                 → LLM classifies: is this task deterministic?
       → yes: "create_tool"
       → no:  "answer"

Design rationale:
  - We avoid creating tools on first encounter.  The LLM classifier has a high
    bar for "create_tool": the query must be clearly deterministic, repeatable,
    and well-defined.  Vague or conversational queries always get "answer".
  - The two-tier similarity check means we only call the LLM when genuinely
    uncertain, saving latency and cost on warm cache hits.
  - On any LLM failure we default to "answer" — the safest fallback.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal, Optional, List

from .tool_registry import ToolRegistry, RetrievalResult

logger = logging.getLogger(__name__)

Action = Literal["answer", "use_tool", "create_tool"]

# Thresholds for cosine similarity (vectors are unit-norm, range 0–1)
HIGH_SIMILARITY = 0.88      # Confident reuse — skip LLM classification
MID_SIMILARITY  = 0.70      # Plausible match — ask LLM to confirm
MIN_SIMILARITY  = 0.50      # Floor — ignore anything below this


CLASSIFIER_SYSTEM = """\
You are a routing layer for an AI agent that manages reusable Python tools.

Given a user query and (optionally) a candidate tool from the registry, decide:

  "use_tool"    — the candidate tool clearly and completely solves this query.
  "create_tool" — no tool exists, but the query is deterministic, well-defined,
                  and repeatable enough that a Python function would solve it reliably
                  (e.g. math, unit conversion, string formatting, date arithmetic,
                  structured data parsing).
  "answer"      — the query is conversational, subjective, explanatory, creative,
                  or too ambiguous for a reusable function.

Be conservative: prefer "answer" over "create_tool" when unsure.

Respond ONLY with valid JSON — no markdown, no commentary:
{"action": "use_tool"|"create_tool"|"answer", "reason": "<one sentence>"}
"""


@dataclass
class RecognitionResult:
    action: Action
    best_match: Optional[RetrievalResult]   # set when action is "use_tool"
    reason: str


def recognize(
    query: str,
    registry: ToolRegistry,
    llm_client,
) -> RecognitionResult:
    """
    Determine the best action for `query`.

    Parameters
    ----------
    query       : raw user query string
    registry    : ToolRegistry instance (searched first, before LLM)
    llm_client  : any object with .complete(system, user, max_tokens) -> str

    Returns
    -------
    RecognitionResult with action + best_match + reason
    """
    # ── Step 1: vector search ─────────────────────────────────────────────────
    candidates: List[RetrievalResult] = registry.search(
        query, top_k=3, min_similarity=MIN_SIMILARITY
    )
    best = candidates[0] if candidates else None

    # ── DEBUG ─────────────────────────────────────────────────────────────────
    print(f"\n[recognizer] query: {query!r}")
    print(f"[recognizer] thresholds: HIGH={HIGH_SIMILARITY} MID={MID_SIMILARITY} MIN={MIN_SIMILARITY}")
    if candidates:
        for r in candidates:
            tier = ("HIGH" if r.similarity >= HIGH_SIMILARITY
                    else "MID" if r.similarity >= MID_SIMILARITY
                    else "LOW")
            print(f"[recognizer] candidate [{tier}] sim={r.similarity:.4f} "
                  f"name={r.tool.name!r} status={r.tool.status}")
    else:
        print("[recognizer] no candidates above MIN_SIMILARITY")
    # ── END DEBUG ─────────────────────────────────────────────────────────────

    # ── Step 2: high-confidence reuse — no LLM needed ────────────────────────
    if best and best.similarity >= HIGH_SIMILARITY:
        print(f"[recognizer] → use_tool (HIGH confidence, no LLM call)")
        return RecognitionResult(
            action="use_tool",
            best_match=best,
            reason=(
                f"Strong similarity ({best.similarity:.2f}) with "
                f"'{best.tool.name}' — reusing without LLM call."
            ),
        )

    # ── Step 3: build context for LLM classifier ──────────────────────────────
    # Show the best candidate to the LLM at any similarity above MIN so it can
    # decide whether to reuse it. Below MID we label it "low confidence" so the
    # LLM knows to be more skeptical, but it still gets to see the candidate.
    if best:
        confidence = "high" if best.similarity >= MID_SIMILARITY else "low"
        print(f"[recognizer] → asking LLM ({confidence} confidence, candidate presented)")
        candidate_block = (
            f"Candidate tool in registry ({confidence} confidence match):\n"
            f"  name:        {best.tool.name}\n"
            f"  description: {best.tool.description}\n"
            f"  status:      {best.tool.status}\n"
            f"  similarity:  {best.similarity:.2f}"
        )
    else:
        print(f"[recognizer] → asking LLM (no usable candidate)")
        candidate_block = "No sufficiently similar tool found in registry."

    user_prompt = (
        f"User query: {query}\n\n"
        f"{candidate_block}\n\n"
        "Decide the action."
    )

    # ── Step 4: LLM classification ────────────────────────────────────────────
    try:
        raw = llm_client.complete(
            system=CLASSIFIER_SYSTEM,
            user=user_prompt,
            max_tokens=120,
        )
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        payload = json.loads(cleaned)
        action: Action = payload.get("action", "answer")
        reason: str = payload.get("reason", "")
        print(f"[recognizer] LLM decided: {action!r} — {reason}")

        if action not in ("answer", "use_tool", "create_tool"):
            logger.warning("Unrecognised action '%s' from LLM — defaulting to 'answer'", action)
            action = "answer"

    except Exception as exc:
        logger.warning("Recognizer LLM call failed (%s) — defaulting to 'answer'", exc)
        action = "answer"
        reason = f"LLM call failed ({exc}); defaulting to safe direct answer."

    # Attach the candidate tool only when the LLM confirmed "use_tool"
    matched = best if (action == "use_tool" and best) else None

    return RecognitionResult(action=action, best_match=matched, reason=reason)
