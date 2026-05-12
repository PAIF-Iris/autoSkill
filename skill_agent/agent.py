"""
agent.py — SkillAgent: the top-level orchestrator.

Entry point:
    agent = SkillAgent()
    result = agent.run("What is 12% compound interest on $5000 over 7 years?")

Flow:
  recognize()
    ├─ "answer"       → _answer_directly()
    ├─ "use_tool"     → _use_existing_tool()
    └─ "create_tool"  → _create_and_use_tool()

New in this version:
  on_event=   — real-time event stream (AgentEvent callbacks)
  tool_decision= — hook to let user choose keep / revise / discard
  llm=        — shorthand for provider string ("anthropic" | "openai")
  use_docker= — run tools in a Docker container instead of subprocess
  permission_handler= — approve/deny filesystem/network permission requests

All paths return an AgentResult.  The agent never raises to the caller.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Tuple

from .events import AgentEvent, EventType, EventHandler
from .recognizer import recognize, RecognitionResult
from .tool_registry import ToolRegistry, Tool
from .tool_writer import write_tool
from .validator import validate_tool
from .executor import execute_tool
from .embeddings import embed
from .reviser import revise_tool
from .permissions import scan_permissions, GrantedPermissions

logger = logging.getLogger(__name__)

DIRECT_ANSWER_SYSTEM = (
    "You are a helpful, concise assistant. "
    "Answer the user's question directly and accurately."
)

KWARGS_EXTRACT_SYSTEM = (
    "Extract the exact keyword arguments for the given Python function from the user query. "
    "Keys must exactly match the function's parameter names. "
    "Values must be the correct Python types (int, float, str, etc). "
    "Respond ONLY with a JSON object — no markdown, no commentary."
)

MAX_WRITE_ATTEMPTS = 3  # initial write + up to 2 revisions via tool_decision


@dataclass
class AgentResult:
    """Structured result returned for every query."""
    answer: Any                         # final answer to the user
    action_taken: str                   # see constants below
    tool_name: Optional[str]            # name of tool used/created (if any)
    validation_passed: Optional[bool]   # None if no validation ran
    latency_ms: Optional[float]         # tool execution latency (None for direct answers)
    notes: list[str] = field(default_factory=list)  # audit trail / debug info


# ── action_taken constants ────────────────────────────────────────────────────
A_DIRECT             = "answered_directly"
A_USED_TOOL          = "used_tool"
A_USED_TOOL_FALLBACK = "used_tool_then_answered_directly"
A_CREATED_AND_USED   = "created_and_used_tool"
A_CREATED_FALLBACK   = "created_tool_then_answered_directly"

# ── Tool decision return values ───────────────────────────────────────────────
KEEP    = "keep"
REVISE  = "revise"
DISCARD = "discard"


class SkillAgent:
    """
    Self-improving tool-learning agent with real-time event streaming.

    Parameters
    ----------
    llm_client : optional
        Any object implementing .complete(system, user, max_tokens) -> str.
        Optionally also .stream(system, user, max_tokens) -> Iterator[str].
        Mutually exclusive with `llm=`.
    llm : str, optional
        Provider shorthand: "anthropic" (default) or "openai".
        Requires the corresponding package to be installed.
    llm_model : str, optional
        Model name override (e.g. "gpt-4o", "claude-opus-4-6").
    llm_api_key : str, optional
        API key override. Falls back to environment variables.
    db_path : str
        Path for the SQLite database file.
    on_event : Callable[[AgentEvent], None], optional
        Called for every agent event.  None = no overhead.
    tool_decision : Callable[[str, str, ValidationResult], Tuple[str, str]], optional
        Called when a tool passes validation, before saving.
        Arguments: (tool_name, code, validation_result)
        Returns: ("keep"|"revise"|"discard", revision_notes)
        None = auto-keep on pass, auto-discard on fail (existing behaviour).
    use_docker : bool
        Run tool code in a Docker container instead of a plain subprocess.
        Requires Docker CLI available on PATH.
    permission_handler : Callable[[list[PermissionRequest]], GrantedPermissions], optional
        Called when a tool's code requires permissions (filesystem, network…).
        None = deny all non-stdlib permissions (tools run with --network none).
    """

    def __init__(
        self,
        llm_client=None,
        db_path: str = "skills.db",
        # ── new parameters ────────────────────────────────────────────────
        llm: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        on_event: Optional[EventHandler] = None,
        tool_decision: Optional[Callable] = None,
        use_docker: bool = False,
        permission_handler: Optional[Callable] = None,
    ):
        if llm_client is not None and llm is not None:
            raise ValueError(
                "Pass either llm_client= (an LLM object) or llm= (a provider string), not both."
            )

        if llm_client is not None:
            self.llm = llm_client
        elif llm is not None:
            from .llm import create_llm
            self.llm = create_llm(provider=llm, model=llm_model, api_key=llm_api_key)
        else:
            from .llm import AnthropicClient
            self.llm = AnthropicClient()

        self.registry = ToolRegistry(db_path=db_path)
        self._on_event = on_event
        self._tool_decision = tool_decision
        self._use_docker = use_docker
        self._permission_handler = permission_handler

        tool_count = len(self.registry.list_tools())
        logger.info("SkillAgent ready. Tools in registry: %d", tool_count)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, query: str) -> AgentResult:
        """
        Process a user query end-to-end.
        Always returns an AgentResult; never raises.
        """
        logger.info("── New query ──────────────────────────────────")
        logger.info("Query: %s", query)

        self._emit(EventType.ROUTING_START, {})

        try:
            recognition = recognize(query, self.registry, self.llm)
        except Exception as exc:
            logger.error("Recognizer crashed: %s — falling back to direct answer", exc)
            self._emit(EventType.ERROR, {"stage": "routing", "message": str(exc), "exc_type": type(exc).__name__})
            return self._answer_directly(query, note=f"Recognizer error: {exc}")

        self._emit(EventType.ROUTING_DONE, {
            "action":    recognition.action,
            "reason":    recognition.reason,
            "tool_name": recognition.best_match.tool.name if recognition.best_match else None,
            "similarity": recognition.best_match.similarity if recognition.best_match else None,
        })
        logger.info("Decision: %s | %s", recognition.action, recognition.reason)

        if recognition.action == "use_tool":
            return self._use_existing_tool(query, recognition)
        if recognition.action == "create_tool":
            return self._create_and_use_tool(query)
        return self._answer_directly(query)

    def feedback(
        self, result: AgentResult, positive: bool, comment: str = ""
    ) -> None:
        """
        Record user satisfaction with the answer produced by `result`.

        Negative feedback on a degraded tool triggers auto-revision.
        """
        if result.tool_name is None:
            logger.debug("feedback() called on a direct-answer result; ignoring.")
            return

        tool = self.registry.get_tool_by_name(result.tool_name)
        if tool is None or tool.tool_id is None:
            logger.warning("feedback(): tool '%s' not found.", result.tool_name)
            return

        self.registry.save_feedback(tool.tool_id, positive, comment)
        logger.info("Feedback for '%s': %s", tool.name, "+" if positive else "-")

        if not positive:
            refreshed = self.registry.get_tool_by_id(tool.tool_id)
            if refreshed and refreshed.status == "degraded":
                self._attempt_revision(refreshed)

    # ── Private handlers ──────────────────────────────────────────────────────

    def _answer_directly(self, query: str, note: str = "") -> AgentResult:
        """No-tool path — stream the LLM response directly."""
        self._emit(EventType.ANSWER_START, {"query": query})
        try:
            answer = self._llm_stream(
                system=DIRECT_ANSWER_SYSTEM,
                user=query,
                max_tokens=1000,
            )
        except Exception as exc:
            answer = f"[Error generating answer: {exc}]"
            self._emit(EventType.ERROR, {"stage": "answer", "message": str(exc)})

        self._emit(EventType.ANSWER_DONE, {"answer": answer})
        notes = [note] if note else []
        return AgentResult(
            answer=answer,
            action_taken=A_DIRECT,
            tool_name=None,
            validation_passed=None,
            latency_ms=None,
            notes=notes,
        )

    def _use_existing_tool(
        self, query: str, recognition: RecognitionResult
    ) -> AgentResult:
        """Execute the best matching tool from the registry."""
        tool = recognition.best_match.tool
        notes = [
            f"Using tool '{tool.name}' "
            f"(similarity={recognition.best_match.similarity:.2f}, "
            f"status={tool.status})"
        ]
        self._emit(EventType.TOOL_FOUND, {
            "name": tool.name,
            "similarity": recognition.best_match.similarity,
            "status": tool.status,
        })

        kwargs = self._extract_kwargs(query, tool.code, tool.name)

        # Permission scan before execution
        granted = self._handle_permissions(tool.code)

        self._emit(EventType.TOOL_EXECUTING, {"name": tool.name, "kwargs": kwargs})
        exec_result = self._execute(tool.code, tool.name, kwargs, granted)

        if tool.tool_id is not None:
            self.registry.record_execution(tool.tool_id, exec_result.success)

        self._emit(EventType.TOOL_EXECUTED, {
            "name":       tool.name,
            "success":    exec_result.success,
            "latency_ms": exec_result.latency_ms,
            "output":     exec_result.output,
            "error":      exec_result.error,
        })

        if exec_result.success:
            logger.info("Tool '%s' succeeded in %.1f ms", tool.name, exec_result.latency_ms)
            return AgentResult(
                answer=exec_result.output,
                action_taken=A_USED_TOOL,
                tool_name=tool.name,
                validation_passed=None,
                latency_ms=exec_result.latency_ms,
                notes=notes,
            )

        logger.warning("Tool '%s' failed: %s — falling back.", tool.name, exec_result.error)
        notes.append(f"Tool failed ({exec_result.error}); answered directly.")
        result = self._answer_directly(query)
        result.action_taken = A_USED_TOOL_FALLBACK
        result.tool_name = tool.name
        result.notes = notes + result.notes
        return result

    def _create_and_use_tool(self, query: str) -> AgentResult:
        """Write → validate → [user decision] → save → execute a brand-new tool."""
        notes: list[str] = []
        revision_notes = ""

        for attempt in range(1, MAX_WRITE_ATTEMPTS + 1):
            # ── 1. Write ──────────────────────────────────────────────────────
            effective_query = (
                f"{query}\n\nRevision notes: {revision_notes}"
                if revision_notes else query
            )
            self._emit(EventType.TOOL_WRITING, {"query": effective_query, "attempt": attempt})
            written = write_tool(effective_query, self.llm)

            if not written:
                self._emit(EventType.TOOL_WRITTEN, {"success": False})
                notes.append(f"Tool writer failed (attempt {attempt}).")
                result = self._answer_directly(query)
                result.notes = notes + result.notes
                result.validation_passed = False
                return result

            self._emit(EventType.TOOL_WRITTEN, {
                "success":     True,
                "name":        written.name,
                "description": written.description,
            })
            notes.append(f"Wrote tool '{written.name}' (attempt {attempt}).")
            logger.info("Tool written: '%s'", written.name)

            # ── 2. Validate ───────────────────────────────────────────────────
            self._emit(EventType.TOOL_VALIDATING, {"name": written.name})
            validation = validate_tool(
                fn_name=written.name,
                description=written.description,
                code=written.code,
                llm_client=self.llm,
            )
            notes.append(f"Validation: {validation.summary}")
            logger.info("Validation: %s", validation.summary)

            # ── 3. Tool decision ──────────────────────────────────────────────
            self._emit(EventType.TOOL_DECISION, {
                "name":               written.name,
                "code":               written.code,
                "validation_summary": validation.summary,
                "passed":             validation.passed,
            })

            if self._tool_decision is not None:
                decision, user_revision_notes = self._tool_decision(
                    written.name, written.code, validation
                )
            else:
                # Default: auto-keep on pass, auto-discard on fail
                decision = KEEP if validation.passed else DISCARD
                user_revision_notes = ""

            if decision == DISCARD:
                for failure in validation.failures:
                    notes.append(f"  ✗ {failure}")
                notes.append("Tool discarded.")
                result = self._answer_directly(query)
                result.notes = notes + result.notes
                result.validation_passed = validation.passed
                return result

            if decision == REVISE:
                revision_notes = user_revision_notes
                notes.append(f"Revision requested: '{revision_notes}'")
                continue  # next attempt

            # decision == KEEP ─────────────────────────────────────────────────

            # ── 4. Permission scan ────────────────────────────────────────────
            granted = self._handle_permissions(written.code)

            # ── 5. Save ───────────────────────────────────────────────────────
            embedding = embed(f"{written.name}: {written.description}")
            tool = Tool(name=written.name, description=written.description, code=written.code)
            try:
                tool_id = self.registry.save_tool(tool, embedding)
                tool.tool_id = tool_id
                notes.append(f"Tool saved (id={tool_id}).")
                logger.info("Tool '%s' saved as id=%d", written.name, tool_id)
            except Exception as exc:
                notes.append(f"Could not save tool ({exc}).")
                logger.warning("Tool save failed: %s", exc)
                tool_id = None

            self._emit(EventType.TOOL_SAVED, {"name": written.name, "tool_id": tool_id})

            # ── 6. Execute ────────────────────────────────────────────────────
            kwargs = self._extract_kwargs(query, written.code, written.name)
            self._emit(EventType.TOOL_EXECUTING, {"name": written.name, "kwargs": kwargs})
            exec_result = self._execute(written.code, written.name, kwargs, granted)

            if tool_id is not None:
                self.registry.record_execution(tool_id, exec_result.success)

            self._emit(EventType.TOOL_EXECUTED, {
                "name":       written.name,
                "success":    exec_result.success,
                "latency_ms": exec_result.latency_ms,
                "output":     exec_result.output,
                "error":      exec_result.error,
            })

            if exec_result.success:
                logger.info(
                    "Tool '%s' executed in %.1f ms → %r",
                    written.name, exec_result.latency_ms, exec_result.output,
                )
                return AgentResult(
                    answer=exec_result.output,
                    action_taken=A_CREATED_AND_USED,
                    tool_name=written.name,
                    validation_passed=True,
                    latency_ms=exec_result.latency_ms,
                    notes=notes,
                )

            notes.append(
                f"Execution failed on real query ({exec_result.error}); answering directly."
            )
            logger.warning("Post-save execution failed for '%s': %s", written.name, exec_result.error)
            result = self._answer_directly(query)
            result.action_taken = A_CREATED_FALLBACK
            result.tool_name = written.name
            result.validation_passed = True
            result.notes = notes + result.notes
            return result

        # Exhausted MAX_WRITE_ATTEMPTS with repeated "revise" decisions
        notes.append(f"Exhausted {MAX_WRITE_ATTEMPTS} write attempts.")
        result = self._answer_directly(query)
        result.notes = notes + result.notes
        result.validation_passed = False
        return result

    def _attempt_revision(self, tool: Tool) -> None:
        """Auto-revise a degraded tool: rewrite → validate → update_tool."""
        logger.info("Attempting auto-revision of '%s'.", tool.name)
        self._emit(EventType.TOOL_WRITING, {
            "query": f"[revision] {tool.name}",
            "attempt": 1,
        })

        revised = revise_tool(tool, self.llm)
        if revised is None:
            logger.warning("Reviser returned nothing for '%s'.", tool.name)
            return

        self._emit(EventType.TOOL_WRITTEN, {
            "success":     True,
            "name":        revised.name,
            "description": revised.description,
        })
        self._emit(EventType.TOOL_VALIDATING, {"name": revised.name})

        validation = validate_tool(revised.name, revised.description, revised.code, self.llm)
        if not validation.passed:
            logger.warning("Revised '%s' failed validation.", tool.name)
            return

        new_embedding = embed(f"{revised.name}: {revised.description}")
        self.registry.update_tool(
            tool_id=tool.tool_id,
            new_code=revised.code,
            new_description=revised.description,
            new_embedding=new_embedding,
            reason="Auto-revised after negative user feedback",
        )
        logger.info("Tool '%s' revised and updated.", tool.name)

    # ── Event / streaming helpers ─────────────────────────────────────────────

    def _emit(self, event_type: EventType, payload: dict | None = None) -> None:
        """Fire an event.  Zero-cost (single None check) when on_event is not set."""
        if self._on_event is None:
            return
        self._on_event(AgentEvent(type=event_type, payload=payload or {}))

    def _llm_stream(self, system: str, user: str, max_tokens: int = 1000) -> str:
        """
        Stream LLM response, emitting ANSWER_CHUNK per chunk.
        Falls back to .complete() if the client has no .stream() method.
        Returns the fully assembled string.
        """
        if not hasattr(self.llm, "stream"):
            answer = self.llm.complete(system, user, max_tokens)
            self._emit(EventType.ANSWER_CHUNK, {"chunk": answer})
            return answer

        chunks: list[str] = []
        for chunk in self.llm.stream(system, user, max_tokens):
            chunks.append(chunk)
            self._emit(EventType.ANSWER_CHUNK, {"chunk": chunk})
        return "".join(chunks)

    # ── Permission / execution helpers ────────────────────────────────────────

    def _handle_permissions(self, code: str) -> GrantedPermissions:
        """
        Scan the tool code for permission requirements.
        If a handler is registered, delegate the grant decision to it.
        Otherwise return empty (deny-all) permissions.
        Emits PERMISSION_REQUEST if any permissions are needed.
        """
        requests = scan_permissions(code)
        if not requests:
            return GrantedPermissions()

        self._emit(EventType.PERMISSION_REQUEST, {
            "permissions": [
                {"type": r.type, "reason": r.reason, "required": r.required}
                for r in requests
            ]
        })

        if self._permission_handler is not None:
            return self._permission_handler(requests)

        return GrantedPermissions()   # deny by default

    def _execute(self, code: str, fn_name: str, kwargs: dict, granted: GrantedPermissions):
        """Route execution to Docker or subprocess based on agent config."""
        if self._use_docker:
            from .docker_executor import execute_in_docker, docker_available
            if docker_available():
                return execute_in_docker(code, fn_name, kwargs, granted)
            logger.warning("Docker requested but not available; falling back to subprocess.")
        return execute_tool(code, fn_name, kwargs)

    # ── Kwargs extraction ─────────────────────────────────────────────────────

    def _extract_kwargs(self, query: str, code: str, fn_name: str) -> dict:
        """
        Ask the LLM to extract function arguments from the natural-language query.
        Returns {} on failure; the executor will produce a clear TypeError.
        """
        user_prompt = (
            f"Function code:\n{code}\n\n"
            f"User query: {query}\n\n"
            f"Extract the keyword arguments for '{fn_name}':"
        )
        try:
            raw = self.llm.complete(
                system=KWARGS_EXTRACT_SYSTEM,
                user=user_prompt,
                max_tokens=300,
            )
            cleaned = re.sub(r"```(?:json)?\s*|```", "", raw).strip()
            kwargs = json.loads(cleaned)
            if not isinstance(kwargs, dict):
                raise ValueError(f"Expected dict, got {type(kwargs)}")
            return kwargs
        except Exception as exc:
            logger.warning("kwargs extraction failed: %s — using empty dict", exc)
            return {}
