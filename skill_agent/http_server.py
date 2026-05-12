"""
http_server.py — FastAPI HTTP middleware layer for the autoSkill runtime.

Exposes the runtime as a REST API so it can sit in front of any LLM
provider or be integrated into existing applications.

Architecture
------------
    User request
        ↓
    POST /run  ←── this server
        ├── tool match found  → execute sandbox → return answer (no LLM call)
        └── no match          → SkillAgent routes to LLM

Endpoints
---------
    POST /run                 Run a query through the full agent pipeline
    POST /tools/search        Semantic search over the tool registry
    GET  /tools               List all active tools
    GET  /tools/{id}          Full details + code for one tool
    POST /tools/{id}/feedback Record thumbs up/down for a tool
    GET  /health              Liveness check

Starting the server
-------------------
    # Via CLI (recommended):
    skill-agent serve-http --port 8000 --llm anthropic

    # Via uvicorn directly:
    uvicorn skill_agent.http_server:app --port 8000
    # (uses SKILL_AGENT_* env vars for config)

    # Programmatically:
    from skill_agent import SkillAgent
    from skill_agent.http_server import create_app
    import uvicorn

    agent = SkillAgent()
    app   = create_app(agent)
    uvicorn.run(app, host="0.0.0.0", port=8000)

Thread safety note
------------------
SkillAgent.run() is synchronous. Concurrent HTTP requests are handled via
asyncio.to_thread(), which dispatches each call to the default thread pool.
The ToolRegistry's FAISS index is protected by an internal lock; SQLite is
opened with WAL mode for concurrent reads.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response
    from pydantic import BaseModel
except ImportError as _e:
    raise ImportError(
        "HTTP server requires FastAPI. "
        "Install with: pip install 'skill-agent[http]'"
    ) from _e

logger = logging.getLogger(__name__)

_VERSION = "0.2.0"


# ── Pydantic models ───────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    query: str


class RunResponse(BaseModel):
    answer: Any
    action_taken: str
    tool_name: Optional[str] = None
    validation_passed: Optional[bool] = None
    latency_ms: Optional[float] = None
    notes: list[str] = []


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchMatch(BaseModel):
    tool_id: int
    name: str
    description: str
    similarity: float
    status: str
    usage_count: int
    success_rate: float


class SearchResponse(BaseModel):
    matches: list[SearchMatch]
    count: int


class ToolSummary(BaseModel):
    tool_id: int
    name: str
    description: str
    status: str
    usage_count: int
    success_rate: float


class ToolDetail(ToolSummary):
    code: str
    created_at: float
    last_used_at: Optional[float] = None
    user_sentiment: Optional[float] = None
    version_count: int


class FeedbackRequest(BaseModel):
    positive: bool
    comment: str = ""


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(agent) -> FastAPI:
    """
    Create and return the FastAPI application bound to `agent`.

    Parameters
    ----------
    agent : SkillAgent
        A fully-initialised SkillAgent instance. All endpoints delegate to
        this single instance — create it with the desired provider and db_path
        before calling create_app().

    Returns
    -------
    FastAPI application ready to be passed to uvicorn.run() or mounted.
    """
    app = FastAPI(
        title="autoSkill",
        description="Executable semantic memory for AI agents",
        version=_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ────────────────────────────────────────────────────────────────

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        """Liveness check. Returns tool count and server version."""
        tool_count = len(agent.registry.list_tools())
        return {"status": "ok", "tools_count": tool_count, "version": _VERSION}

    # ── Core runtime ──────────────────────────────────────────────────────────

    @app.post("/run", response_model=RunResponse, tags=["runtime"])
    async def run_query(req: RunRequest) -> RunResponse:
        """
        Run a query through the full agent pipeline.

        The agent checks the tool registry first (vector search). If a
        high-confidence match is found, the tool executes immediately without
        an LLM call. Otherwise it routes to the configured LLM provider.
        """
        result = await asyncio.to_thread(agent.run, req.query)
        return RunResponse(
            answer           = result.answer,
            action_taken     = result.action_taken,
            tool_name        = result.tool_name,
            validation_passed= result.validation_passed,
            latency_ms       = result.latency_ms,
            notes            = result.notes,
        )

    # ── Tool registry ─────────────────────────────────────────────────────────

    @app.post("/tools/search", response_model=SearchResponse, tags=["tools"])
    def search_tools(req: SearchRequest) -> SearchResponse:
        """Semantic search over the tool registry. Returns ranked matches."""
        results = agent.registry.search(req.query, top_k=req.top_k)
        return SearchResponse(
            matches=[
                SearchMatch(
                    tool_id    = r.tool.tool_id,
                    name       = r.tool.name,
                    description= r.tool.description,
                    similarity = round(r.similarity, 3),
                    status     = r.tool.status,
                    usage_count= r.tool.usage_count,
                    success_rate= round(r.tool.success_rate, 3),
                )
                for r in results
            ],
            count=len(results),
        )

    @app.get("/tools", response_model=list[ToolSummary], tags=["tools"])
    def list_tools(include_retired: bool = False) -> list[ToolSummary]:
        """List all tools in the registry, ordered by usage count."""
        tools = agent.registry.list_tools(include_retired=include_retired)
        return [
            ToolSummary(
                tool_id    = t.tool_id,
                name       = t.name,
                description= t.description,
                status     = t.status,
                usage_count= t.usage_count,
                success_rate= round(t.success_rate, 3),
            )
            for t in tools
        ]

    @app.get("/tools/{tool_id}", response_model=ToolDetail, tags=["tools"])
    def get_tool(tool_id: int) -> ToolDetail:
        """Full details for one tool, including source code and version count."""
        tool = agent.registry.get_tool_by_id(tool_id)
        if tool is None:
            raise HTTPException(status_code=404,
                                detail=f"Tool id={tool_id} not found")

        sentiment = agent.registry.get_user_sentiment(tool_id)
        versions  = agent.registry.get_versions(tool_id)

        return ToolDetail(
            tool_id      = tool_id,
            name         = tool.name,
            description  = tool.description,
            code         = tool.code,
            status       = tool.status,
            usage_count  = tool.usage_count,
            success_rate = round(tool.success_rate, 3),
            created_at   = tool.created_at,
            last_used_at = tool.last_used_at,
            user_sentiment= round(sentiment, 3) if sentiment is not None else None,
            version_count= len(versions),
        )

    @app.post("/tools/{tool_id}/feedback",
              status_code=204, tags=["tools"])
    def record_feedback(tool_id: int, req: FeedbackRequest) -> Response:
        """Record thumbs up or down for a tool. Triggers health re-evaluation."""
        tool = agent.registry.get_tool_by_id(tool_id)
        if tool is None:
            raise HTTPException(status_code=404,
                                detail=f"Tool id={tool_id} not found")
        agent.registry.save_feedback(tool_id, req.positive, req.comment)
        return Response(status_code=204)

    return app


# ── Module-level app for direct uvicorn use ───────────────────────────────────
# `uvicorn skill_agent.http_server:app` reads config from env vars.

def _build_default_app() -> FastAPI:
    from .config import AutoSkillConfig
    cfg   = AutoSkillConfig.from_env()
    agent = cfg.create_agent()
    return create_app(agent)


app = _build_default_app()
