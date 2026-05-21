"""
mcp_http_server.py — Remote MCP server using the official mcp SDK.

Wraps the existing ToolRegistry handlers as FastMCP tools and exposes them
via HTTP (Streamable HTTP transport, MCP spec 2025).  Use this when you
want Claude Desktop, OpenClaw, or any MCP client to connect over the network
instead of via local stdio.

The server is a *dumb runtime* — it stores, searches, and executes tools
but makes zero AI decisions.  The host LLM provides the code directly.

Running:
  python -m skill_agent.mcp_http_server --db skills.db
  python -m skill_agent.mcp_http_server --db skills.db --host 0.0.0.0 --port 8000

Client config (Claude Desktop / OpenClaw):
  {
    "mcpServers": {
      "autoskill": {
        "type": "http",
        "url": "http://<linode-ip>:8000/mcp"
      }
    }
  }
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

MCP_SERVER_NAME    = "autoskill"
MCP_SERVER_VERSION = "1.0.0"


def _build_mcp(registry, host: str = "127.0.0.1", port: int = 8000, path: str = "/mcp") -> Any:
    """Create and return a FastMCP instance with all tools registered."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        name=MCP_SERVER_NAME,
        instructions=(
            "autoSkill is a persistent tool runtime. You are the intelligence — "
            "autoSkill is the storage and execution layer. Always search_tools "
            "before creating a new one. Provide complete Python code when saving "
            "tools. Tools must be pure functions (no filesystem, network, or "
            "subprocess imports)."
        ),
        host=host,
        port=port,
        streamable_http_path=path,
    )

    from .tool_registry import ToolRegistry, Tool
    from .executor import execute_tool
    from .embeddings import embed
    from .mcp_server import _validate_code_static

    # ── search_tools ──────────────────────────────────────────────────────

    @mcp.tool(
        name="search_tools",
        description=(
            "Search the autoSkill registry for tools matching a natural-language "
            "query. Returns ranked results with similarity score, health status, "
            "and usage statistics. Call this before save_tool to avoid duplicates."
        ),
    )
    def search_tools(query: str, top_k: int = 5) -> str:
        results = registry.search(query, top_k=top_k, min_similarity=0.30)
        payload = [
            {
                "tool_id":     r.tool.tool_id,
                "name":        r.tool.name,
                "description": r.tool.description,
                "similarity":  round(r.similarity, 3),
                "status":      r.tool.status,
                "usage_count": r.tool.usage_count,
                "success_rate": round(r.tool.success_rate, 3),
            }
            for r in results
        ]
        return json.dumps({"matches": payload, "count": len(payload)})

    # ── execute_tool ──────────────────────────────────────────────────────

    @mcp.tool(
        name="execute_tool",
        description=(
            "Execute a tool from the registry by its numeric ID. Use search_tools "
            "first to find the tool_id. Runs the tool in a sandboxed subprocess "
            "with a 10-second timeout."
        ),
    )
    def execute_tool_handler(tool_id: int, args: dict | None = None) -> str:
        if args is None:
            args = {}

        tool = registry.get_tool_by_id(tool_id)
        if tool is None:
            return json.dumps({"error": f"No tool found with id={tool_id}"})
        if tool.status == "retired":
            return json.dumps({"error": f"Tool '{tool.name}' (id={tool_id}) is retired"})

        exec_result = execute_tool(tool.code, tool.name, args)
        registry.record_execution(tool_id, exec_result.success)

        if exec_result.success:
            return json.dumps({
                "result":     exec_result.output,
                "tool_name":  tool.name,
                "latency_ms": round(exec_result.latency_ms, 1),
            })
        return json.dumps({"error": f"Execution failed: {exec_result.error}"})

    # ── save_tool ─────────────────────────────────────────────────────────

    @mcp.tool(
        name="save_tool",
        description=(
            "Save a new tool to the registry. You provide the name, description, "
            "and complete Python source code. The code must define a pure function "
            "(no filesystem, network, or subprocess imports). Returns the new tool_id."
        ),
    )
    def save_tool(name: str, description: str, code: str) -> str:
        # Static validation
        valid, err_msg = _validate_code_static(code, name)
        if not valid:
            return json.dumps({"error": f"Code validation failed: {err_msg}"})

        # Check for name collisions
        existing = registry.get_tool_by_name(name)
        if existing is not None:
            return json.dumps({
                "error": (
                    f"A tool named '{name}' already exists (id={existing.tool_id}). "
                    f"Use save_tool_version to update it."
                ),
            })

        # Embed and save
        embedding = embed(f"{name}: {description}")
        tool = Tool(name=name, description=description, code=code)

        try:
            tool_id = registry.save_tool(tool, embedding)
        except Exception as exc:
            return json.dumps({"error": f"Failed to save tool: {exc}"})

        return json.dumps({
            "tool_id": tool_id,
            "name":    name,
            "status":  "active",
            "message": f"Tool '{name}' saved successfully with id={tool_id}",
        })

    # ── save_tool_version ─────────────────────────────────────────────────

    @mcp.tool(
        name="save_tool_version",
        description=(
            "Save a new version of an existing tool. The previous version is "
            "preserved in version history. The tool's status resets to 'active'. "
            "Use this when a tool needs improvement or you've written a better "
            "implementation. The function name must match the existing tool."
        ),
    )
    def save_tool_version(tool_id: int, name: str, description: str, code: str) -> str:
        existing = registry.get_tool_by_id(tool_id)
        if existing is None:
            return json.dumps({"error": f"No tool found with id={tool_id}"})

        if name != existing.name:
            return json.dumps({
                "error": (
                    f"Name '{name}' does not match existing tool name "
                    f"'{existing.name}'. The function name must stay the same "
                    f"across versions."
                ),
            })

        valid, err_msg = _validate_code_static(code, name)
        if not valid:
            return json.dumps({"error": f"Code validation failed: {err_msg}"})

        embedding = embed(f"{name}: {description}")
        old_version_count = len(registry.get_versions(tool_id))

        try:
            registry.update_tool(
                tool_id, code, description, embedding,
                reason="Host LLM revision",
            )
        except Exception as exc:
            return json.dumps({"error": f"Failed to update tool: {exc}"})

        new_version_count = len(registry.get_versions(tool_id))

        return json.dumps({
            "tool_id":         tool_id,
            "name":            name,
            "status":          "active",
            "versions_before": old_version_count,
            "versions_after":  new_version_count,
            "message": (
                f"Tool '{name}' (id={tool_id}) updated — "
                f"version {new_version_count} saved."
            ),
        })

    # ── tool_stats ────────────────────────────────────────────────────────

    @mcp.tool(
        name="tool_stats",
        description=(
            "Return health and usage statistics for a specific tool: "
            "status, success rate, usage count, and user sentiment score."
        ),
    )
    def tool_stats(tool_id: int) -> str:
        tool = registry.get_tool_by_id(tool_id)
        if tool is None:
            return json.dumps({"error": f"No tool found with id={tool_id}"})

        sentiment = registry.get_user_sentiment(tool_id)
        versions  = registry.get_versions(tool_id)

        return json.dumps({
            "tool_id":        tool_id,
            "name":           tool.name,
            "description":    tool.description,
            "status":         tool.status,
            "usage_count":    tool.usage_count,
            "success_rate":   round(tool.success_rate, 3),
            "user_sentiment": round(sentiment, 3) if sentiment is not None else None,
            "version_count":  len(versions),
            "created_at":     tool.created_at,
            "last_used_at":   tool.last_used_at,
        })

    return mcp


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)-7s %(name)s — %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        description="autoSkill MCP HTTP server — exposes the skill registry over the network via Streamable HTTP.",
    )
    parser.add_argument("--db", default="skills.db",
                        help="Path to the SQLite skills database (default: skills.db)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (default: 127.0.0.1; use 0.0.0.0 for public)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to listen on (default: 8000)")
    parser.add_argument("--path", default="/mcp",
                        help="HTTP path for the MCP endpoint (default: /mcp)")
    args = parser.parse_args()

    from .tool_registry import ToolRegistry
    registry = ToolRegistry(db_path=args.db)

    mcp = _build_mcp(registry, host=args.host, port=args.port, path=args.path)

    print(f"autoSkill MCP HTTP server → http://{args.host}:{args.port}{args.path}",
          file=sys.stderr)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
