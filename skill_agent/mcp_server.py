"""
mcp_server.py — Model Context Protocol (MCP) server for the skill registry.

Exposes five stable meta-tool primitives so that external agents (OpenClaw,
Claude Desktop, VS Code MCP extension, etc.) can interact with the autoSkill
runtime without being flooded by the full list of generated tools.

Protocol:
  Transport : stdio, Content-Length framed (identical to Language Server Protocol)
  Encoding  : UTF-8 JSON-RPC 2.0
  Version   : MCP 2024-11-05

Exposed MCP tools (stable — never change regardless of what tools are generated):
  search_tools   — semantic search over the registry
  execute_tool   — run a tool by ID with given args
  create_tool    — generate, validate, and register a new tool for a task
  improve_tool   — trigger LLM revision of a degraded tool
  tool_stats     — health and usage statistics for a tool

Running:
  python -m skill_agent.mcp_server --db skills.db
  python -m skill_agent.mcp_server --db skills.db --llm openai --llm-model gpt-4o
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from typing import Any, Optional

from .tool_registry import ToolRegistry
from .executor import execute_tool

logger = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME          = "autoskill"
SERVER_VERSION       = "1.0.0"


# ── Stable meta-tool definitions ─────────────────────────────────────────────

META_TOOLS = [
    {
        "name": "search_tools",
        "description": (
            "Search the autoSkill registry for tools that match a natural-language "
            "query. Returns ranked results with similarity score, health status, and "
            "usage statistics. Use this before create_tool to check if a tool already "
            "exists."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language description of the task or capability",
                },
                "top_k": {
                    "type": "number",
                    "description": "Maximum number of results to return (default: 5)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "execute_tool",
        "description": (
            "Execute a tool from the registry by its numeric ID. Use search_tools "
            "first to find the tool_id. Runs the tool in a sandboxed subprocess with "
            "a 10-second timeout."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool_id": {
                    "type": "number",
                    "description": "Numeric tool ID returned by search_tools",
                },
                "args": {
                    "type": "object",
                    "description": "Keyword arguments matching the tool function's parameters",
                },
            },
            "required": ["tool_id"],
        },
    },
    {
        "name": "create_tool",
        "description": (
            "Generate, validate, and register a brand-new Python tool for a "
            "deterministic task. The LLM writes the function, validates it, then "
            "saves it to the registry so it can be reused via search_tools in future "
            "calls. Only use for clearly deterministic, repeatable tasks."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Plain-English description of what the tool should compute or transform",
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "improve_tool",
        "description": (
            "Trigger an LLM-driven revision of a degraded or failing tool. "
            "The revised version is validated before replacing the original; "
            "the old version is preserved in the version history."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool_id": {
                    "type": "number",
                    "description": "Numeric ID of the tool to revise",
                },
            },
            "required": ["tool_id"],
        },
    },
    {
        "name": "tool_stats",
        "description": (
            "Return health and usage statistics for a specific tool: "
            "status, success rate, usage count, and user sentiment score."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool_id": {
                    "type": "number",
                    "description": "Numeric ID of the tool",
                },
            },
            "required": ["tool_id"],
        },
    },
]


# ── Input schema extraction (utility — used by tests and external tooling) ────

_PY_TO_JSON_TYPE: dict[str, str] = {
    "int":   "number",
    "float": "number",
    "str":   "string",
    "bool":  "boolean",
    "list":  "array",
    "List":  "array",
    "dict":  "object",
    "Dict":  "object",
}


def _annotation_to_json_type(annotation: ast.expr | None) -> str:
    if annotation is None:
        return "string"
    if isinstance(annotation, ast.Name):
        return _PY_TO_JSON_TYPE.get(annotation.id, "string")
    if isinstance(annotation, ast.Subscript):
        return _annotation_to_json_type(annotation.value)
    return "string"


def _extract_input_schema(code: str, fn_name: str) -> dict:
    """Parse code with AST and return a JSON Schema describing the function's parameters."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"type": "object", "properties": {}, "required": []}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            properties: dict[str, Any] = {}
            required: list[str] = []
            for arg in node.args.args:
                if arg.arg == "self":
                    continue
                json_type = _annotation_to_json_type(arg.annotation)
                properties[arg.arg] = {"type": json_type}
                required.append(arg.arg)
            return {"type": "object", "properties": properties, "required": required}

    return {"type": "object", "properties": {}, "required": []}


# ── Framing ───────────────────────────────────────────────────────────────────

def _read_message() -> dict | None:
    """Read one Content-Length framed JSON-RPC message from stdin. Returns None on EOF."""
    header = sys.stdin.buffer.readline()
    if not header:
        return None

    header = header.decode("utf-8").strip()
    if not header.lower().startswith("content-length:"):
        logger.warning("MCP: unexpected header line: %r", header)
        return None

    try:
        length = int(header.split(":", 1)[1].strip())
    except ValueError:
        logger.warning("MCP: malformed Content-Length header: %r", header)
        return None

    sys.stdin.buffer.readline()  # consume blank separator

    body = sys.stdin.buffer.read(length)
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("MCP: failed to parse JSON body: %s", exc)
        return None


def _write_message(obj: dict) -> None:
    """Write one Content-Length framed JSON-RPC message to stdout."""
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
    sys.stdout.buffer.write(header + body)
    sys.stdout.buffer.flush()


# ── JSON-RPC helpers ──────────────────────────────────────────────────────────

def _ok(req_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(req_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _text(content: Any) -> dict:
    """Wrap a value as an MCP text content block."""
    return {"content": [{"type": "text", "text": json.dumps(content, ensure_ascii=False)}]}


# ── Meta-tool handlers ────────────────────────────────────────────────────────

def _handle_search_tools(req_id: Any, args: dict, registry: ToolRegistry) -> dict:
    query = args.get("query", "").strip()
    if not query:
        return _err(req_id, -32602, "'query' is required and must be non-empty")

    top_k = int(args.get("top_k", 5))
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
    return _ok(req_id, _text({"matches": payload, "count": len(payload)}))


def _handle_execute_tool(req_id: Any, args: dict, registry: ToolRegistry) -> dict:
    raw_id = args.get("tool_id")
    if raw_id is None:
        return _err(req_id, -32602, "'tool_id' is required")

    try:
        tool_id = int(raw_id)
    except (TypeError, ValueError):
        return _err(req_id, -32602, f"'tool_id' must be a number, got {raw_id!r}")

    tool = registry.get_tool_by_id(tool_id)
    if tool is None:
        return _err(req_id, -32602, f"No tool found with id={tool_id}")
    if tool.status == "retired":
        return _err(req_id, -32602, f"Tool '{tool.name}' (id={tool_id}) is retired")

    tool_args = args.get("args", {})
    if not isinstance(tool_args, dict):
        return _err(req_id, -32602, "'args' must be a JSON object")

    exec_result = execute_tool(tool.code, tool.name, tool_args)
    registry.record_execution(tool_id, exec_result.success)

    if exec_result.success:
        return _ok(req_id, _text({
            "result":     exec_result.output,
            "tool_name":  tool.name,
            "latency_ms": round(exec_result.latency_ms, 1),
        }))

    return _err(req_id, -32000, f"Execution failed: {exec_result.error}")


def _handle_create_tool(req_id: Any, args: dict, agent) -> dict:
    if agent is None:
        return _err(
            req_id, -32603,
            "create_tool requires an LLM — start the server with --llm anthropic (or openai)",
        )

    task = args.get("task", "").strip()
    if not task:
        return _err(req_id, -32602, "'task' is required and must be non-empty")

    result = agent._create_and_use_tool(task)

    if result.tool_name is None:
        return _err(req_id, -32000, f"Tool creation failed. Notes: {result.notes}")

    return _ok(req_id, _text({
        "tool_name":  result.tool_name,
        "action":     result.action_taken,
        "answer":     result.answer,
        "notes":      result.notes,
    }))


def _handle_improve_tool(req_id: Any, args: dict, registry: ToolRegistry, agent) -> dict:
    if agent is None:
        return _err(
            req_id, -32603,
            "improve_tool requires an LLM — start the server with --llm anthropic (or openai)",
        )

    raw_id = args.get("tool_id")
    if raw_id is None:
        return _err(req_id, -32602, "'tool_id' is required")

    try:
        tool_id = int(raw_id)
    except (TypeError, ValueError):
        return _err(req_id, -32602, f"'tool_id' must be a number, got {raw_id!r}")

    tool = registry.get_tool_by_id(tool_id)
    if tool is None:
        return _err(req_id, -32602, f"No tool found with id={tool_id}")

    old_version_count = len(registry.get_versions(tool_id))
    agent._attempt_revision(tool)
    new_version_count = len(registry.get_versions(tool_id))

    revised = new_version_count > old_version_count
    return _ok(req_id, _text({
        "tool_id":  tool_id,
        "name":     tool.name,
        "revised":  revised,
        "message":  "Tool successfully revised and updated." if revised else
                    "Revision attempted but the new version did not pass validation.",
    }))


def _handle_tool_stats(req_id: Any, args: dict, registry: ToolRegistry) -> dict:
    raw_id = args.get("tool_id")
    if raw_id is None:
        return _err(req_id, -32602, "'tool_id' is required")

    try:
        tool_id = int(raw_id)
    except (TypeError, ValueError):
        return _err(req_id, -32602, f"'tool_id' must be a number, got {raw_id!r}")

    tool = registry.get_tool_by_id(tool_id)
    if tool is None:
        return _err(req_id, -32602, f"No tool found with id={tool_id}")

    sentiment = registry.get_user_sentiment(tool_id)
    versions  = registry.get_versions(tool_id)

    return _ok(req_id, _text({
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
    }))


# ── MCP method handlers ───────────────────────────────────────────────────────

def handle_initialize(req: dict) -> dict:
    return _ok(req.get("id"), {
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "capabilities":    {"tools": {}},
        "serverInfo":      {"name": SERVER_NAME, "version": SERVER_VERSION},
    })


def handle_tools_list(req: dict) -> dict:
    return _ok(req.get("id"), {"tools": META_TOOLS})


def handle_tools_call(
    req: dict,
    registry: ToolRegistry,
    agent,  # Optional[SkillAgent] — None when started without --llm
) -> dict:
    req_id = req.get("id")
    params = req.get("params", {})
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})

    if tool_name == "search_tools":
        return _handle_search_tools(req_id, arguments, registry)
    if tool_name == "execute_tool":
        return _handle_execute_tool(req_id, arguments, registry)
    if tool_name == "create_tool":
        return _handle_create_tool(req_id, arguments, agent)
    if tool_name == "improve_tool":
        return _handle_improve_tool(req_id, arguments, registry, agent)
    if tool_name == "tool_stats":
        return _handle_tool_stats(req_id, arguments, registry)

    return _err(req_id, -32602, f"Unknown tool: '{tool_name}'")


# ── Dispatch loop ─────────────────────────────────────────────────────────────

def _dispatch(req: dict, registry: ToolRegistry, agent) -> dict | None:
    method = req.get("method", "")

    if method == "initialize":
        return handle_initialize(req)
    if method == "initialized":
        return None  # notification — no response
    if method == "tools/list":
        return handle_tools_list(req)
    if method == "tools/call":
        return handle_tools_call(req, registry, agent)

    req_id = req.get("id")
    if req_id is not None:
        return _err(req_id, -32601, f"Method not found: '{method}'")
    return None


def run_server(registry: ToolRegistry, agent=None) -> None:
    """Main stdio dispatch loop. Runs until EOF on stdin."""
    mode = "with LLM" if agent is not None else "registry-only (no LLM)"
    logger.info("autoSkill MCP server ready [%s] (protocol %s)", mode, MCP_PROTOCOL_VERSION)
    while True:
        msg = _read_message()
        if msg is None:
            break
        response = _dispatch(msg, registry, agent)
        if response is not None:
            _write_message(response)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)-7s %(name)s — %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        description="autoSkill MCP stdio server — exposes the skill registry to OpenClaw and other MCP clients.",
    )
    parser.add_argument("--db", default="skills.db",
                        help="Path to the SQLite skills database (default: skills.db)")
    parser.add_argument("--llm", default=None, choices=["anthropic", "openai"],
                        help="LLM provider for create_tool / improve_tool (omit to disable those methods)")
    parser.add_argument("--llm-model", default=None,
                        help="Model name override (e.g. claude-opus-4-6, gpt-4o)")
    parser.add_argument("--llm-api-key", default=None,
                        help="API key override (falls back to env vars)")
    args = parser.parse_args()

    registry = ToolRegistry(db_path=args.db)

    agent: Optional[object] = None
    if args.llm is not None:
        from .agent import SkillAgent
        agent = SkillAgent(
            llm=args.llm,
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            db_path=args.db,
        )

    run_server(registry, agent)


if __name__ == "__main__":
    main()
