"""
mcp_server.py — Model Context Protocol (MCP) server for the skill registry.

Exposes five stable meta-tool primitives so that external agents (OpenClaw,
Claude Desktop, VS Code MCP extension, etc.) can interact with the autoSkill
runtime without being flooded by the full list of generated tools.

The MCP server is a *dumb runtime* — it stores, searches, and executes tools
but makes zero AI decisions itself.  The host LLM is responsible for all
intelligence: deciding when to create, reuse, or revise tools.

Protocol:
  Transport : stdio, Content-Length framed (identical to Language Server Protocol)
  Encoding  : UTF-8 JSON-RPC 2.0
  Version   : MCP 2024-11-05

Exposed MCP tools (stable — never change regardless of what tools are generated):
  search_tools       — semantic search over the registry
  execute_tool       — run a tool by ID with given args
  save_tool          — register a new tool (host provides name, description, code)
  save_tool_version  — update an existing tool, snapshotting the old version
  tool_stats         — health and usage statistics for a tool

Running:
  python -m skill_agent.mcp_server --db skills.db
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from typing import Any, Optional

from .tool_registry import ToolRegistry, Tool
from .executor import execute_tool
from .embeddings import embed

logger = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME          = "autoskill"
SERVER_VERSION       = "1.0.0"


# ── Dangerous import categories (mirrors permissions.py) ────────────────────────

_FILESYSTEM_MODULES = frozenset({
    "os", "pathlib", "shutil", "glob", "tempfile", "io", "fileinput",
    "fnmatch", "stat", "zipfile", "tarfile", "gzip", "bz2", "lzma",
})

_NETWORK_MODULES = frozenset({
    "requests", "urllib", "http", "httpx", "aiohttp", "socket",
    "ftplib", "smtplib", "poplib", "imaplib", "xmlrpc", "ssl",
})

_SUBPROCESS_MODULES = frozenset({
    "subprocess", "multiprocessing", "concurrent", "threading",
})


# ── Static code validation ─────────────────────────────────────────────────────

def _validate_code_static(code: str, fn_name: str) -> tuple[bool, Optional[str]]:
    """
    AST-validate code without executing it.  Returns (is_valid, error_message).

    Checks:
      1. Code is syntactically valid Python.
      2. A function named `fn_name` is defined.
      3. The function imports no dangerous modules (filesystem, network, subprocess).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"Syntax error: {exc}"

    # Check function exists
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            found = True
            break

    if not found:
        return False, f"No function named '{fn_name}' found in code"

    # Collect top-level imports
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    dangerous: list[str] = []
    for mod in sorted(imported):
        category = None
        if mod in _FILESYSTEM_MODULES:
            category = "filesystem"
        elif mod in _NETWORK_MODULES:
            category = "network"
        elif mod in _SUBPROCESS_MODULES:
            category = "subprocess"
        if category:
            dangerous.append(f"{mod} ({category})")

    if dangerous:
        return False, (
            "Code imports dangerous modules (sandboxed tools must be self-contained "
            "pure Python): " + ", ".join(dangerous)
        )

    return True, None


# ── Stable meta-tool definitions ─────────────────────────────────────────────

META_TOOLS = [
    {
        "name": "search_tools",
        "description": (
            "Search the autoSkill registry for tools that match a natural-language "
            "query. Returns ranked results with similarity score, health status, and "
            "usage statistics. Use this before save_tool to check if a tool already "
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
        "name": "save_tool",
        "description": (
            "Save a brand-new tool to the registry. You (the host LLM) provide the "
            "name, description, and Python code. The server statically validates the "
            "code (must be a pure function — no filesystem, network, or subprocess "
            "imports), embeds it, and registers it. Returns the new tool_id."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Unique snake_case function name for the tool",
                },
                "description": {
                    "type": "string",
                    "description": "Plain-English description of what the tool does (used for semantic search)",
                },
                "code": {
                    "type": "string",
                    "description": "Complete Python source code defining the function",
                },
            },
            "required": ["name", "description", "code"],
        },
    },
    {
        "name": "save_tool_version",
        "description": (
            "Save a new version of an existing tool. The previous version is "
            "preserved in the version history. The tool's status is reset to "
            "'active'. Use this when a tool needs improvement or the host LLM "
            "has written a better implementation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool_id": {
                    "type": "number",
                    "description": "Numeric ID of the tool to update",
                },
                "name": {
                    "type": "string",
                    "description": "Function name (must match the existing tool's name)",
                },
                "description": {
                    "type": "string",
                    "description": "Updated description for the new version",
                },
                "code": {
                    "type": "string",
                    "description": "Complete Python source code for the new version",
                },
            },
            "required": ["tool_id", "name", "description", "code"],
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


def _handle_save_tool(req_id: Any, args: dict, registry: ToolRegistry) -> dict:
    name = args.get("name", "").strip()
    description = args.get("description", "").strip()
    code = args.get("code", "").strip()

    if not name:
        return _err(req_id, -32602, "'name' is required and must be non-empty")
    if not description:
        return _err(req_id, -32602, "'description' is required and must be non-empty")
    if not code:
        return _err(req_id, -32602, "'code' is required and must be non-empty")

    # Static validation
    valid, err_msg = _validate_code_static(code, name)
    if not valid:
        return _err(req_id, -32602, f"Code validation failed: {err_msg}")

    # Check for name collisions
    existing = registry.get_tool_by_name(name)
    if existing is not None:
        return _err(
            req_id, -32602,
            f"A tool named '{name}' already exists (id={existing.tool_id}). "
            f"Use save_tool_version to update it.",
        )

    # Embed and save
    embedding = embed(f"{name}: {description}")
    tool = Tool(name=name, description=description, code=code)

    try:
        tool_id = registry.save_tool(tool, embedding)
    except Exception as exc:
        return _err(req_id, -32603, f"Failed to save tool: {exc}")

    return _ok(req_id, _text({
        "tool_id":     tool_id,
        "name":        name,
        "status":      "active",
        "message":     f"Tool '{name}' saved successfully with id={tool_id}",
    }))


def _handle_save_tool_version(req_id: Any, args: dict, registry: ToolRegistry) -> dict:
    raw_id = args.get("tool_id")
    if raw_id is None:
        return _err(req_id, -32602, "'tool_id' is required")

    try:
        tool_id = int(raw_id)
    except (TypeError, ValueError):
        return _err(req_id, -32602, f"'tool_id' must be a number, got {raw_id!r}")

    name = args.get("name", "").strip()
    description = args.get("description", "").strip()
    code = args.get("code", "").strip()

    if not name:
        return _err(req_id, -32602, "'name' is required and must be non-empty")
    if not description:
        return _err(req_id, -32602, "'description' is required and must be non-empty")
    if not code:
        return _err(req_id, -32602, "'code' is required and must be non-empty")

    # Look up existing tool
    existing = registry.get_tool_by_id(tool_id)
    if existing is None:
        return _err(req_id, -32602, f"No tool found with id={tool_id}")

    # Name must match the existing tool
    if name != existing.name:
        return _err(
            req_id, -32602,
            f"Name '{name}' does not match existing tool name '{existing.name}'. "
            f"The function name must stay the same across versions.",
        )

    # Static validation
    valid, err_msg = _validate_code_static(code, name)
    if not valid:
        return _err(req_id, -32602, f"Code validation failed: {err_msg}")

    # Embed and update
    embedding = embed(f"{name}: {description}")
    old_version_count = len(registry.get_versions(tool_id))

    try:
        registry.update_tool(tool_id, code, description, embedding, reason="Host LLM revision")
    except Exception as exc:
        return _err(req_id, -32603, f"Failed to update tool: {exc}")

    new_version_count = len(registry.get_versions(tool_id))

    return _ok(req_id, _text({
        "tool_id":        tool_id,
        "name":           name,
        "status":         "active",
        "versions_before": old_version_count,
        "versions_after":  new_version_count,
        "message":         f"Tool '{name}' (id={tool_id}) updated — version {new_version_count} saved.",
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


def handle_tools_call(req: dict, registry: ToolRegistry) -> dict:
    req_id = req.get("id")
    params = req.get("params", {})
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})

    if tool_name == "search_tools":
        return _handle_search_tools(req_id, arguments, registry)
    if tool_name == "execute_tool":
        return _handle_execute_tool(req_id, arguments, registry)
    if tool_name == "save_tool":
        return _handle_save_tool(req_id, arguments, registry)
    if tool_name == "save_tool_version":
        return _handle_save_tool_version(req_id, arguments, registry)
    if tool_name == "tool_stats":
        return _handle_tool_stats(req_id, arguments, registry)

    return _err(req_id, -32602, f"Unknown tool: '{tool_name}'")


# ── Dispatch loop ─────────────────────────────────────────────────────────────

def _dispatch(req: dict, registry: ToolRegistry) -> dict | None:
    method = req.get("method", "")

    if method == "initialize":
        return handle_initialize(req)
    if method == "initialized":
        return None  # notification — no response
    if method == "tools/list":
        return handle_tools_list(req)
    if method == "tools/call":
        return handle_tools_call(req, registry)

    req_id = req.get("id")
    if req_id is not None:
        return _err(req_id, -32601, f"Method not found: '{method}'")
    return None


def run_server(registry: ToolRegistry) -> None:
    """Main stdio dispatch loop. Runs until EOF on stdin."""
    logger.info("autoSkill MCP server ready (protocol %s)", MCP_PROTOCOL_VERSION)
    while True:
        msg = _read_message()
        if msg is None:
            break
        response = _dispatch(msg, registry)
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
    args = parser.parse_args()

    registry = ToolRegistry(db_path=args.db)
    run_server(registry)


if __name__ == "__main__":
    main()
