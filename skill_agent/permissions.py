"""
permissions.py — tool permission detection and grant tracking.

Before executing a generated tool, the agent scans its imports to determine
what OS-level access it needs.  These requirements are surfaced as
PermissionRequest objects; the caller (or a registered handler) decides
what to grant via GrantedPermissions.

If no handler is registered the defaults are:
  - filesystem : denied  (Docker/subprocess: no volume mounts)
  - network    : denied  (Docker: --network none)
  - subprocess : denied  (not injected into harness)

MCP dependencies (mcp_call() invocations) are also detected here so the
agent can emit MCP_REQUIRED events.  Actual MCP connections are not yet
established at execution time — this is scaffolding for a future release.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PermissionRequest:
    type: str           # "filesystem" | "network" | "subprocess"
    reason: str         # human-readable: which module triggered this
    required: bool = True


@dataclass
class GrantedPermissions:
    filesystem_paths: List[str] = field(default_factory=list)
    network: bool = False
    subprocess: bool = False


@dataclass
class MCPDependency:
    server_name: str
    tool_name: str
    reason: str


# ── Module category maps ──────────────────────────────────────────────────────

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
    "os",   # also listed under filesystem — dual category
})


# ── Scanners ──────────────────────────────────────────────────────────────────

def scan_permissions(code: str) -> List[PermissionRequest]:
    """
    AST-scan `code` for imports that require OS-level permissions.

    Returns a deduplicated list of PermissionRequest objects.
    Returns [] on SyntaxError (the validator will catch bad code separately).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    requests: List[PermissionRequest] = []
    seen: set[str] = set()

    fs_hits = imported & _FILESYSTEM_MODULES - {"os"}  # os handled below
    net_hits = imported & _NETWORK_MODULES
    sub_hits = imported & _SUBPROCESS_MODULES

    if fs_hits or "os" in imported:
        modules = (fs_hits | ({"os"} if "os" in imported else set()))
        requests.append(PermissionRequest(
            type="filesystem",
            reason=f"Imports: {', '.join(sorted(modules))}",
        ))
        seen.add("filesystem")

    if net_hits:
        requests.append(PermissionRequest(
            type="network",
            reason=f"Imports: {', '.join(sorted(net_hits))}",
        ))
        seen.add("network")

    if sub_hits - ({"os"} if "os" in imported else set()):
        actual = sub_hits - {"os"}
        if actual:
            requests.append(PermissionRequest(
                type="subprocess",
                reason=f"Imports: {', '.join(sorted(actual))}",
            ))

    return requests


def scan_mcp_deps(code: str) -> List[MCPDependency]:
    """
    Detect `mcp_call("server_name", "tool_name", …)` calls in the AST.

    This is scaffolding for a future release where tools can declare
    MCP dependencies.  Currently emits MCP_REQUIRED events but does not
    establish actual connections.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    deps: List[MCPDependency] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "mcp_call"
            and len(node.args) >= 2
        ):
            server = ast.literal_eval(node.args[0]) if isinstance(node.args[0], ast.Constant) else "?"
            tool   = ast.literal_eval(node.args[1]) if isinstance(node.args[1], ast.Constant) else "?"
            deps.append(MCPDependency(
                server_name=str(server),
                tool_name=str(tool),
                reason=f"mcp_call('{server}', '{tool}', …) at line {node.lineno}",
            ))
    return deps
