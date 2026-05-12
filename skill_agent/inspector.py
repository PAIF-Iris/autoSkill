"""
inspector.py — human-readable views of the tool registry database.

All functions accept a ToolRegistry instance and return strings or dicts
so they can be used from both the SDK and the CLI without I/O coupling.

Functions
─────────
  view_tools(registry, status_filter)  → formatted ASCII/rich table string
  inspect_tool(registry, name)          → dict with full tool details
  export_json(registry)                 → JSON string of all tools
"""
from __future__ import annotations

import json
import textwrap
from datetime import datetime
from typing import Optional

from .tool_registry import ToolRegistry


def _fmt_time(ts: Optional[float]) -> str:
    if ts is None:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _bar(rate: float, width: int = 10) -> str:
    filled = round(rate * width)
    return "█" * filled + "░" * (width - filled)


def view_tools(
    registry: ToolRegistry,
    status_filter: Optional[str] = None,
) -> str:
    """
    Return a formatted table of tools in the registry.

    Parameters
    ----------
    registry      : ToolRegistry instance
    status_filter : "active" | "degraded" | "retired" | None (= all non-retired)

    Returns a multi-line string ready for print().
    Uses `rich` tables if the package is available; falls back to plain ASCII.
    """
    include_retired = status_filter in ("retired", "all")
    tools = registry.list_tools(include_retired=include_retired)

    if status_filter and status_filter not in ("all", None):
        tools = [t for t in tools if t.status == status_filter]

    if not tools:
        return "(no tools match the filter)"

    try:
        return _view_tools_rich(tools, registry)
    except ImportError:
        return _view_tools_plain(tools, registry)


def _view_tools_rich(tools, registry) -> str:
    from rich.table import Table
    from rich.console import Console
    from io import StringIO

    table = Table(title="Skill Registry", show_lines=False)
    table.add_column("ID",          style="dim",     width=4)
    table.add_column("Name",        style="bold",    min_width=30)
    table.add_column("Status",      width=9)
    table.add_column("Uses",        justify="right", width=5)
    table.add_column("Success",     width=14)
    table.add_column("Feedback",    width=6)
    table.add_column("Created",     width=17)
    table.add_column("Last Used",   width=17)

    STATUS_COLOR = {"active": "green", "degraded": "yellow", "retired": "red"}

    for t in tools:
        sentiment = registry.get_user_sentiment(t.tool_id) if t.tool_id else None
        sentiment_str = f"{sentiment:.0%}" if sentiment is not None else "—"
        color = STATUS_COLOR.get(t.status, "white")
        table.add_row(
            str(t.tool_id or "?"),
            t.name,
            f"[{color}]{t.status}[/{color}]",
            str(t.usage_count),
            f"{_bar(t.success_rate)} {t.success_rate:.0%}",
            sentiment_str,
            _fmt_time(t.created_at),
            _fmt_time(t.last_used_at),
        )

    buf = StringIO()
    console = Console(file=buf, highlight=False)
    console.print(table)
    return buf.getvalue()


def _view_tools_plain(tools, registry) -> str:
    col_w = [4, 32, 9, 6, 14, 8, 17, 17]
    headers = ["ID", "Name", "Status", "Uses", "Success", "Feedbk", "Created", "Last Used"]
    sep = "  ".join("─" * w for w in col_w)

    def row(*cells):
        return "  ".join(str(c).ljust(w)[:w] for c, w in zip(cells, col_w))

    lines = [row(*headers), sep]
    for t in tools:
        sentiment = registry.get_user_sentiment(t.tool_id) if t.tool_id else None
        lines.append(row(
            t.tool_id or "?",
            t.name,
            t.status,
            t.usage_count,
            f"{t.success_rate:.0%}",
            f"{sentiment:.0%}" if sentiment is not None else "—",
            _fmt_time(t.created_at),
            _fmt_time(t.last_used_at),
        ))
    return "\n".join(lines)


def inspect_tool(registry: ToolRegistry, name: str) -> dict:
    """
    Return a dict with full details of a single tool, including:
      - metadata (name, description, status, stats)
      - full code
      - version history (list of dicts)
      - feedback summary

    Returns None if the tool is not found.
    """
    tool = registry.get_tool_by_name(name)
    if tool is None:
        return None

    versions = []
    if tool.tool_id is not None:
        for v in registry.get_versions(tool.tool_id):
            versions.append({
                "version_num": v.version_num,
                "description": v.description,
                "reason":      v.reason,
                "created_at":  _fmt_time(v.created_at),
                "code":        v.code,
            })

    sentiment = registry.get_user_sentiment(tool.tool_id) if tool.tool_id else None

    return {
        "id":           tool.tool_id,
        "name":         tool.name,
        "description":  tool.description,
        "status":       tool.status,
        "usage_count":  tool.usage_count,
        "success_rate": f"{tool.success_rate:.1%}",
        "sentiment":    f"{sentiment:.1%}" if sentiment is not None else None,
        "created_at":   _fmt_time(tool.created_at),
        "last_used_at": _fmt_time(tool.last_used_at),
        "code":         tool.code,
        "versions":     versions,
    }


def format_inspect(details: dict) -> str:
    """Format the dict returned by inspect_tool() for terminal display."""
    if details is None:
        return "Tool not found."

    lines = [
        f"  Name        : {details['name']}",
        f"  ID          : {details['id']}",
        f"  Status      : {details['status']}",
        f"  Description : {details['description']}",
        f"  Uses        : {details['usage_count']}",
        f"  Success     : {details['success_rate']}",
        f"  Feedback    : {details['sentiment'] or '(none)'}",
        f"  Created     : {details['created_at']}",
        f"  Last used   : {details['last_used_at']}",
        "",
        "  Code:",
    ]
    for line in details["code"].splitlines():
        lines.append("    " + line)

    if details["versions"]:
        lines.append("")
        lines.append(f"  Version history ({len(details['versions'])} revision(s)):")
        for v in details["versions"]:
            lines.append(
                f"    v{v['version_num']}  {v['created_at']}  "
                f"[{v['reason'] or 'no reason'}]"
            )

    return "\n".join(lines)


def export_json(registry: ToolRegistry) -> str:
    """Export all tools (including retired) as a JSON string."""
    tools = registry.list_tools(include_retired=True)
    data = []
    for t in tools:
        details = inspect_tool(registry, t.name)
        if details:
            data.append(details)
    return json.dumps(data, indent=2)
