"""
cli.py — command-line interface for skill-agent.

Architecture: thin wrapper over the SDK.  All business logic lives in the
skill_agent package; this file only handles terminal I/O, argument parsing,
and interactive UI.

Commands
────────
  run <query>          Stream an answer; interactive tool decision toggle
  list                 Show tools in the registry
  inspect <name>       Full details for one tool
  versions <name>      Version history for one tool
  feedback <name>      Record thumbs-up / thumbs-down
  retire <name>        Manually retire a tool
  prune                Run the pruning policies
  serve                Start the MCP stdio server

Usage
─────
  skill-agent run "Calculate compound interest on $5000 at 4.2% for 7 years"
  skill-agent run "…" --llm openai --model gpt-4o
  skill-agent list --status degraded
  skill-agent inspect calculate_compound_interest
  skill-agent feedback calculate_compound_interest --up
  skill-agent prune --stale-days 14
  skill-agent serve --db skills.db
"""
from __future__ import annotations

import argparse
import sys
import os
from typing import Optional

# ── Lazy rich import ──────────────────────────────────────────────────────────
try:
    from rich.console import Console as _RichConsole
    _console = _RichConsole(stderr=True)
    _stdout  = _RichConsole()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def _err(msg: str) -> None:
    if HAS_RICH:
        _console.print(msg)
    else:
        print(msg, file=sys.stderr)


def _out(msg: str) -> None:
    if HAS_RICH:
        _stdout.print(msg)
    else:
        print(msg)


# ── Interactive arrow-key toggle ──────────────────────────────────────────────

def _interactive_choice(prompt: str, options: list[str], descriptions: list[str]) -> str:
    """
    Arrow-key selection menu.  Returns the chosen option string.
    Falls back to numbered input when stdin is not a TTY or curses is unavailable.
    """
    if not sys.stdin.isatty():
        return _numbered_choice(prompt, options, descriptions)
    try:
        import curses
        return _curses_choice(prompt, options, descriptions)
    except Exception:
        return _numbered_choice(prompt, options, descriptions)


def _numbered_choice(prompt: str, options: list[str], descriptions: list[str]) -> str:
    print(f"\n{prompt}")
    for i, (opt, desc) in enumerate(zip(options, descriptions), 1):
        print(f"  {i}. {opt:10}  {desc}")
    while True:
        raw = input("Enter number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"  Please enter a number between 1 and {len(options)}.")


def _curses_choice(prompt: str, options: list[str], descriptions: list[str]) -> str:
    import curses

    selected = [0]

    def _menu(stdscr):
        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)

        while True:
            stdscr.clear()
            h, w = stdscr.getmaxyx()
            stdscr.addstr(0, 0, prompt, curses.A_BOLD)
            for i, (opt, desc) in enumerate(zip(options, descriptions)):
                row = 2 + i
                if row >= h:
                    break
                if i == selected[0]:
                    stdscr.attron(curses.color_pair(1))
                    stdscr.addstr(row, 2, f"► {opt:10}  {desc}")
                    stdscr.attroff(curses.color_pair(1))
                else:
                    stdscr.addstr(row, 2, f"  {opt:10}  {desc}")
            stdscr.refresh()

            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")) and selected[0] > 0:
                selected[0] -= 1
            elif key in (curses.KEY_DOWN, ord("j")) and selected[0] < len(options) - 1:
                selected[0] += 1
            elif key in (ord("\n"), ord("\r"), curses.KEY_ENTER):
                return

    curses.wrapper(_menu)
    return options[selected[0]]


# ── Tool decision handler ─────────────────────────────────────────────────────

def _make_tool_decision(no_interactive: bool):
    """Return a tool_decision callback for SkillAgent."""
    if no_interactive:
        return None   # auto-keep

    def decide(tool_name: str, code: str, validation) -> tuple[str, str]:
        print()
        _err(f"[bold]Tool created:[/bold] {tool_name}" if HAS_RICH
             else f"Tool created: {tool_name}")
        _err(f"Validation  : {validation.summary}")

        if HAS_RICH:
            from rich.syntax import Syntax
            _console.print(Syntax(code, "python", theme="monokai", line_numbers=True))
        else:
            print("\nCode:")
            for line in code.splitlines():
                print("  " + line)

        choice = _interactive_choice(
            prompt="What would you like to do?",
            options=["keep", "revise", "discard"],
            descriptions=[
                "Save and use this tool",
                "Rewrite with your feedback",
                "Answer directly without saving",
            ],
        )

        revision_notes = ""
        if choice == "revise":
            revision_notes = input("Revision instructions (or Enter to auto-revise): ").strip()

        return choice, revision_notes

    return decide


# ── Event renderer ────────────────────────────────────────────────────────────

def _make_event_handler(no_interactive: bool):
    """Return an on_event callback that renders events to the terminal."""
    from .events import EventType

    def handle(event):
        t = event.type

        if t == EventType.ROUTING_DONE:
            action = event.payload.get("action", "?")
            tool   = event.payload.get("tool_name")
            sim    = event.payload.get("similarity")
            parts  = [f"  → {action}"]
            if tool:
                parts.append(f"'{tool}'")
            if sim is not None:
                parts.append(f"(sim={sim:.2f})")
            _err("  ".join(parts))

        elif t == EventType.TOOL_FOUND:
            name = event.payload.get("name", "?")
            sim  = event.payload.get("similarity", 0)
            _err(f"  → tool found: '{name}'  (similarity={sim:.2f})")

        elif t == EventType.TOOL_WRITING:
            attempt = event.payload.get("attempt", 1)
            suffix  = f" (attempt {attempt})" if attempt > 1 else ""
            print(f"  → writing tool{suffix}…", end="\r", flush=True, file=sys.stderr)

        elif t == EventType.TOOL_WRITTEN:
            if event.payload.get("success"):
                name = event.payload.get("name", "?")
                print(f"  → tool written: '{name}'         ", file=sys.stderr)
            else:
                print("  → tool write failed              ", file=sys.stderr)

        elif t == EventType.TOOL_VALIDATING:
            name = event.payload.get("name", "?")
            print(f"  → validating '{name}'…", end="\r", flush=True, file=sys.stderr)

        elif t == EventType.TOOL_SAVED:
            tid  = event.payload.get("tool_id")
            name = event.payload.get("name", "?")
            _err(f"  → saved: '{name}' (id={tid})              ")

        elif t == EventType.TOOL_EXECUTING:
            name = event.payload.get("name", "?")
            print(f"  → executing '{name}'…", end="\r", flush=True, file=sys.stderr)

        elif t == EventType.TOOL_EXECUTED:
            name      = event.payload.get("name", "?")
            success   = event.payload.get("success")
            latency   = event.payload.get("latency_ms", 0)
            mark      = "✓" if success else "✗"
            _err(f"  {mark} executed in {latency:.0f} ms              ")

        elif t == EventType.ANSWER_CHUNK:
            chunk = event.payload.get("chunk", "")
            sys.stdout.write(chunk)
            sys.stdout.flush()

        elif t == EventType.ANSWER_DONE:
            # Final newline after streaming
            print()

        elif t == EventType.PERMISSION_REQUEST:
            perms = event.payload.get("permissions", [])
            _err("\n  [!] Tool requests the following permissions:")
            for p in perms:
                _err(f"      • {p['type']}: {p['reason']}")

        elif t == EventType.MCP_REQUIRED:
            deps = event.payload.get("dependencies", [])
            _err("\n  [!] Tool requires MCP connections:")
            for d in deps:
                _err(f"      • {d['server_name']}/{d['tool_name']}")

        elif t == EventType.ERROR:
            stage = event.payload.get("stage", "?")
            msg   = event.payload.get("message", "")
            _err(f"  [ERROR in {stage}] {msg}")

    return handle


# ── Permission handler ────────────────────────────────────────────────────────

def _interactive_permission_handler(requests):
    from .permissions import GrantedPermissions
    granted = GrantedPermissions()
    print()
    for req in requests:
        answer = input(f"  Grant '{req.type}' permission? ({req.reason}) [y/N] ").strip().lower()
        if answer == "y":
            if req.type == "network":
                granted.network = True
            elif req.type == "filesystem":
                path = input("    Path to expose (read-only): ").strip()
                if path:
                    granted.filesystem_paths.append(path)
            elif req.type == "subprocess":
                granted.subprocess = True
    return granted


# ── CLI commands ──────────────────────────────────────────────────────────────

def _cmd_run(args) -> None:
    from .agent import SkillAgent

    agent = SkillAgent(
        llm=args.llm if args.llm else None,
        llm_model=args.model,
        db_path=args.db,
        on_event=_make_event_handler(args.no_interactive),
        tool_decision=_make_tool_decision(args.no_interactive),
        use_docker=args.docker,
        permission_handler=(
            _interactive_permission_handler
            if (not args.no_interactive and sys.stdin.isatty())
            else None
        ),
    )

    print()
    result = agent.run(args.query)

    # If no streaming happened (MockClient or non-streaming path), print answer now
    if result.action_taken == "answered_directly" and not hasattr(agent.llm, "stream"):
        print(result.answer)

    # Print non-streaming tool results
    if result.action_taken in ("used_tool", "created_and_used_tool"):
        print(result.answer)

    print()
    _err(f"  action : {result.action_taken}")
    if result.tool_name:
        _err(f"  tool   : {result.tool_name}")
    if result.latency_ms:
        _err(f"  latency: {result.latency_ms:.1f} ms")


def _cmd_list(args) -> None:
    from .tool_registry import ToolRegistry
    from .inspector import view_tools
    reg = ToolRegistry(db_path=args.db)
    print(view_tools(reg, status_filter=args.status))


def _cmd_inspect(args) -> None:
    from .tool_registry import ToolRegistry
    from .inspector import inspect_tool, format_inspect
    reg = ToolRegistry(db_path=args.db)
    details = inspect_tool(reg, args.name)
    print(format_inspect(details))


def _cmd_versions(args) -> None:
    from .tool_registry import ToolRegistry
    reg = ToolRegistry(db_path=args.db)
    tool = reg.get_tool_by_name(args.name)
    if tool is None:
        print(f"Tool '{args.name}' not found.")
        sys.exit(1)
    versions = reg.get_versions(tool.tool_id)
    if not versions:
        print("No version history.")
        return
    for v in versions:
        from datetime import datetime
        ts = datetime.fromtimestamp(v.created_at).strftime("%Y-%m-%d %H:%M")
        print(f"v{v.version_num}  {ts}  [{v.reason or 'no reason'}]")
        print("  " + v.description)


def _cmd_feedback(args) -> None:
    from .tool_registry import ToolRegistry
    reg = ToolRegistry(db_path=args.db)
    tool = reg.get_tool_by_name(args.name)
    if tool is None:
        print(f"Tool '{args.name}' not found.")
        sys.exit(1)
    positive = not args.down
    reg.save_feedback(tool.tool_id, positive, args.comment or "")
    sign = "👍" if positive else "👎"
    print(f"{sign}  Feedback recorded for '{args.name}'.")


def _cmd_retire(args) -> None:
    from .tool_registry import ToolRegistry
    reg = ToolRegistry(db_path=args.db)
    tool = reg.get_tool_by_name(args.name)
    if tool is None:
        print(f"Tool '{args.name}' not found.")
        sys.exit(1)
    reg.retire_tool(tool.tool_id)
    print(f"Tool '{args.name}' retired.")


def _cmd_prune(args) -> None:
    from .tool_registry import ToolRegistry
    reg = ToolRegistry(db_path=args.db)
    retired = reg.prune(stale_days=args.stale_days)
    if not retired:
        print("Nothing to prune.")
        return
    print(f"Retired {len(retired)} tool(s):")
    for r in retired:
        print(f"  • {r['name']:40}  {r['reason']}")


def _cmd_serve(args) -> None:
    from .mcp_server import run_server
    from .tool_registry import ToolRegistry
    reg = ToolRegistry(db_path=args.db)

    agent = None
    if getattr(args, "llm", None):
        from .agent import SkillAgent
        agent = SkillAgent(
            llm=args.llm,
            llm_model=getattr(args, "llm_model", None),
            llm_api_key=getattr(args, "llm_api_key", None),
            db_path=args.db,
        )

    mode = f"llm={args.llm}" if agent else "registry-only"
    _err(f"MCP server starting on stdio (db={args.db}, {mode})")
    run_server(reg, agent)


def _cmd_serve_http(args) -> None:
    try:
        import uvicorn
    except ImportError:
        _err("uvicorn not found. Install with: pip install 'skill-agent[http]'")
        sys.exit(1)

    from .config import AutoSkillConfig
    from .http_server import create_app

    cfg = AutoSkillConfig.from_env()
    if getattr(args, "llm", None):
        cfg.llm_provider = args.llm
    if getattr(args, "llm_model", None):
        cfg.llm_model = args.llm_model
    if args.db != "skills.db":          # user passed an explicit --db
        cfg.db_path = args.db
    if getattr(args, "host", None):
        cfg.http_host = args.host
    if getattr(args, "port", None):
        cfg.http_port = args.port

    agent = cfg.create_agent()
    app   = create_app(agent)
    _err(f"HTTP server → http://{cfg.http_host}:{cfg.http_port}  "
         f"(provider={cfg.llm_provider}, db={cfg.db_path})")
    uvicorn.run(app, host=cfg.http_host, port=cfg.http_port)


def _cmd_export(args) -> None:
    from .tool_registry import ToolRegistry
    from .inspector import export_json
    reg = ToolRegistry(db_path=args.db)
    print(export_json(reg))


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skill-agent",
        description="Self-improving tool-learning AI agent",
    )
    parser.add_argument("--db", default="skills.db", help="Path to skills database")

    sub = parser.add_subparsers(dest="command", required=True)

    # ── run ──────────────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Run a query")
    p_run.add_argument("query", help="Natural language query")
    p_run.add_argument("--llm", choices=["anthropic", "openai", "ollama", "mock"],
                       default=None, help="LLM provider (default: anthropic)")
    p_run.add_argument("--model", default=None,
                       help="Model name override (e.g. gpt-4o)")
    p_run.add_argument("--docker", action="store_true",
                       help="Execute tools in Docker container")
    p_run.add_argument("--no-interactive", action="store_true",
                       help="Disable interactive tool-decision prompt (auto-keep)")

    # ── list ─────────────────────────────────────────────────────────────────
    p_list = sub.add_parser("list", help="List tools in the registry")
    p_list.add_argument("--status",
                        choices=["active", "degraded", "retired", "all"],
                        default=None)

    # ── inspect ──────────────────────────────────────────────────────────────
    p_inspect = sub.add_parser("inspect", help="Full details for one tool")
    p_inspect.add_argument("name")

    # ── versions ─────────────────────────────────────────────────────────────
    p_ver = sub.add_parser("versions", help="Show version history for a tool")
    p_ver.add_argument("name")

    # ── feedback ─────────────────────────────────────────────────────────────
    p_fb = sub.add_parser("feedback", help="Record feedback for a tool")
    p_fb.add_argument("name")
    p_fb.add_argument("--up",   dest="down", action="store_false", default=False,
                      help="Positive feedback (default)")
    p_fb.add_argument("--down", dest="down", action="store_true",
                      help="Negative feedback")
    p_fb.add_argument("--comment", default="", help="Optional comment")

    # ── retire ───────────────────────────────────────────────────────────────
    p_retire = sub.add_parser("retire", help="Retire a tool")
    p_retire.add_argument("name")

    # ── prune ────────────────────────────────────────────────────────────────
    p_prune = sub.add_parser("prune", help="Run pruning policies")
    p_prune.add_argument("--stale-days", type=int, default=30,
                         help="Days of inactivity before a tool is stale (default: 30)")

    # ── serve ────────────────────────────────────────────────────────────────
    p_serve = sub.add_parser("serve", help="Start the MCP stdio server")
    p_serve.add_argument("--llm", choices=["anthropic", "openai", "ollama", "mock"],
                         default=None,
                         help="Enable create_tool/improve_tool via this LLM provider")
    p_serve.add_argument("--llm-model", dest="llm_model", default=None,
                         help="Model name override")
    p_serve.add_argument("--llm-api-key", dest="llm_api_key", default=None,
                         help="API key override (falls back to env vars)")

    # ── serve-http ───────────────────────────────────────────────────────────
    p_http = sub.add_parser("serve-http", help="Start the HTTP REST API server")
    p_http.add_argument("--host", default=None,
                        help="Bind host (default: 0.0.0.0)")
    p_http.add_argument("--port", type=int, default=None,
                        help="Port (default: 8000)")
    p_http.add_argument("--llm",
                        choices=["anthropic", "openai", "ollama", "mock"],
                        default=None, help="LLM provider")
    p_http.add_argument("--llm-model", dest="llm_model", default=None,
                        help="Model name override")

    # ── export ───────────────────────────────────────────────────────────────
    sub.add_parser("export", help="Export all tools as JSON to stdout")

    return parser


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    dispatch = {
        "run":      _cmd_run,
        "list":     _cmd_list,
        "inspect":  _cmd_inspect,
        "versions": _cmd_versions,
        "feedback": _cmd_feedback,
        "retire":   _cmd_retire,
        "prune":    _cmd_prune,
        "serve":      _cmd_serve,
        "serve-http": _cmd_serve_http,
        "export":     _cmd_export,
    }

    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
