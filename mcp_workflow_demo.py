"""
mcp_workflow_demo.py — Connect to the autoSkill MCP server and have Claude
drive a full round using all four tools:

  1. search_tools   — check if a tool already exists
  2. save_tool      — create a new tool (Claude writes the code)
  3. execute_tool   — run it
  4. save_tool_version — improve it

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python mcp_workflow_demo.py

Defaults to http://172.105.105.81:8000/mcp — override with MCP_URL env var.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

import anthropic
from mcp.client.streamable_http import streamable_http_client
from mcp.client.session import ClientSession


MCP_URL = os.environ.get("MCP_URL", "http://172.105.105.81:8000/mcp")

SYSTEM_PROMPT = """\
You are testing the autoSkill tool registry.  The user wants you to perform a
full end-to-end walkthrough that exercises every available tool.  Follow these
steps in order:

1. search_tools — look for a "celsius to fahrenheit" converter
   (it probably doesn't exist yet, but always CHECK FIRST — this is critical)

2. save_tool — create a working Python function that converts Celsius to
   Fahrenheit.  Write clean, correct code.  The formula is: F = C * 9/5 + 32.

3. execute_tool — run it with C=100 to verify it returns 212.

4. save_tool_version — improve it by adding proper type hints, a docstring,
   and input validation (reject non-numeric inputs).

Report what happened at each step.  If anything fails, explain why.
"""

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")


def mcp_tool_to_anthropic(tool) -> dict:
    """Convert an MCP tool definition to Anthropic-compatible format."""
    schema = tool.inputSchema
    props = schema.get("properties", {})
    required = schema.get("required", [])

    # Build clean property dict for Anthropic
    clean_props = {}
    for name, prop in props.items():
        entry = {
            "type": prop.get("type", "string"),
            "description": prop.get("description", prop.get("title", "")),
        }
        if "default" in prop:
            entry["default"] = prop["default"]
        clean_props[name] = entry

    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": {
            "type": "object",
            "properties": clean_props,
            "required": required,
        },
    }


async def main():
    print(f"Connecting to MCP server: {MCP_URL}")
    print(f"Model: {ANTHROPIC_MODEL}")
    print()

    # ── 1. Connect to MCP server ──────────────────────────────────────────
    client = anthropic.AsyncAnthropic()

    async with streamable_http_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            # Initialize
            init_result = await session.initialize()
            print(f"Connected to {init_result.serverInfo.name} "
                  f"v{init_result.serverInfo.version}")
            print()

            # Fetch tool list
            tools_result = await session.list_tools()
            mcp_tools = tools_result.tools
            print(f"Server exposes {len(mcp_tools)} tools:")
            for t in mcp_tools:
                print(f"  • {t.name} — {t.description[:80]}...")
            print()

            # ── 2. Convert to Anthropic tool format ────────────────────────
            anthropic_tools = [mcp_tool_to_anthropic(t) for t in mcp_tools]

            # Make the operation tool (not a real tool — it marks completion)
            anthropic_tools.append({
                "name": "report_complete",
                "description": (
                    "Call this when you have finished all steps. Summarize "
                    "what happened with each tool."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Summary of what happened at each step",
                        },
                    },
                    "required": ["summary"],
                },
            })

            # ── 3. Claude conversation loop ────────────────────────────────
            messages = [
                {"role": "user", "content": (
                    "Walk through the full tool lifecycle: search for a "
                    "celsius-to-fahrenheit converter, create it if needed, "
                    "execute it, then improve it. Call each tool exactly once."
                )},
            ]

            while True:
                response = await client.messages.create(
                    model=ANTHROPIC_MODEL,
                    max_tokens=2048,
                    system=SYSTEM_PROMPT,
                    tools=anthropic_tools,
                    messages=messages,
                )

                # Print text responses
                for block in response.content:
                    if block.type == "text":
                        print(f"[Claude] {block.text}")

                # Check for tool calls
                tool_calls = [
                    block for block in response.content
                    if block.type == "tool_use"
                ]

                if not tool_calls:
                    print("\nDone — no more tool calls.")
                    break

                # Add assistant response to history
                messages.append({
                    "role": "assistant",
                    "content": response.content,
                })

                tool_results = []
                for tool_block in tool_calls:
                    name = tool_block.name
                    args = tool_block.input or {}

                    print(f"\n{'─'*60}")
                    print(f"[tool] {name}")

                    if name == "report_complete":
                        print(f"  summary: {args.get('summary', '')}")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": "Report received. Workflow complete.",
                        })
                    elif name in {"search_tools", "execute_tool",
                                  "save_tool", "save_tool_version",
                                  "tool_stats"}:
                        try:
                            result = await session.call_tool(name, args)
                            # Extract text content
                            text = ""
                            for item in result.content:
                                if item.type == "text":
                                    text += item.text
                            print(f"  result: {text[:500]}")
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_block.id,
                                "content": text,
                            })
                        except Exception as exc:
                            err_msg = f"Error: {exc}"
                            print(f"  ERROR: {err_msg}")
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_block.id,
                                "content": err_msg,
                            })
                    else:
                        # Unknown tool
                        err_msg = f"Unknown tool: {name}"
                        print(f"  {err_msg}")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": err_msg,
                        })

                messages.append({
                    "role": "user",
                    "content": tool_results,
                })

            print(f"\n{'─'*60}")
            print("Workflow complete.")


if __name__ == "__main__":
    asyncio.run(main())
