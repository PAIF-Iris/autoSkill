---
name: autoskill
description: >
  Persistent self-improving tool runtime. Provides a dynamic registry of
  Python tools that are generated on demand, validated, sandboxed, and
  reused via semantic search. Connect via the autoSkill MCP server.
---

# autoSkill — Persistent Tool Runtime

Use autoSkill for tasks that are:

- **Deterministic** — same inputs always produce the same output
- **Reusable** — worth caching as a function for future queries
- **Procedural** — data transformation, math, parsing, formatting, conversion

Examples:
- Convert CSV to JSON
- Compute compound interest
- Extract emails from text
- Parse and reformat dates

## Quick start

Connect to autoSkill as an MCP server. Five tools, one workflow:

```
search_tools  →  is there already a tool that does this?
save_tool     →  no? create one (write the Python code yourself)
execute_tool  →  run it with test inputs
save_tool_version →  found a bug or want to improve? update it
tool_stats    →  check health and usage
```

## MCP tools

### 1. `search_tools` — check before creating

Always search first. A matching tool may already exist.

```json
{ "query": "convert fahrenheit to celsius" }
```

Returns ranked results with `tool_id`, similarity score, and health status.

### 2. `execute_tool` — run a matched tool

```json
{ "tool_id": 7, "args": { "fahrenheit": 98.6 } }
```

Runs in a sandboxed subprocess with a 10-second timeout.

### 3. `save_tool` — register a new tool

Use when `search_tools` finds nothing suitable. You (the host LLM) write the
Python function and provide the name, description, and code. The server
validates that the code is a safe pure function before saving.

```json
{
  "name": "fahrenheit_to_celsius",
  "description": "Convert a temperature from Fahrenheit to Celsius",
  "code": "def fahrenheit_to_celsius(fahrenheit: float) -> float:\n    \"\"\"Convert Fahrenheit to Celsius.\"\"\"\n    return (fahrenheit - 32) * 5 / 9"
}
```

### 4. `save_tool_version` — update an existing tool

When a tool needs improvement, provide the new code. The previous version is
preserved in history and the tool's status is reset to "active".

```json
{
  "tool_id": 7,
  "name": "fahrenheit_to_celsius",
  "description": "Convert Fahrenheit to Celsius with input validation",
  "code": "def fahrenheit_to_celsius(fahrenheit: float) -> float:\n    ..."
}
```

### 5. `tool_stats` — inspect health

```json
{ "tool_id": 7 }
```

Returns status, success rate, usage count, and user sentiment score.

## What autoSkill is NOT for

- Conversational or subjective questions — answer those directly
- Tasks requiring live data (API calls, web fetches) — tools must be pure functions
- One-off queries unlikely to recur — the overhead of tool creation is not worth it

## Connecting

**Remote (Linode / any server):**

```json
{
  "mcpServers": {
    "autoskill": {
      "type": "http",
      "url": "http://<host>:8000/mcp"
    }
  }
}
```

Start the server: `skill-agent serve-mcp --host 0.0.0.0 --port 8000`

**Local (stdio, same machine):**

```json
{
  "mcpServers": {
    "autoskill": {
      "command": "skill-agent",
      "args": ["serve"]
    }
  }
}
```

The MCP server is a **dumb runtime** — it stores, searches, and executes tools
but makes zero AI decisions. All intelligence (when to create, reuse, or
revise tools) lives in you, the host LLM.
