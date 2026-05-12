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
- Apply a discount schedule to a price list

## How to use

autoSkill exposes five stable methods through its MCP server.
You never interact with generated tools directly — you go through these primitives.

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

### 3. `create_tool` — generate a new tool

Only use when `search_tools` finds nothing suitable.

```json
{ "task": "convert a temperature from fahrenheit to celsius" }
```

autoSkill will write the Python function, validate it, and save it to the
registry so it can be retrieved by future `search_tools` calls.

### 4. `improve_tool` — revise a degraded tool

```json
{ "tool_id": 7 }
```

Triggers an LLM revision. The original version is preserved in history.

### 5. `tool_stats` — inspect health

```json
{ "tool_id": 7 }
```

Returns status, success rate, usage count, and user sentiment score.

## What autoSkill is NOT for

- Conversational or subjective questions — answer those directly
- Tasks requiring live data (API calls, web fetches) — unless the tool is
  explicitly permitted network access
- One-off queries unlikely to recur — the overhead of tool creation is not
  worth it

## Starting the MCP server

```bash
# Registry-only (search + execute, no LLM generation)
python -m skill_agent.mcp_server --db skills.db

# With Anthropic (enables create_tool + improve_tool)
python -m skill_agent.mcp_server --db skills.db --llm anthropic

# With OpenAI
python -m skill_agent.mcp_server --db skills.db --llm openai --llm-model gpt-4o
```

## Architecture note

The intelligence lives in the MCP server, not in this skill file.
This file is a routing hint — it tells you when and how to reach autoSkill.
The tool registry, vector search, sandbox, and self-improvement loop all
run inside the server process.
