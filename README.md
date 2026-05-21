# autoSkill

**Executable semantic memory for AI agents.**

autoSkill is a self-improving tool-learning runtime that makes AI agents faster and more reliable by building a persistent library of Python tools from the queries they handle.

Instead of sending every request to an LLM, autoSkill:

1. Searches a vector registry of previously generated tools
2. Executes a matching tool directly (deterministic, milliseconds, no API cost)
3. Falls back to the LLM only when no matching tool exists
4. Saves the new tool to the registry so it can be reused

Over time, the system handles more and more queries without LLM calls.

```
User query
    │
    ▼
Vector search (FAISS)
    │
    ├── High match → Execute tool directly (< 100ms, no LLM)
    ├── Mid match  → LLM confirms → Execute tool
    └── No match   → LLM classifies
                        │
                        ├── Deterministic? → Generate tool → Validate → Save → Execute
                        └── Conversational → LLM answers directly
```

---

## Why deterministic execution matters

Most agent frameworks send every query to an LLM. This is expensive, slow, and inconsistent — the same calculation can return slightly different results each time.

autoSkill separates two fundamentally different kinds of work:

| Kind | Examples | Best handled by |
|---|---|---|
| Deterministic | unit conversion, compound interest, CSV parsing, date arithmetic | Python function (fast, exact, free) |
| Generative | explanation, analysis, creative writing, open-ended questions | LLM |

Once a tool is written and validated, it runs in a sandboxed subprocess in under 100ms and costs nothing. The LLM is only involved the first time.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    autoSkill Runtime                     │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │ Recognizer│───▶│ Router   │───▶│  Tool Generator  │  │
│  │ (FAISS + │    │          │    │  (LLM → Python)  │  │
│  │  LLM)    │    └──────────┘    └──────────────────┘  │
│  └──────────┘          │                   │            │
│                        │                   ▼            │
│  ┌─────────────────────▼──┐    ┌──────────────────┐    │
│  │   Tool Registry        │    │   Validator +     │    │
│  │   SQLite + FAISS       │◀───│   Sandbox Exec    │    │
│  │   health scoring       │    └──────────────────┘    │
│  └────────────────────────┘                             │
│                                                         │
│  Interfaces: Python SDK │ HTTP API │ MCP (stdio + HTTP) │ CLI  │
└─────────────────────────────────────────────────────────┘
```

### Core modules

| Module | Role |
|---|---|
| `agent.py` | `SkillAgent` — top-level orchestrator |
| `recognizer.py` | Routes queries via vector search + LLM classification |
| `tool_writer.py` | LLM generates a validated Python function |
| `validator.py` | LLM generates test cases; tool must pass all of them |
| `executor.py` | Subprocess sandbox with timeout, output capture |
| `tool_registry.py` | SQLite (metadata) + FAISS (vector search) |
| `embeddings.py` | sentence-transformers `all-MiniLM-L6-v2` (384d) |
| `reviser.py` | Auto-rewrites degraded tools after negative feedback |
| `providers/` | `BaseProvider` ABC + Anthropic, OpenAI, Ollama, Mock |
| `config.py` | `AutoSkillConfig` dataclass with `from_env()` |
| `http_server.py` | FastAPI REST API |
| `mcp_server.py` | MCP stdio server (5 stable meta-tools) |
| `mcp_http_server.py` | MCP HTTP server for remote clients (Streamable HTTP) |
| `cli.py` | `skill-agent` CLI |

---

## Installation

```bash
# Core (Anthropic provider)
pip install skill-agent

# With OpenAI support
pip install 'skill-agent[openai]'

# With HTTP REST API
pip install 'skill-agent[http]'

# With MCP HTTP server (for remote clients)
pip install 'skill-agent[mcp]'

# Everything
pip install 'skill-agent[all]'
```

From source:
```bash
git clone https://github.com/your-org/autoskill
cd autoskill
pip install -e '.[all]'
```

---

## Python SDK

### Basic usage

```python
from skill_agent import SkillAgent

agent = SkillAgent()   # reads ANTHROPIC_API_KEY from env

result = agent.run("Convert 98.6°F to Celsius")
print(result.answer)        # 37.0
print(result.action_taken)  # "created_and_used_tool" or "used_tool"
print(result.latency_ms)    # e.g. 87.3
```

### With config helper

```python
from skill_agent import AutoSkillConfig

cfg   = AutoSkillConfig.from_env()          # reads SKILL_AGENT_* env vars
agent = cfg.create_agent(db_path="prod.db")
result = agent.run("What is 15% of $240?")
```

### Real-time streaming

```python
from skill_agent import SkillAgent, EventType

def on_event(event):
    if event.type == EventType.ANSWER_CHUNK:
        print(event.payload["chunk"], end="", flush=True)
    elif event.type == EventType.ROUTING_DONE:
        print(f"\n[→ {event.payload['action']}]", flush=True)

agent = SkillAgent(on_event=on_event)
agent.run("Explain the difference between APR and APY")
```

### Feedback (drives self-improvement)

```python
result = agent.run("Convert 100 USD to EUR at 0.92 rate")
agent.feedback(result, positive=True)

# Negative feedback on a degraded tool triggers automatic revision
agent.feedback(result, positive=False, comment="Wrong decimal places")
```

### Full `AgentResult` fields

```python
result.answer            # Any — the answer (tool output or LLM text)
result.action_taken      # str — see action constants below
result.tool_name         # Optional[str] — tool used or created
result.validation_passed # Optional[bool] — None for direct answers
result.latency_ms        # Optional[float] — tool execution time
result.notes             # list[str] — audit trail
```

Action constants: `answered_directly`, `used_tool`, `created_and_used_tool`,
`used_tool_then_answered_directly`, `created_tool_then_answered_directly`.

---

## Provider integrations

### Anthropic (default)

```python
from skill_agent import SkillAgent

agent = SkillAgent()
# or explicitly:
agent = SkillAgent(llm="anthropic", llm_model="claude-opus-4-7")
```

Env var: `ANTHROPIC_API_KEY`

### OpenAI

```python
agent = SkillAgent(llm="openai", llm_model="gpt-4o")
```

Env var: `OPENAI_API_KEY`  
Requires: `pip install 'skill-agent[openai]'`

### Ollama (local models — no API key)

```bash
# Install Ollama: https://ollama.com
ollama pull llama3
ollama serve
```

```python
from skill_agent.providers import OllamaProvider
from skill_agent import SkillAgent

agent = SkillAgent(llm_client=OllamaProvider("llama3"))
```

Or via CLI:
```bash
skill-agent run "Calculate compound interest on $1000 at 5% for 10 years" \
    --llm ollama --model llama3
```

### Custom provider

```python
from skill_agent.providers import BaseProvider
from skill_agent import SkillAgent

class MyProvider(BaseProvider):
    def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        # call your LLM here
        return my_llm.generate(system_prompt=system, user_prompt=user)

agent = SkillAgent(llm_client=MyProvider())
```

Only `.complete()` is required. Override `.stream()` for token-level streaming.

---

## HTTP API

Start the server:
```bash
# With Anthropic:
skill-agent serve-http --port 8000 --llm anthropic

# With Ollama:
skill-agent serve-http --port 8000 --llm ollama --llm-model llama3

# Via env vars only:
export SKILL_AGENT_LLM_PROVIDER=anthropic
export SKILL_AGENT_HTTP_PORT=8000
skill-agent serve-http
```

### Endpoints

#### `POST /run`
Route a query through the full agent pipeline.

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"query": "Convert 100 miles to kilometers"}'
```

```json
{
  "answer": 160.934,
  "action_taken": "used_tool",
  "tool_name": "miles_to_kilometers",
  "latency_ms": 54.2,
  "notes": ["Using tool 'miles_to_kilometers' (similarity=0.94, status=active)"]
}
```

#### `POST /tools/search`
Semantic search over the registry.

```bash
curl -X POST http://localhost:8000/tools/search \
  -H "Content-Type: application/json" \
  -d '{"query": "distance unit conversion", "top_k": 5}'
```

#### `GET /tools`
List all active tools.

#### `GET /tools/{id}`
Full details for one tool, including source code.

#### `POST /tools/{id}/feedback`
Record thumbs up/down.

```bash
curl -X POST http://localhost:8000/tools/3/feedback \
  -H "Content-Type: application/json" \
  -d '{"positive": true, "comment": "Accurate result"}'
```

#### `GET /health`
Liveness check.

### Using as middleware

The HTTP API is designed to sit in front of your LLM provider:

```python
import httpx

class AutoSkillMiddleware:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def run(self, query: str) -> dict:
        resp = httpx.post(f"{self.base_url}/run", json={"query": query})
        return resp.json()

middleware = AutoSkillMiddleware()
result = middleware.run("Convert 72°F to Celsius")

if result["action_taken"] in ("used_tool", "created_and_used_tool"):
    # Tool handled it — no LLM needed
    answer = result["answer"]
else:
    # Fall through to your LLM
    answer = your_llm.generate(query)
```

### Interactive docs

Visit `http://localhost:8000/docs` for the Swagger UI after starting the server.

---

## MCP Server

autoSkill exposes **five stable meta-tools** via the Model Context Protocol,
compatible with Claude Desktop, Claude Code, OpenClaw, Cursor, VS Code agents,
and any MCP-aware runtime. The MCP server is a **dumb runtime** — all
intelligence (decision-making, code generation) lives in the host LLM.

### Two ways to run

| Mode | Command | When to use |
|---|---|---|
| **Local (stdio)** | `skill-agent serve` | Same machine — Claude Desktop, VS Code |
| **Remote (HTTP)** | `skill-agent serve-mcp` | Remote server — Linode, Cloudflare Tunnel |

### Remote setup (MCP HTTP)

```bash
# On your server
pip install 'skill-agent[mcp]'
skill-agent serve-mcp --host 0.0.0.0 --port 8000
```

Client config:
```json
{
  "mcpServers": {
    "autoskill": {
      "type": "http",
      "url": "http://<server-ip>:8000/mcp"
    }
  }
}
```

### Local setup (MCP stdio)

```bash
skill-agent serve
# or
python -m skill_agent.mcp_server --db skills.db
```

Client config:
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

### MCP tools

| Tool | Arguments | Description |
|---|---|---|
| `search_tools` | `query`, `top_k` | Semantic search — call this first |
| `execute_tool` | `tool_id`, `args` | Run a tool by its numeric ID |
| `save_tool` | `name`, `description`, `code` | Register a new tool (host provides code) |
| `save_tool_version` | `tool_id`, `name`, `description`, `code` | Update a tool, snapshotting the old version |
| `tool_stats` | `tool_id` | Health, usage, and sentiment stats |

### Testing with an LLM driver

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export MCP_URL=http://172.105.105.81:8000/mcp   # your server
python mcp_workflow_demo.py
```

This script connects Claude to the remote MCP server and walks through the full
tool lifecycle: search → create → execute → improve.

The `SKILL.md` file contains the prompt-level routing hint for the host LLM.

---

## CLI reference

```bash
skill-agent --help

# Run a query interactively (shows generated code, lets you keep/revise/discard)
skill-agent run "Calculate compound interest on $5000 at 4.2% for 7 years"
skill-agent run "..." --llm openai --model gpt-4o
skill-agent run "..." --llm ollama --model llama3
skill-agent run "..." --no-interactive   # auto-keep generated tools

# Inspect the registry
skill-agent list
skill-agent list --status degraded
skill-agent inspect calculate_compound_interest
skill-agent versions calculate_compound_interest

# Manage tools
skill-agent feedback calculate_compound_interest --up
skill-agent feedback calculate_compound_interest --down --comment "wrong result"
skill-agent retire my_old_tool
skill-agent prune --stale-days 14
skill-agent export > tools.json

# Start servers
skill-agent serve                           # MCP stdio (local)
skill-agent serve-mcp --port 8000          # MCP HTTP (remote)
skill-agent serve-http --port 8000         # HTTP REST API
skill-agent serve-http --llm ollama        # HTTP with Ollama

# Global flag (applies to all commands)
skill-agent --db /path/to/skills.db list
```

---

## Tool lifecycle

```
write → validate → [user decision] → save → execute
                                              │
                        ┌─────────────────────┘
                        ▼
                   record_execution()
                        │
                   _evaluate_health()
                        │
              ┌─────────┴──────────┐
         combined ≥ 0.60       combined < 0.60
              │                    │
           active              degraded
                                   │
                         negative feedback?
                                   │
                            _attempt_revision()
                                   │
                          rewrite → validate → update
```

### Health score

```
combined = 0.70 × execution_success_rate
         + 0.30 × user_sentiment_score    (if feedback exists)
```

| Threshold | After N executions | Outcome |
|---|---|---|
| combined < 0.60 | ≥ 5 | `degraded` — still usable, ranked lower |
| combined < 0.35 | ≥ 10 | `retired` — removed from search |

### Pruning

```bash
skill-agent prune                   # default: 30-day staleness window
skill-agent prune --stale-days 7    # aggressive
```

Three policies run together:
1. **Stale** — not used in N days
2. **Deeply degraded** — success rate below retirement threshold
3. **Duplicate** — cosine similarity > 0.97 with another tool; weaker one retired

---

## Retrieval architecture

### Similarity tiers

| Tier | Threshold | Action |
|---|---|---|
| High confidence | ≥ 0.88 | Reuse tool immediately — no LLM call |
| Mid confidence | ≥ 0.70 | LLM confirms whether the tool applies |
| Low / no match | < 0.50 | LLM classifies: create tool or answer directly |

### Reranking

Search results are sorted by:
1. Status (`active` > `degraded`)
2. Cosine similarity (descending)

This prevents degraded tools from winning on similarity alone.

### Embeddings

- Model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- Index: FAISS `IndexFlatIP` (exact inner-product = cosine similarity on unit-norm vectors)
- Rebuild: FAISS index is rebuilt from SQLite on startup and after any `update_tool()` or retirement

---

## Security and sandboxing

### Subprocess isolation

Every generated tool runs in a **fresh Python interpreter** as a subprocess, not inside the agent process. The harness:

- Passes `kwargs` as a JSON command-line argument
- Captures stdout/stderr
- Enforces a 10-second timeout (SIGKILL on expiry)
- Returns a structured `{"ok": bool, "result": ..., "error": ...}` payload
- Never calls `eval()` or `exec()` in the agent process

### Permissions

The agent scans tool code for permission requirements before execution. If the tool needs network access or filesystem paths, a `PERMISSION_REQUEST` event is emitted. The CLI prompts the user interactively; the SDK delegates to a `permission_handler=` callback.

### Docker sandbox

For stronger isolation:
```python
agent = SkillAgent(use_docker=True)
```

Or via CLI:
```bash
skill-agent run "..." --docker
```

Requires Docker CLI on PATH.

### What is NOT restricted (yet)

The subprocess sandbox currently does not restrict:
- Memory usage (use `ulimit` or Docker for this)
- Network access (unless permission is denied)
- Filesystem reads (unless permission is denied)

These are documented limitations; Docker execution addresses all three.

---

## Configuration reference

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `SKILL_AGENT_LLM_PROVIDER` | `anthropic` | Provider: `anthropic`, `openai`, `ollama`, `mock` |
| `SKILL_AGENT_LLM_MODEL` | provider default | Model name override |
| `SKILL_AGENT_API_KEY` | — | API key (providers also check their own env vars) |
| `SKILL_AGENT_DB_PATH` | `skills.db` | Path to SQLite registry |
| `SKILL_AGENT_USE_DOCKER` | `false` | `1`/`true`/`yes` to enable Docker sandbox |
| `SKILL_AGENT_HTTP_HOST` | `0.0.0.0` | HTTP server bind host |
| `SKILL_AGENT_HTTP_PORT` | `8000` | HTTP server port |
| `SKILL_AGENT_OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |

### `SkillAgent` constructor

```python
SkillAgent(
    llm_client=None,          # BaseProvider instance (mutually exclusive with llm=)
    llm="anthropic",          # provider shorthand
    llm_model=None,           # model override
    llm_api_key=None,         # API key override
    db_path="skills.db",      # SQLite registry path
    on_event=None,            # Callable[[AgentEvent], None]
    tool_decision=None,       # Callable — keep/revise/discard hook
    use_docker=False,         # run tools in Docker
    permission_handler=None,  # Callable — approve/deny filesystem/network
)
```

---

## Running locally

```bash
git clone https://github.com/your-org/autoskill
cd autoskill
python -m venv .venv && source .venv/bin/activate
pip install -e '.[all]'

# Run the demo (no API key needed)
python example.py

# Run with real Anthropic API
export ANTHROPIC_API_KEY=sk-ant-...
python example.py --real

# Run tests (no API key needed)
python tests.py

# Start HTTP REST API
skill-agent serve-http --llm anthropic

# Start MCP server (local stdio)
skill-agent serve

# Start MCP server (remote HTTP)
skill-agent serve-mcp --host 0.0.0.0 --port 8000
```

---

## Example generated tools

autoSkill generates tools like these from natural-language queries:

**Temperature conversion**
```python
def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """Convert Fahrenheit temperature to Celsius."""
    if not isinstance(fahrenheit, (int, float)):
        raise ValueError("fahrenheit must be a number")
    return round((fahrenheit - 32) * 5 / 9, 4)
```

**Compound interest**
```python
def calculate_compound_interest(
    principal: float, rate: float, years: int
) -> float:
    """Return principal grown by compound interest."""
    if principal < 0 or rate < 0 or years < 0:
        raise ValueError("All inputs must be non-negative")
    return round(principal * (1 + rate) ** years, 2)
```

**CSV row count**
```python
def count_csv_rows(csv_text: str) -> int:
    """Count data rows in a CSV string (excludes header)."""
    import csv, io
    rows = list(csv.reader(io.StringIO(csv_text.strip())))
    return max(0, len(rows) - 1)
```

All generated tools are:
- Standard library only
- Deterministic (same inputs → same outputs)
- Input-validated (raise `ValueError` on bad input)
- JSON-serializable return values

---

## Docker support

Build the image:
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -e '.[all]'
ENV SKILL_AGENT_DB_PATH=/data/skills.db
VOLUME /data
EXPOSE 8000
CMD ["skill-agent", "serve-http", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t autoskill .
docker run -p 8000:8000 -v $(pwd)/data:/data \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  autoskill
```

---

## Contributing

Contributions are welcome. Before opening a PR:

1. Run the test suite: `python tests.py`
2. All 57 tests must pass with no API key (uses `MockProvider`)
3. New providers belong in `skill_agent/providers/`
4. New providers must inherit `BaseProvider` and implement `.complete()`
5. Add tests for new functionality in `tests.py`

### Adding a new provider

```python
# skill_agent/providers/my_provider.py
from .base import BaseProvider

class MyProvider(BaseProvider):
    def __init__(self, api_key: str, model: str = "my-model"):
        self._api_key = api_key
        self._model   = model

    def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        # ... call your API ...
        return response_text
```

Register it in `skill_agent/providers/__init__.py` and add the provider name
to `create_llm()` in `llm.py`.

---

## License

MIT
