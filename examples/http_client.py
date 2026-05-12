"""
examples/http_client.py — HTTP API client example (stdlib only, no requests/httpx).

Demonstrates all six endpoints of the autoSkill HTTP server.

Start the server first:
    skill-agent serve-http --port 8000 --llm anthropic
    # or for local models:
    skill-agent serve-http --port 8000 --llm ollama --llm-model llama3

Then run this script:
    python examples/http_client.py
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import Any

BASE_URL = "http://localhost:8000"


def _request(method: str, path: str, body: Any = None) -> Any:
    url     = f"{BASE_URL}{path}"
    data    = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req     = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.read().decode()}", file=sys.stderr)
        return None


def section(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print('═' * 60)


def pp(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def main() -> None:
    # ── 1. Health check ───────────────────────────────────────────────────────
    section("GET /health")
    pp(_request("GET", "/health"))

    # ── 2. Run a query ────────────────────────────────────────────────────────
    section("POST /run  — deterministic query (tool created or reused)")
    result = _request("POST", "/run", {
        "query": "What is 5% compound interest on $1000 over 3 years?"
    })
    pp(result)

    # ── 3. Search for tools ───────────────────────────────────────────────────
    section("POST /tools/search")
    pp(_request("POST", "/tools/search", {
        "query": "compound interest calculator",
        "top_k": 3,
    }))

    # ── 4. List all tools ─────────────────────────────────────────────────────
    section("GET /tools")
    tools = _request("GET", "/tools") or []
    pp(tools)

    # ── 5. Inspect one tool ───────────────────────────────────────────────────
    if tools:
        tool_id = tools[0]["tool_id"]
        section(f"GET /tools/{tool_id}")
        pp(_request("GET", f"/tools/{tool_id}"))

        # ── 6. Record feedback ────────────────────────────────────────────────
        section(f"POST /tools/{tool_id}/feedback")
        _request("POST", f"/tools/{tool_id}/feedback", {
            "positive": True,
            "comment": "Result matches expected value",
        })
        print("  204 No Content — feedback recorded")
    else:
        print("\n  (no tools in registry yet — run a query first)")

    # ── 7. Conversational query (should answer directly) ──────────────────────
    section("POST /run  — conversational query (direct LLM answer)")
    result = _request("POST", "/run", {
        "query": "What is the difference between simple and compound interest?"
    })
    if result:
        print(f"  action_taken : {result['action_taken']}")
        print(f"  answer       : {str(result['answer'])[:200]}…")


if __name__ == "__main__":
    main()
