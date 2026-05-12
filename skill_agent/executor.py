"""
executor.py — safe sandboxed execution via subprocess.

Security model:
  - Tool code runs in a FRESH Python interpreter (subprocess), completely
    isolated from the agent process.
  - We pass kwargs as a command-line JSON argument (not via stdin/pipe)
    to keep the interface simple and auditable.
  - A thin harness script wraps the tool function, catches ALL exceptions,
    and serializes the result as JSON to stdout.
  - The agent process itself never calls eval() or exec().
  - Timeout is enforced by subprocess.run(timeout=...) which sends SIGKILL.

Output protocol:
  The harness always writes exactly one JSON line to stdout:
    {"ok": true,  "result": <value>}   on success
    {"ok": false, "error": "<msg>", "trace": "<traceback>"}  on exception

Limitations:
  - No memory limit enforcement (would need ulimit / cgroups / Docker).
  - No network/filesystem restriction (would need seccomp or a container).
  - These are the natural next hardening steps for production.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

TIMEOUT_SECONDS = 10
MAX_OUTPUT_BYTES = 64 * 1024    # 64 KB — more than enough for any tool result


@dataclass
class ExecutionResult:
    success: bool
    output: Any                  # deserialized return value (None on failure)
    error: Optional[str]         # error message (None on success)
    latency_ms: float
    raw_stdout: str
    raw_stderr: str


def execute_tool(code: str, fn_name: str, kwargs: dict) -> ExecutionResult:
    """
    Execute `fn_name(**kwargs)` safely in a subprocess.

    Parameters
    ----------
    code     : complete Python source defining fn_name (stdlib-only assumed)
    fn_name  : name of the function to call
    kwargs   : dict of keyword arguments; must be JSON-serializable

    Returns
    -------
    ExecutionResult — never raises; all errors are captured.
    """
    # Build the harness script that wraps the tool and serializes output.
    # We concatenate lines explicitly — NOT textwrap.dedent — because dedent
    # measures the minimum indentation across ALL lines including the injected
    # {code}, which destroys the function body's own indentation.
    harness_lines = [
        "import json",
        "import sys",
        "import traceback",
        "",
        "# ── Tool code (injected) ─────────────────────────────────────────",
        code,
        "",
        "# ── Harness ──────────────────────────────────────────────────────",
        "try:",
        "    kwargs = json.loads(sys.argv[1])",
        f"    result = {fn_name}(**kwargs)",
        '    print(json.dumps({"ok": True, "result": result}))',
        "except Exception as exc:",
        '    print(json.dumps({"ok": False, "error": str(exc), "trace": __import__("traceback").format_exc()}))',
    ]
    harness = "\n".join(harness_lines) + "\n"

    # Write harness to a temp file so we can pass it to python3
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, prefix="skill_exec_"
    ) as f:
        f.write(harness)
        tmp_path = Path(f.name)

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            ["python3", str(tmp_path), json.dumps(kwargs)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        stdout = proc.stdout.strip()[:MAX_OUTPUT_BYTES]
        stderr = proc.stderr.strip()[:MAX_OUTPUT_BYTES]

        # Non-zero exit without a structured payload (e.g. import error)
        if proc.returncode != 0 and not stdout:
            return ExecutionResult(
                success=False,
                output=None,
                error=f"Process exited {proc.returncode}: {stderr[:300]}",
                latency_ms=latency_ms,
                raw_stdout=stdout,
                raw_stderr=stderr,
            )

        # Parse the harness's JSON output
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return ExecutionResult(
                success=False,
                output=None,
                error=f"Non-JSON output from harness: {stdout[:200]!r}",
                latency_ms=latency_ms,
                raw_stdout=stdout,
                raw_stderr=stderr,
            )

        if payload.get("ok"):
            return ExecutionResult(
                success=True,
                output=payload["result"],
                error=None,
                latency_ms=latency_ms,
                raw_stdout=stdout,
                raw_stderr=stderr,
            )
        else:
            return ExecutionResult(
                success=False,
                output=None,
                error=payload.get("error", "Unknown error"),
                latency_ms=latency_ms,
                raw_stdout=stdout,
                raw_stderr=stderr,
            )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            output=None,
            error=f"Execution timed out after {TIMEOUT_SECONDS}s",
            latency_ms=TIMEOUT_SECONDS * 1000,
            raw_stdout="",
            raw_stderr="",
        )
    finally:
        # Always clean up the temp file
        tmp_path.unlink(missing_ok=True)
