"""
docker_executor.py — Docker-based sandboxed code execution.

Provides the same interface as executor.py (returns ExecutionResult) but
runs the tool harness inside a fresh Docker container for stronger isolation:

  - No access to host filesystem by default
  - Network disabled by default (--network none)
  - CPU and memory capped
  - Container removed after execution (--rm)

Permissions granted via GrantedPermissions are translated to Docker flags:
  - filesystem_paths  → read-only volume mounts (-v path:/data/N:ro)
  - network=True      → removes --network none restriction
  - subprocess=True   → no extra flag (subprocess is allowed in the harness
                         Python process, but inside the container)

Requires:
  - Docker CLI available on PATH (check with docker_available())
  - python:3.11-slim image (pulled automatically on first use)

Usage:
    from skill_agent.docker_executor import execute_in_docker, docker_available
    if docker_available():
        result = execute_in_docker(code, fn_name, kwargs)
    else:
        result = execute_tool(code, fn_name, kwargs)  # subprocess fallback
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from .executor import ExecutionResult, MAX_OUTPUT_BYTES, TIMEOUT_SECONDS as _DEFAULT_TIMEOUT
from .permissions import GrantedPermissions

DOCKER_IMAGE   = "python:3.11-slim"
DOCKER_MEMORY  = "256m"
DOCKER_CPUS    = "0.5"
DOCKER_TIMEOUT = max(_DEFAULT_TIMEOUT * 3, 30)   # containers have startup overhead


def docker_available() -> bool:
    """Return True if the Docker CLI is installed and the daemon is reachable."""
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def execute_in_docker(
    code: str,
    fn_name: str,
    kwargs: dict,
    granted: Optional[GrantedPermissions] = None,
) -> ExecutionResult:
    """
    Execute `fn_name(**kwargs)` in a Docker container.

    Parameters
    ----------
    code     : complete Python source defining fn_name
    fn_name  : name of the function to call
    kwargs   : dict of keyword arguments; must be JSON-serializable
    granted  : permissions approved for this execution

    Returns
    -------
    ExecutionResult — same type as executor.execute_tool; never raises.
    """
    if granted is None:
        granted = GrantedPermissions()

    # Build harness (same structure as executor.py)
    harness_lines = [
        "import json",
        "import sys",
        "import traceback",
        "",
        "# ── Tool code ──────────────────────────────────────────────────────",
        code,
        "",
        "# ── Harness ────────────────────────────────────────────────────────",
        "try:",
        "    kwargs = json.loads(sys.argv[1])",
        f"    result = {fn_name}(**kwargs)",
        '    print(json.dumps({"ok": True, "result": result}))',
        "except Exception as exc:",
        '    print(json.dumps({"ok": False, "error": str(exc), '
        '"trace": __import__("traceback").format_exc()}))',
    ]
    harness = "\n".join(harness_lines) + "\n"

    with tempfile.TemporaryDirectory(prefix="skill_docker_") as tmp_dir:
        harness_path = Path(tmp_dir) / "harness.py"
        harness_path.write_text(harness, encoding="utf-8")

        # ── Build docker run command ─────────────────────────────────────────
        cmd = [
            "docker", "run", "--rm",
            "--memory", DOCKER_MEMORY,
            "--cpus", DOCKER_CPUS,
            "--network", "none",        # default: no network
            "-v", f"{tmp_dir}:/code:ro",
        ]

        # Filesystem mounts
        for idx, fs_path in enumerate(granted.filesystem_paths):
            cmd.extend(["-v", f"{fs_path}:/data/{idx}:ro"])

        # Network permission
        if granted.network:
            # Remove --network none (last two elements added above)
            cmd = [c for i, c in enumerate(cmd)
                   if not (cmd[i - 1] == "--network" and c == "none")]

        cmd.extend([
            DOCKER_IMAGE,
            "python3", "/code/harness.py", json.dumps(kwargs),
        ])

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=DOCKER_TIMEOUT,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            stdout = proc.stdout.strip()[:MAX_OUTPUT_BYTES]
            stderr = proc.stderr.strip()[:MAX_OUTPUT_BYTES]

            if proc.returncode != 0 and not stdout:
                return ExecutionResult(
                    success=False,
                    output=None,
                    error=f"Container exited {proc.returncode}: {stderr[:300]}",
                    latency_ms=latency_ms,
                    raw_stdout=stdout,
                    raw_stderr=stderr,
                )

            try:
                payload = json.loads(stdout)
            except json.JSONDecodeError:
                return ExecutionResult(
                    success=False,
                    output=None,
                    error=f"Non-JSON output from container: {stdout[:200]!r}",
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
                error=f"Docker execution timed out after {DOCKER_TIMEOUT}s",
                latency_ms=DOCKER_TIMEOUT * 1000,
                raw_stdout="",
                raw_stderr="",
            )
