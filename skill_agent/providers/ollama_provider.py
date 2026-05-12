"""
providers/ollama_provider.py — Ollama local model provider.

No extra dependencies — uses Python stdlib urllib only.
Requires Ollama running locally: https://ollama.com

Quick start:
    ollama pull llama3
    ollama serve          # default: http://localhost:11434
"""
from __future__ import annotations

import json
import urllib.request
from typing import Iterator, Optional

from .base import BaseProvider

DEFAULT_MODEL   = "llama3"
DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(BaseProvider):
    """
    LLM provider backed by a locally running Ollama server.

    Uses the /api/chat endpoint with JSON streaming disabled for .complete()
    and enabled for .stream().  No pip dependencies beyond stdlib.

    Parameters
    ----------
    model : str
        Ollama model tag (e.g. "llama3", "mistral", "phi3", "codellama").
        The model must already be pulled: ``ollama pull <model>``.
    base_url : str
        Base URL of the Ollama server. Defaults to http://localhost:11434.
    timeout : int
        HTTP request timeout in seconds. Defaults to 120.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 120,
    ) -> None:
        self._model    = model
        self._base_url = base_url.rstrip("/")
        self._timeout  = timeout

    def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        payload = json.dumps({
            "model":   self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream":  False,
            "options": {"num_predict": max_tokens},
        }).encode()

        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            data = json.loads(resp.read().decode())
        return data["message"]["content"]

    def stream(self, system: str, user: str, max_tokens: int = 1000) -> Iterator[str]:
        payload = json.dumps({
            "model":   self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream":  True,
            "options": {"num_predict": max_tokens},
        }).encode()

        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            for raw_line in resp:
                if not raw_line:
                    continue
                data = json.loads(raw_line.decode())
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
                if data.get("done"):
                    break
