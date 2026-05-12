"""
providers/anthropic_provider.py — Anthropic Claude provider.

Requires: pip install anthropic
API key:  ANTHROPIC_API_KEY environment variable (or pass api_key= explicitly)
"""
from __future__ import annotations

import os
from typing import Iterator, Optional

from .base import BaseProvider

DEFAULT_MODEL = "claude-sonnet-4-20250514"


class AnthropicProvider(BaseProvider):
    """
    LLM provider backed by the Anthropic Messages API.

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
    model : str, optional
        Model identifier. Defaults to claude-sonnet-4-20250514.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic package not installed. Run: pip install anthropic"
            ) from exc

        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"]
        )
        self._model = model or DEFAULT_MODEL

    def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text

    def stream(self, system: str, user: str, max_tokens: int = 1000) -> Iterator[str]:
        with self._client.messages.stream(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as s:
            yield from s.text_stream
