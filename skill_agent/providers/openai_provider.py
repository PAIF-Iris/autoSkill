"""
providers/openai_provider.py — OpenAI Chat Completions provider.

Requires: pip install openai  (or pip install skill-agent[openai])
API key:  OPENAI_API_KEY environment variable (or pass api_key= explicitly)
"""
from __future__ import annotations

import os
from typing import Iterator, Optional

from .base import BaseProvider

DEFAULT_MODEL = "gpt-4o"


class OpenAIProvider(BaseProvider):
    """
    LLM provider backed by the OpenAI Chat Completions API.

    Parameters
    ----------
    api_key : str, optional
        OpenAI API key. Falls back to OPENAI_API_KEY env var.
    model : str, optional
        Model identifier. Defaults to gpt-4o.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        try:
            import openai as _openai
        except ImportError as exc:
            raise ImportError(
                "OpenAI package not installed. "
                "Run: pip install openai  or  pip install 'skill-agent[openai]'"
            ) from exc

        self._client = _openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )
        self._model = model or DEFAULT_MODEL

    def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return response.choices[0].message.content or ""

    def stream(self, system: str, user: str, max_tokens: int = 1000) -> Iterator[str]:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            stream=True,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
