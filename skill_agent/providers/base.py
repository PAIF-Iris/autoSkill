"""
providers/base.py — Abstract base class for all LLM providers.

Every provider must implement .complete().
.stream() has a working default (single-chunk fallback) so providers that
don't support streaming still work with the streaming agent path.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator


class BaseProvider(ABC):
    """
    Minimal contract every LLM provider must satisfy.

    Implement .complete() to add a new provider.
    Override .stream() to enable real-time token streaming.
    """

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        """Return the full LLM response as a single string."""
        ...

    def stream(self, system: str, user: str, max_tokens: int = 1000) -> Iterator[str]:
        """
        Yield response text incrementally.

        Default: wraps .complete() in a single-chunk iterator so the agent's
        streaming path works even for providers that don't support streaming.
        Override this for true token-level streaming.
        """
        yield self.complete(system, user, max_tokens)
