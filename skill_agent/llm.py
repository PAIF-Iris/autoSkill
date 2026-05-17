"""
llm.py — LLM client abstraction (backward-compatible layer).

All classes here inherit from BaseProvider and satisfy the same two-method
contract used throughout the agent pipeline:
    .complete(system, user, max_tokens) -> str
    .stream(system, user, max_tokens)   -> Iterator[str]

Backward-compatible names (kept forever):
    AnthropicClient  — alias for AnthropicProvider
    OpenAIClient     — alias for OpenAIProvider
    MockClient       — alias for MockProvider

New code should import from skill_agent.providers instead:
    from skill_agent.providers import AnthropicProvider, OllamaProvider

Factory
-------
    create_llm(provider, model=None, api_key=None) -> BaseProvider
    Supports: "anthropic", "openai", "ollama", "mock"
"""
from __future__ import annotations

from typing import Optional

from .providers.base             import BaseProvider          # noqa: F401 (re-exported)
from .providers.anthropic_provider import AnthropicProvider
from .providers.openai_provider  import OpenAIProvider
from .providers.ollama_provider  import OllamaProvider
from .providers.mock_provider    import MockProvider

# ── Backward-compatible aliases ───────────────────────────────────────────────
# Code that does `from skill_agent.llm import AnthropicClient` keeps working.

AnthropicClient = AnthropicProvider
OpenAIClient    = OpenAIProvider
MockClient      = MockProvider   # backward-compatible alias

# Re-export default model constants for any code that imported them from here
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_OPENAI_MODEL    = "gpt-4o"


# ── Factory ───────────────────────────────────────────────────────────────────

def create_llm(
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BaseProvider:
    """
    Instantiate a provider by name.

    Parameters
    ----------
    provider : "anthropic" | "openai" | "ollama" | "mock"
    model    : override the default model for that provider
    api_key  : API key override (not used for ollama or mock)

    Returns a BaseProvider instance.
    """
    if provider == "anthropic":
        return AnthropicProvider(api_key=api_key, model=model)
    if provider == "openai":
        return OpenAIProvider(api_key=api_key, model=model)
    if provider == "ollama":
        return OllamaProvider(model=model or "llama3")
    if provider == "mock":
        return MockProvider()
    raise ValueError(
        f"Unknown provider: {provider!r}. "
        f"Expected 'anthropic', 'openai', 'ollama', or 'mock'."
    )
