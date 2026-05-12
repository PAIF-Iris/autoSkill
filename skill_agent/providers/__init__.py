"""
skill_agent.providers — Provider abstraction layer.

All providers implement BaseProvider's two-method contract:
    .complete(system, user, max_tokens) -> str
    .stream(system, user, max_tokens)   -> Iterator[str]   (default: single-chunk)

Built-in providers
------------------
AnthropicProvider   requires: pip install anthropic
OpenAIProvider      requires: pip install openai
OllamaProvider      requires: Ollama running locally (no pip dep)
MockProvider        no requirements — for tests and offline use

Custom providers
----------------
Subclass BaseProvider and implement .complete():

    from skill_agent.providers import BaseProvider

    class MyProvider(BaseProvider):
        def complete(self, system, user, max_tokens=1000):
            ...

    agent = SkillAgent(llm_client=MyProvider())
"""
from .base             import BaseProvider
from .anthropic_provider import AnthropicProvider
from .openai_provider  import OpenAIProvider
from .ollama_provider  import OllamaProvider
from .mock_provider    import MockProvider

__all__ = [
    "BaseProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "OllamaProvider",
    "MockProvider",
]
