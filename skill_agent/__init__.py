"""
skill_agent — Executable semantic memory for AI agents.

A self-improving tool-learning runtime that routes every query through
vector retrieval first, executes deterministic Python tools when a match
is found, and falls back to LLM generation only when needed.

Quick start
-----------
    from skill_agent import SkillAgent

    agent = SkillAgent()                   # uses ANTHROPIC_API_KEY
    result = agent.run("Convert 98.6°F to Celsius")
    print(result.answer)

With config helper:
    from skill_agent import AutoSkillConfig

    cfg   = AutoSkillConfig(llm_provider="ollama", llm_model="llama3")
    agent = cfg.create_agent()

With real-time events:
    from skill_agent import SkillAgent, EventType

    def on_event(e):
        if e.type == EventType.ANSWER_CHUNK:
            print(e.payload["chunk"], end="", flush=True)

    agent = SkillAgent(on_event=on_event)

Custom provider:
    from skill_agent.providers import BaseProvider

    class MyProvider(BaseProvider):
        def complete(self, system, user, max_tokens=1000):
            ...   # call your LLM here

    agent = SkillAgent(llm_client=MyProvider())
"""
from .agent          import SkillAgent, AgentResult
from .events         import AgentEvent, EventType
from .tool_registry  import ToolVersion
from .permissions    import PermissionRequest, GrantedPermissions, MCPDependency
from .providers.base         import BaseProvider
from .providers.mock_provider import MockProvider
from .config         import AutoSkillConfig

__all__ = [
    # Core runtime
    "SkillAgent",
    "AgentResult",
    # Events
    "AgentEvent",
    "EventType",
    # Registry
    "ToolVersion",
    # Permissions
    "PermissionRequest",
    "GrantedPermissions",
    "MCPDependency",
    # Provider abstraction
    "BaseProvider",
    "MockProvider",
    # Configuration
    "AutoSkillConfig",
]
