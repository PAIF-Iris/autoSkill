"""
config.py — Unified configuration for the autoSkill runtime.

Usage
-----
Programmatic:
    cfg = AutoSkillConfig(llm_provider="ollama", llm_model="llama3")
    agent = cfg.create_agent()

From environment variables:
    cfg = AutoSkillConfig.from_env()
    agent = cfg.create_agent()

Environment variables
---------------------
SKILL_AGENT_LLM_PROVIDER   — "anthropic" | "openai" | "ollama" | "mock"
SKILL_AGENT_LLM_MODEL      — model name override (e.g. "gpt-4o", "llama3")
SKILL_AGENT_API_KEY        — API key (providers also check their own env vars)
SKILL_AGENT_DB_PATH        — path to SQLite registry file
SKILL_AGENT_USE_DOCKER     — "1" | "true" | "yes" to run tools in Docker
SKILL_AGENT_HTTP_HOST      — bind host for serve-http (default: 0.0.0.0)
SKILL_AGENT_HTTP_PORT      — port for serve-http (default: 8000)
SKILL_AGENT_OLLAMA_URL     — Ollama server URL (default: http://localhost:11434)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AutoSkillConfig:
    """
    Configuration container for the autoSkill runtime.

    All fields have sensible defaults so you only need to set what differs
    from the defaults in your environment.
    """

    # ── LLM provider ─────────────────────────────────────────────────────────
    llm_provider: str = "anthropic"
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"

    # ── Storage ───────────────────────────────────────────────────────────────
    db_path: str = "skills.db"

    # ── Execution sandbox ─────────────────────────────────────────────────────
    use_docker: bool = False

    # ── HTTP server (used by serve-http) ──────────────────────────────────────
    http_host: str = "0.0.0.0"
    http_port: int = 8000

    # ── Internal: extra kwargs passed through to SkillAgent ──────────────────
    _extra: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_env(cls) -> "AutoSkillConfig":
        """
        Build config from environment variables.
        All variables are optional; unset ones fall back to dataclass defaults.
        """
        def _bool(val: Optional[str]) -> bool:
            return (val or "").lower() in ("1", "true", "yes")

        return cls(
            llm_provider    = os.environ.get("SKILL_AGENT_LLM_PROVIDER", "anthropic"),
            llm_model       = os.environ.get("SKILL_AGENT_LLM_MODEL"),
            llm_api_key     = os.environ.get("SKILL_AGENT_API_KEY"),
            ollama_base_url = os.environ.get("SKILL_AGENT_OLLAMA_URL",
                                             "http://localhost:11434"),
            db_path         = os.environ.get("SKILL_AGENT_DB_PATH", "skills.db"),
            use_docker      = _bool(os.environ.get("SKILL_AGENT_USE_DOCKER")),
            http_host       = os.environ.get("SKILL_AGENT_HTTP_HOST", "0.0.0.0"),
            http_port       = int(os.environ.get("SKILL_AGENT_HTTP_PORT", "8000")),
        )

    def create_agent(self, **overrides):
        """
        Convenience factory: build a SkillAgent from this config.

        Keyword `overrides` are forwarded directly to SkillAgent() and take
        precedence over config values — useful for one-off customisation.

        Example
        -------
        cfg = AutoSkillConfig.from_env()
        agent = cfg.create_agent(on_event=my_handler)
        """
        from .agent import SkillAgent

        kwargs: dict = dict(
            llm        = self.llm_provider,
            llm_model  = self.llm_model,
            llm_api_key= self.llm_api_key,
            db_path    = self.db_path,
            use_docker = self.use_docker,
        )
        kwargs.update(self._extra)
        kwargs.update(overrides)
        return SkillAgent(**kwargs)
