from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout: float = 60.0


class LLMClient(Protocol):
    """Unified interface for all LLM providers."""

    def generate(self, user_prompt: str, system_prompt: str = "") -> str:
        """Generate text from a user prompt and an optional system prompt."""
        ...
