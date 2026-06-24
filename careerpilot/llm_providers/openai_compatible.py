from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from careerpilot.llm_providers.base import LLMConfig


@dataclass
class LLMMessage:
    """Minimal message object compatible with LangChain-style .content access."""

    content: str


class OpenAICompatibleLLMClient:
    """LLM client for providers compatible with OpenAI Chat Completions API.

    Supported providers include DeepSeek and Qwen/DashScope.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError(
                f"{self.config.provider} returned an empty response."
            )

        return content.strip()

    def invoke(self, prompt: str) -> LLMMessage:
        """LangChain-style compatibility method used by careerpilot.llm.invoke_json."""

        content = self.generate(
            user_prompt=prompt,
            system_prompt=(
                "你是一个严谨的结构化信息抽取和职业规划助手。"
                "当用户要求返回 JSON 时，只返回可解析的 JSON，不要输出 Markdown。"
            ),
        )
        return LLMMessage(content=content)
