from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
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

        # Do not inherit HTTP_PROXY / HTTPS_PROXY / ALL_PROXY from the
        # operating-system environment. This prevents a stale local proxy
        # configuration from blocking online LLM requests.
        self.http_client = httpx.Client(
            trust_env=False,
            timeout=config.timeout,
            follow_redirects=True,
        )

        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=1,
            http_client=self.http_client,
        )

    def _is_deepseek(self) -> bool:
        provider = str(getattr(self.config, "provider", "") or "").lower()
        base_url = str(getattr(self.config, "base_url", "") or "").lower()
        return "deepseek" in provider or "deepseek.com" in base_url

    def _create_completion(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> Any:
        request_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": max_tokens,
        }

        # DeepSeek V4 enables thinking mode by default. CareerPilot mainly
        # needs direct, structured final answers, so disable thinking mode.
        # This prevents the reasoning tokens from consuming the output budget
        # before message.content is produced.
        if self._is_deepseek():
            request_kwargs["extra_body"] = {
                "thinking": {
                    "type": "disabled",
                }
            }

        return self.client.chat.completions.create(**request_kwargs)

    @staticmethod
    def _extract_content(response: Any) -> str:
        if not getattr(response, "choices", None):
            return ""

        message = response.choices[0].message
        content = getattr(message, "content", None)

        if not content:
            return ""

        return str(content).strip()

    @staticmethod
    def _empty_response_details(response: Any) -> str:
        if not getattr(response, "choices", None):
            return "choices=empty"

        choice = response.choices[0]
        message = choice.message

        finish_reason = getattr(choice, "finish_reason", None)
        reasoning_content = getattr(message, "reasoning_content", None)
        reasoning_length = (
            len(reasoning_content)
            if isinstance(reasoning_content, str)
            else 0
        )

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

        return (
            f"finish_reason={finish_reason!r}, "
            f"reasoning_chars={reasoning_length}, "
            f"prompt_tokens={prompt_tokens}, "
            f"completion_tokens={completion_tokens}, "
            f"total_tokens={total_tokens}"
        )

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n"
                    "请直接输出最终答案，不要只输出思考过程。"
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        first_max_tokens = int(self.config.max_tokens)
        response = self._create_completion(
            messages=messages,
            max_tokens=first_max_tokens,
        )

        content = self._extract_content(response)
        if content:
            return content

        # Retry once when the provider returns a successful response object
        # whose final content is empty.
        retry_max_tokens = max(first_max_tokens, 4096)
        retry_messages = [
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n"
                    "你必须直接返回非空的最终答案。"
                    "当用户要求 JSON 时，只输出完整、可解析的 JSON。"
                    "不要输出 Markdown，不要只输出推理过程。"
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        retry_response = self._create_completion(
            messages=retry_messages,
            max_tokens=retry_max_tokens,
        )

        retry_content = self._extract_content(retry_response)
        if retry_content:
            return retry_content

        first_details = self._empty_response_details(response)
        retry_details = self._empty_response_details(retry_response)

        raise RuntimeError(
            f"{self.config.provider} returned an empty final response after retry. "
            f"model={self.config.model!r}; "
            f"first_attempt=({first_details}); "
            f"retry_attempt=({retry_details})"
        )

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
