from __future__ import annotations

import os

from dotenv import load_dotenv

from careerpilot.llm_providers.base import LLMClient, LLMConfig
from careerpilot.llm_providers.openai_compatible import OpenAICompatibleLLMClient


def _get_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float.") from exc


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


def create_llm_client(
    provider: str | None = None,
    model: str | None = None,
) -> LLMClient:
    """Create an LLM client from environment variables.

    Supported providers:
    - deepseek
    - qwen
    """

    load_dotenv()

    provider = (provider or os.getenv("CAREERPILOT_LLM_PROVIDER", "deepseek")).lower()

    temperature = _get_float_env("CAREERPILOT_LLM_TEMPERATURE", 0.2)
    max_tokens = _get_int_env("CAREERPILOT_LLM_MAX_TOKENS", 2048)
    timeout = _get_float_env("CAREERPILOT_LLM_TIMEOUT", 60.0)

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("CAREERPILOT_LLM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. "
                "Please create a .env file or export DEEPSEEK_API_KEY."
            )

        config = LLMConfig(
            provider="deepseek",
            api_key=api_key,
            base_url=os.getenv(
                "CAREERPILOT_DEEPSEEK_BASE_URL",
                "https://api.deepseek.com",
            ),
            model=model
            or os.getenv("CAREERPILOT_DEEPSEEK_MODEL", "deepseek-v4-flash"),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return OpenAICompatibleLLMClient(config)

    if provider == "qwen":
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("CAREERPILOT_LLM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY is not set. "
                "Please create a .env file or export DASHSCOPE_API_KEY."
            )

        config = LLMConfig(
            provider="qwen",
            api_key=api_key,
            base_url=os.getenv(
                "CAREERPILOT_QWEN_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            model=model or os.getenv("CAREERPILOT_QWEN_MODEL", "qwen-plus"),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return OpenAICompatibleLLMClient(config)

    raise ValueError(
        f"Unsupported LLM provider: {provider}. "
        "Supported providers are: deepseek, qwen."
    )
