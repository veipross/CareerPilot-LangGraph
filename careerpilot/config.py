"""Configuration helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model: str = os.getenv("CAREERPILOT_MODEL", "qwen-plus")
    # China mainland endpoint by default. You can override this in .env.
    dashscope_base_url: str = os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    dashscope_api_key: str | None = os.getenv("DASHSCOPE_API_KEY")
    temperature: float = float(os.getenv("CAREERPILOT_TEMPERATURE", "0.2"))


def get_settings() -> Settings:
    return Settings()
