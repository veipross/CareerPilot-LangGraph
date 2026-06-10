"""LLM factory and safe JSON helpers."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from .config import Settings


def build_qwen_llm(settings: Settings):
    """Create a Qwen chat model through DashScope's OpenAI-compatible endpoint.

    The import is intentionally local so offline mode can run without langchain-openai.
    """

    if not settings.dashscope_api_key:
        return None

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - user environment issue
        raise RuntimeError(
            "Missing langchain-openai. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    return ChatOpenAI(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        model=settings.model,
        temperature=settings.temperature,
    )


def extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from a model response.

    Handles raw JSON and fenced ```json blocks. Returns an empty dict if parsing fails.
    """

    if not text:
        return {}

    candidates = []
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates.extend(fenced)
    brace = re.search(r"\{.*\}", text, flags=re.S)
    if brace:
        candidates.append(brace.group(0))
    candidates.append(text)

    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return {}


def invoke_json(llm: Any, prompt: str) -> Dict[str, Any]:
    """Invoke a chat model and parse a JSON object from the response."""

    if llm is None:
        return {}
    message = llm.invoke(prompt)
    content = getattr(message, "content", str(message))
    return extract_json(content)
