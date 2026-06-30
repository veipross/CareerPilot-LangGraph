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

@dataclass(frozen=True)
class RAGSettings:
    """Runtime configuration for local keyword/vector/hybrid retrieval."""

    mode: str = "keyword"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    knowledge_dir: str = "data/knowledge"
    index_dir: str = "data/vector_store"
    top_k: int = 4
    chunk_size: int = 420
    chunk_overlap: int = 60
    vector_weight: float = 0.75
    device: str = "cpu"


def get_rag_settings() -> RAGSettings:
    """Read RAG settings at call time so values loaded from .env are honored."""

    def read_int(name: str, default: int, minimum: int) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except ValueError:
            value = default
        return max(minimum, value)

    def read_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except ValueError:
            return default

    mode = os.getenv("CAREERPILOT_RAG_MODE", "keyword").strip().lower()
    if mode not in {"keyword", "vector", "hybrid"}:
        mode = "keyword"

    vector_weight = min(
        1.0,
        max(0.0, read_float("CAREERPILOT_RAG_VECTOR_WEIGHT", 0.75)),
    )
    chunk_size = read_int("CAREERPILOT_RAG_CHUNK_SIZE", 420, 64)
    chunk_overlap = min(
        chunk_size - 1,
        read_int("CAREERPILOT_RAG_CHUNK_OVERLAP", 60, 0),
    )

    return RAGSettings(
        mode=mode,
        embedding_model=os.getenv(
            "CAREERPILOT_RAG_EMBEDDING_MODEL",
            "BAAI/bge-small-zh-v1.5",
        ),
        knowledge_dir=os.getenv("CAREERPILOT_RAG_KNOWLEDGE_DIR", "data/knowledge"),
        index_dir=os.getenv("CAREERPILOT_RAG_INDEX_DIR", "data/vector_store"),
        top_k=read_int("CAREERPILOT_RAG_TOP_K", 4, 1),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        vector_weight=vector_weight,
        device=os.getenv("CAREERPILOT_RAG_DEVICE", "cpu"),
    )
