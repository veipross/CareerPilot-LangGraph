from pathlib import Path

from careerpilot.config import get_rag_settings
from careerpilot.rag import (
    build_corpus_fingerprint,
    load_knowledge_chunks,
    split_text_into_chunks,
)
from careerpilot.tools import retrieve_knowledge


def test_split_text_into_chunks_has_overlap():
    chunks = split_text_into_chunks("abcdefghij", chunk_size=6, overlap=2)
    assert chunks == ["abcdef", "efghij"]


def test_corpus_fingerprint_changes_with_content(tmp_path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    path = knowledge_dir / "rag.txt"
    path.write_text("RAG 使用向量检索。", encoding="utf-8")

    chunks_v1 = load_knowledge_chunks(knowledge_dir, chunk_size=100, overlap=10)
    fingerprint_v1 = build_corpus_fingerprint(
        chunks_v1,
        embedding_model="test-model",
        chunk_size=100,
        overlap=10,
    )

    path.write_text("RAG 使用向量检索和重排序。", encoding="utf-8")
    chunks_v2 = load_knowledge_chunks(knowledge_dir, chunk_size=100, overlap=10)
    fingerprint_v2 = build_corpus_fingerprint(
        chunks_v2,
        embedding_model="test-model",
        chunk_size=100,
        overlap=10,
    )

    assert fingerprint_v1 != fingerprint_v2


def test_keyword_mode_exposes_rag_metadata(tmp_path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.txt").write_text(
        "RAG 检索增强需要文档切分、向量检索和 Agent 编排。",
        encoding="utf-8",
    )

    results = retrieve_knowledge(
        query_terms=["RAG", "Agent"],
        knowledge_dir=str(knowledge_dir),
        top_k=2,
        mode="keyword",
    )

    assert results
    assert results[0]["retrieval_mode"] == "keyword"
    assert results[0]["keyword_score"] > 0
    assert results[0]["vector_score"] == 0


def test_vector_failure_falls_back_to_keyword(tmp_path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag.txt").write_text(
        "RAG 和 Agent 的本地知识说明。",
        encoding="utf-8",
    )

    def raise_vector_error(**kwargs):
        raise RuntimeError("mock vector backend unavailable")

    monkeypatch.setattr(
        "careerpilot.rag.retrieve_vector_knowledge",
        raise_vector_error,
    )
    results = retrieve_knowledge(
        query_terms=["RAG", "Agent"],
        query_text="如何构建语义检索 Agent",
        knowledge_dir=str(knowledge_dir),
        index_dir=str(tmp_path / "index"),
        top_k=2,
        mode="hybrid",
    )

    assert results
    assert results[0]["retrieval_mode"] == "keyword_fallback"
    assert "mock vector backend unavailable" in results[0]["fallback_reason"]


def test_rag_settings_reject_invalid_mode(monkeypatch):
    monkeypatch.setenv("CAREERPILOT_RAG_MODE", "unknown")
    settings = get_rag_settings()
    assert settings.mode == "keyword"
