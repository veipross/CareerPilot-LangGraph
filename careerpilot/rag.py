"""Local semantic retrieval backed by Sentence Transformers and FAISS.

The module keeps heavy optional imports lazy so CareerPilot's deterministic
keyword mode and test suite can still run in lightweight/offline environments.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, List, Sequence


INDEX_SCHEMA_VERSION = 1
SUPPORTED_KNOWLEDGE_SUFFIXES = {".txt", ".md"}


class VectorRAGError(RuntimeError):
    """Raised when the vector retrieval backend cannot be initialized."""


@dataclass(frozen=True)
class KnowledgeChunk:
    source: str
    source_name: str
    chunk_index: int
    content: str


@dataclass(frozen=True)
class VectorIndexInfo:
    schema_version: int
    fingerprint: str
    embedding_model: str
    chunk_size: int
    overlap: int
    chunks: List[dict[str, Any]]


def normalize_text(text: str) -> str:
    """Collapse whitespace while preserving readable plain text."""

    return " ".join((text or "").split()).strip()


def split_text_into_chunks(
    text: str,
    chunk_size: int = 420,
    overlap: int = 60,
) -> List[str]:
    """Split text into deterministic overlapping character chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    normalized = normalize_text(text)
    if not normalized:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(normalized):
        end = start + chunk_size
        chunks.append(normalized[start:end])
        if end >= len(normalized):
            break
        start = end - overlap
    return chunks


def load_knowledge_chunks(
    knowledge_dir: str | Path,
    chunk_size: int = 420,
    overlap: int = 60,
) -> List[KnowledgeChunk]:
    """Load all supported local knowledge files and attach source metadata."""

    root = Path(knowledge_dir)
    if not root.exists():
        return []

    chunks: List[KnowledgeChunk] = []
    paths = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_KNOWLEDGE_SUFFIXES
    )
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for chunk_index, content in enumerate(
            split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap),
            start=1,
        ):
            chunks.append(
                KnowledgeChunk(
                    source=str(path),
                    source_name=path.name,
                    chunk_index=chunk_index,
                    content=content,
                )
            )
    return chunks


def build_corpus_fingerprint(
    chunks: Sequence[KnowledgeChunk],
    embedding_model: str,
    chunk_size: int,
    overlap: int,
) -> str:
    """Return a stable fingerprint used to decide whether to rebuild an index."""

    digest = hashlib.sha256()
    digest.update(f"schema={INDEX_SCHEMA_VERSION}\n".encode("utf-8"))
    digest.update(f"model={embedding_model}\n".encode("utf-8"))
    digest.update(f"chunk_size={chunk_size}\n".encode("utf-8"))
    digest.update(f"overlap={overlap}\n".encode("utf-8"))

    for chunk in chunks:
        digest.update(chunk.source_name.encode("utf-8"))
        digest.update(str(chunk.chunk_index).encode("ascii"))
        digest.update(chunk.content.encode("utf-8"))
    return digest.hexdigest()


def _load_vector_dependencies() -> tuple[Any, Any]:
    """Import NumPy and FAISS lazily with an actionable error message."""

    try:
        import numpy as np
        import faiss
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise VectorRAGError(
            "向量 RAG 依赖缺失，请安装 sentence-transformers 和 faiss-cpu。"
        ) from exc
    return np, faiss


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str, device: str) -> Any:
    """Load and cache one embedding model per model/device pair."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise VectorRAGError(
            "缺少 sentence-transformers，无法启用向量 RAG。"
        ) from exc

    try:
        return SentenceTransformer(model_name, device=device)
    except Exception as exc:  # pragma: no cover - model/network/runtime specific
        raise VectorRAGError(
            f"无法加载 Embedding 模型 {model_name}：{exc}"
        ) from exc


def _encode_texts(model: Any, texts: Sequence[str], np: Any) -> Any:
    """Encode and L2-normalize text vectors for cosine search via inner product."""

    if not texts:
        return np.empty((0, 0), dtype="float32")

    try:
        vectors = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except TypeError:
        # Supports small fake encoders used by unit tests and older wrappers.
        vectors = model.encode(list(texts))
    except Exception as exc:
        raise VectorRAGError(f"文本向量化失败：{exc}") from exc

    vectors = np.asarray(vectors, dtype="float32")
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise VectorRAGError("Embedding 模型返回了无效向量形状。")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def _metadata_path(index_dir: Path) -> Path:
    return index_dir / "knowledge_metadata.json"


def _index_path(index_dir: Path) -> Path:
    return index_dir / "knowledge.faiss"


def _read_metadata(index_dir: Path) -> dict[str, Any] | None:
    path = _metadata_path(index_dir)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _metadata_is_current(
    metadata: dict[str, Any] | None,
    fingerprint: str,
    embedding_model: str,
    expected_chunk_count: int,
) -> bool:
    if not metadata:
        return False
    return (
        metadata.get("schema_version") == INDEX_SCHEMA_VERSION
        and metadata.get("fingerprint") == fingerprint
        and metadata.get("embedding_model") == embedding_model
        and len(metadata.get("chunks") or []) == expected_chunk_count
    )


def _write_metadata(index_dir: Path, info: VectorIndexInfo) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    _metadata_path(index_dir).write_text(
        json.dumps(asdict(info), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_or_load_faiss_index(
    chunks: Sequence[KnowledgeChunk],
    embedding_model: str,
    index_dir: str | Path,
    chunk_size: int = 420,
    overlap: int = 60,
    device: str = "cpu",
    encoder: Any | None = None,
) -> tuple[Any, List[KnowledgeChunk], bool]:
    """Load a compatible persisted FAISS index or rebuild it.

    Returns ``(index, chunks, rebuilt)``.
    """

    if not chunks:
        raise VectorRAGError("知识库为空，无法建立向量索引。")

    np, faiss = _load_vector_dependencies()
    index_root = Path(index_dir)
    fingerprint = build_corpus_fingerprint(
        chunks,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    metadata = _read_metadata(index_root)
    index_file = _index_path(index_root)

    if (
        index_file.exists()
        and _metadata_is_current(
            metadata,
            fingerprint=fingerprint,
            embedding_model=embedding_model,
            expected_chunk_count=len(chunks),
        )
    ):
        try:
            index = faiss.read_index(str(index_file))
            if index.ntotal == len(chunks):
                return index, list(chunks), False
        except Exception:
            # Corrupt/incompatible local cache: safely rebuild below.
            pass

    model = encoder or _load_sentence_transformer(embedding_model, device)
    embeddings = _encode_texts(model, [chunk.content for chunk in chunks], np)
    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(embeddings)

    index_root.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_file))
    _write_metadata(
        index_root,
        VectorIndexInfo(
            schema_version=INDEX_SCHEMA_VERSION,
            fingerprint=fingerprint,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            overlap=overlap,
            chunks=[asdict(chunk) for chunk in chunks],
        ),
    )
    return index, list(chunks), True


def _matched_terms(content: str, query_terms: Iterable[str]) -> List[str]:
    content_lower = content.lower()
    matched: List[str] = []
    for term in query_terms or []:
        term = str(term).strip()
        if term and (term.lower() in content_lower or term in content):
            matched.append(term)
    return list(dict.fromkeys(matched))


def retrieve_vector_knowledge(
    query_text: str,
    query_terms: Sequence[str],
    knowledge_dir: str | Path,
    index_dir: str | Path,
    embedding_model: str,
    top_k: int = 4,
    chunk_size: int = 420,
    overlap: int = 60,
    device: str = "cpu",
    encoder: Any | None = None,
) -> List[dict[str, Any]]:
    """Retrieve semantically similar chunks with normalized FAISS inner product."""

    query = normalize_text(query_text)
    if not query:
        query = "；".join(str(term) for term in query_terms if str(term).strip())
    if not query or top_k <= 0:
        return []

    chunks = load_knowledge_chunks(
        knowledge_dir,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if not chunks:
        return []

    np, _ = _load_vector_dependencies()
    model = encoder or _load_sentence_transformer(embedding_model, device)
    index, indexed_chunks, rebuilt = build_or_load_faiss_index(
        chunks=chunks,
        embedding_model=embedding_model,
        index_dir=index_dir,
        chunk_size=chunk_size,
        overlap=overlap,
        device=device,
        encoder=model,
    )
    query_vector = _encode_texts(model, [query], np)
    limit = min(int(top_k), len(indexed_chunks))
    similarities, positions = index.search(query_vector, limit)

    results: List[dict[str, Any]] = []
    for rank, (position, similarity) in enumerate(
        zip(positions[0].tolist(), similarities[0].tolist()),
        start=1,
    ):
        if position < 0 or position >= len(indexed_chunks):
            continue
        chunk = indexed_chunks[position]
        bounded_similarity = max(0.0, min(1.0, float(similarity)))
        vector_score = round(bounded_similarity * 100, 1)
        preview = (
            chunk.content
            if len(chunk.content) <= 240
            else chunk.content[:240].rstrip() + "..."
        )
        results.append(
            {
                "source": chunk.source,
                "source_name": chunk.source_name,
                "rank": rank,
                "chunk_index": chunk.chunk_index,
                "score": vector_score,
                "content": chunk.content,
                "preview": preview,
                "matched_terms": _matched_terms(chunk.content, query_terms),
                "retrieval_reason": f"向量语义相似度 {vector_score:.1f} / 100",
                "retrieval_mode": "vector",
                "embedding_model": embedding_model,
                "vector_score": vector_score,
                "keyword_score": 0.0,
                "index_rebuilt": rebuilt,
                "fallback_reason": "",
            }
        )
    return results
