"""Metrics and report helpers for offline RAG retrieval evaluation."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def reciprocal_rank(
    ranked_sources: Sequence[str],
    expected_sources: Iterable[str],
) -> float:
    """Return reciprocal rank of the first relevant source, or 0 when absent."""

    expected = {str(source) for source in expected_sources if str(source)}
    for rank, source in enumerate(ranked_sources, start=1):
        if str(source) in expected:
            return 1.0 / rank
    return 0.0


def hit_at_k(
    ranked_sources: Sequence[str],
    expected_sources: Iterable[str],
    k: int,
) -> float:
    """Return 1.0 when any relevant source appears in the first *k* results."""

    if k <= 0:
        return 0.0
    expected = {str(source) for source in expected_sources if str(source)}
    return float(any(str(source) in expected for source in ranked_sources[:k]))


def percentile(values: Sequence[float], percent: float) -> float:
    """Compute a deterministic nearest-rank percentile without NumPy."""

    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    bounded = max(0.0, min(100.0, float(percent)))
    rank = max(1, math.ceil((bounded / 100.0) * len(ordered)))
    return ordered[rank - 1]


def evaluate_case_result(
    case: Mapping[str, Any],
    retrieved: Sequence[Mapping[str, Any]],
    latency_ms: float,
) -> Dict[str, Any]:
    """Convert one retrieval result into source-level ranking metrics."""

    expected_sources = [str(value) for value in case.get("expected_sources", [])]
    ranked_sources = [str(item.get("source_name", "")) for item in retrieved]
    rr = reciprocal_rank(ranked_sources, expected_sources)
    relevant_rank = int(round(1.0 / rr)) if rr > 0 else None

    return {
        "id": str(case.get("id", "")),
        "query": str(case.get("query", "")),
        "query_type": str(case.get("query_type", "unspecified")),
        "expected_sources": expected_sources,
        "ranked_sources": ranked_sources,
        "relevant_rank": relevant_rank,
        "hit_at_1": hit_at_k(ranked_sources, expected_sources, 1),
        "hit_at_3": hit_at_k(ranked_sources, expected_sources, 3),
        "mrr": rr,
        "latency_ms": round(float(latency_ms), 3),
        "retrieved": [
            {
                "rank": int(item.get("rank", index)),
                "source_name": str(item.get("source_name", "")),
                "chunk_index": int(item.get("chunk_index", 0)),
                "score": float(item.get("score", 0.0)),
                "vector_score": float(item.get("vector_score", 0.0)),
                "keyword_score": float(item.get("keyword_score", 0.0)),
                "retrieval_mode": str(item.get("retrieval_mode", "")),
            }
            for index, item in enumerate(retrieved, start=1)
        ],
    }


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "case_count": 0,
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "mrr": 0.0,
            "non_empty_rate": 0.0,
            "mean_latency_ms": 0.0,
            "median_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }

    latencies = [float(row.get("latency_ms", 0.0)) for row in rows]
    count = len(rows)
    return {
        "case_count": count,
        "hit_at_1": round(sum(float(row.get("hit_at_1", 0.0)) for row in rows) / count, 4),
        "hit_at_3": round(sum(float(row.get("hit_at_3", 0.0)) for row in rows) / count, 4),
        "mrr": round(sum(float(row.get("mrr", 0.0)) for row in rows) / count, 4),
        "non_empty_rate": round(
            sum(bool(row.get("ranked_sources")) for row in rows) / count,
            4,
        ),
        "mean_latency_ms": round(statistics.fmean(latencies), 3),
        "median_latency_ms": round(statistics.median(latencies), 3),
        "p95_latency_ms": round(percentile(latencies, 95.0), 3),
    }


def summarize_mode(
    mode: str,
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate overall and per-query-type metrics for one retrieval mode."""

    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("query_type", "unspecified")), []).append(row)

    return {
        "mode": mode,
        "metrics": _aggregate_rows(rows),
        "by_query_type": {
            name: _aggregate_rows(group_rows)
            for name, group_rows in sorted(groups.items())
        },
        "cases": list(rows),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render an evaluation report suitable for README or interview evidence."""

    lines = [
        "# CareerPilot RAG 检索评估报告",
        "",
        f"- Embedding 模型：`{report.get('embedding_model', '')}`",
        f"- 评估样本数：{report.get('case_count', 0)}",
        f"- Top-K：{report.get('top_k', 0)}",
        f"- 重复次数：{report.get('repeat', 1)}",
        f"- 向量后端预热耗时：{report.get('warmup_ms', 0.0):.1f} ms",
        "",
        "## 总体指标",
        "",
        "| 模式 | Hit@1 | Hit@3 | MRR | 非空召回率 | 平均延迟(ms) | P50(ms) | P95(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for mode_result in report.get("results", []):
        metrics = mode_result.get("metrics", {})
        lines.append(
            "| {mode} | {h1:.3f} | {h3:.3f} | {mrr:.3f} | {nonempty:.3f} | "
            "{mean:.2f} | {median:.2f} | {p95:.2f} |".format(
                mode=mode_result.get("mode", ""),
                h1=float(metrics.get("hit_at_1", 0.0)),
                h3=float(metrics.get("hit_at_3", 0.0)),
                mrr=float(metrics.get("mrr", 0.0)),
                nonempty=float(metrics.get("non_empty_rate", 0.0)),
                mean=float(metrics.get("mean_latency_ms", 0.0)),
                median=float(metrics.get("median_latency_ms", 0.0)),
                p95=float(metrics.get("p95_latency_ms", 0.0)),
            )
        )

    lines.extend(["", "## 分类型指标", ""])
    for mode_result in report.get("results", []):
        lines.append(f"### {mode_result.get('mode', '')}")
        lines.append("")
        lines.append("| 查询类型 | 数量 | Hit@1 | Hit@3 | MRR |")
        lines.append("|---|---:|---:|---:|---:|")
        for query_type, metrics in mode_result.get("by_query_type", {}).items():
            lines.append(
                "| {kind} | {count} | {h1:.3f} | {h3:.3f} | {mrr:.3f} |".format(
                    kind=query_type,
                    count=int(metrics.get("case_count", 0)),
                    h1=float(metrics.get("hit_at_1", 0.0)),
                    h3=float(metrics.get("hit_at_3", 0.0)),
                    mrr=float(metrics.get("mrr", 0.0)),
                )
            )
        lines.append("")

    lines.extend(["## 逐样本结果", ""])
    for mode_result in report.get("results", []):
        lines.append(f"### {mode_result.get('mode', '')}")
        lines.append("")
        lines.append("| ID | 类型 | 相关排名 | 期望来源 | Top-3 来源 | 延迟(ms) |")
        lines.append("|---|---|---:|---|---|---:|")
        for row in mode_result.get("cases", []):
            expected = ", ".join(row.get("expected_sources", [])) or "-"
            ranked = ", ".join(row.get("ranked_sources", [])[:3]) or "-"
            rank = row.get("relevant_rank") or "未命中"
            lines.append(
                f"| {row.get('id', '')} | {row.get('query_type', '')} | {rank} | "
                f"{expected} | {ranked} | {float(row.get('latency_ms', 0.0)):.2f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 指标说明",
            "",
            "- Hit@1：第一个结果是否来自期望知识文件。",
            "- Hit@3：前三个结果中是否至少有一个来自期望知识文件。",
            "- MRR：首个相关结果排名的倒数均值，越接近 1 越好。",
            "- 延迟为完成模型与索引预热后的单次检索时间，不包含首次模型下载。",
            "",
        ]
    )
    return "\n".join(lines)


def save_report(report: Mapping[str, Any], output_dir: str | Path) -> Dict[str, Path]:
    """Persist machine-readable JSON and human-readable Markdown reports."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "rag_evaluation.json"
    markdown_path = root / "rag_evaluation.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
