from careerpilot.rag_evaluation import (
    evaluate_case_result,
    hit_at_k,
    percentile,
    reciprocal_rank,
    render_markdown,
    summarize_mode,
)


def test_reciprocal_rank_and_hit_at_k():
    ranked = ["agent.txt", "rag.txt", "graph.txt"]
    expected = ["rag.txt"]
    assert reciprocal_rank(ranked, expected) == 0.5
    assert hit_at_k(ranked, expected, 1) == 0.0
    assert hit_at_k(ranked, expected, 3) == 1.0


def test_evaluate_case_result_tracks_relevant_rank():
    case = {
        "id": "rag_case",
        "query": "semantic retrieval",
        "query_type": "semantic_paraphrase",
        "expected_sources": ["rag_notes.txt"],
    }
    retrieved = [
        {"rank": 1, "source_name": "agent_notes.txt", "chunk_index": 1, "score": 70},
        {"rank": 2, "source_name": "rag_notes.txt", "chunk_index": 1, "score": 65},
    ]
    row = evaluate_case_result(case, retrieved, 12.3456)
    assert row["relevant_rank"] == 2
    assert row["hit_at_1"] == 0.0
    assert row["hit_at_3"] == 1.0
    assert row["mrr"] == 0.5
    assert row["latency_ms"] == 12.346


def test_percentile_uses_nearest_rank():
    assert percentile([], 95) == 0.0
    assert percentile([1, 2, 3, 4], 50) == 2.0
    assert percentile([1, 2, 3, 4], 95) == 4.0


def test_summarize_mode_aggregates_query_types():
    rows = [
        {
            "query_type": "semantic",
            "hit_at_1": 1.0,
            "hit_at_3": 1.0,
            "mrr": 1.0,
            "ranked_sources": ["a.txt"],
            "latency_ms": 10.0,
        },
        {
            "query_type": "exact",
            "hit_at_1": 0.0,
            "hit_at_3": 1.0,
            "mrr": 0.5,
            "ranked_sources": ["b.txt"],
            "latency_ms": 20.0,
        },
    ]
    summary = summarize_mode("hybrid", rows)
    assert summary["metrics"]["hit_at_1"] == 0.5
    assert summary["metrics"]["hit_at_3"] == 1.0
    assert summary["metrics"]["mrr"] == 0.75
    assert summary["metrics"]["mean_latency_ms"] == 15.0
    assert set(summary["by_query_type"]) == {"exact", "semantic"}


def test_render_markdown_contains_comparison_table():
    report = {
        "embedding_model": "test-model",
        "case_count": 1,
        "top_k": 3,
        "repeat": 1,
        "warmup_ms": 20.0,
        "results": [
            {
                "mode": "keyword",
                "metrics": {
                    "hit_at_1": 1.0,
                    "hit_at_3": 1.0,
                    "mrr": 1.0,
                    "non_empty_rate": 1.0,
                    "mean_latency_ms": 1.0,
                    "median_latency_ms": 1.0,
                    "p95_latency_ms": 1.0,
                },
                "by_query_type": {},
                "cases": [],
            }
        ],
    }
    markdown = render_markdown(report)
    assert "CareerPilot RAG 检索评估报告" in markdown
    assert "| keyword |" in markdown
    assert "Hit@1" in markdown
