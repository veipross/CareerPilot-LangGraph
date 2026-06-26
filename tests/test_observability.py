from careerpilot.service import run_careerpilot
from careerpilot.tools import retrieve_knowledge


EXPECTED_NODE_ORDER = [
    "extract_profile",
    "analyze_jd",
    "match",
    "rag_retriever",
    "project_planner",
    "github_recommender",
    "resume_rewriter",
    "interview_planner",
    "final_report",
]


def test_offline_pipeline_exposes_complete_execution_trace():
    result = run_careerpilot(
        resume_text=(
            "使用 Python、LangGraph 和 Agent 构建求职分析系统，"
            "并通过 FastAPI 提供接口。"
        ),
        jd_text=(
            "岗位要求：Python、LangGraph、RAG、Agent、FastAPI、Docker。"
        ),
        offline=True,
    )

    trace = result["execution_trace"]
    metrics = result["pipeline_metrics"]

    assert [item["node"] for item in trace] == EXPECTED_NODE_ORDER
    assert [item["index"] for item in trace] == list(range(1, 10))
    assert all(item["status"] == "completed" for item in trace)
    assert all(item["duration_ms"] >= 0 for item in trace)
    assert all(item["summary"] for item in trace)

    assert metrics["node_count"] == 9
    assert metrics["completed_count"] == 9
    assert metrics["total_duration_ms"] >= 0
    assert metrics["slowest_node"] in EXPECTED_NODE_ORDER


def test_trace_contains_rag_and_match_summaries():
    result = run_careerpilot(
        resume_text="Python LangGraph Agent FastAPI 项目",
        jd_text="Python LangGraph RAG Agent FastAPI Docker",
        offline=True,
    )

    trace_by_node = {
        item["node"]: item
        for item in result["execution_trace"]
    }

    assert "匹配分" in trace_by_node["match"]["summary"]
    assert "知识片段" in trace_by_node["rag_retriever"]["summary"]
    assert "报告长度" in trace_by_node["final_report"]["summary"]


def test_retrieval_results_expose_source_metadata(tmp_path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "rag_notes.txt").write_text(
        "RAG 检索增强系统需要文档切分、向量检索与 Agent 编排。",
        encoding="utf-8",
    )

    results = retrieve_knowledge(
        query_terms=["RAG", "Agent"],
        knowledge_dir=str(knowledge_dir),
        top_k=2,
    )

    assert results
    first = results[0]
    assert first["rank"] == 1
    assert first["source_name"] == "rag_notes.txt"
    assert first["chunk_index"] == 1
    assert first["preview"]
    assert "命中" in first["retrieval_reason"]
