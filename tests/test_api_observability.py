from fastapi.testclient import TestClient

from careerpilot.api import app


client = TestClient(app)


def test_web_analyze_get_redirects_to_home():
    response = client.get(
        "/web/analyze",
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/"


def test_json_api_returns_trace_metrics_and_rag_sources():
    response = client.post(
        "/analyze",
        json={
            "resume_text": "Python LangGraph Agent FastAPI 项目",
            "jd_text": "Python LangGraph RAG Agent FastAPI Docker",
            "target_role": "大模型/Agent 工程实习生",
            "offline": True,
            "provider": "deepseek",
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert len(payload["execution_trace"]) == 9
    assert payload["pipeline_metrics"]["completed_count"] == 9
    assert payload["rag_context"]
    first_rag_item = payload["rag_context"][0]
    assert first_rag_item["source_name"]
    assert "retrieval_mode" in first_rag_item
    assert "vector_score" in first_rag_item
    assert "keyword_score" in first_rag_item
    assert "embedding_model" in first_rag_item


def test_web_report_renders_observability_sections():
    response = client.post(
        "/web/analyze",
        data={
            "resume_text": "Python LangGraph Agent FastAPI 项目",
            "jd_text": "Python LangGraph RAG Agent FastAPI Docker",
            "target_role": "大模型/Agent 工程实习生",
            "offline": "on",
            "provider": "deepseek",
            "model": "",
        },
    )

    assert response.status_code == 200
    assert "LangGraph 执行轨迹" in response.text
    assert "RAG 检索来源" in response.text
    assert "真实节点输出与耗时" in response.text
    assert "检索模式" in response.text
    assert "综合相关度" in response.text
    assert "向量相似度" in response.text
    assert "关键词相关度" in response.text
    assert "Embedding 模型" in response.text
    assert "索引状态" in response.text
