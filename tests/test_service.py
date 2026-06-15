from careerpilot.service import build_initial_state, run_careerpilot


def test_build_initial_state():
    state = build_initial_state(
        resume_text="Python LangGraph Agent",
        jd_text="需要 LangGraph RAG Agent",
        target_role="大模型应用实习生",
    )

    assert state["resume_text"]
    assert state["jd_text"]
    assert state["target_role"] == "大模型应用实习生"


def test_run_careerpilot_offline():
    result = run_careerpilot(
        resume_text="项目：使用 Python 和 LangGraph 构建 Agent 系统",
        jd_text="岗位要求：Python、LangGraph、RAG、Agent",
        offline=True,
    )

    assert "final_report" in result
    assert result["match_report"]["score"] >= 0
    assert result["project_plan"]["repo_name"] == "CareerPilot-LangGraph"