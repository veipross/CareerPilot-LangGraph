from careerpilot.tools import compute_match, extract_project_lines, extract_skills, recommend_project_features


def test_extract_skills_aliases():
    text = "使用通义千问和 DashScope 构建大模型 Agent，并结合模型量化和 TensorRT。"
    skills = extract_skills(text)
    assert "Qwen" in skills
    assert "DashScope" in skills
    assert "Agent" in skills
    assert "量化" in skills
    assert "TensorRT" in skills


def test_compute_match():
    score, matched, missing = compute_match(["Python", "LangGraph"], ["Python", "RAG", "LangGraph"])
    assert score == 66.7
    assert matched == ["Python", "LangGraph"]
    assert missing == ["RAG"]


def test_recommend_features():
    features = recommend_project_features(["RAG", "Tool Calling"], ["CUDA", "TensorRT"])
    joined = "\n".join(features)
    assert "RAG" in joined
    assert "工具调用" in joined
    assert "推理性能" in joined


def test_extract_project_lines():
    text = "项目一：Agent 系统\n普通描述\n项目二：推理优化框架"
    lines = extract_project_lines(text)
    assert len(lines) == 2



def test_recommend_open_source_projects():
    from careerpilot.tools import recommend_open_source_projects

    recs = recommend_open_source_projects(
        missing_skills=["LangGraph", "Agent", "RAG"],
        matched_skills=["Python", "Qwen", "DashScope"],
        target_role="大模型/Agent 工程实习生",
    )

    assert recs
    assert recs[0]["fit_score"] >= recs[-1]["fit_score"]
    assert "repo" in recs[0]
    assert recs[0]["contribution_ideas"]
