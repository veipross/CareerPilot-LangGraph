from careerpilot.graph import match_node


JD_TEXT = "岗位要求：Python、LangGraph、RAG、Agent、FastAPI、Docker"


def _match(resume_text: str) -> dict:
    state = {
        "resume_text": resume_text,
        "jd_text": JD_TEXT,
        "profile": {"skills": []},
        "jd_profile": {
            "core_requirements": [],
            "tools": [],
            "keywords": [],
        },
    }
    return match_node(state)["match_report"]


def test_high_medium_low_cases_are_ordered():
    high = _match(
        "使用 Python、LangGraph、RAG、Agent、FastAPI 和 Docker 构建求职分析系统。"
    )["score"]
    medium = _match(
        "使用 Python、LangGraph 和 Agent 构建多节点工作流。"
    )["score"]
    low = _match(
        "主要使用 Python 完成传统数据处理项目。"
    )["score"]

    assert high == 100.0
    assert medium == 50.0
    assert low == 16.7
    assert high > medium > low


def test_match_report_contains_explainable_fields():
    report = _match("使用 Python、LangGraph 和 Agent 构建项目。")

    assert report["level"] == "中匹配"
    assert report["matched_count"] == 3
    assert report["required_count"] == 6
    assert report["score_formula"] == "3 / 6 × 100 = 50.0"
    assert report["score_explanation"]
    assert "录用概率" in " ".join(report["score_explanation"])


def test_free_form_llm_sentences_do_not_inflate_denominator():
    state = {
        "resume_text": "Python LangGraph",
        "jd_text": "Python LangGraph RAG Agent",
        "profile": {
            "skills": ["熟练掌握 Python 并具有工程经验", "LangGraph"],
        },
        "jd_profile": {
            "core_requirements": [
                "熟悉 Python 后端开发和大型项目协作",
                "具备 LangGraph、RAG 与 Agent 实践经验",
                "本科及以上学历",
            ],
            "tools": [],
            "keywords": [],
        },
    }

    report = match_node(state)["match_report"]

    assert report["required_count"] == 4
    assert report["matched_count"] == 2
    assert report["score"] == 50.0
