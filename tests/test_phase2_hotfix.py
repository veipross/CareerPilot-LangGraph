from careerpilot.graph import extract_profile_node, interview_planner_node


class StubMessage:
    def __init__(self, content):
        self.content = content


class StubLLM:
    def __init__(self, content):
        self.content = content

    def invoke(self, prompt):
        return StubMessage(self.content)


def test_profile_normalizes_string_and_alias_fields():
    state = {"resume_text": "Python LangGraph 项目"}
    llm = StubLLM(
        '{"技术栈":"Python、LangGraph","项目经历":"CareerPilot Agent"}'
    )
    result = extract_profile_node(state, llm)
    assert result["profile"]["skills"] == ["Python", "LangGraph"]
    assert result["profile"]["projects"] == ["CareerPilot Agent"]


def test_interview_empty_payload_uses_nonempty_default():
    result = interview_planner_node({}, StubLLM("{}"))
    assert len(result["interview_plan"]["questions"]) >= 5
    assert result["errors"]


def test_interview_alias_payload_is_accepted():
    llm = StubLLM(
        '{"高频问题":["为什么使用 LangGraph？"],"讲解要点":["状态图"],"三周计划":["补测试"]}'
    )
    result = interview_planner_node({}, llm)
    assert result["interview_plan"]["questions"] == ["为什么使用 LangGraph？"]
