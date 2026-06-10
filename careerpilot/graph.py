"""LangGraph workflow for the career agent."""
from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from .llm import invoke_json
from .schemas import CareerState, InterviewPlan, JDProfile, MatchReport, ProjectPlan, ResumeProfile, ResumeRewrite
from .tools import (
    compute_match,
    extract_project_lines,
    extract_skills,
    recommend_project_features,
    top_keywords,
)


def _merge_errors(state: CareerState, message: str) -> Dict[str, Any]:
    return {"errors": [*(state.get("errors") or []), message]}


def extract_profile_node(state: CareerState, llm: Any | None = None) -> Dict[str, Any]:
    resume_text = state.get("resume_text", "")
    if llm is None:
        profile = ResumeProfile(
            education="人工智能方向硕士研究生",
            target_roles=["AI 大模型", "算法实习生", "大模型应用实习生"],
            skills=extract_skills(resume_text),
            projects=extract_project_lines(resume_text),
            strengths=["LLM 应用经验", "CV/推理优化背景", "Python/PyTorch 工程能力"],
            missing_signals=["开源贡献", "Agent 项目", "RAG/工具调用评测"],
        )
        return {"profile": profile.model_dump()}

    prompt = f"""
你是资深 AI 秋招简历顾问。请从下面的简历中抽取结构化信息，只返回 JSON，不要 Markdown。
字段：education, target_roles, skills, projects, strengths, missing_signals。
简历：
{resume_text}
"""
    data = invoke_json(llm, prompt)
    try:
        return {"profile": ResumeProfile(**data).model_dump()}
    except Exception:
        return {**_merge_errors(state, "LLM resume profile parsing failed; used offline extractor."), "profile": ResumeProfile(skills=extract_skills(resume_text), projects=extract_project_lines(resume_text)).model_dump()}


def analyze_jd_node(state: CareerState, llm: Any | None = None) -> Dict[str, Any]:
    jd_text = state.get("jd_text", "")
    offline_keywords = top_keywords(jd_text)
    offline_skills = extract_skills(jd_text)

    if llm is None:
        jd = JDProfile(
            role=state.get("target_role") or "大模型/Agent 工程实习生",
            core_requirements=offline_skills[:8],
            preferred_requirements=[kw for kw in offline_keywords if kw not in offline_skills][:6],
            tools=offline_skills,
            keywords=offline_keywords,
        )
        return {"jd_profile": jd.model_dump()}

    prompt = f"""
你是 AI 招聘 JD 分析器。请从下面 JD 中抽取结构化信息，只返回 JSON，不要 Markdown。
字段：role, core_requirements, preferred_requirements, tools, keywords。
JD：
{jd_text}
"""
    data = invoke_json(llm, prompt)
    try:
        return {"jd_profile": JDProfile(**data).model_dump()}
    except Exception:
        return {**_merge_errors(state, "LLM JD parsing failed; used offline extractor."), "jd_profile": JDProfile(role="大模型/Agent 工程实习生", core_requirements=offline_skills, tools=offline_skills, keywords=offline_keywords).model_dump()}


def match_node(state: CareerState) -> Dict[str, Any]:
    profile = state.get("profile", {})
    jd = state.get("jd_profile", {})
    resume_skills = profile.get("skills", [])
    jd_skills = list(dict.fromkeys([*jd.get("core_requirements", []), *jd.get("tools", [])]))
    score, matched, missing = compute_match(resume_skills, jd_skills)

    risk_points = []
    if "LangGraph" in missing:
        risk_points.append("简历中还缺少 LangGraph 明确项目经历。")
    if "RAG" in missing:
        risk_points.append("简历中 RAG/向量检索信号较弱。")
    if "Tool Calling" in missing and "Function Calling" in missing:
        risk_points.append("Agent 工具调用、函数调用经历需要补强。")

    positioning = (
        "适合定位为：有 CV/推理优化背景的大模型应用/Agent 工程候选人。"
        if score >= 45
        else "当前匹配度一般，需要尽快补一个可演示 Agent 项目和开源贡献。"
    )

    report = MatchReport(
        score=score,
        matched_skills=matched,
        missing_skills=missing,
        risk_points=risk_points,
        positioning=positioning,
    )
    return {"match_report": report.model_dump()}


def project_planner_node(state: CareerState, llm: Any | None = None) -> Dict[str, Any]:
    match = state.get("match_report", {})
    features = recommend_project_features(match.get("missing_skills", []), match.get("matched_skills", []))

    base_plan = ProjectPlan(
        repo_name="CareerPilot-LangGraph",
        github_projects_to_study=["langchain-ai/langgraph", "langchain-ai/langchain", "QwenLM/Qwen-Agent", "run-llama/llama_index"],
        features=features,
        milestones=[
            "M1：跑通离线版流程，生成 Markdown 匹配报告",
            "M2：接入 Qwen/DashScope，完成结构化抽取和简历改写",
            "M3：加入 RAG 知识库和 GitHub issue 推荐工具",
            "M4：补测试、README、架构图，准备投开源 PR",
        ],
        resume_bullets=[
            "基于 LangGraph 设计多节点 Agent 工作流，实现简历解析、JD 分析、技能匹配、项目规划与简历改写的端到端自动化。",
            "接入 Qwen/DashScope OpenAI-compatible API，结合结构化输出约束生成可复用的 JSON/Markdown 求职分析报告。",
            "设计确定性匹配工具与可扩展工具调用接口，为后续 RAG、GitHub issue 检索和 vLLM serving benchmark 预留扩展点。",
        ],
    )

    if llm is None:
        return {"project_plan": base_plan.model_dump()}

    prompt = f"""
你是大模型 Agent 项目导师。基于以下匹配报告，优化项目路线。只返回 JSON。
字段：repo_name, github_projects_to_study, features, milestones, resume_bullets。
匹配报告：{match}
初始方案：{base_plan.model_dump()}
"""
    data = invoke_json(llm, prompt)
    try:
        return {"project_plan": ProjectPlan(**data).model_dump()}
    except Exception:
        return {**_merge_errors(state, "LLM project planning failed; used base plan."), "project_plan": base_plan.model_dump()}


def resume_rewriter_node(state: CareerState, llm: Any | None = None) -> Dict[str, Any]:
    project_plan = state.get("project_plan", {})
    match = state.get("match_report", {})
    default = ResumeRewrite(
        summary="有 LLM 应用、CV 与推理优化背景，正在补强 LangGraph/Agent 工程化项目经验，适合大模型应用、Agent 工程和算法实习岗位。",
        rewritten_project_bullets=project_plan.get("resume_bullets", []),
        keywords_to_add=["LangGraph", "Agent", "Tool Calling", "RAG", "结构化输出", "Qwen", "DashScope"],
    )
    if llm is None:
        return {"resume_rewrite": default.model_dump()}

    prompt = f"""
你是 AI 简历优化专家。请把项目方案改写成适合秋招简历的一段项目描述和 3-5 条 bullet。只返回 JSON。
字段：summary, rewritten_project_bullets, keywords_to_add。
项目方案：{project_plan}
匹配报告：{match}
"""
    data = invoke_json(llm, prompt)
    try:
        return {"resume_rewrite": ResumeRewrite(**data).model_dump()}
    except Exception:
        return {**_merge_errors(state, "LLM resume rewrite failed; used default bullets."), "resume_rewrite": default.model_dump()}


def interview_planner_node(state: CareerState, llm: Any | None = None) -> Dict[str, Any]:
    default = InterviewPlan(
        questions=[
            "为什么选择 LangGraph，而不是只用普通 Chain 或单轮 Prompt？",
            "你的 Agent 状态 State 里保存了哪些字段？节点之间如何传递结果？",
            "如何评估 JD 匹配结果是否可靠？有没有确定性规则和 LLM 输出校验？",
            "如果模型输出 JSON 解析失败，你的系统如何降级？",
            "后续如何把这个项目扩展到 RAG、GitHub issue 推荐或 MCP？",
        ],
        talking_points=[
            "强调这是一个状态图编排项目，不是简单聊天机器人。",
            "强调离线规则 + LLM 结构化输出双路径，工程鲁棒性更好。",
            "结合自己已有 DashScope、Prompt、多模态数据增强和推理优化经历。",
        ],
        study_plan=[
            "第 1 周：熟悉 StateGraph、START/END、节点状态更新和持久化思路。",
            "第 2 周：接入 Qwen/DashScope，完成结构化 JSON 抽取和错误降级。",
            "第 3 周：加入 RAG 或 GitHub issue 工具，写 README 和测试。",
        ],
    )
    if llm is None:
        return {"interview_plan": default.model_dump()}

    prompt = f"""
你是大模型岗位面试官。请基于项目内容生成面试准备方案。只返回 JSON。
字段：questions, talking_points, study_plan。
状态：{state}
"""
    data = invoke_json(llm, prompt)
    try:
        return {"interview_plan": InterviewPlan(**data).model_dump()}
    except Exception:
        return {**_merge_errors(state, "LLM interview plan failed; used default plan."), "interview_plan": default.model_dump()}


def final_report_node(state: CareerState) -> Dict[str, Any]:
    profile = state.get("profile", {})
    jd = state.get("jd_profile", {})
    match = state.get("match_report", {})
    project = state.get("project_plan", {})
    rewrite = state.get("resume_rewrite", {})
    interview = state.get("interview_plan", {})
    errors = state.get("errors") or []

    def bullets(items):
        return "\n".join(f"- {item}" for item in items) if items else "- 暂无"

    report = f"""# CareerPilot-LangGraph 求职匹配报告

## 1. 岗位定位
- 目标岗位：{jd.get('role', state.get('target_role', '大模型/Agent 工程实习生'))}
- 候选人定位：{match.get('positioning', '')}
- 匹配分：**{match.get('score', 0)} / 100**

## 2. 已匹配技能
{bullets(match.get('matched_skills', []))}

## 3. 需要补强的技能/信号
{bullets(match.get('missing_skills', []))}

## 4. 风险点
{bullets(match.get('risk_points', []))}

## 5. 推荐 GitHub 项目路线
仓库名：**{project.get('repo_name', 'CareerPilot-LangGraph')}**

### 核心功能
{bullets(project.get('features', []))}

### 里程碑
{bullets(project.get('milestones', []))}

### 建议学习/贡献仓库
{bullets(project.get('github_projects_to_study', []))}

## 6. 可写入简历的项目描述
{rewrite.get('summary', '')}

{bullets(rewrite.get('rewritten_project_bullets', []))}

关键词：{', '.join(rewrite.get('keywords_to_add', []))}

## 7. 面试准备
### 高频问题
{bullets(interview.get('questions', []))}

### 讲解要点
{bullets(interview.get('talking_points', []))}

### 三周计划
{bullets(interview.get('study_plan', []))}
"""
    if errors:
        report += "\n## 8. 运行提示\n" + bullets(errors) + "\n"
    return {"final_report": report}


def build_graph(llm: Any | None = None):
    """Build the LangGraph state machine."""

    workflow = StateGraph(CareerState)
    workflow.add_node("extract_profile", lambda state: extract_profile_node(state, llm))
    workflow.add_node("analyze_jd", lambda state: analyze_jd_node(state, llm))
    workflow.add_node("match", match_node)
    workflow.add_node("project_planner", lambda state: project_planner_node(state, llm))
    workflow.add_node("resume_rewriter", lambda state: resume_rewriter_node(state, llm))
    workflow.add_node("interview_planner", lambda state: interview_planner_node(state, llm))
    workflow.add_node("final_report", final_report_node)

    workflow.add_edge(START, "extract_profile")
    workflow.add_edge("extract_profile", "analyze_jd")
    workflow.add_edge("analyze_jd", "match")
    workflow.add_edge("match", "project_planner")
    workflow.add_edge("project_planner", "resume_rewriter")
    workflow.add_edge("resume_rewriter", "interview_planner")
    workflow.add_edge("interview_planner", "final_report")
    workflow.add_edge("final_report", END)
    return workflow.compile()
