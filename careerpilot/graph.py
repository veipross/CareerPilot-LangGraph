"""LangGraph workflow for the career agent."""
from __future__ import annotations

from time import perf_counter
import re
from typing import Any, Callable, Dict

from langgraph.graph import END, START, StateGraph

from .llm import invoke_json
from .schemas import CareerState, ExecutionTraceItem, InterviewPlan, JDProfile, MatchReport, OpenSourceRecommendation, ProjectPlan, RAGContext, ResumeProfile, ResumeRewrite
from .tools import (
    canonicalize_skill_items,
    classify_match_level,
    compute_match,
    extract_project_lines,
    extract_skills,
    recommend_project_features,
    recommend_open_source_projects,
    retrieve_knowledge,
    top_keywords,
)


def _merge_errors(state: CareerState, message: str) -> Dict[str, Any]:
    return {"errors": [*(state.get("errors") or []), message]}


NODE_LABELS = {
    "extract_profile": "简历解析",
    "analyze_jd": "JD 分析",
    "match": "技能匹配",
    "rag_retriever": "RAG 知识检索",
    "project_planner": "项目路线规划",
    "github_recommender": "开源项目推荐",
    "resume_rewriter": "简历改写",
    "interview_planner": "面试题生成",
    "final_report": "最终报告",
}



def _first_value(data: Dict[str, Any], *keys: str) -> Any:
    """Return the first present, non-empty value from a model payload."""

    for key in keys:
        if key in data and data[key] not in (None, "", [], {}):
            return data[key]
    return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _as_text_list(value: Any) -> list[str]:
    """Normalize common LLM list variants into a clean string list."""

    if value is None:
        return []

    if isinstance(value, dict):
        value = list(value.values())

    if isinstance(value, (tuple, set)):
        value = list(value)

    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = _as_text(
                    _first_value(
                        item,
                        "name",
                        "title",
                        "content",
                        "description",
                        "value",
                    )
                )
            else:
                text = _as_text(item)
            if text:
                items.append(text)
        return list(dict.fromkeys(items))

    text = _as_text(value)
    if not text:
        return []

    parts = re.split(r"[\n；;、]+|(?<!\d),(?!\d)", text)
    items = [part.strip(" -•\t") for part in parts if part.strip(" -•\t")]
    return list(dict.fromkeys(items))


def _normalize_profile_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "education": _as_text(
            _first_value(data, "education", "education_background", "学历", "教育背景")
        ),
        "target_roles": _as_text_list(
            _first_value(data, "target_roles", "target_positions", "目标岗位", "求职方向")
        ),
        "skills": _as_text_list(
            _first_value(data, "skills", "technical_skills", "技能", "技术栈")
        ),
        "projects": _as_text_list(
            _first_value(data, "projects", "project_experience", "项目", "项目经历")
        ),
        "strengths": _as_text_list(
            _first_value(data, "strengths", "advantages", "优势", "亮点")
        ),
        "missing_signals": _as_text_list(
            _first_value(data, "missing_signals", "gaps", "缺失信号", "不足")
        ),
    }


def _normalize_interview_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "questions": _as_text_list(
            _first_value(
                data,
                "questions",
                "interview_questions",
                "high_frequency_questions",
                "高频问题",
                "面试问题",
            )
        ),
        "talking_points": _as_text_list(
            _first_value(data, "talking_points", "key_points", "讲解要点", "回答要点")
        ),
        "study_plan": _as_text_list(
            _first_value(data, "study_plan", "three_week_plan", "三周计划", "学习计划")
        ),
    }


def _summarize_node_output(node_name: str, update: Dict[str, Any]) -> str:
    """Build a concise human-readable summary for one node output."""

    if node_name == "extract_profile":
        profile = update.get("profile") or {}
        return (
            f"抽取 {len(profile.get('skills', []))} 项技能、"
            f"{len(profile.get('projects', []))} 条项目经历"
        )

    if node_name == "analyze_jd":
        jd = update.get("jd_profile") or {}
        requirement_count = len(
            list(
                dict.fromkeys(
                    [
                        *jd.get("core_requirements", []),
                        *jd.get("tools", []),
                    ]
                )
            )
        )
        return f"识别岗位 {jd.get('role', '未命名')}，提取 {requirement_count} 项要求"

    if node_name == "match":
        match = update.get("match_report") or {}
        return (
            f"匹配分 {float(match.get('score', 0)):.1f}，"
            f"命中 {match.get('matched_count', 0)}/{match.get('required_count', 0)} 项技能"
        )

    if node_name == "rag_retriever":
        items = update.get("rag_context") or []
        sources = {
            item.get("source_name") or item.get("source")
            for item in items
            if item.get("source_name") or item.get("source")
        }
        return f"召回 {len(items)} 个知识片段，来自 {len(sources)} 个本地来源"

    if node_name == "project_planner":
        plan = update.get("project_plan") or {}
        return (
            f"生成 {len(plan.get('features', []))} 项功能建议和 "
            f"{len(plan.get('milestones', []))} 个里程碑"
        )

    if node_name == "github_recommender":
        items = update.get("github_recommendations") or []
        return f"推荐 {len(items)} 个开源仓库与贡献切入点"

    if node_name == "resume_rewriter":
        rewrite = update.get("resume_rewrite") or {}
        return f"生成 {len(rewrite.get('rewritten_project_bullets', []))} 条简历项目 bullet"

    if node_name == "interview_planner":
        plan = update.get("interview_plan") or {}
        return f"生成 {len(plan.get('questions', []))} 道面试问题"

    if node_name == "final_report":
        report = update.get("final_report") or ""
        return f"汇总完成，报告长度 {len(report)} 字符"

    return "节点执行完成"


def _build_pipeline_metrics(trace: list[dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate trace entries into lightweight pipeline metrics."""

    total_duration_ms = round(
        sum(float(item.get("duration_ms", 0.0)) for item in trace),
        1,
    )
    slowest = max(
        trace,
        key=lambda item: float(item.get("duration_ms", 0.0)),
        default={},
    )

    return {
        "node_count": len(trace),
        "completed_count": sum(
            1 for item in trace if item.get("status") == "completed"
        ),
        "total_duration_ms": total_duration_ms,
        "slowest_node": slowest.get("node", ""),
        "slowest_label": slowest.get("label", ""),
        "slowest_duration_ms": round(
            float(slowest.get("duration_ms", 0.0)),
            1,
        ),
    }


def _instrument_node(
    node_name: str,
    function: Callable[[CareerState], Dict[str, Any]],
) -> Callable[[CareerState], Dict[str, Any]]:
    """Wrap a LangGraph node and append an actual execution trace entry."""

    label = NODE_LABELS.get(node_name, node_name)

    def wrapped(state: CareerState) -> Dict[str, Any]:
        started_at = perf_counter()
        try:
            update = function(state)
        except Exception as exc:
            duration_ms = round((perf_counter() - started_at) * 1000, 1)
            raise RuntimeError(
                f"LangGraph 节点 {label} ({node_name}) 执行失败，"
                f"耗时 {duration_ms:.1f} ms：{exc}"
            ) from exc

        duration_ms = round((perf_counter() - started_at) * 1000, 1)
        prior_trace = list(state.get("execution_trace") or [])
        trace_item = ExecutionTraceItem(
            index=len(prior_trace) + 1,
            node=node_name,
            label=label,
            status="completed",
            duration_ms=duration_ms,
            summary=_summarize_node_output(node_name, update),
            output_keys=sorted(
                key
                for key in update.keys()
                if key not in {"execution_trace", "pipeline_metrics"}
            ),
        ).model_dump()
        trace = [*prior_trace, trace_item]

        return {
            **update,
            "execution_trace": trace,
            "pipeline_metrics": _build_pipeline_metrics(trace),
        }

    return wrapped


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
        profile = ResumeProfile(**_normalize_profile_payload(data))
        if not profile.skills and not profile.projects:
            raise ValueError("model payload contains no usable skills or projects")
        return {"profile": profile.model_dump()}
    except Exception:
        return {
            **_merge_errors(
                state,
                "LLM resume profile parsing was incomplete; used deterministic extractor.",
            ),
            "profile": ResumeProfile(
                skills=extract_skills(resume_text),
                projects=extract_project_lines(resume_text),
            ).model_dump(),
        }


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
    """Compute a stable and explainable deterministic match score.

    LLM outputs are useful for narrative analysis, but free-form requirement
    sentences must not enter the score denominator. Both online and offline
    modes therefore map skill evidence to the same canonical vocabulary before
    calculating coverage.
    """

    profile = state.get("profile", {})
    jd = state.get("jd_profile", {})
    resume_text = state.get("resume_text", "")
    jd_text = state.get("jd_text", "")

    offline_resume_skills = extract_skills(resume_text)
    offline_jd_skills = extract_skills(jd_text)

    llm_resume_skills = canonicalize_skill_items(profile.get("skills", []))
    llm_jd_skills = canonicalize_skill_items(
        [
            *jd.get("core_requirements", []),
            *jd.get("tools", []),
            *jd.get("keywords", []),
        ]
    )

    resume_skills = list(
        dict.fromkeys([*offline_resume_skills, *llm_resume_skills])
    )
    jd_skills = list(
        dict.fromkeys([*offline_jd_skills, *llm_jd_skills])
    )

    score, matched, missing = compute_match(resume_skills, jd_skills)
    matched_count = len(matched)
    required_count = len(jd_skills)
    level = classify_match_level(score)

    if required_count:
        formula = f"{matched_count} / {required_count} × 100 = {score:.1f}"
    else:
        formula = "JD 中未识别到可评分的标准化技能"

    score_explanation = [
        f"JD 中识别出 {required_count} 个标准化技能，简历命中 {matched_count} 个。",
        f"计算公式：{formula}。" if required_count else formula,
        "总分表示技术技能覆盖率，不等同于录用概率，也不会为了展示效果人为调高。",
        "在线与离线模式使用同一套确定性评分词表，避免模型措辞变化造成分数漂移。",
    ]

    risk_points = []
    if not jd_skills:
        risk_points.append("JD 中没有识别到可评分技能，建议补充更完整的技术要求。")
    if "LangGraph" in missing:
        risk_points.append("简历中还缺少 LangGraph 明确项目经历。")
    if "RAG" in missing:
        risk_points.append("简历中 RAG/向量检索信号较弱。")
    if "Tool Calling" in missing and "Function Calling" in missing:
        risk_points.append("Agent 工具调用、函数调用经历需要补强。")

    if level == "高匹配":
        positioning = "岗位技能覆盖较完整，建议重点强化项目成果、量化指标和工程深度。"
    elif level == "中匹配":
        positioning = "具备部分核心能力，建议围绕缺口技能补充可运行项目与证据。"
    else:
        positioning = "当前岗位技能覆盖偏低，需要优先补齐核心技术栈和相关项目经历。"

    report = MatchReport(
        score=score,
        level=level,
        scoring_method="canonical_skill_coverage",
        score_formula=formula,
        matched_count=matched_count,
        required_count=required_count,
        resume_skill_count=len(resume_skills),
        matched_skills=matched,
        missing_skills=missing,
        score_explanation=score_explanation,
        risk_points=risk_points,
        positioning=positioning,
    )

    return {"match_report": report.model_dump()}

def rag_retriever_node(state: CareerState) -> Dict[str, Any]:
    """Retrieve local knowledge snippets based on JD and missing skills."""

    match = state.get("match_report", {})
    jd = state.get("jd_profile", {})

    query_terms = list(
        dict.fromkeys(
            [
                *match.get("missing_skills", []),
                *match.get("matched_skills", []),
                *jd.get("keywords", []),
                jd.get("role", ""),
            ]
        )
    )

    raw_items = retrieve_knowledge(query_terms=query_terms, knowledge_dir="data/knowledge", top_k=4)
    items = [RAGContext(**item).model_dump() for item in raw_items]
    return {"rag_context": items}


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



def github_recommender_node(state: CareerState) -> Dict[str, Any]:
    """Recommend open-source repos and concrete PR entry points."""

    match = state.get("match_report", {})
    jd = state.get("jd_profile", {})
    target_role = jd.get("role") or state.get("target_role") or "大模型/Agent 工程实习生"

    raw_items = recommend_open_source_projects(
        missing_skills=match.get("missing_skills", []),
        matched_skills=match.get("matched_skills", []),
        target_role=target_role,
        top_k=4,
    )

    items = [OpenSourceRecommendation(**item).model_dump() for item in raw_items]
    return {"github_recommendations": items}


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
        plan = InterviewPlan(**_normalize_interview_payload(data))
        if not plan.questions:
            raise ValueError("model payload contains no interview questions")
        if not plan.talking_points:
            plan.talking_points = default.talking_points
        if not plan.study_plan:
            plan.study_plan = default.study_plan
        return {"interview_plan": plan.model_dump()}
    except Exception:
        return {
            **_merge_errors(
                state,
                "LLM interview plan was empty or invalid; used verified default plan.",
            ),
            "interview_plan": default.model_dump(),
        }


def final_report_node(state: CareerState) -> Dict[str, Any]:
    profile = state.get("profile", {})
    jd = state.get("jd_profile", {})
    match = state.get("match_report", {})
    project = state.get("project_plan", {})
    rewrite = state.get("resume_rewrite", {})
    interview = state.get("interview_plan", {})
    github_recs = state.get("github_recommendations", [])
    rag_context = state.get("rag_context", [])
    errors = state.get("errors") or []

    def bullets(items):
        return "\n".join(f"- {item}" for item in items) if items else "- 暂无"

    def github_cards(items):
        if not items:
            return "- 暂无"

        blocks = []
        for item in items:
            ideas = "\n".join(
                f"  - {idea}" for idea in item.get("contribution_ideas", [])
            )
            skills = ", ".join(item.get("skills_to_learn", []))

            blocks.append(
                f"### {item.get('repo', 'unknown')}\n"
                f"- 适配分：{item.get('fit_score', 0)} / 100\n"
                f"- 推荐理由：{item.get('reason', '')}\n"
                f"- 重点补强：{skills or '暂无'}\n"
                f"- 贡献切入点：\n{ideas if ideas else '  - 暂无'}"
            )

        return "\n\n".join(blocks)

    def score_details(match_report):
        explanation = match_report.get("score_explanation", [])
        lines = [
            f"- 匹配等级：{match_report.get('level', '未评估')}",
            f"- 评分方法：标准化技能覆盖率（canonical skill coverage）",
            f"- 命中技能：{match_report.get('matched_count', 0)} / {match_report.get('required_count', 0)}",
            f"- 计算公式：{match_report.get('score_formula', '') or '暂无'}",
        ]
        lines.extend(f"- {item}" for item in explanation[2:])
        return "\n".join(lines)

    def rag_cards(items):
        if not items:
            return "- 暂无"

        blocks = []
        for item in items:
            content = item.get("content", "")
            if len(content) > 220:
                content = content[:220] + "..."

            matched = ", ".join(item.get("matched_terms", []))
            source_name = item.get("source_name") or item.get("source", "unknown")

            blocks.append(
                f"### Top {item.get('rank', 0)} · {source_name}\n"
                f"- 来源路径：{item.get('source', 'unknown')}\n"
                f"- 文档片段：Chunk {item.get('chunk_index', 0)}\n"
                f"- 关键词相关度：{item.get('score', 0)} / 100\n"
                f"- 命中词：{matched or '暂无'}\n"
                f"- 召回依据：{item.get('retrieval_reason', '') or '关键词重叠'}\n"
                f"- 片段：{content}"
            )

        return "\n\n".join(blocks)

    report = f"""# CareerPilot-LangGraph 求职匹配报告

## 1. 岗位定位
- 目标岗位：{jd.get('role', state.get('target_role', '大模型/Agent 工程实习生'))}
- 候选人定位：{match.get('positioning', '')}
- 匹配分：**{match.get('score', 0)} / 100**

## 2. 评分解释
{score_details(match)}

## 3. 已匹配技能
{bullets(match.get('matched_skills', []))}

## 4. 需要补强的技能/信号
{bullets(match.get('missing_skills', []))}

## 5. 风险点
{bullets(match.get('risk_points', []))}

## 6. 推荐 GitHub 项目路线
仓库名：**{project.get('repo_name', 'CareerPilot-LangGraph')}**

### 核心功能
{bullets(project.get('features', []))}

### 里程碑
{bullets(project.get('milestones', []))}

### 建议学习/贡献仓库
{bullets(project.get('github_projects_to_study', []))}

### 开源贡献推荐
{github_cards(github_recs)}

### RAG 知识库检索结果
{rag_cards(rag_context)}

## 7. 可写入简历的已实现项目描述
基于 LangGraph 构建求职分析 Agent，完成简历解析、JD 分析、技能匹配、本地知识检索、项目规划、简历改写与面试准备的多节点工作流。

- 基于 LangGraph 设计 9 节点状态图，记录节点输出、耗时、最慢节点与完整执行轨迹。
- 接入 DeepSeek/Qwen OpenAI-compatible API，并提供 Offline 确定性降级、空响应重试和结构化 JSON 校验。
- 设计标准化技能覆盖率评分，提供高、中、低匹配评估样例及可解释评分公式。
- 实现本地文本知识库的关键词检索与来源追踪，展示来源文件、Chunk、命中词和关键词相关度。
- 使用 FastAPI 提供 REST API 与 Web Demo，支持 Markdown 报告复制/下载，并通过 pytest 覆盖核心流程。

关键词：LangGraph, DeepSeek, Qwen, FastAPI, Agent, 本地知识检索, 可观测性, pytest

### 尚未实现、不可直接写成已完成
- FAISS/Milvus 向量检索与 Embedding 召回
- Docker 容器化部署
- Function Calling / Tool Calling 真实外部工具执行
- GitHub API / Issue 实时检索
- 分析历史持久化与流式节点进度

## 8. 面试准备
### 高频问题
{bullets(interview.get('questions', []))}

### 讲解要点
{bullets(interview.get('talking_points', []))}

### 三周计划
{bullets(interview.get('study_plan', []))}
"""
    if errors:
        report += "\n## 9. 运行提示\n" + bullets(errors) + "\n"
    return {"final_report": report}


def build_graph(llm: Any | None = None):
    """Build the LangGraph state machine."""

    workflow = StateGraph(CareerState)
    workflow.add_node(
        "extract_profile",
        _instrument_node(
            "extract_profile",
            lambda state: extract_profile_node(state, llm),
        ),
    )
    workflow.add_node(
        "analyze_jd",
        _instrument_node(
            "analyze_jd",
            lambda state: analyze_jd_node(state, llm),
        ),
    )
    workflow.add_node("match", _instrument_node("match", match_node))
    workflow.add_node(
        "rag_retriever",
        _instrument_node("rag_retriever", rag_retriever_node),
    )
    workflow.add_node(
        "project_planner",
        _instrument_node(
            "project_planner",
            lambda state: project_planner_node(state, llm),
        ),
    )
    workflow.add_node(
        "github_recommender",
        _instrument_node("github_recommender", github_recommender_node),
    )
    workflow.add_node(
        "resume_rewriter",
        _instrument_node(
            "resume_rewriter",
            lambda state: resume_rewriter_node(state, llm),
        ),
    )
    workflow.add_node(
        "interview_planner",
        _instrument_node(
            "interview_planner",
            lambda state: interview_planner_node(state, llm),
        ),
    )
    workflow.add_node(
        "final_report",
        _instrument_node("final_report", final_report_node),
    )

    workflow.add_edge(START, "extract_profile")
    workflow.add_edge("extract_profile", "analyze_jd")
    workflow.add_edge("analyze_jd", "match")
    workflow.add_edge("match", "rag_retriever")
    workflow.add_edge("rag_retriever", "project_planner")
    workflow.add_edge("project_planner", "github_recommender")
    workflow.add_edge("github_recommender", "resume_rewriter")
    workflow.add_edge("resume_rewriter", "interview_planner")
    workflow.add_edge("interview_planner", "final_report")
    workflow.add_edge("final_report", END)
    return workflow.compile()
