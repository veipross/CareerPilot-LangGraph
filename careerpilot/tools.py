"""Deterministic tools used by the graph.

These tools make the demo reproducible and give the Agent something concrete to call.
"""
from __future__ import annotations

import re
from pathlib import Path
from collections import Counter
from typing import Dict, Iterable, List, Tuple

CANONICAL_SKILLS = [
    "Python",
    "C++",
    "LLM",
    "PyTorch",
    "NumPy",
    "Pandas",
    "Matplotlib",
    "scikit-learn",
    "Hugging Face",
    "CUDA",
    "TensorRT",
    "vLLM",
    "DeepSpeed",
    "TensorRT-LLM",
    "Transformer",
    "Attention",
    "LoRA",
    "P-Tuning",
    "RAG",
    "LangChain",
    "LangGraph",
    "Agent",
    "Function Calling",
    "Tool Calling",
    "MCP",
    "Prompt Engineering",
    "DashScope",
    "Qwen",
    "OpenAI API",
    "FAISS",
    "Milvus",
    "Vector DB",
    "Redis",
    "SQL",
    "Git",
    "REST API",
    "FastAPI",
    "Docker",
    "Linux",
    "NCCL",
    "量化",
    "剪枝",
    "多模态",
    "计算机视觉",
    "NLP",
]

SKILL_ALIASES = {
    "通义千问": "Qwen",
    "千问": "Qwen",
    "百炼": "DashScope",
    "大语言模型": "LLM",
    "大模型": "LLM",
    "工具调用": "Tool Calling",
    "函数调用": "Function Calling",
    "检索增强": "RAG",
    "检索增强生成": "RAG",
    "向量检索": "RAG",
    "向量数据库": "Vector DB",
    "智能体": "Agent",
    "模型量化": "量化",
    "结构化剪枝": "剪枝",
}

PROJECT_TEMPLATES = [
    {
        "name": "CareerPilot-LangGraph",
        "why": "直接服务秋招场景，可展示 LangGraph 状态图、工具调用、结构化输出与简历/JD 匹配。",
    },
    {
        "name": "Qwen-Agent-RAG-Evaluator",
        "why": "把 Qwen/DashScope、RAG、评测指标结合，适合投大模型应用和 Agent 岗位。",
    },
    {
        "name": "LLM-Serving-Benchmark-vLLM",
        "why": "承接你已有推理优化背景，用 TTFT、tokens/s、显存占用证明工程能力。",
    },
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _contains_term(text: str, term: str) -> bool:
    """Return whether *term* occurs as a standalone technical term.

    ASCII skill names use token boundaries so short names such as ``RAG`` do
    not accidentally match unrelated words such as ``storage``. Chinese terms
    keep substring matching because they are not whitespace-delimited.
    """

    if not text or not term:
        return False

    if re.fullmatch(r"[A-Za-z0-9+_. /-]+", term):
        pattern = rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None

    return term in text


def extract_skills(text: str) -> List[str]:
    """Extract normalized skills for deterministic scoring and fallback."""

    text_norm = normalize_text(text)
    found = []

    for skill in CANONICAL_SKILLS:
        if _contains_term(text_norm, skill):
            found.append(skill)

    for alias, canonical in SKILL_ALIASES.items():
        if _contains_term(text_norm, alias):
            found.append(canonical)

    # Preserve order while deduplicating.
    return list(dict.fromkeys(found))


def canonicalize_skill_items(items: Iterable[str]) -> List[str]:
    """Convert free-form LLM skill strings to the deterministic vocabulary."""

    normalized: List[str] = []
    for item in items or []:
        if not isinstance(item, str):
            continue
        normalized.extend(extract_skills(item))
    return list(dict.fromkeys(normalized))


def extract_project_lines(text: str) -> List[str]:
    """Extract lines likely to describe projects."""

    lines = [line.strip(" -·\t") for line in (text or "").splitlines() if line.strip()]
    project_lines = [
        line
        for line in lines
        if any(key in line for key in ["项目", "系统", "框架", "Agent", "RAG", "推理", "增强"])
    ]
    return project_lines[:8]


def top_keywords(text: str, n: int = 12) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+_.-]{1,}|[\u4e00-\u9fff]{2,}", text or "")
    stop = {"要求", "负责", "相关", "能力", "熟悉", "具备", "优先", "岗位", "项目", "开发", "经验"}
    counts = Counter(token for token in tokens if token not in stop)
    return [word for word, _ in counts.most_common(n)]


def compute_match(resume_skills: Iterable[str], jd_skills: Iterable[str]) -> Tuple[float, List[str], List[str]]:
    """Compute exact normalized skill coverage.

    The score intentionally remains a transparent coverage ratio instead of
    an LLM-generated probability: matched JD skills / recognized JD skills.
    """

    resume_set = {s.lower(): s for s in resume_skills}
    jd_unique = list(dict.fromkeys(jd_skills))
    matched = []
    missing = []
    for skill in jd_unique:
        if skill.lower() in resume_set:
            matched.append(resume_set[skill.lower()])
        else:
            missing.append(skill)
    if not jd_unique:
        return 0.0, matched, missing
    score = round(len(matched) / len(jd_unique) * 100, 1)
    return score, matched, missing


def classify_match_level(score: float) -> str:
    """Map a coverage score to a human-readable evaluation band."""

    if score >= 80:
        return "高匹配"
    if score >= 50:
        return "中匹配"
    return "低匹配"


def recommend_project_features(missing_skills: List[str], matched_skills: List[str]) -> List[str]:
    features = [
        "简历 PDF/文本解析：抽取教育背景、项目经历、技能关键词",
        "JD 结构化分析：抽取核心要求、加分项、技术栈和岗位关键词",
        "LangGraph 状态图编排：ProfileExtractor → JDAnalyzer → Matcher → ProjectPlanner → ResumeRewriter",
        "结构化输出：将匹配分、缺口技能、项目建议输出为 JSON + Markdown 报告",
    ]
    if any(skill in missing_skills for skill in ["RAG", "FAISS", "Milvus", "Vector DB"]):
        features.append("RAG 模块：把公司 JD、面经、开源 issue 建成知识库，支持岗位定制检索")
    if any(skill in missing_skills for skill in ["Tool Calling", "Function Calling", "MCP", "Agent"]):
        features.append("工具调用模块：接入 GitHub issue 查询、项目 README 分析、面试题生成工具")
    if any(skill in matched_skills for skill in ["CUDA", "TensorRT", "vLLM", "量化"]):
        features.append("推理性能扩展：记录模型调用 latency、token 用量，并预留 vLLM serving 接口")
    return features


OPEN_SOURCE_PROJECTS = [
    {
        "repo": "langchain-ai/langgraph",
        "tags": ["LangGraph", "Agent", "Tool Calling", "状态图", "结构化输出"],
        "reason": "Agent 状态图编排核心仓库，最适合补强你报告中的 LangGraph/Agent 信号。",
        "ideas": [
            "复现官方 examples，并补充中文 README/注释版学习笔记",
            "为 CareerPilot 增加一个 LangGraph checkpoint 或 streaming 示例",
            "从 good first issue / docs issue 开始，提交测试或文档 PR",
        ],
    },
    {
        "repo": "langchain-ai/langchain",
        "tags": ["LangChain", "Tool Calling", "Function Calling", "OpenAI API", "Agent"],
        "reason": "大模型应用岗位认可度高，适合学习工具调用、结构化输出和模型接口封装。",
        "ideas": [
            "补充一个 Qwen/DashScope OpenAI-compatible 调用示例",
            "给某个 Tool/Parser 增加最小可复现测试",
            "整理 LangChain 与 LangGraph 的差异对比文档",
        ],
    },
    {
        "repo": "QwenLM/Qwen-Agent",
        "tags": ["Qwen", "DashScope", "Agent", "Function Calling", "Tool Calling"],
        "reason": "与你已有 DashScope/通义千问项目经历最连贯，适合做中文 Agent 应用展示。",
        "ideas": [
            "把 CareerPilot 的简历/JD 分析封装成 Qwen-Agent 工具",
            "实现一个中文求职助手 demo，支持多轮追问和工具调用",
            "补充 DashScope 配置说明、异常处理或示例脚本",
        ],
    },
    {
        "repo": "run-llama/llama_index",
        "tags": ["RAG", "向量检索", "OpenAI API", "Agent", "文档解析"],
        "reason": "用于补强 RAG/向量检索短板，适合把 JD、面经、GitHub issue 做成知识库。",
        "ideas": [
            "为 CareerPilot 增加 JD/面经 RAG 检索模块",
            "对比 naive retrieval、rerank、metadata filter 的效果",
            "补一个中文 PDF 简历解析到索引的示例",
        ],
    },
    {
        "repo": "vllm-project/vllm",
        "tags": ["vLLM", "推理优化", "Qwen", "性能评测", "Serving"],
        "reason": "承接你的 TensorRT/CUDA/量化背景，能形成区别于普通 Agent 项目的性能亮点。",
        "ideas": [
            "写 Qwen 模型在不同并发下的 TTFT/tokens/s benchmark 脚本",
            "把 CareerPilot 的 LLM 调用切换到本地 vLLM OpenAI-compatible 服务",
            "记录显存占用、吞吐和 P95 latency，形成实验报告",
        ],
    },
]


def recommend_open_source_projects(
    missing_skills: List[str],
    matched_skills: List[str],
    target_role: str = "大模型/Agent 工程实习生",
    top_k: int = 4,
) -> List[Dict[str, object]]:
    """Recommend open-source repos and concrete contribution ideas."""

    missing = set(missing_skills or [])
    matched = set(matched_skills or [])
    role_text = target_role or ""
    results = []

    for project in OPEN_SOURCE_PROJECTS:
        tags = set(project["tags"])
        score = 40
        score += 12 * len(tags & missing)
        score += 6 * len(tags & matched)

        if "Agent" in role_text and "Agent" in tags:
            score += 10
        if "大模型" in role_text and ({"Qwen", "OpenAI API", "vLLM"} & tags):
            score += 8

        score = min(score, 100)

        results.append(
            {
                "repo": project["repo"],
                "reason": project["reason"],
                "fit_score": score,
                "contribution_ideas": project["ideas"],
                "skills_to_learn": [
                    skill for skill in project["tags"]
                    if skill in missing or skill not in matched
                ][:5],
            }
        )

    results.sort(key=lambda item: item["fit_score"], reverse=True)
    return results[:top_k]



def split_text_into_chunks(text: str, chunk_size: int = 420, overlap: int = 60) -> List[str]:
    """Split text into small chunks for lightweight local retrieval."""

    text = normalize_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)

    return chunks


def retrieve_knowledge(
    query_terms: List[str],
    knowledge_dir: str = "data/knowledge",
    top_k: int = 4,
) -> List[Dict[str, object]]:
    """A lightweight keyword-overlap retriever.

    This is intentionally deterministic so the project can run without an
    embedding model or vector database. It can be replaced by FAISS/LlamaIndex later.
    """

    root = Path(knowledge_dir)
    if not root.exists():
        return []

    terms = [term for term in dict.fromkeys(query_terms or []) if term]
    lower_terms = [term.lower() for term in terms]

    candidates = []
    for path in sorted(root.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for chunk_index, chunk in enumerate(split_text_into_chunks(text), start=1):
            chunk_lower = chunk.lower()
            matched = [
                terms[i]
                for i, term in enumerate(lower_terms)
                if term and (term in chunk_lower or terms[i] in chunk)
            ]

            if not matched:
                continue

            matched = list(dict.fromkeys(matched))
            # This is a transparent keyword relevance score, not a vector
            # similarity score. Normalize by the query size so several chunks
            # do not all saturate at 100 merely because the query is long.
            denominator = max(1, min(len(terms), 12))
            coverage = len(matched) / denominator
            score = min(99, round(25 + 74 * coverage))
            preview = chunk if len(chunk) <= 240 else chunk[:240].rstrip() + "..."
            candidates.append(
                {
                    "source": str(path),
                    "source_name": path.name,
                    "chunk_index": chunk_index,
                    "score": score,
                    "content": chunk,
                    "preview": preview,
                    "matched_terms": matched,
                    "retrieval_reason": (
                        f"命中 {len(matched)} 个查询词：{', '.join(matched)}"
                    ),
                }
            )

    candidates.sort(
        key=lambda item: (
            item["score"],
            len(item["matched_terms"]),
            item["source_name"],
            -item["chunk_index"],
        ),
        reverse=True,
    )

    selected = candidates[:top_k]
    for rank, item in enumerate(selected, start=1):
        item["rank"] = rank

    return selected
