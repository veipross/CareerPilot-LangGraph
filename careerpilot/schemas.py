"""Shared state and data schemas for CareerPilot-LangGraph."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field


class CareerState(TypedDict, total=False):
    """LangGraph state passed between nodes."""

    resume_text: str
    jd_text: str
    target_role: str
    profile: Dict[str, Any]
    jd_profile: Dict[str, Any]
    match_report: Dict[str, Any]
    project_plan: Dict[str, Any]
    github_recommendations: List[Dict[str, Any]]
    rag_context: List[Dict[str, Any]]
    resume_rewrite: Dict[str, Any]
    interview_plan: Dict[str, Any]
    final_report: str
    execution_trace: List[Dict[str, Any]]
    pipeline_metrics: Dict[str, Any]
    errors: List[str]


class ExecutionTraceItem(BaseModel):
    """One completed LangGraph node execution record."""

    index: int = 0
    node: str
    label: str
    status: str = "completed"
    duration_ms: float = 0.0
    summary: str = ""
    output_keys: List[str] = Field(default_factory=list)


class ResumeProfile(BaseModel):
    education: str = ""
    target_roles: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    missing_signals: List[str] = Field(default_factory=list)


class JDProfile(BaseModel):
    role: str = ""
    core_requirements: List[str] = Field(default_factory=list)
    preferred_requirements: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)


class MatchReport(BaseModel):
    score: float = 0.0
    level: str = "未评估"
    scoring_method: str = "canonical_skill_coverage"
    score_formula: str = ""
    matched_count: int = 0
    required_count: int = 0
    resume_skill_count: int = 0
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    score_explanation: List[str] = Field(default_factory=list)
    risk_points: List[str] = Field(default_factory=list)
    positioning: str = ""


class ProjectPlan(BaseModel):
    repo_name: str = "CareerPilot-LangGraph"
    github_projects_to_study: List[str] = Field(default_factory=list)
    features: List[str] = Field(default_factory=list)
    milestones: List[str] = Field(default_factory=list)
    resume_bullets: List[str] = Field(default_factory=list)




class RAGContext(BaseModel):
    source: str
    source_name: str = ""
    rank: int = 0
    chunk_index: int = 0
    score: float = 0.0
    content: str
    preview: str = ""
    matched_terms: List[str] = Field(default_factory=list)
    retrieval_reason: str = ""
    retrieval_mode: str = "keyword"
    embedding_model: str = ""
    vector_score: float = 0.0
    keyword_score: float = 0.0
    index_rebuilt: bool = False
    fallback_reason: str = ""


class OpenSourceRecommendation(BaseModel):
    repo: str
    reason: str
    fit_score: int = 0
    contribution_ideas: List[str] = Field(default_factory=list)
    skills_to_learn: List[str] = Field(default_factory=list)


class ResumeRewrite(BaseModel):
    summary: str = ""
    rewritten_project_bullets: List[str] = Field(default_factory=list)
    keywords_to_add: List[str] = Field(default_factory=list)


class InterviewPlan(BaseModel):
    questions: List[str] = Field(default_factory=list)
    talking_points: List[str] = Field(default_factory=list)
    study_plan: List[str] = Field(default_factory=list)
