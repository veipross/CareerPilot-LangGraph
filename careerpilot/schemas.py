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
    resume_rewrite: Dict[str, Any]
    interview_plan: Dict[str, Any]
    final_report: str
    errors: List[str]


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
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    risk_points: List[str] = Field(default_factory=list)
    positioning: str = ""


class ProjectPlan(BaseModel):
    repo_name: str = "CareerPilot-LangGraph"
    github_projects_to_study: List[str] = Field(default_factory=list)
    features: List[str] = Field(default_factory=list)
    milestones: List[str] = Field(default_factory=list)
    resume_bullets: List[str] = Field(default_factory=list)


class ResumeRewrite(BaseModel):
    summary: str = ""
    rewritten_project_bullets: List[str] = Field(default_factory=list)
    keywords_to_add: List[str] = Field(default_factory=list)


class InterviewPlan(BaseModel):
    questions: List[str] = Field(default_factory=list)
    talking_points: List[str] = Field(default_factory=list)
    study_plan: List[str] = Field(default_factory=list)
