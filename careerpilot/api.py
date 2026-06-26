"""FastAPI application and Web Demo for CareerPilot-LangGraph."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .service import DEFAULT_TARGET_ROLE, run_careerpilot


# 当前 Python 包所在目录：
# CareerPilot-LangGraph/careerpilot/
PACKAGE_DIR = Path(__file__).resolve().parent

# 页面模板和静态资源目录
TEMPLATE_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"


app = FastAPI(
    title="CareerPilot-LangGraph API",
    description=(
        "A LangGraph-based career preparation agent service "
        "with DeepSeek/Qwen online providers and offline fallback."
    ),
    version="0.3.0",
)

# check_dir=False 可以避免在静态资源目录尚未创建时，
# FastAPI 导入阶段直接报错。
app.mount(
    "/static",
    StaticFiles(
        directory=str(STATIC_DIR),
        check_dir=False,
    ),
    name="static",
)

templates = Jinja2Templates(
    directory=str(TEMPLATE_DIR),
)


class CareerRequest(BaseModel):
    """JSON request model for the CareerPilot API."""

    resume_text: str = Field(
        min_length=1,
        description="Raw resume text.",
    )
    jd_text: str = Field(
        min_length=1,
        description="Raw job description text.",
    )
    target_role: str = Field(
        default=DEFAULT_TARGET_ROLE,
        description="Target position expected by the candidate.",
    )
    offline: bool = Field(
        default=False,
        description="Whether to skip external LLM API calls.",
    )
    provider: Literal["deepseek", "qwen"] = Field(
        default="deepseek",
        description="Online LLM provider.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Optional model-name override.",
    )


class CareerResponse(BaseModel):
    """JSON response model returned by CareerPilot."""

    final_report: str
    match_score: Optional[float] = None
    match_level: Optional[str] = None
    match_breakdown: dict = Field(default_factory=dict)
    execution_trace: list[dict] = Field(default_factory=list)
    pipeline_metrics: dict = Field(default_factory=dict)
    rag_context: list[dict] = Field(default_factory=list)
    provider: str
    offline: bool


def _run_analysis(
    *,
    resume_text: str,
    jd_text: str,
    target_role: str,
    offline: bool,
    provider: str,
    model: str | None = None,
) -> CareerResponse:
    """Run CareerPilot and convert the LangGraph state to an API response."""

    result = run_careerpilot(
        resume_text=resume_text,
        jd_text=jd_text,
        target_role=target_role,
        offline=offline,
        provider=None if offline else provider,
        model=model or None,
    )

    match_report = result.get("match_report") or {}

    return CareerResponse(
        final_report=result.get("final_report", ""),
        match_score=match_report.get("score"),
        match_level=match_report.get("level"),
        match_breakdown={
            "matched_count": match_report.get("matched_count", 0),
            "required_count": match_report.get("required_count", 0),
            "resume_skill_count": match_report.get("resume_skill_count", 0),
            "score_formula": match_report.get("score_formula", ""),
            "scoring_method": match_report.get("scoring_method", ""),
        },
        execution_trace=result.get("execution_trace") or [],
        pipeline_metrics=result.get("pipeline_metrics") or {},
        rag_context=result.get("rag_context") or [],
        provider="offline" if offline else provider,
        offline=offline,
    )


def _build_template_context(
    *,
    target_role: str = DEFAULT_TARGET_ROLE,
    provider: str = "deepseek",
    offline: bool = False,
    resume_text: str = "",
    jd_text: str = "",
    model: str = "",
    report: str = "",
    match_score: float | None = None,
    match_level: str = "",
    match_breakdown: dict | None = None,
    execution_trace: list[dict] | None = None,
    pipeline_metrics: dict | None = None,
    rag_context: list[dict] | None = None,
    error: str = "",
) -> dict:
    """Build the shared context passed to the Jinja2 page template."""

    return {
        "target_role": target_role,
        "provider": provider,
        "offline": offline,
        "resume_text": resume_text,
        "jd_text": jd_text,
        "model": model,
        "report": report,
        "match_score": match_score,
        "match_level": match_level,
        "match_breakdown": match_breakdown or {},
        "execution_trace": execution_trace or [],
        "pipeline_metrics": pipeline_metrics or {},
        "rag_context": rag_context or [],
        "error": error,
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Service health check."""

    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Render the CareerPilot Web Demo homepage."""

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=_build_template_context(),
    )


@app.get(
    "/web/analyze",
    include_in_schema=False,
)
def web_analyze_get() -> RedirectResponse:
    """Redirect browser refreshes back to the CareerPilot homepage."""

    return RedirectResponse(
        url="/",
        status_code=303,
    )


@app.post(
    "/analyze",
    response_model=CareerResponse,
)
def analyze(req: CareerRequest) -> CareerResponse:
    """Run CareerPilot through the JSON API."""

    try:
        return _run_analysis(
            resume_text=req.resume_text,
            jd_text=req.jd_text,
            target_role=req.target_role,
            offline=req.offline,
            provider=req.provider,
            model=req.model,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"CareerPilot analysis failed: {exc}",
        ) from exc


@app.post(
    "/web/analyze",
    response_class=HTMLResponse,
)
def web_analyze(
    request: Request,
    resume_text: str = Form(...),
    jd_text: str = Form(...),
    target_role: str = Form(DEFAULT_TARGET_ROLE),
    provider: str = Form("deepseek"),
    model: str = Form(""),
    offline: Optional[str] = Form(None),
):
    """Run CareerPilot from the browser form and render the report."""

    offline_enabled = offline == "on"

    context = _build_template_context(
        target_role=target_role,
        provider=provider,
        offline=offline_enabled,
        resume_text=resume_text,
        jd_text=jd_text,
        model=model,
    )

    if provider not in {"deepseek", "qwen"}:
        context["error"] = (
            f"Unsupported provider: {provider}. "
            "Supported providers are deepseek and qwen."
        )
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context=context,
            status_code=400,
        )

    if not resume_text.strip():
        context["error"] = "简历内容不能为空。"
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context=context,
            status_code=400,
        )

    if not jd_text.strip():
        context["error"] = "岗位 JD 不能为空。"
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context=context,
            status_code=400,
        )

    try:
        response = _run_analysis(
            resume_text=resume_text.strip(),
            jd_text=jd_text.strip(),
            target_role=target_role.strip() or DEFAULT_TARGET_ROLE,
            offline=offline_enabled,
            provider=provider,
            model=model.strip() or None,
        )

        context["report"] = response.final_report
        context["match_score"] = response.match_score
        context["match_level"] = response.match_level or ""
        context["match_breakdown"] = response.match_breakdown
        context["execution_trace"] = response.execution_trace
        context["pipeline_metrics"] = response.pipeline_metrics
        context["rag_context"] = response.rag_context

    except Exception as exc:
        context["error"] = str(exc)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=context,
    )
