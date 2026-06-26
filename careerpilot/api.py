"""FastAPI application and Web Demo for CareerPilot-LangGraph."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .resume_parser import (
    MAX_RESUME_FILE_BYTES,
    ResumeParseError,
    parse_resume_upload,
)
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
    version="0.4.0",
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


class ResumeExtractResponse(BaseModel):
    """Metadata and text returned by the resume upload endpoint."""

    text: str
    filename: str
    file_type: str
    page_count: int
    char_count: int


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
    model: Optional[str] = None,
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
    match_score: Optional[float] = None,
    match_level: str = "",
    match_breakdown: Optional[dict] = None,
    execution_trace: Optional[list[dict]] = None,
    pipeline_metrics: Optional[dict] = None,
    rag_context: Optional[list[dict]] = None,
    resume_source: str = "text",
    resume_filename: str = "",
    resume_file_type: str = "",
    resume_page_count: int = 0,
    resume_char_count: int = 0,
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
        "resume_source": resume_source,
        "resume_filename": resume_filename,
        "resume_file_type": resume_file_type,
        "resume_page_count": resume_page_count,
        "resume_char_count": resume_char_count,
        "max_resume_file_mb": MAX_RESUME_FILE_BYTES // (1024 * 1024),
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
    "/resume/extract",
    response_model=ResumeExtractResponse,
)
def extract_resume_file(
    resume_file: UploadFile = File(...),
) -> ResumeExtractResponse:
    """Extract resume text from an uploaded PDF/TXT/Markdown file."""

    try:
        data = resume_file.file.read(MAX_RESUME_FILE_BYTES + 1)
        parsed = parse_resume_upload(
            filename=resume_file.filename or "resume",
            content_type=resume_file.content_type,
            data=data,
        )
    except ResumeParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        resume_file.file.close()

    return ResumeExtractResponse(
        text=parsed.text,
        filename=parsed.filename,
        file_type=parsed.file_type,
        page_count=parsed.page_count,
        char_count=parsed.char_count,
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
    resume_text: str = Form(""),
    resume_file: Optional[UploadFile] = File(None),
    jd_text: str = Form(...),
    target_role: str = Form(DEFAULT_TARGET_ROLE),
    provider: str = Form("deepseek"),
    model: str = Form(""),
    offline: Optional[str] = Form(None),
):
    """Run CareerPilot from the browser form and render the report."""

    offline_enabled = offline == "on"
    resolved_resume_text = resume_text.strip()
    upload_metadata = {
        "resume_source": "text",
        "resume_filename": "",
        "resume_file_type": "",
        "resume_page_count": 0,
        "resume_char_count": len(resolved_resume_text),
    }

    if resume_file is not None and resume_file.filename:
        try:
            uploaded_data = resume_file.file.read(MAX_RESUME_FILE_BYTES + 1)
            parsed = parse_resume_upload(
                filename=resume_file.filename,
                content_type=resume_file.content_type,
                data=uploaded_data,
            )
            resolved_resume_text = parsed.text
            upload_metadata = {
                "resume_source": "upload",
                "resume_filename": parsed.filename,
                "resume_file_type": parsed.file_type,
                "resume_page_count": parsed.page_count,
                "resume_char_count": parsed.char_count,
            }
        except ResumeParseError as exc:
            context = _build_template_context(
                target_role=target_role,
                provider=provider,
                offline=offline_enabled,
                resume_text=resume_text,
                jd_text=jd_text,
                model=model,
                error=str(exc),
            )
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context=context,
                status_code=400,
            )
        finally:
            resume_file.file.close()

    context = _build_template_context(
        target_role=target_role,
        provider=provider,
        offline=offline_enabled,
        resume_text=resolved_resume_text,
        jd_text=jd_text,
        model=model,
        **upload_metadata,
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

    if not resolved_resume_text:
        context["error"] = "请上传简历文件，或在文本框中粘贴简历内容。"
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
            resume_text=resolved_resume_text,
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
