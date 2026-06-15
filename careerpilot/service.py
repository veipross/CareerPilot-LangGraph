"""Service layer for running CareerPilot workflows.

This module keeps the LangGraph execution logic independent from CLI, API,
future MCP tools, and UI adapters.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from .config import Settings, get_settings
from .graph import build_graph
from .llm import build_qwen_llm
from .schemas import CareerState


DEFAULT_TARGET_ROLE = "大模型/Agent 工程实习生"


def read_text_file(path: str | Path) -> str:
    """Read a txt/markdown/pdf file as text."""
    file = Path(path)

    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")

    if file.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(file))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    return file.read_text(encoding="utf-8")


def build_initial_state(
    resume_text: str,
    jd_text: str,
    target_role: str = DEFAULT_TARGET_ROLE,
) -> CareerState:
    """Build the initial LangGraph state."""
    return {
        "resume_text": resume_text,
        "jd_text": jd_text,
        "target_role": target_role,
    }


def run_careerpilot(
    resume_text: str,
    jd_text: str,
    target_role: str = DEFAULT_TARGET_ROLE,
    offline: bool = False,
    model: str | None = None,
    settings: Settings | None = None,
) -> CareerState:
    """Run the CareerPilot LangGraph workflow.

    Args:
        resume_text: Raw resume text.
        jd_text: Raw job description text.
        target_role: Target role displayed in the report.
        offline: If true, skip LLM calls and use deterministic fallback logic.
        model: Optional model name override, e.g. qwen-plus.
        settings: Optional runtime settings, mainly useful for tests.

    Returns:
        Final CareerState containing final_report and all intermediate fields.
    """
    load_dotenv()

    runtime_settings = settings or get_settings()
    if model:
        runtime_settings = replace(runtime_settings, model=model)

    llm = None if offline else build_qwen_llm(runtime_settings)
    graph = build_graph(llm=llm)

    return graph.invoke(
        build_initial_state(
            resume_text=resume_text,
            jd_text=jd_text,
            target_role=target_role,
        )
    )


def run_careerpilot_from_files(
    resume_path: str | Path,
    jd_path: str | Path,
    out_path: str | Path | None = None,
    target_role: str = DEFAULT_TARGET_ROLE,
    offline: bool = False,
    model: str | None = None,
) -> CareerState:
    """Run CareerPilot from local files and optionally write the markdown report."""
    result = run_careerpilot(
        resume_text=read_text_file(resume_path),
        jd_text=read_text_file(jd_path),
        target_role=target_role,
        offline=offline,
        model=model,
    )

    if out_path:
        output = Path(out_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.get("final_report", ""), encoding="utf-8")

    return result