"""Command line interface."""
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from .config import get_settings
from .graph import build_graph
from .llm import build_qwen_llm


def read_text(path: str) -> str:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if file.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(file))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="CareerPilot-LangGraph: resume/JD matching agent")
    parser.add_argument("--resume", required=True, help="Path to resume txt/pdf")
    parser.add_argument("--jd", required=True, help="Path to job description txt")
    parser.add_argument("--out", default="outputs/report.md", help="Output markdown report")
    parser.add_argument("--target-role", default="大模型/Agent 工程实习生")
    parser.add_argument("--offline", action="store_true", help="Run without LLM API")
    parser.add_argument("--model", default=None, help="Override Qwen model name, e.g. qwen-plus")
    args = parser.parse_args()

    load_dotenv()
    settings = get_settings()
    if args.model:
        settings = type(settings)(model=args.model, dashscope_base_url=settings.dashscope_base_url, dashscope_api_key=settings.dashscope_api_key, temperature=settings.temperature)

    llm = None if args.offline else build_qwen_llm(settings)
    graph = build_graph(llm=llm)
    result = graph.invoke(
        {
            "resume_text": read_text(args.resume),
            "jd_text": read_text(args.jd),
            "target_role": args.target_role,
        }
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result["final_report"], encoding="utf-8")
    print(f"Report generated: {out}")


if __name__ == "__main__":
    main()
