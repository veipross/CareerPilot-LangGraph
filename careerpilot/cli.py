"""Command line interface."""

from __future__ import annotations

import argparse

from .service import DEFAULT_TARGET_ROLE, run_careerpilot_from_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CareerPilot-LangGraph: resume/JD matching agent"
    )
    parser.add_argument("--resume", required=True, help="Path to resume txt/pdf")
    parser.add_argument("--jd", required=True, help="Path to job description txt/pdf")
    parser.add_argument("--out", default="outputs/report.md", help="Output markdown report")
    parser.add_argument("--target-role", default=DEFAULT_TARGET_ROLE)
    parser.add_argument("--offline", action="store_true", help="Run without LLM API")
    parser.add_argument("--model", default=None, help="Override model name, e.g. qwen-plus")

    args = parser.parse_args()

    run_careerpilot_from_files(
        resume_path=args.resume,
        jd_path=args.jd,
        out_path=args.out,
        target_role=args.target_role,
        offline=args.offline,
        model=args.model,
    )

    print(f"Report generated: {args.out}")


if __name__ == "__main__":
    main()