from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from careerpilot.config import get_rag_settings
from careerpilot.rag_evaluation import (
    evaluate_case_result,
    save_report,
    summarize_mode,
)
from careerpilot.tools import retrieve_knowledge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare keyword, vector, and hybrid RAG retrieval quality.",
    )
    parser.add_argument(
        "--cases",
        default="data/evaluation/rag_queries.json",
        help="Evaluation case JSON path relative to project root.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/rag_evaluation",
        help="Directory for JSON and Markdown reports.",
    )
    parser.add_argument(
        "--modes",
        default="keyword,vector,hybrid",
        help="Comma-separated retrieval modes.",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat every case and average its latency.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Do not pre-load the embedding model and FAISS index.",
    )
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_retrieval(
    case: Dict[str, Any],
    mode: str,
    top_k: int,
    repeat: int,
    settings: Any,
) -> Dict[str, Any]:
    latencies: List[float] = []
    final_results: List[Dict[str, Any]] = []

    for _ in range(max(1, repeat)):
        started = perf_counter()
        final_results = retrieve_knowledge(
            query_terms=list(case.get("query_terms", [])),
            query_text=str(case.get("query", "")),
            knowledge_dir=settings.knowledge_dir,
            index_dir=settings.index_dir,
            embedding_model=settings.embedding_model,
            top_k=top_k,
            mode=mode,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            vector_weight=settings.vector_weight,
            device=settings.device,
        )
        latencies.append((perf_counter() - started) * 1000.0)

    latency_ms = sum(latencies) / len(latencies)
    return evaluate_case_result(case, final_results, latency_ms)


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    settings = get_rag_settings()

    case_path = resolve_project_path(args.cases)
    cases = json.loads(case_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise SystemExit("评估集为空或格式错误。")

    allowed_modes = {"keyword", "vector", "hybrid"}
    modes = [part.strip().lower() for part in args.modes.split(",") if part.strip()]
    invalid = [mode for mode in modes if mode not in allowed_modes]
    if invalid:
        raise SystemExit(f"不支持的检索模式：{', '.join(invalid)}")

    warmup_ms = 0.0
    if not args.skip_warmup and any(mode in {"vector", "hybrid"} for mode in modes):
        warm_case = cases[0]
        started = perf_counter()
        warm_results = retrieve_knowledge(
            query_terms=list(warm_case.get("query_terms", [])),
            query_text=str(warm_case.get("query", "")),
            knowledge_dir=settings.knowledge_dir,
            index_dir=settings.index_dir,
            embedding_model=settings.embedding_model,
            top_k=args.top_k,
            mode="vector",
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            vector_weight=settings.vector_weight,
            device=settings.device,
        )
        warmup_ms = (perf_counter() - started) * 1000.0
        if not warm_results or warm_results[0].get("retrieval_mode") == "keyword_fallback":
            reason = warm_results[0].get("fallback_reason", "未知原因") if warm_results else "无召回结果"
            raise SystemExit(f"向量后端预热失败：{reason}")

    mode_reports = []
    for mode in modes:
        rows = []
        print(f"\n[{mode}] evaluating {len(cases)} cases...")
        for index, case in enumerate(cases, start=1):
            row = run_retrieval(
                case=case,
                mode=mode,
                top_k=max(1, args.top_k),
                repeat=max(1, args.repeat),
                settings=settings,
            )
            rows.append(row)
            rank_text = row["relevant_rank"] if row["relevant_rank"] is not None else "miss"
            print(
                f"  {index:02d}/{len(cases):02d} {row['id']:<28} "
                f"rank={rank_text!s:<4} latency={row['latency_ms']:.2f} ms"
            )
        mode_reports.append(summarize_mode(mode, rows))

    report = {
        "schema_version": 1,
        "case_file": str(case_path),
        "case_count": len(cases),
        "top_k": max(1, args.top_k),
        "repeat": max(1, args.repeat),
        "embedding_model": settings.embedding_model,
        "knowledge_dir": settings.knowledge_dir,
        "index_dir": settings.index_dir,
        "vector_weight": settings.vector_weight,
        "warmup_ms": round(warmup_ms, 3),
        "results": mode_reports,
    }

    paths = save_report(report, resolve_project_path(args.output_dir))

    print("\n" + "=" * 92)
    print(f"{'Mode':<10} {'Hit@1':>8} {'Hit@3':>8} {'MRR':>8} {'Mean ms':>12} {'P95 ms':>12}")
    print("-" * 92)
    for mode_report in mode_reports:
        metrics = mode_report["metrics"]
        print(
            f"{mode_report['mode']:<10} "
            f"{metrics['hit_at_1']:>8.3f} "
            f"{metrics['hit_at_3']:>8.3f} "
            f"{metrics['mrr']:>8.3f} "
            f"{metrics['mean_latency_ms']:>12.2f} "
            f"{metrics['p95_latency_ms']:>12.2f}"
        )
    print("=" * 92)
    print(f"JSON report    : {paths['json']}")
    print(f"Markdown report: {paths['markdown']}")


if __name__ == "__main__":
    main()
