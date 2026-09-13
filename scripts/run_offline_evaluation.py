"""Compare retrieval thresholds on an independent, offline catalog holdout set.

The 18-question acceptance set is intentionally not imported here: it remains a
regression check, while this script uses data/holdout_evaluation_questions.json
to select a temporary MIN_RELEVANCE value.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sanding_rag.chroma_store import ChromaStore  # noqa: E402
from sanding_rag.config import Settings  # noqa: E402
from sanding_rag.embedding import HashingTestEmbedder, SentenceTransformerEmbedder  # noqa: E402
from sanding_rag.ingestion import MarkdownIngestionPipeline  # noqa: E402
from sanding_rag.llm import ContextEchoLLM  # noqa: E402
from sanding_rag.runtime import make_retriever  # noqa: E402
from sanding_rag.service import RAGAnswerService  # noqa: E402
from sanding_rag.splitter import SemanticRecursiveSplitter  # noqa: E402


def _arguments() -> object:
    parser = ArgumentParser(description="Run independent offline catalog retrieval evaluation")
    parser.add_argument(
        "--thresholds",
        help="comma-separated candidate MIN_RELEVANCE values; default comes from EVALUATION_THRESHOLDS",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=("dense", "bm25", "hybrid_rrf"),
        default="dense",
        help="retrieval mode to evaluate; defaults to the production Dense baseline",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "EVALUATION_TRACES.json",
        help="JSON trace artifact path",
    )
    parser.add_argument(
        "--hashing-test-embedder",
        action="store_true",
        help="use deterministic test vectors only; semantic Embedding is the default for evaluation",
    )
    return parser.parse_args()


def _parse_thresholds(raw: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if len(values) < 3:
        raise ValueError("at least three candidate thresholds are required")
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("thresholds must be between 0 and 1")
    return values


def _top_k_trace(matches: list[object]) -> list[dict[str, Any]]:
    return [
        {
            "source": str(match.metadata["source"]),
            "source_url": str(match.metadata["source_url"]),
            "product_id": str(match.metadata["product_id"]),
            "score": round(float(match.score), 4),
        }
        for match in matches
    ]


def _metrics(traces: list[dict[str, Any]]) -> dict[str, Any]:
    answer_traces = [trace for trace in traces if trace["expected"]["result"] == "answer"]
    handoff_traces = [trace for trace in traces if trace["expected"]["result"] == "handoff"]

    hit_at_1 = sum(bool(trace["source_hit_at_1"]) for trace in answer_traces)
    hit_at_3 = sum(bool(trace["source_hit_at_3"]) for trace in answer_traces)
    wrong_answers = sum(not bool(trace["actual"]["answer_is_correct"]) for trace in traces)
    correct_handoffs = sum(
        trace["actual"]["result"] == "handoff" for trace in handoff_traces
    )
    incorrect_handoffs = sum(
        trace["actual"]["result"] == "handoff" for trace in answer_traces
    )
    cross_product_errors = sum(
        bool(trace["expected"]["cross_product"])
        and trace["actual"]["result"] == "answer"
        for trace in traces
    )

    def ratio(numerator: int, denominator: int) -> dict[str, int | float]:
        return {
            "numerator": numerator,
            "denominator": denominator,
            "rate": round(numerator / denominator, 4) if denominator else 0.0,
        }

    return {
        "source_hit_at_1": ratio(hit_at_1, len(answer_traces)),
        "source_hit_at_3": ratio(hit_at_3, len(answer_traces)),
        "incorrect_answer_rate": ratio(wrong_answers, len(traces)),
        "correct_handoff_rate": ratio(correct_handoffs, len(handoff_traces)),
        "answerable_but_incorrectly_handed_off_rate": ratio(incorrect_handoffs, len(answer_traces)),
        "cross_product_error_count": cross_product_errors,
    }


def _trace_case(service: RAGAnswerService, case: dict[str, Any]) -> dict[str, Any]:
    query = str(case["query"])
    hard_reason, raw_matches = service.trace_retrieval(query)
    payload = service.ask(query)
    expected_product_ids = set(str(value) for value in case.get("expected_product_ids", []))
    top_k = _top_k_trace(raw_matches)
    # Product IDs remain an internal evaluation mapping.  Public ``sources``
    # intentionally contain only display-safe provenance fields.
    returned_product_ids = payload.internal_returned_product_ids
    source_hit_at_1 = bool(top_k and top_k[0]["product_id"] in expected_product_ids)
    source_hit_at_3 = any(
        match["product_id"] in expected_product_ids for match in top_k[:3]
    )
    actual_result = "handoff" if payload.handoff_required else "answer"
    if case["expected"] == "answer":
        answer_is_correct = actual_result == "answer" and bool(
            expected_product_ids.intersection(returned_product_ids)
        )
    else:
        answer_is_correct = actual_result == "handoff"

    return {
        "case_id": case["id"],
        "category": case["category"],
        "query": query,
        "hard_safety_gate": {"triggered": bool(hard_reason), "reason": hard_reason},
        "top_k": top_k,
        "final": {
            "result": actual_result,
            "answer": payload.answer,
            "handoff_reason": payload.handoff_reason,
            "sources": payload.sources,
        },
        "expected": {
            "result": case["expected"],
            "product_ids": sorted(expected_product_ids),
            "cross_product": bool(case.get("cross_product", False)),
        },
        "actual": {
            "result": actual_result,
            "source_product_ids": returned_product_ids,
            "answer_is_correct": answer_is_correct,
        },
        "source_hit_at_1": source_hit_at_1,
        "source_hit_at_3": source_hit_at_3,
    }


def main() -> int:
    args = _arguments()
    settings = Settings.from_environment(PROJECT_ROOT)
    settings = replace(settings, retrieval_mode=args.retrieval_mode)
    thresholds = _parse_thresholds(args.thresholds) if args.thresholds else settings.evaluation_thresholds
    if len(thresholds) < 3:
        raise ValueError("EVALUATION_THRESHOLDS must contain at least three values")
    cases = json.loads((PROJECT_ROOT / "data" / "holdout_evaluation_questions.json").read_text(encoding="utf-8"))
    if len(cases) < 24:
        raise ValueError("holdout evaluation must contain at least 24 cases")

    with tempfile.TemporaryDirectory(prefix="sanding-rag-evaluation-") as temporary_dir:
        store = ChromaStore(Path(temporary_dir) / "chroma", "evaluation")
        embedder = (
            HashingTestEmbedder()
            if args.hashing_test_embedder
            else SentenceTransformerEmbedder(settings.embedding_model)
        )
        MarkdownIngestionPipeline(SemanticRecursiveSplitter(), embedder, store).ingest_path(
            PROJECT_ROOT / "data" / "sample"
        )
        comparisons: list[dict[str, Any]] = []
        for threshold in thresholds:
            service = RAGAnswerService(
                embedder=embedder,
                store=store,
                llm=ContextEchoLLM(),
                top_k=settings.top_k,
                min_relevance=threshold,
                retriever=make_retriever(settings, embedder, store),
            )
            traces = [_trace_case(service, case) for case in cases]
            comparisons.append(
                {
                    "min_relevance": threshold,
                    "metrics": _metrics(traces),
                    "traces": traces,
                }
            )

    report = {
        "dataset": "data/holdout_evaluation_questions.json",
        "dataset_case_count": len(cases),
        "embedding_mode": "hashing_test_embedder" if args.hashing_test_embedder else "sentence_transformers",
        "embedding_model": settings.embedding_model,
        "top_k": settings.top_k,
        "retrieval_mode": settings.retrieval_mode,
        "threshold_comparisons": comparisons,
    }
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "dataset_case_count": len(cases),
                "threshold_comparisons": [
                    {"min_relevance": item["min_relevance"], "metrics": item["metrics"]}
                    for item in comparisons
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
