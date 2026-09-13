"""Run the answerable-only catalog retrieval stress suite across three modes.

The suite is independent from the frozen regression and safety-holdout sets.
It measures retrieval provenance only; it does not call an LLM or tune the
shared answer-safety threshold.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sanding_rag.chroma_store import ChromaStore  # noqa: E402
from sanding_rag.config import Settings  # noqa: E402
from sanding_rag.embedding import HashingTestEmbedder, SentenceTransformerEmbedder  # noqa: E402
from sanding_rag.ingestion import MarkdownIngestionPipeline  # noqa: E402
from sanding_rag.retrieval import BM25Retriever, DenseRetriever, HybridRRFRetriever, Retriever  # noqa: E402
from sanding_rag.splitter import SemanticRecursiveSplitter  # noqa: E402


def _arguments() -> object:
    parser = ArgumentParser(description="Compare Dense, BM25 and rank-only RRF retrieval")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "HYBRID_RETRIEVAL_TRACES.json",
        help="JSON comparison artifact path",
    )
    parser.add_argument(
        "--hashing-test-embedder",
        action="store_true",
        help="use deterministic test vectors only; BGE is the experiment default",
    )
    return parser.parse_args()


def _top_k_trace(matches: list[object]) -> list[dict[str, Any]]:
    return [
        {
            "source": str(match.metadata["source"]),
            "product_id": str(match.metadata["product_id"]),
            "score": round(float(match.score), 4),
        }
        for match in matches
    ]


def _trace_case(retriever: Retriever, case: dict[str, Any], top_k: int) -> dict[str, Any]:
    started = time.perf_counter_ns()
    matches = retriever.query(str(case["query"]), top_k)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    trace = _top_k_trace(matches)
    expected_source = str(case["expected_source"])
    expected_product_id = str(case["expected_product_id"])
    expected = lambda match: (  # noqa: E731 - concise, explicit comparison predicate
        match["source"] == expected_source and match["product_id"] == expected_product_id
    )
    expected_rank = next((index for index, match in enumerate(trace, start=1) if expected(match)), None)
    top_product_id = trace[0]["product_id"] if trace else None
    return {
        "case_id": case["id"],
        "category": case["category"],
        "query": case["query"],
        "expected": {"source": expected_source, "product_id": expected_product_id},
        "top_k": trace,
        "expected_rank": expected_rank,
        "latency_ms": round(elapsed_ms, 3),
        "cross_product_error": bool(top_product_id and top_product_id != expected_product_id),
    }


def _metrics(traces: list[dict[str, Any]]) -> dict[str, int | float]:
    count = len(traces)
    return {
        "source_hit_at_1": round(sum(trace["expected_rank"] == 1 for trace in traces) / count, 4),
        "source_hit_at_3": round(
            sum(bool(trace["expected_rank"] and trace["expected_rank"] <= 3) for trace in traces) / count,
            4,
        ),
        "mrr_at_top_k": round(
            sum(1 / int(trace["expected_rank"]) for trace in traces if trace["expected_rank"]) / count,
            4,
        ),
        "average_retrieval_latency_ms": round(sum(float(trace["latency_ms"]) for trace in traces) / count, 3),
        "cross_product_error_count": sum(bool(trace["cross_product_error"]) for trace in traces),
    }


def main() -> int:
    args = _arguments()
    settings = Settings.from_environment(PROJECT_ROOT)
    cases = json.loads((PROJECT_ROOT / "data" / "retrieval_stress_questions.json").read_text(encoding="utf-8"))
    if not 24 <= len(cases) <= 30:
        raise ValueError("retrieval stress suite must contain 24 to 30 cases")

    with tempfile.TemporaryDirectory(prefix="sanding-rag-retrieval-stress-") as temporary_dir:
        store = ChromaStore(Path(temporary_dir) / "chroma", "retrieval_stress")
        embedder = (
            HashingTestEmbedder()
            if args.hashing_test_embedder
            else SentenceTransformerEmbedder(settings.embedding_model)
        )
        MarkdownIngestionPipeline(SemanticRecursiveSplitter(), embedder, store).ingest_path(
            PROJECT_ROOT / "data" / "sample"
        )
        dense = DenseRetriever(embedder, store)
        bm25 = BM25Retriever.from_store(store)
        retrievers: dict[str, Retriever] = {
            "dense": dense,
            "bm25": bm25,
            "hybrid_rrf": HybridRRFRetriever(
                dense,
                bm25,
                rrf_k=settings.rrf_k,
                candidate_depth=settings.rrf_candidate_depth,
            ),
        }
        # Exclude one-time sentence-transformer thread-pool/model warm-up from
        # the per-query latency comparison.  The same shared embedder is used
        # by Dense and Hybrid, so this warm-up does not favour either mode.
        dense.query("检索预热", 1)
        comparisons = []
        for mode, retriever in retrievers.items():
            traces = [_trace_case(retriever, case, settings.top_k) for case in cases]
            comparisons.append({"mode": mode, "metrics": _metrics(traces), "traces": traces})

    report = {
        "dataset": "data/retrieval_stress_questions.json",
        "dataset_case_count": len(cases),
        "embedding_mode": "hashing_test_embedder" if args.hashing_test_embedder else "sentence_transformers",
        "embedding_model": settings.embedding_model,
        "top_k": settings.top_k,
        "rrf": {"rrf_k": settings.rrf_k, "candidate_depth": settings.rrf_candidate_depth},
        "comparisons": comparisons,
    }
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "dataset_case_count": len(cases),
                "comparisons": [
                    {"mode": item["mode"], "metrics": item["metrics"]} for item in comparisons
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
