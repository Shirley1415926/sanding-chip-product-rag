"""Run marketplace catalog acceptance questions in repeatable offline test mode."""

from __future__ import annotations

import json
import sys
import tempfile
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sanding_rag.chroma_store import ChromaStore  # noqa: E402
from sanding_rag.embedding import HashingTestEmbedder, SentenceTransformerEmbedder  # noqa: E402
from sanding_rag.ingestion import MarkdownIngestionPipeline  # noqa: E402
from sanding_rag.llm import ContextEchoLLM  # noqa: E402
from sanding_rag.service import RAGAnswerService  # noqa: E402
from sanding_rag.splitter import SemanticRecursiveSplitter  # noqa: E402


def _arguments() -> object:
    parser = ArgumentParser(description="Run the marketplace RAG acceptance questions")
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="use the real local Sentence Transformers model instead of the offline test embedder",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    cases = json.loads((PROJECT_ROOT / "data/test_questions.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="sanding-rag-acceptance-") as temporary_dir:
        store = ChromaStore(Path(temporary_dir) / "chroma", "acceptance")
        embedder = (
            SentenceTransformerEmbedder("BAAI/bge-small-zh-v1.5")
            if args.semantic
            else HashingTestEmbedder()
        )
        ingestion = MarkdownIngestionPipeline(SemanticRecursiveSplitter(), embedder, store)
        ingestion.ingest_path(PROJECT_ROOT / "data/sample")
        service = RAGAnswerService(
            embedder=embedder,
            store=store,
            llm=ContextEchoLLM(),
            top_k=4,
            min_relevance=0.17,
        )
        results: list[dict[str, object]] = []
        for case in cases:
            payload = service.ask(case["question"])
            sources = payload.sources
            passed = payload.handoff_required if case["expect"] == "handoff" else not payload.handoff_required
            if case["expect"] == "handoff" and case.get("handoff_reason"):
                passed = bool(passed and payload.handoff_reason == case["handoff_reason"])
            if case["expect"] == "answer":
                source_text = " ".join(str(source["source"]) for source in sources)
                passed = bool(passed and sources and case["source_contains"] in source_text)
                if case.get("answer_contains"):
                    passed = bool(passed and case["answer_contains"] in payload.answer)
            results.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "expected": case["expect"],
                    "actual": "handoff" if payload.handoff_required else "answer",
                    "sources": [source["source"] for source in sources],
                    "source_urls": [source["source_url"] for source in sources],
                    "handoff_reason": payload.handoff_reason,
                    "passed": passed,
                }
            )

    summary = {
        "embedding_mode": "sentence_transformers" if args.semantic else "hashing_test_embedder",
        "passed": sum(item["passed"] for item in results),
        "total": len(results),
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
