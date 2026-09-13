"""Command-line entry points for the first-stage closed loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .ingestion import MarkdownIngestionPipeline
from .runtime import LazyProductionLLM, make_production_embedder, make_retriever, make_store
from .service import RAGAnswerService
from .splitter import SemanticRecursiveSplitter


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="叁鼎芯供应链商品知识助手（RAG MVP）")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="导入 Markdown 文件或目录")
    ingest.add_argument("path", type=Path)
    ask = commands.add_parser("ask", help="根据已导入资料回答问题")
    ask.add_argument("question")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = Settings.from_environment(_project_root())
    try:
        embedder = make_production_embedder(settings)
        store = make_store(settings)
        if args.command == "ingest":
            input_path = args.path if args.path.is_absolute() else Path.cwd() / args.path
            results = MarkdownIngestionPipeline(
                splitter=SemanticRecursiveSplitter(),
                embedder=embedder,
                store=store,
            ).ingest_path(input_path)
            print(json.dumps({"ingested": [result.to_dict() for result in results]}, ensure_ascii=False, indent=2))
            return

        service = RAGAnswerService(
            embedder=embedder,
            store=store,
            llm=LazyProductionLLM(settings),
            top_k=settings.top_k,
            min_relevance=settings.min_relevance,
            retriever=make_retriever(settings, embedder, store),
        )
        print(json.dumps(service.ask(args.question).to_dict(), ensure_ascii=False, indent=2))
    except Exception as exc:  # CLI should return an actionable short error, not a stack trace by default.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
