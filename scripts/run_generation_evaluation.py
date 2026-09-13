"""Explicit real-LLM evaluation for final-answer faithfulness and safety.

This command is deliberately excluded from unit tests and normal acceptance
scripts.  It reads LLM_API_BASE, LLM_API_KEY and LLM_MODEL only from a local
``.env`` file, never prints the key, and redacts common secret/contact patterns
before writing local trace artifacts.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import time
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sanding_rag.chroma_store import ChromaStore  # noqa: E402
from sanding_rag.config import Settings  # noqa: E402
from sanding_rag.embedding import EmbeddingProvider  # noqa: E402
from sanding_rag.generation_evaluation import (  # noqa: E402
    apply_manual_review_plan,
    evaluate_case,
    failure_examples,
    public_evidence_text,
    redact_text,
    summarize,
)
from sanding_rag.generation_evaluation_artifacts import demo_gate, render_report  # noqa: E402
from sanding_rag.ingestion import MarkdownIngestionPipeline  # noqa: E402
from sanding_rag.llm import LLMProvider, OpenAICompatibleLLM  # noqa: E402
from sanding_rag.runtime import make_production_embedder, make_retriever  # noqa: E402
from sanding_rag.service import RAGAnswerService  # noqa: E402
from sanding_rag.splitter import SemanticRecursiveSplitter  # noqa: E402


class _CountingLLM:
    """Records whether orchestration actually reached the real LLM boundary."""

    def __init__(self, delegate: LLMProvider) -> None:
        self._delegate = delegate
        self.calls = 0

    def answer(self, question: str, context: str, handoff_message: str) -> str:
        self.calls += 1
        return self._delegate.answer(question, context, handoff_message)


class EvaluationRunAborted(RuntimeError):
    """Stop the whole run for a provider/runtime failure, without a quality result."""

    def __init__(self, *, case_id: str, case_index: int, llm_called: bool, error: Exception) -> None:
        super().__init__(type(error).__name__)
        self.case_id = case_id
        self.case_index = case_index
        self.llm_called = llm_called
        self.error_type = type(error).__name__
        self.diagnostic = _safe_runtime_diagnostic(error)


def _safe_runtime_diagnostic(error: Exception) -> str:
    """Keep diagnostics actionable without persisting provider bodies or secrets."""
    text = redact_text(str(error))
    status_match = re.search(r"HTTP\s+(\d{3})", text)
    if status_match:
        status = int(status_match.group(1))
        if status in {401, 403}:
            return f"LLM authentication or permission was rejected (HTTP {status})"
        if status == 404:
            return "LLM endpoint or model was not found (HTTP 404)"
        if status == 429:
            return "LLM provider rate limit was reached (HTTP 429)"
        if 500 <= status <= 599:
            return f"LLM provider server error (HTTP {status})"
        return f"LLM request failed (HTTP {status})"
    if isinstance(error, TimeoutError) or "timeout" in text.lower() or "timed out" in text.lower():
        return "LLM request timed out"
    if "valid JSON envelope" in text:
        return "LLM provider response envelope was invalid"
    if "choices[0].message.content" in text:
        return "LLM provider response did not contain a usable message"
    if "URLError" in text or "network" in text.lower() or "OSError" in text:
        return "LLM network request failed"
    return f"LLM/runtime failure ({type(error).__name__})"


def _arguments() -> object:
    parser = ArgumentParser(description="Run explicit real-LLM catalog answer evaluation")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="write a transparent not-run report and review plan without reading an API key or calling any model",
    )
    parser.add_argument(
        "--manual-review-file",
        type=Path,
        help="optional local JSON mapping case IDs to {decision: pass|fail, notes}; never commit this file",
    )
    parser.add_argument(
        "--traces-output",
        type=Path,
        default=None,
        help="optional trace JSON path; real runs default to gitignored data/runtime/",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="optional Markdown report path; real runs default to gitignored data/runtime/",
    )
    return parser.parse_args()


def _dotenv_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _llm_from_dotenv(project_root: Path) -> tuple[OpenAICompatibleLLM, str]:
    values = _dotenv_values(project_root / ".env")
    required = ("LLM_API_BASE", "LLM_API_KEY", "LLM_MODEL")
    missing = [key for key in required if not values.get(key) or values.get(key) == "replace-me"]
    if missing:
        raise ValueError(
            "真实生成评估要求在本地 .env 设置 " + ", ".join(missing) + "；不会读取环境变量回退值。"
        )
    provider = values.get("LLM_PROVIDER", "openai_compatible")
    if provider != "openai_compatible":
        raise ValueError("真实生成评估当前只支持 .env 中 LLM_PROVIDER=openai_compatible")
    return (
        OpenAICompatibleLLM(
            api_base=values["LLM_API_BASE"],
            api_key=values["LLM_API_KEY"],
            model=values["LLM_MODEL"],
        ),
        values["LLM_MODEL"],
    )


def _load_manual_reviews(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "reviews" in raw:
        raw = raw["reviews"]
    if not isinstance(raw, dict):
        raise ValueError("manual review JSON must be an object keyed by case ID")
    return {str(case_id): dict(review) for case_id, review in raw.items() if isinstance(review, dict)}


def _known_fact_values(cases: list[dict[str, Any]]) -> list[str]:
    """Use canonical point forms for conservative cross-product fact checking."""
    return sorted(
        {
            str(point["accepted_values"][0])
            for case in cases
            for point in case["expected_answer_points"]
            if point.get("accepted_values")
        }
    )


def _top_k_trace(matches: list[object]) -> list[dict[str, Any]]:
    return [
        {
            "source": str(match.metadata["source"]),
            "source_url": str(match.metadata["source_url"]),
            "product_id": str(match.metadata["product_id"]),
            "score": round(float(match.score), 4),
            "text": redact_text(str(match.text)),
        }
        for match in matches
    ]


def _not_run_traces(cases: list[dict[str, Any]], reason: str) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case["id"],
            "category": case["category"],
            "query": case["query"],
            "expected": {
                "must_handoff": case["must_handoff"],
                "expected_answer_points": case["expected_answer_points"],
                "required_sources": case["required_sources"],
                "required_product_ids": case["required_product_ids"],
            },
            "final_answer": None,
            "handoff_required": None,
            "handoff_reason": None,
            "llm_called": False,
            "retrieval_sources": [],
            "returned_sources": [],
            "model": None,
            "elapsed_ms": None,
            "evaluation": None,
            "manual_review": {
                "required": False,
                "status": "not_run",
                "decision": None,
                "notes": reason,
            },
        }
        for case in cases
    ]


def _run_cases(
    *,
    cases: list[dict[str, Any]],
    settings: Settings,
    embedder: EmbeddingProvider,
    store: ChromaStore,
    llm: _CountingLLM,
    model_name: str,
) -> list[dict[str, Any]]:
    service = RAGAnswerService(
        embedder=embedder,
        store=store,
        llm=llm,
        top_k=settings.top_k,
        min_relevance=settings.min_relevance,
        retriever=make_retriever(settings, embedder, store),
    )
    known_facts = _known_fact_values(cases)
    traces = []
    for index, case in enumerate(cases, start=1):
        query = str(case["query"])
        print(f"[{index}/{len(cases)}] {case['id']}：开始", file=sys.stderr, flush=True)
        hard_safety_reason, matches = service.trace_retrieval(query)
        before_calls = llm.calls
        started = time.perf_counter_ns()
        try:
            payload = service.ask(query)
            answer = payload.answer
            handoff_required = payload.handoff_required
            handoff_reason = payload.handoff_reason
            returned_sources = payload.sources
            returned_product_ids = payload.internal_returned_product_ids
            verified_used_source_ids = payload.internal_used_source_ids
        except Exception as exc:
            # Invalid answer JSON/source IDs are converted by RAGAnswerService
            # into ordinary handoff badcases. An exception means the provider or
            # runtime failed, so no later per-case quality conclusion is valid.
            raise EvaluationRunAborted(
                case_id=str(case["id"]),
                case_index=index,
                llm_called=llm.calls > before_calls,
                error=exc,
            ) from exc
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        llm_called = llm.calls > before_calls
        retrieval_sources = _top_k_trace(matches)
        evidence_text = public_evidence_text(
            (str(match.metadata["product_name"]), str(match.text)) for match in matches
        )
        evaluation = evaluate_case(
            case=case,
            answer=answer,
            handoff_required=handoff_required,
            handoff_reason=handoff_reason,
            returned_sources=returned_sources,
            returned_product_ids=returned_product_ids,
            evidence_text=evidence_text,
            llm_called=llm_called,
            known_fact_values=known_facts,
        )
        traces.append(
            {
                "case_id": case["id"],
                "category": case["category"],
                "query": query,
                "expected": {
                    "must_handoff": case["must_handoff"],
                    "expected_handoff_reason": case.get("expected_handoff_reason"),
                    "expected_answer_points": case["expected_answer_points"],
                    "required_sources": case["required_sources"],
                    "required_product_ids": case["required_product_ids"],
                },
                "hard_safety_gate": {"triggered": bool(hard_safety_reason), "reason": hard_safety_reason},
                "final_answer": redact_text(answer),
                "handoff_required": handoff_required,
                "handoff_reason": handoff_reason,
                "llm_called": llm_called,
                "retrieval_sources": retrieval_sources,
                "returned_sources": returned_sources,
                "returned_source_product_ids": returned_product_ids,
                "verified_used_source_ids": verified_used_source_ids,
                "model": model_name,
                "elapsed_ms": round(elapsed_ms, 3),
                "evaluation": evaluation,
            }
        )
        print(
            f"[{index}/{len(cases)}] {case['id']}：完成；LLM={'是' if llm_called else '否'}；"
            f"结果={'转人工' if handoff_required else '回答'}",
            file=sys.stderr,
            flush=True,
        )
    return traces


def _write_outputs(args: object, report: dict[str, Any]) -> None:
    if args.traces_output is None:
        traces_path = (
            PROJECT_ROOT / "docs" / "GENERATION_EVALUATION_TRACES.json"
            if args.prepare_only
            else PROJECT_ROOT / "data" / "runtime" / "generation_evaluation" / "latest" / "GENERATION_EVALUATION_TRACES.json"
        )
    else:
        traces_path = args.traces_output if args.traces_output.is_absolute() else PROJECT_ROOT / args.traces_output
    if args.report_output is None:
        report_path = (
            PROJECT_ROOT / "docs" / "GENERATION_EVALUATION_REPORT.md"
            if args.prepare_only
            else PROJECT_ROOT / "data" / "runtime" / "generation_evaluation" / "latest" / "GENERATION_EVALUATION_REPORT.md"
        )
    else:
        report_path = args.report_output if args.report_output.is_absolute() else PROJECT_ROOT / args.report_output
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    traces_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(report), encoding="utf-8")


def _aborted_report(
    *,
    run_at: str,
    cases: list[dict[str, Any]],
    model_name: str,
    settings: Settings,
    aborted: EvaluationRunAborted,
) -> dict[str, Any]:
    """Create a diagnostics-only artifact; never attach partial quality traces."""
    return {
        "status": "aborted",
        "reason": "真实 LLM 运行时错误；未生成质量结论",
        "run_at": run_at,
        "aborted_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "dataset": "data/generation_evaluation_questions.json",
        "dataset_case_count": len(cases),
        "model": model_name,
        "retrieval_mode": settings.retrieval_mode,
        "min_relevance": settings.min_relevance,
        "diagnostic": {
            "case_id": aborted.case_id,
            "case_index": aborted.case_index,
            "llm_called": aborted.llm_called,
            "error_type": aborted.error_type,
            "message": aborted.diagnostic,
        },
        "traces": [],
    }


def main() -> int:
    args = _arguments()
    cases = json.loads((PROJECT_ROOT / "data" / "generation_evaluation_questions.json").read_text(encoding="utf-8"))
    if not 24 <= len(cases) <= 30:
        raise ValueError("generation evaluation set must contain 24 to 30 cases")
    run_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    if args.prepare_only:
        report = {
            "status": "not_run",
            "reason": "prepare-only：未读取 .env，也未调用真实 LLM",
            "run_at": run_at,
            "dataset": "data/generation_evaluation_questions.json",
            "dataset_case_count": len(cases),
            "model": None,
            "retrieval_mode": "dense",
            "min_relevance": 0.60,
            "traces": _not_run_traces(cases, "真实 LLM 评估尚未运行"),
        }
        _write_outputs(args, report)
        print(json.dumps({"status": report["status"], "reason": report["reason"]}, ensure_ascii=False))
        return 0

    try:
        llm_delegate, model_name = _llm_from_dotenv(PROJECT_ROOT)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    settings = Settings.from_environment(PROJECT_ROOT)
    if settings.retrieval_mode != "dense" or settings.min_relevance != 0.60:
        raise ValueError("真实生成评估固定验证 Dense 基线和 MIN_RELEVANCE=0.60；请检查本地 .env")
    try:
        with tempfile.TemporaryDirectory(prefix="sanding-rag-generation-evaluation-") as temporary_dir:
            store = ChromaStore(Path(temporary_dir) / "chroma", "generation_evaluation")
            embedder = make_production_embedder(settings)
            MarkdownIngestionPipeline(SemanticRecursiveSplitter(), embedder, store).ingest_path(
                PROJECT_ROOT / "data" / "sample"
            )
            traces = _run_cases(
                cases=cases,
                settings=settings,
                embedder=embedder,
                store=store,
                llm=_CountingLLM(llm_delegate),
                model_name=model_name,
            )
    except EvaluationRunAborted as exc:
        report = _aborted_report(
            run_at=run_at,
            cases=cases,
            model_name=model_name,
            settings=settings,
            aborted=exc,
        )
        _write_outputs(args, report)
        print(
            json.dumps(
                {
                    "status": "aborted",
                    "case_id": exc.case_id,
                    "diagnostic": exc.diagnostic,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    apply_manual_review_plan(traces, _load_manual_reviews(args.manual_review_file))
    summary = summarize(traces)
    report = {
        "status": "completed",
        "run_at": run_at,
        "dataset": "data/generation_evaluation_questions.json",
        "dataset_case_count": len(cases),
        "model": model_name,
        "retrieval_mode": settings.retrieval_mode,
        "min_relevance": settings.min_relevance,
        "evaluation_method": "deterministic rules only; no LLM judge",
        "summary": summary,
        "failure_examples": failure_examples(traces),
        "demo_gate": demo_gate(summary),
        "traces": traces,
    }
    _write_outputs(args, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "model": report["model"],
                "dataset_case_count": report["dataset_case_count"],
                "automatic_overall": summary["automatic_overall"],
                "manual_review": summary["manual_review"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
