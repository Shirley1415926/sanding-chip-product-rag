"""Explicit real-LLM evaluation for final-answer faithfulness and safety.

This command is deliberately excluded from unit tests and normal acceptance
scripts.  It reads LLM_API_BASE, LLM_API_KEY and LLM_MODEL only from a local
``.env`` file, never prints the key, and redacts common secret/contact patterns
before writing local trace artifacts.
"""

from __future__ import annotations

import json
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
    DIMENSIONS,
    apply_manual_review_plan,
    evaluate_case,
    failure_examples,
    redact_text,
    summarize,
)
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
        runtime_error: str | None = None
        try:
            payload = service.ask(query)
            answer = payload.answer
            handoff_required = payload.handoff_required
            handoff_reason = payload.handoff_reason
            returned_sources = payload.sources
            returned_product_ids = payload.internal_returned_product_ids
            verified_used_source_ids = payload.internal_used_source_ids
        except Exception as exc:  # Preserve a redacted diagnostic without headers, keys or raw responses.
            runtime_error = redact_text(str(exc))[:400] or type(exc).__name__
            answer = ""
            handoff_required = False
            handoff_reason = "evaluation_runtime_error"
            returned_sources = []
            returned_product_ids = []
            verified_used_source_ids = []
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        llm_called = llm.calls > before_calls
        retrieval_sources = _top_k_trace(matches)
        evidence_text = "\n\n".join(str(match.text) for match in matches)
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
                "runtime_error": runtime_error,
                "evaluation": evaluation,
            }
        )
        print(
            f"[{index}/{len(cases)}] {case['id']}：完成；LLM={'是' if llm_called else '否'}；"
            f"结果={'转人工' if handoff_required else '回答'}"
            + (f"；错误={runtime_error}" if runtime_error else ""),
            file=sys.stderr,
            flush=True,
        )
    return traces


def _render_report(report: dict[str, Any]) -> str:
    lines = ["# 真实 LLM 回答质量评估", "", f"运行日期：{report['run_at']}", ""]
    if report["status"] != "completed":
        lines.extend(
            [
                "## 当前状态",
                "",
                "本次未调用真实 LLM，因此没有生成质量指标或人工复核结论。",
                f"原因：{report['reason']}",
                f"题目数量：{report['dataset_case_count']}；模型版本：未配置（未读取 `.env`）。",
                "",
                "评估脚本已就绪。仅在本地 `.env` 同时提供 `LLM_API_BASE`、`LLM_API_KEY`、`LLM_MODEL` 后，显式执行下列命令才会调用模型：",
                "",
                "```bash",
                "./.venv/bin/python scripts/run_generation_evaluation.py",
                "```",
                "",
                "这不会使用环境变量回退值，也不会在普通单元测试或 CI 中调用 API。真实运行的原始回答和 trace 默认写入 gitignore 的 `data/runtime/generation_evaluation/latest/`，不会自动进入提交。",
                "",
                "## 自动结果",
                "",
                "未运行，因此不存在 Groundedness、相关性、完整性、来源正确性或安全合规的自动指标。",
                "",
                "## 人工复核结果",
                "",
                "未运行，因此没有可供人工复核的真实模型回答；不能把准备清单当作人工复核已完成。真实运行时，所有自动失败题和至少 20%“已调用 LLM 且自动通过”题须人工复核；硬安全门题不计入生成回答抽样。",
                "",
                "## 已知边界",
                "",
                "未测试英文询盘、真实用户表达、提示注入、长文档或大规模语料。当前不能据此判断是否达到独立演示门槛，更不能宣称可正式商用。",
            ]
        )
        return "\n".join(lines) + "\n"

    summary = report["summary"]
    lines.extend(
        [
            "## 运行范围",
            "",
            f"模型：`{report['model']}`；题目数：{report['dataset_case_count']}；检索模式：`{report['retrieval_mode']}`；MIN_RELEVANCE：`{report['min_relevance']}`。",
            "",
            "自动判定使用确定性规则，不调用 LLM Judge：只读取问题、Top-K 检索证据、最终回答和预期要点。它不是绝对真相，特别是无法完整识别自然语言的隐含事实或高质量同义改写。",
            "",
            "## 自动结果（逐维）",
            "",
            "| 维度 | 通过 | 失败 | 不适用 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    labels = {
        "groundedness": "Groundedness / 忠实度",
        "answer_relevance": "Answer relevance / 相关性",
        "completeness": "Completeness / 完整性",
        "citation_correctness": "Citation correctness / 来源正确性",
        "safety_compliance": "Safety compliance / 安全合规",
    }
    for dimension in DIMENSIONS:
        metric = summary["automatic_dimensions"][dimension]
        lines.append(f"| {labels[dimension]} | {metric['passed']} | {metric['failed']} | {metric['not_applicable']} |")
    overall = summary["automatic_overall"]
    lines.extend(
        [
            "",
            f"自动整体结果（仅用于定位，不替代逐维结果）：通过 {overall['passed']}，失败 {overall['failed']}。",
            "",
            "## 人工复核结果",
            "",
            f"规则要求复核全部自动失败题，以及“已调用 LLM 且自动通过”题中的确定性 20% 抽样；硬安全门题不进入生成回答抽样。需要复核 {summary['manual_review']['required_case_count']} 题；已完成 {summary['manual_review']['completed_case_count']} 题；待人工复核 {summary['manual_review']['pending_case_count']} 题；人工判定通过 {summary['manual_review']['reviewer_pass_count']}，失败 {summary['manual_review']['reviewer_fail_count']}。",
            "",
        ]
    )
    failures = report["failure_examples"]
    lines.extend(["## 自动失败样例", ""])
    if not failures:
        lines.append("本次自动规则没有发现失败；仍需完成规定的人工抽样复核。")
    else:
        for example in failures[:5]:
            reason_text = "；".join(
                f"{dimension}: {', '.join(reasons)}" for dimension, reasons in example["reasons"].items()
            )
            lines.append(f"- `{example['case_id']}`：{example['query']}。{reason_text}")
    gate = report["demo_gate"]
    lines.extend(
        [
            "",
            "## 独立演示门槛",
            "",
            gate,
            "",
            "## 当前不能证明的边界",
            "",
            "本评估未测试英文询盘、真实用户表达、提示注入、长文档、大规模语料、并发、真实库存/履约/报价系统，以及长期资料更新后的表现。即使完成门槛，也仅代表限定目录与测试题下可做独立演示，不代表正式商用就绪。",
        ]
    )
    return "\n".join(lines) + "\n"


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
    report_path.write_text(_render_report(report), encoding="utf-8")


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
    apply_manual_review_plan(traces, _load_manual_reviews(args.manual_review_file))
    summary = summarize(traces)
    manual = summary["manual_review"]
    if summary["automatic_overall"]["failed"] == 0 and manual["pending_case_count"] == 0 and manual["reviewer_fail_count"] == 0:
        demo_gate = "已达到限定商城目录、固定题集和已完成复核条件下的独立演示门槛；不代表正式商用就绪。"
    else:
        demo_gate = "尚未达到独立演示门槛：需要先消除自动失败，并完成全部规定的人工复核。"
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
        "demo_gate": demo_gate,
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
