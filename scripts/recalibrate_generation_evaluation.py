"""Recalculate a completed generation run against its public evidence contract.

This script is deliberately offline.  It never reads ``.env``, initializes an
embedding/Chroma runtime, imports an LLM client or makes a network request.  A
new derived run is written beside, never over, the immutable parent trace.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sanding_rag.generation_evaluation import (  # noqa: E402
    apply_manual_review_plan,
    evaluate_case,
    failure_examples,
    public_evidence_text,
    review_selection,
    summarize,
)
from sanding_rag.generation_evaluation_artifacts import demo_gate, render_report  # noqa: E402
from sanding_rag.markdown_loader import load_markdown_many  # noqa: E402


def _arguments() -> object:
    parser = ArgumentParser(description="Offline recalibration of one completed generation-evaluation trace")
    parser.add_argument(
        "--parent-trace",
        type=Path,
        default=PROJECT_ROOT / "data" / "runtime" / "generation_evaluation" / "latest" / "GENERATION_EVALUATION_TRACES.json",
        help="completed local generation trace to preserve and use as the immutable parent",
    )
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "sample",
        help="public Markdown catalog used only to map source to public product_name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="new derived-run directory; defaults to a gitignored calibrated sibling directory",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _known_fact_values(cases: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(point["accepted_values"][0])
            for case in cases
            for point in case["expected_answer_points"]
            if point.get("accepted_values")
        }
    )


def _public_names_by_source(catalog_dir: Path) -> dict[str, str]:
    """Read only public front-matter names, keyed by their public source path."""
    names: dict[str, str] = {}
    for document in load_markdown_many(catalog_dir):
        source = str(document.metadata["source"])
        product_name = str(document.metadata["product_name"])
        existing = names.get(source)
        if existing is not None and existing != product_name:
            raise ValueError(f"catalog source has conflicting public product names: {source}")
        names[source] = product_name
    return names


def _evidence_from_trace(trace: dict[str, Any], public_names_by_source: dict[str, str]) -> str:
    evidence: list[tuple[str, str]] = []
    for candidate in trace.get("retrieval_sources", []):
        if not isinstance(candidate, dict):
            raise ValueError(f"{trace['case_id']}: retrieval source must be an object")
        source = str(candidate.get("source", ""))
        product_name = public_names_by_source.get(source)
        if not product_name:
            raise ValueError(f"{trace['case_id']}: public product name unavailable for source {source!r}")
        text = str(candidate.get("text", ""))
        if not text:
            raise ValueError(f"{trace['case_id']}: retrieval source {source!r} has no stored text")
        evidence.append((product_name, text))
    return public_evidence_text(evidence)


def _parent_reference(parent_trace_path: Path, parent_report: dict[str, Any]) -> dict[str, Any]:
    try:
        display_path = str(parent_trace_path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        display_path = str(parent_trace_path.resolve())
    return {
        "trace_path": display_path,
        "trace_sha256": hashlib.sha256(parent_trace_path.read_bytes()).hexdigest(),
        "run_at": parent_report.get("run_at"),
        "model": parent_report.get("model"),
        "status": parent_report.get("status"),
    }


def _reusable_reviews(parent_traces: list[dict[str, Any]], selected_case_ids: set[str]) -> dict[str, dict[str, Any]]:
    reusable: dict[str, dict[str, Any]] = {}
    for trace in parent_traces:
        review = trace.get("manual_review")
        case_id = str(trace.get("case_id", ""))
        if (
            case_id in selected_case_ids
            and isinstance(review, dict)
            and review.get("status") == "completed"
            and review.get("decision") in {"pass", "fail"}
        ):
            reusable[case_id] = {
                "decision": review["decision"],
                "notes": str(review.get("notes", "")),
            }
    return reusable


def recalibrate_report(
    *,
    parent_report: dict[str, Any],
    parent_trace_path: Path,
    cases: list[dict[str, Any]],
    public_names_by_source: dict[str, str],
) -> dict[str, Any]:
    """Pure recalculation over stored answers and public catalog metadata only."""
    if parent_report.get("status") != "completed":
        raise ValueError("offline recalibration requires a completed parent generation-evaluation trace")
    parent_traces = parent_report.get("traces")
    if not isinstance(parent_traces, list):
        raise ValueError("completed parent trace must contain a traces list")
    cases_by_id = {str(case["id"]): case for case in cases}
    if set(cases_by_id) != {str(trace.get("case_id", "")) for trace in parent_traces}:
        raise ValueError("parent trace case IDs do not exactly match the frozen generation evaluation set")
    known_facts = _known_fact_values(cases)
    recalibrated_traces: list[dict[str, Any]] = []
    for parent_trace in parent_traces:
        trace = copy.deepcopy(parent_trace)
        case_id = str(trace["case_id"])
        case = cases_by_id[case_id]
        if str(trace.get("query", "")) != str(case["query"]):
            raise ValueError(f"{case_id}: parent trace query does not match the frozen evaluation case")
        parent_review = trace.pop("manual_review", None)
        trace["parent_manual_review"] = parent_review
        evidence_text = _evidence_from_trace(trace, public_names_by_source)
        trace["calibration_evidence"] = {
            "format": "商品名称 + 公开证据正文",
            "excluded_internal_fields": ["product_id", "chunk_id", "score"],
            "source_count": len(trace.get("retrieval_sources", [])),
        }
        # Keep the exact public-only evidence contract auditable in this derived
        # local trace.  It intentionally excludes every internal retrieval field.
        trace["calibration_evidence_text"] = evidence_text
        trace["evaluation"] = evaluate_case(
            case=case,
            answer=str(trace.get("final_answer", "")),
            handoff_required=bool(trace.get("handoff_required")),
            handoff_reason=trace.get("handoff_reason"),
            returned_sources=list(trace.get("returned_sources", [])),
            returned_product_ids=list(trace.get("returned_source_product_ids", [])),
            evidence_text=evidence_text,
            llm_called=bool(trace.get("llm_called")),
            known_fact_values=known_facts,
        )
        recalibrated_traces.append(trace)

    selected_case_ids = review_selection(recalibrated_traces)
    apply_manual_review_plan(recalibrated_traces, _reusable_reviews(parent_traces, selected_case_ids))
    summary = summarize(recalibrated_traces)
    return {
        "status": "completed",
        "run_kind": "offline_evidence_calibration",
        "run_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "dataset": "data/generation_evaluation_questions.json",
        "dataset_case_count": len(cases),
        "model": parent_report.get("model"),
        "retrieval_mode": parent_report.get("retrieval_mode"),
        "min_relevance": parent_report.get("min_relevance"),
        "evaluation_method": "deterministic rules only; no LLM judge or network request",
        "parent_run": _parent_reference(parent_trace_path, parent_report),
        "calibration": {
            "reason": "align groundedness evidence with the public product_name plus Markdown body passed to the model",
            "public_fields_included": ["product_name", "text"],
            "internal_fields_excluded": ["product_id", "chunk_id", "score"],
            "parent_trace_preserved": True,
        },
        "summary": summary,
        "failure_examples": failure_examples(recalibrated_traces),
        "demo_gate": demo_gate(summary),
        "traces": recalibrated_traces,
    }


def _default_output_dir(parent_report: dict[str, Any]) -> Path:
    run_label = re.sub(r"[^A-Za-z0-9._-]+", "-", str(parent_report.get("run_at", "unknown"))).strip("-")
    return PROJECT_ROOT / "data" / "runtime" / "generation_evaluation" / "calibrated" / f"{run_label}-public-evidence-v1"


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "GENERATION_EVALUATION_TRACES.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "GENERATION_EVALUATION_REPORT.md").write_text(render_report(report), encoding="utf-8")


def main() -> int:
    args = _arguments()
    try:
        parent_trace_path = _resolve(args.parent_trace)
        parent_report = json.loads(parent_trace_path.read_text(encoding="utf-8"))
        output_dir = _resolve(args.output_dir) if args.output_dir else _default_output_dir(parent_report)
        if output_dir.resolve() == parent_trace_path.parent.resolve():
            raise ValueError("calibration output directory must differ from the immutable parent run directory")
        cases = json.loads((PROJECT_ROOT / "data" / "generation_evaluation_questions.json").read_text(encoding="utf-8"))
        report = recalibrate_report(
            parent_report=parent_report,
            parent_trace_path=parent_trace_path,
            cases=cases,
            public_names_by_source=_public_names_by_source(_resolve(args.catalog_dir)),
        )
        _write_report(output_dir, report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "run_kind": report["run_kind"],
                "output_dir": str(output_dir),
                "parent_run": report["parent_run"],
                "automatic_overall": report["summary"]["automatic_overall"],
                "manual_review": report["summary"]["manual_review"],
                "demo_gate": report["demo_gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
