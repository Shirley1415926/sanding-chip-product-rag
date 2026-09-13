"""Deterministic checks and review planning for real-LLM answer evaluation.

This module intentionally does not call an LLM judge.  Its rules inspect only
the evaluation question, retrieved evidence, returned answer and declared
expected points, which makes the automatic portion reproducible.  Human review
is still required for every automatic failure and a seeded 20% pass sample.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Iterable


DIMENSIONS = (
    "groundedness",
    "answer_relevance",
    "completeness",
    "citation_correctness",
    "safety_compliance",
)

_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"(?i)(api[_ -]?key\s*[:=]\s*)\S+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+"), r"\1[REDACTED]"),
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[REDACTED_PHONE]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
)
_RISK_TERMS = (
    "库存",
    "现货",
    "交期",
    "发货",
    "报价",
    "经销价",
    "批量价",
    "保质期",
    "配料",
    "过敏原",
    "认证",
    "刻字",
    "logo",
)


def redact_text(text: str) -> str:
    """Remove common credentials/contact patterns before a trace is persisted."""
    redacted = text
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def normalized(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def point_coverage(answer: str, points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Check each declared answer point against allowed faithful surface forms."""
    answer_normalized = normalized(answer)
    coverage = []
    for point in points:
        accepted_values = [str(value) for value in point.get("accepted_values", [])]
        matches = [value for value in accepted_values if normalized(value) in answer_normalized]
        coverage.append(
            {
                "id": str(point["id"]),
                "passed": bool(matches),
                "matched_values": matches,
                "accepted_values": accepted_values,
            }
        )
    return coverage


def _dimension(applicable: bool, passed: bool | None, reasons: Iterable[str]) -> dict[str, Any]:
    return {
        "applicable": applicable,
        "passed": passed,
        "reasons": list(reasons),
    }


def evaluate_case(
    *,
    case: dict[str, Any],
    answer: str,
    handoff_required: bool,
    handoff_reason: str | None,
    returned_sources: list[dict[str, Any]],
    evidence_text: str,
    llm_called: bool,
    known_fact_values: list[str],
) -> dict[str, Any]:
    """Evaluate a generated answer using only declared data and retrieved context."""
    must_handoff = bool(case["must_handoff"])
    answer = redact_text(answer)
    answer_normalized = normalized(answer)
    evidence_normalized = normalized(evidence_text)
    returned_source_names = {str(source["source"]) for source in returned_sources}
    returned_product_ids = {str(source["product_id"]) for source in returned_sources}

    if must_handoff:
        expected_reason = case.get("expected_handoff_reason")
        safety_reasons = []
        safety_passed = True
        if not handoff_required:
            safety_passed = False
            safety_reasons.append("应转人工的问题返回了正常回答")
        if llm_called:
            safety_passed = False
            safety_reasons.append("应转人工的问题仍调用了 LLM")
        if returned_sources:
            safety_passed = False
            safety_reasons.append("应转人工的问题返回了商品来源")
        if expected_reason and handoff_reason != expected_reason:
            safety_passed = False
            safety_reasons.append(f"转人工原因应为 {expected_reason}，实际为 {handoff_reason}")
        citation = _dimension(True, not returned_sources, [] if not returned_sources else ["转人工结果不应附带商品来源"])
        dimensions = {
            "groundedness": _dimension(False, None, ["转人工题不评估正常商品事实回答"]),
            "answer_relevance": _dimension(False, None, ["转人工题不评估正常商品事实回答"]),
            "completeness": _dimension(False, None, ["转人工题没有应回答商品要点"]),
            "citation_correctness": citation,
            "safety_compliance": _dimension(True, safety_passed, safety_reasons),
        }
        return {
            "point_coverage": [],
            "dimensions": dimensions,
            "automatic_overall_pass": bool(citation["passed"] and safety_passed),
        }

    point_results = point_coverage(answer, list(case["expected_answer_points"]))
    missing_points = [point["id"] for point in point_results if not point["passed"]]
    required_sources = {str(value) for value in case["required_sources"]}
    required_product_ids = {str(value) for value in case["required_product_ids"]}
    missing_sources = sorted(required_sources - returned_source_names)
    missing_product_ids = sorted(required_product_ids - returned_product_ids)
    unexpected_sources = sorted(returned_source_names - required_sources)
    unexpected_product_ids = sorted(returned_product_ids - required_product_ids)
    citation_reasons: list[str] = []
    if missing_sources:
        citation_reasons.append(f"缺少要求的来源：{', '.join(missing_sources)}")
    if missing_product_ids:
        citation_reasons.append(f"缺少要求的商品 ID：{', '.join(missing_product_ids)}")
    if unexpected_sources:
        citation_reasons.append(f"包含非目标来源：{', '.join(unexpected_sources)}")
    if unexpected_product_ids:
        citation_reasons.append(f"包含非目标商品 ID：{', '.join(unexpected_product_ids)}")
    citation_passed = not citation_reasons

    disclosed_known_facts = sorted(
        {
            value
            for value in known_fact_values
            if len(normalized(value)) > 1 and normalized(value) in answer_normalized
        }
    )
    unsupported_facts = [
        value for value in disclosed_known_facts if normalized(value) not in evidence_normalized
    ]
    unsupported_risk_terms = [
        term for term in _RISK_TERMS
        if normalized(term) in answer_normalized and normalized(term) not in evidence_normalized
    ]
    groundedness_reasons = []
    if handoff_required:
        groundedness_reasons.append("可回答题被转人工")
    if unsupported_facts:
        groundedness_reasons.append(f"回答包含未出现在检索证据中的已知目录事实：{', '.join(unsupported_facts)}")
    if unsupported_risk_terms:
        groundedness_reasons.append(f"回答包含证据未支持的风险事实：{', '.join(unsupported_risk_terms)}")
    groundedness_passed = not groundedness_reasons

    relevance_reasons = []
    if handoff_required:
        relevance_reasons.append("可回答题被转人工")
    if not answer.strip():
        relevance_reasons.append("回答为空")
    if point_results and not any(point["passed"] for point in point_results):
        relevance_reasons.append("回答未覆盖任何预期要点")
    relevance_passed = not relevance_reasons
    completeness_passed = not handoff_required and not missing_points
    completeness_reasons = [] if completeness_passed else (
        ["可回答题被转人工"] if handoff_required else [f"缺少要点：{', '.join(missing_points)}"]
    )

    dimensions = {
        "groundedness": _dimension(True, groundedness_passed, groundedness_reasons),
        "answer_relevance": _dimension(True, relevance_passed, relevance_reasons),
        "completeness": _dimension(True, completeness_passed, completeness_reasons),
        "citation_correctness": _dimension(True, citation_passed, citation_reasons),
        "safety_compliance": _dimension(False, None, ["可回答题不适用转人工安全维度"]),
    }
    applicable_dimensions = [result for result in dimensions.values() if result["applicable"]]
    return {
        "point_coverage": point_results,
        "dimensions": dimensions,
        "automatic_overall_pass": all(bool(result["passed"]) for result in applicable_dimensions),
    }


def review_selection(traces: list[dict[str, Any]], seed: str = "generation-evaluation-v1") -> set[str]:
    """Select all automatic failures plus a deterministic 20% sample of passes."""
    failures = [trace["case_id"] for trace in traces if not trace["evaluation"]["automatic_overall_pass"]]
    passes = [trace["case_id"] for trace in traces if trace["evaluation"]["automatic_overall_pass"]]
    sample_size = math.ceil(len(passes) * 0.20)
    sample = sorted(
        passes,
        key=lambda case_id: hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest(),
    )[:sample_size]
    return set(failures + sample)


def apply_manual_review_plan(
    traces: list[dict[str, Any]],
    provided_reviews: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Attach a transparent pending/completed human-review state to each trace."""
    selected = review_selection(traces)
    provided_reviews = provided_reviews or {}
    for trace in traces:
        case_id = str(trace["case_id"])
        review = provided_reviews.get(case_id)
        required = case_id in selected
        if required and review:
            decision = str(review.get("decision", "")).lower()
            if decision not in {"pass", "fail"}:
                raise ValueError(f"manual review for {case_id} must set decision to 'pass' or 'fail'")
            trace["manual_review"] = {
                "required": True,
                "status": "completed",
                "decision": decision,
                "notes": redact_text(str(review.get("notes", ""))),
            }
        elif required:
            trace["manual_review"] = {
                "required": True,
                "status": "pending_human_review",
                "decision": None,
                "notes": "",
            }
        else:
            trace["manual_review"] = {
                "required": False,
                "status": "not_selected",
                "decision": None,
                "notes": "",
            }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Return per-dimension automatic and manual-review counts without a total score."""
    automatic: dict[str, dict[str, int]] = {
        dimension: {"passed": 0, "failed": 0, "not_applicable": 0}
        for dimension in DIMENSIONS
    }
    for trace in traces:
        for dimension, result in trace["evaluation"]["dimensions"].items():
            if not result["applicable"]:
                automatic[dimension]["not_applicable"] += 1
            elif result["passed"]:
                automatic[dimension]["passed"] += 1
            else:
                automatic[dimension]["failed"] += 1
    selected = [trace["manual_review"] for trace in traces if trace["manual_review"]["required"]]
    completed = [review for review in selected if review["status"] == "completed"]
    manual = {
        "required_case_count": len(selected),
        "completed_case_count": len(completed),
        "pending_case_count": len(selected) - len(completed),
        "reviewer_pass_count": sum(review["decision"] == "pass" for review in completed),
        "reviewer_fail_count": sum(review["decision"] == "fail" for review in completed),
    }
    return {
        "automatic_dimensions": automatic,
        "automatic_overall": {
            "passed": sum(trace["evaluation"]["automatic_overall_pass"] for trace in traces),
            "failed": sum(not trace["evaluation"]["automatic_overall_pass"] for trace in traces),
        },
        "manual_review": manual,
    }


def failure_examples(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return concise failure evidence for the generated Markdown report."""
    examples = []
    for trace in traces:
        if trace["evaluation"]["automatic_overall_pass"]:
            continue
        reasons = {
            dimension: result["reasons"]
            for dimension, result in trace["evaluation"]["dimensions"].items()
            if result["applicable"] and not result["passed"]
        }
        examples.append(
            {
                "case_id": trace["case_id"],
                "query": trace["query"],
                "reasons": reasons,
            }
        )
    return examples
