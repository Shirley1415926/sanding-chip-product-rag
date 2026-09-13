from __future__ import annotations

import sys
import unittest
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sanding_rag.generation_evaluation import (
    apply_manual_review_plan,
    evaluate_case,
    redact_text,
    review_selection,
    summarize,
)


ANSWER_CASE = {
    "id": "G-ANSWER",
    "must_handoff": False,
    "expected_answer_points": [
        {"id": "origin", "accepted_values": ["福建福州"]},
        {"id": "packaging", "accepted_values": ["单件礼盒"]},
    ],
    "required_sources": ["catalog/cork.md"],
    "required_product_ids": ["cork"],
}
HANDOFF_CASE = {
    "id": "G-HANDOFF",
    "must_handoff": True,
    "expected_handoff_reason": "inventory",
    "expected_answer_points": [],
    "required_sources": [],
    "required_product_ids": [],
}
SOURCE = {
    "source": "catalog/cork.md",
    "product_id": "cork",
    "source_url": "https://txs.wyfdev.com/product/cork/",
}


class GenerationEvaluationTests(unittest.TestCase):
    def test_generation_fixture_is_independent_and_has_required_contract(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        cases = json.loads(
            (project_root / "data" / "generation_evaluation_questions.json").read_text(encoding="utf-8")
        )
        frozen_queries = set()
        for fixture in (
            "test_questions.json",
            "holdout_evaluation_questions.json",
            "retrieval_stress_questions.json",
        ):
            for case in json.loads((project_root / "data" / fixture).read_text(encoding="utf-8")):
                frozen_queries.add(case.get("question", case.get("query")))
        self.assertTrue(24 <= len(cases) <= 30)
        self.assertFalse(any(case["query"] in frozen_queries for case in cases))
        for case in cases:
            for required_key in (
                "expected_answer_points",
                "required_sources",
                "required_product_ids",
                "must_handoff",
            ):
                self.assertIn(required_key, case)

    def test_answer_dimensions_pass_only_when_facts_and_source_are_supported(self) -> None:
        result = evaluate_case(
            case=ANSWER_CASE,
            answer="这款产地为福建福州，采用单件礼盒。",
            handoff_required=False,
            handoff_reason=None,
            returned_sources=[SOURCE],
            evidence_text="产地：福建福州。包装：单件礼盒。",
            llm_called=True,
            known_fact_values=["福建福州", "单件礼盒", "四川成都"],
        )
        self.assertTrue(result["automatic_overall_pass"])
        self.assertTrue(result["dimensions"]["groundedness"]["passed"])
        self.assertTrue(result["dimensions"]["completeness"]["passed"])
        self.assertTrue(result["dimensions"]["citation_correctness"]["passed"])

    def test_groundedness_flags_a_known_fact_not_present_in_evidence(self) -> None:
        result = evaluate_case(
            case=ANSWER_CASE,
            answer="这款产地为福建福州，采用单件礼盒，来自四川成都。",
            handoff_required=False,
            handoff_reason=None,
            returned_sources=[SOURCE],
            evidence_text="产地：福建福州。包装：单件礼盒。",
            llm_called=True,
            known_fact_values=["福建福州", "单件礼盒", "四川成都"],
        )
        self.assertFalse(result["dimensions"]["groundedness"]["passed"])
        self.assertIn("四川成都", result["dimensions"]["groundedness"]["reasons"][0])

    def test_handoff_requires_no_llm_call_or_sources(self) -> None:
        result = evaluate_case(
            case=HANDOFF_CASE,
            answer="抱歉，请转人工。",
            handoff_required=True,
            handoff_reason="inventory",
            returned_sources=[],
            evidence_text="",
            llm_called=False,
            known_fact_values=[],
        )
        self.assertTrue(result["automatic_overall_pass"])
        self.assertTrue(result["dimensions"]["safety_compliance"]["passed"])

    def test_handoff_flags_any_llm_call(self) -> None:
        result = evaluate_case(
            case=HANDOFF_CASE,
            answer="抱歉，请转人工。",
            handoff_required=True,
            handoff_reason="inventory",
            returned_sources=[],
            evidence_text="",
            llm_called=True,
            known_fact_values=[],
        )
        self.assertFalse(result["automatic_overall_pass"])
        self.assertTrue(
            any("仍调用了 LLM" in reason for reason in result["dimensions"]["safety_compliance"]["reasons"])
        )

    def test_review_plan_contains_all_failures_and_twenty_percent_of_passes(self) -> None:
        traces = []
        for index in range(10):
            traces.append({"case_id": f"P{index}", "evaluation": {"automatic_overall_pass": True}})
        traces.append({"case_id": "F1", "evaluation": {"automatic_overall_pass": False}})
        selected = review_selection(traces)
        self.assertIn("F1", selected)
        self.assertEqual(len(selected), 3)  # one failure plus ceil(10 * 20%) pass cases

    def test_manual_review_summary_keeps_pending_state_visible(self) -> None:
        traces = [
            {
                "case_id": "P1",
                "evaluation": {
                    "automatic_overall_pass": True,
                    "dimensions": {
                        "groundedness": {"applicable": True, "passed": True},
                        "answer_relevance": {"applicable": True, "passed": True},
                        "completeness": {"applicable": True, "passed": True},
                        "citation_correctness": {"applicable": True, "passed": True},
                        "safety_compliance": {"applicable": False, "passed": None},
                    },
                },
            },
            {
                "case_id": "F1",
                "evaluation": {
                    "automatic_overall_pass": False,
                    "dimensions": {
                        "groundedness": {"applicable": True, "passed": False},
                        "answer_relevance": {"applicable": True, "passed": True},
                        "completeness": {"applicable": True, "passed": True},
                        "citation_correctness": {"applicable": True, "passed": True},
                        "safety_compliance": {"applicable": False, "passed": None},
                    },
                },
            },
        ]
        apply_manual_review_plan(traces)
        summary = summarize(traces)
        self.assertEqual(summary["manual_review"]["required_case_count"], 2)
        self.assertEqual(summary["manual_review"]["pending_case_count"], 2)

    def test_trace_redaction_removes_common_secret_and_contact_patterns(self) -> None:
        text = "token sk-abcDEF123456, api_key=secret, 电话13800138000，a@example.com"
        redacted = redact_text(text)
        self.assertNotIn("sk-abcDEF123456", redacted)
        self.assertNotIn("13800138000", redacted)
        self.assertNotIn("a@example.com", redacted)


if __name__ == "__main__":
    unittest.main()
