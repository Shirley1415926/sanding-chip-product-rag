from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sanding_rag.markdown_loader import load_markdown_many


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script() -> object:
    spec = importlib.util.spec_from_file_location(
        "recalibrate_generation_evaluation_for_test",
        PROJECT_ROOT / "scripts" / "recalibrate_generation_evaluation.py",
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load recalibrate_generation_evaluation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _public_names() -> dict[str, str]:
    return {
        str(document.metadata["source"]): str(document.metadata["product_name"])
        for document in load_markdown_many(PROJECT_ROOT / "data" / "sample")
    }


def _parent_trace(case: dict[str, object], public_names: dict[str, str]) -> dict[str, object]:
    must_handoff = bool(case["must_handoff"])
    required_sources = [str(value) for value in case["required_sources"]]
    values = [
        str(point["accepted_values"][0])
        for point in case["expected_answer_points"]
        if point.get("accepted_values")
    ]
    retrieval_sources = [
        {
            "source": source,
            "source_url": "https://example.test/catalog",
            "product_id": "internal-only-id",
            "score": 0.99,
            "text": "\n".join(values) or "公开资料正文",
        }
        for source in required_sources
    ]
    if case["id"] == "G13":
        retrieval_sources = [
            {
                "source": "catalog/marketplace-gutian-bamboo-fungus.md",
                "source_url": "https://example.test/fungus",
                "product_id": "internal-only-id",
                "score": 0.6674,
                "text": "产地：福建古田。规格：80g/袋。包装：20 袋/箱。",
            }
        ]
        values = ["古田竹荪干货", "80g/袋", "20袋/箱"]
    return {
        "case_id": case["id"],
        "query": case["query"],
        "final_answer": "抱歉，请转人工。" if must_handoff else "；".join(values),
        "handoff_required": must_handoff,
        "handoff_reason": case.get("expected_handoff_reason") if must_handoff else None,
        "llm_called": not must_handoff,
        "retrieval_sources": [] if must_handoff else retrieval_sources,
        "returned_sources": []
        if must_handoff
        else [{"source": source, "source_url": "https://example.test/catalog"} for source in required_sources],
        "returned_source_product_ids": [] if must_handoff else list(case["required_product_ids"]),
        "evaluation": {"automatic_overall_pass": case["id"] != "G13"},
        "manual_review": {
            "required": True,
            "status": "completed",
            "decision": "pass",
            "notes": "父运行人工复核通过。",
        },
    }


class GenerationEvaluationRecalibrationTests(unittest.TestCase):
    def test_recalibration_uses_public_name_without_internal_ids_and_preserves_parent(self) -> None:
        recalibration = _load_script()
        cases = json.loads((PROJECT_ROOT / "data" / "generation_evaluation_questions.json").read_text(encoding="utf-8"))
        public_names = _public_names()
        parent = {
            "status": "completed",
            "run_at": "2026-09-13T22:24:39+08:00",
            "model": "test-model",
            "retrieval_mode": "dense",
            "min_relevance": 0.60,
            "traces": [_parent_trace(case, public_names) for case in cases],
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            parent_path = Path(temporary_dir) / "parent.json"
            parent_path.write_text(json.dumps(parent, ensure_ascii=False), encoding="utf-8")
            with patch("urllib.request.urlopen", side_effect=AssertionError("recalibration must stay offline")):
                report = recalibration.recalibrate_report(
                    parent_report=parent,
                    parent_trace_path=parent_path,
                    cases=cases,
                    public_names_by_source=public_names,
                )

        g13 = next(trace for trace in report["traces"] if trace["case_id"] == "G13")
        self.assertTrue(g13["evaluation"]["automatic_overall_pass"])
        self.assertEqual(g13["parent_manual_review"]["decision"], "pass")
        self.assertEqual(report["summary"]["automatic_overall"]["failed"], 0)
        self.assertEqual(report["summary"]["manual_review"]["pending_case_count"], 0)
        self.assertIn("trace_sha256", report["parent_run"])
        self.assertEqual(report["calibration"]["internal_fields_excluded"], ["product_id", "chunk_id", "score"])
        evidence = recalibration._evidence_from_trace(g13, public_names)
        self.assertIn("商品名称：古田竹荪干货", evidence)
        self.assertNotIn("internal-only-id", evidence)
        self.assertNotIn("0.6674", evidence)
        self.assertEqual(g13["calibration_evidence_text"], evidence)
        script_text = (PROJECT_ROOT / "scripts" / "recalibrate_generation_evaluation.py").read_text(encoding="utf-8")
        self.assertNotIn("load_dotenv", script_text)
        self.assertNotIn("OpenAICompatibleLLM", script_text)
        self.assertNotIn("make_production_embedder", script_text)
