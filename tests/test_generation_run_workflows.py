from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sanding_rag.domain import RetrievedChunk
from sanding_rag.generation_evaluation import apply_manual_review_plan, failure_examples, summarize
from sanding_rag.generation_evaluation_artifacts import demo_gate


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(module_name: str, filename: str) -> object:
    spec = importlib.util.spec_from_file_location(module_name, PROJECT_ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-cork",
        text="# 三坊七巷主题福州软木画\n- 公开展示价：¥328。",
        metadata={
            "source": "catalog/cork.md",
            "source_url": "https://txs.wyfdev.com/product/cork/",
            "product_id": "cork",
            "product_name": "三坊七巷主题福州软木画",
            "document_type": "product",
            "updated_at": "2026-09-13",
        },
        score=0.99,
    )


def _completed_trace(case_id: str, *, automatic_pass: bool, llm_called: bool) -> dict[str, object]:
    return {
        "case_id": case_id,
        "query": "测试问题",
        "llm_called": llm_called,
        "evaluation": {
            "automatic_overall_pass": automatic_pass,
            "dimensions": {
                "groundedness": {"applicable": True, "passed": automatic_pass, "reasons": []},
                "answer_relevance": {"applicable": True, "passed": True, "reasons": []},
                "completeness": {"applicable": True, "passed": True, "reasons": []},
                "citation_correctness": {"applicable": True, "passed": True, "reasons": []},
                "safety_compliance": {"applicable": False, "passed": None, "reasons": []},
            },
        },
    }


class _StaticRetriever:
    def query(self, question: str, top_k: int) -> list[RetrievedChunk]:
        return [_chunk()]


class _NoopEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.0]


class _NoopStore:
    def query(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        return [_chunk()]


class _RuntimeFailingLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.synthetic_secret = "s" + "k-" + "synthetic-test-token"

    def answer(self, question: str, context: str, handoff_message: str) -> str:
        self.calls += 1
        raise RuntimeError(f"LLM request failed: HTTP 401; Authorization: Bearer {self.synthetic_secret}")


class GenerationRunWorkflowTests(unittest.TestCase):
    def test_provider_runtime_error_stops_on_first_case_and_builds_aborted_artifact(self) -> None:
        runner = _load_script("run_generation_evaluation_for_test", "run_generation_evaluation.py")
        failing = _RuntimeFailingLLM()
        settings = SimpleNamespace(top_k=4, min_relevance=0.60, retrieval_mode="dense")
        cases = [{"id": "G-FAIL", "query": "软木画展示价是多少？", "expected_answer_points": []}]

        with patch.object(runner, "make_retriever", return_value=_StaticRetriever()):
            with self.assertRaises(runner.EvaluationRunAborted) as raised:
                runner._run_cases(
                    cases=cases,
                    settings=settings,
                    embedder=_NoopEmbedder(),
                    store=_NoopStore(),
                    llm=runner._CountingLLM(failing),
                    model_name="test-model",
                )

        aborted = raised.exception
        self.assertEqual(failing.calls, 1)
        self.assertEqual(aborted.case_id, "G-FAIL")
        self.assertTrue(aborted.llm_called)
        self.assertNotIn(failing.synthetic_secret, aborted.diagnostic)
        report = runner._aborted_report(
            run_at="2026-09-13T00:00:00+08:00",
            cases=cases,
            model_name="test-model",
            settings=settings,
            aborted=aborted,
        )
        self.assertEqual(report["status"], "aborted")
        self.assertNotIn("summary", report)
        self.assertNotIn("failure_examples", report)
        self.assertEqual(report["traces"], [])
        with tempfile.TemporaryDirectory() as temporary_dir:
            trace_path = Path(temporary_dir) / "GENERATION_EVALUATION_TRACES.json"
            markdown_path = Path(temporary_dir) / "GENERATION_EVALUATION_REPORT.md"
            runner._write_outputs(
                SimpleNamespace(
                    traces_output=trace_path,
                    report_output=markdown_path,
                    prepare_only=False,
                ),
                report,
            )
            persisted = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["status"], "aborted")
            self.assertNotIn("completed", markdown_path.read_text(encoding="utf-8"))

    def test_main_returns_nonzero_and_never_marks_runtime_abort_completed(self) -> None:
        runner = _load_script("run_generation_evaluation_main_abort_test", "run_generation_evaluation.py")
        settings = SimpleNamespace(top_k=4, min_relevance=0.60, retrieval_mode="dense")
        synthetic_secret = "s" + "k-" + "synthetic-test-token"
        aborted = runner.EvaluationRunAborted(
            case_id="G01",
            case_index=1,
            llm_called=True,
            error=RuntimeError(f"LLM request failed: HTTP 401; Authorization: Bearer {synthetic_secret}"),
        )

        class _NoopIngestion:
            def ingest_path(self, path: Path) -> None:
                return None

        with tempfile.TemporaryDirectory() as temporary_dir:
            traces_path = Path(temporary_dir) / "GENERATION_EVALUATION_TRACES.json"
            markdown_path = Path(temporary_dir) / "GENERATION_EVALUATION_REPORT.md"
            arguments = SimpleNamespace(
                prepare_only=False,
                traces_output=traces_path,
                report_output=markdown_path,
                manual_review_file=None,
            )
            with (
                patch.object(runner, "_arguments", return_value=arguments),
                patch.object(runner, "_llm_from_dotenv", return_value=(object(), "test-model")),
                patch.object(runner.Settings, "from_environment", return_value=settings),
                patch.object(runner, "ChromaStore", return_value=object()),
                patch.object(runner, "make_production_embedder", return_value=object()),
                patch.object(runner, "MarkdownIngestionPipeline", return_value=_NoopIngestion()),
                patch.object(runner, "_run_cases", side_effect=aborted),
            ):
                self.assertEqual(runner.main(), 2)

            report = json.loads(traces_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "aborted")
            self.assertNotIn("summary", report)
            self.assertNotIn(synthetic_secret, markdown_path.read_text(encoding="utf-8"))

    def test_offline_manual_review_updates_only_selected_cases_without_dotenv_or_llm(self) -> None:
        manual = _load_script("apply_generation_manual_review_for_test", "apply_generation_manual_review.py")
        traces = [
            _completed_trace("G-FAIL", automatic_pass=False, llm_called=True),
            _completed_trace("G-PASS", automatic_pass=True, llm_called=True),
            _completed_trace("G-SAFETY", automatic_pass=True, llm_called=False),
        ]
        apply_manual_review_plan(traces)
        initial_summary = summarize(traces)
        report = {
            "status": "completed",
            "run_at": "2026-09-13T00:00:00+08:00",
            "dataset_case_count": len(traces),
            "model": "test-model",
            "retrieval_mode": "dense",
            "min_relevance": 0.60,
            "summary": initial_summary,
            "failure_examples": failure_examples(traces),
            "demo_gate": demo_gate(initial_summary),
            "traces": traces,
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir) / "data" / "runtime" / "generation_evaluation" / "latest"
            run_dir.mkdir(parents=True)
            (run_dir / "GENERATION_EVALUATION_TRACES.json").write_text(
                json.dumps(report, ensure_ascii=False), encoding="utf-8"
            )
            reviews = {
                "G-FAIL": {"decision": "fail", "notes": "人工确认自动失败。"},
                "G-PASS": {"decision": "pass", "notes": "人工抽样通过。"},
            }
            with patch("urllib.request.urlopen", side_effect=AssertionError("offline review must not use network")):
                updated = manual.apply_review_to_run(run_dir, reviews)

            manual_summary = updated["summary"]["manual_review"]
            self.assertEqual(manual_summary["completed_case_count"], 2)
            self.assertEqual(manual_summary["reviewer_pass_count"], 1)
            self.assertEqual(manual_summary["reviewer_fail_count"], 1)
            self.assertEqual(
                updated["traces"][2]["manual_review"]["status"],
                "not_selected",
            )
            self.assertTrue((run_dir / "GENERATION_EVALUATION_REPORT.md").exists())

    def test_offline_manual_review_rejects_unselected_case(self) -> None:
        manual = _load_script("apply_generation_manual_review_rejection_test", "apply_generation_manual_review.py")
        traces = [
            _completed_trace("G-PASS", automatic_pass=True, llm_called=True),
            _completed_trace("G-SAFETY", automatic_pass=True, llm_called=False),
        ]
        apply_manual_review_plan(traces)
        report = {
            "status": "completed",
            "run_at": "2026-09-13T00:00:00+08:00",
            "dataset_case_count": len(traces),
            "model": "test-model",
            "retrieval_mode": "dense",
            "min_relevance": 0.60,
            "summary": summarize(traces),
            "failure_examples": [],
            "demo_gate": "未达到门槛",
            "traces": traces,
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            (run_dir / "GENERATION_EVALUATION_TRACES.json").write_text(
                json.dumps(report, ensure_ascii=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "selected case IDs"):
                manual.apply_review_to_run(run_dir, {"G-SAFETY": {"decision": "pass"}})


if __name__ == "__main__":
    unittest.main()
