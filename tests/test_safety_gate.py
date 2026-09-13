from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sanding_rag.llm import ContextEchoLLM
from sanding_rag.service import HANDOFF_MESSAGE, RAGAnswerService, SafetyGate


class _NoopEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return []


class _EmptyStore:
    def query(self, vector: list[float], top_k: int) -> list[object]:
        return []


class SafetyGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = SafetyGate()

    def test_inventory_question_is_handed_off(self) -> None:
        self.assertEqual(self.gate.blocked_reason("这个型号现在有库存吗？"), "inventory")

    def test_final_quote_question_is_handed_off(self) -> None:
        self.assertEqual(self.gate.blocked_reason("请给我最终报价"), "final_quote")

    def test_static_spec_question_can_retrieve(self) -> None:
        self.assertIsNone(self.gate.blocked_reason("这个型号的供电范围是多少？"))

    def test_no_evidence_uses_the_fixed_handoff_message(self) -> None:
        payload = RAGAnswerService(
            embedder=_NoopEmbedder(),
            store=_EmptyStore(),  # type: ignore[arg-type]
            llm=ContextEchoLLM(),
        ).ask("不存在的材料标准是什么？")
        self.assertTrue(payload.handoff_required)
        self.assertEqual(payload.answer, HANDOFF_MESSAGE)
        self.assertEqual(payload.sources, [])


if __name__ == "__main__":
    unittest.main()
