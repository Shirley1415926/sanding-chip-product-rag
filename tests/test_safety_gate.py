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


class _FailIfCalledLLM:
    def __init__(self) -> None:
        self.calls = 0

    def answer(self, question: str, context: str, handoff_message: str) -> str:
        self.calls += 1
        raise AssertionError("hard safety gate must return before the LLM boundary")


class SafetyGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = SafetyGate()

    def test_inventory_question_is_handed_off(self) -> None:
        self.assertEqual(self.gate.blocked_reason("这个型号现在有库存吗？"), "inventory")

    def test_final_quote_question_is_handed_off(self) -> None:
        self.assertEqual(self.gate.blocked_reason("请给我最终报价"), "final_quote")

    def test_keyword_evasion_for_stock_price_and_lead_time_is_handed_off(self) -> None:
        self.assertEqual(self.gate.blocked_reason("这个商品还能下单吗？"), "inventory")
        self.assertEqual(self.gate.blocked_reason("竹荪这周能出库吗？"), "real_time_lead_time")
        self.assertEqual(self.gate.blocked_reason("买一箱能便宜多少？"), "non_public_price")

    def test_unpublished_cooperation_rules_are_handed_off(self) -> None:
        self.assertEqual(
            self.gate.blocked_reason("供应商入驻要提供哪些资质，审核多久？"),
            "unpublished_cooperation_rules",
        )

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

    def test_hard_safety_gate_never_calls_llm(self) -> None:
        llm = _FailIfCalledLLM()
        payload = RAGAnswerService(
            embedder=_NoopEmbedder(),
            store=_EmptyStore(),  # type: ignore[arg-type]
            llm=llm,  # type: ignore[arg-type]
        ).ask("潮汕沙茶酱现在有多少库存，今天能发吗？")
        self.assertTrue(payload.handoff_required)
        self.assertEqual(payload.handoff_reason, "inventory")
        self.assertEqual(llm.calls, 0)


if __name__ == "__main__":
    unittest.main()
