from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sanding_rag.domain import RetrievedChunk
from sanding_rag.llm import SYSTEM_PROMPT
from sanding_rag.service import RAGAnswerService


def _chunk(
    *,
    product_id: str,
    product_name: str,
    source: str,
    text: str,
    score: float = 0.95,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{product_id}",
        text=text,
        metadata={
            "source": source,
            "source_url": "https://txs.wyfdev.com/product/public-detail/",
            "product_id": product_id,
            "product_name": product_name,
            "document_type": "product",
            "updated_at": "2026-09-13",
        },
        score=score,
    )


class _NoopEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.0]


class _FixedStore:
    def __init__(self, matches: list[RetrievedChunk]) -> None:
        self.matches = matches

    def query(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        return self.matches[:top_k]


class _RecordingLLM:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls = 0
        self.context = ""

    def answer(self, question: str, context: str, handoff_message: str) -> str:
        self.calls += 1
        self.context = context
        return json.dumps(self.response, ensure_ascii=False)


class VerifiedSourceBindingTests(unittest.TestCase):
    def _service(self, matches: list[RetrievedChunk], llm: _RecordingLLM) -> RAGAnswerService:
        return RAGAnswerService(
            embedder=_NoopEmbedder(),
            store=_FixedStore(matches),  # type: ignore[arg-type]
            llm=llm,  # type: ignore[arg-type]
            min_relevance=0.60,
        )

    def test_product_id_never_enters_prompt_or_public_payload(self) -> None:
        cork = _chunk(
            product_id="marketplace-cork-painting-sanfangqixiang",
            product_name="三坊七巷主题福州软木画",
            source="catalog/cork.md",
            text="# 三坊七巷主题福州软木画\n- 展示价：¥328。",
        )
        llm = _RecordingLLM({"answer": "三坊七巷主题福州软木画的页面展示价为 ¥328。", "used_source_ids": ["S1"]})
        payload = self._service([cork], llm).ask("三坊七巷主题福州软木画的展示价是多少？")

        self.assertFalse(payload.handoff_required)
        self.assertIn("商品名称：三坊七巷主题福州软木画", llm.context)
        self.assertNotIn("product_id", llm.context)
        self.assertNotIn("product_id", SYSTEM_PROMPT)
        self.assertNotIn("marketplace-cork-painting-sanfangqixiang", llm.context)
        self.assertNotIn("chunk-marketplace-cork-painting-sanfangqixiang", llm.context)
        self.assertNotIn("score", llm.context)
        self.assertNotIn("marketplace-cork-painting-sanfangqixiang", payload.answer)
        public_payload = json.dumps(payload.to_dict(), ensure_ascii=False)
        self.assertNotIn("product_id", public_payload)
        self.assertNotIn("marketplace-cork-painting-sanfangqixiang", public_payload)

    def test_only_model_cited_candidate_is_returned_to_user(self) -> None:
        target = _chunk(
            product_id="fungus",
            product_name="古田竹荪干货",
            source="catalog/fungus.md",
            text="# 古田竹荪干货\n- 规格：80g/袋。",
        )
        unrelated = _chunk(
            product_id="dragon-plate",
            product_name="景德镇青花龙纹瓷盘",
            source="catalog/dragon-plate.md",
            text="# 景德镇青花龙纹瓷盘\n- 起订量：10 件起订。",
            score=0.90,
        )
        llm = _RecordingLLM({"answer": "古田竹荪干货的规格为 80g/袋。", "used_source_ids": ["S1"]})
        payload = self._service([target, unrelated], llm).ask("福建干货有哪些80g/袋的公开资料？")

        self.assertFalse(payload.handoff_required)
        self.assertEqual([source["source"] for source in payload.sources], ["catalog/fungus.md"])
        self.assertEqual(payload.internal_returned_product_ids, ["fungus"])

    def test_invalid_or_forged_source_id_is_rejected(self) -> None:
        cork = _chunk(
            product_id="cork",
            product_name="三坊七巷主题福州软木画",
            source="catalog/cork.md",
            text="# 三坊七巷主题福州软木画\n- 展示价：¥328。",
        )
        llm = _RecordingLLM({"answer": "展示价为 ¥328。", "used_source_ids": ["S99"]})
        payload = self._service([cork], llm).ask("软木画展示价是多少？")

        self.assertTrue(payload.handoff_required)
        self.assertEqual(payload.handoff_reason, "llm_invalid_source_citation")
        self.assertEqual(payload.sources, [])

    def test_model_disclosure_of_product_id_is_rejected_before_public_output(self) -> None:
        cork = _chunk(
            product_id="marketplace-cork-painting-sanfangqixiang",
            product_name="三坊七巷主题福州软木画",
            source="catalog/cork.md",
            text="# 三坊七巷主题福州软木画\n- 展示价：¥328。",
        )
        llm = _RecordingLLM(
            {
                "answer": "product_id=marketplace-cork-painting-sanfangqixiang，展示价为 ¥328。",
                "used_source_ids": ["S1"],
            }
        )
        payload = self._service([cork], llm).ask("软木画展示价是多少？")

        self.assertTrue(payload.handoff_required)
        self.assertEqual(payload.handoff_reason, "llm_internal_identifier_disclosure")
        self.assertNotIn("marketplace-cork-painting-sanfangqixiang", payload.to_dict()["answer"])

    def test_g11_style_comparison_keeps_two_product_contexts_and_two_sources(self) -> None:
        cork = _chunk(
            product_id="marketplace-cork-painting-sanfangqixiang",
            product_name="三坊七巷主题福州软木画",
            source="catalog/marketplace-cork-painting-sanfangqixiang.md",
            text="# 三坊七巷主题福州软木画\n- 页面展示价：¥328。",
        )
        tea = _chunk(
            product_id="marketplace-oolong-tea-gift-box",
            product_name="乌龙茶叶礼盒",
            source="catalog/marketplace-oolong-tea-gift-box.md",
            text="# 乌龙茶叶礼盒\n- 页面展示价：¥128。",
            score=0.94,
        )
        llm = _RecordingLLM(
            {
                "answer": "软木画的页面展示价为 ¥328；乌龙茶叶礼盒的页面展示价为 ¥128。",
                "used_source_ids": ["S1", "S2"],
            }
        )
        payload = self._service([cork, tea], llm).ask(
            "软木画和乌龙茶叶礼盒各自的公开展示价是多少？请分开说明，并明确这是页面展示价。"
        )

        self.assertFalse(payload.handoff_required)
        self.assertIn("[S1]", llm.context)
        self.assertIn("[S2]", llm.context)
        self.assertIn("商品名称：三坊七巷主题福州软木画", llm.context)
        self.assertIn("商品名称：乌龙茶叶礼盒", llm.context)
        self.assertNotIn("product_id", llm.context)
        self.assertNotIn("订单价", payload.answer)
        self.assertNotIn("批量价", payload.answer)
        self.assertNotIn("经销价", payload.answer)
        self.assertIn("只陈述有直接证据支持的完整商品名称与公开展示价", SYSTEM_PROMPT)
        self.assertIn("不要主动补充“不是订单价、批量价、经销价”等否定性说明", SYSTEM_PROMPT)
        self.assertEqual(
            [source["source"] for source in payload.sources],
            [
                "catalog/marketplace-cork-painting-sanfangqixiang.md",
                "catalog/marketplace-oolong-tea-gift-box.md",
            ],
        )
        self.assertEqual(
            payload.internal_returned_product_ids,
            ["marketplace-cork-painting-sanfangqixiang", "marketplace-oolong-tea-gift-box"],
        )

    def test_g13_and_g15_selection_contexts_include_complete_public_product_names(self) -> None:
        fungus = _chunk(
            product_id="marketplace-gutian-bamboo-fungus",
            product_name="古田竹荪干货",
            source="catalog/marketplace-gutian-bamboo-fungus.md",
            text="# 古田竹荪干货\n- 产地：福建古田。\n- 规格：80g/袋。\n- 包装：20袋/箱。",
        )
        fungus_llm = _RecordingLLM(
            {
                "answer": "古田竹荪干货：产自福建古田，规格为80g/袋，包装为20袋/箱。",
                "used_source_ids": ["S1"],
            }
        )
        fungus_payload = self._service([fungus], fungus_llm).ask(
            "我想挑一份福建干制食材礼品，条件是单袋80g且整箱20袋；应看哪款公开资料？"
        )

        dinnerware = _chunk(
            product_id="marketplace-linglong-porcelain-dinnerware",
            product_name="玲珑瓷餐具套装",
            source="catalog/marketplace-linglong-porcelain-dinnerware.md",
            text="# 玲珑瓷餐具套装\n- 56头套装。\n- 微波炉适用。",
        )
        dinnerware_llm = _RecordingLLM(
            {
                "answer": "玲珑瓷餐具套装：公开资料写明微波炉适用，套装数量为56头。",
                "used_source_ids": ["S1"],
            }
        )
        dinnerware_payload = self._service([dinnerware], dinnerware_llm).ask(
            "需要一套公开写明可用于微波炉的景德镇餐具，资料中能确认的套装数量是多少？"
        )

        self.assertFalse(fungus_payload.handoff_required)
        self.assertIn("商品名称：古田竹荪干货", fungus_llm.context)
        self.assertNotIn("marketplace-gutian-bamboo-fungus", fungus_llm.context)
        self.assertFalse(dinnerware_payload.handoff_required)
        self.assertIn("商品名称：玲珑瓷餐具套装", dinnerware_llm.context)
        self.assertNotIn("marketplace-linglong-porcelain-dinnerware", dinnerware_llm.context)

    def test_hard_safety_gate_still_skips_the_llm(self) -> None:
        cork = _chunk(
            product_id="cork",
            product_name="三坊七巷主题福州软木画",
            source="catalog/cork.md",
            text="# 三坊七巷主题福州软木画\n- 展示价：¥328。",
        )
        llm = _RecordingLLM({"answer": "不应被调用", "used_source_ids": ["S1"]})
        payload = self._service([cork], llm).ask("软木画现在还能下单吗？")

        self.assertTrue(payload.handoff_required)
        self.assertEqual(payload.handoff_reason, "inventory")
        self.assertEqual(llm.calls, 0)


if __name__ == "__main__":
    unittest.main()
