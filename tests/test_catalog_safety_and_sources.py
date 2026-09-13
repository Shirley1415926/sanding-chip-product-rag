from __future__ import annotations

import sys
import unittest
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sanding_rag.domain import RetrievedChunk
from sanding_rag.llm import ContextEchoLLM
from sanding_rag.markdown_loader import load_markdown_many
from sanding_rag.service import RAGAnswerService


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _NoopEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.0]


class _FixedStore:
    def __init__(self, matches: list[RetrievedChunk]) -> None:
        self.matches = matches

    def query(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        return self.matches[:top_k]


class CatalogSourceAndSafetyTests(unittest.TestCase):
    def test_catalog_sources_are_public_https_detail_or_cooperation_pages(self) -> None:
        documents = load_markdown_many(PROJECT_ROOT / "data" / "sample")
        products = [document for document in documents if document.metadata["document_type"] == "product"]
        policies = [document for document in documents if document.metadata["document_type"] == "policy"]
        self.assertEqual(len(products), 10)
        self.assertEqual(len(policies), 1)
        for document in products:
            parsed = urlparse(document.metadata["source_url"])
            self.assertEqual(parsed.scheme, "https")
            self.assertEqual(parsed.netloc, "txs.wyfdev.com")
            self.assertTrue(parsed.path.startswith("/product/"))
        self.assertEqual(
            policies[0].metadata["source_url"],
            "https://txs.wyfdev.com/cooperation-apply/",
        )

    def test_cross_product_material_claim_is_handed_off_when_target_has_no_evidence(self) -> None:
        dragon_plate = RetrievedChunk(
            chunk_id="dragon-plate",
            text="# 景德镇青花龙纹瓷盘\n\n- 起订量：10 件起订。",
            metadata={
                "source": "catalog/marketplace-jingdezhen-blue-white-dragon-plate.md",
                "source_url": "https://txs.wyfdev.com/product/example/",
                "product_id": "marketplace-jingdezhen-blue-white-dragon-plate",
                "product_name": "景德镇青花龙纹瓷盘",
                "document_type": "product",
                "updated_at": "2026-09-13",
            },
            score=0.99,
        )
        payload = RAGAnswerService(
            embedder=_NoopEmbedder(),
            store=_FixedStore([dragon_plate]),  # type: ignore[arg-type]
            llm=ContextEchoLLM(),
            min_relevance=0.45,
        ).ask("景德镇青花龙纹瓷盘是骨瓷材质吗？")
        self.assertTrue(payload.handoff_required)
        self.assertEqual(payload.handoff_reason, "explicit_fact_not_found")

    def test_platform_query_prefers_policy_source_over_product_neighbors(self) -> None:
        policy = RetrievedChunk(
            chunk_id="policy",
            text="# 平台合作说明\n\n- 平台提供供应商入驻入口。",
            metadata={
                "source": "catalog/platform-cooperation.md",
                "source_url": "https://txs.wyfdev.com/cooperation-apply/",
                "product_id": "GENERAL",
                "product_name": "平台合作说明",
                "document_type": "policy",
                "updated_at": "2026-09-13",
            },
            score=0.99,
        )
        product = RetrievedChunk(
            chunk_id="product",
            text="# 景德镇青花龙纹瓷盘\n\n- 起订量：10 件起订。",
            metadata={
                "source": "catalog/marketplace-jingdezhen-blue-white-dragon-plate.md",
                "source_url": "https://txs.wyfdev.com/product/example/",
                "product_id": "marketplace-jingdezhen-blue-white-dragon-plate",
                "product_name": "景德镇青花龙纹瓷盘",
                "document_type": "product",
                "updated_at": "2026-09-13",
            },
            score=0.95,
        )
        payload = RAGAnswerService(
            embedder=_NoopEmbedder(),
            store=_FixedStore([policy, product]),  # type: ignore[arg-type]
            llm=ContextEchoLLM(),
            min_relevance=0.45,
        ).ask("平台有供应商入驻入口吗？")
        self.assertFalse(payload.handoff_required)
        self.assertEqual(payload.internal_returned_product_ids, ["GENERAL"])
        self.assertEqual([source["source"] for source in payload.sources], ["catalog/platform-cooperation.md"])
        self.assertFalse(any("product_id" in source for source in payload.sources))


if __name__ == "__main__":
    unittest.main()
