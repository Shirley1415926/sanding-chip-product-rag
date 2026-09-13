from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sanding_rag.config import Settings
from sanding_rag.domain import RetrievedChunk
from sanding_rag.llm import ContextEchoLLM
from sanding_rag.retrieval import BM25Retriever, ChineseCatalogTokenizer, DenseRetriever, HybridRRFRetriever
from sanding_rag.runtime import make_retriever
from sanding_rag.service import RAGAnswerService


def _chunk(chunk_id: str, product_id: str, text: str, score: float = 0.95) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        metadata={
            "source": f"catalog/{product_id}.md",
            "source_url": f"https://txs.wyfdev.com/product/{product_id}/",
            "product_id": product_id,
            "product_name": product_id,
            "document_type": "product",
            "updated_at": "2026-09-13",
        },
        score=score,
    )


class _NoopEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.0]


class _NoopStore:
    def query(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        return []


class _StaticRetriever:
    def __init__(self, matches: list[RetrievedChunk]) -> None:
        self.matches = matches
        self.calls = 0

    def query(self, question: str, top_k: int) -> list[RetrievedChunk]:
        self.calls += 1
        return self.matches[:top_k]


class RetrievalModeTests(unittest.TestCase):
    def test_chinese_tokenizer_preserves_catalog_specifications(self) -> None:
        tokens = ChineseCatalogTokenizer().tokenize("56头骨瓷套装，300g/瓶，80g/袋，10件起订")
        for expected in ("56头", "300g/瓶", "80g/袋", "10件起订", "56", "300", "80", "10"):
            self.assertIn(expected, tokens)

    def test_bm25_prefers_exact_numeric_specification(self) -> None:
        retriever = BM25Retriever(
            [
                _chunk("tea", "tea", "福建茶产区，100g/罐，双罐礼盒。"),
                _chunk("fungus", "fungus", "福建古田，干货，80g/袋，20袋/箱。"),
            ]
        )
        matches = retriever.query("福建干货80g/袋", 2)
        self.assertEqual(matches[0].metadata["product_id"], "fungus")
        self.assertGreaterEqual(matches[0].score, 0.60)

    def test_rrf_ranking_is_unchanged_when_raw_scores_change(self) -> None:
        a, b, c = (_chunk("a", "a", "a"), _chunk("b", "b", "b"), _chunk("c", "c", "c"))
        first = HybridRRFRetriever(
            _StaticRetriever([_chunk("a", "a", "a", 0.99), _chunk("b", "b", "b", 0.01), c]),
            _StaticRetriever([_chunk("b", "b", "b", 0.02), c, _chunk("a", "a", "a", 0.98)]),
            rrf_k=60,
            candidate_depth=3,
        ).query("query", 3)
        second = HybridRRFRetriever(
            _StaticRetriever([_chunk("a", "a", "a", 0.01), _chunk("b", "b", "b", 0.99), c]),
            _StaticRetriever([_chunk("b", "b", "b", 0.98), c, _chunk("a", "a", "a", 0.02)]),
            rrf_k=60,
            candidate_depth=3,
        ).query("query", 3)
        self.assertEqual([match.chunk_id for match in first], [match.chunk_id for match in second])
        self.assertEqual(first[0].chunk_id, "b")

    def test_safety_gate_and_threshold_are_identical_for_every_mode_seam(self) -> None:
        low_confidence = _chunk("known", "known", "公开资料", 0.59)
        for mode in ("dense", "bm25", "hybrid_rrf"):
            with self.subTest(mode=mode):
                retriever = _StaticRetriever([low_confidence])
                service = RAGAnswerService(
                    embedder=_NoopEmbedder(),
                    store=_NoopStore(),  # type: ignore[arg-type]
                    llm=ContextEchoLLM(),
                    retriever=retriever,
                    min_relevance=0.60,
                )
                inventory_payload = service.ask("这个商品现在还能下单吗？")
                self.assertTrue(inventory_payload.handoff_required)
                self.assertEqual(inventory_payload.handoff_reason, "inventory")
                self.assertEqual(retriever.calls, 0)
                low_relevance_payload = service.ask("公开资料是什么？")
                self.assertTrue(low_relevance_payload.handoff_required)
                self.assertEqual(low_relevance_payload.handoff_reason, "below_relevance_threshold")
                self.assertEqual(retriever.calls, 1)

    def test_runtime_defaults_to_dense_and_min_relevance_point_sixty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings.from_environment(Path(temporary_dir))
        self.assertEqual(settings.retrieval_mode, "dense")
        self.assertEqual(settings.min_relevance, 0.60)
        self.assertIsInstance(make_retriever(settings, _NoopEmbedder(), _NoopStore()), DenseRetriever)  # type: ignore[arg-type]
        service = RAGAnswerService(
            embedder=_NoopEmbedder(),
            store=_NoopStore(),  # type: ignore[arg-type]
            llm=ContextEchoLLM(),
        )
        self.assertEqual(service.min_relevance, 0.60)


if __name__ == "__main__":
    unittest.main()
