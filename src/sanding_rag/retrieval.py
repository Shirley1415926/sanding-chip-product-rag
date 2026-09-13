"""Configurable, deterministic retrieval adapters for the catalog RAG service.

Dense vectors remain the production default.  BM25 is deliberately small and
self-contained so its Chinese tokenisation rules are visible and reproducible.
The hybrid implementation uses reciprocal-rank fusion (RRF): it combines rank
positions only and never adds incomparable Dense and BM25 raw scores.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Protocol, Sequence

from .chroma_store import ChromaStore
from .domain import RetrievedChunk
from .embedding import EmbeddingProvider


class Retriever(Protocol):
    """Minimal retrieval seam used by answer generation and offline evaluation."""

    def query(self, question: str, top_k: int) -> list[RetrievedChunk]:
        """Return ranked evidence with a 0–1 confidence for relevance gating."""


class DenseRetriever:
    """Existing BGE + Chroma cosine retrieval, exposed behind ``Retriever``."""

    def __init__(self, embedder: EmbeddingProvider, store: ChromaStore) -> None:
        self._embedder = embedder
        self._store = store

    def query(self, question: str, top_k: int) -> list[RetrievedChunk]:
        return self._store.query(self._embedder.embed_query(question), top_k)


class ChineseCatalogTokenizer:
    """Deterministic tokenizer for a compact Chinese catalog corpus.

    It retains complete numeric specifications (for example ``80g/袋`` and
    ``10件起订``), their component number/unit tokens, Latin identifiers, CJK
    spans, and CJK character n-grams.  This gives exact product names and
    catalogue specifications strong lexical signals without a non-deterministic
    third-party Chinese segmenter.
    """

    _specification = re.compile(
        r"(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>kg|g|ml|l|件|头|袋|箱|罐|瓶|杯)"
        r"(?:\s*/\s*(?P<per_unit>件|头|袋|箱|罐|瓶|杯))?\s*(?P<minimum>起订)?",
        re.IGNORECASE,
    )
    _cjk_span = re.compile(r"[\u4e00-\u9fff]+")
    _latin_or_number = re.compile(r"[a-z]+[a-z0-9_-]*|\d+(?:\.\d+)?", re.IGNORECASE)

    def tokenize(self, text: str) -> list[str]:
        normalized = re.sub(r"\s+", "", text.lower())
        tokens: list[str] = []

        for match in self._specification.finditer(normalized):
            full = match.group(0)
            number = match.group("number")
            unit = match.group("unit")
            tokens.extend((full, f"{number}{unit}", number, unit))
            if per_unit := match.group("per_unit"):
                tokens.append(f"/{per_unit}")
            if match.group("minimum"):
                tokens.append("起订")

        for span in self._cjk_span.findall(normalized):
            tokens.append(span)
            if len(span) == 1:
                continue
            for ngram_size in (1, 2, 3, 4):
                if len(span) < ngram_size:
                    continue
                tokens.extend(
                    span[index : index + ngram_size]
                    for index in range(len(span) - ngram_size + 1)
                )

        tokens.extend(self._latin_or_number.findall(normalized))
        return tokens


class BM25Retriever:
    """In-memory BM25 over imported chunks with explicit score calibration.

    BM25 raw scores are used only for lexical ranking.  They are mapped to a
    bounded confidence before the shared ``MIN_RELEVANCE`` gate; this prevents a
    lexical raw score from being compared directly with a cosine score.
    """

    def __init__(
        self,
        chunks: Sequence[RetrievedChunk],
        tokenizer: ChineseCatalogTokenizer | None = None,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        confidence_scale: float = 1.0,
    ) -> None:
        if k1 <= 0 or not 0 <= b <= 1 or confidence_scale <= 0:
            raise ValueError("BM25 parameters must be positive and b must be within [0, 1]")
        self._chunks = tuple(sorted(chunks, key=lambda chunk: chunk.chunk_id))
        self._tokenizer = tokenizer or ChineseCatalogTokenizer()
        self._k1 = k1
        self._b = b
        self._confidence_scale = confidence_scale
        self._term_frequencies: list[Counter[str]] = []
        self._document_lengths: list[int] = []
        document_frequency: Counter[str] = Counter()

        for chunk in self._chunks:
            product_name = str(chunk.metadata.get("product_name", ""))
            terms = self._tokenizer.tokenize(f"{product_name}\n{chunk.text}")
            term_frequency = Counter(terms)
            self._term_frequencies.append(term_frequency)
            self._document_lengths.append(len(terms))
            document_frequency.update(term_frequency.keys())

        self._document_frequency = document_frequency
        self._average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths) if self._document_lengths else 0.0
        )

    @classmethod
    def from_store(cls, store: ChromaStore) -> "BM25Retriever":
        return cls(store.all_chunks())

    def query(self, question: str, top_k: int) -> list[RetrievedChunk]:
        if top_k <= 0 or not self._chunks:
            return []
        query_terms = set(self._tokenizer.tokenize(question))
        if not query_terms:
            return []

        scores = [self._raw_score(query_terms, index) for index in range(len(self._chunks))]
        ranked = sorted(
            (
                (raw_score, chunk)
                for raw_score, chunk in zip(scores, self._chunks, strict=True)
                if raw_score > 0
            ),
            key=lambda item: (-item[0], item[1].chunk_id),
        )[:top_k]
        return [
            replace(chunk, score=self._confidence(raw_score))
            for raw_score, chunk in ranked
        ]

    def _raw_score(self, query_terms: set[str], document_index: int) -> float:
        if not self._average_document_length:
            return 0.0
        frequency = self._term_frequencies[document_index]
        length = self._document_lengths[document_index]
        score = 0.0
        document_count = len(self._chunks)
        for term in query_terms:
            term_frequency = frequency.get(term, 0)
            if not term_frequency:
                continue
            document_frequency = self._document_frequency[term]
            inverse_document_frequency = math.log(
                1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = term_frequency + self._k1 * (
                1 - self._b + self._b * length / self._average_document_length
            )
            score += inverse_document_frequency * term_frequency * (self._k1 + 1) / denominator
        return score

    def _confidence(self, raw_score: float) -> float:
        return raw_score / (raw_score + self._confidence_scale)


class HybridRRFRetriever:
    """Rank-only reciprocal-rank fusion of Dense and BM25 candidates."""

    def __init__(
        self,
        dense: Retriever,
        bm25: Retriever,
        *,
        rrf_k: int = 60,
        candidate_depth: int = 12,
    ) -> None:
        if rrf_k <= 0 or candidate_depth <= 0:
            raise ValueError("rrf_k and candidate_depth must be positive")
        self._dense = dense
        self._bm25 = bm25
        self._rrf_k = rrf_k
        self._candidate_depth = candidate_depth

    def query(self, question: str, top_k: int) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        candidate_depth = max(top_k, self._candidate_depth)
        ranked_lists = (
            self._dense.query(question, candidate_depth),
            self._bm25.query(question, candidate_depth),
        )
        rrf_scores: defaultdict[str, float] = defaultdict(float)
        best_confidence: dict[str, float] = {}
        best_match: dict[str, RetrievedChunk] = {}

        for ranked_matches in ranked_lists:
            for rank, match in enumerate(ranked_matches, start=1):
                # The only fusion calculation: rank position, never a raw score.
                rrf_scores[match.chunk_id] += 1.0 / (self._rrf_k + rank)
                best_confidence[match.chunk_id] = max(best_confidence.get(match.chunk_id, 0.0), match.score)
                best_match.setdefault(match.chunk_id, match)

        fused = sorted(
            rrf_scores,
            key=lambda chunk_id: (-rrf_scores[chunk_id], chunk_id),
        )[:top_k]
        return [replace(best_match[chunk_id], score=best_confidence[chunk_id]) for chunk_id in fused]
