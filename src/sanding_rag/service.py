"""Query orchestration, relevance policy and non-negotiable human-handoff guardrails."""

from __future__ import annotations

import re

from .chroma_store import ChromaStore
from .domain import AnswerPayload, RetrievedChunk
from .embedding import EmbeddingProvider
from .llm import LLMProvider


HANDOFF_MESSAGE = "抱歉，当前知识库没有足够的可核验依据。为避免提供不准确的库存、实时交期或最终报价，请转人工客服确认。"


class SafetyGate:
    """Risk categories are blocked before retrieval so static documents cannot become promises."""

    _patterns: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("inventory", re.compile(r"库存|现货|剩余.*(货|量)|有货.*吗|今天能发|当天发", re.IGNORECASE)),
        ("real_time_lead_time", re.compile(r"实时.*交期|实时.*交货|明天.*(到货|交货)|今天.*(到货|交货)|具体.*(交期|发货)|哪天.*(到货|交货|发货)|发货日期|什么时候发货", re.IGNORECASE)),
        ("non_public_price", re.compile(r"批量.*(价|价格)|经销价|批发价|订单.*(价|价格|报价)|专属采购价", re.IGNORECASE)),
        ("final_quote", re.compile(r"最终报价|最终价格|报价多少|具体价格|单价多少|价格多少", re.IGNORECASE)),
        ("unpublished_food_facts", re.compile(r"保质期|配料|过敏原|营养成分|热量", re.IGNORECASE)),
    )

    def blocked_reason(self, question: str) -> str | None:
        for reason, pattern in self._patterns:
            if pattern.search(question):
                return reason
        return None


class RAGAnswerService:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: ChromaStore,
        llm: LLMProvider,
        top_k: int = 4,
        min_relevance: float = 0.45,
        safety_gate: SafetyGate | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._top_k = top_k
        self._min_relevance = min_relevance
        self._safety_gate = safety_gate or SafetyGate()

    def ask(self, question: str) -> AnswerPayload:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        blocked_reason = self._safety_gate.blocked_reason(question)
        if blocked_reason:
            return self._handoff(blocked_reason)

        matches = self._store.query(self._embedder.embed_query(question), self._top_k)
        if not matches:
            return self._handoff("no_retrieval_result")
        if matches[0].score < self._min_relevance:
            return self._handoff("below_relevance_threshold")

        matches = self._restrict_to_named_product(question, matches)
        if self._requires_explicit_fact(question, matches):
            return self._handoff("explicit_fact_not_found")

        context = self._format_context(matches)
        answer = self._llm.answer(question, context, HANDOFF_MESSAGE).strip()
        if not answer or answer == HANDOFF_MESSAGE:
            return self._handoff("llm_insufficient_evidence")
        return AnswerPayload(
            answer=answer,
            sources=self._deduplicated_sources(matches),
            handoff_required=False,
        )

    @staticmethod
    def _format_context(matches: list[RetrievedChunk]) -> str:
        return "\n\n".join(
            "\n".join(
                (
                    f"[证据 {index}] source={match.metadata['source']}",
                    f"product_id={match.metadata['product_id']}; type={match.metadata['document_type']}; updated_at={match.metadata['updated_at']}",
                    match.text,
                )
            )
            for index, match in enumerate(matches, start=1)
        )

    @staticmethod
    def _restrict_to_named_product(question: str, matches: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Do not answer Product A with a fact retrieved from Product B.

        Catalog documents provide ``product_name`` as provenance metadata. When a
        user explicitly names a product, only chunks for that product are passed
        to the answer model. General policy chunks are intentionally not added
        here because this narrow guard is for product-specific attributes.
        """
        named_matches = [
            match
            for match in matches
            if (product_name := str(match.metadata.get("product_name", ""))) and product_name in question
        ]
        return named_matches or matches

    @staticmethod
    def _requires_explicit_fact(question: str, matches: list[RetrievedChunk]) -> bool:
        """Require literal evidence for facts that cannot be safely inferred."""
        explicit_terms = ("微波炉", "保质期", "配料", "过敏原", "营养成分", "热量")
        for term in explicit_terms:
            if term in question and not any(term in match.text for match in matches):
                return True
        return False

    @staticmethod
    def _deduplicated_sources(matches: list[RetrievedChunk]) -> list[dict[str, object]]:
        seen: set[str] = set()
        sources: list[dict[str, object]] = []
        for match in matches:
            # A caller normally needs a document-level provenance list, not four
            # duplicate rows just because four nearby chunks came from one file.
            source = str(match.metadata["source"])
            if source not in seen:
                sources.append(match.source_dict())
                seen.add(source)
        return sources

    @staticmethod
    def _handoff(reason: str) -> AnswerPayload:
        return AnswerPayload(
            answer=HANDOFF_MESSAGE,
            sources=[],
            handoff_required=True,
            handoff_reason=reason,
        )
