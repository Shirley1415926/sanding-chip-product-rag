"""Query orchestration, relevance policy and non-negotiable human-handoff guardrails."""

from __future__ import annotations

import json
import re
from typing import Any

from .chroma_store import ChromaStore
from .domain import AnswerPayload, RetrievedChunk
from .embedding import EmbeddingProvider
from .llm import LLMProvider
from .retrieval import DenseRetriever, Retriever


HANDOFF_MESSAGE = "抱歉，当前知识库没有足够的可核验依据。为避免提供不准确的库存、实时交期或最终报价，请转人工客服确认。"


class SafetyGate:
    """Risk categories are blocked before retrieval so static documents cannot become promises."""

    _patterns: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("inventory", re.compile(r"库存|现货|剩余.*(货|量)|有货.*吗|今天能发|当天发|是否.*有货|还有.*(吗|货)|还能.*(下单|买)|可以.*(下单|购买)|可.*购买", re.IGNORECASE)),
        ("real_time_lead_time", re.compile(r"实时.*交期|实时.*交货|明天.*(到货|交货)|今天.*(到货|交货)|具体.*(交期|发货)|哪天.*(到货|交货|发货)|发货日期|什么时候发货|出库|发得出|多久.*(到|发)|几天.*(到|发)|到手", re.IGNORECASE)),
        ("non_public_price", re.compile(r"批量.*(价|价格)|经销价|批发价|订单.*(价|价格|报价)|专属采购价|拿货价|渠道价|团购价|优惠.*(多少|价)|便宜.*(多少|吗)|折扣|一箱.*(多少钱|价格|价)|大宗.*(价|价格)", re.IGNORECASE)),
        ("final_quote", re.compile(r"最终报价|最终价格|报价多少|具体价格|单价多少|价格多少", re.IGNORECASE)),
        ("unpublished_food_facts", re.compile(r"保质期|配料|过敏原|营养成分|热量", re.IGNORECASE)),
        ("unpublished_cooperation_rules", re.compile(r"(?:供应商|入驻|经销商|经销).*(?:资质|材料|证件|审核|审批|条件|门槛|流程|区域)", re.IGNORECASE)),
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
        min_relevance: float = 0.60,
        safety_gate: SafetyGate | None = None,
        retriever: Retriever | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._top_k = top_k
        self._min_relevance = min_relevance
        self._safety_gate = safety_gate or SafetyGate()
        self._retriever = retriever or DenseRetriever(embedder, store)

    @property
    def min_relevance(self) -> float:
        """Expose the active threshold for configuration regression tests."""
        return self._min_relevance

    def ask(self, question: str) -> AnswerPayload:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        blocked_reason, matches = self.trace_retrieval(question)
        if blocked_reason:
            return self._handoff(blocked_reason)
        if not matches:
            return self._handoff("no_retrieval_result")
        if matches[0].score < self._min_relevance:
            return self._handoff("below_relevance_threshold")

        matches = self._restrict_to_named_product(question, matches)
        if self._requires_explicit_fact(question, matches):
            return self._handoff("explicit_fact_not_found")

        evidence_by_id = self._evidence_by_id(matches)
        context = self._format_context(evidence_by_id)
        raw_response = self._llm.answer(question, context, HANDOFF_MESSAGE).strip()
        generated = self._parse_cited_response(raw_response)
        if generated is None:
            return self._handoff("llm_invalid_response_format")
        answer, used_source_ids = generated
        if not answer or answer == HANDOFF_MESSAGE:
            return self._handoff("llm_insufficient_evidence")
        if self._contains_internal_identifier(answer, evidence_by_id):
            return self._handoff("llm_internal_identifier_disclosure")
        if not used_source_ids or any(source_id not in evidence_by_id for source_id in used_source_ids):
            return self._handoff("llm_invalid_source_citation")
        sources, returned_product_ids = self._returned_sources(used_source_ids, evidence_by_id)
        if not sources:
            return self._handoff("llm_invalid_source_citation")
        return AnswerPayload(
            answer=answer,
            sources=sources,
            handoff_required=False,
            internal_used_source_ids=used_source_ids,
            internal_returned_product_ids=returned_product_ids,
        )

    def trace_retrieval(self, question: str) -> tuple[str | None, list[RetrievedChunk]]:
        """Return the pre-answer safety decision and raw Top-K evidence.

        Evaluation uses this read-only seam to record what retrieval saw without
        asking an LLM to expose its hidden reasoning. A hard safety hit returns
        no evidence because the normal request path must not retrieve for it.
        """
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        blocked_reason = self._safety_gate.blocked_reason(question)
        if blocked_reason:
            return blocked_reason, []
        return None, self._retriever.query(question, self._top_k)

    @staticmethod
    def _evidence_by_id(matches: list[RetrievedChunk]) -> dict[str, RetrievedChunk]:
        """Assign deterministic, request-scoped IDs without exposing metadata IDs."""
        return {f"S{index}": match for index, match in enumerate(matches, start=1)}

    @staticmethod
    def _format_context(evidence_by_id: dict[str, RetrievedChunk]) -> str:
        """Expose only public evidence labels, never routing or retrieval internals."""
        return "\n\n".join(
            "\n".join(
                (
                    f"[{source_id}]",
                    f"商品名称：{str(match.metadata['product_name'])}",
                    "公开证据：",
                    match.text,
                )
            )
            for source_id, match in evidence_by_id.items()
        )

    @staticmethod
    def _parse_cited_response(raw_response: str) -> tuple[str, list[str]] | None:
        """Accept only the small response contract required for source binding."""
        try:
            decoded = json.loads(raw_response)
        except json.JSONDecodeError:
            return None
        if not isinstance(decoded, dict) or set(decoded) != {"answer", "used_source_ids"}:
            return None
        answer = decoded.get("answer")
        source_ids = decoded.get("used_source_ids")
        if not isinstance(answer, str) or not isinstance(source_ids, list):
            return None
        if any(not isinstance(source_id, str) or not re.fullmatch(r"S[1-9]\d*", source_id) for source_id in source_ids):
            return None
        if len(set(source_ids)) != len(source_ids):
            return None
        return answer.strip(), source_ids

    @staticmethod
    def _contains_internal_identifier(answer: str, evidence_by_id: dict[str, RetrievedChunk]) -> bool:
        """Fail closed if a model tries to reveal routing identifiers to a user."""
        if re.search(r"(?i)product[_ -]?id", answer) or re.search(r"S[1-9]\d*", answer):
            return True
        answer_normalized = answer.lower()
        return any(
            str(match.metadata["product_id"]).lower() in answer_normalized
            for match in evidence_by_id.values()
        )

    @staticmethod
    def _restrict_to_named_product(question: str, matches: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Keep explicit product and platform questions inside their evidence scope.

        Catalog documents provide ``product_name`` as provenance metadata. When a
        user explicitly names a product, only chunks for that product are passed
        to the answer model. General policy chunks are intentionally not added
        here because this narrow guard is for product-specific attributes. A clear
        cooperation query similarly prefers policy chunks so product neighbors do
        not appear as unrelated sources in a platform answer.
        """
        candidate_names = {
            str(match.metadata.get("product_name", ""))
            for match in matches
            if str(match.metadata.get("product_name", ""))
        }
        named_matches = [
            match
            for match in matches
            if (product_name := str(match.metadata.get("product_name", "")))
            and RAGAnswerService._product_name_appears_in_question(
                product_name, question, candidate_names
            )
        ]
        if named_matches:
            return named_matches
        if re.search(r"供应商|经销商|批量采购|专属采购|入驻", question):
            policy_matches = [
                match for match in matches if match.metadata.get("document_type") == "policy"
            ]
            if policy_matches:
                return policy_matches
        return matches

    @staticmethod
    def _product_name_appears_in_question(
        product_name: str, question: str, candidate_names: set[str]
    ) -> bool:
        """Recognize an explicit full name or a distinctive three-character name fragment.

        This lets a comparison such as “软木画和乌龙茶叶礼盒” retain both
        product contexts while leaving non-name selection questions to normal
        retrieval.  Two-character generic words such as “礼盒” are not enough.
        """
        compact_question = re.sub(r"\s+", "", question)
        compact_name = re.sub(r"\s+", "", product_name)
        if compact_name and compact_name in compact_question:
            return True
        for index in range(max(0, len(compact_name) - 2)):
            fragment = compact_name[index : index + 3]
            if fragment not in compact_question:
                continue
            matches_this_name_only = sum(fragment in re.sub(r"\s+", "", name) for name in candidate_names)
            if matches_this_name_only == 1:
                return True
        return False

    @staticmethod
    def _requires_explicit_fact(question: str, matches: list[RetrievedChunk]) -> bool:
        """Require literal evidence for facts that cannot be safely inferred."""
        explicit_terms = (
            "微波炉",
            "保质期",
            "配料",
            "过敏原",
            "营养成分",
            "热量",
            "材质",
            "骨瓷",
            "洗碗机",
            "认证",
            "食品安全",
            "出口标准",
            "无添加",
            "有机",
            "定制范围",
            "刻字",
            "logo",
            "56 头",
        )
        for term in explicit_terms:
            if term in question and not any(term in match.text for match in matches):
                return True
        return False

    @staticmethod
    def _returned_sources(
        used_source_ids: list[str], evidence_by_id: dict[str, RetrievedChunk]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Map only verified model citations to public, deduplicated provenance."""
        seen: set[str] = set()
        sources: list[dict[str, Any]] = []
        product_ids: list[str] = []
        for source_id in used_source_ids:
            match = evidence_by_id[source_id]
            source = str(match.metadata["source"])
            if source not in seen:
                sources.append(match.public_source_dict())
                product_ids.append(str(match.metadata["product_id"]))
                seen.add(source)
        return sources, product_ids

    @staticmethod
    def _handoff(reason: str) -> AnswerPayload:
        return AnswerPayload(
            answer=HANDOFF_MESSAGE,
            sources=[],
            handoff_required=True,
            handoff_reason=reason,
        )
