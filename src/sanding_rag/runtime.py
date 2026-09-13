"""Explicit composition root: only this module chooses concrete external providers."""

from __future__ import annotations

from .chroma_store import ChromaStore
from .config import Settings
from .embedding import EmbeddingProvider, SentenceTransformerEmbedder
from .llm import LLMProvider, OpenAICompatibleLLM
from .retrieval import BM25Retriever, DenseRetriever, HybridRRFRetriever, Retriever


def make_production_embedder(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider != "sentence_transformers":
        raise ValueError(
            "Phase 1 production embedding provider must be 'sentence_transformers'; "
            "the hashing provider is test-only."
        )
    return SentenceTransformerEmbedder(settings.embedding_model)


def make_production_llm(settings: Settings) -> LLMProvider:
    if settings.llm_provider != "openai_compatible":
        raise ValueError("Phase 1 LLM provider must be 'openai_compatible'")
    return OpenAICompatibleLLM(
        api_base=settings.llm_api_base,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )


class LazyProductionLLM:
    """Defer LLM setup so safety/no-evidence handoffs never need an API key."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._delegate: LLMProvider | None = None

    def answer(self, question: str, context: str, handoff_message: str) -> str:
        if self._delegate is None:
            self._delegate = make_production_llm(self._settings)
        return self._delegate.answer(question, context, handoff_message)


def make_store(settings: Settings) -> ChromaStore:
    return ChromaStore(settings.chroma_path, settings.collection_name)


def make_retriever(
    settings: Settings,
    embedder: EmbeddingProvider,
    store: ChromaStore,
) -> Retriever:
    """Choose an explicit retrieval mode; production default remains Dense."""
    dense = DenseRetriever(embedder, store)
    if settings.retrieval_mode == "dense":
        return dense
    bm25 = BM25Retriever.from_store(store)
    if settings.retrieval_mode == "bm25":
        return bm25
    if settings.retrieval_mode == "hybrid_rrf":
        return HybridRRFRetriever(
            dense,
            bm25,
            rrf_k=settings.rrf_k,
            candidate_depth=settings.rrf_candidate_depth,
        )
    raise ValueError(f"Unsupported retrieval mode: {settings.retrieval_mode}")
