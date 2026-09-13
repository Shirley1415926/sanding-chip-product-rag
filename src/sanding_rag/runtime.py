"""Explicit composition root: only this module chooses concrete external providers."""

from __future__ import annotations

from .chroma_store import ChromaStore
from .config import Settings
from .embedding import EmbeddingProvider, SentenceTransformerEmbedder
from .llm import LLMProvider, OpenAICompatibleLLM


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
