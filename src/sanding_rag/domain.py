"""第一阶段在模块之间传递的最小、显式数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


REQUIRED_METADATA = ("source", "source_url", "product_id", "document_type", "updated_at")


@dataclass(frozen=True)
class SourceDocument:
    text: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, str | int]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, str | int]
    score: float

    def public_source_dict(self) -> dict[str, Any]:
        """Return document provenance that is safe to show to an end user.

        ``product_id`` and chunk identifiers are implementation metadata.  They
        remain on the retrieved chunk for routing and evaluation, but must never
        cross the public answer boundary.
        """
        return {
            "source": self.metadata["source"],
            "source_url": self.metadata["source_url"],
            "document_type": self.metadata["document_type"],
            "updated_at": self.metadata["updated_at"],
        }


@dataclass(frozen=True)
class AnswerPayload:
    answer: str
    sources: list[dict[str, Any]]
    handoff_required: bool
    handoff_reason: str | None = None
    # These fields are deliberately not serialized by ``to_dict``.  They let
    # local evaluators prove source/product binding without exposing internal
    # identifiers to callers of the CLI or a future user-facing API.
    internal_used_source_ids: list[str] = field(default_factory=list, repr=False)
    internal_returned_product_ids: list[str] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "handoff_required": self.handoff_required,
            "handoff_reason": self.handoff_reason,
        }
