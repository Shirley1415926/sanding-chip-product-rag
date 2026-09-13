"""第一阶段在模块之间传递的最小、显式数据契约。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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

    def source_dict(self) -> dict[str, Any]:
        """Return only fields that can be shown to an operator or caller."""
        return {
            "source": self.metadata["source"],
            "source_url": self.metadata["source_url"],
            "product_id": self.metadata["product_id"],
            "document_type": self.metadata["document_type"],
            "updated_at": self.metadata["updated_at"],
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
        }


@dataclass(frozen=True)
class AnswerPayload:
    answer: str
    sources: list[dict[str, Any]]
    handoff_required: bool
    handoff_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
