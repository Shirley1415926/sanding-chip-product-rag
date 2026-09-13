"""The small offline pipeline: Markdown -> split -> embed -> Chroma."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .chroma_store import ChromaStore
from .domain import SourceDocument
from .embedding import EmbeddingProvider
from .markdown_loader import load_markdown_many
from .splitter import SemanticRecursiveSplitter


@dataclass(frozen=True)
class IngestionResult:
    source: str
    chunk_count: int

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


class MarkdownIngestionPipeline:
    def __init__(
        self,
        splitter: SemanticRecursiveSplitter,
        embedder: EmbeddingProvider,
        store: ChromaStore,
    ) -> None:
        self._splitter = splitter
        self._embedder = embedder
        self._store = store

    def ingest_path(self, path: Path) -> list[IngestionResult]:
        return [self.ingest_document(document) for document in load_markdown_many(path)]

    def ingest_document(self, document: SourceDocument) -> IngestionResult:
        chunks = self._splitter.split(document)
        if not chunks:
            raise ValueError(f"No chunks produced for {document.metadata['source']}")
        vectors = self._embedder.embed_documents([chunk.text for chunk in chunks])
        self._store.replace_source(chunks, vectors)
        return IngestionResult(source=document.metadata["source"], chunk_count=len(chunks))
