"""Small Chroma adapter that stores caller-supplied vectors and traceable metadata."""

from __future__ import annotations

from pathlib import Path

from .domain import Chunk, RetrievedChunk


class ChromaStore:
    def __init__(self, persist_path: Path, collection_name: str) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb is required. Run `pip install -e .` in the project venv.") from exc
        persist_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def replace_source(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError("A vector is required for every chunk")
        source = str(chunks[0].metadata["source"])
        if any(str(chunk.metadata["source"]) != source for chunk in chunks):
            raise ValueError("replace_source accepts chunks from one source only")

        # Delete-first makes re-import of a revised Markdown file idempotent.
        self._collection.delete(where={"source": source})
        metadatas = [{key: value for key, value in chunk.metadata.items()} for chunk in chunks]
        self._collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=vectors,
            metadatas=metadatas,
        )

    def query(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        if not self.count:
            return []
        requested = min(top_k, self.count)
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=requested,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=document,
                metadata=metadata,
                score=max(0.0, 1.0 - float(distance)),
            )
            for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances, strict=True)
        ]

    def all_chunks(self) -> list[RetrievedChunk]:
        """Return every stored chunk for deterministic lexical indexing.

        Chroma remains the source of truth: the BM25 adapter rebuilds a tiny,
        in-memory index from these documents rather than maintaining a second
        persisted copy that could drift after re-ingestion.
        """
        if not self.count:
            return []
        result = self._collection.get(include=["documents", "metadatas"])
        chunks = [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=document,
                metadata=metadata,
                score=0.0,
            )
            for chunk_id, document, metadata in zip(
                result.get("ids", []),
                result.get("documents", []),
                result.get("metadatas", []),
                strict=True,
            )
        ]
        return sorted(chunks, key=lambda chunk: chunk.chunk_id)

    def clear(self) -> None:
        """Testing helper that clears only the configured collection."""
        ids = self._collection.get(include=[]).get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
