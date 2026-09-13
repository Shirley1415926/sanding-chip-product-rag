"""A Markdown-aware splitter implemented here rather than delegated to a framework."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .domain import Chunk, SourceDocument


HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class _Section:
    text: str
    heading: str


class SemanticRecursiveSplitter:
    """Preserve headings and natural boundaries before recursively cutting text.

    This is intentionally a small, inspectable strategy: Markdown headings form
    semantic sections; paragraphs and line boundaries come next; Chinese and
    English sentence punctuation are next; fixed-size windows are the final
    fallback. It is not an LLM semantic parser and makes no external call.
    """

    def __init__(self, chunk_size: int = 460, overlap: int = 60) -> None:
        if chunk_size < 80:
            raise ValueError("chunk_size must be at least 80 characters")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: SourceDocument) -> list[Chunk]:
        sections = self._sections(document.text)
        raw_chunks: list[tuple[str, str]] = []
        for section in sections:
            for part in self._split_recursively(section.text, ("\n\n", "\n", "。", "！", "？", ". ", " ")):
                cleaned = part.strip()
                if cleaned:
                    raw_chunks.append((cleaned, section.heading))

        chunks: list[Chunk] = []
        prior = ""
        search_from = 0
        for index, (text, heading) in enumerate(raw_chunks):
            prefix = prior[-self.overlap :] if prior and self.overlap else ""
            chunk_text = f"{prefix}{text}" if prefix else text
            source_offset = document.text.find(text, search_from)
            if source_offset >= 0:
                search_from = source_offset + len(text)
            else:
                source_offset = 0
            digest = hashlib.sha256(
                f"{document.metadata['source']}:{index}:{text}".encode("utf-8")
            ).hexdigest()[:20]
            metadata: dict[str, str | int] = {
                **document.metadata,
                "chunk_index": index,
                "heading": heading,
                "start_offset": source_offset,
                "end_offset": source_offset + len(text),
            }
            chunks.append(Chunk(chunk_id=f"chunk_{digest}", text=chunk_text, metadata=metadata))
            prior = text
        return chunks

    def _sections(self, markdown: str) -> list[_Section]:
        current_lines: list[str] = []
        sections: list[_Section] = []
        hierarchy: list[str] = []

        def flush() -> None:
            nonlocal current_lines
            text = "\n".join(current_lines).strip()
            if text:
                sections.append(_Section(text=text, heading=" > ".join(hierarchy) or "正文"))
            current_lines = []

        for line in markdown.splitlines():
            match = HEADING.match(line)
            if match:
                flush()
                level, title = len(match.group(1)), match.group(2)
                hierarchy = hierarchy[: level - 1]
                hierarchy.append(title)
                current_lines.append(line)
            else:
                current_lines.append(line)
        flush()
        return sections

    def _split_recursively(self, text: str, separators: tuple[str, ...]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        if not separators:
            return [text[index : index + self.chunk_size] for index in range(0, len(text), self.chunk_size)]

        separator = separators[0]
        if separator not in text:
            return self._split_recursively(text, separators[1:])

        pieces = text.split(separator)
        rebuilt = [piece + separator for piece in pieces[:-1]] + [pieces[-1]]
        result: list[str] = []
        buffer = ""
        for piece in rebuilt:
            if len(piece) > self.chunk_size:
                if buffer.strip():
                    result.append(buffer)
                    buffer = ""
                result.extend(self._split_recursively(piece, separators[1:]))
                continue
            if buffer and len(buffer) + len(piece) > self.chunk_size:
                result.append(buffer)
                buffer = piece
            else:
                buffer += piece
        if buffer.strip():
            result.append(buffer)
        return result
