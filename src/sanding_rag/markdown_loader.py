"""Markdown-only loader with deliberately small front-matter parsing surface."""

from __future__ import annotations

from pathlib import Path

from .domain import REQUIRED_METADATA, SourceDocument


class MetadataValidationError(ValueError):
    """Raised before any non-traceable document can enter the vector store."""


def _parse_front_matter(raw: str, path: Path) -> tuple[dict[str, str], str]:
    lines = raw.replace("\r\n", "\n").split("\n")
    if not lines or lines[0].strip() != "---":
        raise MetadataValidationError(f"{path}: missing opening front-matter delimiter '---'")

    try:
        closing_index = next(
            index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"
        )
    except StopIteration as exc:
        raise MetadataValidationError(f"{path}: missing closing front-matter delimiter '---'") from exc

    metadata: dict[str, str] = {}
    for line in lines[1:closing_index]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            raise MetadataValidationError(f"{path}: invalid front-matter line: {line!r}")
        key, value = line.split(":", maxsplit=1)
        key, value = key.strip(), value.strip().strip("\"'")
        if not key or not value:
            raise MetadataValidationError(f"{path}: blank front-matter key or value")
        metadata[key] = value

    missing = [key for key in REQUIRED_METADATA if not metadata.get(key)]
    if missing:
        raise MetadataValidationError(f"{path}: missing required metadata: {', '.join(missing)}")
    if metadata["document_type"] not in {"product", "policy", "faq"}:
        raise MetadataValidationError(
            f"{path}: document_type must be product, policy or faq; got {metadata['document_type']!r}"
        )

    text = "\n".join(lines[closing_index + 1 :]).strip()
    if not text:
        raise MetadataValidationError(f"{path}: Markdown body must not be empty")
    return metadata, text


def load_markdown(path: Path) -> SourceDocument:
    if path.suffix.lower() != ".md":
        raise ValueError(f"Only Markdown is supported in phase 1: {path}")
    metadata, text = _parse_front_matter(path.read_text(encoding="utf-8"), path)
    return SourceDocument(text=text, metadata=metadata)


def discover_markdown(path: Path) -> list[Path]:
    """Resolve a Markdown file or recursively sorted directory of Markdown files."""
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(candidate for candidate in path.rglob("*.md") if candidate.is_file())
    raise FileNotFoundError(path)


def load_markdown_many(path: Path) -> list[SourceDocument]:
    paths = discover_markdown(path)
    if not paths:
        raise ValueError(f"No Markdown files found under {path}")
    return [load_markdown(candidate) for candidate in paths]
