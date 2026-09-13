"""Small environment configuration layer; no framework or hidden global state."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv_if_present(path: Path) -> None:
    """Load simple KEY=VALUE lines without adding python-dotenv as a dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


@dataclass(frozen=True)
class Settings:
    project_root: Path
    embedding_provider: str
    embedding_model: str
    llm_provider: str
    llm_api_base: str
    llm_api_key: str
    llm_model: str
    chroma_path: Path
    collection_name: str
    top_k: int
    min_relevance: float
    evaluation_thresholds: tuple[float, ...]

    @classmethod
    def from_environment(cls, project_root: Path) -> "Settings":
        load_dotenv_if_present(project_root / ".env")
        chroma_location = Path(os.getenv("CHROMA_PATH", "data/chroma"))
        if not chroma_location.is_absolute():
            chroma_location = project_root / chroma_location
        return cls(
            project_root=project_root,
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "sentence_transformers"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            llm_provider=os.getenv("LLM_PROVIDER", "openai_compatible"),
            llm_api_base=os.getenv("LLM_API_BASE", "https://api.openai.com/v1"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
            chroma_path=chroma_location,
            collection_name=os.getenv("COLLECTION_NAME", "sanding_product_knowledge"),
            top_k=int(os.getenv("TOP_K", "4")),
            min_relevance=float(os.getenv("MIN_RELEVANCE", "0.60")),
            evaluation_thresholds=tuple(
                float(value.strip())
                for value in os.getenv("EVALUATION_THRESHOLDS", "0.30,0.45,0.60,0.70").split(",")
                if value.strip()
            ),
        )
