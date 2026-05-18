"""Pluggable vector backends for RAG retrieval.

Each backend implements `VectorBackend` and is keyed by collection name in
`Retriever`. The default is `ChromaBackend`; large collections (wikipedia,
gutenberg) can later be served by `FaissIvfPqBackend` to stay within the
Jetson's RAM budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Hit:
    text: str
    source: str
    distance: float
    metadata: dict = field(default_factory=dict)
    chunk_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "distance": self.distance,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id,
        }


@runtime_checkable
class VectorBackend(Protocol):
    """Per-collection retrieval backend."""

    name: str

    def query(self, query_text: str, top_k: int) -> list[Hit]: ...
