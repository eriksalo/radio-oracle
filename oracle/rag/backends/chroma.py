"""ChromaDB-backed vector retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from oracle.rag.backends import Hit
from oracle.rag.embedder import Embedder

if TYPE_CHECKING:
    from chromadb.api import ClientAPI


class ChromaBackend:
    """A single ChromaDB collection wrapped as a `VectorBackend`."""

    def __init__(self, name: str, client: ClientAPI, embedder: Embedder):
        self.name = name
        self._client = client
        self._embedder = embedder
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            self._collection = self._client.get_collection(self.name)
        return self._collection

    def query(self, query_text: str, top_k: int) -> list[Hit]:
        query_embedding = self._embedder.embed_single(query_text)
        try:
            hits = self._get_collection().query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )
        except Exception as e:
            logger.warning(f"ChromaBackend '{self.name}' query failed: {e}")
            return []

        docs = hits.get("documents") or [[]]
        if not docs or not docs[0]:
            return []

        distances = (hits.get("distances") or [[0.0] * len(docs[0])])[0]
        metadatas = (hits.get("metadatas") or [[{}] * len(docs[0])])[0]
        ids = (hits.get("ids") or [[None] * len(docs[0])])[0]

        return [
            Hit(
                text=docs[0][i],
                source=self.name,
                distance=distances[i] if i < len(distances) else 0.0,
                metadata=metadatas[i] if i < len(metadatas) and metadatas[i] else {},
                chunk_id=ids[i] if i < len(ids) else None,
            )
            for i in range(len(docs[0]))
        ]
