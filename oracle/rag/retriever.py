"""ChromaDB-based semantic retrieval."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from config.settings import settings
from oracle.rag.embedder import Embedder


class Retriever:
    """Semantic search across ChromaDB collections."""

    def __init__(
        self,
        chroma_path: Path | None = None,
        embedder: Embedder | None = None,
    ):
        self._chroma_path = chroma_path or settings.chroma_path
        self._embedder = embedder or Embedder()
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import chromadb

                self._client = chromadb.PersistentClient(path=str(self._chroma_path))
                logger.info(f"ChromaDB client initialized at {self._chroma_path}")
            except ImportError:
                logger.error("chromadb not installed. Install with: pip install chromadb")
                raise
        return self._client

    def list_collections(self) -> list[str]:
        """List all available collection names."""
        client = self._get_client()
        return [c.name for c in client.list_collections()]

    def query(
        self,
        query_text: str,
        collection_names: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """Search across one or more collections.

        Args:
            query_text: Natural language query
            collection_names: Collections to search (None = all)
            top_k: Number of results per collection

        Returns:
            List of dicts with 'text', 'source', 'distance' keys, sorted by distance
        """
        k = top_k or settings.rag_top_k
        client = self._get_client()

        if collection_names is None:
            collection_names = self.list_collections()

        if not collection_names:
            logger.warning("No collections available for RAG query")
            return []

        query_embedding = self._embedder.embed_single(query_text)
        results: list[dict] = []

        for name in collection_names:
            try:
                collection = client.get_collection(name)
                hits = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                )
                for i, doc in enumerate(hits["documents"][0]):
                    metadata = hits["metadatas"][0][i] if hits["metadatas"] else {}
                    results.append(
                        {
                            "text": doc,
                            "source": name,
                            "distance": hits["distances"][0][i] if hits["distances"] else 0.0,
                            "metadata": metadata,
                        }
                    )
            except Exception as e:
                logger.warning(f"Error querying collection '{name}': {e}")

        results.sort(key=lambda x: x["distance"])
        return results[:k]

    def format_context(self, results: list[dict]) -> str:
        """Format retrieval results into a context string for the LLM."""
        if not results:
            return ""

        parts = ["=== Retrieved Knowledge ==="]
        for i, r in enumerate(results, 1):
            source = r.get("source", "unknown")
            parts.append(f"\n[Source {i}: {source}]\n{r['text']}")
        parts.append("\n=== End Retrieved Knowledge ===")
        return "\n".join(parts)
