"""Semantic retrieval across pluggable vector backends with tiered modes."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from config.settings import settings
from oracle.rag.backends import Hit, VectorBackend
from oracle.rag.backends.chroma import ChromaBackend
from oracle.rag.embedder import Embedder
from oracle.rag.modes import RetrievalMode, params_for
from oracle.rag.reranker import CrossEncoderReranker
from oracle.rag.router import route as router_route


class Retriever:
    """Semantic search dispatching to per-collection `VectorBackend`s."""

    def __init__(
        self,
        chroma_path: Path | None = None,
        embedder: Embedder | None = None,
        reranker: CrossEncoderReranker | None = None,
    ):
        self._chroma_path = chroma_path or settings.chroma_path
        self._embedder = embedder or Embedder()
        self._reranker = reranker  # lazy-built only when first deep query lands
        self._client = None
        self._backends: dict[str, VectorBackend] = {}

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

    def _build_backend(self, name: str) -> VectorBackend:
        kind = settings.collection_backends.get(name, "chroma")
        if kind == "faiss":
            from oracle.rag.backends.faiss_ivfpq import FaissIvfPqBackend

            cfg = settings.faiss_collection_config.get(name, {})
            return FaissIvfPqBackend(
                name=name,
                index_path=settings.faiss_index_dir / f"{name}.index",
                sqlite_path=settings.faiss_index_dir / f"{name}.sqlite",
                model_name=cfg.get("model", settings.embedding_model),
                query_prefix=cfg.get("query_prefix", ""),
                ef_search=cfg.get("ef_search", 64),
                score_scale=cfg.get("score_scale", 20.0),
            )
        return ChromaBackend(name, self._get_client(), self._embedder)

    def _get_backend(self, name: str) -> VectorBackend:
        if name not in self._backends:
            self._backends[name] = self._build_backend(name)
        return self._backends[name]

    def _get_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def list_collections(self) -> list[str]:
        client = self._get_client()
        names = {c.name for c in client.list_collections()}
        # Surface FAISS-backed collections that have no chroma counterpart
        # (e.g. the music collection has no ZIM source, so it never lived
        # in chroma). Without this union, the router can't see them.
        for name, kind in settings.collection_backends.items():
            if kind == "faiss":
                names.add(name)
        return sorted(names)

    def query(
        self,
        query_text: str,
        collection_names: list[str] | None = None,
        top_k: int | None = None,
        mode: RetrievalMode = "snappy",
    ) -> list[dict]:
        """Search across one or more collections.

        Returns dicts shaped `{text, source, distance, metadata, chunk_id}`,
        sorted by distance (lower = more relevant), truncated to the mode's
        `final_top_k` (or the explicit `top_k` override if provided).
        """
        params = params_for(mode, settings)
        per_coll_k = params.per_collection_top_k
        final_k = top_k or params.final_top_k

        if collection_names is None:
            # Explicit override via ORACLE_RAG_COLLECTIONS bypasses the router.
            # Useful for memory-constrained hosts that need to skip heavy
            # collections, or for diagnostic queries.
            if settings.rag_collections:
                collection_names = [
                    n.strip() for n in settings.rag_collections.split(",") if n.strip()
                ]
            else:
                available = self.list_collections()
                routing = router_route(query_text, available=available)
                if routing.matched:
                    logger.debug(f"Router matched: {routing.matched}; order: {routing.order}")
                collection_names = routing.order
        if not collection_names:
            logger.warning("No collections available for RAG query")
            return []

        hits: list[Hit] = []
        for name in collection_names:
            try:
                hits.extend(self._get_backend(name).query(query_text, per_coll_k))
            except Exception as e:
                logger.warning(f"Backend '{name}' raised during query: {e}")

        hits.sort(key=lambda h: h.distance)

        if params.rerank_pool > 0 and hits:
            pool = hits[: params.rerank_pool]
            hits = self._get_reranker().rerank(query_text, pool, final_k)
        else:
            hits = hits[:final_k]

        return [h.to_dict() for h in hits]

    def format_context(self, results: list[dict]) -> str:
        if not results:
            return ""
        parts = ["=== Retrieved Knowledge ==="]
        for i, r in enumerate(results, 1):
            source = r.get("source", "unknown")
            parts.append(f"\n[Source {i}: {source}]\n{r['text']}")
        parts.append("\n=== End Retrieved Knowledge ===")
        return "\n".join(parts)
