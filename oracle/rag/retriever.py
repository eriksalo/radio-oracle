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
from oracle.rag.router import bias_for, question_type
from oracle.rag.router import route as router_route

# How a source is named inside the prompt: the model reads these, so they
# carry what it needs to weigh the passage ("old book").
_SOURCE_LABELS = {
    "gutenberg": "gutenberg (a book from before 1930 — historical, check its advice)",
    "wikimed": "wikimed (medical reference)",
    "wikipedia": "wikipedia",
    "ifixit": "ifixit (repair guide)",
    "wikibooks": "wikibooks (textbook)",
    "crashcourse": "crashcourse (lesson)",
}


class Retriever:
    """Semantic search dispatching to per-collection `VectorBackend`s."""

    def __init__(
        self,
        chroma_path: Path | None = None,
        embedder: Embedder | None = None,
        reranker: CrossEncoderReranker | None = None,
    ):
        self._chroma_path = chroma_path or settings.chroma_path
        # Built lazily: only Chroma backends use it, and constructing it
        # eagerly imported torch (device probe) on a FAISS-only Jetson.
        self._embedder_override = embedder
        self._embedder_cache: Embedder | None = None
        self._reranker = reranker  # lazy-built only when first deep query lands
        self._client = None
        self._backends: dict[str, VectorBackend] = {}

    @property
    def _embedder(self) -> Embedder:
        if self._embedder_cache is None:
            self._embedder_cache = self._embedder_override or Embedder()
        return self._embedder_cache

    def _chroma_needed(self) -> bool:
        """Any collection not routed to FAISS still lives in Chroma."""
        backends = settings.collection_backends
        return not backends or any(kind != "faiss" for kind in backends.values())

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

    def collection_sizes(self) -> dict[str, int]:
        """Vectors per loaded FAISS collection (for the device summary)."""
        out: dict[str, int] = {}
        for name, backend in self._backends.items():
            idx = getattr(backend, "_index", None)
            if idx is not None:
                out[name] = int(idx.ntotal)
        return out

    def list_collections(self) -> list[str]:
        names: set[str] = set()
        # On the Jetson every collection is FAISS: never touch Chroma there
        # (importing chromadb mapped pandas, pyarrow and sklearn into the
        # app — hundreds of MB — for a client that was never queried).
        if self._chroma_needed():
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
        excluded = {n.strip() for n in settings.rag_exclude_collections.split(",") if n.strip()}
        collection_names = [n for n in collection_names if n not in excluded]
        if params.max_collections > 0:
            collection_names = collection_names[: params.max_collections]
        if not collection_names:
            logger.warning("No collections available for RAG query")
            return []

        if (
            settings.rag_collection_bias
            and top_k is None
            and question_type(query_text) == "literature"
        ):
            per_coll_k = max(per_coll_k, settings.rag_literature_top_k)
        hits: list[Hit] = []
        for name in collection_names:
            try:
                hits.extend(self._get_backend(name).query(query_text, per_coll_k))
            except Exception as e:
                logger.warning(f"Backend '{name}' raised during query: {e}")

        # Relevance gate on the *raw* distance: better to inject nothing
        # than off-topic chunks — the persona is instructed to say when the
        # archives have no answer.
        before = len(hits)
        hits = [h for h in hits if h.distance <= settings.rag_max_distance]
        if before and not hits:
            logger.info(
                f"RAG: all {before} hits above distance gate "
                f"{settings.rag_max_distance} — injecting nothing"
            )

        # Merge by biased distance: the question type says which shelves
        # to trust (medical → WikiMed, how-to → iFixit/Wikibooks, plot →
        # Gutenberg + Wikipedia); Gutenberg loses close races everywhere
        # else. Then keep the top hits diverse: one collection may not
        # take every slot while another has a hit under the gate.
        bias = bias_for(query_text) if settings.rag_collection_bias else {}
        qtype = question_type(query_text)
        if qtype == "literature" and top_k is None and settings.rag_collection_bias:
            # Plot and character facts live deep in the article or the book
            # itself; the lead chunk alone had the 4B model inventing endings.
            final_k = max(final_k, settings.rag_literature_top_k)
        hits.sort(key=lambda h: h.distance + bias.get(h.source, 0.0))
        if len(hits) > final_k and len({h.source for h in hits}) > 1:
            cap = max(1, final_k - 1)
            picked: list[Hit] = []
            spill: list[Hit] = []
            per: dict[str, int] = {}
            for h in hits:
                if per.get(h.source, 0) < cap:
                    picked.append(h)
                    per[h.source] = per.get(h.source, 0) + 1
                else:
                    spill.append(h)
            hits = picked + spill
        if hits:
            logger.debug(
                f"RAG[{qtype}]: "
                + ", ".join(f"{h.source}@{h.distance:.2f}" for h in hits[:final_k])
            )

        if params.rerank_pool > 0 and hits and settings.rag_rerank_enabled:
            pool = hits[: params.rerank_pool]
            hits = self._get_reranker().rerank(query_text, pool, final_k)
        else:
            hits = hits[:final_k]

        return [h.to_dict() for h in hits]

    def format_context(self, results: list[dict]) -> str:
        if not results:
            return ""
        parts = [
            "=== Retrieved Knowledge ===",
            "(Where a passage says to seek medical help or call emergency services, give the "
            "steps the user can take themselves and the warning signs instead; do not mention "
            "doctors or their absence.)",
        ]
        limit = settings.rag_chunk_char_limit
        for i, r in enumerate(results, 1):
            source = r.get("source", "unknown")
            # Article/book title lets the persona actually cite sources
            # ("according to the Wikipedia entry on X") instead of just
            # naming the collection.
            title = (r.get("metadata") or {}).get("title") or ""
            shelf = _SOURCE_LABELS.get(source, source)
            label = f"{shelf} — {title}" if title else shelf
            text = r["text"]
            if limit and len(text) > limit:
                # Full 512-word chunks are ~3.3KB each; five of them cost
                # ~10s of prompt prefill per turn on the Jetson. Truncate
                # at a word boundary — the lead of a chunk carries most of
                # the answer-bearing content.
                text = text[:limit].rsplit(" ", 1)[0] + " …"
            parts.append(f"\n[Source {i}: {label}]\n{text}")
        parts.append("\n=== End Retrieved Knowledge ===")
        return "\n".join(parts)
