"""FAISS IVF-PQ backed retrieval for large collections (wiki, gutenberg).

Loads a prebuilt `<collection>.index` (FAISS IVF-PQ) and `<collection>.sqlite`
(row_id -> chunk text + metadata) once at startup. Query time:
    embed(query) -> faiss.search -> sqlite lookup -> Hits

Per-backend embedder so each large collection can use its own model. A
module-level cache keys by model name so multiple backends sharing a model
load the weights once.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from oracle.rag.backends import Hit
from oracle.rag.embedder import Embedder

if TYPE_CHECKING:
    import faiss  # type: ignore


_EMBEDDER_CACHE: dict[str, Embedder] = {}
_EMBEDDER_LOCK = Lock()


def _get_shared_embedder(model_name: str, trust_remote_code: bool = True) -> Embedder:
    with _EMBEDDER_LOCK:
        if model_name not in _EMBEDDER_CACHE:
            _EMBEDDER_CACHE[model_name] = Embedder(model_name=model_name)
        return _EMBEDDER_CACHE[model_name]


class FaissIvfPqBackend:
    """FAISS IVF-PQ over a positional id-map.

    The FAISS index returns integer row ids; the sqlite map resolves each to
    chunk_id + text + metadata. Both files are produced by
    `scripts/build_faiss_ivfpq.py`.
    """

    def __init__(
        self,
        name: str,
        index_path: str | Path,
        sqlite_path: str | Path,
        model_name: str,
        query_prefix: str = "",
        ef_search: int = 64,
        score_scale: float = 20.0,
    ):
        self.name = name
        self._index_path = Path(index_path)
        self._sqlite_path = Path(sqlite_path)
        self._model_name = model_name
        self._query_prefix = query_prefix
        self._ef_search = ef_search
        # FAISS IVF-PQ returns the inner product approximated as the sum across
        # M product-quantizer subspaces, which can run well above 1.0 even with
        # normalized input vectors (e.g. ~14-17 for top hits on nomic-v1.5 +
        # PQ-64). We rescale to a 0..1+ cosine-distance-like range so hits
        # merge sensibly with ChromaBackend's cosine distances. For other
        # model/PQ configs, override score_scale in settings.faiss_collection_config.
        self._score_scale = float(score_scale)
        self._index: faiss.Index | None = None
        self._con: sqlite3.Connection | None = None
        self._embedder: Embedder | None = None

    def _load(self) -> None:
        if self._index is not None:
            return
        import faiss

        if not self._index_path.exists():
            raise FileNotFoundError(f"FAISS index missing: {self._index_path}")
        if not self._sqlite_path.exists():
            raise FileNotFoundError(f"FAISS sqlite missing: {self._sqlite_path}")
        logger.info(f"Loading FAISS index {self._index_path.name} ...")
        # mmap instead of loading resident: the seven indices total ~1.7GB,
        # which on the 8GB Jetson pushed the box into swap once the LLM,
        # STT, TTS, and embedder were also loaded. With mmap the OS pages
        # index data from NVMe on demand and can evict it under pressure;
        # measured query latency stays in the same range (warm pages).
        self._index = faiss.read_index(
            str(self._index_path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
        )
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = self._ef_search
        # Connect to the idmap sqlite read-only.
        self._con = sqlite3.connect(
            f"file:{self._sqlite_path}?mode=ro", uri=True, check_same_thread=False
        )
        self._embedder = _get_shared_embedder(self._model_name)
        logger.info(f"FAISS backend '{self.name}' ready ({self._index.ntotal} vectors)")

    def query(self, query_text: str, top_k: int) -> list[Hit]:
        self._load()
        assert self._index is not None and self._con is not None and self._embedder is not None

        text = f"{self._query_prefix}{query_text}" if self._query_prefix else query_text
        emb = self._embedder.embed_single(text)
        q = np.asarray([emb], dtype=np.float32)
        distances, indices = self._index.search(q, top_k)

        rows = indices[0].tolist()
        dists = distances[0].tolist()
        # Inner-product score: higher = more similar. Convert to a
        # cosine-distance-like "lower = better" value scaled to be roughly
        # comparable to ChromaBackend's distances (0..2 range, with 0..0.3
        # = great match). See __init__ for score_scale rationale.
        placeholders = ",".join("?" for _ in rows if _ != -1)
        valid_rows = [r for r in rows if r != -1]
        if not valid_rows:
            return []

        cur = self._con.cursor()
        cur.execute(
            f"SELECT faiss_row, chunk_id, text, source, url, title, chunk_index "
            f"FROM faiss_idmap WHERE faiss_row IN ({placeholders})",
            valid_rows,
        )
        row_map = {r[0]: r for r in cur.fetchall()}

        hits: list[Hit] = []
        for row_id, score in zip(rows, dists):
            if row_id == -1:
                continue
            row = row_map.get(row_id)
            if row is None:
                continue
            _, chunk_id, text, source, url, title, chunk_index = row
            normalized_distance = max(0.0, 1.0 - score / self._score_scale)
            hits.append(
                Hit(
                    text=text,
                    source=self.name,
                    distance=float(normalized_distance),
                    metadata={
                        "url": url or "",
                        "title": title or "",
                        "chunk_index": chunk_index,
                        "source": source or self.name,
                    },
                    chunk_id=chunk_id,
                )
            )
        return hits
