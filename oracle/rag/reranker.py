"""Cross-encoder reranker for Tier-2 retrieval."""

from __future__ import annotations

from loguru import logger

from config.settings import settings
from oracle.rag.backends import Hit


class CrossEncoderReranker:
    """Wraps sentence-transformers `CrossEncoder` to re-score (query, text) pairs.

    Loaded lazily — the model only hits disk on the first `rerank` call.
    Runs on CPU by default to avoid contending with the LLM for VRAM on Jetson.
    """

    def __init__(self, model_name: str | None = None, device: str = "cpu"):
        self._model_name = model_name or settings.reranker_model
        self._device = device
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            logger.info(
                f"Loading reranker: {self._model_name} (device={self._device})"
            )
            self._model = CrossEncoder(self._model_name, device=self._device)
        except ImportError:
            logger.error(
                "sentence-transformers missing — install via the rag extra."
            )
            raise

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        if top_k >= len(hits):
            top_k = len(hits)
        self.load()
        pairs = [(query, h.text) for h in hits]
        scores = self._model.predict(pairs, show_progress_bar=False)
        # Higher score = more relevant. Convert to "distance" semantics
        # (lower = better) so downstream sorters work uniformly.
        for h, s in zip(hits, scores):
            h.distance = float(-s)
        hits.sort(key=lambda h: h.distance)
        return hits[:top_k]
