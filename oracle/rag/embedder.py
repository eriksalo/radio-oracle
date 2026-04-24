"""Embedding wrapper for sentence-transformers (CPU)."""

from __future__ import annotations

from loguru import logger

from config.settings import settings


class Embedder:
    """Wrapper around sentence-transformers for CPU embedding."""

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or settings.embedding_model
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name, device="cpu")
            logger.info("Embedding model loaded")
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, returns list of float vectors."""
        if self._model is None:
            self.load()
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed([text])[0]
