"""Embedding wrapper for sentence-transformers (CUDA / CPU)."""

from __future__ import annotations

from loguru import logger

from config.settings import settings


def resolve_device(requested: str) -> str:
    """Resolve 'auto' to 'cuda' if a GPU is available, else 'cpu'."""
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class Embedder:
    """Wrapper around sentence-transformers with CUDA + FP16 support."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        fp16: bool | None = None,
        batch_size: int | None = None,
    ):
        self._model_name = model_name or settings.embedding_model
        self._device = resolve_device(device or settings.embedding_device)
        self._fp16 = settings.embedding_fp16 if fp16 is None else fp16
        self._batch_size = batch_size or settings.embedding_batch_size
        self._model = None

    @property
    def device(self) -> str:
        return self._device

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(
                f"Loading embedding model: {self._model_name} "
                f"(device={self._device}, fp16={self._fp16}, batch_size={self._batch_size})"
            )
            # trust_remote_code: nomic-embed ships its modeling code in the
            # repo (nomic-bert-2048); the workstation reembed script already
            # passes this — the runtime side must match.
            self._model = SentenceTransformer(
                self._model_name, device=self._device, trust_remote_code=True
            )
            if self._fp16:
                if str(self._model.device).startswith("cuda"):
                    self._model.half()
                    logger.info("Embedding model converted to FP16")
                else:
                    logger.warning("FP16 requested but device is not CUDA; staying in FP32")
            logger.info(f"Embedding model loaded on {self._model.device}")
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
        embeddings = self._model.encode(
            texts,
            show_progress_bar=False,
            batch_size=self._batch_size,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed([text])[0]
