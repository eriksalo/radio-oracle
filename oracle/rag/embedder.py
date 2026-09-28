"""Query/document embedding wrapper — sentence-transformers or ONNX runtime.

Two runtimes behind one ``Embedder`` surface (``settings.embedding_runtime``):

* ``sentence-transformers`` — the original path (torch). Used on the
  workstation for ingest/re-embed, where CUDA and batch throughput matter.
* ``onnx`` — onnxruntime + the ``tokenizers`` library, no torch. Same
  math as the ST pipeline for nomic-embed-text-v1.5 as configured here
  (mean pooling over the attention mask, *no* normalisation — the query
  vector's magnitude is part of the calibrated distance scale), so vectors
  are interchangeable with the FAISS indices built by
  ``scripts/reembed_collection.py``.
  Chosen for the Jetson: importing torch + a fp32 BERT was ~1 GB of the
  app's 2.5 GB RSS and ~10 s of boot; the int8 ONNX model is 131 MB.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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


class _OnnxBackend:
    """Tokenize → ONNX encoder → masked mean pool → L2 normalise."""

    def __init__(self, model_dir: Path, model_file: str, num_threads: int) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tok.enable_truncation(settings.onnx_embedding_max_tokens)
        self._tok.enable_padding(pad_id=0, pad_token="[PAD]")
        so = ort.SessionOptions()
        so.intra_op_num_threads = num_threads
        so.log_severity_level = 3
        self._sess = ort.InferenceSession(
            str(model_dir / model_file), so, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._sess.get_inputs()}

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        out: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = self._tok.encode_batch(texts[i : i + batch_size])
            ids = np.array([e.ids for e in batch], dtype=np.int64)
            mask = np.array([e.attention_mask for e in batch], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self._sess.run(None, feeds)[0]  # (B, T, 768)
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)
            # No L2 normalisation: the sentence-transformers path this
            # replaces returns raw mean-pooled vectors (norm ≈23 for
            # nomic-v1.5) and the FAISS distance gate / score_scale were
            # calibrated on those inner products. Normalising here made
            # every hit land at distance ~0.96 and get gated (2026-09-27).
            out.append(pooled.astype(np.float32))
        return np.concatenate(out, axis=0)


class Embedder:
    """Wrapper around sentence-transformers (CUDA/CPU) or an ONNX encoder."""

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
        self._onnx: _OnnxBackend | None = None

    @property
    def device(self) -> str:
        return self._device

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _onnx_dir(self) -> Path | None:
        if settings.embedding_runtime != "onnx":
            return None
        d = settings.onnx_embedding_dirs.get(self._model_name)
        return Path(d) if d else None

    def load(self) -> None:
        if self._model is not None or self._onnx is not None:
            return
        onnx_dir = self._onnx_dir()
        if onnx_dir is not None:
            if not onnx_dir.is_dir():
                raise FileNotFoundError(
                    f"ONNX embedding dir not found: {onnx_dir} — run scripts/download_models.sh"
                )
            logger.info(
                f"Loading ONNX embedding model: {self._model_name} from {onnx_dir}/"
                f"{settings.onnx_embedding_file} (threads={settings.onnx_embedding_threads})"
            )
            self._onnx = _OnnxBackend(
                onnx_dir, settings.onnx_embedding_file, settings.onnx_embedding_threads
            )
            self._device = "cpu"
            logger.info("ONNX embedding model loaded")
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
        if self._model is None and self._onnx is None:
            self.load()
        if self._onnx is not None:
            return self._onnx.encode(texts, self._batch_size).tolist()
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
