"""ONNX embedding runtime: shape/normalisation and retrieval sanity.
Exact parity with sentence-transformers is checked on the Jetson by
scripts/probe_embedder.py (the dev venv has no torch)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from config.settings import settings

MODEL_DIR = Path(__file__).parent.parent / "models" / "nomic-embed-text-v1.5-onnx"

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "model_int8.onnx").exists(), reason="nomic ONNX model not present"
)


@pytest.fixture
def onnx_embedder(monkeypatch):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    monkeypatch.setattr(settings, "embedding_runtime", "onnx")
    monkeypatch.setattr(
        settings, "onnx_embedding_dirs", {"nomic-ai/nomic-embed-text-v1.5": str(MODEL_DIR)}
    )
    from oracle.rag.embedder import Embedder

    e = Embedder(model_name="nomic-ai/nomic-embed-text-v1.5")
    e.load()
    return e


def test_onnx_embeddings_are_raw_mean_pooled_768(onnx_embedder):
    v = np.asarray(onnx_embedder.embed_single("search_query: who was Nikola Tesla?"))
    assert v.shape == (768,)
    # Un-normalised, like the sentence-transformers path (norm ≈ 23 for
    # nomic-v1.5): the FAISS score scale depends on it.
    assert 10.0 < np.linalg.norm(v) < 40.0


def test_onnx_embeddings_rank_relevant_document_first(onnx_embedder):
    q = np.asarray(onnx_embedder.embed_single("search_query: where did Nikola Tesla die?"))
    docs = onnx_embedder.embed(
        [
            "search_document: Nikola Tesla died in New York City in January 1943.",
            "search_document: Water boils at a lower temperature at high altitude.",
            "search_document: The northern lights are caused by solar particles.",
        ]
    )
    d = np.asarray(docs)
    sims = (d @ q) / (np.linalg.norm(d, axis=1) * np.linalg.norm(q))
    assert int(np.argmax(sims)) == 0
    assert sims[0] > 0.5


def test_onnx_batch_matches_single(onnx_embedder):
    texts = [
        "search_query: a short one",
        "search_query: a considerably longer query than the first",
    ]
    batch = np.asarray(onnx_embedder.embed(texts))
    singles = np.asarray([onnx_embedder.embed_single(t) for t in texts])
    # Padding must not change the pooled vector beyond quantisation noise
    # (dynamic int8 scales activations per batch tensor, so batched and
    # single runs differ slightly — measured cos ≈ 0.98 for the int8 file).
    cos = (batch * singles).sum(axis=1) / (
        np.linalg.norm(batch, axis=1) * np.linalg.norm(singles, axis=1)
    )
    assert cos.min() > 0.95


def test_onnx_mode_does_not_probe_torch(monkeypatch):
    """resolve_device('auto') imports torch (~350 MB on the Jetson); the
    ONNX runtime must not call it."""
    from oracle.rag import embedder as mod

    monkeypatch.setattr(settings, "embedding_runtime", "onnx")
    monkeypatch.setattr(settings, "embedding_device", "auto")
    monkeypatch.setattr(
        settings, "onnx_embedding_dirs", {"nomic-ai/nomic-embed-text-v1.5": str(MODEL_DIR)}
    )
    called = []
    monkeypatch.setattr(mod, "resolve_device", lambda req: called.append(req) or "cpu")
    e = mod.Embedder(model_name="nomic-ai/nomic-embed-text-v1.5")
    assert e.device == "cpu" and called == []
