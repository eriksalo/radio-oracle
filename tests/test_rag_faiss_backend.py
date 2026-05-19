"""End-to-end smoke test for FaissIvfPqBackend on a tiny synthetic fixture.

Skipped if faiss is not installed in this environment.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from oracle.rag.backends.faiss_ivfpq import FaissIvfPqBackend


@pytest.fixture
def faiss_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Build a tiny IVF-Flat index + sqlite mapping in a temp dir.

    IVF-PQ requires a meaningful training set; for 200 vectors we use IVF
    over a Flat quantizer (same backend API) to keep the test fast.
    """
    rng = np.random.default_rng(42)
    dim = 32
    n = 200
    vecs = rng.standard_normal((n, dim), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, 4, faiss.METRIC_INNER_PRODUCT)
    index.train(vecs)
    index.add(vecs)
    idx_path = tmp_path / "fixture.index"
    faiss.write_index(index, str(idx_path))

    db_path = tmp_path / "fixture.sqlite"
    con = sqlite3.connect(str(db_path))
    con.execute(
        """CREATE TABLE faiss_idmap (
            faiss_row INTEGER PRIMARY KEY,
            chunk_id  TEXT NOT NULL,
            text      TEXT NOT NULL,
            source    TEXT,
            url       TEXT,
            title     TEXT,
            chunk_index INTEGER
        )"""
    )
    con.executemany(
        "INSERT INTO faiss_idmap VALUES (?,?,?,?,?,?,?)",
        [
            (i, f"id{i:03d}", f"document body {i}", "wikipedia", "", "", i)
            for i in range(n)
        ],
    )
    con.commit()
    con.close()

    return idx_path, db_path


def test_faiss_backend_query_returns_hits(faiss_fixture, monkeypatch):
    idx_path, db_path = faiss_fixture

    # Stub the embedder so we don't load a real model. The fixture used a
    # 32-d float32 vectors, so embed_single must return a 32-d list.
    rng = np.random.default_rng(7)
    fake_emb = rng.standard_normal(32, dtype=np.float32)
    fake_emb /= np.linalg.norm(fake_emb)

    from oracle.rag.backends import faiss_ivfpq

    class _FakeEmbedder:
        def embed_single(self, text):
            return fake_emb.tolist()

    monkeypatch.setattr(
        faiss_ivfpq, "_get_shared_embedder", lambda *a, **kw: _FakeEmbedder()
    )

    backend = FaissIvfPqBackend(
        name="wikipedia",
        index_path=idx_path,
        sqlite_path=db_path,
        model_name="fake-model",
        query_prefix="search_query: ",
        ef_search=4,
    )
    hits = backend.query("what is X?", top_k=5)
    assert len(hits) > 0
    assert len(hits) <= 5
    for h in hits:
        assert h.source == "wikipedia"
        assert h.chunk_id is not None
        assert h.text.startswith("document body ")
        # Pseudo-distance should be a finite float in roughly [0, 2]
        assert 0.0 <= h.distance <= 2.5


def test_faiss_backend_protocol_compliance(faiss_fixture):
    from oracle.rag.backends import VectorBackend

    idx_path, db_path = faiss_fixture
    backend = FaissIvfPqBackend(
        name="wikipedia",
        index_path=idx_path,
        sqlite_path=db_path,
        model_name="fake",
    )
    assert isinstance(backend, VectorBackend)
