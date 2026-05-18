"""Backend protocol + ChromaBackend smoke tests (no chromadb network/disk)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from oracle.rag.backends import Hit, VectorBackend
from oracle.rag.backends.chroma import ChromaBackend


def test_hit_to_dict_round_trip():
    h = Hit(text="t", source="c", distance=0.25, metadata={"k": "v"}, chunk_id="abc")
    d = h.to_dict()
    assert d == {
        "text": "t",
        "source": "c",
        "distance": 0.25,
        "metadata": {"k": "v"},
        "chunk_id": "abc",
    }


def test_chroma_backend_satisfies_protocol():
    client = MagicMock()
    embedder = MagicMock()
    backend = ChromaBackend("wikimed", client=client, embedder=embedder)
    assert isinstance(backend, VectorBackend)
    assert backend.name == "wikimed"


def test_chroma_backend_query_shape():
    client = MagicMock()
    embedder = MagicMock()
    embedder.embed_single.return_value = [0.1, 0.2, 0.3]

    fake_collection = MagicMock()
    fake_collection.query.return_value = {
        "ids": [["id1", "id2"]],
        "documents": [["doc one", "doc two"]],
        "distances": [[0.1, 0.5]],
        "metadatas": [[{"src": "a"}, {"src": "b"}]],
    }
    client.get_collection.return_value = fake_collection

    backend = ChromaBackend("wikimed", client=client, embedder=embedder)
    hits = backend.query("what is X", top_k=2)

    assert len(hits) == 2
    assert hits[0].text == "doc one"
    assert hits[0].chunk_id == "id1"
    assert hits[0].distance == pytest.approx(0.1)
    assert hits[0].metadata == {"src": "a"}
    assert hits[0].source == "wikimed"


def test_chroma_backend_empty_result():
    client = MagicMock()
    embedder = MagicMock()
    embedder.embed_single.return_value = [0.0]
    fake_collection = MagicMock()
    fake_collection.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "distances": [[]],
        "metadatas": [[]],
    }
    client.get_collection.return_value = fake_collection

    backend = ChromaBackend("empty", client=client, embedder=embedder)
    assert backend.query("anything", top_k=5) == []


def test_chroma_backend_swallows_collection_errors():
    """Failed collection queries should not raise — they should return []."""
    client = MagicMock()
    embedder = MagicMock()
    embedder.embed_single.return_value = [0.0]
    fake_collection = MagicMock()
    fake_collection.query.side_effect = RuntimeError("HNSW dead")
    client.get_collection.return_value = fake_collection

    backend = ChromaBackend("wikipedia", client=client, embedder=embedder)
    assert backend.query("anything", top_k=5) == []
