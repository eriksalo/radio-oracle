"""Tier-1 / Tier-2 mode plumbing tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from oracle.rag.backends import Hit
from oracle.rag.modes import detect_mode, params_for


def test_detect_mode_default_is_snappy():
    assert detect_mode("What is the boiling point of water?") == "snappy"


def test_detect_mode_picks_up_go_deeper():
    assert detect_mode("Tell me more about that.") == "deep"
    assert detect_mode("Go deeper on the symptoms.") == "deep"
    assert detect_mode("anything else?") == "deep"
    assert detect_mode("Could you elaborate?") == "deep"


def test_detect_mode_is_case_insensitive():
    assert detect_mode("TELL ME MORE") == "deep"


class _FakeSettings:
    tier1_top_k = 5
    tier2_top_k = 20
    tier2_rerank_pool = 100
    tier2_final_top_k = 20


def test_params_for_snappy_skips_reranker():
    p = params_for("snappy", _FakeSettings)
    assert p.mode == "snappy"
    assert p.per_collection_top_k == 5
    assert p.rerank_pool == 0
    assert p.final_top_k == 5


def test_params_for_deep_enables_reranker():
    p = params_for("deep", _FakeSettings)
    assert p.mode == "deep"
    assert p.per_collection_top_k == 20
    assert p.rerank_pool == 100
    assert p.final_top_k == 20


def test_retriever_deep_mode_invokes_reranker(monkeypatch):
    """Retriever should call rerank() exactly once in deep mode and skip it otherwise."""
    from oracle.rag.retriever import Retriever

    fake_backend = MagicMock()
    fake_backend.query.return_value = [
        Hit(text=f"doc {i}", source="x", distance=i * 0.01, chunk_id=str(i)) for i in range(50)
    ]
    fake_reranker = MagicMock()
    fake_reranker.rerank.side_effect = lambda q, hits, k: hits[:k]

    r = Retriever(embedder=MagicMock(), reranker=fake_reranker)
    r._client = MagicMock()
    r._client.list_collections.return_value = [MagicMock(name="c")]
    r._backends = {"x": fake_backend}

    # Bypass list_collections by passing names explicitly
    snappy = r.query("q", collection_names=["x"], mode="snappy")
    assert fake_reranker.rerank.call_count == 0
    assert len(snappy) == 5  # tier1_top_k default

    deep = r.query("q", collection_names=["x"], mode="deep")
    assert fake_reranker.rerank.call_count == 1
    assert len(deep) <= 20  # tier2_final_top_k default
