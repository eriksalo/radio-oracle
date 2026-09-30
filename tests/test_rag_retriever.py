"""Retriever-level behavior: relevance gate, exclusions, context format."""

from __future__ import annotations

from unittest.mock import MagicMock

from config.settings import settings
from oracle.rag.backends import Hit
from oracle.rag.retriever import Retriever


def _retriever_with_hits(monkeypatch, hits_by_collection: dict[str, list[Hit]]):
    r = Retriever(embedder=MagicMock())
    monkeypatch.setattr(r, "list_collections", lambda: sorted(hits_by_collection.keys()))

    def _get_backend(name):
        backend = MagicMock()
        backend.query.return_value = hits_by_collection.get(name, [])
        return backend

    monkeypatch.setattr(r, "_get_backend", _get_backend)
    return r


def _hit(text: str, source: str, distance: float, title: str = "") -> Hit:
    meta = {"title": title} if title else {}
    return Hit(text=text, source=source, distance=distance, metadata=meta, chunk_id=text)


def test_relevance_gate_drops_offtopic(monkeypatch):
    r = _retriever_with_hits(
        monkeypatch,
        {"wikipedia": [_hit("good", "wikipedia", 0.2), _hit("junk", "wikipedia", 0.9)]},
    )
    results = r.query("who was Tesla", collection_names=["wikipedia"])
    assert [x["text"] for x in results] == ["good"]


def test_relevance_gate_can_empty_out(monkeypatch):
    r = _retriever_with_hits(monkeypatch, {"wikipedia": [_hit("junk", "wikipedia", 0.95)]})
    assert r.query("gibberish", collection_names=["wikipedia"]) == []


def test_excluded_collections_never_queried(monkeypatch):
    r = _retriever_with_hits(
        monkeypatch,
        {
            "wikipedia": [_hit("fact", "wikipedia", 0.2)],
            "music": [_hit("track row", "music", 0.05)],
        },
    )
    monkeypatch.setattr(settings, "rag_exclude_collections", "music")
    results = r.query("who was Tesla", collection_names=["music", "wikipedia"])
    assert [x["source"] for x in results] == ["wikipedia"]


def test_rerank_kill_switch(monkeypatch):
    r = _retriever_with_hits(monkeypatch, {"wikipedia": [_hit("fact", "wikipedia", 0.2)]})
    monkeypatch.setattr(settings, "rag_rerank_enabled", False)
    boom = MagicMock(side_effect=AssertionError("reranker must not be built"))
    monkeypatch.setattr(r, "_get_reranker", boom)
    results = r.query("tell me more", collection_names=["wikipedia"], mode="deep")
    assert results and not boom.called


def test_format_context_includes_title(monkeypatch):
    r = Retriever(embedder=MagicMock())
    ctx = r.format_context(
        [
            _hit("Tesla was born in 1856.", "wikipedia", 0.2, title="Nikola Tesla").to_dict(),
            _hit("No title here.", "gutenberg", 0.3).to_dict(),
        ]
    )
    assert "[Source 1: wikipedia — Nikola Tesla]" in ctx
    assert "[Source 2: gutenberg (a book from before 1930" in ctx


def test_followup_detection():
    from oracle.core import _needs_rewrite

    assert _needs_rewrite("where did he die?")
    assert _needs_rewrite("more about that")
    assert _needs_rewrite("why?")  # very short
    assert not _needs_rewrite("describe the construction of medieval aqueducts in detail")


def test_format_context_truncates_long_chunks(monkeypatch):
    monkeypatch.setattr(settings, "rag_chunk_char_limit", 100)
    r = Retriever(embedder=MagicMock())
    long_text = "word " * 100
    ctx = r.format_context([_hit(long_text, "wikipedia", 0.2).to_dict()])
    body = ctx.split("]\n", 1)[1]
    assert len(body) < 150
    assert "…" in body


def test_faiss_only_config_never_touches_chroma_or_torch(monkeypatch):
    """With every collection on FAISS, listing collections must not build
    a Chroma client, and constructing the retriever must not build the
    default (torch-probing) embedder."""
    from config.settings import settings
    from oracle.rag import retriever as mod

    monkeypatch.setattr(settings, "collection_backends", {"wikipedia": "faiss", "music": "faiss"})
    monkeypatch.setattr(
        mod, "Embedder", lambda *a, **k: (_ for _ in ()).throw(AssertionError("embedder built"))
    )
    r = mod.Retriever()
    r._get_client = lambda: (_ for _ in ()).throw(AssertionError("chroma client built"))
    assert r.list_collections() == ["music", "wikipedia"]


def test_bias_lets_wikimed_beat_a_closer_gutenberg_hit_on_medical_questions(monkeypatch):
    r = _retriever_with_hits(
        monkeypatch,
        {
            "gutenberg": [_hit("1900s manual", "gutenberg", 0.11)],
            "wikimed": [_hit("modern", "wikimed", 0.15)],
            "wikipedia": [_hit("encyclopedia", "wikipedia", 0.16)],
        },
    )
    monkeypatch.setattr(settings, "tier1_top_k", 3)
    results = r.query("How do I treat a second-degree burn?")
    assert [x["source"] for x in results][:2] == ["wikimed", "wikipedia"]
    monkeypatch.setattr(settings, "rag_collection_bias", False)
    results = r.query("How do I treat a second-degree burn?")
    assert [x["source"] for x in results][0] == "gutenberg"


def test_diversity_cap_keeps_a_second_collection_in(monkeypatch):
    r = _retriever_with_hits(
        monkeypatch,
        {
            "wikipedia": [
                _hit("w1", "wikipedia", 0.10),
                _hit("w2", "wikipedia", 0.11),
                _hit("w3", "wikipedia", 0.12),
            ],
            "wikibooks": [_hit("b1", "wikibooks", 0.20)],
        },
    )
    monkeypatch.setattr(settings, "tier1_top_k", 3)
    results = r.query("Explain supply and demand.")
    assert [x["source"] for x in results] == ["wikipedia", "wikipedia", "wikibooks"]


def test_gate_still_applies_to_raw_distance(monkeypatch):
    r = _retriever_with_hits(
        monkeypatch,
        {"wikimed": [_hit("far", "wikimed", 0.40)], "gutenberg": [_hit("near", "gutenberg", 0.20)]},
    )
    results = r.query("How do I treat a burn?")
    assert [x["source"] for x in results] == ["gutenberg"]
