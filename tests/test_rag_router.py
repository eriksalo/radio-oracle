"""Tests for the query → collection-priority router."""

from __future__ import annotations

from oracle.rag.router import route

AVAILABLE = ["crashcourse", "wikipedia", "wikimed", "wikibooks", "gutenberg", "ifixit"]


def test_medical_question_routes_wikimed_first():
    r = route("What's the best treatment for a snake bite?", available=AVAILABLE)
    assert r.order[0] == "wikimed"
    assert "wikimed" in r.matched


def test_repair_question_routes_ifixit_first():
    r = route("How do I repair my dishwasher?", available=AVAILABLE)
    assert r.order[0] == "ifixit"


def test_factual_question_routes_wikipedia_first():
    r = route("Who was Augustus Caesar?", available=AVAILABLE)
    assert r.order[0] == "wikipedia"


def test_literary_question_routes_gutenberg_first():
    r = route("Find me a quote from Shakespeare about love.", available=AVAILABLE)
    assert r.order[0] == "gutenberg"


def test_unmatched_query_uses_default_order():
    r = route("foo bar baz", available=AVAILABLE)
    assert r.matched == []
    # Default order puts wikipedia first
    assert r.order[0] == "wikipedia"


def test_unavailable_matched_collection_drops():
    r = route("medical question about infections", available=["wikipedia", "ifixit"])
    # wikimed was matched but isn't in `available` so it must be dropped
    assert "wikimed" not in r.order


def test_no_available_collections():
    r = route("anything", available=[])
    assert r.order == []


def test_router_preserves_all_available():
    r = route("How do I fix a broken bone?", available=AVAILABLE)
    # Even though "how do I fix" matches ifixit AND "broken bone" hints at
    # wikimed, every available collection should still appear in the order.
    assert set(r.order) == set(AVAILABLE)
