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


# ---------------------------------------------------------- question weighting


def test_question_type_classes():
    from oracle.rag.router import question_type

    assert question_type("How do I treat a second-degree burn?") == "medical"
    assert question_type("What are the signs of dehydration?") == "medical"
    assert question_type("How do I fix a flat bicycle tire?") == "howto"
    assert question_type("How do you grow potatoes?") == "howto"
    assert question_type("What happens at the end of A Tale of Two Cities?") == "literature"
    assert question_type("Who were the three musketeers?") == "literature"
    assert question_type("Who was Nikola Tesla?") == "factual"
    assert question_type("Explain supply and demand.") == "general"


def test_bias_pushes_gutenberg_back_except_for_literature():
    from oracle.rag.router import bias_for

    assert bias_for("How do I treat a burn?")["gutenberg"] > 0.1
    assert bias_for("How do I treat a burn?")["wikimed"] < 0
    assert bias_for("How do I fix a flat tire?")["ifixit"] < 0
    assert bias_for("What happens at the end of Moby Dick?")["gutenberg"] < 0
    assert bias_for("Who was Tesla?")["wikipedia"] < 0


def test_augment_only_plot_questions():
    from oracle.rag.router import augment_query

    q = "What happens at the end of A Tale of Two Cities?"
    assert augment_query(q).startswith(q) and "plot" in augment_query(q)
    assert augment_query("Who wrote The Time Machine?") == "Who wrote The Time Machine?"
    assert augment_query("How do I treat a burn?") == "How do I treat a burn?"
