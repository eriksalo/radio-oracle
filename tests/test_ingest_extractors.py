"""Page cleaners of scripts/ingest_sources.py (web pages and wiki articles)."""

from __future__ import annotations

import pytest

pytest.importorskip("selectolax")

from selectolax.lexbor import LexborHTMLParser  # noqa: E402

from scripts import ingest_sources as ingest  # noqa: E402

PROSE = " ".join(["Keep three days of water for every person in the house."] * 12)


def _page(body: str, lang: str = "en") -> LexborHTMLParser:
    return LexborHTMLParser(f"<html lang='{lang}'><body>{body}</body></html>")


def test_merge_short_joins_neighbours_until_the_floor() -> None:
    short = "word " * 30
    merged = ingest._merge_short([short, short, short, "word " * 120, short], floor=90)
    assert len(merged) == 2
    assert all(len(section.split()) >= 90 for section in merged)


def test_merge_short_keeps_a_lone_short_section() -> None:
    assert ingest._merge_short(["only a few words"]) == ["only a few words"]


def test_webpage_keeps_main_text_and_drops_chrome() -> None:
    tree = _page(
        "<nav><p>Home</p></nav><main><h1>Water</h1>"
        f"<p>{PROSE}</p><p>Return to top</p><p>Español</p></main><footer><p>Contact</p></footer>"
    )
    doc = ingest.extract_webpage("example.org/water", "Water | Example", tree)
    assert doc is not None
    title, sections = doc
    assert title == "Water"
    text = " ".join(sections)
    assert "three days of water" in text
    assert "Home" not in text and "Contact" not in text
    assert "Return to top" not in text and "Español" not in text


def test_webpage_skips_other_languages_search_urls_and_hubs() -> None:
    body = f"<main><h1>Agua</h1><p>{PROSE}</p></main>"
    assert ingest.extract_webpage("example.org/es/agua", "Agua", _page(body, "es")) is None
    assert ingest.extract_webpage("example.org/search?q=water", "Water", _page(body)) is None
    hub = _page("<main><h1>Topics</h1><p>Water</p><p>Food</p></main>")
    assert ingest.extract_webpage("example.org/topics", "Topics", hub) is None


def test_webpage_uses_the_heading_when_the_title_is_the_site_name() -> None:
    tree = _page(f"<main><h1>buckwheat tea</h1><p>{PROSE}</p></main>")
    doc = ingest.extract_webpage("grimgrains.com/site/buckwheat_tea.html", "GrimGrains", tree)
    assert doc is not None and doc[0] == "Buckwheat tea"


def test_mediawiki_drops_references_translations_and_namespaces() -> None:
    body = (
        f"<div id='mw-content-text'><p>{PROSE}</p><h2>Building it</h2><p>{PROSE}</p>"
        "<h2>References</h2><p>Smith 1999, a book nobody will hear read aloud.</p>"
        "<table><tr><td>cell</td></tr></table></div>"
    )
    doc = ingest.extract_mediawiki("Rainwater_tank/en", "Rainwater tank/en", _page(body))
    assert doc is not None
    title, sections = doc
    assert title == "Rainwater tank"
    text = " ".join(sections)
    assert "Building it." in text
    assert "Smith 1999" not in text and "cell" not in text
    assert ingest.extract_mediawiki("Rainwater_tank/fr", "Rainwater tank/fr", _page(body)) is None
    assert ingest.extract_mediawiki("Category:Water", "Category:Water", _page(body)) is None
    assert ingest.extract_mediawiki("List_of_pumps", "List of pumps", _page(body)) is None


def test_medlineplus_topic_page_keeps_only_the_summary() -> None:
    body = (
        f"<article><div id='topic-summary'><p>{PROSE}</p></div>"
        "<h2>Start Here</h2><p>Abscess (Some Clinic) Also in Spanish</p></article>"
    )
    doc = ingest.extract_medlineplus("medlineplus.gov/abscess.html", "Abscess | Boils", _page(body))
    assert doc is not None
    assert doc[0] == "Abscess"
    assert "Some Clinic" not in " ".join(doc[1])
    gene = ingest.extract_medlineplus("medlineplus.gov/genetics/gene/npm1/", "NPM1", _page(body))
    assert gene is None
