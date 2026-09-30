"""Routing fixes from the 2026-09-30 quality eval: catalog questions are
exploration commands, "can you play X" is a request, "song by artist" and
tag spelling resolve, and whole-word matching beats substrings."""

from __future__ import annotations

import sqlite3

import pytest

from oracle import commands
from oracle.music.catalog import Catalog, norm


class _VC:
    class _STT:
        def load(self):
            pass

        def unload(self):
            pass

    stt_fast = _STT()


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    async def boom(text):
        return ("llm", text)

    monkeypatch.setattr(commands, "_llm_intent", boom)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,action,query",
    [
        ("Do you have any Springsteen?", "llm", None),  # channel unknown → LLM, not oracle
        ("Any albums by the Beatles?", "list_music", "the Beatles"),
        ("What albums are there by Pink Floyd?", "list_music", "Pink Floyd"),
        ("What kind of music is there?", "list_music", None),
        ("Is there any jazz?", "list_music", "jazz"),
        ("What songs are there by Greg Brown?", "list_music", "Greg Brown"),
        ("Do you have anything by Tolstoy?", "llm", None),
        ("Do you have any books by Tolstoy?", "list_books", "Tolstoy"),
        ("Which books by H. G. Wells do you have?", "list_books", "H. G. Wells"),
        ("How many books do you have?", "list_books", None),
        ("Can you play Mark Knopfler?", "play", "Mark Knopfler"),
        ("Could you put on some Leonard Cohen?", "play", "Leonard Cohen"),
        ("Turn the radio on.", "music_on", None),
        ("I don't like this one.", "next", None),
        ("Skip to the last chapter.", "goto_chapter", None),
        ("Go back to the last chapter.", "prev_chapter", None),
        ("Who wrote Blowin' in the Wind?", "question", None),
        ("Why is the sky blue?", "question", None),
    ],
)
async def test_classify_routes(text, action, query):
    got_action, got_query = await commands.classify(_VC(), text)
    if action == "llm":
        assert got_action == "llm", f"{text!r} should reach the LLM intent step"
    else:
        assert (got_action, got_query) == (action, query)


def test_qualifier_stops_before_trailing_question():
    assert commands._extract_qualifier("Which books by H. G. Wells do you have?") == "H. G. Wells"
    assert commands._extract_qualifier("what music by Greg Brown is there") == "Greg Brown"
    assert commands._extract_qualifier("play music by Bob Dylan.") == "Bob Dylan"


def test_chapter_spec_ordinals():
    assert commands._extract_chapter_spec("Skip to the last chapter.") == "last"
    assert commands._extract_chapter_spec("Go to the final chapter") == "last"
    assert commands._extract_chapter_spec("go to chapter twenty one") == "twenty one"


def test_extract_play_object():
    assert commands._extract_play_object("Can you play Mark Knopfler?") == "Mark Knopfler"
    assert commands._extract_play_object("Would you play some Aerosmith please") == "Aerosmith"


# ---------------------------------------------------------------- catalog


def _catalog(tmp_path) -> Catalog:
    c = Catalog(db_path=tmp_path / "music.db")
    rows = [
        ("Pink Floyd", "The Dark Side of the Moon", "Money", "Progressive Rock"),
        ("Pink Floyd", "Wish You Were Here", "Wish You Were Here", "Progressive Rock"),
        ("Bob Dylan", "Freewheelin'", "Blowin' in the Wind", "Folk"),
        ("Dylan", "Live", "Hurricane", "Folk"),
        ("Bryan Adams", "Reckless", "Summer of '69", "Rock"),
        ("Ryan Adams & The Cardinals", "Cold Roses", "Magnolia Mountain", "Rock"),
        ("Robert Plant & Alison Krauss", "Raising Sand", "Killing the Blues", "Folk"),
        ("Paul McCartney", "Ram", "Maybe I’m Amazed", "Rock"),
        ("Johnny Cash", "American IV", "Hurt", "Country"),
        ("Nine Inch Nails", "The Downward Spiral", "Hurt", "Rock"),
        ("Neil Young", "Harvest", "Heart of Gold", "Folk"),
        ("Barclay James Harvest", "Octoberon", "Rock 'n' Roll Star", "Progressive Rock"),
        ("The Beatles", "Let It Be", "Let It Be", "Rock"),
        ("Everly Brothers", "Greatest", "Let It Be Me", "Pop"),
        ("Moody Blues", "Days of Future Passed", "Nights in White Satin", "Rock"),
        ("Muddy Waters", "Folk Singer", "My Home Is in the Delta", "Blues"),
        ("Celine Dion", "Falling into You", "Because You Loved Me", "Pop"),
        ("Dion DiMucci", "Runaround Sue", "Runaround Sue", "Pop, Rock"),
    ]
    conn = sqlite3.connect(tmp_path / "music.db")
    for i, (artist, album, title, genre) in enumerate(rows):
        conn.execute(
            "INSERT INTO tracks (track_id, title, artist, album, genre, duration_sec, filename,"
            " filepath_rel) VALUES (?, ?, ?, ?, ?, 180, ?, ?)",
            (f"t{i}", title, artist, album, genre, f"{i}.mp3", f"{artist}/{i}.mp3"),
        )
    conn.commit()
    conn.close()
    return c


def test_norm_folds_tag_spelling():
    assert norm("Robert Plant & Alison Krauss") == norm("Robert Plant and Alison Krauss")
    assert norm("Maybe I’m Amazed") == norm("maybe i'm amazed") == "maybe im amazed"
    assert norm("Les Misérables") == "les miserables"


@pytest.mark.parametrize(
    "query,hint,artists,tier",
    [
        ("Ryan Adams", None, {"Ryan Adams & The Cardinals"}, "artist"),
        ("Dylan", None, {"Bob Dylan", "Dylan"}, "artist"),
        ("Robert Plant and Alison Krauss", None, {"Robert Plant & Alison Krauss"}, "artist"),
        ("Maybe I'm Amazed", None, {"Paul McCartney"}, "title"),
        ("Hurt by Johnny Cash", None, {"Johnny Cash"}, "title+artist"),
        ("Harvest", "album", {"Neil Young"}, "album"),
        ("Harvest", None, {"Barclay James Harvest"}, "artist"),
        ("Let It Be", None, {"The Beatles"}, "album"),  # exact album and title; album tier first
        ("blues", None, {"Muddy Waters"}, "genre"),
        (
            "rock",
            None,
            {
                "Bryan Adams",
                "Ryan Adams & The Cardinals",
                "Paul McCartney",
                "Nine Inch Nails",
                "The Beatles",
                "Moody Blues",
                "Pink Floyd",
                "Barclay James Harvest",
                "Dion DiMucci",
            },
            "genre",
        ),
        ("Dion", None, {"Celine Dion", "Dion DiMucci"}, "artist"),
        ("Pink Floyd", None, {"Pink Floyd"}, "artist"),
        ("nothing here", None, set(), "none"),
    ],
)
def test_search_ranked_tiers(tmp_path, query, hint, artists, tier):
    c = _catalog(tmp_path)
    try:
        hits, got_tier = c.search_ranked(query, hint=hint)
        assert got_tier == tier
        assert {t.artist for t in hits} == artists
    finally:
        c.close()
