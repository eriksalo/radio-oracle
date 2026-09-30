"""Tests for the turn pipeline pieces in oracle.core: follow-up rewrite
heuristics, concurrent retrieval, and the speech splitter."""

from __future__ import annotations

import asyncio

import pytest

from config.settings import settings
from oracle import core
from oracle.core import SpeechSplitter, _needs_rewrite, _token_overlap

# ------------------------------------------------------------ heuristics


@pytest.mark.parametrize(
    "text",
    [
        "Where did he die?",
        "What is it made of?",
        "Can you see them from Colorado?",
        "What about the other one?",
        "Tell me more",
        "And after that?",
        "Why is that?",
        "How come?",
    ],
)
def test_rewrite_triggers_on_followups(text):
    assert _needs_rewrite(text, has_prior_answer=True)


@pytest.mark.parametrize(
    "text",
    [
        "Who wrote Pride and Prejudice?",
        "What causes the northern lights?",
        "How does a vacuum tube amplify a signal?",
        "What is the boiling point of water at high altitude?",
        "Tell me about the whaling ship in Moby Dick.",
        "How do I treat a second-degree burn?",
    ],
)
def test_rewrite_skips_self_contained_questions(text):
    assert not _needs_rewrite(text, has_prior_answer=True)


def test_rewrite_never_fires_without_prior_answer():
    assert not _needs_rewrite("Where did he die?", has_prior_answer=False)


def test_token_overlap():
    assert _token_overlap("where did he die", "where did he die") == 1.0
    assert _token_overlap("where did he die", "Where did Nikola Tesla die?") < 0.8
    assert _token_overlap("", "x") == 0.0


# --------------------------------------------------- concurrent retrieval


class _Store:
    def __init__(self, messages):
        self._m = messages

    def get_messages(self, session_id, limit=None):
        return self._m[-limit:] if limit else self._m


@pytest.mark.asyncio
async def test_retrieve_for_turn_runs_rewrite_and_raw_retrieval_concurrently(monkeypatch):
    order: list[str] = []
    calls: list[str] = []

    async def fake_chat(messages, model=None, num_predict=None):
        order.append("rewrite-start")
        await asyncio.sleep(0.05)
        order.append("rewrite-end")
        assert num_predict == settings.ollama_rewrite_num_predict
        return "Where did Nikola Tesla die?"

    def fake_rag(text):
        calls.append(text)
        order.append(f"rag:{text}")
        return f"CTX[{text}]"

    monkeypatch.setattr(core, "chat", fake_chat)
    monkeypatch.setattr(core, "_try_rag_query", fake_rag)
    monkeypatch.setattr(settings, "rag_query_rewrite", True)

    store = _Store(
        [
            {"role": "user", "content": "Who was Nikola Tesla?"},
            {"role": "assistant", "content": "An inventor."},
            {"role": "user", "content": "Where did he die?"},
        ]
    )
    ctx = await core._retrieve_for_turn(store, "s", "Where did he die?")
    # Raw retrieval must not wait for the rewrite to finish.
    assert order.index("rag:Where did he die?") < order.index("rewrite-end")
    # The rewrite changed the query materially → second retrieval wins.
    assert calls == ["Where did he die?", "Where did Nikola Tesla die?"]
    assert ctx == "CTX[Where did Nikola Tesla die?]"


@pytest.mark.asyncio
async def test_retrieve_for_turn_keeps_raw_when_rewrite_is_noop(monkeypatch):
    calls: list[str] = []

    async def fake_chat(messages, model=None, num_predict=None):
        return "where did he die"

    monkeypatch.setattr(core, "chat", fake_chat)
    monkeypatch.setattr(core, "_try_rag_query", lambda t: calls.append(t) or "ctx")
    monkeypatch.setattr(settings, "rag_query_rewrite", True)
    store = _Store(
        [
            {"role": "assistant", "content": "An inventor."},
            {"role": "user", "content": "Where did he die?"},
        ]
    )
    assert await core._retrieve_for_turn(store, "s", "Where did he die?") == "ctx"
    assert calls == ["Where did he die?"]


@pytest.mark.asyncio
async def test_retrieve_for_turn_skips_llm_for_plain_questions(monkeypatch):
    async def boom(*a, **k):
        raise AssertionError("rewrite must not run")

    monkeypatch.setattr(core, "chat", boom)
    monkeypatch.setattr(core, "_try_rag_query", lambda t: "ctx")
    monkeypatch.setattr(settings, "rag_query_rewrite", True)
    store = _Store([{"role": "assistant", "content": "x"}, {"role": "user", "content": "q"}])
    assert await core._retrieve_for_turn(store, "s", "Who wrote Pride and Prejudice?") == "ctx"


# --------------------------------------------------------- speech splitter


def _feed_all(sp: SpeechSplitter, text: str, chunk: int = 3) -> list[str]:
    out: list[str] = []
    for i in range(0, len(text), chunk):
        out.extend(sp.feed(text[i : i + chunk]))
    tail = sp.flush()
    if tail:
        out.append(tail)
    return out


def test_splitter_clauses_and_sentences_are_units():
    sp = SpeechSplitter(clause_min_words=3, soft_cut_words=8, hard_cut_words=12)
    text = (
        "Nikola Tesla was a Serbian-American inventor, best known for AC power. "
        "He died in New York. Poor and alone."
    )
    assert _feed_all(sp, text) == [
        "Nikola Tesla was a Serbian-American inventor,",
        "best known for AC power.",
        "He died in New York.",
        "Poor and alone.",
    ]


def test_splitter_short_clause_below_minimum_waits():
    sp = SpeechSplitter(clause_min_words=3, soft_cut_words=8, hard_cut_words=12)
    # "Yes," is one word: not worth a unit; the sentence closes it.
    assert _feed_all(sp, "Yes, he did. Twice, in fact.") == ["Yes, he did.", "Twice, in fact."]


def test_splitter_does_not_split_numbers_or_abbreviations():
    sp = SpeechSplitter(clause_min_words=3, soft_cut_words=8, hard_cut_words=12)
    units = _feed_all(sp, "About 1,000 feet up, e.g. Denver, water boils lower.")
    assert units[0] == "About 1,000 feet up,"


def test_splitter_disabled_means_sentences_only():
    sp = SpeechSplitter(clause_min_words=0)
    long = "One, two, three and four " * 4 + "end. Five."
    assert _feed_all(sp, long) == [long[: long.index("end.") + 4], "Five."]


def test_splitter_soft_cuts_long_units_before_conjunctions_and_prepositions():
    sp = SpeechSplitter(clause_min_words=3, soft_cut_words=8, hard_cut_words=12)
    text = (
        "A vacuum tube amplifies a signal by using a heated filament "
        "to emit electrons into a vacuum."
    )
    units = _feed_all(sp, text)
    # 8 complete words, then the next cut word ("by" is word 7 → too early;
    # "to" at index 11) closes the unit.
    assert units[0] == "A vacuum tube amplifies a signal by using a heated filament"
    assert units[1] == "to emit electrons into a vacuum."
    assert " ".join(units) == text


def test_splitter_hard_cuts_at_word_limit():
    sp = SpeechSplitter(clause_min_words=3, soft_cut_words=8, hard_cut_words=12)
    text = " ".join(f"w{i}" for i in range(30)) + "."
    units = _feed_all(sp, text)
    assert [len(u.split()) for u in units] == [12, 12, 6]
    assert " ".join(units) == text


def test_splitter_never_cuts_mid_word():
    sp = SpeechSplitter(clause_min_words=0, soft_cut_words=3, hard_cut_words=4)
    sp._min_words = 1  # enable cuts without a clause minimum
    out = sp.feed("one two three four fi")
    # "fi" may be the start of "five": the cut lands before it.
    assert out == ["one two three four"]
    assert sp.feed("ve six. ") == ["five six."]


def test_splitter_flush_returns_tail_once():
    sp = SpeechSplitter()
    assert sp.feed("no punctuation here") == []
    assert sp.flush() == "no punctuation here"
    assert sp.flush() == ""


# ---------------------------------------------------------------- hygiene


def test_strip_foreign_drops_cjk_run_and_keeps_latin():
    from oracle.core import strip_foreign

    assert strip_foreign("Soap doesn't kill germs—它 works by breaking") == (
        "Soap doesn't kill germs— works by breaking"
    )
    assert strip_foreign("Café déjà vu, naïve π") == "Café déjà vu, naïve π"
    assert strip_foreign("") == ""


def test_trim_to_sentence_end():
    from oracle.core import trim_to_sentence_end

    assert trim_to_sentence_end("One. Two. He doesn't claim immortality,") == "One. Two."
    assert trim_to_sentence_end("Complete sentence.") == "Complete sentence."
    assert trim_to_sentence_end("Ends with a quote.”") == "Ends with a quote.”"
    assert trim_to_sentence_end("no terminator at all") == "no terminator at all"


def test_units_beyond_marks_the_unfinished_tail():
    from oracle.core import units_beyond

    units = ["He treats death as natural,", "like birth.", "He doesn't claim immortality,"]
    kept = "He treats death as natural, like birth."
    assert units_beyond(units, kept) == {"He doesn't claim immortality,"}
    assert units_beyond(units, " ".join(units)) == set()


def test_clean_for_speech_drops_foreign_script():
    from oracle.tts import clean_for_speech

    assert clean_for_speech("germs—它 works") == "germs— works"
