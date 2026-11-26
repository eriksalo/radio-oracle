"""Shelves: the library sorted by subject instead of by source.

The collections are sources (Wikipedia, WikiMed, iFixit…); a person asks
by subject. A shelf is one of seven subjects. It does four things:

* a question is put on a shelf (:func:`classify` — keywords first, then
  the nearest shelf by embedding when an embedder is already resident);
* the shelf says which sources to trust when hits are merged
  (:func:`bias`, the finer successor of ``router.bias_for``);
* conversation threads carry their shelf, so "what have we talked about
  on the health shelf" works (``oracle/memory/threads.py``);
* the shelves can be browsed by voice (:func:`list_text`,
  :func:`describe_text`) and an answer's sources can be named
  (:func:`sources_text`).

Music and the books read aloud are not shelves: they are played, not
looked up. No heavy imports here; the embedder is handed in.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock

import numpy as np
from loguru import logger

from oracle.rag.router import _LITERATURE_RE, _MEDICAL_RE

EmbedFn = Callable[[str], object]

# nomic task prefix for the nearest-shelf test (never feeds FAISS).
EMBED_PREFIX = "classification: "


@dataclass(frozen=True)
class Shelf:
    key: str
    name: str  # as spoken: "the … shelf"
    blurb: str  # what is on it, spoken
    aliases: re.Pattern[str]  # how a person names the shelf
    keywords: re.Pattern[str]  # words that put a question on it
    examples: tuple[str, ...]  # typical questions; the first is offered aloud
    bias: dict[str, float]  # per-collection distance bias (see router._BIAS)


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE)


SHELVES: tuple[Shelf, ...] = (
    Shelf(
        key="health",
        name="health and first aid",
        blurb=(
            "Injuries, illness, medicines and the body, from the WikiMed medical reference and "
            "the NHS medicines guide."
        ),
        aliases=_rx(r"health|medical|medicine|first[- ]aid|doctor(?:ing)?"),
        keywords=re.compile(
            _MEDICAL_RE.pattern
            + r"|\b(?:swollen|sore|aches?|aching|stings?|stung|bites?|bitten|cough(?:s|ing)?|"
            r"a cold|the flu|headaches?|migraines?|blood pressure|aspirin|ibuprofen|"
            r"paracetamol|bandages?|stitches|bruises?|infected|sick|illness|unconscious|"
            r"heart attack|breathing|pulse|medicines?|pills?|sunburn|earache|toothache)\b",
            re.IGNORECASE,
        ),
        examples=(
            "How do I treat a second-degree burn?",
            "What are the signs of dehydration?",
            "How do I splint a broken arm?",
            "What is the right dose of ibuprofen for an adult?",
            "How do I know if a cut is infected?",
            "What should I do for a high fever?",
            "How do I stop heavy bleeding?",
            "What causes migraines?",
            "How long does the flu usually last?",
            "How do I wash out and dress a wound?",
            "What do I do about a wasp sting or a spider bite?",
            "Why does my back hurt when I get up?",
        ),
        bias={
            "wikimed": -0.04,
            "wikipedia": 0.0,
            "wikibooks": 0.02,
            "crashcourse": 0.02,
            "ifixit": 0.08,
            "gutenberg": 0.12,
            # The practical guides (config/knowledge_sources.json).
            "guides_health": -0.04,
            "guides_outdoors": 0.02,
            "guides_food": 0.06,
            "guides_workshop": 0.08,
        },
    ),
    Shelf(
        key="survival",
        name="survival and outdoors",
        blurb=(
            "Shelter, water, fire, finding your way and the weather, from the encyclopedia, "
            "the textbooks and people's answers to outdoors questions."
        ),
        aliases=_rx(r"survival|outdoors?|wilderness|camping|bushcraft|emergenc(?:y|ies)"),
        keywords=_rx(
            r"surviv(?:e|al|ing)|wilderness|shelter|campfire|start(?:ing)? a fire|fire ?starter|"
            r"tinder|kindling|purif(?:y|ying) water|water purification|drinking water|"
            r"boil(?:ing)? water|navigat(?:e|ing|ion)|compass|find(?:ing)? north|north star|"
            r"lost in the (?:woods|forest|desert|mountains)|signal for help|rescue|"
            r"knots?|rope|snare|trap(?:ping)?|forag(?:e|ing)|edible (?:plants?|berries|mushrooms)|"
            r"blizzard|storm|tornado|hurricane|flood|earthquake|wildfire|avalanche|lightning|"
            r"bear|wolves|tent|camp(?:ing|site)?|hik(?:e|ing)|backpack(?:ing)?|canoe|"
            r"weather|clouds? mean|barometer|frost tonight|snow cave|lean-to|bug[- ]out|"
            r"power (?:is|goes) out|power outage|blackout"
        ),
        examples=(
            "How do I make water safe to drink?",
            "How do I start a fire without matches?",
            "How do I find north without a compass?",
            "How do I build a shelter in the snow?",
            "What should I do if I get lost in the woods?",
            "Which knots are good for tying down a tarp?",
            "How can I tell that a storm is coming?",
            "What do I do when a bear comes into camp?",
            "How do I signal for rescue?",
            "How do I catch and store rain for drinking?",
            "Which wild plants and mushrooms can I eat?",
            "What supplies should I keep for an emergency?",
        ),
        bias={
            "wikibooks": -0.03,
            "wikipedia": -0.02,
            "wikimed": 0.0,
            "ifixit": 0.03,
            "crashcourse": 0.03,
            "gutenberg": 0.06,
            # The practical guides (config/knowledge_sources.json).
            "guides_outdoors": -0.04,
            "guides_health": 0.0,
            "guides_food": 0.02,
            "guides_workshop": 0.03,
        },
    ),
    Shelf(
        key="food",
        name="food and growing",
        blurb=(
            "Cooking, baking, keeping food, the garden and animals, from the cookbook "
            "textbooks, a recipe collection and the encyclopedia."
        ),
        aliases=_rx(r"food|cook(?:ing|ery)?|recipes?|growing|garden(?:ing)?|farm(?:ing)?|kitchen"),
        keywords=_rx(
            r"cook(?:s|ed|ing)?|bak(?:e|es|ed|ing)|recipes?|bread|sourdough|starter|yeast|dough|"
            r"flour|ferment(?:s|ed|ing|ation)?|pickl(?:e|es|ed|ing)|canning|preserv(?:e|es|ing)|"
            r"smok(?:e|ing) (?:meat|fish)|cur(?:e|ing) meat|jerky|"
            r"dehydrat(?:e|ing) (?:food|fruit|meat)|"
            r"brew(?:s|ed|ing)?|beer|wine|cider|cheese|butter|yogurt|stew|soup|roast|"
            r"garden(?:s|ing)?|grow(?:s|ing)? (?:\w+ )?"
            r"(?:tomato|potato|bean|corn|vegetable|herb|"
            r"fruit|carrot|squash|pepper|lettuce|onion|garlic|wheat)\w*|"
            r"tomato(?:es)?|potato(?:es)?|vegetables?|seeds?|seedlings?|compost|soil|fertili[sz]er|"
            r"harvest(?:s|ed|ing)?|crops?|orchard|prun(?:e|ing)|"
            r"chickens?|hens?|eggs?|goats?|cows?|milk(?:ing)?|bees?|beekeeping|honey|livestock|"
            r"butcher(?:ing)?|root cellar"
        ),
        examples=(
            "How do I make a sourdough starter?",
            "When should I plant tomatoes?",
            "How do I can vegetables safely?",
            "How do I keep chickens through the winter?",
            "How long does bread dough need to rise?",
            "What makes good compost?",
            "How do I store potatoes for the winter?",
            "How do I make cheese from milk?",
            "How do I dry fruit to keep it?",
            "How do I brew cider or make wine?",
            "How long should I cook rice?",
            "How do I raise goats for milk?",
        ),
        bias={
            "wikibooks": -0.03,
            "wikipedia": -0.01,
            "wikimed": 0.03,
            "crashcourse": 0.03,
            "ifixit": 0.06,
            "gutenberg": 0.08,
            # The practical guides (config/knowledge_sources.json).
            "guides_food": -0.04,
            "guides_outdoors": 0.02,
            "guides_health": 0.03,
            "guides_workshop": 0.06,
        },
    ),
    Shelf(
        key="fixing",
        name="fixing and building",
        blurb=(
            "Repairs, tools, engines, plumbing, wiring and woodwork, from the iFixit repair "
            "guides and the textbooks."
        ),
        aliases=_rx(
            r"fixing|fix[- ]it|repairs?|repairing|building|tools?|mechanics?|workshop|how[- ]to"
        ),
        keywords=_rx(
            r"fix(?:es|ed|ing)?|repair(?:s|ed|ing)?|broken|troubleshoot(?:ing)?|"
            r"won'?t (?:start|turn on|"
            r"run|work|charge)|leak(?:s|ing|y)?|plumbing|pipes?|faucet|valve|well pump|pump|"
            r"pressure (?:tank|switch|gauge)|psi|water heater|furnace|wood ?stove|chimney|"
            r"wir(?:e|ing)|solder(?:ing)?|circuit|fuse|breaker|outlet|multimeter|generator|"
            r"batter(?:y|ies)|solar panels?|inverter|engine|carburetor|spark plug|chainsaw|"
            r"lawn ?mower|tractor|truck|car|brakes?|tires?|oil change|bicycle|bike chain|"
            r"weld(?:ing)?|drill|saw|hammer|chisel|sharpen(?:ing)?|lathe|"
            r"carpentry|woodwork(?:ing)?|lumber|roof(?:ing)?|shingles?|concrete|mortar|"
            r"insulat(?:e|ion)|radio|antenna|amplifier|laptop|phone|drain|hinges?|toilet"
        ),
        examples=(
            "Why does my well pump keep turning on and off?",
            "How do I replace a phone battery?",
            "How do I fix a leaking faucet?",
            "Why won't my chainsaw start?",
            "How do I solder two wires together?",
            "How do I sharpen a chisel?",
            "How do I patch a hole in a roof?",
            "How do I wire a light switch?",
            "How do I build a set of shelves?",
            "Why does my car engine overheat?",
            "How do I glue a broken chair leg?",
            "How do I stop a toilet from running?",
        ),
        bias={
            # -0.04 pulled anaesthesia-machine guides in for a pressure-tank
            # question (device probe, 2026-11-25); the router's how-to value.
            "ifixit": -0.03,
            "wikibooks": -0.02,
            "wikipedia": 0.0,
            "crashcourse": 0.03,
            "wikimed": 0.06,
            "gutenberg": 0.08,
            # The practical guides (config/knowledge_sources.json).
            "guides_workshop": -0.03,
            "guides_outdoors": 0.02,
            "guides_food": 0.06,
            "guides_health": 0.06,
        },
    ),
    Shelf(
        key="science",
        name="science and learning",
        blurb=(
            "How the world works: physics, chemistry, biology, the sky, mathematics, from "
            "the Crash Course lessons, the textbooks and the encyclopedia."
        ),
        aliases=_rx(
            r"science|sciences|learning|math(?:s|ematics)?|physics|chemistry|biology|"
            r"astronomy|school|lessons?"
        ),
        keywords=_rx(
            r"science|scientific|physics|chemistry|chemical|biology|astronomy|geology|"
            r"math(?:s|ematics)?|algebra|geometry|calculus|equation|theorem|prime numbers?|"
            r"atoms?|molecules?|electrons?|protons?|neutrons?|elements?|periodic table|"
            r"gravity|energy|electricity|magnet(?:s|ism|ic)?|light|sound waves?|"
            r"speed of (?:light|sound)|relativity|quantum|thermodynamics|entropy|"
            r"cells?|dna|genes?|evolution|photosynthesis|ecosystem|species|bacteria|"
            r"planets?|stars?|sun|moon|galax(?:y|ies)|universe|black holes?|orbit|eclipse|"
            r"northern lights|aurora|sky blue|rainbow|volcano(?:es)?|tides?|seasons?|climate|"
            r"how does (?:a|an|the) \w+ work|why is the|why do(?:es)?"
        ),
        examples=(
            "Why is the sky blue?",
            "How does photosynthesis work?",
            "What is the speed of light?",
            "What causes the northern lights?",
            "How do magnets work?",
            "What is a black hole?",
            "How does a vacuum tube amplify a signal?",
            "What is the Pythagorean theorem?",
            "Why do animals hibernate?",
            "What is water made of?",
            "How far away is the sun?",
            "Why does the sea have tides?",
        ),
        bias={
            "crashcourse": -0.03,
            "wikipedia": -0.02,
            "wikibooks": -0.02,
            "wikimed": 0.02,
            "ifixit": 0.06,
            "gutenberg": 0.08,
            # The practical guides (config/knowledge_sources.json).
            "guides_health": 0.04,
            "guides_outdoors": 0.04,
            "guides_food": 0.04,
            "guides_workshop": 0.04,
        },
    ),
    Shelf(
        key="history",
        name="history and the world",
        blurb="People, places, countries, wars and events, mostly from Wikipedia.",
        aliases=_rx(r"history|historical|the world|world|geography|people and places|countries"),
        keywords=_rx(
            r"history|historical|histor(?:y|ies) of|ancient|medieval|century|centuries|"
            r"wars?|battles?|revolution|empire|emperor|kings?|queens?|presidents?|"
            r"prime minister|dynasty|civilization|colon(?:y|ies|ial)|treaty|invasion|"
            r"who (?:is|was|were|are)|when (?:did|was|were)|where (?:is|was)|"
            r"born|died|founded|invented|discovered|"
            r"capital of|population of|countr(?:y|ies)|continent|rivers?|mountains?|oceans?|"
            r"romans?|greeks?|egypt(?:ians?)?|vikings?|napoleon|world war|civil war|"
            r"cold war|renaissance|industrial revolution|pyramids?"
        ),
        examples=(
            "Who was Napoleon Bonaparte?",
            "When did the Roman Empire fall?",
            "What caused the First World War?",
            "What is the capital of Mongolia?",
            "Who built the pyramids?",
            "When was the printing press invented?",
            "Where is the Danube river?",
            "Who was the first president of the United States?",
            "What was the Ottoman Empire?",
            "What happened at the Battle of Hastings?",
            "How large is Australia?",
            "Who were the Aztecs?",
        ),
        bias={
            "wikipedia": -0.03,
            "crashcourse": -0.02,
            "wikibooks": 0.0,
            "wikimed": 0.04,
            "gutenberg": 0.05,
            "ifixit": 0.08,
            # The practical guides (config/knowledge_sources.json).
            "guides_health": 0.06,
            "guides_outdoors": 0.06,
            "guides_food": 0.06,
            "guides_workshop": 0.06,
        },
    ),
    Shelf(
        key="stories",
        name="stories and literature",
        blurb=(
            "Novels, poems, plays and their authors, from the Gutenberg books and the encyclopedia."
        ),
        aliases=_rx(r"stories|story|literature|fiction|novels?|poetry|poems?"),
        keywords=_LITERATURE_RE,
        examples=(
            "What happens at the end of Moby Dick?",
            "Who wrote Pride and Prejudice?",
            "Who is Captain Ahab?",
            "What is Hamlet about?",
            "What is the first line of A Tale of Two Cities?",
            "Tell me about the poems of Walt Whitman.",
            "Who is Sherlock Holmes's companion?",
            "What is the plot of Treasure Island?",
            "Who is the monster in the old novel?",
            "Which story has a great white whale?",
            "Read me a famous line from Shakespeare.",
            "What did Mark Twain write?",
        ),
        # The router's literature table: only the two sources that hold books.
        bias={
            "gutenberg": -0.02,
            "wikipedia": -0.03,
            "wikibooks": 0.30,
            "crashcourse": 0.30,
            "wikimed": 0.50,
            "ifixit": 0.50,
            # The practical guides (config/knowledge_sources.json).
            "guides_health": 0.5,
            "guides_outdoors": 0.5,
            "guides_food": 0.5,
            "guides_workshop": 0.5,
        },
    ),
)

_BY_KEY: dict[str, Shelf] = {s.key: s for s in SHELVES}
KEYS: tuple[str, ...] = tuple(_BY_KEY)

# A keyword tie is broken in this order: a book title or a body word is a
# stronger sign than "make" or "who was".
_PRECEDENCE: tuple[str, ...] = (
    "stories",
    "health",
    "food",
    "fixing",
    "survival",
    "science",
    "history",
)


def get(key: str | None) -> Shelf | None:
    return _BY_KEY.get(key) if key else None


def name(key: str | None) -> str:
    shelf = get(key)
    return shelf.name if shelf else ""


def bias(key: str | None) -> dict[str, float] | None:
    """Per-collection distance bias for a shelf; None when there is none
    (the caller falls back to the router's question type)."""
    shelf = get(key)
    return dict(shelf.bias) if shelf else None


def named(text: str) -> str | None:
    """The shelf a person names ("the health shelf", "first aid"), or None."""
    for shelf in SHELVES:
        if shelf.aliases.search(text):
            return shelf.key
    return None


def keyword_scores(question: str) -> dict[str, int]:
    """Distinct keyword hits per shelf (shelves with none are left out)."""
    out: dict[str, int] = {}
    for shelf in SHELVES:
        found = {m.group(0).lower() for m in shelf.keywords.finditer(question)}
        if found:
            out[shelf.key] = len(found)
    return out


# ------------------------------------------------------------ nearest shelf

_proto_lock = Lock()
_protos: tuple[tuple[str, ...], np.ndarray] | None = None
_protos_fn_id: int | None = None

# Cosine to the nearest example question, and the lead it must have over
# the best example of any other shelf. On 138 hand-written questions
# (2026-11-25, nomic-v1.5 fp16): 86 % on the right shelf, 3 % on a wrong
# one, the rest left unassigned. Not checked on real conversations.
MIN_SIM = 0.62
MIN_MARGIN = 0.04


def _unit(vec: object) -> np.ndarray | None:
    arr = np.asarray(vec, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else None


def _prototypes(embed_fn: EmbedFn) -> tuple[tuple[str, ...], np.ndarray] | None:
    """Unit vectors of every shelf's example questions, embedded once per
    embedder (84 embeddings, ~5 s on the Jetson — call :func:`warm`
    off the turn's path)."""
    global _protos, _protos_fn_id
    with _proto_lock:
        if _protos is not None and _protos_fn_id == id(embed_fn):
            return _protos
        keys: list[str] = []
        rows: list[np.ndarray] = []
        for shelf in SHELVES:
            for example in shelf.examples:
                raw = embed_fn(EMBED_PREFIX + example)
                vec = _unit(raw) if raw is not None else None
                if vec is None:
                    return None  # no embedder resident yet; try again later
                keys.append(shelf.key)
                rows.append(vec)
        _protos = (tuple(keys), np.vstack(rows))
        _protos_fn_id = id(embed_fn)
        return _protos


def ready(embed_fn: EmbedFn | None) -> bool:
    """True when the example questions are embedded for *embed_fn*, so
    :func:`classify` costs one embedding and no more."""
    return embed_fn is not None and _protos is not None and _protos_fn_id == id(embed_fn)


def warm(embed_fn: EmbedFn | None) -> bool:
    """Embed the example questions now. True when they are ready."""
    if embed_fn is None:
        return False
    try:
        return _prototypes(embed_fn) is not None
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Shelf prototypes unavailable: {e}")
        return False


def nearest(question: str, embed_fn: EmbedFn) -> tuple[str | None, float, float]:
    """(shelf or None, best cosine, lead over the runner-up shelf)."""
    protos = _prototypes(embed_fn)
    if protos is None:
        return None, 0.0, 0.0
    raw = embed_fn(EMBED_PREFIX + question)
    vec = _unit(raw) if raw is not None else None
    if vec is None:
        return None, 0.0, 0.0
    keys, matrix = protos
    sims = matrix @ vec
    best: dict[str, float] = {}
    for key, sim in zip(keys, sims.tolist()):
        if sim > best.get(key, -1.0):
            best[key] = sim
    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    top_key, top = ranked[0]
    margin = top - ranked[1][1] if len(ranked) > 1 else top
    if top >= MIN_SIM and margin >= MIN_MARGIN:
        return top_key, top, margin
    return None, top, margin


def classify(question: str, embed_fn: EmbedFn | None = None) -> str | None:
    """The shelf a question belongs on, or None when it is not clear.

    The nearest example question decides when it is clearly nearest (an
    embedder must already be resident): a lone keyword is a weaker sign
    ("my joints ache" is not carpentry). Otherwise the shelf with the most
    keyword hits, ties by precedence. Never raises.
    """
    if embed_fn is not None:
        try:
            key, _sim, _margin = nearest(question, embed_fn)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Shelf embedding test failed: {e}")
            key = None
        if key is not None:
            return key
    scores = keyword_scores(question)
    if not scores:
        return None
    top = max(scores.values())
    return next(k for k in _PRECEDENCE if scores.get(k) == top)


# ----------------------------------------------------------------- spoken


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


def list_text() -> str:
    names = _join([s.name for s in SHELVES])
    return (
        f"I keep seven shelves: {names}. Music, and the books I read aloud, are separate. "
        "Say, what's on the health shelf, to hear about one."
    )


def describe_text(key: str, thread_labels: list[str] | None = None) -> str:
    """What is on one shelf, what we have talked about there, and a
    question to try."""
    shelf = get(key)
    if shelf is None:
        return list_text()
    parts = [f"The {shelf.name} shelf. {shelf.blurb}"]
    if thread_labels:
        parts.append(f"We have talked about {_join(thread_labels[:3])} there.")
    parts.append(f"You could ask, for example: {shelf.examples[0]}")
    return " ".join(parts)


_SOURCE_PHRASES: dict[str, str] = {
    "wikipedia": "the Wikipedia article on {title}",
    "wikimed": "the WikiMed medical article on {title}",
    "ifixit": "the iFixit repair guide, {title}",
    "wikibooks": "the textbook {title}",
    "crashcourse": "the Crash Course lesson, {title}",
    "gutenberg": "the old book {title}",
    "summary": "my own plot summary of {title}",
}
_SOURCE_BARE: dict[str, str] = {
    "wikipedia": "Wikipedia",
    "wikimed": "the WikiMed medical reference",
    "ifixit": "the iFixit repair guides",
    "wikibooks": "the textbooks",
    "crashcourse": "the Crash Course lessons",
    "gutenberg": "the old books",
    "summary": "my own plot summaries",
}


_GUIDE_LABELS: dict[str, str] | None = None


def _guide_label(source: str) -> str | None:
    """How a practical-guide source is named aloud ("the NHS medicines
    guide"), from config/knowledge_sources.json; None for anything else."""
    global _GUIDE_LABELS
    if _GUIDE_LABELS is None:
        labels: dict[str, str] = {}
        try:
            import json
            from pathlib import Path

            manifest = Path(__file__).resolve().parents[2] / "config" / "knowledge_sources.json"
            for entry in json.loads(manifest.read_text())["sources"]:
                labels[entry["name"]] = entry["label"]
        except Exception as e:  # noqa: BLE001 — a missing list only costs the nicer name
            logger.warning(f"Knowledge source list unreadable: {e}")
        _GUIDE_LABELS = labels
    return _GUIDE_LABELS.get(source)


def sources_text(shelf_key: str | None, sources: list[tuple[str, str]]) -> str:
    """Where the last answer came from. *sources* are (collection, title)
    pairs in the order they were used."""
    if not sources:
        return (
            "I found nothing on the shelves for that one, so it came from what I already "
            "know. Treat it with a little more care."
        )
    phrases: list[str] = []
    seen: set[tuple[str, str]] = set()
    for collection, title in sources:
        title = " ".join((title or "").split())
        key = (collection, title.lower())
        if key in seen:
            continue
        seen.add(key)
        guide = _guide_label(collection)
        if guide and title:
            # A question or a recipe name: said as it is, after its source.
            phrases.append(f"{guide}, {title.removeprefix('Recipe: ').rstrip(' .')}")
        elif guide:
            phrases.append(guide)
        elif title:
            template = _SOURCE_PHRASES.get(collection, "{title}")
            if collection == "wikibooks" and title.lower().startswith("cookbook:"):
                template, title = "the cookbook page on {title}", title[len("cookbook:") :]
            # "Horticulture/Tomato", "Sourdough Starter (recipe)" — said aloud.
            title = re.sub(r"\s*\([^)]*\)\s*$", "", title).replace("/", ", ").strip()
            phrases.append(template.format(title=title))
        else:
            phrases.append(_SOURCE_BARE.get(collection, collection))
        if len(phrases) == 3:
            break
    shelf = get(shelf_key)
    lead = f"That came from the {shelf.name} shelf: " if shelf else "That came from "
    return lead + _join(phrases) + "."
