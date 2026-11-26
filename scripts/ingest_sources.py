#!/usr/bin/env python3
"""Extra knowledge sources: ZIM → clean text → embeddings → FAISS, no ChromaDB.

The sources (Stack Exchange sites, NHS medicines, recipe sites …) are
listed in ``config/knowledge_sources.json``. Several sources share one
collection, because the retriever takes ``tier1_top_k`` hits *per
collection*: one collection per subject shelf keeps the prompt short.

Stages, each resumable and each per source:

    model      fetch the embedding model (fp32 ONNX + tokenizer) into models/
    download   fetch the ZIM into data/knowledge/
    extract    pages → cleaned, chunked text in data/embeddings/<source>.text.sqlite
    embed      chunks → data/embeddings/<source>.vectors.f32 (row N = chunk row N)
    build      a collection's sources → data/faiss/<collection>.{index,sqlite}

    python scripts/ingest_sources.py download extract --source nhs_medicines
    python scripts/ingest_sources.py embed --source nhs_medicines
    python scripts/ingest_sources.py build --collection guides_health
    python scripts/ingest_sources.py status

What goes in is chosen for a radio that *speaks*: navigation, comments,
user cards and code blocks are dropped; a Stack Exchange question is kept
only when it and its best answer were voted up, and only its best answers
are kept. Every chunk starts with the page title, because an answer on
its own ("Use a 5/32 file") does not say what it is about.

Document vectors match the existing indices (scripts/reembed_collection.py):
nomic-embed-text-v1.5, ``search_document: `` prefix, mean-pooled, L2
normalised. Only the *query* side stays un-normalised (see CLAUDE.md).
The embed stage runs the ONNX model, so it needs no torch — on a 4-core
CPU that is 1.5–5 chunks a second (measured 2026-11-26); with
``--provider cuda`` in a venv that has onnxruntime-gpu it uses the GPU.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "config" / "knowledge_sources.json"
ZIM_DIR = Path("data/knowledge")
EMB_DIR = Path("data/embeddings")
FAISS_DIR = Path("data/faiss")
DIM = 768
DOC_PREFIX = "search_document: "
MODEL = "nomic-ai/nomic-embed-text-v1.5"


@dataclass(frozen=True)
class Source:
    name: str
    file: str
    collection: str
    extractor: str
    label: str
    skip: tuple[str, ...] = ()  # pages whose path contains any of these are left out

    @property
    def zim_path(self) -> Path:
        return ZIM_DIR / Path(self.file).name

    @property
    def text_path(self) -> Path:
        return EMB_DIR / f"{self.name}.text.sqlite"

    @property
    def vec_path(self) -> Path:
        return EMB_DIR / f"{self.name}.vectors.f32"


def load_manifest() -> tuple[str, list[Source]]:
    data = json.loads(MANIFEST.read_text())
    return data["base_url"], [Source(**s) for s in data["sources"]]


# ------------------------------------------------------------- extraction

_BLOCKS = "p, li, h1, h2, h3, h4, blockquote, dt, dd, td, th"
_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _has_inner(el, selector: str) -> bool:
    """True when *el* contains a match other than itself (lexbor's css()
    also returns the element it is called on)."""
    return any(c.mem_id != el.mem_id for c in el.css(selector))


def _paragraphs(node) -> list[str]:
    """Leaf block elements of *node* as clean paragraphs, in page order."""
    out: list[str] = []
    for el in node.css(_BLOCKS):
        if el.mem_id == node.mem_id:
            continue
        if _has_inner(el, "li, p" if el.tag == "li" else _BLOCKS):
            continue  # a container: its children are yielded themselves
        text = _clean(el.text(separator=" "))
        if not text:
            continue
        if el.tag in ("h1", "h2", "h3", "h4") and text[-1] not in ".?!:":
            text += "."
        out.append(text)
    return out


def _drop(tree, selector: str) -> None:
    for el in tree.css(selector):
        el.decompose()


def _vote(node) -> int:
    el = node.css_first(".js-vote-count")
    try:
        return int(_clean(el.text())) if el is not None else 0
    except ValueError:
        return 0


def extract_stackexchange(path: str, title: str, tree) -> tuple[str, list[str]] | None:
    """A question and its best answers. Sections: the question, then one
    per kept answer."""
    if not path.startswith("questions/") or path.startswith("questions/tagged"):
        return None
    _drop(tree, "pre, code, .comment, .comments, .s-user-card, .post-signature, script, style")
    _drop(tree, ".js-post-notice, .s-notice, aside, .post-taglist, img, figure")
    question = tree.css_first(".question") or tree.css_first("#question")
    if question is None:
        return None
    answers = [
        (_vote(a), "accepted-answer" in (a.attributes.get("class") or ""), a)
        for a in tree.css(".answer")
    ]
    answers = [a for a in answers if a[0] >= 2 or (a[1] and a[0] >= 1)]
    if not answers or _vote(question) < 1:
        return None
    answers.sort(key=lambda a: (a[1], a[0]), reverse=True)
    q_body = question.css_first(".s-prose")
    q_text = " ".join(_paragraphs(q_body)) if q_body is not None else ""
    q_words = q_text.split()
    if len(q_words) > 150:  # the question sets the scene; the answers are the content
        q_text = " ".join(q_words[:150]) + " …"
    sections = []
    for i, (_, _, node) in enumerate(answers[:3]):
        body = node.css_first(".s-prose")
        paras = _paragraphs(body) if body is not None else []
        if sum(len(p.split()) for p in paras) < 12:
            continue
        lead = f"Question: {q_text}" if i == 0 and q_text else ""
        sections.append("\n\n".join(([lead] if lead else []) + ["Answer: " + paras[0]] + paras[1:]))
    return (title, sections) if sections else None


def extract_nhs(path: str, title: str, tree) -> tuple[str, list[str]] | None:
    if "page-no-longer-available" in path:
        return None
    main = tree.css_first("main")
    if main is None:
        return None
    _drop(main, "nav, script, style, form, svg, .nhsuk-hub-key-links, .beta-hub-related-links")
    _drop(main, ".nhsuk-breadcrumb, .nhsuk-back-link, .nhsuk-pagination, .nhsuk-contents-list")
    _drop(
        main, ".nhsuk-review-date, .nhsuk-u-visually-hidden, [class*=related-links], video, figure"
    )
    title = re.sub(r"\s*-\s*NHS\s*$", "", title)
    skip = ("page last reviewed", "next review due", "more in ", title.lower())
    paras = [p for p in _paragraphs(main) if not p.lower().startswith(skip)]
    if sum(len(p.split()) for p in paras) < 60:
        return None  # a hub page: links only
    return title, ["\n\n".join(paras)]


def extract_recipes(path: str, title: str, tree) -> tuple[str, list[str]] | None:
    slug = path.strip("/").split("/")
    if len(slug) > 2 and slug[1] == "recipe":
        pass  # foss.cooking/recipe/<author>/<name>
    elif len(slug) != 2 or slug[1] in ("tags", "index.html", "about", "contribute"):
        return None  # the front page, tag and author lists
    main = tree.css_first("article") or tree.css_first("main")
    if main is None:
        return None
    _drop(main, "nav, script, style, img, figure, .taglist, footer, header")
    paras = _paragraphs(main)
    if sum(len(p.split()) for p in paras) < 40:
        return None
    title = re.split(r"\s+[|–-]\s+", title)[0].strip()
    return f"Recipe: {title}", ["\n\n".join(paras)]


_PAGE_CHROME = (
    "nav, header, footer, aside, script, style, noscript, form, svg, img, figure, iframe, "
    "button, video, audio, select, [role=navigation], [role=banner], [role=contentinfo], "
    "[class*=breadcrumb], [class*=share], [class*=social], [class*=cookie], [class*=sidebar], "
    "[class*=newsletter], [class*=comment], [class*=pagination], [class*=skip], "
    "[class*=related], [class*=menu], [class*=advert], [class*=affiliate], [class*=author-box], "
    "[class*=visually-hidden], [class*=sr-only], [aria-hidden=true], [class*=language], "
    "[class*=lang-switch], [class*=toc], [class*=byline], [class*=post-meta], [class*=widget]"
)
_END_SECTIONS = (
    "references",
    "external links",
    "see also",
    "further reading",
    "notes",
    "sources",
    "footnotes",
    "bibliography",
    "related",
    "gallery",
    "navigation",
    "contents",
    "species",
    "selected species",
    "cultivars",
    "synonyms",
)
_TITLE_TAIL = re.compile(r"\s+[|–—-]\s+.*$")
# Lines a site puts on every page; never part of the text.
_PAGE_JUNK = re.compile(
    r"^(return to top|back to top|top|top of page|last (updated|reviewed|modified)\b.*|english|espa.ol|print|"
    r"share( this)?( page)?|skip to .*|table of contents|read more|learn more|click here|"
    r"an official website of .*|page last reviewed.*|related (content|pages|links|articles)|"
    r"jump to:? .*|tags?:.*|categor(y|ies):.*|posted (on|by|in)\b.*|leave a (comment|reply).*)\.?$",
    re.I,
)


def _is_english(tree) -> bool:
    html = tree.css_first("html")
    lang = (html.attributes.get("lang") or "") if html is not None else ""
    return not lang or lang.lower().startswith("en")


def _sections(node, min_words: int = 25) -> list[str]:
    """*node*'s text cut at its h2/h3 headings, each section led by its
    heading; reference-style sections and stubs are dropped."""
    out: list[str] = []
    heading, body = "", []

    def close() -> None:
        name = heading.rstrip(".:").lower()
        if body and name not in _END_SECTIONS and sum(len(p.split()) for p in body) >= min_words:
            out.append("\n\n".join(([heading] if heading else []) + body))

    for el in node.css(_BLOCKS):
        if el.mem_id == node.mem_id:
            continue
        if _has_inner(el, "li, p" if el.tag == "li" else _BLOCKS):
            continue
        text = _clean(el.text(separator=" "))
        if not text or _PAGE_JUNK.match(text):
            continue
        if len(text) < 30 and not text.isascii():
            continue  # a language switcher's entries
        if el.tag in ("h1", "h2", "h3"):
            close()
            heading, body = (text if text[-1] in ".?!:" else text + "."), []
        else:
            body.append(text)
    close()
    return _merge_short(out)


def _merge_short(sections: list[str], floor: int = 90) -> list[str]:
    """Join neighbouring sections until each has *floor* words: a passage
    of two sentences answers nothing, but it matches a question as well as
    a full one does."""
    merged: list[str] = []
    pending = ""
    for section in sections:
        pending = f"{pending}\n\n{section}" if pending else section
        if len(pending.split()) >= floor:
            merged.append(pending)
            pending = ""
    if pending:
        if merged:
            merged[-1] = f"{merged[-1]}\n\n{pending}"
        else:
            merged.append(pending)
    return merged


def extract_webpage(path: str, title: str, tree) -> tuple[str, list[str]] | None:
    """An ordinary web page from a site crawl: the main content, cut at its
    headings. English pages only; search and listing URLs are skipped."""
    if "?" in path or not _is_english(tree):
        return None
    main = (
        tree.css_first("main")
        or tree.css_first("[role=main]")
        or tree.css_first("article")
        or tree.css_first("#content")
        or tree.css_first("#main")
        or tree.body
    )
    if main is None:
        return None
    _drop(main, _PAGE_CHROME)
    sections = _sections(main)
    if sum(len(s.split()) for s in sections) < 80:
        return None  # a hub or listing page
    title = _TITLE_TAIL.sub("", title).strip() or title
    site = path.split("/")[0].replace(".", "").lower()
    if title.replace(" ", "").lower() in site:
        # Every page carries the site's name as its title: use the heading.
        first_line = sections[0].split("\n")[0].rstrip(".")
        if 0 < len(first_line.split()) <= 12:
            title = first_line[0].upper() + first_line[1:]
    return title, sections


_WIKI_NAMESPACES = re.compile(
    r"^(A/)?(Category|File|User|Talk|Template|Help|Special|MediaWiki|Portal|Project|Module|"
    r"[A-Za-z_ ]+[ _]talk|Appropedia|Gardenology|Restarters|Glossary|Image|Draft|_assets_)[:_/]",
    re.I,
)
# Translated copies of a wiki page live at Page/fr, Page/zh-hans …
_WIKI_TRANSLATION = re.compile(r"/([a-z]{2,3}(?:-[a-z]{2,4})?)$")


def extract_mediawiki(path: str, title: str, tree) -> tuple[str, list[str]] | None:
    """A wiki article as packed by mwoffliner: prose only, one section per
    heading; lists of links, tables of data and other namespaces are skipped."""
    if _WIKI_NAMESPACES.match(path) or _WIKI_NAMESPACES.match(title):
        return None
    if title.lower().startswith(("list of ", "index of ", "main page")):
        return None
    translation = _WIKI_TRANSLATION.search(path)
    if translation and translation.group(1) != "en":
        return None
    title = re.sub(r"/en$", "", title)
    main = tree.css_first("#mw-content-text") or tree.css_first(".mw-parser-output")
    if main is None:
        return None
    _drop(main, "script, style, img, figure, table, .reference, .references, .reflist, .navbox")
    _drop(main, ".mw-editsection, .thumb, .gallery, .noprint, .infobox, .toc, #toc, .hatnote")
    _drop(
        main, ".metadata, .ambox, .catlinks, .printfooter, sup, .mw-cite-backlink, .external.free"
    )
    sections = _sections(main)
    if sum(len(s.split()) for s in sections) < 80:
        return None  # a stub, a redirect page or a list of links
    return title, sections


def extract_medlineplus(path: str, title: str, tree) -> tuple[str, list[str]] | None:
    """MedlinePlus health topics, the medical encyclopedia, drug and
    lab-test pages. Spanish pages, indexes and news are skipped."""
    if "?" in path or not path.startswith("medlineplus.gov/") or not _is_english(tree):
        return None
    if any(part in path for part in _MEDLINEPLUS_SKIP):
        return None
    is_topic = path.count("/") == 1  # medlineplus.gov/asthma.html: a summary, then link lists
    main = tree.css_first("#topic-summary") if is_topic else tree.css_first("article")
    if main is None:
        return None
    _drop(main, _PAGE_CHROME)
    _drop(main, ".section-button, .adam-info, .page-actions, .side, .attribution, #section-version")
    sections = [s for s in _sections(main) if not s.lower().startswith(_MEDLINEPLUS_LISTS)]
    if sum(len(s.split()) for s in sections) < 60:
        return None
    title = re.split(r"\s*[:|]\s", title)[0].strip() or title
    return title, sections


_MEDLINEPLUS_SKIP = (
    "/spanish/",
    "/languages/",
    "/genetics/gene/",
    "/genetics/chromosome/",
    "/ency/imagepages/",
    "/ency/presentations/",
    "/ency/quiz/",
    "/ency/anatomyvideos/",
    "/ency/encyclopedia_",
    "/about/",
    "/medwords/",
    "/webeval/",
    "/recipes/",
    "/medlineplus-connect/",
    "/medlineplus-videos/",
    "/videofiles/",
)
_MEDLINEPLUS_LISTS = (
    "alternative names",
    "patient instructions",
    "images",
    "review date",
    "other names for this condition",
    "additional information",
    "related health topics",
    "related medlineplus",
    "learn more",
    "brand names",
    "other names",
    "browse the encyclopedia",
)


EXTRACTORS: dict[str, Callable] = {
    "stackexchange": extract_stackexchange,
    "nhs": extract_nhs,
    "recipes": extract_recipes,
    "webpage": extract_webpage,
    "mediawiki": extract_mediawiki,
    "medlineplus": extract_medlineplus,
}


def iter_documents(src: Source) -> Iterator[tuple[str, str, list[str]]]:
    """(path, title, sections) for every page of the ZIM worth keeping."""
    from libzim.reader import Archive
    from selectolax.lexbor import LexborHTMLParser

    extractor = EXTRACTORS[src.extractor]
    zim = Archive(str(src.zim_path))
    get_entry = getattr(zim, "_get_entry_by_id", None) or zim.get_entry_by_id
    seen: set[int] = set()  # a site crawl holds the same page under several URLs
    for i in range(zim.entry_count):
        entry = get_entry(i)
        if entry.is_redirect or any(part in entry.path for part in src.skip):
            continue
        item = entry.get_item()
        if not item.mimetype.startswith("text/html"):
            continue
        try:
            tree = LexborHTMLParser(bytes(item.content).decode("utf-8", errors="replace"))
            doc = extractor(entry.path, _clean(entry.title), tree)
        except Exception as e:  # noqa: BLE001 — one bad page must not end the run
            logger.warning(f"{src.name}: skipped {entry.path}: {e}")
            continue
        if doc is None:
            continue
        fingerprint = hash((doc[0], doc[1][0][:600]))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        yield entry.path, doc[0], doc[1]


def _open_chunks(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    # Same table as scripts/reembed_collection.py::FlatVectorStore, so
    # scripts/build_faiss_ivfpq.py reads it unchanged.
    con.execute(
        """CREATE TABLE IF NOT EXISTS chunks (
            row_id INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL UNIQUE,
            text TEXT NOT NULL,
            source TEXT,
            url TEXT,
            title TEXT,
            chunk_index INTEGER
        )"""
    )
    return con


def extract(src: Source, force: bool = False) -> int:
    """Write the source's chunks; returns how many. Re-extracting a source
    that already has vectors needs *force* (the vectors are removed: row
    numbers would no longer match)."""
    from oracle.rag.chunker import chunk_text

    if src.text_path.exists():
        if not force:
            con = sqlite3.connect(str(src.text_path))
            n = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            con.close()
            logger.info(f"{src.name}: already extracted ({n:,} chunks); --force to redo")
            return n
        src.text_path.unlink()
        src.vec_path.unlink(missing_ok=True)
    con = _open_chunks(src.text_path)
    row = docs = 0
    t0 = time.time()
    for path, title, sections in iter_documents(src):
        docs += 1
        index = 0
        for section in sections:
            # 300 words: with the title that stays under the embedder's 512
            # tokens, and a spoken answer never uses more of a passage.
            for chunk in chunk_text(section, chunk_size=300, chunk_overlap=40):
                text = f"{title}\n\n{chunk}"
                con.execute(
                    "INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                    (row, f"{src.name}_{row}", text, src.name, path, title, index),
                )
                row += 1
                index += 1
        if docs % 2000 == 0:
            con.commit()
            logger.info(f"{src.name}: {docs:,} pages, {row:,} chunks")
    con.commit()
    con.close()
    logger.info(f"{src.name}: {docs:,} pages -> {row:,} chunks in {time.time() - t0:.0f}s")
    return row


# -------------------------------------------------------------- embedding


class DocEmbedder:
    """nomic-v1.5 ONNX → masked mean pool → L2 normalise (document side)."""

    def __init__(self, model_path: Path, provider: str, threads: int) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(model_path.parent / "tokenizer.json"))
        self._tok.enable_truncation(512)
        self._tok.enable_padding(pad_id=0, pad_token="[PAD]")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        providers = ["CUDAExecutionProvider"] if provider == "cuda" else []
        if provider == "cuda" and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()  # the CUDA/cuDNN wheels pip installed, on Windows too
        self._sess = ort.InferenceSession(
            str(model_path), so, providers=[*providers, "CPUExecutionProvider"]
        )
        self.providers = self._sess.get_providers()
        if provider == "cuda" and self.providers[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                "CUDA was asked for but onnxruntime fell back to "
                f"{self.providers[0]}: install onnxruntime-gpu with its CUDA and cuDNN "
                "libraries (see docs/guides-ingest-gpu-runbook.md)"
            )
        self._inputs = {i.name for i in self._sess.get_inputs()}

    def encode(self, texts: list[str]) -> np.ndarray:
        batch = self._tok.encode_batch([DOC_PREFIX + t for t in texts])
        ids = np.array([e.ids for e in batch], dtype=np.int64)
        mask = np.array([e.attention_mask for e in batch], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self._sess.run(None, feeds)[0].astype(np.float32)
        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)
        return pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9)


def embedded_rows(src: Source) -> int:
    if not src.vec_path.exists():
        return 0
    size = src.vec_path.stat().st_size
    if size % (DIM * 4):
        raise RuntimeError(f"{src.vec_path} is not a whole number of vectors")
    return size // (DIM * 4)


def embed(src: Source, embedder: DocEmbedder, batch: int, window: int = 256) -> None:
    """Append vectors for the chunks that have none yet, in row order.
    Safe to stop and restart."""
    con = sqlite3.connect(str(src.text_path))
    total = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    done = embedded_rows(src)
    logger.info(f"{src.name}: {done:,}/{total:,} chunks embedded ({embedder.providers[0]})")
    t0, start = time.time(), done
    with open(src.vec_path, "ab") as fp:
        while done < total:
            rows = con.execute(
                "SELECT text FROM chunks WHERE row_id >= ? ORDER BY row_id LIMIT ?", (done, window)
            ).fetchall()
            texts = [r[0] for r in rows]
            # Similar lengths share a batch (padding is wasted work), but
            # the file is written in row order.
            order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            vectors = np.zeros((len(texts), DIM), dtype=np.float32)
            for i in range(0, len(order), batch):
                idx = order[i : i + batch]
                vectors[idx] = embedder.encode([texts[j] for j in idx])
            fp.write(vectors.tobytes(order="C"))
            fp.flush()
            done += len(texts)
            rate = (done - start) / max(time.time() - t0, 1e-6)
            eta = (total - done) / max(rate, 1e-6) / 60
            logger.info(f"{src.name}: {done:,}/{total:,} ({rate:.1f}/s, {eta:.0f} min left)")
    con.close()


# ------------------------------------------------------------------ build


def build(collection: str, sources: list[Source]) -> int:
    """Join the collection's fully embedded sources and write its index."""
    import faiss

    from scripts.build_faiss_ivfpq import build_index, write_idmap_sqlite

    ready = []
    for src in sources:
        if not src.text_path.exists():
            continue
        con = sqlite3.connect(str(src.text_path))
        n = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        con.close()
        if n and embedded_rows(src) == n:
            ready.append((src, n))
        else:
            logger.warning(
                f"{collection}: {src.name} left out ({embedded_rows(src):,}/{n:,} embedded)"
            )
    if not ready:
        raise RuntimeError(f"{collection}: no source is fully embedded")
    joined_text = EMB_DIR / f"{collection}.text.sqlite"
    joined_text.unlink(missing_ok=True)
    out = _open_chunks(joined_text)
    parts = []
    offset = 0
    for src, n in ready:
        con = sqlite3.connect(str(src.text_path))
        rows = con.execute(
            "SELECT row_id, chunk_id, text, source, url, title, chunk_index FROM chunks ORDER BY row_id"
        )
        out.executemany(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?)", ((r[0] + offset, *r[1:]) for r in rows)
        )
        con.close()
        parts.append(np.fromfile(src.vec_path, dtype=np.float32).reshape(n, DIM))
        offset += n
    out.commit()
    out.close()
    vectors = np.concatenate(parts)
    nlist = max(16, min(4096, int(round(len(vectors) ** 0.5))))
    index = build_index(vectors, nlist, 64, 8, 1_000_000)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_DIR / f"{collection}.index"))
    write_idmap_sqlite(FAISS_DIR / f"{collection}.sqlite", joined_text)
    joined_text.unlink()
    logger.info(f"{collection}: {len(vectors):,} vectors from {[s.name for s, _ in ready]}")
    return len(vectors)


# -------------------------------------------------------------------- cli


def download(base_url: str, src: Source) -> None:
    if src.zim_path.exists():
        logger.info(f"{src.name}: have {src.zim_path}")
        return
    ZIM_DIR.mkdir(parents=True, exist_ok=True)
    tmp = src.zim_path.with_suffix(".part")
    logger.info(f"{src.name}: downloading {base_url}{src.file}")
    request = urllib.request.Request(base_url + src.file, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as resp, open(tmp, "wb") as fp:
        while block := resp.read(1 << 20):
            fp.write(block)
    tmp.rename(src.zim_path)
    logger.info(f"{src.name}: {src.zim_path.stat().st_size / 1e6:.0f} MB")


MODEL_URL = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/"
MODEL_FILES = ("onnx/model.onnx", "tokenizer.json", "tokenizer_config.json", "config.json")


def fetch_model(model_path: Path) -> None:
    """The fp32 ONNX encoder and its tokenizer, next to each other."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        target = model_path.parent / Path(name).name
        if target.exists():
            continue
        logger.info(f"downloading {name}")
        request = urllib.request.Request(MODEL_URL + name, headers={"User-Agent": "Mozilla/5.0"})
        tmp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(request, timeout=60) as resp, open(tmp, "wb") as fp:
            while block := resp.read(1 << 20):
                fp.write(block)
        tmp.rename(target)


def status(sources: list[Source]) -> None:
    for src in sources:
        chunks = 0
        if src.text_path.exists():
            con = sqlite3.connect(str(src.text_path))
            chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            con.close()
        zim = (
            f"{src.zim_path.stat().st_size / 1e6:6.0f} MB" if src.zim_path.exists() else "       --"
        )
        logger.info(
            f"{src.name:20s} {src.collection:16s} zim {zim}  chunks {chunks:8,}  "
            f"embedded {embedded_rows(src):8,}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "stages", nargs="+", choices=["model", "download", "extract", "embed", "build", "status"]
    )
    p.add_argument("--source", action="append", help="source name (repeatable); default: all")
    p.add_argument("--collection", action="append", help="limit to these collections")
    p.add_argument("--force", action="store_true", help="extract again, discarding vectors")
    p.add_argument(
        "--model", type=Path, default=Path("models/nomic-embed-text-v1.5-onnx/model.onnx")
    )
    p.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()

    base_url, sources = load_manifest()
    chosen = [
        s
        for s in sources
        if (not args.source or s.name in args.source)
        and (not args.collection or s.collection in args.collection)
    ]
    if not chosen:
        raise SystemExit("no source matches")
    embedder: DocEmbedder | None = None
    for stage in args.stages:
        if stage == "status":
            status(chosen)
        elif stage == "model":
            fetch_model(args.model)
        elif stage == "build":
            for collection in dict.fromkeys(s.collection for s in chosen):
                build(collection, [s for s in sources if s.collection == collection])
        else:
            for src in chosen:
                if stage == "download":
                    download(base_url, src)
                elif stage == "extract":
                    extract(src, args.force)
                elif stage == "embed":
                    embedder = embedder or DocEmbedder(args.model, args.provider, args.threads)
                    embed(src, embedder, args.batch, window=max(256, args.batch * 8))


if __name__ == "__main__":
    main()
