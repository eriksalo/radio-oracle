#!/usr/bin/env python3
"""Extra knowledge sources: ZIM → clean text → embeddings → FAISS, no ChromaDB.

The sources (Stack Exchange sites, NHS medicines, recipe sites …) are
listed in ``config/knowledge_sources.json``. Several sources share one
collection, because the retriever takes ``tier1_top_k`` hits *per
collection*: one collection per subject shelf keeps the prompt short.

Stages, each resumable and each per source:

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
    if len(slug) != 2 or slug[1] in ("tags", "index.html", "about", "contribute"):
        return None  # the front page, tag lists
    main = tree.css_first("article") or tree.css_first("main")
    if main is None:
        return None
    _drop(main, "nav, script, style, img, figure, .taglist, footer, header")
    paras = _paragraphs(main)
    if sum(len(p.split()) for p in paras) < 40:
        return None
    title = re.split(r"\s+[|–-]\s+", title)[0].strip()
    return f"Recipe: {title}", ["\n\n".join(paras)]


EXTRACTORS: dict[str, Callable] = {
    "stackexchange": extract_stackexchange,
    "nhs": extract_nhs,
    "recipes": extract_recipes,
}


def iter_documents(src: Source) -> Iterator[tuple[str, str, list[str]]]:
    """(path, title, sections) for every page of the ZIM worth keeping."""
    from libzim.reader import Archive
    from selectolax.lexbor import LexborHTMLParser

    extractor = EXTRACTORS[src.extractor]
    zim = Archive(str(src.zim_path))
    get_entry = getattr(zim, "_get_entry_by_id", None) or zim.get_entry_by_id
    for i in range(zim.entry_count):
        entry = get_entry(i)
        if entry.is_redirect:
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
        if doc is not None:
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
        self._sess = ort.InferenceSession(
            str(model_path), so, providers=[*providers, "CPUExecutionProvider"]
        )
        self.providers = self._sess.get_providers()
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
    con = sqlite3.connect(f"file:{src.text_path}?mode=ro", uri=True)
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
        con = sqlite3.connect(f"file:{src.text_path}?mode=ro", uri=True)
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
        con = sqlite3.connect(f"file:{src.text_path}?mode=ro", uri=True)
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


def status(sources: list[Source]) -> None:
    for src in sources:
        chunks = 0
        if src.text_path.exists():
            con = sqlite3.connect(f"file:{src.text_path}?mode=ro", uri=True)
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
    p.add_argument("stages", nargs="+", choices=["download", "extract", "embed", "build", "status"])
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
                    embed(src, embedder, args.batch)


if __name__ == "__main__":
    main()
