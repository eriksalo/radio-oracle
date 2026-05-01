#!/usr/bin/env python3
"""
Unified ZIM ingestion script — replaces ingest_wikipedia.py, ingest_generic_zim.py,
and ingest_gutenberg.py (Gutenberg is now a ZIM too).

Reads ZIM files, extracts article text, chunks it, embeds with
all-MiniLM-L6-v2 (auto-detects GPU, FP16 on CUDA), and stores in ChromaDB.

Features over the old scripts:
  - Single script for all ZIM sources
  - Auto-detects collection name from ZIM filename
  - GPU auto-detection + explicit --device / --fp16 flags
  - Producer/consumer pipeline so HTML extraction overlaps GPU embedding
  - Deterministic doc IDs for resume/dedup
  - --all mode to ingest every ZIM in a directory
  - Rate logging and progress tracking

Usage:
    python scripts/ingest_zim.py <file.zim> [--collection <name>] [--batch-size 2000] [--dry-run]
    python scripts/ingest_zim.py --all --zim-dir /path/to/zims [--batch-size 2000] [--dry-run]

Workstation tuning (e.g. RTX 4070):
    python scripts/ingest_zim.py --all --zim-dir /path/to/zims \
        --device cuda --fp16 --encode-batch-size 512 --batch-size 4000
"""

import argparse
import hashlib
import multiprocessing as mp
import os
import pickle
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

from loguru import logger
from selectolax.parser import HTMLParser


# ---------------------------------------------------------------------------
# ZIM reading
# ---------------------------------------------------------------------------

def iter_zim_articles(
    zim_path: str,
    entry_start: int = 0,
    entry_end: int | None = None,
    existing_urls: set[str] | None = None,
    log_progress: bool = True,
    progress_prefix: str = "",
    stats: dict | None = None,
):
    """Yield (url, title, html) for entries in [entry_start, entry_end).

    `existing_urls` is checked against `entry.path[:500]` BEFORE the (expensive)
    HTML decode — this is the resume hot path. Without it, we'd UTF-8 decode
    every already-ingested article only to throw it away.

    If `stats` dict is provided, increments stats["scanned"] for every entry
    (yielded or skipped) and stats["skipped_url"] for URL-skipped ones, so
    callers can monitor real progress.
    """
    from libzim.reader import Archive  # type: ignore[import-untyped]

    archive = Archive(zim_path)
    if entry_end is None:
        entry_end = archive.entry_count
    if log_progress:
        logger.info(
            f"ZIM: {Path(zim_path).name} — entries [{entry_start}, {entry_end})"
        )

    skip_urls = existing_urls if existing_urls is not None else frozenset()
    skipped_url_count = 0

    for i in range(entry_start, entry_end):
        if stats is not None:
            stats["scanned"] = stats.get("scanned", 0) + 1
        try:
            entry = archive._get_entry_by_id(i)
        except Exception:
            continue

        if entry.is_redirect:
            continue

        # URL-skip BEFORE HTML decode: on resume, this avoids decoding
        # millions of already-ingested articles.
        url = entry.path
        if url[:500] in skip_urls:
            skipped_url_count += 1
            if stats is not None:
                stats["skipped_url"] = stats.get("skipped_url", 0) + 1
            if log_progress and (i + 1) % 50_000 == 0:
                logger.info(
                    f"{progress_prefix}  scanned {i + 1} / {entry_end} entries "
                    f"(skipped {skipped_url_count} known URLs)"
                )
            continue

        item = entry.get_item()
        if "html" not in item.mimetype:
            continue

        try:
            html = bytes(item.content).decode("utf-8", errors="replace")
        except Exception:
            continue

        yield url, entry.title or "", html

        if log_progress and (i + 1) % 50_000 == 0:
            logger.info(
                f"{progress_prefix}  scanned {i + 1} / {entry_end} entries "
                f"(skipped {skipped_url_count} known URLs)"
            )


def get_zim_entry_count(zim_path: str) -> int:
    """Cheap entry-count read for slicing across workers."""
    from libzim.reader import Archive  # type: ignore[import-untyped]

    return Archive(zim_path).entry_count


# ---------------------------------------------------------------------------
# Text extraction & chunking
# ---------------------------------------------------------------------------

def extract_text(html: str) -> str:
    """Strip HTML to clean text, removing scripts/styles/tables/footnotes."""
    parser = HTMLParser(html)

    for tag in ("script", "style", "table", "sup", "footer", "nav", "aside"):
        for node in parser.css(tag):
            node.decompose()

    text = parser.text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """Split text into word-based chunks with overlap."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if len(text) >= 100 else []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if len(chunk) >= 100:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def make_doc_id(collection: str, url: str, chunk_idx: int) -> str:
    """Deterministic document ID for dedup/resume."""
    raw = f"{collection}:{url}:{chunk_idx}"
    return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# ChromaDB + embedding
# ---------------------------------------------------------------------------

def get_chroma_collection(db_path: str, collection_name: str):
    import chromadb

    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def build_embedder(device: str = "auto", fp16: bool = True):
    """Load sentence-transformers model with explicit device + FP16.

    device: 'auto' (cuda if available), 'cpu', 'cuda', or 'cuda:N'.
    fp16: convert weights to half precision on CUDA (~2x faster, negligible
    quality loss for MiniLM). No-op on CPU.
    """
    from sentence_transformers import SentenceTransformer

    from oracle.rag.embedder import resolve_device

    resolved = resolve_device(device)
    logger.info(f"Loading embedding model all-MiniLM-L6-v2 (device={resolved}, fp16={fp16}) ...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=resolved)

    if fp16 and str(model.device).startswith("cuda"):
        model.half()
        logger.info("Embedding model converted to FP16")
    elif fp16:
        logger.warning("FP16 requested but device is not CUDA; staying in FP32")

    logger.info(f"Embedding model loaded (device: {model.device})")
    return model


def encode_batch(
    embedder,
    texts: list[str],
    encode_batch_size: int = 2048,
):
    """Embed a batch on the GPU. Pure compute, no DB I/O."""
    return embedder.encode(
        texts,
        show_progress_bar=False,
        batch_size=encode_batch_size,
        convert_to_numpy=True,
    )


def upsert_batch(collection, doc_ids, texts, embeddings, metadatas) -> None:
    """Upsert into ChromaDB. Pure DB I/O, no GPU."""
    collection.upsert(
        ids=doc_ids,
        documents=texts,
        embeddings=embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings,
        metadatas=metadatas,
    )


def flush_batch(
    collection,
    embedder,
    doc_ids: list[str],
    texts: list[str],
    metadatas: list[dict],
    encode_batch_size: int = 2048,
):
    """Encode + upsert in one call (legacy single-thread path)."""
    embeddings = encode_batch(embedder, texts, encode_batch_size)
    upsert_batch(collection, doc_ids, texts, embeddings, metadatas)


def load_existing_ids(collection, db_path: str) -> set[str]:
    """Load all existing doc IDs from a collection for fast skip-checking.

    Reads directly from chroma.sqlite3 (METADATA segment) — orders of magnitude
    faster than collection.get() with offset pagination, which is O(n^2) on
    large collections (9.86M IDs takes ~2 hours via API vs ~25s via SQL).
    Falls back to the API if the direct read fails for any reason.
    """
    import sqlite3

    sqlite_path = Path(db_path) / "chroma.sqlite3"
    if not sqlite_path.exists():
        logger.warning(f"SQLite not found at {sqlite_path}, falling back to API")
        return _load_existing_ids_via_api(collection)

    try:
        t0 = time.time()
        conn = sqlite3.connect(str(sqlite_path))
        cur = conn.cursor()
        cur.execute(
            """
            SELECT e.embedding_id
            FROM embeddings e
            JOIN segments s ON s.id = e.segment_id
            JOIN collections c ON c.id = s.collection
            WHERE c.name = ? AND s.scope = 'METADATA'
            """,
            (collection.name,),
        )
        ids = {row[0] for row in cur.fetchall()}
        conn.close()
        logger.info(f"Loaded {len(ids)} existing IDs via SQL in {time.time() - t0:.1f}s")
        return ids
    except Exception as e:
        logger.warning(f"Direct SQL ID load failed ({e}), falling back to API")
        return _load_existing_ids_via_api(collection)


def _load_existing_ids_via_api(collection) -> set[str]:
    count = collection.count()
    if count == 0:
        return set()
    logger.info(f"Loading {count} existing IDs via API (slow path)...")
    all_ids: set[str] = set()
    batch = 10_000
    for offset in range(0, count, batch):
        result = collection.get(limit=batch, offset=offset, include=[])
        all_ids.update(result["ids"])
    logger.info(f"Loaded {len(all_ids)} existing IDs")
    return all_ids


def load_existing_urls(collection_name: str, db_path: str) -> set[str]:
    """Load every URL that already has stored chunks, so we can skip
    extract_text/chunk_text on resume. URLs are stored truncated to 500 chars
    in metadata — caller must compare against url[:500].

    Tradeoff: if a previous run was killed mid-article, that one in-progress
    article's remaining chunks will be lost on resume. Acceptable for RAG.
    """
    import sqlite3

    sqlite_path = Path(db_path) / "chroma.sqlite3"
    if not sqlite_path.exists():
        return set()

    try:
        t0 = time.time()
        conn = sqlite3.connect(str(sqlite_path))
        cur = conn.cursor()
        cur.execute(
            """
            SELECT s.id FROM segments s
            JOIN collections c ON c.id = s.collection
            WHERE c.name = ? AND s.scope = 'METADATA'
            """,
            (collection_name,),
        )
        row = cur.fetchone()
        if row is None:
            conn.close()
            return set()
        segment_id = row[0]
        cur.execute(
            """
            SELECT DISTINCT em.string_value
            FROM embedding_metadata em
            JOIN embeddings e ON e.id = em.id
            WHERE em.key = 'url' AND e.segment_id = ?
            """,
            (segment_id,),
        )
        urls = {row[0] for row in cur.fetchall() if row[0] is not None}
        conn.close()
        logger.info(f"Loaded {len(urls)} existing URLs via SQL in {time.time() - t0:.1f}s")
        return urls
    except Exception as e:
        logger.warning(f"URL load failed ({e}); URL-skip disabled, will rely on chunk-level skip")
        return set()


# ---------------------------------------------------------------------------
# Multi-process ingestion pipeline
# ---------------------------------------------------------------------------
#
# Producer pool (N processes) -> mp.Queue -> Encode thread (1 GPU)
#                                            -> upsert.Queue -> Upsert thread (1 chroma client)
#
# Why processes for producers? libzim read + HTML strip + chunking is pure
# CPU/Python and the GIL serializes threads. With 32 logical cores idle,
# multi-process is the only way to scale producer throughput.
#
# Existing URLs/IDs are pickled to a tempfile in the main process (one SQL
# read), then each worker loads the pickle once at startup. Avoids re-running
# the multi-minute SQL load N times.
# ---------------------------------------------------------------------------

_PRODUCER_DONE = "__PRODUCER_DONE__"  # final sentinel; must be picklable for mp.Queue
_STATS_TICK = "__STATS_TICK__"        # periodic progress update from workers


def _dump_lookup_sets(existing_ids: set[str], existing_urls: set[str]) -> str:
    """Pickle lookup sets to a tempfile and return the path."""
    fd, path = tempfile.mkstemp(prefix="ingest_lookup_", suffix=".pkl")
    with os.fdopen(fd, "wb") as f:
        pickle.dump(
            {"ids": existing_ids, "urls": existing_urls},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return path


def _load_lookup_sets(path: str) -> tuple[set[str], set[str]]:
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["ids"], d["urls"]


def _producer_worker(
    worker_id: int,
    zim_path: str,
    collection_name: str,
    entry_start: int,
    entry_end: int,
    chunk_size: int,
    chunk_overlap: int,
    lookup_path: str,
    batch_size: int,
    out_queue,  # mp.Queue
) -> None:
    """One ZIM producer process. Walks [entry_start, entry_end), URL-skips,
    extracts, chunks, dedup-checks against existing_ids, and emits batches."""
    # Stdlib logging into the worker — loguru in subprocs is finicky on Windows.
    import logging

    log = logging.getLogger(f"producer-{worker_id}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    try:
        existing_ids, existing_urls = _load_lookup_sets(lookup_path)
    except Exception as e:
        log.exception(f"failed loading lookup sets: {e}")
        out_queue.put((_PRODUCER_DONE, worker_id, {"error": repr(e)}))
        return

    stats = {
        "total_articles": 0,
        "total_chunks": 0,
        "skipped_short": 0,
        "skipped_existing": 0,
        "skipped_url": 0,
        "scanned": 0,
    }

    batch_ids: list[str] = []
    batch_texts: list[str] = []
    batch_metas: list[dict] = []

    last_tick_scanned = 0
    tick_every = 25_000

    def maybe_tick():
        nonlocal last_tick_scanned
        if stats["scanned"] - last_tick_scanned >= tick_every:
            out_queue.put((_STATS_TICK, worker_id, dict(stats)))
            last_tick_scanned = stats["scanned"]

    try:
        article_iter = iter_zim_articles(
            zim_path,
            entry_start=entry_start,
            entry_end=entry_end,
            existing_urls=existing_urls,
            log_progress=False,  # main process logs aggregate
            stats=stats,         # iter_zim_articles updates scanned/skipped_url
        )
        for url, title, html in article_iter:
            maybe_tick()
            text = extract_text(html)
            if len(text) < 100:
                stats["skipped_short"] += 1
                continue

            chunks = chunk_text(text, chunk_size, chunk_overlap)
            if not chunks:
                stats["skipped_short"] += 1
                continue

            stats["total_articles"] += 1
            url_key = url[:500]
            title_key = title[:500]

            for idx, chunk in enumerate(chunks):
                doc_id = make_doc_id(collection_name, url, idx)
                stats["total_chunks"] += 1

                if doc_id in existing_ids:
                    stats["skipped_existing"] += 1
                    continue

                batch_ids.append(doc_id)
                batch_texts.append(chunk)
                batch_metas.append({
                    "source": collection_name,
                    "title": title_key,
                    "url": url_key,
                    "chunk_index": idx,
                })

                if len(batch_ids) >= batch_size:
                    out_queue.put((batch_ids, batch_texts, batch_metas))
                    batch_ids, batch_texts, batch_metas = [], [], []

        if batch_ids:
            out_queue.put((batch_ids, batch_texts, batch_metas))
    except Exception as e:
        log.exception(f"producer {worker_id} crashed: {e}")
        out_queue.put((_PRODUCER_DONE, worker_id, {"error": repr(e), "stats": stats}))
        return

    out_queue.put((_PRODUCER_DONE, worker_id, {"stats": stats}))


def ingest_zim(
    zim_path: str,
    collection_name: str,
    db_path: str = "data/chroma",
    batch_size: int = 4000,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    dry_run: bool = False,
    device: str = "auto",
    fp16: bool = True,
    encode_batch_size: int = 2048,
    queue_depth: int = 8,
    workers: int = 0,  # 0 = auto: min(cpu_count, 12)
) -> None:
    if workers <= 0:
        workers = min(mp.cpu_count(), 12)

    logger.info(f"=== Ingesting {Path(zim_path).name} -> collection '{collection_name}' ===")
    logger.info(
        f"ChromaDB: {db_path} | workers: {workers} | upsert batch: {batch_size} | "
        f"encode batch: {encode_batch_size} | chunk: {chunk_size}w / {chunk_overlap} overlap | "
        f"device: {device} | fp16: {fp16}"
    )

    entry_count = get_zim_entry_count(zim_path)
    logger.info(f"ZIM has {entry_count} entries; partitioning across {workers} workers")

    if not dry_run:
        collection = get_chroma_collection(db_path, collection_name)
        embedder = build_embedder(device=device, fp16=fp16)
        existing_ids = load_existing_ids(collection, db_path)
        existing_urls = load_existing_urls(collection_name, db_path)
        logger.info(
            f"Collection '{collection_name}' has {len(existing_ids)} existing chunks "
            f"across {len(existing_urls)} URLs"
        )
    else:
        logger.info("DRY RUN — will not embed or store anything")
        collection = None
        embedder = None
        existing_ids = set()
        existing_urls = set()

    # Pickle lookup sets once so workers don't re-hit SQL N times.
    lookup_path = _dump_lookup_sets(existing_ids, existing_urls)
    pickle_size_mb = os.path.getsize(lookup_path) / 1024 / 1024
    logger.info(f"Wrote lookup pickle ({pickle_size_mb:.1f} MB) at {lookup_path}")

    new_chunks = 0
    t0 = time.time()

    # mp.Queue for producer→encode handoff. Bounded to apply backpressure when
    # the GPU can't keep up.
    work_queue: mp.Queue = mp.Queue(maxsize=queue_depth)
    # threading.Queue for encode→upsert handoff (same process, so cheap).
    upsert_queue: queue.Queue = queue.Queue(maxsize=queue_depth)
    upsert_done = threading.Event()
    upsert_error: dict = {}

    # ---- Spawn N producer processes with disjoint entry slices ----
    procs: list[mp.Process] = []
    slice_size = (entry_count + workers - 1) // workers
    for wid in range(workers):
        start = wid * slice_size
        end = min(start + slice_size, entry_count)
        if start >= end:
            continue
        p = mp.Process(
            target=_producer_worker,
            args=(
                wid, zim_path, collection_name, start, end,
                chunk_size, chunk_overlap, lookup_path, batch_size, work_queue,
            ),
            name=f"zim-producer-{wid}",
            daemon=True,
        )
        p.start()
        procs.append(p)

    # ---- Upsert thread (chroma I/O, runs concurrent with GPU encode) ----
    def _upsert_loop():
        try:
            while True:
                item = upsert_queue.get()
                if item is None:  # sentinel
                    break
                ids, texts, embeddings, metas = item
                if not dry_run:
                    upsert_batch(collection, ids, texts, embeddings, metas)
                upsert_queue.task_done()
        except Exception as e:
            upsert_error["err"] = repr(e)
            logger.exception(f"Upsert thread crashed: {e}")
        finally:
            upsert_done.set()

    upsert_thread = threading.Thread(target=_upsert_loop, name="chroma-upsert", daemon=True)
    upsert_thread.start()

    # ---- Encode loop (main thread, GPU-resident) ----
    workers_remaining = len(procs)
    aggregate_stats = {
        "total_articles": 0, "total_chunks": 0, "skipped_short": 0,
        "skipped_existing": 0, "skipped_url": 0, "scanned": 0,
    }
    last_log = t0
    log_interval = 5.0  # seconds

    # Per-worker latest stats snapshot (so re-aggregating from the latest tick
    # is correct — workers send cumulative-per-worker stats, not deltas).
    worker_stats: dict[int, dict] = {}

    def recompute_aggregate():
        agg = {k: 0 for k in aggregate_stats}
        for s in worker_stats.values():
            for k, v in s.items():
                if k in agg:
                    agg[k] += v
        return agg

    try:
        while workers_remaining > 0:
            item = work_queue.get()

            # Done sentinel from a worker
            if isinstance(item, tuple) and len(item) == 3 and item[0] == _PRODUCER_DONE:
                _, wid, payload = item
                workers_remaining -= 1
                if "error" in payload:
                    logger.error(f"Worker {wid} errored: {payload['error']}")
                if "stats" in payload:
                    worker_stats[wid] = payload["stats"]
                aggregate_stats.update(recompute_aggregate())
                logger.info(
                    f"Worker {wid} done; {workers_remaining} workers still running"
                )
                continue

            # Periodic stats tick from a worker
            if isinstance(item, tuple) and len(item) == 3 and item[0] == _STATS_TICK:
                _, wid, snapshot = item
                worker_stats[wid] = snapshot
                aggregate_stats.update(recompute_aggregate())
                continue

            ids, texts, metas = item
            new_chunks += len(ids)

            if not dry_run:
                embeddings = encode_batch(embedder, texts, encode_batch_size)
                upsert_queue.put((ids, texts, embeddings, metas))

            now = time.time()
            if now - last_log >= log_interval:
                elapsed = now - t0
                rate = new_chunks / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  {new_chunks} new chunks "
                    f"(scanned {aggregate_stats['scanned']}, "
                    f"articles {aggregate_stats['total_articles']}, "
                    f"skipped existing {aggregate_stats['skipped_existing']}) "
                    f"— {rate:.0f} new/sec, upsert q={upsert_queue.qsize()}"
                )
                last_log = now
    except KeyboardInterrupt:
        logger.warning("Interrupted; terminating workers and draining queues")
        for p in procs:
            if p.is_alive():
                p.terminate()
        # Drain work_queue to unblock any pending puts
        try:
            while True:
                work_queue.get_nowait()
        except Exception:
            pass
        raise
    finally:
        # Tell upsert thread to stop and wait for it to drain
        upsert_queue.put(None)
        upsert_thread.join(timeout=300)
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                logger.warning(f"Worker {p.name} did not exit cleanly; killing")
                p.terminate()
        try:
            os.unlink(lookup_path)
        except OSError:
            pass

    if upsert_error:
        logger.error(f"Upsert error: {upsert_error['err']}")

    elapsed = time.time() - t0
    logger.info(f"=== Done: {collection_name} ===")
    logger.info(
        f"  Articles: {aggregate_stats['total_articles']} | "
        f"Total chunks (incl. dupes): {aggregate_stats['total_chunks']} | "
        f"New: {new_chunks} | Skipped existing: {aggregate_stats['skipped_existing']} | "
        f"Skipped URL: {aggregate_stats['skipped_url']} | "
        f"Skipped short: {aggregate_stats['skipped_short']} | "
        f"Scanned: {aggregate_stats['scanned']}"
    )
    rate = new_chunks / elapsed if elapsed > 0 else 0
    logger.info(f"  Time: {elapsed / 60:.1f} min | Rate: {rate:.0f} new chunks/sec")

    if not dry_run:
        logger.info(f"  Collection total: {collection.count()} chunks")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ZIM_COLLECTIONS: dict[str, str] = {
    "wikipedia_en_all": "wikipedia",
    "wikipedia_en_medicine": "wikimed",
    "ifixit": "ifixit",
    "wikibooks": "wikibooks",
    "gutenberg": "gutenberg",
    "crashcourse": "crashcourse",
}


def detect_collection(zim_filename: str) -> str | None:
    """Auto-detect collection name from ZIM filename."""
    name = zim_filename.lower()
    for prefix, coll in ZIM_COLLECTIONS.items():
        if name.startswith(prefix):
            return coll
    return None


def find_all_zims(directory: str) -> list[tuple[str, str]]:
    """Find all .zim files in directory and pair with collection names."""
    results = []
    for zim_file in sorted(Path(directory).glob("*.zim")):
        coll = detect_collection(zim_file.name)
        if coll:
            results.append((str(zim_file), coll))
        else:
            logger.warning(f"Unknown ZIM file (skipping): {zim_file.name}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ZIM files into ChromaDB")
    parser.add_argument("zim_file", nargs="?", help="Path to ZIM file")
    parser.add_argument("--collection", "-c", help="ChromaDB collection name")
    parser.add_argument("--all", action="store_true", help="Ingest all ZIM files in --zim-dir")
    parser.add_argument("--zim-dir", default=".", help="Directory containing ZIM files (default: cwd)")
    parser.add_argument("--db-path", default="data/chroma", help="ChromaDB path (default: data/chroma)")
    parser.add_argument("--batch-size", type=int, default=4000, help="Chunks per ChromaDB upsert (default: 4000)")
    parser.add_argument("--encode-batch-size", type=int, default=2048, help="Sub-batch passed to model.encode (default: 2048; safe on 12GB VRAM with MiniLM-FP16)")
    parser.add_argument("--chunk-size", type=int, default=512, help="Words per chunk (default: 512)")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="Overlap between chunks (default: 64)")
    parser.add_argument("--device", default="auto", help="Embedding device: auto | cpu | cuda | cuda:N (default: auto)")
    fp16_group = parser.add_mutually_exclusive_group()
    fp16_group.add_argument("--fp16", dest="fp16", action="store_true", default=True, help="Use FP16 weights on CUDA (default: on)")
    fp16_group.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable FP16, use FP32")
    parser.add_argument("--queue-depth", type=int, default=8, help="Bounded queue depth between stages (default: 8)")
    parser.add_argument("--workers", type=int, default=0, help="Producer process count (default: min(cpu_count, 12))")
    parser.add_argument("--dry-run", action="store_true", help="Count chunks without embedding/storing")
    args = parser.parse_args()

    if args.all:
        zims = find_all_zims(args.zim_dir)
        if not zims:
            logger.error(f"No recognized ZIM files found in {args.zim_dir}")
            sys.exit(1)
        logger.info(f"Found {len(zims)} ZIM files to ingest:")
        for path, coll in zims:
            logger.info(f"  {Path(path).name} -> {coll}")
        for path, coll in zims:
            ingest_zim(
                path, coll,
                db_path=args.db_path,
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                dry_run=args.dry_run,
                device=args.device,
                fp16=args.fp16,
                encode_batch_size=args.encode_batch_size,
                queue_depth=args.queue_depth,
                workers=args.workers,
            )
    elif args.zim_file:
        collection = args.collection or detect_collection(Path(args.zim_file).name)
        if not collection:
            logger.error("Cannot auto-detect collection name. Use --collection <name>")
            sys.exit(1)
        ingest_zim(
            args.zim_file, collection,
            db_path=args.db_path,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            dry_run=args.dry_run,
            device=args.device,
            fp16=args.fp16,
            encode_batch_size=args.encode_batch_size,
            queue_depth=args.queue_depth,
            workers=args.workers,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
