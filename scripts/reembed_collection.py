#!/usr/bin/env python3
"""Re-embed an existing ChromaDB collection with a new embedding model.

Reads chunk text + metadata directly from `chroma.sqlite3` (parallel
SQL reads — no slow chromadb API pagination), embeds them on the GPU with
a configurable model (default: nomic-embed-text-v1.5, 768-d Matryoshka),
and writes to a new collection. Doc IDs are preserved so resume is
trivial — already-embedded IDs in the target are skipped.

Architecture mirrors `ingest_zim.py`:
    N producer processes (SQL readers)
        -> mp.Queue (bounded)
            -> GPU encode loop (main thread, model on CUDA FP16)
                -> threading.Queue
                    -> upsert thread (single chromadb writer)

Why processes for SQL producers? The GIL serializes threads on SQL row
unpacking; on a 32-core box we want them in their own interpreters.

Usage:
    python scripts/reembed_collection.py \\
        --source wikipedia \\
        --target wikipedia_v2 \\
        --model nomic-ai/nomic-embed-text-v1.5 \\
        --db-path data/chroma \\
        --workers 8 --encode-batch-size 256 --batch-size 1000
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import queue
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

from loguru import logger


# --- SQL helpers ------------------------------------------------------------

_DOC_KEY = "chroma:document"


def get_source_id_range(db_path: str, collection_name: str) -> tuple[str, int, int, int]:
    """Return (segment_id, min_embedding_row_id, max_embedding_row_id, total_rows)
    for the METADATA segment of `collection_name`.

    Workers need the segment_id so they can filter the embeddings range scan
    — chromadb assigns embeddings.id globally across all collections, so an
    id range from one collection's MIN/MAX may overlap with rows from other
    collections that were ingested in the same span.
    """
    sqlite_path = Path(db_path) / "chroma.sqlite3"
    con = sqlite3.connect(str(sqlite_path))
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT s.id FROM segments s
            JOIN collections c ON c.id = s.collection
            WHERE c.name = ? AND s.scope = 'METADATA'
            """,
            (collection_name,),
        )
        seg_row = cur.fetchone()
        if seg_row is None:
            raise RuntimeError(f"No METADATA segment for {collection_name!r}")
        segment_id = seg_row[0]
        cur.execute(
            """
            SELECT MIN(id), MAX(id), COUNT(id)
            FROM embeddings WHERE segment_id = ?
            """,
            (segment_id,),
        )
        row = cur.fetchone()
        if row is None or row[0] is None:
            raise RuntimeError(f"Collection {collection_name!r} segment is empty")
        return segment_id, int(row[0]), int(row[1]), int(row[2])
    finally:
        con.close()


def load_target_existing_ids(out_dir: Path, target: str) -> set[str]:
    """Chunk IDs already written to the target flat-file store. Reading the
    sqlite chunk table is O(n) but cheap (~1M IDs/sec)."""
    text_path = out_dir / f"{target}.text.sqlite"
    if not text_path.exists():
        return set()
    con = sqlite3.connect(str(text_path))
    try:
        cur = con.execute("SELECT chunk_id FROM chunks")
        return {r[0] for r in cur.fetchall()}
    except sqlite3.OperationalError:
        return set()
    finally:
        con.close()


# --- Flat-file writer ------------------------------------------------------


class FlatVectorStore:
    """Append-only float32 .vectors.f32 + chunk-metadata .text.sqlite.

    Row N in the .f32 file (offset = N * dim * 4) corresponds to chunks.row_id
    = N in the sqlite. Both are produced in lockstep by the upsert thread, so
    there's never a partial commit visible to other readers.
    """

    def __init__(self, out_dir: Path, name: str, dim: int):
        self.dim = dim
        self.vec_path = out_dir / f"{name}.vectors.f32"
        self.text_path = out_dir / f"{name}.text.sqlite"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Append-mode binary file; row count derives from filesize.
        self._fp = open(self.vec_path, "ab")
        existing_bytes = self.vec_path.stat().st_size
        if existing_bytes % (dim * 4) != 0:
            raise RuntimeError(
                f"{self.vec_path} size {existing_bytes} is not a multiple of "
                f"dim*4 ({dim*4}); refusing to append to a corrupted file"
            )
        self.next_row = existing_bytes // (dim * 4)
        # check_same_thread=False so the upsert thread can use this connection
        # (single writer, no concurrency concerns).
        self._con = sqlite3.connect(str(self.text_path), check_same_thread=False)
        self._con.execute("PRAGMA journal_mode = WAL")
        self._con.execute("PRAGMA synchronous = NORMAL")
        self._con.execute(
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
        self._con.execute(
            "CREATE INDEX IF NOT EXISTS chunks_chunk_id ON chunks (chunk_id)"
        )
        self._con.commit()

    def append(
        self,
        vectors: "np.ndarray",  # noqa: F821 — numpy imported in caller
        chunk_ids: list[str],
        texts: list[str],
        metas: list[dict],
    ) -> int:
        n = vectors.shape[0]
        assert vectors.shape[1] == self.dim
        assert vectors.dtype.name == "float32"
        assert n == len(chunk_ids) == len(texts) == len(metas)

        first_row = self.next_row
        # Append vectors as raw bytes — fastest path, no header to update.
        self._fp.write(vectors.tobytes(order="C"))
        self._fp.flush()
        rows = [
            (
                first_row + i,
                chunk_ids[i],
                texts[i],
                metas[i].get("source", ""),
                metas[i].get("url", ""),
                metas[i].get("title", ""),
                int(metas[i].get("chunk_index", 0)),
            )
            for i in range(n)
        ]
        self._con.executemany(
            "INSERT OR IGNORE INTO chunks VALUES (?,?,?,?,?,?,?)", rows
        )
        self._con.commit()
        self.next_row += n
        return first_row

    def close(self) -> None:
        try:
            self._fp.flush()
            self._fp.close()
        except Exception:
            pass
        try:
            self._con.close()
        except Exception:
            pass


def _dump_lookup(existing: set[str]) -> str:
    fd, path = tempfile.mkstemp(prefix="reembed_lookup_", suffix=".pkl")
    with os.fdopen(fd, "wb") as f:
        pickle.dump(existing, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _load_lookup(path: str) -> set[str]:
    with open(path, "rb") as f:
        return pickle.load(f)


# --- Producer (multi-process) ----------------------------------------------

_PRODUCER_DONE = "__PRODUCER_DONE__"
_STATS_TICK = "__STATS_TICK__"


def _producer_worker(
    worker_id: int,
    db_path: str,
    source_collection: str,
    source_segment_id: str,
    id_start: int,
    id_end: int,
    lookup_path: str,
    batch_size: int,
    out_queue: mp.Queue,
) -> None:
    """Read rows in [id_start, id_end) and emit (ids, texts, metadatas) batches.

    Uses two streaming index range scans merged in Python — one over
    `embeddings` (id range, PK index) and one over `embedding_metadata`
    (id range, autoindex on (id, key)). Cross-table joins on a 400 GB
    sqlite turned out to be ~10x slower because each outer row triggers
    multiple random-seek index lookups; sequential range scans avoid that.
    """
    import logging

    log = logging.getLogger(f"reembed-prod-{worker_id}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    try:
        existing = _load_lookup(lookup_path)
    except Exception as e:
        out_queue.put((_PRODUCER_DONE, worker_id, {"error": repr(e)}))
        return

    stats = {"scanned": 0, "skipped_existing": 0, "emitted": 0}
    last_tick = 0

    sqlite_path = Path(db_path) / "chroma.sqlite3"
    # Each cursor needs its own connection — Python's sqlite3 serializes
    # concurrent cursors on a shared connection (~10x slowdown observed).
    con_e = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    con_m = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    for c in (con_e, con_m):
        c.execute("PRAGMA cache_size = -32768")  # 32 MB per connection
        c.execute("PRAGMA mmap_size = 268435456")  # 256 MB mmap hint

    cur_e = con_e.cursor()
    cur_m = con_m.cursor()
    # `embeddings` is a rowid table; PRIMARY KEY = id. Filter by segment_id
    # so we don't accidentally pick up rows belonging to other collections
    # that share the id range (chromadb assigns ids globally across
    # collections, not per-collection).
    cur_e.execute(
        """SELECT id, embedding_id FROM embeddings
           WHERE segment_id = ? AND id >= ? AND id < ? ORDER BY id""",
        (source_segment_id, id_start, id_end),
    )
    # `embedding_metadata` has PRIMARY KEY (id, key). Range scan by id reads
    # all keys for each row in order with no extra seek per key. We don't
    # need to filter by segment here — the merge below only consumes
    # metadata for ids that came out of cur_e, which is already segment-filtered.
    cur_m.execute(
        """SELECT id, key, string_value, int_value FROM embedding_metadata
           WHERE id >= ? AND id < ? ORDER BY id, key""",
        (id_start, id_end),
    )

    meta_iter = iter(cur_m)
    m_row = next(meta_iter, None)

    batch_ids: list[str] = []
    batch_texts: list[str] = []
    batch_metas: list[dict] = []
    tick_every = 25_000

    def maybe_tick():
        nonlocal last_tick
        if stats["scanned"] - last_tick >= tick_every:
            out_queue.put((_STATS_TICK, worker_id, dict(stats)))
            last_tick = stats["scanned"]

    try:
        for e_id, emb_id in cur_e:
            stats["scanned"] += 1
            maybe_tick()

            # Skip any straggling metadata for ids before this embedding's
            # (would only happen if a row is missing in embeddings, but be
            # defensive).
            while m_row is not None and m_row[0] < e_id:
                m_row = next(meta_iter, None)

            doc = ""
            url = ""
            title = ""
            chunk_idx = 0
            source = ""
            while m_row is not None and m_row[0] == e_id:
                _, key, sval, ival = m_row
                if key == "chroma:document":
                    doc = sval or ""
                elif key == "url":
                    url = sval or ""
                elif key == "title":
                    title = sval or ""
                elif key == "chunk_index":
                    chunk_idx = int(ival) if ival is not None else 0
                elif key == "source":
                    source = sval or ""
                m_row = next(meta_iter, None)

            if not doc or not doc.strip():
                continue
            if emb_id in existing:
                stats["skipped_existing"] += 1
                continue

            batch_ids.append(emb_id)
            batch_texts.append(doc)
            batch_metas.append(
                {
                    "source": source or source_collection,
                    "url": url,
                    "title": title,
                    "chunk_index": chunk_idx,
                }
            )
            stats["emitted"] += 1
            if len(batch_ids) >= batch_size:
                out_queue.put((batch_ids, batch_texts, batch_metas))
                batch_ids, batch_texts, batch_metas = [], [], []

        if batch_ids:
            out_queue.put((batch_ids, batch_texts, batch_metas))
    except Exception as e:
        log.exception(f"producer {worker_id} crashed: {e}")
        out_queue.put((_PRODUCER_DONE, worker_id, {"error": repr(e), "stats": stats}))
        return
    finally:
        con_e.close()
        con_m.close()

    out_queue.put((_PRODUCER_DONE, worker_id, {"stats": stats}))


# --- GPU encode ------------------------------------------------------------


def build_embedder(
    model_name: str, device: str = "auto", fp16: bool = True, trust_remote_code: bool = True
):
    """Load the new embedding model on GPU FP16. nomic-v1.5 needs
    `trust_remote_code=True` because its modeling code ships with the repo."""
    from sentence_transformers import SentenceTransformer

    from oracle.rag.embedder import resolve_device

    resolved = resolve_device(device)
    logger.info(f"Loading {model_name} (device={resolved}, fp16={fp16}) ...")
    model = SentenceTransformer(
        model_name, device=resolved, trust_remote_code=trust_remote_code
    )
    if fp16 and str(model.device).startswith("cuda"):
        model.half()
        logger.info("Model converted to FP16")
    logger.info(f"Model loaded on {model.device}")
    return model


def encode_batch(model, texts: list[str], batch_size: int, prefix: str = ""):
    if prefix:
        texts = [f"{prefix}{t}" for t in texts]
    return model.encode(
        texts,
        show_progress_bar=False,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


# --- Driver ----------------------------------------------------------------


def reembed(
    source: str,
    target: str,
    db_path: str,
    out_dir: Path,
    dim: int,
    model_name: str,
    workers: int,
    batch_size: int,
    encode_batch_size: int,
    queue_depth: int,
    device: str,
    fp16: bool,
    prefix: str,
    dry_run: bool,
) -> None:
    import numpy as np

    logger.info(f"=== Re-embedding {source} -> {target} with {model_name} ===")
    source_segment_id, id_min, id_max, source_total = get_source_id_range(db_path, source)
    logger.info(
        f"Source has {source_total} chunks (segment {source_segment_id}, "
        f"id range {id_min}..{id_max})"
    )

    target_existing = load_target_existing_ids(out_dir, target)
    logger.info(f"Target {target!r} already has {len(target_existing)} chunks (resume)")

    lookup_path = _dump_lookup(target_existing)
    size_mb = os.path.getsize(lookup_path) / 1024 / 1024
    logger.info(f"Wrote lookup pickle ({size_mb:.1f} MB) at {lookup_path}")

    if dry_run:
        store = None
        model = None
        logger.info("DRY RUN — will count emit candidates only")
    else:
        store = FlatVectorStore(out_dir, target, dim=dim)
        logger.info(
            f"Output: {store.vec_path} (resume from row {store.next_row}) + {store.text_path}"
        )
        model = build_embedder(model_name, device=device, fp16=fp16)

    work_queue: mp.Queue = mp.Queue(maxsize=queue_depth)
    upsert_queue: queue.Queue = queue.Queue(maxsize=queue_depth)
    upsert_done = threading.Event()
    upsert_error: dict = {}

    span = id_max - id_min + 1
    slice_size = (span + workers - 1) // workers
    procs: list[mp.Process] = []
    for wid in range(workers):
        start = id_min + wid * slice_size
        end = min(start + slice_size, id_max + 1)
        if start >= end:
            continue
        p = mp.Process(
            target=_producer_worker,
            args=(
                wid, db_path, source, source_segment_id,
                start, end, lookup_path, batch_size, work_queue,
            ),
            name=f"reembed-prod-{wid}",
            daemon=True,
        )
        p.start()
        procs.append(p)

    def _upsert_loop():
        try:
            while True:
                item = upsert_queue.get()
                if item is None:
                    break
                ids, texts, embs, metas = item
                if not dry_run:
                    embs32 = embs.astype("float32", copy=False)
                    store.append(embs32, ids, texts, metas)
                upsert_queue.task_done()
        except Exception as e:
            upsert_error["err"] = repr(e)
            logger.exception(f"upsert thread crashed: {e}")
        finally:
            upsert_done.set()

    upsert_thread = threading.Thread(target=_upsert_loop, name="flat-writer", daemon=True)
    upsert_thread.start()

    new_chunks = 0
    t0 = time.time()
    last_log = t0
    log_interval = 5.0
    workers_remaining = len(procs)
    worker_stats: dict[int, dict] = {}

    def recompute():
        agg = {"scanned": 0, "skipped_existing": 0, "emitted": 0}
        for s in worker_stats.values():
            for k in agg:
                agg[k] += s.get(k, 0)
        return agg

    try:
        while workers_remaining > 0:
            item = work_queue.get()

            if isinstance(item, tuple) and len(item) == 3 and item[0] == _PRODUCER_DONE:
                _, wid, payload = item
                workers_remaining -= 1
                if "error" in payload:
                    logger.error(f"Worker {wid} errored: {payload['error']}")
                if "stats" in payload:
                    worker_stats[wid] = payload["stats"]
                logger.info(f"Worker {wid} done; {workers_remaining} workers running")
                continue
            if isinstance(item, tuple) and len(item) == 3 and item[0] == _STATS_TICK:
                _, wid, snapshot = item
                worker_stats[wid] = snapshot
                continue

            ids, texts, metas = item
            new_chunks += len(ids)

            if not dry_run:
                embs = encode_batch(model, texts, encode_batch_size, prefix=prefix)
                upsert_queue.put((ids, texts, embs, metas))

            now = time.time()
            if now - last_log >= log_interval:
                agg = recompute()
                rate = new_chunks / (now - t0) if (now - t0) > 0 else 0
                logger.info(
                    f"  {new_chunks} new (scanned {agg['scanned']}, "
                    f"skipped {agg['skipped_existing']}) — {rate:.0f} new/sec, "
                    f"upsert q={upsert_queue.qsize()}"
                )
                last_log = now
    except KeyboardInterrupt:
        logger.warning("Interrupted; terminating workers")
        for p in procs:
            if p.is_alive():
                p.terminate()
        raise
    finally:
        upsert_queue.put(None)
        upsert_thread.join(timeout=300)
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        try:
            os.unlink(lookup_path)
        except OSError:
            pass

    if upsert_error:
        logger.error(f"Upsert error: {upsert_error['err']}")
    if store is not None:
        store.close()
    elapsed = time.time() - t0
    agg = recompute()
    rate = new_chunks / elapsed if elapsed > 0 else 0
    logger.info(
        f"=== Done: {target} ===  new={new_chunks} scanned={agg['scanned']} "
        f"skipped={agg['skipped_existing']}  {elapsed/60:.1f} min  {rate:.0f}/sec"
    )
    if not dry_run and store is not None:
        logger.info(f"  Total rows in {store.vec_path.name}: {store.next_row}")


def main() -> None:
    p = argparse.ArgumentParser(description="Re-embed a ChromaDB collection with a new model")
    p.add_argument("--source", required=True, help="Source collection name")
    p.add_argument("--target", required=True, help="Target collection name (must differ)")
    p.add_argument(
        "--model", default="nomic-ai/nomic-embed-text-v1.5", help="HF embedding model"
    )
    p.add_argument(
        "--prefix", default="search_document: ",
        help="Per-text prefix required by some models (default for nomic-v1.5)",
    )
    p.add_argument("--db-path", default="data/chroma", help="Source chromadb persist dir")
    p.add_argument(
        "--out-dir", default="data/embeddings",
        help="Where to write <target>.vectors.f32 and <target>.text.sqlite",
    )
    p.add_argument("--dim", type=int, default=768, help="Embedding dimension (nomic-v1.5 = 768)")
    p.add_argument("--workers", type=int, default=0, help="0 = min(cpu_count, 12)")
    p.add_argument("--batch-size", type=int, default=2000, help="Producer emit batch")
    p.add_argument("--encode-batch-size", type=int, default=256, help="GPU sub-batch")
    p.add_argument("--queue-depth", type=int, default=8)
    p.add_argument("--device", default="auto")
    fp16 = p.add_mutually_exclusive_group()
    fp16.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    fp16.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.source == args.target:
        print("--source and --target must differ", file=sys.stderr)
        sys.exit(2)

    workers = args.workers if args.workers > 0 else min(mp.cpu_count(), 12)
    reembed(
        source=args.source,
        target=args.target,
        db_path=args.db_path,
        out_dir=Path(args.out_dir),
        dim=args.dim,
        model_name=args.model,
        workers=workers,
        batch_size=args.batch_size,
        encode_batch_size=args.encode_batch_size,
        queue_depth=args.queue_depth,
        device=args.device,
        fp16=args.fp16,
        prefix=args.prefix,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
