#!/usr/bin/env python3
"""Build a FAISS IVF-PQ index from a flat-file embedding store.

Consumes the artifacts produced by `scripts/reembed_collection.py`:

    <in-dir>/<name>.vectors.f32   raw float32, shape (n, dim)
    <in-dir>/<name>.text.sqlite   chunks table: (row_id, chunk_id, text, ...)

Writes:

    <out-dir>/<name>.index        FAISS IVF-PQ index (IDMap = positional)
    <out-dir>/<name>.sqlite       faiss_idmap table: (faiss_row, chunk_id, text, ...)

`faiss_row` in the output matches `row_id` in the input — same vectors, same
order, just one renamed table to keep the FAISS backend happy.

Usage:
    python scripts/build_faiss_ivfpq.py \\
        --name wikipedia_v2 \\
        --in-dir data/embeddings \\
        --out-dir data/faiss \\
        --dim 768 --nlist 4096 --pq-m 64
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import numpy as np
from loguru import logger


def load_vectors(path: Path, dim: int) -> np.ndarray:
    """Memmap the .vectors.f32 file into an (n, dim) float32 array."""
    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size % (dim * 4) != 0:
        raise RuntimeError(
            f"{path} size {size} is not a multiple of dim*4 ({dim*4})"
        )
    n = size // (dim * 4)
    logger.info(f"Memory-mapping {path.name}: {n:,} vectors x {dim} dim ({size/1024/1024/1024:.2f} GB)")
    return np.memmap(path, dtype=np.float32, mode="r", shape=(n, dim))


def build_index(
    vectors: np.ndarray, nlist: int, pq_m: int, pq_bits: int, train_sample: int
):
    import faiss

    n, dim = vectors.shape
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, pq_bits)
    index.metric_type = faiss.METRIC_INNER_PRODUCT

    if not index.is_trained:
        sample = vectors
        if n > train_sample:
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(n, size=train_sample, replace=False)
            sample = np.ascontiguousarray(vectors[idx])
        else:
            sample = np.ascontiguousarray(sample)
        logger.info(f"Training IVF-PQ on {len(sample):,} vectors (dim={dim})")
        t0 = time.time()
        index.train(sample)
        logger.info(f"Training done in {time.time()-t0:.1f}s")

    t0 = time.time()
    chunk = 200_000
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        index.add(np.ascontiguousarray(vectors[start:end]))
        if (start // chunk) % 10 == 0:
            elapsed = time.time() - t0
            rate = end / elapsed if elapsed > 0 else 0
            logger.info(f"  added {end:,}/{n:,} ({rate:.0f}/s)")
    logger.info(f"Added {n:,} vectors in {time.time()-t0:.1f}s")
    return index


def write_idmap_sqlite(out_path: Path, in_path: Path) -> None:
    """Copy the chunks table from the input store into a faiss_idmap-shaped
    table, preserving row_id = faiss_row."""
    if out_path.exists():
        out_path.unlink()
    con_in = sqlite3.connect(f"file:{in_path}?mode=ro", uri=True)
    con_out = sqlite3.connect(str(out_path))
    try:
        con_out.execute(
            """CREATE TABLE faiss_idmap (
                faiss_row INTEGER PRIMARY KEY,
                chunk_id  TEXT NOT NULL,
                text      TEXT NOT NULL,
                source    TEXT,
                url       TEXT,
                title     TEXT,
                chunk_index INTEGER
            )"""
        )
        rows = con_in.execute(
            "SELECT row_id, chunk_id, text, source, url, title, chunk_index FROM chunks"
        )
        batch: list[tuple] = []
        total = 0
        for r in rows:
            batch.append(r)
            if len(batch) >= 10_000:
                con_out.executemany(
                    "INSERT INTO faiss_idmap VALUES (?,?,?,?,?,?,?)", batch
                )
                total += len(batch)
                batch.clear()
        if batch:
            con_out.executemany(
                "INSERT INTO faiss_idmap VALUES (?,?,?,?,?,?,?)", batch
            )
            total += len(batch)
        con_out.commit()
        logger.info(f"Wrote {out_path} ({total:,} rows)")
    finally:
        con_in.close()
        con_out.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Build FAISS IVF-PQ from flat-file embeddings")
    p.add_argument("--name", required=True, help="Basename for input + output files")
    p.add_argument("--in-dir", default="data/embeddings")
    p.add_argument("--out-dir", default="data/faiss")
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--nlist", type=int, default=4096)
    p.add_argument("--pq-m", type=int, default=64)
    p.add_argument("--pq-bits", type=int, default=8)
    p.add_argument("--train-sample", type=int, default=1_000_000)
    args = p.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    vec_path = in_dir / f"{args.name}.vectors.f32"
    text_path = in_dir / f"{args.name}.text.sqlite"
    vectors = load_vectors(vec_path, args.dim)

    index = build_index(
        vectors, args.nlist, args.pq_m, args.pq_bits, args.train_sample
    )

    import faiss

    idx_path = out_dir / f"{args.name}.index"
    map_path = out_dir / f"{args.name}.sqlite"
    faiss.write_index(index, str(idx_path))
    logger.info(f"Wrote {idx_path}")

    write_idmap_sqlite(map_path, text_path)


if __name__ == "__main__":
    main()
