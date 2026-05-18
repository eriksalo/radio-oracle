#!/usr/bin/env python3
"""Build a FAISS IVF-PQ index from a ChromaDB collection's HNSW data.

Reads `data_level0.bin` directly via mmap (much faster than chromadb's
get-by-id API), trains an IVF-PQ over a sample of vectors, adds all
vectors in HNSW row order, and writes:

    <out-dir>/<name>.index    FAISS index (IVF-PQ, IDMap = positional)
    <out-dir>/<name>.sqlite   id-map: row_id -> chunk_id + text + metadata

The companion `oracle/rag/backends/faiss_ivfpq.py` loads these two
artifacts and serves queries.

Why direct mmap instead of `collection.get(include=['embeddings'])`?
For 11.4 M vectors a batched API read takes ~hours; mmap parses the
file in ~minutes.

Usage:
    python scripts/build_faiss_ivfpq.py \\
        --source wikipedia_v2 \\
        --db-path data/chroma \\
        --out-dir data/faiss \\
        --nlist 4096 --pq-m 64
"""

from __future__ import annotations

import argparse
import pickle
import sqlite3
import struct
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger


# --- Chroma HNSW reader -----------------------------------------------------

# Chroma-fork hnswlib header.bin layout (100 bytes total):
#   0x00  u32 PERSISTENCE_VERSION
#   0x04  u64 offsetLevel0_
#   0x0C  u64 max_elements_
#   0x14  u64 cur_element_count
#   0x1C  u64 size_data_per_element_
#   0x24  u64 label_offset_
#   0x2C  u64 offsetData_         (= 132; vector starts here in each record)
#   0x34  i32 maxlevel_
#   0x38  u32 enterpoint_node_
#   0x3C  u64 maxM_
#   0x44  u64 maxM0_
#   0x4C  u64 M_
#   0x54  f64 mult_
#   0x5C  u64 ef_construction_

def read_hnsw_header(header_path: Path) -> dict:
    b = header_path.read_bytes()
    if len(b) != 100:
        raise ValueError(f"Unexpected header.bin size: {len(b)}")
    pv = struct.unpack_from("<I", b, 0)[0]
    offL0, max_el, cur_el, sde, lbloff, datoff = struct.unpack_from("<6Q", b, 4)
    return {
        "persistence_version": pv,
        "max_elements": max_el,
        "cur_element_count": cur_el,
        "size_data_per_element": sde,
        "label_offset": lbloff,
        "offset_data": datoff,
    }


def get_segment_dir(db_path: Path, collection_name: str) -> Path:
    """Locate the VECTOR segment directory for a chromadb collection."""
    sqlite_path = db_path / "chroma.sqlite3"
    con = sqlite3.connect(str(sqlite_path))
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT s.id FROM segments s
            JOIN collections c ON c.id = s.collection
            WHERE c.name = ? AND s.scope = 'VECTOR'
            """,
            (collection_name,),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"No VECTOR segment for {collection_name!r}")
        return db_path / row[0]
    finally:
        con.close()


def read_vectors_and_labels(
    segment_dir: Path, expected_dim: int | None = None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (vectors[n,dim] float32, labels[n] uint64, header dict).

    Vectors are mmap-backed; copy into a contiguous array before training to
    avoid surprises in FAISS C++ code that assumes ownership."""
    header = read_hnsw_header(segment_dir / "header.bin")
    n = header["cur_element_count"]
    sde = header["size_data_per_element"]
    off = header["offset_data"]

    # dim is implied by size_data_per_element_:
    # sde = 4 (linklistsizeint) + 4 * maxM0 (links) + 4 * dim + 8 (label)
    # The chunk between offsetData and (offsetData + 4*dim) is the vector,
    # and the trailing 8 bytes are the label. So:
    bytes_for_vector_and_label = sde - off
    dim = (bytes_for_vector_and_label - 8) // 4
    if expected_dim is not None and dim != expected_dim:
        raise ValueError(
            f"Computed dim={dim} from header but expected {expected_dim}; "
            f"check the segment {segment_dir.name}"
        )
    logger.info(f"HNSW header: n={n}, sde={sde}, offsetData={off}, dim={dim}")

    data_path = segment_dir / "data_level0.bin"
    expected_size = n * sde
    actual = data_path.stat().st_size
    if actual != expected_size:
        logger.warning(
            f"data_level0.bin size mismatch: expected {expected_size} got {actual}"
        )

    raw = np.memmap(data_path, dtype=np.uint8, mode="r", shape=(n, sde))
    vec_bytes = raw[:, off : off + 4 * dim]
    # .view() requires contiguous memory along the last axis; raw rows are
    # contiguous so a per-row slice is also contiguous in C order.
    vectors_view = vec_bytes.reshape(-1).view(np.float32).reshape(n, dim)
    # Copy out of the memmap to free file resources for FAISS training.
    vectors = np.ascontiguousarray(vectors_view, dtype=np.float32)
    label_bytes = raw[:, off + 4 * dim : off + 4 * dim + 8]
    labels = label_bytes.reshape(-1).view(np.uint64).reshape(n)
    labels_arr = np.ascontiguousarray(labels)
    del raw
    return vectors, labels_arr, header


def load_label_to_id(segment_dir: Path) -> dict[int, str]:
    """Read index_metadata.pickle from a VECTOR segment to map
    hnswlib internal label (uint64) to chromadb external id (md5)."""
    pkl = segment_dir / "index_metadata.pickle"
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    # PersistentData has `id_to_label: dict[str -> int]` and `label_to_id`
    # depending on chromadb version. Try both shapes.
    if hasattr(data, "label_to_id"):
        return dict(data.label_to_id)
    if hasattr(data, "id_to_label"):
        return {v: k for k, v in data.id_to_label.items()}
    # Plain dict fallback
    if isinstance(data, dict):
        if "label_to_id" in data:
            return dict(data["label_to_id"])
        if "id_to_label" in data:
            return {v: k for k, v in data["id_to_label"].items()}
    raise RuntimeError("Could not find label_to_id mapping in index_metadata.pickle")


# --- Text reader from chromadb sqlite --------------------------------------


def read_chunk_metadata(db_path: Path, collection_name: str, chunk_ids: list[str]) -> dict:
    """Map md5 -> {text, url, title, chunk_index, source} from chromadb sqlite.

    Bulk-load all metadata for the collection rather than per-id round trips."""
    sqlite_path = db_path / "chroma.sqlite3"
    con = sqlite3.connect(str(sqlite_path))
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT
                e.embedding_id,
                em_doc.string_value,
                em_url.string_value,
                em_title.string_value,
                em_chunk.int_value,
                em_source.string_value
            FROM embeddings e
            JOIN segments s ON s.id = e.segment_id
            JOIN collections c ON c.id = s.collection
            LEFT JOIN embedding_metadata em_doc
                ON em_doc.id = e.id AND em_doc.key = 'chroma:document'
            LEFT JOIN embedding_metadata em_url
                ON em_url.id = e.id AND em_url.key = 'url'
            LEFT JOIN embedding_metadata em_title
                ON em_title.id = e.id AND em_title.key = 'title'
            LEFT JOIN embedding_metadata em_chunk
                ON em_chunk.id = e.id AND em_chunk.key = 'chunk_index'
            LEFT JOIN embedding_metadata em_source
                ON em_source.id = e.id AND em_source.key = 'source'
            WHERE c.name = ? AND s.scope = 'METADATA'
            """,
            (collection_name,),
        )
        return {
            row[0]: {
                "text": row[1] or "",
                "url": row[2] or "",
                "title": row[3] or "",
                "chunk_index": int(row[4]) if row[4] is not None else 0,
                "source": row[5] or collection_name,
            }
            for row in cur.fetchall()
        }
    finally:
        con.close()


# --- FAISS index writer -----------------------------------------------------


def build_index(
    vectors: np.ndarray, nlist: int, pq_m: int, pq_bits: int, train_sample: int
):
    import faiss

    n, dim = vectors.shape
    quantizer = faiss.IndexFlatIP(dim)  # inner product for normalized vectors
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, pq_bits)
    index.metric_type = faiss.METRIC_INNER_PRODUCT

    if not index.is_trained:
        sample = vectors
        if n > train_sample:
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(n, size=train_sample, replace=False)
            sample = np.ascontiguousarray(vectors[idx])
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


def write_idmap_sqlite(
    out_path: Path,
    labels: np.ndarray,
    label_to_id: dict,
    metadata: dict,
) -> None:
    """Write (faiss_row, chunk_id, text, source, url, title, chunk_index)."""
    if out_path.exists():
        out_path.unlink()
    con = sqlite3.connect(str(out_path))
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE faiss_idmap (
            faiss_row INTEGER PRIMARY KEY,
            chunk_id  TEXT NOT NULL,
            text      TEXT NOT NULL,
            source    TEXT,
            url       TEXT,
            title     TEXT,
            chunk_index INTEGER
        )
        """
    )
    misses = 0
    rows = []
    for row_idx, label in enumerate(labels):
        cid = label_to_id.get(int(label))
        if cid is None:
            misses += 1
            continue
        m = metadata.get(cid)
        if m is None:
            misses += 1
            continue
        rows.append(
            (row_idx, cid, m["text"], m["source"], m["url"], m["title"], m["chunk_index"])
        )
        if len(rows) >= 10000:
            cur.executemany(
                "INSERT INTO faiss_idmap VALUES (?,?,?,?,?,?,?)", rows
            )
            rows.clear()
    if rows:
        cur.executemany("INSERT INTO faiss_idmap VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    logger.info(f"Wrote {out_path} ({misses} misses skipped)")


def main() -> None:
    p = argparse.ArgumentParser(description="Build FAISS IVF-PQ from a chromadb collection")
    p.add_argument("--source", required=True, help="ChromaDB collection name")
    p.add_argument("--db-path", default="data/chroma", help="ChromaDB persist path")
    p.add_argument("--out-dir", default="data/faiss", help="Where to write index + sqlite")
    p.add_argument("--name", default=None, help="Output basename (default: --source)")
    p.add_argument("--nlist", type=int, default=4096, help="IVF cluster count")
    p.add_argument("--pq-m", type=int, default=64, help="PQ sub-quantizers")
    p.add_argument("--pq-bits", type=int, default=8, help="Bits per PQ code")
    p.add_argument("--train-sample", type=int, default=1_000_000)
    p.add_argument("--expected-dim", type=int, default=None)
    args = p.parse_args()

    db_path = Path(args.db_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or args.source

    seg = get_segment_dir(db_path, args.source)
    logger.info(f"Source segment: {seg}")

    vectors, labels, header = read_vectors_and_labels(seg, expected_dim=args.expected_dim)
    label_to_id = load_label_to_id(seg)
    logger.info(f"label_to_id has {len(label_to_id)} entries")

    logger.info("Loading chunk metadata from chromadb sqlite ...")
    t0 = time.time()
    metadata = read_chunk_metadata(db_path, args.source, list(label_to_id.values()))
    logger.info(f"Loaded {len(metadata)} metadata rows in {time.time()-t0:.1f}s")

    index = build_index(vectors, args.nlist, args.pq_m, args.pq_bits, args.train_sample)

    import faiss

    idx_path = out_dir / f"{name}.index"
    map_path = out_dir / f"{name}.sqlite"
    faiss.write_index(index, str(idx_path))
    logger.info(f"Wrote {idx_path}")

    write_idmap_sqlite(map_path, labels, label_to_id, metadata)


if __name__ == "__main__":
    main()
