#!/usr/bin/env python3
"""
Unified ZIM ingestion script — replaces ingest_wikipedia.py, ingest_generic_zim.py,
and ingest_gutenberg.py (Gutenberg is now a ZIM too).

Reads ZIM files, extracts article text, chunks it, embeds with
all-MiniLM-L6-v2 (auto-detects GPU), and stores in ChromaDB.

Features over the old scripts:
  - Single script for all ZIM sources
  - Auto-detects collection name from ZIM filename
  - GPU auto-detection (huge speedup on workstations)
  - Deterministic doc IDs for resume/dedup
  - --all mode to ingest every ZIM in a directory
  - Rate logging and progress tracking

Usage:
    python scripts/ingest_zim.py <file.zim> [--collection <name>] [--batch-size 2000] [--dry-run]
    python scripts/ingest_zim.py --all --zim-dir /path/to/zims [--batch-size 2000] [--dry-run]
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

from loguru import logger
from selectolax.parser import HTMLParser


# ---------------------------------------------------------------------------
# ZIM reading
# ---------------------------------------------------------------------------

def iter_zim_articles(zim_path: str):
    """Yield (url, title, html) for every article in a ZIM file."""
    from libzim.reader import Archive  # type: ignore[import-untyped]

    archive = Archive(zim_path)
    entry_count = archive.entry_count
    logger.info(f"ZIM: {Path(zim_path).name} — {entry_count} entries")

    for i in range(entry_count):
        try:
            entry = archive._get_entry_by_id(i)
        except Exception:
            continue

        if entry.is_redirect:
            continue

        item = entry.get_item()
        if "html" not in item.mimetype:
            continue

        try:
            html = bytes(item.content).decode("utf-8", errors="replace")
        except Exception:
            continue

        yield entry.path, entry.title or "", html

        if (i + 1) % 50_000 == 0:
            logger.info(f"  scanned {i + 1} / {entry_count} entries")


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


def build_embedder():
    """Load sentence-transformers model, auto-detecting GPU."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info(f"Embedding model loaded (device: {model.device})")
    return model


def flush_batch(
    collection,
    embedder,
    doc_ids: list[str],
    texts: list[str],
    metadatas: list[dict],
):
    """Embed a batch and upsert into ChromaDB."""
    embeddings = embedder.encode(texts, show_progress_bar=False, batch_size=256)
    collection.upsert(
        ids=doc_ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )


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
# Main ingestion loop
# ---------------------------------------------------------------------------

def ingest_zim(
    zim_path: str,
    collection_name: str,
    db_path: str = "data/chroma",
    batch_size: int = 2000,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    dry_run: bool = False,
) -> None:
    logger.info(f"=== Ingesting {Path(zim_path).name} -> collection '{collection_name}' ===")
    logger.info(f"ChromaDB: {db_path} | batch: {batch_size} | chunk: {chunk_size}w / {chunk_overlap} overlap")

    if not dry_run:
        collection = get_chroma_collection(db_path, collection_name)
        embedder = build_embedder()
        existing_ids = load_existing_ids(collection, db_path)
        existing_urls = load_existing_urls(collection_name, db_path)
        logger.info(f"Collection '{collection_name}' has {len(existing_ids)} existing chunks across {len(existing_urls)} URLs")
    else:
        logger.info("DRY RUN — will not embed or store anything")
        existing_ids = set()
        existing_urls = set()

    total_articles = 0
    total_chunks = 0
    new_chunks = 0
    skipped_short = 0
    skipped_existing = 0
    skipped_url = 0
    batch_ids: list[str] = []
    batch_texts: list[str] = []
    batch_metas: list[dict] = []
    t0 = time.time()

    for url, title, html in iter_zim_articles(zim_path):
        if url[:500] in existing_urls:
            skipped_url += 1
            continue
        text = extract_text(html)
        if len(text) < 100:
            skipped_short += 1
            continue

        chunks = chunk_text(text, chunk_size, chunk_overlap)
        if not chunks:
            skipped_short += 1
            continue

        total_articles += 1

        for idx, chunk in enumerate(chunks):
            doc_id = make_doc_id(collection_name, url, idx)
            total_chunks += 1

            if doc_id in existing_ids:
                skipped_existing += 1
                continue

            if dry_run:
                new_chunks += 1
                continue

            batch_ids.append(doc_id)
            batch_texts.append(chunk)
            batch_metas.append({
                "source": collection_name,
                "title": title[:500],
                "url": url[:500],
                "chunk_index": idx,
            })
            new_chunks += 1

            if len(batch_ids) >= batch_size:
                flush_batch(collection, embedder, batch_ids, batch_texts, batch_metas)
                elapsed = time.time() - t0
                rate = new_chunks / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  {new_chunks} new chunks ({total_chunks} total, "
                    f"{skipped_existing} skipped, {total_articles} articles) "
                    f"— {rate:.0f} new/sec"
                )
                batch_ids.clear()
                batch_texts.clear()
                batch_metas.clear()

    if batch_ids and not dry_run:
        flush_batch(collection, embedder, batch_ids, batch_texts, batch_metas)

    elapsed = time.time() - t0
    logger.info(f"=== Done: {collection_name} ===")
    logger.info(
        f"  Articles: {total_articles} | Total chunks: {total_chunks} | "
        f"New: {new_chunks} | Skipped existing: {skipped_existing} | "
        f"Skipped URL: {skipped_url} | Skipped short: {skipped_short}"
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
    parser.add_argument("--batch-size", type=int, default=2000, help="Embedding batch size (default: 2000)")
    parser.add_argument("--chunk-size", type=int, default=512, help="Words per chunk (default: 512)")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="Overlap between chunks (default: 64)")
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
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
