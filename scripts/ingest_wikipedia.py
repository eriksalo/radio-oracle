#!/usr/bin/env python3
"""Ingest Wikipedia ZIM file into ChromaDB for RAG.

Usage:
    python scripts/ingest_wikipedia.py data/knowledge/wikipedia_en_all_nopic_latest.zim [--dry-run]

Run on a workstation with GPU for faster embedding. Then rsync data/chroma/ to Jetson.
"""

import argparse
import sys
from pathlib import Path

from loguru import logger


def ingest_zim(zim_path: Path, dry_run: bool = False, batch_size: int = 500) -> None:
    from libzim.reader import Archive
    from selectolax.parser import HTMLParser

    # Only import heavy deps if not dry-run
    if not dry_run:
        import chromadb

        from config.settings import settings
        from oracle.rag.chunker import chunk_text
        from oracle.rag.embedder import Embedder

        embedder = Embedder()
        client = chromadb.PersistentClient(path=str(settings.chroma_path))
        collection = client.get_or_create_collection("wikipedia")

    zim = Archive(str(zim_path))
    entry_count = zim.entry_count
    logger.info(f"ZIM archive: {entry_count} entries")

    processed = 0
    chunks_total = 0
    batch_docs: list[str] = []
    batch_ids: list[str] = []
    batch_metas: list[dict] = []

    for i in range(entry_count):
        try:
            entry = zim._get_entry_by_id(i)
            if not entry.is_redirect and entry.get_item().mimetype == "text/html":
                html = bytes(entry.get_item().content).decode("utf-8", errors="replace")
                tree = HTMLParser(html)

                # Remove scripts, styles
                for tag in tree.css("script, style, table, sup.reference"):
                    tag.decompose()

                text = tree.text(separator="\n\n")
                if len(text) < 100:
                    continue

                title = entry.title

                if dry_run:
                    if processed < 5:
                        logger.info(f"  [{processed}] {title}: {len(text)} chars")
                    processed += 1
                    continue

                chunks = chunk_text(text)
                for j, chunk in enumerate(chunks):
                    doc_id = f"wiki_{i}_{j}"
                    batch_docs.append(chunk)
                    batch_ids.append(doc_id)
                    batch_metas.append({"title": title, "source": "wikipedia"})
                    chunks_total += 1

                if len(batch_docs) >= batch_size:
                    embeddings = embedder.embed(batch_docs)
                    collection.add(
                        documents=batch_docs,
                        embeddings=embeddings,
                        ids=batch_ids,
                        metadatas=batch_metas,
                    )
                    logger.info(f"  Ingested {chunks_total} chunks ({processed} articles)")
                    batch_docs, batch_ids, batch_metas = [], [], []

                processed += 1

        except Exception as e:
            logger.warning(f"Error processing entry {i}: {e}")
            continue

    # Flush remaining batch
    if batch_docs and not dry_run:
        embeddings = embedder.embed(batch_docs)
        collection.add(
            documents=batch_docs,
            embeddings=embeddings,
            ids=batch_ids,
            metadatas=batch_metas,
        )

    logger.info(f"Done: {processed} articles, {chunks_total} chunks")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Wikipedia ZIM into ChromaDB")
    parser.add_argument("zim_file", type=Path, help="Path to Wikipedia ZIM file")
    parser.add_argument("--dry-run", action="store_true", help="Preview without ingesting")
    parser.add_argument("--batch-size", type=int, default=500, help="Embedding batch size")
    args = parser.parse_args()

    if not args.zim_file.exists():
        logger.error(f"ZIM file not found: {args.zim_file}")
        sys.exit(1)

    ingest_zim(args.zim_file, dry_run=args.dry_run, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
