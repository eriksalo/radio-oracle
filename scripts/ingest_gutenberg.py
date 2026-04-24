#!/usr/bin/env python3
"""Ingest Project Gutenberg text files into ChromaDB.

Usage:
    python scripts/ingest_gutenberg.py data/knowledge/gutenberg/ --dry-run

Expects a directory of .txt files (one per book).
"""

import argparse
import sys
from pathlib import Path

from loguru import logger


def ingest_gutenberg(
    text_dir: Path,
    dry_run: bool = False,
    batch_size: int = 500,
) -> None:
    if not dry_run:
        import chromadb

        from config.settings import settings
        from oracle.rag.chunker import chunk_text
        from oracle.rag.embedder import Embedder

        embedder = Embedder()
        client = chromadb.PersistentClient(path=str(settings.chroma_path))
        collection = client.get_or_create_collection("gutenberg")

    txt_files = sorted(text_dir.glob("*.txt"))
    logger.info(f"Found {len(txt_files)} Gutenberg text files")

    processed = 0
    chunks_total = 0
    batch_docs: list[str] = []
    batch_ids: list[str] = []
    batch_metas: list[dict] = []

    for txt_file in txt_files:
        try:
            text = txt_file.read_text(encoding="utf-8", errors="replace")
            if len(text) < 200:
                continue

            title = txt_file.stem.replace("_", " ").title()

            if dry_run:
                if processed < 5:
                    logger.info(f"  [{processed}] {title}: {len(text)} chars")
                processed += 1
                continue

            chunks = chunk_text(text)
            for j, chunk in enumerate(chunks):
                doc_id = f"gutenberg_{txt_file.stem}_{j}"
                batch_docs.append(chunk)
                batch_ids.append(doc_id)
                batch_metas.append({"title": title, "source": "gutenberg"})
                chunks_total += 1

            if len(batch_docs) >= batch_size:
                embeddings = embedder.embed(batch_docs)
                collection.add(
                    documents=batch_docs,
                    embeddings=embeddings,
                    ids=batch_ids,
                    metadatas=batch_metas,
                )
                logger.info(f"  Ingested {chunks_total} chunks ({processed} books)")
                batch_docs, batch_ids, batch_metas = [], [], []

            processed += 1

        except Exception as e:
            logger.warning(f"Error processing {txt_file.name}: {e}")
            continue

    if batch_docs and not dry_run:
        embeddings = embedder.embed(batch_docs)
        collection.add(
            documents=batch_docs,
            embeddings=embeddings,
            ids=batch_ids,
            metadatas=batch_metas,
        )

    logger.info(f"Done (gutenberg): {processed} books, {chunks_total} chunks")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Gutenberg texts into ChromaDB")
    parser.add_argument("text_dir", type=Path, help="Directory of .txt files")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    if not args.text_dir.is_dir():
        logger.error(f"Not a directory: {args.text_dir}")
        sys.exit(1)

    ingest_gutenberg(args.text_dir, args.dry_run, args.batch_size)


if __name__ == "__main__":
    main()
