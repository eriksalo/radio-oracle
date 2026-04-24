#!/usr/bin/env python3
"""Generic ZIM ingestion script — works for iFixit, Wikibooks, WikiMed, etc.

Usage:
    python scripts/ingest_generic_zim.py data/knowledge/ifixit.zim --collection ifixit
    python scripts/ingest_generic_zim.py data/knowledge/wikibooks.zim --collection wikibooks
"""

import argparse
import sys
from pathlib import Path

from loguru import logger


def ingest_zim(
    zim_path: Path,
    collection_name: str,
    dry_run: bool = False,
    batch_size: int = 500,
) -> None:
    from libzim.reader import Archive
    from selectolax.parser import HTMLParser

    if not dry_run:
        import chromadb

        from config.settings import settings
        from oracle.rag.chunker import chunk_text
        from oracle.rag.embedder import Embedder

        embedder = Embedder()
        client = chromadb.PersistentClient(path=str(settings.chroma_path))
        collection = client.get_or_create_collection(collection_name)

    zim = Archive(str(zim_path))
    entry_count = zim.entry_count
    logger.info(f"ZIM archive ({collection_name}): {entry_count} entries")

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

                for tag in tree.css("script, style"):
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
                    doc_id = f"{collection_name}_{i}_{j}"
                    batch_docs.append(chunk)
                    batch_ids.append(doc_id)
                    batch_metas.append(
                        {
                            "title": title,
                            "source": collection_name,
                        }
                    )
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

    if batch_docs and not dry_run:
        embeddings = embedder.embed(batch_docs)
        collection.add(
            documents=batch_docs,
            embeddings=embeddings,
            ids=batch_ids,
            metadatas=batch_metas,
        )

    logger.info(f"Done ({collection_name}): {processed} articles, {chunks_total} chunks")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ZIM file into ChromaDB")
    parser.add_argument("zim_file", type=Path, help="Path to ZIM file")
    parser.add_argument("--collection", required=True, help="ChromaDB collection name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    if not args.zim_file.exists():
        logger.error(f"ZIM file not found: {args.zim_file}")
        sys.exit(1)

    ingest_zim(args.zim_file, args.collection, args.dry_run, args.batch_size)


if __name__ == "__main__":
    main()
