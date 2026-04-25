#!/usr/bin/env python3
"""
Verify ChromaDB contents and test RAG queries.

Usage:
    python scripts/verify_chroma.py                    # show collection stats
    python scripts/verify_chroma.py "how to purify water"  # test retrieval
"""

import sys

import chromadb
from loguru import logger


def show_stats(db_path: str = "data/chroma") -> None:
    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()
    if not collections:
        logger.warning("No collections found.")
        return

    total = 0
    for coll in collections:
        name = coll if isinstance(coll, str) else coll.name
        c = client.get_collection(name)
        count = c.count()
        total += count
        print(f"  {name}: {count:,} chunks")
    print(f"  TOTAL: {total:,} chunks")


def test_query(query: str, db_path: str = "data/chroma", top_k: int = 5) -> None:
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=db_path)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = model.encode([query])[0].tolist()

    print(f'\nQuery: "{query}"\n')

    for coll in client.list_collections():
        name = coll if isinstance(coll, str) else coll.name
        collection = client.get_collection(name)
        if collection.count() == 0:
            continue
        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, collection.count()),
        )
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                source = meta.get("source", name)
                title = meta.get("title", "?")
                print(f"[{source}] {title} (dist={dist:.3f})")
                print(f"  {doc[:200]}...")
                print()


def main() -> None:
    if len(sys.argv) > 1:
        test_query(sys.argv[1])
    else:
        show_stats()


if __name__ == "__main__":
    main()
