# Workstream 1: RAG Ingest

Runs on a workstation with GPU. Produces a ChromaDB directory that gets rsync'd to the Jetson.

## Scope

- ZIM file parsing (Wikipedia, iFixit, Wikibooks, WikiMed)
- Gutenberg book ingestion
- Stack Exchange XML ingestion
- Text chunking and embedding
- ChromaDB collection management
- Verification scripts

## Key Files

```
scripts/
  ingest_wikipedia.py      # Wikipedia ZIM -> ChromaDB
  ingest_generic_zim.py    # Any ZIM file -> ChromaDB
  ingest_zim.py            # Unified ZIM ingestion with GPU + resume
  ingest_gutenberg.py      # Project Gutenberg -> ChromaDB
  verify_chroma.py         # Validate ChromaDB after ingest
oracle/rag/
  chunker.py               # Text splitting (size + overlap)
  embedder.py              # Sentence-transformer wrapper
  retriever.py             # Query-time retrieval (Jetson-side)
```

## Dependencies

```
pip install -e ".[ingest,rag]"
```

## Interface Contract

**Output**: `data/chroma/` directory with named collections (wikipedia, ifixit, gutenberg, etc.)

**Consumer**: `oracle/rag/retriever.py` queries these collections at runtime. Retriever is read-only — ingest scripts are the only writers.

## Testing

```bash
# Verify collections exist and have expected doc counts
python scripts/verify_chroma.py data/chroma/

# Unit tests for chunker
pytest tests/test_chunker.py
```

## TODO

- [ ] Stack Exchange XML ingest script
- [ ] Army field manual ingest (PDF -> text -> chunks)
- [ ] Incremental re-ingest (skip already-embedded docs)
- [ ] Collection stats dashboard / summary script
