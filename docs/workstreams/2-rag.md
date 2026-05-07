# Workstream 2: Large-data ingest / RAG

Build the offline knowledge base. Runs on a workstation with a GPU; the
output is a `data/chroma/` directory rsync'd to the Jetson.

## Status

Working end-to-end. Wikipedia, iFixit, Wikibooks, Gutenberg, etc. all
ingest via the unified `scripts/ingest_zim.py`. GPU/FP16/pipelined embedding
is in. Resume via direct SQLite reads of existing IDs.

## Scope

- ZIM file parsing (Wikipedia, iFixit, Wikibooks, WikiMed, Gutenberg, CrashCourse)
- Stack Exchange XML / PDF / arbitrary text ingestion (planned)
- Text chunking and embedding (sentence-transformers, all-MiniLM-L6-v2)
- ChromaDB collection management
- Query-time retrieval (read-only on Jetson)
- Embedding device/FP16/batch tuning for workstation GPUs

## File ownership

```
scripts/
  ingest_zim.py            # Unified ZIM ingestion (GPU + resume)
  ingest_wikipedia.py      # Legacy single-source ingest scripts
  ingest_generic_zim.py    # (kept for reference; ingest_zim.py is preferred)
  ingest_gutenberg.py
  download_knowledge.sh    # Download ZIMs from Kiwix mirrors
  download_knowledge.ps1   # Same, Windows PowerShell
  verify_chroma.py         # Validate ChromaDB after ingest
oracle/rag/
  chunker.py               # Text splitting (size + overlap)
  embedder.py              # sentence-transformers wrapper (CUDA/FP16)
  retriever.py             # Query-time retrieval
tests/
  test_chunker.py
```

## Settings

```bash
ORACLE_CHROMA_PATH=data/chroma
ORACLE_EMBEDDING_MODEL=all-MiniLM-L6-v2
ORACLE_EMBEDDING_DEVICE=auto    # auto | cpu | cuda | cuda:N
ORACLE_EMBEDDING_FP16=true      # only honored on CUDA
ORACLE_EMBEDDING_BATCH_SIZE=256
ORACLE_RAG_TOP_K=5
ORACLE_CHUNK_SIZE=512
ORACLE_CHUNK_OVERLAP=64
ORACLE_RAG_COLLECTIONS=         # comma-separated allowlist; empty = all.
                                # Set this on memory-constrained hosts so a
                                # full Wikipedia HNSW index doesn't load.
```

## Dependencies

```bash
pip install -e ".[ingest,rag]"
# also need a CUDA-enabled torch wheel on a workstation with NVIDIA GPU
```

External: ChromaDB on disk (no server). Sentence-transformers downloads model
weights on first use.

## Interface contract

**Output** (consumed by Workstream 6 — LLM):
- `data/chroma/` directory with named collections: `wikipedia`, `ifixit`,
  `wikibooks`, `wikimed`, `gutenberg`, `crashcourse`, etc.

**Consumer**: `oracle/rag/retriever.py::Retriever.query()` returns ranked
chunks. `oracle/core.py::_try_rag_query` calls the retriever lazily —
missing collections degrade to "no RAG context" without crashing.

**Consumes**: nothing at runtime. Ingest is offline / one-shot.

## Standalone exercise

```bash
# Workstation: ingest one ZIM with full GPU acceleration
python scripts/ingest_zim.py path/to/wikipedia.zim \
    --device cuda --fp16 --encode-batch-size 512 --batch-size 4000

# Verify the resulting database
python scripts/verify_chroma.py data/chroma/

# Query the retriever directly (Jetson or workstation)
python -c "
from oracle.rag.retriever import Retriever
r = Retriever()
print(r.list_collections())
for hit in r.query('how does a vacuum tube work', top_k=3):
    print(hit['source'], '-', hit['text'][:120])
"

pytest tests/test_chunker.py
```

## TODO

- [ ] Stack Exchange XML ingest script
- [ ] Generic PDF ingestion (army field manuals, etc.)
- [ ] Collection stats summary script (counts + disk usage)
- [ ] Re-rank with cross-encoder before returning top-k
- [ ] Hybrid search (BM25 + dense)
