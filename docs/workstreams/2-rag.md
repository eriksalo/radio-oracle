# Workstream 2: Large-data ingest / RAG

Build the offline knowledge base. Runs on a workstation with an NVIDIA GPU;
the output is a `data/faiss/` directory rsync'd to the Jetson. ChromaDB
lives on the workstation only — it's the read-only staging area for chunk
text + legacy MiniLM embeddings, not a production retrieval target.

## Status

In production. 22.5M chunks across 7 collections (wikipedia, gutenberg,
wikibooks, wikimed, ifixit, crashcourse, music) deployed to the Jetson as
FAISS IVF-PQ indices on 2026-05-19. ~1.65 GB of indices + ~83 GB of text
sqlite. Pluggable retrieval backends so the Jetson can run pure-FAISS
while the workstation can still query the legacy ChromaDB collections for
ingestion validation.

Cutover history and step-by-step rebuild commands live in
[`docs/rag-migration-runbook.md`](../rag-migration-runbook.md). The
HNSW-on-Jetson failure that motivated the switch is in
[`docs/hnsw-jetson-load-failure.md`](../hnsw-jetson-load-failure.md).

## Scope

- ZIM file parsing (Wikipedia, iFixit, Wikibooks, WikiMed, Gutenberg, CrashCourse)
- Music tag scrape via mutagen (no ZIM; synthesizes a sentence doc from ID3/MP4 tags)
- Text chunking + nomic-v1.5 embedding on the GPU (FP16, max_seq_length=512)
- FAISS IVF-PQ build per collection (PQ-64, METRIC_INNER_PRODUCT, score_scale=20)
- Per-query collection routing + cross-encoder rerank for "deep" mode
- Stack Exchange XML / PDF / arbitrary text ingestion (planned)

## Pipeline

```
ZIM (raw HTML)               → scripts/ingest_zim.py        → chromadb (MiniLM-L6, 384-d)
chromadb chunks (text only)  → scripts/reembed_collection.py → data/embeddings/<name>.{vectors.f32, text.sqlite}
flat .f32 + .sqlite          → scripts/build_faiss_ivfpq.py → data/faiss/<name>.{index, sqlite}
data/faiss/                  → rsync                         → Jetson /opt/radio-oracle/data/faiss/
```

`ingest_zim.py` is now a staging step — its chromadb output is what
`reembed_collection.py` consumes. The chromadb store is read-only at
runtime on the workstation and is **not** shipped to the Jetson.

## File ownership

```
scripts/
  ingest_zim.py            # ZIM → chromadb (chunk text staging)
  reembed_collection.py    # chromadb → flat .f32 + .sqlite (nomic-v1.5, GPU+FP16, resume-safe)
  build_faiss_ivfpq.py     # flat files → FAISS IVF-PQ index + sqlite
  download_knowledge.sh    # Download ZIMs from Kiwix mirrors
  download_knowledge.ps1   # Same, Windows PowerShell
  verify_chroma.py         # Validate ChromaDB staging area after ingest
  ingest_wikipedia.py      # Legacy single-source ingest scripts
  ingest_generic_zim.py    # (kept for reference; ingest_zim.py is preferred)
  ingest_gutenberg.py
oracle/rag/
  chunker.py               # Text splitting (size + overlap)
  embedder.py              # sentence-transformers wrapper (CUDA/FP16, nomic prefixes)
  retriever.py             # Pluggable backend dispatch + tiered mode wiring
  backends/
    chroma.py              # Legacy ChromaDB backend
    faiss_ivfpq.py         # Production FAISS backend (cosine via IP, score_scale calibrated)
  modes.py                 # Snappy / deep retrieval parameter sets
  reranker.py              # Cross-encoder reranker (CPU-only)
  router.py                # Per-query collection routing (intent regex)
tests/
  test_chunker.py
  test_rag_modes.py
```

## Settings

```bash
# Where staged chunk text + legacy embeddings live (workstation only)
ORACLE_CHROMA_PATH=data/chroma

# Where the production FAISS indices live (workstation and Jetson)
ORACLE_FAISS_INDEX_DIR=data/faiss

# Per-collection backend routing — anything not listed falls back to chroma
ORACLE_COLLECTION_BACKENDS='{"wikipedia":"faiss","gutenberg":"faiss","wikimed":"faiss","wikibooks":"faiss","ifixit":"faiss","crashcourse":"faiss","music":"faiss"}'
# NOTE: the JSON value must be single-quoted inside .env / EnvironmentFile
# or pydantic will fail to parse on startup. See migration runbook §5f.

# Re-embed / encode controls (workstation only)
ORACLE_EMBEDDING_DEVICE=auto    # auto | cpu | cuda | cuda:N
ORACLE_EMBEDDING_FP16=true      # only honored on CUDA
ORACLE_EMBEDDING_BATCH_SIZE=256

# Tiered retrieval — snappy is the first-pass that returns to the LLM
ORACLE_RAG_TOP_K=5
ORACLE_TIER1_TOP_K=5
ORACLE_TIER2_TOP_K=20
ORACLE_TIER2_RERANK_POOL=100
ORACLE_TIER2_FINAL_TOP_K=20
ORACLE_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Chunking (ingestion side)
ORACLE_CHUNK_SIZE=512
ORACLE_CHUNK_OVERLAP=64

ORACLE_RAG_COLLECTIONS=         # comma-separated allowlist; empty = all.
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
- `data/faiss/` directory with named collections: `wikipedia`, `gutenberg`,
  `wikibooks`, `wikimed`, `ifixit`, `crashcourse`, `music`. Each is a pair
  of files: `<name>.index` (IVF-PQ) + `<name>.sqlite` (faiss_row → chunk
  text + metadata).

**Consumer**: `oracle/rag/retriever.py::Retriever.query()` returns ranked
chunks via the per-collection backend. `oracle/core.py::_try_rag_query`
calls the retriever lazily — missing collections degrade to "no RAG
context" without crashing.

**Consumes**: nothing at runtime. Ingest + index build is offline.

## Standalone exercise

```bash
# Workstation, end-to-end build for one collection (replace <NAME>):
python scripts/ingest_zim.py path/to/<NAME>.zim --collection <NAME>
python scripts/reembed_collection.py --source <NAME> --target <NAME> \
    --model nomic-ai/nomic-embed-text-v1.5 --dim 768 \
    --db-path data/chroma --out-dir data/embeddings \
    --workers 12 --batch-size 2000 --encode-batch-size 256
python scripts/build_faiss_ivfpq.py --name <NAME> \
    --in-dir data/embeddings --out-dir data/faiss --dim 768

# Query the FAISS retriever directly (workstation or Jetson)
ORACLE_COLLECTION_BACKENDS='{"<NAME>":"faiss"}' \
ORACLE_FAISS_INDEX_DIR=data/faiss \
python -c "
from oracle.rag.retriever import Retriever
from oracle.rag.modes import detect_mode
r = Retriever()
print(r.list_collections())
for hit in r.query('how does a vacuum tube work', mode=detect_mode('how does a vacuum tube work'))[:3]:
    print(hit['source'], '-', hit['text'][:120])
"

pytest tests/test_chunker.py tests/test_rag_modes.py
```

## TODO

- [ ] Stack Exchange XML ingest script
- [ ] Generic PDF ingestion (army field manuals, etc.)
- [ ] Collection stats summary script (counts + disk usage)
- [x] Re-rank with cross-encoder before returning top-k — see `reranker.py` / deep mode
- [ ] Hybrid search (BM25 + dense)
- [ ] Per-collection extra columns in `faiss_idmap` (artist/year/genre/has_drm) to support precise filters on the music collection
