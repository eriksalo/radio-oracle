# RAG v2 — re-embedding the archives with EmbeddingGemma-300M

Runs on the **GPU workstation** (needs the ChromaDB staging store and an
NVIDIA GPU; the Jetson only receives the final `data/faiss/` artifacts).
Expect a day-plus of GPU time for the full 22.5M-chunk corpus.

## Why

- `nomic-embed-text-v1.5` (current, ~62.4 MTEB English v2) →
  **google/embeddinggemma-300m** (69.67): the single biggest retrieval
  quality lever available without new hardware. Same 768d, MRL-capable
  (truncatable to 256/128d later), <200MB quantized on the Jetson query
  side, 2K-token context (fine — chunks are ≤512 tokens).
- The old chunker cut mid-sentence across paragraph boundaries; it was
  fixed 2026-07 (`oracle/rag/chunker.py`) but every existing chunk was
  produced by the old one. Re-chunking requires re-ingesting from ZIM.

Two paths — pick per collection:

| Path | What | Cost | Gain |
|---|---|---|---|
| **A: re-embed only** | existing chroma chunks → new vectors | GPU embed time only | embedder upgrade |
| **B: re-chunk + re-embed** | ZIM → ingest (new chunker) → embed | ingest CPU-days + embed | embedder + clean chunk boundaries |

Recommendation: Path A for everything first (one decision, one eval);
Path B later for `wikipedia` only if the eval shows boundary artifacts.

## 0. Prereqs (workstation)

```bash
cd ~/projects/radio-oracle
pip install -e ".[rag,ingest]"
# EmbeddingGemma needs a recent sentence-transformers and a HF login
# (gated model): huggingface-cli login
```

## 1. Golden-set baseline (before touching anything)

```bash
cp data/eval/golden.example.json data/eval/golden.json  # then extend to ~50 Qs
ORACLE_FAISS_INDEX_DIR=data/faiss \
python scripts/eval_rag.py --golden data/eval/golden.json --k 5 | tee eval-baseline.txt
```

## 2. Re-embed each collection (Path A)

EmbeddingGemma prompt format (must match at query time — see step 5):
- documents: `title: none | text: {chunk}`
- queries:   `task: search result | query: {question}`

```bash
for coll in wikipedia gutenberg wikimed wikibooks ifixit crashcourse music; do
  python scripts/reembed_collection.py \
      --source "$coll" --target "${coll}_eg" \
      --model google/embeddinggemma-300m \
      --prefix "title: none | text: " \
      --dim 768 --max-seq-length 512 \
      --db-path data/chroma --out-dir data/embeddings_v2
done
```

Notes:
- Resume-safe: rerun after interruption; embedded ids are skipped.
- If VRAM allows, raise `--encode-batch-size`.
- MRL option: keep 768d now; truncation to 256d is a later, separate
  decision (rebuild indices from the same .f32 files, sliced).

## 3. Build FAISS indices

```bash
mkdir -p data/faiss_v2
for coll in wikipedia gutenberg wikimed wikibooks ifixit crashcourse music; do
  python scripts/build_faiss_ivfpq.py \
      --name "${coll}_eg" --in-dir data/embeddings_v2 --out-dir data/faiss_v2 \
      --dim 768 --pq-m 64
done
# rename <coll>_eg.{index,sqlite} -> <coll>.{index,sqlite} inside data/faiss_v2
```

`--nlist` defaults to clamp(sqrt(n), 64, 4096) as before. Runtime nprobe
comes from settings `ef_search` (128 since 2026-07).

## 4. Eval old vs new (ship gate)

```bash
# Point the retriever at the new indices; query model comes from settings,
# so override it for the v2 run (step 5 makes this permanent):
ORACLE_FAISS_INDEX_DIR=data/faiss_v2 \
python scripts/eval_rag.py --golden data/eval/golden.json --k 5 \
    --min-recall 0.80 | tee eval-v2.txt
```

Wait — the per-collection query model/prefix live in
`settings.faiss_collection_config`, so for the v2 eval either edit that
dict (step 5) first, or export a temporary settings override. Do step 5
on a branch, run the eval, and only merge when v2 recall@5 beats the
baseline.

## 5. Flip the query side (code change, one commit)

In `config/settings.py` `faiss_collection_config`, change every
collection to:

```python
{"model": "google/embeddinggemma-300m",
 "query_prefix": "task: search result | query: ",
 "ef_search": 128, "score_scale": 20.0}
```

`score_scale` calibrates the FAISS-IP → pseudo-distance mapping
(`1 - score/scale`) that the relevance gate (`rag_max_distance=0.65`)
and cross-collection merge use. EmbeddingGemma's normalized-IP score
distribution differs from nomic's: after re-embedding, run a handful of
queries, log raw scores (`faiss_ivfpq.py` debug), and pick a scale that
puts good hits at distance ≲0.35 — then re-check the gate with absurd
queries ("flurbon coefficient" → nothing injected).

## 6. Ship to the Jetson

```bash
rsync -av data/faiss_v2/ erik@radio-oracle.local:/opt/radio-oracle/data/faiss_v2/
# On the Jetson: point ORACLE_FAISS_INDEX_DIR=data/faiss_v2 in .env,
# restart, spot-check + timing (first query downloads the embedder to the
# HF cache — do it once while attended). Keep data/faiss/ until burned in.
```

Jetson note: query embedding stays on CPU (no CUDA torch wheel for
JP6.2/cp311 — see rag-migration-runbook.md "Known follow-up").
EmbeddingGemma-300m is similar compute to nomic-v1.5 — expect ~1s/query;
if it's slower, the ONNX int8 export of EmbeddingGemma is the fix.

## Path B (re-chunk wikipedia) — only if eval shows boundary artifacts

```bash
python scripts/ingest_zim.py data/knowledge/wikipedia_en_all_nopic_latest.zim \
    --collection wikipedia_v2   # new chunker is picked up automatically
# then steps 2-4 with --source wikipedia_v2
```

## Alternatives considered (documented for the future, not this run)

- **int8/binary quantized vectors + float rescoring**: 97-100% / ~96%
  quality retention, 4x/32x memory. Worth it if collections grow past
  RAM; FAISS IVF-PQ64 already gets us ~48x vs fp32.
- **LanceDB**: mmap on-disk IVF-PQ, ships aarch64 wheels, easiest exit
  from FAISS if index RAM becomes the constraint.
- **DiskANN**: best recall/GB (graph on NVMe), aarch64 support landed
  Feb 2026 but source-build only — revisit when there's a prebuilt wheel.
