# RAG migration runbook

Two-tier retrieval cutover for wikipedia + gutenberg. Sequence + commands.

All paths assume the workstation layout: corpus + venv under
`C:\Users\erik\Desktop\Huge Information Stores\`, code at
`C:\Users\erik\Desktop\radio-oracle\`.

## 0. Sanity check

Existing chromadb collections + counts:
```powershell
& "C:\Users\erik\Desktop\Huge Information Stores\.venv\Scripts\python.exe" -c @'
import chromadb
c = chromadb.PersistentClient(path=r"C:\Users\erik\Desktop\Huge Information Stores\data\chroma")
for col in c.list_collections(): print(f"{col.name:20s} {col.count():>12,}")
'@
```

Clean up smoke-test leftovers if you don't want them around:
```powershell
& "C:\Users\erik\Desktop\Huge Information Stores\.venv\Scripts\python.exe" -c @'
import chromadb
c = chromadb.PersistentClient(path=r"C:\Users\erik\Desktop\Huge Information Stores\data\chroma")
try: c.delete_collection("crashcourse_v2")
except: pass
try: c.delete_collection("wikipedia_v2")
except: pass
'@
```

## 1. Re-embed wikipedia (~31 h on RTX 4070 SUPER)

```powershell
cd C:\Users\erik\Desktop\radio-oracle
$env:PYTHONPATH = "."
& "C:\Users\erik\Desktop\Huge Information Stores\.venv\Scripts\python.exe" `
  scripts/reembed_collection.py `
  --source wikipedia --target wikipedia_v2 `
  --model nomic-ai/nomic-embed-text-v1.5 `
  --db-path "C:\Users\erik\Desktop\Huge Information Stores\data\chroma" `
  --workers 12 --batch-size 2000 --encode-batch-size 256
```

Ctrl-C is safe — restart with the same command and it'll skip
already-embedded IDs (resume reads target's existing IDs from
`chroma.sqlite3`).

Monitor in a second terminal:
```powershell
nvidia-smi -l 2
# in another terminal, watch chunk count grow:
while ($true) {
  & "C:\Users\erik\Desktop\Huge Information Stores\.venv\Scripts\python.exe" -c @'
import chromadb
c = chromadb.PersistentClient(path=r"C:\Users\erik\Desktop\Huge Information Stores\data\chroma")
print(c.get_or_create_collection("wikipedia_v2").count())
'@
  Start-Sleep 30
}
```

## 2. Build the FAISS IVF-PQ index (~15-20 min)

After re-embed finishes:

```powershell
cd C:\Users\erik\Desktop\radio-oracle
$env:PYTHONPATH = "."
& "C:\Users\erik\Desktop\Huge Information Stores\.venv\Scripts\python.exe" `
  scripts/build_faiss_ivfpq.py `
  --source wikipedia_v2 `
  --db-path "C:\Users\erik\Desktop\Huge Information Stores\data\chroma" `
  --out-dir "C:\Users\erik\Desktop\Huge Information Stores\data\faiss" `
  --name wikipedia `
  --nlist 4096 --pq-m 64
```

Outputs:
- `data/faiss/wikipedia.index` (~730 MB)
- `data/faiss/wikipedia.sqlite` (chunk text + metadata keyed by FAISS row id)

## 3. Cut over the retriever to FAISS for wikipedia

Either edit `config/settings.py`:
```python
collection_backends: dict[str, str] = {"wikipedia": "faiss"}
```
or set an env var per-session:
```powershell
$env:ORACLE_COLLECTION_BACKENDS = '{"wikipedia":"faiss"}'
```

Validate quality:
```powershell
& "C:\Users\erik\Desktop\Huge Information Stores\.venv\Scripts\python.exe" -c @'
import sys; sys.path.insert(0, r"C:\Users\erik\Desktop\radio-oracle")
import os
os.environ["ORACLE_COLLECTION_BACKENDS"] = '{"wikipedia":"faiss"}'
os.environ["ORACLE_FAISS_INDEX_DIR"] = r"C:\Users\erik\Desktop\Huge Information Stores\data\faiss"
os.environ["ORACLE_CHROMA_PATH"] = r"C:\Users\erik\Desktop\Huge Information Stores\data\chroma"
from oracle.rag.retriever import Retriever
r = Retriever()
for q in ["Who was Augustus Caesar?", "explain Newton's laws of motion", "history of penicillin"]:
    out = r.query(q, collection_names=["wikipedia"], mode="snappy")
    print(f"\n{q!r}")
    for h in out[:3]: print(f"  d={h['distance']:.3f} {h['text'][:100]}")
'@
```

## 4. Repeat 1-3 for gutenberg

```powershell
# Re-embed (~28 h)
& "C:\Users\erik\Desktop\Huge Information Stores\.venv\Scripts\python.exe" `
  scripts/reembed_collection.py `
  --source gutenberg --target gutenberg_v2 `
  --model nomic-ai/nomic-embed-text-v1.5 `
  --db-path "C:\Users\erik\Desktop\Huge Information Stores\data\chroma" `
  --workers 12 --batch-size 2000 --encode-batch-size 256

# Build FAISS
& "C:\Users\erik\Desktop\Huge Information Stores\.venv\Scripts\python.exe" `
  scripts/build_faiss_ivfpq.py `
  --source gutenberg_v2 `
  --db-path "C:\Users\erik\Desktop\Huge Information Stores\data\chroma" `
  --out-dir "C:\Users\erik\Desktop\Huge Information Stores\data\faiss" `
  --name gutenberg

# Cutover
$env:ORACLE_COLLECTION_BACKENDS = '{"wikipedia":"faiss","gutenberg":"faiss"}'
```

## 5. Ship to Jetson

Copy `data/faiss/*.index` and `data/faiss/*.sqlite` to the Jetson at
`/opt/radio-oracle/data/faiss/`. The Jetson does NOT need the
chromadb `wikipedia` / `gutenberg` collections anymore — the FAISS
backend has everything it needs.

After validation on Jetson, you can reclaim disk on the workstation by
dropping the v2 chromadb collections (their data is fully baked into
the FAISS artifacts):
```powershell
& "C:\Users\erik\Desktop\Huge Information Stores\.venv\Scripts\python.exe" -c @'
import chromadb
c = chromadb.PersistentClient(path=r"C:\Users\erik\Desktop\Huge Information Stores\data\chroma")
c.delete_collection("wikipedia_v2")
c.delete_collection("gutenberg_v2")
'@
```

## Throughput notes

`reembed_collection.py` on the RTX 4070 SUPER with nomic-v1.5
(`encode_batch_size=256`, 12 workers) clocked ~80 chunks/sec on the
crashcourse smoke run. Wikipedia/gutenberg should see similar rates
once warm.

If throughput is lower than expected, check:
- `nvidia-smi` GPU util — should be 80-100%. If <50%, producers can't
  keep up; bump `--workers` (up to `cpu_count`).
- Upsert queue depth in the log — if it climbs while GPU is idle, the
  chromadb write is the bottleneck; lower `--batch-size`.
- If you see "out of memory" on GPU, drop `--encode-batch-size` to 128.
