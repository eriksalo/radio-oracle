# RAG migration runbook

Two-tier retrieval cutover for the six knowledge collections. Sequence + commands as actually run on 2026-05-17/18 — see `hnsw-jetson-load-failure.md` for the diagnosis that motivated the switch.

All paths assume the workstation layout: corpus + venv under
`C:\Users\erik\Desktop\Huge Information Stores\`, code at
`C:\Users\erik\Desktop\radio-oracle\`.

## 0. Sanity check

Existing chromadb collections + counts (source of truth for chunk text):

```powershell
& "C:\Users\erik\Desktop\Huge Information Stores\.venv\Scripts\python.exe" -c @'
import chromadb
c = chromadb.PersistentClient(path=r"C:\Users\erik\Desktop\Huge Information Stores\data\chroma")
for col in c.list_collections(): print(f"{col.name:20s} {col.count():>12,}")
'@
```

FAISS artifacts:

```powershell
Get-ChildItem "C:\Users\erik\Desktop\Huge Information Stores\data\faiss" | Select Name, @{N='MB';E={[math]::Round($_.Length/1MB,1)}}
```

## 1. Re-embed a collection (chromadb chunks → flat float32 + sqlite)

For each collection (replace `<NAME>`; ~250 chunks/sec on RTX 4070 SUPER means crashcourse <10s, wikimed/wikibooks ~15–25 min, wikipedia/gutenberg ~11 h):

```powershell
cd C:\Users\erik\Desktop\radio-oracle
$env:PYTHONPATH = "."
& "C:\Users\erik\Desktop\Huge Information Stores\.venv\Scripts\python.exe" `
  scripts/reembed_collection.py `
  --source <NAME> --target <NAME> `
  --model nomic-ai/nomic-embed-text-v1.5 `
  --db-path "C:\Users\erik\Desktop\Huge Information Stores\data\chroma" `
  --out-dir "C:\Users\erik\Desktop\Huge Information Stores\data\embeddings" `
  --dim 768 `
  --workers 12 --batch-size 2000 --encode-batch-size 256
```

Defaults that matter: `--max-seq-length 512` (5× throughput vs nomic's 8192 default; only ~1% of chunks affected). Ctrl-C is safe — resume reads existing `chunk_id` set from the target text.sqlite and skips them.

Monitor (separate terminal):

```powershell
nvidia-smi -l 2
# row count grows in the .vectors.f32 file:
while ($true) {
  $f = "C:\Users\erik\Desktop\Huge Information Stores\data\embeddings\<NAME>.vectors.f32"
  if (Test-Path $f) { "{0:N0}" -f ((Get-Item $f).Length / (768*4)) }
  Start-Sleep 30
}
```

## 2. Build FAISS IVF-PQ from the flat files (~1–20 min)

```powershell
cd C:\Users\erik\Desktop\radio-oracle
$env:PYTHONPATH = "."
& "C:\Users\erik\Desktop\Huge Information Stores\.venv\Scripts\python.exe" `
  scripts/build_faiss_ivfpq.py `
  --name <NAME> `
  --in-dir "C:\Users\erik\Desktop\Huge Information Stores\data\embeddings" `
  --out-dir "C:\Users\erik\Desktop\Huge Information Stores\data\faiss" `
  --dim 768
```

`--nlist` defaults to auto (`clamp(sqrt(n), 64, 4096)`). Outputs `<name>.index` + `<name>.sqlite` (chunk text/metadata keyed by faiss_row).

## 3. (Optional) Reclaim intermediate disk

Once the FAISS index is verified, the flat-file workspace is redundant — the FAISS sqlite has a copy of the text:

```powershell
Remove-Item "C:\Users\erik\Desktop\Huge Information Stores\data\embeddings\<NAME>.vectors.f32"
Remove-Item "C:\Users\erik\Desktop\Huge Information Stores\data\embeddings\<NAME>.text.sqlite"
```

Verify counts match first if paranoid: `SELECT COUNT(*) FROM chunks` (text.sqlite) should equal `SELECT COUNT(*) FROM faiss_idmap` (FAISS sqlite).

## 4. Flip the retriever over

Either edit `config/settings.py` (`collection_backends: {"wikipedia": "faiss", ...}`) or set an env var for the session:

```powershell
$env:ORACLE_COLLECTION_BACKENDS = '{"wikipedia":"faiss","gutenberg":"faiss","wikimed":"faiss","wikibooks":"faiss","ifixit":"faiss","crashcourse":"faiss"}'
$env:ORACLE_FAISS_INDEX_DIR = "C:\Users\erik\Desktop\Huge Information Stores\data\faiss"
$env:ORACLE_CHROMA_PATH = "C:\Users\erik\Desktop\Huge Information Stores\data\chroma"
```

Validate end-to-end:

```powershell
& "C:\Users\erik\Desktop\Huge Information Stores\.venv\Scripts\python.exe" -c @'
import os, sys, time
sys.path.insert(0, r"C:\Users\erik\Desktop\radio-oracle")
from oracle.rag.retriever import Retriever
from oracle.rag.modes import detect_mode
r = Retriever()
for q in ["Who was Augustus Caesar?", "How do I fix a leaky faucet?", "What are stroke symptoms?"]:
    t0 = time.time()
    out = r.query(q, mode=detect_mode(q))
    print(f"\n{q!r}  ({(time.time()-t0)*1000:.0f} ms)")
    for h in out[:3]: print(f"  [{h['source']}] d={h['distance']:.3f} {h['text'][:90]}")
'@
```

After first-query warmup (loads nomic + FAISS into RAM, ~60–90 s cold), subsequent queries are 80–300 ms.

## 5. Ship to Jetson

Done 2026-05-19 to `radio-oracle` at `10.0.0.190` (direct ethernet — the wifi `.186` would also work but caps at lower throughput). The steps that actually worked:

### 5a. Create target dir on Jetson

```bash
ssh erik@10.0.0.190 "sudo mkdir -p /opt/radio-oracle/data/faiss && sudo chown -R erik:erik /opt/radio-oracle/data/faiss"
```

### 5b. Transfer ~80 GB of FAISS artifacts

Git Bash on Windows doesn't ship rsync; `scp -r` works fine and saturated gigabit at ~115 MB/s (~12 min total):

```bash
cd "C:/Users/erik/Desktop/Huge Information Stores/data/faiss"
scp -r ./*.index ./*.sqlite erik@10.0.0.190:/opt/radio-oracle/data/faiss/
ssh erik@10.0.0.190 "sudo chown -R oracle:oracle /opt/radio-oracle/data/faiss"
```

### 5c. Sync the new RAG code

`/opt/radio-oracle` is **not a git checkout** — code updates are manual. The Jetson has hardware-specific extensions (mode `Literal["text","voice","hardware"]`, GPIO config, music player). **DO NOT** blanket-copy `oracle/core.py` or `config/settings.py` from the workstation — patch them instead. `oracle/rag/` can be replaced wholesale.

```bash
# Stage on Jetson:
ssh erik@10.0.0.190 "mkdir -p /tmp/radio-oracle-update/oracle/rag/backends"
cd C:/Users/erik/Desktop/radio-oracle
scp oracle/rag/backends/*.py        erik@10.0.0.190:/tmp/radio-oracle-update/oracle/rag/backends/
scp oracle/rag/{modes,reranker,router,retriever}.py erik@10.0.0.190:/tmp/radio-oracle-update/oracle/rag/

# Backup, then install:
ssh erik@10.0.0.190 "
  TS=\$(date +%Y%m%d-%H%M%S)
  sudo cp -r /opt/radio-oracle/{oracle/rag,config/settings.py,.env} /opt/radio-oracle.bak-\$TS/ 2>/dev/null || true
  sudo mkdir -p /opt/radio-oracle/oracle/rag/backends
  sudo cp /tmp/radio-oracle-update/oracle/rag/backends/*.py /opt/radio-oracle/oracle/rag/backends/
  sudo cp /tmp/radio-oracle-update/oracle/rag/*.py /opt/radio-oracle/oracle/rag/
  sudo chown -R oracle:oracle /opt/radio-oracle/oracle/rag
"
```

### 5d. Patch settings.py (don't replace it)

The Jetson's settings.py has GPIO/I²C/music fields the workstation's doesn't. Inject only the three FAISS-related fields right before `settings = OracleSettings()`:

```python
# Add to /opt/radio-oracle/config/settings.py inside class OracleSettings:
collection_backends: dict[str, str] = {}
faiss_index_dir: Path = Path("data/faiss")
faiss_collection_config: dict[str, dict] = {
    "wikipedia":   {"model": "nomic-ai/nomic-embed-text-v1.5", "query_prefix": "search_query: ", "ef_search": 64, "score_scale": 20.0},
    "gutenberg":   {"model": "nomic-ai/nomic-embed-text-v1.5", "query_prefix": "search_query: ", "ef_search": 64, "score_scale": 20.0},
    "wikimed":     {"model": "nomic-ai/nomic-embed-text-v1.5", "query_prefix": "search_query: ", "ef_search": 64, "score_scale": 20.0},
    "wikibooks":   {"model": "nomic-ai/nomic-embed-text-v1.5", "query_prefix": "search_query: ", "ef_search": 64, "score_scale": 20.0},
    "ifixit":      {"model": "nomic-ai/nomic-embed-text-v1.5", "query_prefix": "search_query: ", "ef_search": 64, "score_scale": 20.0},
    "crashcourse": {"model": "nomic-ai/nomic-embed-text-v1.5", "query_prefix": "search_query: ", "ef_search": 64, "score_scale": 20.0},
    "music":       {"model": "nomic-ai/nomic-embed-text-v1.5", "query_prefix": "search_query: ", "ef_search": 64, "score_scale": 20.0},
}
```

### 5e. Install faiss-cpu in the Jetson venv

```bash
ssh erik@10.0.0.190 "sudo /opt/radio-oracle/.venv/bin/pip install faiss-cpu"
```

`faiss-cpu` 1.13.2 has aarch64 wheels on PyPI; no build needed.

### 5f. Update `.env`

Append to `/opt/radio-oracle/.env`:

```ini
ORACLE_FAISS_INDEX_DIR=/opt/radio-oracle/data/faiss
ORACLE_COLLECTION_BACKENDS='{"wikipedia":"faiss","gutenberg":"faiss","wikimed":"faiss","wikibooks":"faiss","ifixit":"faiss","crashcourse":"faiss","music":"faiss"}'
```

**Critical: the JSON value MUST be single-quoted.** systemd's `EnvironmentFile=` parser (and bash `source`) will strip unquoted inner double-quotes, breaking pydantic JSON parsing on startup.

### 5g. Restart services

```bash
ssh erik@10.0.0.190 "sudo systemctl restart radio-oracle.service oracle-web.service && sleep 8 && sudo systemctl is-active radio-oracle.service oracle-web.service"
```

Both should report `active`. Tail `journalctl -u radio-oracle -n 50` to confirm hardware init (RGB LED, ADS1115, button, power switch) and Ollama-ready come up clean.

### 5h. Smoke-test retrieval

```bash
ssh erik@10.0.0.190 "
  sudo -u oracle bash -c '
    set -a; source /opt/radio-oracle/.env; set +a
    cd /opt/radio-oracle
    /opt/radio-oracle/.venv/bin/python -c \"
from oracle.rag.retriever import Retriever
from oracle.rag.modes import detect_mode
r = Retriever()
for q in [\\\"Who was Augustus Caesar?\\\", \\\"How do I fix a leaky faucet?\\\"]:
    out = r.query(q, mode=detect_mode(q))
    print(q, \\\"->\\\", out[0][\\\"source\\\"], out[0][\\\"distance\\\"])
\"
  '
"
```

Each query should land the right collection (wikipedia, ifixit respectively). Cold first-query takes ~14 s (model + FAISS load); subsequent queries ~1.2 s warm.

### 5i. Remove legacy ChromaDB data

Done 2026-05-19. All 7 collections now served by FAISS; the 442 GB ChromaDB store is redundant.

```bash
ssh erik@10.0.0.190 "sudo rm -rf /opt/radio-oracle/data/chroma/"
# Remove stale env var
ssh erik@10.0.0.190 "sudo sed -i '/ORACLE_RAG_CHROMA_PATH/d' /opt/radio-oracle/.env"
```

Disk reclaimed: 635 GB → 195 GB used. The `ChromaBackend` code remains as a fallback for any future collection that doesn't warrant a FAISS build.

### Known follow-up: torch CUDA on Jetson (deferred)

The Jetson currently runs nomic-v1.5 embedding on CPU at ~1.1 s/query. GPU embedding would drop that to ~200 ms but is **harder than it looks**:

- The Jetson venv is **Python 3.11** (`/opt/uv-python/cpython-3.11.15-linux-aarch64-gnu/`).
- The torch in it (`2.12.0+cu130`) is the generic PyPI build for CUDA 13; the Jetson driver is 12.6, so `torch.cuda.is_available()` is False.
- **No prebuilt cp311 CUDA-enabled torch wheel exists for JetPack 6.2.** Nvidia's `developer.download.nvidia.com/compute/redist/jp/v62/pytorch/` and the Jetson AI Lab community repo (`pypi.jetson-ai-lab.io/jp6/cu126`) both ship cp310 only.

Three real fix paths (in order of decreasing recommendation):

1. **ONNX Runtime swap.** Add an `onnxruntime-gpu` (cp311 aarch64 + CUDA 12.6) embedder backend that consumes a nomic-v1.5 ONNX export. ~half-day of code in `oracle/rag/embedder.py`. Isolated, reversible.
2. **Build torch from source** for cp311 + CUDA 12.6 on the Jetson. ~4–6 hours of compile, swap tuning required. Authoritative but slow.
3. **Recreate the venv on Python 3.10** and use the official Nvidia cp310 wheels. Forces reinstall of every dep including hardware-sensitive ones (Jetson.GPIO, ADS1115, sounddevice, miniaudio); high ABI-breakage risk on a live production system. Last resort.

End-to-end latency is dominated by Llama 3.2 3B generation, so the 1.1 s embedder cost is a small fraction of the wall-clock response time. Defer until/unless that math changes.

## Throughput notes (observed 2026-05-18)

- Re-embed throughput on the RTX 4070 SUPER, 12 producers, batch 2000, encode_batch 256, `max_seq_length=512`: ~250 chunks/sec end-to-end. GPU ~80–100% saturated, VRAM ~7 GB / 12 GB.
- FAISS build: training (1M sample IVF-PQ on dim=768) ~45 s, add throughput ~125k vectors/sec.
- Wikipedia: 11.4 M chunks re-embedded in ~12 h. Gutenberg should be similar.

If throughput drops materially below 200/sec:
- `nvidia-smi`: GPU below 70% steady-state means producers are starving the encoder. Check sqlite read pattern.
- Upsert queue growing: writer is the bottleneck. The flat-file writer should never bottleneck below ~10K rows/sec; if it does, suspect disk.
- GPU > 95% but rate is low: input sequence length is high. Lower `--max-seq-length`.
