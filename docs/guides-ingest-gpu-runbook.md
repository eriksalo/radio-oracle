# Practical guides: build the indices on the GPU PC

Hand-over from the dev box (2026-11-26). The dev box has no GPU and embeds
about 1 chunk a second; the first batch is roughly 250,000 chunks. This
runbook builds everything on the Windows PC with the NVIDIA card and puts
the result on the radio.

Commands are PowerShell unless a block says **Jetson**. Run them from the
repository folder. If you resume with Claude Code on the PC, point it at
this file — the "State" and "Open items" sections are the context it needs.

## State when this was written

- Code is on `main`: `scripts/ingest_sources.py` (the whole pipeline),
  `config/knowledge_sources.json` (13 sources → 4 collections),
  `scripts/probe_guides.py` (device check). The shelves, router and spoken
  source names already know the four collections.
- **Live on the radio:** `guides_health`, built from the NHS medicines guide
  only (2,873 chunks). The Jetson's `.env` lists it in
  `ORACLE_COLLECTION_BACKENDS`; the previous `.env` is `.env.bak-2026-11-26`.
- **Not on the radio:** `guides_outdoors`, `guides_food`, `guides_workshop`,
  and the medical Q&A part of `guides_health`.
- Nothing built on the dev box needs copying. The PC redoes all of it; with
  a GPU that is the quick part.

Chunk counts measured on the dev box (after the quality filter):

| Collection | Source | Chunks |
|---|---|---|
| guides_health | NHS medicines | 2,873 |
| | medical Q&A | 8,028 |
| guides_outdoors | outdoors Q&A | 16,217 |
| | ham radio Q&A | 8,961 |
| | sustainable living Q&A | 4,442 |
| guides_food | based.cooking recipes | 502 |
| | foss.cooking recipes | 489 |
| | cooking Q&A | 43,957 |
| | home brewing Q&A | 8,905 |
| | gardening Q&A | not downloaded yet (890 MB) |
| guides_workshop | woodworking Q&A | 7,654 |
| | vehicle repair Q&A | 24,731 |
| | home improvement Q&A | not downloaded yet (1.9 GB) |

## 1. Get the code

```powershell
git pull origin main
```

## 2. A Python environment for the ingest

Python 3.10–3.12. This is separate from any existing environment so nothing
else is disturbed.

```powershell
python -m venv .venv-ingest
```

```powershell
.\.venv-ingest\Scripts\python.exe -m pip install --upgrade pip
```

```powershell
.\.venv-ingest\Scripts\python.exe -m pip install "onnxruntime-gpu[cuda,cudnn]" tokenizers numpy loguru libzim selectolax faiss-cpu pydantic pydantic-settings
```

The `[cuda,cudnn]` extra pulls NVIDIA's CUDA 12 and cuDNN 9 libraries as pip
packages, so no separate CUDA install is needed; only a current NVIDIA
driver. Not tested on this PC — if step 5 stops with "CUDA was asked for but
onnxruntime fell back to CPUExecutionProvider", that is the thing to fix
(driver version, or install the CUDA 12 toolkit and cuDNN 9 by hand).

## 3. The embedding model

Downloads the fp32 nomic-embed-text-v1.5 ONNX file (about 520 MB) and its
tokenizer into `models\nomic-embed-text-v1.5-onnx\`:

```powershell
.\.venv-ingest\Scripts\python.exe scripts\ingest_sources.py model
```

## 4. Download and extract every source

About 4 GB of downloads into `data\knowledge\`, then a few minutes of text
extraction into `data\embeddings\`:

```powershell
.\.venv-ingest\Scripts\python.exe scripts\ingest_sources.py download extract status
```

`status` prints one line per source. The chunk counts should match the table
above (the two missing ones will now have numbers). A source with 0 chunks
means its page layout was not recognised.

## 5. Embed on the GPU

```powershell
.\.venv-ingest\Scripts\python.exe scripts\ingest_sources.py embed --provider cuda --batch 64
```

It logs the rate and time left per source. It can be stopped and started
again; it carries on where it was. The speed on this card has not been
measured — if `--batch 64` runs out of GPU memory, use 32.

## 6. Build the four indices

```powershell
.\.venv-ingest\Scripts\python.exe scripts\ingest_sources.py build
```

Writes `data\faiss\guides_health.index` / `.sqlite` and the same for
`guides_outdoors`, `guides_food`, `guides_workshop`. A source that is not
fully embedded is left out with a warning.

## 7. Copy to the radio

The radio's files belong to the `oracle` user, so copy to your home folder
first:

```powershell
scp data\faiss\guides_*.index data\faiss\guides_*.sqlite erik@radio-oracle.local:/home/erik/
```

Then on the **Jetson** (`ssh erik@radio-oracle.local`):

```bash
sudo install -o oracle -g oracle -m 644 /home/erik/guides_*.index /home/erik/guides_*.sqlite /opt/radio-oracle/data/faiss/ && rm /home/erik/guides_*.index /home/erik/guides_*.sqlite
```

Register the three new collections (keeps a backup; `guides_health` is
already there):

```bash
cd /opt/radio-oracle && sudo cp -p .env .env.bak-guides && sudo sed -i 's/"guides_health":"faiss"}/"guides_health":"faiss","guides_outdoors":"faiss","guides_food":"faiss","guides_workshop":"faiss"}/' .env && sudo grep COLLECTION .env
```

The printed line must end with all four `guides_…` names.

## 8. Check on the radio

On the **Jetson**. This stops the radio for a couple of minutes and starts
it again by itself, with the new collections loaded:

```bash
nohup sudo SIM_SCRIPT=scripts/probe_guides.py /opt/radio-oracle/scripts/sim_turn.sh > /dev/null 2>&1 &
```

```bash
tail -f /tmp/sim_turn.log
```

What to look for:

- `# guide collections:` lists all four.
- `# as expected: 13/13`. On 2026-11-26, with only the NHS guide loaded, it
  was 10/13: the hiking, bear-bag and ketchup questions had no guide yet.
- "What are the side effects of amoxicillin?" still has the NHS page first
  at about 0.10. A much worse number would mean the GPU vectors differ from
  the CPU ones.
- Napoleon, the sky and Moby Dick get **no** `guides_…` passage.
- `lookup … ms` per question: it was 250–380 ms with one guide collection.
- The last `Mem:` line: "available" was about 1.8 GB before.

To undo: `sudo cp -p /opt/radio-oracle/.env.bak-guides /opt/radio-oracle/.env`
and `sudo systemctl restart radio-oracle`.

## Open items

- The shelf weights for the new collections (`oracle/rag/shelves.py`,
  `oracle/rag/router.py`) are first guesses. Only the health one has been
  seen working, on four medicine questions.
- Nothing has been tried with a real voice.
- A Q&A passage starts with the question as its title, so it can match a
  spoken question very closely. Watch for guides crowding out Wikipedia on
  plain factual questions; raise their bias on the science and history
  shelves if so.
- "Where did that come from" repeats the label when two passages come from
  one source ("the NHS medicines guide, X, the NHS medicines guide, Y").
- Not in the source list yet: army field manuals (`zimit/armypubs_en_all`,
  7.7 GB, PDF-based — needs a PDF text step), MedlinePlus (1.8 GB), CDC
  (170 MB), ready.gov (1.9 GB). Each needs an extractor in
  `scripts/ingest_sources.py` and an entry in `config/knowledge_sources.json`.
