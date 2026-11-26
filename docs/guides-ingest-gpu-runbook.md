# Practical guides: build the indices on the GPU PC

Hand-over from the dev box (2026-11-26). The dev box has no GPU and embeds
about 1 chunk a second; the first batch is roughly 250,000 chunks. This
runbook builds everything on the Windows PC with the NVIDIA card and puts
the result on the radio.

Commands are PowerShell unless a block says **Jetson**. Run them from the
repository folder. If you resume with Claude Code on the PC, point it at
this file — the "State" and "Open items" sections are the context it needs.

## Done on 2026-11-25 (evening, US Mountain)

The whole runbook was run on the GPU PC (RTX 4070 SUPER, Ryzen 9 9950X3D)
and the result is live on the radio.

- 13 sources, 225,622 chunks, embedded in 18 minutes (140–210 chunks/s) with
  `--batch 16`; four indices built and installed on the Jetson.
- All four `guides_…` collections are in the Jetson's
  `ORACLE_COLLECTION_BACKENDS`. Previous `.env`: `.env.bak-guides`. The
  NHS-only `guides_health` files it replaced are in
  `/opt/radio-oracle/data/faiss-bak-guides/`.
- `probe_guides.py` on the radio: 13/13 as expected (details in step 8).
- The PC keeps `data\knowledge\` (the ZIMs), `data\embeddings\` (text and
  vectors) and `.venv-ingest`, so adding a source is download → extract →
  embed → build for that source only.

The sections below are the original hand-over, corrected where the run
showed something different.

## More sources, 2026-11-26

Seventeen sources were added to `config/knowledge_sources.json` after a pass
through the Kiwix catalogue (`zimit/`, `other/`, `stack_exchange/`) for
anything that fits the four guide shelves. Three new page cleaners in
`scripts/ingest_sources.py` handle them: `medlineplus`, `webpage` (any
crawled site: main content, English only, cut at headings) and `mediawiki`
(wiki articles packed by mwoffliner). Sections shorter than 90 words are
joined to their neighbours, and a page that appears under several URLs is
taken once. A source can list `skip` path fragments in the manifest.

| Collection | New source | Chunks |
|---|---|---|
| guides_health | MedlinePlus (encyclopedia, drugs, health topics, genetic conditions) | 38,224 |
| | CDC travel health (Yellow Book, disease pages) | 2,487 |
| | fitness Q&A | 15,151 |
| | pet care Q&A | 13,754 |
| | parenting Q&A | 21,412 |
| guides_outdoors | Ready.gov | 361 |
| | UK "Prepare" campaign | 146 |
| guides_food | Grimgrains | 484 |
| | Gardenology plant encyclopedia | 7,455 |
| | coffee Q&A | 2,259 |
| | beer, wine and spirits Q&A | 2,158 |
| | vegetarian Q&A | 1,489 |
| guides_workshop | Appropedia | 60,815 |
| | Restarters repair wiki | 1,027 |
| | bicycles Q&A | 40,020 |
| | arts and crafts Q&A | 4,735 |
| | household tips Q&A | 5,102 |

That is 217,079 new chunks, embedded in 18 minutes; with the first batch,
442,701 in 30 sources. Collection
sizes: `guides_health` 101,929, `guides_outdoors` 30,127, `guides_food`
90,749, `guides_workshop` 219,896.

Looked at and left out, with the reason:

- `publicdomainrecipes.com`: a fork of based.cooking — 471 of its 473
  chunks have a title based.cooking already has. `foss.cooking`, which was
  in the first batch and is still in, is the same: 486 of 489. That is why
  the radio names "the recipe collection, Sourdough starter" twice. One of
  the two should go.
- `survivors_en_all_maxi`: a video-game wiki (Vampire Survivors).
- `scoutwiki.org`: scouting organisations and jamborees, not outdoor skills.
- `trueprepper.com`: mostly gear reviews with affiliate copy.
- `energypedia`: energy policy, projects and NGO pages rather than how-to.
- `cd3wdproject.org` (546 MB): a large practical library (farming, food
  processing, metalwork, health), but its pages carry no usable titles,
  several languages are mixed without `lang` attributes, and books are
  split into frames. Worth a dedicated cleaner that takes each book's
  title from its index page — an estimated 80,000 chunks.
- Stack Exchange `electronics` (4.0 GB): circuit design at a depth that
  would outnumber everything else on the workshop shelf. `3dprinting`,
  `engineering`, `arduino`, `raspberrypi` likewise left for a decision.
- PDF collections (`zimgit-water`, `-medicine`, `-food-preparation`,
  `-knots`, `-post-disaster`, `armypubs`, `irp.fas.org` military medicine):
  still need the PDF text step.
- `theworldfactbook` and the science Stack Exchange sites do not belong on
  a guide shelf; they would need their own collection.

Ready.gov is 2 GB for 97 English pages: the crawl holds every page in a
dozen languages, and the business, CERT and social-media-toolkit pages are
skipped.

The radio names a source aloud from this manifest, so
`config/knowledge_sources.json` has to be copied to the Jetson together
with the indices.

Checked on the PC only (ranking, sixteen questions straight against the
indices): the NHS page is still first for amoxicillin; the second-degree
burn question now gets MedlinePlus "Minor burns - aftercare" first; broken
bone → MedlinePlus; chocolate and dogs → pet care Q&A; malaria → CDC;
emergency kit and tornado → Ready.gov; tomatoes → Gardenology; flat tyre →
bicycles Q&A; composting toilet → Appropedia; kettle → Restarters.
`scripts/probe_guides.py` has seven more questions for these.

**Not on the radio yet** as of this writing: the four rebuilt indices are
in `data/faiss/` on the PC (about 730 MB with their text). Steps 7 and 8
apply unchanged, plus the manifest. The index files the radio loads grow
from 21 MB to 39 MB; check the last `Mem:` line of the probe.

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

Chunk counts after the quality filter (dev box; the PC's extraction matched
every one, and supplied the two that were not downloaded there):

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
| | gardening Q&A | 23,051 |
| guides_workshop | woodworking Q&A | 7,654 |
| | vehicle repair Q&A | 24,731 |
| | home improvement Q&A | 75,812 |

Per collection: `guides_health` 10,901, `guides_outdoors` 29,620,
`guides_food` 76,904, `guides_workshop` 108,197.

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

The `[cuda,cudnn]` extra pulls NVIDIA's CUDA and cuDNN libraries as pip
packages, so no separate CUDA install is needed; only a current NVIDIA
driver. Worked first time on the PC (2026-11-25): Python 3.13.6 (the only
Python installed there — outside the 3.10–3.12 range above, but every wheel
installed), onnxruntime-gpu 1.32.0, which pulled CUDA 13 and cuDNN 9, driver
610.88. If step 5 ever stops with "CUDA was asked for but onnxruntime fell
back to CPUExecutionProvider", the driver is the first thing to check.

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
.\.venv-ingest\Scripts\python.exe scripts\ingest_sources.py embed --provider cuda --batch 16
```

It logs the rate and time left per source. It can be stopped and started
again; it carries on where it was. Stage names go before the options:
`embed status --provider cuda` works, `embed --provider cuda status` does
not.

**Use `--batch 16` on the 12 GB card, not 64.** Measured 2026-11-25 on
worst-case 512-token chunks:

| Batch | Chunks/s | GPU memory used |
|---|---|---|
| 8 | 120 | 2.9 GB |
| 16 | 111 | 5.0 GB |
| 32 | 104 | 9.1 GB |
| 64 | 21 | 11.8 GB (card full, spills to system memory) |

A bigger batch buys nothing and 64 is five times slower. Other programs on
the PC (Fusion 360, DJI Studio) were holding 7 GB of the card at one point;
with that much taken even batch 32 does not fit. The real run averaged
140–210 chunks/s because most chunks are shorter than 512 tokens.

GPU and CPU vectors for the same text agree to cosine 0.9999998.

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

### Result, 2026-11-25

- All four collections listed, with the sizes above. `# as expected: 13/13`.
- Amoxicillin: NHS page first at 0.10, unchanged from the CPU-built index.
- Hiking, bear bag and ketchup now get a guide passage first (0.15, 0.20,
  0.18). Napoleon, the sky and Moby Dick get none.
- `lookup` 260–383 ms per question with all four loaded.
- "available" 1.9 GB after the restart.
- Flagged by the probe for a look, not failed: "How do I treat a
  second-degree burn?" has the medical Q&A "Third-degree burn - Home
  treatment" first at 0.22, ahead of WikiMed's "Burn" at 0.23.

A PC that has never connected to the radio needs its host key accepted
first. The dev box already trusts it: compare `ssh-keyscan -t ed25519
radio-oracle.local` with `ssh-keygen -F radio-oracle.local` there.

## Open items

- The shelf weights for the new collections (`oracle/rag/shelves.py`,
  `oracle/rag/router.py`) are first guesses. Each has now been seen working
  on the probe's questions (2026-11-25), which is one to five questions per
  shelf.
- The second-degree burn question (step 8 result): a Q&A page about a
  different degree of burn narrowly outranks the WikiMed article. Wrong
  first-aid advice is the costly kind of near-miss.
- Nothing has been tried with a real voice.
- A Q&A passage starts with the question as its title, so it can match a
  spoken question very closely. Watch for guides crowding out Wikipedia on
  plain factual questions; raise their bias on the science and history
  shelves if so.
- "Where did that come from" repeats the label when two passages come from
  one source ("the NHS medicines guide, X, the NHS medicines guide, Y").
- MedlinePlus, CDC and ready.gov are in (see "More sources, 2026-11-26").
  Still not in the source list: army field manuals (`zimit/armypubs_en_all`,
  7.7 GB) and the other PDF collections, which need a PDF text step, and
  CD3WD, which needs its own cleaner.
- Parenting Q&A sits in `guides_health` for want of a better shelf; much of
  it is about behaviour rather than health. Watch whether it turns up on
  medical questions.
