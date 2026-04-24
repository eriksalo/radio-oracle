# Radio Oracle — Complete Setup Guide

This guide covers everything from unboxing the Jetson to a working Oracle that answers questions from 60GB+ of offline knowledge.

There are three major sections:

1. **Jetson Orin Nano Setup** — OS, hardware, system config
2. **Downloads** — models, knowledge bases, and what each one is
3. **Ingestion & RAG** — turning raw files into searchable vector embeddings

---

## Part 1: Jetson Orin Nano Super Setup

### 1.1 What You Need

| Item | Notes |
|------|-------|
| Jetson Orin Nano Super Developer Kit (8GB) | The "Super" variant has 40 TOPS and faster clocks |
| 1TB M.2 2280 NVMe SSD | Samsung 980 Pro, WD SN770, or similar. The built-in SD card slot is too slow. |
| USB microphone | ReSpeaker USB Mic Array v2.0 recommended (has onboard VAD/AEC). Any USB mic works. |
| USB DAC + speaker | A cheap USB sound card + any speaker. Or a USB-powered speaker with built-in DAC. |
| MicroSD card (32GB+) | Only needed for initial JetPack flash — the NVMe takes over after. |
| USB keyboard + HDMI monitor | For initial setup only. The Oracle runs headless after. |
| Ethernet cable or WiFi adapter | For initial setup and downloading. Not needed after. |
| PTT momentary button | Any normally-open momentary push button. Wired to GPIO. |
| LEDs (3x) + resistors (3x 330ohm) | Status indicators: idle (green), listening (blue), thinking (yellow) |
| Jumper wires | For GPIO connections |
| Power supply | The Jetson kit includes one (19V barrel jack or USB-C depending on revision) |

### 1.2 Flash JetPack OS

The Jetson needs NVIDIA's JetPack OS (Ubuntu 22.04 + CUDA).

**Option A: SD Card Image (easiest)**

1. Download JetPack 6.2+ SD card image from [NVIDIA Developer](https://developer.nvidia.com/embedded/jetpack)
2. Flash to microSD with [balenaEtcher](https://etcher.balena.io/) or `dd`
3. Insert SD card, connect monitor/keyboard, power on
4. Walk through the Ubuntu first-boot wizard (username, password, locale)
5. Once booted, verify CUDA:
   ```bash
   nvcc --version    # should show CUDA 12.x
   ```

**Option B: SDK Manager (from an x86 Ubuntu host)**

1. Install [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager) on an Ubuntu 20.04/22.04 x86 machine
2. Connect Jetson via USB-C in recovery mode (hold recovery button while powering on)
3. SDK Manager flashes JetPack to the NVMe directly — no SD card needed
4. This is cleaner if you want NVMe-only from the start

### 1.3 Move Root to NVMe (if you used SD card)

If you flashed via SD card, the OS is on the SD. Move it to NVMe for speed:

```bash
# Partition and format the NVMe
sudo fdisk /dev/nvme0n1
# Create one partition using the full disk (type: Linux, 83)

sudo mkfs.ext4 /dev/nvme0n1p1

# Copy root filesystem
sudo mkdir /mnt/nvme
sudo mount /dev/nvme0n1p1 /mnt/nvme
sudo rsync -axHAWX --numeric-ids --info=progress2 / /mnt/nvme/

# Update boot config to use NVMe as root
# Edit /boot/extlinux/extlinux.conf — change root= to point at /dev/nvme0n1p1
sudo nano /boot/extlinux/extlinux.conf
# Change: root=/dev/mmcblk0p1  →  root=/dev/nvme0n1p1

sudo reboot
```

After reboot, verify with `df -h /` — it should show `/dev/nvme0n1p1`.

You can remove the SD card once confirmed.

### 1.4 System Configuration

```bash
# Update everything
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y \
    python3-venv python3-dev python3-pip \
    portaudio19-dev libasound2-dev alsa-utils \
    wget curl git build-essential

# Set hostname
sudo hostnamectl set-hostname radio-oracle

# Disable GUI — the Oracle runs headless to save ~1GB RAM
sudo systemctl set-default multi-user.target
# (To re-enable desktop later: sudo systemctl set-default graphical.target)

# Increase swap (helps during heavy operations)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Set power mode to maximum performance
sudo nvpmodel -m 0
sudo jetson_clocks
```

### 1.5 Audio Hardware Setup

Plug in your USB mic and USB DAC, then identify them:

```bash
# List capture devices (microphones)
arecord -l

# List playback devices (speakers)
aplay -l
```

You'll see output like:
```
card 1: Device [USB Audio Device], device 0: USB Audio [USB Audio]
card 2: ArrayUAC10 [ReSpeaker 4 Mic Array (UAC1.0)], device 0: ...
```

Note the card numbers. Create `/etc/asound.conf`:

```bash
sudo nano /etc/asound.conf
```

```
# Adjust card numbers to match your hardware
# Playback = USB DAC, Capture = USB mic
pcm.!default {
    type asym
    playback.pcm "plughw:1,0"   # <-- your DAC card number
    capture.pcm "plughw:2,0"    # <-- your mic card number
}
ctl.!default {
    type hw
    card 1
}
```

Test:
```bash
# Record 5 seconds
arecord -d 5 -f S16_LE -r 16000 /tmp/test.wav

# Play it back
aplay /tmp/test.wav
```

If you hear your recording, audio is good.

### 1.6 GPIO Wiring

The PTT button and status LEDs connect to the Jetson's 40-pin GPIO header.

**Pin assignments (BCM numbering):**

| Function | GPIO Pin | Physical Pin | Notes |
|----------|----------|-------------|-------|
| PTT Button | GPIO 18 | Pin 12 | Connect between pin 12 and GND (pin 6). Uses internal pull-up. |
| LED: Idle | GPIO 23 | Pin 16 | Green LED + 330ohm resistor to GND |
| LED: Listening | GPIO 24 | Pin 18 | Blue LED + 330ohm resistor to GND |
| LED: Thinking | GPIO 25 | Pin 22 | Yellow LED + 330ohm resistor to GND |
| GND | — | Pin 6, 9, 14, 20, 25 | Any ground pin works |

**PTT button wiring:**
```
GPIO 18 (pin 12) ----[button]---- GND (pin 6)
```
The software enables an internal pull-up resistor. When you press the button, it pulls the pin LOW.

**LED wiring (each LED):**
```
GPIO pin ----[330ohm]----[LED+]----[LED-]---- GND
```

Pin numbers can be changed via environment variables:
```bash
export ORACLE_PTT_GPIO_PIN=18
export ORACLE_LED_IDLE_PIN=23
export ORACLE_LED_LISTEN_PIN=24
export ORACLE_LED_THINK_PIN=25
```

### 1.7 Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh

# Verify it's running
ollama --version

# Limit to one model loaded at a time (saves memory)
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf << 'EOF'
[Service]
Environment="OLLAMA_MAX_LOADED_MODELS=1"
EOF
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### 1.8 Install Radio Oracle

```bash
# Clone the repo
sudo mkdir -p /opt/radio-oracle
sudo chown $USER:$USER /opt/radio-oracle
git clone https://github.com/eriksalo/radio-oracle.git /opt/radio-oracle
cd /opt/radio-oracle

# Create venv and install
python3 -m venv .venv
.venv/bin/pip install -e ".[all,dev]"

# Quick sanity check
.venv/bin/python -m oracle --help
```

### 1.9 Install as System Service

```bash
# Create a dedicated user
sudo useradd -r -m -s /bin/bash -G gpio,audio,video oracle
sudo chown -R oracle:oracle /opt/radio-oracle

# Install the service
sudo cp /opt/radio-oracle/systemd/radio-oracle.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable radio-oracle

# Don't start it yet — we need models and knowledge first
```

---

## Part 2: Downloads

Everything below needs to be downloaded once. After that, the Oracle is fully offline.

### 2.1 AI Models

These are the brains of the Oracle. Download them on the Jetson (or on a workstation and rsync over).

#### LLM — Llama 3.2 3B (via Ollama)

```bash
ollama pull llama3.2:3b
```

- **Size:** ~2 GB download, ~3 GB VRAM at runtime
- **What it is:** The main language model. Handles all conversation, reasoning, and answer generation.
- **Why this one:** Best quality-per-GB at the 3B parameter size. Q4_K_M quantization fits comfortably in 8GB unified memory.
- **Alternatives:** If answers aren't good enough, try `qwen2.5:3b` or `phi3.5:3.8b` — same memory footprint.

#### STT — Whisper small.en

```bash
cd /opt/radio-oracle
./scripts/download_models.sh
```

Downloads to `models/whisper-small.en.bin`:
- **Size:** ~460 MB
- **What it is:** OpenAI's Whisper speech recognition model, converted to GGML format for whisper.cpp. English-only variant (more accurate than multilingual for English).
- **Why small.en:** Good accuracy vs. speed tradeoff. The "tiny" model is faster but misses words. The "medium" model is better but uses 1.5GB VRAM.

#### TTS — Piper lessac-medium

Also downloaded by `download_models.sh`, to `models/en_US-lessac-medium.onnx`:
- **Size:** ~75 MB (model) + JSON config
- **What it is:** A neural text-to-speech voice trained on the LJSpeech/Lessac dataset. ONNX format runs on CPU.
- **Why this one:** Natural-sounding American English voice. Runs entirely on CPU so it doesn't compete with the LLM for GPU memory.
- **Voice alternatives:** Browse voices at [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) — download the `.onnx` + `.onnx.json` pair and set `ORACLE_PIPER_MODEL_PATH`.

#### Embedding Model — all-MiniLM-L6-v2

- **Size:** ~80 MB
- **What it is:** A sentence-transformer model that converts text into 384-dimensional vectors for semantic search.
- **Download:** Automatic on first use. The `sentence-transformers` library downloads it from HuggingFace and caches it in `~/.cache/huggingface/`.
- **To pre-download:**
  ```bash
  .venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
  ```

### 2.2 Knowledge Bases

These are the Oracle's "archives" — the offline knowledge it searches to answer questions.

**Important:** Download and ingest these on a workstation with a GPU. Ingestion involves computing millions of embeddings and takes hours to days on CPU but is much faster on a GPU. Then rsync the finished ChromaDB to the Jetson.

#### Automated Downloads (ZIM files via Kiwix)

```bash
# Preview what will be downloaded
./scripts/download_knowledge.sh --dry-run

# Download all (~28 GB total)
./scripts/download_knowledge.sh
```

| Source | File | Size | What It Contains |
|--------|------|------|-----------------|
| **Wikipedia EN** | `wikipedia_en_all_nopic_latest.zim` | 22 GB | All English Wikipedia articles, text only (no images). ~6.7 million articles covering every topic imaginable. This is the Oracle's primary knowledge source. |
| **iFixit** | `ifixit_en_all_latest.zim` | 2.5 GB | Thousands of step-by-step repair guides for electronics, appliances, vehicles, and more. Invaluable for fixing things when you can't Google it. |
| **Wikibooks** | `wikibooks_en_all_latest.zim` | 2 GB | Open textbooks on math, science, programming, cooking, languages, and more. Structured educational content. |
| **WikiMed** | `wikipedia_en_medicine_nopic_latest.zim` | 1 GB | Medical subset of Wikipedia — diseases, treatments, anatomy, pharmacology. Curated by medical professionals for offline medical reference. |

ZIM is an offline archive format used by [Kiwix](https://www.kiwix.org/). Each file is a self-contained compressed archive of an entire website.

#### Manual Downloads

These sources require manual downloading because they don't have simple single-file downloads:

**Project Gutenberg (~20 GB)**
- **What:** 70,000+ free ebooks — literature, philosophy, science, history, reference
- **Why:** Deep knowledge in humanities, historical texts, classic references
- **How to get:**
  ```bash
  mkdir -p data/knowledge/gutenberg

  # Option 1: Use the Gutenberg mirror (recommended)
  # Download the complete collection of UTF-8 text files:
  rsync -av --include='*.txt' --exclude='*' \
      aleph.gutenberg.org::gutenberg data/knowledge/gutenberg/

  # Option 2: Use wget to mirror (slower, gets everything)
  # See: https://www.gutenberg.org/help/mirroring.html

  # Option 3: Download the Gutenberg DVD ISO (~8GB, curated subset)
  # https://www.gutenberg.org/wiki/Gutenberg:The_CD_and_DVD_Project
  ```

**Stack Exchange Data Dump (~10 GB subset)**
- **What:** Q&A from Stack Overflow, SuperUser, ServerFault, and other Stack Exchange sites
- **Why:** Practical how-to knowledge, troubleshooting, programming, system administration
- **How to get:**
  ```bash
  mkdir -p data/knowledge/stackexchange

  # Download from Internet Archive
  # Full dump: https://archive.org/details/stackexchange
  # Recommended subsets (download the 7z files):
  #   - stackoverflow.com-Posts.7z (most useful, but huge ~20GB compressed)
  #   - superuser.com.7z
  #   - serverfault.com.7z
  #   - diy.stackexchange.com.7z (home improvement)
  #   - electronics.stackexchange.com.7z
  #   - mechanics.stackexchange.com.7z

  # Each site has a Posts.xml inside the 7z
  # Extract: 7z x superuser.com.7z -odata/knowledge/stackexchange/
  sudo apt install p7zip-full
  ```

**Army Field Manuals & Survival Guides (~2 GB)**
- **What:** US Army field manuals (FM series), survival guides, first aid, engineering references
- **Why:** Practical survival knowledge — water purification, shelter building, navigation, first aid
- **How to get:**
  ```bash
  mkdir -p data/knowledge/manuals

  # Key manuals to find (public domain, freely available as PDFs/text):
  # FM 21-76  - Survival
  # FM 3-05.70 - Survival (updated version)
  # FM 5-434  - Earthmoving Operations
  # FM 4-25.11 - First Aid
  # FM 21-10  - Field Hygiene and Sanitation
  # FM 3-34.5 - Environmental Considerations
  # SAS Survival Handbook (not public domain, but widely available)
  # Where There Is No Doctor (Hesperian Health Guides — free PDF)

  # Sources:
  #   https://www.globalsecurity.org/military/library/policy/army/fm/
  #   https://armypubs.army.mil/ (official, some are restricted)
  #   https://archive.org/search?query=army+field+manual

  # Convert PDFs to text for ingestion:
  sudo apt install poppler-utils
  for f in data/knowledge/manuals/*.pdf; do
      pdftotext "$f" "${f%.pdf}.txt"
  done
  ```

**CrashCourse Transcripts (<1 GB)**
- **What:** Transcripts from CrashCourse YouTube educational series (biology, chemistry, history, etc.)
- **Why:** Well-structured educational content, good for explanations
- **How to get:**
  ```bash
  mkdir -p data/knowledge/crashcourse

  # Use yt-dlp to download auto-generated subtitles
  pip install yt-dlp

  # Download subtitles for a playlist (e.g., CrashCourse Biology)
  yt-dlp --write-auto-sub --sub-lang en --skip-download \
      --output "data/knowledge/crashcourse/%(playlist)s/%(title)s" \
      "https://www.youtube.com/c/crashcourse/playlists"

  # Convert .vtt subtitle files to plain text
  for f in data/knowledge/crashcourse/**/*.vtt; do
      # Strip timing info, keep just text
      sed '/^[0-9]/d; /^$/d; /-->/d' "$f" | sort -u > "${f%.vtt}.txt"
  done
  ```

### 2.3 Download Summary

| Category | Item | Size | Downloaded To |
|----------|------|------|---------------|
| LLM | Llama 3.2 3B | 2 GB | Ollama internal (`~/.ollama/models/`) |
| STT | Whisper small.en | 460 MB | `models/whisper-small.en.bin` |
| TTS | Piper lessac-medium | 75 MB | `models/en_US-lessac-medium.onnx` |
| Embeddings | all-MiniLM-L6-v2 | 80 MB | `~/.cache/huggingface/` (auto) |
| Knowledge | Wikipedia EN | 22 GB | `data/knowledge/` |
| Knowledge | iFixit | 2.5 GB | `data/knowledge/` |
| Knowledge | Wikibooks | 2 GB | `data/knowledge/` |
| Knowledge | WikiMed | 1 GB | `data/knowledge/` |
| Knowledge | Gutenberg | 20 GB | `data/knowledge/gutenberg/` |
| Knowledge | Stack Exchange | 10 GB | `data/knowledge/stackexchange/` |
| Knowledge | Field manuals | 2 GB | `data/knowledge/manuals/` |
| Knowledge | CrashCourse | <1 GB | `data/knowledge/crashcourse/` |
| **Total** | | **~62 GB** | |

After ingestion, the ChromaDB vector store will add another ~30-50 GB (embeddings + metadata). Total disk usage: **~120 GB** on a 1TB drive — plenty of room.

---

## Part 3: Ingestion & RAG Pipeline

### 3.1 How RAG Works (Overview)

RAG (Retrieval-Augmented Generation) is how the Oracle finds relevant knowledge before answering:

```
User asks: "How do I purify water?"
         │
         ▼
   ┌─────────────┐
   │  Embed the   │  Convert question to a 384-dim vector
   │  question    │  using all-MiniLM-L6-v2
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │  Search      │  Find the 5 closest vectors in ChromaDB
   │  ChromaDB    │  across all collections
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │  Build       │  System prompt + retrieved chunks + conversation
   │  LLM prompt  │  history → send to Llama 3.2
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │  LLM answers │  Grounded in retrieved knowledge,
   │  in character │  framed as "archive entries"
   └─────────────┘
```

**Ingestion** is the offline step where we:
1. Read each knowledge source (ZIM, text, XML)
2. Strip HTML, extract clean text
3. Split text into ~512-word chunks with 64-word overlap
4. Compute an embedding vector for each chunk
5. Store chunks + vectors in ChromaDB

This only happens once. The Jetson just reads the finished ChromaDB at runtime.

### 3.2 Workstation Setup for Ingestion

Ingestion is CPU/GPU intensive. Do this on a workstation, not the Jetson.

```bash
# Clone the repo on your workstation
git clone https://github.com/eriksalo/radio-oracle.git
cd radio-oracle

# Create venv with ingestion deps
python3 -m venv .venv
.venv/bin/pip install -e ".[rag,ingest]"

# Verify
.venv/bin/python -c "from oracle.rag.embedder import Embedder; e = Embedder(); e.load(); print('OK')"
```

If you have an NVIDIA GPU on your workstation, sentence-transformers will use it automatically. This speeds up embedding by 10-50x.

```bash
# Check if GPU is available for embeddings
.venv/bin/python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 3.3 Ingest Wikipedia

The biggest and most important source. Expect ~20 million chunks.

```bash
# Dry run first — see what you're dealing with
.venv/bin/python scripts/ingest_wikipedia.py \
    data/knowledge/wikipedia_en_all_nopic_latest.zim \
    --dry-run

# Full ingestion
# With GPU: ~8-12 hours
# CPU only: ~2-3 days
.venv/bin/python scripts/ingest_wikipedia.py \
    data/knowledge/wikipedia_en_all_nopic_latest.zim \
    --batch-size 1000
```

**What happens during ingestion:**
1. Opens the 22GB ZIM file and iterates every entry
2. Skips redirects and non-HTML entries
3. Parses HTML, strips `<script>`, `<style>`, `<table>`, and footnote refs
4. Extracts clean text (minimum 100 characters)
5. Chunks text at ~512 words with 64-word overlap, respecting paragraph boundaries
6. Every 1000 chunks: batch-embeds with all-MiniLM-L6-v2 and stores in ChromaDB

Progress is logged. You can safely Ctrl+C and resume (ChromaDB handles duplicates via document IDs).

**Tuning `--batch-size`:**
- GPU with 8GB+ VRAM: `--batch-size 2000` (faster, uses more VRAM)
- GPU with 4GB VRAM: `--batch-size 500` (default)
- CPU only: `--batch-size 100` (avoid OOM)

### 3.4 Ingest Other ZIM Sources

Same process, different collections:

```bash
# iFixit repair guides
.venv/bin/python scripts/ingest_generic_zim.py \
    data/knowledge/ifixit_en_all_latest.zim \
    --collection ifixit

# Wikibooks
.venv/bin/python scripts/ingest_generic_zim.py \
    data/knowledge/wikibooks_en_all_latest.zim \
    --collection wikibooks

# WikiMed medical
.venv/bin/python scripts/ingest_generic_zim.py \
    data/knowledge/wikipedia_en_medicine_nopic_latest.zim \
    --collection wikimed
```

Each creates a separate ChromaDB collection. At query time, the Oracle searches all collections and returns the best matches regardless of source.

### 3.5 Ingest Project Gutenberg

```bash
# Expects a directory of .txt files (one per book)
.venv/bin/python scripts/ingest_gutenberg.py \
    data/knowledge/gutenberg/ \
    --dry-run

.venv/bin/python scripts/ingest_gutenberg.py \
    data/knowledge/gutenberg/ \
    --batch-size 1000
```

### 3.6 Ingest Stack Exchange (Custom)

Stack Exchange uses XML dumps. You'll need a small custom script. Here's the approach:

```bash
# Extract the Posts.xml from the 7z
7z x superuser.com.7z -odata/knowledge/stackexchange/superuser/
```

Create `scripts/ingest_stackexchange.py` (or use the pattern from the existing scripts):

```python
# The key logic for parsing Stack Exchange XML:
import xml.etree.ElementTree as ET
from selectolax.parser import HTMLParser

for event, elem in ET.iterparse("Posts.xml", events=("end",)):
    if elem.tag == "row" and elem.get("PostTypeId") == "1":  # Questions
        title = elem.get("Title", "")
        body_html = elem.get("Body", "")
        body_text = HTMLParser(body_html).text()
        score = int(elem.get("Score", 0))

        if score > 5 and len(body_text) > 100:  # Filter quality
            # chunk and embed as with other sources
            ...
    elem.clear()  # Free memory (the XML files are huge)
```

Filter by score (>5 or >10) to keep only high-quality answers. Otherwise you'll ingest millions of low-quality posts.

### 3.7 Ingest Field Manuals / Plain Text

For any collection of `.txt` files:

```bash
# Same as Gutenberg, just point at the directory and name the collection
.venv/bin/python scripts/ingest_gutenberg.py \
    data/knowledge/manuals/ \
    --dry-run

# Or modify the script to use --collection for a different name
```

You can reuse `ingest_gutenberg.py` for any directory of text files — it just reads `*.txt`, chunks, and embeds.

### 3.8 Verify the Vector Store

After ingestion, check what you've got:

```bash
.venv/bin/python -c "
import chromadb
client = chromadb.PersistentClient(path='data/chroma')
for c in client.list_collections():
    print(f'{c.name}: {c.count()} chunks')
"
```

Expected output (approximate):
```
wikipedia: 20000000
ifixit: 500000
wikibooks: 400000
wikimed: 200000
gutenberg: 5000000
```

Test a query:
```bash
.venv/bin/python -c "
from oracle.rag.retriever import Retriever
r = Retriever()
results = r.query('how to purify water')
for hit in results:
    print(f'[{hit[\"source\"]}] (dist={hit[\"distance\"]:.3f})')
    print(f'  {hit[\"text\"][:150]}...')
    print()
"
```

You should see relevant chunks from Wikipedia, survival manuals, etc.

### 3.9 Transfer to Jetson

The ChromaDB directory is self-contained. Just rsync it:

```bash
# From your workstation:
rsync -avz --progress \
    data/chroma/ \
    jetson:/opt/radio-oracle/data/chroma/

# This transfers ~30-50 GB depending on how many sources you ingested.
# Over gigabit Ethernet: ~5-10 minutes
# Over WiFi: hours. Use Ethernet.
```

Also transfer any models you downloaded on the workstation:
```bash
rsync -avz --progress \
    models/ \
    jetson:/opt/radio-oracle/models/
```

### 3.10 ChromaDB Internals (What's on Disk)

After ingestion, `data/chroma/` contains:

```
data/chroma/
├── chroma.sqlite3          # Metadata, document text, collection info
├── {uuid}/                 # One directory per collection
│   ├── data_level0.bin     # HNSW index (the actual vector data)
│   ├── header.bin          # Index metadata
│   ├── index_metadata.json
│   └── length.bin
└── ...
```

- The `.sqlite3` file stores all document text and metadata
- The `data_level0.bin` files are the HNSW vector indices — these are what make search fast
- Everything is memory-mapped at runtime, so the Jetson only loads what it needs

### 3.11 Tuning RAG Quality

After setup, you can tune retrieval via environment variables:

```bash
# Number of chunks returned per query (default 5)
ORACLE_RAG_TOP_K=5

# Chunk size during ingestion (default 512 words)
ORACLE_CHUNK_SIZE=512

# Overlap between chunks (default 64 words)
ORACLE_CHUNK_OVERLAP=64
```

**If answers are too vague:** Increase `RAG_TOP_K` to 8-10 to give the LLM more context.

**If answers contain irrelevant info:** Decrease `RAG_TOP_K` to 3 or add a distance threshold in `retriever.py`.

**If chunks cut off mid-sentence:** Increase `CHUNK_OVERLAP` to 128.

---

## Final Checklist

After completing all three parts, run through this checklist:

```bash
# On the Jetson:

# 1. Ollama running with model loaded?
ollama list                              # should show llama3.2:3b
curl http://localhost:11434/api/tags     # should return JSON

# 2. Models present?
ls -lh /opt/radio-oracle/models/
# whisper-small.en.bin  (~460 MB)
# en_US-lessac-medium.onnx  (~75 MB)
# en_US-lessac-medium.onnx.json

# 3. ChromaDB populated?
cd /opt/radio-oracle
.venv/bin/python -c "
import chromadb
client = chromadb.PersistentClient(path='data/chroma')
for c in client.list_collections():
    print(f'{c.name}: {c.count()} chunks')
"

# 4. Audio working?
arecord -d 3 -f S16_LE -r 16000 /tmp/test.wav && aplay /tmp/test.wav

# 5. Health check
.venv/bin/python -c "
import asyncio
from oracle.health import run_health_checks
results = asyncio.run(run_health_checks())
for k, v in results.items():
    print(f'  {k}: {\"OK\" if v else \"FAIL\"}')"

# 6. Text mode test
.venv/bin/python -m oracle --mode text
# Ask: "How do I purify water?"
# Verify it answers with RAG-grounded knowledge in character

# 7. Voice mode test
.venv/bin/python -m oracle --mode voice
# Speak into the mic, verify you get a spoken response

# 8. Start the service
sudo systemctl start radio-oracle
journalctl -fu radio-oracle              # watch the logs

# 9. Reboot test
sudo reboot
# After reboot, the Oracle should auto-start and play its greeting
```

Once everything checks out, seal it up in the radio enclosure. The Oracle is operational.
