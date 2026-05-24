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
| USB microphone | **Seeed reSpeaker Lite** (XMOS XU316, 2-mic) — this is what we ship with. On-chip IC/NS/AGC/VNR help with noise; on-chip AEC only works if playback also routes through the same chip (we don't — see "Echo cancellation" below). Any USB mic works in principle. |
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

You'll see output like (current shipping hardware):
```
card 0: UACDemoV10 [UACDemoV1.0], device 0: ...    # USB speaker (Jieli)
card 1: Lite [ReSpeaker Lite], device 0: ...       # USB mic (XMOS XU316)
```

In production, oracle does **not** route through `/etc/asound.conf` — it uses
oracle's per-user PulseAudio. Music plays directly to the speaker sink at
its native 48 kHz; the mic captures through an AEC source. See "Audio
routing" below for the full picture.

### 1.6 Audio routing (PulseAudio: AEC mic, direct-speaker music)

Two audio paths, deliberately asymmetric:

| Path | Routing | Why |
|---|---|---|
| **Mic capture** (wake-word, STT) | mic → `module-echo-cancel` → `aec_source` | NS/AGC are useful; the WebRTC backend also tries to AEC, but with no music going through `aec_sink` it has no reference signal — see notes below. |
| **Music + TTS output** | client → real USB speaker sink directly | Music decoded by `mpg123` at 44.1 kHz, Pulse does one C-side resample to the speaker's 48 kHz, no AEC processing. ~1 % CPU, zero underruns. |

The earlier architecture pushed music through `aec_sink` to get an AEC
reference. The WebRTC AEC backend in this Pulse build only accepts default
settings (32 kHz mono), which murdered music quality — and the in-process
Python audio pipeline that fed it pegged 100 % CPU. We dropped both: the
music player is now an `mpg123` subprocess (see `oracle/music/player.py`)
and `aec_sink` is unused for output.

Trade-off: wake-word reliability degrades while music plays, because AEC
no longer subtracts speaker-from-mic. The action button is the reliable
fallback wake mechanism during music. If you want AEC-during-music back,
the proper fix is `module-combine-sink` fanning music to both the speaker
and `aec_sink` (left as a follow-up).

Why not the XU316's on-chip AEC: it needs a reference signal of what the
speaker is playing. Mic and speaker are separate USB devices (Lite and
UACDemoV1.0); the XU316 sees no reference, so its on-chip AEC is a no-op
for the echo path. The XU316's IC/NS/AGC/VNR still work since those
operate on the mic input alone.

Setup (run once on the Jetson):

```bash
# Enable lingering so oracle's PulseAudio runs without a login session
sudo loginctl enable-linger oracle

# Drop the canonical pulse config into oracle's home
sudo -u oracle mkdir -p /home/oracle/.config/pulse
sudo cp systemd/pulse-default.pa /home/oracle/.config/pulse/default.pa
sudo chown oracle:oracle /home/oracle/.config/pulse/default.pa

# Restart oracle's PulseAudio so it picks up the config
sudo -u oracle XDG_RUNTIME_DIR=/run/user/999 pulseaudio -k
```

The exact config lives in `systemd/pulse-default.pa` (tracked in this
repo). The systemd unit (`systemd/radio-oracle.service`) is already wired
to oracle's user PulseAudio via `XDG_RUNTIME_DIR=/run/user/999` and
`PULSE_SERVER=unix:/run/user/999/pulse/native`.

The `.env` keeps `ORACLE_AUDIO_INPUT_DEVICE=pulse` and
`ORACLE_AUDIO_OUTPUT_DEVICE=pulse`. With our `default.pa`, those resolve
to `aec_source` (AEC'd mic) and the real speaker sink (direct), giving
the asymmetric routing described above.

Verify:

```bash
sudo -u oracle XDG_RUNTIME_DIR=/run/user/999 pactl info | grep Default
# Default Source: aec_source
# Default Sink: alsa_output.usb-Jieli_Technology_UACDemoV1.0_415035313136340C-00.analog-stereo
```

### 1.7 Mic firmware (ReSpeaker Lite)

The XU316 firmware is field-upgradable over USB DFU. Current shipping
firmware is **v2.0.7** (binary in `firmware/`). To check the installed
version:

```bash
sudo lsusb -v -d 2886:0019 | grep bcdDevice
# bcdDevice            2.07
```

To flash (will not brick — alt 0 FACTORY is recovery):

```bash
sudo apt install -y dfu-util
sudo systemctl stop radio-oracle
sudo dfu-util -R -a 1 -D firmware/respeaker_lite_usb_dfu_firmware_v2.0.7.bin
sudo systemctl start radio-oracle
```

### 1.8 Audio test

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

#### Embedding Model — nomic-embed-text-v1.5

- **Size:** ~550 MB
- **What it is:** A 768-dim sentence-transformer that converts text into vectors for FAISS semantic search. Matryoshka-truncatable. Replaces the legacy `all-MiniLM-L6-v2` (384-d) used during the ChromaDB era.
- **Why this one:** Higher recall than MiniLM-L6 on long-form passages; the 768-d vectors pair well with FAISS IVF-PQ at PQ-64 to compress the index 8× while staying within recall budget. Required by the nomic model are the `search_query: ` and `search_document: ` prefixes — already wired into the embedder and the FAISS backend.
- **Download:** Automatic on first use. `sentence-transformers` fetches it from HuggingFace and caches it in `~/.cache/huggingface/`. Loading requires `trust_remote_code=True` (handled by `oracle/rag/embedder.py`).
- **To pre-download:**
  ```bash
  .venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)"
  ```

> **Note on the Jetson:** the embedder runs on CPU there today (~1.2 s/query
> warm). No cp311 CUDA torch wheel exists for JetPack 6.2. See
> `docs/rag-migration-runbook.md` §"Known follow-up" for the three viable
> fix paths if that latency becomes a blocker.

### 2.2 Knowledge Bases

These are the Oracle's "archives" — the offline knowledge it searches to answer questions.

**Important:** Download and ingest these on a workstation with a GPU. The
three-stage pipeline (ZIM → ChromaDB → flat .f32+.sqlite → FAISS IVF-PQ)
computes millions of embeddings and takes hours to days on CPU but is
much faster on a GPU. Only the final `data/faiss/` directory rsyncs to
the Jetson.

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
| Embeddings | nomic-embed-text-v1.5 | 550 MB | `~/.cache/huggingface/` (auto) |
| Knowledge | Wikipedia EN | 22 GB | `data/knowledge/` |
| Knowledge | iFixit | 2.5 GB | `data/knowledge/` |
| Knowledge | Wikibooks | 2 GB | `data/knowledge/` |
| Knowledge | WikiMed | 1 GB | `data/knowledge/` |
| Knowledge | Gutenberg | 20 GB | `data/knowledge/gutenberg/` |
| Knowledge | Stack Exchange | 10 GB | `data/knowledge/stackexchange/` |
| Knowledge | Field manuals | 2 GB | `data/knowledge/manuals/` |
| Knowledge | CrashCourse | <1 GB | `data/knowledge/crashcourse/` |
| **Total** | | **~62 GB** | |

After ingestion, the FAISS layer adds ~85 GB to the Jetson (~83 GB of
chunk-text sqlite, ~1.65 GB of indices). The workstation also keeps
~450 GB of ChromaDB as the rebuild source. Total disk on the Jetson:
**~150 GB** on a 1TB drive — plenty of room.

---

## Part 3: Ingestion & RAG Pipeline

### 3.1 How RAG Works (Overview)

RAG (Retrieval-Augmented Generation) is how the Oracle finds relevant knowledge before answering:

```
User asks: "How do I purify water?"
         │
         ▼
   ┌──────────────┐
   │  Embed the   │  Prefix with "search_query: ", encode to a 768-d
   │  question    │  vector with nomic-embed-text-v1.5
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  Route to    │  Intent regex picks the right collection(s)
   │  collection  │  (oracle/rag/router.py)
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  Search      │  FAISS IVF-PQ: probe ef_search cells, return top-k
   │  FAISS index │  candidates with chunk text from the sister sqlite
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  (Deep mode) │  Cross-encoder rerank on a wider candidate pool
   │  Rerank      │  before returning to the LLM. Snappy mode skips this.
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  Build       │  System prompt + retrieved chunks + conversation
   │  LLM prompt  │  history → send to Llama 3.2
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  LLM answers │  Grounded in retrieved knowledge,
   │  in character │  framed as "archive entries"
   └──────────────┘
```

**Ingestion** is a three-stage offline pipeline:
1. **Stage chunk text in ChromaDB.** `scripts/ingest_zim.py` reads each
   ZIM, strips HTML, splits into ~512-word chunks with 64-word overlap,
   and writes the text + a legacy MiniLM-L6 embedding into ChromaDB.
   ChromaDB is the source of truth for chunk content.
2. **Re-embed with nomic-v1.5 to flat files.** `scripts/reembed_collection.py`
   reads chunks back out of ChromaDB and writes `<name>.vectors.f32` +
   `<name>.text.sqlite` to `data/embeddings/`. GPU+FP16, resume-safe.
3. **Build FAISS IVF-PQ.** `scripts/build_faiss_ivfpq.py` reads the flat
   files and writes `<name>.index` + `<name>.sqlite` to `data/faiss/`.

Only `data/faiss/` ships to the Jetson. ChromaDB stays on the
workstation as the rebuild source for future model swaps. See
[`docs/rag-migration-runbook.md`](rag-migration-runbook.md) for the
command-by-command runbook.

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

### 3.3 Stage Chunk Text into ChromaDB

This is the first ingestion stage — read raw ZIMs, chunk the text,
embed with the legacy MiniLM-L6 model, and write to ChromaDB. The
embedding here is **not** what the Jetson queries; it's just used to
key chunks in ChromaDB. The nomic-v1.5 re-embed in stage 3.4 is what
production retrieval uses.

```bash
# Wikipedia (biggest source — ~11.5M chunks after dedup, ~8-12 h on GPU)
.venv/bin/python scripts/ingest_zim.py \
    data/knowledge/wikipedia_en_all_nopic_latest.zim \
    --collection wikipedia --batch-size 1000

# iFixit repair guides
.venv/bin/python scripts/ingest_zim.py \
    data/knowledge/ifixit_en_all_latest.zim \
    --collection ifixit

# Wikibooks
.venv/bin/python scripts/ingest_zim.py \
    data/knowledge/wikibooks_en_all_latest.zim \
    --collection wikibooks

# WikiMed medical
.venv/bin/python scripts/ingest_zim.py \
    data/knowledge/wikipedia_en_medicine_nopic_latest.zim \
    --collection wikimed
```

`ingest_zim.py` is resume-safe — Ctrl+C and rerun; chunks already in
ChromaDB are skipped via a fast SQL preload of existing IDs (no HNSW
load required).

**Tuning `--batch-size`:**
- GPU with 8GB+ VRAM: `--batch-size 2000` (faster, uses more VRAM)
- GPU with 4GB VRAM: `--batch-size 500` (default)
- CPU only: `--batch-size 100` (avoid OOM)

### 3.4 Re-embed Each Collection with nomic-v1.5

For each collection staged above, run `reembed_collection.py` to produce
the flat `<name>.vectors.f32` + `<name>.text.sqlite` in `data/embeddings/`.
The re-embed step is the slow one (~250 chunks/sec end-to-end on an RTX
4070 SUPER) but it only runs once per model version.

```bash
.venv/bin/python scripts/reembed_collection.py \
    --source wikipedia --target wikipedia \
    --model nomic-ai/nomic-embed-text-v1.5 --dim 768 \
    --db-path data/chroma --out-dir data/embeddings \
    --workers 12 --batch-size 2000 --encode-batch-size 256
```

Defaults to `max_seq_length=512`, which gives a ~5× throughput win over
nomic's 8192 default and only affects ~1% of chunks. Resume-safe: rereads
existing `chunk_id`s from the target text.sqlite on startup and skips them.

### 3.5 Build the FAISS IVF-PQ Index

```bash
.venv/bin/python scripts/build_faiss_ivfpq.py \
    --name wikipedia \
    --in-dir data/embeddings --out-dir data/faiss --dim 768
```

`--nlist` defaults to auto (`clamp(sqrt(n), 64, 4096)`). Outputs
`data/faiss/wikipedia.index` (the IVF-PQ index) + `data/faiss/wikipedia.sqlite`
(`faiss_row` → `chunk_id` / text / source / url / title / chunk_index).
At PQ-64 the indices are tiny: full Wikipedia fits in ~840 MB.

Once the FAISS pair is verified, you can reclaim the flat-file workspace
(`rm data/embeddings/wikipedia.*`); the FAISS sqlite has the same text.

### 3.6 Music Collection (different path)

Music has no ZIM source. Use the workstation-only scripts in the
`Huge Information Stores/` working directory: `_scan_music_metadata.py`
(builds `data/music.db` from mutagen tags) then `_ingest_music_faiss.py`
(synthesizes a sentence doc per track, embeds with nomic-v1.5, writes
the same flat-file pair). Then run `build_faiss_ivfpq.py --name music`
to build the index. See the workstation `CLAUDE.md` for full detail.

### 3.7 Ingest Stack Exchange (Custom)

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

### 3.8 Ingest Field Manuals / Plain Text

For any collection of `.txt` files, stage them via `ingest_zim.py`'s
generic text path (or the legacy `ingest_gutenberg.py` for backward
compatibility — it reads `*.txt`, chunks, and writes to ChromaDB), then
re-embed and build FAISS exactly as in 3.4-3.5.

### 3.9 Verify the FAISS Indices

After all three stages complete, you should have one `.index` + one
`.sqlite` per collection under `data/faiss/`. Counts and disk usage:

```bash
.venv/bin/python -c "
import sqlite3, os
from pathlib import Path
total = 0
for sqlite_path in sorted(Path('data/faiss').glob('*.sqlite')):
    name = sqlite_path.stem
    n = sqlite3.connect(sqlite_path).execute('SELECT COUNT(*) FROM faiss_idmap').fetchone()[0]
    idx_mb = os.path.getsize(sqlite_path.with_suffix('.index')) / 1024**2
    sql_mb = os.path.getsize(sqlite_path) / 1024**2
    print(f'{name:12s} {n:>12,}  index={idx_mb:>8.1f}MB  sqlite={sql_mb:>9.1f}MB')
    total += n
print(f'{\"TOTAL\":12s} {total:>12,}')
"
```

Expected output (production snapshot, 2026-05-19):
```
crashcourse        1,654  index=     1.1MB  sqlite=     6.4MB
gutenberg     10,301,735  index=   717.6MB  sqlite= 40402.0MB
ifixit           181,502  index=    14.5MB  sqlite=   538.3MB
music              4,216  index=     1.2MB  sqlite=     0.9MB
wikibooks        313,401  index=    23.9MB  sqlite=  1057.3MB
wikimed          258,730  index=    20.0MB  sqlite=   934.3MB
wikipedia     11,476,000  index=   800.8MB  sqlite= 36664.7MB
TOTAL         22,537,238
```

Test a query end-to-end (set the env vars first; see Workstream 2):
```bash
ORACLE_COLLECTION_BACKENDS='{"wikipedia":"faiss","gutenberg":"faiss","wikimed":"faiss","wikibooks":"faiss","ifixit":"faiss","crashcourse":"faiss","music":"faiss"}' \
ORACLE_FAISS_INDEX_DIR=data/faiss \
.venv/bin/python -c "
from oracle.rag.retriever import Retriever
from oracle.rag.modes import detect_mode
r = Retriever()
results = r.query('how to purify water', mode=detect_mode('how to purify water'))
for hit in results[:5]:
    print(f'[{hit[\"source\"]}] (d={hit[\"distance\"]:.3f})')
    print(f'  {hit[\"text\"][:150]}...')
    print()
"
```

You should see relevant chunks from Wikipedia, WikiMed, survival manuals, etc.

### 3.10 Transfer to Jetson

Only the FAISS artifacts ship. ChromaDB stays on the workstation as the
rebuild source.

```bash
# From your workstation (~83 GB total over gigabit ethernet, ~12 min):
rsync -avz --progress \
    data/faiss/ \
    jetson:/opt/radio-oracle/data/faiss/

# Also transfer any models you downloaded on the workstation:
rsync -avz --progress \
    models/ \
    jetson:/opt/radio-oracle/models/
```

The Jetson then needs `faiss-cpu` in its venv and three new env vars in
`/opt/radio-oracle/.env` (`ORACLE_FAISS_INDEX_DIR`, `ORACLE_COLLECTION_BACKENDS`,
and the corresponding settings) — see the rsync + Jetson-side patch
detail in [`docs/rag-migration-runbook.md`](rag-migration-runbook.md) §5.

### 3.11 FAISS Artifacts on Disk

After build, `data/faiss/` contains one pair of files per collection:

```
data/faiss/
├── wikipedia.index         # IVF-PQ index (FAISS-native binary)
├── wikipedia.sqlite        # faiss_row → chunk_id / text / source / url / title / chunk_index
├── gutenberg.index
├── gutenberg.sqlite
├── ifixit.index
├── ifixit.sqlite
... (one pair per collection)
```

- The `.index` file is the FAISS IVF-PQ index — small (~PQ-64 compresses
  8× from raw vectors).
- The `.sqlite` file holds the full chunk text + metadata keyed by row
  id. At query time the FAISS backend resolves top-k vector ids and
  fetches text from this sqlite — no chromadb dependency at runtime.
- Both files are memory-mapped lazily, so the Jetson only loads index
  pages it actually probes.

### 3.12 Tuning RAG Quality

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

# 3. FAISS indices populated?
cd /opt/radio-oracle
ls -lh data/faiss/*.index data/faiss/*.sqlite
.venv/bin/python -c "
import sqlite3, glob
for p in sorted(glob.glob('data/faiss/*.sqlite')):
    n = sqlite3.connect(p).execute('SELECT COUNT(*) FROM faiss_idmap').fetchone()[0]
    print(f'{p}: {n:,} chunks')
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
