# September 2026 latency & turn-taking upgrade — deploy + measurements

Plan: `~/.claude/plans/buzzing-munching-kazoo.md` (workstation). Goal:
end-of-speech → first audio ≤ 4 s on a RAG question (was ~11 s in July),
turns that end when the user stops talking, no gaps between spoken
sentences. Every phase is gated by a setting and measured here before the
next one starts.

## How to measure

Per-stage timings are logged for every turn as `TURN <label>: …` (see
`oracle/timing.py`) and shown in the diag dashboard's activity feed as
`TIMING`. `ttfa` = end of user speech → first audio; `prefill` is Ollama's
own `prompt_eval_duration` (the only honest cache-hit signal —
`prompt_eval_count` reports the whole prompt even on a hit).

Regression harness (no mic; stops/starts the service around itself):

```bash
ssh erik@radio-oracle.local
nohup sudo /opt/radio-oracle/scripts/sim_turn.sh > /dev/null 2>&1 &
tail -f /tmp/sim_turn.log        # one row per question + medians
```

Questions: `docs/golden_questions.txt` (`+` lines are follow-ups → rewrite path).

## On-device facts (2026-09-27, before any change)

- JetPack 6.2 (R36.5.0), 25 W, Ollama 0.24.0, flash-attn, q8_0 KV, `OLLAMA_NUM_PARALLEL=1`.
- Idle RAM with the service up: 6.0 GB used / 1.2 GB available (ollama 3.3 GB, app 2.5 GB RSS).
- Synthetic prefix-cache probe (1542-token prompt, `num_predict=1`):
  cold 2.99 s (515 tok/s) · identical repeat 0.08 s · same persona+memory,
  new RAG chunk 2.3 s · after an intervening rewrite-style call 2.86 s.
  → prefill runs 515-680 tok/s; the rewrite call evicts the prefix; the
  RAG payload volume dominates.
- `sherpa_onnx` 1.13.4 (`OnlineRecognizer`, `SileroVad`, `TenVad` available), `onnxruntime` 1.26, docker present.

## Phase 0 — baseline (2026-09-27, instrumented, no behaviour change)

Golden set, simulated playback. `first_token` is measured from the end of
context build (so it includes prefill); `ttfa` from end of speech.

| question | ttfa | rewrite | retrieve | prefill | first_token | tok/s | prompt tok | rag chars |
|---|---|---|---|---|---|---|---|---|
| Who was Nikola Tesla? | 16.05 | 0 | 1.20 | 3.27 | 3.62 | 14.2 | 2074 | 6287 |
| + Where did he die? | 15.65 | 1.18 | 1.26 | 4.16 | 4.48 | 14.5 | 2185 | 6270 |
| How does a vacuum tube amplify a signal? | 12.81 | 0 | 1.39 | 3.77 | 4.11 | 14.1 | 2339 | 6391 |
| Boiling point at high altitude? | 11.96 | 0 | 1.44 | 4.29 | 4.64 | 13.9 | 2587 | 6304 |
| + Why is that? | 15.23 | 2.27 | 1.52 | 5.10 | 5.42 | 13.9 | 2672 | 6315 |
| Whaling ship in Moby Dick | 18.07 | 0 | 1.37 | 5.07 | 5.35 | 13.6 | 2949 | 6327 |
| Second-degree burn | 20.89 | 0 | 1.37 | 5.17 | 5.54 | 13.6 | 3012 | 6370 |
| Northern lights | 18.36 | 1.80 | 1.26 | 6.19 | 6.47 | 13.5 | 3194 | 6344 |
| + Colorado? | 22.16 | 1.83 | 1.42 | 6.16 | 6.50 | 13.4 | 3187 | 6270 |
| Who wrote Pride and Prejudice? | 14.34 | 1.79 | 1.28 | 6.72 | 7.03 | 13.4 | 3438 | 6359 |
| **median** | **15.85** | 0.59 | 1.37 | 5.08 | 5.39 | 13.75 | 2811 | |

Findings that changed the plan:
- **Nothing was prefix-cached**: RAG text sat inside the persona message,
  so prompt tokens (and prefill) grew every turn with history (2074 → 3438).
- **The question was in the prompt twice** (stored before `build()`, then
  appended again).
- **The rewrite fired on "Who wrote Pride and Prejudice?"** (≤5-word rule).
- **Kokoro is the biggest single cost**: 6–11 s from first token to first
  audio. Probe (`scripts/probe_tts_contention.py`, `probe_kokoro.py`):
  Kokoro-82M fp32 on this CPU is **RTF 0.77** (9.4 s for a 12 s sentence),
  with *no* contention from Ollama decode; int8 is 3× slower (ARM kernels);
  Supertonic 3 RTF 0.91; Piper lessac-medium (sherpa-onnx) **RTF 0.12**;
  Piper -high RTF ~1.0. Kokoro on CUDA (cp310 sidecar venv, JP6
  onnxruntime-gpu 1.24) **RTF 0.13–0.21** but ~0.7–0.9 GB resident — only
  fits after the torch embedder leaves the app process (Phase 5).
- Replies ran 136–178 tokens (40 s of speech) despite the persona.

## Phase 1 — prompt/pipeline (deployed 2026-09-27)

Changes: RAG block last (after history), question once, `tier1_top_k` 5→3,
`rag_chunk_char_limit` 1200→700, rewrite only on real follow-ups and run
concurrently with raw retrieval, three-stage TTS pipeline (split →
synthesize → play) with first-clause start, speaker lock, acks pre-warmed
and only when the archives are consulted, `num_predict` 220→160.

| question | ttfa | rewrite | retrieve | prefill | first_token | tok/s | prompt tok | rag chars |
|---|---|---|---|---|---|---|---|---|
| Who was Nikola Tesla? | 7.68 | – | 1.16 | 2.22 | 2.56 | 15.2 | 1186 | 2295 |
| + Where did he die? | 8.36 | 1.00 | 2.46 | 2.43 | 2.74 | 15.1 | 1290 | 2295 |
| Vacuum tube | 9.54 | – | 1.26 | 1.34 | 1.67 | 14.9 | 1363 | 2383 |
| Boiling point | 9.15 | – | 1.27 | 1.71 | 2.06 | 14.7 | 1570 | 2323 |
| + Why is that? | 13.15 | 1.00 | 4.53 | 3.11 | 3.42 | 14.8 | 1659 | 2310 |
| Moby Dick | 9.55 | – | 1.35 | 2.14 | 2.43 | 14.4 | 1798 | 2313 |
| Burn | 11.37 | – | 1.26 | 2.18 | 2.46 | 14.5 | 1823 | 2292 |
| Northern lights | 8.90 | – | 1.14 | 2.34 | 2.67 | 14.5 | 1906 | 2284 |
| + Colorado? | 15.63 | 1.00 | 2.76 | 3.56 | 3.83 | 14.4 | 1912 | 2294 |
| Pride and Prejudice | 8.24 | – | 1.17 | 2.59 | 2.88 | 14.4 | 2017 | 2318 |
| **median** | **9.35** | 1.00 | 1.27 | 2.28 | 2.61 | 14.6 | 1729 | |

Where the remaining ~9 s goes: retrieval ~1.3 s, prefill ~2.3 s (the RAG
block + last exchange are new tokens every turn), then **~4–6 s of Kokoro
synthesizing the first unit** — when the model's first sentence has no
comma for 20+ words the first unit is a whole sentence. Follow-ups that
trigger a second retrieval cost ~2.5 s extra.

### Phase 1b/1c — short TTS units (`num_predict` 140, drop unfinished tail)

1b (cut only the *first* unit, ≥12 words at a conjunction, hard 20): median
ttfa **11.18 s** — no better; the model's first sentences ("A vacuum tube
amplifies a signal by using a heated filament to emit electrons into a
vacuum.") had no comma or listed conjunction, so the whole sentence was
synthesized first. Run-to-run variance from the sampled reply text is
±2 s, so single runs are indicative only.

1c (every unit short: clause ≥3 words, soft cut before a conjunction /
preposition after 8 words, hard cut at 12 — `tts_clause_min_words` /
`tts_soft_cut_words` / `tts_hard_cut_words`): median ttfa **8.26 s**;
first token → first audio 1.8–5.9 s. This is the floor for CPU Kokoro.

## Phase 2 — Silero VAD + Smart Turn (code deployed, `energy` still active)

Measured on the Jetson (`scripts/probe_endpoint.py`, Kokoro-synthesized
speech scaled to a quiet-mic peak of 0.03, `vad_input_gain` 8):

- Silero: 1.5 ms per 100 ms block; 46–49/51 blocks flagged speech at peaks
  0.5 / 0.1 / 0.03 / 0.01 (i.e. robust across the ReSpeaker's level range).
- Smart Turn v3.2 int8: 148 ms per checkpoint (20 ms of that is the numpy
  Whisper-feature port, verified against transformers to 2e-3).
  P(complete): finished sentence 0.99 · trailing "and" **0.02** · finished
  sentence + 1 s 0.99. A sentence cut abruptly mid-word scores 0.99 (no
  prosodic cue), so a *real* hesitation is what it keys on.
- Pipeline (sentence, 0.6 s pause, second sentence): endpoint "complete"
  0.3 s after the first sentence — i.e. it returns ~0.6 s sooner than the
  0.9 s energy timer, and keeps listening through "…and ⏸".

**Enable** (needs a voice session to confirm on the real mic):
```
ORACLE_VAD_BACKEND=silero+smartturn     # or: silero (VAD only, 0.25 s)
```
Rollback: remove the line (energy VAD, 0.9 s). Watch the journal for
`Endpoint: complete|incomplete|cap after N.NNs` (DEBUG) and the TIMING
`record` stage.

## Phase 3 — Nemotron streaming STT (code deployed, **not enabled**)

`scripts/probe_stt_streaming.py` on the Jetson, 8 radio phrases:

| | keyword hits | after-endpoint latency |
|---|---|---|
| Parakeet-TDT-0.6B offline (current) | 7/8 | **256 ms** p50, 436 max |
| Nemotron streaming 560 ms int8 | 6/8 | `finish()` 235 ms p50, 454 max (+ per-block feed ≤246 ms) |

The streaming model still has to flush its encoder after the endpoint, so
it saves ~nothing over Parakeet's already-fast batch decode, and it
dropped leading words ("Was Nicola Tesla") and trailing syllables
("Moby", "altitud") — it needs a pre-roll and a longer tail (added:
0.5 s pre-roll, 1.2 s tail), which cost back the latency. **Parakeet
stays.** The backend remains available as `ORACLE_STT_BACKEND=nemotron-streaming`
(~630 MB; do not run with Parakeet resident) for a future early-retrieval
experiment on partial transcripts.

## Phase 3.5 — memory: ONNX embedder, then Kokoro on the GPU (deployed 2026-09-27)

The Kokoro finding above made GPU TTS the only route to ≤4 s, and GPU
TTS needs memory the app didn't have. Two moves:

**ONNX query embedder** (`ORACLE_EMBEDDING_RUNTIME=onnx`,
`oracle/rag/embedder.py`, no torch in the app process).
`scripts/probe_embedder.py` on the Jetson, golden questions:

| runtime | load | per query | cosine vs sentence-transformers |
|---|---|---|---|
| nomic fp32 `model.onnx` (chosen) | 1.7 s | **64 ms** | **1.0000** |
| nomic int8 `model_int8.onnx` | 3.7 s | 23 ms | 0.966–0.976 (batch-dependent) |
| sentence-transformers (torch, CPU) | 17.2 s | 137 ms | — |

App RSS 2.5 GB → 2.18 GB with fp32 ONNX (→ 1.91 GB once Kokoro moved out).

**Kokoro sidecar on CUDA** (`oracle/tts_server.py`, `systemd/radio-oracle-tts.service`,
cp310 venv `.venv-tts` with `onnxruntime-gpu==1.24.0` from
pypi.jetson-ai-lab.io/jp6/cu126 + `kokoro-onnx`; `ORACLE_TTS_BACKEND=server`).
fp16 model, CUDA arena capped at 512 MB. Ready in 4.5 s; **7.8 s of
speech in 1.73 s over loopback HTTP (RTF 0.22)** vs 0.77 on the CPU.
Pitfall fixed: the JP6 wheel also exposes TensorrtExecutionProvider and
kokoro-onnx's default picks it first — minutes of engine building and
~1 GB before failing; the sidecar builds the CUDA session itself
(`Kokoro.from_session`). The `oracle` user needed the `video`,`render`
groups (added 2026-09-27) or CUDA fails with error 801.

Memory with everything up (ollama 3.0 GB RSS, app 1.9 GB, sidecar
cgroup 0.95 GB / 2.2 GB RSS incl. shared CUDA libs): **~0.5 GB available,
~1 GB of the 8 GB swapfile in use** after the day's probes. Tight; see
the int8 embedder note below for the next 0.5 GB.

**Gotcha found on the first GPU-TTS harness run**: every question was
gated ("all 18 hits above distance gate 0.32"). The sentence-transformers
path returns *un-normalised* mean-pooled vectors (norm ≈ 23) and the
FAISS `score_scale`/gate were calibrated on those inner products; the
ONNX backend L2-normalised. Identical direction (cos 0.99999994, same
hits in the same order) but distance 0.96 instead of 0.14. Fixed by not
normalising — vectors are now bit-identical (norm 23.1100). That run's
4.47 s median is therefore *without* RAG context and is superseded below.

### Phase 3.5 result (GPU Kokoro + ONNX embedder, RAG active, Ollama)

| question | ttfa | rewrite | retrieve | prefill | first_token | 1st tok→audio | tok/s | prompt tok |
|---|---|---|---|---|---|---|---|---|
| Who was Nikola Tesla? | 5.03 | – | 0.67 | 2.26 | 2.51 | 1.86 | 11.6 | 1186 |
| + Where did he die? | 6.15 | 1.00 | 1.82 | 2.42 | 2.67 | 1.66 | 11.5 | 1302 |
| Vacuum tube | 4.47 | – | 0.68 | 1.38 | 1.62 | 2.17 | 11.9 | 1391 |
| Boiling point | 5.80 | – | 0.84 | 1.85 | 2.11 | 2.85 | 11.6 | 1606 |
| + Why is that? | 8.42 | 1.00 | 2.61 | 3.27 | 3.55 | 2.26 | 11.1 | 1759 |
| Moby Dick | 5.98 | – | 0.78 | 2.20 | 2.42 | 2.77 | 11.9 | 1830 |
| Burn | 4.36 | – | 0.72 | 2.17 | 2.40 | 1.23 | 11.6 | 1809 |
| Northern lights | 5.98 | – | 0.63 | 2.29 | 2.56 | 2.79 | 11.5 | 1858 |
| + Colorado? | 8.48 | 1.00 | 2.17 | 3.46 | 3.70 | 2.62 | 11.3 | 1845 |
| Pride and Prejudice | 5.28 | – | 0.60 | 2.35 | 2.59 | 2.09 | 11.1 | 1914 |
| **median** | **5.89** | | 0.75 | 2.28 | 2.54 | | 11.55 | 1784 |

**15.85 s → 5.89 s median.** Where the rest goes: prefill ~2.3 s (the
RAG block + last exchange are new tokens every turn, ~800 of the ~1800;
prefill also runs ~20% slower now that the sidecar's CUDA context shares
the GPU — decode fell 14.5 → 11.5 tok/s), retrieval 0.7 s, Kokoro first
unit 1.2–2.9 s. Follow-ups that need a rewrite + second retrieval sit at
~8.4 s. Note the earlier 1c run's 8.26 s → this run's 5.89 s is the GPU
TTS alone (prefill and retrieval unchanged).

Next levers, in order: llama-server (Phase 4), a smaller RAG block
(2 chunks / 500 chars ≈ −0.5 s prefill), and the int8 embedder for memory
headroom (swap is in use).

## Phase 4 — llama-server (code deployed; `ORACLE_LLM_BACKEND=ollama` still active)

`oracle/llm.py` speaks both APIs; `systemd/llama-server.service` runs
NVIDIA's `ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin` (JP6 /
L4T r36) with `--runtime=nvidia --network host`, serving **the same GGUF
Ollama pulled** (`/usr/share/ollama/.ollama/models/blobs/sha256-85e4…`,
bind-mounted read-only) on 127.0.0.1:8080 with `-ngl 99 -c 4096 -np 1
-fa on -ctk q8_0 -ctv q8_0 --cache-ram 512 --cache-reuse 256 --jinja`.
`Conflicts=ollama.service`: only one may hold the model.

Bring-up:
```bash
sudo cp /opt/radio-oracle/systemd/llama-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl stop ollama && sudo systemctl start llama-server
curl -s 127.0.0.1:8080/health          # {"status":"ok"}
# /opt/radio-oracle/.env:  ORACLE_LLM_BACKEND=llama-server
sudo systemctl restart radio-oracle
```
Rollback: remove the env line, `systemctl stop llama-server && systemctl start ollama`.
The turn timer's `prefill` / `cached_tokens` come from the server's
`timings` (`cache_n`), so prefix-cache hits are visible per turn.

Pitfalls met: `docker pull` of the NVIDIA image wedged three times on the
same layer (`failed to cleanup "extract-…" NotFound`, containerd
snapshotter), even after `docker system prune -af` and a daemon restart.
**Native build used instead**: `apt install cmake cuda-toolkit-12-6`
(~2.5 GB, nvcc 12.6) then `sudo scripts/build_llama_cpp.sh` →
`/opt/llama.cpp/bin/llama-server` (sm_87, `-j4`), served by
`systemd/llama-server-native.service` (install it as
`/etc/systemd/system/llama-server.service`). No docker dependency.

More gotchas on the way to a running server (all fixed in the units):
- `--mlock` is not a flag in llama.cpp 0.5.0-dev (it moved under a
  `--mmap`-style option); dropped.
- Ollama's blob is mode 644 but `~ollama/.ollama` blocks traversal;
  `models/qwen3-4b-instruct-2507-q4_K_M.gguf` is a **hard link** to the
  blob (same filesystem, no copy) and the units load that.
- `radio-oracle.service` had `Wants=ollama.service`: every app restart
  pulled Ollama up, and `Conflicts=ollama.service` then stopped
  llama-server mid-load. Now only `After=ollama.service llama-server.service`.
  `ollama.service` is disabled; whichever server is enabled wins.

First measurements (native build, `-c 4096 -fa on -ctk/-ctv q8_0`):
same 274-token prompt twice → call 1 `prompt_ms` 604 (454 tok/s cold),
decode 25 tok in 1.57 s (**16.0 tok/s**; Ollama gave 14.5 alone, 11.5
next to the TTS sidecar); call 2 `cache_n` 273, `prompt_ms` 61. Memory
with llama-server + sidecar + app: ~0.8 GB available, 0.67 GB swap.

### Phase 4 result (llama-server + GPU Kokoro + ONNX embedder, RAG active) — **current production state**

| question | ttfa | rewrite | retrieve | prefill | first_token | 1st tok→audio | cached tok | prompt tok |
|---|---|---|---|---|---|---|---|---|
| Who was Nikola Tesla? | 4.52 | – | 0.59 | 2.17 | 2.21 | 1.72 | 1 | 1195 |
| + Where did he die? | 4.05 | 1.00 | 1.57 | 1.15 | 1.24 | 1.25 | 694 | 1296 |
| Vacuum tube | 3.91 | – | 0.65 | 1.17 | 1.20 | 2.06 | 795 | 1398 |
| Boiling point | 3.97 | – | 0.67 | 1.38 | 1.41 | 1.88 | 886 | 1594 |
| + Why is that? | 5.97 | 1.00 | 2.23 | 1.39 | 1.50 | 2.24 | 1013 | 1771 |
| Moby Dick | 5.12 | – | 0.71 | 1.58 | 1.75 | 2.66 | 1142 | 1843 |
| Burn | 3.87 | – | 0.68 | 1.58 | 2.02 | 1.17 | 1143 | 1845 |
| Northern lights | 4.97 | – | 0.56 | 1.57 | 1.76 | 2.64 | 1203 | 1904 |
| + Colorado? | 6.77 | 1.00 | 2.10 | 2.20 | 2.32 | 2.35 | 696 | 1898 |
| Pride and Prejudice | 4.56 | – | 0.57 | 1.76 | 2.00 | 1.99 | 1204 | 1999 |
| **median** | **4.54** | | 0.68 | 1.57 | 1.76 | | | 1807 |

**Baseline 15.85 s → 4.54 s median** (plain questions 3.9–5.1 s;
follow-ups with a rewrite + second retrieval 6.0–6.8 s). The prefix cache
is doing its job (`cached_tokens` 700–1200 of ~1800 per turn; the rewrite
call no longer evicts it because llama-server keeps the slot's cache in
RAM — note turn 2's prefill 1.15 s right after a rewrite). Decode ~12
tok/s in the harness (16 in isolation) because Kokoro synthesis on the
GPU overlaps generation.

## Where it stands / what's next (2026-09-27 evening)

Active on the Jetson (`/opt/radio-oracle/.env`): `ORACLE_LLM_BACKEND=llama-server`,
`ORACLE_TTS_BACKEND=server`, `ORACLE_EMBEDDING_RUNTIME=onnx`,
`ORACLE_STT_BACKEND=parakeet`, energy VAD. Services: `llama-server`
(native, enabled; `ollama` disabled), `radio-oracle-tts`, `radio-oracle`.

Remaining levers, in order of expected payoff:
1. **Enable `ORACLE_VAD_BACKEND=silero+smartturn`** after a live-mic
   check (Phase 2): ends turns ~0.6 s sooner and stops mid-thought
   cut-offs. Needs Erik at the radio.
2. **Smaller RAG block** (2 chunks or 500 chars): prefill is now the
   biggest fixed cost (~1.2–1.6 s for ~700 new tokens); −0.4 s.
3. **Rewrite path** (follow-ups +2 s): a tiny resident rewrite model is
   now affordable on llama-server? No — memory. Better: skip the second
   retrieval when the raw-query hits already pass the gate.
4. **Memory headroom**: with llama-server (3.2 GB, unified/GPU, not
   pageable) + the sidecar (0.9 GB) resident, the app's CPU-side pages
   are what gets swapped (1.7–3.3 GB of swap seen right after restarts;
   retrieval latency stayed 0.6–0.7 s, so the hot set fits). **int8
   embedder rejected**: `scripts/probe_embedder_recall.py` gives a mean
   top-5 Jaccard overlap of only 0.69 vs fp32 (0.20 on two questions, two
   top-1 changes). Applied instead: llama-server `--cache-ram 256`,
   `-c 3584` (prompts peak ~2.5k tokens). Further options: shorter
   `max_context_turns`, or moving Parakeet (0.7 GB) to the GPU via a
   sherpa-onnx CUDA build.
5. Workstation jobs: livekit-wakeword retrain (`docs/wakeword-retrain-runbook.md`).

Not committed to git as of this writing — the working tree in
`~/projects/radio-oracle` holds everything; the Jetson was updated by rsync.

## Deferred

- JetPack 7.2.1 / TensorRT Edge-LLM — only if Phase 4 leaves ttfa > 4 s.
- LLM swap: Qwen3.5-4B and Granite 4 hybrids re-process the whole prompt in
  llama.cpp (open bugs) — would undo Phase 1. Gemma 4 E2B Q4_K_S is the one
  A/B candidate after Phase 4.
- TTS swap: Kokoro stays. Chatterbox Turbo (GPU) is a later personality experiment.
- Software AEC during music: two USB clocks; see the July mono-gambit notes.


## 2026-09-28 — reader fixes, activity memory, speaker identification

Plan: `~/.claude/plans/vast-beaming-hammock.md`. Commits `09b2e20`,
`6ca7c55`, `3efb982`, `1424676`.

**Reader (Phase A).** Fresh books start at the first real chapter heading
(the Gutenberg preamble is chapter 0 and is skipped; "read the preface"
opts in). Chapter navigation by voice: "go to chapter three", "chapter
XV", "the last chapter", "go back a chapter", "start the book over",
"the chapter called Loomings", "what am I reading?". Chapter names come
from the heading plus the short first paragraph the indexer left behind
("CHAPTER I: LOOMINGS"). In music mode chapter words only enter the
reader when a book is in progress; "next slide" (Parakeet's version of
"next song") is `next`. Loose title matches are confirmed ("Did you mean
X, by Y?"). Paragraphs are spoken as ≤30-word pipelined units (60-word
units overflowed the GPU sidecar's CUDA arena → HTTP 500 → CPU Kokoro).
`scripts/probe_reader.py` exercises all of it on the production DB and
restores any bookmark it touches; `scripts/check_books_db.py` verified
books/chapters ids agree (150 random + the bookmarked ones: 0 mismatches).
Also: a phantom long press at boot (ADC settling) toggled reader mode —
presses in the poller's first 2 s are ignored.

**Activity memory (Phase B).** `oracle/memory/journal.py`: `events` table
in oracle.db fed by the activity feed (playing / asked / answered /
music_request) and the reader (book started / resumed / chapter /
stopped / finished, with the spoken status). The prompt gets a
deterministic "What you remember doing with Erik" block (current book +
position, artists played, music asked for, questions from earlier
sessions — earlier sessions only, so the prefix cache survives). Session
summaries include the activity log; the profile prompt is structured
(Name only if said, Music, Books, Interests, Projects & people,
Preferences) and the drifted old profile was reset once (`meta.profile_version=2`).

**Speaker identification (Phase C).** TitaNet-small
(`models/nemo_en_titanet_small.onnx`, sherpa-onnx) embeddings of each
command, 76–170 ms on the CPU. On Kokoro voices: same speaker 0.85–0.90,
other speakers ≤0.41 (probe: 8/8 at threshold 0.6). CAM++ was
content-sensitive (same voice/different text 0.1) and rejected. Once per
session, an unknown voice gets "Is this Erik?" / "Who am I talking to?";
a "yes" or a name enrols that utterance plus the next two. Memory,
journal, session and bookmarks are all per user
(`users`/`voiceprints`/`profiles` tables; `bookmarks(user, book_id)`).
Settings: `speaker_id_enabled`, `speaker_threshold` 0.6,
`speaker_ask_threshold` 0.4.

**Memory: zram.** The Jetson's `nvzramconfig` creates six 634 MB zram
swap devices at priority 5, above the 8 GB NVMe swapfile. With
llama-server + the TTS sidecar pinned, the app's idle pages went to zram
— 3.7 GB of "swap" held *compressed in RAM* (~2.6 GB), 60 MB free, the
app's RSS down to 47 MB, and every turn paging in. `nvzramconfig` is now
disabled and the swapfile takes the overflow (~0.9–1.3 GB) from NVMe.
Still tight (~0.3–0.4 GB free); next candidates are Parakeet on the GPU
or the fp16 embedder.
