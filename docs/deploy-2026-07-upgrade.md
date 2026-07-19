# Deploying the July 2026 upgrade to the Jetson

The 2026-07 push (catalog fix, wired e-reader, persistent memory, latency
work, RAG quality, Qwen3 swap) was developed and unit-tested off-device —
the Jetson was unreachable at the time. This is the on-device deploy +
verification checklist. Work through it top to bottom.

## 1. Deploy

```bash
# From the workstation checkout
rsync -avz -e ssh --rsync-path="sudo rsync" \
    oracle config scripts systemd pyproject.toml CLAUDE.md \
    erik@radio-oracle.local:/opt/radio-oracle/

ssh erik@radio-oracle.local
sudo -u oracle /opt/radio-oracle/.venv/bin/pip install -e "/opt/radio-oracle[all]"
ollama pull qwen3:4b-instruct-2507-q4_K_M   # ~2.5 GB
sudo systemctl restart radio-oracle
sudo journalctl -fu radio-oracle
```

Rollback at any point: `ORACLE_OLLAMA_MODEL=llama3.2:3b` in
`/opt/radio-oracle/.env` and restart (llama3.2:3b stays pulled).

## 2. Smoke checks (in order)

1. **Boot**: service reaches "radio" mode; music starts; journal shows
   Kokoro + base.en + retriever preloaded during `voice_init`.
2. **Music**: "librarian … next song" — skip should be instant (no 0.4s
   lag on single button press either). "play <artist>" still works.
   Then prove the catalog fix: index a temp dir
   (`.venv/bin/python scripts/index_music.py /tmp/testmusic` with 2-3
   MP3s into a scratch `ORACLE_MUSIC_DB_PATH`) and `--list` it — the
   pre-fix code raised `no such column: track_id` here.
3. **Memory pressure with Qwen3-4B**: `tegrastats` while asking a
   librarian question — the new model is ~0.7GB bigger than Llama 3.2 3B
   plus a larger KV cache (num_ctx=8192). If OOM/swapping: first drop
   `ORACLE_OLLAMA_NUM_CTX=4096`; if still tight, roll back the model.
4. **Qwen3 behavior**:
   - Persona tone: a few librarian turns; check for `<think>` leakage or
     un-persona-like throat-clearing (the -instruct-2507 variant is
     non-thinking, so none is expected). Tune `config/persona.toml` if
     the voice reads differently than Llama's did.
   - Radio LLM-JSON intent: "play some jazz", "read me Moby Dick" —
     journal should log clean `{"action": ...}` parses (expect fewer
     failures than Llama).
   - Tok/s: journal timestamps around a streamed reply; target ≥15
     decode tok/s. Record the number in this file.
5. **Reader (new)**: "librarian … I'd like to read a book" → asks for a
   title → say "Moby Dick" → announces + reads. Short press pauses,
   double press next chapter, long press back to radio (music resumes).
   Power-cycle, "read a book" again → resumes from the bookmark.
6. **Memory (new)**: have a short conversation, restart the service,
   ask "what did we talk about earlier?" — it should answer from the
   injected session summary (finalized at shutdown, or by the boot
   catch-up sweep ~a minute after restart).
7. **RAG quality knobs** (each has an env kill-switch):
   - Follow-up rewrite: "who was Nikola Tesla?" then "where did he
     die?" — second answer should be grounded (journal logs the
     rewritten query). Off: `ORACLE_RAG_QUERY_REWRITE=false`.
   - Deep mode: "tell me more about that" triggers the cross-encoder —
     time it in the journal; if the rerank step costs >1.5s, set
     `ORACLE_RAG_RERANK_ENABLED=false` until Phase 3.
   - nprobe 64→128: compare retrieval latency in the journal against
     the old ~per-collection numbers; revert per-collection `ef_search`
     in settings if it hurts.
   - Distance gate: ask something absurd ("what is the flurbon
     coefficient?") — journal should show "injecting nothing" and the
     answer should admit the archives are silent. Tune
     `ORACLE_RAG_MAX_DISTANCE` if real questions get gated (watch for
     the log line).
8. **Latency probe**: time wake→first-audio for (a) "next song"
   (target <3s) and (b) a librarian question (target <5s first
   sentence). Record below.

## 2b. Parakeet STT (opt-in, after the Qwen3 swap is verified)

```bash
# On the Jetson: install the k2-fsa CUDA (JetPack 6.2 / CUDA 12.6) wheel per
# https://k2-fsa.github.io/sherpa/onnx/install/linux.html , then:
./scripts/download_models.sh          # fetches the parakeet bundle (~700MB)
# /opt/radio-oracle/.env:
#   ORACLE_STT_BACKEND=parakeet
#   ORACLE_PARAKEET_PROVIDER=cuda
sudo systemctl restart radio-oracle
```

Verify: radio commands and librarian questions both transcribe correctly
(one model now serves both — journal should show no whisper load/unload
churn); measure RTF on a ~5s utterance vs faster-whisper base/small
(expect a large win); check RAM: ~700MB resident alongside the LLM.
Rollback: ORACLE_STT_BACKEND=faster-whisper.

## 3. Recorded results

| Check | Date | Result |
|---|---|---|
| Qwen3 decode tok/s | 2026-07-19 | 11.1 @ 15W → **16.2 @ 25W** (nvpmodel -m 1; temp 60°C under load) |
| Qwen3 footprint | 2026-07-19 | 3.3GB @ num_ctx 4096 + FA + q8 KV (was 4.2GB @ 8192) |
| RAG recall@5 (golden 32) | 2026-07-19 | **93.75%** (misses: 2 verbatim-quote literary lookups) |
| Retrieval latency (warm, nprobe 128) | 2026-07-19 | p50 ~1.0s (0.14s embed + FAISS) |
| Distance gate | 2026-07-19 | calibrated 0.65 → **0.32** (real 0.10-0.17, junk 0.38+); junk → 0 injected |
| Follow-up rewrite | 2026-07-19 | works: "where did he die?" → "Where did Nikola Tesla die?" (+~2s) |
| Sim. librarian turn (no STT) | 2026-07-19 | first audio ~11s warm / ~19s LLM-cold; full reply 16-28s |
| RAM during full turn | 2026-07-19 | 6.0GB used / 1.2GB avail (FAISS mmap'd; no swap) |
| voice_init (STT+TTS+warm retriever) | 2026-07-19 | ~28s one-time at boot |
| Radio command wake→action | | (manual: needs voice/button) |
| Rerank cost (deep mode) | | (manual: say "tell me more") |
| Parakeet RTF vs whisper | | (not yet enabled) |

**Notes from the 2026-07-19 session:** device now at radio-oracle.local
(DHCP moved it off .186); power mode set to 25W (revert:
`sudo nvpmodel -m 0`); RAG had been silently broken on-device since
2026-05-24 (transformers/torch conflict — fixed, pinned); books FTS index
built (60k titles, ~8ms searches); music catalog verified on fresh AND
production DBs.
