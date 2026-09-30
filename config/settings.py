from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class OracleSettings(BaseSettings):
    model_config = {"env_prefix": "ORACLE_"}

    # LLM server. "ollama" (default) or "llama-server" — llama.cpp's own
    # server (NVIDIA's JP6 container, systemd/llama-server.service) with
    # the same GGUF Ollama pulled; see docs/deploy-2026-09-latency.md §4.
    llm_backend: Literal["ollama", "llama-server"] = "ollama"
    llama_server_url: str = "http://127.0.0.1:8080"
    llama_server_model: str = "qwen3-4b"  # --alias in the unit; informational
    # Ollama
    ollama_host: str = "http://localhost:11434"
    # Qwen3-4B-Instruct-2507 (Q4_K_M, ~2.5GB): best-in-class instruction
    # following + JSON/tool calling for its size as of mid-2026; replaced
    # Llama 3.2 3B (set ORACLE_OLLAMA_MODEL=llama3.2:3b to roll back).
    ollama_model: str = "qwen3:4b-instruct-2507-q4_K_M"
    ollama_timeout: float = 120.0
    # Context window: Ollama defaults to 2048, which silently truncates
    # persona + RAG context + history. 4096 fits the trimmed RAG payload
    # (rag_chunk_char_limit) + memory + 10-message history with room to
    # spare; 8192 measured 4.2GB total for the 4B model on the Jetson and
    # pushed the box into swap once STT/TTS/retriever were resident.
    ollama_num_ctx: int = 4096
    # Factuality-leaning sampling for a RAG-grounded archive persona.
    ollama_temperature: float = 0.6
    ollama_top_p: float = 0.9
    # Hard cap on streamed (spoken) replies. Baseline 2026-09-27: replies
    # ran 136-178 tokens ≈ 40 s of speech despite the persona's "two to
    # four sentences"; 140 ≈ 10 s of generation at 14 tok/s ≈ 100 words,
    # and the user can always ask for more. A reply that hits the cap has
    # its unfinished last sentence dropped rather than spoken. Applies to
    # stream_chat only — summarizer/intent calls stay uncapped.
    ollama_num_predict: int = 140
    # Follow-up query rewrite: model (None = main model; a small resident
    # model would avoid evicting the main model's prefix cache) and output
    # cap — a search query is a dozen tokens.
    ollama_rewrite_model: str | None = None
    ollama_rewrite_num_predict: int = 32

    # STT
    # "parakeet" (NVIDIA Parakeet-TDT-0.6B via sherpa-onnx) replaces both
    # whisper models with one better/faster recognizer; whisper backends
    # remain the fallback. See oracle/stt_parakeet.py.
    # "nemotron-streaming" (NVIDIA Nemotron-speech-streaming-en-0.6b via
    # sherpa-onnx OnlineRecognizer) decodes while the user talks so the
    # transcript is final ~one chunk after the endpoint. See
    # oracle/stt_streaming.py. Replaces Parakeet (don't keep both resident).
    stt_backend: Literal["faster-whisper", "pywhispercpp", "parakeet", "nemotron-streaming"] = (
        "faster-whisper"
    )
    # The Jetson runs the 110M variant (models/sherpa-onnx-nemo-
    # parakeet_tdt_transducer_110m-en-36000-int8, set in .env): same WER on
    # the radio phrases as 0.6B (0.05, 19/24 exact), 102 vs 232 ms, and
    # ~900 MB less resident (0.6B int8 inflates to 0.85-1.1 GB in RAM).
    parakeet_model_dir: Path = Path("models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8")
    parakeet_provider: str = "cpu"  # "cuda" on the Jetson (JetPack 6.2 build)
    parakeet_num_threads: int = 4
    streaming_stt_model_dir: Path = Path(
        "models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25"
    )
    streaming_stt_provider: str = "cpu"
    # 3 of the 6 cores: decoding runs on the capture thread while the
    # wake-word detector and (soon) Kokoro synthesis share the rest.
    streaming_stt_num_threads: int = 3
    # Gain on blocks fed to the streaming recognizer (the finished-buffer
    # path applies its own peak normalisation instead).
    stt_input_gain: float = 8.0
    # faster-whisper: HuggingFace model name (downloaded on first use)
    faster_whisper_model: str = "small.en"
    # Radio-dispatcher path uses a smaller model — vocab is a handful of
    # keywords, so transcript quality doesn't need to match the librarian
    # turn. tiny.en proved too lossy (mis-hearing "next song" as "Okay"
    # cascaded into an LLM-hallucinated 20 s TTS error), so settled on
    # base.en: ~140 MB, ~2x faster than small.en, accurate enough for
    # the keyword set.
    faster_whisper_radio_model: str = "base.en"
    faster_whisper_device: str = "cpu"
    faster_whisper_compute: str = "int8"
    # pywhispercpp (legacy): path to .bin ggml model
    whisper_model_path: Path = Path("models/whisper-base.en.bin")
    whisper_force_cpu: bool = True
    whisper_language: str = "en"

    # Kokoro TTS. "server" delegates synthesis to the GPU sidecar
    # (oracle/tts_server.py, systemd radio-oracle-tts.service) and falls
    # back to in-process CPU Kokoro if it isn't reachable.
    tts_backend: Literal["local", "server"] = "local"
    tts_server_url: str = "http://127.0.0.1:8781"
    tts_server_connect_timeout: float = 2.0
    tts_server_timeout: float = 30.0
    tts_model_path: Path = Path("models/kokoro-v1.0.onnx")
    tts_voices_path: Path = Path("models/voices-v1.0.bin")
    tts_voice: str = "am_eric"  # American male; Erik switched from am_michael 2026-09-30
    tts_speed: float = 1.0
    # Peak level speech is normalized to (0-1; 0 disables). Kokoro output
    # is well below full scale — unnormalized it sits quiet next to
    # loudness-mastered music on the same sink.
    tts_peak: float = 0.9
    # Streaming replies are cut into TTS units as they arrive (see
    # oracle.core.SpeechSplitter). Kokoro on the Jetson CPU is ~0.8× real
    # time, so units are kept short: a clause (, ; : —) closes a unit once
    # it has tts_clause_min_words; with no punctuation a unit is cut before
    # a conjunction/preposition after tts_soft_cut_words or anywhere after
    # tts_hard_cut_words. tts_clause_min_words=0 → whole sentences only
    # (right once TTS runs on the GPU).
    tts_clause_min_words: int = 3
    tts_soft_cut_words: int = 8
    tts_hard_cut_words: int = 12

    # Audio
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    vad_energy_threshold: float = 0.004
    # Librarian questions can have brief mid-sentence pauses, but 1.5s made
    # every turn feel sluggish; 0.9s tested as a good balance. Raise via env
    # if slow speakers get cut off.
    vad_silence_duration: float = 0.9
    # Trailing-silence window after the wake word. 0.6s was tuned for terse
    # commands ("next song") but truncated real questions at the first
    # natural pause ("Why… is the weather…" recorded 2.2s → "Why?").
    # Radio turns now handle questions too, so match the librarian window.
    vad_silence_duration_radio: float = 0.9
    # End-of-utterance detection (oracle/endpoint.py). "energy" is the
    # legacy RMS threshold + fixed trailing silence (vad_silence_duration*).
    # "silero" ends the turn vad_silence_min after Silero VAD last heard
    # speech; "silero+smartturn" additionally asks Smart Turn v3 whether the
    # speaker sounds finished at each checkpoint and keeps listening (up to
    # vad_silence_max) when they don't.
    vad_backend: Literal["energy", "silero", "silero+smartturn"] = "energy"
    silero_vad_model: Path = Path("models/silero_vad.onnx")
    silero_vad_threshold: float = 0.5
    smart_turn_model: Path = Path("models/smart-turn-v3.2-cpu.onnx")
    smart_turn_threshold: float = 0.5
    smart_turn_interval: float = 0.2  # seconds of extra silence between checks
    # Gain applied to the copy of each block the VAD / Smart Turn see (the
    # recording itself is untouched — STT gets its own post-gain). The
    # ReSpeaker path is quiet enough that the energy threshold sits at
    # 0.004 RMS; Silero was trained on normal-level speech.
    vad_input_gain: float = 8.0
    vad_silence_min: float = 0.25
    vad_silence_max: float = 2.0
    # After the oracle answers a question, the mic stays open this many
    # seconds for a follow-up (no wake word needed). Silence resumes the
    # music. 0 disables.
    followup_window_s: float = 5.0
    # PortAudio can't address ALSA `plughw`/`asym` PCMs, so we pin both
    # devices by name and open them at their native rates. Capture is
    # resampled to `audio_sample_rate` for Whisper.
    audio_input_device: str = "ReSpeaker"
    audio_output_device: str = "UACDemoV1.0"
    audio_capture_sample_rate: int = 16000
    audio_playback_sample_rate: int = 48000

    # RAG
    chroma_path: Path = Path("data/chroma")
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "auto"  # auto | cpu | cuda | cuda:N
    embedding_fp16: bool = True  # only honored on CUDA
    embedding_batch_size: int = 256
    # Query-side runtime (oracle/rag/embedder.py). "onnx" runs the encoder
    # through onnxruntime + tokenizers with no torch in the process — on
    # the Jetson that is ~1 GB less RSS and ~10 s less boot than
    # sentence-transformers. Vectors are identical up to int8 noise, so the
    # FAISS indices (built with sentence-transformers) are unchanged.
    embedding_runtime: Literal["sentence-transformers", "onnx"] = "sentence-transformers"
    onnx_embedding_dirs: dict[str, str] = {
        "nomic-ai/nomic-embed-text-v1.5": "models/nomic-embed-text-v1.5-onnx",
    }
    # fp16 export: cosine 1.0000 vs fp32 (max |diff| 0.003 on norm-23
    # vectors), 85 vs 49 ms/query on the workstation, ~270 MB less RAM —
    # chosen 2026-09-28 for memory headroom. model_int8.onnx: 23 ms but
    # cos 0.97 — retrieval changes, rejected.
    onnx_embedding_file: str = "model_fp16.onnx"
    onnx_embedding_threads: int = 4
    onnx_embedding_max_tokens: int = 512  # queries are a sentence; caps padding cost
    rag_top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64
    # Relevance gate: drop hits with normalized distance above this before
    # injecting into the LLM (0 = best). Without it, off-topic chunks are
    # always injected even when nothing relevant exists. Calibrated on the
    # Jetson 2026-07-19 (nomic-v1.5, score_scale=20): real hits 0.10-0.17,
    # junk 0.38+. Re-calibrate when the embedder or score_scale changes.
    rag_max_distance: float = 0.32
    # Per-chunk character cap at injection time. Full 512-word chunks are
    # ~3.3KB; five uncapped chunks cost ~10s of prompt prefill per turn on
    # the Jetson. Measured 2026-09-27: 5 × 1200 chars = ~6.3 KB ≈ 1500
    # tokens ≈ 2.5 s of prefill at ~600 tok/s — the largest per-turn cost
    # once the persona prefix is cached. 700 keeps the answer-bearing lead.
    # 0 disables.
    rag_chunk_char_limit: int = 700
    # Rewrite short/pronoun-heavy follow-ups ("where did he die?") into
    # self-contained queries using recent turns, via a quick LLM call.
    rag_query_rewrite: bool = True
    # Collections never used for knowledge answers (the music catalog is
    # searched via Catalog, not RAG — its rows polluted answers as
    # "Retrieved Knowledge").
    rag_exclude_collections: str = "music"
    # Kill-switch for cross-encoder reranking (deep mode) if it proves too
    # slow on the Jetson CPU.
    rag_rerank_enabled: bool = True
    # Comma-separated collection names to search by default. Empty/None means
    # all collections — but on memory-constrained hosts, large HNSW indexes
    # (e.g. a full Wikipedia embedding) won't fit in RAM, so restrict the set.
    rag_collections: str | None = None

    # Two-tier retrieval (snappy first answer + optional deep cross-encoder rerank)
    tier1_top_k: int = 3  # results per collection in snappy mode (5 → 3: prefill cost, 2026-09)
    tier2_top_k: int = 10  # results per collection in deep mode (pre-rerank)
    tier2_rerank_pool: int = 30  # max candidates fed to the cross-encoder
    tier2_final_top_k: int = 10  # results returned to the LLM after rerank
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str = "cpu"  # keep off the Jetson's shared VRAM

    # Per-collection backends. "chroma" = current ChromaBackend; "faiss" =
    # FaissIvfPqBackend reading from faiss_index_dir/<collection>.{index,sqlite}.
    collection_backends: dict[str, str] = {}
    faiss_index_dir: Path = Path("data/faiss")
    faiss_collection_config: dict[str, dict] = {
        "wikipedia": {
            "model": "nomic-ai/nomic-embed-text-v1.5",
            "query_prefix": "search_query: ",
            "ef_search": 128,
            "score_scale": 20.0,
        },
        "gutenberg": {
            "model": "nomic-ai/nomic-embed-text-v1.5",
            "query_prefix": "search_query: ",
            "ef_search": 128,
            "score_scale": 20.0,
        },
        "wikimed": {
            "model": "nomic-ai/nomic-embed-text-v1.5",
            "query_prefix": "search_query: ",
            "ef_search": 128,
            "score_scale": 20.0,
        },
        "wikibooks": {
            "model": "nomic-ai/nomic-embed-text-v1.5",
            "query_prefix": "search_query: ",
            "ef_search": 128,
            "score_scale": 20.0,
        },
        "ifixit": {
            "model": "nomic-ai/nomic-embed-text-v1.5",
            "query_prefix": "search_query: ",
            "ef_search": 128,
            "score_scale": 20.0,
        },
        "crashcourse": {
            "model": "nomic-ai/nomic-embed-text-v1.5",
            "query_prefix": "search_query: ",
            "ef_search": 128,
            "score_scale": 20.0,
        },
        "music": {
            "model": "nomic-ai/nomic-embed-text-v1.5",
            "query_prefix": "search_query: ",
            "ef_search": 128,
            "score_scale": 20.0,
        },
    }

    # Memory
    db_path: Path = Path("data/oracle.db")
    # Who the memory belongs to until speaker identification says otherwise.
    default_user: str = "erik"
    # Speaker identification (oracle/speaker.py): TitaNet-small embeddings of
    # each command, scored against enrolled voiceprints. Same speaker ≥0.90,
    # different ≤0.41 measured on Kokoro voices; 0.6 is the match line,
    # 0.4 the "worth asking 'Is this X?'" line.
    speaker_id_enabled: bool = True
    speaker_model: Path = Path("models/nemo_en_titanet_small.onnx")
    speaker_threshold: float = 0.6
    speaker_ask_threshold: float = 0.4
    speaker_min_seconds: float = 1.0  # shorter clips are skipped
    speaker_enroll_prints: int = 3  # the confirmed utterance + this many - 1 more
    speaker_answer_timeout: float = 6.0
    max_context_turns: int = 10
    summary_threshold: int = 20

    # Music player
    music_path: Path = Path("music")
    music_db_path: Path = Path("data/music.db")

    # Books / e-reader
    books_path: Path = Path("data/books")
    books_db_path: Path = Path("data/books.db")
    reading_paragraph_pause: float = 0.6  # seconds between paragraphs
    # Paragraphs are spoken as pipelined units of at most this many words
    # (sentence-aligned): the first word comes sooner and no single TTS
    # request is long. 60 (~22 s of speech) overflowed the GPU sidecar's
    # 1 GB CUDA arena on the Jetson; 30 (~11 s) fits.
    reading_unit_max_words: int = 24
    # Units synthesized ahead of playback in the reader pipeline.
    reading_prefetch_units: int = 3
    reading_chapter_pause: float = 2.0  # seconds between chapters

    # Mode
    mode: Literal["text", "voice", "hardware"] = "text"
    log_level: str = "INFO"
    # Hardware
    # NOTE: action_button_pin / power_switch_pin are retained for documentation
    # only — the switches are read via the ADS1115 (channels below), not GPIO,
    # because the Tegra234 GPIO INPUT register has a loopback bug on JP 6.2.x
    # for these pads. See memory/hdr40-pinmux-overlay.md.
    action_button_pin: int = 18  # (legacy wiring on BCM 18; now unused for reads)
    led_red_pin: int = 16  # BOARD pin 16 (via 330Ω to common-anode RGB LED)
    led_green_pin: int = 18  # BOARD pin 18
    led_blue_pin: int = 22  # BOARD pin 22
    power_switch_pin: int = 17  # (legacy wiring on BCM 17; now unused for reads)
    # Power-on routine (oracle/welcome.py): chime + listen, then a greeting,
    # then the four options, before falling back to music.
    welcome_enabled: bool = True
    welcome_first_wait: float = 7.0
    welcome_second_wait: float = 5.0
    welcome_third_wait: float = 5.0
    # Play the "ready to listen" chime after the wake word (oracle/chime.py).
    wake_chime: bool = True
    wake_chime_path: Path = Path("chime-clean-short.wav")
    # Peak level the chime is normalized to (0-1). Safe to run loud: it
    # plays synchronously before the mic opens.
    wake_chime_peak: float = 0.85
    wake_word: str = "librarian"  # (legacy) spoken keyword checked in STT transcript
    wakeword_model: str = "models/librarian.onnx"  # custom-trained openWakeWord model
    # Source feeding wake detection: "raw" (ReSpeaker direct — right when
    # music bypasses AEC), "aec" (aec_source — right for the mono-gambit
    # topology where music plays through aec_sink), or "default".
    wakeword_source: Literal["raw", "aec", "default"] = "raw"
    wakeword_threshold: float = 0.7  # detection confidence threshold (0–1)
    long_press_threshold: float = 1.0  # seconds — long press triggers Librarian-mode toggle
    pot_i2c_bus: int = 7  # /dev/i2c-N for the ADS1115 (header pins 3/5)
    pot_ads1115_addr: int = 0x48  # default ADDR-floating address
    pot_ads1115_channel: int = 0  # AIN0 (single-ended)
    power_switch_ads1115_channel: int = 1  # AIN1, SPST toggle → GND, 10k pull-up to 3V3
    action_button_ads1115_channel: int = 2  # AIN2, momentary push-button → GND, 10k pull-up to 3V3


settings = OracleSettings()
