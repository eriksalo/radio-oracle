from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class OracleSettings(BaseSettings):
    model_config = {"env_prefix": "ORACLE_"}

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout: float = 120.0

    # Whisper STT
    stt_backend: Literal["faster-whisper", "pywhispercpp"] = "faster-whisper"
    # faster-whisper: HuggingFace model name (downloaded on first use)
    faster_whisper_model: str = "small.en"
    faster_whisper_device: str = "cpu"
    faster_whisper_compute: str = "int8"
    # pywhispercpp (legacy): path to .bin ggml model
    whisper_model_path: Path = Path("models/whisper-base.en.bin")
    whisper_force_cpu: bool = True
    whisper_language: str = "en"

    # Kokoro TTS
    tts_model_path: Path = Path("models/kokoro-v1.0.onnx")
    tts_voices_path: Path = Path("models/voices-v1.0.bin")
    tts_voice: str = "am_michael"   # American male, natural
    tts_speed: float = 1.0

    # Audio
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    vad_energy_threshold: float = 0.004
    vad_silence_duration: float = 1.5
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
    rag_top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64
    # Comma-separated collection names to search by default. Empty/None means
    # all collections — but on memory-constrained hosts, large HNSW indexes
    # (e.g. a full Wikipedia embedding) won't fit in RAM, so restrict the set.
    rag_collections: str | None = None

    # Two-tier retrieval (snappy first answer + optional deep cross-encoder rerank)
    tier1_top_k: int = 5  # results per collection in snappy mode
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
        "wikipedia":   {"model": "nomic-ai/nomic-embed-text-v1.5", "query_prefix": "search_query: ", "ef_search": 64, "score_scale": 20.0},
        "gutenberg":   {"model": "nomic-ai/nomic-embed-text-v1.5", "query_prefix": "search_query: ", "ef_search": 64, "score_scale": 20.0},
        "wikimed":     {"model": "nomic-ai/nomic-embed-text-v1.5", "query_prefix": "search_query: ", "ef_search": 64, "score_scale": 20.0},
        "wikibooks":   {"model": "nomic-ai/nomic-embed-text-v1.5", "query_prefix": "search_query: ", "ef_search": 64, "score_scale": 20.0},
        "ifixit":      {"model": "nomic-ai/nomic-embed-text-v1.5", "query_prefix": "search_query: ", "ef_search": 64, "score_scale": 20.0},
        "crashcourse": {"model": "nomic-ai/nomic-embed-text-v1.5", "query_prefix": "search_query: ", "ef_search": 64, "score_scale": 20.0},
        "music":       {"model": "nomic-ai/nomic-embed-text-v1.5", "query_prefix": "search_query: ", "ef_search": 64, "score_scale": 20.0},
    }

    # Memory
    db_path: Path = Path("data/oracle.db")
    max_context_turns: int = 10
    summary_threshold: int = 20

    # Music player
    music_path: Path = Path("music")
    music_db_path: Path = Path("data/music.db")
    music_radio_filter: bool = True    # apply AM bandpass to music playback

    # Books / e-reader
    books_path: Path = Path("data/books")
    books_db_path: Path = Path("data/books.db")
    reading_paragraph_pause: float = 0.6   # seconds between paragraphs
    reading_chapter_pause: float = 2.0     # seconds between chapters

    # Mode
    mode: Literal["text", "voice", "hardware"] = "text"
    log_level: str = "INFO"
    # Hardware
    # NOTE: action_button_pin / power_switch_pin are retained for documentation
    # only — the switches are read via the ADS1115 (channels below), not GPIO,
    # because the Tegra234 GPIO INPUT register has a loopback bug on JP 6.2.x
    # for these pads. See memory/hdr40-pinmux-overlay.md.
    action_button_pin: int = 18  # (legacy wiring on BCM 18; now unused for reads)
    led_red_pin: int = 16       # BOARD pin 16 (via 330Ω to common-anode RGB LED)
    led_green_pin: int = 18     # BOARD pin 18
    led_blue_pin: int = 22      # BOARD pin 22
    power_switch_pin: int = 17   # (legacy wiring on BCM 17; now unused for reads)
    wake_word: str = "librarian"       # (legacy) spoken keyword checked in STT transcript
    wakeword_model: str = "alexa"        # openWakeWord pretrained model name
    wakeword_threshold: float = 0.5     # detection confidence threshold (0–1)
    long_press_threshold: float = 1.0  # seconds — long press triggers Librarian-mode toggle
    pot_i2c_bus: int = 7                  # /dev/i2c-N for the ADS1115 (header pins 3/5)
    pot_ads1115_addr: int = 0x48          # default ADDR-floating address
    pot_ads1115_channel: int = 0          # AIN0 (single-ended)
    power_switch_ads1115_channel: int = 1   # AIN1, SPST toggle → GND, 10k pull-up to 3V3
    action_button_ads1115_channel: int = 2  # AIN2, momentary push-button → GND, 10k pull-up to 3V3


settings = OracleSettings()
