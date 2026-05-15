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
    whisper_model_path: Path = Path("models/whisper-base.en.bin")
    # pywhispercpp doesn't expose use_gpu; we hide CUDA from the STT subprocess
    # via CUDA_VISIBLE_DEVICES so whisper.cpp falls back to CPU and the GPU
    # stays dedicated to ollama. Set to False for GPU mode.
    whisper_force_cpu: bool = True
    whisper_language: str = "en"

    # Piper TTS
    piper_model_path: Path = Path("models/en_US-lessac-medium.onnx")
    piper_sample_rate: int = 22050

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

    # Memory
    db_path: Path = Path("data/oracle.db")
    max_context_turns: int = 10
    summary_threshold: int = 20

    # Mode
    mode: Literal["text", "voice", "hardware"] = "text"
    log_level: str = "INFO"
    voice_play_greeting: bool = True

    # Hardware
    # NOTE: action_button_pin / power_switch_pin are retained for documentation
    # only — the switches are read via the ADS1115 (channels below), not GPIO,
    # because the Tegra234 GPIO INPUT register has a loopback bug on JP 6.2.x
    # for these pads. See memory/hdr40-pinmux-overlay.md.
    action_button_pin: int = 18  # (legacy wiring on BCM 18; now unused for reads)
    led_red_pin: int = 23
    led_green_pin: int = 24
    led_blue_pin: int = 25
    power_switch_pin: int = 17   # (legacy wiring on BCM 17; now unused for reads)
    long_press_threshold: float = 1.0  # seconds — long press triggers Librarian-mode toggle
    pot_i2c_bus: int = 7                  # /dev/i2c-N for the ADS1115 (header pins 3/5)
    pot_ads1115_addr: int = 0x48          # default ADDR-floating address
    pot_ads1115_channel: int = 0          # AIN0 (single-ended)
    action_button_ads1115_channel: int = 2  # AIN2, momentary push-button → GND, 10k pull-up to 3V3
    power_switch_ads1115_channel: int = 1   # AIN1, SPST toggle → GND, 10k pull-up to 3V3


settings = OracleSettings()
