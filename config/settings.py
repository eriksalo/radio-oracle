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
    whisper_model_path: Path = Path("models/whisper-small.en.bin")
    whisper_language: str = "en"

    # Piper TTS
    piper_model_path: Path = Path("models/en_US-lessac-medium.onnx")
    piper_sample_rate: int = 22050

    # Audio
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    vad_energy_threshold: float = 0.004
    vad_silence_duration: float = 1.5

    # RAG
    chroma_path: Path = Path("data/chroma")
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "auto"  # auto | cpu | cuda | cuda:N
    embedding_fp16: bool = True  # only honored on CUDA
    embedding_batch_size: int = 256
    rag_top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Memory
    db_path: Path = Path("data/oracle.db")
    max_context_turns: int = 10
    summary_threshold: int = 20

    # Mode
    mode: Literal["text", "voice", "hardware"] = "text"
    log_level: str = "INFO"

    # Hardware
    action_button_pin: int = 18  # momentary push-button (short = action, long = mode toggle)
    led_red_pin: int = 23
    led_green_pin: int = 24
    led_blue_pin: int = 25
    power_switch_pin: int = 17  # SPST toggle: closed (LOW w/ pull-up) = device on
    long_press_threshold: float = 1.0  # seconds — long press triggers Librarian-mode toggle


settings = OracleSettings()
