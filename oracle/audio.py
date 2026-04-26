"""Audio capture and playback with energy-based VAD."""

from __future__ import annotations

import io
import wave

import numpy as np
from loguru import logger

from config.settings import settings


def record_until_silence(
    sample_rate: int | None = None,
    channels: int | None = None,
    energy_threshold: float | None = None,
    silence_duration: float | None = None,
) -> np.ndarray:
    """Record audio from default mic until silence is detected.

    Returns float32 numpy array of audio samples.
    """
    import sounddevice as sd

    sr = sample_rate or settings.audio_sample_rate
    ch = channels or settings.audio_channels
    threshold = energy_threshold or settings.vad_energy_threshold
    max_silence = silence_duration or settings.vad_silence_duration

    block_duration = 0.1  # 100ms blocks
    block_size = int(sr * block_duration)
    silence_blocks = 0
    max_silence_blocks = int(max_silence / block_duration)
    started = False
    frames: list[np.ndarray] = []

    logger.debug(f"Recording: sr={sr}, threshold={threshold}, silence={max_silence}s")

    stream_opts = dict(samplerate=sr, channels=ch, dtype="float32", blocksize=block_size)
    with sd.InputStream(**stream_opts) as stream:
        while True:
            data, _ = stream.read(block_size)
            energy = np.sqrt(np.mean(data**2))

            if energy > threshold:
                started = True
                silence_blocks = 0
                frames.append(data.copy())
            elif started:
                silence_blocks += 1
                frames.append(data.copy())
                if silence_blocks >= max_silence_blocks:
                    break
            # If not started and below threshold, keep waiting

    audio = np.concatenate(frames, axis=0).flatten()
    duration = len(audio) / sr
    # Boost gain so quiet USB mics still produce signal Whisper can transcribe.
    # Target peak ~0.5; cap gain at 50x to avoid blowing up pure noise.
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-5:
        gain = min(0.5 / peak, 50.0)
        if gain > 1.0:
            audio = (audio * gain).astype(np.float32)
            logger.info(f"Recorded {duration:.1f}s of audio (peak {peak:.3f}, applied {gain:.0f}x gain)")
        else:
            logger.info(f"Recorded {duration:.1f}s of audio (peak {peak:.3f})")
    else:
        logger.info(f"Recorded {duration:.1f}s of audio (silent)")
    return audio


def play_audio(audio: np.ndarray, sample_rate: int | None = None) -> None:
    """Play audio through default output device."""
    import sounddevice as sd

    sr = sample_rate or settings.audio_sample_rate
    sd.play(audio, samplerate=sr)
    sd.wait()


def play_wav_bytes(wav_bytes: bytes) -> None:
    """Play WAV data from bytes."""
    import sounddevice as sd

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}[wf.getsampwidth()]
        audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
        if dtype == np.int16:
            audio /= 32768.0
        elif dtype == np.int32:
            audio /= 2147483648.0
        if channels > 1:
            audio = audio.reshape(-1, channels)
        sd.play(audio, samplerate=sr)
        sd.wait()


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int | None = None) -> bytes:
    """Convert float32 audio to WAV bytes."""
    sr = sample_rate or settings.audio_sample_rate
    int16_audio = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16_audio.tobytes())
    return buf.getvalue()


def apply_radio_filter(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Bandpass filter (300-3400Hz) for AM radio speaker feel."""
    from scipy.signal import butter, sosfilt

    low = 300.0 / (sample_rate / 2)
    high = 3400.0 / (sample_rate / 2)
    sos = butter(4, [low, high], btype="band", output="sos")
    return sosfilt(sos, audio).astype(np.float32)
