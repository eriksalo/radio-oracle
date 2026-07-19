"""Audio capture and playback with energy-based VAD."""

from __future__ import annotations

import io
import threading
import wave
from collections.abc import Callable

import numpy as np
from loguru import logger

from config.settings import settings

# Type alias for abort callbacks used across recording and playback.
AbortCheck = Callable[[], bool] | None


def record_until_silence(
    sample_rate: int | None = None,
    channels: int | None = None,
    energy_threshold: float | None = None,
    silence_duration: float | None = None,
    should_abort: AbortCheck = None,
) -> np.ndarray:
    """Record audio from default mic until silence is detected.

    Returns float32 numpy array of audio samples.  If *should_abort*
    returns True mid-recording, returns whatever has been captured so far
    (may be empty).
    """
    import sounddevice as sd
    from scipy.signal import resample_poly

    out_sr = sample_rate or settings.audio_sample_rate
    capture_sr = settings.audio_capture_sample_rate
    ch = channels or settings.audio_channels
    threshold = energy_threshold or settings.vad_energy_threshold
    max_silence = silence_duration or settings.vad_silence_duration
    device = _get_input_device()

    block_duration = 0.1  # 100ms blocks
    block_size = int(capture_sr * block_duration)
    silence_blocks = 0
    max_silence_blocks = int(max_silence / block_duration)
    started = False
    frames: list[np.ndarray] = []

    logger.debug(
        f"Recording: device={device} capture_sr={capture_sr} out_sr={out_sr} "
        f"threshold={threshold} silence={max_silence}s"
    )

    stream_opts = dict(
        samplerate=capture_sr, channels=ch, dtype="float32",
        blocksize=block_size, device=device,
    )
    with sd.InputStream(**stream_opts) as stream:
        while True:
            if should_abort and should_abort():
                logger.debug("Recording aborted")
                break
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

    if not frames:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(frames, axis=0).flatten()
    if capture_sr != out_sr:
        # Whisper expects 16k mono; downsample from device native rate.
        from math import gcd
        g = gcd(capture_sr, out_sr)
        audio = resample_poly(audio, out_sr // g, capture_sr // g).astype(np.float32)
    duration = len(audio) / out_sr
    # Boost gain so quiet USB mics still produce signal Whisper can transcribe.
    # Target peak ~0.5; cap gain at 50x to avoid blowing up pure noise.
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-5:
        gain = min(0.5 / peak, 50.0)
        if gain > 1.0:
            audio = (audio * gain).astype(np.float32)
            logger.info(
                f"Recorded {duration:.1f}s of audio (peak {peak:.3f}, {gain:.0f}x gain)"
            )
        else:
            logger.info(f"Recorded {duration:.1f}s of audio (peak {peak:.3f})")
    else:
        logger.info(f"Recorded {duration:.1f}s of audio (silent)")
    return audio


def _resample_to_playback(audio: np.ndarray, src_sr: int) -> tuple[np.ndarray, int]:
    """Resample audio to the speaker's native rate so PortAudio's hw path accepts it."""
    dst_sr = settings.audio_playback_sample_rate
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False), dst_sr
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(src_sr, dst_sr)
    out = resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)
    return out, dst_sr


def _stream_play(
    audio: np.ndarray,
    sample_rate: int,
    should_abort: AbortCheck,
) -> None:
    """Play *audio* via an OutputStream callback that re-reads the pot
    every block, so turning the volume knob is audible immediately
    (e.g. mid-paragraph while the librarian is reading).

    The previous implementation pre-scaled the whole buffer once and
    handed it to sd.play(), which baked the volume in at playback start.
    """
    import sounddevice as sd

    from oracle.hardware.volume import get_volume_control

    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        channels = 1
    else:
        channels = audio.shape[1]

    vc = get_volume_control()
    cursor = 0
    total = len(audio)
    finished = threading.Event()

    def callback(outdata, frames, _time_info, status) -> None:
        nonlocal cursor
        if status:
            logger.debug(f"playback status: {status}")
        take = min(frames, total - cursor)
        gain = vc.gain
        if take > 0:
            chunk = audio[cursor:cursor + take]
            if channels == 1:
                outdata[:take, 0] = chunk * gain
            else:
                outdata[:take] = chunk * gain
            cursor += take
        if take < frames:
            outdata[take:] = 0
            raise sd.CallbackStop()

    stream = sd.OutputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        device=_get_output_device(),
        callback=callback,
        finished_callback=finished.set,
    )
    with stream:
        while not finished.is_set():
            if should_abort and should_abort():
                logger.debug("Playback aborted")
                return
            finished.wait(timeout=0.05)


def _resolve_device(name: str, kind: str) -> int | None:
    """Resolve a device name to its integer index.

    Returns the index if found, or ``None`` to use the system default
    (which /etc/asound.conf should route to the correct USB device).
    PortAudio often misses USB devices under systemd, so falling back
    to None is the expected path on the Jetson.
    """
    import sounddevice as sd

    kind_key = f"max_{kind}_channels"
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if name in dev["name"] and dev[kind_key] > 0:
            logger.info(f"{kind.title()} device: {name!r} → index {idx}")
            return idx
    logger.info(f"{kind.title()} device {name!r} not in PortAudio list; using system default")
    return None


# Cache resolved device indices (None = system default).
_input_device_resolved = False
_input_device_id: int | None = None
_output_device_resolved = False
_output_device_id: int | None = None


def _get_input_device() -> int | None:
    global _input_device_resolved, _input_device_id
    if not _input_device_resolved:
        _input_device_id = _resolve_device(settings.audio_input_device, "input")
        _input_device_resolved = True
    return _input_device_id


def _get_output_device() -> int | None:
    global _output_device_resolved, _output_device_id
    if not _output_device_resolved:
        _output_device_id = _resolve_device(settings.audio_output_device, "output")
        _output_device_resolved = True
    return _output_device_id


def play_audio(
    audio: np.ndarray,
    sample_rate: int | None = None,
    should_abort: AbortCheck = None,
) -> None:
    """Play audio through configured output device."""
    src_sr = sample_rate or settings.audio_sample_rate
    out, dst_sr = _resample_to_playback(audio, src_sr)
    _stream_play(out, dst_sr, should_abort)


def play_wav_bytes(wav_bytes: bytes, should_abort: AbortCheck = None) -> None:
    """Play WAV data from bytes."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        src_sr = wf.getframerate()
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
        out, dst_sr = _resample_to_playback(audio, src_sr)
        _stream_play(out, dst_sr, should_abort)


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
