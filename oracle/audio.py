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
    onset_timeout: float | None = None,
    on_block: Callable[[np.ndarray], None] | None = None,
) -> np.ndarray:
    """Record audio from default mic until silence is detected.

    Returns float32 numpy array of audio samples.  If *should_abort*
    returns True mid-recording, returns whatever has been captured so far
    (may be empty). If *onset_timeout* is set and no speech starts within
    that many seconds, returns empty — used for the post-answer follow-up
    window, where silence means "no follow-up, resume the music".
    *on_block* receives each raw mono block (capture rate) from speech
    onset on — the streaming STT decodes as the user talks.
    """
    import sounddevice as sd
    from scipy.signal import resample_poly

    from oracle.endpoint import VadEndpointer, build_endpointer

    out_sr = sample_rate or settings.audio_sample_rate
    capture_sr = settings.audio_capture_sample_rate
    ch = channels or settings.audio_channels
    threshold = energy_threshold or settings.vad_energy_threshold
    max_silence = silence_duration or settings.vad_silence_duration
    device = _get_input_device()

    block_duration = 0.1  # 100ms blocks
    block_size = int(capture_sr * block_duration)
    silence_s = 0.0
    onset_blocks_left = int(onset_timeout / block_duration) if onset_timeout else None
    started = False
    frames: list[np.ndarray] = []
    # The VAD endpointers classify 16 kHz mono; capture is 16 kHz today
    # (audio_capture_sample_rate) so blocks pass straight through.
    endpointer = build_endpointer(threshold, max_silence)
    endpointer.start()
    vad_gain = 1.0 if settings.vad_backend == "energy" else settings.vad_input_gain

    logger.debug(
        f"Recording: device={device} capture_sr={capture_sr} out_sr={out_sr} "
        f"endpoint={settings.vad_backend} threshold={threshold} silence={max_silence}s"
    )

    stream_opts = dict(
        samplerate=capture_sr,
        channels=ch,
        dtype="float32",
        blocksize=block_size,
        device=device,
        # Smart Turn can take a few hundred ms at a checkpoint; a deeper
        # input buffer keeps PortAudio from dropping samples meanwhile.
        latency="high",
    )
    with sd.InputStream(**stream_opts) as stream:
        while True:
            if should_abort and should_abort():
                logger.debug("Recording aborted")
                break
            data, _ = stream.read(block_size)
            mono = data[:, 0] if data.ndim > 1 else data
            # The VAD models see a gain-boosted copy (quiet USB mic); the
            # energy endpointer's threshold is tuned to the raw level.
            vad_view = mono if vad_gain == 1.0 else np.clip(mono * vad_gain, -1.0, 1.0)

            if endpointer.is_speech(vad_view):
                started = True
                silence_s = 0.0
                frames.append(data.copy())
                if on_block is not None:
                    on_block(mono)
            elif started:
                silence_s += block_duration
                frames.append(data.copy())
                if on_block is not None:
                    on_block(mono)
                so_far = np.concatenate(frames)[:, 0]
                if vad_gain != 1.0:
                    so_far = np.clip(so_far * vad_gain, -1.0, 1.0)
                if endpointer.turn_complete(so_far, silence_s):
                    if isinstance(endpointer, VadEndpointer):
                        logger.debug(f"Endpoint: {endpointer.last_decision} after {silence_s:.2f}s")
                    break
            elif onset_blocks_left is not None:
                onset_blocks_left -= 1
                if onset_blocks_left <= 0:
                    logger.debug("No speech within onset timeout")
                    return np.array([], dtype=np.float32)
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
            logger.info(f"Recorded {duration:.1f}s of audio (peak {peak:.3f}, {gain:.0f}x gain)")
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


# Speech playback buffers this much audio inside PortAudio. The Python
# interpreter feeding the stream stalls for tens of milliseconds at a time
# (GIL held by a fork, a numpy call, a page fault under memory pressure);
# the USB DAC's own "high" latency is 32 ms, so a callback-driven stream
# underran on nearly every stall ("Tell me about this device": 1,018
# underflows in a 9 s clip, 2026-09-30). Blocking writes into a quarter
# second of buffer ride those out, and no Python runs on the audio thread.
_PLAYBACK_LATENCY_S = 0.25
_PLAYBACK_CHUNK_S = 0.05
_PLAYBACK_TAIL_MARGIN_S = 0.1


def _stream_latency(stream) -> float:
    try:
        return float(stream.latency)
    except (TypeError, ValueError, AttributeError):
        return _PLAYBACK_LATENCY_S


def _stream_play(
    audio: np.ndarray,
    sample_rate: int,
    should_abort: AbortCheck,
) -> None:
    """Play *audio* at unity gain. The physical volume knob acts on the
    PulseAudio *sink* (oracle.volume_bridge), which scales every stream —
    music, speech, chime — once and live. Applying pot gain here too made
    speech quieter than music by roughly the knob position squared.

    Blocking-write mode: PortAudio's own thread drains its ring buffer; this
    thread just keeps it topped up in 50 ms chunks and checks for abort in
    between.
    """
    import sounddevice as sd

    audio = np.ascontiguousarray(audio, dtype=np.float32)
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    total = len(audio)
    if total == 0:
        return
    chunk = max(1, int(sample_rate * _PLAYBACK_CHUNK_S))
    underflows = 0
    cursor = 0

    stream = sd.OutputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        device=_get_output_device(),
        latency=_PLAYBACK_LATENCY_S,
    )
    try:
        # PortAudio refuses writes before start(); the first write goes in
        # right after, as large as the ring buffer will take, so the DAC
        # never sees an empty period while the loop is spinning up. The
        # flag from that first write only says the buffer began empty.
        stream.start()
        first = max(chunk, min(total, int(stream.write_available or 0)))
        stream.write(audio[:first])
        cursor = first
        while cursor < total:
            if should_abort and should_abort():
                logger.debug("Playback aborted")
                stream.abort()
                return
            end = min(total, cursor + chunk)
            if stream.write(audio[cursor:end]):
                underflows += 1
            cursor = end
        # stop() does NOT drain on this path (PortAudio ALSA → pulse): what
        # is still in the ring buffer is dropped. Measured on the sink
        # monitor, every clip lost its last ~270 ms, heard as the end of
        # each sentence being clipped (2026-09-30). Push a buffer's worth
        # of silence behind the audio so only silence is discarded.
        tail = np.zeros((chunk, channels), dtype=np.float32)
        pad = int(sample_rate * (_stream_latency(stream) + _PLAYBACK_TAIL_MARGIN_S))
        while pad > 0:
            if should_abort and should_abort():
                stream.abort()
                return
            n = min(pad, chunk)
            stream.write(tail[:n])
            pad -= n
        stream.stop()
    finally:
        stream.close()
    if underflows:
        # The feeder fell more than the whole buffer behind: the process was
        # starved (page faults under memory pressure, GIL held by a long C
        # call). Heard as crackle / dropped words.
        logger.warning(f"Playback: {underflows} underflows in a {total / sample_rate:.1f}s clip")


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


# One speaker: the "checking the archives" ack runs in its own thread and
# the answer's first audio can now land while it is still playing. Two
# concurrent OutputStreams mix; serialize instead.
_speaker_lock = threading.Lock()


def play_audio(
    audio: np.ndarray,
    sample_rate: int | None = None,
    should_abort: AbortCheck = None,
) -> None:
    """Play audio through configured output device (one clip at a time)."""
    src_sr = sample_rate or settings.audio_sample_rate
    out, dst_sr = _resample_to_playback(audio, src_sr)
    with _speaker_lock:
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
