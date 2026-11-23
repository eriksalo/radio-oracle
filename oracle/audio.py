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


# Soft pre-roll: the always-on wake detector's mic buffer (registered by the
# app). Opening a recording stream takes a few hundred ms; someone who
# answers a prompt at once ("keep going", "yes") lost their first word in
# that gap — "keep going" arrived as "going." twice running (2026-11-23).
_preroll_source: Callable[[float], np.ndarray] | None = None


def set_preroll_source(fn: Callable[[float], np.ndarray] | None) -> None:
    global _preroll_source
    _preroll_source = fn


def record_until_silence(
    sample_rate: int | None = None,
    channels: int | None = None,
    energy_threshold: float | None = None,
    silence_duration: float | None = None,
    should_abort: AbortCheck = None,
    onset_timeout: float | None = None,
    on_block: Callable[[np.ndarray], None] | None = None,
    preroll_fn: Callable[[], np.ndarray] | None = None,
) -> np.ndarray:
    """Record audio from default mic until silence is detected.

    *preroll_fn*, called once the mic stream is open, returns 16 kHz mono
    audio captured just before it (what was said straight after the wake
    word); it becomes the start of the recording and speech counts as
    already begun.

    Returns float32 numpy array of audio samples.  If *should_abort*
    returns True mid-recording, returns whatever has been captured so far
    (may be empty). If *onset_timeout* is set and no speech starts within
    that many seconds, returns empty — used for the post-answer follow-up
    window, where silence means "no follow-up, resume the music".
    *on_block* receives each raw mono block (capture rate) from speech
    onset on — the streaming STT decodes as the user talks.
    """
    import time

    import sounddevice as sd
    from scipy.signal import resample_poly

    from oracle.endpoint import VadEndpointer, build_endpointer

    t_call = time.monotonic()

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
    pending: list[np.ndarray] = []  # speech blocks not yet enough to call it onset
    min_onset = max(1, settings.vad_min_onset_blocks)
    max_blocks = (
        int(settings.vad_max_utterance_s / block_duration) if settings.vad_max_utterance_s else 0
    )
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
        # Explicit pre-roll (after a wake word) always starts the recording.
        # Otherwise the detector's buffer since this call is used only if
        # the person is already speaking at the end of it.
        soft = preroll_fn is None
        pre = None
        if capture_sr == 16000:
            if preroll_fn is not None:
                pre = preroll_fn()
            elif _preroll_source is not None and settings.wake_preroll:
                pre = _preroll_source(t_call)
        if pre is not None and len(pre) >= block_size:
            pre = np.asarray(pre, dtype=np.float32)
            blocks = [
                pre[i : i + block_size] for i in range(0, len(pre) - block_size + 1, block_size)
            ]
            # Keep the endpointer's model state in step with the audio.
            flags = [
                endpointer.is_speech(b if vad_gain == 1.0 else np.clip(b * vad_gain, -1.0, 1.0))
                for b in blocks
            ]
            first = 0
            if soft:
                first = len(blocks)
                if flags[-1]:  # speaking right now: back up to where it began
                    first = len(blocks) - 1
                    while first > 0 and flags[first - 1]:
                        first -= 1
                    first = max(0, first - 1)  # one block of lead-in
            if first < len(blocks):
                for mono in blocks[first:]:
                    frames.append(np.repeat(mono.reshape(-1, 1), ch, axis=1))
                    if on_block is not None:
                        on_block(mono)
                started = True
                kept = (len(blocks) - first) * block_duration
                logger.info(f"Recording starts with {kept:.1f}s said before the mic opened")
        while True:
            if should_abort and should_abort():
                logger.debug("Recording aborted")
                break
            data, _ = stream.read(block_size)
            mono = data[:, 0] if data.ndim > 1 else data
            # The VAD models see a gain-boosted copy (quiet USB mic); the
            # energy endpointer's threshold is tuned to the raw level.
            vad_view = mono if vad_gain == 1.0 else np.clip(mono * vad_gain, -1.0, 1.0)

            if max_blocks and len(frames) >= max_blocks:
                logger.info(f"Recording hit the {settings.vad_max_utterance_s:.0f}s cap")
                break
            if endpointer.is_speech(vad_view):
                if not started:
                    pending.append(data.copy())
                    if len(pending) < min_onset:
                        continue
                    started = True
                    for blk in pending:
                        frames.append(blk)
                        if on_block is not None:
                            on_block(blk[:, 0] if blk.ndim > 1 else blk)
                    pending = []
                else:
                    frames.append(data.copy())
                    if on_block is not None:
                        on_block(mono)
                silence_s = 0.0
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
                pending = []
                onset_blocks_left -= 1
                if onset_blocks_left <= 0:
                    logger.debug("No speech within onset timeout")
                    return np.array([], dtype=np.float32)
            else:
                pending = []  # an isolated blip, not the start of speech

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
    if peak < settings.vad_blip_peak and duration < settings.vad_blip_max_s:
        logger.info(f"Ignored a {duration:.1f}s blip (peak {peak:.3f}) — room noise, not speech")
        return np.array([], dtype=np.float32)
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


# What the speaker is saying right now, for voice barge-in (oracle/app.py):
# the wake word only interrupts while a clip is playing, and never while
# the radio itself is saying the wake word — the mic hears the speaker
# 15-20 dB above the room and the ReSpeaker Lite's on-chip AEC does not
# remove it (measured 2026-11-23: the radio's own "Librarian." scored up
# to 0.96 on the mic).
_SELF_ECHO_HOLD_S = 1.5
_playing = threading.Event()
_playing_text = ""
_last_text = ""
_last_text_end = 0.0


def speech_playing(grace: float = 0.0) -> bool:
    """True while a clip is going out of the speaker — or, with *grace*,
    if one ended within that many seconds (the gap between two sentences
    of one answer still counts as "talking")."""
    import time

    if _playing.is_set():
        return True
    return grace > 0 and _last_text_end > 0 and time.monotonic() - _last_text_end < grace


# Button pause for speech (oracle/app.py): while the request is set, the
# clip that is playing stops, the radio says "Paused", and play_audio holds
# the speaker until the request clears (the clip then starts over, so the
# answer picks up at the sentence it stopped in) or the caller aborts.
_pause_req = threading.Event()
_pause_cue: tuple[np.ndarray, int] | None = None


def set_pause_cue(audio: np.ndarray | None, sample_rate: int | None = None) -> None:
    """The spoken "Paused." clip (synthesized once by the app)."""
    global _pause_cue
    _pause_cue = None if audio is None else (audio, sample_rate or settings.audio_sample_rate)


def pause_speech() -> None:
    _pause_req.set()


def resume_speech() -> None:
    _pause_req.clear()


def speech_paused() -> bool:
    return _pause_req.is_set()


def _hold_pause(should_abort: AbortCheck) -> None:
    """Say "Paused", then wait for resume_speech() or an abort. The abort
    check is polled the whole time: it is what reads the button."""
    import time

    def done() -> bool:
        return not _pause_req.is_set() or bool(should_abort and should_abort())

    if _pause_cue is not None and not done():
        try:
            cue, sr = _resample_to_playback(*_pause_cue)
            _stream_play(cue, sr, done)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Pause cue failed: {e}")
    while not done():
        time.sleep(0.05)


def speaker_said(word: str) -> bool:
    """True if the clip playing now, or one that ended within the last
    1.5 s, contains *word* (the detector fires a beat after the word)."""
    import time

    w = word.lower()
    if _playing.is_set() and w in _playing_text:
        return True
    return w in _last_text and time.monotonic() - _last_text_end < _SELF_ECHO_HOLD_S


def play_audio(
    audio: np.ndarray,
    sample_rate: int | None = None,
    should_abort: AbortCheck = None,
    text: str = "",
) -> None:
    """Play audio through configured output device (one clip at a time).

    *text* is what the clip says, when the caller knows it."""
    import time

    global _playing_text, _last_text, _last_text_end
    src_sr = sample_rate or settings.audio_sample_rate
    out, dst_sr = _resample_to_playback(audio, src_sr)

    def aborted() -> bool:
        return bool(should_abort and should_abort())

    def interrupted() -> bool:
        return aborted() or _pause_req.is_set()

    with _speaker_lock:
        _playing_text = text.lower()
        _playing.set()
        try:
            while True:
                if should_abort is None:
                    # Nobody is reading the button for this clip (chimes,
                    # book narration): it cannot be paused, or resumed.
                    _stream_play(out, dst_sr, None)
                    break
                if not _pause_req.is_set():
                    _stream_play(out, dst_sr, interrupted)
                if aborted() or not _pause_req.is_set():
                    break
                # Paused mid-clip: hold, then say this clip again from its
                # start. Nothing is "being said" meanwhile, so the wake
                # word is never mistaken for the radio's own voice.
                said, _playing_text = _playing_text, ""
                _hold_pause(should_abort)
                _playing_text = said
                if aborted():
                    break
        finally:
            _playing.clear()
            _last_text, _last_text_end = _playing_text, time.monotonic()
            _playing_text = ""


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
