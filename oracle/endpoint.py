"""End-of-utterance detection: Silero VAD + Pipecat Smart Turn.

The energy VAD in ``oracle.audio`` ends a turn after a fixed run of quiet
blocks (0.9 s) — dead time on every turn, and still cuts people off who
pause mid-thought. This module offers two better endpointers behind
``settings.vad_backend``:

* ``silero`` — Silero VAD (via sherpa-onnx) decides speech/non-speech per
  block; the turn ends after ``vad_silence_min`` seconds of non-speech.
* ``silero+smartturn`` — as above, but at each silence checkpoint the last
  8 s of audio go through Smart Turn v3.2 (Pipecat, BSD-2, ~8 MB int8), an
  acoustic model that says whether the speaker sounds *finished*. A
  trailing "…and" or a rising pause keeps the mic open (up to
  ``vad_silence_max``); a finished sentence ends the turn at the first
  checkpoint.

Smart Turn expects Whisper-style features: 80-bin log-mel over the last
8 s at 16 kHz, computed exactly as transformers' WhisperFeatureExtractor
does (zero-mean/unit-var waveform, reflect-padded 400-pt STFT, hop 160,
slaney mel, log10, clamp to max-8, (x+4)/4). Re-implemented in numpy so
the runtime needs neither torch nor transformers; the port is checked
against a transformers-generated reference tensor in tests.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from loguru import logger

from config.settings import settings

SAMPLE_RATE = 16000
_N_FFT = 400
_HOP = 160
_N_SAMPLES = 8 * SAMPLE_RATE  # Smart Turn window
_N_FRAMES = _N_SAMPLES // _HOP  # 800


# ----------------------------------------------------------------- features


def _hz_to_mel_slaney(f: np.ndarray) -> np.ndarray:
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * f / 200.0
    hi = f >= 1000.0
    mels = np.where(hi, 15.0 + np.log(np.maximum(f, 1e-9) / 1000.0) * logstep, mels)
    return mels


def _mel_to_hz_slaney(m: np.ndarray) -> np.ndarray:
    logstep = np.log(6.4) / 27.0
    f = 200.0 * m / 3.0
    hi = m >= 15.0
    return np.where(hi, 1000.0 * np.exp((m - 15.0) * logstep), f)


def whisper_mel_filters(n_mels: int = 80) -> np.ndarray:
    """Whisper's (201, n_mels) slaney-normalised mel filterbank — a numpy
    port of transformers.audio_utils.mel_filter_bank(norm="slaney",
    mel_scale="slaney") so no model file needs shipping."""
    fft_freqs = np.linspace(0.0, SAMPLE_RATE / 2, _N_FFT // 2 + 1)
    mel_pts = np.linspace(
        _hz_to_mel_slaney(np.array(0.0)), _hz_to_mel_slaney(np.array(8000.0)), n_mels + 2
    )
    filter_freqs = _mel_to_hz_slaney(mel_pts)
    diff = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    fb = np.maximum(0.0, np.minimum(down, up))
    enorm = 2.0 / (filter_freqs[2 : n_mels + 2] - filter_freqs[:n_mels])
    return (fb * enorm[None, :]).astype(np.float32)


def whisper_log_mel(audio: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """Whisper log-mel features for Smart Turn: (80, 800) float32.

    *audio* is float32 at 16 kHz; the last 8 s are kept (shorter input is
    right-padded with zeros after normalisation, as the reference does).
    """
    x = np.asarray(audio, dtype=np.float32).ravel()
    if len(x) > _N_SAMPLES:
        x = x[-_N_SAMPLES:]
    # zero_mean_unit_var_norm over the real samples only, pad with 0.
    if len(x):
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    x = np.pad(x, (0, _N_SAMPLES - len(x)))

    window = np.hanning(_N_FFT + 1)[:-1].astype(np.float32)  # periodic Hann
    padded = np.pad(x, (_N_FFT // 2, _N_FFT // 2), mode="reflect")
    n_frames = 1 + (len(padded) - _N_FFT) // _HOP  # 801 (center=True)
    idx = np.arange(_N_FFT)[None, :] + _HOP * np.arange(n_frames)[:, None]
    frames = padded[idx] * window
    power = np.abs(np.fft.rfft(frames, n=_N_FFT, axis=1)) ** 2  # (T, 201)
    mel = np.maximum(power @ mel_filters, 1e-10)  # (T, 80)
    log_spec = np.log10(mel).T[:, :-1]  # drop the last frame → (80, 800)
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


# ---------------------------------------------------------------- Smart Turn


class SmartTurn:
    """Pipecat Smart Turn v3 (ONNX, CPU): P(speaker is done) for the last 8 s."""

    def __init__(self, model_path: Path | None = None):
        self._model_path = model_path or settings.smart_turn_model
        self._sess = None
        self._mel: np.ndarray | None = None

    def load(self) -> None:
        if self._sess is not None:
            return
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.log_severity_level = 3
        self._sess = ort.InferenceSession(
            str(self._model_path), so, providers=["CPUExecutionProvider"]
        )
        self._mel = whisper_mel_filters()
        logger.info(f"Smart Turn loaded: {self._model_path}")

    def probability(self, audio: np.ndarray) -> float:
        if self._sess is None:
            self.load()
        assert self._sess is not None and self._mel is not None
        feats = whisper_log_mel(audio, self._mel)[None]
        out = self._sess.run(None, {self._sess.get_inputs()[0].name: feats})[0]
        return float(out.ravel()[0])

    def is_complete(self, audio: np.ndarray, threshold: float | None = None) -> bool:
        thr = settings.smart_turn_threshold if threshold is None else threshold
        return self.probability(audio) > thr


# --------------------------------------------------------------- Silero VAD


class SileroVad:
    """Per-block speech/non-speech via sherpa-onnx's Silero VAD."""

    def __init__(self, model_path: Path | None = None, threshold: float | None = None):
        self._model_path = model_path or settings.silero_vad_model
        self._threshold = settings.silero_vad_threshold if threshold is None else threshold
        self._vad = None

    def load(self) -> None:
        if self._vad is not None:
            return
        import sherpa_onnx

        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(self._model_path)
        cfg.silero_vad.threshold = self._threshold
        # We do our own silence timing; make the detector flip quickly.
        cfg.silero_vad.min_silence_duration = 0.1
        cfg.silero_vad.min_speech_duration = 0.1
        cfg.silero_vad.window_size = 512
        cfg.sample_rate = SAMPLE_RATE
        cfg.num_threads = 1
        self._vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)
        logger.info(f"Silero VAD loaded: {self._model_path}")

    def reset(self) -> None:
        if self._vad is not None:
            self._vad.reset()

    def is_speech(self, block: np.ndarray) -> bool:
        if self._vad is None:
            self.load()
        assert self._vad is not None
        self._vad.accept_waveform(np.ascontiguousarray(block, dtype=np.float32).ravel())
        # Drain finished segments so the internal buffer doesn't grow.
        while not self._vad.empty():
            self._vad.pop()
        return bool(self._vad.is_speech_detected())


# --------------------------------------------------------------- endpointers


class Endpointer:
    """Decides, block by block, whether the user has started and finished.

    ``is_speech(block)`` classifies one capture block; ``turn_complete
    (audio_so_far, silence_s)`` is asked after each non-speech block once
    speech has started and returns True to end the recording.
    """

    def start(self) -> None:  # noqa: B027 — hook, default no-op
        pass

    def is_speech(self, block: np.ndarray) -> bool:
        raise NotImplementedError

    def turn_complete(self, audio: np.ndarray, silence_s: float) -> bool:
        raise NotImplementedError


class EnergyEndpointer(Endpointer):
    """The legacy behaviour: RMS threshold, fixed trailing-silence window."""

    def __init__(self, threshold: float, max_silence: float):
        self._threshold = threshold
        self._max_silence = max_silence

    def is_speech(self, block: np.ndarray) -> bool:
        return float(np.sqrt(np.mean(block**2))) > self._threshold

    def turn_complete(self, audio: np.ndarray, silence_s: float) -> bool:
        return silence_s >= self._max_silence


class VadEndpointer(Endpointer):
    """Silero VAD, optionally confirmed by Smart Turn at each checkpoint.

    Without Smart Turn the turn ends after ``min_silence`` seconds of
    non-speech. With it, from ``min_silence`` on, Smart Turn is consulted
    every ``interval`` seconds of continued silence; "complete" ends the
    turn, "incomplete" keeps listening until ``max_silence``.
    """

    def __init__(
        self,
        vad: SileroVad | Callable[[np.ndarray], bool],
        smart_turn: SmartTurn | Callable[[np.ndarray], bool] | None,
        min_silence: float,
        max_silence: float,
        interval: float = 0.2,
    ):
        self._vad = vad
        self._smart_turn = smart_turn
        self._min = min_silence
        self._max = max_silence
        self._interval = interval
        self._next_check = min_silence
        self.last_decision = "none"  # for logs: complete | incomplete | cap | vad

    def start(self) -> None:
        self._next_check = self._min
        self.last_decision = "none"
        if isinstance(self._vad, SileroVad):
            self._vad.reset()

    def is_speech(self, block: np.ndarray) -> bool:
        if isinstance(self._vad, SileroVad):
            return self._vad.is_speech(block)
        return self._vad(block)

    def turn_complete(self, audio: np.ndarray, silence_s: float) -> bool:
        if silence_s < self._min:
            return False
        if self._smart_turn is None:
            self.last_decision = "vad"
            return True
        if silence_s >= self._max:
            self.last_decision = "cap"
            return True
        if silence_s + 1e-6 < self._next_check:
            return False
        self._next_check = silence_s + self._interval
        t = time.monotonic()
        if isinstance(self._smart_turn, SmartTurn):
            done = self._smart_turn.is_complete(audio)
        else:
            done = self._smart_turn(audio)
        logger.debug(
            f"Smart Turn @ {silence_s:.2f}s silence: "
            f"{'complete' if done else 'incomplete'} ({(time.monotonic() - t) * 1000:.0f} ms)"
        )
        self.last_decision = "complete" if done else "incomplete"
        return done


# ------------------------------------------------------------------ factory

_vad_singleton: SileroVad | None = None
_smart_turn_singleton: SmartTurn | None = None


def build_endpointer(
    energy_threshold: float,
    max_silence: float,
    backend: str | None = None,
) -> Endpointer:
    """The endpointer for ``settings.vad_backend`` (models are shared and
    loaded once). *max_silence* is the legacy fixed window — for the VAD
    backends it becomes the hard cap only when longer than
    ``settings.vad_silence_max``."""
    global _vad_singleton, _smart_turn_singleton
    backend = backend or settings.vad_backend
    if backend == "energy":
        return EnergyEndpointer(energy_threshold, max_silence)
    if _vad_singleton is None:
        _vad_singleton = SileroVad()
    smart: SmartTurn | None = None
    if backend == "silero+smartturn":
        if _smart_turn_singleton is None:
            _smart_turn_singleton = SmartTurn()
        smart = _smart_turn_singleton
    return VadEndpointer(
        _vad_singleton,
        smart,
        min_silence=settings.vad_silence_min,
        max_silence=max(settings.vad_silence_max, max_silence if smart is None else 0.0),
        interval=settings.smart_turn_interval,
    )


def warm() -> None:
    """Load the configured endpoint models (call at boot, off the loop)."""
    if settings.vad_backend == "energy":
        return
    try:
        ep = build_endpointer(settings.vad_energy_threshold, settings.vad_silence_duration)
        if isinstance(ep, VadEndpointer):
            if isinstance(ep._vad, SileroVad):
                ep._vad.load()
            if isinstance(ep._smart_turn, SmartTurn):
                ep._smart_turn.load()
                ep._smart_turn.probability(np.zeros(SAMPLE_RATE, dtype=np.float32))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Endpoint model warm-up failed ({settings.vad_backend}): {e}")
