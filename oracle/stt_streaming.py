"""Streaming speech-to-text: NVIDIA Nemotron-speech-streaming-en-0.6b
(cache-aware FastConformer transducer) via sherpa-onnx's OnlineRecognizer.

Selected with ORACLE_STT_BACKEND=nemotron-streaming. Same NeMo family and
runtime as the offline Parakeet backend, but it decodes *while* the user
is talking: ``oracle.stt.listen`` feeds each 100 ms capture block into an
open session, so when the endpointer closes the mic the transcript is
final within one model chunk (560 ms of audio) instead of after a
0.2–0.7 s batch decode. The offline ``transcribe(audio)`` surface is kept
for callers that already have a buffer.

Bundle: sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25
(~630 MB resident, CPU) — downloaded by scripts/download_models.sh. It
replaces Parakeet-offline; the two must not both be resident on 8 GB.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from config.settings import settings

SAMPLE_RATE = 16000
# Flush the encoder's look-ahead at the end of an utterance (sherpa-onnx
# examples pad 0.66 s; the 560 ms chunk model needs at least one chunk).
_TAIL_PADDING_S = 1.2  # 0.66 lost trailing syllables ("Moby Dick" → "Moby") on the Jetson
# Silence fed before the first block: the recorder starts feeding at speech
# onset, and the cache-aware encoder dropped the first word without it.
_PRE_ROLL_S = 0.5


class StreamingSession:
    """One utterance: feed blocks as they are captured, then finish()."""

    def __init__(self, recognizer, gain: float) -> None:
        self._rec = recognizer
        self._stream = recognizer.create_stream()
        self._gain = gain
        self._seconds = 0.0
        self._stream.accept_waveform(
            SAMPLE_RATE, np.zeros(int(_PRE_ROLL_S * SAMPLE_RATE), dtype=np.float32)
        )

    def feed(self, block: np.ndarray, sample_rate: int | None = None) -> None:
        sr = sample_rate or settings.audio_capture_sample_rate
        x = np.asarray(block, dtype=np.float32).ravel()
        if sr != SAMPLE_RATE:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(sr, SAMPLE_RATE)
            x = resample_poly(x, SAMPLE_RATE // g, sr // g).astype(np.float32)
        if self._gain != 1.0:
            x = np.clip(x * self._gain, -1.0, 1.0)
        self._stream.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(x))
        self._seconds += len(x) / SAMPLE_RATE
        while self._rec.is_ready(self._stream):
            self._rec.decode_stream(self._stream)

    def partial(self) -> str:
        """Best transcript so far (for early retrieval / logs)."""
        return self._rec.get_result(self._stream).strip()

    def finish(self) -> str:
        tail = np.zeros(int(_TAIL_PADDING_S * SAMPLE_RATE), dtype=np.float32)
        self._stream.accept_waveform(SAMPLE_RATE, tail)
        self._stream.input_finished()
        while self._rec.is_ready(self._stream):
            self._rec.decode_stream(self._stream)
        text = self._rec.get_result(self._stream).strip()
        logger.info(f"STT result: {text!r} ({self._seconds:.1f}s, streaming)")
        if text:
            from oracle.activity import emit

            emit("heard", text=text)
        return text


class StreamingSTT:
    """Nemotron streaming transducer via sherpa-onnx (load/transcribe/unload
    surface compatible with WhisperSTT/ParakeetSTT, plus open_stream())."""

    def __init__(self, model_name: str | None = None) -> None:
        # model_name accepted for factory-signature compatibility; ignored.
        self._model_dir = settings.streaming_stt_model_dir
        self._recognizer = None
        logger.info(
            f"STT backend: nemotron-streaming dir={self._model_dir} "
            f"threads={settings.streaming_stt_num_threads}"
        )

    def load(self) -> None:
        if self._recognizer is not None:
            return
        import sherpa_onnx

        d = self._model_dir
        if not d.is_dir():
            raise FileNotFoundError(
                f"Streaming STT model dir not found: {d} — run scripts/download_models.sh"
            )

        def _pick(stem: str) -> str:
            for name in (f"{stem}.int8.onnx", f"{stem}.onnx"):
                if (d / name).exists():
                    return str(d / name)
            raise FileNotFoundError(f"missing {stem}*.onnx in {d}")

        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"),
            encoder=_pick("encoder"),
            decoder=_pick("decoder"),
            joiner=_pick("joiner"),
            num_threads=settings.streaming_stt_num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            # Endpointing is the VAD/Smart Turn's job (oracle.endpoint).
            enable_endpoint_detection=False,
            provider=settings.streaming_stt_provider,
        )
        logger.info("Nemotron streaming recognizer loaded")

    def unload(self) -> None:
        """No-op by design — one resident model serves both roles."""

    def release(self) -> None:
        self._recognizer = None

    def open_stream(self) -> StreamingSession:
        if self._recognizer is None:
            self.load()
        return StreamingSession(self._recognizer, settings.stt_input_gain)

    def transcribe(self, audio: np.ndarray, sample_rate: int | None = None) -> str:
        """Offline-style transcription of a whole buffer (same model)."""
        session = self.open_stream()
        sr = sample_rate or settings.audio_sample_rate
        # record_until_silence already applied its own peak gain to a
        # finished buffer; don't stack ours on top.
        session._gain = 1.0
        session.feed(audio, sample_rate=sr)
        return session.finish()
