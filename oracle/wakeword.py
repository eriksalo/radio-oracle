"""Always-on wake word detection via openWakeWord.

Runs continuously on the AEC-cleaned mic source, processing 80ms audio
chunks through a lightweight ONNX model (~7ms/chunk on Jetson CPU).
Fires a callback when the wake word score exceeds the threshold.

Supports both pretrained models (by name) and custom-trained models
(by .onnx file path).
"""

from __future__ import annotations

import collections
import threading
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from config.settings import settings

# openWakeWord expects 16 kHz mono int16 or float32, in 1280-sample chunks (80ms)
_CHUNK_SAMPLES = 1280
_SAMPLE_RATE = 16000
_RING_CHUNKS = 100  # 8 s


class WakeWordDetector:
    """Continuously listens for a wake word on the default input device.

    Call :meth:`start` to begin background detection.  When the wake word is
    detected, ``on_wake`` is called from the detector thread.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        threshold: float | None = None,
        on_wake: Callable[[], None] | None = None,
    ) -> None:
        self._model_path = str(model_path) if model_path else settings.wakeword_model
        self._threshold = threshold or settings.wakeword_threshold
        self.on_wake = on_wake
        self._model = None
        self._model_key: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._muted = threading.Event()
        self._muted.set()  # not muted by default (set = listening)
        # The last few seconds of mic audio, so what is said straight after
        # the wake word is not lost while the app reacts (see
        # audio_since_wake). (sequence number, int16 chunk).
        self._ring: collections.deque = collections.deque(maxlen=_RING_CHUNKS)
        self._ring_lock = threading.Lock()
        self._seq = 0
        self._wake_seq: int | None = None

    def _load_model(self) -> None:
        from openwakeword.model import Model

        path = self._model_path

        # If it looks like a file path, load custom model
        if path.endswith(".onnx") or "/" in path:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"Wake word model not found: {p}")
            self._model = Model(
                wakeword_models=[str(p)],
                inference_framework="onnx",
            )
            # The prediction key is the model filename stem
            self._model_key = p.stem
        else:
            # Pretrained model by name
            self._model = Model(
                wakeword_models=[path],
                inference_framework="onnx",
            )
            self._model_key = path

        logger.info(
            f"Wake word detector: model={self._model_key!r} (threshold={self._threshold}, onnx)"
        )

    def start(self) -> None:
        """Start background wake word detection."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="wakeword",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop detection and release resources."""
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._model = None

    def mute(self) -> None:
        """Temporarily suspend detection (e.g. during own TTS playback)."""
        self._muted.clear()

    def unmute(self) -> None:
        """Resume detection after mute."""
        if self._model is not None:
            self._model.reset()
        self._muted.set()

    def _loop(self) -> None:
        import os

        import sounddevice as sd

        try:
            self._load_model()
        except Exception as e:
            logger.error(f"Wake word model load failed: {e}")
            return

        # Which source feeds wake detection (ORACLE_WAKEWORD_SOURCE):
        #   raw     — the ReSpeaker directly, bypassing echo-cancel. Right
        #             when AEC has no playback reference (music routed
        #             direct): its NS-only processing mangled the custom
        #             model's spectral features.
        #   aec     — aec_source. Right for the "mono gambit" topology
        #             where music/TTS play through aec_sink: the canceller
        #             subtracts the music, so the wake word works DURING
        #             playback.
        #   default — whatever the pulse default source is.
        # Set PULSE_SOURCE around stream open so this stream alone is
        # affected; STT recordings opened later still use the pulse default.
        src_map = {
            "raw": "alsa_input.usb-Seeed_Studio_ReSpeaker_Lite_0000000001-00.analog-stereo",
            "aec": "aec_source",
        }
        wanted_src = src_map.get(settings.wakeword_source)
        prev_src = os.environ.get("PULSE_SOURCE")
        logger.debug("Wake word listener started")
        # The mic and the speaker are one USB device: when the board resets
        # or is re-plugged the stream dies. Reopen it for as long as the
        # detector is wanted — before 2026-11-23 the thread logged the error
        # and exited, and the radio stayed deaf until a restart.
        failures = 0
        while not self._stop.is_set():
            try:
                self._listen(sd, wanted_src, prev_src)
                failures = 0
            except Exception as e:
                if self._stop.is_set():
                    break
                failures += 1
                if failures == 1 or failures % 20 == 0:
                    logger.error(f"Wake word loop error (retrying every 3 s): {e}")
                self._stop.wait(3.0)
        logger.debug("Wake word listener stopped")

    def _remember(self, chunk) -> None:
        import time

        with self._ring_lock:
            self._seq += 1
            self._ring.append((self._seq, time.monotonic(), chunk.copy()))

    def audio_since_wake(self):
        """Mic audio captured after the most recent wake-word detection, as
        float32 mono at 16 kHz (empty if there was none or it has aged out
        of the 8 s buffer)."""
        import numpy as np

        with self._ring_lock:
            mark = self._wake_seq
            chunks = [c for s, _t, c in self._ring if mark is not None and s > mark]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32) / 32768.0

    def audio_since_time(self, t: float):
        """Mic audio that arrived after monotonic time *t* (float32 mono,
        16 kHz) — what was said while a recording stream was still opening."""
        import numpy as np

        with self._ring_lock:
            chunks = [c for _s, ts, c in self._ring if ts > t]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32) / 32768.0

    def _listen(self, sd, wanted_src: str | None, prev_src: str | None) -> None:
        """Open the mic and run detection until stopped or the stream fails."""
        import os

        if wanted_src is not None:
            os.environ["PULSE_SOURCE"] = wanted_src
        try:
            # openWakeWord expects int16-range samples (not normalized float).
            # Read as int16 directly so model.predict() gets the scale it expects.
            stream = sd.InputStream(
                samplerate=_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=_CHUNK_SAMPLES,
            )
        finally:
            # Restore env as soon as the stream has connected (or failed
            # to) so other threads opening pulse streams aren't affected.
            if wanted_src is not None:
                if prev_src is None:
                    os.environ.pop("PULSE_SOURCE", None)
                else:
                    os.environ["PULSE_SOURCE"] = prev_src
        with stream:
            if self._model is not None:
                self._model.reset()
            while not self._stop.is_set():
                # If muted, drain audio but don't process
                if not self._muted.is_set():
                    data, _ = stream.read(_CHUNK_SAMPLES)
                    self._remember(data.flatten())
                    self._stop.wait(0.01)
                    continue

                data, _ = stream.read(_CHUNK_SAMPLES)
                chunk = data.flatten()
                self._remember(chunk)

                predictions = self._model.predict(chunk)
                score = predictions.get(self._model_key, 0.0)

                if score >= self._threshold:
                    logger.info(f"Wake word detected: {self._model_key} (score={score:.3f})")
                    self._model.reset()
                    self._wake_seq = self._seq
                    if self.on_wake is not None:
                        self.on_wake()
