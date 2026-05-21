"""Always-on wake word detection via openWakeWord.

Runs continuously on the AEC-cleaned mic source, processing 80ms audio
chunks through a lightweight ONNX model (~7ms/chunk on Jetson CPU).
Fires a callback when the wake word score exceeds the threshold.
"""

from __future__ import annotations

import threading
from typing import Callable

import numpy as np
from loguru import logger

from config.settings import settings

# openWakeWord expects 16 kHz mono int16 or float32, in 1280-sample chunks (80ms)
_CHUNK_SAMPLES = 1280
_SAMPLE_RATE = 16000


class WakeWordDetector:
    """Continuously listens for a wake word on the default input device.

    Call :meth:`start` to begin background detection.  When the wake word is
    detected, ``on_wake`` is called from the detector thread.
    """

    def __init__(
        self,
        model_name: str | None = None,
        threshold: float | None = None,
        on_wake: Callable[[], None] | None = None,
    ) -> None:
        self._model_name = model_name or settings.wakeword_model
        self._threshold = threshold or settings.wakeword_threshold
        self.on_wake = on_wake
        self._model = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._muted = threading.Event()
        self._muted.set()  # not muted by default (set = listening)

    def _load_model(self) -> None:
        from openwakeword.model import Model

        self._model = Model(
            wakeword_models=[self._model_name],
            inference_framework="onnx",
        )
        logger.info(
            f"Wake word detector: {self._model_name!r} "
            f"(threshold={self._threshold}, onnx)"
        )

    def start(self) -> None:
        """Start background wake word detection."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="wakeword", daemon=True,
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
        import sounddevice as sd

        try:
            self._load_model()
        except Exception as e:
            logger.error(f"Wake word model load failed: {e}")
            return

        logger.debug("Wake word listener started")
        try:
            with sd.InputStream(
                samplerate=_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=_CHUNK_SAMPLES,
            ) as stream:
                while not self._stop.is_set():
                    # If muted, drain audio but don't process
                    if not self._muted.is_set():
                        stream.read(_CHUNK_SAMPLES)
                        self._stop.wait(0.01)
                        continue

                    data, _ = stream.read(_CHUNK_SAMPLES)
                    chunk = data.flatten()

                    predictions = self._model.predict(chunk)
                    score = predictions.get(self._model_name, 0.0)

                    if score >= self._threshold:
                        logger.info(
                            f"Wake word detected: {self._model_name} "
                            f"(score={score:.3f})"
                        )
                        self._model.reset()
                        if self.on_wake is not None:
                            self.on_wake()
        except Exception as e:
            if not self._stop.is_set():
                logger.error(f"Wake word loop error: {e}")
        finally:
            logger.debug("Wake word listener stopped")
