"""Speech-to-text via NVIDIA Parakeet-TDT-0.6B (sherpa-onnx runtime).

Drop-in alternative to WhisperSTT (same load/transcribe/unload surface),
selected with ORACLE_STT_BACKEND=parakeet. One model serves both the radio
dispatcher and the librarian turn — better WER than whisper small.en at a
fraction of the latency, so the two-model split (and the unload/reload
dance around LLM calls) disappears.

Model bundle: sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8 (~700MB resident)
downloaded by scripts/download_models.sh. sherpa-onnx ships documented
CUDA builds for JetPack 6.2 — set ORACLE_PARAKEET_PROVIDER=cuda there;
the default is CPU so dev machines work out of the box.

unload() is a deliberate no-op: this single model replaces both whisper
models and is sized to stay resident alongside the pinned LLM, so the
callers' load/unload choreography (written for whisper's memory juggling)
must not evict it.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from config.settings import settings


class ParakeetSTT:
    """Parakeet-TDT offline transducer via sherpa-onnx."""

    def __init__(self, model_name: str | None = None) -> None:
        # model_name is accepted (and ignored) for WhisperSTT signature
        # compatibility — there is only one Parakeet model.
        self._model_dir = settings.parakeet_model_dir
        self._recognizer = None
        logger.info(
            f"STT backend: parakeet dir={self._model_dir} provider={settings.parakeet_provider}"
        )

    def load(self) -> None:
        """Load the recognizer (idempotent)."""
        if self._recognizer is not None:
            return
        import sherpa_onnx

        d = self._model_dir
        if not d.is_dir():
            raise FileNotFoundError(
                f"Parakeet model dir not found: {d} — run scripts/download_models.sh"
            )

        def _pick(stem: str) -> str:
            # Prefer the int8 file; fall back to fp32 if the bundle has it.
            for name in (f"{stem}.int8.onnx", f"{stem}.onnx"):
                if (d / name).exists():
                    return str(d / name)
            raise FileNotFoundError(f"missing {stem}*.onnx in {d}")

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=_pick("encoder"),
            decoder=_pick("decoder"),
            joiner=_pick("joiner"),
            tokens=str(d / "tokens.txt"),
            num_threads=settings.parakeet_num_threads,
            sample_rate=16000,
            feature_dim=80,
            model_type="nemo_transducer",
            provider=settings.parakeet_provider,
        )
        logger.info("Parakeet recognizer loaded")

    def unload(self) -> None:
        """No-op by design — see module docstring."""

    def release(self) -> None:
        """Actually free the recognizer (shutdown only)."""
        self._recognizer = None

    def transcribe(self, audio: np.ndarray, sample_rate: int | None = None) -> str:
        """Transcribe float32 audio to text."""
        sr = sample_rate or settings.audio_sample_rate
        if sr != 16000:
            from scipy.signal import resample_poly

            audio = resample_poly(audio, 16000, sr).astype(np.float32)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        audio = np.ascontiguousarray(audio)

        if self._recognizer is None:
            self.load()

        logger.debug(f"Transcribing {len(audio) / 16000:.1f}s of audio (parakeet)")
        stream = self._recognizer.create_stream()
        stream.accept_waveform(16000, audio)
        self._recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        logger.info(f"STT result: {text!r}")
        return text
