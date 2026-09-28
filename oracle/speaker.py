"""Who is talking — speaker identification and the "Is this Erik?" flow.

Every voice command already yields a 16 kHz recording; a TitaNet-small
speaker embedding of it (sherpa-onnx, ~76 ms on the Jetson CPU, 40 MB)
is scored against the voiceprints in ``oracle.memory.users``. Measured on
Kokoro voices as stand-ins: same speaker ≥ 0.90, different speakers
≤ 0.41 cosine (CAM++ was content-sensitive and unusable).

Flow, once per session (``SpeakerSession``):
  * match ≥ ``speaker_threshold`` → that user's memory, silently;
  * otherwise, the first time only, the Librarian asks — "Is this Erik?"
    when there is a plausible candidate (score ≥ ``speaker_ask_threshold``,
    or nobody is enrolled yet), else "Who am I talking to?" — and enrols
    the answer's voice: this utterance plus the next few, for a robust
    print. Silence or nonsense leaves the default user in place and the
    question isn't repeated until the next session.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from config.settings import settings
from oracle.memory.users import UserStore, normalize_name

if TYPE_CHECKING:
    from oracle.core import VoiceContext

SAMPLE_RATE = 16000
_YES_RE = re.compile(
    r"^\W*(yes|yeah|yep|yup|sure|correct|right|it is|that'?s (me|right)|i am|indeed|ok(?:ay)?)\b",
    re.IGNORECASE,
)
_NO_RE = re.compile(r"^\W*(no|nope|nah|not|it'?s not|wrong)\b", re.IGNORECASE)


class SpeakerId:
    """TitaNet embeddings + the voiceprint store."""

    def __init__(self, model_path: Path | None = None, users: UserStore | None = None) -> None:
        self._model_path = model_path or settings.speaker_model
        self._extractor = None
        self._users = users

    @property
    def users(self) -> UserStore:
        if self._users is None:
            self._users = UserStore()
        return self._users

    def load(self) -> None:
        if self._extractor is not None:
            return
        import sherpa_onnx

        if not Path(self._model_path).exists():
            raise FileNotFoundError(
                f"Speaker model not found: {self._model_path} — run scripts/download_models.sh"
            )
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(self._model_path), num_threads=2, provider="cpu"
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self.users  # open the store
        logger.info(f"Speaker ID loaded: {self._model_path} (dim {self._extractor.dim})")

    def embed(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        if self._extractor is None:
            self.load()
        x = np.asarray(audio, dtype=np.float32).ravel()
        if sample_rate != SAMPLE_RATE:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(sample_rate, SAMPLE_RATE)
            x = resample_poly(x, SAMPLE_RATE // g, sample_rate // g).astype(np.float32)
        s = self._extractor.create_stream()
        s.accept_waveform(sample_rate=SAMPLE_RATE, waveform=np.ascontiguousarray(x))
        s.input_finished()
        return np.asarray(self._extractor.compute(s), dtype=np.float32)

    def identify(self, audio: np.ndarray) -> tuple[str | None, float]:
        return self.users.identify(self.embed(audio))

    def enroll(self, name: str, audio: np.ndarray, source: str = "") -> int:
        return self.users.enroll(name, self.embed(audio), source=source)


@dataclass
class SpeakerSession:
    """Per-session identification state."""

    user: str = field(default_factory=lambda: settings.default_user)
    asked: bool = False
    enroll_pending: int = 0
    last_score: float = 0.0
    identified: bool = False
    # Set by observe() when the voice is unknown: ask at the next natural
    # moment (after the command completes), not in the middle of it.
    pending_candidate: str | None = None
    pending_audio: np.ndarray | None = None
    pending: bool = False


def _long_enough(audio: np.ndarray) -> bool:
    return audio is not None and len(audio) >= settings.speaker_min_seconds * SAMPLE_RATE


async def _speak(vc: VoiceContext, text: str) -> None:
    from oracle.core import speak_text

    await speak_text(vc, text)


async def _ask(vc: VoiceContext, prompt: str) -> str:
    """Say *prompt*, chime, listen for a short answer. "" on silence/abort."""
    from oracle.stt import listen

    await _speak(vc, prompt)
    if settings.wake_chime:
        from oracle.chime import play_wake_chime

        await asyncio.to_thread(play_wake_chime)
    try:
        _audio, text = await asyncio.to_thread(
            listen, vc.stt_fast, onset_timeout=settings.speaker_answer_timeout
        )
    except (ValueError, OSError) as e:
        logger.warning(f"Mic unavailable for speaker check-in: {e}")
        return ""
    return text.strip()


def apply_user(vc: VoiceContext, name: str, reader=None) -> None:
    """Switch the whole session — memory, journal, bookmarks — to *name*."""
    if vc.speaker is not None:
        vc.speaker.user = name
        vc.speaker.identified = True
    vc.ctx_builder.set_user(name)
    if reader is not None:
        reader.set_user(name)


async def observe(vc: VoiceContext, audio: np.ndarray, reader=None) -> str:
    """Identify the speaker of *audio* silently. A recognised voice switches
    the session; an unknown one is remembered for ``maybe_ask``. Returns
    the user in effect."""
    sp = vc.speaker
    sid = vc.speaker_id
    if sp is None or sid is None or not settings.speaker_id_enabled or not _long_enough(audio):
        return sp.user if sp is not None else settings.default_user

    try:
        if sp.enroll_pending > 0:
            # Building a robust print from the utterances after a "yes".
            await asyncio.to_thread(sid.enroll, sp.user, audio, "follow-on")
            sp.enroll_pending -= 1
            return sp.user
        name, score = await asyncio.to_thread(sid.identify, audio)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Speaker ID failed: {e}")
        return sp.user
    sp.last_score = score
    logger.debug(f"Speaker: best {name!r} score {score:.2f}")

    if name is not None and score >= settings.speaker_threshold:
        if name != sp.user or not sp.identified:
            apply_user(vc, name, reader)
            logger.info(f"Speaker identified: {name} ({score:.2f})")
        sp.pending = False
        return sp.user

    if not sp.asked and not sp.pending:
        nobody_enrolled = sid.users.voiceprint_count(settings.default_user) == 0
        candidate = None
        if name is not None and score >= settings.speaker_ask_threshold:
            candidate = name
        elif nobody_enrolled:
            candidate = settings.default_user
        sp.pending = True
        sp.pending_candidate = candidate
        sp.pending_audio = audio
    return sp.user


async def maybe_ask(vc: VoiceContext, reader=None) -> str:
    """If observe() flagged an unknown voice, ask now — once per session."""
    sp = vc.speaker
    sid = vc.speaker_id
    if sp is None or sid is None or not sp.pending or sp.asked:
        return sp.user if sp is not None else settings.default_user
    sp.asked = True
    sp.pending = False
    audio = sp.pending_audio
    candidate = sp.pending_candidate
    sp.pending_audio = None
    if audio is None:
        return sp.user

    if candidate is not None:
        answer = await _ask(vc, f"Is this {candidate.title()}?")
        logger.info(f"Speaker check-in ({candidate}): {answer!r}")
        if _YES_RE.match(answer):
            await asyncio.to_thread(sid.enroll, candidate, audio, "confirmed")
            sp.enroll_pending = settings.speaker_enroll_prints - 1
            apply_user(vc, candidate, reader)
            await _speak(vc, f"Thanks, {candidate.title()}. I'll remember your voice.")
            return sp.user
        if not _NO_RE.match(answer):
            return sp.user  # silence / unclear: leave things as they are

    answer = await _ask(vc, "Who am I talking to?")
    new_name = normalize_name(answer)
    logger.info(f"Speaker check-in name: {answer!r} -> {new_name!r}")
    if not new_name:
        await _speak(vc, "No problem.")
        return sp.user
    await asyncio.to_thread(sid.enroll, new_name, audio, "introduced")
    sp.enroll_pending = settings.speaker_enroll_prints - 1
    apply_user(vc, new_name, reader)
    await _speak(vc, f"Nice to meet you, {new_name.title()}.")
    return sp.user


async def check_in(vc: VoiceContext, audio: np.ndarray, reader=None, allow_ask: bool = True) -> str:
    """observe() and, when allowed, maybe_ask() right away."""
    user = await observe(vc, audio, reader)
    if allow_ask:
        user = await maybe_ask(vc, reader)
    return user
