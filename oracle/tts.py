"""Text-to-speech via Kokoro TTS (ONNX, CPU)."""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from loguru import logger

from config.settings import settings

_SENTENCE_END = re.compile(r"[.!?\n]+\s*")

SAMPLE_RATE = 24000


def split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries for streaming TTS."""
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_EMPHASIS_RE = re.compile(r"(\*{1,3}|_{1,3})(?=\S)(.+?)(?<=\S)\1")
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_BULLET_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.MULTILINE)
_MD_LEFTOVER_RE = re.compile(r"[*_`#~^|>]+")


_FOREIGN_SCRIPT_RE = re.compile(
    "[\u0400-\u04ff\u0530-\u058f\u0590-\u06ff\u0900-\u0dff\u0e00-\u0e7f\u1100-\u11ff"
    "\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]+"
)


def clean_for_speech(text: str) -> str:
    """Strip markdown the LLM likes to emit — *Moby Dick* was being read
    aloud as "asterisk Moby Dick asterisk" (2026-09-28). Emphasis, links,
    headings, bullets and code ticks go; the words stay. Runs of a script
    the English voice cannot say (a stray Chinese token, 2026-09-30) go too."""
    t = _FOREIGN_SCRIPT_RE.sub("", text)
    t = _MD_LINK_RE.sub(r"\1", t)
    t = _MD_HEADING_RE.sub("", t)
    t = _MD_BULLET_RE.sub("", t)
    for _ in range(2):  # nested ***bold italic***
        t = _MD_EMPHASIS_RE.sub(r"\2", t)
    t = _MD_LEFTOVER_RE.sub("", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


# Sentence boundaries that keep the terminal punctuation (prosody).
_SENTENCE_KEEP_RE = re.compile(r"(?<=[.!?])\s+")
_DIGIT_RE = re.compile(r"\d")


def spoken_words(text: str) -> int:
    """How many words this will be *spoken* as, roughly: a token with
    digits is read out ("60,030" → "sixty thousand and thirty", "11.5" →
    "eleven point five"), so it counts as one word per digit plus one.
    Units were sized by whitespace words; the device description's
    number-heavy sentence became a 13 s unit that needed more GPU arena
    than a whole paragraph of prose (2026-09-28)."""
    n = 0
    for tok in text.split():
        d = len(_DIGIT_RE.findall(tok))
        n += 1 + d if d else 1
    return n


_LEADING_PUNCT_RE = re.compile(r"^[^A-Za-z0-9\(\[\"'$]+")


def speech_units(text: str, max_words: int | None = None) -> list[str]:
    """Cut *text* into TTS units: sentence-aligned, at most *max_words*
    each (a long sentence stays whole), leading punctuation dropped and
    unvoiceable fragments skipped. Every spoken line goes through this —
    the GPU sidecar fails on requests longer than ~10 s of speech, and a
    whole paragraph or a long answer was exactly that (2026-09-28)."""
    limit = settings.reading_unit_max_words if max_words is None else max_words
    text = clean_for_speech(text)
    sentences = [_LEADING_PUNCT_RE.sub("", x).strip() for x in _SENTENCE_KEEP_RE.split(text)]
    sentences = [x for x in sentences if re.search(r"[A-Za-z0-9]", x)]
    # A sentence longer than the limit is cut at clause boundaries (, ; :)
    # and, failing that, at the limit: the sidecar fails on ~10 s of speech.
    pieces: list[str] = []
    for sent in sentences:
        pieces.extend(_split_long(sent, limit))
    units: list[str] = []
    cur: list[str] = []
    n = 0
    for sent in pieces:
        w = spoken_words(sent)
        if cur and n + w > limit:
            units.append(" ".join(cur))
            cur, n = [], 0
        cur.append(sent)
        n += w
    if cur:
        units.append(" ".join(cur))
    return units


_CLAUSE_KEEP_RE = re.compile(r"(?<=[,;:—])\s+")


def _split_long(sentence: str, limit: int) -> list[str]:
    if spoken_words(sentence) <= limit:
        return [sentence]
    out: list[str] = []
    cur: list[str] = []
    n = 0
    for clause in _CLAUSE_KEEP_RE.split(sentence):
        w = spoken_words(clause)
        if cur and n + w > limit:
            out.append(" ".join(cur))
            cur, n = [], 0
        if w > limit:  # no punctuation to cut at: hard-split by spoken weight
            run: list[str] = []
            rn = 0
            for word in clause.split():
                ww = spoken_words(word)
                if run and rn + ww > limit:
                    out.append(" ".join(run))
                    run, rn = [], 0
                run.append(word)
                rn += ww
            if run:
                out.append(" ".join(run))
            continue
        cur.append(clause)
        n += w
    if cur:
        out.append(" ".join(cur))
    return out


def say(tts: KokoroTTS, text: str, should_abort=None, prefetch: int | None = None) -> None:
    """Speak *text* as a pipeline of short units: a worker synthesizes
    unit N+1 while unit N plays. Blocking; returns when done or aborted."""
    import queue
    import threading

    from oracle.audio import play_audio

    units = speech_units(text)
    if not units:
        return
    q: queue.Queue = queue.Queue(maxsize=prefetch or settings.reading_prefetch_units)
    stop = threading.Event()
    done = object()  # distinct sentinel: a stubbed synthesize() may return None

    def aborted() -> bool:
        return stop.is_set() or bool(should_abort and should_abort())

    def produce() -> None:
        try:
            for u in units:
                if aborted():
                    return
                try:
                    q.put(tts.synthesize(u))
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"TTS failed on a unit: {e}")
        finally:
            q.put(done)

    worker = threading.Thread(target=produce, name="say-tts", daemon=True)
    worker.start()
    try:
        for i in range(len(units) + 1):
            t0 = time.monotonic()
            audio = q.get()
            waited = time.monotonic() - t0
            if i and waited > 0.5:
                # The speaker sat silent waiting for synthesis: the sidecar
                # is slow (memory pressure, arena rebuild) or the box is
                # thrashing. Audible as a gap mid-sentence.
                logger.warning(f"TTS pipeline stall: waited {waited:.1f}s for unit {i + 1}")
            if audio is done:
                break
            if aborted():
                break
            play_audio(audio, tts.sample_rate, should_abort=should_abort)
    finally:
        stop.set()
        try:
            while q.get_nowait() is not done:
                pass
        except queue.Empty:
            pass
        worker.join(timeout=60)


class KokoroTTS:
    """Kokoro TTS — in-process on the CPU, or via the GPU sidecar.

    With ``settings.tts_backend == "server"`` synthesis is delegated to
    ``oracle.tts_server`` (Kokoro on CUDA in a cp310 venv; see that
    module). If the sidecar is unreachable at load time the in-process CPU
    model is used instead, so the radio still talks.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        voices_path: Path | None = None,
    ):
        self._model_path = model_path or settings.tts_model_path
        self._voices_path = voices_path or settings.tts_voices_path
        self._kokoro = None
        self._server: str | None = None

    def _try_server(self) -> bool:
        if settings.tts_backend != "server":
            return False
        import httpx

        url = settings.tts_server_url.rstrip("/")
        try:
            r = httpx.get(f"{url}/health", timeout=settings.tts_server_connect_timeout)
            r.raise_for_status()
            self._server = url
            logger.info(f"Kokoro TTS via sidecar {url} ({r.text.strip()})")
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"TTS sidecar {url} unavailable ({e}); using in-process CPU Kokoro")
            return False

    def load(self) -> None:
        """Load Kokoro model and voice data (or connect to the sidecar)."""
        if self._kokoro is not None or self._server is not None:
            return
        if self._try_server():
            return
        self._load_local()

    def _load_local(self) -> None:
        """Load the in-process CPU model (idempotent)."""
        if self._kokoro is not None:
            return
        try:
            from kokoro_onnx import Kokoro

            logger.info(
                f"Loading Kokoro model from {self._model_path}, voices from {self._voices_path}"
            )
            self._kokoro = Kokoro(
                str(self._model_path),
                str(self._voices_path),
            )
            logger.info("Kokoro TTS loaded")
        except ImportError:
            logger.error("kokoro-onnx not installed. Install with: pip install kokoro-onnx")
            raise

    def _synth_remote(self, text: str) -> np.ndarray | None:
        """None only when the sidecar itself is gone (connection error);
        a per-unit synthesis failure (HTTP 5xx / X-Error) yields a short
        silence so one bad input never demotes the session to CPU Kokoro."""
        import httpx

        try:
            r = httpx.post(
                f"{self._server}/synth",
                params={"voice": settings.tts_voice, "speed": settings.tts_speed},
                content=text.encode("utf-8"),
                timeout=settings.tts_server_timeout,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"TTS sidecar unreachable ({e}); falling back to in-process Kokoro")
            self._server = None
            return None
        if r.status_code >= 500 or r.headers.get("X-Error"):
            logger.warning(
                f"TTS sidecar could not voice {text[:60]!r}: "
                f"{r.headers.get('X-Error') or r.status_code}; skipping the unit"
            )
            return np.zeros(int(0.3 * SAMPLE_RATE), dtype=np.float32)
        r.raise_for_status()
        return np.frombuffer(r.content, dtype="<f4").astype(np.float32)

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to float32 audio array at 24 kHz."""
        if self._kokoro is None and self._server is None:
            self.load()
        text = clean_for_speech(text) or text

        audio = self._synth_remote(text) if self._server is not None else None
        if audio is None:
            # Remote failed (or not configured): the in-process model must
            # actually be loaded here — load() would happily re-attach to
            # a sidecar that just failed us and leave _kokoro None.
            self._load_local()
            samples, _sr = self._kokoro.create(
                text,
                voice=settings.tts_voice,
                speed=settings.tts_speed,
            )
            audio = samples.astype(np.float32)
        # Peak-normalize: Kokoro synthesizes well below full scale, so
        # speech sounded quiet next to loudness-mastered music on the
        # same sink. 0 disables.
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if settings.tts_peak and peak > 1e-4:
            audio = audio * (settings.tts_peak / peak)
        return audio

    def synthesize_streaming(self, text: str) -> Iterator[np.ndarray]:
        """Yield audio chunks per sentence for low-latency playback."""
        for sentence in split_sentences(text):
            yield self.synthesize(sentence)

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE
