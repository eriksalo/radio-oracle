"""Speaker identification: the voiceprint store, and the once-per-session
check-in flow ("Is this Erik?" / "Who am I talking to?") with a fake
embedder and fake speech I/O."""

from __future__ import annotations

import numpy as np
import pytest

from config.settings import settings
from oracle import speaker as sp
from oracle.memory.users import UserStore, normalize_name


def _voice(seed: int, seconds: float = 2.0) -> np.ndarray:
    """A fake recording whose first sample encodes the 'voice'."""
    a = np.zeros(int(seconds * 16000), dtype=np.float32)
    a[0] = seed
    return a


def _fake_embed(audio: np.ndarray) -> np.ndarray:
    """Deterministic embedding per 'voice': unit vectors, distinct voices orthogonal."""
    seed = int(audio[0])
    e = np.zeros(8, dtype=np.float32)
    e[seed % 8] = 1.0
    return e


# --------------------------------------------------------------- store


def test_user_store_enroll_identify(tmp_path):
    us = UserStore(tmp_path / "u.db")
    assert us.known_users() == [settings.default_user]
    assert us.identify(_fake_embed(_voice(1))) == (None, 0.0)
    us.enroll("erik", _fake_embed(_voice(1)))
    us.enroll("sarah", _fake_embed(_voice(2)))
    name, score = us.identify(_fake_embed(_voice(1)))
    assert (name, round(score, 3)) == ("erik", 1.0)
    name, score = us.identify(_fake_embed(_voice(2)))
    assert name == "sarah"
    assert us.identify(_fake_embed(_voice(3)))[1] == 0.0
    assert us.voiceprint_count("erik") == 1
    assert sorted(us.known_users()) == ["erik", "sarah"]


@pytest.mark.parametrize(
    "spoken,expected",
    [
        ("It's Erik.", "erik"),
        ("this is erik", "erik"),
        ("My name is Erik Salo", "erik salo"),
        ("I'm Sarah!", "sarah"),
        ("no", None),
        ("", None),
        ("Hi, I am Bob", "bob"),
    ],
)
def test_normalize_name(spoken, expected):
    assert normalize_name(spoken) == expected


# ---------------------------------------------------------- check-in flow


class _Ctx:
    """The parts of VoiceContext the flow touches."""

    def __init__(self, tmp_path):
        self.speaker_id = sp.SpeakerId(
            model_path=tmp_path / "none.onnx", users=UserStore(tmp_path / "u.db")
        )
        self.speaker_id.embed = _fake_embed  # type: ignore[method-assign]
        self.speaker = sp.SpeakerSession()
        self.stt_fast = object()
        self.users_switched: list[str] = []

        class _CB:
            def __init__(self, outer):
                self.outer = outer
                self.user = settings.default_user

            def set_user(self, name):
                self.user = name
                self.outer.users_switched.append(name)

        self.ctx_builder = _CB(self)


@pytest.fixture
def flow(tmp_path, monkeypatch):
    spoken: list[str] = []
    answers: list[str] = []

    async def fake_speak(vc, text):
        spoken.append(text)

    def fake_listen(stt, **kw):
        return np.zeros(16000, dtype=np.float32), (answers.pop(0) if answers else "")

    monkeypatch.setattr(sp, "_speak", fake_speak)
    monkeypatch.setattr("oracle.stt.listen", fake_listen)
    monkeypatch.setattr(settings, "speaker_enroll_prints", 3)
    return _Ctx(tmp_path), spoken, answers


@pytest.mark.asyncio
async def test_first_ever_user_is_asked_and_enrolled(flow):
    vc, spoken, answers = flow
    answers.append("Yes.")
    user = await sp.check_in(vc, _voice(1))
    assert user == "erik"
    assert spoken == ["Is this Erik?", "Thanks, Erik. I'll remember your voice."]
    assert vc.speaker_id.users.voiceprint_count("erik") == 1
    assert vc.speaker.enroll_pending == 2 and vc.users_switched == ["erik"]
    # The next two utterances silently add prints; the third identifies.
    await sp.check_in(vc, _voice(1))
    await sp.check_in(vc, _voice(1))
    assert vc.speaker_id.users.voiceprint_count("erik") == 3
    assert len(spoken) == 2
    await sp.check_in(vc, _voice(1))
    assert vc.speaker.last_score == pytest.approx(1.0) and len(spoken) == 2


@pytest.mark.asyncio
async def test_known_voice_switches_silently(flow):
    vc, spoken, answers = flow
    vc.speaker_id.users.enroll("erik", _fake_embed(_voice(1)))
    vc.speaker_id.users.enroll("sarah", _fake_embed(_voice(2)))
    assert await sp.check_in(vc, _voice(2)) == "sarah"
    assert vc.users_switched == ["sarah"] and spoken == []
    assert await sp.check_in(vc, _voice(1)) == "erik"
    assert vc.users_switched == ["sarah", "erik"]


@pytest.mark.asyncio
async def test_unknown_voice_says_no_then_introduces(flow):
    vc, spoken, answers = flow
    vc.speaker_id.users.enroll("erik", _fake_embed(_voice(1)))
    answers.append("It's Sarah")
    # Voice 3 is orthogonal to erik (score 0) → below the ask threshold →
    # straight to "Who am I talking to?" (no "Is this Erik?" first)
    user = await sp.check_in(vc, _voice(3))
    assert user == "sarah"
    assert spoken == ["Who am I talking to?", "Nice to meet you, Sarah."]
    assert vc.speaker_id.users.voiceprint_count("sarah") == 1


@pytest.mark.asyncio
async def test_silence_leaves_default_and_asks_only_once(flow):
    vc, spoken, answers = flow
    # nobody enrolled, no answer
    assert await sp.check_in(vc, _voice(5)) == settings.default_user
    assert spoken == ["Is this Erik?"]
    assert await sp.check_in(vc, _voice(5)) == settings.default_user
    assert spoken == ["Is this Erik?"]  # not asked again this session
    assert vc.users_switched == []


@pytest.mark.asyncio
async def test_disabled_or_short_audio_is_noop(flow, monkeypatch):
    vc, spoken, answers = flow
    assert await sp.check_in(vc, _voice(1, seconds=0.3)) == settings.default_user
    monkeypatch.setattr(settings, "speaker_id_enabled", False)
    assert await sp.check_in(vc, _voice(1)) == settings.default_user
    assert spoken == []
