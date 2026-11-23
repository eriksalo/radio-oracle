"""A voice turn never ends in unexplained silence: garbled speech gets one
spoken "say it again", a false wake closes with a tone, and the wake
listener reopens the mic when the USB board comes back."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from oracle import chime, commands, wakeword


class _VC:
    tts = None
    stt_fast = object()
    user = None


@pytest.fixture
def turn(monkeypatch):
    """Scripted mic + classifier; records what was spoken and cued."""
    log: list[str] = []
    script = {"listens": [], "actions": []}

    def fake_listen(stt, **kw):
        log.append(f"listen:{kw.get('onset_timeout')}")
        return script["listens"].pop(0)

    async def fake_classify(vc, text):
        return script["actions"].pop(0), None

    async def fake_observe(vc, audio, reader=None):
        return None

    async def fake_maybe_ask(vc):
        return None

    monkeypatch.setattr(commands, "listen", fake_listen)
    monkeypatch.setattr(commands, "classify", fake_classify)
    monkeypatch.setattr(commands, "_speak", lambda vc, text, abort=None: log.append(f"say:{text}"))
    monkeypatch.setattr(chime, "play_close_cue", lambda: log.append("close-cue"))
    monkeypatch.setattr("oracle.speaker.observe", fake_observe)
    monkeypatch.setattr("oracle.speaker.maybe_ask", fake_maybe_ask)
    monkeypatch.setattr(
        commands, "_do_action", lambda action, *a, **k: commands.DispatchResult("radio")
    )
    return script, log


SPEECH = np.ones(1600, dtype=np.float32)
NOTHING = np.array([], dtype=np.float32)


@pytest.mark.asyncio
async def test_false_wake_times_out_and_closes_with_a_tone(turn):
    script, log = turn
    script["listens"] = [(NOTHING, "")]
    await commands.dispatch_radio_command(None, None, _VC())
    assert log == [f"listen:{commands.settings.wake_onset_timeout_s}", "close-cue"]


@pytest.mark.asyncio
async def test_untranscribable_speech_gets_one_spoken_retry(turn):
    script, log = turn
    script["listens"] = [(SPEECH, ""), (SPEECH, "next song")]
    script["actions"] = ["next"]
    await commands.dispatch_radio_command(None, None, _VC())
    assert sum(e.startswith("listen") for e in log) == 2
    assert "say:Sorry, I didn't catch that. Say it again?" in log
    assert "close-cue" not in log


@pytest.mark.asyncio
async def test_nonsense_twice_ends_with_words_not_silence(turn):
    script, log = turn
    script["listens"] = [(SPEECH, "red in the printing grass"), (SPEECH, "grass printing")]
    script["actions"] = ["none", "none"]
    await commands.dispatch_radio_command(None, None, _VC())
    says = [e for e in log if e.startswith("say:")]
    assert says == [
        "say:Sorry, I didn't catch that. Say it again?",
        "say:I still didn't get that. Call me when you're ready.",
    ]


def test_ack_phrases_never_repeat_back_to_back(monkeypatch):
    played: list[int] = []
    monkeypatch.setattr(commands, "_ack_cache", [0, 1, 2])
    monkeypatch.setattr(commands, "warm_thinking_acks", lambda vc: None)
    monkeypatch.setattr(commands, "play_audio", lambda a, sr, should_abort=None: played.append(a))

    class VC:
        class tts:
            sample_rate = 24000

    for _ in range(40):
        commands._play_thinking_ack(VC())
    assert len(played) == 40
    assert all(a != b for a, b in zip(played, played[1:]))


def test_wake_listener_reopens_the_mic_after_a_failure(monkeypatch):
    import sys
    import types

    monkeypatch.setitem(sys.modules, "sounddevice", types.ModuleType("sounddevice"))
    det = wakeword.WakeWordDetector(model_path="x.onnx")
    monkeypatch.setattr(det, "_load_model", lambda: None)
    calls = {"n": 0}
    recovered = threading.Event()

    def fake_listen(sd, wanted, prev):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("Input/output error")  # board unplugged
        recovered.set()
        det._stop.wait(5)

    monkeypatch.setattr(det, "_listen", fake_listen)
    monkeypatch.setattr(det._stop, "wait", lambda t=None: recovered.wait(0.01) and False)
    t = threading.Thread(target=det._loop, daemon=True)
    t.start()
    assert recovered.wait(3)
    det._stop.set()
