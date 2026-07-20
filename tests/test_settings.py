from config.settings import OracleSettings


def test_default_settings():
    s = OracleSettings()
    assert s.ollama_model == "qwen3:4b-instruct-2507-q4_K_M"
    assert s.mode == "text"
    assert s.audio_sample_rate == 16000
    assert s.rag_top_k == 5


def test_env_override(monkeypatch):
    monkeypatch.setenv("ORACLE_OLLAMA_MODEL", "phi3:mini")
    monkeypatch.setenv("ORACLE_MODE", "voice")
    s = OracleSettings()
    assert s.ollama_model == "phi3:mini"
    assert s.mode == "voice"


def test_create_stt_backend_switch(monkeypatch):
    from config.settings import settings
    from oracle.stt import WhisperSTT, create_stt

    assert isinstance(create_stt(), WhisperSTT)

    monkeypatch.setattr(settings, "stt_backend", "parakeet")
    stt = create_stt()
    from oracle.stt_parakeet import ParakeetSTT

    assert isinstance(stt, ParakeetSTT)
    # unload must be a no-op — the single shared model stays resident.
    stt.unload()


def test_wake_chirp_is_sane():
    from oracle.chime import _SAMPLE_RATE, wake_chirp

    chirp = wake_chirp()
    assert chirp.dtype.name == "float32"
    dur = len(chirp) / _SAMPLE_RATE
    assert 0.2 < dur < 0.6  # short enough not to delay listening
    assert abs(chirp).max() <= 0.35  # modest level → negligible mic leak
    assert wake_chirp() is chirp  # cached, no re-synthesis per wake
