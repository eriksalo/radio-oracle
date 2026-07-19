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
