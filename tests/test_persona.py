from oracle.persona import build_system_prompt, get_greeting, load_persona


def test_load_persona():
    persona = load_persona()
    assert persona["oracle"]["name"] == "The Oracle"
    assert "greeting" in persona["oracle"]


def test_build_system_prompt():
    prompt = build_system_prompt()
    assert "The Oracle" in prompt
    assert "mid-century" in prompt
    assert "archives" in prompt.lower()


def test_get_greeting():
    greeting = get_greeting()
    assert "Oracle" in greeting
    assert len(greeting) > 10
