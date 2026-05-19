"""Persona loader sanity tests.

The persona name was rewritten from 'The Oracle' to 'The Librarian' in commit
9c7a2e2; tests now key on stable structural invariants instead of brittle
exact strings, so future persona tweaks won't break them.
"""

from oracle.persona import build_system_prompt, get_greeting, load_persona


def test_load_persona():
    persona = load_persona()
    assert "oracle" in persona
    assert isinstance(persona["oracle"].get("name"), str)
    assert persona["oracle"]["name"]  # non-empty
    assert "greeting" in persona["oracle"]
    assert "{user_name}" in persona["oracle"]["greeting"]


def test_build_system_prompt():
    persona = load_persona()
    prompt = build_system_prompt()
    # The configured persona name must end up in the system prompt.
    assert persona["oracle"]["name"] in prompt
    # The user's name is interpolated into the prompt.
    assert persona["user"]["name"] in prompt
    # Some non-trivial body was rendered (rules + traits expand to a long
    # multi-paragraph prompt).
    assert len(prompt) > 200


def test_get_greeting():
    persona = load_persona()
    greeting = get_greeting()
    assert greeting
    assert len(greeting) > 10
    # The user's name from persona.toml must be interpolated.
    assert persona["user"]["name"] in greeting
    # No leftover unfilled placeholders.
    assert "{user_name}" not in greeting
    assert "{" not in greeting
