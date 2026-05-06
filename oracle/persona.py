"""Persona system — builds system prompt from persona config."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from loguru import logger

_DEFAULT_PERSONA_PATH = Path("config/persona.toml")
_USER_NAME_OVERRIDE = Path("data/user_name.txt")
_NAME_RE = re.compile(r"^[\w .,'-]{1,40}$")


def load_persona(path: Path | None = None) -> dict:
    """Load persona configuration from TOML file."""
    p = path or _DEFAULT_PERSONA_PATH
    with open(p, "rb") as f:
        data = tomllib.load(f)
    logger.debug(f"Loaded persona from {p}")
    return data


def get_user_name(persona: dict | None = None) -> str:
    """Return the name the assistant should address the user by.

    Order of precedence: override file > persona TOML default > 'Erik'.
    """
    if _USER_NAME_OVERRIDE.exists():
        try:
            override = _USER_NAME_OVERRIDE.read_text(encoding="utf-8").strip()
            if override:
                return override
        except OSError:
            pass
    if persona is None:
        persona = load_persona()
    return persona.get("user", {}).get("name", "Erik")


def set_user_name(name: str) -> str:
    """Persist a new user name override. Returns the saved value."""
    cleaned = name.strip()
    if not cleaned or not _NAME_RE.match(cleaned):
        raise ValueError("name must be 1-40 chars, letters/digits/space/.,'-")
    _USER_NAME_OVERRIDE.parent.mkdir(parents=True, exist_ok=True)
    _USER_NAME_OVERRIDE.write_text(cleaned + "\n", encoding="utf-8")
    logger.info(f"User name set to {cleaned!r}")
    return cleaned


def build_system_prompt(persona: dict | None = None) -> str:
    """Build the system prompt from persona config."""
    if persona is None:
        persona = load_persona()

    oracle = persona["oracle"]
    template = oracle["system_prompt_template"]["template"]

    return template.format(
        name=oracle["name"],
        user_name=get_user_name(persona),
        **oracle["traits"],
        **oracle["rules"],
    )


def get_greeting(persona: dict | None = None) -> str:
    """Get the assistant's greeting message."""
    if persona is None:
        persona = load_persona()
    return persona["oracle"]["greeting"].format(
        user_name=get_user_name(persona),
        name=persona["oracle"]["name"],
    )
