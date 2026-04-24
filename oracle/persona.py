"""Persona system — builds system prompt from persona config."""

from __future__ import annotations

import tomllib
from pathlib import Path

from loguru import logger

_DEFAULT_PERSONA_PATH = Path("config/persona.toml")


def load_persona(path: Path | None = None) -> dict:
    """Load persona configuration from TOML file."""
    p = path or _DEFAULT_PERSONA_PATH
    with open(p, "rb") as f:
        data = tomllib.load(f)
    logger.debug(f"Loaded persona from {p}")
    return data


def build_system_prompt(persona: dict | None = None) -> str:
    """Build the system prompt from persona config.

    Args:
        persona: Persona dict (loaded from TOML). If None, loads default.

    Returns:
        Formatted system prompt string
    """
    if persona is None:
        persona = load_persona()

    oracle = persona["oracle"]
    traits = oracle["traits"]
    rules = oracle["rules"]
    template = oracle["system_prompt_template"]["template"]

    return template.format(
        name=oracle["name"],
        **traits,
        **rules,
    )


def get_greeting(persona: dict | None = None) -> str:
    """Get the Oracle's greeting message."""
    if persona is None:
        persona = load_persona()
    return persona["oracle"]["greeting"]
