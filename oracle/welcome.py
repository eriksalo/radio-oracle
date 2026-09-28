"""Power-on routine: chime, listen, introduce yourself, offer the menu.

Instead of going straight to music when the radio is switched on:

  1. play the chime and listen for ``welcome_first_wait`` seconds;
  2. nothing → "This is the Librarian. Can I help you with something?",
     chime, listen ``welcome_second_wait`` seconds;
  3. nothing → the four options (ask a question / play or explore the
     music / read or explore the books / about this device), chime,
     listen ``welcome_third_wait`` seconds;
  4. nothing → say nothing more; the LED blinks blue slowly and the radio
     waits for the wake word or the button. Music only starts when asked.

Anything heard at any step goes through the normal command dispatcher.
The flow takes its side effects (speak, listen, dispatch) as callables so
it can be tested without hardware.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import numpy as np
from loguru import logger

from config.settings import settings

GREETING = "This is the Librarian. Can I help you with something?"
OPTIONS = (
    "You can ask me a question, play or explore the music, "
    "read or explore the books, or ask me about this device."
)


@dataclass
class WelcomeOutcome:
    heard: str | None = None  # what the user said, if anything
    dispatched: object | None = None  # the DispatchResult, if a command ran
    steps: int = 0  # how many prompts were spoken (0-2)


async def run_welcome(
    speak: Callable[[str], Awaitable[None]],
    chime: Callable[[], Awaitable[None]],
    listen: Callable[[float], Awaitable[tuple[np.ndarray, str]]],
    dispatch: Callable[[str, np.ndarray], Awaitable[object]],
    should_abort: Callable[[], bool] | None = None,
) -> WelcomeOutcome:
    out = WelcomeOutcome()

    def aborted() -> bool:
        return bool(should_abort and should_abort())

    waits = (
        settings.welcome_first_wait,
        settings.welcome_second_wait,
        settings.welcome_third_wait,
    )
    prompts = (None, GREETING, OPTIONS)
    for step, (prompt, wait) in enumerate(zip(prompts, waits, strict=True)):
        if aborted():
            return out
        if prompt:
            await speak(prompt)
            out.steps = step
        await chime()
        try:
            audio, text = await listen(wait)
        except (ValueError, OSError) as e:
            logger.warning(f"Mic unavailable during welcome: {e}")
            return out
        text = (text or "").strip()
        if text:
            logger.info(f"Welcome: heard {text!r} at step {step}")
            out.heard = text
            out.dispatched = await dispatch(text, audio)
            return out
    return out
