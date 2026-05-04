"""Momentary action button with short/long-press detection.

Press < long_press_threshold = "short" (next track / action)
Press >= long_press_threshold = "long"  (toggle Radio ↔ Librarian mode)

Events are pushed to a thread-safe queue. Falls back to keyboard input
when GPIO is unavailable so the state machine still exercises in dev.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Literal

from loguru import logger

from config.settings import settings

PressKind = Literal["short", "long"]
_DEBOUNCE_S = 0.03
_POLL_S = 0.01


@dataclass(frozen=True)
class ButtonEvent:
    kind: PressKind
    duration: float


class ActionButton:
    """Polled momentary button that emits short/long press events."""

    def __init__(
        self,
        pin: int | None = None,
        long_press_threshold: float | None = None,
    ) -> None:
        self._pin = pin if pin is not None else settings.action_button_pin
        self._long_threshold = (
            long_press_threshold
            if long_press_threshold is not None
            else settings.long_press_threshold
        )
        self._gpio = None
        self.events: Queue[ButtonEvent] = Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._setup()

    def _setup(self) -> None:
        try:
            import Jetson.GPIO as GPIO  # type: ignore[import-not-found]

            self._gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            logger.info(
                f"Action button on GPIO {self._pin} "
                f"(long-press ≥ {self._long_threshold:.2f}s)"
            )
        except (ImportError, RuntimeError) as e:
            logger.warning(f"GPIO unavailable ({e}), button will use keyboard fallback")
            self._gpio = None

    def start(self) -> None:
        """Begin monitoring in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        target = self._gpio_loop if self._gpio is not None else self._keyboard_loop
        self._thread = threading.Thread(target=target, name="button-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def classify(self, duration: float) -> PressKind:
        return "long" if duration >= self._long_threshold else "short"

    def _gpio_loop(self) -> None:
        prev = self._gpio.input(self._pin)
        press_start: float | None = None
        while not self._stop.is_set():
            level = self._gpio.input(self._pin)
            if prev == self._gpio.HIGH and level == self._gpio.LOW:
                press_start = time.monotonic()
                self._stop.wait(_DEBOUNCE_S)
            elif prev == self._gpio.LOW and level == self._gpio.HIGH and press_start is not None:
                duration = time.monotonic() - press_start
                press_start = None
                if duration >= _DEBOUNCE_S:
                    kind = self.classify(duration)
                    logger.debug(f"Button {kind} press ({duration:.2f}s)")
                    self.events.put(ButtonEvent(kind=kind, duration=duration))
            prev = level
            self._stop.wait(_POLL_S)

    def _keyboard_loop(self) -> None:
        # Dev fallback: type 'l' + Enter for long press, just Enter for short.
        while not self._stop.is_set():
            try:
                line = input("[button] Enter=short, 'l' Enter=long, 'q' Enter=quit: ")
            except (EOFError, KeyboardInterrupt):
                return
            if self._stop.is_set():
                return
            text = line.strip().lower()
            if text.startswith("q"):
                return
            kind: PressKind = "long" if text.startswith("l") else "short"
            duration = self._long_threshold if kind == "long" else 0.1
            self.events.put(ButtonEvent(kind=kind, duration=duration))

    def cleanup(self) -> None:
        self.stop()
        if self._gpio is not None:
            self._gpio.cleanup(self._pin)
