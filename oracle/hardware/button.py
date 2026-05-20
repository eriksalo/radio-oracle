"""Momentary action button with short/long-press detection.

Press < long_press_threshold = "short" (next track / action)
Press >= long_press_threshold = "long"  (toggle Radio ↔ Librarian mode)

Events are pushed to a thread-safe queue. Falls back to keyboard input
when the ADC is unavailable so the state machine still exercises in dev.

Read via ADS1115 (single channel) rather than GPIO — the Tegra234 GPIO INPUT
register has a loopback bug on JP 6.2.x for the pads we'd otherwise use.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Literal

from loguru import logger

from config.settings import settings
from oracle.hardware.switch_adc import make_action_button_switch

PressKind = Literal["short", "long"]
_DEBOUNCE_S = 0.03
_POLL_S = 0.04      # ADS1115 double-read ~10 ms; 40 ms gap ≈ 20 Hz, fine for press timing


@dataclass(frozen=True)
class ButtonEvent:
    kind: PressKind
    duration: float


class ActionButton:
    """Polled momentary button that emits short/long press events.

    Reads the switch state from one ADS1115 channel; ``closed`` (line shorted
    to GND) means pressed.
    """

    def __init__(self, long_press_threshold: float | None = None) -> None:
        self._long_threshold = (
            long_press_threshold
            if long_press_threshold is not None
            else settings.long_press_threshold
        )
        self._switch = make_action_button_switch()
        self.events: Queue[ButtonEvent] = Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if self._switch.available:
            logger.info(
                f"Action button on ADS1115 AIN{self._switch.channel} "
                f"(long-press ≥ {self._long_threshold:.2f}s)"
            )
        else:
            logger.warning(
                f"ADS1115 unavailable ({self._switch.error}); "
                "button will use keyboard fallback"
            )

    def start(self) -> None:
        """Begin monitoring in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        target = self._adc_loop if self._switch.available else self._keyboard_loop
        self._thread = threading.Thread(target=target, name="button-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def classify(self, duration: float) -> PressKind:
        return "long" if duration >= self._long_threshold else "short"

    def _adc_loop(self) -> None:
        prev = self._switch.is_closed() or False  # treat unknown as released
        press_start: float | None = None
        while not self._stop.is_set():
            level = self._switch.is_closed()
            if level is None:
                self._stop.wait(_POLL_S)
                continue
            if not prev and level:
                # released → pressed
                press_start = time.monotonic()
                self._stop.wait(_DEBOUNCE_S)
            elif prev and not level and press_start is not None:
                # pressed → released
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
