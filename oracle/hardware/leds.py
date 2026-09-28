"""Status RGB LED for the Oracle enclosure.

Single common-anode RGB LED driven by 3 GPIO pins (R/G/B).
Anode tied to 3.3 V; each pin pulls its cathode LOW to light that channel
(LOW = lit, HIGH = off). Color encodes the current mode.

Falls back to log-only output if Jetson.GPIO is unavailable (dev machines).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal

from loguru import logger

from config.settings import settings

Mode = Literal[
    "off",
    "boot",
    "waiting",
    "error",
    "q_listen",
    "q_think",
    "q_speak",
    "music_play",
    "music_listen",
    "music_think",
    "music_speak",
    "book_read",
    "book_listen",
    "book_think",
    "book_speak",
    "book_paused",
]


@dataclass(frozen=True)
class Color:
    r: bool
    g: bool
    b: bool


# Mode → channel intent (True = that channel lit). Polarity to GPIO pins
# is handled in ``_write`` for a common-anode LED.
# Colour = what you're dealing with: blue = questions/conversation,
# green = music, purple = books; white = idle, amber = booting, red = error.
# Pattern = what it's doing: solid = output (speaking / playing / reading),
# fast blink = thinking, slow blink = listening to you (mic open),
# very slow = idle/paused. (Erik's scheme, 2026-09-28.)
_BLUE = Color(False, False, True)
_GREEN = Color(False, True, False)
_PURPLE = Color(True, False, True)
_WHITE = Color(True, True, True)
_AMBER = Color(True, True, False)
_RED = Color(True, False, False)

MODE_COLORS: dict[str, Color] = {
    "off": Color(False, False, False),
    "boot": _AMBER,
    "waiting": _WHITE,
    "error": _RED,
    "q_listen": _BLUE,
    "q_think": _BLUE,
    "q_speak": _BLUE,
    "music_play": _GREEN,
    "music_listen": _GREEN,
    "music_think": _GREEN,
    "music_speak": _GREEN,
    "book_read": _PURPLE,
    "book_listen": _PURPLE,
    "book_think": _PURPLE,
    "book_speak": _PURPLE,
    "book_paused": _PURPLE,
}

# Blink full period (seconds) per mode; absent = solid.
_FAST = 0.3  # thinking
_SLOW = 1.0  # listening
_IDLE = 2.0  # waiting / paused / booting
_BLINK_PERIOD_S: dict[str, float] = {
    "boot": _IDLE,
    "waiting": _IDLE,
    "error": 0.5,
    "q_listen": _SLOW,
    "q_think": _FAST,
    "music_listen": _SLOW,
    "music_think": _FAST,
    "book_listen": _SLOW,
    "book_think": _FAST,
    "book_paused": _IDLE,
}


class StatusLEDs:
    """Drive a single common-anode RGB LED, encoding mode as color."""

    def __init__(self) -> None:
        self._pins: dict[str, int] = {
            "r": settings.led_red_pin,
            "g": settings.led_green_pin,
            "b": settings.led_blue_pin,
        }
        self._gpio = None
        self._mode: Mode = "off"
        self._lock = threading.Lock()
        self._blink_stop: threading.Event | None = None
        self._blink_thread: threading.Thread | None = None
        self._setup()

    def _setup(self) -> None:
        try:
            import Jetson.GPIO as GPIO  # type: ignore[import-not-found]

            self._gpio = GPIO
            GPIO.setmode(GPIO.BOARD)
            # Common-anode: park each cathode HIGH so the LED is dark at boot.
            for pin in self._pins.values():
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.HIGH)
            logger.info(f"RGB LED on R={self._pins['r']} G={self._pins['g']} B={self._pins['b']}")
        except (ImportError, RuntimeError) as e:
            logger.warning(f"GPIO unavailable ({e}), LED will log only")
            self._gpio = None

    def _write(self, color: Color) -> None:
        if self._gpio is None:
            return
        # Common-anode: lit = pin LOW, off = pin HIGH.
        on, off = self._gpio.LOW, self._gpio.HIGH
        self._gpio.output(self._pins["r"], on if color.r else off)
        self._gpio.output(self._pins["g"], on if color.g else off)
        self._gpio.output(self._pins["b"], on if color.b else off)

    def _stop_blink(self) -> None:
        if self._blink_stop is not None:
            self._blink_stop.set()
        if self._blink_thread is not None and self._blink_thread.is_alive():
            self._blink_thread.join(timeout=1.0)
        self._blink_stop = None
        self._blink_thread = None

    def _start_blink(self, color: Color, period: float) -> None:
        stop = threading.Event()

        def _loop() -> None:
            on = True
            while not stop.is_set():
                self._write(color if on else MODE_COLORS["off"])
                on = not on
                stop.wait(period / 2)
            self._write(MODE_COLORS["off"])

        thread = threading.Thread(target=_loop, name="led-blink", daemon=True)
        self._blink_stop = stop
        self._blink_thread = thread
        thread.start()

    def set_mode(self, mode: Mode) -> None:
        """Set the LED to the color for the given mode."""
        with self._lock:
            if mode == self._mode:
                return
            logger.debug(f"LED: {self._mode} -> {mode}")
            from oracle.activity import emit

            emit("phase", phase=mode)
            self._stop_blink()
            color = MODE_COLORS.get(mode, MODE_COLORS["off"])
            blink_period = _BLINK_PERIOD_S.get(mode)
            if blink_period is not None:
                self._start_blink(color, blink_period)
            else:
                self._write(color)
            self._mode = mode

    def set_rgb(self, r: bool, g: bool, b: bool) -> Color:
        """Drive the channels directly (diag/test path).

        Bypasses the mode state machine — useful for the diagnostic UI.
        Subsequent ``set_mode`` calls will overwrite this. Returns the
        Color that was actually written.
        """
        color = Color(bool(r), bool(g), bool(b))
        with self._lock:
            self._stop_blink()
            self._write(color)
            self._mode = "off"  # force the next set_mode to re-apply
        return color

    @property
    def mode(self) -> Mode:
        return self._mode

    def all_off(self) -> None:
        self.set_mode("off")

    def cleanup(self) -> None:
        self._stop_blink()
        self._write(MODE_COLORS["off"])
        if self._gpio is not None:
            self._gpio.cleanup(list(self._pins.values()))
