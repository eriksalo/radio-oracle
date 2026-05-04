"""SPST toggle (virtual power) switch monitor.

Closed (LOW with pull-up) = device on / radio active.
Open  (HIGH)              = device standby (LED off, no audio).

Polls the GPIO in a background thread; listeners are invoked on edge.
"""

from __future__ import annotations

import threading
from typing import Callable

from loguru import logger

from config.settings import settings

_DEBOUNCE_S = 0.05


class PowerSwitch:
    """Monitor a SPST toggle switch wired between GPIO and GND."""

    def __init__(self, pin: int | None = None, poll_interval: float = 0.05) -> None:
        self._pin = pin if pin is not None else settings.power_switch_pin
        self._poll = poll_interval
        self._gpio = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._listeners: list[Callable[[bool], None]] = []
        self._is_on = False
        self._setup()

    def _setup(self) -> None:
        try:
            import Jetson.GPIO as GPIO  # type: ignore[import-not-found]

            self._gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            self._is_on = GPIO.input(self._pin) == GPIO.LOW
            logger.info(
                f"Power switch on GPIO {self._pin} "
                f"(initial: {'on' if self._is_on else 'off'})"
            )
        except (ImportError, RuntimeError) as e:
            logger.warning(f"GPIO unavailable ({e}); power switch fixed 'on' (dev)")
            self._gpio = None
            self._is_on = True

    @property
    def is_on(self) -> bool:
        return self._is_on

    def add_listener(self, callback: Callable[[bool], None]) -> None:
        self._listeners.append(callback)

    def start(self) -> None:
        if self._gpio is None or (self._thread is not None and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="power-switch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def _read(self) -> bool:
        return self._gpio.input(self._pin) == self._gpio.LOW

    def _loop(self) -> None:
        while not self._stop.is_set():
            new_state = self._read()
            if new_state != self._is_on:
                self._stop.wait(_DEBOUNCE_S)
                if self._stop.is_set():
                    return
                if self._read() == new_state:
                    self._is_on = new_state
                    logger.info(f"Power switch: {'on' if new_state else 'off'}")
                    for cb in list(self._listeners):
                        try:
                            cb(new_state)
                        except Exception as e:  # noqa: BLE001
                            logger.warning(f"Power switch listener error: {e}")
            self._stop.wait(self._poll)

    def cleanup(self) -> None:
        self.stop()
        if self._gpio is not None:
            self._gpio.cleanup(self._pin)
