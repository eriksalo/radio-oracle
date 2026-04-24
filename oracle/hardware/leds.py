"""Status LED control for the Oracle enclosure."""

from __future__ import annotations

from loguru import logger

from config.settings import settings


class StatusLEDs:
    """Control status LEDs: idle, listening, thinking, speaking.

    Falls back to log messages if GPIO is unavailable.
    """

    STATES = ("idle", "listening", "thinking", "speaking")

    def __init__(self):
        self._pins = {
            "idle": settings.led_idle_pin,
            "listening": settings.led_listen_pin,
            "thinking": settings.led_think_pin,
        }
        self._gpio = None
        self._setup()

    def _setup(self) -> None:
        try:
            import Jetson.GPIO as GPIO

            self._gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            for name, pin in self._pins.items():
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            logger.info(f"Status LEDs configured: {self._pins}")
        except (ImportError, RuntimeError) as e:
            logger.warning(f"GPIO unavailable ({e}), LEDs will log only")
            self._gpio = None

    def set_state(self, state: str) -> None:
        """Set the current state LED (turns off others)."""
        if state not in self.STATES:
            logger.warning(f"Unknown LED state: {state}")
            return

        logger.debug(f"LED state: {state}")
        if self._gpio is None:
            return

        for name, pin in self._pins.items():
            self._gpio.output(pin, self._gpio.HIGH if name == state else self._gpio.LOW)

    def all_off(self) -> None:
        """Turn off all LEDs."""
        if self._gpio is None:
            return
        for pin in self._pins.values():
            self._gpio.output(pin, self._gpio.LOW)

    def cleanup(self) -> None:
        """Release GPIO resources."""
        if self._gpio is not None:
            self.all_off()
            self._gpio.cleanup(list(self._pins.values()))
