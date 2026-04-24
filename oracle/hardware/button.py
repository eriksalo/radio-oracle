"""PTT (Push-to-Talk) button via Jetson GPIO."""

from __future__ import annotations

from loguru import logger

from config.settings import settings


class PTTButton:
    """GPIO-based push-to-talk button.

    Hold button to talk, release to submit.
    Falls back to keyboard if GPIO is unavailable (dev mode).
    """

    def __init__(self, pin: int | None = None):
        self._pin = pin or settings.ptt_gpio_pin
        self._gpio = None
        self._setup()

    def _setup(self) -> None:
        try:
            import Jetson.GPIO as GPIO

            self._gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            logger.info(f"PTT button configured on GPIO pin {self._pin}")
        except (ImportError, RuntimeError) as e:
            logger.warning(f"GPIO unavailable ({e}), PTT will use keyboard fallback")
            self._gpio = None

    def wait_for_press(self) -> None:
        """Block until the PTT button is pressed (held down)."""
        if self._gpio is None:
            input("Press Enter to start recording...")
            return

        logger.debug("Waiting for PTT press...")
        self._gpio.wait_for_edge(self._pin, self._gpio.FALLING)
        logger.debug("PTT pressed")

    def is_held(self) -> bool:
        """Check if the button is currently held down."""
        if self._gpio is None:
            return False
        return self._gpio.input(self._pin) == self._gpio.LOW

    def wait_for_release(self) -> None:
        """Block until the PTT button is released."""
        if self._gpio is None:
            return

        self._gpio.wait_for_edge(self._pin, self._gpio.RISING)
        logger.debug("PTT released")

    def cleanup(self) -> None:
        """Release GPIO resources."""
        if self._gpio is not None:
            self._gpio.cleanup(self._pin)
