"""SPST toggle (virtual power) switch monitor.

Closed (line shorted to GND through the switch) = device on / radio active.
Open  (held at 3V3 by external 10 kΩ pull-up)    = device standby.

Polls one ADS1115 channel in a background thread; listeners are invoked on
edge. ADC-based read sidesteps the Tegra234 GPIO INPUT loopback bug on
JP 6.2.x.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from loguru import logger

from oracle.hardware.switch_adc import make_power_switch_switch

_DEBOUNCE_S = 0.05


class PowerSwitch:
    """Monitor a SPST toggle switch via the ADS1115."""

    def __init__(self, poll_interval: float = 0.05) -> None:
        self._poll = poll_interval
        self._switch = make_power_switch_switch()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._listeners: list[Callable[[bool], None]] = []
        if self._switch.available:
            initial = self._switch.is_closed()
            self._is_on = bool(initial)
            logger.info(
                f"Power switch on ADS1115 AIN{self._switch.channel} "
                f"(initial: {'on' if self._is_on else 'off'})"
            )
        else:
            logger.warning(
                f"ADS1115 unavailable ({self._switch.error}); "
                "power switch fixed 'on' (dev)"
            )
            self._is_on = True

    @property
    def is_on(self) -> bool:
        return self._is_on

    def add_listener(self, callback: Callable[[bool], None]) -> None:
        self._listeners.append(callback)

    def start(self) -> None:
        if not self._switch.available or (self._thread is not None and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="power-switch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def _read(self) -> bool | None:
        return self._switch.is_closed()

    def _loop(self) -> None:
        while not self._stop.is_set():
            new_state = self._read()
            if new_state is None:
                self._stop.wait(self._poll)
                continue
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
