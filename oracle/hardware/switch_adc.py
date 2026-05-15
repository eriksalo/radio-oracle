"""Digital switch read via ADS1115 (workaround for Tegra234 GPIO input bug).

Switch wiring: external 10 kΩ pull-up to 3.3 V, switch shorts the line to GND.
Idle  → ~3.3 V (open contact)
Closed → ~0 V

This replaces direct GPIO reads on Jetson Orin Nano / JP 6.2.x, where the
Tegra234 GPIO controller's INPUT_VALUE register exhibits a loopback bug — it
reads the OUT_VAL latch instead of the actual pin. See memory file
``hdr40-pinmux-overlay.md`` for the diagnosis.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from loguru import logger

from config.settings import settings
from oracle.hardware.pot import ADS1115

# Default hysteresis thresholds for a 3.3 V rail with 10 kΩ pull-up.
# Anything below LOW is unambiguously "closed" (shorted to GND).
# Anything above HIGH is unambiguously "open" (held by pull-up).
# In between, keep previous state.
_THRESH_LOW = 0.8
_THRESH_HIGH = 2.5


@dataclass(frozen=True)
class SwitchReading:
    channel: int
    voltage: float
    closed: bool          # True = switch shorted to GND (active-low style)


class DigitalSwitch:
    """Read a switch as a boolean via one ADS1115 channel.

    ``closed`` semantics match the prior GPIO-with-PUD_UP convention:
    pressed/on = switch shorted to GND = ``True``.

    Construct with a pre-made :class:`ADS1115` (preferred — one I²C fd shared
    across all switches and the pot) or pass ``adc=None`` to open a private
    bus.
    """

    def __init__(
        self,
        channel: int,
        adc: ADS1115 | None = None,
        thresh_low: float = _THRESH_LOW,
        thresh_high: float = _THRESH_HIGH,
        active_low: bool = True,
    ) -> None:
        self._adc = adc if adc is not None else ADS1115()
        self._channel = channel
        self._thresh_low = thresh_low
        self._thresh_high = thresh_high
        self._active_low = active_low
        self._state: bool | None = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._adc.available

    @property
    def error(self) -> str | None:
        return self._adc.error

    @property
    def channel(self) -> int:
        return self._channel

    def read(self) -> SwitchReading | None:
        """Sample the channel and return a hysteresis-filtered boolean."""
        v = self._adc.read_voltage(self._channel)
        if v is None:
            return None
        with self._lock:
            if v <= self._thresh_low:
                new_state = True if self._active_low else False
            elif v >= self._thresh_high:
                new_state = False if self._active_low else True
            elif self._state is not None:
                # voltage in dead-band — keep previous
                new_state = self._state
            else:
                # first read in dead-band — pick the closer threshold
                midpoint = (self._thresh_low + self._thresh_high) / 2
                if v < midpoint:
                    new_state = True if self._active_low else False
                else:
                    new_state = False if self._active_low else True
            self._state = new_state
        return SwitchReading(channel=self._channel, voltage=round(v, 4), closed=new_state)

    def is_closed(self) -> bool | None:
        r = self.read()
        return r.closed if r is not None else None


_shared_adc: ADS1115 | None = None
_shared_lock = threading.Lock()


def shared_adc() -> ADS1115:
    """Return a process-wide shared :class:`ADS1115` instance.

    Lets the pot and all switches share one /dev/i2c-N fd instead of opening
    one per consumer.
    """
    global _shared_adc
    with _shared_lock:
        if _shared_adc is None:
            _shared_adc = ADS1115()
            if not _shared_adc.available:
                logger.warning(f"Shared ADS1115 unavailable: {_shared_adc.error}")
        return _shared_adc


def make_action_button_switch() -> DigitalSwitch:
    return DigitalSwitch(channel=settings.action_button_ads1115_channel, adc=shared_adc())


def make_power_switch_switch() -> DigitalSwitch:
    return DigitalSwitch(channel=settings.power_switch_ads1115_channel, adc=shared_adc())
