"""Volume control via potentiometer — maps pot % to audio gain."""

from __future__ import annotations

from loguru import logger

from config.settings import settings
from oracle.hardware.pot import Potentiometer

# Calibrated voltage range of the physical pot wiper.
# Measured: 0.00 V at full CCW, ~3.01 V at full CW.
_V_MIN = 0.0
_V_MAX = 3.01


class VolumeControl:
    """Reads the physical pot and exposes a 0.0–1.0 gain value.

    Uses quadratic scaling so the knob feels more natural (linear pots
    have most of their perceptual loudness change crammed into the first
    quarter of rotation with linear mapping).

    Pulls voltage from the SharedAdcPoller cache (no i2c on the read
    path), so .gain is cheap enough to call freely.
    """

    def __init__(self, pot: Potentiometer | None = None) -> None:
        if pot is None:
            from oracle.hardware.switch_adc import shared_adc, shared_adc_poller
            poller = shared_adc_poller()
            poller.register(settings.pot_ads1115_channel)
            pot = Potentiometer(adc=shared_adc(), poller=poller)
        self._pot = pot
        self._last_gain: float = 1.0  # fallback if ADC unavailable

    @property
    def available(self) -> bool:
        return self._pot.available

    @property
    def gain(self) -> float:
        """Current volume as 0.0–1.0, read from the poller cache."""
        reading = self._pot.read()
        if reading is None:
            return self._last_gain
        # Map calibrated voltage range to 0.0–1.0, then apply quadratic curve
        linear = max(0.0, min(1.0, (reading.voltage - _V_MIN) / (_V_MAX - _V_MIN)))
        g = linear * linear
        self._last_gain = g
        return g

    def cleanup(self) -> None:
        self._pot.cleanup()


# Module-level singleton, lazily created.
_instance: VolumeControl | None = None


def get_volume_control() -> VolumeControl:
    """Return the shared VolumeControl singleton."""
    global _instance
    if _instance is None:
        _instance = VolumeControl()
        if _instance.available:
            logger.info(f"Volume control active (gain={_instance.gain:.2f})")
        else:
            logger.info("Volume control: pot unavailable, gain fixed at 1.0")
    return _instance
