"""Volume control via potentiometer — maps pot % to audio gain."""

from __future__ import annotations

import time

from loguru import logger

from oracle.hardware.pot import Potentiometer

# Calibrated voltage range of the physical pot wiper.
# Measured: 0.00 V at full CCW, ~3.01 V at full CW.
_V_MIN = 0.0
_V_MAX = 3.01

# Cache TTL for .gain reads. The ADS1115 over i2c-7 takes ~12 ms per read
# and shares the bus with the action button and power switch. Calling
# .gain from inside an audio callback (every ~5–20 ms) blows the
# callback's time budget and causes underruns. 50 ms is imperceptible
# latency for a human turning a knob and gives the bus headroom.
_GAIN_CACHE_TTL_S = 0.05


class VolumeControl:
    """Reads the physical pot and exposes a 0.0–1.0 gain value.

    Uses quadratic scaling so the knob feels more natural (linear pots
    have most of their perceptual loudness change crammed into the first
    quarter of rotation with linear mapping).
    """

    def __init__(self, pot: Potentiometer | None = None) -> None:
        if pot is None:
            from oracle.hardware.switch_adc import shared_adc
            pot = Potentiometer(adc=shared_adc())
        self._pot = pot
        self._last_gain: float = 1.0  # fallback if ADC unavailable
        self._last_read_t: float = 0.0  # monotonic ts of last successful read

    @property
    def available(self) -> bool:
        return self._pot.available

    @property
    def gain(self) -> float:
        """Current volume as 0.0–1.0. Cached for _GAIN_CACHE_TTL_S to keep
        audio callbacks cheap — the underlying ADS1115 i2c read is ~12 ms.
        """
        now = time.monotonic()
        if now - self._last_read_t < _GAIN_CACHE_TTL_S:
            return self._last_gain
        reading = self._pot.read()
        # Update the cache timestamp regardless of read success so a
        # missing ADC doesn't cause callers to hammer the bus.
        self._last_read_t = now
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
