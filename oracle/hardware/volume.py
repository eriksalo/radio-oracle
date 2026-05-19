"""Volume control via potentiometer — maps pot % to audio gain."""

from __future__ import annotations

from loguru import logger

from oracle.hardware.pot import Potentiometer


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

    @property
    def available(self) -> bool:
        return self._pot.available

    @property
    def gain(self) -> float:
        """Current volume as 0.0–1.0, read fresh from hardware."""
        reading = self._pot.read()
        if reading is None:
            return self._last_gain
        # pct is 0–100; map to 0.0–1.0 with quadratic curve
        linear = reading.pct / 100.0
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
