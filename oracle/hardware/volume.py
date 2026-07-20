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

    The mapping is LINEAR: gain feeds ``pactl set-sink-volume N%``, and
    PulseAudio's volume percentage is already perceptually (cubically)
    mapped. The previous quadratic here stacked onto that — a ~6th-power
    curve end to end, leaving most of the knob's travel inaudible and all
    the loudness crammed into the last few degrees.

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

    def reading(self):
        """Full pot reading (raw/voltage/pct) or None — for telemetry."""
        try:
            return self._pot.read()
        except Exception:  # noqa: BLE001
            return None

    @property
    def gain(self) -> float:
        """Current volume as 0.0–1.0, read from the poller cache."""
        reading = self._pot.read()
        if reading is None:
            return self._last_gain
        # Map calibrated voltage range to 0.0–1.0; Pulse's cubic volume
        # scale provides the perceptual shaping.
        g = max(0.0, min(1.0, (reading.voltage - _V_MIN) / (_V_MAX - _V_MIN)))
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
