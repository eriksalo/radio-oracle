"""Read a 10 kΩ potentiometer wired through an ADS1115 I²C ADC.

Wiper sits on AIN0; the breakout is at I²C address 0x48 by default. We talk
to the ADS1115 directly over /dev/i2c via smbus2 to avoid pulling in the
full Adafruit Blinka stack on the Jetson.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from loguru import logger

from config.settings import settings

# ADS1115 register pointers
_REG_CONVERSION = 0x00
_REG_CONFIG = 0x01

# Config bits (datasheet §9.5.3)
_OS_SINGLE = 0x8000          # start a single conversion
_MUX_SINGLE = {              # single-ended on AINx
    0: 0x4000, 1: 0x5000, 2: 0x6000, 3: 0x7000,
}
_PGA_4_096V = 0x0200         # ±4.096 V — covers a 3.3 V rail with headroom
_MODE_SINGLE = 0x0100        # one-shot
_DR_128SPS = 0x0080          # 128 samples/s
_COMP_DISABLE = 0x0003

_FS_VOLTAGE = 4.096          # full-scale voltage for _PGA_4_096V
_FS_CODE = 32767             # 16-bit signed; positive range only used here
_VREF_NOMINAL = 3.3          # the rail across the pot, for percent calc


@dataclass(frozen=True)
class PotReading:
    raw: int
    voltage: float
    pct: float                # 0..100, clamped


class Potentiometer:
    """One-shot ADS1115 reader for a single-ended potentiometer wiper.

    Falls back to ``available=False`` if /dev/i2c-N or smbus2 isn't usable;
    callers should treat unavailability as soft (diag UI shows "—").
    """

    def __init__(
        self,
        bus: int | None = None,
        addr: int | None = None,
        channel: int | None = None,
    ) -> None:
        self._bus_num = bus if bus is not None else settings.pot_i2c_bus
        self._addr = addr if addr is not None else settings.pot_ads1115_addr
        self._channel = channel if channel is not None else settings.pot_ads1115_channel
        self._bus = None
        self._lock = threading.Lock()
        self._error: str | None = None
        self._setup()

    @property
    def available(self) -> bool:
        return self._bus is not None

    @property
    def error(self) -> str | None:
        return self._error

    def _setup(self) -> None:
        if self._channel not in _MUX_SINGLE:
            self._error = f"invalid channel {self._channel}"
            return
        try:
            from smbus2 import SMBus  # type: ignore[import-not-found]

            self._bus = SMBus(self._bus_num)
            logger.info(
                f"Potentiometer on i2c-{self._bus_num} addr=0x{self._addr:02x} "
                f"AIN{self._channel}"
            )
        except (ImportError, FileNotFoundError, PermissionError, OSError) as e:
            self._error = repr(e)
            logger.warning(f"Pot ADC unavailable ({e})")
            self._bus = None

    def read(self) -> PotReading | None:
        """Trigger a single conversion and return the wiper voltage.

        Returns None if the ADC isn't available or the bus errored.
        """
        if self._bus is None:
            return None
        config = (
            _OS_SINGLE
            | _MUX_SINGLE[self._channel]
            | _PGA_4_096V
            | _MODE_SINGLE
            | _DR_128SPS
            | _COMP_DISABLE
        )
        cfg_bytes = [(config >> 8) & 0xFF, config & 0xFF]
        with self._lock:
            try:
                self._bus.write_i2c_block_data(self._addr, _REG_CONFIG, cfg_bytes)
                # 128 SPS → ~7.8 ms; small margin for jitter.
                time.sleep(0.010)
                hi, lo = self._bus.read_i2c_block_data(self._addr, _REG_CONVERSION, 2)
            except OSError as e:
                self._error = repr(e)
                logger.debug(f"Pot read failed: {e}")
                return None
        raw = (hi << 8) | lo
        if raw & 0x8000:  # two's-complement; pot can't go negative but be safe
            raw -= 1 << 16
        voltage = (raw / _FS_CODE) * _FS_VOLTAGE
        pct = max(0.0, min(100.0, (voltage / _VREF_NOMINAL) * 100.0))
        return PotReading(raw=raw, voltage=round(voltage, 4), pct=round(pct, 1))

    def cleanup(self) -> None:
        bus = self._bus
        self._bus = None
        if bus is not None:
            try:
                bus.close()
            except OSError:
                pass
