"""ADS1115 I²C ADC driver + potentiometer reader.

The low-level :class:`ADS1115` class is reused by :mod:`oracle.hardware.switch_adc`
to read switches as digital signals (10 kΩ pull-up to 3.3 V, switch shorts the
line to GND — the ADC samples the voltage and a threshold converts to bool).
We talk to the chip directly over /dev/i2c via smbus2 to avoid pulling in the
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


class ADS1115:
    """Single-ended one-shot reader for an ADS1115.

    Thread-safe: each instance holds its own SMBus fd and a lock; multiple
    instances against the same /dev/i2c-N are safe (kernel arbitrates).
    """

    def __init__(self, bus: int | None = None, addr: int | None = None) -> None:
        self._bus_num = bus if bus is not None else settings.pot_i2c_bus
        self._addr = addr if addr is not None else settings.pot_ads1115_addr
        self._bus = None
        self._lock = threading.Lock()
        self._error: str | None = None
        self._open()

    @property
    def available(self) -> bool:
        return self._bus is not None

    @property
    def error(self) -> str | None:
        return self._error

    def _open(self) -> None:
        try:
            from smbus2 import SMBus  # type: ignore[import-not-found]

            self._bus = SMBus(self._bus_num)
            logger.info(f"ADS1115 on i2c-{self._bus_num} addr=0x{self._addr:02x}")
        except (ImportError, FileNotFoundError, PermissionError, OSError) as e:
            self._error = repr(e)
            logger.warning(f"ADS1115 unavailable ({e})")
            self._bus = None

    def read_raw(self, channel: int) -> int | None:
        """Return the raw signed conversion value, or None on error/unavailable."""
        if self._bus is None or channel not in _MUX_SINGLE:
            return None
        config = (
            _OS_SINGLE
            | _MUX_SINGLE[channel]
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
                logger.debug(f"ADS1115 read failed: {e}")
                return None
        raw = (hi << 8) | lo
        if raw & 0x8000:  # two's-complement
            raw -= 1 << 16
        return raw

    def read_voltage(self, channel: int) -> float | None:
        raw = self.read_raw(channel)
        if raw is None:
            return None
        return (raw / _FS_CODE) * _FS_VOLTAGE

    def cleanup(self) -> None:
        bus = self._bus
        self._bus = None
        if bus is not None:
            try:
                bus.close()
            except OSError:
                pass


@dataclass(frozen=True)
class PotReading:
    raw: int
    voltage: float
    pct: float                # 0..100, clamped


class Potentiometer:
    """One-shot ADS1115 reader for a single-ended potentiometer wiper."""

    def __init__(
        self,
        bus: int | None = None,
        addr: int | None = None,
        channel: int | None = None,
    ) -> None:
        self._adc = ADS1115(bus=bus, addr=addr)
        self._channel = channel if channel is not None else settings.pot_ads1115_channel

    @property
    def available(self) -> bool:
        return self._adc.available

    @property
    def error(self) -> str | None:
        return self._adc.error

    def read(self) -> PotReading | None:
        raw = self._adc.read_raw(self._channel)
        if raw is None:
            return None
        voltage = (raw / _FS_CODE) * _FS_VOLTAGE
        pct = max(0.0, min(100.0, (voltage / _VREF_NOMINAL) * 100.0))
        return PotReading(raw=raw, voltage=round(voltage, 4), pct=round(pct, 1))

    def cleanup(self) -> None:
        self._adc.cleanup()
