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
_DR_64SPS = 0x0060           # 64 samples/s — 15.6 ms/conversion
_DR_128SPS = 0x0080          # 128 samples/s
_DR_250SPS = 0x00A0          # 250 samples/s — 4 ms/conversion
_COMP_DISABLE = 0x0003

# Default data rate. With the centralized SharedAdcPoller cycling through
# three channels at ~10 Hz, conversion time isn't a bottleneck and a
# slower rate means a single conversion captures more averaged samples
# (less noise on the pot/switch lines). The double-read mux-settle still
# applies, so per-channel cost is ~32 ms — three channels per cycle fit
# comfortably in 100 ms.
_DEFAULT_DATA_RATE = _DR_64SPS

_FS_VOLTAGE = 4.096          # full-scale voltage for _PGA_4_096V
_FS_CODE = 32767             # 16-bit signed; positive range only used here
_VREF_NOMINAL = 3.3          # the rail across the pot, for percent calc

# Conversion timing — wait this long after triggering a single-shot
# conversion before reading the result. At 64 SPS conversion is ~15.6 ms,
# plus a few ms margin for mux settle + i2c slack. Deterministic sleep
# beats polling the OS bit: the bit is racy in the first ms after a
# config write (chip hasn't yet started the new conversion, so OS still
# reads 1 from the prior one — that fooled the polling read into
# returning the stale result and caused channel-swapped readings).
_CONV_WAIT_S = 0.020


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
        """Return the raw signed conversion value, or None on error/unavailable.

        Single conversion per call: writes config (which switches mux and
        triggers the conversion in single-shot mode), sleeps through the
        conversion, then reads the result. The mux settles in well under
        1 ms vs the 15.6 ms conversion time, so contamination is not a
        concern at sane data rates.

        Thread-safety: only one caller (the SharedAdcPoller) should be
        invoking this. If multiple threads call concurrently they will
        race on the chip's mux/config state and produce wrong-channel
        readings — use the poller cache instead.
        """
        if self._bus is None or channel not in _MUX_SINGLE:
            return None
        config = (
            _OS_SINGLE
            | _MUX_SINGLE[channel]
            | _PGA_4_096V
            | _MODE_SINGLE
            | _DEFAULT_DATA_RATE
            | _COMP_DISABLE
        )
        cfg_bytes = [(config >> 8) & 0xFF, config & 0xFF]
        with self._lock:
            try:
                self._bus.write_i2c_block_data(self._addr, _REG_CONFIG, cfg_bytes)
                time.sleep(_CONV_WAIT_S)
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
    """Single-ended ADS1115 reader for a potentiometer wiper.

    Two modes:
      * No poller (default): each ``read()`` performs a fresh i2c
        conversion. Used by tests and standalone scripts.
      * With poller (production): ``read()`` consults the shared
        ``SharedAdcPoller``'s cache — no i2c traffic on the read path.
    """

    def __init__(
        self,
        bus: int | None = None,
        addr: int | None = None,
        channel: int | None = None,
        adc: ADS1115 | None = None,
        poller: object | None = None,
    ) -> None:
        self._adc = adc if adc is not None else ADS1115(bus=bus, addr=addr)
        self._channel = channel if channel is not None else settings.pot_ads1115_channel
        self._poller = poller

    @property
    def available(self) -> bool:
        return self._adc.available

    @property
    def error(self) -> str | None:
        return self._adc.error

    def read(self) -> PotReading | None:
        if self._poller is not None:
            voltage = self._poller.get_voltage(self._channel)
            if voltage is None:
                return None
            raw = int(voltage / _FS_VOLTAGE * _FS_CODE)
        else:
            raw = self._adc.read_raw(self._channel)
            if raw is None:
                return None
            voltage = (raw / _FS_CODE) * _FS_VOLTAGE
        pct = max(0.0, min(100.0, (voltage / _VREF_NOMINAL) * 100.0))
        return PotReading(raw=raw, voltage=round(voltage, 4), pct=round(pct, 1))

    def cleanup(self) -> None:
        self._adc.cleanup()
