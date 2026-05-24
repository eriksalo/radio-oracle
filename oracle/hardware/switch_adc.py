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
import time
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
        poller: "SharedAdcPoller | None" = None,
    ) -> None:
        self._adc = adc if adc is not None else ADS1115()
        self._channel = channel
        self._thresh_low = thresh_low
        self._thresh_high = thresh_high
        self._active_low = active_low
        self._state: bool | None = None
        self._lock = threading.Lock()
        # When a poller is supplied, read() returns cached values instead
        # of hitting i2c directly. Caller is responsible for starting it.
        self._poller = poller

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
        """Sample the channel and return a hysteresis-filtered boolean.

        If a SharedAdcPoller was attached, returns its cached voltage —
        no i2c traffic on this call. Otherwise reads the ADC directly.
        """
        if self._poller is not None:
            v = self._poller.get_voltage(self._channel)
        else:
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


# ---------------------------------------------------------------------------
# Shared ADC poller — single background thread that cycles through all
# configured channels and caches the latest voltage. Pot / DigitalSwitch /
# VolumeControl read from the cache instead of hitting i2c themselves.
#
# Why: previously each consumer (action button, power switch, volume knob)
# either ran its own poll thread or read on demand from inside a hot path.
# That gave us three threads contending for one ADS1115 lock, three
# separate mux-settle double-reads per cycle, and i2c traffic proportional
# to consumer count. With one poller, traffic is bounded by the cycle rate
# regardless of how many consumers subscribe.
# ---------------------------------------------------------------------------


class SharedAdcPoller:
    """Background thread that reads N channels in sequence and caches results.

    Channels are registered before ``start()``. After start, ``get_voltage(ch)``
    returns the most recent cached value (or ``None`` if no successful read
    yet / ADC unavailable).
    """

    def __init__(self, adc: ADS1115, period_s: float = 0.1) -> None:
        self._adc = adc
        self._period = period_s
        self._channels: list[int] = []
        self._channels_lock = threading.Lock()
        self._cache: dict[int, float | None] = {}
        self._cache_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def register(self, channel: int) -> None:
        """Add a channel to the polling cycle. Auto-starts the poller."""
        with self._channels_lock:
            if channel in self._channels:
                self.start()
                return
            self._channels.append(channel)
        with self._cache_lock:
            self._cache.setdefault(channel, None)
        self.start()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if not self._adc.available:
            logger.warning("SharedAdcPoller: ADC unavailable, not starting")
            return
        with self._channels_lock:
            if not self._channels:
                logger.debug("SharedAdcPoller: no channels registered yet, skipping start")
                return
            chans = sorted(self._channels)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="adc-poller", daemon=True)
        self._thread.start()
        logger.info(
            f"ADC poller started on channels {chans} "
            f"@ {1.0 / self._period:.0f} Hz target cycle rate"
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def get_voltage(self, channel: int) -> float | None:
        with self._cache_lock:
            return self._cache.get(channel)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self) -> None:
        while not self._stop.is_set():
            cycle_start = time.monotonic()
            with self._channels_lock:
                chans = list(self._channels)  # snapshot to avoid mutation during iter
            for ch in chans:
                if self._stop.is_set():
                    break
                v = self._adc.read_voltage(ch)
                with self._cache_lock:
                    self._cache[ch] = v
            elapsed = time.monotonic() - cycle_start
            sleep_left = self._period - elapsed
            if sleep_left > 0:
                self._stop.wait(sleep_left)


_shared_poller: SharedAdcPoller | None = None
_poller_lock = threading.Lock()


def shared_adc_poller() -> SharedAdcPoller:
    """Return the process-wide shared ADC poller (lazy-initialized)."""
    global _shared_poller
    with _poller_lock:
        if _shared_poller is None:
            _shared_poller = SharedAdcPoller(shared_adc())
        return _shared_poller


def make_action_button_switch() -> DigitalSwitch:
    poller = shared_adc_poller()
    poller.register(settings.action_button_ads1115_channel)
    return DigitalSwitch(
        channel=settings.action_button_ads1115_channel,
        adc=shared_adc(),
        poller=poller,
    )


def make_power_switch_switch() -> DigitalSwitch:
    poller = shared_adc_poller()
    poller.register(settings.power_switch_ads1115_channel)
    return DigitalSwitch(
        channel=settings.power_switch_ads1115_channel,
        adc=shared_adc(),
        poller=poller,
    )
