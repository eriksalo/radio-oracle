"""An ADS1115 that stops answering is reported once, not on every read."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from loguru import logger

from oracle.hardware import pot, switch_adc


class _Bus:
    """smbus2.SMBus stand-in: answers, or raises EREMOTEIO like a missing chip."""

    def __init__(self) -> None:
        self.dead = False
        self._config = [0x80, 0x00]

    def write_i2c_block_data(self, addr: int, register: int, data: list[int]) -> None:
        if self.dead:
            raise OSError(121, "Remote I/O error")
        self._config = [data[0] | 0x80, data[1]]

    def read_i2c_block_data(self, addr: int, register: int, length: int) -> list[int]:
        if self.dead:
            raise OSError(121, "Remote I/O error")
        return list(self._config) if register == 0x01 else [0x40, 0x00]

    def close(self) -> None:
        return None


@pytest.fixture
def adc(monkeypatch: pytest.MonkeyPatch) -> pot.ADS1115:
    monkeypatch.setattr(pot.ADS1115, "_open", lambda self: None)
    monkeypatch.setattr(pot, "_CONV_WAIT_S", 0.0)
    chip = pot.ADS1115(bus=7, addr=0x48)
    chip._bus = _Bus()
    return chip


@pytest.fixture
def log_lines() -> Iterator[list[str]]:
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(str(message)), level="DEBUG", format="{message}")
    yield lines
    logger.remove(sink)


def test_failure_is_logged_once_and_recovery_once(adc: pot.ADS1115, log_lines: list[str]) -> None:
    assert adc.read_raw(1) == 0x4000 and adc.responding
    adc._bus.dead = True
    for _ in range(50):
        assert adc.read_raw(1) is None
    assert not adc.responding and adc.error is not None
    assert len(log_lines) == 1 and "stopped answering" in log_lines[0]

    adc._bus.dead = False
    assert adc.read_raw(1) == 0x4000
    assert adc.responding and adc.error is None
    assert len(log_lines) == 2 and "answering again" in log_lines[1]
    assert "50 failed reads" in log_lines[1]


def test_a_long_outage_is_repeated_at_the_reminder_interval(
    adc: pot.ADS1115, log_lines: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [1000.0]
    monkeypatch.setattr(pot.time, "monotonic", lambda: now[0])
    adc._bus.dead = True
    adc.read_raw(1)
    now[0] += pot._DOWN_REMINDER_S - 1
    adc.read_raw(1)
    assert len(log_lines) == 1
    now[0] += 2
    adc.read_raw(1)
    adc.read_raw(1)
    assert len(log_lines) == 2 and "still not answering" in log_lines[1]


def test_poller_slows_down_while_the_chip_is_silent(
    adc: pot.ADS1115, monkeypatch: pytest.MonkeyPatch
) -> None:
    poller = switch_adc.SharedAdcPoller(adc, period_s=0.1)
    poller._channels = [1]
    waits: list[float] = []

    def wait(seconds: float) -> None:
        waits.append(seconds)
        if len(waits) == 1:
            adc._bus.dead = True
        else:
            poller._stop.set()

    monkeypatch.setattr(poller._stop, "wait", wait)
    poller._loop()
    assert waits[0] <= 0.1
    assert waits[1] > 1.0
    assert poller.get_voltage(1) is None
