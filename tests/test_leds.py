"""Tests for StatusLEDs mode → color mapping and blink behaviour."""

import time

from oracle.hardware.leds import _BLINK_PERIOD_S, MODE_COLORS, Color, StatusLEDs


def test_mode_colors_cover_all_modes():
    expected = {"off", "radio", "librarian", "reader", "thinking", "speaking", "error"}
    assert set(MODE_COLORS.keys()) == expected


def test_mode_color_values():
    assert MODE_COLORS["off"] == Color(False, False, False)
    assert MODE_COLORS["radio"] == Color(False, True, False)
    assert MODE_COLORS["librarian"] == Color(False, False, True)
    assert MODE_COLORS["reader"] == Color(True, False, True)
    # thinking and speaking are both blue; thinking blinks (see below).
    assert MODE_COLORS["thinking"] == Color(False, False, True)
    assert MODE_COLORS["speaking"] == Color(False, False, True)
    assert MODE_COLORS["error"] == Color(True, False, False)


def test_blink_modes_table():
    # thinking should blink at ~2 Hz; error stays around 1.7 Hz.
    assert "thinking" in _BLINK_PERIOD_S
    assert "error" in _BLINK_PERIOD_S
    assert _BLINK_PERIOD_S["thinking"] <= 0.5  # 2 Hz or faster
    # Solid modes must NOT have an entry.
    for solid in ("off", "radio", "librarian", "reader", "speaking"):
        assert solid not in _BLINK_PERIOD_S


def test_status_leds_init_without_gpio_logs_only():
    leds = StatusLEDs()
    assert leds.mode == "off"
    leds.set_mode("radio")
    assert leds.mode == "radio"
    leds.set_mode("error")
    assert leds.mode == "error"
    leds.cleanup()


def test_thinking_mode_starts_a_blink_thread():
    leds = StatusLEDs()
    try:
        leds.set_mode("thinking")
        assert leds.mode == "thinking"
        assert leds._blink_thread is not None
        assert leds._blink_thread.is_alive()
    finally:
        leds.set_mode("off")
        leds.cleanup()


def test_speaking_mode_is_solid_no_blink():
    leds = StatusLEDs()
    try:
        leds.set_mode("speaking")
        assert leds.mode == "speaking"
        # No blink thread should be running for solid modes.
        assert leds._blink_thread is None
    finally:
        leds.cleanup()


def test_switching_from_blink_to_solid_stops_thread():
    leds = StatusLEDs()
    try:
        leds.set_mode("thinking")
        assert leds._blink_thread is not None
        leds.set_mode("librarian")
        # Allow the join to settle.
        time.sleep(0.05)
        assert leds._blink_thread is None
    finally:
        leds.cleanup()
