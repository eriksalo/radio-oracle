"""Tests for StatusLEDs mode → color mapping."""

from oracle.hardware.leds import MODE_COLORS, Color, StatusLEDs


def test_mode_colors_cover_all_modes():
    expected = {"off", "radio", "librarian", "thinking", "speaking", "error"}
    assert set(MODE_COLORS.keys()) == expected


def test_mode_color_values():
    assert MODE_COLORS["off"] == Color(False, False, False)
    assert MODE_COLORS["radio"] == Color(False, True, False)
    assert MODE_COLORS["librarian"] == Color(False, False, True)
    assert MODE_COLORS["thinking"] == Color(True, True, False)
    assert MODE_COLORS["speaking"] == Color(False, True, True)
    assert MODE_COLORS["error"] == Color(True, False, False)


def test_status_leds_init_without_gpio_logs_only():
    # On dev machines without Jetson.GPIO, this must not raise.
    leds = StatusLEDs()
    assert leds.mode == "off"
    leds.set_mode("radio")
    assert leds.mode == "radio"
    leds.set_mode("error")
    assert leds.mode == "error"
    leds.cleanup()
