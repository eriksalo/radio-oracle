"""Tests for StatusLEDs mode → color mapping and blink behaviour."""

import time

from oracle.hardware.leds import _BLINK_PERIOD_S, MODE_COLORS, Color, StatusLEDs


def test_mode_colors_cover_all_modes():
    expected = {
        "off", "boot", "waiting", "error",
        "q_listen", "q_think", "q_speak",
        "music_play", "music_listen", "music_think", "music_speak",
        "book_read", "book_listen", "book_think", "book_speak", "book_paused",
    }  # fmt: skip
    assert set(MODE_COLORS.keys()) == expected


def test_channel_colours():
    blue, green, purple = (
        Color(False, False, True),
        Color(False, True, False),
        Color(True, False, True),
    )
    for m in ("q_listen", "q_think", "q_speak"):
        assert MODE_COLORS[m] == blue
    for m in ("music_play", "music_listen", "music_think", "music_speak"):
        assert MODE_COLORS[m] == green
    for m in ("book_read", "book_listen", "book_think", "book_speak", "book_paused"):
        assert MODE_COLORS[m] == purple
    assert MODE_COLORS["off"] == Color(False, False, False)
    assert MODE_COLORS["error"] == Color(True, False, False)
    assert MODE_COLORS["waiting"] == Color(True, True, True)
    assert MODE_COLORS["boot"] == Color(True, True, False)


def test_blink_modes_table():
    # Thinking blinks fast, listening slow, output is solid.
    for m in ("q_think", "music_think", "book_think"):
        assert _BLINK_PERIOD_S[m] <= 0.3
    for m in ("q_listen", "music_listen", "book_listen"):
        assert 0.5 <= _BLINK_PERIOD_S[m] <= 1.5
    for m in ("waiting", "boot", "book_paused"):
        assert _BLINK_PERIOD_S[m] >= 1.5
    for solid in ("off", "q_speak", "music_play", "music_speak", "book_read", "book_speak"):
        assert solid not in _BLINK_PERIOD_S


def test_status_leds_init_without_gpio_logs_only():
    leds = StatusLEDs()
    assert leds.mode == "off"
    leds.set_mode("music_play")
    assert leds.mode == "music_play"
    leds.set_mode("error")
    assert leds.mode == "error"
    leds.cleanup()


def test_thinking_mode_starts_a_blink_thread():
    leds = StatusLEDs()
    try:
        leds.set_mode("q_think")
        assert leds.mode == "q_think"
        assert leds._blink_thread is not None
        assert leds._blink_thread.is_alive()
    finally:
        leds.set_mode("off")
        leds.cleanup()


def test_speaking_mode_is_solid_no_blink():
    leds = StatusLEDs()
    try:
        leds.set_mode("q_speak")
        assert leds.mode == "q_speak"
        # No blink thread should be running for solid modes.
        assert leds._blink_thread is None
    finally:
        leds.cleanup()


def test_switching_from_blink_to_solid_stops_thread():
    leds = StatusLEDs()
    try:
        leds.set_mode("q_think")
        assert leds._blink_thread is not None
        leds.set_mode("librarian")
        # Allow the join to settle.
        time.sleep(0.05)
        assert leds._blink_thread is None
    finally:
        leds.cleanup()
