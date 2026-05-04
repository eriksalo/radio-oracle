"""Tests for ActionButton press classification."""

from oracle.hardware.button import ActionButton


def test_classify_threshold():
    btn = ActionButton(long_press_threshold=1.0)
    assert btn.classify(0.0) == "short"
    assert btn.classify(0.5) == "short"
    assert btn.classify(0.99) == "short"
    assert btn.classify(1.0) == "long"
    assert btn.classify(2.5) == "long"


def test_custom_threshold():
    btn = ActionButton(long_press_threshold=0.3)
    assert btn.classify(0.2) == "short"
    assert btn.classify(0.31) == "long"


def test_init_without_gpio_uses_keyboard_fallback():
    # On dev machines this should configure without raising.
    btn = ActionButton()
    assert btn.events.empty()
    btn.cleanup()
