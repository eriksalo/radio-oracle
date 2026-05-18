"""Hardware integration: RGB LED, action button, power switch, audio, volume."""

from oracle.hardware.button import ActionButton, ButtonEvent
from oracle.hardware.leds import MODE_COLORS, Color, StatusLEDs
from oracle.hardware.power_switch import PowerSwitch
from oracle.hardware.volume import VolumeControl, get_volume_control

__all__ = [
    "ActionButton",
    "ButtonEvent",
    "Color",
    "MODE_COLORS",
    "PowerSwitch",
    "StatusLEDs",
    "VolumeControl",
    "get_volume_control",
]
