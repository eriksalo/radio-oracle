"""Auto-detect and configure USB audio devices via ALSA."""

from __future__ import annotations

import subprocess

from loguru import logger


def list_audio_devices() -> str:
    """List available ALSA audio devices."""
    try:
        result = subprocess.run(
            ["arecord", "-l"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning(f"Could not list audio devices: {e}")
        return ""


def find_usb_device(device_list: str, direction: str = "capture") -> str | None:
    """Find USB audio device in ALSA listing.

    Returns device string like 'hw:1,0' or None.
    """
    for line in device_list.splitlines():
        if "USB" in line.upper() and "card" in line.lower():
            # Extract card number
            parts = line.split(":")
            if len(parts) >= 2:
                card = parts[0].strip().split()[-1]
                return f"hw:{card},0"
    return None


def configure_default_audio() -> dict[str, str | None]:
    """Auto-detect USB mic and DAC, return device config.

    Returns dict with 'input_device' and 'output_device' keys.
    """
    capture_devices = list_audio_devices()
    try:
        playback_result = subprocess.run(
            ["aplay", "-l"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        playback_devices = playback_result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        playback_devices = ""

    input_dev = find_usb_device(capture_devices, "capture")
    output_dev = find_usb_device(playback_devices, "playback")

    if input_dev:
        logger.info(f"USB input device: {input_dev}")
    else:
        logger.warning("No USB input device found, using default")

    if output_dev:
        logger.info(f"USB output device: {output_dev}")
    else:
        logger.warning("No USB output device found, using default")

    return {"input_device": input_dev, "output_device": output_dev}
