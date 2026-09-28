"""Probe: do the ADS1115 switch channels dip while the speaker plays?

Phantom button presses (and one power-switch drop) all happened while
the radio was speaking or playing music loudly. This samples the button
(AIN2) and power-switch (AIN1) channels at ~50 Hz for a quiet phase and
then while ~20 s of loud speech plays through the real speaker, and
reports min/max voltage and how many samples fell below the "pressed"
threshold (0.8 V) in each phase. Must run with the app stopped (it owns
the ADC):

    SIM_SCRIPT=scripts/probe_adc_audio.py sudo -E scripts/sim_turn.sh
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from config.settings import settings  # noqa: E402
from oracle.hardware.pot import ADS1115  # noqa: E402

CH = {
    "button": settings.action_button_ads1115_channel,
    "power": settings.power_switch_ads1115_channel,
    "pot": settings.pot_ads1115_channel,
}


def sample(adc: ADS1115, seconds: float, label: str) -> None:
    stats = {k: {"n": 0, "min": 9.0, "max": 0.0, "low": 0} for k in CH}
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        for k, ch in CH.items():
            v = adc.read_voltage(ch)
            if v is None:
                continue
            s = stats[k]
            s["n"] += 1
            s["min"] = min(s["min"], v)
            s["max"] = max(s["max"], v)
            if v < 0.8:
                s["low"] += 1
        time.sleep(0.005)
    for k, s in stats.items():
        print(
            f"{label:14s} {k:7s} n={s['n']:4d} min={s['min']:.3f} max={s['max']:.3f} "
            f"below-0.8V={s['low']}",
            flush=True,
        )


def main() -> None:
    adc = ADS1115(bus=settings.pot_i2c_bus, addr=settings.pot_ads1115_addr)
    print("channels:", CH, flush=True)
    sample(adc, 6.0, "quiet")

    from oracle.tts import KokoroTTS, say

    tts = KokoroTTS()
    tts.load()
    text = (
        "Testing the speaker at full volume while I watch the switch inputs. "
        "Call me Ishmael. Some years ago, never mind how long precisely, having little or "
        "no money in my purse, I thought I would sail about a little and see the watery part "
        "of the world."
    )
    settings.tts_peak = 0.95
    th = threading.Thread(target=say, args=(tts, text), daemon=True)
    th.start()
    time.sleep(1.0)
    sample(adc, 18.0, "during speech")
    th.join(timeout=30)
    sample(adc, 4.0, "quiet after")
    adc.cleanup()


if __name__ == "__main__":
    main()
