# Workstream 1: Electronics & Wiring

Physical hardware: GPIO buttons/switches, RGB LED, USB audio routing.
The schematic is at [`docs/wiring/jetson-wiring.svg`](../wiring/jetson-wiring.svg);
the per-net cable schedule is in [`docs/wiring/netlist.md`](../wiring/netlist.md).

## Status

Hardware loop runs end-to-end on the Jetson via `--mode hardware`. Solid colors
on the LED; volume-pot reading via I²C is the open work item.

## Scope

- Action button (BCM 18) — momentary, short/long-press detection
- Power switch (BCM 17) — SPST toggle that gates the device on/off
- Status LED — single common-cathode RGB LED on BCM 23/24/25
- Volume potentiometer via ADS1115 I²C ADC (planned)
- USB audio device auto-detection
- Physical enclosure / wiring diagram / netlist

## File ownership

```
oracle/hardware/
  __init__.py
  button.py                # ActionButton — short/long press events
  leds.py                  # StatusLEDs — RGB color = mode
  power_switch.py          # PowerSwitch — toggle gates the app
  audio_routing.py         # USB audio device detection
docs/wiring/
  jetson-wiring.svg        # Stylized schematic
  netlist.md               # Per-net cable schedule + BOM
tests/
  test_leds.py
  test_button.py
```

## Settings

```bash
ORACLE_ACTION_BUTTON_PIN=18
ORACLE_LED_RED_PIN=23
ORACLE_LED_GREEN_PIN=24
ORACLE_LED_BLUE_PIN=25
ORACLE_POWER_SWITCH_PIN=17
ORACLE_LONG_PRESS_THRESHOLD=1.0   # seconds
```

## Dependencies

- `Jetson.GPIO` (system package on Jetson) — falls back to keyboard/log on dev machines
- No pip extras required for the hardware modules themselves

## Interface contract

**Provides** (consumed by Workstream 7 — Orchestration):
- `ActionButton` — bg thread; emits `ButtonEvent(kind="short"|"long")` on `.events: Queue`
- `PowerSwitch` — bg thread; `is_on: bool` + `add_listener(cb)`
- `StatusLEDs` — `set_mode("off"|"radio"|"librarian"|"thinking"|"speaking"|"error")`
- `find_audio_device()` — for `sounddevice`

**Consumes**: nothing. This workstream has no upstream dependencies.

**Fallback**: when `Jetson.GPIO` import fails, every module logs a warning
and degrades — button uses keyboard, power switch fixed-on, LEDs log only.
This means workstreams 5/6/7 keep working on any laptop.

## Standalone exercise

```bash
# Unit tests (any machine, no GPIO needed)
pytest tests/test_leds.py tests/test_button.py

# Smoke-test on Jetson
python -c "
from oracle.hardware.leds import StatusLEDs
import time
leds = StatusLEDs()
for m in ('radio','librarian','thinking','speaking','error'):
    leds.set_mode(m); time.sleep(2)
leds.cleanup()
"
```

The full set of smoke tests (LED, button, power switch) is in
`docs/wiring/netlist.md` under *Smoke test*.

## TODO

- [ ] Read volume pot via ADS1115, feed ALSA mixer
- [ ] LED breathing/pulse for "thinking" / "speaking" (currently solid)
- [ ] Audio device hot-plug detection
- [ ] Short-press in Librarian = interrupt current TTS playback
- [ ] Power-off clean shutdown (currently just enters standby)
