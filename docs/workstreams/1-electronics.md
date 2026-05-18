# Workstream 1: Electronics & Wiring

Physical hardware: ADS1115-based switches/pot, RGB LED, USB audio routing.
The schematic is at [`docs/wiring/jetson-wiring.svg`](../wiring/jetson-wiring.svg);
the per-net cable schedule is in [`docs/wiring/netlist.md`](../wiring/netlist.md).

## Status

Hardware loop runs end-to-end on the Jetson via `--mode hardware`. All switches
and the volume pot read through the ADS1115 ADC (GPIO input is broken on
JP 6.2.x — Tegra234 INPUT_VALUE loopback bug). Volume pot now controls audio
gain on all playback. Power switch immediately halts mic/speaker/LED on open.

## Scope

- Action button (ADS1115 AIN2) — momentary, short/long-press detection
- Power switch (ADS1115 AIN1) — SPST toggle, virtual on/off (standby halts all I/O)
- Volume potentiometer (ADS1115 AIN0) — quadratic gain curve, applied at playback time
- Status LED — single common-anode RGB LED on BOARD pins 16/18/22
- USB audio device auto-detection
- Physical enclosure / wiring diagram / netlist

## File ownership

```
oracle/hardware/
  __init__.py
  pot.py                 # ADS1115 driver + Potentiometer reader
  switch_adc.py          # DigitalSwitch via ADS1115 (shared ADC singleton)
  button.py              # ActionButton — short/long press events via ADS1115
  leds.py                # StatusLEDs — RGB color = mode (common-anode, BOARD mode)
  power_switch.py        # PowerSwitch — toggle gates the app, aborts all I/O
  volume.py              # VolumeControl — pot % → quadratic gain 0.0–1.0
  audio_routing.py       # USB audio device detection
docs/wiring/
  jetson-wiring.svg      # Stylized schematic
  jetson-schematic.svg   # Detailed schematic
  netlist.md             # Per-net cable schedule + BOM
```

## Settings

```bash
ORACLE_POT_I2C_BUS=7                    # /dev/i2c-7 (header pins 3/5)
ORACLE_POT_ADS1115_ADDR=0x48            # default ADDR-floating
ORACLE_POT_ADS1115_CHANNEL=0            # AIN0 — pot wiper
ORACLE_POWER_SWITCH_ADS1115_CHANNEL=1   # AIN1 — SPST toggle
ORACLE_ACTION_BUTTON_ADS1115_CHANNEL=2  # AIN2 — momentary button
ORACLE_LED_RED_PIN=16                   # BOARD pin 16
ORACLE_LED_GREEN_PIN=18                 # BOARD pin 18
ORACLE_LED_BLUE_PIN=22                  # BOARD pin 22
ORACLE_LONG_PRESS_THRESHOLD=1.0         # seconds
```

## Dependencies

- `Jetson.GPIO` (system package on Jetson) — LED control in BOARD mode
- `smbus2` — I²C communication with ADS1115
- Falls back to log-only on dev machines without hardware

## Interface contract

**Provides** (consumed by Workstream 7 — Orchestration):
- `ActionButton` — bg thread; emits `ButtonEvent(kind="short"|"long")` on `.events: Queue`
- `PowerSwitch` — bg thread; `is_on: bool` + `add_listener(cb)`. Abort callback
  propagates to `record_until_silence()` and `play_audio()` for immediate cutoff.
- `StatusLEDs` — `set_mode("off"|"radio"|"librarian"|"thinking"|"speaking"|"error")`
- `VolumeControl` / `get_volume_control()` — singleton, reads pot gain at playback time
- `find_audio_device()` — for `sounddevice`

**Consumes**: nothing. This workstream has no upstream dependencies.

**Fallback**: when `Jetson.GPIO` or `smbus2` import fails, every module logs a
warning and degrades — button uses keyboard, power switch fixed-on, LEDs log only,
volume fixed at 1.0. This means all other workstreams keep working on any laptop.

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

- [x] Read volume pot via ADS1115, apply to audio playback
- [x] Power-off immediately halts mic, speaker, and LED
- [ ] LED breathing/pulse for "thinking" / "speaking" (currently solid)
- [ ] Audio device hot-plug detection
- [ ] Short-press in Librarian = interrupt current TTS playback
