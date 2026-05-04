# Workstream 4: Electronics & Wiring

GPIO hardware integration for the vintage radio enclosure.

See `docs/wiring/jetson-wiring.svg` for the schematic.

## Scope

- Action button — momentary, short/long-press detection
- Power switch — SPST toggle that gates the device on/off
- Status LED — single common-cathode RGB LED, color encodes mode
- USB audio device auto-detection and routing
- Physical enclosure integration

## Key Files

```
oracle/hardware/
  __init__.py
  button.py                # ActionButton (short/long press events)
  leds.py                  # StatusLEDs (RGB LED, mode → color)
  power_switch.py          # PowerSwitch (toggle to wake/standby)
  audio_routing.py         # USB audio device detection
oracle/app.py              # Hardware-driven state machine (Standby/Radio/Librarian)
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

## Interface Contract

**Provides to oracle/app.py**:
- `ActionButton` — background thread; emits `ButtonEvent(kind="short"|"long")` on the `.events` queue
- `PowerSwitch` — background thread; `is_on` property + `add_listener(cb)` callbacks on edges
- `StatusLEDs` — `set_mode("off"|"radio"|"librarian"|"thinking"|"speaking"|"error")`
- `find_audio_device()` — returns device index for sounddevice

**State machine** (in `oracle/app.py::OracleApp`):
- `power_switch` open → STANDBY (LED off)
- `power_switch` closed → RADIO (default; LED green)
- long-press button: RADIO ↔ LIBRARIAN (LED blue, runs voice loop)
- short-press in RADIO: next-track placeholder (until music player lands)

**Fallback**: When `Jetson.GPIO` import fails (dev machine), each module logs and degrades:
button → keyboard input, power switch → fixed "on", LEDs → log only. Text REPL and headless
voice mode (`--mode voice`) keep working without hardware.

## Testing

```bash
# Unit tests for color/event mapping
pytest tests/test_leds.py tests/test_button.py

# On Jetson — full hardware loop
python -m oracle --mode hardware
```

## TODO

- [ ] LED breathing/pulse effect for "thinking" / "speaking" (currently solid)
- [ ] Audio device hot-plug detection
- [ ] Volume pot via ADS1115 — read I²C and feed ALSA mixer
- [ ] Short-press in Librarian = interrupt current TTS playback
- [ ] Power-off clean shutdown (currently just enters standby)
