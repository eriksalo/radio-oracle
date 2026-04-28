# Workstream 4: Electronics & Wiring

GPIO hardware integration for the vintage radio enclosure.

## Scope

- Push-to-talk button (GPIO input with debounce)
- Status LEDs (idle / listening / thinking / speaking)
- USB audio device auto-detection and routing
- Physical enclosure integration
- Power management

## Key Files

```
oracle/hardware/
  __init__.py
  button.py                # GPIO PTT button with debounce
  leds.py                  # Status LED state machine
  audio_routing.py         # USB audio device detection
```

## Settings

```bash
ORACLE_PTT_GPIO_PIN=18
ORACLE_LED_IDLE_PIN=23
ORACLE_LED_LISTEN_PIN=24
ORACLE_LED_THINK_PIN=25
```

## Interface Contract

**Provides to core.py**:
- `Button` — async context manager, yields press/release events
- `LEDController` — `set_state(state: Literal["idle", "listening", "thinking", "speaking"])`
- `find_audio_device()` — returns device index for sounddevice

**Fallback**: When GPIO is unavailable (dev machine), hardware modules should no-op gracefully so text REPL and voice-on-laptop still work.

## Testing

Hardware modules can be tested standalone on the Jetson:

```bash
# Test LEDs cycle through states
python -c "from oracle.hardware.leds import LEDController; ..."

# Test button reads
python -c "from oracle.hardware.button import Button; ..."
```

## TODO

- [ ] LED breathing/pulse effect for "thinking" state
- [ ] Audio device hot-plug detection
- [ ] Rotary encoder for volume knob (if adding one)
- [ ] Power button hold for clean shutdown
- [ ] Wiring diagram in docs/
