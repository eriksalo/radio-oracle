# Radio Oracle — Wiring Netlist

Companion to [`jetson-wiring.svg`](jetson-wiring.svg). Use the SVG to see the
layout; use this file to actually solder/breadboard the connections.

All Jetson pins are referenced by **header position** (1–40) with their **BCM**
GPIO number in parentheses. BCM numbers match `config/settings.py`.

---

## Components (BOM with reference designators)

| Ref     | Part                                | Notes                                      |
|---------|-------------------------------------|--------------------------------------------|
| **J1**  | Jetson Orin Nano 40-pin header (J12)| 2 × 20 male pins                           |
| **LED1**| Common-cathode RGB LED, 5/8 mm      | 4 leads — typical order: R · GND · G · B   |
| **R1–R3**| 330 Ω, ¼ W resistor                | One per RGB channel; orient either way     |
| **SW1** | Momentary push-button, NO           | 2 terminals; panel-mount (≥ 12 mm)         |
| **SW2** | SPST toggle (or rocker) switch      | 2 terminals; panel-mount                   |
| **RV1** | 10 kΩ linear potentiometer          | 3 terminal: CW · Wiper · CCW               |
| **U1**  | ADS1115 I²C ADC breakout            | Adafruit/generic 5-pin power + 4-ch analog |

---

## Jetson J1 pin map (only pins this design uses)

| Pin | BCM | Direction | Function                  | Notes                       |
|-----|-----|-----------|---------------------------|-----------------------------|
| 1   | —   | power-out | +3.3 V                    | Feeds U1.VDD and RV1.CW     |
| 3   | 2   | I²C       | SDA1                      | Hardware I²C bus 1 data     |
| 5   | 3   | I²C       | SCL1                      | Hardware I²C bus 1 clock    |
| 6   | —   | ground    | GND                       | Star ground for everything  |
| 11  | 17  | input     | Power toggle              | Internal pull-up; LOW = on  |
| 12  | 18  | input     | Action button             | Internal pull-up; LOW = pressed |
| 16  | 23  | output    | RGB R drive               | HIGH = R channel lit        |
| 18  | 24  | output    | RGB G drive               | HIGH = G channel lit        |
| 22  | 25  | output    | RGB B drive               | HIGH = B channel lit        |

All other J1 pins are unused.

---

## Nets

A **net** is a set of pins that are electrically the same node (same wire).
The ten nets in this design:

### N1 · `+3V3`  (Jetson 3.3 V rail)
- J1.1 — 3V3 (power source)
- U1.VDD
- RV1.CW (pot left terminal)

### N2 · `GND`  (star ground)
- J1.6 — GND (ground source)
- LED1.cathode (common cathode)
- SW1.t2 (action button, second terminal)
- SW2.t2 (power toggle, second terminal)
- RV1.CCW (pot right terminal)
- U1.GND

### N3 · `I2C_SDA` (BCM 2)
- J1.3
- U1.SDA

### N4 · `I2C_SCL` (BCM 3)
- J1.5
- U1.SCL

### N5 · `POWER_TOGGLE` (BCM 17)
- J1.11
- SW2.t1 (power toggle, first terminal)

### N6 · `ACTION_BUTTON` (BCM 18)
- J1.12
- SW1.t1 (action button, first terminal)

### N7 · `RGB_R` (BCM 23)
- J1.16  ──[ R1 = 330 Ω ]──  LED1.R_anode

### N8 · `RGB_G` (BCM 24)
- J1.18  ──[ R2 = 330 Ω ]──  LED1.G_anode

### N9 · `RGB_B` (BCM 25)
- J1.22  ──[ R3 = 330 Ω ]──  LED1.B_anode

### N10 · `POT_WIPER`
- RV1.W (wiper)
- U1.A0

---

## Cable schedule (physical wires to run)

Each row = one piece of hookup wire. Suggested colors match the SVG legend.
Resistor bodies count as inline parts on the wire, not as endpoints.

| #  | Net           | From               | To                  | Suggested color  |
|----|---------------|--------------------|---------------------|------------------|
| 1  | +3V3          | J1.1               | U1.VDD              | red              |
| 2  | +3V3          | U1.VDD             | RV1.CW              | red (jumper)     |
| 3  | GND           | J1.6               | U1.GND              | black            |
| 4  | GND           | U1.GND             | RV1.CCW             | black (jumper)   |
| 5  | GND           | J1.6 (or U1.GND)   | LED1.cathode        | black            |
| 6  | GND           | J1.6               | SW1.t2              | black            |
| 7  | GND           | J1.6               | SW2.t2              | black            |
| 8  | I2C_SDA       | J1.3               | U1.SDA              | purple           |
| 9  | I2C_SCL       | J1.5               | U1.SCL              | brown            |
| 10 | POWER_TOGGLE  | J1.11              | SW2.t1              | teal             |
| 11 | ACTION_BUTTON | J1.12              | SW1.t1              | yellow           |
| 12 | RGB_R         | J1.16              | R1 (lead a)         | hot pink         |
| 13 | RGB_R         | R1 (lead b)        | LED1.R_anode        | hot pink (short) |
| 14 | RGB_G         | J1.18              | R2 (lead a)         | green            |
| 15 | RGB_G         | R2 (lead b)        | LED1.G_anode        | green (short)    |
| 16 | RGB_B         | J1.22              | R3 (lead a)         | blue             |
| 17 | RGB_B         | R3 (lead b)        | LED1.B_anode        | blue (short)     |
| 18 | POT_WIPER     | RV1.W              | U1.A0               | orange           |

**Wire count: 18** (5 ground, 2 power, 2 I²C, 2 switch signals, 6 RGB driver+anode, 1 pot wiper).

In practice, ground and 3.3 V are often distributed via a breadboard rail
rather than running 5 separate ground wires from J1.6 — the netlist treats
them as one node either way.

---

## Off-header (not in netlist)

- **USB DAC** — plugs into any Jetson USB-A port. The DAC's 3.5 mm jack feeds
  the speaker. No GPIO involvement.
- **Microphone** — USB mic, also on USB-A.
- **Power input** — DC barrel jack on the Jetson dev kit (not GPIO).

---

## Verification checklist

Before powering on:

- [ ] Confirm RGB LED is **common-cathode** (the long lead is GND, not VCC).
      If yours is common-anode, swap LED1.cathode → +3V3 and invert the
      Jetson outputs (active-LOW); see notes in `4-electronics.md`.
- [ ] Resistors are 330 Ω (orange-orange-brown, or 5-band: orange-orange-
      black-black-brown). Anything between 220 Ω and 1 kΩ is safe.
- [ ] SW1 and SW2 are wired to GND (not 3.3 V) — code uses `PUD_UP`.
- [ ] U1 (ADS1115) ADDR pin is left floating or tied to GND for I²C address
      `0x48` (the default the code expects).
- [ ] Pot CW vs CCW: orientation only affects which knob direction is
      "louder"; software can invert.

## Smoke test

```bash
# RGB LED
python -c "
from oracle.hardware.leds import StatusLEDs
import time
leds = StatusLEDs()
for m in ('radio','librarian','thinking','speaking','error'):
    print(m); leds.set_mode(m); time.sleep(2)
leds.cleanup()
"

# Action button (interactive — press it a few times)
python -c "
from oracle.hardware.button import ActionButton
btn = ActionButton(); btn.start()
import time; time.sleep(15)
while not btn.events.empty(): print(btn.events.get_nowait())
btn.cleanup()
"

# Power switch (interactive — flip it once each way)
python -c "
from oracle.hardware.power_switch import PowerSwitch
sw = PowerSwitch(); sw.add_listener(lambda on: print('ON' if on else 'OFF'))
sw.start()
import time; time.sleep(15); sw.cleanup()
"
```
