# Radio Oracle — Wiring Connection List

Bench-side companion to the diagrams. Two views of the same information:

- **[`jetson-schematic.svg`](jetson-schematic.svg)** — proper EE schematic
  with standard symbols and named net flags. Use this to *verify the design
  is correct*.
- **[`jetson-wiring.svg`](jetson-wiring.svg)** — pictorial / Fritzing-style
  layout with colored wires. Use this to *see what it should look like*.

This file is the *checkbox-driven build sheet*. Print or open it next to
the bench, work top-to-bottom through the matrix, tick each row as you make
the connection.

All Jetson pins use **header position 1–40** (with BCM number in
parentheses). Reference designators (J1, R1–R5, LED1, SW1, SW2, RV1, U1)
match the schematic.

> **Why the switches go through U1 instead of GPIO:** the Tegra234 GPIO
> input register has a loopback bug on JP 6.2.x for the pads we'd otherwise
> use, so SW1 and SW2 are read as analog voltages through the ADS1115
> (10 kΩ pull-up to 3V3 + switch-to-GND ≈ a clean 3.3 V / 0 V binary).
> See `memory/hdr40-pinmux-overlay.md` for the diagnosis.

---

## Bill of materials

| Ref       | Qty | Part                                    | Notes                                      |
|-----------|-----|-----------------------------------------|--------------------------------------------|
| **J1**    | 1   | Jetson Orin Nano dev kit (40-pin J12)   | uses 7 of 40 pins                          |
| **LED1**  | 1   | Common-**anode** RGB LED, 5 mm or 8 mm  | 4 leads, common = longest; A-R-G-B typical |
| **R1**    | 1   | 330 Ω resistor, ¼ W                     | RGB R series resistor (LED1.R → BCM 23)    |
| **R2**    | 1   | 330 Ω resistor, ¼ W                     | RGB G series resistor (LED1.G → BCM 24)    |
| **R3**    | 1   | 330 Ω resistor, ¼ W                     | RGB B series resistor (LED1.B → BCM 25)    |
| **R4**    | 1   | 10 kΩ resistor, ¼ W                     | Pull-up for SW2 (U1.A1 → +3V3)             |
| **R5**    | 1   | 10 kΩ resistor, ¼ W                     | Pull-up for SW1 (U1.A2 → +3V3)             |
| **SW1**   | 1   | Momentary push-button, NO               | 12 mm panel-mount fits a vintage radio     |
| **SW2**   | 1   | SPST toggle / rocker switch             | panel-mount                                |
| **RV1**   | 1   | 10 kΩ linear potentiometer              | 3-terminal: CW · W · CCW                   |
| **U1**    | 1   | ADS1115 I²C ADC breakout                | 5-pin power + 4-channel analog             |
| —         | ~14 | hookup wire, 22 AWG stranded            | ≥ 6 colors recommended                     |
| —         | 1   | breadboard, 400-tie-point or larger     | optional but recommended                   |

---

## Jetson J1 pin map (only pins this design uses)

| Pin | BCM | Net           | Direction | Notes                                       |
|-----|-----|---------------|-----------|---------------------------------------------|
| 1   | —   | `+3V3`        | power-out | feeds U1.VDD, RV1.CW, LED1 common, R4, R5   |
| 3   | 2   | `SDA`         | I²C       | hardware I²C bus 1 data                     |
| 5   | 3   | `SCL`         | I²C       | hardware I²C bus 1 clock                    |
| 6   | —   | `GND`         | ground    | star ground for everything                  |
| 16  | 23  | `BCM23`       | output    | **LOW = R channel lit** (common-anode LED)  |
| 18  | 24  | `BCM24`       | output    | **LOW = G channel lit** (common-anode LED)  |
| 22  | 25  | `BCM25`       | output    | **LOW = B channel lit** (common-anode LED)  |

Pins 11 (BCM 17) and 12 (BCM 18) are **no longer used** — switches read
via U1.A1 / U1.A2 instead. Every other pin on J1 is also unused — leave
them open.

---

## Connection matrix (build sheet — tick as you wire)

Each row is one electrical link. Resistor rows have the resistor itself
*on the wire* (one lead at "From", the other at "To"); they're not separate
endpoints. Wire colors match `jetson-wiring.svg` for easy cross-reference.

| ✓ | #  | Net      | From               | To                   | Inline part    | Wire color |
|---|----|----------|--------------------|----------------------|----------------|------------|
| ☐ |  1 | `+3V3`   | J1.1 (3V3)         | U1.VDD               | —              | red        |
| ☐ |  2 | `+3V3`   | U1.VDD             | RV1.CW               | —              | red        |
| ☐ |  3 | `+3V3`   | U1.VDD             | LED1.anode (common)  | —              | red        |
| ☐ |  4 | `GND`    | J1.6 (GND)         | U1.GND               | —              | black      |
| ☐ |  5 | `GND`    | U1.GND             | RV1.CCW              | —              | black      |
| ☐ |  6 | `GND`    | U1.GND             | SW1.t2               | —              | black      |
| ☐ |  7 | `GND`    | U1.GND             | SW2.t2               | —              | black      |
| ☐ |  8 | `SDA`    | J1.3 (BCM 2)       | U1.SDA               | —              | purple     |
| ☐ |  9 | `SCL`    | J1.5 (BCM 3)       | U1.SCL               | —              | brown      |
| ☐ | 10 | `A1`     | SW2.t1             | U1.A1                | —              | teal       |
| ☐ | 11 | `A1`     | SW2.t1 / U1.A1     | +3V3 rail            | **R4 = 10 kΩ** | teal       |
| ☐ | 12 | `A2`     | SW1.t1             | U1.A2                | —              | yellow     |
| ☐ | 13 | `A2`     | SW1.t1 / U1.A2     | +3V3 rail            | **R5 = 10 kΩ** | yellow     |
| ☐ | 14 | `BCM23`  | J1.16 (BCM 23)     | LED1.R\_cathode      | **R1 = 330 Ω** | hot pink   |
| ☐ | 15 | `BCM24`  | J1.18 (BCM 24)     | LED1.G\_cathode      | **R2 = 330 Ω** | green      |
| ☐ | 16 | `BCM25`  | J1.22 (BCM 25)     | LED1.B\_cathode      | **R3 = 330 Ω** | blue       |
| ☐ | 17 | `A0`     | RV1.W (wiper)      | U1.A0                | —              | orange     |

**Total: 17 wires** + 5 inline resistors. (Rows 4–7 share the `GND` net
and rows 1–3 share the `+3V3` rail; on a breadboard you'd land them on
the power/ground rails rather than running separate wires from J1.1/J1.6.)

The two pull-ups R4/R5 sit between each switch's signal terminal and the
+3V3 rail. The switch shorts that signal to GND when closed: open ≈ 3.3 V
(rail), closed ≈ 0 V, which the ADS1115 thresholds in software.

---

## Per-component pinout (which pin is which)

Quick reference for orienting parts on the bench. "Looking at the part with
its leads pointing down" unless noted.

### LED1 — common-anode RGB LED

```
       (R)   (A)   (G)   (B)
        │     │     │     │       (A = common anode, longest lead)
        ▷     ▷     ▷     ▷       arrowheads = anode side of each diode
```

Common anode goes to **+3V3**. Each colour cathode goes through its
330 Ω series resistor to a BCM output pin — pulling that pin **LOW** lights
the channel (the firmware in `oracle/hardware/leds.py` drives active-LOW
and parks all three pins HIGH at boot so the LED is dark until a mode is
set).

Verify polarity with a 3 V coin cell + 330 Ω in series before installing.
If you only have a common-cathode part, swap LED1's common to GND and
flip the active-low convention in `leds.py`.

### SW1 / SW2 — switches

Two terminals, polarity-free.

- **SW1** is *momentary* (closes only while held) — short press = next /
  action, long press (≥ 1 s) = toggle Librarian mode.
- **SW2** is *latching* (toggles open ↔ closed) — closed = device on.

Both switches are wired in a classic pull-up-to-rail configuration: one
terminal goes to U1's analog input (with R4 / R5 pulling that input up to
+3V3), the other terminal goes to GND. Closing the switch shorts the
input to GND, which the ADS1115 reads as ≈ 0 V and software thresholds to
`closed = True`.

### RV1 — 10 kΩ linear pot

```
   CW  ──┐
         ├──[ resistive track ]
   CCW ──┘
            wiper (W) ─── slides along the track
```
Three terminals in a row on most panel-mount pots; the middle one is the
wiper. CW vs CCW is whichever way you mount the knob — software can flip
the sense.

### U1 — ADS1115 (Adafruit/generic breakout)

```
  VDD  SCL  ALERT  A1   A3
   │    │    │    │    │
  ┌──────────────────────┐
  │      ADS1115         │
  └──────────────────────┘
   │    │    │    │    │
  GND  SDA  ADDR  A0   A2
```

Pin order varies by breakout vendor — **double-check yours by silkscreen**.
Leave `ADDR` floating or tie to GND for I²C address `0x48` (the default
the code expects).

---

## Verification checklist (before powering on)

- [ ] J1.1 = 3.3 V (not 5 V). Pin 2 and pin 4 are 5 V — don't mix them up.
- [ ] LED1 is common-**anode** (long lead = +3V3).
      `leds.py` already drives active-LOW. If you only have a common-cathode
      part, swap LED1.common ↔ GND and invert the polarity in `leds.py`.
- [ ] R1, R2, R3 are all 330 Ω (orange-orange-brown, or 5-band
      orange-orange-black-black-brown). Anything 220 Ω–1 kΩ is safe.
- [ ] R4, R5 are 10 kΩ (brown-black-orange, 5-band brown-black-black-red-
      brown). 4.7 kΩ–47 kΩ is fine; lower wastes more current.
- [ ] SW1's signal goes to **U1.A2** and SW2's signal goes to **U1.A1**
      (NOT to BCM 17 / BCM 18 — those are unused). The pull-up resistor
      sits on the same node as the signal, tied to +3V3.
- [ ] U1 ADDR pin is floating or tied to GND (I²C address `0x48`).
- [ ] Pot CW/CCW orientation only affects which way is "louder"; software
      can invert.
- [ ] Continuity-check every row in the matrix above with a multimeter
      *before* applying power. With nothing pressed, A1 and A2 should both
      sit at ≈ 3.3 V; pressing each switch drops its line to ≈ 0 V.

---

## Smoke tests (after wiring, before assembly)

Run each on the Jetson with the venv active. They each touch one piece
of hardware so you can localise faults.

### RGB LED (rows 14–16 + +3V3 rail)

```bash
python -c "
from oracle.hardware.leds import StatusLEDs
import time
leds = StatusLEDs()
for m in ('radio','librarian','thinking','speaking','error'):
    print(m); leds.set_mode(m); time.sleep(2)
leds.cleanup()
"
```
Expected: green → blue → amber → cyan → red-blink, then off.

### Action button (rows 12–13 — U1.A2 + pull-up)

```bash
python -c "
from oracle.hardware.button import ActionButton
btn = ActionButton(); btn.start()
import time; time.sleep(15)
while not btn.events.empty(): print(btn.events.get_nowait())
btn.cleanup()
"
```
Press a few times in 15 s. Expected: one `ButtonEvent(kind='short', …)`
per quick tap, `kind='long'` for ≥ 1 s holds.

### Power switch (rows 10–11 — U1.A1 + pull-up)

```bash
python -c "
from oracle.hardware.power_switch import PowerSwitch
sw = PowerSwitch(); sw.add_listener(lambda on: print('ON' if on else 'OFF'))
sw.start()
import time; time.sleep(15); sw.cleanup()
"
```
Flip the toggle. Expected: `ON` and `OFF` printed on each edge.

### ADS1115 — all three channels (rows 1–9 + 17)

```bash
python -c "
from oracle.hardware.switch_adc import shared_adc
from oracle.hardware.pot import Potentiometer
adc = shared_adc()
pot = Potentiometer()
import time
for _ in range(20):
    a1 = adc.read_voltage(1)   # SW2 (power toggle)
    a2 = adc.read_voltage(2)   # SW1 (action button)
    p  = pot.read()
    print(f'A1={a1:.3f}V  A2={a2:.3f}V  pot={p.pct:5.1f}% ({p.voltage:.3f}V)')
    time.sleep(0.25)
"
```
Turn the knob, then flip / press each switch in turn. Expected: A1 and A2
idle near 3.3 V and drop to ≈ 0 V when their switch is closed; pot voltage
sweeps roughly 0.0 V → 3.3 V end-to-end.

---

## Off-header (not in the matrix)

| Item        | Where it goes                  |
|-------------|--------------------------------|
| USB DAC     | any Jetson USB-A port          |
| USB mic     | any Jetson USB-A port          |
| Speaker     | DAC's 3.5 mm jack              |
| DC power    | barrel jack on the dev kit     |
