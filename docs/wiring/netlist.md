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
parentheses). Reference designators (J1, R1–R3, LED1, SW1, SW2, RV1, U1)
match the schematic.

---

## Bill of materials

| Ref       | Qty | Part                                    | Notes                                      |
|-----------|-----|-----------------------------------------|--------------------------------------------|
| **J1**    | 1   | Jetson Orin Nano dev kit (40-pin J12)   | uses 9 of 40 pins                          |
| **LED1**  | 1   | Common-cathode RGB LED, 5 mm or 8 mm    | 4 leads, typical order R-K-G-B             |
| **R1**    | 1   | 330 Ω resistor, ¼ W                     | RGB R series resistor (BCM 23 → LED1.R)   |
| **R2**    | 1   | 330 Ω resistor, ¼ W                     | RGB G series resistor (BCM 24 → LED1.G)   |
| **R3**    | 1   | 330 Ω resistor, ¼ W                     | RGB B series resistor (BCM 25 → LED1.B)   |
| **SW1**   | 1   | Momentary push-button, NO               | 12 mm panel-mount fits a vintage radio     |
| **SW2**   | 1   | SPST toggle / rocker switch             | panel-mount                                |
| **RV1**   | 1   | 10 kΩ linear potentiometer              | 3-terminal: CW · W · CCW                   |
| **U1**    | 1   | ADS1115 I²C ADC breakout                | 5-pin power + 4-channel analog             |
| —         | ~12 | hookup wire, 22 AWG stranded            | ≥ 6 colors recommended                     |
| —         | 1   | breadboard, 400-tie-point or larger     | optional but recommended                   |

---

## Jetson J1 pin map (only pins this design uses)

| Pin | BCM | Net           | Direction | Notes                       |
|-----|-----|---------------|-----------|-----------------------------|
| 1   | —   | `+3V3`        | power-out | feeds U1.VDD and RV1.CW     |
| 3   | 2   | `SDA`         | I²C       | hardware I²C bus 1 data     |
| 5   | 3   | `SCL`         | I²C       | hardware I²C bus 1 clock    |
| 6   | —   | `GND`         | ground    | star ground for everything  |
| 11  | 17  | `BCM17`       | input     | internal pull-up; LOW = on  |
| 12  | 18  | `BCM18`       | input     | internal pull-up; LOW = pressed |
| 16  | 23  | `BCM23`       | output    | HIGH = R channel lit        |
| 18  | 24  | `BCM24`       | output    | HIGH = G channel lit        |
| 22  | 25  | `BCM25`       | output    | HIGH = B channel lit        |

Every other pin on J1 is **unused** — leave them open.

---

## Connection matrix (build sheet — tick as you wire)

Each row is one electrical link. Resistor rows have the resistor itself
*on the wire* (one lead at "From", the other at "To"); they're not separate
endpoints. Wire colors match `jetson-wiring.svg` for easy cross-reference.

| ✓ | #  | Net      | From               | To                   | Inline part   | Wire color |
|---|----|----------|--------------------|----------------------|---------------|------------|
| ☐ |  1 | `+3V3`   | J1.1 (3V3)         | U1.VDD               | —             | red        |
| ☐ |  2 | `+3V3`   | U1.VDD             | RV1.CW               | —             | red        |
| ☐ |  3 | `GND`    | J1.6 (GND)         | U1.GND               | —             | black      |
| ☐ |  4 | `GND`    | U1.GND             | RV1.CCW              | —             | black      |
| ☐ |  5 | `GND`    | U1.GND             | LED1.cathode         | —             | black      |
| ☐ |  6 | `GND`    | U1.GND             | SW1.t2               | —             | black      |
| ☐ |  7 | `GND`    | U1.GND             | SW2.t2               | —             | black      |
| ☐ |  8 | `SDA`    | J1.3 (BCM 2)       | U1.SDA               | —             | purple     |
| ☐ |  9 | `SCL`    | J1.5 (BCM 3)       | U1.SCL               | —             | brown      |
| ☐ | 10 | `BCM17`  | J1.11 (BCM 17)     | SW2.t1               | —             | teal       |
| ☐ | 11 | `BCM18`  | J1.12 (BCM 18)     | SW1.t1               | —             | yellow     |
| ☐ | 12 | `BCM23`  | J1.16 (BCM 23)     | LED1.R\_anode        | **R1 = 330 Ω** | hot pink   |
| ☐ | 13 | `BCM24`  | J1.18 (BCM 24)     | LED1.G\_anode        | **R2 = 330 Ω** | green      |
| ☐ | 14 | `BCM25`  | J1.22 (BCM 25)     | LED1.B\_anode        | **R3 = 330 Ω** | blue       |
| ☐ | 15 | `A0`     | RV1.W (wiper)      | U1.A0                | —             | orange     |

**Total: 15 wires** + 3 inline resistors. (Rows 3–7 share the `GND` net.
On a breadboard you'd land them all on the GND rail rather than running
five separate wires from J1.6.)

---

## Per-component pinout (which pin is which)

Quick reference for orienting parts on the bench. "Looking at the part with
its leads pointing down" unless noted.

### LED1 — common-cathode RGB LED

```
       (R)   (K)   (G)   (B)
        │     │     │     │       (K = common cathode, longest lead)
        ▽     ▽     ▽     ▽
```
Verify with a 3 V coin cell + 330 Ω in series before installing — a
common-anode part will need the wiring inverted.

### SW1 / SW2 — switches

Two terminals, polarity-free. SW1 is *momentary* (closes only while held);
SW2 is *latching* (toggles open ↔ closed).

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
- [ ] LED1 is common-**cathode** (long lead = GND).
      If it's common-anode, swap LED1.cathode ↔ +3V3 and tell the firmware
      it's active-LOW (TODO in `oracle/hardware/leds.py`).
- [ ] R1, R2, R3 are all 330 Ω (orange-orange-brown, or 5-band
      orange-orange-black-black-brown). Anything 220 Ω–1 kΩ is safe.
- [ ] SW1 and SW2 wired between BCM pin and **GND** (not 3.3 V) —
      firmware uses `PUD_UP`.
- [ ] U1 ADDR pin is floating or tied to GND (I²C address `0x48`).
- [ ] Pot CW/CCW orientation only affects which way is "louder"; software
      can invert.
- [ ] Continuity-check every row in the matrix above with a multimeter
      *before* applying power. Ring out every GND row to J1.6.

---

## Smoke tests (after wiring, before assembly)

Run each on the Jetson with the venv active. They each touch one piece
of hardware so you can localise faults.

### RGB LED (rows 12–14 + GND rail)

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

### Action button (row 11)

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

### Power switch (row 10)

```bash
python -c "
from oracle.hardware.power_switch import PowerSwitch
sw = PowerSwitch(); sw.add_listener(lambda on: print('ON' if on else 'OFF'))
sw.start()
import time; time.sleep(15); sw.cleanup()
"
```
Flip the toggle. Expected: `ON` and `OFF` printed on each edge.

### ADS1115 + pot (rows 1–4 + 8–9 + 15)

```bash
python -c "
import board, busio
from adafruit_ads1x15.ads1115 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS1115(i2c)
chan = AnalogIn(ads, 0)   # A0 = pot wiper
for _ in range(20):
    print(f'pot = {chan.value:5d}  ({chan.voltage:.3f} V)')
    __import__('time').sleep(0.25)
"
```
Turn the knob. Expected: voltage sweeps roughly 0.0 V → 3.3 V end-to-end.

---

## Off-header (not in the matrix)

| Item        | Where it goes                  |
|-------------|--------------------------------|
| USB DAC     | any Jetson USB-A port          |
| USB mic     | any Jetson USB-A port          |
| Speaker     | DAC's 3.5 mm jack              |
| DC power    | barrel jack on the dev kit     |
