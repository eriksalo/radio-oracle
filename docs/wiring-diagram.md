# Radio Oracle — Wiring Diagram & Wire Harness

## System Overview

```
                          ┌─────────────────────────────────┐
                          │    VINTAGE RADIO ENCLOSURE       │
                          │                                  │
    ┌──���────────┐         │  ┌────��──────────────────────┐  │
    │  USB-C    │─────────┼──│  Jetson Orin Nano Super   │  │
    │  Power    │  power  │  │  8GB                      │  │
    │  Supply   │         │  │                           │  │
    └───────────���         │  │  ┌─────────┐  ┌────────┐ │  │
                          │  │  │ 40-Pin  │  │ USB-A  │ │  │
                          │  │  │ Header  │  │ Ports  │ │  │
                          │  │  └────┬────┘  └───┬────┘ │  │
                          │  └───────┼───────────┼──────┘  │
                          │          │           │         │
                          │    ┌─────┴─────┐  ┌──┴───┐    │
                          │    │ Breakout  │  │ USB  │    │
                          │    │ Board     │  │ Audio│    │
                          │    │           │  │      │    │
                          │    │ • ADS1115 │  │ • Mic│    │
                          │    │ • LED     │  │ • Spk│    │
                          │    │ • Resistor│  └──────┘    │
                          │    └─────┬─────┘              │
                          │          │                     │
                          │    ┌─────┴─────────────────┐  │
                          │    │   Front Panel          │  │
                          │    │                        │  │
                          │    │   (○) LED              │  │
                          │    │   [◎] Switched Pot     │  │
                          │    │   (●) Momentary Button │  │
                          │    └────────────────────────┘  │
                          └─────────────────────────────────┘
```

---

## Jetson 40-Pin Header — Pins Used

```
            Jetson Orin Nano Super — 40-Pin GPIO Header
          (component side up, USB ports facing away from you)

                     Pin 2  ●  ● Pin 1
                     Pin 4  ○  ● Pin 3
                     Pin 6  ●  ● Pin 5
                     Pin 8  ○  ○ Pin 7
                    Pin 10  ○  ○ Pin 9
                    Pin 12  ○  ○ Pin 11
                    Pin 14  ○  ○ Pin 13
                    Pin 16  ●  ○ Pin 15
                    Pin 18  ●  ○ Pin 17
                    Pin 20  ○  ○ Pin 19
                    Pin 22  ●  ○ Pin 21
                    Pin 24  ○  ○ Pin 23
                    Pin 26  ○  ○ Pin 25
                    Pin 28  ○  ○ Pin 27
                    Pin 30  ○  ○ Pin 29
                    Pin 32  ○  ○ Pin 31
                    Pin 34  ●  ○ Pin 33
                    Pin 36  ○  ○ Pin 35
                    Pin 38  ○  ○ Pin 37
                    Pin 40  ○  ○ Pin 39

                ● = used     ○ = unused

    ┌──────┬────────┬────────────┬──────────────────────┐
    │ Pin  │ Color  │ Function   │ Connects To          │
    ├──────┼────────┼────────────┼──────────────────────┤
    │  1   │ RED    │ 3.3V       │ Pot high-side        │
    │  2   │ ORG    │ 5V         │ ADS1115 VDD          │
    │  3   │ YEL    │ I2C SDA    │ ADS1115 SDA          │
    │  5   │ GRN    │ I2C SCL    │ ADS1115 SCL          │
    │  6   │ BLK    │ GND        │ Ground bus           │
    │ 16   │ RED-W  │ GPIO       │ RGB LED red (via 330Ω)   │
    │ 18   │ GRN-W  │ GPIO       │ RGB LED green (via 330Ω) │
    │ 22   │ BLU-W  │ GPIO       │ RGB LED blue (via 330Ω)  │
    │ 34   │ BLK    │ GND        │ Ground bus           │
    └──────┴────────┴────────────┴──────────────────────┘
```

---

## Schematic

```
    3.3V (Pin 1) ─── RED ───┐
                             │
    5V   (Pin 2) ─── ORG ───┼──────────────────────────────────────────┐
                             │                                          │
    SDA  (Pin 3) ─── YEL ───┼─────────────────────────────────┐        │
                             │                                  │        │
    SCL  (Pin 5) ─── GRN ───┼────────────────────────┐        │        │
                             │                         │        │        │
    GND  (Pin 6) ─── BLK ───┼───┬──────┬──────┬──────┼────────┼───┐    │
                             │   │      │      │      │        │   │    │
                             │   │      │      │      │        │   │    │
                             │   │      │      │    ┌─┴────────┴───┴────┴──┐
    SWITCHED POTENTIOMETER   │   │      │      │    │      ADS1115         │
    ┌────��─────────────────┐ │   │      │      │    │                      │
    │                      │ │   │      │      │    │  VDD ── 5V (ORG)     │
    │   On/Off Switch      │ │   │      │      │    │  GND ── GND (BLK)   │
    │   ┌──┐               │ │   │      │      │    │  SCL ── SCL (GRN)   │
    │   │  ├── PUR ────────┼─┼───┼──────┼──────┼────│  SDA ── SDA (YEL)   │
    │   │  ├── BLK ────────┼─┼───┤      │      │    │  ADDR ── GND (BLK)  │
    │   └──┘               │ │   │      │      │    │                      │
    │      Pin 33 (GPIO)   │ │   │      │      │    │  A0 ──┐              │
    │      internal pull-up│ │   │      │      │    └───────┼──────────────┘
    │                      │ │   │      │      │            │
    │   Potentiometer      │ │   │      │      │            │
    │   ┌──────────────┐   │ │   │      │      │            │
    │   │  ┌─ RED ─────┼───┼─┘   │      │      │            │
    ��   │  │  (3.3V)   │   │     │      │      │            │
    │   │  │           │   │     │      │      │            │
    │   │  ├─ GRY ─────┼───┼─────┼──────┼──────┼────────────┘
    │   │  │  (wiper)  │   │     │      │      │   to ADS1115 A0
    │   │  │           │   │     │      │      │
    │   │  └─ BLK ─────┼───┼─────┘      │      │
    │   │    (GND)     │   │             │      │
    │   └──────────────┘   │             │      │
    └──────────────────────┘             │      │
                                         │      │
    MOMENTARY BUTTON                     │      │
    ┌──────────────────────┐             │      │
    │   ┌──┐               │             │      │
    │   │  ├── BLU ────────┼─────────────┼──────┼──── Pin 31 (GPIO)
    │   │  │               │             │      │     internal pull-up
    │   │  ├── BLK ────────┼─────────────┘      │
    │   └──┘  (GND)        │                    │
    └──────────────────────┘                    │
                                                │
    RGB LED (common anode)                      │
    ┌──────────────────────────────────────┐    │
    │  Anode ── 3.3V (Pin 1)              │    │
    │                                      │    │
    │  Pin 16 ──┤330Ω├── Red cathode      │    │
    │  Pin 18 ──┤330Ω├── Green cathode    │    │
    │  Pin 22 ──┤330Ω├── Blue cathode     │    │
    │                                      │    │
    │  LOW = lit, HIGH = off              │    │
    └──────────────────────────────────────┘
```

---

## Wire Harness

Nine wires from the 40-pin header fan out to four components. Bundle into two looms for tidy routing.

### Loom A — I2C + Power (to ADS1115 / Pot)

Runs from the top of the 40-pin header (pins 1–6) to the breakout board area.

```
    40-Pin Header                         Breakout Board
    ┌────────────┐                        ┌────────────────────────┐
    │            │   Loom A               │                        │
    │  Pin 1 ────┼── RED ─── 18cm ───────►│ Pot high-side (3.3V)   │
    │  Pin 2 ────┼── ORG ─── 18cm ───────►│ ADS1115 VDD (5V)      │
    │  Pin 3 ────┼── YEL ─── 18cm ───────►│ ADS1115 SDA           │
    │  Pin 5 ────┼── GRN ─── 18cm ───────►│ ADS1115 SCL           │
    │  Pin 6 ────┼── BLK ─── 18cm ───┬───►│ ADS1115 GND + ADDR    │
    │            │                    ├───►│ Pot low-side (GND)     │
    │            │                    └───►│ Switch leg 2 (GND)     │
    └────────────┘                        └────────────────────────┘

    5 wires, ~18cm, bundled with spiral wrap or sleeving
    GND wire splits to 3 destinations at the breakout board
```

### Loom B — RGB LED (pins 16, 18, 22)

Runs from the middle of the 40-pin header to the front panel RGB LED.

```
    40-Pin Header                         Front Panel
    Pin 1  (3.3V) ── 22cm ──────────────► RGB LED anode
    Pin 16 (GPIO) ── 22cm ──┤330Ω├──────► RGB LED red cathode
    Pin 18 (GPIO) ── 22cm ──┤330Ω├──────► RGB LED green cathode
    Pin 22 (GPIO) ── 22cm ──┤330Ω├──────► RGB LED blue cathode

    Common anode: LOW = lit, HIGH = off
```

### Interconnect — Breakout to Front Panel

```
    Breakout Board                        Front Panel
    ┌──────────────┐                      ┌──────────────────┐
    │              │                      │                  │
    │ ADS1115 A0 ──┼── GRY ── 10cm ─────►│ Pot wiper        │
    │              │                      │                  │
    └──────────────┘                      └──────────────────┘

    1 wire, ~10cm (or as needed based on component placement)
```

---

## Wire Color Code

```
    ┌────────┬───────────────────────────────────────┐
    │ Color  │ Signal                                │
    ├────────┼───────────────────────────────────────┤
    │ RED    │ 3.3V power                            │
    │ ORG    │ 5V power                              │
    │ YEL    │ I2C SDA (data)                        │
    │ GRN    │ I2C SCL (clock)                       │
    │ RED-W  │ RGB LED red cathode                    │
    │ GRN-W  │ RGB LED green cathode                  │
    │ BLU-W  │ RGB LED blue cathode                   │
    │ GRY    │ Pot wiper (analog)                     │
    │ BLK    │ Ground (all GND connections)           │
    └────────┴───────────────────────────────────────┘
```

---

## Breakout Board Layout

Small perfboard or breadboard holding the ADS1115 and LED resistor.

```
         ┌─────────────────────────────────────────┐
         │          BREAKOUT BOARD (perfboard)      │
         │                                          │
         │   ┌─────────────────┐                    │
         │   │    ADS1115      │                    │
         │   │    Breakout     │                    │
         │   │                 │                    │
         │   │  VDD  GND  SCL │                    │
         │   │   │    │    │  │                    │
         │   │  SDA  ADDR  A0 │                    │
         │   │   │    │    │  │                    │
         │   └───┼────┼────┼──┘                    │
         │       │    │    │                        │
         │       │    │    └─── GRY → pot wiper     │
         │       │    └──── jumper to GND rail      │
         │       │                                  │
         │   GND rail ━━━━━━━━━━━━━━━━━━━━━━━━━━   │
         │       ▲         ▲         ▲              │
         │       │         │         │              │
         │     from       pot       switch          │
         │     Pin 6     low-side   leg 2           │
         │                                          │
         │   (RGB LED wired directly from pins       │
         │    16/18/22 to front panel, each via 330Ω) │
         └─────────────────────────────────────────┘
```

---

## Connection Checklist

```
    Component          Wire     From              To                  ✓
    ─────────────────────────────────────────────────────────────────────
    ADS1115 VDD        ORG      Pin 2  (5V)       ADS1115 VDD         □
    ADS1115 GND        BLK      Pin 6  (GND)      ADS1115 GND         □
    ADS1115 SCL        GRN      Pin 5  (SCL)      ADS1115 SCL         □
    ADS1115 SDA        YEL      Pin 3  (SDA)      ADS1115 SDA         □
    ADS1115 ADDR       —        jumper on board    ADS1115 GND rail    □
    ADS1115 A0         GRY      ADS1115 A0        Pot wiper           □
    Pot high-side      RED      Pin 1  (3.3V)     Pot pin 1           □
    Pot low-side       BLK      GND rail          Pot pin 3           □
    ADS1115 A1         —        ADS1115 A1        Power switch (10k PU) □
    ADS1115 A2         —        ADS1115 A2        Action button (10k PU) □
    LED anode          RED      Pin 1  (3.3V)     RGB LED anode       □
    LED red cathode    RED-W    Pin 16 (GPIO)     330Ω → LED red      □
    LED green cathode  GRN-W    Pin 18 (GPIO)     330Ω → LED green    □
    LED blue cathode   BLU-W    Pin 22 (GPIO)     330Ω → LED blue     □
    USB mic            —        USB-A port        —                   □
    USB speaker        —        USB-A port        —                   □
    ─────────────────────────────────────────────────────────────────────
    Total wires from 40-pin header: 10
    Total unique pins used: 8 (1, 2, 3, 5, 6, 16, 18, 22)
```

---

## Auto-Boot (J14 Button Header)

```
    J14 Button Header (separate from 40-pin header)
    Located near the carrier board edge

    ┌─────────────────────────────────┐
    │  1  [PWR]     [GND]  2         │
    │  3  [...]     [...]  4         │
    │  5  [DIS_AUTO][GND]  6    ◄── Do NOT jumper (leaves auto-boot ON)
    │  7  [RST]     [GND]  8         │
    │  9  [GND]     [REC]  10        │
    │ 11  [SLP]     [GND]  12        │
    └─────────────────────────────────┘

    Auto-power-on is ENABLED by default (no jumper).
    Jumpering pins 5-6 would DISABLE it.
    Leave J14 alone. Board boots when USB-C power is applied.
```
