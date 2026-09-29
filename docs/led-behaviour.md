# Status LED behaviour

One common-anode RGB LED (header pins 16/18/22, see `docs/wiring-diagram.md`),
driven by `oracle/hardware/leds.py` (`StatusLEDs.set_mode`). Erik's scheme
(2026-09-28): the **colour says what you're dealing with**, the **pattern
says what the radio is doing**.

| colour | meaning |
|---|---|
| off | standby (power switch off) |
| amber | booting (models loading) |
| white | waiting: powered on, idle, nothing playing — say "Librarian" or press the button |
| blue | questions / conversation |
| green | music |
| purple | books |
| red | error |

| pattern | meaning |
|---|---|
| solid | output: speaking, playing, reading |
| fast blink (0.3 s) | thinking (transcribing / LLM / retrieval) |
| slow blink (1 s) | listening: the mic is open, talk now |
| very slow blink (2 s) | idle: booting, waiting, book paused |

Every case, by mode name in the code:

| mode | when | LED |
|---|---|---|
| `off` | standby | off |
| `boot` | service start until the models are loaded | amber, 2 s blink |
| `waiting` | power on, nothing happening (after the welcome, after a stop) | white, 2 s blink |
| `error` | unrecoverable turn error | red, 0.5 s blink |
| `q_listen` | mic open after "Librarian" / button in question mode, follow-up window | blue, 1 s blink |
| `q_think` | STT, retrieval, LLM generating an answer | blue, 0.3 s blink |
| `q_speak` | speaking an answer, greeting, options menu, "about this device" | blue solid |
| `music_play` | music playing | green solid |
| `music_listen` | mic open while music is on (wake or button) | green, 1 s blink |
| `music_think` | classifying / finding what to play | green, 0.3 s blink |
| `music_speak` | announcing a track, confirming a command | green solid |
| `book_read` | reading a book aloud | purple solid |
| `book_listen` | mic open in reader mode | purple, 1 s blink |
| `book_think` | finding a book / chapter, classifying a command | purple, 0.3 s blink |
| `book_speak` | announcing a book / chapter, answering "what am I reading?" | purple solid |
| `book_paused` | reader paused (button) | purple, 2 s blink |

Rules that keep it honest:

- The base colour comes from the *context* of the command, not from what
  it might become: a command heard while music plays is green until it
  starts something else; in reader mode it is purple; otherwise blue.
- Standby always wins (power switch off → `off`, whatever else is running).
- The dashboard (`radio-oracle-diag`) shows the current mode text in the
  activity feed (`PHASE_TEXT` in `oracle/diag/static/index.html`) and can
  drive the LED directly from the Hardware I/O card for bench tests.
