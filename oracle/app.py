"""Hardware-driven application: power switch + button + RGB LED state machine.

States:
  STANDBY   — power switch open. LED off, no audio, button ignored.
  RADIO     — power on, default. Music player (placeholder until implemented).
  LIBRARIAN — long-press toggles into here. Voice conversation loop.

Transitions:
  power-on          : STANDBY  -> RADIO
  power-off         : *        -> STANDBY
  long-press button : RADIO   <-> LIBRARIAN
  short-press button: RADIO    -> "next track" (placeholder); no-op in LIBRARIAN
"""

from __future__ import annotations

import asyncio
from queue import Empty
from typing import Literal

from loguru import logger

from oracle.hardware import ActionButton, ButtonEvent, PowerSwitch, StatusLEDs
from oracle.state import StateWriter

State = Literal["standby", "radio", "librarian"]


class OracleApp:
    """Top-level hardware-driven event loop."""

    def __init__(self) -> None:
        self.leds = StatusLEDs()
        self.button = ActionButton()
        self.power = PowerSwitch()
        self._state: State = "standby"
        self._state_writer = StateWriter()
        self._state_writer.set_mode(self._state)
        self._state_writer.set_power(self.power.is_on)
        self.power.add_listener(self._state_writer.set_power)

    async def run(self) -> None:
        from oracle.core import (
            VoiceContext,
            voice_close,
            voice_init,
            voice_turn,
            wake_word_listen,
        )

        self.button.start()
        self.power.start()
        self.leds.set_mode("off")

        voice_ctx: VoiceContext | None = None

        try:
            # Wait for power-on before doing anything expensive.
            logger.info("Waiting for power switch (close to start)...")
            while not self.power.is_on:
                self._drain_events()
                await asyncio.sleep(0.1)

            voice_ctx = await voice_init()
            self._enter("radio")

            while True:
                if not self.power.is_on:
                    self._enter("standby")
                    while not self.power.is_on:
                        self._drain_events()
                        await asyncio.sleep(0.1)
                    self._enter("radio")
                    continue

                self._handle_buttons()

                if self._state == "librarian":
                    await voice_turn(
                        voice_ctx,
                        leds=self.leds,
                        should_abort=self._should_exit_librarian,
                    )
                    # Re-assert color after a turn (it may have ended in
                    # "speaking"/"thinking"); the next loop iteration will
                    # honor any pending state change from buttons.
                    if self._state == "librarian":
                        self.leds.set_mode("librarian")
                elif self._state == "radio":
                    # Listen for wake word; on detection, do one voice turn.
                    remainder = await wake_word_listen(
                        voice_ctx,
                        leds=self.leds,
                        should_abort=lambda: not self.power.is_on,
                    )
                    self._handle_buttons()
                    if remainder is not None and self._state == "radio":
                        self.leds.set_mode("librarian")
                        if remainder:
                            await voice_turn(
                                voice_ctx,
                                leds=self.leds,
                                pre_text=remainder,
                                should_abort=lambda: not self.power.is_on,
                            )
                        else:
                            await voice_turn(
                                voice_ctx,
                                leds=self.leds,
                                should_abort=lambda: not self.power.is_on,
                            )
                        self.leds.set_mode("radio")
        except KeyboardInterrupt:
            logger.info("Oracle interrupted")
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Oracle error: {e}")
            self.leds.set_mode("error")
            await asyncio.sleep(2)
        finally:
            await self._shutdown(voice_ctx)

    # ---------------------------------------------------------------- state

    def _enter(self, state: State) -> None:
        if state == self._state:
            return
        logger.info(f"Mode: {self._state} -> {state}")
        self._state = state
        self._state_writer.set_mode(state)
        if state == "standby":
            self.leds.set_mode("off")
        elif state == "radio":
            self.leds.set_mode("radio")
        elif state == "librarian":
            self.leds.set_mode("librarian")

    def _drain_events(self) -> list[ButtonEvent]:
        out: list[ButtonEvent] = []
        while True:
            try:
                out.append(self.button.events.get_nowait())
            except Empty:
                return out

    def _handle_buttons(self) -> None:
        for evt in self._drain_events():
            self._state_writer.record_button(evt.kind, evt.duration)
            if evt.kind == "long":
                if self._state == "radio":
                    self._enter("librarian")
                elif self._state == "librarian":
                    self._enter("radio")
            elif evt.kind == "short":
                if self._state == "radio":
                    logger.info("Radio: next track (placeholder)")

    def _should_exit_librarian(self) -> bool:
        # Peek (not pop) — _handle_buttons drains the queue at top of loop.
        if not self.power.is_on:
            return True
        for evt in list(self.button.events.queue):
            if evt.kind == "long":
                return True
        return False

    async def _shutdown(self, voice_ctx) -> None:
        from oracle.hardware.volume import get_volume_control

        for op in (self.button.cleanup, self.power.cleanup, self.leds.cleanup, get_volume_control().cleanup):
            try:
                op()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Cleanup error in {op.__qualname__}: {e}")
        if voice_ctx is not None:
            try:
                from oracle.core import voice_close

                await voice_close(voice_ctx)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Voice close error: {e}")
        self._state_writer.clear()
