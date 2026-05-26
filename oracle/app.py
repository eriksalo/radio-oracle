"""Hardware-driven application: power switch + button + RGB LED state machine.

States:
  STANDBY   — power switch open. LED off, no audio, button ignored.
  RADIO     — power on, default. Music plays; wake word triggers a command turn
              (next song / play artist / "I have a question" / etc.).
  LIBRARIAN — long-press button or wake-phrase "I have a question". Voice
              conversation loop with clarifying questions; music paused.
  READER    — wake-phrase "I'd like to read a book". Book TTS playback; music
              paused. (Stub for now — returns to RADIO immediately.)

Transitions:
  power-on               : STANDBY -> RADIO (music starts)
  power-off              : *       -> STANDBY (music stops, all I/O halted)
  long-press button      : RADIO  <-> LIBRARIAN; READER -> RADIO
  short-press button     : RADIO   -> next track; no-op elsewhere
  double-press button    : RADIO   -> next album (with AM intro); no-op elsewhere
  wake word + voice cmd  : RADIO   -> player action OR transition to LIBRARIAN/READER
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path
from queue import Empty
from typing import Literal

from loguru import logger

# Chime played when the wake word fires, before the voice turn starts.
_WAKE_CHIME = Path(__file__).resolve().parent.parent / "chime-clean-short.mp3"
_SPEAKER_SINK = "alsa_output.usb-Jieli_Technology_UACDemoV1.0_415035313136340C-00.stereo-fallback"

from oracle.hardware import ActionButton, ButtonEvent, PowerSwitch, StatusLEDs
from oracle.state import StateWriter

State = Literal["standby", "radio", "librarian", "reader"]


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
        self.power.add_listener(self._on_power_change)
        self._player = None  # lazy init
        self._wakeword = None  # lazy init
        self._wake_event: asyncio.Event | None = None
        # Double-press detection: buffer the first short press and wait
        # up to _DOUBLE_PRESS_S to see if a second one arrives.
        self._pending_short_press: float | None = None
        self._double_press_window = 0.4  # seconds

    def _get_player(self):
        """Lazily create the music player (only if catalog has tracks)."""
        if self._player is not None:
            return self._player
        try:
            from oracle.music.player import Player

            self._player = Player()
            count = self._player._catalog.count()
            if count > 0:
                logger.info(f"Music player ready ({count} tracks)")
            else:
                logger.info("Music catalog empty — player disabled")
                self._player = None
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Music player unavailable: {e}")
            self._player = None
        return self._player

    def _start_wakeword(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start the always-on wake word detector."""
        from oracle.wakeword import WakeWordDetector

        self._wake_event = asyncio.Event()

        def on_wake() -> None:
            # Flip LED from the detector thread for zero perceived delay.
            # StatusLEDs.set_mode is lock-guarded; GPIO writes are sub-ms.
            # Doing this in the asyncio path queues behind the chime + pause.
            if self._state == "radio":
                self.leds.set_mode("librarian")
            loop.call_soon_threadsafe(self._wake_event.set)

        self._wakeword = WakeWordDetector(on_wake=on_wake)
        self._wakeword.start()

    def _stop_wakeword(self) -> None:
        if self._wakeword is not None:
            self._wakeword.stop()
            self._wakeword = None

    async def run(self) -> None:
        from oracle.core import (
            VoiceContext,
            voice_close,
            voice_init,
            voice_turn,
        )

        loop = asyncio.get_event_loop()

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
            self._start_wakeword(loop)
            self._enter("radio")

            while True:
                if not self.power.is_on:
                    self._enter("standby")
                    if self._wakeword:
                        self._wakeword.mute()
                    while not self.power.is_on:
                        self._drain_events()
                        await asyncio.sleep(0.1)
                    if self._wakeword:
                        self._wakeword.unmute()
                    self._enter("radio")
                    continue

                self._handle_buttons()

                if self._state == "librarian":
                    await voice_turn(
                        voice_ctx,
                        leds=self.leds,
                        should_abort=self._should_exit_librarian,
                    )
                    if self._state == "librarian":
                        self.leds.set_mode("librarian")
                elif self._state == "reader":
                    # Stub: book-reading loop not wired up yet.
                    logger.info("Reader mode: not yet implemented; returning to radio")
                    self._enter("radio")
                elif self._state == "radio":
                    self._ensure_music()
                    await self._radio_wait(voice_ctx)

        except KeyboardInterrupt:
            logger.info("Oracle interrupted")
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Oracle error: {e}")
            self.leds.set_mode("error")
            await asyncio.sleep(2)
        finally:
            await self._shutdown(voice_ctx)

    async def _radio_wait(self, voice_ctx) -> None:
        """Wait for wake word, button, or power-off in radio mode."""
        from oracle.commands import DispatchResult, dispatch_radio_command

        if self._wake_event is None:
            await asyncio.sleep(0.1)
            return

        # Clear any stale wake event
        self._wake_event.clear()

        # Poll: check wake event, buttons, and power switch
        while self._state == "radio" and self.power.is_on:
            self._handle_buttons()
            if self._state != "radio":
                return

            if self._wake_event.is_set():
                self._wake_event.clear()
                logger.info("Wake word triggered — radio command turn")

                # LED was already flipped to blue in the on_wake callback.

                # Mute wake detector during our own voice interaction
                if self._wakeword:
                    self._wakeword.mute()

                self._pause_music()
                self._play_wake_chime()

                player = self._get_player()
                catalog = player._catalog if player is not None else None
                # If dispatch raises (mic, STT, LLM, TTS), we still must
                # unmute the wake detector — otherwise the next wake never
                # fires. Catch locally so a transient failure doesn't tear
                # the whole event loop down.
                try:
                    result = await dispatch_radio_command(
                        player=player,
                        catalog=catalog,
                        vc=voice_ctx,
                        leds=self.leds,
                        should_abort=lambda: not self.power.is_on,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("dispatch_radio_command failed; recovering to radio")
                    result = DispatchResult(next_mode="radio", resume_music=True)
                finally:
                    if self._wakeword:
                        self._wakeword.unmute()

                if result.next_mode == "radio":
                    self.leds.set_mode("radio")
                    if result.resume_music:
                        self._resume_music()
                else:
                    self._enter(result.next_mode)
                return

            await asyncio.sleep(0.05)

    # ---------------------------------------------------------------- music

    def _ensure_music(self) -> None:
        """Start music if not already playing."""
        player = self._get_player()
        if player and not player.is_playing:
            player.play()

    def _pause_music(self) -> None:
        if self._player and self._player.is_playing:
            self._player.pause()

    def _resume_music(self) -> None:
        if self._player and self._player.is_playing:
            self._player.resume()

    def _stop_music(self) -> None:
        if self._player:
            self._player.stop()

    def _play_wake_chime(self) -> None:
        """Fire-and-forget chime through PulseAudio.

        The mp3 is ~3.4 s; blocking on it used to delay recording so the
        user's "next song" got eaten by the chime. We Popen and move on
        — recording starts immediately, and PulseAudio's AEC suppresses
        the chime from the mic input (see [[project-radio-oracle-audio]]
        for the empirical AEC behaviour).
        """
        if not _WAKE_CHIME.exists():
            logger.debug(f"Wake chime not found: {_WAKE_CHIME}")
            return
        try:
            subprocess.Popen(
                ["mpg123", "-q", "-o", "pulse", str(_WAKE_CHIME)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, FileNotFoundError) as e:
            logger.warning(f"Failed to spawn wake chime: {e}")

    def _next_track(self) -> None:
        player = self._get_player()
        if player:
            player.next()

    def _next_album(self) -> None:
        player = self._get_player()
        if player:
            player.next_album()

    # ---------------------------------------------------------------- state

    def _enter(self, state: State) -> None:
        if state == self._state:
            return
        old = self._state
        logger.info(f"Mode: {old} -> {state}")
        self._state = state
        self._state_writer.set_mode(state)

        if state == "standby":
            self._stop_music()
            self.leds.set_mode("off")
        elif state == "radio":
            self.leds.set_mode("radio")
            if old in ("librarian", "reader"):
                self._resume_music()
        elif state == "librarian":
            self._pause_music()
            self.leds.set_mode("librarian")
        elif state == "reader":
            self._pause_music()
            self.leds.set_mode("reader")

    def _on_power_change(self, is_on: bool) -> None:
        """Called from power switch thread — immediately update LED."""
        if not is_on:
            self.leds.set_mode("off")
            logger.info("Power off — LED off (immediate)")

    def _drain_events(self) -> list[ButtonEvent]:
        out: list[ButtonEvent] = []
        while True:
            try:
                out.append(self.button.events.get_nowait())
            except Empty:
                return out

    def _handle_buttons(self) -> None:
        now = time.monotonic()
        for evt in self._drain_events():
            self._state_writer.record_button(evt.kind, evt.duration)
            if evt.kind == "long":
                # Long press cancels any pending short press.
                self._pending_short_press = None
                if self._state == "radio":
                    self._enter("librarian")
                elif self._state in ("librarian", "reader"):
                    self._enter("radio")
            elif evt.kind == "short" and self._state == "radio":
                if (
                    self._pending_short_press is not None
                    and now - self._pending_short_press < self._double_press_window
                ):
                    # Second short press within window → next album.
                    self._pending_short_press = None
                    self._next_album()
                    logger.debug("Double press → next album")
                else:
                    # Buffer this press; fire next_track if no second press comes.
                    self._pending_short_press = now

        # Flush an expired pending short press as a single next-track.
        if (
            self._pending_short_press is not None
            and now - self._pending_short_press >= self._double_press_window
        ):
            self._pending_short_press = None
            if self._state == "radio":
                self._next_track()

    def _should_exit_librarian(self) -> bool:
        if not self.power.is_on:
            return True
        for evt in list(self.button.events.queue):
            if evt.kind == "long":
                return True
        return False

    async def _shutdown(self, voice_ctx) -> None:
        from oracle.hardware.volume import get_volume_control

        self._stop_music()
        if self._player:
            self._player.close()
        self._stop_wakeword()

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
