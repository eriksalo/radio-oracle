"""Hardware-driven application: two channels plus a conversation overlay.

The radio is always tuned to one CHANNEL — music (RADIO state) or a book
(READER state). The oracle is not a mode: any wake-word turn can be a
channel command ("next song", "next chapter"), a channel switch ("read my
book", "play music"), or a question — questions interrupt the channel,
answer with a follow-up window, and the channel resumes by itself.

States:
  STANDBY — power switch open. LED off, no audio, button ignored.
  RADIO   — music channel (default). Wake word → chime → say anything.
  READER  — book channel. Same wake-word surface; narration pauses at the
            sentence for the turn and resumes after. Bookmarks persist.
  (LIBRARIAN remains in the State literal for the legacy --mode voice
  path; the hardware flow no longer enters it — "I have a question" is
  just a question turn with a longer follow-up window.)

Transitions:
  power-on               : STANDBY -> RADIO (music starts)
  power-off              : *       -> STANDBY (music stops, all I/O halted)
  long-press button      : RADIO <-> READER (channel toggle)
  short-press button     : RADIO -> next track; READER -> pause/resume;
                           during any voice turn -> barge-in (stop talking)
  double-press button    : RADIO -> next album (AM intro); READER -> next chapter
  wake word              : chime, then one utterance — command, switch, or question
"""

from __future__ import annotations

import asyncio
import time
from queue import Empty
from typing import Literal

from loguru import logger

from config.settings import settings
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
        self._reader_session = None  # lazy init
        self._voice_ctx = None  # set in run() after voice_init
        # Book title/author requested via "read me <title>"; consumed by
        # the reader loop on entry.
        self._pending_book_query: str | None = None
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
            if self._state in ("radio", "reader"):
                self.leds.set_mode("librarian")
            loop.call_soon_threadsafe(self._wake_event.set)

        self._wakeword = WakeWordDetector(on_wake=on_wake)
        self._wakeword.start()

    def _stop_wakeword(self) -> None:
        if self._wakeword is not None:
            self._wakeword.stop()
            self._wakeword = None

    async def run(self) -> None:
        from oracle.core import VoiceContext, voice_init

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
            self._voice_ctx = voice_ctx
            from oracle import volume_bridge

            volume_bridge.start()  # knob works for speech even without music
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

                if self._state == "reader":
                    await self._run_reader(voice_ctx)
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
                # "Ready to listen" chirp — synthesized in memory and
                # played synchronously, so it neither pays the ~3.4s
                # subprocess cost of the old MP3 chime nor leaks its
                # tail into the recording (both killed the previous
                # attempt; see oracle/chime.py).
                if settings.wake_chime:
                    from oracle.chime import play_wake_chime

                    play_wake_chime()

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
                        should_abort=self._make_turn_abort(),
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("dispatch_radio_command failed; recovering to radio")
                    result = DispatchResult(next_mode="radio")
                finally:
                    if self._wakeword:
                        self._wakeword.unmute()

                if result.next_mode == "radio":
                    self.leds.set_mode("radio")
                    if result.resume_channel:
                        self._resume_music()
                else:
                    self._pending_book_query = result.reader_query
                    self._enter(result.next_mode)
                return

            await asyncio.sleep(0.05)

    # ---------------------------------------------------------------- reader

    def _get_reader(self, voice_ctx):
        """Lazily create the reading session (shares the voice TTS)."""
        if self._reader_session is not None:
            return self._reader_session
        try:
            from oracle.books.session import ReaderSession

            session = ReaderSession(tts=voice_ctx.tts)
            if session.book_count() == 0:
                logger.info("Book library empty — reader disabled")
                session.close()
                return None
            self._reader_session = session
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Book reader unavailable: {e}")
            self._reader_session = None
        return self._reader_session

    async def _run_reader(self, voice_ctx) -> None:
        """Book channel: pick a book, read it aloud, service buttons + wake.

        Controls while reading: short press = pause/resume, double press =
        next chapter, long press = back to the music. The wake word stays
        active (same reliability caveat as during music): "librarian" →
        reading pauses mid-sentence → chime → any command or question →
        reading resumes from that paragraph. Book choice by voice on entry
        ("read me Moby Dick" carries the title in; otherwise resume the
        current book, else ask).
        """
        from oracle.commands import DispatchResult
        from oracle.core import speak_text

        session = self._get_reader(voice_ctx)
        if session is None:
            await speak_text(voice_ctx, "The book archive isn't available.")
            self._enter("radio")
            return

        # Presses queued while something else was happening (e.g. button
        # mashing during a slow turn) are stale intents — acting on them
        # here caused rapid radio<->reader flapping. Start clean.
        self._drain_events()

        play_query: str | None = None
        while self._state == "reader" and self.power.is_on:
            query = self._pending_book_query
            self._pending_book_query = None

            # --- book selection (wake muted: we're prompting/announcing) ---
            if self._wakeword:
                self._wakeword.mute()
            try:
                book = None
                if query:
                    book = session.find_book(query)
                    if book is None:
                        await speak_text(voice_ctx, f"I couldn't find {query} in the archive.")
                if book is None and not query:
                    book = session.current_book()
                    if book is not None:
                        logger.info(f"Reader: resuming current book {book.title!r}")
                if book is None:
                    logger.info("Reader: no current book — asking")
                    book = await self._ask_which_book(voice_ctx, session)
                if book is None:
                    logger.info("Reader: no book chosen — back to music")
                    break

                if session.has_bookmark(book.id):
                    announce = f"Resuming {book.title}."
                elif book.author:
                    announce = f"Reading {book.title}, by {book.author}."
                else:
                    announce = f"Reading {book.title}."
                await speak_text(voice_ctx, announce)

                if not session.start(book):
                    await speak_text(voice_ctx, "I couldn't open that book.")
                    break
            finally:
                if self._wakeword:
                    self._wakeword.unmute()

            # --- reading loop: buttons + wake-word interrupts ---
            exit_requested = False
            switch: DispatchResult | None = None

            def should_stop() -> bool:
                return exit_requested or not self.power.is_on

            if self._wake_event is not None:
                self._wake_event.clear()
            read_task = asyncio.create_task(asyncio.to_thread(session.read_continuous, should_stop))
            pending_press: float | None = None
            try:
                while not read_task.done():
                    now = time.monotonic()
                    for evt in self._drain_events():
                        self._state_writer.record_button(evt.kind, evt.duration)
                        if evt.kind == "long":
                            pending_press = None
                            exit_requested = True
                        elif evt.kind == "short":
                            if (
                                pending_press is not None
                                and now - pending_press < self._double_press_window
                            ):
                                pending_press = None
                                if not session.next_chapter():
                                    logger.info("Already at the last chapter")
                            else:
                                pending_press = now
                    if (
                        pending_press is not None
                        and now - pending_press >= self._double_press_window
                    ):
                        pending_press = None
                        paused = session.toggle_pause()
                        self.leds.set_mode("thinking" if paused else "reader")
                    if self._wake_event is not None and self._wake_event.is_set():
                        self._wake_event.clear()
                        result = await self._reader_wake_turn(voice_ctx, session)
                        if result is not None:
                            switch = result
                            exit_requested = True
                    if not self.power.is_on:
                        exit_requested = True
                    await asyncio.sleep(0.05)
            finally:
                exit_requested = True
                await read_task
                session.stop()  # persists the bookmark

            if switch is None:
                break  # finished, long-press, or power-off → music
            if switch.next_mode == "reader" and switch.reader_query:
                self._pending_book_query = switch.reader_query
                continue  # reselect and keep reading
            play_query = switch.play_query
            break

        self._drain_events()  # discard presses accumulated during reading
        if play_query:
            self._start_specific_music(play_query)
        if self._state == "reader":
            self._enter("radio")

    async def _reader_wake_turn(self, voice_ctx, session):
        """One wake-word turn during reading. Returns a DispatchResult when
        the reader loop must exit (channel switch / different book), else
        None (reading continues or stays paused)."""
        from oracle.commands import DispatchResult, dispatch_radio_command

        logger.info("Wake word triggered — book command turn")
        if self._wakeword:
            self._wakeword.mute()
        session.pause()  # aborts narration mid-sentence; bookmark saved
        if settings.wake_chime:
            from oracle.chime import play_wake_chime

            play_wake_chime()
        player = self._get_player()
        try:
            result = await dispatch_radio_command(
                player=player,
                catalog=player._catalog if player is not None else None,
                vc=voice_ctx,
                leds=self.leds,
                should_abort=self._make_turn_abort(),
                context="book",
                reader=session,
            )
        except Exception:  # noqa: BLE001
            logger.exception("book dispatch failed; resuming reading")
            result = DispatchResult("reader")
        finally:
            if self._wakeword:
                self._wakeword.unmute()

        if result.next_mode == "reader" and not result.reader_query:
            if result.resume_channel:
                session.resume()
                self.leds.set_mode("reader")
            else:
                self.leds.set_mode("thinking")  # paused, awaiting button/wake
            return None
        return result

    def _start_specific_music(self, query: str) -> None:
        """Start playback of a searched-for track/artist (channel switch)."""
        player = self._get_player()
        if player is None:
            return
        from oracle.commands import _play_query

        label = _play_query(player, player._catalog, query)
        logger.info(f"Channel switch to music: {query!r} -> {label!r}")

    async def _ask_which_book(self, voice_ctx, session):
        """Prompt for a title/author by voice and search the library."""
        from oracle.audio import record_until_silence
        from oracle.core import speak_text

        await speak_text(voice_ctx, "Which book? Say a title or an author.")
        try:
            audio = record_until_silence(should_abort=lambda: not self.power.is_on)
        except (ValueError, OSError) as e:
            logger.warning(f"Mic unavailable for book choice: {e}")
            return None
        if len(audio) == 0 or not self.power.is_on:
            return None

        self.leds.set_mode("thinking")
        voice_ctx.stt.load()
        try:
            text = voice_ctx.stt.transcribe(audio)
        finally:
            voice_ctx.stt.unload()
        if not text.strip():
            await speak_text(voice_ctx, "I didn't catch that.")
            return None

        logger.info(f"Book request: {text!r}")
        book = session.find_book(text)
        if book is None:
            await speak_text(voice_ctx, f"I couldn't find {text.strip()} in the archive.")
        return book

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
            if old == "reader":
                self._resume_music()
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
                    self._enter("reader")  # channel toggle (resumes current book)
                elif self._state == "reader":
                    self._enter("radio")
            elif evt.kind == "short" and self._state == "radio":
                if (
                    self._pending_short_press is not None
                    and now - self._pending_short_press < self._double_press_window
                ):
                    # Second short press within window → upgrade to next album.
                    self._pending_short_press = None
                    self._next_album()
                    logger.debug("Double press → next album")
                else:
                    # Act immediately — buffering the press added a fixed
                    # 0.4s lag to every single skip. A second press within
                    # the window upgrades the skip to a new album, which is
                    # fine: the caller wanted to move on either way.
                    self._pending_short_press = now
                    self._next_track()

        # Expire the double-press window.
        if (
            self._pending_short_press is not None
            and now - self._pending_short_press >= self._double_press_window
        ):
            self._pending_short_press = None

    def _make_turn_abort(self):
        """Abort check for one radio voice turn: power-off, or any button
        press = barge-in ("stop talking"). The press is consumed here so it
        doesn't later fire next-track; the latch keeps the whole turn
        (recording, LLM stream, TTS queue) aborted once tripped. Called
        from worker threads — Queue.get_nowait is thread-safe."""
        latch = {"hit": False}

        def check() -> bool:
            if not self.power.is_on:
                return True
            if latch["hit"]:
                return True
            try:
                evt = self.button.events.get_nowait()
            except Empty:
                return False
            self._state_writer.record_button(evt.kind, evt.duration)
            logger.info(f"Button {evt.kind}-press during voice turn — interrupting")
            latch["hit"] = True
            return True

        return check

    async def _shutdown(self, voice_ctx) -> None:
        from oracle import volume_bridge
        from oracle.hardware.volume import get_volume_control

        volume_bridge.stop()
        self._stop_music()
        if self._player:
            self._player.close()
        if self._reader_session:
            try:
                self._reader_session.close()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Reader close error: {e}")
        self._stop_wakeword()

        cleanup_ops = (
            self.button.cleanup,
            self.power.cleanup,
            self.leds.cleanup,
            get_volume_control().cleanup,
        )
        for op in cleanup_ops:
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
