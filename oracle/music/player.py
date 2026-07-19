"""Music player — album-based background playback with pause/resume/skip.

Plays MP3s via an mpg123 subprocess writing to PulseAudio. Decoding +
resampling + audio I/O all happen in native C in a separate process, so
this module touches the audio path approximately zero times per sample.
Earlier in-process designs (miniaudio decode + scipy resample + a Python
sounddevice callback) pegged the python process at 100%+ CPU and
underran constantly because the callback contended with the wake-word
STT thread and the i2c-polled volume knob. mpg123 plays the same file
through the same PulseAudio sink at ~1% CPU with zero underruns.

Volume: the physical pot drives the speaker sink's PulseAudio volume
via a background daemon that polls VolumeControl and shells out to
``pactl set-sink-volume`` when the gain changes. mpg123 itself plays at
unity; per-stream volume is handled by Pulse.

Album mode: tracks are grouped by album and played in order. An AM
radio tuning sound plays when the radio first starts and between
albums. Skipping a single track (short press) does *not* replay the
intro; skipping to a new album does.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from loguru import logger

from oracle.music.catalog import Catalog, Track

# Speaker sink. We target Pulse's *default* sink rather than hardcoding
# the USB DAC's full name — the Jieli DAC's profile suffix flips between
# `.stereo-fallback` and `.analog-stereo` depending on whether PortAudio
# has opened it for capture/playback yet, so any hardcoded suffix breaks
# the volume knob the first time STT/TTS runs. The default sink is the
# USB DAC in our setup (music bypasses aec_sink — see docs/SETUP.md §1.6),
# and `@DEFAULT_SINK@` follows it through profile changes.
_SPEAKER_SINK = "@DEFAULT_SINK@"

# AM radio tuning sound — played on first start and between albums.
_INTRO_MP3 = Path(__file__).resolve().parent.parent.parent / "AMradioSound.mp3"

# Polling period for the pot→PA-volume bridge. 100 ms is well below
# human perception of knob lag and trivial for pactl.
_VOLUME_POLL_S = 0.1
# Don't bother updating Pulse unless gain changed by at least this
# fraction; avoids pactl spam when the pot wiggles a few mV.
_VOLUME_DELTA = 0.01


def _set_pa_sink_volume(gain: float) -> None:
    """Set the speaker sink volume in Pulse. gain is 0.0–1.0."""
    pct = max(0, min(100, int(round(gain * 100))))
    proc = subprocess.run(
        ["pactl", "set-sink-volume", _SPEAKER_SINK, f"{pct}%"],
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        # Most common cause: the sink name changed (profile flip, USB
        # reconnect). Log enough to diagnose without spamming on every
        # poll: the volume_loop only calls us when gain *changed*.
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        logger.warning(f"pactl set-sink-volume rc={proc.returncode}: {err[:160]}")


class Player:
    """Plays music tracks through mpg123 → PulseAudio → speaker.

    Runs the playback loop in a background thread; mpg123 itself runs in
    a child process. Pause/resume use SIGSTOP/SIGCONT on the mpg123
    process — instant, no audio thread to babysit.

    Tracks are played album-by-album. An AM radio tuning sound plays on
    first start and between albums for that authentic dial-surfing feel.
    """

    def __init__(self, catalog: Catalog | None = None) -> None:
        self._catalog = catalog or Catalog()
        self._current: Track | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._paused.set()  # starts unpaused (set = playing)
        self._continuous = True
        self._proc: subprocess.Popen | None = None
        self._proc_lock = threading.Lock()
        self._volume_thread: threading.Thread | None = None
        self._volume_stop = threading.Event()
        # Album skip: set to break out of current album in the play thread.
        self._skip_album = threading.Event()
        # Suppress intro: set by next() so a track skip at an album
        # boundary doesn't replay the AM tuning sound.
        self._suppress_intro = False

    @property
    def now_playing(self) -> Track | None:
        return self._current

    @property
    def is_playing(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_paused(self) -> bool:
        return not self._paused.is_set()

    def play(self, track: Track | None = None, continuous: bool = True) -> Track | None:
        """Start playing. Picks a random album if no track given.

        With *continuous=True* (default), automatically advances to the
        next random album when the current one ends — true radio behaviour.
        """
        self.stop()
        self._stop_event.clear()
        self._paused.set()
        self._continuous = continuous
        self._suppress_intro = False
        self._skip_album.clear()
        self._start_volume_bridge()
        self._thread = threading.Thread(
            target=self._play_thread,
            kwargs={"first_track": track, "play_intro": True},
            name="music-player",
            daemon=True,
        )
        self._thread.start()
        return track

    def stop(self) -> None:
        """Stop playback."""
        self._stop_event.set()
        self._skip_album.set()
        self._paused.set()  # unblock if paused
        self._kill_proc()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._stop_volume_bridge()
        if self._current:
            logger.info(f"Stopped: {self._current.title}")
        self._current = None

    def pause(self) -> None:
        """Pause via SIGSTOP — mpg123 halts instantly, frees CPU."""
        self._paused.clear()
        with self._proc_lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.send_signal(signal.SIGSTOP)
                except OSError:
                    pass
        logger.debug("Music paused")

    def resume(self) -> None:
        """Resume via SIGCONT."""
        self._paused.set()
        with self._proc_lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.send_signal(signal.SIGCONT)
                except OSError:
                    pass
        logger.debug("Music resumed")

    def next(self) -> None:
        """Skip to the next track in the current album (no AM intro)."""
        self._suppress_intro = True
        self._kill_proc()

    def next_album(self) -> None:
        """Skip to a new random album (plays AM intro)."""
        self._suppress_intro = False
        self._skip_album.set()
        self._kill_proc()

    # --------------------------------------------------------------- playback

    def _play_thread(
        self,
        *,
        first_track: Track | None,
        play_intro: bool,
    ) -> None:
        """Background thread: play albums continuously."""
        try:
            is_first_album = True
            while not self._stop_event.is_set():
                # --- pick the next album ---
                if is_first_album and first_track and first_track.album:
                    tracks = self._catalog.random_album_tracks()
                    # If a specific first track was given, try to honour it
                    if first_track:
                        for i, t in enumerate(tracks):
                            if t.path == first_track.path:
                                tracks = tracks[i:]
                                break
                        else:
                            tracks = self._catalog.random_album_tracks()
                else:
                    tracks = self._catalog.random_album_tracks()

                if not tracks:
                    logger.warning("No tracks in catalog")
                    break

                album_label = tracks[0].album or tracks[0].artist or "unknown"

                # --- play AM intro if appropriate ---
                if self._stop_event.is_set():
                    break
                if self._suppress_intro:
                    self._suppress_intro = False
                elif play_intro or not is_first_album:
                    logger.info(f"Tuning to: {album_label}")
                    self._play_intro()

                is_first_album = False
                play_intro = True  # always intro after the first album
                self._skip_album.clear()

                # --- play tracks in album ---
                for track in tracks:
                    if self._stop_event.is_set():
                        return
                    if self._skip_album.is_set():
                        self._skip_album.clear()
                        break
                    self._current = track
                    logger.info(f"Playing: {track.artist} — {track.title}")
                    self._play_file(track)
                    if self._stop_event.is_set():
                        return
                    if self._skip_album.is_set():
                        self._skip_album.clear()
                        break
                    # Small gap between tracks in the same album.
                    time.sleep(0.3)

                if not self._continuous:
                    break
                # Brief pause between albums before the next intro.
                time.sleep(0.3)

        except Exception as e:  # noqa: BLE001
            logger.exception(f"Music thread crashed: {e}")
        finally:
            self._current = None

    def _play_intro(self) -> None:
        """Play the AM radio tuning sound through mpg123 → PulseAudio."""
        if not _INTRO_MP3.exists():
            logger.debug(f"AM radio intro not found: {_INTRO_MP3}")
            return
        try:
            proc = subprocess.Popen(
                ["mpg123", "-q", "-o", "pulse", str(_INTRO_MP3)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, FileNotFoundError) as e:
            logger.warning(f"Failed to play AM intro: {e}")
            return
        # Wait for the sound to finish, but bail if stop requested.
        while proc.poll() is None:
            if self._stop_event.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return
            time.sleep(0.05)

    def _play_file(self, track: Track) -> None:
        """Play a single file via mpg123 → PulseAudio. Blocks until done."""
        try:
            proc = subprocess.Popen(
                ["mpg123", "-q", "-o", "pulse", track.path],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )
        except FileNotFoundError:
            logger.error("mpg123 not installed — apt install mpg123")
            return
        except OSError as e:
            logger.warning(f"Failed to spawn mpg123 for {track.path}: {e}")
            return

        with self._proc_lock:
            self._proc = proc

        try:
            while True:
                if self._stop_event.is_set():
                    self._kill_proc()
                    break
                try:
                    rc = proc.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    continue
                if rc != 0 and rc != -signal.SIGTERM and not self._stop_event.is_set():
                    err = ""
                    if proc.stderr:
                        err = proc.stderr.read().decode("utf-8", errors="replace").strip()
                    logger.warning(f"mpg123 exit {rc} on {track.path}: {err[:200]}")
                break
        finally:
            with self._proc_lock:
                self._proc = None

    def _kill_proc(self) -> None:
        with self._proc_lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.send_signal(signal.SIGCONT)
                    self._proc.terminate()
                except OSError:
                    pass
                try:
                    self._proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()

    # ---------------------------------------------------------------- volume

    def _start_volume_bridge(self) -> None:
        """Spawn the pot→PA-volume daemon if not already running."""
        if self._volume_thread is not None and self._volume_thread.is_alive():
            return
        self._volume_stop.clear()
        self._volume_thread = threading.Thread(
            target=self._volume_loop, name="music-volume", daemon=True
        )
        self._volume_thread.start()

    def _stop_volume_bridge(self) -> None:
        self._volume_stop.set()
        if self._volume_thread is not None and self._volume_thread.is_alive():
            self._volume_thread.join(timeout=0.5)
        self._volume_thread = None

    def _volume_loop(self) -> None:
        """Poll the pot and push the result to Pulse when it changes."""
        try:
            from oracle.hardware.volume import get_volume_control

            ctl = get_volume_control()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Volume bridge unavailable: {e}")
            return
        logger.debug(f"Music volume bridge started (initial gain={ctl.gain:.2f})")
        last_applied = -1.0
        while not self._volume_stop.is_set():
            gain = ctl.gain
            if abs(gain - last_applied) >= _VOLUME_DELTA:
                _set_pa_sink_volume(gain)
                last_applied = gain
            self._volume_stop.wait(_VOLUME_POLL_S)
        logger.debug("Music volume bridge stopped")

    def close(self) -> None:
        self.stop()
        self._catalog.close()
