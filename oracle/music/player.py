"""Music player — background playback with pause/resume/skip.

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
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from typing import Callable

from loguru import logger

from oracle.music.catalog import Catalog, Track

# Speaker sink name. This is the real USB DAC, not aec_sink — music
# bypasses AEC entirely (see docs/SETUP.md §1.6).
_SPEAKER_SINK = "alsa_output.usb-Jieli_Technology_UACDemoV1.0_415035313136340C-00.analog-stereo"

# Polling period for the pot→PA-volume bridge. 100 ms is well below
# human perception of knob lag and trivial for pactl.
_VOLUME_POLL_S = 0.1
# Don't bother updating Pulse unless gain changed by at least this
# fraction; avoids pactl spam when the pot wiggles a few mV.
_VOLUME_DELTA = 0.01


def _set_pa_sink_volume(gain: float) -> None:
    """Set the speaker sink volume in Pulse. gain is 0.0–1.0."""
    pct = max(0, min(100, int(round(gain * 100))))
    subprocess.run(
        ["pactl", "set-sink-volume", _SPEAKER_SINK, f"{pct}%"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class Player:
    """Plays music tracks through mpg123 → PulseAudio → speaker.

    Runs the playback loop in a background thread; mpg123 itself runs in
    a child process. Pause/resume use SIGSTOP/SIGCONT on the mpg123
    process — instant, no audio thread to babysit.
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
        """Start playing a track. If None, pick a random one.

        With *continuous=True* (default), automatically advances to the
        next random track when the current one ends — true radio behaviour.
        """
        self.stop()
        if track is None:
            track = self._catalog.random_track()
        if track is None:
            logger.warning("No tracks in catalog")
            return None

        self._current = track
        self._stop_event.clear()
        self._paused.set()
        self._continuous = continuous
        self._start_volume_bridge()
        self._thread = threading.Thread(
            target=self._play_thread, args=(track,), name="music-player", daemon=True
        )
        self._thread.start()
        logger.info(f"Playing: {track.artist} — {track.title}")
        return track

    def stop(self) -> None:
        """Stop playback."""
        self._stop_event.set()
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

    def next(self) -> Track | None:
        """Skip to a random track."""
        return self.play()

    def play_continuous(
        self,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        """Play random tracks in a loop until stopped (blocks)."""
        self._start_volume_bridge()
        try:
            while True:
                if should_stop and should_stop():
                    break
                track = self._catalog.random_track()
                if track is None:
                    logger.warning("No tracks in catalog for continuous play")
                    break
                self._current = track
                self._stop_event.clear()
                logger.info(f"Playing: {track.artist} — {track.title}")
                self._play_file(track)
                if self._stop_event.is_set():
                    break
                time.sleep(1.0)
        finally:
            self._current = None
            self._stop_volume_bridge()

    def _play_thread(self, track: Track) -> None:
        """Background thread: play track, then auto-advance if continuous."""
        try:
            self._play_file(track)
            while self._continuous and not self._stop_event.is_set():
                next_track = self._catalog.random_track()
                if next_track is None:
                    break
                self._current = next_track
                logger.info(f"Playing: {next_track.artist} — {next_track.title}")
                # Small gap between tracks; long enough that PA sees the
                # stream close cleanly, short enough not to feel laggy.
                time.sleep(0.3)
                if self._stop_event.is_set():
                    break
                self._play_file(next_track)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Music thread crashed: {e}")
        finally:
            self._current = None

    def _play_file(self, track: Track) -> None:
        """Play a single file via mpg123 → PulseAudio. Blocks until done."""
        # -q: no banner spam on stdout/stderr
        # -o pulse: route to PulseAudio (uses default sink = real speaker)
        try:
            proc = subprocess.Popen(
                ["mpg123", "-q", "-o", "pulse", track.path],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                # New process group so SIGSTOP/SIGCONT don't bleed.
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

        # Wait for playback to finish, polling for stop. Pause/resume are
        # handled signal-driven in pause()/resume() — no extra work here.
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
                    err = proc.stderr.read().decode("utf-8", errors="replace").strip() if proc.stderr else ""
                    logger.warning(f"mpg123 exit {rc} on {track.path}: {err[:200]}")
                break
        finally:
            with self._proc_lock:
                self._proc = None

    def _kill_proc(self) -> None:
        with self._proc_lock:
            if self._proc and self._proc.poll() is None:
                try:
                    # Make sure we're not stuck in SIGSTOP — resume so
                    # SIGTERM can actually be delivered.
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
        last_applied = -1.0
        while not self._volume_stop.is_set():
            gain = ctl.gain
            if abs(gain - last_applied) >= _VOLUME_DELTA:
                _set_pa_sink_volume(gain)
                last_applied = gain
            self._volume_stop.wait(_VOLUME_POLL_S)

    def close(self) -> None:
        self.stop()
        self._catalog.close()
