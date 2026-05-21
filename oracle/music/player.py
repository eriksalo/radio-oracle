"""Music player — background playback with pause/resume/skip."""

from __future__ import annotations

import threading
import time
from typing import Callable

from loguru import logger

from config.settings import settings
from oracle.music.catalog import Catalog, Track


class Player:
    """Plays music tracks through the configured audio output.

    Runs playback in a background thread. The pot-based volume control
    is applied automatically via ``play_audio`` in ``oracle.audio``.
    """

    def __init__(self, catalog: Catalog | None = None) -> None:
        self._catalog = catalog or Catalog()
        self._current: Track | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._paused.set()  # starts unpaused
        self._continuous = True
        self._lock = threading.Lock()

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
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._current:
            logger.info(f"Stopped: {self._current.title}")
        self._current = None

    def pause(self) -> None:
        self._paused.clear()
        logger.debug("Music paused")

    def resume(self) -> None:
        self._paused.set()
        logger.debug("Music resumed")

    def next(self) -> Track | None:
        """Skip to a random track."""
        return self.play()

    def play_continuous(
        self,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        """Play random tracks in a loop until stopped.

        Blocks the calling thread. Use ``play()`` for background playback.
        """
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
            # Brief pause between tracks
            time.sleep(1.0)
        self._current = None

    def _play_thread(self, track: Track) -> None:
        """Background thread: play track, then auto-advance if continuous."""
        try:
            self._play_file(track)
            # Auto-advance to next random track in continuous mode.
            while self._continuous and not self._stop_event.is_set():
                next_track = self._catalog.random_track()
                if next_track is None:
                    break
                self._current = next_track
                logger.info(f"Playing: {next_track.artist} — {next_track.title}")
                time.sleep(1.0)  # brief pause between tracks
                if self._stop_event.is_set():
                    break
                self._play_file(next_track)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Music thread crashed: {e}")
        finally:
            self._current = None

    def _play_file(self, track: Track) -> None:
        """Decode and play a music file, respecting pause/stop."""
        try:
            import miniaudio  # type: ignore[import-not-found]
        except ImportError:
            logger.error("miniaudio not installed — pip install miniaudio")
            return

        try:
            decoded = miniaudio.decode_file(
                track.path,
                output_format=miniaudio.SampleFormat.FLOAT32,
                nchannels=1,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to decode {track.path}: {e}")
            return

        import numpy as np
        import sounddevice as sd

        from oracle.audio import _get_output_device, _resample_to_playback, apply_radio_filter
        from oracle.hardware.volume import get_volume_control

        samples = np.frombuffer(decoded.samples, dtype=np.float32)
        sample_rate = decoded.sample_rate

        # Apply AM radio filter if enabled
        if settings.music_radio_filter:
            samples = apply_radio_filter(samples, sample_rate)

        # Resample once for the output device
        samples, dst_sr = _resample_to_playback(samples, sample_rate)

        # Stream via a single OutputStream — no gaps between chunks.
        chunk_duration = 0.5  # seconds per write
        chunk_size = int(dst_sr * chunk_duration)
        offset = 0
        volume_ctl = get_volume_control()

        try:
            with sd.OutputStream(
                samplerate=dst_sr,
                channels=1,
                dtype="float32",
                device=_get_output_device(),
                blocksize=chunk_size,
                latency="high",
            ) as stream:
                while offset < len(samples):
                    # Pause gate
                    while not self._paused.is_set():
                        if self._stop_event.is_set():
                            return
                        time.sleep(0.1)

                    if self._stop_event.is_set():
                        return

                    end = min(offset + chunk_size, len(samples))
                    chunk = samples[offset:end].copy()

                    # Apply volume from hardware knob
                    gain = volume_ctl.gain
                    if gain < 1.0:
                        chunk *= gain

                    stream.write(chunk.reshape(-1, 1))
                    offset = end
        except (sd.PortAudioError, OSError) as e:
            logger.warning(f"Audio stream error during playback: {e}")

    def close(self) -> None:
        self.stop()
        self._catalog.close()
