"""Tests for the music player's control logic (no mpg123, no audio)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from config.settings import settings
from oracle.music import catalog as catalog_mod
from oracle.music import player as player_mod
from oracle.music.catalog import Catalog
from oracle.music.player import Player


@pytest.fixture()
def cat(tmp_path, monkeypatch):
    music = tmp_path / "music"
    music.mkdir()

    def _tags(path: Path):
        return path.stem.split(" - ", 1)[-1], "Artist", path.parent.name, "", 10.0

    monkeypatch.setattr(settings, "music_path", music)
    monkeypatch.setattr(catalog_mod, "_extract_tags", _tags)
    album = music / "Artist" / "Album"
    album.mkdir(parents=True)
    for i, title in enumerate(["One", "Two", "Three"], 1):
        (album / f"{i:02d} - {title}.mp3").write_bytes(b"\0")

    c = Catalog(db_path=tmp_path / "music.db")
    c.index_directory(music)
    yield c
    c.close()


@pytest.fixture()
def player(cat, monkeypatch):
    monkeypatch.setattr(Player, "_start_volume_bridge", lambda self: None)
    monkeypatch.setattr(Player, "_stop_volume_bridge", lambda self: None)
    monkeypatch.setattr(Player, "_play_intro", lambda self: None)
    p = Player(catalog=cat)
    yield p
    p.stop()


def test_plays_album_tracks_in_order(player, monkeypatch):
    played: list[str] = []
    monkeypatch.setattr(Player, "_play_file", lambda self, track: played.append(track.title))
    player._continuous = False
    player._play_thread(first_track=None, play_intro=False)
    assert played == ["One", "Two", "Three"]


def test_stop_event_halts_album(player, monkeypatch):
    played: list[str] = []

    def _play(self, track):
        played.append(track.title)
        self._stop_event.set()

    monkeypatch.setattr(Player, "_play_file", _play)
    player._continuous = False
    player._play_thread(first_track=None, play_intro=False)
    assert played == ["One"]


def test_pause_resume_state(player):
    assert not player.is_paused
    player.pause()
    assert player.is_paused
    player.resume()
    assert not player.is_paused


def test_next_suppresses_intro(player):
    player.next()
    assert player._suppress_intro is True


def test_next_album_sets_skip(player):
    player.next_album()
    assert player._skip_album.is_set()
    assert player._suppress_intro is False


def test_play_and_stop_lifecycle(player, monkeypatch):
    monkeypatch.setattr(Player, "_play_file", lambda self, track: time.sleep(0.02))
    player.play(continuous=False)
    assert player.is_playing
    deadline = time.time() + 2.0
    while player.is_playing and time.time() < deadline:
        time.sleep(0.01)
    assert not player.is_playing
    player.stop()
    assert player.now_playing is None


def test_set_pa_sink_volume_clamps(monkeypatch):
    calls: list[list[str]] = []

    class _Proc:
        returncode = 0
        stderr = b""

    monkeypatch.setattr(
        player_mod.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or _Proc()
    )
    player_mod._set_pa_sink_volume(1.5)
    player_mod._set_pa_sink_volume(-0.2)
    assert calls[0][-1] == "100%"
    assert calls[1][-1] == "0%"
