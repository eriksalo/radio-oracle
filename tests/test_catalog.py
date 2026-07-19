"""Tests for the music catalog — schema, indexing, and query round-trips."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from config.settings import settings
from oracle.music import catalog as catalog_mod
from oracle.music.catalog import Catalog


@pytest.fixture()
def music_dir(tmp_path, monkeypatch):
    d = tmp_path / "music"
    d.mkdir()
    monkeypatch.setattr(settings, "music_path", d)
    return d


@pytest.fixture()
def fake_tags(monkeypatch):
    """Derive tags from the file path: music/<artist>/<album>/<NN - title>.mp3."""

    def _tags(path: Path):
        artist = path.parent.parent.name
        album = path.parent.name
        title = path.stem.split(" - ", 1)[-1]
        return title, artist, album, "", 180.0

    monkeypatch.setattr(catalog_mod, "_extract_tags", _tags)


def _touch_album(music_dir: Path, artist: str, album: str, titles: list[str]) -> None:
    d = music_dir / artist / album
    d.mkdir(parents=True)
    for i, title in enumerate(titles, 1):
        (d / f"{i:02d} - {title}.mp3").write_bytes(b"\0")


@pytest.fixture()
def cat(music_dir, fake_tags, tmp_path):
    _touch_album(music_dir, "Zebra", "Stripes", ["Charlie", "Alpha", "Bravo"])
    _touch_album(music_dir, "Aardvark", "Burrow", ["Dig"])
    c = Catalog(db_path=tmp_path / "music.db")
    c.index_directory(music_dir)
    yield c
    c.close()


def test_index_and_list_round_trip(cat):
    tracks = cat.list_tracks()
    assert len(tracks) == 4
    assert {t.artist for t in tracks} == {"Zebra", "Aardvark"}
    assert all(t.duration == 180.0 for t in tracks)


def test_get_track_by_id(cat):
    track = cat.list_tracks()[0]
    assert cat.get_track(track.id).path == track.path
    assert cat.get_track("nonexistent") is None


def test_search_matches_artist_and_title(cat):
    assert len(cat.search("zebra")) == 3
    assert len(cat.search("dig")) == 1
    assert cat.search("nomatch") == []


def test_album_tracks_in_filename_order(cat, music_dir):
    # Filename prefixes 01/02/03 carry track order; titles are shuffled.
    rows = cat._conn.execute(
        f"{catalog_mod._TRACK_SELECT} WHERE album = ? ORDER BY filename", ("Stripes",)
    ).fetchall()
    titles = [r["title"] for r in rows]
    assert titles == ["Charlie", "Alpha", "Bravo"]


def test_random_album_tracks_returns_whole_album(cat):
    tracks = cat.random_album_tracks()
    assert len(tracks) in (1, 3)
    assert len({t.album for t in tracks}) == 1


def test_reindex_is_idempotent(cat, music_dir):
    assert cat.index_directory(music_dir) == 0
    assert cat.count() == 4


def test_paths_resolve_to_existing_files(cat):
    for t in cat.list_tracks():
        p = Path(t.path)
        assert p.is_absolute()
        assert p.exists()


def test_stored_paths_are_relative_to_music_dir(cat):
    rows = cat._conn.execute("SELECT filepath_rel FROM tracks").fetchall()
    for r in rows:
        assert not Path(r["filepath_rel"]).is_absolute()


def test_legacy_jetson_schema_accepted(tmp_path, music_dir):
    """A pre-existing DB with the deployed 16-column-ish schema must work."""
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE tracks (track_id TEXT PRIMARY KEY, title TEXT, artist TEXT,"
        " album TEXT, genre TEXT, duration_sec REAL, filename TEXT,"
        " filepath_rel TEXT, bitrate INTEGER, sample_rate INTEGER)"
    )
    f = music_dir / "song.mp3"
    f.write_bytes(b"\0")
    conn.execute(
        "INSERT INTO tracks (track_id, title, artist, album, genre, duration_sec,"
        " filename, filepath_rel) VALUES ('abc', 'Song', 'Artist', 'Album', '',"
        " 60.0, 'song.mp3', ?)",
        (str(f),),
    )
    conn.commit()
    conn.close()

    c = Catalog(db_path=db)
    tracks = c.list_tracks()
    assert len(tracks) == 1
    assert tracks[0].id == "abc"
    assert tracks[0].path == str(f)
    c.close()


def test_incompatible_schema_raises(tmp_path):
    """The old repo-created schema (id/duration/path) must fail loudly."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE tracks (id INTEGER PRIMARY KEY, title TEXT, artist TEXT,"
        " album TEXT, genre TEXT, duration REAL, path TEXT)"
    )
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="incompatible tracks schema"):
        Catalog(db_path=db)
