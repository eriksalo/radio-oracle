# Workstream 3: Music player

Index a local music library and play tracks continuously in Radio mode.
The Oracle acts as a vintage radio — music plays between voice interactions.

## Status

**Complete.** Catalog indexes 4,037 tracks (297.5 hours) from `/opt/radio-oracle/music/`.
Player runs in a background thread with auto-advance, AM radio filter, and
pause/resume coordinated with the voice pipeline.

## Architecture

```
oracle/music/
  __init__.py              # exports Catalog, Player, Track
  catalog.py               # SQLite catalog — index, search, random pick
  player.py                # threaded playback — mpg123 subprocess → PulseAudio
scripts/
  index_music.py           # CLI: index dir, list, search, stats
```

### Catalog (`catalog.py`)
- SQLite at `data/music.db` with `tracks` table
  (track_id, title, artist, album, genre, duration_sec, filename, filepath_rel)
- Tag extraction via mutagen (`easy=True`)
- `index_directory(path)` — scans for `.mp3/.flac/.ogg/.opus/.m4a/.wav/.aac/.wma`
- `search(query)` — LIKE search across title, artist, album, genre
- `random_album_tracks()` — random album, tracks ordered by filename (carries track numbers)
- Startup schema check fails loudly if the DB shape doesn't match the queries

### Player (`player.py`)
- Playback = an `mpg123 -q -o pulse <file>` subprocess per track — decode,
  resample, and audio I/O all in native C at ~1% CPU. (The earlier in-process
  miniaudio/scipy/sounddevice pipeline pegged 100%+ CPU and underran.)
- Pause/resume = SIGSTOP/SIGCONT on the mpg123 pid — instant
- Album mode: random album, tracks in order, AM tuning sound between albums
- Volume: pot → background daemon → `pactl set-sink-volume @DEFAULT_SINK@`

### State machine integration (`oracle/app.py`)
- **Radio mode**: `_ensure_music()` starts continuous playback if not already running
- **Wake word detected**: pause music → one voice turn → resume music
- **Long-press → Librarian**: pause music; long-press back → resume
- **Short-press in Radio**: `_next_track()` skips immediately (double press → next album)
- **Standby / power-off**: `_stop_music()` halts playback
- **Shutdown**: `player.close()` stops playback and closes catalog DB

## Settings

```bash
ORACLE_MUSIC_PATH=music                  # relative to project root
ORACLE_MUSIC_DB_PATH=data/music.db
```

## Dependencies

```toml
# pyproject.toml [music] extra
music = [
    "mutagen>=1.47",         # tag extraction
]
# plus the mpg123 system package (installed by scripts/setup_jetson.sh)
```

## CLI usage

```bash
# Index a music directory
python scripts/index_music.py ~/Music/

# List all indexed tracks
python scripts/index_music.py --list

# Search
python scripts/index_music.py --search "beatles"

# Stats
python scripts/index_music.py --stats
```

## Future ideas

- [ ] DJ mode: Oracle introduces tracks in-character via TTS between songs
- [ ] LLM intent detection for music requests ("play some jazz")
- [ ] Playlist / queue support
- [ ] Track history (don't repeat recently played)
