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
  player.py                # threaded playback — miniaudio decode, chunked output
scripts/
  index_music.py           # CLI: index dir, list, search, stats
```

### Catalog (`catalog.py`)
- SQLite at `data/music.db` with `tracks` table (id, title, artist, album, genre, duration, path)
- Tag extraction via mutagen (`easy=True`)
- `index_directory(path)` — scans for `.mp3/.flac/.ogg/.opus/.m4a/.wav/.aac/.wma`
- `search(query)` — LIKE search across title, artist, album
- `random_track()` — `ORDER BY RANDOM() LIMIT 1`

### Player (`player.py`)
- Background `threading.Thread` with `_stop_event` and `_paused` Events
- `play(track, continuous=True)` — starts thread, auto-advances to next random track
- `_play_file()` — decodes via `miniaudio.decode_file()` to float32 mono, applies AM radio
  filter if `ORACLE_MUSIC_RADIO_FILTER=true`, plays in 2-second chunks checking pause/stop
- Volume applied at playback time via `oracle.audio.play_audio()` → pot-based hardware gain

### State machine integration (`oracle/app.py`)
- **Radio mode**: `_ensure_music()` starts continuous playback if not already running
- **Wake word detected**: pause music → one voice turn → resume music
- **Long-press → Librarian**: pause music; long-press back → resume
- **Short-press in Radio**: `_next_track()` skips to random track
- **Standby / power-off**: `_stop_music()` halts playback
- **Shutdown**: `player.close()` stops playback and closes catalog DB

## Settings

```bash
ORACLE_MUSIC_PATH=music                  # relative to project root
ORACLE_MUSIC_DB_PATH=data/music.db
ORACLE_MUSIC_RADIO_FILTER=true           # AM bandpass 300-3400 Hz
```

## Dependencies

```toml
# pyproject.toml [music] extra
music = [
    "mutagen>=1.47",         # tag extraction
    "miniaudio>=1.59",       # decode (libminiaudio, no ffmpeg needed)
]
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
