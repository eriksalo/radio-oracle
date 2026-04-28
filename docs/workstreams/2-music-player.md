# Workstream 2: Music Player

Index a local music library and play tracks on command, with the Oracle acting as a retro radio DJ.

## Scope

- Scan and index music files (MP3, FLAC, OGG) with metadata
- SQLite catalog (artist, album, title, genre, duration, path)
- Playback engine (play, pause, stop, skip, volume)
- LLM intent detection ("play some jazz", "next track", "what's playing")
- AM radio filter applied to music output for vintage feel
- DJ mode: Oracle introduces tracks in-character

## Key Files

```
oracle/music/
  __init__.py
  indexer.py               # Scan + tag extraction -> SQLite
  catalog.py               # Query the music catalog
  player.py                # Playback engine (sounddevice/miniaudio)
scripts/
  ingest_music.py          # CLI: scan music directory, build catalog
```

## Dependencies (new)

Add to pyproject.toml:
```toml
music = [
    "mutagen>=1.47",       # Audio metadata extraction
    "miniaudio>=1.59",     # Lightweight audio playback
]
```

## Settings (new)

```bash
ORACLE_MUSIC_PATH=/opt/radio-oracle/data/music
ORACLE_MUSIC_DB_PATH=/opt/radio-oracle/data/music.db
ORACLE_MUSIC_VOLUME=0.8
```

## Interface Contract

**Integration point**: `core.py` (both text_repl and voice_loop)

The LLM determines user intent. When it detects a music command, core.py delegates to the music player. While music is playing:
- PTT button pauses music, starts listening
- After the voice interaction, music can resume
- "Stop the music" / "quiet" stops playback

**Audio output**: Music player and TTS must not play simultaneously. The player exposes `pause()` / `resume()` so the conversation loop can take over audio when needed.

## TODO

- [ ] `oracle/music/__init__.py` — package init
- [ ] `oracle/music/indexer.py` — scan directory, extract tags with mutagen
- [ ] `oracle/music/catalog.py` — SQLite catalog with search (by genre, artist, mood)
- [ ] `oracle/music/player.py` — playback with pause/resume/stop/volume
- [ ] `scripts/ingest_music.py` — CLI entry point for indexing
- [ ] Intent detection prompt additions in persona.toml
- [ ] Integration in core.py voice_loop and text_repl
- [ ] AM radio filter on music playback
- [ ] DJ announcements between tracks
