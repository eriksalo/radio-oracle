# Workstream 3: Music player

Index a local music library and play tracks on command. The Oracle plays
DJ — introducing tracks in-character — between songs.

## Status

Stub package only (`oracle/music/__init__.py` exists, nothing else).
Radio mode in the orchestration layer logs "next track (placeholder)" on
short-press; this workstream replaces that with real playback.

## Scope

- Scan and index music files (MP3, FLAC, OGG) with metadata
- SQLite catalog (artist, album, title, genre, duration, path)
- Playback engine (play, pause, stop, skip, volume)
- LLM intent detection ("play some jazz", "next track", "what's playing")
- AM-radio filter on music output for the vintage feel
- DJ mode: Oracle introduces tracks in-character between songs

## File ownership

```
oracle/music/
  __init__.py              # (stub)
  indexer.py               # (TODO) scan + tag extraction → SQLite
  catalog.py               # (TODO) query the catalog
  player.py                # (TODO) playback engine
scripts/
  ingest_music.py          # (TODO) CLI: scan music dir, build catalog
```

## Settings (planned)

```bash
ORACLE_MUSIC_PATH=/opt/radio-oracle/data/music
ORACLE_MUSIC_DB_PATH=/opt/radio-oracle/data/music.db
ORACLE_MUSIC_VOLUME=0.8
ORACLE_MUSIC_RADIO_FILTER=true   # apply AM filter to playback
```

## Dependencies (planned)

```toml
# pyproject.toml — already declared as the [music] extra
music = [
    "mutagen>=1.47",         # tag extraction
    "miniaudio>=1.59",       # lightweight playback (libminiaudio)
]
```

`pip install -e ".[music]"` once the modules exist.

## Interface contract (planned)

**Provides** (consumed by Workstream 7 — Orchestration):
- `Player.play(track_id)`, `Player.pause()`, `Player.resume()`,
  `Player.stop()`, `Player.next()`, `Player.set_volume(0..1)`
- `Player.now_playing` → `Track | None`
- `Catalog.search(query: str) → list[Track]` for LLM intent matching

**Consumes**:
- Workstream 5 (TTS) for DJ announcements (`tts.synthesize(...)`)
- Workstream 6 (LLM) for intent classification — "play some jazz" → search
- Workstream 1 (Hardware) for the AM-radio filter (already exists in
  `oracle/audio.py::apply_radio_filter`)

**Audio coordination**: the player owns the speaker while playing. When
voice mode kicks in (long-press → Librarian, or wake word), the player
must `pause()` so TTS can take over. Resume on Librarian exit.

## Standalone exercise (once implemented)

```bash
# Index a music directory
python scripts/ingest_music.py ~/Music/

# Smoke test playback without the rest of the app
python -c "
from oracle.music.player import Player
from oracle.music.catalog import Catalog
cat = Catalog()
tracks = cat.search('jazz')
p = Player(); p.play(tracks[0].id)
import time; time.sleep(15); p.stop()
"
```

## TODO

- [ ] `oracle/music/indexer.py` — scan dir, extract tags via mutagen
- [ ] `oracle/music/catalog.py` — SQLite catalog with text search
- [ ] `oracle/music/player.py` — miniaudio playback with pause/resume/volume
- [ ] `scripts/ingest_music.py` — CLI entry point
- [ ] Persona prompt additions for music intent in `config/persona.toml`
- [ ] Wire Radio mode in `oracle/app.py` to call `Player.next()` on short-press
- [ ] DJ announcement on track change (TTS-prefixed playback)
- [ ] AM-radio filter applied to PCM stream (reuse `apply_radio_filter`)
