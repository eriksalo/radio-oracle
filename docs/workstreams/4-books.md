# Workstream 4: Books & book reader

Audiobook-style reading of a local EPUB/text library. The Oracle reads
chapters aloud, paragraph-by-paragraph, with bookmark/resume so a long
book survives reboots.

## Status

Not started. Stub package will live at `oracle/books/`.

This is distinct from RAG (Workstream 2): RAG retrieves *short snippets*
to ground the LLM. Book Reader plays *long-form linear text* through TTS.

## Scope

- Local library of EPUB / plain-text books
- Library index (title, author, length, last position) in SQLite
- Reader: stream paragraph-by-paragraph through TTS
- Bookmarking: per-book cursor (chapter + paragraph) persisted across reboots
- Voice commands: "read me Moby Dick", "next chapter", "pause", "resume",
  "what was I reading", "bookmark this page"
- Optional: speed control via the volume pot's secondary mode

## File ownership

```
oracle/books/
  __init__.py              # (stub)
  library.py               # (TODO) scan dir, parse EPUB/TXT, build index
  reader.py                # (TODO) paragraph stream + bookmark state
  bookmarks.py             # (TODO) SQLite-backed cursor per book
scripts/
  ingest_books.py          # (TODO) CLI: scan books dir, build the index
```

## Settings (planned)

```bash
ORACLE_BOOKS_PATH=/opt/radio-oracle/data/books
ORACLE_BOOKS_DB_PATH=/opt/radio-oracle/data/books.db
ORACLE_READING_PARAGRAPH_PAUSE=0.6   # seconds between paragraphs
ORACLE_READING_CHAPTER_PAUSE=2.0     # seconds between chapters
```

## Dependencies (planned)

```toml
# pyproject.toml — new [books] extra
books = [
    "ebooklib>=0.18",        # EPUB parsing
    "beautifulsoup4>=4.12",  # XHTML cleanup inside EPUB chapters
]
```

Plain-text books need no extra deps. EPUB is the priority since most public-
domain books on Standard Ebooks / Project Gutenberg ship as EPUB.

## Interface contract (planned)

**Provides** (consumed by Workstream 7 — Orchestration):
- `Library.search(query: str) → list[Book]`
- `Reader.start(book_id)` / `Reader.pause()` / `Reader.resume()` /
  `Reader.next_paragraph()` / `Reader.next_chapter()` / `Reader.stop()`
- `Reader.position(book_id) → (chapter_idx, paragraph_idx)` for "where was I"

**Consumes**:
- Workstream 5 (TTS) for the actual speech
- Workstream 6 (LLM) for intent classification ("read me X" → which book?)

**Audio coordination**: same rule as the music player — Reader owns the
speaker while reading; voice interactions pause the read.

## Standalone exercise (once implemented)

```bash
# Index a books directory
python scripts/ingest_books.py ~/Books/

# Read a chapter without involving voice/LLM
python -c "
from oracle.books.library import Library
from oracle.books.reader import Reader
lib = Library()
book = lib.search('moby dick')[0]
r = Reader()
r.start(book.id, chapter=2)
import time; time.sleep(60); r.pause()
print('Stopped at', r.position(book.id))
"
```

## TODO

- [ ] `oracle/books/library.py` — scan dir, parse EPUB metadata + chapter list
- [ ] `oracle/books/reader.py` — paragraph iterator, TTS playback, callbacks
- [ ] `oracle/books/bookmarks.py` — SQLite cursor per book
- [ ] `scripts/ingest_books.py` — CLI indexer
- [ ] Voice intent prompt additions in `config/persona.toml`
- [ ] Wire into orchestration as a third top-level mode (Standby/Radio/Librarian/**Reader**) — or as a Radio sub-mode, TBD
- [ ] EPUB-to-SSML conversion so TTS handles dialogue/headings correctly
- [ ] "Continue reading" on power-on (resume from last bookmark)
