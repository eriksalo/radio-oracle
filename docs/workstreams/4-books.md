# Workstream 4: Books & e-reader

Audiobook-style reading of a local text library. The Oracle reads
chapters aloud, paragraph-by-paragraph, with bookmark/resume so a long
book survives reboots.

## Status

Working end-to-end. 60,030 Project Gutenberg books extracted from ZIM,
indexed into SQLite with chapter/paragraph splitting. Reader plays
paragraph-by-paragraph via Kokoro TTS with bookmark persistence.

This is distinct from RAG (Workstream 2): RAG retrieves *short snippets*
to ground the LLM. Book Reader plays *long-form linear text* through TTS.

## Scope

- Local library of plain-text books (extracted from Gutenberg ZIM)
- SQLite library index: books → chapters → paragraphs
- Gutenberg boilerplate stripping (header/footer regex)
- Chapter detection (regex for "Chapter I", "Book II", "Part III", etc.)
- Reader: stream paragraph-by-paragraph through TTS
- Bookmarking: per-book cursor (chapter + paragraph) persisted across reboots
- ZIM extraction script for bulk conversion

## File ownership

```
oracle/books/
  __init__.py              # exports Book, Bookmark, BookmarkStore, Library, Reader
  library.py               # SQLite-backed book index, text parsing, chapter splitting
  reader.py                # paragraph-by-paragraph TTS playback with pause/resume
  bookmarks.py             # SQLite-backed cursor per book (upsert on every paragraph)
scripts/
  extract_gutenberg_zim.py # Extract English texts from Gutenberg ZIM archive
  index_books.py           # CLI: index dir, list, search, info
```

## Settings

```bash
ORACLE_BOOKS_PATH=data/books
ORACLE_BOOKS_DB_PATH=data/books.db
ORACLE_READING_PARAGRAPH_PAUSE=0.6   # seconds between paragraphs
ORACLE_READING_CHAPTER_PAUSE=2.0     # seconds between chapters
```

## Dependencies

No extra pip deps — plain-text parsing uses stdlib only. ZIM extraction
requires `libzim` (only needed on the machine doing extraction, not on
the Jetson at runtime).

## Interface contract

**Provides** (consumed by Workstream 7 — Orchestration):
- `Library.search(query) → list[Book]`
- `Library.get_book(id) → Book`
- `Library.get_chapter(book_id, chapter_idx) → Chapter`
- `Library.get_paragraph(book_id, chapter_idx, para_idx) → str`
- `Reader.start(book_id)` — resumes from bookmark if exists
- `Reader.read_paragraph()` — TTS one paragraph, advance, save bookmark
- `Reader.read_continuous(should_stop)` — hands-free playback loop
- `Reader.pause()` / `Reader.resume()` / `Reader.stop()`
- `BookmarkStore.get(book_id) → Bookmark` / `.save()` / `.list_in_progress()`

**Consumes**:
- Workstream 5 (TTS) for the actual speech (`KokoroTTS.synthesize()`)

**Audio coordination**: same rule as the music player — Reader owns the
speaker while reading; voice interactions pause the read.

## Standalone exercise

```bash
# Index a books directory
python scripts/index_books.py data/books/

# List indexed books
python scripts/index_books.py --list

# Search for a book
python scripts/index_books.py --search "moby dick"

# Show book details
python scripts/index_books.py --info 42

# Read a chapter without involving voice/LLM
python -c "
from oracle.books.library import Library
from oracle.books.reader import Reader
lib = Library()
book = lib.search('moby dick')[0]
r = Reader(library=lib)
r.start(book.id, chapter_idx=2)
r.read_paragraph()  # reads one paragraph aloud
r.stop()
"
```

## TODO

- [x] `oracle/books/library.py` — scan dir, parse plain-text, chapter splitting
- [x] `oracle/books/reader.py` — paragraph iterator, TTS playback, bookmark save
- [x] `oracle/books/bookmarks.py` — SQLite cursor per book
- [x] `scripts/index_books.py` — CLI indexer
- [x] `scripts/extract_gutenberg_zim.py` — ZIM → plain text extraction
- [x] Index 60k Gutenberg books on Jetson
- [ ] Voice intent prompt additions in `config/persona.toml`
- [ ] Wire into orchestration (voice command: "read me Moby Dick")
- [ ] "Continue reading" on power-on (resume from last bookmark)
- [ ] Short-press in Radio mode = next paragraph (when reader is active)
- [ ] EPUB support (ebooklib + beautifulsoup4)
