"""Book library and reader (Workstream 4).

Audiobook-style linear reading of a local EPUB/text library, with bookmark
state per book so a long read survives reboots. Distinct from RAG
(Workstream 2): RAG retrieves snippets to ground the LLM; this package
plays long-form text through TTS.

Status: not started. See docs/workstreams/4-books.md for the design and
file plan. Submodules to be added:
  - library.py    — scan dir, parse EPUB metadata, build index
  - reader.py     — paragraph stream + TTS playback
  - bookmarks.py  — SQLite-backed cursor per book
"""
