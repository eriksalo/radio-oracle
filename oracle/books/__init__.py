"""Book library and voice e-reader for Radio Oracle.

Audiobook-style linear reading of a local text library, with bookmark
state per book so a long read survives reboots. Distinct from RAG:
RAG retrieves snippets to ground the LLM; this package plays
long-form text through TTS paragraph-by-paragraph.
"""

from oracle.books.bookmarks import Bookmark, BookmarkStore
from oracle.books.library import Book, Library
from oracle.books.reader import Reader

__all__ = ["Book", "Bookmark", "BookmarkStore", "Library", "Reader"]
