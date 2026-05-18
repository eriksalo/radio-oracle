#!/usr/bin/env python3
"""Extract author metadata from Gutenberg text headers.

Scans the first N lines of each book for patterns like:
  "by Author Name"
  "BY AUTHOR NAME"
  "Author: Author Name"

Updates the books.db with extracted authors.

Usage:
    python scripts/extract_book_authors.py             # run extraction
    python scripts/extract_book_authors.py --dry-run   # preview without writing
    python scripts/extract_book_authors.py --stats      # show current author stats
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

from loguru import logger

from config.settings import settings

# Lines to scan from the start of each file.
SCAN_LINES = 40

# Patterns to match author lines. Order matters — first match wins.
# Each yields a group(1) with the raw author string.
AUTHOR_PATTERNS = [
    # "by Author Name" on its own line (case-insensitive)
    re.compile(r"^\s*by\s+(.+)$", re.IGNORECASE),
    # "Author: Author Name"
    re.compile(r"^\s*author:\s*(.+)$", re.IGNORECASE),
    # "Written by Author Name"
    re.compile(r"^\s*written\s+by\s+(.+)$", re.IGNORECASE),
    # "Translated by" — use translator as a fallback
    re.compile(r"^\s*translated\s+by\s+(.+)$", re.IGNORECASE),
    # "Edited by"
    re.compile(r"^\s*edited\s+by\s+(.+)$", re.IGNORECASE),
]

# Junk strings that look like authors but aren't.
REJECT = {
    "the author",
    "the same author",
    "anonymous",
    "unknown",
    "various",
    "various authors",
    "the author of",
    "a lady",
    "a gentleman",
    "",
}

# Suffixes/noise to strip from captured author strings.
STRIP_SUFFIXES = [
    re.compile(r",?\s*author of\b.*$", re.IGNORECASE),
    re.compile(r",?\s*with .*$", re.IGNORECASE),
    re.compile(r",?\s*illustrated by\b.*$", re.IGNORECASE),
    re.compile(r"\s*\[.*\]\s*$"),
    re.compile(r"\s*\(.*\)\s*$"),
    re.compile(r"[.,;:]+\s*$"),
]


def clean_author(raw: str) -> str | None:
    """Clean and validate an extracted author string."""
    s = raw.strip()
    # Strip markdown/formatting
    s = s.strip("*_#")
    # Apply suffix strippers
    for pat in STRIP_SUFFIXES:
        s = pat.sub("", s)
    s = s.strip().strip(".,;:")
    # Reject known junk
    if s.lower() in REJECT:
        return None
    # Reject if too short or too long
    if len(s) < 3 or len(s) > 120:
        return None
    # Reject if it looks like a title (all caps and > 40 chars likely a title)
    if s.isupper() and len(s) > 40:
        return None
    # Reject lines that are clearly not names (contain certain keywords)
    lower = s.lower()
    for word in ("chapter", "project gutenberg", "ebook", "contents", "http", "www."):
        if word in lower:
            return None
    return s


_BY_STANDALONE = re.compile(r"^\s*by\s*$", re.IGNORECASE)
_COPYRIGHT_LINE = re.compile(r"copyright|published|press of|printed|entered according", re.IGNORECASE)
_PUBLISHER_WORDS = {
    "company", "co.", "inc.", "inc", "ltd", "ltd.", "press", "publishers",
    "publishing", "sons", "brothers", "bros", "house", "books", "edition",
    "editions", "library", "society", "association", "institute", "university",
    "printshop", "printer", "printers",
}


def _looks_like_publisher(name: str) -> bool:
    """Reject strings that look like publisher names rather than authors."""
    words = set(name.lower().split())
    return bool(words & _PUBLISHER_WORDS)


def extract_author(filepath: Path) -> str | None:
    """Extract author from the first SCAN_LINES of a Gutenberg text file."""
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= SCAN_LINES:
                    break
                lines.append(line.rstrip())
    except OSError:
        return None

    for i, line in enumerate(lines):
        # Skip lines that are part of copyright/publisher blocks
        if _COPYRIGHT_LINE.search(line):
            continue

        # Handle "By" on its own line — author is on the next non-empty line
        if _BY_STANDALONE.match(line):
            for j in range(i + 1, min(i + 3, len(lines))):
                candidate = lines[j].strip()
                if candidate:
                    author = clean_author(candidate)
                    if author and not _looks_like_publisher(author):
                        return author
                    break
            continue

        # Standard same-line patterns
        for pat in AUTHOR_PATTERNS:
            m = pat.match(line)
            if m:
                author = clean_author(m.group(1))
                if author and not _looks_like_publisher(author):
                    return author
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract author metadata from Gutenberg books")
    parser.add_argument("--dry-run", action="store_true", help="Preview without updating DB")
    parser.add_argument("--stats", action="store_true", help="Show current author stats")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of books to process (0=all)")
    args = parser.parse_args()

    db_path = settings.books_db_path
    conn = sqlite3.connect(db_path)

    if args.stats:
        total = conn.execute("SELECT count(*) FROM books").fetchone()[0]
        with_author = conn.execute("SELECT count(*) FROM books WHERE author IS NOT NULL AND author != ''").fetchone()[0]
        without = total - with_author
        print(f"Total books:    {total}")
        print(f"With author:    {with_author}")
        print(f"Without author: {without}")
        if with_author > 0:
            print(f"\nTop 20 authors:")
            for r in conn.execute(
                "SELECT author, count(*) FROM books WHERE author IS NOT NULL AND author != '' "
                "GROUP BY 1 ORDER BY 2 DESC LIMIT 20"
            ):
                print(f"  {r[1]:5d}  {r[0]}")
        conn.close()
        return

    # Get books missing authors
    query = "SELECT id, path FROM books WHERE author IS NULL OR author = ''"
    if args.limit:
        query += f" LIMIT {args.limit}"
    rows = conn.execute(query).fetchall()
    logger.info(f"Processing {len(rows)} books without authors")

    found = 0
    updated = 0
    samples: list[tuple[str, str]] = []

    for book_id, path in rows:
        filepath = Path(path)
        if not filepath.is_absolute():
            filepath = Path("/opt/radio-oracle") / filepath

        author = extract_author(filepath)
        if author:
            found += 1
            if len(samples) < 20:
                samples.append((author, path))
            if not args.dry_run:
                conn.execute("UPDATE books SET author = ? WHERE id = ?", (author, book_id))
                updated += 1
                if updated % 5000 == 0:
                    conn.commit()
                    logger.info(f"  ...updated {updated} so far")

    if not args.dry_run and updated > 0:
        conn.commit()

    print(f"\nProcessed: {len(rows)}")
    print(f"Authors found: {found} ({100*found/max(len(rows),1):.1f}%)")
    if args.dry_run:
        print("(dry run — no changes written)")
    else:
        print(f"Updated: {updated}")

    if samples:
        print(f"\nSample extractions:")
        for author, path in samples:
            fname = Path(path).name
            print(f"  {author:40s}  ← {fname}")

    conn.close()


if __name__ == "__main__":
    main()
