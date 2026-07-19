#!/usr/bin/env python3
"""Index a directory of book files into the books database.

Usage:
    python scripts/index_books.py [BOOKS_DIR]
    python scripts/index_books.py --list
    python scripts/index_books.py --search "moby dick"

Defaults to ORACLE_BOOKS_PATH (data/books/) if no directory given.
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Index books for the Oracle e-reader")
    parser.add_argument("books_dir", nargs="?", type=Path, help="Directory of .txt files")
    parser.add_argument("--list", action="store_true", help="List all indexed books")
    parser.add_argument("--search", type=str, help="Search for a book by title/author")
    parser.add_argument("--info", type=int, help="Show details for a book by ID")
    args = parser.parse_args()

    from oracle.books.library import Library

    lib = Library()

    if args.list:
        books = lib.list_books()
        if not books:
            print("No books indexed yet.")
            return
        for b in books:
            print(f"  [{b.id:3d}] {b.title} — {b.author or 'Unknown'} ({b.total_chapters} ch, {b.total_paragraphs} para)")
        print(f"\n{len(books)} books total")
        return

    if args.search:
        results = lib.search(args.search)
        if not results:
            print(f"No books matching '{args.search}'")
            return
        for b in results:
            print(f"  [{b.id:3d}] {b.title} — {b.author or 'Unknown'}")
        return

    if args.info:
        book = lib.get_book(args.info)
        if not book:
            print(f"Book {args.info} not found")
            return
        print(f"Title:      {book.title}")
        print(f"Author:     {book.author or 'Unknown'}")
        print(f"Chapters:   {book.total_chapters}")
        print(f"Paragraphs: {book.total_paragraphs}")
        print(f"Path:       {book.path}")
        print("\nChapters:")
        for ch_idx in range(book.total_chapters):
            ch = lib.get_chapter(book.id, ch_idx)
            if ch:
                para_count = lib.get_paragraph_count(book.id, ch_idx)
                print(f"  {ch_idx:3d}. {ch.title} ({para_count} paragraphs)")
        return

    # Default: index a directory
    books_dir = args.books_dir
    added = lib.index_directory(books_dir)
    if added:
        print(f"Indexed {added} new books")

    books = lib.list_books()
    print(f"\nLibrary now contains {len(books)} books")
    lib.close()


if __name__ == "__main__":
    main()
