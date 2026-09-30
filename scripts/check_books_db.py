"""Integrity check for books.db: do chapters/paragraphs belong to the book
rows they point at?

Compares each sampled ``books.title`` with the "Title:" line that
Gutenberg texts carry in their preamble paragraphs. A mismatch means the
``books`` table was rebuilt with different ids than ``chapters`` /
``paragraphs`` — the reader would then open one title and read another.

    python scripts/check_books_db.py [--sample 200]
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from config.settings import settings  # noqa: E402

_TITLE_RE = re.compile(r"^\s*Title:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--ids", type=int, nargs="*", default=[])
    args = ap.parse_args()

    c = sqlite3.connect(str(settings.books_db_path))
    c.row_factory = sqlite3.Row
    n_books = c.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    n_ch_books = c.execute("SELECT COUNT(DISTINCT book_id) FROM chapters").fetchone()[0]
    print(f"books rows: {n_books}  distinct chapters.book_id: {n_ch_books}")
    print(
        "id ranges: books",
        c.execute("SELECT MIN(id), MAX(id) FROM books").fetchone()[:],
        "chapters",
        c.execute("SELECT MIN(book_id), MAX(book_id) FROM chapters").fetchone()[:],
    )

    ids = args.ids or [
        r[0]
        for r in c.execute(
            "SELECT id FROM books ORDER BY RANDOM() LIMIT ?", (args.sample,)
        ).fetchall()
    ]
    checked = matched = mismatched = no_title = 0
    examples = []
    for bid in ids:
        b = c.execute("SELECT id, title, path FROM books WHERE id = ?", (bid,)).fetchone()
        if b is None:
            continue
        paras = c.execute(
            "SELECT text FROM paragraphs WHERE book_id = ? AND chapter_idx = 0 "
            "ORDER BY para_idx LIMIT 6",
            (bid,),
        ).fetchall()
        blob = "\n".join(r[0] for r in paras)
        m = _TITLE_RE.search(blob)
        checked += 1
        if not m:
            no_title += 1
            continue
        pt = _norm(m.group(1))
        bt = _norm(b["title"])
        ok = pt[:25] == bt[:25] or pt in bt or bt in pt
        if ok:
            matched += 1
        else:
            mismatched += 1
            if len(examples) < 8:
                examples.append((bid, b["title"][:40], m.group(1)[:40], Path(b["path"]).name))
    print(
        f"checked {checked}: matched {matched}, MISMATCHED {mismatched}, "
        f"no Title: line in preamble {no_title}"
    )
    for ex in examples:
        print(f"  id {ex[0]}: books.title={ex[1]!r}  paragraphs say={ex[2]!r}  file={ex[3]}")
    if mismatched:
        # Is it a constant offset? Look up where the paragraph title really lives.
        for bid, _, ptitle, _ in examples[:3]:
            rows = c.execute(
                "SELECT id, path FROM books WHERE title LIKE ? LIMIT 3", (f"%{ptitle[:20]}%",)
            ).fetchall()
            print(
                f"  paragraphs of id {bid} match books rows: {[(r[0], Path(r[1]).name) for r in rows]}"
            )


if __name__ == "__main__":
    main()
