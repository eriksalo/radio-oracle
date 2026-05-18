#!/usr/bin/env python3
"""Extract English plain-text books from a Gutenberg ZIM archive.

Usage:
    python scripts/extract_gutenberg_zim.py data/books/gutenberg_mul_all_2025-11.zim
    python scripts/extract_gutenberg_zim.py data/books/gutenberg_mul_all_2025-11.zim --dry-run
    python scripts/extract_gutenberg_zim.py data/books/gutenberg_mul_all_2025-11.zim --limit 100
"""

from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

from loguru import logger


class _HTMLToText(HTMLParser):
    """Minimal HTML→plain text converter."""

    _SKIP_TAGS = {"script", "style", "head", "nav", "footer"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag in ("p", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "li"):
            self._parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        text = "".join(self._parts)
        # Collapse whitespace within lines but preserve paragraph breaks
        lines = text.split("\n")
        lines = [" ".join(line.split()) for line in lines]
        text = "\n".join(lines)
        # Collapse multiple blank lines to double newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def html_to_text(html: str) -> str:
    parser = _HTMLToText()
    parser.feed(html)
    return parser.get_text()


def _safe_filename(name: str) -> str:
    """Convert a book title to a safe filename."""
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:150]


def extract(zim_path: Path, output_dir: Path, dry_run: bool = False, limit: int = 0) -> None:
    from libzim.reader import Archive

    zim = Archive(str(zim_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"ZIM: {zim.entry_count} entries, {zim.article_count} articles")

    extracted = 0
    skipped_lang = 0
    skipped_short = 0

    for i in range(zim.entry_count):
        if limit and extracted >= limit:
            break

        entry = zim._get_entry_by_id(i)
        path = entry.path

        # Skip non-book entries
        if path.endswith(".epub") or "_cover" in path:
            continue

        try:
            item = entry.get_item()
        except Exception:
            continue

        if item.mimetype != "text/html":
            continue

        html = bytes(item.content).decode("utf-8", errors="replace")

        # English only
        if 'lang="en"' not in html[:500]:
            skipped_lang += 1
            continue

        text = html_to_text(html)

        # Skip very short texts (under 1000 chars ~= less than a page)
        if len(text) < 1000:
            skipped_short += 1
            continue

        # Extract Gutenberg ID from path (e.g. "Title.12345")
        gid_match = re.search(r"\.(\d+)$", path)
        gid = gid_match.group(1) if gid_match else str(i)

        # Build filename
        title_part = path.rsplit(".", 1)[0] if gid_match else path
        filename = f"{_safe_filename(title_part)}.{gid}.txt"
        out_path = output_dir / filename

        if dry_run:
            if extracted < 10:
                logger.info(f"  [{gid}] {title_part[:60]} — {len(text)} chars")
            extracted += 1
            continue

        # Write with Gutenberg-style header for the library indexer
        header = f"Title: {title_part}\n\n"
        out_path.write_text(header + text, encoding="utf-8")
        extracted += 1

        if extracted % 500 == 0:
            logger.info(f"  Extracted {extracted} books...")

    logger.info(
        f"Done: {extracted} English books extracted, "
        f"{skipped_lang} non-English skipped, {skipped_short} too short"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Gutenberg books from ZIM")
    parser.add_argument("zim_path", type=Path, help="Path to gutenberg .zim file")
    parser.add_argument("--output", type=Path, default=None, help="Output directory (default: data/books/gutenberg/)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Max books to extract (0 = all)")
    args = parser.parse_args()

    if not args.zim_path.exists():
        logger.error(f"ZIM not found: {args.zim_path}")
        sys.exit(1)

    output = args.output or Path("data/books/gutenberg")
    extract(args.zim_path, output, args.dry_run, args.limit)


if __name__ == "__main__":
    main()
