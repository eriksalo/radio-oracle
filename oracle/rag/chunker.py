"""Text chunking for RAG ingestion."""

from __future__ import annotations

from config.settings import settings


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping chunks along paragraph boundaries.

    Whole paragraphs are packed into a chunk until adding the next one
    would exceed *chunk_size* words; the chunk closes at the paragraph
    boundary and the next one starts with the closed chunk's last
    *chunk_overlap* words as context (when they fit). Paragraphs longer
    than *chunk_size* on their own are split by words with a
    size-overlap stride. Chunks never exceed *chunk_size* words.

    (The pre-2026-07 version streamed words across paragraph boundaries
    and only cut at the size limit, so most chunks started and ended
    mid-sentence — one reason for the RAG v2 re-embed.)

    Args:
        text: Source text to chunk
        chunk_size: Max words per chunk (proxy for tokens)
        chunk_overlap: Overlap between consecutive chunks in words

    Returns:
        List of text chunks
    """
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current: list[str] = []

    def close_chunk() -> list[str]:
        """Emit the open chunk; return its overlap tail for the next one."""
        if not current:
            return []
        chunks.append(" ".join(current))
        return current[-overlap:] if overlap else []

    for para in paragraphs:
        words = para.split()

        if len(words) > size:
            # Oversize paragraph: flush, then split it with an overlap stride.
            close_chunk()
            current = []
            stride = max(size - overlap, 1)
            for start in range(0, len(words), stride):
                piece = words[start : start + size]
                if start > 0 and len(piece) <= overlap:
                    break  # tail fully covered by the previous piece's overlap
                chunks.append(" ".join(piece))
            continue

        if current and len(current) + len(words) > size:
            tail = close_chunk()
            # Seed with overlap context only when the paragraph leaves room.
            current = list(tail) if len(tail) + len(words) <= size else []

        current.extend(words)

    close_chunk()
    return chunks
