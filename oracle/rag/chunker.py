"""Text chunking for RAG ingestion."""

from __future__ import annotations

from config.settings import settings


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping chunks, preserving paragraph boundaries.

    Args:
        text: Source text to chunk
        chunk_size: Max tokens per chunk (approximated as words)
        chunk_overlap: Overlap between chunks in tokens

    Returns:
        List of text chunks
    """
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current_words: list[str] = []

    for para in paragraphs:
        words = para.split()
        for word in words:
            current_words.append(word)
            if len(current_words) >= size:
                chunks.append(" ".join(current_words))
                # Keep overlap
                current_words = current_words[-overlap:]

    # Flush remaining
    if current_words:
        chunks.append(" ".join(current_words))

    return chunks
