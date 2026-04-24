from oracle.rag.chunker import chunk_text


def test_chunk_basic():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=30, chunk_overlap=5)
    assert len(chunks) >= 3
    assert all(len(c.split()) <= 30 for c in chunks)


def test_chunk_preserves_content():
    text = "Hello world. " * 50
    chunks = chunk_text(text, chunk_size=20, chunk_overlap=3)
    assert len(chunks) > 1
    # All chunks should contain words from the source
    for chunk in chunks:
        assert "Hello" in chunk or "world" in chunk


def test_chunk_short_text():
    text = "Short text here."
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == "Short text here."
