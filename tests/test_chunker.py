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


def test_chunk_never_exceeds_size():
    text = "\n\n".join(" ".join(f"w{i}_{j}" for j in range(17)) for i in range(30))
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=8)
    assert all(len(c.split()) <= 50 for c in chunks)


def test_chunk_respects_paragraph_boundaries():
    paras = [f"para{i} " + " ".join(f"p{i}w{j}" for j in range(20)) for i in range(6)]
    text = "\n\n".join(paras)
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=0)
    # With no overlap, every chunk must start exactly at a paragraph start —
    # the old chunker cut mid-sentence.
    for c in chunks:
        assert c.split()[0].startswith("para")


def test_chunk_overlap_carries_context():
    paras = [" ".join(f"p{i}w{j}" for j in range(20)) for i in range(4)]
    text = "\n\n".join(paras)
    chunks = chunk_text(text, chunk_size=40, chunk_overlap=5)
    assert len(chunks) >= 2
    # Second chunk starts with the tail of the first.
    first_tail = chunks[0].split()[-5:]
    assert chunks[1].split()[:5] == first_tail


def test_chunk_oversize_paragraph_split_with_overlap():
    text = " ".join(f"w{j}" for j in range(120))  # single huge paragraph
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
    assert all(len(c.split()) <= 50 for c in chunks)
    joined = set()
    for c in chunks:
        joined.update(c.split())
    assert joined == {f"w{j}" for j in range(120)}  # nothing lost
