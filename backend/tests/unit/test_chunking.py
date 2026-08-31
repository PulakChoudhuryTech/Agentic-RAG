from backend.app.rag.chunking import chunk_child, chunk_parent


def test_chunk_parent_respects_max_chars_roughly():
    text = "\n\n".join(f"Paragraph {i}. " * 20 for i in range(10))
    chunks = chunk_parent(text, max_chars=500, overlap=50)

    assert len(chunks) > 1
    for chunk in chunks:
        # allow some slack: a single paragraph longer than max_chars is kept whole
        assert len(chunk.content) <= 700


def test_chunk_parent_indices_are_sequential():
    text = "\n\n".join(f"Paragraph {i} content here." for i in range(20))
    chunks = chunk_parent(text, max_chars=100, overlap=20)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_parent_empty_text_returns_no_chunks():
    assert chunk_parent("") == []
    assert chunk_parent("   \n\n   ") == []


def test_chunk_parent_overlap_carries_tail_forward():
    text = "\n\n".join(f"Sentence number {i} with some extra words to pad length." for i in range(15))
    chunks = chunk_parent(text, max_chars=200, overlap=50)

    assert len(chunks) >= 2
    tail_of_first = chunks[0].content[-30:]
    assert tail_of_first[:10] in chunks[1].content


def test_chunk_child_covers_the_whole_parent_text():
    parent_text = "word " * 200  # 1000 chars
    children = chunk_child(parent_text, max_chars=100, overlap=20)

    assert len(children) > 1
    assert children[0].char_start == 0
    assert children[-1].char_end == len(parent_text)


def test_chunk_child_empty_text_returns_no_chunks():
    assert chunk_child("") == []
