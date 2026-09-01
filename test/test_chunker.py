from rag.chunker import chunk_text

def test_chunk_count():
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 3

test_chunk_count()
print("Test passed!")