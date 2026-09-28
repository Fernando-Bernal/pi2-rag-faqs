from chunking import chunk_document, load_document
from config import DATA_PATH, MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS


def test_load_document_reads_full_text():
    text = load_document(DATA_PATH)
    assert len(text.split()) > 1000


def test_chunk_document_generates_at_least_20_chunks():
    text = load_document(DATA_PATH)
    chunks = chunk_document(text)
    assert len(chunks) >= 20


def test_chunks_respect_token_bounds():
    text = load_document(DATA_PATH)
    chunks = chunk_document(text)
    for chunk in chunks[:-1]:
        assert MIN_CHUNK_TOKENS <= chunk["num_tokens"] <= MAX_CHUNK_TOKENS


def test_chunks_have_unique_sequential_ids():
    text = load_document(DATA_PATH)
    chunks = chunk_document(text)
    ids = [c["chunk_id"] for c in chunks]
    assert ids == list(range(len(chunks)))
