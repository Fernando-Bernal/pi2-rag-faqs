from query import has_relevant_chunks


def test_has_relevant_chunks_true_above_threshold():
    chunks = [{"chunk_id": 0, "text": "x", "score": 0.6}]
    assert has_relevant_chunks(chunks, min_score=0.5) is True


def test_has_relevant_chunks_true_exactly_at_threshold():
    chunks = [{"chunk_id": 0, "text": "x", "score": 0.5}]
    assert has_relevant_chunks(chunks, min_score=0.5) is True


def test_has_relevant_chunks_false_below_threshold():
    chunks = [{"chunk_id": 0, "text": "x", "score": 0.494}]
    assert has_relevant_chunks(chunks, min_score=0.5) is False


def test_has_relevant_chunks_false_when_empty():
    assert has_relevant_chunks([], min_score=0.5) is False


def test_has_relevant_chunks_only_checks_best_score():
    chunks = [
        {"chunk_id": 0, "text": "x", "score": 0.55},
        {"chunk_id": 1, "text": "y", "score": 0.1},
    ]
    assert has_relevant_chunks(chunks, min_score=0.5) is True
