"""Tests for analysis service — batch processing utilities."""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

from app.schemas.analysis import PaperAnalysis


def test_split_into_batches():
    from app.services.analysis import _split_into_batches

    items = list(range(12))
    batches = _split_into_batches(items, batch_size=5)
    assert len(batches) == 3
    assert batches[0] == [0, 1, 2, 3, 4]
    assert batches[1] == [5, 6, 7, 8, 9]
    assert batches[2] == [10, 11]


def test_split_into_batches_empty():
    from app.services.analysis import _split_into_batches

    assert _split_into_batches([], batch_size=5) == []


def test_split_into_batches_exact():
    from app.services.analysis import _split_into_batches

    items = list(range(10))
    batches = _split_into_batches(items, batch_size=5)
    assert len(batches) == 2


def test_validate_paper_ids():
    from app.services.analysis import _validate_paper_ids

    input_ids = {"paper-1", "paper-2", "paper-3"}
    analyses = [
        PaperAnalysis(
            paper_id="paper-1", relevance_score=0.5, quality_score=0.5,
            key_findings=["F"], methodology="M", methodology_rigor="high",
            limitations=[], theories_used=[],
        ),
        PaperAnalysis(
            paper_id="paper-999",  # not in input
            relevance_score=0.5, quality_score=0.5,
            key_findings=["F"], methodology="M", methodology_rigor="high",
            limitations=[], theories_used=[],
        ),
    ]
    valid = _validate_paper_ids(analyses, input_ids)
    assert len(valid) == 1
    assert valid[0].paper_id == "paper-1"
