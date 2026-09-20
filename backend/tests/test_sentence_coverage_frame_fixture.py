"""The shared frame-classification fixture.

``demo/check_delivered.py`` deliberately never imports the backend (its own module
docstring), so the residue/frame classifier exists twice: once here, in
``app.services.sentence_coverage``, and once as an independent copy in
``demo/check_delivered.py``. ``demo/fixtures/frame_spans.json`` is the single
committed fixture both sides' tests read, so the two copies cannot diverge silently.
``demo/tests/test_check_delivered.py::test_frame_spans_fixture_matches_this_modules_own_classifier``
checks the checker's own copy; this test checks this module's.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from app.services.sentence_coverage import Residue, is_frame, token_list  # noqa: E402

DEMO = Path(__file__).resolve().parents[2] / "demo"


def test_frame_spans_fixture_matches_this_modules_own_classifier():
    """The fixture stores each span's own text and class, not the surrounding
    sentence, so only the two POSITION-INDEPENDENT classes are checked here: a
    non-frame residue (``class`` null -- content-bearing, or a never-frame/numeric/
    quantifier/over-budget term) stays non-frame regardless of position, and an F1
    (pure attribution, or a bare leading subject with no boundary punctuation of its
    own) reproduces with no position at all. F2/F3 need the residue's own position in
    its sentence (a colon or comma-plus-opener right after it) -- covered instead by
    this module's own per-sentence tests in ``test_sentence_coverage.py``, exactly as
    the demo side's equivalent test documents for its own copy."""
    fixture_path = DEMO / "fixtures" / "frame_spans.json"
    spans = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert spans, "expected the shared frame_spans.json fixture to be non-empty"
    checked = 0
    for entry in spans:
        if entry["class"] not in (None, "f1"):
            continue
        # Most entries carry no position at all (``start`` defaults to -1,
        # never sentence-initial), which is exactly the input the
        # bare-leading-subject branch never fires on. An entry that DOES carry a
        # position (``start``/``before_first_claim``/``trailing_gap``) is what lets
        # this fixture see that branch at all, so it is read here when present instead
        # of being forced back to the position-independent defaults.
        residue = Residue(
            text=entry["text"],
            tokens=token_list(entry["text"]),
            start=entry.get("start", -1),
            end=-1,
            before_first_claim=entry.get("before_first_claim", False),
            is_trailing=False,
            trailing_gap=entry.get("trailing_gap", ""),
            leading_gap="",
        )
        claim_tokens: set[str] = set()
        assert is_frame(residue, claim_tokens) == entry["class"], entry
        checked += 1
    assert checked > 0
