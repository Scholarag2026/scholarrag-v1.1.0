"""The shared heading-shaped-paragraph fixture.

``demo/check_delivered.py`` deliberately never imports the backend (its own module
docstring), so the heading-shape predicate exists twice: once here, in
``app.services.sentence_coverage``, and once as an independent copy in
``demo/check_delivered.py``. ``demo/fixtures/heading_shaped_paragraphs.json`` is the
single committed fixture both sides' tests read, so the two copies cannot diverge
silently. ``demo/tests/test_check_delivered.py::
test_is_heading_shaped_paragraph_matches_the_shared_fixture_file`` checks the
checker's own copy; this test checks this module's.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from app.services.sentence_coverage import is_heading_shaped_paragraph  # noqa: E402

DEMO = Path(__file__).resolve().parents[2] / "demo"


def test_heading_shaped_paragraph_fixture_matches_this_modules_own_predicate():
    fixture_path = DEMO / "fixtures" / "heading_shaped_paragraphs.json"
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert cases, "expected the shared heading_shaped_paragraphs.json fixture to be non-empty"
    for case in cases:
        assert is_heading_shaped_paragraph(case["text"]) == case["expected"], case
