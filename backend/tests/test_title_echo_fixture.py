"""The shared title-echo fixture.

``demo/check_delivered.py`` deliberately never imports the backend (its own module
docstring), so the title-echo predicate exists twice: once here, in
``app.services.fulltext``, and once as an independent copy in
``demo/check_delivered.py``. ``demo/fixtures/title_echo_cases.json`` is the single
committed fixture both sides' tests read, so the two copies cannot diverge silently.
``demo/tests/test_check_delivered.py::test_title_echo_fixture_matches_this_modules_own_predicate``
checks the checker's own copy; this test checks this module's.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from app.services.fulltext import _is_title_echo  # noqa: E402

DEMO = Path(__file__).resolve().parents[2] / "demo"


def test_title_echo_fixture_matches_this_modules_own_predicate():
    fixture_path = DEMO / "fixtures" / "title_echo_cases.json"
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert cases, "expected the shared title_echo_cases.json fixture to be non-empty"
    for case in cases:
        assert _is_title_echo(case["paragraph"], case["references"]) == case["expected"], case
