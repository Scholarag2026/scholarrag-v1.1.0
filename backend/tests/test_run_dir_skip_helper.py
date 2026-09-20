"""``tests.conftest.skip_unless_run_dir`` decides whether a backend test that reads a
tracked ``demo/output/<run>`` directory directly can even run: the published release
export ships only the run directory backing the promoted ``demo/expected/`` baseline
(submission-checklist.md, release items), so every other named run directory is absent
there, and a test built against one of those must skip with a plain reason instead of
failing on a missing file.
"""

import pytest

from tests.conftest import skip_unless_run_dir


def test_returns_the_directory_unchanged_when_it_exists(tmp_path):
    run_dir = tmp_path / "20260101-000000"
    run_dir.mkdir()
    assert skip_unless_run_dir(run_dir) == run_dir


def test_skips_with_a_plain_reason_naming_the_directory_when_it_is_absent(tmp_path):
    run_dir = tmp_path / "20260101-000000"
    with pytest.raises(pytest.skip.Exception) as exc_info:
        skip_unless_run_dir(run_dir)
    assert str(exc_info.value) == "run directory 20260101-000000 is not shipped"


def test_skips_when_the_path_exists_but_is_a_file_not_a_directory(tmp_path):
    """A stray file at that path is not a usable run directory either."""
    run_dir = tmp_path / "20260101-000000"
    run_dir.write_text("not a directory", encoding="utf-8")
    with pytest.raises(pytest.skip.Exception):
        skip_unless_run_dir(run_dir)
