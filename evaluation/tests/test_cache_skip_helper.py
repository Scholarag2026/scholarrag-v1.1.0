"""``conftest.skip_unless_cache_dir`` decides whether a claims test that reads a
gitignored fetch cache directly (``evaluation/claims/data/...``) can even run: the
published release export ships none of that cache (submission-checklist.md, release
items), so a test built against it must skip with a plain reason instead of failing on
a missing file.
"""

import pytest
from conftest import skip_unless_cache_dir


def test_returns_the_directory_unchanged_when_it_exists(tmp_path):
    cache_dir = tmp_path / "hss_fulltext"
    cache_dir.mkdir()
    assert skip_unless_cache_dir(cache_dir) == cache_dir


def test_skips_with_the_plain_fetch_cache_reason_when_the_directory_is_absent(tmp_path):
    cache_dir = tmp_path / "hss_fulltext"
    with pytest.raises(pytest.skip.Exception) as exc_info:
        skip_unless_cache_dir(cache_dir)
    assert str(exc_info.value) == (
        "fetch cache not shipped; rebuild it with the documented fetch command"
    )


def test_skips_when_the_path_exists_but_is_a_file_not_a_directory(tmp_path):
    cache_dir = tmp_path / "hss_fulltext"
    cache_dir.write_text("not a directory", encoding="utf-8")
    with pytest.raises(pytest.skip.Exception):
        skip_unless_cache_dir(cache_dir)
