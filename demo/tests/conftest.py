"""Make ``demo/run_demo.py`` importable from the tests without installing anything."""

import sys
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).resolve().parents[1]
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))


def skip_unless_run_dir(run_dir: Path) -> Path:
    """Return *run_dir* when the tracked ``demo/output/<run>`` directory it names is
    present in this checkout; otherwise skip the calling test with a plain reason.

    The published release export ships only the one run directory backing the
    promoted ``demo/expected/`` baseline (submission-checklist.md, release items);
    every other tracked ``demo/output/<run>`` directory a demo test reads directly is
    excluded from that export, so a test built against one of those must skip there
    rather than fail on a missing file.
    """
    if not run_dir.is_dir():
        pytest.skip(f"run directory {run_dir.name} is not shipped")
    return run_dir
