"""Make ``common`` and the screening/claims script modules importable in tests.

``screening/summarize.py`` and ``claims/summarize.py`` share a module name, so tests must
load either one through :func:`load_script_module` instead of ``import summarize``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

EVAL_ROOT = Path(__file__).resolve().parents[1]
for sub in ("", "screening", "claims"):
    p = EVAL_ROOT / sub if sub else EVAL_ROOT
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def skip_unless_cache_dir(path: Path) -> Path:
    """Return *path* when the gitignored fetch-cache directory it names is present in
    this checkout; otherwise skip the calling test with a plain reason.

    ``evaluation/**/data/`` (``.gitignore``) holds full-text caches the fetch scripts
    build from the network; the published release export ships none of it
    (submission-checklist.md, release items). ``evaluation/README.md``'s "Commands"
    section names the fetch command that rebuilds each one.
    """
    if not path.is_dir():
        pytest.skip("fetch cache not shipped; rebuild it with the documented fetch command")
    return path


def load_script_module(subdir: str, name: str) -> ModuleType:
    """Load ``evaluation/<subdir>/<name>.py`` under the unique name ``<subdir>_<name>``."""
    qualified = f"{subdir}_{name}"
    if qualified in sys.modules:
        return sys.modules[qualified]
    path = EVAL_ROOT / subdir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(qualified, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module
