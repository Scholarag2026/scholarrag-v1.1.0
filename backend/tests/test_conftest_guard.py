"""The test suite must never be able to point itself at a live database.

v1.0.0's conftest fell back to ``DATABASE_URL`` and dropped every table it found there
(an earlier version of the suite lost a live database this way). The guard below is
what now stands between ``pytest`` and whatever ``DATABASE_URL`` happens to name.
"""

import ast
from pathlib import Path

import pytest
from sqlalchemy.engine import make_url

from tests.conftest import (
    TEST_DATABASE_URL,
    assert_safe_test_database,
    modules_with_session_factory,
)

LIVE = "postgresql+asyncpg://deepresearch:secret@localhost:5432/deepresearch"
SAFE = "postgresql+asyncpg://deepresearch:secret@localhost:5432/deepresearch_test"


@pytest.fixture(autouse=True)
def _no_bypass(monkeypatch):
    monkeypatch.delenv("SCHOLARRAG_ALLOW_DESTRUCTIVE_TESTS", raising=False)


def test_refuses_a_url_identical_to_a_live_url():
    """Even a ``_test`` name is refused when the app itself is pointed at that database."""
    with pytest.raises(pytest.UsageError, match="live"):
        assert_safe_test_database(SAFE, [SAFE])


def test_refuses_the_same_database_reached_through_a_different_driver():
    sync_driver_live = "postgresql://deepresearch:secret@localhost:5432/deepresearch_test"
    with pytest.raises(pytest.UsageError, match="live"):
        assert_safe_test_database(SAFE, ["", sync_driver_live])


def test_refuses_the_same_database_reached_through_a_loopback_alias():
    """127.0.0.1, ::1 and localhost are the same server; spelling must not defeat the guard."""
    for alias in ("127.0.0.1", "[::1]", ""):
        via_alias = f"postgresql+asyncpg://deepresearch:secret@{alias}:5432/x_test"
        via_localhost = "postgresql+asyncpg://deepresearch:secret@localhost:5432/x_test"
        with pytest.raises(pytest.UsageError, match="live"):
            assert_safe_test_database(via_alias, [via_localhost])
        with pytest.raises(pytest.UsageError, match="live"):
            assert_safe_test_database(via_localhost, [via_alias])


def test_refuses_a_database_whose_name_lacks_the_test_suffix():
    with pytest.raises(pytest.UsageError, match="_test"):
        assert_safe_test_database(LIVE, [])


def test_accepts_a_distinct_test_database():
    assert assert_safe_test_database(SAFE, [LIVE, ""]) is None


def test_env_override_bypasses_the_guard(monkeypatch):
    monkeypatch.setenv("SCHOLARRAG_ALLOW_DESTRUCTIVE_TESTS", "1")
    assert assert_safe_test_database(LIVE, [LIVE]) is None


def test_the_configured_test_url_names_a_test_database():
    assert TEST_DATABASE_URL.rsplit("/", 1)[-1].endswith("_test")


# --- source scan: no test module may reach the live database by another route ------------

TESTS_DIR = Path(__file__).resolve().parent
SCAN_EXEMPT = {"conftest.py", Path(__file__).name}
# These two modules import ``app.database.engine`` only to inspect its pool / dialect
# configuration; they never open a connection on it.
READ_ONLY_ENGINE_INSPECTION = {"test_db_pool.py", "test_json_serializer.py"}
LIVE_DATABASE_ATTRS = {"engine", "async_session_factory"}


def _is_database_url_env_read(node: ast.AST) -> bool:
    """``os.environ.get("DATABASE_URL")``, ``os.getenv(...)`` or ``os.environ["DATABASE_URL"]``."""
    if isinstance(node, ast.Subscript):
        return (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "environ"
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "DATABASE_URL"
        )
    if isinstance(node, ast.Call) and node.args:
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and first.value == "DATABASE_URL"):
            return False
        func = node.func
        if isinstance(func, ast.Attribute):
            return func.attr == "getenv" or (
                func.attr == "get"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "environ"
            )
        return isinstance(func, ast.Name) and func.id == "getenv"
    return False


def _live_database_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if _is_database_url_env_read(node):
            found.append(f"{path.name}:{node.lineno} reads DATABASE_URL from the environment")
        elif isinstance(node, ast.Attribute) and node.attr == "database_url":
            found.append(f"{path.name}:{node.lineno} uses settings.database_url")
        elif isinstance(node, ast.ImportFrom) and node.module == "app.database":
            for alias in node.names:
                if alias.name in LIVE_DATABASE_ATTRS and not (
                    alias.name == "engine" and path.name in READ_ONLY_ENGINE_INSPECTION
                ):
                    found.append(f"{path.name}:{node.lineno} imports app.database.{alias.name}")
        elif (
            isinstance(node, ast.Attribute)
            and node.attr in LIVE_DATABASE_ATTRS
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "database"
        ):
            found.append(f"{path.name}:{node.lineno} uses app.database.{node.attr}")
        elif isinstance(node, ast.Call) and (
            (isinstance(node.func, ast.Name) and node.func.id == "create_async_engine")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "create_async_engine")
        ):
            url_arg = node.args[0] if node.args else next(
                (kw.value for kw in node.keywords if kw.arg == "url"), None
            )
            if not (isinstance(url_arg, ast.Name) and url_arg.id == "TEST_DATABASE_URL"):
                found.append(
                    f"{path.name}:{node.lineno} create_async_engine() not given TEST_DATABASE_URL"
                )
    return found


def test_no_test_module_can_reach_the_live_database():
    """Every engine built by the suite must come from TEST_DATABASE_URL (via conftest).

    Flags: reading ``DATABASE_URL`` from the environment by any spelling, ``settings.database_url``,
    importing or referencing ``app.database.engine`` / ``app.database.async_session_factory``,
    and any ``create_async_engine()`` call whose URL is not the ``TEST_DATABASE_URL`` name.
    """
    offenders = sorted(
        violation
        for p in TESTS_DIR.rglob("*.py")
        if p.name not in SCAN_EXEMPT
        for violation in _live_database_violations(p)
    )
    assert offenders == []


def test_scan_catches_every_spelling(tmp_path):
    sample = tmp_path / "test_sample.py"
    sample.write_text(
        "import os\n"
        "from app.config import settings\n"
        "from app.database import async_session_factory\n"
        "a = os.getenv('DATABASE_URL')\n"
        "b = os.environ['DATABASE_URL']\n"
        "c = os.environ.get('DATABASE_URL')\n"
        "d = settings.database_url\n"
        "e = create_async_engine(a)\n"
        "f = create_async_engine(TEST_DATABASE_URL)\n",
        encoding="utf-8",
    )
    violations = _live_database_violations(sample)
    assert [v.split(" ", 1)[1] for v in violations] == [
        "imports app.database.async_session_factory",
        "reads DATABASE_URL from the environment",
        "reads DATABASE_URL from the environment",
        "reads DATABASE_URL from the environment",
        "uses settings.database_url",
        "create_async_engine() not given TEST_DATABASE_URL",
    ]


# --- runtime guard: module-level session factories are rebound to the test database -------


def test_every_module_level_session_factory_is_bound_to_the_test_database():
    """app.api modules keep a module-level ``_session_factory`` for background jobs, bound at
    import time to the live engine; ``app.database.async_session_factory`` is the source of
    all of them, and ``app.main`` binds the same name for its lifespan services. The autouse
    fixture in conftest must have rebound every one of them."""
    expected = make_url(TEST_DATABASE_URL)
    targets = modules_with_session_factory()
    assert len(targets) >= 10, [m.__name__ for m, _ in targets]
    assert any(m.__name__ == "app.database" for m, _ in targets)
    # app.main imports async_session_factory by name and hands it to the lifespan services.
    assert any(m.__name__ == "app.main" and a == "async_session_factory" for m, a in targets)
    for module, attr in targets:
        bound = getattr(module, attr).kw["bind"].url
        assert bound == expected, f"{module.__name__}.{attr} is bound to {bound}"
