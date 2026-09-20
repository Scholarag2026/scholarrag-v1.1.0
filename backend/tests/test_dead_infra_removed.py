"""Dead infrastructure must not be declared."""

from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _requirement_names() -> set[str]:
    names = set()
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split(">=")[0].split("==")[0].split("[")[0].strip().lower()
        names.add(name)
    return names


def test_celery_is_not_a_declared_dependency():
    assert "celery" not in _requirement_names()


def test_pybreaker_is_not_a_declared_dependency():
    assert "pybreaker" not in _requirement_names()


def test_redis_is_still_declared():
    """Redis stays a declared dependency: it is provisioned on Zeabur for the future
    worker migration."""
    assert "redis" in _requirement_names()


def test_structlog_is_still_declared():
    """structlog is genuinely used by app/clients/crossref.py."""
    assert "structlog" in _requirement_names()


def test_rabbitmq_url_setting_is_removed():
    from app.config import Settings

    assert "rabbitmq_url" not in Settings.model_fields


def test_redis_url_setting_is_kept():
    """No code path reads ``Settings.redis_url`` today, but the field stays for the
    future Zeabur worker migration; only the local dev stack's own redis service
    and REDIS_URL wiring were removed (see the two tests below)."""
    from app.config import Settings

    assert "redis_url" in Settings.model_fields


def test_env_example_does_not_declare_redis_url():
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "REDIS_URL" not in text


def test_docker_compose_does_not_declare_a_redis_service():
    for name in ("docker-compose.yml", "docker-compose.prod.yml"):
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert "redis" not in text.lower()


def test_analysis_max_retries_setting_is_kept():
    """This setting is wired into Agent(retries=...) elsewhere; do not delete it here."""
    from app.config import Settings

    assert "analysis_max_retries" in Settings.model_fields


def test_no_source_file_references_rabbitmq():
    app_dir = Path(__file__).resolve().parents[1] / "app"
    offenders = [
        str(p)
        for p in app_dir.rglob("*.py")
        if "rabbitmq" in p.read_text(encoding="utf-8").lower()
    ]
    assert offenders == []
