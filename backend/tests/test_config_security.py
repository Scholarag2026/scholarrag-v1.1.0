"""Startup security validation of settings."""

import logging

import pytest


def test_settings_has_app_env_defaulting_to_development():
    from app.config import Settings

    assert Settings().app_env == "development"


def test_production_with_placeholder_secret_raises():
    from app.config import Settings, validate_security_settings

    s = Settings(app_env="production", jwt_secret_key="change-me")
    with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
        validate_security_settings(s)


def test_production_with_empty_secret_raises():
    from app.config import Settings, validate_security_settings

    s = Settings(app_env="production", jwt_secret_key="")
    with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
        validate_security_settings(s)


def test_production_with_real_secret_passes():
    from app.config import Settings, validate_security_settings

    s = Settings(app_env="production", jwt_secret_key="a-real-32-byte-secret-value-here")
    validate_security_settings(s)


def test_production_with_short_secret_raises():
    """A non-placeholder secret under 32 characters is still forgeable-by-guessing; the
    audit's docker-compose.prod.yml finding (APP_ENV not set to production there, so this
    branch never ran against the compose file's own secrets) is closed by this length floor
    plus the compose file itself now setting APP_ENV=production."""
    from app.config import Settings, validate_security_settings

    s = Settings(app_env="production", jwt_secret_key="short-but-not-the-placeholder")
    with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
        validate_security_settings(s)


def test_production_with_secret_of_exactly_31_chars_raises():
    from app.config import Settings, validate_security_settings

    s = Settings(app_env="production", jwt_secret_key="a" * 31)
    with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
        validate_security_settings(s)


def test_production_with_secret_of_exactly_32_chars_passes():
    from app.config import Settings, validate_security_settings

    s = Settings(app_env="production", jwt_secret_key="a" * 32)
    validate_security_settings(s)


@pytest.mark.parametrize("env_name", ["development", "test", "local"])
def test_dev_environments_allow_a_short_non_placeholder_secret(env_name):
    """The length floor is a production-only refusal (a production-only requirement); a
    short secret in a dev environment is unaffected, exactly like the pre-existing
    placeholder tolerance above."""
    from app.config import Settings, validate_security_settings

    s = Settings(app_env=env_name, jwt_secret_key="short")
    validate_security_settings(s)


def test_unknown_env_name_is_treated_as_non_dev():
    from app.config import Settings, validate_security_settings

    s = Settings(app_env="staging", jwt_secret_key="change-me")
    with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
        validate_security_settings(s)


@pytest.mark.parametrize("env_name", ["development", "test", "local"])
def test_dev_environments_allow_the_placeholder_secret(env_name, caplog):
    from app.config import Settings, validate_security_settings

    s = Settings(app_env=env_name, jwt_secret_key="change-me")
    with caplog.at_level(logging.WARNING, logger="app.config"):
        validate_security_settings(s)
    messages = [r.getMessage() for r in caplog.records if r.name == "app.config"]
    assert any("insecure default" in m for m in messages), messages


def test_importing_app_main_does_not_raise_under_test_defaults():
    """The default (development) configuration must never break test/CI boot."""
    import app.main

    assert app.main.app is not None
