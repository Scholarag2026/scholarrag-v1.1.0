"""``Settings.writing_model``, ``writing_thinking`` and ``writing_max_tokens``: three
writer-only overrides (env ``WRITING_MODEL``, ``WRITING_THINKING``, ``WRITING_MAX_TOKENS``).

Only ``call_deepseek`` in ``app.services.writing`` reads any of these fields (see
``test_writing_provenance.py``); every other DeepSeek-calling agent keeps
``settings.deepseek_model`` and its own fixed sampling parameters regardless of what
these are set to.
"""

from app.config import Settings


def _settings(**overrides) -> Settings:
    """Settings built without reading a local .env, so defaults are deterministic."""
    return Settings(_env_file=None, **overrides)


def test_writing_model_defaults_to_none():
    assert _settings().writing_model is None


def test_writing_model_reads_env_override(monkeypatch):
    monkeypatch.setenv("WRITING_MODEL", "deepseek-v4-pro")
    assert _settings().writing_model == "deepseek-v4-pro"


def test_writing_model_blank_env_counts_as_unset(monkeypatch):
    """``WRITING_MODEL=`` left blank in `.env` (an empty string, not an unset variable)
    is coerced to ``None``, not kept as ``""``."""
    monkeypatch.setenv("WRITING_MODEL", "")
    assert _settings().writing_model is None


def test_writing_thinking_defaults_to_none():
    """``None`` means the request carries no ``thinking`` field at all, so the
    provider's own default applies."""
    assert _settings().writing_thinking is None


def test_writing_thinking_reads_env_override(monkeypatch):
    monkeypatch.setenv("WRITING_THINKING", "enabled")
    assert _settings().writing_thinking == "enabled"


def test_writing_thinking_reads_disabled(monkeypatch):
    monkeypatch.setenv("WRITING_THINKING", "disabled")
    assert _settings().writing_thinking == "disabled"


def test_writing_thinking_blank_env_counts_as_unset(monkeypatch):
    """``WRITING_THINKING=`` left blank in `.env` is coerced to ``None``, the same
    treatment ``writing_model`` gets."""
    monkeypatch.setenv("WRITING_THINKING", "")
    assert _settings().writing_thinking is None


def test_writing_max_tokens_defaults_to_4096():
    """4096 is the constant every writing call sent before this setting existed."""
    assert _settings().writing_max_tokens == 4096


def test_writing_max_tokens_reads_env_override(monkeypatch):
    monkeypatch.setenv("WRITING_MAX_TOKENS", "16384")
    assert _settings().writing_max_tokens == 16384


def test_writing_max_tokens_blank_env_counts_as_default(monkeypatch):
    """``WRITING_MAX_TOKENS=`` left blank (docker-compose.yml forwards
    ``${WRITING_MAX_TOKENS:-}``, an empty string, whenever the shell variable is unset)
    must not crash Settings construction; it is treated as the default 4096."""
    monkeypatch.setenv("WRITING_MAX_TOKENS", "")
    assert _settings().writing_max_tokens == 4096
