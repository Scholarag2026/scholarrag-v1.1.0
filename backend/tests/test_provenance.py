"""Shared LLM-call provenance record.

Every evaluated LLM step must record which model actually answered (the provider's
reported model name, not the floating alias we configured), the sampling temperature,
a version hash of the system prompt, and token usage.
"""

import os
from datetime import timezone
from types import SimpleNamespace

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from app.agents.model_config import (  # noqa: E402
    DETERMINISTIC_FAST_MODEL_SETTINGS,
    DETERMINISTIC_LONG_MODEL_SETTINGS,
    prompt_version,
)
from app.schemas.provenance import LLMCallProvenance, provenance_from_run  # noqa: E402


def _fake_run(*, provider_details=None):
    response = SimpleNamespace(
        model_name="deepseek-v4-flash",
        provider_response_id="resp-123",
        provider_details=provider_details,
        provider_name="deepseek",
    )
    usage = SimpleNamespace(input_tokens=10, output_tokens=2)
    return SimpleNamespace(response=response, usage=lambda: usage)


def test_prompt_version_is_stable_and_prefixed():
    v1 = prompt_version("You are a screener.")
    v2 = prompt_version("You are a screener.")
    assert v1 == v2
    assert v1.startswith("sha256:")
    assert len(v1) == len("sha256:") + 12
    assert prompt_version("You are a screener!") != v1


def test_deterministic_settings_pin_temperature_zero():
    assert DETERMINISTIC_FAST_MODEL_SETTINGS["temperature"] == 0.0
    assert DETERMINISTIC_LONG_MODEL_SETTINGS["temperature"] == 0.0
    assert DETERMINISTIC_FAST_MODEL_SETTINGS["timeout"] > 0
    fast_timeout = DETERMINISTIC_FAST_MODEL_SETTINGS["timeout"]
    assert DETERMINISTIC_LONG_MODEL_SETTINGS["timeout"] > fast_timeout


def test_provenance_from_run_captures_reported_model_and_usage():
    run = _fake_run(provider_details={"system_fingerprint": "fp-abc"})
    prov = provenance_from_run(
        "relevance_screener",
        run,
        model_configured="deepseek-chat",
        temperature=0.0,
        prompt="SYSTEM PROMPT",
    )
    assert isinstance(prov, LLMCallProvenance)
    assert prov.agent == "relevance_screener"
    assert prov.model_configured == "deepseek-chat"
    assert prov.model_reported == "deepseek-v4-flash"
    assert prov.provider == "deepseek"
    assert prov.provider_response_id == "resp-123"
    assert prov.system_fingerprint == "fp-abc"
    assert prov.temperature == 0.0
    assert prov.prompt_version == prompt_version("SYSTEM PROMPT")
    assert prov.input_tokens == 10
    assert prov.output_tokens == 2
    assert prov.called_at.tzinfo is not None
    assert prov.called_at.utcoffset() == timezone.utc.utcoffset(None)


def test_provenance_from_run_reads_usage_exposed_as_a_property():
    """pydantic-ai 2.x exposes ``AgentRunResult.usage`` as a property (1.x: a method).
    Observed 2026-09-03 in the v1.1.0 container (pydantic-ai 2.37.0): token counts were
    recorded as None for every claim verification because only the callable was handled."""
    usage = SimpleNamespace(input_tokens=321, output_tokens=45)
    run = SimpleNamespace(
        response=SimpleNamespace(
            model_name="deepseek-v4-flash",
            provider_response_id="resp-9",
            provider_details={},
            provider_name="deepseek",
        ),
        usage=usage,
    )
    prov = provenance_from_run(
        "claim_verification", run, model_configured="deepseek-chat", temperature=0.0, prompt="P"
    )
    assert prov.input_tokens == 321
    assert prov.output_tokens == 45


def test_provenance_from_run_tolerates_missing_provider_details():
    run = _fake_run(provider_details=None)
    prov = provenance_from_run(
        "claim_verification", run, model_configured="deepseek-chat", temperature=0.0, prompt="P"
    )
    assert prov.system_fingerprint is None
    assert prov.model_reported == "deepseek-v4-flash"


def test_provenance_is_json_serialisable():
    run = _fake_run(provider_details={})
    prov = provenance_from_run("writing", run, model_configured="m", temperature=0.7, prompt="P")
    dumped = prov.model_dump(mode="json")
    assert dumped["called_at"].endswith("Z") or "+00:00" in dumped["called_at"]
    assert dumped["temperature"] == 0.7
