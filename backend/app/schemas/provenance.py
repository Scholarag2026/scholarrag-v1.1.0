"""Provenance of a single LLM call.

Reviewers of the SoftwareX submission pointed out that the screening and claim-
verification steps ran against a floating model alias (``deepseek-chat``) with no
record of which model actually answered, at what temperature, or under which prompt.
``LLMCallProvenance`` is the shared record every evaluated agent now attaches to its
result so that a job's outcome can be tied to the model that produced it.

``model_configured`` is what the deployment asked for; ``model_reported`` is what the
provider says it served (``ModelResponse.model_name``), which for DeepSeek resolves the
alias to a concrete family/version string such as ``deepseek-v4-flash``.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def prompt_version(prompt: str) -> str:
    """Stable short identifier for a system prompt: ``sha256:`` + first 12 hex digits."""
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"sha256:{digest[:12]}"


class LLMCallProvenance(BaseModel):
    """Which model answered, under which settings, for one agent invocation."""

    agent: str
    model_configured: str
    model_reported: str | None = None
    provider: str = "deepseek"
    provider_response_id: str | None = None
    system_fingerprint: str | None = None
    temperature: float | None = None
    prompt_version: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    #: How this call asked the model for its structured answer: "tool" (pydantic-ai's own
    #: tool-calling output, a forced tool_choice) or "prompted" (the schema is put in the
    #: prompt and the model answers in plain JSON text, validated after the fact -- no
    #: tool_choice sent). ``None`` for a call this concept does not apply to (every agent
    #: except the relevance screener's own V2 second pass, at present).
    output_mode: str | None = None
    #: The reasoning effort level reported for a thinking-mode call -- the value pydantic-ai
    #: actually sent (when the caller configures one explicitly), or a documented default
    #: name (e.g. "high") when the caller leaves it to the provider's own default and so has
    #: no explicit value to report. ``None`` for a call this concept does not apply to (a
    #: call made with thinking disabled, or any agent that does not use thinking mode at
    #: all).
    reasoning_effort: str | None = None
    called_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    #: How many decisions a deterministic code guard changed after the model answered
    #: (the relevance screener's unanchored-EXCLUDE demotion). Zero for
    #: every agent that has no such guard; the relevance screener populates it via
    #: :func:`provenance_from_run`'s ``guard_conversions`` keyword.
    guard_conversions: int = 0


def _get(obj: Any, name: str, default: Any = None) -> Any:
    try:
        value = getattr(obj, name)
    except AttributeError:
        return default
    return default if value is None else value


def provenance_from_run(
    agent: str,
    run_result: Any,
    *,
    model_configured: str,
    temperature: float | None,
    prompt: str,
    guard_conversions: int = 0,
    output_mode: str | None = None,
    reasoning_effort: str | None = None,
) -> LLMCallProvenance:
    """Build a provenance record from a pydantic-ai ``AgentRunResult``.

    Tolerates partial objects: any field the run result does not expose is ``None``.
    ``guard_conversions`` defaults to 0 for every caller that has no post-model guard;
    the relevance screener is, at present, the only caller that passes a non-zero value.
    ``output_mode``/``reasoning_effort`` default to ``None`` for every caller this does not
    apply to; the relevance screener's own V2 second pass is, at present, the only caller
    that passes them.
    """
    response = _get(run_result, "response")
    details = _get(response, "provider_details") or {}
    if not isinstance(details, dict):
        details = {}

    input_tokens: int | None = None
    output_tokens: int | None = None
    # pydantic-ai 1.x exposes ``usage()`` as a method, 2.x as a property; accept both.
    usage = _get(run_result, "usage")
    if callable(usage):
        try:
            usage = usage()
        except Exception:  # pragma: no cover - defensive
            usage = None
    input_tokens = _get(usage, "input_tokens")
    output_tokens = _get(usage, "output_tokens")

    return LLMCallProvenance(
        agent=agent,
        model_configured=model_configured,
        model_reported=_get(response, "model_name"),
        provider=_get(response, "provider_name", "deepseek") or "deepseek",
        provider_response_id=_get(response, "provider_response_id"),
        system_fingerprint=details.get("system_fingerprint"),
        temperature=temperature,
        prompt_version=prompt_version(prompt),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        called_at=datetime.now(timezone.utc),
        guard_conversions=guard_conversions,
        output_mode=output_mode,
        reasoning_effort=reasoning_effort,
    )
