"""Shared pydantic-ai model settings for every agent (issue LLM-NO-TIMEOUT / D3).

Two tiers:

``FAST_MODEL_SETTINGS``
    Short prompts on the search / interactive path — query generation, expansion,
    relevance screening, metadata extraction, rule merges, scope refinement.

``LONG_MODEL_SETTINGS``
    Reasoner-backed agents and whole-document analysis — quality scoring, gap
    analysis, research design, data collection, quant/qual, deep analysis and claim
    verification.

``ANALYSIS_MODEL_SETTINGS`` adds an explicit output budget to the long tier for the
full-text analysis agent, whose reply is the longest any agent produces; see its own
comment below.

Without an explicit ``timeout`` pydantic-ai uses a 600s default per attempt and the
underlying OpenAI SDK retries timeouts twice, so one stalled DeepSeek call can hold a
job for ~1800s. ``AGENT_RETRIES`` finally wires ``settings.analysis_max_retries``,
which was declared in config and referenced nowhere.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.settings import ModelSettings

from app.config import settings
from app.schemas.provenance import prompt_version  # noqa: F401  (re-exported for agents)

logger = logging.getLogger(__name__)

#: Every agent in this module uses pydantic-ai's structured ``output_type``, which
#: forces a specific ``tool_choice`` under the hood -- DeepSeek's API rejects that combination
#: outright ("Thinking mode does not support this tool_choice", a 400) once thinking mode is
#: on, and "deepseek-flash" (unlike the retired "deepseek-chat" alias) defaults to it on. The
#: same call shape ``writing.py``'s own ``settings.writing_thinking`` toggle already disables
#: for the (non-tool-calling) writer path. Sent as ``extra_body`` because DeepSeek's ``thinking``
#: field is a provider extension, not part of pydantic-ai's own generic ``thinking`` setting
#: (which maps to a reasoning-effort level, not this on/off switch). Harmless for a call that
#: was already working (the field is simply absent from the response either way); a
#: deployment that stops needing it can drop this dict without touching any agent module.
_DISABLE_THINKING: dict[str, Any] = {"extra_body": {"thinking": {"type": "disabled"}}}

FAST_MODEL_SETTINGS = ModelSettings(timeout=settings.llm_timeout_seconds, **_DISABLE_THINKING)
LONG_MODEL_SETTINGS = ModelSettings(
    timeout=settings.llm_long_timeout_seconds, **_DISABLE_THINKING
)

# Deterministic variants for the two evaluated decision agents (relevance screening and
# claim verification): same timeout tiers, sampling temperature pinned to 0 so repeated
# runs of the same protocol are as reproducible as the provider allows.
DETERMINISTIC_FAST_MODEL_SETTINGS = ModelSettings(
    timeout=settings.llm_timeout_seconds, temperature=0.0, **_DISABLE_THINKING
)
DETERMINISTIC_LONG_MODEL_SETTINGS = ModelSettings(
    timeout=settings.llm_long_timeout_seconds, temperature=0.0, **_DISABLE_THINKING
)

#: The relevance screener's V2 second pass (``app.agents.relevance_screener_agent
#: .confirm_inclusions`` with a non-empty ``research_question``, i.e. the inclusion-only
#: population/outcome/study-type re-check) needs reasoning left ON, not disabled: an
#: independent review found that disabling it (as every other tier above does, for
#: ``deepseek-flash`` tool-calling compatibility) collapsed this judge's own output from
#: about 2,300 tokens per record to about 70 and its demotion rate from 18 of 38 candidates
#: to 5 of 38, on the same model, prompt, schema and shown text -- reasoning off is not
#: behaviour-preserving for this one call shape, even though it is for every other tier's
#: agent (each of which ran the same served build with reasoning off under the retired
#: ``deepseek-chat`` alias already). No ``extra_body`` thinking-disable here, deliberately.
#: The timeout widens past ``DETERMINISTIC_LONG_MODEL_SETTINGS``'s 300s: reasoning-on calls
#: at this judge's own configured model were observed to take 7.4 to 55.2 seconds each, so
#: 300s would time out the slowest ones and route their batch to
#: ``second_pass_unavailable``.
SECOND_PASS_V2_MODEL_SETTINGS = ModelSettings(timeout=420.0, temperature=0.0)

#: Full-text analysis (``app.agents.deep_analysis_agent``): the long timeout tier plus an
#: explicit output budget. Without ``max_tokens`` the ceiling is whatever the provider
#: defaults to on the day, and a reply that runs past it comes back as
#: ``IncompleteToolCall`` with no analysis and no evidence at all for that paper -- which
#: is what happened to 8 of the 12 demo seed papers, all of them mid-tool-call. Pinning
#: the budget here makes the
#: ceiling a property of this deployment rather than of the provider's defaults, and the
#: analysis agent's own per-paper item cap and per-field length limits keep a reply well
#: inside it.
ANALYSIS_MODEL_SETTINGS = ModelSettings(
    timeout=settings.llm_long_timeout_seconds, max_tokens=16384, **_DISABLE_THINKING
)

AGENT_RETRIES = settings.analysis_max_retries


def attach_system_fingerprint(model_response: Any, raw_response: Any) -> Any:
    """Copy ``raw_response.system_fingerprint`` into ``model_response.provider_details``.

    pydantic-ai's OpenAI mapping keeps ``finish_reason``/``logprobs`` but drops the
    top-level ``system_fingerprint`` DeepSeek returns. Best-effort by design: any failure
    leaves the response untouched and the provenance record reports ``None``.
    """
    try:
        fingerprint = getattr(raw_response, "system_fingerprint", None)
        if not fingerprint:
            return model_response
        details = getattr(model_response, "provider_details", None)
        if not isinstance(details, dict):
            details = {}
        details["system_fingerprint"] = str(fingerprint)
        model_response.provider_details = details
    except Exception:
        logger.debug("Could not attach system_fingerprint to the model response", exc_info=True)
    return model_response


class DeepSeekChatModel(OpenAIChatModel):
    """OpenAI-compatible chat model that also records DeepSeek's ``system_fingerprint``."""

    def _process_response(self, response):  # type: ignore[override]
        model_response = super()._process_response(response)
        return attach_system_fingerprint(model_response, response)


def build_deepseek_model(model_name: str) -> Model | str:
    """DeepSeek chat model for ``model_name`` with best-effort fingerprint capture.

    Falls back to the plain ``"deepseek:<name>"`` model string (pydantic-ai's default
    wiring) if the subclass cannot be constructed, so a pydantic-ai change can only
    degrade ``system_fingerprint`` to ``None`` — it can never take an agent down.
    """
    try:
        provider = DeepSeekProvider(api_key=settings.deepseek_api_key or None)
        return DeepSeekChatModel(model_name, provider=provider)
    except Exception:
        logger.warning("Falling back to default DeepSeek model wiring", exc_info=True)
        return f"deepseek:{model_name}"
