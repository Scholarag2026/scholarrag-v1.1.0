"""Defaults for the T2 external-client / LLM-timeout settings.

These values are the program-wide reliability policy;
a silent default change would silently un-fix LLM-NO-TIMEOUT or the retry policy.
"""

from app.config import Settings


def _settings() -> Settings:
    """Settings built without reading a local .env, so defaults are deterministic."""
    return Settings(_env_file=None)


def test_openalex_client_defaults():
    s = _settings()
    assert s.openalex_timeout == 10.0
    assert s.openalex_rate_limit == 4.0
    assert s.openalex_batch_size == 50
    assert s.openalex_max_query_length == 500


def test_other_external_client_defaults():
    s = _settings()
    assert s.crossref_timeout == 10.0
    assert s.unpaywall_timeout == 10.0


def test_retry_policy_defaults():
    """Retry ONLY transient failures: 3 attempts, exponential backoff 1s -> 8s."""
    s = _settings()
    assert s.external_retry_attempts == 3
    assert s.external_retry_min_wait == 1.0
    assert s.external_retry_max_wait == 8.0


def test_concurrency_and_circuit_breaker_defaults():
    s = _settings()
    assert s.external_concurrency == 8
    assert s.circuit_breaker_threshold == 5


def test_llm_timeout_defaults():
    """D3: 60s for search-path agents, 300s for reasoner/long-document agents."""
    s = _settings()
    assert s.llm_timeout_seconds == 60
    assert s.llm_long_timeout_seconds == 300
    # analysis_max_retries already existed but was dead; T2 wires it into Agent(retries=).
    assert s.analysis_max_retries == 2


def test_deepseek_model_defaults_move_off_the_retired_names():
    """"deepseek-chat" and "deepseek-reasoner" are retired aliases/names; the batch pass and
    the five reasoner-backed agents (analysis, data_collection, gap, qualitative,
    quantitative) default to "deepseek-flash", the provider's current live name for the same
    served deployment. A run's own provenance record still carries the served
    model_reported and system_fingerprint regardless of which name was configured."""
    s = _settings()
    assert s.deepseek_model == "deepseek-flash"
    assert s.deepseek_reasoner_model == "deepseek-flash"


def test_screener_second_pass_model_defaults_to_the_same_model_reasoning_on():
    """Measurement found deepseek-flash with reasoning on
    and screener_second_pass_output_mode "prompted" upholds the same inclusions the second
    pass's own reasoning-on judgement needs, at a fraction of the cost and latency of a
    materially stronger model -- the judge's own gain comes from reasoning being enabled,
    not from the model's own size, so the second pass defaults to the same model
    deepseek_model uses."""
    s = _settings()
    assert s.screener_second_pass_model == "deepseek-flash"
    assert s.screener_second_pass_model == s.deepseek_model


def test_screener_second_pass_output_mode_defaults_to_prompted():
    """deepseek-flash (this setting's own default judge model) rejects a forced tool_choice
    outright while reasoning is on; "prompted" avoids ever sending one, which is what
    keeps reasoning on."""
    s = _settings()
    assert s.screener_second_pass_output_mode == "prompted"


def test_screener_second_pass_stage_defaults():
    """The second-pass stage's own concurrency and budget
    defaults, pinned the same way the sibling model default already is -- a silent change
    to either would silently change how many judge calls run at once and how long the
    once-per-job stage is allowed to run for. The concurrency=None fallback path itself
    (what run_smart_search and the harness both take) is covered in
    test_relevance_screener_agent.py, next to run_second_pass_stage's other tests."""
    s = _settings()
    assert s.screener_second_pass_concurrency == 8
    assert s.screener_second_pass_stage_budget_seconds == 1800.0
