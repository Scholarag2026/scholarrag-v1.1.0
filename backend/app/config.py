import logging

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("app.config")

DEV_ENVIRONMENTS = frozenset({"development", "test", "local"})
INSECURE_JWT_SECRET = "change-me"
#: Minimum JWT_SECRET_KEY length outside DEV_ENVIRONMENTS. A non-placeholder secret shorter
#: than this is still weak enough to be guessed or brute-forced; the placeholder check alone
#: (INSECURE_JWT_SECRET) does not catch a short real-looking value someone typed by hand.
MIN_PRODUCTION_JWT_SECRET_LENGTH = 32


class Settings(BaseSettings):
    app_env: str = "development"
    database_url: str = "postgresql+asyncpg://deepresearch:secret@localhost:5432/deepresearch"
    redis_url: str = "redis://localhost:6379/0"
    jwt_secret_key: str = "change-me"
    jwt_access_token_expire_minutes: int = 15
    jwt_refresh_token_expire_days: int = 7
    frontend_url: str = "http://localhost:3000"
    backend_url: str = "http://localhost:8000"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    # "deepseek-chat" is a retired alias; "deepseek-flash" is the provider's current
    # official name for the same served deployment (confirmed from model_reported on live
    # calls -- see evaluation/common.py's DEEPSEEK_PRICES alias table). Every agent that reads
    # this field (query, screener, verifier, citation-link, writer fallback) gets the live
    # name with no other code change; a run's own provenance record still carries the served
    # model_reported and system_fingerprint regardless of which name was configured.
    deepseek_model: str = "deepseek-flash"
    # The model app.agents.relevance_screener_agent.confirm_inclusions uses for its V2
    # (research-question-and-criteria-aware) second pass, distinct from deepseek_model so a
    # caller can run this judge over the (small) INCLUDE-only set without changing the batch
    # pass or any other agent. The shipped judge runs with reasoning left on, at the
    # provider's own default effort (no reasoning_effort parameter is sent), and answers in
    # "prompted" output mode (screener_second_pass_output_mode below): the SecondPassAnswerV2
    # schema is put in the prompt and the model answers in plain JSON text, since it rejects
    # tool-calling outright while reasoning is on. The stage runs up to
    # screener_second_pass_concurrency (8) calls concurrently, within an 1800s stage budget
    # (screener_second_pass_stage_budget_seconds) and a 420s per-call timeout
    # (SECOND_PASS_V2_MODEL_SETTINGS). On the 140-record demo set (37 second-pass candidates,
    # 8 calls) this measured a 55.4s stage wall time and 7.4 to 55.2s per call. A run's own
    # provenance record carries the served model_reported and system_fingerprint for this
    # pass, same as every other call.
    screener_second_pass_model: str = "deepseek-flash"
    # Upper bound on concurrent app.agents.relevance_screener_agent.run_second_pass_stage
    # calls in flight at once: a reasoning-on second-pass call
    # was observed to take 7.4 to 55.2s each. The stage runs once per job, after retrieval
    # (run_smart_search's own round loop), over every provisional INCLUDE the whole job
    # produced -- not once per round, which on a live search's many small rounds (the
    # demonstration corpus runs about 27 of them) would have paid one such latency before
    # every next round could even start, well over an hour of judge wall time alone against
    # a 30-minute search budget. 8 keeps the whole job's candidate set well within the
    # stage's own budget below while staying a small fraction of DeepSeek's own per-account
    # rate limit.
    screener_second_pass_concurrency: int = 8
    # Wall-clock ceiling on the one run_second_pass_stage call a job makes (every
    # provisional INCLUDE the whole job produced, judged once after run_smart_search's own
    # round loop ends), entirely separate from smart_search_max_time_minutes' own search-
    # loop ceiling: the judge is off the search's own critical path, so a job's own stopping
    # decisions are made exactly as they were before the second pass existed and are never
    # shortened by how long the judge takes. A chunk already in flight when this elapses is
    # still awaited (a call already paid for is not abandoned); a chunk that has not yet
    # started is routed to NEEDS_REVIEW with second_pass_unavailable instead. 1800s (30
    # minutes) covers roughly 250 candidates (50 chunks, 7 waves at concurrency 8) at this
    # freeze's own measured per-call latencies in the typical case; an unlucky run whose
    # every wave draws this measurement's own slowest observed call could exceed it, in
    # which case the slowest-launching chunks are queued for a human instead.
    screener_second_pass_stage_budget_seconds: float = 1800.0
    # How the V2 second-pass judge is asked to produce its structured answer: "tool"
    # (pydantic-ai's own tool-calling output, which forces a tool_choice the model must use)
    # or "prompted" (the shipped default: the SecondPassAnswerV2 schema is put in the prompt
    # instead, the model answers in plain JSON text, and pydantic-ai validates that text
    # against the same schema -- no tool_choice is ever sent). Both modes produce identical
    # decision and provenance fields. This exists because deepseek-flash (this setting's own
    # default judge model above) rejects tool_choice outright while reasoning is on
    # ("Thinking mode does not support this tool_choice"); "prompted" is what lets it answer
    # with reasoning on rather than falling back to a stronger, costlier model. "tool" is
    # still available for a judge model that does not reject the combination.
    screener_second_pass_output_mode: str = "prompted"
    # Writer-only override (env WRITING_MODEL). Unset (the default) or an empty string
    # both mean the writer uses deepseek_model like every other agent; only
    # call_deepseek in app.services.writing reads this field -- the analysis,
    # citation-link, query, screener and verifier agents always use deepseek_model.
    writing_model: str | None = None
    # Writer-only thinking-mode toggle (env WRITING_THINKING, values "enabled" or
    # "disabled"). None (unset, the default) or an empty string both mean the request
    # carries no "thinking" field at all, so the provider's own default applies; only
    # call_deepseek reads this field.
    writing_thinking: str | None = None
    # Writer-only max_tokens override (env WRITING_MAX_TOKENS). Defaults to 4096, the
    # fixed value every writing call sent before this setting existed; only
    # call_deepseek reads this field.
    writing_max_tokens: int = 4096
    openalex_email: str = ""  # For polite pool (higher rate limits)
    # OpenAlex account API key (openalex.org/settings/api). Without it, requests
    # bill the anonymous shared-egress-IP quota, which 429s when co-tenants use it
    # up; with it, usage draws on the account's daily budget + prepaid balance.
    openalex_api_key: str = ""

    # Unpaywall
    unpaywall_email: str = ""  # Required for Unpaywall API

    # Deep search
    deep_search_yield_threshold: int = 3
    deep_search_max_rounds: int = 6
    deep_search_max_scanned: int = 2000
    deep_search_max_time_minutes: float = 30.0

    # Per-round query fan-out (bounded asyncio.gather in services/search.py)
    search_fanout_concurrency: int = 4

    # Analysis agents
    analysis_batch_size: int = 5
    analysis_max_retries: int = 2
    # "deepseek-reasoner" is a retired name (absent from the provider's current price
    # table); the five reasoner-backed agents (analysis, data_collection, gap, qualitative,
    # quantitative) move to "deepseek-flash", a live name, so their calls keep being served
    # rather than erroring or silently routing somewhere uncosted. The field itself is kept so
    # a deployment that still has a working reasoner deployment can point it back with one env
    # var; a run's own provenance record carries the served model_reported either way.
    #
    # The field's own name is now a misnomer: these five agents run on the same
    # tool-calling structured-output path as everything else (app.agents.model_config's
    # LONG_MODEL_SETTINGS / DETERMINISTIC_LONG_MODEL_SETTINGS), which disables thinking mode
    # for deepseek-flash tool-calling compatibility, so none of them reasons today whatever
    # this field is set to. It is not renamed (the DEEPSEEK_REASONER_MODEL env var would
    # break for any deployment that already sets it); a future reasoning-capable deployment
    # can still use this field, but "reasoner" in its name describes the historical
    # deepseek-reasoner deployment, not this default's current behaviour.
    deepseek_reasoner_model: str = "deepseek-flash"
    gap_analysis_min_papers: int = 5
    gap_analysis_max_papers: int = 150

    # Bounded fan-out for analysis / verification loops (T5)
    analysis_concurrency: int = 4
    verification_concurrency: int = 8
    job_progress_update_every: int = 5

    # LLM call budgets (issue LLM-NO-TIMEOUT / decision D3).
    # Without an explicit timeout pydantic-ai defaults to 600s per attempt and the
    # OpenAI SDK retries twice, so one stalled call can block a job for ~1800s.
    llm_timeout_seconds: int = 60          # search-path agents (short prompts)
    llm_long_timeout_seconds: int = 300    # reasoner / whole-document analysis agents

    # External HTTP clients (OpenAlex / CrossRef / Unpaywall)
    openalex_timeout: float = 10.0
    openalex_rate_limit: float = 4.0        # requests per second, process-wide
    openalex_batch_size: int = 50           # max OR values per OpenAlex filter
    openalex_max_query_length: int = 500    # chars, after wildcard sanitization
    crossref_timeout: float = 10.0
    unpaywall_timeout: float = 10.0

    # Shared retry policy: transient failures only (429/5xx/connect/read timeouts)
    external_retry_attempts: int = 3
    external_retry_min_wait: float = 1.0
    external_retry_max_wait: float = 8.0
    # 429s with Retry-After above this many seconds fail fast (quota-length blocks
    # cannot be waited out); at or below it the retry waits exactly that long.
    external_retry_after_cap: float = 20.0

    # Default bounded-gather width for callers fanning out over external calls
    external_concurrency: int = 8

    # Circuit breaker: consecutive failures per source per job before skipping (D2)
    circuit_breaker_threshold: int = 5

    # File storage
    storage_path: str = "./data/uploads"

    # Smart search
    smart_search_max_scanned: int = 10000
    smart_search_max_time_minutes: float = 30.0
    smart_search_batch_size: int = 10
    # Stop rule: the loop never stops for lack of new inclusions before
    # smart_search_min_rounds rounds have run, and afterwards stops only once
    # smart_search_dry_round_patience consecutive rounds have each added zero new
    # included papers. Other stop reasons (time, max_scanned, cancellation, retrieval
    # failure, query-generation failure) are unaffected by either setting.
    smart_search_min_rounds: int = 3
    smart_search_dry_round_patience: int = 2

    # Citation graph
    graph_max_nodes: int = 80
    graph_expand_max_refs: int = 15
    graph_expand_max_cites: int = 15

    # Citation graph build (OpenAlex). The shared rate limiter in the OpenAlex client
    # does the throttling — this semaphore only bounds in-flight papers, and there is
    # deliberately no sleep anywhere in the build loop.
    graph_build_concurrency: int = 4
    graph_build_max_minutes: int = 20        # per-job wall-clock budget
    graph_max_refs_per_paper: int = 100      # referenced_works hydrated per paper
    graph_max_cites_per_paper: int = 50      # citing works fetched per paper
    graph_stale_job_minutes: int = 10        # D12c: abandoned-build window for the 409 guard

    # Database connection pool
    db_pool_size: int = 10
    db_max_overflow: int = 10
    db_pool_timeout: int = 10
    db_pool_recycle: int = 1800

    # Job durability
    job_stale_after_minutes: int = 10
    job_reaper_interval_seconds: int = 300

    model_config = SettingsConfigDict(env_file=".env")

    @field_validator("writing_model", mode="before")
    @classmethod
    def _blank_writing_model_is_unset(cls, v):
        """``WRITING_MODEL=`` left blank in `.env` is an empty string, not an unset
        variable; treat it the same as unset rather than sending "" as a model name."""
        if v == "":
            return None
        return v

    @field_validator("writing_thinking", mode="before")
    @classmethod
    def _blank_writing_thinking_is_unset(cls, v):
        """``WRITING_THINKING=`` left blank in `.env` is an empty string, not an unset
        variable; treat it the same as unset rather than sending "" as a thinking type."""
        if v == "":
            return None
        return v

    @field_validator("writing_max_tokens", mode="before")
    @classmethod
    def _blank_writing_max_tokens_is_default(cls, v):
        """``WRITING_MAX_TOKENS=`` left blank -- including docker-compose.yml's own
        ``${WRITING_MAX_TOKENS:-}``, which forwards the empty string whenever the shell
        variable is unset -- is not a valid integer; treat it as the field's own
        default (4096) rather than failing Settings construction."""
        if v == "":
            return 4096
        return v

settings = Settings()


def validate_security_settings(s: Settings) -> None:
    """Fail fast when a non-development deployment still uses the placeholder JWT secret.

    ``jwt_secret_key`` defaults to ``"change-me"``; if the env var is ever unset in a
    real deployment every access/refresh token becomes forgeable. Called from
    ``app.main`` at import time, before the FastAPI app is created.
    """
    placeholder = not s.jwt_secret_key or s.jwt_secret_key == INSECURE_JWT_SECRET
    if s.app_env in DEV_ENVIRONMENTS:
        if placeholder:
            logger.warning(
                "jwt_secret_key is unset or the insecure default %r (app_env=%r). "
                "This is tolerated for local development only — set JWT_SECRET_KEY "
                "and APP_ENV=production before deploying.",
                INSECURE_JWT_SECRET,
                s.app_env,
            )
        return
    if placeholder:
        raise RuntimeError(
            f"JWT_SECRET_KEY is unset or still the insecure default "
            f"{INSECURE_JWT_SECRET!r} while APP_ENV={s.app_env!r}. Set a strong "
            f"JWT_SECRET_KEY, or set APP_ENV to one of "
            f"{sorted(DEV_ENVIRONMENTS)} for local development."
        )
    if len(s.jwt_secret_key) < MIN_PRODUCTION_JWT_SECRET_LENGTH:
        raise RuntimeError(
            f"JWT_SECRET_KEY is only {len(s.jwt_secret_key)} character(s) while "
            f"APP_ENV={s.app_env!r}; it must be at least "
            f"{MIN_PRODUCTION_JWT_SECRET_LENGTH} characters. Set a longer JWT_SECRET_KEY, "
            f"or set APP_ENV to one of {sorted(DEV_ENVIRONMENTS)} for local development."
        )
