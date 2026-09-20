"""Smart Search pipeline: PRISMA two-stage filtering with iterative query expansion.

Every identified record ends the job in exactly one place: ``papers`` (included),
``needs_review`` (stage 2 protocol-eligibility screening reached the record but the shown
text could not decide it), ``excluded`` (stage ``wos`` or ``llm``, with the reason) or
``unscreened`` (no decision: failed model call, time limit, cancellation), so that
``flow`` reconciles: ``identified == duplicates_removed + stage1_excluded +
stage2_excluded + needs_review + unscreened + included``. Stage 1 (Web of Science venue
filter) is optional; the screening decisions (INCLUDE/EXCLUDE/NEEDS_REVIEW, plus
any criterion id and verbatim quote) and the LLM provenance behind them
are recorded for export. INCLUDE is the only automatic route into the library;
NEEDS_REVIEW records are never included nor excluded, only flagged for a human to read.
"""

from __future__ import annotations

import logging
import re
import time
import traceback
from datetime import date
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.model_config import DETERMINISTIC_FAST_MODEL_SETTINGS
from app.agents.query_generator_agent import (
    QUERY_GENERATOR_PROMPT,
    generate_search_queries_with_provenance,
)
from app.agents.relevance_screener_agent import (
    GUARD_REASON_SECOND_PASS_UNAVAILABLE,
    SCREENER_PROMPT_VERSION,
    SECOND_PASS_PROMPT_V2_VERSION,
    SECOND_PASS_REASONING_EFFORT_DEFAULT,
    SECOND_PASS_STAGE_BATCH_SIZE,
    SecondPassStageDecision,
    SecondPassStageResult,
    apply_table_of_contents_demotion,
    apply_type_demotion,
    build_shown_texts,
    run_second_pass_stage,
    screen_papers,
)
from app.config import settings
from app.models.analysis_job import JobStatus
from app.schemas.provenance import LLMCallProvenance, prompt_version
from app.services.screening_record import STAGE_LLM, STAGE_WOS, paper_record
from app.services.search import SearchService, gather_search_results
from app.services.task import should_abort, update_job_status
from app.services.wos_import import get_wos_journal_count, is_wos_indexed_bulk

logger = logging.getLogger(__name__)

# Regex: strip anything that is not a letter or digit, then lowercase
_PUNCT_RE = re.compile(r"[^a-z0-9]")

WOS_FILTER_VALUES = ("auto", "on", "off")
QUERY_GENERATOR_PROMPT_VERSION = prompt_version(QUERY_GENERATOR_PROMPT)

# Reasons written to the screening record for papers that did not get an LLM decision.
REASON_NOT_INDEXED = "excluded at stage 1: journal not in the imported Web of Science list"
REASON_NO_ISSN = "excluded at stage 1: no ISSN to match against the Web of Science journal list"
REASON_TIME_LIMIT = "not screened: time limit reached before screening"
REASON_CANCELLED = "not screened: job cancelled before screening"


def _normalize_title(title: str) -> str:
    """Normalize a paper title for deduplication: strip punctuation and lowercase."""
    return _PUNCT_RE.sub("", title.lower())


class _ScreenerProvenance:
    """Aggregate per-call ``LLMCallProvenance`` records into the job-level summary.

    ``add`` is called once per model call, not once per batch: a batch whose response left
    a paper undecided makes a second, re-ask call, so the caller adds
    ``screened.provenance`` and, when present, ``screened.reask_provenance`` -- two calls,
    two additions -- so ``calls``/``input_tokens``/``output_tokens`` reflect what the batch
    actually spent, and a model or fingerprint change on the re-ask still lands in the
    ``models``/``fingerprints`` sets rather than being dropped.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.models: set[str] = set()
        self.fingerprints: set[str] = set()

    def add(self, provenance: LLMCallProvenance) -> None:
        self.calls += 1
        self.input_tokens += provenance.input_tokens or 0
        self.output_tokens += provenance.output_tokens or 0
        if provenance.model_reported:
            self.models.add(provenance.model_reported)
        if provenance.system_fingerprint:
            self.fingerprints.add(provenance.system_fingerprint)

    def summary(self) -> dict:
        return {
            "agent": "relevance_screener",
            "model_configured": settings.deepseek_model,
            "model_reported": sorted(self.models),
            "system_fingerprints": sorted(self.fingerprints),
            "temperature": DETERMINISTIC_FAST_MODEL_SETTINGS.get("temperature"),
            "prompt_version": SCREENER_PROMPT_VERSION,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class _SecondPassProvenance:
    """The inclusion-only second pass's own job-level block: one
    :func:`run_second_pass_stage` call for the whole job, made
    once the round loop ends, over every provisional INCLUDE any round produced. ``add`` is
    called once per chunk the stage actually sent -- a job with no provisional INCLUDE at
    all never calls it, and this block's every field stays at its zero/empty default.

    ``add_stage_wall_time`` records the stage's own single wall-clock span, distinct from
    the sum of the calls' own latencies (``per_call_latencies_s``), since those overlap
    under ``settings.screener_second_pass_concurrency`` and summing them would double-count
    the time actually spent. ``record_outcome_counts`` records how the stage's own decisions
    landed: how many candidates it judged, how many it demoted for an unconfirmed
    population/outcome/study-type ("not_established"), and how many it could not reach at
    all (a chunk that raised, timed out, or never launched for lack of remaining stage
    budget -- "second_pass_unavailable"), so a reader of the job's own provenance (and the
    exported screening record, which carries this block verbatim) sees the whole stage as
    one unit: candidates in, calls made, tokens spent, and how each candidate's own
    provisional INCLUDE was resolved.

    ``per_call_input_tokens``/``per_call_output_tokens``
    are kept parallel to ``per_call_latencies_s`` -- one entry per chunk actually sent, in
    the same order -- rather than only the stage-level sum, so a reader can see each call's
    own token usage the way ``per_call_latencies_s`` already shows each call's own latency,
    not just the total. ``calls_missing_usage`` counts a real call (one this block's own
    ``calls`` already includes) whose own ``LLMCallProvenance`` reported both token counts
    as zero -- the shape ``pydantic_ai.models.openai._map_usage`` returns
    (``RequestUsage()``, every field zero) when the raw provider response's own ``usage``
    field is ``None``, rather than an accounting bug in this block or in
    :func:`app.schemas.provenance.provenance_from_run`. A genuinely free call (this stage
    never makes one -- every chunk sends real prompt and record text) would be
    indistinguishable from this without the counter; with it, a reader can tell "the
    provider omitted usage on N of the stage's own calls" from "this run really used zero
    tokens", which a bare
    ``input_tokens: 0``/``output_tokens: 0`` on the whole stage cannot say on its own.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.models: set[str] = set()
        self.fingerprints: set[str] = set()
        self.stage_wall_time_s = 0.0
        self.per_call_latencies_s: list[float] = []
        self.per_call_input_tokens: list[int] = []
        self.per_call_output_tokens: list[int] = []
        self.calls_missing_usage = 0
        self.n_candidates = 0
        self.n_demoted = 0
        self.n_unavailable = 0

    def add(self, provenance: LLMCallProvenance, latency_s: float | None = None) -> None:
        self.calls += 1
        call_input_tokens = provenance.input_tokens or 0
        call_output_tokens = provenance.output_tokens or 0
        self.input_tokens += call_input_tokens
        self.output_tokens += call_output_tokens
        self.per_call_input_tokens.append(call_input_tokens)
        self.per_call_output_tokens.append(call_output_tokens)
        if call_input_tokens == 0 and call_output_tokens == 0:
            self.calls_missing_usage += 1
        if provenance.model_reported:
            self.models.add(provenance.model_reported)
        if provenance.system_fingerprint:
            self.fingerprints.add(provenance.system_fingerprint)
        if latency_s is not None:
            self.per_call_latencies_s.append(latency_s)

    def add_stage_wall_time(self, seconds: float) -> None:
        self.stage_wall_time_s += seconds

    def record_outcome_counts(
        self, *, n_candidates: int, n_demoted: int, n_unavailable: int
    ) -> None:
        self.n_candidates += n_candidates
        self.n_demoted += n_demoted
        self.n_unavailable += n_unavailable

    def summary(self) -> dict:
        return {
            "agent": "relevance_screener_second_pass_v2",
            "model_configured": settings.screener_second_pass_model,
            "model_reported": sorted(self.models),
            "system_fingerprints": sorted(self.fingerprints),
            # DeepSeek's own
            # thinking-mode guide states temperature is not supported in thinking mode, so
            # the 0.0 this judge's own ModelSettings sends is silently ignored by the API --
            # reporting it as this call's own temperature would be false. output_mode and
            # reasoning_effort take its place; see confirm_inclusions's own per-call
            # provenance for the same fields, this block's own source.
            "output_mode": settings.screener_second_pass_output_mode,
            "reasoning_effort": SECOND_PASS_REASONING_EFFORT_DEFAULT,
            "prompt_version": SECOND_PASS_PROMPT_V2_VERSION,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "stage_wall_time_s": round(self.stage_wall_time_s, 3),
            "per_call_latencies_s": [round(x, 3) for x in self.per_call_latencies_s],
            # One entry per chunk actually sent, parallel to
            # ``per_call_latencies_s`` above and to ``calls`` -- see this class's own
            # docstring for why ``calls_missing_usage`` is kept alongside them.
            "per_call_input_tokens": list(self.per_call_input_tokens),
            "per_call_output_tokens": list(self.per_call_output_tokens),
            "calls_missing_usage": self.calls_missing_usage,
            #: Candidates the stage judged, how many it demoted to NEEDS_REVIEW for an
            #: unconfirmed answer, and how many it could not reach at all (call failure,
            #: timeout, or stage-budget exhaustion) -- candidates - demoted - unavailable is
            #: the count still standing as INCLUDE once the stage finished.
            "candidates": self.n_candidates,
            "demotions": self.n_demoted,
            "unavailable": self.n_unavailable,
        }


class _QueryGeneratorProvenance:
    """Aggregate per-call ``LLMCallProvenance`` records from the query generator into
    the job-level summary, the same pattern
    ``_ScreenerProvenance`` above uses for the relevance screener. ``add`` is called
    once per round that actually calls the generator -- a round that replays a prior
    run's own query list (``queries_override``) never calls it at all, and adds
    nothing here, exactly as it adds nothing to ``rounds_log``'s own query-generation
    accounting.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.models: set[str] = set()
        self.fingerprints: set[str] = set()

    def add(self, provenance: LLMCallProvenance) -> None:
        self.calls += 1
        self.input_tokens += provenance.input_tokens or 0
        self.output_tokens += provenance.output_tokens or 0
        if provenance.model_reported:
            self.models.add(provenance.model_reported)
        if provenance.system_fingerprint:
            self.fingerprints.add(provenance.system_fingerprint)

    def summary(self) -> dict:
        return {
            "agent": "query_generator",
            "model_configured": settings.deepseek_model,
            "model_reported": sorted(self.models),
            "system_fingerprints": sorted(self.fingerprints),
            "prompt_version": QUERY_GENERATOR_PROMPT_VERSION,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


async def run_smart_search(
    project_id: UUID,
    job_id: UUID,
    query: str,
    session_factory: async_sessionmaker[AsyncSession],
    inclusion_criteria: list[str] | None = None,
    exclusion_criteria: list[str] | None = None,
    wos_filter: str = "auto",
    inclusion_criteria_stages: list[str] | None = None,
    exclusion_criteria_stages: list[str] | None = None,
    queries_override: list[list[str]] | None = None,
    publication_date_max: date | None = None,
) -> None:
    """Execute the Smart Search pipeline as a background task.

    Stages per round:
      1. AI-generated queries (or raw user query as fallback)
      2. Multi-API search via SearchService
      3. Dedup by DOI + normalized title against all previously seen papers
      4. Stage 1 filter — WoS Core Collection journal lookup (optional, see ``wos_filter``)
      5. Stage 2 filter — AI relevance screening (PRISMA recall-first)
      6. Data-driven stopping: a round that screens cleanly but includes nothing new does
         not by itself end the job -- the loop keeps going until
         ``settings.smart_search_min_rounds`` rounds have run, and after that stops only
         once ``settings.smart_search_dry_round_patience`` such rounds have happened in a
         row, still reported as ``stop_reason="no_new_included"``.
         ``stop_reason`` is ``"screening_failed"`` rather than ``"no_new_included"`` when a
         round left papers unscreened because the model call failed (this stop is immediate,
         not gated by min_rounds/patience), and ``"retrieval_failed"`` rather than
         ``"no_new_papers"`` when every query of the round raised, e.g. an OpenAlex quota
         block; the failures are stored under ``result["retrieval_failures"]`` as
         ``(query, "<ExceptionClass>: <message>")``

    ``inclusion_criteria_stages``/``exclusion_criteria_stages``: parallel to
    ``inclusion_criteria``/``exclusion_criteria``, each "abstract" or "full_text". A
    full-text *inclusion* criterion is never tested at this stage, and an INCLUDE that has
    not already confirmed it in the shown text is never shipped as INCLUDE: it is routed to
    NEEDS_REVIEW with guard reason ``"full_text_to_confirm"`` (``screening_guard_reason`` on
    the paper dict), structurally, whatever the model itself answered, while the still-open
    ids stay visible on ``screening_to_confirm`` on the paper dict for audit. A full-text
    *exclusion* criterion can never ground an EXCLUDE, and grounds a NEEDS_REVIEW only behind
    a verbatim quote of explicit contrary evidence (``screening_guard_reason`` on the paper
    dict, and ``flow["needs_review_by_reason"]`` in the job result). Empty (the default)
    means every criterion is "abstract", the pre-S8 behaviour.

    Two further checks run on every record still standing as INCLUDE once the batch pass and
    its guard have run: a record OpenAlex's own ``openalex_type``/``is_paratext`` marks as paratext,
    book, book review, reference entry, conference abstract or similar, or whose shown
    abstract reads as a table of contents, is demoted to NEEDS_REVIEW (guard reason
    ``"nonarticle_type"`` or ``"table_of_contents"``) before ever reaching the second pass
    below. Whatever survives both checks is sent, batched, to a second, inclusion-only pass
    (:func:`app.agents.relevance_screener_agent.confirm_inclusions`, at
    ``settings.screener_second_pass_model``) that re-asks the population, outcome and study-
    type questions directly against the research question and the same numbered criteria;
    a "not established" answer to any of them demotes the record to NEEDS_REVIEW (guard
    reason ``"not_established"``), and the per-question answers are kept on
    ``screening_second_pass`` on the paper dict. The second pass only ever demotes or leaves
    a record unchanged -- it never excludes and never promotes -- and a second-pass call
    that fails outright never silently ships its batch's records as INCLUDE: they are
    routed to NEEDS_REVIEW instead, with guard reason ``"second_pass_unavailable"``, the
    same "never default to a lenient outcome silently" rule the rest of this pipeline
    applies to a failed model call.

    ``wos_filter``: ``"on"`` always applies Stage 1, ``"off"`` never does, and
    ``"auto"`` applies it only when the WoS journal table is non-empty. The journal count
    is read once per job. When Stage 1 is inactive every paper gets ``wos_indexed=None``
    and proceeds to Stage 2, and the ``no_wos_papers`` stop reason cannot fire.

    Screening record: a batch whose model call fails is recorded under
    ``unscreened`` with ``"<ExceptionClass>: <message>"`` — it is never silently included.
    Papers still waiting when the time budget or a cancellation ends the round are also
    ``unscreened``. ``flow`` counts reconcile to ``identified``. ``result["provenance"]``
    also carries ``rounds`` — one ``{round, queries: [{query, returned,
    new_unique}], screened, included_new, needs_review_new, replayed}`` entry per round
    whose queries ran, so a reviewer can see exactly what was searched and found each
    round — beside ``settings`` (``min_rounds``, ``dry_round_patience``), the stop rule's
    two thresholds.

    ``queries_override``: a list of query lists, one per round. Round
    ``r`` (1-indexed) uses ``queries_override[r - 1]`` verbatim instead of calling the
    query generator agent, and its round log entry carries ``replayed: true`` and
    ``source: "queries_override"``. A round past the end of the given lists falls back to
    normal query generation, so the run only stops right after the last given list when
    the data-driven stopping rule (Step F) would also have stopped there; if that rule
    would instead run another round, that round is generated normally. This makes a Smart
    Search run reproduce a prior run's own queries, round by round, without asking the
    query generator agent to invent new ones.

    ``publication_date_max``: forwarded to OpenAlex as a
    ``to_publication_date`` filter on every search call this job makes, and recorded in
    ``result["criteria"]``. Together with ``queries_override`` this lets a run reproduce a
    prior run's own corpus and queries, though it does not eliminate every source of
    drift: OpenAlex's index can still add or correct records dated on or before the cutoff
    after the fact, and the relevance screener is not perfectly deterministic run to run.

    Safety limits (all checked at round boundaries AND inside a round):
      - total_scanned > settings.smart_search_max_scanned
      - elapsed time > settings.smart_search_max_time_minutes — re-checked before every
        individual query in Step B and before Steps D and E, so a slow round cannot overshoot
      - job cancellation (task_service.should_abort) — checked between rounds and between
        queries; a cancelled job finishes with JobStatus.cancelled and its partial results
    """
    start_time = time.monotonic()
    time_budget_seconds = settings.smart_search_max_time_minutes * 60.0
    search_service = SearchService()
    batch_size = settings.smart_search_batch_size

    # Accumulate results across rounds
    included_papers: list[dict] = []
    # Every pass-1 INCLUDE that has
    # cleared the batch pass's own guard and the two deterministic post-model devices, from
    # every round, whatever a later round's own stopping decision does -- the second pass
    # runs once, over this whole list, after the round loop ends (see below "Stage 2b").
    # During the round loop these are provisional includes, exactly the shipped-before-the-
    # second-pass v2 behaviour: they join included_papers immediately, so a round's own
    # stopping decision (Step F) and a later round's own query generation (existing_papers)
    # see them right away, unaffected by whether the second pass later demotes any of them.
    job_provisional_includes: list[dict] = []
    job_provisional_shown: list[str] = []
    needs_review_records: list[dict] = []
    excluded_records: list[dict] = []
    unscreened_records: list[dict] = []
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    total_scanned = 0
    duplicates_removed = 0
    retrieval_failures: list[tuple[str, str]] = []
    stage1_screened = 0
    stage1_excluded = 0
    stage2_screened = 0
    stage2_excluded = 0
    needs_review_by_reason: dict[str, int] = {}
    rounds_log: list[dict] = []
    round_number = 0
    consecutive_dry_rounds = 0  # rounds in a row that screened cleanly but added no INCLUDE
    stop_reason: str | None = None
    was_cancelled = False
    stage1_applied = False
    screener_provenance = _ScreenerProvenance()
    second_pass_provenance = _SecondPassProvenance()
    query_generator_provenance = _QueryGeneratorProvenance()

    def _elapsed_minutes() -> float:
        return (time.monotonic() - start_time) / 60.0

    def _budget_exhausted() -> bool:
        return (time.monotonic() - start_time) >= time_budget_seconds

    async def _cancelled() -> bool:
        async with session_factory() as abort_db:
            return await should_abort(abort_db, job_id)

    async def _stop_search() -> bool:
        """Consulted by the fan-out before every individual query in Step B."""
        nonlocal stop_reason, was_cancelled
        if _budget_exhausted():
            stop_reason = "time_limit"
            return True
        if await _cancelled():
            stop_reason = "cancelled"
            was_cancelled = True
            return True
        return False

    def _mark_unscreened(papers: list[dict], stage: str, reason: str) -> None:
        for paper_dict in papers:
            unscreened_records.append(paper_record(paper_dict, stage=stage, reason=reason))

    def _route_needs_review(
        paper_dict: dict, guard_reason: str, *, round_log: dict | None
    ) -> None:
        """Common bookkeeping for every record routed to NEEDS_REVIEW, whatever demoted it
        (the batch pass's own guard, the two deterministic post-model checks, or the
        job-level second pass stage). ``round_log`` is a required keyword argument, not a
        closed-over one: a demotion discovered during the round
        loop passes the *current* round's own dict, so that round's own ``needs_review_new``
        is incremented; a demotion Stage 2b makes after the round loop has ended passes
        ``None``, since attributing a job-level outcome to whichever round happened to run
        last would be wrong -- that count lives in the job-level ``screener_second_pass``
        provenance block instead."""
        needs_review_records.append(paper_dict)
        # The guard already attributes "full_text_criterion" /
        # "unquoted_criterion" on a NEEDS_REVIEW naming a full-text exclusion criterion, and
        # now "no_abstract" on an EXCLUDE it demoted for having no abstract at all, whether
        # or not it demoted anything, so guard_reason alone is authoritative here; a
        # guard-silent row is a genuine model NEEDS_REVIEW, bucketed by whether the record
        # had an abstract at all -- landing in the same "no_abstract" bucket as the guard's
        # own attribution above.
        if guard_reason:
            bucket = guard_reason
        elif not (paper_dict.get("abstract") or "").strip():
            bucket = "no_abstract"
        else:
            bucket = "undecidable"
        needs_review_by_reason[bucket] = needs_review_by_reason.get(bucket, 0) + 1
        if round_log is not None:
            round_log["needs_review_new"] += 1

    # -- Mark job as running ------------------------------------------------
    async with session_factory() as db:
        await update_job_status(
            db, job_id,
            status=JobStatus.running,
            progress=0.0,
            progress_message="Smart Search starting...",
        )

    try:
        if wos_filter not in WOS_FILTER_VALUES:
            raise ValueError(f"wos_filter must be one of {WOS_FILTER_VALUES}, got {wos_filter!r}")

        # -- Stage 1 availability: one journal-count read per job ------------------------
        journal_count = 0
        if wos_filter != "off":
            async with session_factory() as db:
                journal_count = await get_wos_journal_count(db)
        stage1_applied = wos_filter == "on" or (wos_filter == "auto" and journal_count > 0)
        if not stage1_applied:
            logger.info(
                "Smart Search %s: Stage 1 (WoS venue filter) inactive (wos_filter=%s, "
                "journals=%d)",
                job_id, wos_filter, journal_count,
            )

        while True:
            round_number += 1

            # --- Safety limits (round boundary) ---
            if await _cancelled():
                was_cancelled = True
                stop_reason = "cancelled"
                logger.info(
                    "Smart Search %s: cancelled before round %d", job_id, round_number,
                )
                break
            if total_scanned >= settings.smart_search_max_scanned:
                stop_reason = "max_scanned"
                logger.info("Smart Search %s: hit max scanned (%d)", job_id, total_scanned)
                break
            if _budget_exhausted():
                stop_reason = "time_limit"
                logger.info(
                    "Smart Search %s: hit time limit (%.1f min)", job_id, _elapsed_minutes(),
                )
                break

            # --- Step A: Select this round's queries: replay the given list, or generate ---
            replayed_this_round = (
                queries_override is not None and round_number <= len(queries_override)
            )
            if replayed_this_round:
                queries = queries_override[round_number - 1]
            else:
                found_titles = [p["title"] for p in included_papers] if included_papers else None
                try:
                    generated = await generate_search_queries_with_provenance(
                        query, existing_papers=found_titles,
                    )
                    queries = generated.queries
                    query_generator_provenance.add(generated.provenance)
                except Exception:
                    if round_number == 1:
                        # Fallback to raw user query on first round
                        logger.warning(
                            "Smart Search %s: query generation failed on round 1, using raw query",
                            job_id,
                        )
                        queries = [query]
                    else:
                        # On later rounds, stop if query generation fails
                        stop_reason = "query_generation_failed"
                        logger.info(
                            "Smart Search %s: query generation failed on round %d, stopping",
                            job_id, round_number,
                        )
                        break

            # --- Step B: fan the round's queries out concurrently ---
            round_papers: list[dict] = []
            query_results, _stopped_early, round_failures = await gather_search_results(
                search_service, queries, stop_check=_stop_search,
                to_publication_date=publication_date_max,
            )
            retrieval_failures.extend(round_failures)

            returned_by_query: dict[str, int] = {}
            new_unique_by_query: dict[str, int] = {}
            for _q, papers in query_results:
                returned_by_query[_q] = len(papers)
                new_unique_by_query.setdefault(_q, 0)
                for paper in papers:
                    total_scanned += 1
                    paper_dict = paper.model_dump(mode="json")

                    # --- Step C: Deduplicate by DOI then normalized title ---
                    doi = paper.doi
                    if doi:
                        doi_lower = doi.lower()
                        if doi_lower in seen_dois:
                            duplicates_removed += 1
                            continue
                        seen_dois.add(doi_lower)

                    norm_title = _normalize_title(paper.title) if paper.title else ""
                    if norm_title:
                        if norm_title in seen_titles:
                            duplicates_removed += 1
                            continue
                        seen_titles.add(norm_title)

                    round_papers.append(paper_dict)
                    new_unique_by_query[_q] += 1

            # Round log: one entry per round whose
            # queries actually ran, so an export can show every query issued and how many
            # results it returned even when the round included nothing new. A query that
            # raised (already in retrieval_failures) or one skipped by a time/cancellation
            # stop before it ran both show as returned=0, new_unique=0; ``screened``,
            # ``included_new`` and ``needs_review_new`` default to zero here and are filled
            # in as Stage 2 (Step E) runs, staying zero if the round never reaches it.
            # ``included_new`` counts this round's own *provisional* includes -- a pass-1
            # INCLUDE that cleared the batch pass's own guard and the two deterministic
            # post-model devices, before the job-level second pass has had any chance to run
            # -- never a stage-confirmed count; some of a round's
            # own provisional includes may be demoted after every round has finished (see
            # Stage 2b below), and that demotion is counted only in the job-level
            # ``screener_second_pass`` provenance block (``demotions``/``unavailable``), not
            # folded into any round's own ``needs_review_new``, since the stage runs once for
            # the whole job and attributing its outcome to whichever round happened to run
            # last would be arbitrary and wrong. ``replayed`` states whether this
            # round's queries came from ``queries_override`` rather than the query generator
            # agent; ``source`` names where they came from, present only when ``replayed`` is
            # true.
            round_log: dict = {
                "round": round_number,
                "queries": [
                    {
                        "query": q,
                        "returned": returned_by_query.get(q, 0),
                        "new_unique": new_unique_by_query.get(q, 0),
                    }
                    for q in queries
                ],
                "screened": 0,
                "included_new": 0,
                "needs_review_new": 0,
                "replayed": replayed_this_round,
            }
            if replayed_this_round:
                round_log["source"] = "queries_override"
            rounds_log.append(round_log)

            # Stage the round's papers are waiting for, should the round end early.
            pending_stage = STAGE_WOS if stage1_applied else STAGE_LLM

            if was_cancelled:
                _mark_unscreened(round_papers, pending_stage, REASON_CANCELLED)
                logger.info(
                    "Smart Search %s: cancelled during round %d", job_id, round_number,
                )
                break

            if not round_papers:
                if round_failures and len(round_failures) == len(queries):
                    # Every query raised: the provider is down or over quota. Do not report
                    # this as the literature running dry.
                    stop_reason = stop_reason or "retrieval_failed"
                    logger.warning(
                        "Smart Search %s: all %d queries failed in round %d (%s), stopping",
                        job_id, len(queries), round_number, round_failures[0][1],
                    )
                else:
                    stop_reason = stop_reason or "no_new_papers"
                    logger.info(
                        "Smart Search %s: no new papers after dedup in round %d, stopping",
                        job_id, round_number,
                    )
                break

            # --- Step D: Stage 1 — WoS filter (one bulk query for the whole round) ---
            if _budget_exhausted():
                stop_reason = "time_limit"
                _mark_unscreened(round_papers, pending_stage, REASON_TIME_LIMIT)
                logger.info(
                    "Smart Search %s: time limit hit before WoS filter in round %d",
                    job_id, round_number,
                )
                break

            wos_passed: list[dict] = []
            if stage1_applied:
                async with session_factory() as db:
                    wos_lookup = await is_wos_indexed_bulk(
                        db, [p.get("journal_issn") for p in round_papers],
                    )

                for paper_dict in round_papers:
                    issn = paper_dict.get("journal_issn")
                    key = issn.strip().upper() if issn else ""
                    indexed, collection, categories = wos_lookup.get(key, (False, None, None))
                    paper_dict["wos_indexed"] = indexed
                    paper_dict["wos_collection"] = collection
                    paper_dict["wos_categories"] = categories
                    stage1_screened += 1
                    if indexed:
                        wos_passed.append(paper_dict)
                    else:
                        stage1_excluded += 1
                        excluded_records.append(
                            paper_record(
                                paper_dict,
                                stage=STAGE_WOS,
                                reason=REASON_NOT_INDEXED if key else REASON_NO_ISSN,
                            )
                        )

                if not wos_passed:
                    stop_reason = "no_wos_papers"
                    logger.info(
                        "Smart Search %s: no WoS-indexed papers in round %d, stopping",
                        job_id, round_number,
                    )
                    break
            else:
                for paper_dict in round_papers:
                    paper_dict["wos_indexed"] = None
                    paper_dict["wos_collection"] = None
                    paper_dict["wos_categories"] = None
                wos_passed = round_papers

            # --- Step E: Stage 2 — AI relevance screening ---
            new_included: list[dict] = []
            round_unscreened = 0  # papers whose screening call failed in this round
            for i in range(0, len(wos_passed), batch_size):
                if _budget_exhausted():
                    stop_reason = "time_limit"
                    _mark_unscreened(wos_passed[i:], STAGE_LLM, REASON_TIME_LIMIT)
                    logger.info(
                        "Smart Search %s: time limit hit during screening in round %d",
                        job_id, round_number,
                    )
                    break
                batch = wos_passed[i : i + batch_size]
                try:
                    screened = await screen_papers(
                        query, batch, inclusion_criteria, exclusion_criteria,
                        inclusion_stages=inclusion_criteria_stages,
                        exclusion_stages=exclusion_criteria_stages,
                    )
                except Exception as exc:
                    # Never default to INCLUDE: the batch is recorded as unscreened.
                    reason = f"{type(exc).__name__}: {exc}"
                    logger.warning(
                        "Smart Search %s: screening failed for a batch of %d papers (%s); "
                        "recorded as unscreened",
                        job_id, len(batch), reason,
                    )
                    _mark_unscreened(batch, STAGE_LLM, reason)
                    round_unscreened += len(batch)
                    continue

                # The model was called: record it even if the result turns out unusable.
                screener_provenance.add(screened.provenance)
                if screened.reask_provenance is not None:
                    # A re-ask is a second paid call --
                    # add its own tokens/model/fingerprint too, or the job summary would
                    # describe only one of the two calls this batch actually made.
                    screener_provenance.add(screened.reask_provenance)
                if len(screened.include) != len(batch) or len(screened.reasons) != len(batch):
                    reason = "ScreeningError: decision count does not match the batch size"
                    logger.warning(
                        "Smart Search %s: %s (batch of %d); recorded as unscreened",
                        job_id, reason, len(batch),
                    )
                    _mark_unscreened(batch, STAGE_LLM, reason)
                    round_unscreened += len(batch)
                    continue

                guard_reasons = screened.guard_reasons or [""] * len(batch)
                to_confirm_lists = screened.to_confirm or [[] for _ in batch]
                shown_texts = build_shown_texts(batch)
                for (
                    paper_dict, status, criterion, quote, reason, guard_reason, to_confirm,
                    shown_text,
                ) in zip(
                    batch, screened.statuses, screened.criteria_ids, screened.quotes,
                    screened.reasons, guard_reasons, to_confirm_lists, shown_texts,
                ):
                    stage2_screened += 1
                    round_log["screened"] += 1
                    paper_dict["screening_criterion"] = criterion
                    paper_dict["screening_quote"] = quote
                    paper_dict["screening_reason"] = reason
                    paper_dict["screening_to_confirm"] = list(to_confirm or [])
                    paper_dict["screening_second_pass"] = {}

                    # Two narrow
                    # deterministic devices, INCLUDE only, applied after the batch pass's own
                    # guard -- OpenAlex's own type/is_paratext are not shown to the model, so
                    # they cannot be checked inside screen_papers itself.
                    status, type_reason = apply_type_demotion(
                        status,
                        work_type=paper_dict.get("openalex_type"),
                        is_paratext=paper_dict.get("is_paratext"),
                    )
                    if type_reason:
                        guard_reason = type_reason
                    status, toc_reason = apply_table_of_contents_demotion(status, shown_text)
                    if toc_reason:
                        guard_reason = toc_reason

                    paper_dict["screening_status"] = status
                    paper_dict["screening_guard_reason"] = guard_reason

                    if status == "INCLUDE":
                        # Provisional: joins
                        # included_papers immediately below, so this round's own stopping
                        # decision and a later round's own query generation see it right
                        # away. The second pass runs once for the whole job, after every
                        # round has run, over every provisional include collected here.
                        new_included.append(paper_dict)
                        job_provisional_includes.append(paper_dict)
                        job_provisional_shown.append(shown_text)
                    elif status == "NEEDS_REVIEW":
                        # Kept as a full paper dict (not the trimmed record shape) so the
                        # results panel can render it with the same PaperCard the included
                        # list uses.
                        _route_needs_review(paper_dict, guard_reason, round_log=round_log)
                    else:  # EXCLUDE
                        stage2_excluded += 1
                        excluded_records.append(
                            paper_record(
                                paper_dict, stage=STAGE_LLM, reason=reason,
                                status=status, criterion=criterion, quote=quote,
                            )
                        )

            included_papers.extend(new_included)
            round_log["included_new"] = len(new_included)

            if stop_reason == "time_limit":
                break

            # --- Step F: Data-driven stopping ---
            if not new_included:
                if round_unscreened:
                    # Do not blame the data for a failing model; do not spin against it either.
                    stop_reason = "screening_failed"
                    logger.warning(
                        "Smart Search %s: screening failed for %d paper(s) and no paper was "
                        "included in round %d, stopping",
                        job_id, round_unscreened, round_number,
                    )
                    break

                # A round that screened cleanly but included nothing new: the loop
                # never stops on this alone before smart_search_min_rounds rounds have run,
                # and afterwards stops only once smart_search_dry_round_patience such rounds
                # have happened in a row -- one quiet round no longer ends a job that would
                # have found more with a couple more query rounds.
                consecutive_dry_rounds += 1
                if (
                    round_number >= settings.smart_search_min_rounds
                    and consecutive_dry_rounds >= settings.smart_search_dry_round_patience
                ):
                    stop_reason = "no_new_included"
                    logger.info(
                        "Smart Search %s: %d consecutive round(s) with no new included "
                        "papers (patience=%d, min_rounds=%d), stopping",
                        job_id, consecutive_dry_rounds,
                        settings.smart_search_dry_round_patience,
                        settings.smart_search_min_rounds,
                    )
                    break
                logger.info(
                    "Smart Search %s: zero new included papers in round %d "
                    "(%d/%d dry round(s), min_rounds=%d), continuing",
                    job_id, round_number, consecutive_dry_rounds,
                    settings.smart_search_dry_round_patience, settings.smart_search_min_rounds,
                )
            else:
                consecutive_dry_rounds = 0

            # --- Update job progress ---
            progress = min(
                0.95,
                total_scanned / settings.smart_search_max_scanned,
            )
            async with session_factory() as db:
                await update_job_status(
                    db, job_id,
                    status=JobStatus.running,
                    progress=progress,
                    progress_message=(
                        f"Round {round_number}: {len(included_papers)} papers included, "
                        f"{total_scanned} scanned, {_elapsed_minutes():.1f} min elapsed"
                    ),
                )

        # -- Stage 2b: inclusion-only second pass, once for the whole job ----------------
        # Every provisional INCLUDE
        # collected above, from every round, judged together as a single concurrent stage
        # once the round loop above has ended -- not once per round. A live search job runs
        # many small rounds (the demonstration corpus runs about 27 of them, roughly 35
        # screened records and 9 INCLUDE candidates each); paying one ~50-380s judge chunk
        # latency before every next round could even start added up to over an hour of judge
        # wall time alone, well past the search's own 30-minute budget, and would have
        # stopped the search on time_limit long before it identified the records it does
        # today. Running the stage once, after retrieval, removes the judge from the
        # search's own critical path entirely: the stage gets its own separate time budget
        # (settings.screener_second_pass_stage_budget_seconds), never
        # settings.smart_search_max_time_minutes, so the search's own stopping decisions
        # (Step F above) are made exactly as they were before the second pass existed, on
        # the provisional INCLUDE count alone, and are never shortened by how long the judge
        # takes.
        if job_provisional_includes:
            n_candidates_total = len(job_provisional_includes)
            n_chunks_total = -(-n_candidates_total // SECOND_PASS_STAGE_BATCH_SIZE)  # ceil
            if was_cancelled:
                # A job already cancelled during the round loop
                # must not wait on the judge at all -- every provisional include is queued
                # for a human instead of paying for (or waiting on) a stage the job will
                # only discard.
                logger.info(
                    "Smart Search %s: cancelled before the second-pass stage; routing %d "
                    "provisional include(s) to needs_review without calling the judge",
                    job_id, n_candidates_total,
                )
                stage_result = SecondPassStageResult(
                    decisions=[
                        SecondPassStageDecision(
                            status="NEEDS_REVIEW",
                            guard_reason=GUARD_REASON_SECOND_PASS_UNAVAILABLE,
                            second_pass=None, error=None,
                        )
                        for _ in job_provisional_includes
                    ],
                    calls=[], wall_time_s=0.0, n_skipped_budget=n_candidates_total,
                )
            else:
                # One update before the stage starts, so the
                # interface shows it running rather than sitting on the last round's own
                # message for as long as the stage takes; chunks_done below drives one more
                # per completed chunk.
                #
                # This is a progress display only, not the
                # stage's own screening result -- a transient database error here (a pool
                # exhausted, a dropped connection) must not fail the stage or the job, and a
                # search that otherwise completed must never be discarded because writing
                # "N/M judged" failed. Logged and swallowed, never re-raised.
                try:
                    async with session_factory() as db:
                        await update_job_status(
                            db, job_id, status=JobStatus.running, progress=0.97,
                            progress_message=(
                                f"Confirming {n_candidates_total} inclusion"
                                f"{'s' if n_candidates_total != 1 else ''} "
                                f"(0/{n_chunks_total} judged)"
                            ),
                        )
                except Exception as exc:
                    logger.warning(
                        "Smart Search %s: second-pass stage's own pre-stage progress "
                        "update failed (%s); continuing without it",
                        job_id, exc,
                    )
                chunks_done = 0

                async def _stage_should_abort() -> bool:
                    # Checked at the same per-chunk boundaries
                    # the deadline already is, so a cancellation noticed mid-stage stops the
                    # next chunk from ever being launched -- a chunk already in flight still
                    # finishes cleanly, bounded by its own timeout, never abandoned mid-call.
                    nonlocal was_cancelled, stop_reason
                    if was_cancelled:
                        return True
                    if await _cancelled():
                        was_cancelled = True
                        stop_reason = "cancelled"
                        return True
                    return False

                async def _stage_chunk_done() -> None:
                    # Same as the pre-stage update above --
                    # a progress display only, never allowed to fail the stage it is
                    # reporting on. run_second_pass_stage awaits this from a bare
                    # ``finally`` (no try/except of its own around it), so an exception
                    # raised here would otherwise propagate out of the whole concurrent
                    # gather and discard every chunk's own result, including chunks that
                    # already completed successfully.
                    nonlocal chunks_done
                    chunks_done += 1
                    try:
                        async with session_factory() as db:
                            await update_job_status(
                                db, job_id, status=JobStatus.running, progress=0.97,
                                progress_message=(
                                    f"Confirming {n_candidates_total} inclusion"
                                    f"{'s' if n_candidates_total != 1 else ''} "
                                    f"({chunks_done}/{n_chunks_total} judged)"
                                ),
                            )
                    except Exception as exc:
                        logger.warning(
                            "Smart Search %s: second-pass stage's own per-chunk progress "
                            "update failed (%s); continuing without it",
                            job_id, exc,
                        )

                stage_deadline = (
                    time.monotonic() + settings.screener_second_pass_stage_budget_seconds
                )
                stage_result = await run_second_pass_stage(
                    job_provisional_includes,
                    job_provisional_shown,
                    research_question=query,
                    inclusion_criteria=inclusion_criteria,
                    exclusion_criteria=exclusion_criteria,
                    model=settings.screener_second_pass_model,
                    concurrency=settings.screener_second_pass_concurrency,
                    stage_deadline=stage_deadline,
                    should_abort=_stage_should_abort,
                    on_chunk_done=_stage_chunk_done,
                )
            for call in stage_result.calls:
                second_pass_provenance.add(call.provenance, call.latency_s)
                # Attribute this chunk's own tokens/latency to
                # every candidate it covered, exactly as ``evaluation/screening/
                # run_screening.py``'s own harness already does for its per-record
                # ``second_pass_input_tokens``/``second_pass_output_tokens`` export columns
                # (one code path -- ``run_second_pass_stage`` -- backing both callers; this
                # is production's own copy of that same attribution, not a second
                # implementation of it). ``call.indices`` are positions directly into
                # ``job_provisional_includes``, the exact list this stage was given as
                # ``candidates``, so no remapping is needed here the way the evaluation
                # harness needs one for its own filtered candidate subset.
                call_info = {
                    "input_tokens": call.provenance.input_tokens,
                    "output_tokens": call.provenance.output_tokens,
                    "latency_s": round(call.latency_s, 3),
                    "model_reported": call.provenance.model_reported,
                }
                for idx in call.indices:
                    job_provisional_includes[idx]["screening_second_pass_call"] = call_info
            second_pass_provenance.add_stage_wall_time(stage_result.wall_time_s)
            if stage_result.n_skipped_budget and not was_cancelled:
                logger.warning(
                    "Smart Search %s: second-pass stage budget exhausted before %d of %d "
                    "candidate(s) could be judged; routed to needs_review",
                    job_id, stage_result.n_skipped_budget, n_candidates_total,
                )
            demoted_ids: set[int] = set()
            n_demoted = 0
            n_unavailable = 0
            # Every decision from the same failed chunk
            # carries the exact same ``error`` string object (run_second_pass_stage's own
            # _route_unavailable assigns one f-string to every index in that chunk), so
            # deduping on id() here is deduping on chunk, not on error text -- two
            # different chunks that happen to fail with byte-identical text are still
            # logged separately below, correctly, since they really are two failures.
            logged_chunk_errors: set[int] = set()
            for paper_dict, decision in zip(job_provisional_includes, stage_result.decisions):
                if decision.second_pass is not None:
                    paper_dict["screening_second_pass"] = decision.second_pass
                if decision.status != "INCLUDE":
                    paper_dict["screening_status"] = decision.status
                    paper_dict["screening_guard_reason"] = decision.guard_reason
                    demoted_ids.add(id(paper_dict))
                    if decision.guard_reason == GUARD_REASON_SECOND_PASS_UNAVAILABLE:
                        n_unavailable += 1
                        if decision.error and id(decision.error) not in logged_chunk_errors:
                            logged_chunk_errors.add(id(decision.error))
                            # The stage already logs this once
                            # per failed chunk; logging it again here, with the job id, is
                            # what makes it findable from this job's own log lines. Once
                            # per chunk, not once per record the chunk covered.
                            logger.warning(
                                "Smart Search %s: second pass could not reach a record "
                                "(%s)",
                                job_id, decision.error,
                            )
                    else:
                        n_demoted += 1
                    # No round_log -- the stage runs once for
                    # the whole job, after every round has already finished, so this
                    # demotion is never attributed to whichever round happened to run
                    # last. It is counted in second_pass_provenance.record_outcome_counts
                    # below instead, the stage's own job-level block.
                    _route_needs_review(paper_dict, decision.guard_reason, round_log=None)
            second_pass_provenance.record_outcome_counts(
                n_candidates=len(job_provisional_includes),
                n_demoted=n_demoted,
                n_unavailable=n_unavailable,
            )
            if demoted_ids:
                # A provisional INCLUDE the stage demoted is no longer included -- removed
                # here, not filtered out earlier, since every round up to this point (and
                # any stopping decision a round made) legitimately saw it as included.
                included_papers = [p for p in included_papers if id(p) not in demoted_ids]

        # -- Completion -----------------------------------------------------
        final_status = JobStatus.cancelled if was_cancelled else JobStatus.completed
        headline = "Cancelled" if was_cancelled else "Completed"
        flow = {
            "identified": total_scanned,
            "duplicates_removed": duplicates_removed,
            "stage1_screened": stage1_screened,
            "stage1_excluded": stage1_excluded,
            "stage2_screened": stage2_screened,
            "stage2_excluded": stage2_excluded,
            "needs_review": len(needs_review_records),
            "unscreened": len(unscreened_records),
            "included": len(included_papers),
            "rounds": round_number,
            "stop_reason": stop_reason,
            #: Counts of NEEDS_REVIEW records keyed by the guard
            #: reason ("no_abstract", "full_text_criterion", "cut_abstract",
            #: "unanchored_exclude", "unquoted_criterion") plus "undecidable" (a genuine model
            #: NEEDS_REVIEW, no guard involved). The guard attributes "full_text_criterion" /
            #: "unquoted_criterion" whether or not it demoted anything, so guard_reason alone
            #: decides the bucket here; no record can be silently folded into "undecidable"
            #: for naming a full-text exclusion criterion. "no_abstract" is now
            #: also a guard reason in its own right (an EXCLUDE on a record with no abstract
            #: at all, demoted unconditionally) -- it lands in the same bucket as the
            #: pre-existing guard-silent, no-abstract fallback just below, since both describe
            #: the same thing. A dict, not a flow row, so ``FLOW_KEYS``/``flowRows`` in
            #: ``provenance.ts`` are unaffected and stay at seven keys.
            "needs_review_by_reason": dict(needs_review_by_reason),
        }
        result = {
            "papers": included_papers,
            "total_included": len(included_papers),
            "total_scanned": total_scanned,
            "rounds": round_number,
            "elapsed_minutes": round(_elapsed_minutes(), 2),
            "stop_reason": stop_reason,
            "stage1_applied": stage1_applied,
            "criteria": {
                "query": query,
                "inclusion_criteria": list(inclusion_criteria or []),
                "exclusion_criteria": list(exclusion_criteria or []),
                "wos_filter": wos_filter,
                #: The one input needed to
                #: reproduce the routing -- which side of each criteria list was marked
                #: full-text -- so the exported screening record is self-describing.
                "inclusion_criteria_stages": list(inclusion_criteria_stages or []),
                "exclusion_criteria_stages": list(exclusion_criteria_stages or []),
                #: The OpenAlex to_publication_date filter applied to
                #: every search call this job made, or None when the job used the live
                #: index with no cutoff.
                "publication_date_max": (
                    publication_date_max.isoformat() if publication_date_max else None
                ),
            },
            "flow": flow,
            "excluded": excluded_records,
            "needs_review": needs_review_records,
            "unscreened": unscreened_records,
            "retrieval_failures": retrieval_failures,
            "provenance": {
                "screener": screener_provenance.summary(),
                #: The
                #: inclusion-only second pass's own calls, separate from "screener" above
                #: since it runs at a different model (settings.screener_second_pass_model)
                #: and a different prompt. Every field is empty/zero on a job that never
                #: sent any batch to the second pass (no INCLUDE survived to it).
                "screener_second_pass": second_pass_provenance.summary(),
                "query_generator": query_generator_provenance.summary(),
                #: One entry per round whose queries ran (see the ``rounds_log``
                #: comment above Step B), so the exported screening record can show what
                #: was searched and found round by round, not just the final totals.
                "rounds": rounds_log,
                #: The two settings the stop rule (Step F) applied this run, so a report
                #: reading ``rounds`` also knows the min-rounds/patience thresholds behind it.
                "settings": {
                    "min_rounds": settings.smart_search_min_rounds,
                    "dry_round_patience": settings.smart_search_dry_round_patience,
                },
            },
        }
        progress_message = (
            f"{headline}: {len(included_papers)} papers after "
            f"{round_number} round(s), {total_scanned} scanned"
        )
        if unscreened_records:
            progress_message += f", {len(unscreened_records)} unscreened"
        if stop_reason == "screening_failed":
            progress_message += " (screening failed)"
        if stop_reason == "retrieval_failed":
            failed_queries = len(retrieval_failures)
            progress_message += (
                f" (retrieval failed: {failed_queries} of {failed_queries} queries; "
                f"{retrieval_failures[0][1]})"
            )
        async with session_factory() as db:
            await update_job_status(
                db, job_id,
                status=final_status,
                progress=1.0,
                progress_message=progress_message,
                result=result,
            )

        logger.info(
            "Smart Search %s %s: %d included / %d scanned in %d round(s) "
            "(stop_reason=%s, stage1_applied=%s, excluded=%d, unscreened=%d)",
            job_id, final_status.value, len(included_papers), total_scanned,
            round_number, stop_reason, stage1_applied, len(excluded_records),
            len(unscreened_records),
        )

    except Exception as exc:
        logger.exception("Smart Search %s failed: %s", job_id, exc)
        async with session_factory() as db:
            await update_job_status(
                db, job_id,
                status=JobStatus.failed,
                progress=0.0,
                progress_message="Smart Search failed",
                error=traceback.format_exc(),
            )
