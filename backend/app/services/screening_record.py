"""Screening record — the auditable per-paper trail of a Smart Search job.

Reviewers asked for the screening decisions to be inspectable and exportable, not just
the list of survivors. Every record the pipeline identified ends up in exactly one of
four outcomes (the screener widened from binary INCLUDE/EXCLUDE to a
three-status protocol-eligibility decision, adding ``needs_review``):

``included``
    passed both stages (stage ``llm``; ``status`` is ``"INCLUDE"``, ``reason`` is the
    model's stated reason)
``needs_review``
    reached Stage 2 but the shown text could not decide (stage ``llm``; ``status`` is
    ``"NEEDS_REVIEW"``, ``criterion``/``quote`` carry any anchor the guard demoted)
``excluded``
    dropped at stage ``wos`` (venue filter) or ``llm`` (relevance screening; ``status`` is
    ``"EXCLUDE"`` with the ``criterion`` id and verbatim ``quote`` that justified it)
``unscreened``
    never received a decision (failed model call, time limit, cancellation); ``reason``
    names the cause, e.g. ``"ScreeningError: TimeoutError: ..."``

These are pure functions over the ``analysis_jobs.result`` JSON written by
:func:`app.services.smart_search.run_smart_search`. :func:`paper_record` is the single
definition of a record row's paper fields: the pipeline uses it when it writes
``excluded``/``unscreened`` entries and the export uses it again when it reads them back.
Tolerant of a v1.0.0 result (only ``papers``, no ``status``/``criterion``/``quote``): those
fields default to ``""`` so an old stored job result still projects onto the same shape.
"""

from __future__ import annotations

import csv
import io
from typing import Any

STAGE_WOS = "wos"
STAGE_LLM = "llm"

#: Byte-order mark so Excel on Windows decodes the CSV as UTF-8 (bilingual titles/reasons).
CSV_BOM = "\ufeff"
#: Leading characters a spreadsheet would evaluate as a formula (CSV injection).
_FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")

#: ``abstract`` is appended last so field order stays
#: stable for existing readers; it is the paper's abstract as stored, not the possibly-cut
#: text the screener's own prompt actually showed the model,
#: empty string when there is none.
RECORD_PAPER_FIELDS = (
    "title", "doi", "year", "journal", "journal_issn", "openalex_id", "abstract",
)
#: ``to_confirm`` sits after ``needs_review_reason``.
#: ``abstract`` is appended last. ``second_pass`` is added after
#: ``to_confirm``;
#: ``second_pass_input_tokens``/``second_pass_output_tokens``/``second_pass_latency_s``
#: come right after it (nineteen columns now); the demo's own frozen ``demo/expected/**``
#: baseline and its ``startswith("outcome,stage")`` assertion are unaffected -- both column
#: count and order after ``outcome``/``stage`` are free to grow, and the demo's own
#: duplicated fifteen-column tuple in ``demo/tests/test_expected_baseline.py`` reads only
#: the already-frozen golden file, never this module.
CSV_HEADER = (
    "outcome", "stage", "status", "criterion", "quote", "needs_review_reason", "to_confirm",
    "second_pass", "second_pass_input_tokens", "second_pass_output_tokens",
    "second_pass_latency_s", "reason", *RECORD_PAPER_FIELDS,
)


#: The order the v2 second pass's three questions are joined into one CSV/JSON
#: cell (``_format_second_pass``), matching the order ``SECOND_PASS_PROMPT_V2`` asks them in.
_SECOND_PASS_KEYS = ("population", "outcome", "study_type")


def _format_second_pass(second_pass: dict[str, str] | None) -> str:
    """One compact ``key=value`` cell for the v2 second pass's per-record
    answer (``screening_second_pass`` on the paper dict: ``population``/``outcome`` each
    ``"established"``/``"not_established"``, ``study_type`` ``"study"``/``"synthesis"``/
    ``"not_established"``), so a user reading the export sees exactly what the judge decided
    without opening the raw job JSON. ``""`` when the second pass never ran for this record
    (every outcome but ``"included"``/``"needs_review"``, and any record demoted before
    reaching the second pass)."""
    if not second_pass:
        return ""
    return ";".join(
        f"{key}={second_pass[key]}" for key in _SECOND_PASS_KEYS if second_pass.get(key)
    )


def paper_record(
    paper: dict[str, Any],
    *,
    stage: str,
    reason: str,
    status: str = "",
    criterion: str = "",
    quote: str = "",
    needs_review_reason: str = "",
    to_confirm: list[str] | tuple[str, ...] = (),
    second_pass: dict[str, str] | None = None,
    second_pass_call: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project a pipeline paper dict (or an existing record entry) onto the record shape.

    Accepts both the ``PaperData`` dump the pipeline works with (``journal_name``,
    ``external_id``) and an already-built record entry (``journal``, ``openalex_id``).
    ``status``, ``criterion`` and ``quote`` are the screener's protocol-eligibility
    decision: ``""`` for a record the LLM screener never reached
    (Stage 1 exclusions, unscreened records), the screener's stated criterion id and
    verbatim quote for an EXCLUDE, and ``""``/``""`` for an INCLUDE or a NEEDS_REVIEW with
    no anchor. ``needs_review_reason`` names why a NEEDS_REVIEW record
    was flagged -- a guard reason (``"full_text_criterion"``, ``"cut_abstract"``,
    ``"unanchored_exclude"``, ``"unquoted_criterion"``,
    ``"full_text_to_confirm"``, ``"nonarticle_type"``, ``"table_of_contents"``,
    ``"not_established"`` or ``"second_pass_unavailable"``) or ``"no_abstract"``/
    ``"undecidable"`` -- and is ``""`` for every other outcome. ``to_confirm`` is the
    full-text inclusion criterion ids still to be confirmed at full text, comma-joined into
    a single cell: non-empty for
    an ``"included"`` record and, since such a record is now routed to NEEDS_REVIEW rather
    than shipped as INCLUDE (guard reason ``"full_text_to_confirm"``), for a
    ``"needs_review"`` record that still carries them too, so a reviewer sees exactly which
    criteria the record was queued to confirm. ``second_pass`` is the v2
    second pass's own per-record answer, compacted by :func:`_format_second_pass`; ``""``
    when the second pass never ran for this record. Both are passed explicitly by the
    caller rather than read off ``paper`` internally, the same convention already used for
    ``reason``. ``abstract`` is read straight off
    ``paper`` -- the paper's abstract as stored, not the possibly-cut text the screener's own
    prompt actually showed the model (``relevance_screener_agent.py`` caps the shown text at
    ``ABSTRACT_CHAR_LIMIT`` characters, marking a cut with ``ABSTRACT_CUT_MARKER``; the guard
    reason ``"cut_abstract"`` on a NEEDS_REVIEW record is the signal that a cut happened)
    -- so it round-trips through both the raw pipeline paper dict
    and an already-built record entry; ``""`` when absent. ``second_pass_call`` is the
    second pass's own chunk-level call this record's own answer
    came from (``paper.get("screening_second_pass_call")`` on the pipeline's own paper dict:
    ``input_tokens``/``output_tokens``/``latency_s``/``model_reported``, the
    ``SECOND_PASS_STAGE_BATCH_SIZE``-sized batching applied); every record a chunk covered
    shares the identical call info (the chunk's own cost cannot be split further per record),
    the same accepted simplification ``evaluation/screening/run_screening.py``'s own harness
    already applies to its equivalent export column. ``""`` for the three cells when the
    second pass never ran for this record.
    """
    openalex_id = paper.get("openalex_id")
    if openalex_id is None and paper.get("source_api") == "openalex":
        openalex_id = paper.get("external_id")
    call = second_pass_call or {}

    def _call_cell(key: str) -> Any:
        # A real, meaningful ``0`` (exactly the value made visible when
        # the provider omits usage) must survive this cell -- never collapsed to "" the way
        # a bare ``or ""`` would.
        value = call.get(key)
        return "" if value is None else value

    return {
        "title": paper.get("title"),
        "doi": paper.get("doi"),
        "year": paper.get("year"),
        "journal": paper.get("journal_name") or paper.get("journal"),
        "journal_issn": paper.get("journal_issn"),
        "openalex_id": openalex_id,
        "abstract": paper.get("abstract") or "",
        "stage": stage,
        "status": status,
        "criterion": criterion,
        "quote": quote,
        "needs_review_reason": needs_review_reason,
        "to_confirm": ",".join(to_confirm),
        "second_pass": _format_second_pass(second_pass),
        "second_pass_input_tokens": _call_cell("input_tokens"),
        "second_pass_output_tokens": _call_cell("output_tokens"),
        "second_pass_latency_s": _call_cell("latency_s"),
        "reason": reason,
    }


def _row(
    outcome: str,
    paper: dict[str, Any],
    *,
    stage: str,
    reason: str,
    status: str = "",
    criterion: str = "",
    quote: str = "",
    needs_review_reason: str = "",
    to_confirm: list[str] | tuple[str, ...] = (),
    second_pass: dict[str, str] | None = None,
    second_pass_call: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "outcome": outcome,
        **paper_record(
            paper, stage=stage, reason=reason, status=status, criterion=criterion, quote=quote,
            needs_review_reason=needs_review_reason, to_confirm=to_confirm,
            second_pass=second_pass, second_pass_call=second_pass_call,
        ),
    }


def _needs_review_reason(paper: dict[str, Any]) -> str:
    """Bucket a NEEDS_REVIEW paper dict's reason: the guard's own attribution (a demotion,
    or a compliant NEEDS_REVIEW naming
    any criterion -- "full_text_criterion" for a full-text exclusion id backed by a verbatim
    quote, "unquoted_criterion" for that same id or any other non-empty criterion id with an
    empty or non-verbatim quote -- either way non-empty), else "no_abstract" for a record
    with no abstract text, else "undecidable" for every other genuine model NEEDS_REVIEW.
    Mirrors the same bucketing ``app.services.smart_search.run_smart_search`` applies to
    ``flow.needs_review_by_reason``."""
    guard_reason = paper.get("screening_guard_reason")
    if guard_reason:
        return guard_reason
    return "no_abstract" if not (paper.get("abstract") or "").strip() else "undecidable"


def build_screening_record(job_result: dict[str, Any]) -> dict[str, Any]:
    """Return ``{criteria, flow, provenance, retrieval_failures, records}`` for a job result.

    ``retrieval_failures`` lists ``[query, "<ExceptionClass>: <message>"]`` for every
    search query that raised (e.g. an OpenAlex quota block), so a record whose ``flow``
    says ``stop_reason: retrieval_failed`` also carries the provider's reason.

    Tolerates results written by v1.0.0 (only ``papers``): the summary blocks are empty
    and included papers get an empty reason.
    """
    result = job_result or {}
    records: list[dict[str, Any]] = []
    for paper in result.get("papers") or []:
        records.append(
            _row(
                "included",
                paper,
                stage=STAGE_LLM,
                reason=paper.get("screening_reason") or "",
                status=paper.get("screening_status") or "",
                criterion=paper.get("screening_criterion") or "",
                quote=paper.get("screening_quote") or "",
                to_confirm=paper.get("screening_to_confirm") or [],
                second_pass=paper.get("screening_second_pass") or None,
                second_pass_call=paper.get("screening_second_pass_call") or None,
            )
        )
    # ``needs_review`` entries are full paper dicts, like ``papers``: the frontend
    # renders them with the same ``PaperCard`` the included list uses, not
    # the trimmed shape ``excluded``/``unscreened`` already carry.
    for paper in result.get("needs_review") or []:
        records.append(
            _row(
                "needs_review",
                paper,
                stage=STAGE_LLM,
                reason=paper.get("screening_reason") or "",
                status=paper.get("screening_status") or "",
                criterion=paper.get("screening_criterion") or "",
                quote=paper.get("screening_quote") or "",
                needs_review_reason=_needs_review_reason(paper),
                # A record demoted for an unconfirmed full-text inclusion criterion
                # (guard reason "full_text_to_confirm") still carries the ids it was queued
                # to confirm, so a reviewer sees exactly why without opening the raw job
                # JSON; "" for every other needs_review record, same as before.
                to_confirm=paper.get("screening_to_confirm") or [],
                second_pass=paper.get("screening_second_pass") or None,
                second_pass_call=paper.get("screening_second_pass_call") or None,
            )
        )
    for outcome, key in (("excluded", "excluded"), ("unscreened", "unscreened")):
        for entry in result.get(key) or []:
            records.append(
                _row(
                    outcome,
                    entry,
                    stage=entry.get("stage") or "",
                    reason=entry.get("reason") or "",
                    status=entry.get("status") or "",
                    criterion=entry.get("criterion") or "",
                    quote=entry.get("quote") or "",
                )
            )
    return {
        "criteria": result.get("criteria") or {},
        "flow": result.get("flow") or {},
        "provenance": result.get("provenance") or {},
        "retrieval_failures": [list(f) for f in result.get("retrieval_failures") or []],
        "records": records,
    }


def _csv_cell(value: Any) -> Any:
    """Neutralise spreadsheet formula triggers in untrusted text (titles, model reasons)."""
    if isinstance(value, str) and value.startswith(_FORMULA_TRIGGERS):
        return "'" + value
    return value


def screening_record_csv(job_result: dict[str, Any]) -> str:
    """One CSV row per record, header exactly ``CSV_HEADER``; ``None`` becomes an empty cell.

    The text starts with a UTF-8 BOM and string cells that begin with ``= + - @``, tab or
    CR are prefixed with a single quote so they open as text, not formulas, in Excel and
    LibreOffice. The JSON export keeps the raw values.
    """
    buffer = io.StringIO()
    buffer.write(CSV_BOM)
    writer = csv.DictWriter(
        buffer, fieldnames=list(CSV_HEADER), extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    for row in build_screening_record(job_result)["records"]:
        writer.writerow({key: _csv_cell(value) for key, value in row.items()})
    return buffer.getvalue()
