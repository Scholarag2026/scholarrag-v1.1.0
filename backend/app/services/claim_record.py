"""Claim-verification record — the auditable per-claim trace of a claim-verify job.

Reviewers asked for the claim-to-source trail to be inspectable and exportable, just like
the screening record. This is a pure function over the flat
``analysis_jobs.result`` JSON written by :func:`app.services.fulltext.verify_and_heal_claims`
/ :func:`app.services.fulltext.verify_user_edits`: one record per claim, carrying the
citation text it was matched against, the source paper's identity, how its full text was
acquired (if the fulltext pipeline recorded a source), and exactly what the model reported.
Also carries ``model_status`` and ``machine_reasons``, so a reader can tell when a
code guard, not the model, produced the final ``status``.

``diagnostics`` (guard flags that never
change ``status``), ``unstated_details`` (peripheral assertions a ``needs_nuance`` claim's
precedence rule marked absent) and ``evidence_quotes`` (every verbatim span the model
composed) are siblings of the fields above, plus the full, ten-field ``assertions`` list.
``assertions`` is variable length and reachable only in the JSON export: ``claim_record_csv``
never flattens it into a cell, and instead appends two summary columns, ``n_assertions`` and
``central_assertion``, computed from it.

``claim_sentence`` sits
next to ``claim``: the full sentence the claim was cut from, equal to ``claim`` when no
citation-link proposition narrowed the claim to a shorter span of that sentence.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping
from typing import Any

#: Byte-order mark so Excel on Windows decodes the CSV as UTF-8 (bilingual titles/claims).
CSV_BOM = "﻿"
#: Leading characters a spreadsheet would evaluate as a formula (CSV injection).
_FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")

CSV_HEADER = (
    "status",
    "model_status",
    "machine_reasons",
    "claim",
    # The full sentence
    # the claim was cut from, next to `claim` itself. Equal to `claim` when no
    # citation-link proposition narrowed the claim to a shorter span.
    "claim_sentence",
    "citation",
    "doi",
    "title",
    "acquisition_route",
    "evidence_quote",
    "evidence_location",
    "model_reported",
    # Five appended columns. diagnostics,
    # unstated_details and evidence_quotes are each "; ".join(...), the rule already used
    # for machine_reasons above. n_assertions and central_assertion summarise the
    # variable-length assertions list, which is not flattened into any CSV cell (it stays
    # JSON-only, see build_claim_record / _record).
    "diagnostics",
    "unstated_details",
    "evidence_quotes",
    "n_assertions",
    "central_assertion",
    # How many model calls
    # this claim actually made (1, or 2 when the second pass fired) and the first pass's own
    # guard reasons, so a reader can see a first-pass demotion beside the final `status`
    # without opening the JSON export's full `passes` list.
    "pass_count",
    "first_pass_machine_reasons",
)

#: The ten fields of ``ClaimAssertion`` (backend/app/schemas/fulltext.py), in schema order.
#: ``central``, ``entity``, ``measure`` and ``quotes`` default when absent, so
#: an older assertion dict carries only the first six keys.
_ASSERTION_DEFAULTS: dict[str, Any] = {
    "text": None,
    "kind": None,
    "verdict": None,
    "claim_value": None,
    "source_value": None,
    "quote": None,
    "central": False,
    "entity": None,
    "measure": None,
    "quotes": [],
}


def _assertion_record(assertion: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise one assertion mapping to all ten ``ClaimAssertion`` fields.

    Fills in the schema's own defaults for any field an older-shape assertion (recorded
    that predates ``central``/``entity``/``measure``/``quotes``) does not carry, so
    every exported assertion has the same shape regardless of when it was written.
    """
    return {field: assertion.get(field, default) for field, default in _ASSERTION_DEFAULTS.items()}


def _record(index: int, verification: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": index,
        "claim": verification.get("claim_text"),
        "claim_sentence": verification.get("claim_sentence"),
        "citation": verification.get("citation"),
        "paper_id": verification.get("paper_id"),
        "doi": verification.get("paper_doi"),
        "title": verification.get("paper_title"),
        "acquisition_route": verification.get("acquisition_route"),
        "status": verification.get("status"),
        # The status a code guard would have reported before any guard capped it stricter,
        # and the machine-readable slugs of every guard that fired: without these, this
        # export -- the whole point of which is an
        # inspectable claim-to-source trail -- cannot show that a code guard, rather than
        # the model, produced a stricter verdict.
        "model_status": verification.get("model_status"),
        "machine_reasons": verification.get("machine_reasons") or [],
        # Diagnostics never change status and are always a sibling of
        # machine_reasons, never a member of it. unstated_details are the peripheral
        # assertions a needs_nuance claim's precedence rule marked absent, in prose.
        # evidence_quotes is every verbatim span the model composed; evidence_quote below
        # keeps its own name, type and position (the service sets it to the first span).
        # assertions carries all ten ClaimAssertion fields per entry; it is variable
        # length and reachable only here, in the JSON export (see CSV_HEADER above).
        "diagnostics": verification.get("diagnostics") or [],
        "unstated_details": verification.get("unstated_details") or [],
        "evidence_quotes": verification.get("evidence_quotes") or [],
        # One record per quote
        # segment `relocate_evidence_quotes` repointed onto the source before the guards
        # ran, ``{"type", "original", "replacement", "edit_distance", "chunk_index"}`` each
        # -- the diagnostics cell already carries the "quote_relocated" slug, but only this
        # field lets a human reviewer see what the model actually wrote versus what was
        # stored. JSON-only, like assertions above; the CSV already carries the slug.
        "quote_relocations": verification.get("quote_relocations") or [],
        "assertions": [_assertion_record(a) for a in verification.get("assertions") or []],
        "evidence_quote": verification.get("evidence_quote"),
        "evidence_location": verification.get("evidence_location"),
        "explanation": verification.get("explanation"),
        "model_reported": verification.get("model_reported"),
        # The full,
        # per-call record (``{"model_status", "quotes", "machine_reasons", "diagnostics",
        # "tokens", "fingerprint"}`` each), JSON-only like ``assertions``/``quote_
        # relocations`` above; ``pass_count`` and ``first_pass_machine_reasons`` (both also
        # in the CSV) summarise it for a reader who only wants to see that a first-pass
        # demotion happened, beside the final ``status``, without opening the JSON export.
        "passes": verification.get("passes") or [],
        "pass_count": len(verification.get("passes") or []),
        "first_pass_machine_reasons": (
            (verification.get("passes") or [{}])[0].get("machine_reasons") or []
        ),
    }


def build_claim_record(job_result: dict[str, Any] | None) -> dict[str, Any]:
    """Return ``{provenance, full_text_coverage, counts, records}`` for a job result.

    ``counts`` mirrors the report's own tally so the export and the on-screen summary bar
    can never disagree: ``total`` is the number of records, the rest count ``status``
    values (``verified``, ``needs_nuance``, ``unsupported``, ``no_full_text``, ``error``).
    """
    result = job_result or {}
    verifications = result.get("verifications") or []
    records = [_record(i, v) for i, v in enumerate(verifications, start=1)]
    counts = {
        "total": len(records),
        "verified": sum(1 for r in records if r["status"] == "verified"),
        "needs_nuance": sum(1 for r in records if r["status"] == "needs_nuance"),
        "unsupported": sum(1 for r in records if r["status"] == "unsupported"),
        "no_full_text": sum(1 for r in records if r["status"] == "no_full_text"),
        "error": sum(1 for r in records if r["status"] == "error"),
    }
    return {
        "provenance": result.get("provenance") or {},
        "full_text_coverage": result.get("full_text_coverage") or 0.0,
        "counts": counts,
        "records": records,
    }


def _csv_cell(value: Any) -> Any:
    """Neutralise spreadsheet formula triggers in untrusted text (titles, model text)."""
    if isinstance(value, str) and value.startswith(_FORMULA_TRIGGERS):
        return "'" + value
    return value


def claim_record_csv(job_result: dict[str, Any] | None) -> str:
    """One CSV row per claim record, header exactly ``CSV_HEADER``; ``None`` -> empty cell.

    The text starts with a UTF-8 BOM and string cells that begin with ``= + - @``, tab or
    CR are prefixed with a single quote so they open as text, not formulas, in Excel and
    LibreOffice (mirrors ``screening_record_csv``). The JSON export keeps raw values.

    ``assertions`` is not a CSV cell (it is variable length and JSON-only); instead this
    derives two summary columns from it, ``n_assertions`` and ``central_assertion`` (the
    ``text`` of the single assertion marked ``central``, or an empty string when zero or
    several assertions are marked ``central`` -- the same "several" case the
    ``assertion_status_inconsistent`` guard records as the ``centrality_unmarked``
    diagnostic).
    """
    buffer = io.StringIO()
    buffer.write(CSV_BOM)
    writer = csv.DictWriter(
        buffer, fieldnames=list(CSV_HEADER), extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    for row in build_claim_record(job_result)["records"]:
        cells = dict(row)
        cells["machine_reasons"] = "; ".join(cells.get("machine_reasons") or [])
        cells["diagnostics"] = "; ".join(cells.get("diagnostics") or [])
        cells["unstated_details"] = "; ".join(cells.get("unstated_details") or [])
        cells["evidence_quotes"] = "; ".join(cells.get("evidence_quotes") or [])
        cells["first_pass_machine_reasons"] = "; ".join(
            cells.get("first_pass_machine_reasons") or []
        )
        assertions = cells.get("assertions") or []
        central_matches = [a for a in assertions if a.get("central")]
        cells["n_assertions"] = len(assertions)
        cells["central_assertion"] = (
            central_matches[0].get("text") or "" if len(central_matches) == 1 else ""
        )
        writer.writerow({key: _csv_cell(value) for key, value in cells.items()})
    return buffer.getvalue()
