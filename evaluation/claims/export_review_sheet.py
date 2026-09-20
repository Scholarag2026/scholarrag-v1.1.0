"""Build the Excel human-review workbook for a frozen claim-verification run.

Interpreter: the evaluation venv's python (``openpyxl``; see evaluation/requirements.txt) or
plain system python -- no backend import, no network, no LLM call. Importers:
``tests/test_claims_export_review_sheet*.py``.

The human sees every constructed and every real item (no sampling on the Review sheet -- that
overrides an earlier suggestion of sampling HSS agreements), and the workbook
carries five sheets: ``Guide``, ``Review``, ``Model_labels``, ``Sources``, ``Provenance``, plus
``Screening`` when ``--screening-record`` is given and ``Delivered`` when
``--delivered-evidence`` is given.

Review sheet
------------
One row per item across every ``--items`` file, columns in this exact order (see
:func:`review_header`): ``item_id`` (opaque, e.g. ``R-001`` -- the mapping back to the real
``item_id``, its construction category and its expected label lives only in ``--key-out``, so
the reviewer's judgement is not primed by either), ``set`` (``constructed`` or ``real`` --
unlike ``item_id`` this is shown openly, since the acceptance report's blinding concern was the
construction *rule*/expected label, not whether an item was constructed at all), ``claim``,
``cited_title``, ``cited_doi``, ``evidence_quote`` (the run's first evidence span, or, for an
item with no ``chunk_doi`` at all -- the ``no_full_text`` construction rule -- the literal
marker :data:`NO_FULL_TEXT_EVIDENCE_MARKER`, so the cell itself says the tool was denied the
text even where the Sources sheet happens to carry the same paper for a different item that
does have a ``chunk_doi``),
``quote_relocated_original`` (the model's own, pre-relocation quote text, next to the
stored, already-relocated ``evidence_quote``, whenever ``app.services.fulltext.
relocate_evidence_quotes`` repointed at least one quote onto the source's own text; blank
otherwise),
``evidence_quotes_all`` (every span, joined with ``"; "``), ``verifier_status`` (the run's
post-guard ``predicted_status``), ``model_status`` (pre-guard), ``first_pass_demotion`` (the
first pass's own guard reasons when the shared verification policy's bounded second pass
fired -- ``row["passes"]`` has two entries -- blank otherwise; Part B), ``machine_reasons``,
``diagnostics`` (guard notes that never change a status -- said so on the Guide sheet),
``unstated_details``, ``central_assertion`` (the one assertion marked ``central``; empty when
zero or several are), ``run_B_status`` (filled only where a second run's status differs), then
the empty human columns ``human_verdict``, ``human_agrees_with_verifier``, ``disagreement_axis``,
``human_quote``, ``human_justification``, ``countersigned_by``, ``countersigned_at``.

Model_labels sheet
-------------------
Keyed by the same opaque ``item_id``, so a reviewer who opens it before deciding gains nothing:
the adjudicated model annotation label, its rationale, and the annotator agreement (``unanimous``
or ``adjudicated``), read from one or more ``--annotation`` csv files (the
``hss_annotation_adjudicated_v3.csv`` / ``real_annotation_adjudicated_v3.csv`` shape: ``item_id,
final_label, source, rationale, ...``). Present only when at least one ``--annotation`` file is
given.

Sources sheet
-------------
Keyed by the citing item's own opaque ``item_id`` (the same id the Review sheet's own ``item_id``
column carries), not by DOI (keying by DOI put a
``no_full_text`` item's full text one lookup away, findable under a *different* item's own row
even though the row itself carried no ``chunk_doi``). One row per item per ``--sources-max-chars``
part of the paper the verifier actually showed that item; an item with no ``chunk_doi`` (the
``no_full_text`` rule) contributes nothing at all, under any key. The same underlying paper's text
is duplicated once per item that cites it -- a Review row's own ``item_id`` on this sheet holds
exactly that row's own parts and nothing else.

Provenance sheet
----------------
One row per fact, carrying the full nine-fact table on the sheet itself (a README is not the
sheet): the verifier prompt sha (read from the
primary run's own ``*.meta.json`` files, not asserted by the caller), the guard commit
(``--guard-commit``, else the last commit touching ``backend/app/services/fulltext.py`` if this
is a git checkout, else ``"unknown"``; suffixed with a ``(dirty: ...)`` marker when that file has
uncommitted changes in the checkout the export ran from, so the recorded commit is never
silently stale), the results directory, the run shown (the ``*_run<run>.jsonl`` file names
actually read for the primary run), the run commit (the last commit that touched the results
directory, auto-detected the same way the guard commit is, with its own ``(dirty: ...)``
marker), the model (the model, temperature and session count read from the primary run's own
meta files), the labels (the ``--annotation`` file names and their own sha256, computed from
the files, not asserted), the labels provenance and the blindness facts (fixed narrative facts,
not derivable from any file the exporter reads), the export time, the
demo run id (``--demo-run-id``, else the screening record's file stem, else ``"n/a"``), the
screening prompt_version actually found in ``--screening-record`` (``"n/a"`` when no screening
record was given); when a Screening sheet was built, a ``retrieval_cutoff`` row carrying the
record's own ``criteria.publication_date_max`` (present only when the record carries that
field), and one ``screening_stratum_<name>`` row per stratum
giving ``"<drawn> of <available>"`` ("reported as counts so the human can see
what was and was not sampled").

Screening sheet
----------------
Sampled from a screening-record json (``backend/app/services/screening_record.py``'s
``build_screening_record`` shape) given with ``--screening-record``. Refuses
(:class:`ExportError`) unless ``--screening-prompt-version-expect`` matches the record's own
``provenance.screener.prompt_version`` exactly, so a screening record built under a different
prompt version than the caller names is never silently sampled into the reviewer's workbook
as though it reflected the shipped screener. It also refuses
(:func:`check_screening_abstracts`) when any
sampled row that carries an actual LLM decision (:data:`EVIDENCE_STRATA`) has no
``abstract_shown`` text and is not itself exempt (:func:`is_no_abstract_exempt`) -- the record
shape the exporter reads has to carry the abstract the model actually saw, or the row's own
blank ``abstract`` field has to be a screener decision made from the title alone, for this
refusal to ever clear; see that function's docstring. Exempted from that refusal: any of the
four evidence strata whose own ``abstract`` field is blank. Originally this was restricted to a
``needs_review`` row the screener's own guard routed there
for having no abstract at all, since only a routing decision could ever lack one. A demo run
(see ``demo/README.md``) shows the screener can and does return an
``included`` decision from the title alone with no abstract at all (7 of 143 included records
on that run), so the exemption now covers all four evidence strata, not only ``needs_review``.
Such a row's ``abstract_shown`` cell carries the literal placeholder ``"(no abstract)"`` instead
of being left blank. A non-exempt row's ``abstract_shown`` is rendered through
:func:`render_abstract_for_screening`, the screener's own v2 head/marker/tail cut at
``SCREENER_ABSTRACT_CHAR_LIMIT`` (10,000 characters) -- not the record's whole stored abstract
-- so the column shows the human exactly what the model saw rather than text the screener
itself never had.

Sampling design (:func:`build_screening_sample`), drawn only from ``stage: llm`` records (a
``stage: wos`` exclusion, the venue filter, is evaluated separately and carries no LLM decision
for a human to agree with): ``included`` -- a random sample of up to
``--screening-included-n`` (default 40); ``needs_review`` -- up to
``--screening-needs-review-n`` (default 40), allocated per ``needs_review_reason`` bucket by
:func:`allocate_stratified_with_floor` -- see that function's docstring for the exact rule, and
its own ``--help`` text for a summary -- with a floor of ``--screening-floor`` (default 5) applied
to any bucket with at least ``--screening-min-bucket`` (default 5) records (a smaller bucket is
taken whole instead); ``excluded`` on a numbered criterion -- a plain random sample of up to
``--screening-excluded-n`` (default 40), drawn uniformly (unlike ``needs_review``'s
proportional-with-floor stratification: stratifying this draw let a criterion holding most of
the exclusions contribute a small minority of the sample, with no per-criterion weight recorded
anywhere to reweight by afterwards); ``excluded`` on the reserved off-topic id
(:data:`TOPIC_CRITERION_ID`)
-- a separate random sample of up to ``--screening-off-topic-n`` (default 20). Plus four
census strata, appended whole and marked with their own stratum, never counted against the
samples above: an excluded record whose
criterion the protocol's own ``criteria.inclusion_criteria_stages``/
``exclusion_criteria_stages`` marks ``"full_text"`` (:func:`full_text_criterion_ids`), an
excluded record with an empty ``quote`` or ``criterion``, an excluded record with no abstract
at all, and every ``unscreened`` record -- each expected to be zero; a non-empty one is a bug
report, not a result. The final row order is reshuffled with
``--screening-seed`` so the sheet is not sorted by outcome (a reviewer who sees forty
consecutive excludes anchors on excluding).

Columns, in the export's own order (:func:`screening_header`): ``record_id`` (opaque, shuffled
order), then the export's own eight -- ``outcome``, ``stage``, ``status``, ``criterion``,
``quote``, ``needs_review_reason``, ``to_confirm``, ``reason`` (the model's own sentence) --
then six identifying columns -- ``title``, ``doi``, ``year``, ``journal``, ``journal_issn``,
``openalex_id`` -- then ``abstract_shown`` (whole, up to ``--max-cell-chars``, with a
truncation marker -- the paper's stored abstract, or the literal placeholder ``"(no abstract)"``
for an evidence row exempted from :func:`check_screening_abstracts` by
:func:`is_no_abstract_exempt`), ``run_B_status`` (filled only on a
difference, present only when ``--screening-record-b`` is given), the human columns
``human_status``, ``human_agrees``, ``quote_is_verbatim``, ``criterion_is_right``,
``human_note``, and finally a hidden ``stratum`` column (which of the strata above the row was
drawn from) for the analysis only. The research question and the numbered criteria are shown
once, on the Guide sheet, not repeated on every row.

Delivered sheet (multi-section)
--------------------------------
Judges the final product a user actually receives, rather than the tool's own internal
verification decision (the Review sheet's own subject) -- present only when
``--delivered-evidence`` names a ``demo/output/<run>/delivered_evidence.json`` file (schema:
``run_id``, ``draft_id``, ``section_title``, ``rows``; see :func:`load_delivered_evidence`). A
single demo run can write more than one generated section into one delivered-evidence file; a
row that names its own ``section_title`` and/or ``draft_id`` is shown under that value, and a
row that does not falls back to the file's own top-level ``section_title``/``draft_id``
(:func:`build_delivered_row`) -- so a one-section file (every row silent on both fields) reads
exactly as it always did.
One row per delivered sentence, in ascending order of the record's own ``row`` field (the
1-based order the sentence appears in the final report): ``row``, ``section_title`` (which
generated section this sentence belongs to), ``draft_id`` (which saved draft this row came
from), ``sentence`` (the verifier's own recorded sentence for this claim; usually the delivered
sentence verbatim from the saved draft, except on a row where finalize rewrote the sentence in
place after dropping a co-cited ``no_full_text`` citation -- ``sentence_in_draft`` records
whether this exact sentence was found verbatim in the saved draft), ``citation_text``,
``paper_title``, ``paper_doi``, ``evidence_quotes`` (the stored quotes, joined the same way the
Review sheet's own ``evidence_quotes_all`` is), ``source_passage`` (one verbatim excerpt per
evidence quote, each from the chunk that quote was actually located in -- a row's quotes are
not always in the same chunk -- joined with a separator; empty when no quote was located at
all), ``source_located`` (the chunk ``evidence_location`` named was fetched and parsed at all --
not that any evidence passage was found in it), ``passage_located`` (every one of
``evidence_quotes`` was actually located and ``source_passage`` shows a
full reading; false, not ``source_located``, is what marks a row whose passage cell is empty or
covers only some of the quotes), ``unlocated_quotes`` (which quotes, if any, a false
``passage_located`` is reporting), ``location_section_mismatch`` (the fetched chunk's own
section disagreed with the section ``evidence_location`` recorded -- one cheap sign the passage
shown may not be from the chunk the claim was verified against), then the human columns
``human_sentence_correct`` (yes/no), ``human_quote_supports`` (yes/no) and ``human_note``.
Refuses (:class:`ExportError`, named reason) when the file does not exist, is not valid json, or
carries no ``rows`` list. Carries no construction category or expected label -- every row is a
real, delivered sentence; there is nothing to blind. Adds ``delivered_rows``,
``delivered_run_id`` and ``delivered_draft_id`` to the Provenance sheet
(:func:`append_delivered_provenance_rows`, still the file's own top-level values) and one
paragraph to the Guide sheet (:data:`DELIVERED_GUIDE_NOTE_TEMPLATE`) naming how many sections
and how many rows the sheet carries. ``score_review_sheet.py`` reports the headline over every
row plus a table broken down by ``section_title`` (:func:`score_review_sheet.score_delivered`'s
own ``by_section``).

Author page templates
----------------------
The covering instructions sent to the reviewer alongside the workbook (this module's own
work-package reports carry the current text) are assembled from
:func:`render_review_singleton_note`, :func:`render_delivered_author_paragraph` and
:func:`render_screening_author_paragraph` rather than freehand prose, and
:func:`assert_author_page_safe` is run over the assembled text. This exists because a freehand
covering note once named one
row's real item id and then told the author that row was "not identifiable from the opaque
sheet" -- false, and the id itself disclosed that row's construction category and expected
label -- plus two paragraphs that misdescribed what changed since the previous copy. None of
the three template functions below takes a real item id as a parameter, so there is no
argument through which one could reappear.

Output
------
``--out``: the ``.xlsx`` workbook. ``--key-out``: a csv with ``opaque_id, real_item_id, set,
construction_category, expected_label, workbook_id`` -- the only place the mapping from a
Review-sheet row back to its real item, category and expected label exists;
``score_review_sheet.py`` joins this file back onto the filled workbook by ``opaque_id`` and
checks the ``workbook_id`` stamps match before scoring anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import openpyxl
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from build_hss_set import load_cache  # noqa: E402
from common import now_iso, read_json, read_jsonl  # noqa: E402

REPO_ROOT = HERE.parent.parent

DEFAULT_MAX_CELL_CHARS = 32_000
DEFAULT_SOURCES_MAX_CHARS = 30_000
# Frozen for reproducibility: re-running the exporter with the same --items, in the same
# order, and this same default seed reproduces the identical opaque-id mapping and row order.
DEFAULT_SHUFFLE_SEED = 20260906
DEFAULT_SCREENING_SEED = 20260906
DEFAULT_SCREENING_INCLUDED_N = 40
DEFAULT_SCREENING_NEEDS_REVIEW_N = 40
#: The numbered-criterion excluded sample (splits the old single 60-record
#: "excluded" draw into this 40 plus the reserved off-topic id's own 20 below).
DEFAULT_SCREENING_EXCLUDED_N = 40
DEFAULT_SCREENING_OFF_TOPIC_N = 20
DEFAULT_SCREENING_FLOOR = 5
DEFAULT_SCREENING_MIN_BUCKET = 5
TRUNCATION_MARKER = "\n[... truncated]"

HUMAN_VERDICT_CHOICES = ("verified", "needs_nuance", "unsupported", "no_full_text")
#: The exact marker text ``evaluation/claims/annotation/aggregate_annotations_v2.py``'s
#: ``WITHHELD_MARKER`` uses for an item the model annotators had no passage or paper text for --
#: reused verbatim (not imported: this module makes no import from the annotation directory) so
#: a ``no_full_text`` Review row's own ``evidence_quote`` cell states plainly, in the reviewer's
#: own words from the earlier annotation round, that the tool was denied the text, rather than
#: leaving the cell blank (a blank cell alongside the Sources
#: sheet happening to carry that same paper's text -- shown there for a *different* item that
#: does have a ``chunk_doi`` -- was silently overriding the ``no_full_text`` label).
NO_FULL_TEXT_EVIDENCE_MARKER = (
    "Evidence withheld: no passage or paper text is available for this item"
)
#: Written into an evidence row's own ``abstract_shown`` cell when :func:`is_no_abstract_exempt`
#: exempts it from :func:`check_screening_abstracts`: the screener made its decision on that row
#: with no abstract at all, so the cell states that plainly rather than being left blank (which
#: would read as a build defect, indistinguishable from a genuine data-loss bug). Originally
#: only a ``needs_review`` row could earn this placeholder,
#: since the screener routed a record to NEEDS_REVIEW precisely for lacking an abstract; a demo
#: run showed an ``included`` decision can be made from the title alone too, so
#: any of the four evidence strata can now carry this placeholder -- see
#: :func:`is_no_abstract_exempt`.
NO_ABSTRACT_PLACEHOLDER = "(no abstract)"
#: Mirrors ``backend/app/agents/relevance_screener_agent.py``'s ``ABSTRACT_CHAR_LIMIT`` and
#: ``ABSTRACT_CUT_MARKER`` -- kept as literal constants rather than imported, for the same
#: reason :data:`TOPIC_CRITERION_ID` and :data:`STAGE_LLM` are (see the module docstring's
#: interpreter note: no backend import). The sheet's own
#: ``abstract_shown`` column used to carry the record's whole stored abstract, so a sampled
#: row whose abstract ran past this limit showed the human text the screener itself never saw;
#: :func:`render_abstract_for_screening` now applies the screener's own v2 cut rule so the
#: column means what its name says.
SCREENER_ABSTRACT_CHAR_LIMIT = 10_000
SCREENER_ABSTRACT_CUT_MARKER = " [...] "
YES_NO_CHOICES = ("yes", "no")
DISAGREEMENT_AXIS_CHOICES = (
    "main finding", "added detail", "scope or population", "strength or hedge",
    "quote fidelity", "construct naming", "other",
)
SCREENING_HUMAN_STATUS_CHOICES = ("include", "needs_review", "exclude")
#: The reserved off-topic criterion id -- mirrors
#: ``backend/app/agents/relevance_screener_agent.py``'s ``TOPIC_CRITERION_ID``. Kept as a
#: literal constant rather than imported: this module makes no backend import (see the module
#: docstring's interpreter note).
TOPIC_CRITERION_ID = "TOPIC"
#: Mirrors ``backend/app/services/screening_record.py``'s ``STAGE_LLM``, same reason.
STAGE_LLM = "llm"

#: The four strata whose sampled rows carry an actual LLM screening decision -- usually made
#: from an abstract the model was shown, occasionally (:func:`is_no_abstract_exempt`) from the
#: title alone -- exactly the rows :func:`check_screening_abstracts` refuses the export over
#: when the abstract is blank and the row is not itself exempt.
EVIDENCE_STRATA = ("included", "needs_review", "excluded_numbered", "excluded_off_topic")
#: The four whole-population invariant checks -- "the point of the sheet as
#: much as the samples are" -- the first three are expected to be empty (a guard failure would
#: put a row there), ``unscreened`` is a census because the human needs to see why whenever it
#: is not.
CENSUS_STRATA = (
    "excluded_full_text_violation", "excluded_missing_anchor", "excluded_no_abstract",
    "unscreened",
)
#: Display/provenance order for every stratum a screening sample can tag a row with.
SCREENING_STRATA = (*EVIDENCE_STRATA, *CENSUS_STRATA)

REVIEW_COLUMN_WIDTHS: dict[str, float] = {
    "item_id": 10, "set": 12, "claim": 45, "cited_title": 30, "cited_doi": 20,
    "evidence_quote": 35, "quote_relocated_original": 35,
    "evidence_quotes_all": 40, "verifier_status": 14, "model_status": 14,
    "first_pass_demotion": 22,
    "machine_reasons": 22, "diagnostics": 22, "unstated_details": 30, "central_assertion": 35,
    "run_B_status": 14, "human_verdict": 14, "human_agrees_with_verifier": 14,
    "disagreement_axis": 18, "human_quote": 30, "human_justification": 35,
    "countersigned_by": 16, "countersigned_at": 16,
}
SCREENING_COLUMN_WIDTHS: dict[str, float] = {
    "record_id": 10, "outcome": 12, "stage": 10, "status": 14, "criterion": 14, "quote": 30,
    "needs_review_reason": 20, "to_confirm": 20, "reason": 35, "title": 40, "doi": 20,
    "year": 8, "journal": 24, "journal_issn": 14, "openalex_id": 16, "abstract_shown": 60,
    "run_B_status": 14, "human_status": 14, "human_agrees": 12, "quote_is_verbatim": 16,
    "criterion_is_right": 16, "human_note": 30, "stratum": 22,
}
#: Delivered sheet: the final product a user receives, one row per delivered sentence.
#: ``section_title``/``draft_id`` added (a single demo run can now deliver more than
#: one generated section into one sheet).
DELIVERED_COLUMN_WIDTHS: dict[str, float] = {
    "row": 8, "section_title": 26, "draft_id": 20, "sentence": 45, "claim_checked": 45,
    "citation_text": 20, "paper_title": 30, "paper_authors": 30, "paper_doi": 20,
    "evidence_quotes": 40, "source_passage": 60, "source_located": 14, "passage_located": 16,
    "unlocated_quotes": 30, "location_section_mismatch": 22, "sentence_in_draft": 16,
    "human_sentence_correct": 18, "human_quote_supports": 16, "human_note": 30,
}


class ExportError(ValueError):
    """A structural problem :func:`build_workbook` (or :func:`build_screening_sheet`) refuses
    to silently paper over (e.g. an item with no primary-run result, a wrong ``--cache-dir``,
    or a screening record produced by a screener prompt other than the one the caller
    expects)."""


# --------------------------------------------------------------------------------------
# Pure helpers -- Review sheet columns
# --------------------------------------------------------------------------------------


def classify_set(item: Mapping[str, Any]) -> str:
    """``"real"`` when the item's own construction ``rule`` is literally ``"real"`` (every
    harvested real-claims item carries ``rule: "real"``), else ``"constructed"`` (every HSS
    rule name -- ``verbatim``, ``paraphrase``, ``altered``, ``over_specified``, ``wrong_paper``,
    ``no_full_text``). Both sets carry a truthy ``rule``, so this cannot use plain truthiness."""
    return "real" if item.get("rule") == "real" else "constructed"


def cited_title_and_doi(item: Mapping[str, Any]) -> tuple[str, str]:
    """The paper the claim cites, for a reviewer to look the paper up: a real-claims item
    carries its own ``cited_title``/``cited_doi`` directly; a constructed item has no such
    field and instead carries ``title`` for the paper it was built from, and either
    ``chunk_doi`` (the paper the verifier actually saw) or, for a ``no_full_text`` item with no
    ``chunk_doi``, ``source_doi`` (the paper being cited, even though its text was withheld)."""
    if item.get("cited_doi") or item.get("cited_title"):
        return str(item.get("cited_title") or ""), str(item.get("cited_doi") or "")
    title = str(item.get("title") or "")
    doi = str(item.get("chunk_doi") or item.get("source_doi") or "")
    return title, doi


def census_counts(items: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """``(n_constructed, n_real)`` -- how many of *items* :func:`classify_set` calls each way.
    Used for the Guide sheet's own census sentence: the Review
    sheet carries every one of both counts, not a sample of either."""
    n_real = sum(1 for it in items if classify_set(it) == "real")
    return len(items) - n_real, n_real


def joined_list(values: Sequence[str] | None) -> str:
    return "; ".join(v for v in (values or []) if v)


def numbered_list(values: Sequence[str] | None) -> str:
    """One value to a line, each prefixed ``"<n>) "`` and newline-joined -- used for the
    Delivered sheet's ``evidence_quotes``/``unlocated_quotes`` cells so a semicolon inside a
    quote is never mistaken for a boundary between quotes (unlike :func:`joined_list`'s
    ``"; "`` join, which some of this run's quote strings contain themselves). The Review
    sheet's own :func:`joined_list`-based columns are untouched."""
    items = [v for v in (values or []) if v]
    return "\n".join(f"{i}) {v}" for i, v in enumerate(items, start=1))


def _is_null_string(value: Any) -> bool:
    """``True`` when *value* is the literal four-character string ``"null"`` (any case, with
    surrounding whitespace) -- the verifier's own placeholder for "no evidence" on a couple of
    run rows, otherwise indistinguishable in a cell from a real quote. Rows with no evidence at
    all carry an empty ``evidence_quote``/``evidence_quotes``
    instead, so this check only ever fires on the placeholder string, not on a genuinely blank
    field."""
    return isinstance(value, str) and value.strip().casefold() == "null"


def first_evidence_quote(row: Mapping[str, Any]) -> str:
    quotes = [q for q in (row.get("evidence_quotes") or []) if not _is_null_string(q)]
    if quotes:
        return str(quotes[0] or "")
    quote = row.get("evidence_quote")
    return "" if _is_null_string(quote) else str(quote or "")


def all_evidence_quotes(row: Mapping[str, Any]) -> str:
    quotes = row.get("evidence_quotes") or (
        [row["evidence_quote"]] if row.get("evidence_quote") else []
    )
    quotes = [q for q in quotes if not _is_null_string(q)]
    return joined_list(quotes)


def relocated_quote_original(row: Mapping[str, Any]) -> str:
    """The model's own, pre-relocation quote text for every quote-relocation ``run["quote_
    relocations"]`` recorded on *row* (``app.services.fulltext.relocate_evidence_quotes``),
    joined the same way ``machine_reasons``/``diagnostics`` are --
    ``""`` when *row* carries no relocation at all.

    The Review sheet's own ``evidence_quote``
    column shows the *stored* quote, which for a relocated row is the source's own text, not
    what the model wrote. Without this, the human judge whose verdict the paper reports
    countersigns a ``verified`` row against a quote the model never produced. Read from
    ``row["quote_relocations"][*]["original"]``, already present on the row
    (`ProductionVerifier`/``run_hss.py``/``run_scifact.py`` all carry it forward)."""
    return joined_list(
        [str(r.get("original") or "") for r in (row.get("quote_relocations") or [])]
    )


def first_pass_demotion(row: Mapping[str, Any]) -> str:
    """The first pass's own guard reasons, joined the same way ``machine_reasons`` is, when
    the shared verification policy's bounded second pass fired for this row
    (``row["passes"]`` has two entries) -- ``""`` for the common, single-pass case.

    Task authorisation 2026-09-10 (Part B): the second pass fires only when the first pass
    was ``needs_nuance`` with ``machine_reasons == ["quote_not_verbatim"]`` after a model
    status of ``verified``; showing that first-pass reason beside the final
    ``verifier_status`` lets the reviewer see that a first-pass demotion happened even though
    the tool went on to resolve it, without opening the JSON run file's full ``passes`` list.
    """
    passes = row.get("passes") or []
    if len(passes) < 2:
        return ""
    return joined_list(passes[0].get("machine_reasons"))


def central_assertion_text(row: Mapping[str, Any]) -> str:
    """The ``text`` of the single assertion marked ``central`` in the run row's own
    ``assertions`` list; ``""`` when zero or more than one assertion is marked central (the
    acceptance report's own rule: an ambiguous inventory should not be silently collapsed to
    one sentence)."""
    assertions = row.get("assertions") or []
    centrals = [a for a in assertions if isinstance(a, Mapping) and a.get("central")]
    if len(centrals) == 1:
        return str(centrals[0].get("text") or "")
    return ""


def run_b_status_if_differs(row_a: Mapping[str, Any], row_b: Mapping[str, Any] | None) -> str:
    """Run B's ``predicted_status``, but only when it differs from run A's own -- an empty
    cell otherwise, so the column reads as a list of the items worth a second look, not a
    second full column to compare by eye."""
    if not row_b:
        return ""
    status_a = row_a.get("predicted_status")
    status_b = row_b.get("predicted_status")
    return str(status_b) if status_b and status_b != status_a else ""


def cap_text(text: str | None, limit: int, marker: str = TRUNCATION_MARKER) -> str:
    """Caps *text* at *limit* characters, appending *marker* when it was cut."""
    text = text or ""
    if len(text) <= limit:
        return text
    if limit <= 0:
        return ""
    marker = marker[:limit]
    keep = max(limit - len(marker), 0)
    return (text[:keep] + marker)[:limit]


def sanitize_for_excel(value: Any) -> Any:
    """Strips XML-illegal control characters (e.g. a stray form-feed or NUL surviving a PDF
    text extraction) from a string cell value -- openpyxl otherwise raises
    ``IllegalCharacterError`` and the whole export fails partway through writing the Sources
    sheet. Non-strings (``None``, ints, ...) pass through unchanged."""
    if isinstance(value, str):
        return ILLEGAL_CHARACTERS_RE.sub("", value)
    return value


def _append_row(ws: Any, values: Sequence[Any]) -> None:
    """Appends *values* as a new worksheet row, running every value through
    :func:`sanitize_for_excel` first. The one place any sheet in this module ever appends a
    row -- every ``write_*_sheet``/``append_*`` function below calls this rather than
    ``ws.append`` directly, so a control character surviving a PDF extraction or carried in a
    screening record's own free text can never reach openpyxl unsanitised regardless of which
    sheet it lands on."""
    ws.append([sanitize_for_excel(v) for v in values])


def _set_cell(ws: Any, row: int, column: int, value: Any) -> Any:
    """Sets a single cell to *value*, sanitised the same way as :func:`_append_row` -- used by
    the Guide sheet, which is written cell by cell rather than row by row."""
    return ws.cell(row=row, column=column, value=sanitize_for_excel(value))


def split_into_parts(text: str, limit: int) -> list[str]:
    text = text or ""
    if not text:
        return []
    return [text[i:i + limit] for i in range(0, len(text), limit)]


# --------------------------------------------------------------------------------------
# Item / cache / result loading
# --------------------------------------------------------------------------------------


def resolve_item_specs(
    items: Sequence[Path], cache_dirs: Sequence[Path],
) -> list[tuple[Path, Path]]:
    """Pairs each ``--items`` path with its cache dir: one ``--cache-dir`` applies to every
    item file; otherwise there must be exactly one per item file, in the same order."""
    dirs = list(cache_dirs)
    if len(dirs) == 1:
        dirs = dirs * len(items)
    if len(dirs) != len(items):
        raise SystemExit(
            f"--cache-dir must give one path (applied to every --items file) or exactly as "
            f"many as --items ({len(items)}); got {len(dirs)}"
        )
    return list(zip(items, dirs, strict=True))


def read_items(item_specs: Sequence[tuple[Path, Path]]) -> list[dict[str, Any]]:
    """One dict per item across every (item file, cache dir) pair, each carrying an added
    ``_cache_dir`` (str) recording which cache dir its chunks live in."""
    items: list[dict[str, Any]] = []
    for item_path, cache_dir in item_specs:
        for row in read_jsonl(item_path):
            row = dict(row)
            row["_cache_dir"] = str(cache_dir)
            items.append(row)
    return items


def build_caches(item_specs: Sequence[tuple[Path, Path]]) -> dict[str, dict[str, Any]]:
    """``{str(cache_dir): {doi: payload}}`` -- one cache per distinct cache dir referenced."""
    dirs = {str(cache_dir) for _, cache_dir in item_specs}
    return {d: load_cache(Path(d)) for d in dirs}


def chunk_texts_for_item(
    item: Mapping[str, Any], caches: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    doi = item.get("chunk_doi")
    if not doi:
        return []
    cache = caches.get(item.get("_cache_dir"), {})
    paper = cache.get(doi)
    if not paper:
        return []
    return [str(c.get("text", "")) for c in paper.get("chunks", [])]


def full_text_for_doi(doi: str, cache: Mapping[str, Any]) -> str:
    paper = cache.get(doi)
    if not paper:
        return ""
    return "\n\n".join(str(c.get("text", "")) for c in paper.get("chunks", []))


def chunk_coverage(
    items: Sequence[Mapping[str, Any]], caches: Mapping[str, Mapping[str, Any]],
) -> tuple[int, int]:
    """``(n_found, n_with_chunk_doi)``: how many of the items whose ``chunk_doi`` is truthy
    actually find a cached chunk for it -- used to detect a wrong ``--cache-dir`` (which
    otherwise exports a complete-looking Sources sheet missing every paper)."""
    total = 0
    found = 0
    for item in items:
        if not item.get("chunk_doi"):
            continue
        total += 1
        if chunk_texts_for_item(item, caches):
            found += 1
    return found, total


MIN_CHUNK_COVERAGE = 0.9


def load_results(
    results_dir: Path, runs: Sequence[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """``{run: {item_id: row}}``: merges every ``*_run<run>.jsonl`` found directly under
    *results_dir*, so items from different ``--items`` files are joined by ``item_id``."""
    out: dict[str, Any] = {}
    for run in runs:
        rows: dict[str, Any] = {}
        for path in sorted(Path(results_dir).glob(f"*_run{run}.jsonl")):
            for row in read_jsonl(path):
                item_id = row.get("item_id")
                if item_id:
                    rows[item_id] = row
        out[run] = rows
    return out


def load_run_meta_field_versions(results_dir: Path, run: str, field: str) -> list[str]:
    """The distinct string values of *field* across every ``*_run<run>.meta.json`` under
    *results_dir* -- read from the run's own recorded metadata, never asserted by the caller,
    so the Provenance sheet cannot silently drift from what actually produced the rows. Used
    for ``prompt_version`` (:func:`load_run_prompt_versions`) and, since the repair-turn
    results (``results/v5``) introduced them, ``verification_policy_version`` and
    ``repair_prompt_version`` -- both absent from every ``v3``/``v4`` meta file, so a run
    predating them yields ``[]`` rather than a missing-key error."""
    versions: list[str] = []
    for path in sorted(Path(results_dir).glob(f"*_run{run}.meta.json")):
        meta = read_json(path, {}) or {}
        value = meta.get(field)
        if value and str(value) not in versions:
            versions.append(str(value))
    return versions


def load_run_prompt_versions(results_dir: Path, run: str) -> list[str]:
    """The distinct ``prompt_version`` values across every ``*_run<run>.meta.json`` under
    *results_dir*. See :func:`load_run_meta_field_versions`."""
    return load_run_meta_field_versions(results_dir, run, "prompt_version")


def missing_primary_result_ids(
    items: Sequence[Mapping[str, Any]], results_by_run: Mapping[str, Mapping[str, Any]],
    primary_run: str,
) -> list[str]:
    primary_rows = results_by_run.get(primary_run, {})
    return [it["item_id"] for it in items if it["item_id"] not in primary_rows]


# --------------------------------------------------------------------------------------
# Blinding: seeded shuffle and opaque item ids
# --------------------------------------------------------------------------------------


def assign_opaque_ids(
    items: Sequence[Mapping[str, Any]], seed: int,
) -> tuple[list[Mapping[str, Any]], dict[str, str]]:
    """Shuffles *items* with ``random.Random(seed)`` and assigns each its rank in that shuffle
    as an opaque ``"R-001"``-style id -- this is the Review sheet's own ``item_id`` column;
    the real ``item_id`` (whose ``hss-<rule>-<nn>`` shape would name the construction rule, and
    through it the expected label) never appears on the sheet, only in ``--key-out``."""
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    shuffled = [items[i] for i in order]
    opaque_ids = {
        it["item_id"]: f"R-{rank:03d}" for rank, it in enumerate(shuffled, start=1)
    }
    return shuffled, opaque_ids


def compute_workbook_id(item_ids: Sequence[str], seed: int) -> str:
    """A short, deterministic id binding one export's workbook to its key csv: a sha256 of
    the shuffle *seed* and the exact ordered list of *item_ids*, truncated to 16 hex
    characters."""
    h = hashlib.sha256()
    h.update(str(seed).encode("utf-8"))
    for item_id in item_ids:
        h.update(b"\x00")
        h.update(str(item_id).encode("utf-8"))
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------------------
# Review row assembly
# --------------------------------------------------------------------------------------

REVIEW_TEXT_COLUMNS_TO_CAP = (
    "claim", "cited_title", "evidence_quote", "quote_relocated_original",
    "evidence_quotes_all", "first_pass_demotion", "machine_reasons", "diagnostics",
    "unstated_details", "central_assertion",
)


def evidence_withheld(item: Mapping[str, Any]) -> bool:
    """``True`` for an item with no ``chunk_doi`` -- the ``no_full_text`` construction rule, or
    any other item the verifier never had the paper's text for -- the same gate
    :func:`build_source_rows` already uses to decide whether an item contributes anything of
    its own to the Sources sheet. Kept as its own named check (rather than the condition
    inlined at each call site) so :func:`build_review_row` and any future caller agree on
    exactly one definition of "withheld"."""
    return not item.get("chunk_doi")


def review_header() -> list[str]:
    return [
        "item_id", "set", "claim", "cited_title", "cited_doi", "evidence_quote",
        "quote_relocated_original", "evidence_quotes_all", "verifier_status", "model_status",
        "first_pass_demotion",
        "machine_reasons", "diagnostics", "unstated_details", "central_assertion",
        "run_B_status", "human_verdict", "human_agrees_with_verifier", "disagreement_axis",
        "human_quote", "human_justification", "countersigned_by", "countersigned_at",
    ]


def build_review_row(
    item: Mapping[str, Any],
    results_by_run: Mapping[str, Mapping[str, Mapping[str, Any]]],
    opaque_id: str,
    *,
    runs: Sequence[str],
    max_cell_chars: int,
) -> dict[str, Any]:
    item_id = item["item_id"]
    primary_run = runs[0]
    row_a = results_by_run.get(primary_run, {}).get(item_id, {})
    row_b = results_by_run.get(runs[1], {}).get(item_id) if len(runs) > 1 else None
    title, doi = cited_title_and_doi(item)
    evidence_quote = (
        NO_FULL_TEXT_EVIDENCE_MARKER if evidence_withheld(item) else first_evidence_quote(row_a)
    )

    out: dict[str, Any] = {
        "item_id": opaque_id,
        "set": classify_set(item),
        "claim": item.get("claim") or "",
        "cited_title": title,
        "cited_doi": doi,
        "evidence_quote": evidence_quote,
        "quote_relocated_original": (
            "" if evidence_withheld(item) else relocated_quote_original(row_a)
        ),
        "evidence_quotes_all": all_evidence_quotes(row_a),
        "verifier_status": row_a.get("predicted_status") or "",
        "model_status": row_a.get("model_status") or "",
        "first_pass_demotion": first_pass_demotion(row_a),
        "machine_reasons": joined_list(row_a.get("machine_reasons")),
        "diagnostics": joined_list(row_a.get("diagnostics")),
        "unstated_details": joined_list(row_a.get("unstated_details")),
        "central_assertion": central_assertion_text(row_a),
        "run_B_status": run_b_status_if_differs(row_a, row_b),
        "human_verdict": "", "human_agrees_with_verifier": "", "disagreement_axis": "",
        "human_quote": "", "human_justification": "", "countersigned_by": "",
        "countersigned_at": "",
    }
    for key in REVIEW_TEXT_COLUMNS_TO_CAP:
        out[key] = cap_text(out[key], max_cell_chars)
    return out


def build_key_row(
    item: Mapping[str, Any], opaque_id: str, workbook_id: str,
) -> dict[str, Any]:
    return {
        "opaque_id": opaque_id,
        "real_item_id": item["item_id"],
        "set": classify_set(item),
        "construction_category": item.get("rule") or "",
        "expected_label": ";".join(item.get("expected") or []) if item.get("expected") else "",
        "workbook_id": workbook_id,
    }


# --------------------------------------------------------------------------------------
# Model_labels sheet
# --------------------------------------------------------------------------------------


def load_annotation_labels(paths: Sequence[Path]) -> dict[str, dict[str, str]]:
    """``{item_id: {"label": final_label, "rationale": ..., "agreement": source}}`` merged
    across one or more adjudicated-annotation csv files (``item_id, final_label, source,
    rationale, ...`` -- the ``hss_annotation_adjudicated_v3.csv``/``real_annotation_
    adjudicated_v3.csv`` shape). ``source`` is either ``"unanimous"`` or ``"adjudicated"``,
    which is the annotator-agreement fact the acceptance report asks the sheet to carry."""
    out: dict[str, dict[str, str]] = {}
    for path in paths:
        with Path(path).open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                item_id = row.get("item_id")
                if not item_id:
                    continue
                out[item_id] = {
                    "label": row.get("final_label") or "",
                    "rationale": row.get("rationale") or "",
                    "agreement": row.get("source") or "",
                }
    return out


def model_labels_header() -> list[str]:
    return ["item_id", "model_annotation_label", "rationale", "annotator_agreement"]


def build_model_labels_rows(
    items: Sequence[Mapping[str, Any]], opaque_ids: Mapping[str, str],
    annotations: Mapping[str, Mapping[str, str]], *, max_cell_chars: int,
) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        ann = annotations.get(item["item_id"])
        if not ann:
            continue
        rows.append({
            "item_id": opaque_ids[item["item_id"]],
            "model_annotation_label": ann.get("label") or "",
            "rationale": cap_text(ann.get("rationale") or "", max_cell_chars),
            "annotator_agreement": ann.get("agreement") or "",
        })
    rows.sort(key=lambda r: r["item_id"])
    return rows


# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------


def _guard_path_is_dirty() -> bool:
    """``True`` when ``git status --porcelain`` reports an uncommitted change to
    ``backend/app/services/fulltext.py`` in this checkout -- meaning whatever commit
    :func:`resolve_guard_commit` names (explicit or auto-detected) is not actually what ran.
    ``False``, never raises, when git is unavailable or this is not a checkout at all: a
    missing git binary is a normal case for a dry-run export, not a build error."""
    guard_path = REPO_ROOT / "backend" / "app" / "services" / "fulltext.py"
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", str(guard_path)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return bool((result.stdout or "").strip())


def resolve_guard_commit(explicit: str | None) -> str:
    """*explicit* when given; otherwise the last commit that touched
    ``backend/app/services/fulltext.py`` (the file the two acceptance reports name as carrying
    the verification guards), from ``git log`` if this is a git checkout; ``"unknown"``
    otherwise. Never raises: a missing git binary or a non-repo checkout is a normal case for
    a dry-run export, not a build error. Appends a ``" (dirty: ...)"`` marker (see
    :func:`_guard_path_is_dirty`) whenever that file has an uncommitted change in this
    checkout, explicit commit or not -- an export run against a dirty guard file asserts a
    guard state the tree does not actually match, and the run's own metadata has no other way
    to catch it later."""
    guard_path = REPO_ROOT / "backend" / "app" / "services" / "fulltext.py"
    if explicit:
        base = explicit
    else:
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%H", "--", str(guard_path)],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return "unknown"
        sha = (result.stdout or "").strip()
        base = sha if result.returncode == 0 and sha else "unknown"
    if base != "unknown" and _guard_path_is_dirty():
        return f"{base} (dirty: backend/app/services/fulltext.py has uncommitted changes)"
    return base


def _run_results_are_dirty(results_dir: Path) -> bool:
    """``True`` when ``git status --porcelain`` reports an uncommitted change under
    *results_dir* -- meaning :func:`resolve_run_commit`'s answer is not actually what produced
    the rows on the Review sheet. ``False``, never raises, when git is unavailable or this is
    not a checkout at all, mirroring :func:`_guard_path_is_dirty`. *results_dir* is resolved to
    an absolute path first (against the process's own cwd, e.g. ``--results-dir claims/results/v3``
    typed from ``evaluation/``) before being handed to git with ``cwd=REPO_ROOT``, so a
    ``--results-dir`` given relative to some other directory than the repo root still resolves
    to the right files instead of silently matching nothing."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", str(Path(results_dir).resolve())],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return bool((result.stdout or "").strip())


def resolve_run_commit(results_dir: Path) -> str:
    """The last commit that touched *results_dir* -- the commit whose files actually produced
    the Review sheet's own rows (the "run commit" fact) -- from ``git log`` if
    this is a git checkout, ``"unknown"`` otherwise. Auto-detected only: a run's commit is a
    fact about files already on disk, not a choice the export makes, so there is no CLI
    override, unlike :func:`resolve_guard_commit`. Appends a ``" (dirty: ...)"`` marker
    whenever *results_dir* has an uncommitted change in this checkout, for the same reason
    :func:`resolve_guard_commit` does. *results_dir* is resolved to an absolute path first; see
    :func:`_run_results_are_dirty`."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(Path(results_dir).resolve())],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = (result.stdout or "").strip()
    base = sha if result.returncode == 0 and sha else "unknown"
    if base != "unknown" and _run_results_are_dirty(results_dir):
        return f"{base} (dirty: {results_dir} has uncommitted changes)"
    return base


def run_files_shown(results_dir: Path, run: str) -> list[str]:
    """The file names of every ``*_run<run>.jsonl`` under *results_dir*, in the same sorted
    order :func:`load_results` reads them -- the "run shown" fact."""
    return [p.name for p in sorted(Path(results_dir).glob(f"*_run{run}.jsonl"))]


def load_run_model_facts(results_dir: Path, run: str) -> str:
    """The model, temperature and session count actually recorded in every
    ``*_run<run>.meta.json`` under *results_dir* -- read from the run's own metadata, never
    asserted by the caller, matching :func:`load_run_prompt_versions`'s own rule (the "model"
    fact). ``"unknown"`` when no meta file carries this run; when the meta files
    disagree with each other, names the disagreeing values instead of silently averaging or
    picking one, since a mismatch there is a build defect, not a fact to summarise."""
    models: set[str] = set()
    temperatures: set[Any] = set()
    session_counts: set[int] = set()
    for path in sorted(Path(results_dir).glob(f"*_run{run}.meta.json")):
        meta = read_json(path, {}) or {}
        reported = meta.get("model_reported") or []
        models.add(
            ", ".join(str(m) for m in reported) or str(meta.get("model_configured") or "unknown")
        )
        temperatures.add(meta.get("temperature"))
        session_counts.add(len(meta.get("sessions") or []))
    if not models:
        return "unknown"
    if len(models) > 1 or len(temperatures) > 1 or len(session_counts) > 1:
        return (
            f"inconsistent across the run {run} meta files (models={sorted(models)}, "
            f"temperatures={sorted(temperatures, key=str)}, "
            f"session_counts={sorted(session_counts)})"
        )
    model = next(iter(models))
    temperature = next(iter(temperatures))
    sessions = next(iter(session_counts))
    plural = "" if sessions == 1 else "s"
    return f"{model}, temperature {temperature}, {sessions} session{plural} per run"


def annotation_label_facts(annotation_paths: Sequence[Path]) -> str:
    """One ``"<file name> (sha256:<hexdigest>)"`` per ``--annotation`` file, joined with
    ``"; "`` -- the hash computed from the file's own bytes, never asserted by the caller, so
    this fact cannot drift from the actual labels file (the "labels" fact).
    ``"n/a"`` when no annotation file was given."""
    if not annotation_paths:
        return "n/a"
    parts = []
    for path in annotation_paths:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        parts.append(f"{Path(path).name} (sha256:{digest})")
    return "; ".join(parts)


#: The "labels provenance" and "blindness" facts: narrative facts about how the
#: adjudicated labels and the frozen prompt/guards came to be, not derivable from any file the
#: exporter reads, so they are fixed strings rather than computed values.
LABELS_PROVENANCE_FACT = (
    "three sessions of one large language model, not human experts, no author countersignature"
)
BLINDNESS_FACT = (
    "not blind; prompt revised after round 1, guards revised after rounds 1 and 2, and "
    "since then the quote-relocation policy (quote_relocation_version) and the quote-repair "
    "turn (repair_prompt_version) were both added with full knowledge of the annotation "
    "labels, and both can change a row's status on this sheet"
)


def provenance_rows(
    *, verifier_prompt_shas: Sequence[str], guard_commit: str, results_dir: Path,
    export_time: str, demo_run_id: str, screening_prompt_version: str,
    run_shown: Sequence[str] = (), run_commit: str = "unknown", model: str = "unknown",
    labels: str = "n/a", labels_provenance: str = LABELS_PROVENANCE_FACT,
    blindness: str = BLINDNESS_FACT, policy_version: str = "unknown",
    repair_prompt_version: str = "unknown",
) -> list[dict[str, str]]:
    """The nine facts required on the Provenance sheet itself, plus the export
    mechanics (``export_time``, ``demo_run_id``, ``screening_prompt_version``) the sheet already
    carried, plus the two facts the repair turn added
    (``verification_policy_version``, ``repair_prompt_version``) -- both default to ``"unknown"``
    for a results directory (``v3``, ``v4``) whose meta files predate the repair turn and never
    recorded them."""
    return [
        {"field": "verifier_prompt_sha", "value": "; ".join(verifier_prompt_shas) or "unknown"},
        {"field": "guard_commit", "value": guard_commit},
        {"field": "verification_policy_version", "value": policy_version},
        {"field": "repair_prompt_version", "value": repair_prompt_version},
        {"field": "results_directory", "value": str(results_dir)},
        {"field": "run_shown", "value": "; ".join(run_shown) or "unknown"},
        {"field": "run_commit", "value": run_commit},
        {"field": "model", "value": model},
        {"field": "labels", "value": labels},
        {"field": "labels_provenance", "value": labels_provenance},
        {"field": "blindness", "value": blindness},
        {"field": "export_time", "value": export_time},
        {"field": "demo_run_id", "value": demo_run_id},
        {"field": "screening_prompt_version", "value": screening_prompt_version},
    ]


def _populate_provenance_sheet(ws: Any, rows: Sequence[Mapping[str, str]]) -> None:
    _append_row(ws, ["field", "value"])
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for row in rows:
        _append_row(ws, [row["field"], row["value"]])
    ws.freeze_panes = "A2"


def write_provenance_sheet(wb: Any, rows: Sequence[Mapping[str, str]]) -> None:
    ws = wb.create_sheet("Provenance")
    _populate_provenance_sheet(ws, rows)
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 80


# --------------------------------------------------------------------------------------
# Guide sheet
# --------------------------------------------------------------------------------------


def _wrap_paragraph(text: str, limit: int) -> list[str]:
    """Word-wraps *text* at *limit* characters, never merging words that were on opposite
    sides of an explicit newline in *text* onto the same output line. A section's own internal
    newlines (for example the numbered-criteria block, one criterion per line) are a real
    boundary, not a run of whitespace: the naive whole-string word-wrap used to fold the tail of
    one criterion and the head of the next into a single 900-character line, so a reviewer read
    a criterion that had visibly changed mid-sentence."""
    lines: list[str] = []
    for raw_line in text.split("\n"):
        words = raw_line.split(" ")
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) > limit and current:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
    return lines


#: The instruction the Guide sheet must carry, verbatim, as the first
#: thing the reviewer reads: the sheet is scored against
#: ``verifier_status`` and the model labels, so a reviewer who instead rules on whether the
#: tool's own explanation is convincing is answering a different question from the one those
#: labels answer.
GUIDE_OPENING_SENTENCE = (
    "You are judging the claim against the cited paper, not against the verifier's explanation."
)
#: The four ``human_verdict`` definitions, copied verbatim (word for word, not paraphrased)
#: from ``evaluation/claims/annotation/INSTRUCTIONS_v3.md``'s own "## Labels" section -- the
#: rubric the adjudicated ``Model_labels`` sheet was itself produced under (an earlier
#: paraphrase used by the exporter differed from that rubric, and for ``no_full_text``
#: the paraphrase omitted the "regardless of how plausible
#: the claim sounds" rule). Nothing enforces agreement between this tuple and
#: INSTRUCTIONS_v3.md going forward; a future edit to either needs to update both.
STATUS_DEFINITIONS_V3: tuple[str, ...] = (
    "verified: the cited paper's text supports the claim as stated.",
    "needs_nuance: the paper supports only a weaker, narrower or conditional version; the "
    "claim overstates, generalises or drops a qualifier. This also covers the case where the "
    "paper supports the claim's main finding but the claim also states a detail, such as a "
    "setting, a population, a time window or an instrument, that the cited paper does not "
    "state.",
    "unsupported: the cited paper contradicts the claim, or does not address it, so the "
    "citation does not support it.",
    "no_full_text: the item shows the marker \"Evidence withheld\" and no passage or paper "
    "text is available for it; apply this label in that case regardless of how plausible the "
    "claim sounds.",
)
#: The required wording: a withheld row is still answered no_full_text even where the same
#: underlying paper happens to appear on the Sources sheet under a *different* item's own
#: item_id (one that does have a chunk_doi).
NO_FULL_TEXT_SOURCES_NOTE = (
    "A row whose evidence_quote reads \"" + NO_FULL_TEXT_EVIDENCE_MARKER + "\" is answered "
    "no_full_text regardless of how plausible the claim sounds. The Sources sheet is keyed by "
    "item_id, so this row's own item_id has no Sources rows at all -- but the same underlying "
    "paper may still appear there under a different item's own item_id, if another item cites "
    "the same paper and does have a chunk_doi; that text was shown to the tool for that other "
    "item, not for this row."
)


#: Nothing on the Guide, Review or Provenance sheet used to
#: say the Review sheet is a census rather than a sample. Kept as a template rather than a
#: fixed string since the two counts vary export to export.
REVIEW_CENSUS_SENTENCE_TEMPLATE = (
    "The Review sheet is a census, not a sample: it carries every one of the {n_constructed} "
    "constructed item(s) and every one of the {n_real} real item(s) given to the exporter, "
    "not a subset of either. A reviewer's own agreement rate is therefore a rate over the "
    "whole set, not an estimate from a sample of it."
)
#: Describes the census strata correctly when the record's own
#: criteria stage no criterion full-text (:func:`full_text_criterion_ids` empty) -- the case
#: every run has carried so far. Only two of the four census strata are then a live guard
#: invariant a future run could actually trip; a third, unscreened, is a real but non-guard
#: outcome; the fourth is a structural fact about this run's criteria, not a passed check.
#: unscreened must not be folded in with the two genuine guard invariants (a non-empty
#: unscreened row is not a bug report --
#: ``backend/app/services/smart_search.py`` records one for a cancelled job, an exhausted time
#: budget or a failed screening batch call, all deliberate operational outcomes), and the
#: "fill in the same human columns" instruction the live variant below carries still applies to
#: it.
FULL_TEXT_CENSUS_STRUCTURAL_TEXT = (
    "Four of the sheet's strata are censuses. Two of them -- excluded_missing_anchor (an "
    "excluded record with no quote or criterion) and excluded_no_abstract (an excluded "
    "record with no abstract at all) -- are guard invariants expected to be empty; a "
    "non-empty one is a bug report, not a result. unscreened is its own case, not a guard "
    "invariant: the tool records it when a job is cancelled, a time budget runs out, or a "
    "screening batch's model call fails; it is usually zero, and a non-empty one is worth a "
    "note but not itself a bug report. The fourth, excluded_full_text_violation, cannot "
    "occur at all under this run's own criteria, since no criterion is staged full-text; it "
    "prints 0 of 0 as a structural fact, not a passed guard check. Fill in the same human "
    "columns for these rows as for any other."
)
#: The wording used instead of :data:`FULL_TEXT_CENSUS_STRUCTURAL_TEXT` when the record's own
#: criteria do stage at least one criterion full-text, so an excluded_full_text_violation row
#: could genuinely be drawn -- every one of the four strata is then a live invariant.
FULL_TEXT_CENSUS_LIVE_TEXT = (
    "Four of the sheet's strata are censuses of a guard invariant expected to be "
    "empty -- an excluded record whose criterion the protocol marks full-text, an "
    "excluded record with no quote or criterion, an excluded record with no abstract "
    "at all -- plus every unscreened record. Fill in the same human columns for these "
    "rows as for any other; a non-empty one of the first three is a bug report, not a "
    "result, and worth a note either way."
)
#: Parallel to the no-abstract note below, for the other
#: needs_review_reason a human is otherwise given no explanation of. Also tells the reviewer
#: that this dictated answer is excluded from
#: score_review_sheet.py's own per_status and overall numbers and reported on its own instead
#: (see score_review_sheet.py::_split_unanchored_exclude) -- otherwise a reviewer who follows
#: this instruction everywhere would see their own agreement rate deflated by an answer the
#: Guide itself dictated, not one they disagreed on.
UNANCHORED_EXCLUDE_NOTE = (
    "A row whose needs_review_reason reads \"unanchored_exclude\" is a needs_review row for a "
    "different reason than a missing abstract: the tool's own substantive judgement was an "
    "exclusion, but a guard held it back because the quote or criterion it gave was not "
    "anchored, verbatim, in the shown text (the same question quote_is_verbatim asks you to "
    "answer). If you read the criteria and agree the record should be excluded, write "
    "human_status exclude and human_agrees no -- you are disagreeing with the needs_review "
    "routing, not with the underlying exclusion -- and say so in human_note, so this reads "
    "as the guard working as intended, not a routing error. This dictated answer is "
    "excluded from per_status and overall and reported there on its own. Still answer "
    "quote_is_verbatim and criterion_is_right on their own merits; that is a separate "
    "question from the routing itself. Your quote_is_verbatim answer on this row is also "
    "pulled out of the headline quote_is_verbatim_rate and reported in that rate's own "
    "nested unanchored_exclude rate instead, for the same reason; criterion_is_right has no "
    "such nested rate and stays inside the headline criterion_is_right_rate."
)
#: Parallel to the no-abstract paragraph above, for the other
#: synthetic marker abstract_shown can carry -- the Guide had a paragraph for the placeholder
#: but none for the cut marker :func:`render_abstract_for_screening` puts on a very long
#: abstract.
ABSTRACT_CUT_NOTE = (
    "A row whose abstract_shown contains the marker \"" + SCREENER_ABSTRACT_CUT_MARKER + "\" "
    "has had its stored abstract cut to a head, the marker, and a tail, the same render the "
    "screener itself was shown when the paper's own abstract ran past the screener's own "
    "character limit -- what you read is exactly what the model read, not the paper's whole "
    "abstract with a piece missing."
)
#: A relocated row's evidence_quote cell shows
#: the source's own text, not what the model wrote, and nothing on the Guide sheet said so.
QUOTE_RELOCATED_NOTE = (
    "A row whose diagnostics contains \"quote_relocated\" had at least one evidence quote "
    "automatically repointed onto the cited paper's own nearby text before verification, "
    "because the tool's own quote was close to, but not exactly, verbatim; "
    "quote_relocated_original then carries the tool's own, pre-relocation quote text, next "
    "to the stored evidence_quote, and is blank on every other row."
)
#: Part B, the bounded second pass: a row whose first pass
#: was demoted, then resolved by a second call to the model, has no diagnostics slug of its
#: own and nothing on the Guide sheet said so before this.
SECOND_PASS_NOTE = (
    "A row whose first_pass_demotion is not blank was verified twice: the tool's first "
    "answer was \"verified\" but at least one of its quotes did not match the source, so the "
    "tool sent a follow-up message in the same conversation naming the quote segment(s) that "
    "were not verbatim and asking the model for a corrected, character-for-character span of "
    "the supplied source for each one, or to change status when no such "
    "span exists -- not the same question asked again from scratch. verifier_status/"
    "model_status and evidence_quote/evidence_quotes_all show that follow-up's own, final "
    "answer, which can drop a quote the first pass offered if no verbatim span for it could "
    "be found, so a quote you saw on an earlier copy of this row may have been withdrawn "
    "here; first_pass_demotion shows the first pass's own demotion reason so you can see that "
    "it happened even though the tool went on to resolve it."
)
def _count_phrase(n: int | None, unit: str) -> str:
    """``"1 generated section"`` / ``"5 generated sections"`` / ``"an unknown number of
    generated sections"`` when *n* is ``None`` -- shared by :data:`DELIVERED_GUIDE_NOTE_TEMPLATE`
    and the author-page templates below so a row/section count is always stated the same way."""
    if n is None:
        return f"an unknown number of {unit}s"
    return f"{n} {unit}" if n == 1 else f"{n} {unit}s"


#: The Review sheet judges the tool's internal verification decision; the Delivered
#: sheet (present only when ``--delivered-evidence`` is given) judges the final product a user
#: actually receives, but only the part of it the writer both cited and verified -- one row per
#: entry of the saved run's own ``final_report.verifications``
#: (``demo/run_demo.py::build_delivered_evidence_rows``, which raises on any entry whose own
#: status is not ``"verified"``), in the order it appears, next to its citation, the cited
#: paper, the evidence quotes the verifier found and the source passage those quotes came from.
#: A generated sentence with no citation at all, or a cited
#: sentence whose verification did not reach ``"verified"``, gets no row here and is not judged
#: on this sheet. A single demo run can deliver more than one generated
#: section into this sheet, so the note states how many sections as well as how many rows.
#: `_finalize_paragraph_text` in
#: `backend/app/services/fulltext.py` removes only an uncited entry tagged "finding" that was
#: not left ``"unclassified"``, and a cited sentence whose verification did not reach
#: ``"verified"``; a ``"framing"``-tagged uncited sentence is never removed and is delivered
#: along with the cited and verified ones, so a section's own saved draft can, and on the
#: promoted run does, carry sentences this sheet never rows and never judges. This note names
#: the two removal rules rather than claiming they are exhaustive.
#: A promoted run's own `_drop_dangling_framing_sentences`
#: (`backend/app/services/fulltext.py`) removes a THIRD kind of sentence, an uncited framing
#: sentence left dangling by one of the first two removals (an enumeration opener with too few
#: fragments left to fulfil it, or a discourse connective whose antecedent was just removed),
#: and increments the same `finalize_stats["sentences_removed_dangling"]` counter doing it. The
#: note names all three removal rules and says "remaining" framing sentences are delivered, not
#: "own".
#: Kept as a tuple of thirteen already-short cells (rather than one long string that would be
#: re-split at the wrapper's own boundaries) so each cell prints as its own Guide row, in the
#: fixed order the Delivered sheet's own layout uses. Both the full-export and delivered-only
#: paths format this same tuple, so the two Delivered Guide texts cannot drift apart; the
#: full-export path alone appends one further cell (see :func:`guide_paragraphs`) naming the
#: Review sheet's separate role.
def _passage_phrase(n: int | None) -> str:
    if n is None:
        return "an unknown number of"
    return f"{n:,}"


DELIVERED_GUIDE_CELLS: tuple[str, ...] = (
    "The Delivered sheet carries the final product this tool hands to a user, but only the "
    "cited and verified part of it. One row is one claim the tool checked: one proposition "
    "taken from one delivered sentence and checked against one cited paper, with the passage "
    "it was checked against stored on the same row. A sentence that carried two citations, or "
    "that the tool checked as more than one proposition, has one row for each. Two rows of the "
    "same sentence can carry overlapping claims, so judge each row exactly as it is printed, "
    "and do not assume two such rows must get the same answer. This copy has {n_rows_phrase} "
    "drawn from {n_sections_phrase} of one demo run, and those rows come from "
    "{n_sentences_phrase}. The two questions you answer are defined in the numbered items "
    "below, and those numbered definitions are the only ones that apply.",
    "Judge each row on its own, from what that row shows: not from other rows, and not from "
    "anything you know about these papers outside the sheet. Rows are grouped by section, in "
    "the order the sections were generated, and within each section in the order the sentence "
    "appears there. The row column restarts at 1 in every section, so it identifies a row only "
    "within its own section, never across the whole sheet. Name a row by its spreadsheet row "
    "number.",
    "A sentence the report generated with no citation at all, and a cited sentence whose "
    "verification did not reach a verified status, get no row here and are not judged on this "
    "sheet: finalize removes an uncited sentence it classifies as an empirical finding, a "
    "cited sentence whose verification did not reach a verified status, and an uncited framing "
    "sentence left dangling by one of those removals. A section's remaining framing sentences "
    "are delivered along with the cited and verified ones and are not judged on this sheet.",
    "Each row shows the delivered sentence, the section it came from (section_title), the one "
    "proposition of that sentence this row is about (claim_checked), the citation as the "
    "report printed it (citation_text), the cited paper (paper_title, paper_doi, and "
    "paper_authors, which lists that paper's authors in the order the publisher records them, "
    "first author first, followed by the year), the quotes the verifier found "
    "(evidence_quotes, one quote to a numbered line), and the passage of the cited paper those "
    "quotes were taken from (source_passage). Read claim_checked first: it names the part of "
    "the sentence this row asks about. paper_authors tells you whose paper the passage "
    "belongs to, so you can check that citation_text names that paper.",
    "source_passage can be much longer than the row shows on screen. Click the cell and read "
    "it in the formula bar, or widen the row, before you answer: the longest passage in this "
    "copy runs to {longest_passage_phrase} characters. source_passage is also a window on the "
    "paper rather than the whole of it, so a quote that the tool did locate can still fall "
    "outside the window shown on the row.",
    "Five columns are filled in by the tool itself, not by you. source_located is no when the "
    "chunk the verifier named as evidence could not be fetched or parsed at all. It does not "
    "mean the evidence passage was found. passage_located is the column that answers that: it "
    "is no when the fetch succeeded but one or more of evidence_quotes could not be found word "
    "for word anywhere in the paper's stored text, so source_passage may be empty, or may show "
    "only some of evidence_quotes. The quotes that were not found are named in "
    "unlocated_quotes. sentence_in_draft is no when this row's own sentence no longer matches "
    "the sentence actually delivered in the draft, which a later automatic edit can rewrite in "
    "place after this record was built.",
    "Answer both human columns on every row regardless of what those five say. A row where "
    "source_located, passage_located or sentence_in_draft is no is excluded from the headline "
    "rates and reported separately, because it is not a full, source-verified reading of the "
    "claim the user received. The fifth tool column, location_section_mismatch, is yes when "
    "the fetched chunk's own section disagreed with the section the verifier recorded. That is "
    "one cheap sign the passage shown may not be from the chunk the claim was verified "
    "against. Such a row is listed separately but is not excluded from the headline.",
    "1. human_sentence_correct: read claim_checked against source_passage, the passage of the "
    "cited paper's own text shown on this row, and answer yes when that passage supports the "
    "claim as written. Judge claim_checked only. Any other part of the sentence belongs to "
    "another row. claim_checked is quoted from the sentence, so read it inside the sentence "
    "shown on this row. Use the sentence and citation_text to fill in a missing subject or a "
    "back reference inside the claim, such as this effect, this comparison or their design. "
    "You still judge only what claim_checked itself asserts.",
    "Answer no when the claim overstates the passage, generalises beyond it, drops a "
    "qualifier the passage stated, changes a number, a sample or a population, is supported "
    "by the passage in part only, or is not supported by it at all. Partial support is no, "
    "not yes. Also answer no when citation_text does not name the paper shown in "
    "paper_authors, that is when the author names or the year in the citation do not belong "
    "to that paper.",
    "Compare names as whole words. Every author surname printed in citation_text must appear "
    "as a whole word among the names in paper_authors, in the order paper_authors lists them, "
    "and the year in citation_text must equal the year in paper_authors. A different "
    "spelling, a missing or added accent, or a different transliteration of the same name is "
    "not a mismatch.",
    "Words of the sentence that sit outside claim_checked are not judged on this row, and "
    "that includes connective and evaluative framing such as similarly, nevertheless, a "
    "parallel pattern, an earlier study, and the strongest direct evidence comes from. Ignore "
    "such words unless the passage and paper_authors on this row show the framing to be false "
    "of this paper. When claim_checked opens with a connective (however, yet, although, "
    "whereas, nevertheless, while), ignore the connective word itself and judge every "
    "assertion the claim then makes, including a summarising one such as Uptake was "
    "selective.",
    "2. human_quote_supports: answer yes when the quotes listed in evidence_quotes on this "
    "row, which were taken from source_passage, are enough on their own to support "
    "claim_checked, without the rest of source_passage and without the rest of the sentence. "
    "Each quote is on its own numbered line, so a semicolon inside a quote is not a boundary "
    "between quotes. Answer no otherwise, for example when the quotes report a different "
    "result, carry none of the numbers the claim states, or cover part of the claim only. "
    "This question is about the quotes alone, so it can be no on a row where "
    "human_sentence_correct is yes.",
    "human_note is required whenever human_sentence_correct or human_quote_supports is no. "
    "Give the reason in one sentence. Name the part of claim_checked that failed when only "
    "part of it did. When the citation does not name the paper on the row, say that instead. "
    "It is optional only when both answers are yes. Both columns must carry yes or no on "
    "every row of the sheet.",
)
#: The full-export path alone appends this after :data:`DELIVERED_GUIDE_CELLS`, since only the
#: full export also carries a Review sheet for this sentence to contrast the Delivered sheet
#: against.
DELIVERED_GUIDE_REVIEW_CONTRAST_CELL = (
    "The Delivered sheet judges the final product, not the tool's own internal verification "
    "decision, which the Review sheet covers separately."
)


def guide_paragraphs(
    *, research_question: str = "", criteria_text: str = "", has_screening: bool = False,
    n_constructed: int | None = None, n_real: int | None = None,
    full_text_stratum_possible: bool = False,
    has_delivered: bool = False, n_delivered: int | None = None,
    n_delivered_sections: int | None = None,
    delivered_only: bool = False,
    n_delivered_sentences: int | None = None,
    longest_passage_chars: int | None = None,
) -> list[str]:
    if delivered_only:
        sections = [
            cell.format(
                n_rows_phrase=_count_phrase(n_delivered, "row"),
                n_sections_phrase=_count_phrase(n_delivered_sections, "generated section"),
                n_sentences_phrase=_count_phrase(n_delivered_sentences, "distinct sentence"),
                longest_passage_phrase=_passage_phrase(longest_passage_chars),
            )
            for cell in DELIVERED_GUIDE_CELLS
        ]
        return [line for section in sections for line in _wrap_paragraph(section, 900)]
    sections = [
        GUIDE_OPENING_SENTENCE,
        "The four human_verdict values, copied word for word from the annotation rubric "
        "(INSTRUCTIONS_v3.md) the Model_labels sheet was itself produced under:",
        *STATUS_DEFINITIONS_V3,
        NO_FULL_TEXT_SOURCES_NOTE,
    ]
    if n_constructed is not None and n_real is not None:
        sections.append(
            REVIEW_CENSUS_SENTENCE_TEMPLATE.format(n_constructed=n_constructed, n_real=n_real)
        )
    sections += [
        "This workbook has two decisions to make, per row: do you agree with the tool, and if "
        "not, why. On the Review sheet, read the claim, the cited paper (cited_title / "
        "cited_doi -- look this row's own item_id up on the Sources sheet, or the paper by DOI "
        "outside the workbook; a withheld row has no Sources rows of its own), the evidence "
        "quotes, and the tool's verifier_status. Then fill in human_verdict with your own "
        "judgement "
        "(never leave it blank) and human_agrees_with_verifier with yes or no. This is the "
        "confirm-or-override decision: yes means you confirm the tool's verifier_status is "
        "correct (your human_verdict should then match it); no means you override it, and your "
        "human_verdict is your own corrected status.",
        "When you set human_agrees_with_verifier to no, also fill in disagreement_axis (one "
        "of: " + ", ".join(DISAGREEMENT_AXIS_CHOICES) + "), human_quote (one sentence copied "
        "from the cited paper) and human_justification (one or two sentences of reasoning). "
        "Leave these blank when you agree.",
        "machine_reasons and diagnostics are automatic checks the tool ran on its own verdict, "
        "for reference only. diagnostics in particular are notes that never change the tool's "
        "verifier_status by themselves -- they explain the verdict, they do not adjust it.",
        QUOTE_RELOCATED_NOTE,
        SECOND_PASS_NOTE,
        "model_status is the tool's answer before its automatic guards ran; verifier_status is "
        "the answer after them. run_B_status is filled in only when a second, independent run "
        "of the tool reached a different status on the same item -- an empty cell there means "
        "the two runs agreed.",
        "A second sheet, Model_labels, carries a large-language-model-adjudicated annotation "
        "label and its rationale for a subset of items, keyed by the same item_id. Decide your "
        "own human_verdict before reading it -- it is there so the two judgements can be "
        "compared afterwards, not so you can defer to it.",
        "countersigned_by and countersigned_at are for whoever finalises this workbook; leave "
        "them blank unless you are doing that.",
    ]
    if has_delivered:
        for cell in DELIVERED_GUIDE_CELLS:
            sections.append(
                cell.format(
                    n_rows_phrase=_count_phrase(n_delivered, "row"),
                    n_sections_phrase=_count_phrase(n_delivered_sections, "generated section"),
                    n_sentences_phrase=_count_phrase(n_delivered_sentences, "distinct sentence"),
                    longest_passage_phrase=_passage_phrase(longest_passage_chars),
                )
            )
        sections.append(DELIVERED_GUIDE_REVIEW_CONTRAST_CELL)
    if has_screening:
        sections.append(
            "The Screening sheet samples records from a demo relevance-screening run. Its "
            "research question was: " + (research_question or "(not recorded)") + ". Its "
            "numbered inclusion/exclusion criteria: " + (criteria_text or "(not recorded)") + "."
        )
        sections.append(
            "For each sampled record, read title, abstract_shown and the identifying columns "
            "(doi/year/journal/openalex_id) against the criteria above and the tool's own "
            "status/criterion/quote/reason, then fill in human_status (one of: "
            + ", ".join(SCREENING_HUMAN_STATUS_CHOICES) + ", never blank), human_agrees (yes "
            "or no), quote_is_verbatim (yes or no -- does quote actually appear, verbatim, in "
            "this row's own shown text, that is the title column together with "
            "abstract_shown, ignoring differences of case, whitespace, and dash or quote "
            "style; a quote taken from the title is allowed and still counts as verbatim; "
            "required whenever the row carries a quote), criterion_is_right "
            "(yes or no -- does the named criterion actually cover the stated ground; "
            "required whenever the row carries a criterion), and human_note (required "
            "whenever human_agrees is no)."
        )
        sections.append(
            "A row whose abstract_shown reads \"(no abstract)\" means the paper had no "
            "abstract at all when the screener decided it, from the title alone. On a "
            "needs_review row that is the guard routing the gap to you to confirm; on an "
            "included, excluded_numbered or excluded_off_topic row it is the model's own "
            "status/criterion decision, made from the title alone rather than an abstract. "
            "Either way you are judging the row on the title and reason shown, not on "
            "abstract evidence the sheet never had to give you. The absence of abstract "
            "text is not on its own a reason to mark a row exclude, or to disagree with a "
            "needs_review routing. On an included row decided from the title alone, "
            "judging that the title-only evidence is not enough to support an include is a "
            "legitimate disagreement in its own right -- record it as you would any other, "
            "not withheld by the rule above."
        )
        sections.append(ABSTRACT_CUT_NOTE)
        sections.append(UNANCHORED_EXCLUDE_NOTE)
        census_text = (
            FULL_TEXT_CENSUS_LIVE_TEXT if full_text_stratum_possible
            else FULL_TEXT_CENSUS_STRUCTURAL_TEXT
        )
        sections.append(
            census_text + " A hidden stratum column records which stratum each row was drawn "
            "from, for the analysis only."
        )
    return [line for section in sections for line in _wrap_paragraph(section, 900)]


WORKBOOK_ID_PREFIX = "Workbook id: "
SCREENING_SEED_PREFIX = "Screening sample seed: "


def write_guide_sheet(
    ws: Any, *, workbook_id: str, research_question: str = "", criteria_text: str = "",
    has_screening: bool = False, screening_seed: int | None = None,
    demo_run_id: str = "n/a", export_time: str = "",
    n_constructed: int | None = None, n_real: int | None = None,
    full_text_stratum_possible: bool = False,
    has_delivered: bool = False, n_delivered: int | None = None,
    n_delivered_sections: int | None = None,
    delivered_only: bool = False,
    n_delivered_sentences: int | None = None,
    longest_passage_chars: int | None = None,
) -> None:
    ws.title = "Guide"
    ws.column_dimensions["A"].width = 120
    ws.freeze_panes = None
    row = 0
    for row, paragraph in enumerate(
        guide_paragraphs(
            research_question=research_question, criteria_text=criteria_text,
            has_screening=has_screening, n_constructed=n_constructed, n_real=n_real,
            full_text_stratum_possible=full_text_stratum_possible,
            has_delivered=has_delivered, n_delivered=n_delivered,
            n_delivered_sections=n_delivered_sections,
            delivered_only=delivered_only,
            n_delivered_sentences=n_delivered_sentences,
            longest_passage_chars=longest_passage_chars,
        ),
        start=1,
    ):
        cell = _set_cell(ws, row, 1, paragraph)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    # Printed next to the workbook id so a returned copy
    # identifies its own export, rather than only the key file (which the reviewer never sees)
    # knowing which run it came from. The delivered-only path carries no Review sheet and no
    # key-out binding worth stating (--key-out is optional in that mode and the scorer only
    # requires --key when a Review sheet is present), so its id line says nothing about either.
    if delivered_only:
        id_text = (
            f"{WORKBOOK_ID_PREFIX}{workbook_id} (exported {export_time or '(not recorded)'} "
            f"against demo run {demo_run_id})."
        )
    else:
        id_text = (
            f"{WORKBOOK_ID_PREFIX}{workbook_id} (must match --key-out's own workbook_id "
            "column; score_review_sheet.py refuses to score a mismatched pair). Exported "
            f"{export_time or '(not recorded)'} against demo run {demo_run_id}."
        )
    id_cell = _set_cell(ws, row + 1, 1, id_text)
    id_cell.alignment = Alignment(wrap_text=True, vertical="top")
    if has_screening and screening_seed is not None:
        seed_cell = _set_cell(ws, row + 2, 1, f"{SCREENING_SEED_PREFIX}{screening_seed}")
        seed_cell.alignment = Alignment(wrap_text=True, vertical="top")


# --------------------------------------------------------------------------------------
# Sheet writers
# --------------------------------------------------------------------------------------


def _write_plain_sheet(
    wb: Any, name: str, rows: Sequence[Mapping[str, Any]], header: list[str],
    widths: Mapping[str, float],
) -> Any:
    ws = wb.create_sheet(name)
    _append_row(ws, header)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.freeze_panes = "A2"
    for row in rows:
        _append_row(ws, [row.get(col, "") for col in header])
    for r in range(2, ws.max_row + 1):
        for c in range(1, len(header) + 1):
            ws.cell(row=r, column=c).alignment = Alignment(wrap_text=True, vertical="top")
    for i, col in enumerate(header, start=1):
        ws.column_dimensions[get_column_letter(i)].width = widths.get(col, 22)
    return ws


def _add_list_validation(
    ws: Any, header: list[str], column: str, choices: Sequence[str],
) -> None:
    last_row = max(ws.max_row, 2)
    col_letter = get_column_letter(header.index(column) + 1)
    dv = DataValidation(
        type="list", formula1=f'"{",".join(choices)}"', allow_blank=True,
        showErrorMessage=True, errorTitle=f"Invalid {column}",
        error=f"{column} must be one of: {', '.join(choices)}.",
    )
    dv.add(f"{col_letter}2:{col_letter}{last_row}")
    ws.add_data_validation(dv)


def write_review_sheet(wb: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    header = review_header()
    ws = _write_plain_sheet(wb, "Review", rows, header, REVIEW_COLUMN_WIDTHS)
    _add_list_validation(ws, header, "human_verdict", HUMAN_VERDICT_CHOICES)
    _add_list_validation(ws, header, "human_agrees_with_verifier", YES_NO_CHOICES)
    _add_list_validation(ws, header, "disagreement_axis", DISAGREEMENT_AXIS_CHOICES)


def write_model_labels_sheet(wb: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    header = model_labels_header()
    _write_plain_sheet(
        wb, "Model_labels", rows, header,
        {"item_id": 10, "model_annotation_label": 20, "rationale": 70, "annotator_agreement": 16},
    )


def build_source_rows(
    items: Sequence[Mapping[str, Any]], caches: Mapping[str, Mapping[str, Any]],
    opaque_ids: Mapping[str, str], *, max_chars: int,
) -> list[tuple[str, int, str]]:
    """One row per (item, text part), keyed on the citing item's own opaque ``item_id`` -- the
    same id the Review sheet's own ``item_id`` column carries -- not the cited paper's DOI
    (keying by DOI put a ``no_full_text`` item's full text one
    lookup away, findable under a *different* item's own row on the Sources sheet even though
    that item's own row carried no ``chunk_doi``). An item with no ``chunk_doi`` contributes
    nothing, under any key. The same underlying paper's text is duplicated once per item that
    cites it (no dedup by DOI), so a Review row's own ``item_id`` on this sheet holds exactly
    that row's own parts."""
    out: list[tuple[str, int, str]] = []
    for item in items:
        doi = item.get("chunk_doi")
        if not doi:
            continue
        cache = caches.get(item.get("_cache_dir"), {})
        text = full_text_for_doi(doi, cache)
        if not text:
            continue
        opaque_id = opaque_ids[item["item_id"]]
        for i, part in enumerate(split_into_parts(text, max_chars), start=1):
            out.append((opaque_id, i, part))
    return out


def write_sources_sheet(wb: Any, source_rows: Sequence[tuple[str, int, str]]) -> None:
    ws = wb.create_sheet("Sources")
    _append_row(ws, ["item_id", "part", "text"])
    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.freeze_panes = "A2"
    for item_id, part, text in source_rows:
        _append_row(ws, [item_id, part, text])
    ws.column_dimensions["A"].width = 12
    ws.column_dimensions["B"].width = 8
    ws.column_dimensions["C"].width = 120
    for r in range(2, ws.max_row + 1):
        ws.cell(row=r, column=3).alignment = Alignment(wrap_text=True, vertical="top")


def write_key_csv(
    path: Path, items: Sequence[Mapping[str, Any]], opaque_ids: Mapping[str, str], *,
    workbook_id: str,
) -> None:
    fieldnames = ["opaque_id", "real_item_id", "set", "construction_category",
                  "expected_label", "workbook_id"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow(build_key_row(item, opaque_ids[item["item_id"]], workbook_id))


def write_delivered_only_key_csv(path: Path, *, workbook_id: str) -> None:
    """The delivered-only path's own key file: the same 6-name header as :func:`write_key_csv`,
    plus exactly one data row carrying the workbook id, with the five item-identifying fields
    left blank -- there is no item list in this mode to bind them to. The scorer requires
    ``--key`` only when a ``Review`` sheet is present, so this file is never joined against
    anything and the blank fields cannot leak or mismatch; a reader who does open it can still
    recover the workbook id (``score_review_sheet.load_key_workbook_id``) from a file that is
    not header-only, unlike v3's."""
    fieldnames = ["opaque_id", "real_item_id", "set", "construction_category",
                  "expected_label", "workbook_id"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "opaque_id": "", "real_item_id": "", "set": "", "construction_category": "",
            "expected_label": "", "workbook_id": workbook_id,
        })


# --------------------------------------------------------------------------------------
# Screening sheet
# --------------------------------------------------------------------------------------


def load_screening_record(path: Path) -> dict[str, Any]:
    data = read_json(path, None)
    if not isinstance(data, dict):
        raise ExportError(f"--screening-record {path} not found or not valid json")
    return data


def screening_prompt_version(record: Mapping[str, Any]) -> str | None:
    return ((record.get("provenance") or {}).get("screener") or {}).get("prompt_version")


def check_screening_prompt_version(record: Mapping[str, Any], expected: str) -> None:
    """Raises :class:`ExportError` unless the screening record's own
    ``provenance.screener.prompt_version`` equals *expected* exactly -- see the module
    docstring; this is the blocking check the acceptance report requires."""
    actual = screening_prompt_version(record)
    if actual != expected:
        raise ExportError(
            f"screening record's provenance.screener.prompt_version is {actual!r}, expected "
            f"{expected!r} (--screening-prompt-version-expect); refusing to sample a screening "
            "record produced by a different screener build"
        )


def bucket_by(records: Sequence[Mapping[str, Any]], key_fn) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = {}
    for r in records:
        k = str(key_fn(r) or "unspecified")
        out.setdefault(k, []).append(r)
    return out


def allocate_stratified_with_floor(
    bucket_sizes: Mapping[str, int], *, total_n: int, floor: int, min_bucket: int,
) -> dict[str, int]:
    """How many records to draw from each bucket in *bucket_sizes* (``{key: size}``, in the
    caller's own iteration order -- that order is the tie-break every step below uses).

    This replaced an earlier proportional-with-floor allocation
    (which could never trim a bucket smaller than *min_bucket*, and so could overshoot
    *total_n*, or leave the largest bucket at zero once it was the only one left to trim) with
    this three-step rule:

    1. Each bucket starts at ``min(floor, size)`` when it holds at least *min_bucket* records;
       a smaller bucket is taken whole instead (never padded above its own size).
    2. If that total exceeds *total_n*, reduce one record at a time from whichever bucket
       currently holds the largest allocation (ties broken by the caller's own bucket order,
       earliest first), never below 1 for a non-empty bucket -- unless *total_n* itself is
       smaller than the number of non-empty buckets, in which case giving every one of them at
       least 1 is not possible and some are reduced to 0, largest-allocation-first exactly as
       above.
    3. If the total is still below *total_n* (most buckets taken whole under *min_bucket*, or a
       small floor), add one record at a time to whichever bucket has the largest remaining
       unallocated size (``size - allocation``), same tie-break, until every bucket is either
       exhausted or the total reaches *total_n*.

    The result always totals ``min(total_n, sum(bucket_sizes))``, and step 2's floor of 1 means
    a bucket that holds any records at all keeps at least one unless *total_n* cannot even give
    one to every non-empty bucket."""
    order = list(bucket_sizes)
    if not order or total_n <= 0:
        return dict.fromkeys(order, 0)
    order_index = {k: i for i, k in enumerate(order)}

    # Step 1: floor or whole-bucket baseline.
    alloc = {
        k: (min(floor, size) if size >= min_bucket else size)
        for k, size in bucket_sizes.items()
    }
    total_alloc = sum(alloc.values())

    non_empty = [k for k in order if bucket_sizes[k] > 0]
    # A non-empty bucket is never reduced below 1 unless total_n can't even give one to every
    # non-empty bucket -- then the floor for reduction relaxes to 0.
    reduction_floor = 1 if total_n >= len(non_empty) else 0

    # Step 2: trim the overshoot, largest current allocation first.
    guard = 0
    while total_alloc > total_n and guard < 100_000:
        candidates = [k for k in order if alloc[k] > reduction_floor]
        if not candidates:
            break
        k = max(candidates, key=lambda k: (alloc[k], -order_index[k]))
        alloc[k] -= 1
        total_alloc -= 1
        guard += 1

    # Step 3: fill the shortfall, largest remaining unallocated size first.
    guard = 0
    while total_alloc < total_n and guard < 100_000:
        candidates = [k for k in order if alloc[k] < bucket_sizes[k]]
        if not candidates:
            break
        k = max(candidates, key=lambda k: (bucket_sizes[k] - alloc[k], -order_index[k]))
        alloc[k] += 1
        total_alloc += 1
        guard += 1

    return alloc


def sample_bucketed(
    buckets: Mapping[str, Sequence[Mapping[str, Any]]], alloc: Mapping[str, int], *, seed: int,
) -> list[Mapping[str, Any]]:
    rng = random.Random(seed)
    out: list[Mapping[str, Any]] = []
    for k in sorted(buckets):
        pool = list(buckets[k])
        take = min(alloc.get(k, 0), len(pool))
        out.extend(pool if take >= len(pool) else rng.sample(pool, take))
    return out


def sample_uniform(
    records: Sequence[Mapping[str, Any]], n: int, *, seed: int,
) -> list[Mapping[str, Any]]:
    if len(records) <= n:
        return list(records)
    return random.Random(seed).sample(list(records), n)


def full_text_criterion_ids(criteria: Mapping[str, Any]) -> set[str]:
    """The criterion ids (``"I1"``, ``"E3"``, ...) the record's own
    ``criteria.inclusion_criteria_stages``/``exclusion_criteria_stages`` mark ``"full_text"``.
    Recomputed here from the record's own carried stages -- mirrors
    ``backend/app/agents/relevance_screener_agent.py``'s ``full_text_inclusion_criterion_ids``/
    ``full_text_exclusion_criterion_ids`` without a backend import. A v1-shaped record (see
    the module docstring) carries neither stages list, so this returns the empty set for one
    -- correct, since v1 has no full-text-deferred criteria concept at all."""
    ids: set[str] = set()
    for i, stage in enumerate(criteria.get("inclusion_criteria_stages") or [], start=1):
        if stage == "full_text":
            ids.add(f"I{i}")
    for i, stage in enumerate(criteria.get("exclusion_criteria_stages") or [], start=1):
        if stage == "full_text":
            ids.add(f"E{i}")
    return ids


def classify_excluded_llm_census(
    record: Mapping[str, Any], full_text_ids: set[str],
) -> str | None:
    """One of :data:`CENSUS_STRATA`'s three exclusion labels for a ``stage: llm`` excluded
    *record*, or ``None`` for a normal exclusion the numbered/off-topic samples draw from.
    Checked in this order because a record excluded with no abstract at all is blamed for
    that alone -- a criterion or quote naming ground the model was never shown is moot once
    the abstract itself is missing, so it is not also counted against the anchor or full-text
    invariants."""
    if not str(record.get("abstract") or "").strip():
        return "excluded_no_abstract"
    if not str(record.get("quote") or "").strip() or not str(record.get("criterion") or "").strip():
        return "excluded_missing_anchor"
    if str(record.get("criterion") or "") in full_text_ids:
        return "excluded_full_text_violation"
    return None


def _tag_stratum(records: Sequence[Mapping[str, Any]], stratum: str) -> list[dict[str, Any]]:
    out = []
    for r in records:
        row = dict(r)
        row["_stratum"] = stratum
        out.append(row)
    return out


def build_screening_sample(
    records: Sequence[Mapping[str, Any]], *, seed: int, included_n: int, needs_review_n: int,
    excluded_numbered_n: int, excluded_off_topic_n: int, floor: int, min_bucket: int,
    criteria: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """The full sampling design. A random sample of *included_n*
    ``included`` records; a ``needs_review_reason``-stratified sample of *needs_review_n*
    ``needs_review`` records (see :func:`allocate_stratified_with_floor`); within the
    ``stage: llm`` exclusions only (a ``stage: wos`` exclusion is the venue filter, evaluated
    separately, and carries no LLM decision for a human to agree with) -- a plain uniform
    random sample of *excluded_numbered_n* records excluded on a numbered criterion (not a
    stratified one: stratifying by ``criterion`` let a criterion holding most of the exclusions
    contribute a small minority of the sample, with no per-criterion weight recorded anywhere
    to reweight by after the human round), and a separate uniform random sample of
    *excluded_off_topic_n* records excluded on the reserved off-topic id
    (:data:`TOPIC_CRITERION_ID`). Plus four census strata
    (:data:`CENSUS_STRATA`), appended whole and never counted against the samples above: three
    guard-invariant checks on the ``stage: llm`` exclusions
    (:func:`classify_excluded_llm_census`), each expected to be empty, plus every
    ``unscreened`` record. Returns ``(rows, stratum_counts)``: every row carries its own
    ``_stratum`` (one of :data:`SCREENING_STRATA`), and *stratum_counts* is
    ``{stratum: {"drawn": n, "available": n}}`` for all eight, so the export can record what
    was and was not sampled."""
    included = [r for r in records if r.get("outcome") == "included"]
    needs_review = [r for r in records if r.get("outcome") == "needs_review"]
    excluded_llm = [
        r for r in records if r.get("outcome") == "excluded" and r.get("stage") == STAGE_LLM
    ]
    unscreened = [r for r in records if r.get("outcome") == "unscreened"]

    full_text_ids = full_text_criterion_ids(criteria or {})
    census: dict[str, list[Mapping[str, Any]]] = {k: [] for k in CENSUS_STRATA[:-1]}
    normal_excluded: list[Mapping[str, Any]] = []
    for r in excluded_llm:
        label = classify_excluded_llm_census(r, full_text_ids)
        (census[label] if label else normal_excluded).append(r)

    off_topic_pool = [r for r in normal_excluded if r.get("criterion") == TOPIC_CRITERION_ID]
    numbered_pool = [r for r in normal_excluded if r.get("criterion") != TOPIC_CRITERION_ID]

    counts: dict[str, dict[str, int]] = {}
    rows: list[dict[str, Any]] = []

    def _draw_uniform(pool: Sequence[Mapping[str, Any]], n: int, stratum: str) -> None:
        drawn = sample_uniform(pool, n, seed=seed)
        counts[stratum] = {"drawn": len(drawn), "available": len(pool)}
        rows.extend(_tag_stratum(drawn, stratum))

    _draw_uniform(included, included_n, "included")

    nr_buckets = bucket_by(needs_review, lambda r: r.get("needs_review_reason"))
    nr_alloc = allocate_stratified_with_floor(
        {k: len(v) for k, v in nr_buckets.items()}, total_n=needs_review_n, floor=floor,
        min_bucket=min_bucket,
    )
    nr_sample = sample_bucketed(nr_buckets, nr_alloc, seed=seed)
    counts["needs_review"] = {"drawn": len(nr_sample), "available": len(needs_review)}
    rows.extend(_tag_stratum(nr_sample, "needs_review"))

    _draw_uniform(numbered_pool, excluded_numbered_n, "excluded_numbered")
    _draw_uniform(off_topic_pool, excluded_off_topic_n, "excluded_off_topic")

    for stratum in CENSUS_STRATA[:-1]:
        pool = census[stratum]
        counts[stratum] = {"drawn": len(pool), "available": len(pool)}
        rows.extend(_tag_stratum(pool, stratum))

    counts["unscreened"] = {"drawn": len(unscreened), "available": len(unscreened)}
    rows.extend(_tag_stratum(unscreened, "unscreened"))

    return rows, counts


def shuffle_records(records: Sequence[Mapping[str, Any]], seed: int) -> list[Mapping[str, Any]]:
    """Reshuffles the sampled *records* so the sheet is not sorted by outcome (a reviewer who
    sees forty consecutive excludes anchors on excluding)."""
    order = list(range(len(records)))
    random.Random(seed + 1).shuffle(order)
    return [records[i] for i in order]


def match_screening_key(record: Mapping[str, Any]) -> str:
    """The join key used to line up the same paper across two demo screening runs: the
    normalised DOI when present, else the OpenAlex id, else the title -- a demo run has no
    other stable identifier."""
    doi = str(record.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    openalex_id = str(record.get("openalex_id") or "").strip()
    if openalex_id:
        return f"openalex:{openalex_id}"
    return f"title:{str(record.get('title') or '').strip().lower()}"


def screening_run_b_status(
    record: Mapping[str, Any], records_b_by_key: Mapping[str, Mapping[str, Any]],
) -> str:
    other = records_b_by_key.get(match_screening_key(record))
    if not other:
        return ""
    status_a = screening_status_value(record)
    status_b = screening_status_value(other)
    return status_b if status_b and status_b != status_a else ""


def screening_header(*, has_run_b: bool) -> list[str]:
    header = [
        "record_id",
        # The export's own eight, unchanged and in the export's own order,
        # so a reader can diff the sheet against the CSV.
        "outcome", "stage", "status", "criterion", "quote", "needs_review_reason",
        "to_confirm", "reason",
        # Six identifying columns, so the human can find the paper.
        "title", "doi", "year", "journal", "journal_issn", "openalex_id",
        # Added: the exact text the model saw.
        "abstract_shown",
    ]
    if has_run_b:
        header.append("run_B_status")
    header += [
        "human_status", "human_agrees", "quote_is_verbatim", "criterion_is_right",
        "human_note",
        # Hidden (see write_screening_sheet): for the analysis only, never for the reviewer.
        "stratum",
    ]
    return header


def is_no_abstract_exempt(record: Mapping[str, Any]) -> bool:
    """``True`` for any :data:`EVIDENCE_STRATA` *record* (identified by its own ``_stratum``,
    set by :func:`build_screening_sample` before this is ever called) whose ``abstract`` field
    is blank -- the screener made its decision on that row from the title alone, not a build
    defect that silently dropped the text. ``False`` for a record not in one of the four
    evidence strata at all (a census or ``unscreened`` row): :func:`check_screening_abstracts`
    never checks those either way.

    Originally this was ``True`` only for a ``needs_review``
    record, since at the time only a routing decision could ever lack an abstract; ``included``,
    ``excluded_numbered`` and ``excluded_off_topic`` were never exempted, so a blank cell on any
    of those was always a build defect. A demo run overturned that assumption:
    ``demo/README.md``
    records the screener returning an ``included`` decision from the title alone for 7 of 143
    included records with no abstract at all. A blank abstract on any of the four evidence
    strata is therefore a real, reviewed screener outcome now, not exclusively a defect."""
    return (
        record.get("_stratum") in EVIDENCE_STRATA
        and not str(record.get("abstract") or "").strip()
    )


def render_abstract_for_screening(
    abstract: str | None, limit: int = SCREENER_ABSTRACT_CHAR_LIMIT,
) -> str:
    """Mirrors ``backend/app/agents/relevance_screener_agent.py::render_abstract``'s v2 branch
    without importing it (see the module docstring's interpreter note): the abstract shown
    whole when it is at or under *limit* characters, otherwise a head,
    :data:`SCREENER_ABSTRACT_CUT_MARKER`, and a tail, split 7:3 of *limit* the same way the
    backend function does. A sampled row's ``abstract_shown``
    cell must show the human what the model actually saw, not the record's whole stored
    abstract -- a 24,685-character abstract used to appear on the sheet whole while the
    screener's own prompt showed the model a 10,007-character head-and-tail render of it.

    Strips *abstract* before measuring its length, mirroring ``build_shown_texts``'s own
    ``(paper.get("abstract") or "").strip()`` -- run before it ever calls ``render_abstract``
    -- so a stored abstract carrying leading or trailing whitespace cannot push the length
    test to the wrong side of *limit*, or shift every head and tail offset once it is over."""
    abstract = (abstract or "").strip()
    if len(abstract) <= limit:
        return abstract
    head_chars = round(limit * 7 / 10)
    tail_chars = limit - head_chars
    head = abstract[:head_chars]
    tail = abstract[-tail_chars:] if tail_chars > 0 else ""
    return f"{head}{SCREENER_ABSTRACT_CUT_MARKER}{tail}"


def build_screening_row(
    record: Mapping[str, Any], record_id: str, *, max_cell_chars: int,
    records_b_by_key: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    no_abstract_exempt = is_no_abstract_exempt(record)
    abstract_shown = (
        NO_ABSTRACT_PLACEHOLDER if no_abstract_exempt
        else cap_text(
            render_abstract_for_screening(str(record.get("abstract") or "")), max_cell_chars,
        )
    )
    out: dict[str, Any] = {
        "record_id": record_id,
        "outcome": str(record.get("outcome") or ""),
        "stage": str(record.get("stage") or ""),
        "status": screening_status_value(record),
        "criterion": str(record.get("criterion") or ""),
        "quote": cap_text(str(record.get("quote") or ""), max_cell_chars),
        "needs_review_reason": str(record.get("needs_review_reason") or ""),
        "to_confirm": str(record.get("to_confirm") or ""),
        "reason": cap_text(str(record.get("reason") or ""), max_cell_chars),
        "title": cap_text(str(record.get("title") or ""), max_cell_chars),
        "doi": str(record.get("doi") or ""),
        "year": record.get("year") if record.get("year") is not None else "",
        "journal": str(record.get("journal") or ""),
        "journal_issn": str(record.get("journal_issn") or ""),
        "openalex_id": str(record.get("openalex_id") or ""),
        "abstract_shown": abstract_shown,
        "human_status": "", "human_agrees": "", "quote_is_verbatim": "",
        "criterion_is_right": "", "human_note": "",
        "stratum": str(record.get("_stratum") or ""),
        # Not a header column (see screening_header) -- read only by check_screening_abstracts,
        # so it is dropped automatically by _write_plain_sheet rather than ever reaching a cell.
        "_no_abstract_exempt": no_abstract_exempt,
    }
    if records_b_by_key is not None:
        out["run_B_status"] = screening_run_b_status(record, records_b_by_key)
    return out


def check_screening_abstracts(rows: Sequence[Mapping[str, Any]]) -> None:
    """Refuses (:class:`ExportError`), rather than silently exporting a blank evidence column,
    when a row in one of :data:`EVIDENCE_STRATA` -- a sampled row carrying an actual LLM
    screening decision -- has no ``abstract_shown`` text and is not itself exempt. Not refused:
    a blank abstract on an ``excluded_no_abstract`` census row (that is what the stratum is
    for), on an ``unscreened`` one (never shown one at all), or on any evidence row
    :func:`is_no_abstract_exempt` exempts -- such a row carries the literal placeholder
    :data:`NO_ABSTRACT_PLACEHOLDER` in ``abstract_shown`` instead, so this checks the row's own
    exemption flag rather than relying on that text alone being non-blank. Since
    :func:`is_no_abstract_exempt` is now unconditional on the four evidence strata (any blank
    abstract there is exempt), a row built by :func:`build_screening_row` can no longer trip
    this in practice; the check is kept as a defensive assertion of that same invariant, so a
    future regression in the exemption logic or in ``cap_text`` still fails loudly here rather
    than shipping a silently blank cell."""
    missing = [
        r["record_id"] for r in rows
        if r.get("stratum") in EVIDENCE_STRATA
        and not r.get("_no_abstract_exempt")
        and not str(r.get("abstract_shown") or "").strip()
    ]
    if not missing:
        return
    preview = ", ".join(missing[:10])
    more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
    raise ExportError(
        f"{len(missing)} sampled screening record(s) carry no abstract text for the human to "
        f"judge against (record_id(s): {preview}{more}); refusing to export a Screening sheet "
        "whose evidence column is blank. The record shape the exporter reads must carry the "
        "abstract the model actually saw for every included/needs_review/excluded-on-a-"
        "criterion row (see backend/app/services/screening_record.py::paper_record) before "
        "this can be sampled."
    )


#: A v1-shaped screening record (see the module docstring's Screening sheet section) carries
#: no ``status`` field at all, only the past-tense ``outcome`` (``included``/``excluded``/
#: ``needs_review``/``unscreened``); folded to the screener's own present-tense vocabulary so
#: the sheet's ``status`` column and ``human_status``'s dropdown always compare on the same
#: three words -- without this, a v1 record's ``status`` cell reads ``"included"`` while the
#: only values ``human_status`` can hold are ``include``/``needs_review``/``exclude``, so a
#: reviewer who correctly writes ``include`` is scored as wrong against every one of them.
_OUTCOME_TO_STATUS = {
    "included": "include", "excluded": "exclude", "needs_review": "needs_review",
    "unscreened": "unscreened",
}


def screening_status_value(record: Mapping[str, Any]) -> str:
    raw = record.get("status")
    if raw:
        return str(raw)
    outcome = str(record.get("outcome") or "")
    return _OUTCOME_TO_STATUS.get(outcome, outcome)


def render_numbered_criteria(criteria: Mapping[str, Any]) -> str:
    inclusion = [str(c) for c in (criteria.get("inclusion_criteria") or [])]
    exclusion = [str(c) for c in (criteria.get("exclusion_criteria") or [])]
    lines = [f"I{i}. {c}" for i, c in enumerate(inclusion, start=1)]
    lines += [f"E{i}. {c}" for i, c in enumerate(exclusion, start=1)]
    return "\n".join(lines)


def _hide_column(ws: Any, header: list[str], column: str) -> None:
    ws.column_dimensions[get_column_letter(header.index(column) + 1)].hidden = True


def write_screening_sheet(
    wb: Any, rows: Sequence[Mapping[str, Any]], *, has_run_b: bool,
) -> None:
    header = screening_header(has_run_b=has_run_b)
    ws = _write_plain_sheet(wb, "Screening", rows, header, SCREENING_COLUMN_WIDTHS)
    _add_list_validation(ws, header, "human_status", SCREENING_HUMAN_STATUS_CHOICES)
    _add_list_validation(ws, header, "human_agrees", YES_NO_CHOICES)
    _add_list_validation(ws, header, "quote_is_verbatim", YES_NO_CHOICES)
    _add_list_validation(ws, header, "criterion_is_right", YES_NO_CHOICES)
    _hide_column(ws, header, "stratum")


def retrieval_cutoff_value(criteria: Mapping[str, Any]) -> str | None:
    """The screening record's own ``criteria.publication_date_max`` -- the
    ``to_publication_date`` cap ``run_smart_search`` forwards to every OpenAlex call on the run
    behind the sheet -- or ``None`` when the record carries no such field (an older record
    shape, or a run with no cap)."""
    value = criteria.get("publication_date_max")
    return str(value) if value else None


def append_retrieval_cutoff_row(ws: Any, value: str) -> None:
    """Appends one ``retrieval_cutoff`` field/value row to the Provenance worksheet *ws*
    (already written by :func:`write_provenance_sheet`) -- see :func:`retrieval_cutoff_value`.
    Never called when that function returns ``None``: a run with no recorded cap gets no row,
    rather than a ``"None"`` or blank one."""
    _append_row(ws, ["retrieval_cutoff", value])


def append_screening_stratum_rows(
    ws: Any, stratum_counts: Mapping[str, Mapping[str, int]],
) -> None:
    """Appends one ``field, value`` row per :data:`SCREENING_STRATA` entry to the Provenance
    worksheet *ws* (already written by :func:`write_provenance_sheet`), ``"<drawn> of
    <available>"`` -- reported as counts so the human can see what was and
    was not sampled, which the seed alone (already on the Guide sheet) does not answer."""
    for stratum in SCREENING_STRATA:
        c = stratum_counts.get(stratum, {"drawn": 0, "available": 0})
        _append_row(ws, [f"screening_stratum_{stratum}", f"{c['drawn']} of {c['available']}"])


def build_screening_sheet(
    wb: Any,
    record_path: Path,
    *,
    expected_prompt_version: str,
    seed: int,
    included_n: int,
    needs_review_n: int,
    excluded_numbered_n: int,
    excluded_off_topic_n: int,
    floor: int,
    min_bucket: int,
    max_cell_chars: int,
    record_b_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Adds the ``Screening`` sheet to *wb*. Returns ``(record, rows, stratum_counts)`` --
    *record* is the loaded screening-record json (its ``criteria`` block is what the Guide
    sheet's research question / numbered criteria, and :func:`full_text_criterion_ids`'s
    invariant check, come from), *rows* the rows actually written, *stratum_counts* the
    drawn-of-available counts from :func:`build_screening_sample` for
    :func:`append_screening_stratum_rows`. Raises :class:`ExportError` (refuses, writes
    nothing) on a screener-version mismatch or a sampled evidence row with no abstract -- see
    :func:`check_screening_prompt_version` and :func:`check_screening_abstracts`."""
    record = load_screening_record(record_path)
    check_screening_prompt_version(record, expected_prompt_version)
    records = record.get("records") or []
    sampled, stratum_counts = build_screening_sample(
        records, seed=seed, included_n=included_n, needs_review_n=needs_review_n,
        excluded_numbered_n=excluded_numbered_n, excluded_off_topic_n=excluded_off_topic_n,
        floor=floor, min_bucket=min_bucket, criteria=record.get("criteria") or {},
    )
    shuffled = shuffle_records(sampled, seed)

    records_b_by_key = None
    if record_b_path is not None:
        record_b = load_screening_record(record_b_path)
        records_b_by_key = {
            match_screening_key(r): r for r in (record_b.get("records") or [])
        }

    rows = [
        build_screening_row(
            r, f"S-{rank:03d}", max_cell_chars=max_cell_chars,
            records_b_by_key=records_b_by_key,
        )
        for rank, r in enumerate(shuffled, start=1)
    ]
    check_screening_abstracts(rows)
    write_screening_sheet(wb, rows, has_run_b=records_b_by_key is not None)
    return record, rows, stratum_counts


# --------------------------------------------------------------------------------------
# Delivered sheet -- the final product a user receives, judged sentence by sentence
# --------------------------------------------------------------------------------------


def delivered_header() -> list[str]:
    return [
        "row", "section_title", "draft_id", "sentence", "claim_checked", "citation_text",
        "paper_title", "paper_authors", "paper_doi", "evidence_quotes", "source_passage",
        "source_located", "passage_located", "unlocated_quotes", "location_section_mismatch",
        "sentence_in_draft", "human_sentence_correct", "human_quote_supports", "human_note",
    ]


def _yes_no_or_blank(value: Any) -> str:
    """``True``/``False`` -> ``"yes"``/``"no"``; ``None`` (the field was never recorded on
    this row, e.g. a ``delivered_evidence.json`` written before the field existed) ->
    ``""``, rather than fabricating a judgement the source record never made."""
    if value is None:
        return ""
    return "yes" if value else "no"


def load_delivered_evidence(path: Path) -> dict[str, Any]:
    """Reads ``demo/output/<run>/delivered_evidence.json`` (schema: ``run_id``, ``draft_id``,
    ``section_title``, ``rows``). Refuses (:class:`ExportError`, named reason) when *path* does
    not exist, is not valid json, or carries no ``rows`` list -- the file is written by the
    demo run itself, so a missing or malformed one usually means the wrong path was given, or
    the run this export documents never produced one."""
    try:
        data = read_json(path, None)
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        raise ExportError(f"--delivered-evidence {path} not found or not valid json")
    if not isinstance(data.get("rows"), list):
        raise ExportError(f"--delivered-evidence {path}: missing or malformed 'rows' list")
    return data


def build_delivered_row(
    row: Mapping[str, Any], *, max_cell_chars: int,
    fallback_section_title: str = "", fallback_draft_id: str = "",
    paper_authors: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """One Delivered-sheet row from one ``delivered_evidence.json`` row: which generated
    section and saved draft it came from (``section_title``/``draft_id``: read from
    the row itself when present, else *fallback_section_title*/*fallback_draft_id* -- the
    file's own top-level values, so a one-section file where no row carries either field
    reads exactly as before), the delivered sentence, its citation and cited paper, the
    verifier's own evidence quotes (joined the same way :func:`all_evidence_quotes` joins the
    Review sheet's own), the source passage those quotes were drawn from, and the record's own
    integrity flags -- ``source_located`` (the named chunk was fetched at all),
    ``passage_located`` (every one of ``evidence_quotes`` was actually found in it -- this,
    not ``source_located``, is what tells a reader whether
    ``source_passage`` shows a full reading), ``unlocated_quotes`` (which quotes, if any,
    ``passage_located`` false is reporting), ``location_section_mismatch`` and
    ``sentence_in_draft`` -- the human's own three judgement columns start blank."""
    doi = row.get("paper_doi") or ""
    authors_text = ""
    if paper_authors is not None:
        authors_text = paper_authors.get(str(doi).lower(), "")
        if not authors_text:
            raise ExportError(
                f"--paper-authors has no record for paper_doi {doi!r} on sheet row "
                f"{row.get('row')!r}"
            )
    out: dict[str, Any] = {
        "row": row.get("row"),
        "section_title": row.get("section_title") or fallback_section_title,
        "draft_id": row.get("draft_id") or fallback_draft_id,
        "sentence": row.get("sentence") or "",
        "claim_checked": row.get("claim_text") or "",
        "citation_text": row.get("citation_text") or "",
        "paper_title": row.get("paper_title") or "",
        "paper_authors": authors_text,
        "paper_doi": doi,
        "evidence_quotes": numbered_list(row.get("evidence_quotes")),
        "source_passage": row.get("source_passage") or "",
        "source_located": _yes_no_or_blank(row.get("source_located")),
        "passage_located": _yes_no_or_blank(row.get("passage_located")),
        "unlocated_quotes": numbered_list(row.get("unlocated_quotes")),
        "location_section_mismatch": _yes_no_or_blank(row.get("location_section_mismatch")),
        "sentence_in_draft": _yes_no_or_blank(row.get("sentence_in_draft")),
        "human_sentence_correct": "", "human_quote_supports": "", "human_note": "",
    }
    for key in (
        "sentence", "claim_checked", "citation_text", "paper_title", "evidence_quotes",
        "source_passage", "unlocated_quotes", "section_title",
    ):
        out[key] = cap_text(out[key], max_cell_chars)
    return out


def write_delivered_sheet(wb: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    header = delivered_header()
    ws = _write_plain_sheet(wb, "Delivered", rows, header, DELIVERED_COLUMN_WIDTHS)
    _add_list_validation(ws, header, "human_sentence_correct", YES_NO_CHOICES)
    _add_list_validation(ws, header, "human_quote_supports", YES_NO_CHOICES)


def build_delivered_sheet(
    wb: Any, delivered_evidence_path: Path, *, max_cell_chars: int,
    paper_authors: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Adds the ``Delivered`` sheet to *wb*. Returns ``(record, rows)`` -- *record* is the
    loaded ``delivered_evidence.json`` (its ``run_id``/``draft_id`` are what
    :func:`append_delivered_provenance_rows` records; its ``section_title``/``draft_id`` are
    the fallback every row without its own value takes), *rows* the rows actually
    written, ordered by section -- in the order each section's own rows first appear in the
    record's own ``rows`` list, which is itself always section-grouped
    (``demo.run_demo.merge_delivered_evidence_records`` appends one whole section's rows
    before the next's) -- and, within each section, in ascending order of the record's own
    ``row`` field (the 1-based order the sentence appears in that section).

    ``row`` restarts at 1 in every section
    (``demo.run_demo.build_delivered_evidence_rows``), so on a multi-section record it is
    unique only within its own section, never across the merged record. Sorting by ``row``
    alone would interleave every section's own row 1, then every
    section's own row 2, and so on, instead of keeping each section's own sentences
    together in the order the sheet's own instructions describe (and
    :data:`DELIVERED_GUIDE_NOTE_TEMPLATE` and :func:`render_delivered_author_paragraph`
    both state)."""
    record = load_delivered_evidence(delivered_evidence_path)
    raw_rows = record["rows"]
    section_order: dict[Any, int] = {}
    for r in raw_rows:
        title = r.get("section_title")
        if title not in section_order:
            section_order[title] = len(section_order)
    ordered = sorted(
        raw_rows,
        key=lambda r: (
            section_order.get(r.get("section_title"), 0),
            r.get("row") if r.get("row") is not None else 0,
        ),
    )
    fallback_section_title = str(record.get("section_title") or "")
    fallback_draft_id = str(record.get("draft_id") or "")
    rows = [
        build_delivered_row(
            r, max_cell_chars=max_cell_chars,
            fallback_section_title=fallback_section_title,
            fallback_draft_id=fallback_draft_id,
            paper_authors=paper_authors,
        )
        for r in ordered
    ]
    write_delivered_sheet(wb, rows)
    return record, rows


def load_paper_authors(path: Path) -> dict[str, str]:
    """Reads a ``demo/tools/selected.json``-shaped list of records (each with ``doi``,
    ``authors_crossref``, ``year``) into ``{doi.lower(): "<names joined by '; '> (<year>)"}``,
    keeping the publisher's own author order (first author first) rather than inverting any
    name to ``Surname, Given``: ``authors_crossref`` stores one full name per author with no
    family-name boundary marked, and two of the twelve demo papers (a name with a middle
    two-word given-name segment, and a two-part surname) would invert wrongly if guessed."""
    data = read_json(path, None)
    if not isinstance(data, list):
        raise ExportError(f"--paper-authors {path} not found or not a json list")
    out: dict[str, str] = {}
    for record in data:
        doi = str(record.get("doi") or "").strip()
        if not doi:
            continue
        names = record.get("authors_crossref") or []
        year = record.get("year")
        joined = "; ".join(str(n) for n in names)
        out[doi.lower()] = f"{joined} ({year})" if year is not None else joined
    return out


def assert_delivered_claims_present(record: Mapping[str, Any], path: Path) -> None:
    """Raises :class:`ExportError` naming *path* and the row numbers of every entry of
    ``record["rows"]`` whose ``claim_text`` is empty. Called from :func:`main` whenever
    ``--delivered-evidence`` is given, so an older ``delivered_evidence.json`` written before
    ``claim_text`` existed is refused with one clear message rather than shipping a workbook
    whose ``claim_checked`` column is silently blank."""
    missing = [
        r.get("row") for r in (record.get("rows") or []) if not (r.get("claim_text") or "")
    ]
    if missing:
        preview = ", ".join(str(r) for r in missing[:10])
        more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        raise ExportError(
            f"--delivered-evidence {path}: {len(missing)} row(s) have no claim_text: "
            f"{preview}{more}"
        )


def distinct_delivered_sentence_count(rows: Sequence[Mapping[str, Any]]) -> int:
    """The number of distinct ``(draft_id, sentence)`` pairs over *all* rows given, so a
    sentence repeated verbatim in two different drafts is counted twice rather than merged."""
    return len({(r.get("draft_id"), r.get("sentence")) for r in rows})


def longest_source_passage_chars(rows: Sequence[Mapping[str, Any]]) -> int:
    """``max(len(source_passage))`` over the built Delivered rows -- measured from the cells
    the sheet actually carries, post-capping, so Guide cell A5's stated length is never a
    number the export did not measure. ``0`` when *rows* is empty."""
    if not rows:
        return 0
    return max(len(str(r.get("source_passage") or "")) for r in rows)


def count_containing_claim_pairs(rows: Sequence[Mapping[str, Any]]) -> int:
    """The number of ordered pairs of rows within one ``(draft_id, sentence)`` group where one
    row's ``claim_checked`` is a substring of the other's (and the two are not identical, since
    two rows never share the exact same claim text on this run). Reported in the exporter's own
    stdout summary so an overlap is visible, not silently asserted against."""
    groups: dict[tuple[Any, Any], list[str]] = {}
    for r in rows:
        key = (r.get("draft_id"), r.get("sentence"))
        groups.setdefault(key, []).append(str(r.get("claim_checked") or ""))
    count = 0
    for claims in groups.values():
        for i, a in enumerate(claims):
            for j, b in enumerate(claims):
                if i != j and a and a != b and a in b:
                    count += 1
    return count


def compute_delivered_workbook_id(
    demo_run_id: str, rows: Sequence[Mapping[str, Any]],
) -> str:
    """A short, deterministic id for a delivered-only export: a sha256 of the run id and every
    built row's ``sentence``, ``claim_checked`` and ``paper_doi``, in sheet order, truncated to
    16 hex characters. Does not depend on export time, Guide text or quote formatting, so a
    re-export of the same run gives the same id; does depend on the run id and every row's own
    text, so a different run or a changed claim gives a different one. Distinct from
    :func:`compute_workbook_id`, which the full-export (item-list) path still uses unchanged."""
    h = hashlib.sha256()
    h.update(b"delivered-v1")
    h.update(b"\x00")
    h.update(str(demo_run_id or "").encode("utf-8"))
    for row in rows:
        for field in ("sentence", "claim_checked", "paper_doi"):
            h.update(b"\x00")
            h.update(str(row.get(field) or "").encode("utf-8"))
    return h.hexdigest()[:16]


def distinct_delivered_draft_ids(rows: Sequence[Mapping[str, Any]]) -> str:
    """Every DISTINCT ``draft_id`` carried by *rows* (a :func:`build_delivered_sheet`
    row list), in first-seen order, semicolon-joined -- the value
    :func:`append_delivered_provenance_rows` records.

    A multi-section run's own merged ``delivered_evidence.json``
    (``demo.run_demo.merge_delivered_evidence_records``) carries no top-level
    ``draft_id`` at all -- each row is tagged with its own section's draft id instead
    -- so reading ``record.get("draft_id")`` (the previous behaviour) always returned
    ``None``/blank, on every run, not only a multi-section one. A single-section run's
    own list here is exactly its one draft id, unchanged; a multi-section run's is
    every section's draft id, so the Provenance sheet is never silently blank."""
    seen: list[str] = []
    for row in rows:
        draft_id = row.get("draft_id")
        if draft_id and draft_id not in seen:
            seen.append(draft_id)
    return "; ".join(seen)


def append_delivered_provenance_rows(
    ws: Any, *, n_rows: int, run_id: str, draft_id: str,
) -> None:
    """Appends ``delivered_rows``/``delivered_run_id``/``delivered_draft_id`` field/value rows
    to the Provenance worksheet *ws* (already written by :func:`write_provenance_sheet`)
    whenever the export carries a Delivered sheet -- the product-level headline the paper
    reports, so the run and draft that produced it are identifiable from the workbook itself,
    not only the file name on disk. *draft_id* is normally
    :func:`distinct_delivered_draft_ids`'s own return value, not the merged record's own
    top-level field."""
    _append_row(ws, ["delivered_rows", str(n_rows)])
    _append_row(ws, ["delivered_run_id", run_id])
    _append_row(ws, ["delivered_draft_id", draft_id])


def build_delivered_only_workbook(
    delivered_evidence_path: Path, *, max_cell_chars: int, demo_run_id: str,
    export_time: str, paper_authors: Mapping[str, str],
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], str]:
    """Builds a workbook carrying exactly three sheets, ``Guide``, ``Provenance`` and
    ``Delivered`` -- no ``Review``, ``Sources``, ``Model_labels`` or ``Screening`` -- for the
    ``--delivered-only`` export path. Returns ``(workbook, record, rows, workbook_id)``.
    Refuses (:class:`ExportError`) via :func:`assert_delivered_claims_present` before writing
    anything, and runs :func:`assert_author_page_safe` over the assembled Guide text before
    returning, exactly as the full-export path does."""
    record = load_delivered_evidence(delivered_evidence_path)
    assert_delivered_claims_present(record, delivered_evidence_path)

    wb = openpyxl.Workbook()
    guide_ws = wb.active
    guide_ws.title = "Guide"
    prov_ws = wb.create_sheet("Provenance")
    # build_delivered_sheet creates its own "Delivered" sheet via wb.create_sheet, which
    # appends it after the two sheets already present, giving the required sheet order:
    # Guide, Provenance, Delivered.
    _record, rows = build_delivered_sheet(
        wb, delivered_evidence_path, max_cell_chars=max_cell_chars,
        paper_authors=paper_authors,
    )
    workbook_id = compute_delivered_workbook_id(demo_run_id, rows)
    n_delivered_sections = len({r.get("section_title") or "" for r in rows})
    write_guide_sheet(
        guide_ws, workbook_id=workbook_id, demo_run_id=demo_run_id, export_time=export_time,
        delivered_only=True, has_delivered=True, has_screening=False,
        n_delivered=len(rows), n_delivered_sentences=distinct_delivered_sentence_count(rows),
        n_delivered_sections=n_delivered_sections,
        longest_passage_chars=longest_source_passage_chars(rows),
    )
    assert_author_page_safe("\n".join(str(c.value or "") for c in guide_ws["A"]))

    prov_rows = [
        {"field": "demo_run_id", "value": demo_run_id},
        {"field": "export_time", "value": export_time},
    ]
    _populate_provenance_sheet(prov_ws, prov_rows)
    append_delivered_provenance_rows(
        prov_ws, n_rows=len(rows),
        run_id=str(record.get("run_id") or ""),
        draft_id=distinct_delivered_draft_ids(rows),
    )
    return wb, record, rows, workbook_id


# --------------------------------------------------------------------------------------
# Author page templates -- the covering instructions sent to the reviewer
# alongside the workbook. Built from these functions rather than freehand prose so
# past defects cannot recur on the next export: a real
# item id (and the false "not identifiable" assurance that went with it) leaked through a
# hand-typed Review-sheet paragraph and disclosed that row's construction category and
# expected label; the Delivered paragraph framed its rows as the whole final product; the
# Screening paragraph called an instruction "unchanged" that the same export had changed.
# --------------------------------------------------------------------------------------

#: A real item id (``hss_test_v3_claims.jsonl``/``real_claims_test.jsonl``) is always
#: ``hss-<slug>-<NN>`` or ``real-<slug>-<NN>`` -- lowercase, hyphen separated, ending in two
#: digits -- never the opaque ``R-###`` Review row id the sheet itself uses (that pattern is
#: deliberately not matched here: naming an opaque row id is not a leak).
#:
#: The slug segment's own character class must allow ``_`` -- 26 of the 60
#: ids in ``hss_test_v3_claims.jsonl`` (every ``hss-no_full_text-NN``, ``hss-over_specified-NN``
#: and ``hss-wrong_paper-NN``) carry an underscore inside a slug segment, and those three
#: categories name the most disclosive thing about a row's construction and expected label --
#: exactly the leak this guard exists to stop. Without ``_`` in the class, the pattern requires
#: every hyphen-delimited segment to match ``[a-z0-9]+`` alone, so a segment carrying an
#: underscore never matches at all and the guard silently passes text naming one of these ids.
REAL_ITEM_ID_PATTERN = re.compile(r"\b(?:hss|real)-[a-z0-9_]+(?:-[a-z0-9_]+)*-\d{2}\b")
#: The exact false assurance an earlier export made: a row was called "not
#: identifiable from the opaque sheet" in the same sentence that named its real id.
FALSE_ASSURANCE_PHRASE = "not identifiable"
#: The phrase that framed 6 Delivered rows as the whole
#: final product rather than the cited-and-verified part of it.
DELIVERED_AUTHOR_QUALIFYING_CLAUSE = "only the cited and verified part of it"


class AuthorPageError(ValueError):
    """:func:`assert_author_page_safe` refuses author-page text that names a real item id or
    repeats the false "not identifiable" assurance an earlier export made."""


def assert_author_page_safe(text: str) -> None:
    """Raises :class:`AuthorPageError` (named reason) when *text* -- the assembled author-page
    covering note -- names a real item id, or claims a row is "not identifiable" (the exact
    combination an earlier export made: a real id named right next to a
    false assurance that it could not be traced back to a Review row). Run this over the whole
    assembled page before it is sent, not only over each template's own output, since the
    finding was in how two true facts were combined, not in either fact alone."""
    match = REAL_ITEM_ID_PATTERN.search(text)
    if match:
        raise AuthorPageError(f"author page text names a real item id: {match.group(0)!r}")
    if FALSE_ASSURANCE_PHRASE in text:
        raise AuthorPageError(
            f"author page text contains the forbidden phrase {FALSE_ASSURANCE_PHRASE!r}"
        )


def render_review_singleton_note(what_happened: str, count: int) -> str:
    """The Review-sheet paragraph fragment for a column that is non-blank on a small number of
    rows this run (e.g. ``first_pass_demotion``). Takes *what_happened* (a description of the
    event, never an item id) and *count* only -- there is no parameter through which a real or
    opaque row id, a construction category or an expected label could be named, so this
    function cannot repeat that defect regardless of what its caller
    passes in."""
    noun = "row" if count == 1 else "rows"
    verb = "carries" if count == 1 else "carry"
    return f"{count} {noun} {verb} it in this run, after {what_happened}"


def render_delivered_author_paragraph(*, n_rows: int, n_sections: int) -> str:
    """The Delivered-sheet paragraph for the author page. Always opens with the same
    qualifying clause (:data:`DELIVERED_AUTHOR_QUALIFYING_CLAUSE`) :data:`
    DELIVERED_GUIDE_NOTE_TEMPLATE` uses on the Guide sheet, so this paragraph cannot drift back
    to the earlier framing (the rows as the whole final product) on
    a future freehand rewrite that forgets to carry it over."""
    rows_phrase = _count_phrase(n_rows, "sentence")
    sections_phrase = _count_phrase(n_sections, "generated section")
    return (
        "This is the final product a user actually receives, but "
        f"{DELIVERED_AUTHOR_QUALIFYING_CLAUSE}: {rows_phrase} from {sections_phrase} of one "
        "demo run, judged one sentence at a time, grouped by section in the order the "
        "sections were generated and, within each section, in the order the sentence appears "
        "there -- not the tool's own internal verification decision, which the Review sheet "
        "covers separately."
    )


def render_screening_author_paragraph(changes: Sequence[str]) -> str:
    """The Screening-sheet paragraph for the author page. *changes* must name every
    instruction that changed since the reviewer's previous copy; every string in it is
    required to appear in the rendered text (so a caller cannot silently drop one, the way
    an earlier export dropped the ``unanchored_exclude`` sentence),
    and an empty sequence is the only way to render "unchanged" -- so a caller can no longer
    claim "unchanged" while separately failing to update this list."""
    if not changes:
        return "Screening sheet. Unchanged in shape and instructions from your last copy."
    return (
        "Screening sheet. Same shape as your last copy, but these instructions changed: "
        + "; ".join(changes) + "."
    )


# --------------------------------------------------------------------------------------
# Workbook orchestration
# --------------------------------------------------------------------------------------


def build_workbook(
    item_specs: Sequence[tuple[Path, Path]],
    results_dir: Path,
    runs: Sequence[str],
    *,
    max_cell_chars: int,
    sources_max_chars: int,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    allow_missing_results: bool = False,
    allow_missing_chunks: bool = False,
    annotation_paths: Sequence[Path] = (),
    guard_commit: str | None = None,
    demo_run_id: str | None = None,
    screening_prompt_version_value: str = "n/a",
    export_time: str | None = None,
) -> tuple[Any, list[dict[str, Any]], dict[str, str], list[str], str]:
    """Returns ``(workbook, kept_items, opaque_ids, skipped_result_ids, workbook_id)``.
    *export_time* defaults to :func:`common.now_iso` when not given; a caller that also writes
    a Guide sheet reflecting the same export (see :func:`write_guide_sheet`'s own
    ``export_time``/``demo_run_id`` parameters) should compute
    it once and pass the same value to both, so the two sheets never print two different
    export timestamps for one export."""
    items = read_items(item_specs)
    caches = build_caches(item_specs)
    results_by_run = load_results(results_dir, runs)

    missing = missing_primary_result_ids(items, results_by_run, runs[0])
    if missing and not allow_missing_results:
        preview = ", ".join(missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise ExportError(
            f"{len(missing)} of {len(items)} item(s) have no run {runs[0]} result (check "
            f"--results-dir and --runs): {preview}{more}. Pass --allow-missing-results to "
            "export anyway (those items are then dropped from the workbook)."
        )
    missing_set = set(missing)
    kept_items = [it for it in items if it["item_id"] not in missing_set]

    found, total_with_chunk = chunk_coverage(kept_items, caches)
    if total_with_chunk and (found / total_with_chunk) < MIN_CHUNK_COVERAGE and (
        not allow_missing_chunks
    ):
        raise ExportError(
            f"only {found} of {total_with_chunk} item(s) with a chunk_doi found a cached "
            f"chunk (below the {MIN_CHUNK_COVERAGE:.0%} floor) -- this usually means the "
            "wrong --cache-dir. Pass --allow-missing-chunks to export anyway."
        )

    shuffled_items, opaque_ids = assign_opaque_ids(kept_items, shuffle_seed)
    rows = [
        build_review_row(
            it, results_by_run, opaque_ids[it["item_id"]], runs=runs,
            max_cell_chars=max_cell_chars,
        )
        for it in shuffled_items
    ]
    workbook_id = compute_workbook_id([it["item_id"] for it in kept_items], shuffle_seed)

    annotations = load_annotation_labels(annotation_paths) if annotation_paths else {}
    model_rows = build_model_labels_rows(
        kept_items, opaque_ids, annotations, max_cell_chars=max_cell_chars,
    ) if annotations else []

    source_rows = build_source_rows(kept_items, caches, opaque_ids, max_chars=sources_max_chars)

    prompt_shas = load_run_prompt_versions(results_dir, runs[0])
    policy_versions = load_run_meta_field_versions(
        results_dir, runs[0], "verification_policy_version",
    )
    repair_prompt_versions = load_run_meta_field_versions(
        results_dir, runs[0], "repair_prompt_version",
    )
    guard_commit_value = resolve_guard_commit(guard_commit)
    export_time_value = export_time or now_iso()
    demo_run_id_value = demo_run_id or "n/a"
    prov_rows = provenance_rows(
        verifier_prompt_shas=prompt_shas, guard_commit=guard_commit_value,
        results_dir=results_dir, export_time=export_time_value,
        demo_run_id=demo_run_id_value, screening_prompt_version=screening_prompt_version_value,
        run_shown=run_files_shown(results_dir, runs[0]), run_commit=resolve_run_commit(results_dir),
        model=load_run_model_facts(results_dir, runs[0]),
        labels=annotation_label_facts(annotation_paths),
        policy_version="; ".join(policy_versions) or "unknown",
        repair_prompt_version="; ".join(repair_prompt_versions) or "unknown",
    )

    n_constructed, n_real = census_counts(kept_items)
    wb = openpyxl.Workbook()
    write_guide_sheet(
        wb.active, workbook_id=workbook_id, demo_run_id=demo_run_id_value,
        export_time=export_time_value, n_constructed=n_constructed, n_real=n_real,
    )
    write_review_sheet(wb, rows)
    if model_rows:
        write_model_labels_sheet(wb, model_rows)
    write_sources_sheet(wb, source_rows)
    write_provenance_sheet(wb, prov_rows)
    return wb, kept_items, opaque_ids, missing, workbook_id


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--items", nargs="+", default=None, type=Path)
    ap.add_argument("--cache-dir", nargs="+", default=None, type=Path)
    ap.add_argument("--results-dir", default=None, type=Path)
    ap.add_argument("--runs", default="A,B", help="comma-separated run letters, e.g. A,B")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--key-out", default=None, type=Path)
    ap.add_argument(
        "--delivered-only", action="store_true",
        help="write only Guide, Provenance and Delivered (no item list, no Review/Sources/"
             "Model_labels/Screening); requires --delivered-evidence and --paper-authors, "
             "makes --items/--cache-dir/--results-dir/--key-out optional",
    )
    ap.add_argument(
        "--paper-authors", type=Path, default=None,
        help="demo/tools/selected.json -- doi-keyed author lists for the Delivered sheet's "
             "paper_authors column. Required in --delivered-only mode, and required on the "
             "full-export path whenever --delivered-evidence is also given.",
    )
    ap.add_argument(
        "--annotation", nargs="*", default=[], type=Path,
        help="one or more adjudicated-annotation csv files, for the Model_labels sheet",
    )
    ap.add_argument("--max-cell-chars", type=int, default=DEFAULT_MAX_CELL_CHARS)
    ap.add_argument("--sources-max-chars", type=int, default=DEFAULT_SOURCES_MAX_CHARS)
    ap.add_argument("--shuffle-seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    ap.add_argument("--allow-missing-results", action="store_true")
    ap.add_argument("--allow-missing-chunks", action="store_true")
    ap.add_argument("--guard-commit", default=None)
    ap.add_argument("--demo-run-id", default=None)
    ap.add_argument("--screening-record", type=Path, default=None)
    ap.add_argument("--screening-record-b", type=Path, default=None)
    ap.add_argument(
        "--screening-prompt-version-expect", default=None,
        help="required with --screening-record: refuses the record unless its own "
             "provenance.screener.prompt_version equals this exactly",
    )
    ap.add_argument("--screening-seed", type=int, default=DEFAULT_SCREENING_SEED)
    ap.add_argument("--screening-included-n", type=int, default=DEFAULT_SCREENING_INCLUDED_N)
    ap.add_argument(
        "--screening-needs-review-n", type=int, default=DEFAULT_SCREENING_NEEDS_REVIEW_N,
        help=(
            "needs_review's own sample size, allocated per needs_review_reason bucket "
            "(allocate_stratified_with_floor): each bucket with at least --screening-min-bucket "
            "records starts at min(--screening-floor, its own size); a smaller bucket is taken "
            "whole. If that total then exceeds this value, one record at a time is removed from "
            "whichever bucket currently holds the most (ties broken by bucket order), never "
            "below 1 per non-empty bucket unless this value is itself smaller than the number "
            "of non-empty buckets. If the total is still short, one record at a time is added "
            "to whichever bucket has the largest remaining unallocated size. The result always "
            "totals min(this value, the sum of every bucket's size), and the largest bucket is "
            "never left at zero unless it has to be."
        ),
    )
    ap.add_argument(
        "--screening-excluded-n", type=int, default=DEFAULT_SCREENING_EXCLUDED_N,
        help="the numbered-criterion excluded sample size (stage: llm only)",
    )
    ap.add_argument(
        "--screening-off-topic-n", type=int, default=DEFAULT_SCREENING_OFF_TOPIC_N,
        help="the reserved off-topic criterion's own excluded sample size (stage: llm only)",
    )
    ap.add_argument(
        "--screening-floor", type=int, default=DEFAULT_SCREENING_FLOOR,
        help="needs_review's per-reason-bucket floor only -- the excluded samples are now "
             "drawn uniformly and do not use this; see --screening-needs-review-n's help for "
             "the full allocation rule",
    )
    ap.add_argument(
        "--screening-min-bucket", type=int, default=DEFAULT_SCREENING_MIN_BUCKET,
        help="needs_review's own bucket-size threshold for --screening-floor to apply; see "
             "--screening-needs-review-n's help for the full allocation rule",
    )
    ap.add_argument(
        "--delivered-evidence", type=Path, default=None,
        help="demo/output/<run>/delivered_evidence.json -- the final product a user "
             "receives, one row per delivered sentence. Adds a Delivered sheet when given; "
             "refuses (named reason) when the path does not exist or is not valid json.",
    )
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.delivered_only:
        if args.annotation:
            print("ERROR: --annotation is refused with --delivered-only "
                  "(the Model_labels sheet is not written in this mode)")
            return 2
        if args.screening_record is not None or args.screening_record_b is not None:
            print("ERROR: --screening-record/--screening-record-b are refused with "
                  "--delivered-only (the Screening sheet is not written in this mode)")
            return 2
        if args.delivered_evidence is None:
            print("ERROR: --delivered-evidence is required with --delivered-only")
            return 2
        if args.paper_authors is None:
            print("ERROR: --paper-authors is required with --delivered-only")
            return 2
        if not args.demo_run_id:
            print("ERROR: --demo-run-id is required with --delivered-only")
            return 2
        export_time = now_iso()
        try:
            paper_authors = load_paper_authors(args.paper_authors)
            wb, record, rows, workbook_id = build_delivered_only_workbook(
                args.delivered_evidence, max_cell_chars=args.max_cell_chars,
                demo_run_id=args.demo_run_id, export_time=export_time,
                paper_authors=paper_authors,
            )
        except ExportError as exc:
            print(f"ERROR: {exc}")
            return 2
        args.out.parent.mkdir(parents=True, exist_ok=True)
        wb.save(args.out)
        if args.key_out is not None:
            write_delivered_only_key_csv(args.key_out, workbook_id=workbook_id)
        n_sentences = distinct_delivered_sentence_count(rows)
        n_pairs = count_containing_claim_pairs(rows)
        longest = longest_source_passage_chars(rows)
        print(
            f"delivered-only: {len(rows)} rows, {n_sentences} distinct sentences, "
            f"{n_pairs} overlapping claim pairs, longest passage {longest} characters, "
            f"workbook id {workbook_id}"
        )
        print(f"wrote {args.out}" + (f" and {args.key_out}" if args.key_out is not None else ""))
        return 0

    if args.items is None or args.cache_dir is None or args.results_dir is None:
        print("ERROR: --items, --cache-dir and --results-dir are required "
              "(unless --delivered-only is given)")
        return 2
    if args.key_out is None:
        print("ERROR: --key-out is required (unless --delivered-only is given)")
        return 2
    if args.delivered_evidence is not None and args.paper_authors is None:
        print("ERROR: --paper-authors is required whenever --delivered-evidence is given")
        return 2

    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    item_specs = resolve_item_specs(args.items, args.cache_dir)

    if args.screening_record is not None and not args.screening_prompt_version_expect:
        print("ERROR: --screening-prompt-version-expect is required with --screening-record")
        return 2

    # Computed once and reused for every sheet this export touches (the Provenance sheet and,
    # when there is a Screening sheet, the second write_guide_sheet call below), so the Guide
    # and Provenance sheets never print two different export times or demo run ids for one
    # export.
    export_time = now_iso()
    demo_run_id_value = args.demo_run_id or (
        args.screening_record.stem if args.screening_record else None
    )

    try:
        wb, items, opaque_ids, skipped_results, workbook_id = build_workbook(
            item_specs, args.results_dir, runs,
            max_cell_chars=args.max_cell_chars, sources_max_chars=args.sources_max_chars,
            shuffle_seed=args.shuffle_seed, allow_missing_results=args.allow_missing_results,
            allow_missing_chunks=args.allow_missing_chunks,
            annotation_paths=args.annotation, guard_commit=args.guard_commit,
            demo_run_id=demo_run_id_value,
            screening_prompt_version_value=(
                args.screening_prompt_version_expect if args.screening_record else "n/a"
            ),
            export_time=export_time,
        )
    except ExportError as exc:
        print(f"ERROR: {exc}")
        return 2

    # The Guide sheet build_workbook already wrote reflects neither a Screening nor a
    # Delivered sheet; whichever of the two blocks below runs, each folds its own facts into
    # this one shared kwargs dict, and a single rewrite at the end reflects the union of both
    # (rather than each block rewriting the sheet on its own and the earlier one's facts being
    # silently lost -- see build_workbook's own docstring on the export_time/demo_run_id
    # sharing rule this generalises).
    n_constructed, n_real = census_counts(items)
    guide_kwargs: dict[str, Any] = dict(
        workbook_id=workbook_id, demo_run_id=demo_run_id_value or "n/a",
        export_time=export_time, n_constructed=n_constructed, n_real=n_real,
    )
    needs_guide_rewrite = False

    if args.screening_record is not None:
        try:
            record, screening_rows, stratum_counts = build_screening_sheet(
                wb, args.screening_record,
                expected_prompt_version=args.screening_prompt_version_expect,
                seed=args.screening_seed, included_n=args.screening_included_n,
                needs_review_n=args.screening_needs_review_n,
                excluded_numbered_n=args.screening_excluded_n,
                excluded_off_topic_n=args.screening_off_topic_n, floor=args.screening_floor,
                min_bucket=args.screening_min_bucket, max_cell_chars=args.max_cell_chars,
                record_b_path=args.screening_record_b,
            )
        except ExportError as exc:
            print(f"ERROR: {exc}")
            return 2
        criteria = record.get("criteria") or {}
        full_text_possible = bool(full_text_criterion_ids(criteria))
        guide_kwargs.update(
            research_question=str(criteria.get("query") or ""),
            criteria_text=render_numbered_criteria(criteria), has_screening=True,
            screening_seed=args.screening_seed, full_text_stratum_possible=full_text_possible,
        )
        needs_guide_rewrite = True
        cutoff = retrieval_cutoff_value(criteria)
        if cutoff:
            append_retrieval_cutoff_row(wb["Provenance"], cutoff)
        append_screening_stratum_rows(wb["Provenance"], stratum_counts)
        print(f"Screening sheet: sampled {len(screening_rows)} of "
              f"{len(record.get('records') or [])} record(s) from {args.screening_record}")
        for stratum in SCREENING_STRATA:
            c = stratum_counts.get(stratum, {"drawn": 0, "available": 0})
            print(f"  {stratum}: {c['drawn']} of {c['available']}")

    if args.delivered_evidence is not None:
        try:
            _delivered_record_for_claims = load_delivered_evidence(args.delivered_evidence)
            assert_delivered_claims_present(
                _delivered_record_for_claims, args.delivered_evidence,
            )
            paper_authors = (
                load_paper_authors(args.paper_authors) if args.paper_authors else None
            )
            delivered_record, delivered_rows = build_delivered_sheet(
                wb, args.delivered_evidence, max_cell_chars=args.max_cell_chars,
                paper_authors=paper_authors,
            )
        except ExportError as exc:
            print(f"ERROR: {exc}")
            return 2
        n_delivered_sections = len({r.get("section_title") or "" for r in delivered_rows})
        guide_kwargs.update(
            has_delivered=True, n_delivered=len(delivered_rows),
            n_delivered_sections=n_delivered_sections,
            n_delivered_sentences=distinct_delivered_sentence_count(delivered_rows),
            longest_passage_chars=longest_source_passage_chars(delivered_rows),
        )
        needs_guide_rewrite = True
        append_delivered_provenance_rows(
            wb["Provenance"], n_rows=len(delivered_rows),
            run_id=str(delivered_record.get("run_id") or ""),
            draft_id=distinct_delivered_draft_ids(delivered_rows),
        )
        print(f"Delivered sheet: {len(delivered_rows)} row(s) from {args.delivered_evidence}")

    if needs_guide_rewrite:
        write_guide_sheet(wb["Guide"], **guide_kwargs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(args.out)
    write_key_csv(args.key_out, items, opaque_ids, workbook_id=workbook_id)
    print(f"wrote {args.out} ({len(items)} rows, shuffle seed {args.shuffle_seed}) and "
          f"{args.key_out}")
    if skipped_results:
        preview = ", ".join(skipped_results[:5])
        more = f" (+{len(skipped_results) - 5} more)" if len(skipped_results) > 5 else ""
        print(f"skipped {len(skipped_results)} item(s) with no run {runs[0]} result: "
              f"{preview}{more}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
