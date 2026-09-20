"""Offline checks over the promoted baseline in demo/expected/.

``test_run_demo.py`` exercises ``run_demo.py`` against a fake HTTP server and never reads
``demo/expected/`` itself. This file reads the committed baseline directly: no network, no
database, no LLM, no import of ``run_demo``.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[1]
EXPECTED = DEMO / "expected"

# The frozen build's prompt hashes, duplicated here rather than imported from the backend,
# because demo/tests runs under the system Python, which does not have the backend's
# dependencies (pydantic_ai etc.) installed.
SCREENER_PROMPT_VERSION = "sha256:9bc741046cd6"
CLAIM_VERIFICATION_PROMPT_VERSION = "sha256:49fcfbfaf2f6"
#: The second-pass screening judge's own frozen prompt, made once per job rather than once
#: per candidate record.
SECOND_PASS_PROMPT_VERSION = "sha256:7aa4b622c72e"

# app.services.screening_record.CSV_HEADER, copied verbatim: nineteen columns, the four
# second-pass fields inserted between `to_confirm` and `reason`, `abstract` last.
CSV_HEADER = (
    "outcome", "stage", "status", "criterion", "quote", "needs_review_reason", "to_confirm",
    "second_pass", "second_pass_input_tokens", "second_pass_output_tokens",
    "second_pass_latency_s",
    "reason", "title", "doi", "year", "journal", "journal_issn", "openalex_id", "abstract",
)

# The six record fields added on top of the base nine-column schema (outcome, stage, reason,
# title, doi, year, journal, journal_issn, openalex_id).
ADDED_RECORD_FIELDS = (
    "status", "criterion", "quote", "needs_review_reason", "to_confirm", "abstract",
)

# The four columns the once-per-job second-pass judge adds on top of the six above.
SECOND_PASS_FIELDS = (
    "second_pass", "second_pass_input_tokens", "second_pass_output_tokens",
    "second_pass_latency_s",
)

# `protocol.json`'s own note: `expected_status` is what the fixture's construction implies, not
# a measurement, and the status actually observed may differ under model drift. On the promoted
# baseline both protocol fixtures resolve exactly as designed: `supported-1` verifies on its
# second pass, after one repair turn (`model_status: verified`, `machine_reasons: []`,
# `guard_demotion: False`, `matches_expected: True`, no quote relocation) -- pass 1 quoted a
# non-verbatim paraphrase (`machine_reasons: ["quote_not_verbatim"]`), and the repair turn
# returned the source's own verbatim wording. `unsupported-1` is correctly caught (`model_status:
# unsupported`, machine reason `assertion_status_inconsistent`, one pass), and is removed from the
# delivered text once the standalone verify-and-heal action runs (`final_status: "not_in_report"`).
FIXTURE_STATUSES = {"supported-1": "verified", "unsupported-1": "unsupported"}
FIXTURE_MATCHES_EXPECTED = {"supported-1": True, "unsupported-1": True}


def _load(name: str):
    return json.loads((EXPECTED / name).read_text(encoding="utf-8"))


def _promoted_backing_run_dir() -> Path:
    """The ``demo/output/<run>`` directory the current ``demo/expected/`` baseline was
    promoted from, identified by CONTENT, not a pinned run id -- the directory under
    ``demo/output`` whose own ``claim_report.json`` is byte-identical to the promoted
    ``demo/expected/claim_report.json`` -- so a future promotion (a new run directory, the
    old one pruned) is picked up automatically instead of this suite silently comparing
    against a stale, hardcoded name.

    Raises :class:`AssertionError` by name when no such directory exists (deleted, or the
    baseline was promoted without its own backing run directory ever being kept on disk)
    rather than letting a missing-file error surface deep inside whichever test called this,
    or being swallowed by an ``xfail`` wrapper somewhere upstream."""
    expected_report = _load("claim_report.json")
    output_root = DEMO / "output"
    if output_root.is_dir():
        for run_dir in sorted(output_root.iterdir()):
            report_path = run_dir / "claim_report.json"
            if not report_path.exists():
                continue
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if report == expected_report:
                return run_dir
    raise AssertionError(
        "no directory under demo/output/ has a claim_report.json matching the promoted "
        "demo/expected/claim_report.json; the backing run directory for this baseline "
        "is missing or the baseline was promoted without one"
    )


def test_promoted_backing_run_dir_raises_assertion_error_when_demo_output_is_entirely_absent(
    monkeypatch, tmp_path
):
    """``demo/output/`` itself can be entirely absent (a checkout that ships no run
    directories at all), not merely lacking a matching one. ``_promoted_backing_run_dir``
    must still raise its own documented ``AssertionError`` with a plain message in that
    case, not let a bare ``FileNotFoundError`` from ``output_root.iterdir()`` escape
    instead."""
    fake_expected = tmp_path / "expected"
    fake_expected.mkdir()
    (fake_expected / "claim_report.json").write_text("{}", encoding="utf-8")
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "DEMO", tmp_path)
    monkeypatch.setattr(module, "EXPECTED", fake_expected)

    with pytest.raises(AssertionError, match="no directory under demo/output/"):
        _promoted_backing_run_dir()


def test_csv_header_is_the_nineteen_columns_in_order():
    text = (EXPECTED / "screening_record.csv").read_text(encoding="utf-8-sig")
    header = next(csv.reader(io.StringIO(text.splitlines()[0])))
    assert tuple(header) == CSV_HEADER
    assert len(header) == 19


def test_csv_rows_carry_the_added_fields_and_agree_with_the_header():
    text = (EXPECTED / "screening_record.csv").read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows, "screening_record.csv has no data rows"
    assert set(rows[0]) == set(CSV_HEADER)
    for field in ADDED_RECORD_FIELDS:
        assert field in rows[0]
    for field in SECOND_PASS_FIELDS:
        assert field in rows[0]


def test_csv_second_pass_columns_agree_with_the_once_per_job_block():
    """The four ``second_pass*`` columns are filled in only for the candidates the once-per-
    job second-pass judge actually reviewed; the row count that carries a non-blank
    ``second_pass`` value must equal that block's own ``candidates`` count, and the block's
    own arithmetic (candidates minus demotions minus unavailable) must equal the flow's own
    ``included`` count."""
    summary = _load("summary.json")
    second_pass = summary["screening"]["provenance"]["screener_second_pass"]
    flow = summary["screening"]["flow"]
    assert second_pass["candidates"] - second_pass["demotions"] - second_pass["unavailable"] \
        == flow["included"]

    text = (EXPECTED / "screening_record.csv").read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    non_blank = [r for r in rows if r["second_pass"].strip()]
    assert len(non_blank) == second_pass["candidates"]


def test_summary_carries_no_comparison_key():
    """``comparison`` is a *run's* own comparison against whatever ``demo/expected/`` happened
    to be at the time it ran -- never a comparison of the baseline against itself. A promotion
    that ships it verbatim would show the baseline appearing to drift from itself; this is the
    one check that it was stripped."""
    summary = _load("summary.json")
    assert "comparison" not in summary


def test_summary_carries_both_frozen_prompt_versions():
    summary = _load("summary.json")
    screener_version = summary["screening"]["provenance"]["screener"]["prompt_version"]
    verifier_version = summary["verification"]["provenance"]["prompt_version"]
    assert screener_version == SCREENER_PROMPT_VERSION
    assert verifier_version == CLAIM_VERIFICATION_PROMPT_VERSION


def test_summary_carries_the_second_pass_prompt_version():
    summary = _load("summary.json")
    second_pass = summary["screening"]["provenance"]["screener_second_pass"]
    assert second_pass["prompt_version"] == SECOND_PASS_PROMPT_VERSION


def test_claim_report_carries_the_verifier_prompt_version():
    report = _load("claim_report.json")
    assert report["provenance"]["prompt_version"] == CLAIM_VERIFICATION_PROMPT_VERSION


def test_every_screening_record_has_the_six_added_fields():
    record = _load("screening_record.json")
    records = record["records"]
    assert records, "screening_record.json has no records"
    missing = {field: 0 for field in ADDED_RECORD_FIELDS}
    for row in records:
        for field in ADDED_RECORD_FIELDS:
            if field not in row:
                missing[field] += 1
    assert missing == {field: 0 for field in ADDED_RECORD_FIELDS}


def test_screening_record_has_the_needs_review_outcome_and_flow_keys():
    record = _load("screening_record.json")
    outcomes = {row["outcome"] for row in record["records"]}
    assert outcomes <= {"included", "excluded", "needs_review", "unscreened"}
    assert "needs_review" in outcomes
    flow = record["flow"]
    assert "needs_review" in flow and "needs_review_by_reason" in flow
    assert flow["needs_review"] == sum(flow["needs_review_by_reason"].values())
    review_rows = sum(1 for row in record["records"] if row["outcome"] == "needs_review")
    assert flow["needs_review"] == review_rows


def test_the_two_protocol_claims_resolve_to_their_fixture_statuses():
    """``matches_expected`` is not required to be ``True`` for every claim: it compares the
    observed status against ``protocol.json``'s by-construction ``expected_status``, and a
    live, non-deterministic verifier can legitimately land on a different status for a
    genuinely borderline claim. What must hold is that the observed statuses and their
    match/mismatch verdicts are exactly what this baseline recorded, not that every claim
    matched its by-construction expectation.
    """
    summary = _load("summary.json")
    claims = {c["id"]: c["status"] for c in summary["verification"]["claims"]}
    assert claims == FIXTURE_STATUSES
    matches_expected = {
        c["id"]: c["matches_expected"] for c in summary["verification"]["claims"]
    }
    assert matches_expected == FIXTURE_MATCHES_EXPECTED


def test_citation_coverage_is_present_and_internally_consistent():
    """The citation-coverage object on the promoted baseline must add up on its own terms,
    independent of what the counts happen to be for this particular live run."""
    report = _load("claim_report.json")
    coverage = report["citation_coverage"]
    assert coverage is not None
    for field in ("found", "linked", "sent_to_verifier", "unresolved", "by_source"):
        assert field in coverage
    assert coverage["found"] == coverage["linked"] + coverage["unresolved"]
    assert sum(coverage["by_source"].values()) == coverage["found"]
    assert len(coverage["unresolved_citations"]) == min(coverage["unresolved"], 50)


def test_citation_coverage_agrees_between_claim_report_and_summary():
    report = _load("claim_report.json")
    summary = _load("summary.json")
    assert summary["verification"]["citation_coverage"] == report["citation_coverage"]


def test_citation_coverage_has_no_unresolved_citations_on_this_baseline():
    """On this baseline every citation occurrence the audit recognises resolves to a citation
    key, so ``unresolved`` is zero and ``found`` equals ``linked``. A future baseline may carry
    a non-zero ``unresolved`` again (live model output varies), in which case this test is
    updated alongside the promotion, not silently left to fail.
    """
    report = _load("claim_report.json")
    coverage = report["citation_coverage"]
    assert coverage["unresolved"] == 0
    assert coverage["unresolved_citations"] == []
    assert coverage["found"] == coverage["linked"]


def test_writing_provenance_carries_the_citation_link_calls_and_folded_totals():
    """The gated write loop can make anywhere from two DeepSeek calls (one generation call,
    one citation-link call, if every claim verifies on the first pass and the body never
    exceeds the hard maximum) up to a first-pass generation, an over-length regeneration, a
    full-section revision, one citation-link call per map build, and one or more
    verification-agent batches, each folded into one running total. This checks the
    properties that hold regardless of the exact count: the citation-link call list is
    non-empty, its last entry is the same call ``citation_link_call`` names, and the folded
    totals are strictly larger than the writing-and-citation-link tokens alone (the analysis
    and verification calls contribute the rest).
    """
    summary = _load("summary.json")
    writing = summary["writing"]
    citation_link_calls = writing["citation_link_calls"]
    assert citation_link_calls, "citation_link_calls is empty"
    assert citation_link_calls[-1] == writing["citation_link_call"]
    assert writing["citation_link_call"]["model_reported"]

    citation_link_input = sum(c["input_tokens"] for c in citation_link_calls)
    citation_link_output = sum(c["output_tokens"] for c in citation_link_calls)
    assert writing["total_input_tokens"] > writing["input_tokens"] + citation_link_input
    assert writing["total_output_tokens"] > writing["output_tokens"] + citation_link_output
    assert writing["total_calls"] > 1 + len(citation_link_calls)


def test_claim_text_in_draft_is_complete_on_this_baseline():
    summary = _load("summary.json")
    claim_text_in_draft = summary["verification"]["claim_text_in_draft"]
    n, _, m = claim_text_in_draft.partition(" of ")
    assert n == m, f"claim_text_in_draft is {claim_text_in_draft!r}, not complete"


def test_every_claim_text_occurs_in_the_promoted_run_directorys_draft():
    """The assertion above only compares the two halves of the ``"n of m"`` string
    ``run_demo.py`` itself computed and never reads a draft paragraph, so a promotion that
    swapped in a mismatched ``claim_report.json`` while keeping that recorded string would
    still pass it. ``demo/expected/`` does not ship ``draft_content.json``, but the run
    directory this baseline was promoted from (``_promoted_backing_run_dir``, resolved by
    content, not a pinned run id) does, kept on disk alongside a ``claim_report.json`` that is
    byte-identical (sha256) to ``demo/expected/claim_report.json``. This test reads both files
    from that run directory directly and checks the real property: every ``claim_text`` occurs
    verbatim in the draft's concatenated paragraph text.

    This checks ``claim_report["final_report"]["verifications"]`` (the post-heal survivors
    that are actually still in the saved draft), not the top-level, pre-heal
    ``verifications`` (the standalone verify-and-heal action's own finalize step removes any
    sentence that is not verified before this check runs, so on this baseline the one claim
    that was not verified, ``unsupported-1``, the pre-heal list's own ``claim_text`` value
    includes text no longer in the draft; checking against that list would make this test
    fail on every baseline with a genuine removal instead of on a real defect).
    """
    run_dir = _promoted_backing_run_dir()
    draft = json.loads((run_dir / "draft_content.json").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "claim_report.json").read_text(encoding="utf-8"))

    # Join each paragraph's own text nodes with no separator first (matching
    # ``run_demo.py``'s ``_draft_paragraph_texts``, which concatenates a paragraph's text
    # nodes with ``"".join(...)``), then join the paragraphs together. Joining every text
    # node across every paragraph with "\n\n" instead would insert a separator between two
    # text nodes of the *same* sentence whenever a mark (e.g. italics) splits it into more
    # than one node, and a claim_text spanning that split would then wrongly report missing
    # even though the runner's own join would have found it.
    paragraphs = [node for node in draft["content"] if node["type"] == "paragraph"]
    draft_text = "\n\n".join(
        "".join(
            text_node["text"]
            for text_node in node.get("content", [])
            if text_node["type"] == "text"
        )
        for node in paragraphs
    )

    final_verifications = report["final_report"]["verifications"]
    missing = [
        v["claim_text"] for v in final_verifications if v["claim_text"] not in draft_text
    ]
    assert missing == [], f"{len(missing)} claim_text value(s) not found in the draft: {missing}"


# --------------------------------------------------------------------------------------
# The extra-sections feature. `demo/expected/` ships summary.json, screening_record.json,
# screening_record.csv, claim_report.json and delivered_evidence.json, and never a
# per-section `claim_report_<n>.json`, even once a promoted run carries one -- those files
# are read from the promoted run's own directory instead (`_promoted_backing_run_dir`). The
# per-section-claim-report test below therefore keeps skipping on a baseline that ships no
# such file; the sections-list test does not skip, because `summary.json` itself (which IS
# promoted) already carries the `sections` list once a run used `--sections`.
# --------------------------------------------------------------------------------------


def test_delivered_evidence_is_promoted_and_tags_every_row_with_its_own_section():
    path = EXPECTED / "delivered_evidence.json"
    if not path.exists():
        pytest.skip("demo/expected/delivered_evidence.json not yet promoted")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["rows"], "delivered_evidence.json has no rows"
    for row in record["rows"]:
        assert row.get("section_title")
        assert row.get("draft_id")
    assert record["unlocated_rows"] == sum(
        1 for row in record["rows"] if not row.get("passage_located")
    )


def test_per_section_claim_reports_are_promoted_once_extra_sections_are_shipped():
    section_reports = sorted(EXPECTED.glob("claim_report_*.json"))
    if not section_reports:
        pytest.skip("no demo/expected/claim_report_<n>.json shipped on this baseline")
    for path in section_reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["provenance"]["prompt_version"] == CLAIM_VERIFICATION_PROMPT_VERSION
        final_report = report["final_report"]
        assert final_report["verified_count"] == len(final_report["verifications"])
        for verification in final_report["verifications"]:
            assert verification["status"] == "verified"


def test_summary_carries_a_sections_list_once_extra_sections_are_shipped():
    summary = _load("summary.json")
    sections = summary.get("sections")
    if not sections:
        pytest.skip("summary.json carries no extra sections on this baseline")
    for entry in sections:
        assert entry.get("title")
        assert entry.get("draft_id")
        assert "loop_stats" in entry["writing"]
        assert "verified_count" in entry["verification"]
        assert "rows" in entry["delivered_evidence"]


# --------------------------------------------------------------------------------------
# check_delivered.py: the promoted baseline's own backing run, checked through the same
# offline delivered-text checker every demo run now runs against itself.
# --------------------------------------------------------------------------------------


def test_check_delivered_against_the_promoted_baselines_backing_run():
    """``demo/expected/`` does not ship ``draft_content.json`` or ``writing_result.json``
    (see ``test_every_claim_text_occurs_in_the_promoted_run_directorys_draft``, above, which
    already establishes that its own ``claim_report.json`` is byte-identical to that run
    directory's), so the checker is run against the run directory the baseline was promoted
    from instead (``_promoted_backing_run_dir``).

    This asserts the EXACT, currently known violation set rather than an ``xfail`` wrapper,
    so a missing backing directory fails loudly in ``_promoted_backing_run_dir`` itself, and a
    baseline that changes its own violation set (a re-promotion, or a regression) fails this
    assertion directly, with the full violation list in the failure message. The promoted
    baseline is clean on all seventeen rules across all seven sections."""
    import check_delivered

    protocol = json.loads((DEMO / "protocol.json").read_text(encoding="utf-8"))
    run_dir = _promoted_backing_run_dir()
    result = check_delivered.check_run_directory(run_dir, protocol=protocol)
    rules = sorted(v.rule for v in result.violations)
    assert rules == [], [str(v) for v in result.violations]
