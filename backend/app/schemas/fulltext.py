"""Schemas for full-text retrieval and claim verification."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel


class ClaimAssertion(BaseModel):
    """One atomic assertion extracted from a claim, with the source's verdict on it.

    The v2 verification prompt
    decomposes a claim into these before deciding a status, so a "one refuted proposition
    condemns the whole claim" aggregation rule is auditable rather than buried in free-text
    ``explanation``. Model-facing (part of ``ClaimVerificationOutput``); never edited by the
    service.
    """

    text: str
    #: "population" | "intervention" | "quantity" | "direction" | "scope" | "attribution"
    #: | "other"
    kind: str
    #: "supported" | "contradicted" | "absent"
    verdict: str
    #: What the claim asserts; required (by the prompt, not the schema) when kind=="quantity".
    claim_value: str | None = None
    #: What the source says; required (by the prompt, not the schema) when
    #: verdict=="contradicted".
    source_value: str | None = None
    #: Verbatim span from a chunk.
    quote: str | None = None
    #: The proposition the
    #: sentence exists to state, its main predicate over its main subject. Exactly one
    #: assertion is marked True by the prompt; all others are peripheral. Additive with
    #: a default, so an old-shape dict (no ``central`` key) still validates.
    central: bool = False
    #: What the assertion is about (the "principal entity" the v3 prompt reasons over).
    #: Additive with a default.
    entity: str | None = None
    #: What is being measured or compared (paired with ``entity`` for the v3 contradiction
    #: and centrality checks). Additive with a default.
    measure: str | None = None
    #: Every verbatim span composed to reach this assertion's verdict (v3's ``supported``
    #: composition rule spans more than one chunk). Additive with a default, so an
    #: old-shape dict still validates.
    quotes: list[str] = []


class ClaimVerificationOutput(BaseModel):
    """What the claim-verification model returns for one claim.

    This is the agent's ``output_type``: it must contain only fields the model is asked
    to fill, so that the model-facing JSON schema stays stable and provenance fields
    are never offered to the model.
    """

    claim_text: str
    paper_id: UUID
    status: str  # "verified" | "unsupported" | "needs_nuance" | "no_full_text" | "error"
    evidence_quote: str | None = None
    explanation: str
    suggested_revision: str | None = None
    #: One entry per atomic assertion (steps 1-2 of the v2 prompt).
    #: Additive with a default, so an old-shape dict without this key still validates.
    assertions: list[ClaimAssertion] = []
    #: Peripheral assertions the v3
    #: precedence rule marks absent, in prose, for a ``needs_nuance`` claim. Additive
    #: with a default, so an old-shape dict without this key still validates.
    unstated_details: list[str] = []
    #: Every verbatim span the model used across all assertions (STEP 4 of the
    #: v3 prompt). ``evidence_quote`` keeps its own name, type and position; the service
    #: (not this schema) sets it to this list's first span. Additive with a default.
    evidence_quotes: list[str] = []


class ClaimVerification(ClaimVerificationOutput):
    """Stored result for one claim: the model's output plus system-filled provenance."""

    # Which model actually answered (provider-reported name); set by the service after
    # the call, never by the model.
    model_reported: str | None = None
    # The remaining fields are filled by the service, never by the model, so the
    # claim-verification record export can read them back without a
    # database lookup (app.services.claim_record).
    #: The full sentence the
    #: claim was cut from. Equal to ``claim_text`` when no citation-link proposition
    #: narrowed the claim to a shorter span of that sentence. Set by the service, never
    #: by the model.
    claim_sentence: str | None = None
    #: The raw "(Author, Year)"-style citation text the claim was matched against.
    citation: str | None = None
    #: DOI of the matched paper, if any.
    paper_doi: str | None = None
    #: Title of the matched paper, if any.
    paper_title: str | None = None
    #: Full-text source recorded by the fulltext pipeline for the matched paper, if any
    #: (e.g. "unpaywall", "stored_url", "metadata_oa_url").
    acquisition_route: str | None = None
    #: Which numbered chunk (as shown to the model) contained the evidence quote, if found.
    evidence_location: str | None = None
    #: The status the model returned, before any code guard ran (``app/services/fulltext.py``
    #: guards 0-3 do the guarding). ``None`` when no model call was made (the deterministic
    #: zero-chunk ``no_full_text`` path, or an agent-call error). Compared against
    #: ``status`` by the frontend to show "Model answered X; automated check made this
    #: stricter" only when a guard actually changed the verdict.
    model_status: str | None = None
    #: Machine-readable slugs for every guard that fired, in guard order (section 4.5):
    #: "no_full_text_with_chunks", "attribution_mismatch", "assertion_status_inconsistent",
    #: "quote_not_verbatim", "numeric_outside_cited_passage",
    #: "numeric_value_absent_from_source". "numeric_not_in_source" is a
    #: diagnostic (see ``diagnostics`` below) and never appears here. A guard may append its
    #: slug here without changing ``status`` (e.g. the attribution guard in report-only mode).
    machine_reasons: list[str] = []
    #: Machine-readable slugs from guards that
    #: never change ``status`` (e.g. "numeric_not_in_source", "centrality_unmarked"). This
    #: field is always a sibling of ``machine_reasons`` and never one of its members, so a
    #: reader cannot mistake a diagnostic for something that changed the verdict. Additive
    #: with a default, so an old-shape dict without this key still validates.
    diagnostics: list[str] = []
    #: Authorised 2026-09-10: one record per quote segment `relocate_evidence_quotes`
    #: repointed at the source's own text before the guards above ran, each
    #: ``{"type": "quote_relocated", "original", "replacement", "edit_distance",
    #: "chunk_index"}``. ``[]`` when nothing needed relocating (the common case). The slug
    #: ``"quote_relocated"`` is also appended to ``diagnostics`` above whenever this is
    #: non-empty, so a reader of ``diagnostics`` alone still sees that something happened;
    #: this field is where the detail of what happened lives. Additive with a default.
    quote_relocations: list[dict] = []
    #: Task authorisation 2026-09-10 (Part B, the bounded second pass): one record per
    #: model call actually made, in call order -- one entry for the common case, two when
    #: the second pass fired. Each entry is ``{"model_status", "quotes", "machine_reasons",
    #: "diagnostics", "tokens", "fingerprint"}``: the model's own raw status for that call,
    #: the quote(s) that call ended with (after Part A relocation), the guard reasons and
    #: diagnostics that call's own guard run produced, that call's token counts, and its
    #: provider-reported fingerprint. The claim report and the claim-record export (``app.
    #: services.claim_record``) read ``len(passes)`` and ``passes[0]["machine_reasons"]`` to
    #: show a reader that the first pass was demoted and a second pass changed the outcome.
    #: Additive with a default, so a report written before this field existed still parses.
    passes: list[dict] = []


class CitationCoverage(BaseModel):
    """How the citations in a draft were resolved, independent of how they were rendered.

    Produced by `app.services.fulltext.extract_claims_from_document`, which recognises a
    citation through the writer's own sentence-to-reference map, the pre-existing
    author-year regexes, or a numbered-citation resolver, in that order.
    """

    #: Every citation occurrence recognised in the draft, by any of the three sources.
    found: int = 0
    #: Of ``found``, how many resolved to a paper actually in the project library.
    linked: int = 0
    #: How many citations were actually turned into a claim and sent to the verifier
    #: (a mapping-derived key absent from the library is still sent, and still counted
    #: here, exactly as ``no_full_text`` does today; a numbered citation that could not
    #: be resolved to any reference-list entry is not, since there is no key to check).
    sent_to_verifier: int = 0
    #: Of ``found``, how many never resolved to a paper in the library at all.
    unresolved: int = 0
    #: Raw citation text for up to the first 50 unresolved citations, for a human to spot
    #: check; ``unresolved`` itself is the true count even when this list is capped.
    unresolved_citations: list[str] = []
    #: How many of ``found`` came from each source: ``"mapping"`` (the writer's own
    #: citation-link map), ``"author-year"`` (the pre-existing regexes) or ``"numbered"``
    #: (the reference-list resolver). Sums to ``found``.
    by_source: dict[str, int] = {}


class FinalClaimReport(BaseModel):
    """The "ONE report of the final text": the claims that
    survive in the saved, healed draft, every one of them verified by construction."""

    verifications: list[ClaimVerification] = []
    verified_count: int = 0


class ClaimVerificationReport(BaseModel):
    """Aggregated verification results for all claims in a draft.

    ``needs_rewrite`` / ``needs_removal`` are honest labels (product decision D4): claim
    verification NEVER edits the draft, it only reports what a human should change.
    This still describes ``verifications`` in full, exactly as before -- ``healed``
    below is what changed.
    """

    draft_id: UUID
    verifications: list[ClaimVerification]
    verified_count: int
    unsupported_count: int
    nuance_count: int
    abstract_only_count: int
    error_count: int = 0
    needs_rewrite: int = 0
    needs_removal: int = 0
    # Aggregated LLM-call provenance (agent, model_configured, model_reported,
    # system_fingerprints, temperature, prompt_version, calls, input_tokens,
    # output_tokens) and the share of claims whose source had full text available.
    provenance: dict | None = None
    full_text_coverage: float = 0.0
    # Which job produced this report, so the frontend can fetch its exportable claim
    # record (GET /tasks/{task_id}/claim-record). Filled by the API
    # route from the job row itself, not stored in the job's own result JSON, so it is
    # never stale even for reports written before this field existed.
    task_id: UUID | None = None
    #: Informational counts, additive with defaults so a report written before
    #: this field existed still parses.
    #: Verifications with >=1 assertion verdict "contradicted".
    contradicted_count: int = 0
    #: Verifications whose status a guard changed (``model_status is not None and
    #: model_status != status``).
    guarded_count: int = 0
    #: How citations in the draft
    #: were resolved, independent of how they were rendered. Additive with a default, so
    #: a report written before this field existed (or one from ``verify_user_edits``,
    #: which only ever sees plain strings, never a Tiptap document with a reference list)
    #: still parses.
    citation_coverage: CitationCoverage | None = None
    #: ``app.services.fulltext.verify_and_heal_claims`` (the
    #: standalone action for an already-saved draft) actually heals the draft, in
    #: addition to reporting on it, superseding product decision D4's "verification only,
    #: the draft is never modified". ``healed`` is ``True`` on every
    #: report that action produces; ``None`` on an older report, or on a
    #: write job's own result (``generate_section`` sets these same three keys
    #: directly on its own job-result dict, not through this schema, since that result
    #: is read as a plain job JSON, never through this endpoint). ``finalize_stats`` is
    #: the removal/drop counts ``finalize_draft_document``/``finalize_generated_section``
    #: returned; ``final_report`` is the clean, every-row-verified view (the
    #: "ONE report of the final text").
    healed: bool | None = None
    finalize_stats: dict[str, int] | None = None
    final_report: FinalClaimReport | None = None


class PasteFulltextRequest(BaseModel):
    """Request body for pasting full text of a paper."""

    text: str


class FulltextChunkResponse(BaseModel):
    """One paper's full-text chunk, by index: the
    delivered-evidence record (``demo/run_demo.py``) and, more generally, any reader that
    only has ``ClaimVerification.evidence_location`` (a "chunk N (section)" string) need a
    paper's chunk text back, and no existing endpoint returned one.

    ``chunk_index`` is 0-based, into the same filtered order (reference/back-matter chunks
    dropped, ``app.services.fulltext.drop_reference_and_backmatter_chunks``) that
    ``_verify_one`` verifies against and that ``evidence_location``'s 1-based "chunk N"
    numbers -- ``chunk_index == N - 1`` resolves to the same text a claim was verified
    against, for a caller that never passed ``evidence_quotes_by_claim``
    and so never had its chunks reordered (``_evidence_first_chunk_order``'s own
    unreordered-by-default case, which is every claim `verify_and_heal_claims` verifies).

    ``abstract`` is the paper's own abstract, the same on
    every chunk of the same paper regardless of ``chunk_index`` -- not itself a numbered
    chunk (``evidence_location`` never names it as one), but part of the context the
    verifier's own prompt gives the model alongside the numbered chunks, so a quote
    genuinely verbatim only there is real evidence the guard already accepted, not
    something this endpoint's own chunk list should leave a caller with no way to see.
    """

    paper_id: UUID
    chunk_index: int
    #: Total number of chunks in the paper's filtered chunk list, so a caller can stop
    #: asking for a higher index without guessing from repeated 404s.
    chunk_count: int
    section: str | None = None
    text: str
    abstract: str | None = None
