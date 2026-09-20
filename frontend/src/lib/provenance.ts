/**
 * Pure helpers for rendering LLM-call provenance and screening-flow counts.
 *
 * The backend records, for every LLM-backed job, which model the provider actually
 * served (`model_reported`), the sampling temperature and a hash of the system prompt
 * (`prompt_version`). These helpers turn that record into display strings and never
 * throw on partial data, because the fields are optional at runtime.
 */

const EM_DASH = "\u2014";

export interface ModelProvenanceLike {
  model_configured?: string | null;
  model_reported?: string[] | string | null;
  system_fingerprints?: string[] | null;
  temperature?: number | null;
  prompt_version?: string | null;
  calls?: number | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
}

export interface ProvenanceSummary {
  /** Distinct models the provider reported, comma-joined; configured model as fallback. */
  models: string;
  temperature: string;
  promptVersion: string;
}

export function summarizeProvenance(
  provenance: ModelProvenanceLike | null | undefined,
): ProvenanceSummary | null {
  if (!provenance) return null;

  const reported = Array.isArray(provenance.model_reported)
    ? provenance.model_reported
    : provenance.model_reported
      ? [provenance.model_reported]
      : [];
  const distinct = Array.from(new Set(reported.filter((m) => typeof m === "string" && m)));
  const models = distinct.length > 0 ? distinct.join(", ") : provenance.model_configured || EM_DASH;

  const temperature =
    typeof provenance.temperature === "number" && Number.isFinite(provenance.temperature)
      ? String(provenance.temperature)
      : EM_DASH;

  return {
    models,
    temperature,
    promptVersion: provenance.prompt_version || EM_DASH,
  };
}

/** 0..1 coverage ratio to a whole percent, clamped; null when the backend did not send one. */
export function formatCoveragePercent(coverage: number | null | undefined): number | null {
  if (typeof coverage !== "number" || !Number.isFinite(coverage)) return null;
  return Math.round(Math.min(1, Math.max(0, coverage)) * 100);
}

export interface CitationAuditLike {
  matched?: string[] | number | null;
  unmatched?: string[] | number | null;
  needs_citation_flags?: number | null;
  total?: number | null;
}

export interface CitationAuditCounts {
  matched: number;
  unmatched: number;
  flags: number;
}

function countOf(value: string[] | number | null | undefined): number {
  if (Array.isArray(value)) return value.length;
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

export function citationAuditCounts(
  audit: CitationAuditLike | null | undefined,
): CitationAuditCounts | null {
  if (!audit) return null;
  return {
    matched: countOf(audit.matched),
    unmatched: countOf(audit.unmatched),
    flags: countOf(audit.needs_citation_flags),
  };
}

/**
 * How the citations in a verified draft were resolved, independent of how they were
 * rendered.
 */
export interface CitationCoverageLike {
  found?: number | null;
  linked?: number | null;
  unresolved?: number | null;
}

export interface CitationCoverageSummary {
  found: number;
  linked: number;
  unresolved: number;
}

function numberOf(value: number | null | undefined): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

/** `null` when the report carries no citation_coverage at all (an older cached report,
 * or one from `verify_user_edits`), so the caller can skip the segment entirely rather
 * than render a "0 found, 0 linked" line that never happened. */
export function citationCoverageSummary(
  coverage: CitationCoverageLike | null | undefined,
): CitationCoverageSummary | null {
  if (!coverage) return null;
  return {
    found: numberOf(coverage.found),
    linked: numberOf(coverage.linked),
    unresolved: numberOf(coverage.unresolved),
  };
}

/** Flow counts shown in the screening record, in reporting order (PRISMA-compatible).
 * "needs_review" sits between the stage2 exclusions and the unscreened count,
 * matching the backend's flow-reconciliation identity in
 * `app/services/smart_search.py`. */
export const FLOW_KEYS = [
  "identified",
  "duplicates_removed",
  "stage1_excluded",
  "stage2_excluded",
  "needs_review",
  "unscreened",
  "included",
] as const;

export type FlowKey = (typeof FLOW_KEYS)[number];

export interface FlowRow {
  key: FlowKey;
  value: number;
}

/**
 * Why the screener's guard flagged a record NEEDS_REVIEW, mapped to the i18n key under
 * `smartSearch.needsReviewReasons` that names it for a human reader. `undecidable` is not
 * a guard demotion -- it marks a genuine model NEEDS_REVIEW the guard never touched -- but
 * shares the same lookup and rendering path as the guard reasons, so one map covers all
 * six. `unquoted_criterion` is a guard reason: a NEEDS_REVIEW that names a full-text
 * exclusion criterion without backing it with a verbatim quote of contrary evidence.
 * `no_abstract` is a guard reason too -- an EXCLUDE on a record with no abstract at all is
 * always demoted with this slug -- as well as the pre-existing fallback label for a
 * guard-silent NEEDS_REVIEW on a record with no abstract; the two cases share the slug
 * because they describe the same thing (a record with no abstract cannot be decided) and
 * the backend already keys its own `needs_review_by_reason` bucket the same way.
 */
export const NEEDS_REVIEW_REASON_LABEL_KEYS: Record<string, string> = {
  full_text_criterion: "needsReviewReasons.full_text_criterion",
  cut_abstract: "needsReviewReasons.cut_abstract",
  unanchored_exclude: "needsReviewReasons.unanchored_exclude",
  unquoted_criterion: "needsReviewReasons.unquoted_criterion",
  no_abstract: "needsReviewReasons.no_abstract",
  undecidable: "needsReviewReasons.undecidable",
};

/**
 * i18n message key for a needs-review reason slug, or `null` for a slug this build does not
 * recognise. Mirrors `claimReasonLabelKey` exactly: callers should fall back to rendering
 * the raw slug rather than skip it, so an unrecognised reason is still visible.
 */
export function needsReviewReasonLabelKey(slug: string): string | null {
  return NEEDS_REVIEW_REASON_LABEL_KEYS[slug] ?? null;
}

export function flowRows(flow: object | null | undefined): FlowRow[] {
  if (!flow) return [];
  const record = flow as Record<string, unknown>;
  const rows: FlowRow[] = [];
  for (const key of FLOW_KEYS) {
    const value = record[key];
    if (typeof value === "number" && Number.isFinite(value)) rows.push({ key, value });
  }
  return rows;
}

/**
 * Machine-readable guard slugs (`ClaimVerification.machine_reasons`) mapped to the i18n
 * key that names them for a human reader. Kept as a plain object, not a switch, so
 * `messages-parity.test.mjs`'s required-key list and this map can be diffed against each
 * other in a test.
 *
 * This map holds only slugs that can change a claim's `status` (the "automated check
 * made this stricter" row). A slug that never changes status belongs in
 * `CLAIM_DIAGNOSTIC_LABEL_KEYS` instead, never in both -- see the disjointness test in
 * `provenance.test.mjs`.
 */
export const CLAIM_REASON_LABEL_KEYS: Record<string, string> = {
  attribution_mismatch: "claimReasonAttributionMismatch",
  quote_not_verbatim: "claimReasonQuoteNotVerbatim",
  numeric_outside_cited_passage: "claimReasonNumericOutsidePassage",
  no_full_text_with_chunks: "claimReasonNoFullTextWithChunks",
  assertion_status_inconsistent: "claimReasonAssertionStatusInconsistent",
};

/**
 * Machine-readable slugs from `ClaimVerification.diagnostics`, a field that is always a
 * sibling of `machine_reasons` and never one of its members (backend `DIAGNOSTIC_SLUGS`,
 * `app/services/fulltext.py`). A diagnostic never changes a claim's status; it is
 * rendered as a separate, neutral row so it can never be mistaken for something that
 * made the check stricter.
 */
export const CLAIM_DIAGNOSTIC_LABEL_KEYS: Record<string, string> = {
  numeric_not_in_source: "claimDiagnosticNumericNotInSource",
  centrality_unmarked: "claimDiagnosticCentralityUnmarked",
};

/**
 * i18n message key for a guard slug, or `null` for a slug this build does not recognise
 * (e.g. a newer backend added a guard the frontend has not shipped a label for yet).
 * Callers should fall back to rendering the raw slug rather than skip it, so an
 * unrecognised reason is still visible instead of silently dropped.
 */
export function claimReasonLabelKey(slug: string): string | null {
  return CLAIM_REASON_LABEL_KEYS[slug] ?? null;
}

/** i18n message key for a diagnostic slug (`ClaimVerification.diagnostics`), or `null`
 * for a slug this build does not recognise. Mirrors `claimReasonLabelKey` exactly,
 * kept as a separate function so a reason and a diagnostic can never be looked up
 * from the same map by mistake. */
export function claimDiagnosticLabelKey(slug: string): string | null {
  return CLAIM_DIAGNOSTIC_LABEL_KEYS[slug] ?? null;
}

/**
 * True when a code guard, not the model, produced the final `status`: `model_status` is
 * present and differs from the final `status`. Mirrors the backend's own `guarded_count`
 * predicate (`backend/app/schemas/fulltext.py::ClaimVerificationReport.guarded_count`)
 * exactly, so the per-claim "automated check made this stricter" line and the
 * report-level guarded count can never disagree about which claims were guarded.
 */
export function isGuardStricter(
  status: string | null | undefined,
  modelStatus: string | null | undefined,
): boolean {
  return !!modelStatus && modelStatus !== status;
}

/**
 * A report stored before the backend started normalising model output
 * (`app/services/fulltext.py`) may still carry the model's literal, doubly-escaped
 * quote/apostrophe/newline sequences in `explanation` or `evidence_quote` -- e.g. the two
 * raw characters ``\`` + ``"`` where a single ``"`` belongs. This is a defensive,
 * display-only unescape for those old-shape results: a no-op on already-clean text, so it
 * is always safe to run. `stripWrappingQuotes` mirrors the backend's `explanation`-only
 * behaviour: a model that quoted its whole answer (e.g. `"The paper supports this."`) has
 * its outer quote pair removed.
 *
 * Mirroring the backend's `_strip_wrapping_quotes` (`app/services/fulltext.py`):
 * stripping only fires when the whole string is one quoted span, i.e. when the inner text
 * (`out.slice(1, -1)`) contains neither the opening nor the closing mark. This keeps a
 * string holding two separate quoted spans joined by plain text (e.g. `"a" and "b"`)
 * intact, rather than losing its two real quote marks and gaining a stray one on each
 * side (`a" and "b`).
 */
const WRAPPING_QUOTE_PAIRS: Record<string, string> = {
  '"': '"',
  "'": "'",
  "“": "”",
  "‘": "’",
};

export function unescapeVerificationText(
  text: string | null | undefined,
  options: { stripWrappingQuotes?: boolean } = {},
): string {
  if (!text) return "";
  let out = text.replace(/\\"/g, '"').replace(/\\'/g, "'").replace(/\\n/g, "\n");
  if (options.stripWrappingQuotes && out.length >= 2) {
    const open = out[0];
    const close = WRAPPING_QUOTE_PAIRS[open];
    if (close && out[out.length - 1] === close) {
      const inner = out.slice(1, -1);
      if (!inner.includes(open) && !inner.includes(close)) {
        out = inner;
      }
    }
  }
  return out;
}

/**
 * The claim-verification report shows the sentence a claim was cut from, and the
 * narrower checked clause only when a citation-link proposition actually narrowed it.
 * `claim_sentence` is set by the service (`backend/app/services/fulltext.py`) and equals
 * `claim_text` whenever nothing narrowed the claim; an older cached report never carrying
 * `claim_sentence` at all falls back to showing `claim_text` alone, with no checked
 * clause (there is no way to tell whether it was narrowed).
 */
export interface ClaimTextLike {
  claim_text: string;
  claim_sentence?: string | null;
}

export interface ClaimTextView {
  /** The full sentence the claim was cut from. */
  sentence: string;
  /** The narrower clause actually checked, or `null` when the sentence itself was checked. */
  checkedClause: string | null;
}

export function claimTextView(claim: ClaimTextLike): ClaimTextView {
  const sentence = claim.claim_sentence || claim.claim_text;
  const checkedClause =
    claim.claim_sentence && claim.claim_text && claim.claim_text !== claim.claim_sentence
      ? claim.claim_text
      : null;
  return { sentence, checkedClause };
}

/**
 * Every verbatim evidence span a verification carries, newest/richest first:
 * `evidence_quotes` (every span the model composed) when present and non-empty,
 * else the single `evidence_quote` as a one-item list, else an empty list. Shared by
 * every reader of a `ClaimVerification` so the same fallback rule is never duplicated.
 */
export interface ClaimEvidenceLike {
  evidence_quotes?: string[] | null;
  evidence_quote?: string | null;
}

export function claimEvidenceQuotes(claim: ClaimEvidenceLike): string[] {
  if (claim.evidence_quotes && claim.evidence_quotes.length > 0) return claim.evidence_quotes;
  return claim.evidence_quote ? [claim.evidence_quote] : [];
}

/**
 * The two Smart Search stop-rule settings a run applied
 * (`backend/app/services/smart_search.py`, `provenance.settings`). `null` when the
 * report carries neither (an older cached result, from before this setting existed).
 */
export interface SmartSearchSettingsLike {
  min_rounds?: number | null;
  dry_round_patience?: number | null;
}

export interface StopRuleSettings {
  minRounds: number;
  dryRoundPatience: number;
}

export function stopRuleSettings(
  settings: SmartSearchSettingsLike | null | undefined,
): StopRuleSettings | null {
  if (!settings) return null;
  if (
    typeof settings.min_rounds !== "number" ||
    typeof settings.dry_round_patience !== "number"
  ) {
    return null;
  }
  return { minRounds: settings.min_rounds, dryRoundPatience: settings.dry_round_patience };
}

/**
 * The per-round query log (`provenance.rounds`,
 * `backend/app/services/smart_search.py`), flattened into rows a plain HTML table can
 * render directly -- one "round" row per round (its screened/included/flagged totals),
 * followed by one "query" row per query that round issued, in the order the round
 * generated them. A round with no queries at all still gets its "round" row, so a round
 * that found nothing is not silently missing from the table.
 */
export interface SmartSearchRoundQueryLike {
  query?: string | null;
  returned?: number | null;
  new_unique?: number | null;
}

export interface SmartSearchRoundLogLike {
  round?: number | null;
  queries?: SmartSearchRoundQueryLike[] | null;
  screened?: number | null;
  included_new?: number | null;
  needs_review_new?: number | null;
}

export type RoundLogRow =
  | {
      kind: "round";
      round: number;
      screened: number;
      includedNew: number;
      needsReviewNew: number;
    }
  | {
      kind: "query";
      round: number;
      query: string;
      returned: number;
      newUnique: number;
    };

export function roundLogRows(
  rounds: SmartSearchRoundLogLike[] | null | undefined,
): RoundLogRow[] {
  if (!rounds) return [];
  const rows: RoundLogRow[] = [];
  for (const entry of rounds) {
    const round = numberOf(entry.round);
    rows.push({
      kind: "round",
      round,
      screened: numberOf(entry.screened),
      includedNew: numberOf(entry.included_new),
      needsReviewNew: numberOf(entry.needs_review_new),
    });
    for (const q of entry.queries ?? []) {
      rows.push({
        kind: "query",
        round,
        query: q.query || "",
        returned: numberOf(q.returned),
        newUnique: numberOf(q.new_unique),
      });
    }
  }
  return rows;
}

/**
 * The inclusion-only second pass runs once per job, after every search round has
 * finished, as its own concurrent stage (`provenance.screener_second_pass`) -- never
 * attributed to any one round. Rendered as its own separate line, distinct from the
 * per-round query log
 * `roundLogRows` above renders: that table's own `included_new`/`needs_review_new` counts
 * stay the pass-1-and-guard-only provisional counts they always were, unaffected by
 * whichever round happened to run last when the stage's own demotions landed.
 */
export interface SecondPassStageLike {
  candidates?: number | null;
  calls?: number | null;
  demotions?: number | null;
  unavailable?: number | null;
  stage_wall_time_s?: number | null;
}

export interface SecondPassStageSummary {
  candidates: number;
  calls: number;
  demotions: number;
  unavailable: number;
  wallTimeSeconds: number;
}

/** `null` when the job never reached the stage at all (no provisional INCLUDE survived the
 * batch pass's own guard and the two deterministic post-model devices), so the caller can
 * skip the line entirely rather than render an all-zero one that never ran. */
export function secondPassStageSummary(
  secondPass: SecondPassStageLike | null | undefined,
): SecondPassStageSummary | null {
  if (!secondPass || numberOf(secondPass.candidates) <= 0) return null;
  return {
    candidates: numberOf(secondPass.candidates),
    calls: numberOf(secondPass.calls),
    demotions: numberOf(secondPass.demotions),
    unavailable: numberOf(secondPass.unavailable),
    wallTimeSeconds: numberOf(secondPass.stage_wall_time_s),
  };
}
