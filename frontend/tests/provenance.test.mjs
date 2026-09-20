import { test } from "node:test";
import assert from "node:assert/strict";

import {
  summarizeProvenance,
  formatCoveragePercent,
  citationAuditCounts,
  flowRows,
  FLOW_KEYS,
  CLAIM_REASON_LABEL_KEYS,
  CLAIM_DIAGNOSTIC_LABEL_KEYS,
  claimReasonLabelKey,
  claimDiagnosticLabelKey,
  NEEDS_REVIEW_REASON_LABEL_KEYS,
  needsReviewReasonLabelKey,
  isGuardStricter,
  unescapeVerificationText,
  secondPassStageSummary,
} from "../src/lib/provenance.ts";

test("summarizeProvenance returns null without a provenance object", () => {
  assert.equal(summarizeProvenance(null), null);
  assert.equal(summarizeProvenance(undefined), null);
});

test("summarizeProvenance joins distinct reported models and formats the rest", () => {
  const out = summarizeProvenance({
    model_configured: "deepseek-chat",
    model_reported: ["deepseek-v4-flash", "deepseek-v4-flash"],
    temperature: 0,
    prompt_version: "sha256:0123456789ab",
  });
  assert.deepEqual(out, {
    models: "deepseek-v4-flash",
    temperature: "0",
    promptVersion: "sha256:0123456789ab",
  });
});

test("summarizeProvenance falls back to the configured model and em dashes", () => {
  const out = summarizeProvenance({ model_configured: "deepseek-chat", model_reported: [] });
  assert.deepEqual(out, { models: "deepseek-chat", temperature: "\u2014", promptVersion: "\u2014" });
});

test("summarizeProvenance accepts a single reported model string", () => {
  const out = summarizeProvenance({ model_reported: "deepseek-v4-flash", temperature: 0.7 });
  assert.equal(out.models, "deepseek-v4-flash");
  assert.equal(out.temperature, "0.7");
});

test("summarizeProvenance with nothing usable shows an em dash for the model", () => {
  assert.equal(summarizeProvenance({}).models, "\u2014");
});

test("formatCoveragePercent rounds and clamps to 0..100, null when absent", () => {
  assert.equal(formatCoveragePercent(0.6667), 67);
  assert.equal(formatCoveragePercent(1), 100);
  assert.equal(formatCoveragePercent(1.2), 100);
  assert.equal(formatCoveragePercent(-0.5), 0);
  assert.equal(formatCoveragePercent(0), 0);
  assert.equal(formatCoveragePercent(undefined), null);
  assert.equal(formatCoveragePercent(null), null);
  assert.equal(formatCoveragePercent(Number.NaN), null);
});

test("citationAuditCounts reads list lengths or numeric counts", () => {
  assert.deepEqual(
    citationAuditCounts({ matched: ["a", "b"], unmatched: ["c"], needs_citation_flags: 1, total: 3 }),
    { matched: 2, unmatched: 1, flags: 1 },
  );
  assert.deepEqual(
    citationAuditCounts({ matched: 4, unmatched: 0, needs_citation_flags: 2 }),
    { matched: 4, unmatched: 0, flags: 2 },
  );
  assert.deepEqual(citationAuditCounts({}), { matched: 0, unmatched: 0, flags: 0 });
  assert.equal(citationAuditCounts(null), null);
  assert.equal(citationAuditCounts(undefined), null);
});

test("flowRows keeps the PRISMA-style order and skips missing counts", () => {
  assert.deepEqual(FLOW_KEYS, [
    "identified",
    "duplicates_removed",
    "stage1_excluded",
    "stage2_excluded",
    "needs_review",
    "unscreened",
    "included",
  ]);
  const rows = flowRows({
    identified: 100,
    duplicates_removed: 5,
    stage1_screened: 95,
    stage1_excluded: 10,
    stage2_screened: 85,
    stage2_excluded: 60,
    needs_review: 8,
    unscreened: 2,
    included: 23,
    rounds: 3,
    stop_reason: "target_reached",
  });
  assert.deepEqual(rows, [
    { key: "identified", value: 100 },
    { key: "duplicates_removed", value: 5 },
    { key: "stage1_excluded", value: 10 },
    { key: "stage2_excluded", value: 60 },
    { key: "needs_review", value: 8 },
    { key: "unscreened", value: 2 },
    { key: "included", value: 23 },
  ]);
  assert.deepEqual(flowRows({ identified: 7, included: "x" }), [{ key: "identified", value: 7 }]);
  assert.deepEqual(flowRows(null), []);
  assert.deepEqual(flowRows(undefined), []);
});

test("flowRows omits needs_review for an older stored result shape (only six flow keys)", () => {
  const rows = flowRows({
    identified: 10,
    duplicates_removed: 1,
    stage1_excluded: 1,
    stage2_excluded: 1,
    unscreened: 1,
    included: 6,
  });
  assert.deepEqual(
    rows.map((r) => r.key),
    ["identified", "duplicates_removed", "stage1_excluded", "stage2_excluded", "unscreened", "included"],
  );
});

// The inclusion-only
// second pass's own job-level stage block, rendered as its own line, separate from the
// per-round table (whose own included_new/needs_review_new stay pass-1-and-guard-only
// provisional counts, never the stage's own demotions).
test("secondPassStageSummary reads the stage block's own counts and wall time", () => {
  const summary = secondPassStageSummary({
    candidates: 37,
    calls: 8,
    demotions: 15,
    unavailable: 1,
    stage_wall_time_s: 379.164,
  });
  assert.deepEqual(summary, {
    candidates: 37,
    calls: 8,
    demotions: 15,
    unavailable: 1,
    wallTimeSeconds: 379.164,
  });
});

test("secondPassStageSummary is null when the job never reached the stage", () => {
  assert.equal(secondPassStageSummary(null), null);
  assert.equal(secondPassStageSummary(undefined), null);
  assert.equal(secondPassStageSummary({ candidates: 0, calls: 0 }), null);
});

test("secondPassStageSummary treats a missing field as zero, not a skipped stage", () => {
  const summary = secondPassStageSummary({ candidates: 5 });
  assert.deepEqual(summary, {
    candidates: 5,
    calls: 0,
    demotions: 0,
    unavailable: 0,
    wallTimeSeconds: 0,
  });
});

// Guard-reason slug -> i18n key, and the "a code guard made this stricter" predicate.

test("claimReasonLabelKey maps every documented guard slug to its i18n key", () => {
  assert.equal(claimReasonLabelKey("attribution_mismatch"), "claimReasonAttributionMismatch");
  assert.equal(claimReasonLabelKey("quote_not_verbatim"), "claimReasonQuoteNotVerbatim");
  assert.equal(
    claimReasonLabelKey("numeric_outside_cited_passage"),
    "claimReasonNumericOutsidePassage",
  );
  assert.equal(
    claimReasonLabelKey("no_full_text_with_chunks"),
    "claimReasonNoFullTextWithChunks",
  );
  assert.equal(
    claimReasonLabelKey("assertion_status_inconsistent"),
    "claimReasonAssertionStatusInconsistent",
  );
});

test("claimReasonLabelKey returns null for an unrecognised slug", () => {
  assert.equal(claimReasonLabelKey("some_future_guard"), null);
  assert.equal(claimReasonLabelKey(""), null);
  // numeric_not_in_source moved entirely to the diagnostic map -- it must not
  // still resolve as a reason (a status-changing slug).
  assert.equal(claimReasonLabelKey("numeric_not_in_source"), null);
});

test("CLAIM_REASON_LABEL_KEYS has exactly the five documented guard slugs", () => {
  assert.deepEqual(
    Object.keys(CLAIM_REASON_LABEL_KEYS).sort(),
    [
      "assertion_status_inconsistent",
      "attribution_mismatch",
      "no_full_text_with_chunks",
      "numeric_outside_cited_passage",
      "quote_not_verbatim",
    ].sort(),
  );
});

// Diagnostics never change
// `status`, so they live in a map disjoint from `CLAIM_REASON_LABEL_KEYS`.

test("claimDiagnosticLabelKey maps both documented diagnostic slugs to their i18n key", () => {
  assert.equal(
    claimDiagnosticLabelKey("numeric_not_in_source"),
    "claimDiagnosticNumericNotInSource",
  );
  assert.equal(
    claimDiagnosticLabelKey("centrality_unmarked"),
    "claimDiagnosticCentralityUnmarked",
  );
});

test("claimDiagnosticLabelKey returns null for an unrecognised slug", () => {
  assert.equal(claimDiagnosticLabelKey("some_future_diagnostic"), null);
  assert.equal(claimDiagnosticLabelKey(""), null);
  // A reason slug is not a diagnostic slug either.
  assert.equal(claimDiagnosticLabelKey("attribution_mismatch"), null);
});

test("CLAIM_DIAGNOSTIC_LABEL_KEYS has exactly the two documented diagnostic slugs", () => {
  assert.deepEqual(
    Object.keys(CLAIM_DIAGNOSTIC_LABEL_KEYS).sort(),
    ["centrality_unmarked", "numeric_not_in_source"].sort(),
  );
});

test("CLAIM_REASON_LABEL_KEYS and CLAIM_DIAGNOSTIC_LABEL_KEYS are exhaustive and disjoint", () => {
  const reasonSlugs = Object.keys(CLAIM_REASON_LABEL_KEYS);
  const diagnosticSlugs = Object.keys(CLAIM_DIAGNOSTIC_LABEL_KEYS);
  const overlap = reasonSlugs.filter((slug) => diagnosticSlugs.includes(slug));
  assert.deepEqual(overlap, [], "a slug must never be both a reason and a diagnostic");
  assert.deepEqual(
    [...reasonSlugs, ...diagnosticSlugs].sort(),
    [
      "assertion_status_inconsistent",
      "attribution_mismatch",
      "centrality_unmarked",
      "no_full_text_with_chunks",
      "numeric_not_in_source",
      "numeric_outside_cited_passage",
      "quote_not_verbatim",
    ].sort(),
  );
});

// Guard-reason slug -> i18n key
// for a NEEDS_REVIEW record's flagged reason.

test("needsReviewReasonLabelKey maps every documented reason slug to its i18n key", () => {
  assert.equal(
    needsReviewReasonLabelKey("full_text_criterion"),
    "needsReviewReasons.full_text_criterion",
  );
  assert.equal(needsReviewReasonLabelKey("cut_abstract"), "needsReviewReasons.cut_abstract");
  assert.equal(
    needsReviewReasonLabelKey("unanchored_exclude"),
    "needsReviewReasons.unanchored_exclude",
  );
  assert.equal(needsReviewReasonLabelKey("no_abstract"), "needsReviewReasons.no_abstract");
  assert.equal(needsReviewReasonLabelKey("undecidable"), "needsReviewReasons.undecidable");
  assert.equal(
    needsReviewReasonLabelKey("unquoted_criterion"),
    "needsReviewReasons.unquoted_criterion",
  );
});

test("needsReviewReasonLabelKey returns null for an unrecognised slug", () => {
  assert.equal(needsReviewReasonLabelKey("some_future_reason"), null);
  assert.equal(needsReviewReasonLabelKey(""), null);
});

test("NEEDS_REVIEW_REASON_LABEL_KEYS has exactly the six documented reason slugs", () => {
  assert.deepEqual(
    Object.keys(NEEDS_REVIEW_REASON_LABEL_KEYS).sort(),
    [
      "cut_abstract",
      "full_text_criterion",
      "no_abstract",
      "unanchored_exclude",
      "unquoted_criterion",
      "undecidable",
    ].sort(),
  );
});

test("isGuardStricter is true only when model_status is present and differs from status", () => {
  assert.equal(isGuardStricter("unsupported", "verified"), true);
  assert.equal(isGuardStricter("needs_nuance", "verified"), true);
  assert.equal(isGuardStricter("verified", "verified"), false);
  assert.equal(isGuardStricter("verified", null), false);
  assert.equal(isGuardStricter("verified", undefined), false);
  assert.equal(isGuardStricter(null, null), false);
  assert.equal(isGuardStricter(undefined, undefined), false);
});

// ── unescapeVerificationText (defensive display for old-shape results) ──

test("unescapeVerificationText collapses doubly-escaped quotes, apostrophes and newlines", () => {
  assert.equal(unescapeVerificationText('He said \\"hi\\" and left.'), 'He said "hi" and left.');
  assert.equal(unescapeVerificationText("It\\'s fine."), "It's fine.");
  assert.equal(unescapeVerificationText("line one\\nline two"), "line one\nline two");
});

test("unescapeVerificationText returns an empty string for null, undefined or empty input", () => {
  assert.equal(unescapeVerificationText(null), "");
  assert.equal(unescapeVerificationText(undefined), "");
  assert.equal(unescapeVerificationText(""), "");
});

test("unescapeVerificationText is a no-op on already-clean text", () => {
  const clean = "The paper's results support this claim directly.";
  assert.equal(unescapeVerificationText(clean), clean);
});

test("unescapeVerificationText strips wrapping quotes only when asked", () => {
  assert.equal(
    unescapeVerificationText('\\"wrapped\\"', { stripWrappingQuotes: true }),
    "wrapped",
  );
  // Same input, no flag (the evidence_quote call site): escapes collapse, wrapping marks
  // stay -- they mark the quoted span for a reader.
  assert.equal(unescapeVerificationText('\\"wrapped\\"'), '"wrapped"');
});

test("unescapeVerificationText does not strip a lone, unmatched leading quote", () => {
  const text = '"quoted mid-way through, not wrapped';
  assert.equal(unescapeVerificationText(text, { stripWrappingQuotes: true }), text);
});

// Mirrors the backend's
// `_strip_wrapping_quotes` fix. Strip only when the whole string is ONE quoted span --
// never when the first/last characters merely happen to be a matched pair around two
// separate quoted spans joined by plain text.

test("unescapeVerificationText does not strip across two separate quoted spans", () => {
  assert.equal(
    unescapeVerificationText('"a" and "b"', { stripWrappingQuotes: true }),
    '"a" and "b"',
  );
  assert.equal(
    unescapeVerificationText("'a' and 'b'", { stripWrappingQuotes: true }),
    "'a' and 'b'",
  );
  assert.equal(
    unescapeVerificationText("“a” and “b”", { stripWrappingQuotes: true }),
    "“a” and “b”",
  );
});

test("unescapeVerificationText still strips a single quoted span containing other punctuation", () => {
  assert.equal(
    unescapeVerificationText('"The paper supports this, plainly."', {
      stripWrappingQuotes: true,
    }),
    "The paper supports this, plainly.",
  );
});

// Excerpt copied verbatim (including the literal backslash-quote defect) from the
// `supported-1` explanation in `demo/expected/claim_report.json` before the backend fix.
const DEFECTIVE_DEMO_EXPLANATION =
  "Every atomic assertion in the claim is directly supported by the source text. The " +
  "paper is a synthesis of naturalistic classroom studies where the type and amount of " +
  "feedback were not manipulated or controlled (\\\"there is not yet a synthesis of " +
  "naturalistic classroom studies where the type and amount of feedback provided on " +
  "students' writing performance is not manipulated or controlled\\\").";

test("unescapeVerificationText fixes the defective demo explanation fixture", () => {
  const cleaned = unescapeVerificationText(DEFECTIVE_DEMO_EXPLANATION, {
    stripWrappingQuotes: true,
  });
  assert.equal(cleaned.includes("\\"), false);
  assert.match(
    cleaned,
    /\("there is not yet a synthesis of naturalistic classroom studies.*controlled"\)\.$/,
  );
});
