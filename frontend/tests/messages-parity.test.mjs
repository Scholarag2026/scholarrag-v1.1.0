import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const load = (name) => JSON.parse(readFileSync(join(here, "..", "messages", name), "utf8"));

function flatten(obj, prefix = "", out = {}) {
  for (const [key, value] of Object.entries(obj)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (value && typeof value === "object" && !Array.isArray(value)) flatten(value, path, out);
    else out[path] = value;
  }
  return out;
}

const en = flatten(load("en.json"));
const zh = flatten(load("zh.json"));

test("en.json and zh.json have identical leaf-key sets", () => {
  const onlyEn = Object.keys(en).filter((k) => !(k in zh));
  const onlyZh = Object.keys(zh).filter((k) => !(k in en));
  assert.deepEqual({ onlyEn, onlyZh }, { onlyEn: [], onlyZh: [] });
});

test("every message value is a non-empty string", () => {
  for (const catalog of [en, zh]) {
    for (const [key, value] of Object.entries(catalog)) {
      assert.equal(typeof value, "string", `${key} is not a string`);
      assert.ok(value.length > 0, `${key} is empty`);
    }
  }
});

test("ICU placeholders match between locales", () => {
  const placeholders = (s) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();
  for (const key of Object.keys(en)) {
    if (key in zh) assert.deepEqual(placeholders(zh[key]), placeholders(en[key]), key);
  }
});

test("the v1.1.0 revision strings exist in both catalogs", () => {
  const required = [
    "smartSearch.venueFilter",
    "smartSearch.venueFilterAuto",
    "smartSearch.venueFilterOn",
    "smartSearch.venueFilterOff",
    "smartSearch.venueFilterInactive",
    "smartSearch.screeningRecord",
    "smartSearch.modelLine",
    "smartSearch.downloadScreeningRecord",
    "smartSearch.flow.identified",
    "smartSearch.flow.included",
    // Three-way screening routing adds a needs_review flow key and a
    // "Flagged for review" filter chip.
    "smartSearch.flow.needs_review",
    "smartSearch.filterIncluded",
    "smartSearch.filterNeedsReview",
    "smartSearch.needsReviewHelp",
    "smartSearch.screeningCriterion",
    "smartSearch.screeningQuote",
    // Stage-aware criteria -- the per-criterion "needs full text" marking and the
    // flagged-record reason labels.
    "smartSearch.fullTextOnly",
    "smartSearch.fullTextOnlyHelp",
    "smartSearch.needsReviewReason",
    "smartSearch.needsReviewReasons.full_text_criterion",
    "smartSearch.needsReviewReasons.cut_abstract",
    "smartSearch.needsReviewReasons.no_abstract",
    "smartSearch.needsReviewReasons.unanchored_exclude",
    "smartSearch.needsReviewReasons.undecidable",
    // A deferred full-text inclusion criterion's to-confirm note on an INCLUDE, and the
    // fifth guard reason for a NEEDS_REVIEW that names a full-text exclusion criterion
    // without a verbatim quote.
    "smartSearch.toConfirm",
    "smartSearch.toConfirmHelp",
    "smartSearch.needsReviewReasons.unquoted_criterion",
    // The reserved `TOPIC` criterion id an off-topic EXCLUDE names when no numbered
    // criterion covers the mismatch, rendered as this literal label.
    "smartSearch.criterionTopic",
    "papers.showMore",
    "papers.showLess",
    "verification.recordVerified",
    "verification.metadataMismatch",
    "verification.doiNotFound",
    "verification.noDoi",
    "verification.indicatorWos",
    "verification.indicatorNotWos",
    "verification.indicatorCitations",
    "search.claimProvenanceLine",
    "search.claimFullTextCoverage",
    "search.claimReasonAttributionMismatch",
    "search.claimReasonQuoteNotVerbatim",
    "search.claimReasonNumericOutsidePassage",
    "search.claimReasonNoFullTextWithChunks",
    // `claimModelStatusStricter` (the model-vs-guard "before-and-after" line) was removed
    // on purpose -- the product now shows one final report, not a comparison of a raw
    // pass against a healed one.
    "search.claimGuardCountsLine",
    "search.claimAssertionsToggle",
    "search.claimAssertionSupported",
    "search.claimAssertionContradicted",
    "search.claimAssertionAbsent",
    "search.claimAssertionClaimValue",
    "search.claimAssertionSourceValue",
    // Six keys -- five new, plus the rename of claimReasonNumericNotInSource to
    // claimDiagnosticNumericNotInSource.
    "search.claimDiagnosticsLabel",
    "search.claimDiagnosticsNote",
    "search.claimUnstatedDetails",
    "search.claimReasonAssertionStatusInconsistent",
    "search.claimDiagnosticCentralityUnmarked",
    "search.claimDiagnosticNumericNotInSource",
    "drafts.citationAuditLine",
    "drafts.modelLine",
    "auth.dataNotice",
  ];
  for (const key of required) {
    assert.ok(key in en, `en missing ${key}`);
    assert.ok(key in zh, `zh missing ${key}`);
  }
});
