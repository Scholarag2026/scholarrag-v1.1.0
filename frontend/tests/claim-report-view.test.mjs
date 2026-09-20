import { test } from "node:test";
import assert from "node:assert/strict";

import { claimTextView, claimEvidenceQuotes } from "../src/lib/provenance.ts";

// The claim-verification report shows the sentence a claim was cut
// from, and the narrower checked clause only when a citation-link proposition actually
// narrowed it (`claim_sentence` vs `claim_text`, `backend/app/schemas/fulltext.py`).

test("claimTextView shows no checked clause when the sentence itself was checked", () => {
  const out = claimTextView({
    claim_text: "Prior work found this effect.",
    claim_sentence: "Prior work found this effect.",
  });
  assert.deepEqual(out, {
    sentence: "Prior work found this effect.",
    checkedClause: null,
  });
});

test("claimTextView surfaces the narrower checked clause when a proposition narrowed it", () => {
  const out = claimTextView({
    claim_text: "X improves Y",
    claim_sentence: "X improves Y (Smith, 2020), while Z remains contested (Jones, 2021).",
  });
  assert.deepEqual(out, {
    sentence: "X improves Y (Smith, 2020), while Z remains contested (Jones, 2021).",
    checkedClause: "X improves Y",
  });
});

test("claimTextView falls back to claim_text alone when claim_sentence is absent (older report)", () => {
  const out = claimTextView({ claim_text: "Prior work found this effect." });
  assert.deepEqual(out, {
    sentence: "Prior work found this effect.",
    checkedClause: null,
  });
});

test("claimEvidenceQuotes prefers the multi-span list when non-empty", () => {
  assert.deepEqual(
    claimEvidenceQuotes({ evidence_quotes: ["span one.", "span two."], evidence_quote: "span one." }),
    ["span one.", "span two."],
  );
});

test("claimEvidenceQuotes falls back to the single evidence_quote", () => {
  assert.deepEqual(
    claimEvidenceQuotes({ evidence_quotes: [], evidence_quote: "the only span." }),
    ["the only span."],
  );
});

test("claimEvidenceQuotes returns an empty list when neither field is set", () => {
  assert.deepEqual(claimEvidenceQuotes({}), []);
});
