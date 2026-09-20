import { test } from "node:test";
import assert from "node:assert/strict";

import { citationCoverageSummary } from "../src/lib/provenance.ts";

test("citationCoverageSummary returns null without a coverage object", () => {
  assert.equal(citationCoverageSummary(null), null);
  assert.equal(citationCoverageSummary(undefined), null);
});

test("citationCoverageSummary extracts found, linked and unresolved", () => {
  const out = citationCoverageSummary({
    found: 14,
    linked: 14,
    sent_to_verifier: 14,
    unresolved: 0,
    unresolved_citations: [],
    by_source: { mapping: 14, "author-year": 0, numbered: 0 },
  });
  assert.deepEqual(out, { found: 14, linked: 14, unresolved: 0 });
});

test("citationCoverageSummary reports a real unresolved count", () => {
  const out = citationCoverageSummary({
    found: 3,
    linked: 2,
    sent_to_verifier: 3,
    unresolved: 1,
    unresolved_citations: ["(Nguyen, 2021)"],
    by_source: { mapping: 0, "author-year": 3, numbered: 0 },
  });
  assert.deepEqual(out, { found: 3, linked: 2, unresolved: 1 });
});

test("citationCoverageSummary tolerates missing or non-numeric fields", () => {
  assert.deepEqual(citationCoverageSummary({}), { found: 0, linked: 0, unresolved: 0 });
  assert.deepEqual(
    citationCoverageSummary({ found: "12", linked: null, unresolved: undefined }),
    { found: 0, linked: 0, unresolved: 0 },
  );
});
