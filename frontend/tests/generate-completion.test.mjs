import { test } from "node:test";
import assert from "node:assert/strict";

import { describeGenerateCompletion } from "../src/lib/writing.ts";

// A section that finalizes to nothing (every
// cited sentence unverified, every uncited sentence tagged "finding") is still a
// completed job -- `generateTaskStatus.result.content` is the empty string, not
// missing -- but the drafts-tab completion effect used to be gated on `result?.content`
// being truthy, so an empty result ran neither branch: no draft-query invalidation, no
// message, and the editor kept showing text the backend had already deleted.

test("a job still running or with no status yet decides nothing", () => {
  assert.equal(describeGenerateCompletion(undefined, null), null);
  assert.equal(describeGenerateCompletion("running", null), null);
});

test("a completed job with no result yet decides nothing", () => {
  assert.equal(describeGenerateCompletion("completed", null), null);
  assert.equal(describeGenerateCompletion("completed", undefined), null);
});

test("a failed job decides nothing here -- the caller's own failed branch handles it", () => {
  assert.equal(
    describeGenerateCompletion("failed", { content: "", section_type: "methods", papers_used: 0 }),
    null,
  );
});

test("a completed job with real content is a normal generation", () => {
  assert.deepEqual(
    describeGenerateCompletion("completed", {
      content: "Tutoring improves outcomes (Smith, 2020).",
      section_type: "literature_review",
      papers_used: 3,
    }),
    { kind: "generated", sectionType: "literature_review", papersUsed: 3 },
  );
});

test("a completed job whose finalized content is empty is a removal, not nothing", () => {
  assert.deepEqual(
    describeGenerateCompletion("completed", {
      content: "",
      section_type: "literature_review",
      papers_used: 3,
    }),
    { kind: "emptyRemoved", sectionType: "literature_review" },
  );
});
