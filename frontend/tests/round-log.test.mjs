import { test } from "node:test";
import assert from "node:assert/strict";

import { roundLogRows, stopRuleSettings } from "../src/lib/provenance.ts";

// The screening record shows the rounds and their queries from
// `provenance.rounds` and the two stop-rule settings from `provenance.settings`
// (`backend/app/services/smart_search.py`).

test("stopRuleSettings returns null without both settings", () => {
  assert.equal(stopRuleSettings(null), null);
  assert.equal(stopRuleSettings(undefined), null);
  assert.equal(stopRuleSettings({ min_rounds: 3 }), null);
});

test("stopRuleSettings extracts both thresholds", () => {
  assert.deepEqual(stopRuleSettings({ min_rounds: 3, dry_round_patience: 2 }), {
    minRounds: 3,
    dryRoundPatience: 2,
  });
});

test("roundLogRows returns an empty list without a round log", () => {
  assert.deepEqual(roundLogRows(null), []);
  assert.deepEqual(roundLogRows(undefined), []);
});

test("roundLogRows emits one round row and one query row per query, in order", () => {
  const rows = roundLogRows([
    {
      round: 1,
      queries: [
        { query: "translation quality AI", returned: 12, new_unique: 10 },
        { query: "post-editing effort", returned: 5, new_unique: 2 },
      ],
      screened: 12,
      included_new: 3,
      needs_review_new: 1,
    },
    {
      round: 2,
      queries: [{ query: "translation quality AI review", returned: 4, new_unique: 0 }],
      screened: 0,
      included_new: 0,
      needs_review_new: 0,
    },
  ]);

  assert.deepEqual(rows, [
    { kind: "round", round: 1, screened: 12, includedNew: 3, needsReviewNew: 1 },
    { kind: "query", round: 1, query: "translation quality AI", returned: 12, newUnique: 10 },
    { kind: "query", round: 1, query: "post-editing effort", returned: 5, newUnique: 2 },
    { kind: "round", round: 2, screened: 0, includedNew: 0, needsReviewNew: 0 },
    {
      kind: "query",
      round: 2,
      query: "translation quality AI review",
      returned: 4,
      newUnique: 0,
    },
  ]);
});

test("roundLogRows still emits a round row for a round whose queries all failed", () => {
  const rows = roundLogRows([
    { round: 1, queries: [], screened: 0, included_new: 0, needs_review_new: 0 },
  ]);
  assert.deepEqual(rows, [
    { kind: "round", round: 1, screened: 0, includedNew: 0, needsReviewNew: 0 },
  ]);
});
