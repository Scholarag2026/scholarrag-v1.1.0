import { test } from "node:test";
import assert from "node:assert/strict";

import { describeIndicator, describeIndicators } from "../src/lib/indicators.ts";

const wosIn = {
  check_type: "wos_indexed",
  status: "info",
  message: "Journal is indexed in Web of Science Core Collection",
  details: null,
};
const wosOut = {
  check_type: "wos_indexed",
  status: "note",
  message: "Journal not found in Web of Science Core Collection",
  details: null,
};

test("skipped indicators are hidden", () => {
  assert.equal(
    describeIndicator({ check_type: "recency", status: "skipped", message: "No year", details: null }),
    null,
  );
});

test("wos_indexed uses details.collection when present", () => {
  assert.deepEqual(
    describeIndicator({ ...wosIn, details: { collection: "SSCI" } }),
    { kind: "wos", collection: "SSCI" },
  );
  assert.deepEqual(
    describeIndicator({ ...wosIn, details: { wos_collection: "AHCI" } }),
    { kind: "wos", collection: "AHCI" },
  );
});

test("wos_indexed honours details.indexed and the message wording", () => {
  assert.deepEqual(describeIndicator({ ...wosIn, details: { indexed: false } }), { kind: "notWos" });
  assert.deepEqual(describeIndicator(wosOut), { kind: "notWos" });
  assert.deepEqual(describeIndicator(wosIn), { kind: "wos", collection: null });
});

test("wos_indexed takes the collection from the row context when the check has none", () => {
  assert.deepEqual(describeIndicator(wosIn, { wosCollection: "SSCI" }), {
    kind: "wos",
    collection: "SSCI",
  });
  assert.deepEqual(describeIndicator(wosOut, { wosCollection: "SSCI" }), { kind: "notWos" });
});

test("recency yields the year from details or the message", () => {
  assert.deepEqual(
    describeIndicator({ check_type: "recency", status: "info", message: "x", details: { year: 2019 } }),
    { kind: "year", year: 2019 },
  );
  assert.deepEqual(
    describeIndicator({
      check_type: "recency",
      status: "note",
      message: "Published in 2012 \u2014 older than 10 years (14 years ago)",
      details: null,
    }),
    { kind: "year", year: 2012 },
  );
});

test("citation_count yields the count from details or the message", () => {
  assert.deepEqual(
    describeIndicator({
      check_type: "citation_count",
      status: "info",
      message: "x",
      details: { count: 7 },
    }),
    { kind: "citations", count: 7 },
  );
  assert.deepEqual(
    describeIndicator({
      check_type: "citation_count",
      status: "note",
      message: "Only 3 citations \u2014 consider higher-impact sources",
      details: null,
    }),
    { kind: "citations", count: 3 },
  );
  assert.deepEqual(
    describeIndicator({ check_type: "citation_count", status: "info", message: "12 citations" }),
    { kind: "citations", count: 12 },
  );
});

test("unknown indicator types fall back to their message text", () => {
  assert.deepEqual(
    describeIndicator({ check_type: "something_new", status: "info", message: "Hello" }),
    { kind: "text", text: "Hello" },
  );
});

test("describeIndicators drops hidden entries and tolerates missing input", () => {
  const out = describeIndicators(
    [
      wosIn,
      { check_type: "recency", status: "skipped", message: "No year", details: null },
      { check_type: "citation_count", status: "info", message: "7 citations", details: null },
    ],
    { wosCollection: "SSCI" },
  );
  assert.deepEqual(out, [
    { kind: "wos", collection: "SSCI" },
    { kind: "citations", count: 7 },
  ]);
  assert.deepEqual(describeIndicators(undefined), []);
  assert.deepEqual(describeIndicators(null), []);
});
