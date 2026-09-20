import { test } from "node:test";
import assert from "node:assert/strict";

import { downloadBlob, claimRecordFilename, screeningRecordFilename } from "../src/lib/download.ts";

test("screeningRecordFilename derives a csv name from the task id", () => {
  assert.equal(
    screeningRecordFilename("3f9b2c1e-0000-4000-8000-000000000000"),
    "screening-record-3f9b2c1e-0000-4000-8000-000000000000.csv",
  );
});

test("claimRecordFilename derives a csv name from the task id", () => {
  assert.equal(
    claimRecordFilename("3f9b2c1e-0000-4000-8000-000000000000"),
    "claim-record-3f9b2c1e-0000-4000-8000-000000000000.csv",
  );
});

test("downloadBlob creates, clicks and removes an anchor and revokes the object URL", () => {
  const events = [];
  const anchor = {
    href: "",
    download: "",
    click() {
      events.push("click");
    },
    remove() {
      events.push("remove");
    },
  };
  const fakeDoc = {
    createElement(tag) {
      events.push(`create:${tag}`);
      return anchor;
    },
    body: {
      appendChild(el) {
        events.push("append");
        assert.equal(el, anchor);
      },
    },
  };
  const revoked = [];
  const fakeUrl = {
    createObjectURL() {
      return "blob:fake";
    },
    revokeObjectURL(u) {
      revoked.push(u);
    },
  };

  downloadBlob(new Blob(["a,b\n"], { type: "text/csv" }), "record.csv", {
    doc: fakeDoc,
    url: fakeUrl,
  });

  assert.equal(anchor.href, "blob:fake");
  assert.equal(anchor.download, "record.csv");
  assert.deepEqual(events, ["create:a", "append", "click", "remove"]);
  assert.deepEqual(revoked, ["blob:fake"]);
});
