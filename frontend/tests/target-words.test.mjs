import { test } from "node:test";
import assert from "node:assert/strict";

import {
  clampTargetWords,
  DEFAULT_TARGET_WORDS,
  MIN_TARGET_WORDS,
  MAX_TARGET_WORDS,
} from "../src/lib/writing.ts";

test("bounds match the AI Write dialog's stated range", () => {
  assert.equal(DEFAULT_TARGET_WORDS, 400);
  assert.equal(MIN_TARGET_WORDS, 100);
  assert.equal(MAX_TARGET_WORDS, 3000);
});

test("clampTargetWords keeps an in-range value", () => {
  assert.equal(clampTargetWords(500), 500);
});

test("clampTargetWords floors below the minimum", () => {
  assert.equal(clampTargetWords(50), 100);
});

test("clampTargetWords caps above the maximum", () => {
  assert.equal(clampTargetWords(5000), 3000);
});

test("clampTargetWords defaults an unparsable or missing value", () => {
  assert.equal(clampTargetWords(null), 400);
  assert.equal(clampTargetWords(undefined), 400);
  assert.equal(clampTargetWords(NaN), 400);
});

test("clampTargetWords rounds a fractional value", () => {
  assert.equal(clampTargetWords(250.6), 251);
});
