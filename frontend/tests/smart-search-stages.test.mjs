import { test } from "node:test";
import assert from "node:assert/strict";
import { register } from "node:module";

// Pure helpers behind the Smart Search
// panel's per-criterion "Needs full text" checkbox, so the stage-array bookkeeping is
// covered without rendering React. Kept in use-smart-search.ts (this phase's ownership)
// rather than a new lib file.
//
// use-smart-search.ts imports its sibling "../lib/api" without an extension (the style
// every hook file in this codebase uses; TypeScript's "bundler" moduleResolution and
// Next.js's webpack both resolve that to lib/api.ts). Plain `node --experimental-strip-
// types`, unlike a bundler, does not guess extensions for relative specifiers, so
// loading the hook module directly would fail before this file's own tests ever ran.
// Registering a tiny resolve hook that retries a dot-relative, extensionless specifier
// with ".ts" appended -- scoped to this test file's own process, via a data: URL so no
// extra file is needed -- lets the test import and exercise the real, shipped
// functions instead of a re-implementation that could drift from them.
const toTsExtensionLoader = `
  import { extname } from "node:path";
  export async function resolve(specifier, context, nextResolve) {
    if (specifier.startsWith(".") && !extname(specifier)) {
      try {
        return await nextResolve(specifier + ".ts", context);
      } catch {
        // Not a bare .ts sibling (e.g. it really is extension-free some other way) --
        // fall through to normal resolution below.
      }
    }
    return nextResolve(specifier, context);
  }
`;
register(`data:text/javascript,${encodeURIComponent(toTsExtensionLoader)}`, import.meta.url);

const {
  resetCriteriaFlags,
  toggleCriterionFlag,
  criteriaStagesFromFlags,
} = await import("../src/hooks/use-smart-search.ts");

test("resetCriteriaFlags returns an all-false array of the given length", () => {
  assert.deepEqual(resetCriteriaFlags(0), []);
  assert.deepEqual(resetCriteriaFlags(1), [false]);
  assert.deepEqual(resetCriteriaFlags(3), [false, false, false]);
});

test("toggleCriterionFlag flips only the targeted index, without mutating the input", () => {
  const flags = [false, false, false];
  const next = toggleCriterionFlag(flags, 1);
  assert.deepEqual(next, [false, true, false]);
  assert.deepEqual(flags, [false, false, false], "input array must not be mutated");
  assert.deepEqual(toggleCriterionFlag(next, 1), [false, false, false], "toggling twice is a no-op");
});

test("toggleCriterionFlag leaves other flags untouched at either end of the array", () => {
  assert.deepEqual(toggleCriterionFlag([false, false], 0), [true, false]);
  assert.deepEqual(toggleCriterionFlag([false, false], 1), [false, true]);
});

test("criteriaStagesFromFlags sends the empty (all-abstract) shape when nothing is checked", () => {
  assert.deepEqual(criteriaStagesFromFlags([]), []);
  assert.deepEqual(criteriaStagesFromFlags([false]), []);
  assert.deepEqual(criteriaStagesFromFlags([false, false, false]), []);
});

test("criteriaStagesFromFlags maps each checked flag to full_text and the rest to abstract", () => {
  assert.deepEqual(criteriaStagesFromFlags([true]), ["full_text"]);
  assert.deepEqual(criteriaStagesFromFlags([true, false, true]), [
    "full_text",
    "abstract",
    "full_text",
  ]);
  assert.deepEqual(criteriaStagesFromFlags([false, true]), ["abstract", "full_text"]);
});

test("criteriaStagesFromFlags output is always parallel in length to the input, once non-empty", () => {
  const flags = [true, false, false, true, false];
  const stages = criteriaStagesFromFlags(flags);
  assert.equal(stages.length, flags.length);
});
