import { test } from "node:test";
import assert from "node:assert/strict";

import { PRIVACY_NOTICE_URL } from "../src/lib/links.ts";

test("the privacy notice points at PRIVACY.md in the public repository", () => {
  assert.equal(
    PRIVACY_NOTICE_URL,
    "https://github.com/Scholarag2026/scholarrag-v1.1.0/blob/main/PRIVACY.md",
  );
});
