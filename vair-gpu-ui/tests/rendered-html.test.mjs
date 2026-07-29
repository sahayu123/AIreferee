import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

test("ships the VAIR review console", async () => {
  const [page, layout, styles] = await Promise.all([
    readFile(new URL("app/page.tsx", root), "utf8"),
    readFile(new URL("app/layout.tsx", root), "utf8"),
    readFile(new URL("app/globals.css", root), "utf8"),
  ]);
  assert.match(page, /Virtual Artificial Intelligence Referee/);
  assert.match(page, /HIGH_CONFIDENCE_OVERRIDE|97%/);
  assert.match(page, /≤4% arm gap/);
  assert.match(page, /api\/analyze/);
  assert.match(layout, /VAIR — Handball Project \+ Foul GPU Referee/);
  assert.match(styles, /\.review-grid/);
  assert.doesNotMatch(page, /_sites-preview|codex-preview/);
});
