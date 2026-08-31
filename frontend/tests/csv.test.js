const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const fs = require("node:fs");
const { parseCSV } = require("../js/csv.js");

test("parses a simple header + rows", () => {
  const text = "a,b,c\n1,2,3\n4,5,6\n";
  const rows = parseCSV(text);
  assert.equal(rows.length, 2);
  assert.deepEqual(rows[0], { a: "1", b: "2", c: "3" });
  assert.deepEqual(rows[1], { a: "4", b: "5", c: "6" });
});

test("handles a trailing blank value (blank distance field)", () => {
  const text = "event_id,nearest_osm_flare_m\nEVT1,\nEVT2,1038.2\n";
  const rows = parseCSV(text);
  assert.equal(rows[0].nearest_osm_flare_m, "");
  assert.equal(rows[1].nearest_osm_flare_m, "1038.2");
});

test("handles quoted fields with embedded commas", () => {
  const text = 'a,b\n"hello, world",2\n';
  const rows = parseCSV(text);
  assert.equal(rows[0].a, "hello, world");
  assert.equal(rows[0].b, "2");
});

test("handles escaped quotes inside quoted fields", () => {
  const text = 'a\n"she said ""hi"""\n';
  const rows = parseCSV(text);
  assert.equal(rows[0].a, 'she said "hi"');
});

test("works with no trailing newline at end of file", () => {
  const text = "a,b\n1,2";
  const rows = parseCSV(text);
  assert.equal(rows.length, 1);
  assert.deepEqual(rows[0], { a: "1", b: "2" });
});

test("parses the real gujarat_2026_inference.csv header + first rows without loss", () => {
  const csvPath = path.join(__dirname, "..", "..", "data", "processed", "gujarat_2026_inference.csv");
  if (!fs.existsSync(csvPath)) {
    // Real artifact not present in this checkout -- skip rather than fail.
    return;
  }
  const text = fs.readFileSync(csvPath, "utf8");
  const rows = parseCSV(text);
  assert.ok(rows.length > 0, "expected at least one real event row");
  const first = rows[0];
  assert.ok("event_id" in first);
  assert.ok("predicted_class" in first);
  assert.ok("prob_Crop_Residue" in first);
  assert.ok("class_capability_note" in first);
  assert.equal(first.data_type, "2026_NRT_INFERENCE_NOT_GROUND_TRUTH");
});
