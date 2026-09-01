const test = require("node:test");
const assert = require("node:assert/strict");
const fmt = require("../js/format.js");

test("formatDistanceM renders blank as an explicit non-zero placeholder", () => {
  assert.equal(fmt.formatDistanceM(""), "— (none within search radius)");
  assert.equal(fmt.formatDistanceM(undefined), "— (none within search radius)");
});

test("formatDistanceM never silently renders a blank as 0", () => {
  const out = fmt.formatDistanceM("");
  assert.notEqual(out, "0 m");
});

test("formatDistanceM formats a real numeric distance", () => {
  assert.equal(fmt.formatDistanceM("1038.206656159879"), "1,038 m");
});

test("formatPercent formats a probability", () => {
  assert.equal(fmt.formatPercent("0.9971"), "99.7%");
  assert.equal(fmt.formatPercent(""), "—");
});

test("formatClusterOverlap handles blank vs present overlap", () => {
  assert.equal(fmt.formatClusterOverlap({ overlaps_cluster_id: "" }), "No persistent-cluster overlap.");
  assert.equal(
    fmt.formatClusterOverlap({ overlaps_cluster_id: "23", overlaps_cluster_recurrence_strength: "Moderate" }),
    "Overlaps persistent cluster #23 (recurrence: Moderate)."
  );
});

test("isStale is false just under the threshold and true just over it", () => {
  const now = new Date("2026-09-01T00:00:00Z");
  const justUnder = new Date(now.getTime() - (fmt.STALE_THRESHOLD_SECONDS - 60) * 1000).toISOString();
  const justOver = new Date(now.getTime() - (fmt.STALE_THRESHOLD_SECONDS + 60) * 1000).toISOString();
  assert.equal(fmt.isStale(justUnder, now), false);
  assert.equal(fmt.isStale(justOver, now), true);
});

test("isStale reflects the real freshness timestamp relative to a later now", () => {
  // Real value observed in data/processed/gujarat_2026_inference_freshness.json
  const realLastUpdated = "2026-08-31T18:11:37.576935+00:00";
  const wellPast = new Date("2026-09-02T00:00:00Z"); // more than 2h later
  assert.equal(fmt.isStale(realLastUpdated, wellPast), true);
});

test("formatRelativeTime covers minute/hour/day ranges", () => {
  assert.equal(fmt.formatRelativeTime(30), "less than a minute ago");
  assert.equal(fmt.formatRelativeTime(90), "1 minute ago");
  assert.equal(fmt.formatRelativeTime(4000), "1 hour ago");
  assert.equal(fmt.formatRelativeTime(200000), "2 days ago");
});

test("sortByStartDateDesc orders most recent first", () => {
  const rows = [{ start_date: "2026-01-01" }, { start_date: "2026-06-15" }, { start_date: "2026-03-02" }];
  const sorted = fmt.sortByStartDateDesc(rows);
  assert.deepEqual(sorted.map((r) => r.start_date), ["2026-06-15", "2026-03-02", "2026-01-01"]);
});

test("applyFilters: class filter", () => {
  const rows = [{ predicted_class: "Gas_Flare", status: "closed", land_cover_class: "Built-up", start_date: "2026-01-01", event_id: "EVT2026_000001" },
                { predicted_class: "Industrial", status: "closed", land_cover_class: "Built-up", start_date: "2026-01-01", event_id: "EVT2026_000002" }];
  const out = fmt.applyFilters(rows, { classes: ["Gas_Flare"] });
  assert.equal(out.length, 1);
  assert.equal(out[0].predicted_class, "Gas_Flare");
});

test("applyFilters: search matches event_id by prefix, case-insensitive", () => {
  const rows = [{ predicted_class: "Gas_Flare", status: "closed", land_cover_class: "", start_date: "2026-01-01", event_id: "EVT2026_000042" }];
  assert.equal(fmt.applyFilters(rows, { search: "evt2026_0000" }).length, 1);
  assert.equal(fmt.applyFilters(rows, { search: "000042" }).length, 0); // not a prefix match
});

test("applyFilters: date range on start_date", () => {
  const rows = [{ predicted_class: "Gas_Flare", status: "closed", land_cover_class: "", start_date: "2026-03-10", event_id: "EVT2026_000001" }];
  assert.equal(fmt.applyFilters(rows, { startDateFrom: "2026-01-01", startDateTo: "2026-02-01" }).length, 0);
  assert.equal(fmt.applyFilters(rows, { startDateFrom: "2026-01-01", startDateTo: "2026-12-31" }).length, 1);
});

test("TRAINED_CLASSES is exactly the 4 real trained classes, alphabetical", () => {
  assert.deepEqual(fmt.TRAINED_CLASSES, ["Crop_Residue", "Forest_Wildfire", "Gas_Flare", "Industrial"]);
});

test("CLASS_COLORS defines a distinct color for every trained class, no severity duplication", () => {
  const colors = fmt.TRAINED_CLASSES.map((c) => fmt.CLASS_COLORS[c]);
  assert.equal(new Set(colors).size, 4, "all 4 class colors must be distinct");
});

test("POLL_INTERVAL_MS is approximately 60 seconds (2026-09 NRT-operation milestone)", () => {
  assert.equal(fmt.POLL_INTERVAL_MS, 60000);
});

test("STALE_THRESHOLD_SECONDS tracks the 30-minute scheduler interval, not a stale hourly constant", () => {
  // ops/com.thermoscope.nrt_update.plist StartInterval=1800; threshold is
  // defined as 2 missed cycles, so this must be 3600, not the old 7200.
  assert.equal(fmt.STALE_THRESHOLD_SECONDS, 3600);
});

test("CLASS_COLORS_UI has the same 4 keys as CLASS_COLORS and stays distinct", () => {
  assert.deepEqual(Object.keys(fmt.CLASS_COLORS_UI).sort(), Object.keys(fmt.CLASS_COLORS).sort());
  const colors = fmt.TRAINED_CLASSES.map((c) => fmt.CLASS_COLORS_UI[c]);
  assert.equal(new Set(colors).size, 4, "all 4 dark-UI class colors must be distinct");
});

test("addDaysISO adds/subtracts days including across month and year boundaries", () => {
  assert.equal(fmt.addDaysISO("2026-08-31", -1), "2026-08-30");
  assert.equal(fmt.addDaysISO("2026-09-01", -1), "2026-08-31");
  assert.equal(fmt.addDaysISO("2026-01-01", -1), "2025-12-31");
  assert.equal(fmt.addDaysISO("2026-08-24", -6), "2026-08-18");
});

test("computeTemporalRange: today is a single-day window", () => {
  assert.deepEqual(fmt.computeTemporalRange("2026-09-01", "today"), { from: "2026-09-01", to: "2026-09-01" });
});

test("computeTemporalRange: 24h is the date-level equivalent (today + preceding day), not a fabricated rolling window", () => {
  assert.deepEqual(fmt.computeTemporalRange("2026-09-01", "24h"), { from: "2026-08-31", to: "2026-09-01" });
});

test("computeTemporalRange: 7d spans today plus the preceding 6 calendar days (7 days total)", () => {
  const range = fmt.computeTemporalRange("2026-09-01", "7d");
  assert.deepEqual(range, { from: "2026-08-26", to: "2026-09-01" });
  // exactly 7 calendar days inclusive
  const days = (new Date(range.to) - new Date(range.from)) / 86400000 + 1;
  assert.equal(days, 7);
});

test("computeTemporalRange: calendar returns null (caller supplies explicit dates, nothing computed)", () => {
  assert.equal(fmt.computeTemporalRange("2026-09-01", "calendar"), null);
});

test("dateBounds reflects the real min/max start_date present, not an invented range", () => {
  const rows = [{ start_date: "2026-03-10" }, { start_date: "2026-01-05" }, { start_date: "2026-06-20" }];
  assert.deepEqual(fmt.dateBounds(rows), { min: "2026-01-05", max: "2026-06-20" });
});

test("dateBounds returns null for an empty dataset rather than fabricating a range", () => {
  assert.equal(fmt.dateBounds([]), null);
});

test("nextScheduledCheckUTC adds exactly the real scheduler interval (1800s), not a hardcoded guess", () => {
  const next = fmt.nextScheduledCheckUTC("2026-08-31T19:16:58.000Z");
  assert.equal(next.toISOString(), "2026-08-31T19:46:58.000Z");
});

test("formatClockIST converts UTC to IST (UTC+5:30), no DST", () => {
  // 19:16 UTC -> 00:46 IST (next calendar day)
  assert.equal(fmt.formatClockIST("2026-08-31T19:16:00.000Z"), "00:46 IST");
  // 04:06 UTC -> 09:36 IST (same calendar day)
  assert.equal(fmt.formatClockIST("2026-09-01T04:06:00.000Z"), "09:36 IST");
});

test("formatDateIST reflects the IST calendar day, which can differ from the UTC calendar day", () => {
  // 19:16 UTC on Aug 31 is already Sep 1 in IST
  assert.equal(fmt.formatDateIST("2026-08-31T19:16:00.000Z"), "01 Sep 2026");
});

test("formatCalendarDate formats a date-only (no time-of-day) value without inventing a clock time", () => {
  assert.equal(fmt.formatCalendarDate("2026-08-31"), "31 Aug 2026");
  assert.equal(fmt.formatCalendarDate(""), "—");
});

test("isValidGujaratCoordinate accepts real in-bounds coordinates", () => {
  // EVT2026_003488's real centroid (see data/processed/gujarat_2026_inference.csv)
  assert.equal(fmt.isValidGujaratCoordinate(21.1013, 72.6377), true);
});

test("isValidGujaratCoordinate accepts a coordinate at the exact bbox edge", () => {
  assert.equal(fmt.isValidGujaratCoordinate(fmt.GUJARAT_BOUNDS.latMin, fmt.GUJARAT_BOUNDS.lonMin), true);
  assert.equal(fmt.isValidGujaratCoordinate(fmt.GUJARAT_BOUNDS.latMax, fmt.GUJARAT_BOUNDS.lonMax), true);
});

test("isValidGujaratCoordinate rejects coordinates outside the pipeline's own bounding box", () => {
  assert.equal(fmt.isValidGujaratCoordinate(30.0, 72.0), false); // lat too high
  assert.equal(fmt.isValidGujaratCoordinate(21.0, 80.0), false); // lon too high
  assert.equal(fmt.isValidGujaratCoordinate(10.0, 72.0), false); // lat too low
});

test("isValidGujaratCoordinate rejects NaN/non-finite/non-numeric input rather than crashing", () => {
  assert.equal(fmt.isValidGujaratCoordinate(NaN, 72.0), false);
  assert.equal(fmt.isValidGujaratCoordinate(21.0, Infinity), false);
  assert.equal(fmt.isValidGujaratCoordinate(null, 72.0), false);
  assert.equal(fmt.isValidGujaratCoordinate("21.0", "72.0"), false); // strings, not numbers -- caller must parse first
});
