/*
 * Pure formatting/domain helpers -- no DOM, no fetch, so these are
 * directly unit-testable under Node (see frontend/tests/format.test.js)
 * and reused unmodified by app.js in the browser.
 *
 * Field names, class names, and thresholds here are drawn directly from
 * DESIGN.md and the real pipeline (src/infer_2026_events.py,
 * src/update_2026_nrt.py, src/spatial_recurrence.py) -- nothing here is
 * invented.
 */
(function (root, factory) {
  var mod = factory();
  if (typeof module !== "undefined" && module.exports) {
    module.exports = mod;
  } else {
    root.ThermoScopeFormat = mod;
  }
})(typeof self !== "undefined" ? self : this, function () {
  // The 4 trained classes (src/train_source_classifier.py TRAINED_CLASSES),
  // in the fixed alphabetical order DESIGN.md §3/§9 requires everywhere.
  var TRAINED_CLASSES = ["Crop_Residue", "Forest_Wildfire", "Gas_Flare", "Industrial"];

  // Categorical, non-severity-ordered class colors (DESIGN.md §3). Used
  // for MAP markers, which sit on a light basemap regardless of the
  // app's dark chrome.
  var CLASS_COLORS = {
    Crop_Residue: "#c1791f",
    Forest_Wildfire: "#2f9959",
    Gas_Flare: "#0ea89b",
    Industrial: "#3163d9",
  };

  // Same 4 categorical identities (same hue per class -- amber, green,
  // teal, blue), lightened for legibility against the dark UI chrome
  // (chips, list rows, detail panel, probability bars). This is a
  // contrast adjustment for a different background, not a new semantic
  // -- CLASS_LABELS/ordering/meaning are identical to CLASS_COLORS.
  var CLASS_COLORS_UI = {
    Crop_Residue: "#d99a52",
    Forest_Wildfire: "#6fb98a",
    Gas_Flare: "#4fc9bb",
    Industrial: "#8aa9d9",
  };

  // Gujarat study-area bounds (src/spatial_recurrence.py LAT_MIN/MAX/LON_MIN/MAX).
  var GUJARAT_BOUNDS = { latMin: 20.0, latMax: 24.7, lonMin: 68.0, lonMax: 74.5 };

  // Data staleness threshold: a small multiple of the NRT scheduler's
  // interval (ops/com.thermoscope.nrt_update.plist StartInterval=1800s,
  // i.e. every 30 minutes) -- kept as "2 missed cycles" rather than a
  // fixed constant so it stays meaningful if the plist interval changes.
  var SCHEDULER_INTERVAL_SECONDS = 1800;
  var STALE_THRESHOLD_SECONDS = 2 * SCHEDULER_INTERVAL_SECONDS;

  // How often the browser checks the local freshness sidecar for a
  // change (DESIGN.md §10 / 2026-09 NRT-operation milestone). This is a
  // local file re-fetch only -- never a NASA/OSM API call.
  var POLL_INTERVAL_MS = 60 * 1000;

  var CLASS_LABELS = {
    Crop_Residue: "Crop Residue",
    Forest_Wildfire: "Forest / Wildfire",
    Gas_Flare: "Gas Flare",
    Industrial: "Industrial",
  };

  var STATUS_DEFINITIONS = {
    closed: "No further detections linked to this event for at least 1 day.",
    provisional: "A future detection could still extend this event.",
  };

  var EVIDENCE_DISTANCE_FIELDS = [
    { key: "nearest_osm_industrial_power_m", label: "Nearest OSM industrial/power" },
    { key: "nearest_gppd_thermal_plant_m", label: "Nearest GPPD thermal plant" },
    { key: "nearest_osm_flare_m", label: "Nearest OSM flare" },
    { key: "nearest_osm_kiln_m", label: "Nearest OSM kiln" },
  ];

  function isBlank(v) {
    return v === undefined || v === null || v === "";
  }

  function toNumber(v) {
    if (isBlank(v)) return null;
    var n = Number(v);
    return isNaN(n) ? null : n;
  }

  function formatDistanceM(v) {
    var n = toNumber(v);
    if (n === null) return "— (none within search radius)";
    return Math.round(n).toLocaleString() + " m";
  }

  function formatPercent(v, digits) {
    var n = toNumber(v);
    if (n === null) return "—";
    return (n * 100).toFixed(digits === undefined ? 1 : digits) + "%";
  }

  function formatFRP(v) {
    var n = toNumber(v);
    if (n === null) return "—";
    return n.toFixed(2) + " MW";
  }

  function formatCoordinate(v, digits) {
    var n = toNumber(v);
    if (n === null) return "—";
    return n.toFixed(digits === undefined ? 4 : digits);
  }

  function formatClusterOverlap(row) {
    if (isBlank(row.overlaps_cluster_id)) return "No persistent-cluster overlap.";
    var strength = isBlank(row.overlaps_cluster_recurrence_strength)
      ? "unknown"
      : row.overlaps_cluster_recurrence_strength;
    return "Overlaps persistent cluster #" + row.overlaps_cluster_id + " (recurrence: " + strength + ").";
  }

  // ISO 8601 UTC timestamp -> whole seconds elapsed, relative to `now`
  // (a Date, injectable for testing).
  function secondsSince(isoTimestamp, now) {
    var then = new Date(isoTimestamp).getTime();
    var nowMs = (now instanceof Date ? now : new Date()).getTime();
    return Math.max(0, Math.round((nowMs - then) / 1000));
  }

  // Derived, not hardcoded: the real last-successful-update timestamp
  // plus the real scheduler interval (ops/com.thermoscope.nrt_update.plist
  // StartInterval). This is an estimate of the next automatic check, not
  // a guarantee -- a run can be skipped by the lock guard or delayed.
  function nextScheduledCheckUTC(isoTimestamp) {
    var then = new Date(isoTimestamp).getTime();
    return new Date(then + SCHEDULER_INTERVAL_SECONDS * 1000);
  }

  function formatClockUTC(date) {
    return date.toISOString().slice(11, 16) + " UTC";
  }

  // ---------------------------------------------------------------
  // IST display (2026-09 redesign). All user-facing times are shown in
  // India Standard Time (UTC+5:30, no DST) via Intl's IANA tz database --
  // not manual offset arithmetic, so it's correct regardless of the
  // viewer's own locale/timezone. The underlying data (run_date,
  // last_updated_utc) is unchanged; this only affects display formatting.
  // ---------------------------------------------------------------
  var IST_TIMEZONE = "Asia/Kolkata";

  function formatClockIST(dateOrIso) {
    var d = dateOrIso instanceof Date ? dateOrIso : new Date(dateOrIso);
    var formatted = new Intl.DateTimeFormat("en-GB", {
      timeZone: IST_TIMEZONE, hour: "2-digit", minute: "2-digit", hour12: false,
    }).format(d);
    return formatted + " IST";
  }

  var MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  function formatDateIST(dateOrIso) {
    var d = dateOrIso instanceof Date ? dateOrIso : new Date(dateOrIso);
    // Read day/month/year as numeric parts in the IST timezone, then apply
    // our own fixed month abbreviations -- Intl's locale-formatted month
    // names vary by locale/ICU version (e.g. en-GB renders "Sept", not
    // "Sep"), which would make this display inconsistent across browsers.
    var parts = new Intl.DateTimeFormat("en-US", {
      timeZone: IST_TIMEZONE, day: "2-digit", month: "numeric", year: "numeric",
    }).formatToParts(d);
    var day, month, year;
    parts.forEach(function (p) {
      if (p.type === "day") day = p.value;
      if (p.type === "month") month = p.value;
      if (p.type === "year") year = p.value;
    });
    return day + " " + MONTH_ABBR[parseInt(month, 10) - 1] + " " + year;
  }

  function formatDateTimeIST(dateOrIso) {
    return formatDateIST(dateOrIso) + " " + formatClockIST(dateOrIso);
  }

  // A plain "YYYY-MM-DD" (calendar-day granularity, e.g. event start_date)
  // has no time-of-day to convert -- format it as a plain IST calendar
  // date without implying a clock time that doesn't exist in the data.
  function formatCalendarDate(isoDate) {
    if (isBlank(isoDate)) return "—";
    return formatDateIST(isoDate + "T00:00:00Z");
  }

  // ---------------------------------------------------------------
  // Defensive coordinate validation (2026-09 map bug-fix milestone).
  // Every event plotted on the map must pass this check first; an event
  // that fails is skipped (never rendered) and logged -- never silently
  // coerced/mutated to "fit". Uses the exact same GUJARAT_BOUNDS the
  // rest of the app already uses (the pipeline's own bounding box,
  // src/spatial_recurrence.py LAT_MIN/MAX/LON_MIN/MAX) -- not a
  // different/tighter boundary invented to hide real data.
  // ---------------------------------------------------------------
  function isValidGujaratCoordinate(lat, lon) {
    if (typeof lat !== "number" || typeof lon !== "number") return false;
    if (isNaN(lat) || isNaN(lon)) return false;
    if (!isFinite(lat) || !isFinite(lon)) return false;
    return (
      lat >= GUJARAT_BOUNDS.latMin && lat <= GUJARAT_BOUNDS.latMax &&
      lon >= GUJARAT_BOUNDS.lonMin && lon <= GUJARAT_BOUNDS.lonMax
    );
  }

  function isStale(isoTimestamp, now, thresholdSeconds) {
    var threshold = thresholdSeconds === undefined ? STALE_THRESHOLD_SECONDS : thresholdSeconds;
    return secondsSince(isoTimestamp, now) > threshold;
  }

  function formatRelativeTime(seconds) {
    if (seconds < 60) return "less than a minute ago";
    var minutes = Math.floor(seconds / 60);
    if (minutes < 60) return minutes + (minutes === 1 ? " minute ago" : " minutes ago");
    var hours = Math.floor(minutes / 60);
    if (hours < 24) return hours + (hours === 1 ? " hour ago" : " hours ago");
    var days = Math.floor(hours / 24);
    return days + (days === 1 ? " day ago" : " days ago");
  }

  // ---------------------------------------------------------------
  // Temporal window controls (2026-09 milestone). The inference CSV
  // exposes event start_date at CALENDAR-DAY granularity only (no
  // per-event timestamp) -- these helpers deliberately never claim
  // finer precision than that. "24 HRS" is a disclosed date-level
  // approximation (today + the preceding calendar day), not a true
  // rolling 24-hour window.
  // ---------------------------------------------------------------
  var TEMPORAL_WINDOWS = ["today", "24h", "7d", "calendar"];

  function addDaysISO(isoDate, days) {
    var d = new Date(isoDate + "T00:00:00Z");
    d.setUTCDate(d.getUTCDate() + days);
    return d.toISOString().slice(0, 10);
  }

  // anchorDateISO is "today" as the pipeline understands it (the real
  // freshness sidecar's run_date) -- never the browser's own clock, so
  // the window is anchored to the same "now" the backend used.
  function computeTemporalRange(anchorDateISO, windowKey) {
    if (windowKey === "today") return { from: anchorDateISO, to: anchorDateISO };
    if (windowKey === "24h") return { from: addDaysISO(anchorDateISO, -1), to: anchorDateISO };
    if (windowKey === "7d") return { from: addDaysISO(anchorDateISO, -6), to: anchorDateISO };
    return null; // "calendar": caller supplies explicit user-picked dates
  }

  // Real min/max start_date actually present in the loaded rows -- used
  // to describe "all dates" honestly instead of an invented range.
  function dateBounds(rows) {
    var dates = (rows || []).map(function (r) { return r.start_date; }).filter(function (v) { return !isBlank(v); });
    if (dates.length === 0) return null;
    dates.sort();
    return { min: dates[0], max: dates[dates.length - 1] };
  }

  // Sort by start_date descending (most recent first) -- DESIGN.md §7 default sort.
  function sortByStartDateDesc(rows) {
    return rows.slice().sort(function (a, b) {
      if (a.start_date === b.start_date) return 0;
      return a.start_date < b.start_date ? 1 : -1;
    });
  }

  function applyFilters(rows, filters) {
    filters = filters || {};
    return rows.filter(function (r) {
      if (filters.classes && filters.classes.length > 0 && filters.classes.indexOf(r.predicted_class) === -1) {
        return false;
      }
      if (filters.statuses && filters.statuses.length > 0 && filters.statuses.indexOf(r.status) === -1) {
        return false;
      }
      if (filters.landCovers && filters.landCovers.length > 0 && filters.landCovers.indexOf(r.land_cover_class) === -1) {
        return false;
      }
      if (filters.startDateFrom && r.start_date < filters.startDateFrom) return false;
      if (filters.startDateTo && r.start_date > filters.startDateTo) return false;
      if (filters.search) {
        var needle = filters.search.trim().toLowerCase();
        if (needle && r.event_id.toLowerCase().indexOf(needle) !== 0) return false;
      }
      return true;
    });
  }

  return {
    TRAINED_CLASSES: TRAINED_CLASSES,
    CLASS_COLORS: CLASS_COLORS,
    CLASS_COLORS_UI: CLASS_COLORS_UI,
    CLASS_LABELS: CLASS_LABELS,
    TEMPORAL_WINDOWS: TEMPORAL_WINDOWS,
    addDaysISO: addDaysISO,
    computeTemporalRange: computeTemporalRange,
    dateBounds: dateBounds,
    GUJARAT_BOUNDS: GUJARAT_BOUNDS,
    SCHEDULER_INTERVAL_SECONDS: SCHEDULER_INTERVAL_SECONDS,
    STALE_THRESHOLD_SECONDS: STALE_THRESHOLD_SECONDS,
    POLL_INTERVAL_MS: POLL_INTERVAL_MS,
    STATUS_DEFINITIONS: STATUS_DEFINITIONS,
    EVIDENCE_DISTANCE_FIELDS: EVIDENCE_DISTANCE_FIELDS,
    isBlank: isBlank,
    toNumber: toNumber,
    formatDistanceM: formatDistanceM,
    formatPercent: formatPercent,
    formatFRP: formatFRP,
    formatCoordinate: formatCoordinate,
    formatClusterOverlap: formatClusterOverlap,
    secondsSince: secondsSince,
    isStale: isStale,
    nextScheduledCheckUTC: nextScheduledCheckUTC,
    formatClockUTC: formatClockUTC,
    formatClockIST: formatClockIST,
    formatDateIST: formatDateIST,
    formatDateTimeIST: formatDateTimeIST,
    formatCalendarDate: formatCalendarDate,
    isValidGujaratCoordinate: isValidGujaratCoordinate,
    formatRelativeTime: formatRelativeTime,
    sortByStartDateDesc: sortByStartDateDesc,
    applyFilters: applyFilters,
  };
});
