/*
 * Main application controller. Loads the two real pipeline artifacts,
 * wires filters/list/map/detail, and renders loading/error/empty/stale
 * states. No data is invented here -- every rendered field traces back
 * to a column in gujarat_2026_inference.csv or a key in
 * gujarat_2026_inference_freshness.json.
 */
(function () {
  var fmt = window.ThermoScopeFormat;
  var CsvLib = window.ThermoScopeCSV;
  var MapLib = window.ThermoScopeMap;

  var CSV_URL = "../data/processed/gujarat_2026_inference.csv";
  var FRESHNESS_URL = "../data/processed/gujarat_2026_inference_freshness.json";

  var state = {
    rows: [],
    freshness: null,
    filters: { classes: [], statuses: [], landCovers: [], startDateFrom: "", startDateTo: "", search: "" },
    temporalWindow: null, // null = all dates; else one of fmt.TEMPORAL_WINDOWS
    selectedEventId: null,
    map: null,
    mapHandle: null,
  };

  function $(id) {
    return document.getElementById(id);
  }

  // ---------------------------------------------------------------
  // Data loading
  // ---------------------------------------------------------------
  function loadData() {
    setLoading(true);
    setError(null);
    return Promise.all([
      fetch(CSV_URL).then(function (r) {
        if (!r.ok) throw new Error("Could not load gujarat_2026_inference.csv (HTTP " + r.status + ")");
        return r.text();
      }),
      fetch(FRESHNESS_URL).then(function (r) {
        if (!r.ok) throw new Error("Could not load gujarat_2026_inference_freshness.json (HTTP " + r.status + ")");
        return r.json();
      }),
    ])
      .then(function (results) {
        state.rows = CsvLib.parseCSV(results[0]);
        state.freshness = results[1];
        setLoading(false);
        onDataLoaded();
      })
      .catch(function (err) {
        setLoading(false);
        setError(err.message || String(err));
      });
  }

  function setLoading(isLoading) {
    $("ts-list").innerHTML = isLoading ? '<p class="ts-state-text">Loading events…</p>' : "";
    if (isLoading) {
      $("ts-summary").textContent = "";
    }
  }

  function setError(message) {
    var banner = $("ts-error-banner");
    if (!message) {
      banner.hidden = true;
      banner.textContent = "";
      return;
    }
    var cached = state.freshness ? " Last known update: " + state.freshness.last_updated_utc + "." : "";
    banner.hidden = false;
    banner.innerHTML =
      '<span>' + message + cached + '</span> <button id="ts-retry-btn" class="ts-retry-btn">Retry</button>';
    $("ts-retry-btn").addEventListener("click", loadData);
  }

  function onDataLoaded() {
    renderFreshnessBar();
    renderFilterOptions();
    updateActiveFilterCounts();
    updateTemporalUI();
    initMapIfNeeded();
    applyFiltersAndRender();
    startPolling();
  }

  // ---------------------------------------------------------------
  // Automatic refresh (2026-09 NRT-operation milestone). The backend
  // (src/update_2026_nrt.py) now runs on its own schedule, independent
  // of the browser; this only checks the LOCAL freshness sidecar every
  // POLL_INTERVAL_MS (~60s) and reloads the LOCAL CSV when it changes.
  // Neither call ever reaches NASA FIRMS or OSM -- both URLs below are
  // the same two local artifact paths loadData() already uses.
  // ---------------------------------------------------------------
  var pollTimerId = null;

  function startPolling() {
    if (pollTimerId) return; // idempotent -- loadData() may be called again (e.g. Retry)
    pollTimerId = setInterval(tick, fmt.POLL_INTERVAL_MS);
  }

  function tick() {
    // Cheap, local, no network: keep "Updated X ago" and the stale
    // banner accurate even on ticks where the backend hasn't produced
    // new data (e.g. a missed or failed scheduled run).
    if (state.freshness) renderFreshnessBar();

    fetch(FRESHNESS_URL, { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (freshness) {
        if (!freshness) return;
        if (!state.freshness || freshness.last_updated_utc !== state.freshness.last_updated_utc) {
          return refreshDataInPlace(freshness);
        }
      })
      .catch(function () {
        // Transient poll failure -- leave the UI exactly as-is and let
        // the stale banner (driven by real elapsed time, not by this
        // fetch) surface a genuine outage; retry on the next tick.
      });
  }

  // Reloads the CSV in place and re-renders, preserving the user's
  // active filters/search/selection -- never a full page reload, never
  // a loading spinner (this is a quiet background refresh, not a
  // user-initiated action).
  function refreshDataInPlace(newFreshness) {
    return fetch(CSV_URL, { cache: "no-store" })
      .then(function (r) {
        if (!r.ok) throw new Error("background refresh failed (HTTP " + r.status + ")");
        return r.text();
      })
      .then(function (text) {
        state.rows = CsvLib.parseCSV(text);
        state.freshness = newFreshness;
        renderFreshnessBar();
        syncLandCoverFilterOptions();
        updateTemporalUI(); // "All dates" bounds and any active window's description may have shifted
        applyFiltersAndRender({ silent: true });
        refreshDetailIfOpen();
      })
      .catch(function () {
        // Leave the currently-displayed data in place; the next tick retries.
      });
  }

  // ---------------------------------------------------------------
  // Freshness / staleness (DESIGN.md §10)
  // ---------------------------------------------------------------
  function renderFreshnessBar() {
    var f = state.freshness;
    var seconds = fmt.secondsSince(f.last_updated_utc);
    var stale = fmt.isStale(f.last_updated_utc);

    $("ts-last-updated").textContent = "Updated " + fmt.formatRelativeTime(seconds);
    $("ts-data-period").textContent = "2026 data through " + f.nrt_window_end;
    $("ts-data-period").title = "NRT window: " + f.nrt_window_start + " to " + f.nrt_window_end;

    // Derived from the real last-successful-update timestamp + the real
    // scheduler interval (fmt.SCHEDULER_INTERVAL_SECONDS, kept in sync
    // with ops/com.thermoscope.nrt_update.plist) -- an estimate, not a
    // guarantee (a cycle can be skipped by the lock guard or delayed).
    var nextCheck = fmt.nextScheduledCheckUTC(f.last_updated_utc);
    $("ts-next-check").textContent = stale
      ? "Next automatic check overdue"
      : "Next automatic check ~" + fmt.formatClockUTC(nextCheck);

    var indicator = $("ts-freshness-indicator");
    indicator.hidden = false;
    indicator.title = f.last_updated_utc + " (UTC) — " + f.n_events + " events, " +
      f.n_closed + " closed, " + f.n_provisional + " provisional";
    $("ts-freshness-dot").classList.toggle("is-stale", stale);

    var staleBanner = $("ts-stale-banner");
    if (stale) {
      staleBanner.hidden = false;
      staleBanner.textContent =
        "Data last updated " + fmt.formatRelativeTime(seconds) +
        " — the automatic update may have failed. Showing the last successfully loaded data.";
    } else {
      staleBanner.hidden = true;
      staleBanner.textContent = "";
    }
  }

  // ---------------------------------------------------------------
  // Filters (DESIGN.md §7) -- built from values actually present in the data
  // ---------------------------------------------------------------
  function renderFilterOptions() {
    var landCovers = Array.from(
      new Set(state.rows.map(function (r) { return r.land_cover_class; }).filter(function (v) { return v !== ""; }))
    ).sort();

    var classGroup = $("ts-filter-classes");
    classGroup.innerHTML = fmt.TRAINED_CLASSES.map(function (cls) {
      return checkboxHtml("class", cls, fmt.CLASS_LABELS[cls], fmt.CLASS_COLORS[cls]);
    }).join("");

    var statusGroup = $("ts-filter-statuses");
    statusGroup.innerHTML = ["closed", "provisional"].map(function (s) {
      return checkboxHtml("status", s, s.charAt(0).toUpperCase() + s.slice(1));
    }).join("");

    var landCoverGroup = $("ts-filter-landcovers");
    landCoverGroup.innerHTML = landCovers.map(function (lc) {
      return checkboxHtml("landcover", lc, lc);
    }).join("");

    document.querySelectorAll("#ts-filterbar input[type=checkbox]").forEach(function (el) {
      el.addEventListener("change", onFilterChange);
    });
    $("ts-filter-search").addEventListener("input", onFilterChange);
    $("ts-filter-date-from").addEventListener("change", onFilterChange);
    $("ts-filter-date-to").addEventListener("change", onFilterChange);
    $("ts-filter-clear").addEventListener("click", clearFilters);
  }

  // Appends checkboxes only for land-cover values not already present in
  // the DOM -- unlike renderFilterOptions() (full rebuild, used only on
  // initial load), this never touches/recreates existing inputs, so a
  // background refresh can never silently uncheck a filter the user set.
  function syncLandCoverFilterOptions() {
    var landCovers = Array.from(
      new Set(state.rows.map(function (r) { return r.land_cover_class; }).filter(function (v) { return v !== ""; }))
    ).sort();
    var group = $("ts-filter-landcovers");
    var existing = {};
    group.querySelectorAll('input[data-group="landcover"]').forEach(function (el) { existing[el.value] = true; });
    landCovers.forEach(function (lc) {
      if (existing[lc]) return;
      var wrapper = document.createElement("div");
      wrapper.innerHTML = checkboxHtml("landcover", lc, lc);
      var label = wrapper.firstChild;
      group.appendChild(label);
      label.querySelector("input").addEventListener("change", onFilterChange);
    });
  }

  function checkboxHtml(group, value, label, color) {
    var swatch = color ? '<span class="ts-chip-swatch" style="background:' + color + '"></span>' : "";
    var id = "ts-filter-" + group + "-" + value;
    return (
      '<label class="ts-chip" for="' + id + '">' +
      '<input type="checkbox" id="' + id + '" data-group="' + group + '" value="' + value + '">' +
      swatch + '<span class="ts-chip-label">' + label + "</span></label>"
    );
  }

  function onFilterChange() {
    state.filters.classes = checkedValues("class");
    state.filters.statuses = checkedValues("status");
    state.filters.landCovers = checkedValues("landcover");
    state.filters.startDateFrom = $("ts-filter-date-from").value;
    state.filters.startDateTo = $("ts-filter-date-to").value;
    state.filters.search = $("ts-filter-search").value;
    // A manual edit to the date pickers IS the calendar control -- reflect
    // that in the temporal selector's active state/description.
    state.temporalWindow = (state.filters.startDateFrom || state.filters.startDateTo) ? "calendar" : null;
    updateTemporalUI();
    updateActiveFilterCounts();
    applyFiltersAndRender();
  }

  // ---------------------------------------------------------------
  // Temporal window control (TODAY / 24 HRS / 7 DAYS / CALENDAR).
  // "Today" is anchored to the real freshness sidecar's run_date, never
  // the browser's own clock, so the window matches what the backend
  // considers "now". The inference CSV only carries start_date at
  // calendar-day granularity -- 24 HRS is therefore a disclosed
  // date-level approximation (today + the preceding day), never
  // presented as a true rolling 24-hour window. See format.js
  // computeTemporalRange for the exact, tested definition of each.
  // ---------------------------------------------------------------
  function anchorToday() {
    return state.freshness ? state.freshness.run_date : null;
  }

  function onTemporalButtonClick(windowKey) {
    if (state.temporalWindow === windowKey) {
      // toggle off -- back to all dates
      state.temporalWindow = null;
      state.filters.startDateFrom = "";
      state.filters.startDateTo = "";
    } else {
      state.temporalWindow = windowKey;
      var today = anchorToday();
      if (windowKey === "calendar") {
        // Reveal the pickers; don't overwrite any dates the user already chose.
      } else if (today) {
        var range = fmt.computeTemporalRange(today, windowKey);
        state.filters.startDateFrom = range.from;
        state.filters.startDateTo = range.to;
      }
    }
    $("ts-filter-date-from").value = state.filters.startDateFrom;
    $("ts-filter-date-to").value = state.filters.startDateTo;
    updateTemporalUI();
    updateActiveFilterCounts();
    applyFiltersAndRender();
  }

  function updateTemporalUI() {
    document.querySelectorAll(".ts-temporal-btn").forEach(function (btn) {
      btn.classList.toggle("is-active", btn.getAttribute("data-window") === state.temporalWindow);
    });
    $("ts-temporal-calendar").hidden = state.temporalWindow !== "calendar";
    renderTemporalDescription();
  }

  function renderTemporalDescription() {
    var el = $("ts-temporal-description");
    var today = anchorToday();
    if (!state.temporalWindow) {
      var bounds = fmt.dateBounds(state.rows);
      el.textContent = bounds ? "All dates — " + bounds.min + " to " + bounds.max : "All dates";
      return;
    }
    if (state.temporalWindow === "today") {
      el.textContent = today ? "Today — " + today : "Today";
    } else if (state.temporalWindow === "24h") {
      el.textContent = today
        ? "24 hrs (date-level) — from " + fmt.addDaysISO(today, -1) + " 00:00 UTC to present"
        : "24 hrs (date-level)";
    } else if (state.temporalWindow === "7d") {
      el.textContent = today ? "7 days — from " + fmt.addDaysISO(today, -6) + " to present" : "7 days";
    } else if (state.temporalWindow === "calendar") {
      var from = state.filters.startDateFrom, to = state.filters.startDateTo;
      el.textContent = (from || to)
        ? "Custom range — " + (from || "…") + " → " + (to || "present")
        : "Custom range — select start and end dates";
    }
  }

  function updateActiveFilterCounts() {
    setCount("ts-filter-classes-count", state.filters.classes.length);
    setCount("ts-filter-statuses-count", state.filters.statuses.length);
    setCount("ts-filter-landcovers-count", state.filters.landCovers.length);
    var anyActive =
      state.filters.classes.length || state.filters.statuses.length || state.filters.landCovers.length ||
      state.filters.startDateFrom || state.filters.startDateTo || state.filters.search;
    $("ts-filter-clear").disabled = !anyActive;
  }

  function setCount(id, n) {
    var el = $(id);
    if (el) el.textContent = n > 0 ? " (" + n + ")" : "";
  }

  function checkedValues(group) {
    var out = [];
    document.querySelectorAll('#ts-filterbar input[data-group="' + group + '"]:checked').forEach(function (el) {
      out.push(el.value);
    });
    return out;
  }

  function clearFilters() {
    document.querySelectorAll("#ts-filterbar input[type=checkbox]").forEach(function (el) { el.checked = false; });
    $("ts-filter-search").value = "";
    $("ts-filter-date-from").value = "";
    $("ts-filter-date-to").value = "";
    state.filters = { classes: [], statuses: [], landCovers: [], startDateFrom: "", startDateTo: "", search: "" };
    state.temporalWindow = null;
    updateTemporalUI();
    updateActiveFilterCounts();
    applyFiltersAndRender();
  }

  // ---------------------------------------------------------------
  // Map
  // ---------------------------------------------------------------
  function initMapIfNeeded() {
    if (state.map) return;
    state.map = MapLib.createMap("ts-map");
  }

  // ---------------------------------------------------------------
  // List + map render
  // ---------------------------------------------------------------
  function applyFiltersAndRender(opts) {
    opts = opts || {};
    var filtered = fmt.sortByStartDateDesc(fmt.applyFilters(state.rows, state.filters));
    renderSummary(filtered.length);
    renderList(filtered);
    if (state.mapHandle) {
      state.map.removeLayer(state.mapHandle.clusterGroup);
    }
    state.mapHandle = MapLib.renderEvents(state.map, filtered, selectEvent);
    if (state.selectedEventId) {
      // silent (background refresh): restyle the marker only, never pan/
      // zoom -- the user didn't ask the map to move.
      if (opts.silent) {
        state.mapHandle.highlightOnly(state.selectedEventId);
      } else {
        state.mapHandle.selectEvent(state.selectedEventId);
      }
    }
  }

  function renderSummary(filteredCount) {
    var f = state.freshness;
    $("ts-summary").textContent =
      filteredCount + " of " + f.n_events + " events shown · " +
      f.n_closed + " closed · " + f.n_provisional + " provisional";
  }

  function renderList(rows) {
    var list = $("ts-list");
    if (rows.length === 0) {
      list.innerHTML =
        '<p class="ts-state-text">No events match the current filters. ' +
        '<button id="ts-empty-clear" class="ts-retry-btn">Clear filters</button></p>';
      var btn = document.getElementById("ts-empty-clear");
      if (btn) btn.addEventListener("click", clearFilters);
      return;
    }
    list.innerHTML = rows.map(rowHtml).join("");
    list.querySelectorAll(".ts-list-row").forEach(function (el) {
      el.addEventListener("click", function () {
        selectEvent(el.getAttribute("data-event-id"));
      });
    });
  }

  function rowHtml(r) {
    var color = fmt.CLASS_COLORS[r.predicted_class] || "#5b6169";
    var topProb = fmt.formatPercent(r["prob_" + r.predicted_class], 0);
    return (
      '<div class="ts-list-row" data-event-id="' + r.event_id + '" tabindex="0">' +
      '<span class="ts-list-marker" style="background:' + color + '"></span>' +
      '<div class="ts-list-main">' +
      '<div class="ts-list-top">' +
      '<span class="ts-list-class">' + fmt.CLASS_LABELS[r.predicted_class] + "</span>" +
      '<span class="ts-list-prob" title="Highest model-predicted class probability -- not verified confidence">p=' +
      topProb + "</span>" +
      "</div>" +
      '<div class="ts-list-sub">' +
      '<span class="ts-list-id">' + r.event_id + "</span>" +
      '<span class="ts-list-date">' + r.start_date + "</span>" +
      "</div>" +
      "</div>" +
      '<span class="ts-status-badge ts-status-' + r.status + '">' + r.status + "</span>" +
      "</div>"
    );
  }

  // ---------------------------------------------------------------
  // Event detail (DESIGN.md §6: OBSERVED DATA / EXTERNAL EVIDENCE / MODEL OUTPUT)
  // ---------------------------------------------------------------
  function selectEvent(eventId) {
    state.selectedEventId = eventId;
    var row = state.rows.filter(function (r) { return r.event_id === eventId; })[0];
    if (!row) return;
    if (state.mapHandle) state.mapHandle.selectEvent(eventId);
    renderDetail(row);
    $("ts-panel-list").hidden = true;
    $("ts-panel-detail").hidden = false;
  }

  function backToList() {
    state.selectedEventId = null;
    if (state.mapHandle) state.mapHandle.clearSelection();
    $("ts-panel-detail").hidden = true;
    $("ts-panel-list").hidden = false;
  }

  // Re-renders the currently-open detail panel's content from (possibly
  // updated) state.rows after a background refresh -- only if the panel
  // is actually open, and without touching panel visibility or the map.
  function refreshDetailIfOpen() {
    if (!state.selectedEventId) return;
    if ($("ts-panel-detail").hidden) return;
    var row = state.rows.filter(function (r) { return r.event_id === state.selectedEventId; })[0];
    if (row) renderDetail(row);
  }

  function renderDetail(r) {
    var color = fmt.CLASS_COLORS[r.predicted_class] || "#5b6169";
    var statusDef = fmt.STATUS_DEFINITIONS[r.status] || "";

    var evidenceRows = fmt.EVIDENCE_DISTANCE_FIELDS.map(function (f) {
      return (
        '<div class="ts-kv"><span class="ts-kv-label">' + f.label + "</span>" +
        '<span class="ts-kv-value">' + fmt.formatDistanceM(r[f.key]) + "</span></div>"
      );
    }).join("");

    var probRows = fmt.TRAINED_CLASSES.map(function (cls) {
      var pct = fmt.formatPercent(r["prob_" + cls], 1);
      var width = Math.max(0, Math.min(100, (fmt.toNumber(r["prob_" + cls]) || 0) * 100));
      var isTop = cls === r.predicted_class;
      return (
        '<div class="ts-prob-row' + (isTop ? " is-top" : "") + '">' +
        '<span class="ts-prob-label">' + fmt.CLASS_LABELS[cls] + "</span>" +
        '<span class="ts-prob-bar"><span class="ts-prob-fill" style="width:' + width + "%;background:" +
        fmt.CLASS_COLORS[cls] + '"></span></span>' +
        '<span class="ts-prob-value">' + pct + "</span>" +
        "</div>"
      );
    }).join("");

    $("ts-detail-content").innerHTML =
      '<div class="ts-detail-header">' +
      '<div class="ts-detail-id">' + r.event_id + "</div>" +
      '<div class="ts-detail-class-row">' +
      '<span class="ts-detail-class-dot" style="background:' + color + '"></span>' +
      '<span class="ts-detail-class" style="color:' + color + '">' + fmt.CLASS_LABELS[r.predicted_class] + "</span>" +
      "</div>" +
      '<div class="ts-detail-class-caption">Model-predicted class</div>' +
      '<span class="ts-status-badge ts-status-' + r.status + '" title="' + statusDef + '">' + r.status + "</span>" +
      "</div>" +

      '<section class="ts-detail-section">' +
      '<h3 class="ts-eyebrow"><span class="ts-eyebrow-index">01</span>Observed data</h3>' +
      kv("Start date", r.start_date) +
      kv("End date", r.end_date) +
      kv("Duration", r.duration_days + (r.duration_days === "1" ? " day" : " days")) +
      kv("Detection count", r.detection_count) +
      kv("Mean fire radiative power", fmt.formatFRP(r.mean_frp)) +
      kv("Max fire radiative power", fmt.formatFRP(r.max_frp)) +
      kv("Share of detections at night", fmt.formatPercent(r.night_fraction, 0)) +
      kv("Centroid", fmt.formatCoordinate(r.centroid_lat) + ", " + fmt.formatCoordinate(r.centroid_lon)) +
      kv("Spatial footprint radius", Math.round(fmt.toNumber(r.spatial_extent_m) || 0) + " m") +
      "</section>" +

      '<section class="ts-detail-section">' +
      '<h3 class="ts-eyebrow"><span class="ts-eyebrow-index">02</span>External evidence</h3>' +
      kv("Land cover", r.land_cover_class || "—") +
      evidenceRows +
      '<div class="ts-kv ts-kv-wide"><span class="ts-kv-label">Persistent cluster</span>' +
      '<span class="ts-kv-value">' + fmt.formatClusterOverlap(r) + "</span></div>" +
      "</section>" +

      '<section class="ts-detail-section ts-detail-model">' +
      '<h3 class="ts-eyebrow"><span class="ts-eyebrow-index">03</span>Model output</h3>' +
      '<div class="ts-model-flag">Model output — not verified ground truth</div>' +
      probRows +
      '<p class="ts-capability-note">' + r.class_capability_note + "</p>" +
      "</section>";
  }

  function kv(label, value) {
    return '<div class="ts-kv"><span class="ts-kv-label">' + label + '</span><span class="ts-kv-value">' + value + "</span></div>";
  }

  // ---------------------------------------------------------------
  // View switching (Monitoring / Methodology)
  // ---------------------------------------------------------------
  function switchView(view) {
    document.querySelectorAll(".ts-view").forEach(function (el) { el.hidden = true; });
    $("ts-view-" + view).hidden = false;
    document.querySelectorAll(".ts-nav-btn").forEach(function (el) {
      el.classList.toggle("is-active", el.getAttribute("data-view") === view);
    });
    if (view === "monitoring" && state.map) {
      setTimeout(function () { state.map.invalidateSize(); }, 0);
    }
  }

  // ---------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------
  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".ts-nav-btn").forEach(function (btn) {
      btn.addEventListener("click", function () { switchView(btn.getAttribute("data-view")); });
    });
    document.querySelectorAll(".ts-temporal-btn").forEach(function (btn) {
      btn.addEventListener("click", function () { onTemporalButtonClick(btn.getAttribute("data-window")); });
    });
    $("ts-back-btn").addEventListener("click", backToList);
    loadData();
  });
})();
