/*
 * Map rendering (Leaflet). Browser-only -- relies on the global `L`
 * (Leaflet) and `L.markerClusterGroup` (Leaflet.markercluster), both
 * loaded via <script> tags in index.html. Not unit-tested under Node
 * (no DOM/map engine there); verified manually per the test plan in
 * frontend/README.md. All domain formatting/color logic it calls comes
 * from format.js so there is exactly one place that logic is defined.
 */
(function (root) {
  var fmt = root.ThermoScopeFormat;

  // Selection is shown as a neutral dark ring around the marker's own
  // class-color fill (not a generic blue accent) -- selection is a UI
  // state, not a 5th data category, so it must never borrow or compete
  // with the categorical class-color palette.
  var SELECTED_RING_COLOR = "#14171a";
  var CLUSTER_COLOR = "#6d7178"; // matches .ts-cluster background in styles.css

  // Esri World Light Gray Canvas (base + reference-label layers) -- shared
  // by the main map and the event-detail mini map so both use the exact
  // same muted, keyless basemap (DESIGN.md §3/§5).
  var TILE_BASE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}";
  var TILE_REF_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}";

  function createMap(containerId) {
    var bounds = [
      [fmt.GUJARAT_BOUNDS.latMin, fmt.GUJARAT_BOUNDS.lonMin],
      [fmt.GUJARAT_BOUNDS.latMax, fmt.GUJARAT_BOUNDS.lonMax],
    ];
    var map = L.map(containerId, {
      zoomControl: true,
      attributionControl: true,
      // Explicit, not just relying on Leaflet's default: never duplicate
      // markers into adjacent "world copies" when zoomed/panned out --
      // see map-bounds bug investigation (2026-09) in PROGRESS.md. This
      // wasn't the cause of the Pakistan-area marker (that marker is a
      // real, single, correctly-plotted event -- see
      // isValidGujaratCoordinate below and the investigation notes), but
      // it's correct, defensible map hygiene regardless.
      worldCopyJump: false,
      maxBoundsViscosity: 0.8,
    });
    map.fitBounds(bounds, { padding: [16, 16] });

    // Constrain how far a user can zoom/pan away from the Gujarat study
    // area -- this is a Gujarat monitoring app, not a world map. Derived
    // entirely from the fit-to-bounds zoom Leaflet just computed and from
    // fmt.GUJARAT_BOUNDS itself (no invented geography): minZoom allows
    // exactly 1 level out from the Gujarat fit, enough to see neighboring
    // Rajasthan/Maharashtra/the Arabian Sea for context (the fit itself,
    // against GUJARAT_BOUNDS' own 68-74.5E/20-24.7N extent, already shows
    // this much) without reaching Iran/Sri Lanka-scale views. maxBounds is
    // the same bbox padded by 1.6x its own span, with viscosity so panning
    // resists (but does not hard-clip) leaving that padded region.
    // Markers/data are never hidden or altered by this -- it is purely a
    // viewport constraint.
    var fitZoom = map.getZoom();
    map.setMinZoom(Math.max(fitZoom - 1, 4));
    var padLat = (fmt.GUJARAT_BOUNDS.latMax - fmt.GUJARAT_BOUNDS.latMin) * 1.6;
    var padLon = (fmt.GUJARAT_BOUNDS.lonMax - fmt.GUJARAT_BOUNDS.lonMin) * 1.6;
    map.setMaxBounds([
      [fmt.GUJARAT_BOUNDS.latMin - padLat, fmt.GUJARAT_BOUNDS.lonMin - padLon],
      [fmt.GUJARAT_BOUNDS.latMax + padLat, fmt.GUJARAT_BOUNDS.lonMax + padLon],
    ]);

    // (An earlier CARTO Positron tile set was tried first but now requires
    // an API key -- verified live, see PROGRESS.md.)
    L.tileLayer(TILE_BASE_URL, { attribution: "Esri, HERE, Garmin, © OpenStreetMap contributors", maxZoom: 16 }).addTo(map);
    L.tileLayer(TILE_REF_URL, { maxZoom: 16 }).addTo(map); // added after the base layer -- same pane, stacks on top by DOM order

    map._thermoscopeBounds = bounds;
    buildLegendControl(window.innerWidth <= 640 ? "bottomright" : "bottomleft").addTo(map);
    buildResetControl(map).addTo(map);
    return map;
  }

  function markerRadius(zoom) {
    // Modest, consistent size with standard growth on zoom-in (DESIGN.md §5).
    return Math.max(5, Math.min(11, zoom - 4));
  }

  function styleForRow(row, zoom) {
    var color = fmt.CLASS_COLORS[row.predicted_class] || CLUSTER_COLOR;
    var closed = row.status === "closed";
    return {
      radius: markerRadius(zoom),
      // Closed (solid) markers get a white halo ring, not a same-color
      // outline -- keeps the bright class fill readable/separated against
      // both the basemap and adjacent markers. Provisional markers keep
      // their own class-colored dashed outline (that dash pattern, not
      // outline color, is what marks them "not yet closed").
      color: closed ? "#ffffff" : color,
      weight: closed ? 1.5 : 2.25,
      fillColor: color,
      fillOpacity: closed ? 0.92 : 0.18, // solid fill = closed, hollow/outline = provisional
      dashArray: closed ? null : "3,2",
    };
  }

  function clusterIconCreateFunction(cluster) {
    var count = cluster.getChildCount();
    var tier = count < 10 ? "sm" : count < 50 ? "md" : "lg";
    return L.divIcon({
      html: '<div class="ts-cluster ts-cluster-' + tier + '">' + count + "</div>",
      className: "ts-cluster-wrap",
      iconSize: null,
    });
  }

  function selectedStyleFor(row, zoom) {
    var base = styleForRow(row, zoom);
    // Ring in a neutral dark tone, fill stays the marker's own class
    // color -- selection reads as "this one, highlighted", not as a
    // 5th category or an implied severity color.
    base.color = SELECTED_RING_COLOR;
    base.weight = 3;
    base.radius = base.radius + 2;
    return base;
  }

  function buildLegendControl(position) {
    // bottomleft, per the reference layout -- except on narrow (<=640px)
    // viewports, where the map area is short enough (~32vh) that the
    // legend's own height reaches up into the topleft zoom/reset control
    // stack; bottomright is empty there (nothing else uses that corner)
    // so it sidesteps the collision entirely rather than fighting it with
    // ever-smaller text. The event-detail drawer also overlays the bottom
    // of the map (see index.html #ts-detail-drawer) -- rather than risk
    // the drawer's opaque panel covering the legend on short viewports,
    // styles.css hides .ts-legend whenever the drawer is open (`:has()`),
    // so the two never visually collide either.
    var control = L.control({ position: position || "bottomleft" });
    control.onAdd = function () {
      var div = L.DomUtil.create("div", "ts-legend");
      var classRows = fmt.TRAINED_CLASSES.map(function (cls) {
        return (
          '<div class="ts-legend-row">' +
          '<span class="ts-legend-swatch" style="background:' + fmt.CLASS_COLORS[cls] + '"></span>' +
          '<span>' + fmt.CLASS_LABELS[cls] + "</span>" +
          "</div>"
        );
      }).join("");
      div.innerHTML =
        '<div class="ts-legend-title ts-legend-title-main">Legend</div>' +
        '<div class="ts-legend-title">Predicted class</div>' +
        classRows +
        '<div class="ts-legend-title ts-legend-title-status">Status</div>' +
        '<div class="ts-legend-row"><span class="ts-legend-marker ts-legend-marker-closed"></span><span>Closed</span></div>' +
        '<div class="ts-legend-row"><span class="ts-legend-marker ts-legend-marker-provisional"></span><span>Provisional</span></div>';
      L.DomEvent.disableClickPropagation(div);
      return div;
    };
    return control;
  }

  function buildResetControl(map) {
    var control = L.control({ position: "topleft" });
    control.onAdd = function () {
      var div = L.DomUtil.create("div", "ts-reset-control leaflet-bar");
      var btn = L.DomUtil.create("a", "", div);
      btn.href = "#";
      btn.title = "Reset to Gujarat extent";
      btn.setAttribute("aria-label", "Reset to Gujarat extent");
      btn.innerHTML =
        '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">' +
        '<circle cx="8" cy="8" r="4.5" stroke="currentColor" stroke-width="1.4"/>' +
        '<circle cx="8" cy="8" r="1.2" fill="currentColor"/>' +
        '<line x1="8" y1="0.5" x2="8" y2="3" stroke="currentColor" stroke-width="1.4"/>' +
        '<line x1="8" y1="13" x2="8" y2="15.5" stroke="currentColor" stroke-width="1.4"/>' +
        '<line x1="0.5" y1="8" x2="3" y2="8" stroke="currentColor" stroke-width="1.4"/>' +
        '<line x1="13" y1="8" x2="15.5" y2="8" stroke="currentColor" stroke-width="1.4"/>' +
        "</svg>";
      L.DomEvent.on(btn, "click", function (e) {
        L.DomEvent.preventDefault(e);
        map.fitBounds(map._thermoscopeBounds, { padding: [16, 16] });
      });
      L.DomEvent.disableClickPropagation(div);
      return div;
    };
    return control;
  }

  // Renders all rows as clustered circle markers. onSelect(eventId) is
  // called on marker click. Returns a handle with `selectEvent(eventId)`
  // and `clearSelection()` so app.js can drive selection from the list.
  function renderEvents(map, rows, onSelect) {
    var clusterGroup = L.markerClusterGroup({
      iconCreateFunction: clusterIconCreateFunction,
      maxClusterRadius: 50,
    });

    var circlesById = {};
    var rejected = [];
    rows.forEach(function (row) {
      var lat = fmt.toNumber(row.centroid_lat);
      var lon = fmt.toNumber(row.centroid_lon);
      // Defensive validation before ever creating a marker: an event with
      // an unparseable or out-of-study-area coordinate is skipped and
      // logged, never plotted -- and never "corrected" by clamping/
      // reprojecting it into bounds, which would silently misrepresent
      // where the event actually is. See PROGRESS.md for the 2026-09
      // investigation this guards against recurring undetected.
      if (lat === null || lon === null || !fmt.isValidGujaratCoordinate(lat, lon)) {
        rejected.push({ event_id: row.event_id, centroid_lat: row.centroid_lat, centroid_lon: row.centroid_lon });
        return;
      }
      var circle = L.circleMarker([lat, lon], styleForRow(row, map.getZoom()));
      circle.on("click", function () {
        onSelect(row.event_id);
      });
      circle.bindTooltip(row.event_id + " — " + fmt.CLASS_LABELS[row.predicted_class], { direction: "top" });
      circlesById[row.event_id] = { circle: circle, row: row };
      clusterGroup.addLayer(circle);
    });

    clusterGroup.addTo(map);

    if (rejected.length > 0) {
      // eslint-disable-next-line no-console
      console.warn(
        "ThermoScope: " + rejected.length + " event(s) skipped on the map -- " +
        "invalid or outside the Gujarat study area bounds (never plotted, never coordinate-corrected):",
        rejected
      );
    }

    var selectedId = null;
    function clearSelection() {
      if (selectedId && circlesById[selectedId]) {
        var entry = circlesById[selectedId];
        entry.circle.setStyle(styleForRow(entry.row, map.getZoom()));
      }
      selectedId = null;
    }
    function selectEvent(eventId) {
      clearSelection();
      var entry = circlesById[eventId];
      if (!entry) return; // filtered out of the current map layer -- nothing to highlight
      entry.circle.setStyle(selectedStyleFor(entry.row, map.getZoom()));
      selectedId = eventId;
      clusterGroup.zoomToShowLayer(entry.circle, function () {
        map.panTo(entry.circle.getLatLng());
      });
    }

    // Same visual highlight as selectEvent, but never pans/zooms the map.
    // Used after a background data refresh (60s poll) so an already-open
    // event stays highlighted without the map jumping under the user --
    // no pan/zoom counts as a "visible distracting animation".
    function highlightOnly(eventId) {
      clearSelection();
      var entry = circlesById[eventId];
      if (!entry) return;
      entry.circle.setStyle(selectedStyleFor(entry.row, map.getZoom()));
      selectedId = eventId;
    }

    return {
      clusterGroup: clusterGroup,
      selectEvent: selectEvent,
      highlightOnly: highlightOnly,
      clearSelection: clearSelection,
      rejectedCoordinates: rejected, // exposed for tests/diagnostics; never used to alter rendering
    };
  }

  // ---------------------------------------------------------------
  // Event-detail mini map (drawer column 1): a small, non-interactive
  // local preview centered on the selected event, reusing the exact same
  // tile layers, class colors, cluster styling, and selection-ring style
  // as the main map -- created once and updated in place on every
  // selection change (Leaflet errors if you re-init a map on a container
  // that already has one), never rebuilt via innerHTML like the other 3
  // columns are, so its Leaflet instance is never leaked.
  // ---------------------------------------------------------------
  var detailMap = null;
  var detailClusterGroup = null;
  var detailSelectedCircle = null;

  // Nearby-event radius in degrees (~13km at Gujarat's latitude) -- purely
  // a local-context radius for this thumbnail, unrelated to and no wider
  // than the event's own data; every marker shown is a real row from
  // state.rows, never fabricated.
  var DETAIL_MAP_CONTEXT_RADIUS_DEG = 0.12;

  function renderDetailMap(containerId, row, allRows) {
    var lat = fmt.toNumber(row.centroid_lat);
    var lon = fmt.toNumber(row.centroid_lon);
    if (lat === null || lon === null || !fmt.isValidGujaratCoordinate(lat, lon)) return;

    if (!detailMap) {
      detailMap = L.map(containerId, {
        zoomControl: false,
        attributionControl: false,
        dragging: false,
        scrollWheelZoom: false,
        doubleClickZoom: false,
        boxZoom: false,
        keyboard: false,
        tap: false,
        worldCopyJump: false,
      });
      L.tileLayer(TILE_BASE_URL, { maxZoom: 16 }).addTo(detailMap);
      L.tileLayer(TILE_REF_URL, { maxZoom: 16 }).addTo(detailMap);
      detailClusterGroup = L.markerClusterGroup({
        iconCreateFunction: clusterIconCreateFunction,
        maxClusterRadius: 40,
      });
      detailMap.addLayer(detailClusterGroup);
    } else {
      detailClusterGroup.clearLayers();
      if (detailSelectedCircle) {
        detailMap.removeLayer(detailSelectedCircle);
        detailSelectedCircle = null;
      }
    }

    detailMap.setView([lat, lon], 13);

    // Other real nearby events, for local context only -- rendered through
    // the same neutral cluster styling as the main map (never class-colored,
    // never implying severity). The selected event itself is excluded here
    // and drawn separately below so it can never be merged into a cluster.
    allRows.forEach(function (other) {
      if (other.event_id === row.event_id) return;
      var oLat = fmt.toNumber(other.centroid_lat);
      var oLon = fmt.toNumber(other.centroid_lon);
      if (oLat === null || oLon === null) return;
      if (Math.abs(oLat - lat) > DETAIL_MAP_CONTEXT_RADIUS_DEG || Math.abs(oLon - lon) > DETAIL_MAP_CONTEXT_RADIUS_DEG) return;
      detailClusterGroup.addLayer(L.circleMarker([oLat, oLon], {
        radius: 5, color: "#ffffff", weight: 1.5, fillColor: CLUSTER_COLOR, fillOpacity: 0.95,
      }));
    });

    detailSelectedCircle = L.circleMarker([lat, lon], selectedStyleFor(row, 13)).addTo(detailMap);

    // The container may have been hidden (drawer closed, or just unhidden
    // this tick) when Leaflet computed its size -- re-measure once the
    // layout has settled.
    setTimeout(function () {
      if (detailMap) detailMap.invalidateSize();
    }, 0);
  }

  root.ThermoScopeMap = {
    createMap: createMap,
    renderEvents: renderEvents,
    renderDetailMap: renderDetailMap,
  };
})(typeof window !== "undefined" ? window : this);
