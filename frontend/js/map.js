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

  var SELECTED_COLOR = "#2f6fed";
  var CLUSTER_COLOR = "#5b6169";

  function createMap(containerId) {
    var bounds = [
      [fmt.GUJARAT_BOUNDS.latMin, fmt.GUJARAT_BOUNDS.lonMin],
      [fmt.GUJARAT_BOUNDS.latMax, fmt.GUJARAT_BOUNDS.lonMax],
    ];
    var map = L.map(containerId, { zoomControl: true, attributionControl: true });
    map.fitBounds(bounds, { padding: [16, 16] });

    // Esri World Light Gray Canvas (base + reference-label layers stacked):
    // a muted, desaturated basemap with minimal labeling (DESIGN.md §3/§5),
    // free and keyless. (An earlier CARTO Positron tile set was tried
    // first but now requires an API key -- verified live, see PROGRESS.md.)
    L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
      { attribution: "Esri, HERE, Garmin, © OpenStreetMap contributors", maxZoom: 16 }
    ).addTo(map);
    L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 16 }
    ).addTo(map); // added after the base layer -- same pane, stacks on top by DOM order

    map._thermoscopeBounds = bounds;
    buildLegendControl().addTo(map);
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
      color: color,
      weight: closed ? 1.5 : 2.25,
      fillColor: color,
      fillOpacity: closed ? 0.88 : 0.18, // solid fill = closed, hollow/outline = provisional
      dashArray: closed ? null : "3,2",
    };
  }

  function selectedStyleFor(row, zoom) {
    var base = styleForRow(row, zoom);
    base.color = SELECTED_COLOR;
    base.weight = 3;
    base.radius = base.radius + 2;
    return base;
  }

  function buildLegendControl() {
    var control = L.control({ position: "bottomleft" });
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
      btn.textContent = "Reset view";
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
      iconCreateFunction: function (cluster) {
        var count = cluster.getChildCount();
        var tier = count < 10 ? "sm" : count < 50 ? "md" : "lg";
        return L.divIcon({
          html: '<div class="ts-cluster ts-cluster-' + tier + '">' + count + "</div>",
          className: "ts-cluster-wrap",
          iconSize: null,
        });
      },
      maxClusterRadius: 50,
    });

    var circlesById = {};
    rows.forEach(function (row) {
      var lat = fmt.toNumber(row.centroid_lat);
      var lon = fmt.toNumber(row.centroid_lon);
      if (lat === null || lon === null) return; // never plot a fabricated location
      var circle = L.circleMarker([lat, lon], styleForRow(row, map.getZoom()));
      circle.on("click", function () {
        onSelect(row.event_id);
      });
      circle.bindTooltip(row.event_id + " — " + fmt.CLASS_LABELS[row.predicted_class], { direction: "top" });
      circlesById[row.event_id] = { circle: circle, row: row };
      clusterGroup.addLayer(circle);
    });

    clusterGroup.addTo(map);

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
    };
  }

  root.ThermoScopeMap = {
    createMap: createMap,
    renderEvents: renderEvents,
  };
})(typeof window !== "undefined" ? window : this);
