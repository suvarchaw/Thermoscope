/*
 * Minimal RFC4180-style CSV parser: handles quoted fields (including
 * embedded commas, embedded newlines, and escaped "" quotes), which a
 * naive String.split(",") would break on. No external dependency --
 * the real inference CSV is small enough (a few thousand rows) that a
 * hand-written parser is simpler than pulling in a parsing library.
 *
 * parseCSV(text) -> array of row objects keyed by the header row.
 * Universal module: usable via <script> in the browser (attaches to
 * window.ThermoScopeCSV) and via require() under Node for testing.
 */
(function (root, factory) {
  var mod = factory();
  if (typeof module !== "undefined" && module.exports) {
    module.exports = mod;
  } else {
    root.ThermoScopeCSV = mod;
  }
})(typeof self !== "undefined" ? self : this, function () {
  function parseCSV(text) {
    var rows = [];
    var row = [];
    var field = "";
    var inQuotes = false;
    var i = 0;
    var len = text.length;

    function endField() {
      row.push(field);
      field = "";
    }
    function endRow() {
      endField();
      rows.push(row);
      row = [];
    }

    while (i < len) {
      var c = text[i];
      if (inQuotes) {
        if (c === '"') {
          if (text[i + 1] === '"') {
            field += '"';
            i += 2;
            continue;
          }
          inQuotes = false;
          i += 1;
          continue;
        }
        field += c;
        i += 1;
        continue;
      }
      if (c === '"') {
        inQuotes = true;
        i += 1;
        continue;
      }
      if (c === ",") {
        endField();
        i += 1;
        continue;
      }
      if (c === "\r") {
        i += 1;
        continue;
      }
      if (c === "\n") {
        endRow();
        i += 1;
        continue;
      }
      field += c;
      i += 1;
    }
    // last field/row (file may or may not end with a trailing newline)
    if (field.length > 0 || row.length > 0) {
      endRow();
    }

    if (rows.length === 0) return [];
    var header = rows[0];
    var out = [];
    for (var r = 1; r < rows.length; r++) {
      var raw = rows[r];
      if (raw.length === 1 && raw[0] === "") continue; // skip trailing blank line
      var obj = {};
      for (var c2 = 0; c2 < header.length; c2++) {
        obj[header[c2]] = raw[c2] !== undefined ? raw[c2] : "";
      }
      out.push(obj);
    }
    return out;
  }

  return { parseCSV: parseCSV };
});
