# ThermoScope Frontend

A static, vanilla HTML/CSS/JS monitoring console implementing
[`DESIGN.md`](../DESIGN.md). No build step, no framework, no npm
dependency for the app itself — the only external code is Leaflet +
Leaflet.markercluster, loaded via `<script>`/`<link>` tags from a CDN.

It reads the real pipeline artifacts directly at runtime:
- `data/processed/gujarat_2026_inference.csv`
- `data/processed/gujarat_2026_inference_freshness.json`

Nothing is copied, cached, or duplicated — every page load re-fetches
whatever `src/update_2026_nrt.py` last wrote.

## Running it

From the repository root:

```
python3 -m http.server 8000
```

Then open `http://localhost:8000/frontend/`.

The server must be started from the repo root (not from `frontend/`)
so that the relative fetch of `../data/processed/...` resolves
correctly.

## Tests

Pure logic (CSV parsing, formatting, filtering, staleness) is unit
tested under Node (no browser needed):

```
cd frontend
node --test tests/*.test.js
```

Map rendering and DOM wiring (`js/map.js`, `js/app.js`) are
browser-only and were verified manually against the live app per the
implementation report, not via this test suite.
