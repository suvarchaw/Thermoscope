# ThermoScope

**Smart India Hackathon 2026 — Problem Statement SIH26162 (NTRO)**
"AI-Based Detection and Classification of Industrial Fires and Persistent
Thermal Sources Using NASA FIRMS, OSM & Satellite Data."
Theme: Disaster Management · Category: Software · Team: Hacksmiths

> **Status: working prototype, inference-only.** Every number and screenshot
> below is produced by the actual code in this repository — nothing here is
> a mockup or a projection. The trained model is disclosed to generalize
> well for two of its four classes and poorly for the other two (see
> [Evaluation](#evaluation--honest-results)) — that limitation is shown
> directly in the app, not hidden.

---

## 1. What ThermoScope Does

NASA FIRMS already detects thermal anomalies (fire/heat hotspots) from
satellites anywhere on Earth. A single raw FIRMS detection only says
*"something hot was here, at this time"* — it does not say what caused it,
whether it is a one-off grass fire or a recurring industrial source, or
which of thousands of daily detections actually deserves an analyst's
attention.

ThermoScope does not try to re-detect fires. It adds a layer on top of
FIRMS:

1. **Groups** raw FIRMS detections into spatiotemporal **events**.
2. **Attaches evidence** to each event — nearby OpenStreetMap industrial/
   power/flare/kiln sites, WRI power-plant records, ESA WorldCover land
   cover, and overlap with a known persistent hotspot cluster.
3. **Classifies** each event into one of four source types (Industrial,
   Gas Flare, Crop Residue, Forest/Wildfire) using a trained LightGBM
   model.
4. **Re-runs automatically every 30 minutes** on fresh 2026 FIRMS data (the
   "NRT" part — see [§5](#5-near-real-time-nrt-operation)), without ever
   retraining the model.
5. **Displays results on a live monitoring dashboard** — map, filters,
   time windows, and a per-event detail view with model probabilities,
   always labeled as model output, never as verified ground truth.

---

## 2. Architecture

```
NASA FIRMS (VIIRS_SNPP_SP historical / VIIRS_SNPP_NRT recent)
        │
        ▼
firms_ingestion.py ──► data/raw/firms_gujarat_{year}.csv   (2019-2025, historical)
        │                                            firms_gujarat_2026.csv (2026, dedup-merged)
        ▼
┌────────────────────────────────┬─────────────────────────────────────┐
│ HISTORICAL / TRAINING BRANCH   │ 2026 NRT BRANCH                      │
│                                 │                                       │
│ event_construction.py          │ event_construction_2026.py           │
│  → gujarat_thermal_events.csv  │  → gujarat_2026_events.csv           │
│                                 │    (EVT2026_* ids, separate file)    │
│ event_evidence.py              │ event_evidence_2026.py               │
│  → gujarat_event_evidence.csv  │  → gujarat_event_evidence_2026.csv   │
│                                 │                                       │
│ event_behavior_features.py     │ (same feature functions reused       │
│ build_event_silver_labels.py   │  directly inside infer_2026_events)  │
│  → gujarat_event_silver_labels.csv                                    │
│                                 │                                       │
│ train_source_classifier_lightgbm.py                                   │
│  → models/source_classifier_lightgbm_v2.joblib (TRAINED ONCE, LOCKED) │
│                                 │                                       │
│                                 │ infer_2026_events.py                 │
│                                 │  loads the .joblib, .predict only    │
│                                 │  → gujarat_2026_inference.csv        │
└────────────────────────────────┴─────────────────────────────────────┘
                                             │
                     update_2026_nrt.py orchestrates the whole 2026
                     branch every 30 minutes, and writes
                     gujarat_2026_inference_freshness.json
                                             │
                                             ▼
                          frontend/ (reads the CSV + JSON directly)
```

A second, earlier spatial-clustering track (60 DBSCAN-derived hotspot
clusters from the fixed 2023 dataset — `thermal_clustering.py`,
`build_cluster_intelligence_layer.py`, and related scripts) exists
alongside events, not instead of them. It is used only as one evidence
signal (`overlaps_cluster_id` / recurrence strength) attached to events —
it is not what gets classified or shown on the 2026 map.

---

## 3. Repository Structure

```
src/                # All pipeline code — ingestion, event construction,
                     # evidence joins, feature engineering, training,
                     # evaluation, and the 2026 NRT engine (flat, 44 files)
data/
  raw/               # Original FIRMS CSVs, per year — read-only in practice
  processed/         # Every derived artifact: events, evidence, silver
                      # labels, features, 2026 inference output + freshness
  processed/models/  # The persisted trained model (.joblib) + provenance.json
  external_cache/    # Cached ESA WorldCover raster tiles (gitignored)
ops/                 # ops/com.thermoscope.nrt_update.plist — the macOS
                      # LaunchAgent that schedules the NRT engine
frontend/            # Static HTML/CSS/vanilla-JS monitoring dashboard
                      # (Leaflet map, filters, event-detail drawer)
tests/               # 33 Python test files (496 test methods), one per
                      # src/ module
notebooks/           # Exploratory notebooks (not part of the pipeline)
results/             # Exploratory maps/figures output
logs/                # Runtime stdout/stderr from the scheduled NRT job
```

Data-directory convention: `data/raw/` is original and unmodified;
`data/processed/` holds every derived/cleaned artifact including the final
2026 inference output.

---

## 4. Machine Learning

- **Labels**: there is no human-annotated ground truth. `build_event_silver_labels.py`
  applies five deterministic, evidence-based rules (Industrial, Gas_Flare,
  Brick_Kiln, Crop_Residue, Forest_Wildfire) to produce a rule-derived
  **silver label** per event — explicitly not verified ground truth. An
  event matching no rule, or more than one, is labeled `Unknown_Ambiguous`
  and excluded from training (9,774 of 20,409 historical events — 47.9%).
- **Features (11, leakage-audited)**: `centroid_lat`, `centroid_lon`,
  `spatial_extent_m`, `status` (categorical), plus 7 behavioral features
  from raw FIRMS fields (`frac_high_confidence`, `frac_low_confidence`,
  `mean_scan`, `mean_track`, `elongation_ratio`,
  `time_of_day_std_minutes`, `detections_per_day`). Every column any
  silver-label rule reads is enforced (at runtime, not just by convention)
  to be excluded from this feature set, so the model cannot simply
  re-derive the labeling rule.
- **Model**: LightGBM, trained once on historical events with a confident
  single-class silver label, train = 2019–2022, test = 2023–2025. A
  Logistic Regression model is kept only as a comparison baseline
  (`train_source_classifier.py`).
- **Trained classes (4)**: `Industrial`, `Gas_Flare`, `Crop_Residue`,
  `Forest_Wildfire`. `Brick_Kiln` and `Unknown_Ambiguous` are excluded from
  training — on the real Gujarat data, zero events ever matched the
  Brick_Kiln rule (no OSM kiln/brickyard evidence within 500m in this
  region — a real external-data gap, not a disabled code path). This means
  the model is **forced** to output one of the 4 trained classes for every
  event, even a genuine brick kiln or ambiguous case — there is no "none
  of the above."
- **Inference**: `infer_2026_events.py` only calls `.predict`/`.predict_proba`
  on the already-fitted `.joblib` — it never calls `.fit`. This is verified
  by a source-inspection test, not just a docstring claim.

### Evaluation — honest results

| Split | Balanced accuracy | What it tests |
|---|---|---|
| **Temporal** (train 2019–2022, test 2023–2025) | **0.935** | Generalizes forward in time |
| **Spatial holdout** (0.5°×0.5° checkerboard, entire cells withheld) | **0.459** | Generalizes to *new locations* |

These numbers come from `data/processed/source_classifier_lightgbm_metrics.csv`
and `data/processed/spatial_evaluation_metrics.csv` — not restated from
memory. **Do not quote "93.5% accuracy" without the spatial number next to
it.** The gap is real and traced: 99.0% of test-set Industrial events and
~85% of Gas_Flare events sit within 300m of a same-class *training* event —
both are fixed physical infrastructure, so the temporal split largely
measures "recognizing a known site," not learning what a gas flare looks
like. Under spatial holdout, Gas_Flare's F1 collapses to 0.000 and
Industrial's to ~0.11, while Crop_Residue (F1 ≈ 0.78) and Forest_Wildfire
(F1 ≈ 0.81) — which aren't tied to fixed sites — generalize much better.
This distinction is surfaced directly in the frontend as a per-class
`class_capability_note`, never hidden behind the headline number.

---

## 5. Near-Real-Time (NRT) Operation

"NRT" here means **scheduled polling**, not streaming — there is no
push/websocket mechanism anywhere.

- Every **30 minutes**, a macOS LaunchAgent (`ops/com.thermoscope.nrt_update.plist`,
  `StartInterval=1800`) runs `python3 src/update_2026_nrt.py`.
- That script fetches the **last 7 days** of `VIIRS_SNPP_NRT` detections,
  merges/dedupes them into the existing 2026 raw store, then **re-derives
  all 2026 events, evidence, and predictions from scratch** from that
  store — safe and idempotent because `event_construction.py`'s
  connected-components algorithm is provably "online-safe": a full re-run
  and an incremental run agree on every already-closed event.
- The **same locked model** is used every cycle via predict-only calls —
  it is never retrained in production.
- A PID lock file (atomic `O_CREAT|O_EXCL`) guards each cycle so an
  overlapping scheduled run exits cleanly as "SKIPPED" instead of
  corrupting output; a lock is only reclaimed once the recorded PID is
  confirmed no longer running.
- A small `gujarat_2026_inference_freshness.json` sidecar is written only
  after a fully successful cycle — the frontend polls this file every
  ~60 seconds and re-fetches the CSV (re-rendering in place, no page
  reload) only when it changes.
- If a cycle fails (e.g. missing API key), it's caught and printed as
  `BLOCKED`, the lock is always released, and the freshness file simply
  isn't updated — which is exactly what the frontend's staleness banner
  is designed to detect.

---

## 6. Frontend

A static HTML/CSS/vanilla-JS dashboard — **no build step, no framework**,
served directly from disk. The only external code is Leaflet +
Leaflet.markercluster, loaded via CDN tags.

It reads `data/processed/gujarat_2026_inference.csv` and
`gujarat_2026_inference_freshness.json` directly via `fetch()` — nothing
is copied, cached, or pre-processed. See [`frontend/README.md`](frontend/README.md)
for frontend-specific details.

Features: a Gujarat map with class-colored event markers and neutral
(never class-colored) clusters; a time-window control (Today / 24 Hrs /
7 Days / Calendar, all in IST); class/status/land-cover filters and
event-ID search; an event list; and a 4-column event-detail drawer
(local map, Observed Data, External Evidence, Model Output) with
per-class probability bars explicitly labeled "not verified ground
truth." A background poll checks for new data every ~60 seconds and
updates in place, preserving whatever filters/selection are active.

---

## 7. Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

To fetch live data from NASA FIRMS, create a `.env` file in the repo root
(gitignored, never commit it):

```
FIRMS_MAP_KEY=your_key_here
```

Get a key at https://firms.modaps.eosdis.nasa.gov/api/map_key/. No key is
needed to run the frontend against the data already committed in
`data/processed/`, or to run the test suite.

---

## 8. Running It

**Frontend dashboard** (from the repo root, so relative paths to
`data/processed/` resolve):

```bash
python3 -m http.server 8000
```

then open `http://localhost:8000/frontend/`.

**Run one NRT update cycle manually:**

```bash
python3 src/update_2026_nrt.py
```

**Schedule it to run automatically every 30 minutes** (macOS):

```bash
mkdir -p ~/Library/LaunchAgents
cp ops/com.thermoscope.nrt_update.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.thermoscope.nrt_update.plist
```

To stop it: `launchctl unload ~/Library/LaunchAgents/com.thermoscope.nrt_update.plist`

---

## 9. Testing

```bash
# Backend (33 files, 496 test methods)
for f in tests/test_*.py; do python3 "$f"; done

# Frontend (38 tests, pure-logic modules only — no browser needed)
cd frontend && node --test tests/*.test.js
```

Map rendering and DOM behavior are verified manually against the live app
(not covered by the Node suite, which tests only `format.js`/`csv.js`).

---

## 10. Data Sources

| Source | Contributes |
|---|---|
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) (VIIRS 375m) | Raw thermal detections — the only external source ThermoScope observes anomalies from |
| [OpenStreetMap](https://overpass-api.de/) (Overpass API) | Nearest industrial/power, flare, and kiln/brickyard evidence |
| [WRI Global Power Plant Database](https://datasets.wri.org/dataset/globalpowerplantdatabase) | Nearest coal/gas/oil thermal power plant evidence |
| [ESA WorldCover](https://esa-worldcover.org/) | Land-cover class at each event's centroid |

## Evidence vs. features vs. prediction

These are three different things, and ThermoScope keeps them separate on
purpose:

- **Evidence** — raw observed facts (a distance, a land-cover string) —
  never itself a classification.
- **Model features** — the narrow, audited 11-column set actually fed to
  the classifier. Evidence columns are excluded so the model can't just
  memorize "near a flare → predict Gas Flare."
- **Prediction** — the model's own output, produced only from the
  features. Evidence and prediction are shown side by side in the
  frontend so a human can sanity-check the model's call.

---

## 11. Limitations

- Silver labels are rule-derived, not human-verified ground truth.
- The trained model cannot output `Brick_Kiln` or `Unknown_Ambiguous` —
  every 2026 event is forced into one of 4 classes even when neither
  genuinely fits.
- Industrial and Gas Flare predictions primarily reflect known-site
  recognition, not generalization to new, unseen locations (see
  [Evaluation](#evaluation--honest-results)).
- "24 Hrs" in the frontend is a disclosed date-level approximation, not a
  true rolling 24-hour window (the inference CSV carries date-level, not
  timestamp-level, event boundaries).
- Gujarat / the current bounding box is a working assumption, not a
  validated final scope.
- No license file is currently included in this repository.

---


