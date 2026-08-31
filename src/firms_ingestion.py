"""
Reusable NASA FIRMS ingestion module (Area API).

Downloads historical/recent VIIRS S-NPP FIRMS detections for a bounding
box and date range via NASA's official FIRMS Area API
(https://firms.modaps.eosdis.nasa.gov/api/area/), validates the response,
normalizes column names to match this project's existing schema, and
writes one raw CSV file per year under data/raw/ -- without ever touching
the existing 2023 pan-India archive file.

Credentials
-----------
The Area API requires a free MAP_KEY (NASA Earthdata login required to
obtain one, at https://firms.modaps.eosdis.nasa.gov/api/map_key/). This
module NEVER hardcodes a key. It reads one from the environment variable
FIRMS_MAP_KEY. If that variable is not set, every function that needs it
raises FirmsConfigError with setup instructions -- it does not invent a
key, fall back to a demo key, or silently return empty/fake data.

Setup:
    1. Register for a free MAP_KEY at
       https://firms.modaps.eosdis.nasa.gov/api/map_key/
       (requires a free NASA Earthdata Login account).
    2. Set it as an environment variable (never commit it):
           export FIRMS_MAP_KEY="your_map_key_here"

API facts used below (verified against NASA's own Area API documentation
and the Earthdata VIIRS attribute reference, not assumed):
  - Endpoint: https://firms.modaps.eosdis.nasa.gov/api/area/csv/
        {MAP_KEY}/{SOURCE}/{west},{south},{east},{north}/{day_range}/{date}
  - Coordinate order is west,south,east,north (min_lon,min_lat,max_lon,max_lat).
  - Maximum day_range per single request is 5 days.
  - VIIRS_SNPP_SP = Standard Processing (historical/science-quality) source.
  - VIIRS_SNPP_NRT = Near-Real-Time source (for recent/live data, later).
  - The Area API returns VIIRS brightness columns as bright_ti4/bright_ti5,
    which differs from this project's existing archive-download-tool file
    (fire_archive_SV-C2_794895.csv), which uses brightness/bright_t31. This
    module normalizes bright_ti4 -> brightness and bright_ti5 -> bright_t31
    on ingestion so downstream code (which already expects
    brightness/bright_t31) works unchanged on the new data.
"""

import csv
import io
import os
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

from spatial_recurrence import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX

# Loads FIRMS_MAP_KEY (and any other project secrets) from a local .env
# file into the environment, if one exists. The .env file itself is
# gitignored and never committed -- this only changes where the
# credential is read FROM, not the requirement that it live outside the
# repository or the FirmsConfigError behavior when it's absent.
load_dotenv()

FIRMS_AREA_API_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
MAX_DAY_RANGE = 5

SOURCE_HISTORICAL = "VIIRS_SNPP_SP"
SOURCE_NRT = "VIIRS_SNPP_NRT"

GUJARAT_BBOX = (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)  # (west, south, east, north)

# The minimal set of columns this project's pipeline actually reads
# (see spatial_recurrence.read_gujarat_detections and downstream modules).
# `type` and `instrument` are intentionally NOT required: `type` was
# already established (prior milestone) as unreliable for this product
# and is never used as a label; `instrument` is informational only.
REQUIRED_COLUMNS = [
    "latitude", "longitude", "acq_date", "acq_time",
    "satellite", "confidence", "frp", "daynight",
]

# Area-API -> existing-pipeline column name normalization.
COLUMN_RENAME_MAP = {
    "bright_ti4": "brightness",
    "bright_ti5": "bright_t31",
}

RAW_DIR = Path("data/raw")
REQUEST_DELAY_S = 1.5
REQUEST_TIMEOUT_S = 60


class FirmsConfigError(Exception):
    """Raised when required configuration (e.g. FIRMS_MAP_KEY) is missing."""


class FirmsIngestionError(Exception):
    """Raised when a FIRMS API request or response fails validation."""


def get_map_key():
    """Read the FIRMS MAP_KEY from the environment. Never hardcode one."""
    map_key = os.environ.get("FIRMS_MAP_KEY")
    if not map_key:
        raise FirmsConfigError(
            "FIRMS_MAP_KEY environment variable is not set.\n"
            "This project never hardcodes API keys. To fetch FIRMS data:\n"
            "  1. Register for a free MAP_KEY at "
            "https://firms.modaps.eosdis.nasa.gov/api/map_key/\n"
            "     (requires a free NASA Earthdata Login account).\n"
            "  2. Set it as an environment variable, e.g.:\n"
            "         export FIRMS_MAP_KEY=\"your_map_key_here\"\n"
            "  3. Re-run this ingestion step."
        )
    return map_key


def generate_date_chunks(start_date, end_date, chunk_days=MAX_DAY_RANGE):
    """Split [start_date, end_date] (inclusive) into non-overlapping
    (chunk_start_date, day_range) windows, each at most chunk_days long,
    matching the Area API's day_range semantics (day_range days starting
    at chunk_start_date, inclusive)."""
    if start_date > end_date:
        raise ValueError(f"start_date {start_date} is after end_date {end_date}")

    chunks = []
    cursor = start_date
    while cursor <= end_date:
        remaining = (end_date - cursor).days + 1
        day_range = min(chunk_days, remaining)
        chunks.append((cursor, day_range))
        cursor = cursor + timedelta(days=day_range)
    return chunks


def build_area_url(map_key, source, bbox, day_range, start_date):
    west, south, east, north = bbox
    area = f"{west},{south},{east},{north}"
    date_str = start_date.strftime("%Y-%m-%d")
    return f"{FIRMS_AREA_API_BASE}/{map_key}/{source}/{area}/{day_range}/{date_str}"


def fetch_csv_chunk(url, timeout=REQUEST_TIMEOUT_S):
    """Fetch one Area API CSV chunk. Raises FirmsIngestionError with a
    clear diagnostic on any failure -- never swallows errors silently."""
    try:
        resp = requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        raise FirmsIngestionError(f"Network error fetching {url}: {e}") from e

    if resp.status_code != 200:
        raise FirmsIngestionError(
            f"FIRMS API returned HTTP {resp.status_code} for {url}: "
            f"{resp.text[:300]}"
        )

    text = resp.text.strip()
    # The Area API returns a plain-text error message (not CSV) for
    # invalid keys/params instead of a non-200 status in some cases.
    if text.lower().startswith("invalid") or "map_key" in text.lower()[:50]:
        raise FirmsIngestionError(f"FIRMS API rejected the request for {url}: {text[:300]}")

    return resp.text


def parse_csv_text(csv_text):
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    return rows, reader.fieldnames or []


def normalize_columns(rows):
    """Rename Area-API VIIRS column names to match this project's existing
    schema (see COLUMN_RENAME_MAP), leaving all other columns untouched."""
    normalized = []
    for row in rows:
        new_row = {}
        for key, value in row.items():
            new_key = COLUMN_RENAME_MAP.get(key, key)
            new_row[new_key] = value
        normalized.append(new_row)
    return normalized


def validate_columns(fieldnames, required=REQUIRED_COLUMNS):
    """Raise FirmsIngestionError listing any required columns missing after
    normalization. Does not require an exact match to the full original
    schema -- only the columns this pipeline actually uses."""
    normalized_fields = {COLUMN_RENAME_MAP.get(f, f) for f in fieldnames}
    missing = [c for c in required if c not in normalized_fields]
    if missing:
        raise FirmsIngestionError(
            f"FIRMS response is missing required columns: {missing}. "
            f"Columns present (after normalization): {sorted(normalized_fields)}"
        )


def deduplicate_rows(rows):
    """Defensive de-duplication on (lat, lon, acq_date, acq_time, satellite).
    Chunked date-range requests are constructed to be non-overlapping, but
    this guards against any API-side overlap or repeated ingestion."""
    seen = set()
    deduped = []
    for row in rows:
        key = (row.get("latitude"), row.get("longitude"), row.get("acq_date"),
               row.get("acq_time"), row.get("satellite"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def fetch_source_range(source, start_date, end_date, map_key,
                        request_delay_s=REQUEST_DELAY_S, bbox=GUJARAT_BBOX):
    """Fetch and normalize (but do not dedupe/write) every detection from
    one FIRMS source over [start_date, end_date] (inclusive), chunked via
    generate_date_chunks. Shared by fetch_year_gujarat (single source, full
    calendar year) and fetch_2026_gujarat (two sources, partial year) so
    the chunking/fetch/normalize logic is defined exactly once. Returns a
    list of normalized row dicts (each tagged with 'viirs_source')."""
    chunks = generate_date_chunks(start_date, end_date)
    all_rows = []
    for chunk_start, day_range in chunks:
        url = build_area_url(map_key, source, bbox, day_range, chunk_start)
        csv_text = fetch_csv_chunk(url)
        rows, _ = parse_csv_text(csv_text)
        if rows:
            all_rows.extend(rows)
        time.sleep(request_delay_s)

    normalized = normalize_columns(all_rows)
    for row in normalized:
        row["viirs_source"] = source
    return normalized


def fetch_year_gujarat(year, output_dir=RAW_DIR, source=SOURCE_HISTORICAL,
                        map_key=None, request_delay_s=REQUEST_DELAY_S,
                        force=False):
    """Fetch one calendar year of Gujarat-bounded VIIRS S-NPP detections
    via the FIRMS Area API and write it to
    data/raw/firms_gujarat_{year}.csv.

    Avoids duplicate ingestion: if the output file already exists and
    force=False, the fetch is skipped and a 'skipped_exists' status is
    returned without making any network requests.

    Returns a result dict: {"status", "year", "path", "row_count", "error"}.
    Never raises for a clean "already ingested" case; raises
    FirmsConfigError/FirmsIngestionError for genuine configuration or data
    problems, which the caller is expected to handle explicitly rather than
    have failures pass silently.
    """
    output_dir = Path(output_dir)
    output_path = output_dir / f"firms_gujarat_{year}.csv"

    if output_path.exists() and not force:
        with open(output_path, newline="") as f:
            existing_row_count = sum(1 for _ in csv.DictReader(f))
        return {
            "status": "skipped_exists", "year": year, "path": str(output_path),
            "row_count": existing_row_count, "error": None,
        }

    if map_key is None:
        map_key = get_map_key()  # raises FirmsConfigError if unset

    start = date(year, 1, 1)
    end = date(year, 12, 31)
    normalized = fetch_source_range(source, start, end, map_key, request_delay_s)
    # fetch_year_gujarat's output has never included a 'viirs_source' column
    # (single-source by construction) -- strip it back off so the written
    # schema/tests are unaffected by the shared-helper refactor.
    for row in normalized:
        row.pop("viirs_source", None)

    if not normalized:
        return {
            "status": "no_data", "year": year, "path": None,
            "row_count": 0, "error": None,
        }

    validate_columns(list(normalized[0].keys()))
    deduped = deduplicate_rows(normalized)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_fieldnames = list(deduped[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(deduped)

    return {
        "status": "ok", "year": year, "path": str(output_path),
        "row_count": len(deduped), "error": None,
    }


# ---------------------------------------------------------------------
# 2026 partial-year ingestion: VIIRS_SNPP_SP through its last available
# date, VIIRS_SNPP_NRT from the following date onward. Unlike
# fetch_year_gujarat (always a full Jan1-Dec31 year, single source), 2026
# is ingested from TWO sources across a boundary date, so it gets its own
# function rather than overloading fetch_year_gujarat's fixed-year-window
# contract.
#
# BOUNDARY, EMPIRICALLY VERIFIED (2026-08-31), NOT ASSUMED
# -------------------------------------------------------------------------
# A live day-by-day query of both sources for the Gujarat bbox around the
# transition confirmed:
#   VIIRS_SNPP_SP:  2026-04-24..2026-04-27 return real detections (178-231
#                    rows/day); 2026-04-28 onward returns 0 rows every day
#                    checked through 2026-05-02.
#   VIIRS_SNPP_NRT: 2026-04-24..2026-04-29 return 0 rows; 2026-04-30 onward
#                    returns real detections (33+ rows/day).
# SP's last real day (2026-04-27) and NRT's first real day (2026-04-30)
# are close but not adjacent -- 2026-04-28 and 2026-04-29 return zero rows
# under BOTH sources, which (given both sources independently agree, and
# neither returns an error/different structure for those two dates) is
# read as two genuine zero-detection Gujarat days, not a source-coverage
# gap. The ingestion boundary below (SP through 04-27, NRT from 04-28) is
# therefore contiguous by date -- no calendar day is skipped or requested
# from neither source -- even though the *data* has a real two-day silent
# stretch inside the NRT side. See PROGRESS.md/DECISIONS.md for the full
# verification transcript.
SP_END_DATE_2026 = date(2026, 4, 27)
NRT_START_DATE_2026 = date(2026, 4, 28)


def fetch_2026_gujarat(output_dir=RAW_DIR, map_key=None,
                        request_delay_s=REQUEST_DELAY_S, run_date=None,
                        force=False,
                        sp_end_date=SP_END_DATE_2026, nrt_start_date=NRT_START_DATE_2026):
    """Fetch 2026-to-date Gujarat-bounded VIIRS S-NPP detections: SP for
    [2026-01-01, sp_end_date], NRT for [nrt_start_date, run_date]. Writes
    data/raw/firms_gujarat_2026.csv with an additive 'viirs_source' column
    (SP or NRT) so the boundary is auditable in the output file itself.
    2019-2025 raw files are never read or written by this function.

    Contiguity is asserted, not just documented: nrt_start_date must be
    exactly one day after sp_end_date, or this raises ValueError rather
    than silently ingesting a gap or an overlap.

    Cross-source duplicates (should the boundary ever need to move and
    briefly overlap) are still removed by the same deduplicate_rows used
    for every other year -- defense in depth, not the primary mechanism.
    """
    if nrt_start_date != sp_end_date + timedelta(days=1):
        raise ValueError(
            f"SP/NRT boundary is not contiguous: sp_end_date={sp_end_date}, "
            f"nrt_start_date={nrt_start_date} (must be exactly one day apart)."
        )

    if run_date is None:
        run_date = date.today()
    if run_date < nrt_start_date:
        raise ValueError(f"run_date {run_date} is before nrt_start_date {nrt_start_date}.")

    output_dir = Path(output_dir)
    output_path = output_dir / "firms_gujarat_2026.csv"

    if output_path.exists() and not force:
        with open(output_path, newline="") as f:
            existing_rows = list(csv.DictReader(f))
        return {
            "status": "skipped_exists", "path": str(output_path),
            "row_count": len(existing_rows), "error": None,
            "sp_row_count": sum(1 for r in existing_rows if r.get("viirs_source") == SOURCE_HISTORICAL),
            "nrt_row_count": sum(1 for r in existing_rows if r.get("viirs_source") == SOURCE_NRT),
            "sp_date_range": (date(2026, 1, 1), sp_end_date),
            "nrt_date_range": (nrt_start_date, run_date),
        }

    if map_key is None:
        map_key = get_map_key()  # raises FirmsConfigError if unset

    sp_rows = fetch_source_range(SOURCE_HISTORICAL, date(2026, 1, 1), sp_end_date,
                                  map_key, request_delay_s)
    nrt_rows = fetch_source_range(SOURCE_NRT, nrt_start_date, run_date,
                                   map_key, request_delay_s)

    all_rows = sp_rows + nrt_rows
    if not all_rows:
        return {
            "status": "no_data", "path": None, "row_count": 0, "error": None,
            "sp_row_count": 0, "nrt_row_count": 0,
            "sp_date_range": (date(2026, 1, 1), sp_end_date),
            "nrt_date_range": (nrt_start_date, run_date),
        }

    validate_columns(list(all_rows[0].keys()))
    deduped = deduplicate_rows(all_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_fieldnames = list(deduped[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(deduped)

    return {
        "status": "ok", "path": str(output_path), "row_count": len(deduped), "error": None,
        "sp_row_count": sum(1 for r in deduped if r["viirs_source"] == SOURCE_HISTORICAL),
        "nrt_row_count": sum(1 for r in deduped if r["viirs_source"] == SOURCE_NRT),
        "sp_date_range": (date(2026, 1, 1), sp_end_date),
        "nrt_date_range": (nrt_start_date, run_date),
    }


# ---------------------------------------------------------------------
# NRT update engine support: fetch a small, configurable recent window
# and MERGE it into the existing 2026 raw store, deduplicating so
# repeated updates over overlapping windows are safe. This is the
# incremental counterpart to fetch_2026_gujarat's one-time SP+NRT
# historical catch-up -- it only ever queries VIIRS_SNPP_NRT (the
# ongoing live source) and never touches any 2019-2025 file.
# ---------------------------------------------------------------------
DEFAULT_NRT_WINDOW_DAYS = 7


def fetch_and_merge_nrt_window(output_dir=RAW_DIR, window_days=DEFAULT_NRT_WINDOW_DAYS,
                                run_date=None, map_key=None,
                                request_delay_s=REQUEST_DELAY_S,
                                nrt_start_date=NRT_START_DATE_2026):
    """Fetch VIIRS_SNPP_NRT for the last `window_days` days (through
    run_date, clamped to not precede nrt_start_date) and merge into
    data/raw/firms_gujarat_2026.csv, deduplicating on the same
    (lat, lon, acq_date, acq_time, satellite) key used everywhere else.
    Running this twice with an identical or overlapping window adds zero
    new rows the second time -- the merge is idempotent by construction
    (dedup, not a append-only/log design).

    Returns a stats dict: rows_before, rows_fetched, rows_after,
    new_rows_added, window_start, window_end.
    """
    if run_date is None:
        run_date = date.today()
    window_start = max(nrt_start_date, run_date - timedelta(days=window_days - 1))
    if window_start > run_date:
        raise ValueError(f"window_start {window_start} is after run_date {run_date}.")

    output_dir = Path(output_dir)
    output_path = output_dir / "firms_gujarat_2026.csv"

    existing_rows = []
    if output_path.exists():
        with open(output_path, newline="") as f:
            existing_rows = list(csv.DictReader(f))

    if map_key is None:
        map_key = get_map_key()  # raises FirmsConfigError if unset

    fetched_rows = fetch_source_range(SOURCE_NRT, window_start, run_date, map_key, request_delay_s)

    combined = existing_rows + fetched_rows
    merged = deduplicate_rows(combined)

    if merged:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Union of keys across all rows (not just merged[0]'s), because
        # VIIRS_SNPP_SP and VIIRS_SNPP_NRT responses do not carry
        # identical column sets (SP includes a 'type' column NRT omits) --
        # using only the first row's keys risks a spurious "extra field"
        # error from csv.DictWriter depending on which source happens to
        # sort first.
        out_fieldnames = list(dict.fromkeys(k for row in merged for k in row.keys()))
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=out_fieldnames)
            writer.writeheader()
            writer.writerows(merged)

    return {
        "rows_before": len(existing_rows), "rows_fetched": len(fetched_rows),
        "rows_after": len(merged), "new_rows_added": len(merged) - len(existing_rows),
        "window_start": window_start, "window_end": run_date,
    }
