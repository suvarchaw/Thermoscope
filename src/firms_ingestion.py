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
    chunks = generate_date_chunks(start, end)

    all_rows = []
    fieldnames = None
    for chunk_start, day_range in chunks:
        url = build_area_url(map_key, source, GUJARAT_BBOX, day_range, chunk_start)
        csv_text = fetch_csv_chunk(url)
        rows, chunk_fieldnames = parse_csv_text(csv_text)
        if rows:
            fieldnames = fieldnames or chunk_fieldnames
            all_rows.extend(rows)
        time.sleep(request_delay_s)

    if not all_rows:
        return {
            "status": "no_data", "year": year, "path": None,
            "row_count": 0, "error": None,
        }

    validate_columns(fieldnames)
    normalized = normalize_columns(all_rows)
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
