"""
Multi-year Gujarat FIRMS processing runner.

Combines the existing, unmodified 2023 Gujarat detections (derived from
data/raw/fire_archive_SV-C2_794895.csv via the existing, unchanged
spatial_recurrence.read_gujarat_detections) with any additional years
ingested via src/firms_ingestion.py / src/ingest_multi_year_gujarat.py
(data/raw/firms_gujarat_{year}.csv), into one combined multi-year dataset.

Designed to work correctly regardless of how many additional years have
actually been ingested -- including the current state, where none have
been (see PROGRESS.md / DECISIONS.md for the documented FIRMS_MAP_KEY
credential blocker). Every row keeps all existing FIRMS fields plus a
`year` column derived from acq_date.

Gujarat bounds are re-applied defensively to every year (including years
fetched already bounded via the Area API), reusing the exact same
LAT_MIN/LAT_MAX/LON_MIN/LON_MAX constants as the rest of the project --
not redefined here.
"""

import csv
import glob
import re
from pathlib import Path

from spatial_recurrence import (
    read_gujarat_detections, RAW_CSV, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
)

HISTORICAL_RAW_GLOB = "data/raw/firms_gujarat_*.csv"
OUTPUT_CSV = Path("data/processed/gujarat_multi_year_detections.csv")

FIELDS_TO_KEEP = [
    "latitude", "longitude", "brightness", "scan", "track", "acq_date",
    "acq_time", "satellite", "confidence", "bright_t31", "frp", "daynight",
    "year",
]


def _in_gujarat_bounds(row):
    try:
        lat = float(row["latitude"])
        lon = float(row["longitude"])
    except (KeyError, ValueError):
        return False
    return LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX


def _year_from_acq_date(acq_date):
    return int(acq_date[:4])


def load_2023_gujarat():
    """Reuses the existing, unchanged 2023 Gujarat loading path."""
    rows = list(read_gujarat_detections(RAW_CSV))
    for row in rows:
        row["year"] = _year_from_acq_date(row["acq_date"])
    return rows


def discover_historical_raw_files(pattern=HISTORICAL_RAW_GLOB):
    """Find any ingested data/raw/firms_gujarat_{year}.csv files."""
    files = {}
    for path_str in glob.glob(pattern):
        m = re.search(r"firms_gujarat_(\d{4})\.csv$", path_str)
        if m:
            files[int(m.group(1))] = Path(path_str)
    return files


def load_historical_year(path):
    """Load one ingested historical-year raw file, defensively re-applying
    the Gujarat bounds and tagging each row with its year."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    kept = []
    for row in rows:
        if not _in_gujarat_bounds(row):
            continue
        row["year"] = _year_from_acq_date(row["acq_date"])
        kept.append(row)
    return kept


def build_combined_dataset():
    """Combine 2023 (existing pipeline) with any ingested historical years.
    Returns (combined_rows, years_present, years_missing)."""
    combined = load_2023_gujarat()
    years_present = {2023: len(combined)}

    historical_files = discover_historical_raw_files()
    for year, path in sorted(historical_files.items()):
        rows = load_historical_year(path)
        combined.extend(rows)
        years_present[year] = len(rows)

    return combined, years_present


def write_combined_csv(rows, path=OUTPUT_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS_TO_KEEP, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    combined, years_present = build_combined_dataset()

    print(f"Combined Gujarat multi-year dataset: {len(combined)} detections")
    print("\nDetections per year:")
    for year in sorted(years_present):
        print(f"  {year}: {years_present[year]}")

    attempted_but_missing = [
        y for y in [2019, 2020, 2021, 2022] if y not in years_present
    ]
    if attempted_but_missing:
        print(f"\nYears not yet ingested (see DECISIONS.md for the "
              f"documented FIRMS_MAP_KEY blocker): {attempted_but_missing}")

    write_combined_csv(combined)
    print(f"\nWrote {OUTPUT_CSV}")

    return combined, years_present


if __name__ == "__main__":
    main()
