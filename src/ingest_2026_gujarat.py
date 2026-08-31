"""
Thin runner for 2026 Gujarat FIRMS ingestion (see
src/firms_ingestion.py::fetch_2026_gujarat for the actual fetch/boundary
logic). Mirrors ingest_multi_year_gujarat.py's role for 2019-2025: this
script's only job is to call the ingestion function and report the
result -- no parsing, no event construction, no inference here.

Does NOT touch data/raw/firms_gujarat_{2019..2025}.csv or
data/raw/fire_archive_SV-C2_794895.csv.
"""

from firms_ingestion import (
    fetch_2026_gujarat, FirmsConfigError, FirmsIngestionError,
    SP_END_DATE_2026, NRT_START_DATE_2026,
)


def main():
    print(f"Ingesting 2026 Gujarat detections: VIIRS_SNPP_SP through {SP_END_DATE_2026}, "
          f"VIIRS_SNPP_NRT from {NRT_START_DATE_2026} onward.\n")
    try:
        result = fetch_2026_gujarat()
    except FirmsConfigError as e:
        print("BLOCKED (missing configuration)")
        print(f"  {e}")
        return {"status": "config_error", "row_count": 0}
    except FirmsIngestionError as e:
        print("FAILED")
        print(f"  {e}")
        return {"status": "failed", "row_count": 0}

    print(f"Status: {result['status']}")
    print(f"Total rows: {result['row_count']}")
    print(f"  SP rows ({result['sp_date_range'][0]}..{result['sp_date_range'][1]}): {result['sp_row_count']}")
    print(f"  NRT rows ({result['nrt_date_range'][0]}..{result['nrt_date_range'][1]}): {result['nrt_row_count']}")
    if result["path"]:
        print(f"Wrote {result['path']}")
    return result


if __name__ == "__main__":
    main()
