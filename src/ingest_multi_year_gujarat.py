"""
Orchestrates fetching multiple years of Gujarat-bounded VIIRS S-NPP FIRMS
data via src/firms_ingestion.py.

2023 is intentionally NOT re-fetched here: it already exists as the
project's original, unmodified raw archive
(data/raw/fire_archive_SV-C2_794895.csv) and is reused as-is by
src/multi_year_gujarat_processing.py, per "reuse existing data wherever
possible" and to avoid any risk of a re-fetched 2023 subtly differing from
the file every prior milestone was built and validated against.

This script targets the remaining years needed for a 2019-2023 span:
2019, 2020, 2021, 2022.

Requires the FIRMS_MAP_KEY environment variable (see
src/firms_ingestion.py for setup instructions). If it is not set, this
script reports that clearly per year and exits without fabricating data or
crashing uninformatively -- ingestion can be completed later simply by
setting the key and re-running this script (already-ingested years are
skipped automatically).
"""

from firms_ingestion import fetch_year_gujarat, FirmsConfigError, FirmsIngestionError

TARGET_YEARS = [2019, 2020, 2021, 2022]


def main():
    results = []
    for year in TARGET_YEARS:
        print(f"Year {year}: ", end="")
        try:
            result = fetch_year_gujarat(year)
        except FirmsConfigError as e:
            print("BLOCKED (missing configuration)")
            print(f"  {e}")
            results.append({"year": year, "status": "config_error", "row_count": 0})
            continue
        except FirmsIngestionError as e:
            print("FAILED")
            print(f"  {e}")
            results.append({"year": year, "status": "failed", "row_count": 0})
            continue

        print(f"{result['status']} ({result['row_count']} rows)")
        results.append(result)

    print("\nSummary:")
    for r in results:
        print(f"  {r['year']}: {r['status']}, rows={r.get('row_count', 0)}")

    return results


if __name__ == "__main__":
    main()
