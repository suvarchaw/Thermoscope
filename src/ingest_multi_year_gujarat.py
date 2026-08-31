"""
Orchestrates fetching multiple years of Gujarat-bounded VIIRS S-NPP FIRMS
data via src/firms_ingestion.py.

2023 is intentionally NOT re-fetched here: it already exists as the
project's original, unmodified raw archive
(data/raw/fire_archive_SV-C2_794895.csv) and is reused as-is by
src/multi_year_gujarat_processing.py, per "reuse existing data wherever
possible" and to avoid any risk of a re-fetched 2023 subtly differing from
the file every prior milestone was built and validated against.

This script targets 2019-2022 (needed for the original 2019-2023 span) plus
2024 and 2025, added once confirmed fully available via VIIRS_SNPP_SP
(read-only investigation, see PROGRESS.md/DECISIONS.md). 2026 is
deliberately NOT included here: as of the investigation date it is only a
partial year (VIIRS_SNPP_SP through 2026-04-27, VIIRS_SNPP_NRT from
2026-04-28 onward) and this script's underlying fetch_year_gujarat always
fetches a complete Jan 1 - Dec 31 calendar year, which would be incorrect
for 2026 without further changes not made in this milestone.

Requires the FIRMS_MAP_KEY environment variable (see
src/firms_ingestion.py for setup instructions). If it is not set, this
script reports that clearly per year and exits without fabricating data or
crashing uninformatively -- ingestion can be completed later simply by
setting the key and re-running this script (already-ingested years are
skipped automatically).
"""

from firms_ingestion import fetch_year_gujarat, FirmsConfigError, FirmsIngestionError

TARGET_YEARS = [2019, 2020, 2021, 2022, 2024, 2025]


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
