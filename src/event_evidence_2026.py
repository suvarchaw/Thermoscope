"""
Additive evidence join for 2026 events (parallel to event_evidence.py's
historical join). Reuses event_evidence.build_evidence_table UNCHANGED --
no new evidence source, threshold, or join rule -- pointed at the 2026
events table instead of the historical one. Writes a separate, additive
output; never touches gujarat_event_evidence.csv.
"""

from pathlib import Path

import event_evidence as ee
from event_construction_2026 import OUTPUT_EVENTS_CSV as EVENTS_2026_CSV

OUTPUT_CSV = Path("data/processed/gujarat_event_evidence_2026.csv")


def main():
    print("Ensuring external evidence sources are cached (fetch-if-missing, unchanged)...")
    ee.fetch_osm_industrial_power()
    ee.fetch_osm_flare()
    ee.fetch_osm_kiln()
    ee.fetch_gppd_thermal_plants()

    events = ee.load_events(EVENTS_2026_CSV)
    print(f"Loaded {len(events)} 2026 events from {EVENTS_2026_CSV} (read-only).")

    rows = ee.build_evidence_table(events)
    ee.write_csv(rows, path=OUTPUT_CSV)
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")

    n_industrial = sum(1 for r in rows if r["nearest_osm_industrial_power_m"] not in (None, ""))
    n_gppd = sum(1 for r in rows if r["nearest_gppd_thermal_plant_m"] not in (None, ""))
    n_flare = sum(1 for r in rows if r["nearest_osm_flare_m"] not in (None, ""))
    n_kiln = sum(1 for r in rows if r["nearest_osm_kiln_m"] not in (None, ""))
    n_overlap = sum(1 for r in rows if r["overlaps_cluster_id"] != "")
    n_landcover = sum(1 for r in rows if r["land_cover_class"] != "")
    print(f"Events with industrial/power evidence within {ee.DEFAULT_MAX_SEARCH_M/1000:.0f}km: {n_industrial}")
    print(f"Events with GPPD thermal-plant evidence within {ee.DEFAULT_MAX_SEARCH_M/1000:.0f}km: {n_gppd}")
    print(f"Events with flare evidence within {ee.DEFAULT_MAX_SEARCH_M/1000:.0f}km: {n_flare}")
    print(f"Events with kiln evidence within {ee.DEFAULT_MAX_SEARCH_M/1000:.0f}km: {n_kiln}")
    print(f"Events overlapping a persistent cluster: {n_overlap}")
    print(f"Events with a sampled land-cover class: {n_landcover}")
    return rows


if __name__ == "__main__":
    main()
