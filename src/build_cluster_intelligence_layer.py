"""
Cluster Intelligence Layer -- the single authoritative, additive 60-row
table consolidating all already-validated cluster evidence for the 60
Gujarat DBSCAN clusters.

This module does NOT recompute any existing methodology. It reads:
  - gujarat_cluster_integrated_evidence.csv (68 columns, unchanged) as the
    authoritative base -- itself already a verified, non-redundant merge
    of DBSCAN cluster geometry, OSM context, the 2023 Recurrence Profile,
    the 2019-2023 cross-year recurrence result, and the longitudinal
    feature table. A full consistency sweep across all 60 clusters and
    every shared field, run before this module was written, found ZERO
    conflicts between these source milestones.
  - gujarat_cluster_unsupervised_groups.csv, for the exploratory
    unsupervised_group label only (not pca_1/pca_2, not the ML feature
    matrix -- see DECISIONS.md).

...and adds exactly two new things:
  1. `unsupervised_group` (Section H) -- reference-only, explicitly
     exploratory.
  2. Three boolean "corroboration" fields plus their count (Section I) --
     see below.

TERMINOLOGY, STATED PRECISELY (do not describe this differently elsewhere)
----------------------------------------------------------------------
The evidence corroboration count summarizes how many distinct,
already-approved evidence dimensions support a cluster. This is
deliberately NOT called an "independent evidence" count: statistical
independence between the three dimensions has not been established, and
for two of them it is empirically false. Specifically, verified against
the current 60-cluster data: every one of the 15 clusters with
recurrence_strength in (Strong, Moderate) is ALSO 5-year persistent
(unique_years == 5) -- 15 of 15, no exceptions. recurrence_strength and
5-year persistence are therefore RELATED dimensions, not independent
ones, and a count of "2" from those two should not be read as two
separate pieces of corroborating evidence -- it is largely the same
underlying persistence signal observed at two different time scales.

The count is:
  - NOT a risk score.
  - NOT a priority ranking.
  - NOT a probability.
  - NOT a severity measure.
  - NOT a measure of how "real" a cluster is.
It is a transparent, equally-weighted (1 point each, no weighting scheme)
tally of which already-approved fields currently corroborate a cluster,
kept next to the three underlying booleans so every value in it can be
audited by eye against the source columns.

OSM corroboration (`corroboration_osm_context`) means only that some
OSM-tagged feature was found within the cluster's adaptive search radius
-- i.e. contextual information exists nearby to review. It does NOT mean
an OSM feature explains, causes, or is the source of the thermal activity.

`unsupervised_group` is excluded from the corroboration count. It is kept
as a separate, clearly-labeled reference-only column because the
unsupervised exploration milestone found: (a) weak clustering structure
(best silhouette 0.243, below what is conventionally considered a strong
result), and (b) strong sensitivity to feature choice (Adjusted Rand
Index dropped to 0.110 between the full feature set and the same
clustering with OSM features removed). Folding a result this unstable
into a corroboration tally would implicitly grant it the same evidentiary
weight as the validated Recurrence Profile / cross-year / OSM-presence
dimensions, which it has not earned. It is not a validated taxonomy.

Corroboration rule definitions (each reuses an existing, already-approved
field verbatim -- no new threshold is introduced anywhere in this module):
  corroboration_2023_recurrence = True if recurrence_strength in
      {"Strong", "Moderate"}           [existing Recurrence Profile tier]
  corroboration_5yr_persistence = True if unique_years == 5
      [existing cross-year recurrence result]
  corroboration_osm_context     = True if has_notable_osm_context is True
      [existing OSM milestone flag]
  evidence_corroboration_count  = sum of the three booleans above (0-3)
"""

import csv
from pathlib import Path

INTEGRATED_EVIDENCE_CSV = Path("data/processed/gujarat_cluster_integrated_evidence.csv")
UNSUPERVISED_GROUPS_CSV = Path("data/processed/gujarat_cluster_unsupervised_groups.csv")
OUTPUT_CSV = Path("data/processed/gujarat_cluster_intelligence_layer.csv")

CORROBORATION_RECURRENCE_TIERS = {"Strong", "Moderate"}


def load_integrated_evidence(path=INTEGRATED_EVIDENCE_CSV):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_unsupervised_groups(path=UNSUPERVISED_GROUPS_CSV):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {int(r["cluster_id"]): r["unsupervised_group"] for r in rows}


def corroboration_2023_recurrence(row):
    return row["recurrence_strength"] in CORROBORATION_RECURRENCE_TIERS


def corroboration_5yr_persistence(row):
    return int(row["unique_years"]) == 5


def corroboration_osm_context(row):
    val = row["has_notable_osm_context"]
    return val is True or val == "True"


def compute_corroboration_fields(row):
    c1 = corroboration_2023_recurrence(row)
    c2 = corroboration_5yr_persistence(row)
    c3 = corroboration_osm_context(row)
    return {
        "corroboration_2023_recurrence": c1,
        "corroboration_5yr_persistence": c2,
        "corroboration_osm_context": c3,
        "evidence_corroboration_count": int(c1) + int(c2) + int(c3),
    }


def build_intelligence_layer(evidence_rows, unsupervised_groups):
    base_columns = list(evidence_rows[0].keys())

    missing_group = set(int(r["cluster_id"]) for r in evidence_rows) - set(unsupervised_groups.keys())
    if missing_group:
        raise ValueError(f"No unsupervised_group found for cluster_id(s): {sorted(missing_group)}")

    output_rows = []
    for row in evidence_rows:
        cid = int(row["cluster_id"])
        out_row = dict(row)
        out_row["unsupervised_group"] = unsupervised_groups[cid]
        out_row.update(compute_corroboration_fields(row))
        output_rows.append(out_row)

    output_rows.sort(key=lambda r: int(r["cluster_id"]))
    fieldnames = base_columns + [
        "unsupervised_group",
        "corroboration_2023_recurrence", "corroboration_5yr_persistence",
        "corroboration_osm_context", "evidence_corroboration_count",
    ]
    return output_rows, fieldnames


def write_csv(rows, fieldnames, path=OUTPUT_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    evidence_rows = load_integrated_evidence()
    unsupervised_groups = load_unsupervised_groups()

    output_rows, fieldnames = build_intelligence_layer(evidence_rows, unsupervised_groups)

    assert len(output_rows) == 60, f"expected 60 clusters, got {len(output_rows)}"
    assert len({r["cluster_id"] for r in output_rows}) == 60, "duplicate cluster_id"

    write_csv(output_rows, fieldnames)
    print(f"Wrote {len(output_rows)} rows, {len(fieldnames)} columns to {OUTPUT_CSV}")

    from collections import Counter
    count_dist = Counter(r["evidence_corroboration_count"] for r in output_rows)
    print(f"evidence_corroboration_count distribution: {dict(sorted(count_dist.items()))}")

    return output_rows, fieldnames


if __name__ == "__main__":
    main()
