"""
Exploratory unsupervised clustering of the 60 Gujarat DBSCAN clusters,
using the ML feature matrix built in src/ml_feature_inventory.py.

Purpose: investigate whether the 60 clusters separate into behavioral/
contextual groups that are not already fully captured by the existing
hand-designed categories (recurrence_strength, persistence_category,
activity_category, trend_direction, seasonality_category). Those
categories are used ONLY for post-hoc comparison here -- never as
training input (see ml_feature_inventory.py for that exclusion).

This is exploratory unsupervised ML only:
  - no supervised model is trained,
  - no risk score is computed,
  - no ground-truth label is invented,
  - discovered groups are named neutrally ("Group A/B/C...") and never
    given risk- or source-type-sounding names,
  - no causal interpretation is made about what a discovered group "is".

Pipeline
--------
1. Standardize features (StandardScaler -- zero mean, unit variance per
   feature, required since features span very different natural scales,
   e.g. centroid_lat ~20-25 vs total_detections_5yr ~10-10,000).
2. Evaluate KMeans for k=2..8 using silhouette score (higher is better
   separation) and inertia (elbow heuristic), on three preprocessing
   variants to test sensitivity to feature-set choices:
     V1 (full)        -- all 22 included features
     V2 (no-spatial)  -- drops centroid_lat/lon/extent_radius_m, testing
                          whether geographic location is driving groups
     V3 (no-OSM)      -- drops the 8 OSM-derived features, testing
                          whether OSM context is driving groups
3. Choose a final k for V1 by silhouette score, with the choice and
   reasoning stated explicitly (not just "argmax silhouette" blindly --
   ties/near-ties are resolved toward interpretability, documented).
4. Fit final KMeans on V1; cross-check stability by also fitting
   AgglomerativeClustering (Ward linkage) at the same k on V1, and
   compare via Adjusted Rand Index (ARI). Cross-check sensitivity by
   fitting KMeans at the same k on V2 and V3 and comparing each to V1 via
   ARI.
5. PCA to 2 components on V1 (standardized) for visualization.
6. Compare final groups against existing hand-designed categories via
   contingency tables and ARI/NMI (read back from the source file, never
   used as training input).
7. Build a descriptive profile per discovered group.

All randomized steps (KMeans init, PCA) use a fixed random_state so
results are reproducible across runs.
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

FEATURES_CSV = Path("data/processed/gujarat_cluster_ml_features.csv")
INTEGRATED_EVIDENCE_CSV = Path("data/processed/gujarat_cluster_integrated_evidence.csv")

GROUPS_CSV = Path("data/processed/gujarat_cluster_unsupervised_groups.csv")
PROFILES_CSV = Path("data/processed/gujarat_cluster_group_profiles.csv")
FIGURE_K_SELECTION = Path("results/figures/unsupervised_k_selection.png")
FIGURE_PCA = Path("results/figures/unsupervised_pca_groups.png")
FIGURE_AGREEMENT = Path("results/figures/unsupervised_vs_existing_categories.png")

RANDOM_STATE = 42
K_RANGE = range(2, 9)

SPATIAL_FEATURES = ["centroid_lat", "centroid_lon", "extent_radius_m"]
OSM_FEATURES = ["n_industrial", "n_power", "n_waste", "n_agricultural", "n_transport",
                 "n_other", "features_found_in_radius", "nearest_distance_m", "nearest_is_named"]

EXISTING_CATEGORY_COLUMNS = [
    "recurrence_strength", "persistence_category", "activity_category",
    "trend_direction", "seasonality_category",
]


def load_feature_matrix(path=FEATURES_CSV):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cluster_ids = [int(r["cluster_id"]) for r in rows]
    feature_names = [c for c in rows[0].keys() if c != "cluster_id"]
    X = np.array([[float(r[c]) for c in feature_names] for r in rows])
    return cluster_ids, feature_names, X


def variant_column_indices(feature_names, exclude):
    return [i for i, name in enumerate(feature_names) if name not in exclude]


def evaluate_k_range(X_scaled, k_range=K_RANGE, random_state=RANDOM_STATE):
    """Returns {k: {"silhouette": ..., "inertia": ...}}"""
    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        results[k] = {"silhouette": sil, "inertia": km.inertia_}
    return results


def choose_k(k_results, min_k=2):
    """Choose k by highest silhouette score. Ties (within 0.01) broken
    toward the smaller, more interpretable k. Documented, not a silent
    argmax."""
    best_sil = max(v["silhouette"] for v in k_results.values())
    candidates = [k for k, v in k_results.items() if v["silhouette"] >= best_sil - 0.01]
    return min(candidates)


def fit_kmeans(X_scaled, k, random_state=RANDOM_STATE):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels, km


def fit_agglomerative(X_scaled, k):
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    return agg.fit_predict(X_scaled)


def relabel_by_size_desc(labels, cluster_ids=None, tiebreak_key=None):
    """Map raw integer cluster labels to neutral, deterministic names
    (Group A, Group B, ...), ordered by group size descending. Ties in
    size broken by the mean of tiebreak_key (ascending), for a fully
    deterministic ordering regardless of KMeans' internal label numbering."""
    counts = Counter(labels)
    if tiebreak_key is not None:
        key_by_label = defaultdict(list)
        for lbl, val in zip(labels, tiebreak_key):
            key_by_label[lbl].append(val)
        order = sorted(counts.keys(), key=lambda lbl: (-counts[lbl], np.mean(key_by_label[lbl])))
    else:
        order = sorted(counts.keys(), key=lambda lbl: -counts[lbl])
    name_map = {raw: f"Group {chr(65 + i)}" for i, raw in enumerate(order)}
    return [name_map[lbl] for lbl in labels]


def load_existing_categories(path=INTEGRATED_EVIDENCE_CSV):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {int(r["cluster_id"]): {c: r[c] for c in EXISTING_CATEGORY_COLUMNS} for r in rows}


def compare_to_categories(group_labels, cluster_ids, categories_by_cluster):
    """Returns {category_column: {"ari": ..., "nmi": ..., "contingency": {(group, cat_value): count}}}"""
    results = {}
    for col in EXISTING_CATEGORY_COLUMNS:
        cat_values = [categories_by_cluster[cid][col] for cid in cluster_ids]
        ari = adjusted_rand_score(cat_values, group_labels)
        nmi = normalized_mutual_info_score(cat_values, group_labels)
        contingency = Counter(zip(group_labels, cat_values))
        results[col] = {"ari": ari, "nmi": nmi, "contingency": contingency}
    return results


def build_group_profiles(group_labels, cluster_ids, feature_names, X, categories_by_cluster):
    profiles = []
    unique_groups = sorted(set(group_labels))
    for group in unique_groups:
        idx = [i for i, g in enumerate(group_labels) if g == group]
        profile = {"group": group, "n_clusters": len(idx),
                   "cluster_ids": ";".join(str(cluster_ids[i]) for i in idx)}
        for j, fname in enumerate(feature_names):
            profile[f"mean_{fname}"] = float(np.mean(X[idx, j]))
        for col in EXISTING_CATEGORY_COLUMNS:
            values = [categories_by_cluster[cluster_ids[i]][col] for i in idx]
            most_common = Counter(values).most_common()
            profile[f"breakdown_{col}"] = "; ".join(f"{v}:{c}" for v, c in most_common)
        profiles.append(profile)
    profiles.sort(key=lambda p: -p["n_clusters"])
    return profiles


def write_groups_csv(cluster_ids, group_labels, pca_coords, path=GROUPS_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "unsupervised_group", "pca_1", "pca_2"])
        for cid, grp, coords in zip(cluster_ids, group_labels, pca_coords):
            writer.writerow([cid, grp, coords[0], coords[1]])


def write_profiles_csv(profiles, path=PROFILES_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(profiles[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(profiles)


def main():
    cluster_ids, feature_names, X = load_feature_matrix()
    print(f"Loaded feature matrix: {X.shape[0]} clusters x {X.shape[1]} features.")
    print(f"Features: {feature_names}\n")

    scaler = StandardScaler()
    X_v1 = scaler.fit_transform(X)

    v2_idx = variant_column_indices(feature_names, exclude=SPATIAL_FEATURES)
    v3_idx = variant_column_indices(feature_names, exclude=OSM_FEATURES)
    X_v2 = StandardScaler().fit_transform(X[:, v2_idx])
    X_v3 = StandardScaler().fit_transform(X[:, v3_idx])

    print("=" * 70)
    print("K-SELECTION (silhouette score, all 3 preprocessing variants)")
    print("=" * 70)
    k_results_v1 = evaluate_k_range(X_v1)
    k_results_v2 = evaluate_k_range(X_v2)
    k_results_v3 = evaluate_k_range(X_v3)
    print("  V1 (full):")
    for k, v in k_results_v1.items():
        print(f"    k={k}: silhouette={v['silhouette']:.3f}, inertia={v['inertia']:.1f}")
    print("  V2 (no-spatial):")
    for k, v in k_results_v2.items():
        print(f"    k={k}: silhouette={v['silhouette']:.3f}")
    print("  V3 (no-OSM):")
    for k, v in k_results_v3.items():
        print(f"    k={k}: silhouette={v['silhouette']:.3f}")

    final_k = choose_k(k_results_v1)
    k_v2_own_choice = choose_k(k_results_v2)
    k_v3_own_choice = choose_k(k_results_v3)
    print(f"\n  Chosen k (V1, used as the final solution) = {final_k} "
          f"(highest silhouette, ties within 0.01 broken toward smaller k)")
    print(f"  For reference, V2's own best k = {k_v2_own_choice}, "
          f"V3's own best k = {k_v3_own_choice} (each variant's independent silhouette-best choice)")

    print("\n" + "=" * 70)
    print("FINAL CLUSTERING (V1, KMeans, k=%d)" % final_k)
    print("=" * 70)
    labels_v1, km_v1 = fit_kmeans(X_v1, final_k)
    sil_final = silhouette_score(X_v1, labels_v1)
    print(f"  Final silhouette score: {sil_final:.3f}")

    print("\n" + "=" * 70)
    print("STABILITY CHECK: KMeans vs. Agglomerative (Ward), same k, same features (V1)")
    print("=" * 70)
    labels_agg = fit_agglomerative(X_v1, final_k)
    ari_algo = adjusted_rand_score(labels_v1, labels_agg)
    print(f"  Adjusted Rand Index (KMeans vs Agglomerative): {ari_algo:.3f}")

    print("\n" + "=" * 70)
    print("SENSITIVITY CHECK: KMeans on V1 vs. V2 (no-spatial) vs. V3 (no-OSM), same k")
    print("=" * 70)
    labels_v2, _ = fit_kmeans(X_v2, final_k)
    labels_v3, _ = fit_kmeans(X_v3, final_k)
    ari_v1_v2 = adjusted_rand_score(labels_v1, labels_v2)
    ari_v1_v3 = adjusted_rand_score(labels_v1, labels_v3)
    print(f"  ARI (V1 full vs V2 no-spatial): {ari_v1_v2:.3f}")
    print(f"  ARI (V1 full vs V3 no-OSM):     {ari_v1_v3:.3f}")

    group_names = relabel_by_size_desc(labels_v1, tiebreak_key=cluster_ids)
    group_sizes = Counter(group_names)
    print(f"\n  Final group sizes: {dict(sorted(group_sizes.items()))}")

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_coords = pca.fit_transform(X_v1)
    print(f"\n  PCA explained variance ratio (2 components): "
          f"{pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f} "
          f"(total {sum(pca.explained_variance_ratio_):.3f})")
    top_loadings = np.argsort(-np.abs(pca.components_[0]))[:5]
    print(f"  Top PC1 loadings: {[(feature_names[i], round(pca.components_[0][i], 2)) for i in top_loadings]}")

    write_groups_csv(cluster_ids, group_names, pca_coords)
    print(f"\nWrote {GROUPS_CSV}")

    categories_by_cluster = load_existing_categories()

    print("\n" + "=" * 70)
    print("COMPARISON AGAINST EXISTING HAND-DESIGNED CATEGORIES")
    print("=" * 70)
    comparison = compare_to_categories(group_names, cluster_ids, categories_by_cluster)
    for col, res in comparison.items():
        print(f"  {col}: ARI={res['ari']:.3f}, NMI={res['nmi']:.3f}")

    profiles = build_group_profiles(group_names, cluster_ids, feature_names, X, categories_by_cluster)
    write_profiles_csv(profiles)
    print(f"\nWrote {PROFILES_CSV}")

    plot_k_selection(k_results_v1, final_k, k_results_v2=k_results_v2, k_results_v3=k_results_v3)
    print(f"\nSaved figure to {FIGURE_K_SELECTION}")
    plot_pca_groups(pca_coords, group_names, cluster_ids, categories_by_cluster)
    print(f"Saved figure to {FIGURE_PCA}")
    plot_agreement(comparison)
    print(f"Saved figure to {FIGURE_AGREEMENT}")

    return {
        "cluster_ids": cluster_ids, "feature_names": feature_names,
        "final_k": final_k, "group_names": group_names,
        "ari_algo": ari_algo, "ari_v1_v2": ari_v1_v2, "ari_v1_v3": ari_v1_v3,
        "comparison": comparison, "profiles": profiles,
        "silhouette": sil_final,
    }


def plot_k_selection(k_results, chosen_k, k_results_v2=None, k_results_v3=None,
                      output_path=FIGURE_K_SELECTION):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = sorted(k_results.keys())
    sils = [k_results[k]["silhouette"] for k in ks]
    inertias = [k_results[k]["inertia"] for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(ks, sils, marker="o", color="#2b6cb0", label="V1 (full)")
    if k_results_v2:
        axes[0].plot(ks, [k_results_v2[k]["silhouette"] for k in ks], marker="s",
                     color="#38a169", alpha=0.7, label="V2 (no-spatial)")
    if k_results_v3:
        axes[0].plot(ks, [k_results_v3[k]["silhouette"] for k in ks], marker="^",
                     color="#d69e2e", alpha=0.7, label="V3 (no-OSM)")
    axes[0].axvline(chosen_k, color="#c53030", linestyle="--", label=f"chosen k={chosen_k} (V1)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Silhouette score")
    axes[0].set_title("Silhouette score vs. k, all 3 preprocessing variants")
    axes[0].legend(fontsize=8)

    axes[1].plot(ks, inertias, marker="o", color="#805ad5")
    axes[1].axvline(chosen_k, color="#c53030", linestyle="--", label=f"chosen k={chosen_k}")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Inertia (within-cluster SSE)")
    axes[1].set_title("Elbow plot")
    axes[1].legend()

    fig.suptitle("Exploratory Unsupervised Clustering: k Selection (descriptive only)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pca_groups(pca_coords, group_names, cluster_ids, categories_by_cluster,
                     output_path=FIGURE_PCA):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    unique_groups = sorted(set(group_names))
    palette = ["#c53030", "#2b6cb0", "#38a169", "#d69e2e", "#805ad5", "#dd6b20", "#319795", "#b83280"]
    group_colors = {g: palette[i % len(palette)] for i, g in enumerate(unique_groups)}

    ax = axes[0]
    for g in unique_groups:
        idx = [i for i, gg in enumerate(group_names) if gg == g]
        ax.scatter(pca_coords[idx, 0], pca_coords[idx, 1], color=group_colors[g],
                   label=g, edgecolor="black", s=70, alpha=0.85)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Discovered unsupervised groups")
    ax.legend(fontsize=8)

    ax = axes[1]
    strength_values = [categories_by_cluster[cid]["recurrence_strength"] for cid in cluster_ids]
    unique_strengths = ["Strong", "Moderate", "Limited"]
    strength_colors = {"Strong": "#c53030", "Moderate": "#d69e2e", "Limited": "#2b6cb0"}
    for s in unique_strengths:
        idx = [i for i, sv in enumerate(strength_values) if sv == s]
        ax.scatter(pca_coords[idx, 0], pca_coords[idx, 1], color=strength_colors[s],
                   label=s, edgecolor="black", s=70, alpha=0.85)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Existing recurrence_strength (for comparison only)")
    ax.legend(fontsize=8)

    fig.suptitle("PCA Projection: Unsupervised Groups vs. Existing Category (descriptive only, not causal)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_agreement(comparison, output_path=FIGURE_AGREEMENT):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = list(comparison.keys())
    aris = [comparison[c]["ari"] for c in cols]
    nmis = [comparison[c]["nmi"] for c in cols]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(cols))
    ax.bar(x - 0.2, aris, width=0.4, label="Adjusted Rand Index", color="#2b6cb0")
    ax.bar(x + 0.2, nmis, width=0.4, label="Normalized Mutual Info", color="#805ad5")
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=20, ha="right")
    ax.set_ylabel("Agreement score (0 = no agreement, 1 = identical)")
    ax.set_title("Unsupervised Groups vs. Existing Hand-Designed Categories\n"
                  "(comparison only -- categories were never used as training input)")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
