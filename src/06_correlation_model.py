"""
Module 6: Default Correlation Modelling
=========================================
Models correlation between defaults using:
  a) Factor-based approach (single-factor Vasicek model)
  b) K-Means clustering for borrower segmentation

Produces borrower segments with correlated risk profiles.

Run: python src/06_correlation_model.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from utils import CLEANED_DATA_PATH, FIGURES_DIR

# ============================================================
# CONFIGURATION
# ============================================================
MAX_CLUSTERS = 8
RANDOM_STATE = 42
CLUSTER_FEATURES = [
    "bureau_score", "ltv", "monthly_income", "age", "interest_rate",
    "sanctioned_loan_amount", "emi_to_income_ratio", "loan_to_income_ratio",
    "peak_dpd", "no_of_installments"
]


def load_data():
    """Load the most recent dataset with PD/LGD."""
    for suffix in ["_with_el.csv", "_with_lgd.csv", "_with_pd.csv", ".csv"]:
        path = CLEANED_DATA_PATH.replace(".csv", suffix)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded: {path} ({len(df)} loans)")
            return df
    raise FileNotFoundError("No data found.")


# ============================================================
# PART A: VASICEK SINGLE-FACTOR MODEL
# ============================================================
def estimate_asset_correlation(df, segments):
    """
    Estimate pairwise asset correlation using the Vasicek single-factor model.

    The model assumes:
    - PD = Phi( (Phi^-1(PD_bar) + sqrt(rho) * Z) / sqrt(1-rho) )
    - Where Z is the systematic factor and rho is the asset correlation

    We estimate rho by matching observed default rate variance across segments.
    """
    print("\n" + "=" * 50)
    print("ASSET CORRELATION ESTIMATION (Vasicek Model)")
    print("=" * 50)

    results = {}

    for seg_name, seg_col in segments.items():
        print(f"\n--- Segmentation by: {seg_name} ---")

        groups = df.groupby(seg_col).agg(
            count=("is_stressed", "count"),
            default_rate=("is_stressed", "mean"),
            mean_pd=("pd_final", "mean") if "pd_final" in df.columns else ("is_stressed", "mean")
        ).reset_index()

        # Filter out tiny groups
        groups = groups[groups["count"] >= 50]

        if len(groups) < 2:
            print(f"  Not enough segments (need >=2, got {len(groups)}). Skipping.")
            continue

        # Portfolio average PD
        pd_bar = df["is_stressed"].mean()

        # Observed variance of default rates across segments
        obs_variance = groups["default_rate"].var()

        # Estimate rho by method of moments
        # Under Vasicek: Var(DR) ≈ rho * pd_bar * (1 - pd_bar) for small rho
        if pd_bar > 0 and (1 - pd_bar) > 0:
            rho_estimate = obs_variance / (pd_bar * (1 - pd_bar))
            rho_estimate = np.clip(rho_estimate, 0.001, 0.999)
        else:
            rho_estimate = 0.15  # default

        print(f"  Portfolio PD:       {pd_bar:.4f}")
        print(f"  DR variance:        {obs_variance:.6f}")
        print(f"  Estimated ρ:        {rho_estimate:.4f}")

        print(f"\n  Segment breakdown:")
        for _, row in groups.iterrows():
            print(f"    {str(row[seg_col]):25s}  DR={row['default_rate']:.4f}  n={int(row['count']):,}")

        results[seg_name] = {
            "rho": rho_estimate,
            "pd_bar": pd_bar,
            "segments": groups,
            "dr_variance": obs_variance
        }

    return results


def analyze_intra_segment_correlation(df, cluster_col="risk_cluster"):
    """Compute default correlation within and between segments."""
    print("\n" + "=" * 50)
    print("INTRA vs INTER-SEGMENT CORRELATION")
    print("=" * 50)

    if cluster_col not in df.columns:
        print("  No cluster column found. Skipping.")
        return

    clusters = df[cluster_col].unique()
    target = "is_stressed"

    print(f"\n  {'Segment':<15} {'DR':<10} {'Count':<10} {'Intra-ρ (est.)':<15}")
    print("  " + "-" * 50)

    for c in sorted(clusters):
        seg = df[df[cluster_col] == c]
        dr = seg[target].mean()
        n = len(seg)

        # Estimate intra-segment correlation
        # Higher DR variance within segment → lower correlation (more idiosyncratic)
        intra_rho = dr * (1 - dr) * 0.5 if dr > 0 else 0  # simplified estimate

        print(f"  Cluster {c:<8} {dr:<10.4f} {n:<10,} {intra_rho:<15.4f}")


# ============================================================
# PART B: K-MEANS BORROWER SEGMENTATION
# ============================================================
def find_optimal_clusters(X_scaled, max_k=MAX_CLUSTERS):
    """Use silhouette score to find optimal number of clusters."""
    print("\n  Finding optimal k...")
    scores = {}
    inertias = {}

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores[k] = score
        inertias[k] = kmeans.inertia_
        print(f"    k={k}: Silhouette={score:.4f}, Inertia={kmeans.inertia_:,.0f}")

    optimal_k = max(scores, key=scores.get)
    print(f"\n  → Optimal k = {optimal_k} (Silhouette = {scores[optimal_k]:.4f})")

    return optimal_k, scores, inertias


def run_kmeans_segmentation(df):
    """Perform K-Means clustering on borrower features."""
    print("\n" + "=" * 50)
    print("K-MEANS BORROWER SEGMENTATION")
    print("=" * 50)

    # Select available features
    available = [f for f in CLUSTER_FEATURES if f in df.columns]
    print(f"  Clustering features: {available}")

    X = df[available].copy()

    # Fill NaN with median
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal k
    optimal_k, sil_scores, inertias = find_optimal_clusters(X_scaled)

    # Final clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
    df["risk_cluster"] = kmeans.fit_predict(X_scaled)

    # Analyze each cluster
    print(f"\n  Cluster Profiles:")
    print(f"  {'Cluster':<10} {'Count':<8} {'DR':<8} {'Avg PD':<10} "
          f"{'Avg Income':<12} {'Avg LTV':<10} {'Avg Bureau':<12}")
    print("  " + "-" * 70)

    for c in sorted(df["risk_cluster"].unique()):
        seg = df[df["risk_cluster"] == c]
        pd_col = "pd_final" if "pd_final" in df.columns else "is_stressed"
        print(f"  {c:<10} {len(seg):<8,} {seg['is_stressed'].mean():<8.4f} "
              f"{seg[pd_col].mean():<10.4f} "
              f"₹{seg['monthly_income'].mean():<10,.0f} "
              f"{seg['ltv'].mean():<10.3f} "
              f"{seg['bureau_score'].mean():<12.1f}")

    return df, kmeans, scaler, optimal_k, sil_scores, inertias


# ============================================================
# VISUALIZATION
# ============================================================
def plot_segmentation_results(df, sil_scores, inertias, optimal_k):
    """Generate segmentation visualizations."""
    print("\n[PLOTS] Generating correlation/segmentation visualizations...")

    # 1. Silhouette & Elbow plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ks = sorted(sil_scores.keys())
    sils = [sil_scores[k] for k in ks]
    ines = [inertias[k] for k in ks]

    ax1.plot(ks, sils, "o-", color="#e63946", linewidth=2, markersize=8)
    ax1.axvline(optimal_k, color="#2a9d8f", linestyle="--", label=f"Optimal k={optimal_k}")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Silhouette Score")
    ax1.set_title("Silhouette Analysis")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ks, ines, "o-", color="#457b9d", linewidth=2, markersize=8)
    ax2.axvline(optimal_k, color="#2a9d8f", linestyle="--", label=f"Optimal k={optimal_k}")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Inertia (Within-Cluster SS)")
    ax2.set_title("Elbow Method")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Cluster Selection Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "cluster_selection.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [PLOT] Saved: cluster_selection.png")

    # 2. Cluster risk profiles
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Bureau score by cluster
    ax = axes[0, 0]
    for c in sorted(df["risk_cluster"].unique()):
        data = df[df["risk_cluster"] == c]["bureau_score"].dropna()
        ax.hist(data, bins=30, alpha=0.5, label=f"Cluster {c} (n={len(data):,})")
    ax.set_title("Bureau Score by Cluster")
    ax.set_xlabel("Bureau Score")
    ax.legend(fontsize=8)

    # Income by cluster
    ax = axes[0, 1]
    cluster_means = df.groupby("risk_cluster")["monthly_income"].mean().sort_index()
    ax.bar(range(len(cluster_means)), cluster_means.values, color=sns.color_palette("mako", len(cluster_means)))
    ax.set_xticks(range(len(cluster_means)))
    ax.set_xticklabels([f"C{c}" for c in cluster_means.index])
    ax.set_title("Avg Monthly Income by Cluster")
    ax.set_ylabel("Income (₹)")

    # Default rate by cluster
    ax = axes[1, 0]
    cluster_dr = df.groupby("risk_cluster").agg(
        stress_rate=("is_stressed", "mean"),
        default_rate=("is_default", "mean")
    ).sort_index()
    x = range(len(cluster_dr))
    width = 0.35
    ax.bar([i - width/2 for i in x], cluster_dr["stress_rate"] * 100, width,
           label="Stress Rate (DPD≥30)", color="#f4a261")
    ax.bar([i + width/2 for i in x], cluster_dr["default_rate"] * 100, width,
           label="Default Rate (DPD≥90)", color="#e63946")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in cluster_dr.index])
    ax.set_title("Default/Stress Rate by Cluster")
    ax.set_ylabel("Rate (%)")
    ax.legend()

    # LTV by cluster
    ax = axes[1, 1]
    for c in sorted(df["risk_cluster"].unique()):
        data = df[df["risk_cluster"] == c]["ltv"].dropna()
        ax.hist(data, bins=30, alpha=0.5, label=f"Cluster {c}")
    ax.set_title("LTV Distribution by Cluster")
    ax.set_xlabel("LTV")
    ax.legend(fontsize=8)

    plt.suptitle(f"Risk Cluster Profiles (k={optimal_k})", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "cluster_profiles.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [PLOT] Saved: cluster_profiles.png")

    # 3. Scatter: PD vs LGD colored by cluster
    if "pd_final" in df.columns and "lgd_base_case" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(
            df["pd_final"], df["lgd_base_case"],
            c=df["risk_cluster"], cmap="Set2", alpha=0.3, s=10
        )
        ax.set_xlabel("Probability of Default (PD)")
        ax.set_ylabel("Loss Given Default (LGD)")
        ax.set_title("PD vs LGD — Colored by Risk Cluster", fontsize=14, fontweight="bold")
        plt.colorbar(scatter, label="Cluster")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "pd_vs_lgd_clusters.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("  [PLOT] Saved: pd_vs_lgd_clusters.png")


def main():
    print("=" * 60)
    print("CREDIT RISK MODELLING — DEFAULT CORRELATION & SEGMENTATION")
    print("=" * 60)

    df = load_data()

    # Part A: Asset Correlation Estimation
    segments = {}
    for col in ["state_group", "manufacturer", "income_band", "bureau_score_band"]:
        if col in df.columns:
            segments[col] = col

    if segments:
        corr_results = estimate_asset_correlation(df, segments)
    else:
        print("  No suitable segmentation columns found for correlation estimation.")
        corr_results = {}

    # Part B: K-Means Segmentation
    df, kmeans, scaler, optimal_k, sil_scores, inertias = run_kmeans_segmentation(df)

    # Part C: Intra vs Inter-segment correlation
    analyze_intra_segment_correlation(df)

    # Visualizations
    plot_segmentation_results(df, sil_scores, inertias, optimal_k)

    # Save
    output_path = CLEANED_DATA_PATH.replace(".csv", "_with_clusters.csv")
    df.to_csv(output_path, index=False)
    print(f"\nData with clusters saved to: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("CORRELATION & SEGMENTATION COMPLETE")
    print(f"  Optimal clusters: {optimal_k}")
    if corr_results:
        for seg, res in corr_results.items():
            print(f"  ρ ({seg}): {res['rho']:.4f}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
