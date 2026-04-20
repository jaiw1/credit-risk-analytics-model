"""
Module 2: Exploratory Data Analysis (EDA)
==========================================
Generates comprehensive visualizations of the cleaned loan dataset.
All plots are saved to outputs/figures/.

Run: python src/02_eda.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from utils import CLEANED_DATA_PATH, FIGURES_DIR

# Plot style
plt.rcParams.update({
    "figure.figsize": (12, 7),
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})
sns.set_style("whitegrid")
PALETTE = sns.color_palette("mako", 10)


def save_fig(name):
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    plt.savefig(path)
    plt.close()
    print(f"  [PLOT] Saved: {name}.png")


def plot_default_rate_by_category(df, col, title, filename, top_n=None):
    """Bar chart of default rate across a categorical variable."""
    if top_n:
        top_vals = df[col].value_counts().head(top_n).index
        data = df[df[col].isin(top_vals)]
    else:
        data = df

    rates = data.groupby(col).agg(
        default_rate=("is_default", "mean"),
        stress_rate=("is_stressed", "mean"),
        count=("is_default", "count")
    ).sort_values("default_rate", ascending=False)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = range(len(rates))
    width = 0.35

    bars1 = ax1.bar([i - width/2 for i in x], rates["default_rate"] * 100,
                     width, label="Default Rate (DPD≥90)", color="#e63946", alpha=0.85)
    bars2 = ax1.bar([i + width/2 for i in x], rates["stress_rate"] * 100,
                     width, label="Stress Rate (DPD≥30)", color="#f4a261", alpha=0.85)

    ax1.set_xlabel(col.replace("_", " ").title())
    ax1.set_ylabel("Rate (%)")
    ax1.set_title(title)
    ax1.set_xticks(x)
    ax1.set_xticklabels(rates.index, rotation=45, ha="right")
    ax1.legend(loc="upper left")

    # Add count annotations
    ax2 = ax1.twinx()
    ax2.plot(x, rates["count"], "ko--", markersize=5, alpha=0.5, label="Loan Count")
    ax2.set_ylabel("Loan Count")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    save_fig(filename)


def plot_distributions(df):
    """Plot distributions of key numeric variables."""
    print("\n[2] Distribution plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    vars_to_plot = [
        ("sanctioned_loan_amount", "Sanctioned Loan Amount (₹)"),
        ("interest_rate", "Interest Rate (%)"),
        ("monthly_income", "Monthly Income (₹)"),
        ("ltv", "Loan-to-Value Ratio"),
        ("age", "Borrower Age"),
        ("bureau_score", "Bureau Score"),
    ]

    for ax, (col, title) in zip(axes.flatten(), vars_to_plot):
        data = df[col].dropna()
        ax.hist(data, bins=50, color="#457b9d", alpha=0.7, edgecolor="white")
        ax.axvline(data.mean(), color="#e63946", linestyle="--", linewidth=2, label=f"Mean: {data.mean():.1f}")
        ax.axvline(data.median(), color="#f4a261", linestyle="--", linewidth=2, label=f"Median: {data.median():.1f}")
        ax.set_title(title)
        ax.legend(fontsize=9)

    plt.suptitle("Distribution of Key Variables", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig("distributions")


def plot_dpd_distribution(df):
    """Histogram of Peak DPD values."""
    print("\n[3] DPD distribution...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Full distribution
    ax1.hist(df["peak_dpd"], bins=80, color="#457b9d", edgecolor="white", alpha=0.8)
    ax1.axvline(30, color="#f4a261", linestyle="--", linewidth=2, label="Stressed (DPD=30)")
    ax1.axvline(90, color="#e63946", linestyle="--", linewidth=2, label="Default (DPD=90)")
    ax1.set_title("Peak DPD Distribution (All Loans)")
    ax1.set_xlabel("Peak DPD (days)")
    ax1.set_ylabel("Count")
    ax1.legend()

    # Zoomed into 30+
    stressed = df[df["peak_dpd"] >= 30]
    ax2.hist(stressed["peak_dpd"], bins=40, color="#e63946", edgecolor="white", alpha=0.8)
    ax2.axvline(90, color="black", linestyle="--", linewidth=2, label="Default threshold (DPD=90)")
    ax2.set_title(f"Peak DPD Distribution (DPD ≥ 30, n={len(stressed)})")
    ax2.set_xlabel("Peak DPD (days)")
    ax2.set_ylabel("Count")
    ax2.legend()

    plt.suptitle("Days Past Due Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_fig("dpd_distribution")


def plot_correlation_heatmap(df):
    """Heatmap of correlations between numeric features."""
    print("\n[4] Correlation heatmap...")

    num_cols = [
        "interest_rate", "age", "monthly_income", "sanctioned_loan_amount",
        "no_of_installments", "installment_amount", "peak_dpd", "total_pos",
        "ltv", "property_value", "bureau_score", "loan_to_income_ratio",
        "emi_to_income_ratio", "is_default", "is_stressed"
    ]
    existing = [c for c in num_cols if c in df.columns]
    corr = df[existing].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.5, ax=ax,
        annot_kws={"size": 8}
    )
    ax.set_title("Correlation Heatmap — Numeric Features", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_fig("correlation_heatmap")


def plot_geographic_default(df):
    """Default rate by state."""
    print("\n[5] Geographic default distribution...")

    state_stats = df.groupby("state").agg(
        default_rate=("is_default", "mean"),
        stress_rate=("is_stressed", "mean"),
        count=("is_default", "count")
    ).sort_values("count", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(14, 7))
    x = range(len(state_stats))
    width = 0.3

    ax.bar([i - width for i in x], state_stats["default_rate"] * 100,
           width, label="Default Rate (DPD≥90)", color="#e63946", alpha=0.85)
    ax.bar(x, state_stats["stress_rate"] * 100,
           width, label="Stress Rate (DPD≥30)", color="#f4a261", alpha=0.85)

    # Loan count as line on twin axis
    ax2 = ax.twinx()
    ax2.bar([i + width for i in x], state_stats["count"],
            width, label="Loan Count", color="#a8dadc", alpha=0.6)
    ax2.set_ylabel("Loan Count")
    ax2.legend(loc="upper right")

    ax.set_xticks(x)
    ax.set_xticklabels(state_stats.index, rotation=45, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Default & Stress Rate by State (Top 15 by Volume)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")

    plt.tight_layout()
    save_fig("geographic_default_rates")


def plot_risk_segmentation(df):
    """Heatmap showing default rate by bureau score band vs income band."""
    print("\n[6] Risk segmentation heatmap...")

    if "bureau_score_band" in df.columns and "income_band" in df.columns:
        pivot = df.pivot_table(
            values="is_stressed", index="bureau_score_band",
            columns="income_band", aggfunc="mean"
        ) * 100

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            pivot, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=1, ax=ax, cbar_kws={"label": "Stress Rate (%)"}
        )
        ax.set_title("Stress Rate (%) by Bureau Score Band × Income Band",
                      fontsize=14, fontweight="bold")
        ax.set_xlabel("Income Band")
        ax.set_ylabel("Bureau Score Band")
        plt.tight_layout()
        save_fig("risk_segmentation_heatmap")


def plot_ltv_analysis(df):
    """LTV distribution colored by default status."""
    print("\n[7] LTV analysis...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # LTV distribution by default status
    ltv_perform = df[df["is_default"] == 0]["ltv"].dropna()
    ltv_default = df[df["is_default"] == 1]["ltv"].dropna()

    ax1.hist(ltv_perform, bins=50, alpha=0.7, label=f"Performing (n={len(ltv_perform)})",
             color="#457b9d", edgecolor="white")
    ax1.hist(ltv_default, bins=20, alpha=0.8, label=f"Default (n={len(ltv_default)})",
             color="#e63946", edgecolor="white")
    ax1.set_title("LTV Distribution by Default Status")
    ax1.set_xlabel("LTV Ratio")
    ax1.legend()

    # LTV bands vs default rate
    df["ltv_band"] = pd.cut(df["ltv"], bins=[0, 0.5, 0.7, 0.8, 0.9, 1.0, 1.5],
                             labels=["<50%", "50-70%", "70-80%", "80-90%", "90-100%", ">100%"])
    ltv_rates = df.groupby("ltv_band", observed=True)["is_stressed"].mean() * 100

    ax2.bar(range(len(ltv_rates)), ltv_rates.values, color=PALETTE[:len(ltv_rates)], alpha=0.85)
    ax2.set_xticks(range(len(ltv_rates)))
    ax2.set_xticklabels(ltv_rates.index, rotation=45)
    ax2.set_ylabel("Stress Rate (%)")
    ax2.set_title("Stress Rate by LTV Band")

    plt.suptitle("Loan-to-Value Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_fig("ltv_analysis")


def generate_eda_summary(df):
    """Print a text summary of the dataset for quick reference."""
    print("\n" + "=" * 60)
    print("EDA SUMMARY")
    print("=" * 60)
    print(f"Total loans:          {len(df):,}")
    print(f"Default rate (90+):   {df['is_default'].mean()*100:.2f}%  ({df['is_default'].sum()} loans)")
    print(f"Stress rate (30+):    {df['is_stressed'].mean()*100:.2f}%  ({df['is_stressed'].sum()} loans)")
    print(f"Avg loan amount:      ₹{df['sanctioned_loan_amount'].mean():,.0f}")
    print(f"Avg interest rate:    {df['interest_rate'].mean():.2f}%")
    print(f"Avg monthly income:   ₹{df['monthly_income'].mean():,.0f}")
    print(f"Avg LTV:              {df['ltv'].mean():.3f}")
    print(f"Gender (M/F):         {(df['gender']=='M').sum():,} / {(df['gender']=='F').sum():,}")
    print("=" * 60)


def main():
    print("=" * 60)
    print("CREDIT RISK MODELLING — EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"Loaded cleaned data: {df.shape[0]} rows × {df.shape[1]} columns")

    # 1. Default rates by category
    print("\n[1] Default rate by category plots...")
    plot_default_rate_by_category(df, "manufacturer", "Default Rate by Vehicle Manufacturer",
                                  "default_by_manufacturer")
    plot_default_rate_by_category(df, "bureau_score_band", "Default Rate by Bureau Score Band",
                                  "default_by_bureau_score")
    plot_default_rate_by_category(df, "age_band", "Default Rate by Age Band",
                                  "default_by_age")
    plot_default_rate_by_category(df, "income_band", "Default Rate by Income Band",
                                  "default_by_income")
    plot_default_rate_by_category(df, "business_sourced_by", "Default Rate by Business Source",
                                  "default_by_source")

    # 2-7. Other plots
    plot_distributions(df)
    plot_dpd_distribution(df)
    plot_correlation_heatmap(df)
    plot_geographic_default(df)
    plot_risk_segmentation(df)
    plot_ltv_analysis(df)

    generate_eda_summary(df)

    print(f"\nAll plots saved to: {FIGURES_DIR}/")
    print("EDA complete!")


if __name__ == "__main__":
    main()
