"""
Module 4: Loss Given Default (LGD) Modelling
==============================================
Estimates LGD based on collateral characteristics and
performs scenario analysis under different stress conditions.

LGD = 1 - Recovery Rate
Recovery is driven by: collateral value, LTV, insurance status,
vehicle registration status, and manufacturer.

Run: python src/04_lgd_model.py
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

# ============================================================
# SCENARIO PARAMETERS
# ============================================================
SCENARIOS = {
    "Base Case": {
        "collateral_haircut": 0.0,       # No haircut on collateral
        "recovery_delay_months": 6,       # 6-month recovery process
        "recovery_cost_pct": 0.10,        # 10% recovery costs
        "depreciation_annual": 0.15,      # 15% annual depreciation for two-wheelers
        "description": "Normal economic conditions, standard recovery timeline"
    },
    "Moderate Stress": {
        "collateral_haircut": 0.20,       # 20% drop in resale values
        "recovery_delay_months": 12,      # 12-month recovery delay
        "recovery_cost_pct": 0.15,        # 15% recovery costs
        "depreciation_annual": 0.20,      # 20% depreciation under stress
        "description": "Moderate economic downturn, extended recovery"
    },
    "Severe Stress": {
        "collateral_haircut": 0.40,       # 40% crash in resale values
        "recovery_delay_months": 18,      # 18-month recovery delay
        "recovery_cost_pct": 0.20,        # 20% recovery costs
        "depreciation_annual": 0.25,      # 25% depreciation under severe stress
        "description": "Severe recession, flooded secondary market, legal delays"
    }
}


def estimate_collateral_value_at_default(df, scenario):
    """
    Estimate collateral value at the time of default considering depreciation.
    Two-wheelers depreciate rapidly — typically 15-25% per year.
    """
    haircut = scenario["collateral_haircut"]
    annual_depr = scenario["depreciation_annual"]

    # Time since disbursement (proxy for vehicle age under loan)
    if "seasoning_months" in df.columns:
        years_since_purchase = df["seasoning_months"].clip(lower=0) / 12
    else:
        years_since_purchase = df["emi_paid"] / 12

    # Depreciated collateral value
    # V(t) = V(0) * (1 - depr)^t * (1 - haircut)
    depreciated_value = (
        df["property_value"]
        * (1 - annual_depr) ** years_since_purchase
        * (1 - haircut)
    )

    return depreciated_value.clip(lower=0)


def compute_recovery_rate(df, scenario):
    """
    Compute recovery rate considering collateral value, costs, and insurance.
    Recovery Rate = min(1, (Collateral Value - Recovery Costs) / EAD)
    """
    recovery_cost_pct = scenario["recovery_cost_pct"]

    # Estimate collateral value at default
    collateral_at_default = estimate_collateral_value_at_default(df, scenario)

    # Exposure at Default (EAD) = Total Principal Outstanding
    ead = df["total_pos"].clip(lower=1)

    # Net recovery from collateral
    gross_recovery = collateral_at_default
    recovery_costs = gross_recovery * recovery_cost_pct
    net_recovery = (gross_recovery - recovery_costs).clip(lower=0)

    # Insurance bonus: insured vehicles have better recovery
    insurance_multiplier = np.where(
        df["insurance_status"] == "Received", 1.05, 0.90
    )
    net_recovery = net_recovery * insurance_multiplier

    # Registration bonus: registered vehicles easier to repossess
    reg_multiplier = np.where(
        df["reg_info_status"] == "Received", 1.0, 0.85
    )
    net_recovery = net_recovery * reg_multiplier

    # Recovery rate capped at 1.0
    recovery_rate = (net_recovery / ead).clip(0, 1)

    return recovery_rate


def compute_lgd(df, scenario):
    """Compute LGD = 1 - Recovery Rate for each loan under a given scenario."""
    recovery_rate = compute_recovery_rate(df, scenario)
    lgd = 1 - recovery_rate
    return lgd.clip(0, 1), recovery_rate


def run_scenario_analysis(df):
    """Run LGD estimation under all scenarios and compare."""
    print("\n" + "=" * 60)
    print("LGD SCENARIO ANALYSIS")
    print("=" * 60)

    results = {}

    for name, params in SCENARIOS.items():
        print(f"\n--- {name} ---")
        print(f"  {params['description']}")
        print(f"  Collateral haircut: {params['collateral_haircut']*100:.0f}%")
        print(f"  Recovery delay: {params['recovery_delay_months']} months")
        print(f"  Recovery costs: {params['recovery_cost_pct']*100:.0f}%")

        lgd, recovery = compute_lgd(df, params)
        df[f"lgd_{name.lower().replace(' ', '_')}"] = lgd
        df[f"recovery_{name.lower().replace(' ', '_')}"] = recovery

        print(f"\n  LGD Statistics:")
        print(f"    Mean LGD:        {lgd.mean():.4f} ({lgd.mean()*100:.1f}%)")
        print(f"    Median LGD:      {lgd.median():.4f}")
        print(f"    Std LGD:         {lgd.std():.4f}")
        print(f"    Min LGD:         {lgd.min():.4f}")
        print(f"    Max LGD:         {lgd.max():.4f}")
        print(f"    Mean Recovery:   {recovery.mean():.4f} ({recovery.mean()*100:.1f}%)")

        results[name] = {
            "mean_lgd": lgd.mean(),
            "median_lgd": lgd.median(),
            "mean_recovery": recovery.mean(),
            "lgd_values": lgd
        }

    return df, results


def analyze_lgd_drivers(df):
    """Analyze what drives LGD variation across the portfolio."""
    print("\n" + "=" * 60)
    print("LGD DRIVER ANALYSIS (Base Case)")
    print("=" * 60)

    lgd_col = "lgd_base_case"

    # By manufacturer
    print("\n  By Manufacturer:")
    mfr_lgd = df.groupby("manufacturer")[lgd_col].agg(["mean", "median", "count"])
    mfr_lgd = mfr_lgd.sort_values("mean", ascending=False)
    for idx, row in mfr_lgd.iterrows():
        print(f"    {idx:15s}  Mean LGD: {row['mean']:.3f}  Count: {row['count']:,}")

    # By LTV band
    print("\n  By LTV Band:")
    df["ltv_band_lgd"] = pd.cut(df["ltv"], bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0],
                                  labels=["<60%", "60-70%", "70-80%", "80-90%", "90-100%", ">100%"])
    ltv_lgd = df.groupby("ltv_band_lgd", observed=True)[lgd_col].agg(["mean", "count"])
    for idx, row in ltv_lgd.iterrows():
        print(f"    {str(idx):15s}  Mean LGD: {row['mean']:.3f}  Count: {row['count']:,}")

    # By insurance status
    print("\n  By Insurance Status:")
    ins_lgd = df.groupby("insurance_status")[lgd_col].mean()
    for idx, val in ins_lgd.items():
        print(f"    {idx:15s}  Mean LGD: {val:.3f}")


def plot_lgd_results(df, results):
    """Generate LGD visualizations."""
    print("\n[PLOTS] Generating LGD visualizations...")

    # 1. LGD distribution by scenario
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["#2a9d8f", "#f4a261", "#e63946"]

    for ax, (name, res), color in zip(axes, results.items(), colors):
        ax.hist(res["lgd_values"], bins=50, color=color, edgecolor="white", alpha=0.8)
        ax.axvline(res["mean_lgd"], color="black", linestyle="--",
                   label=f"Mean: {res['mean_lgd']:.3f}")
        ax.set_title(f"{name}\n(Mean LGD = {res['mean_lgd']*100:.1f}%)")
        ax.set_xlabel("LGD")
        ax.set_ylabel("Count")
        ax.legend()

    plt.suptitle("LGD Distribution Under Different Scenarios", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "lgd_scenarios.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [PLOT] Saved: lgd_scenarios.png")

    # 2. Scenario comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    scenario_names = list(results.keys())
    mean_lgds = [r["mean_lgd"] for r in results.values()]
    mean_recoveries = [r["mean_recovery"] for r in results.values()]

    x = np.arange(len(scenario_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, mean_lgds, width, label="Mean LGD", color="#e63946", alpha=0.85)
    bars2 = ax.bar(x + width/2, mean_recoveries, width, label="Mean Recovery Rate", color="#2a9d8f", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names)
    ax.set_ylabel("Rate")
    ax.set_title("LGD vs Recovery Rate — Scenario Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "lgd_scenario_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [PLOT] Saved: lgd_scenario_comparison.png")

    # 3. LGD by manufacturer
    fig, ax = plt.subplots(figsize=(12, 6))
    lgd_col = "lgd_base_case"
    mfr_lgd = df.groupby("manufacturer")[lgd_col].mean().sort_values(ascending=False)

    ax.barh(range(len(mfr_lgd)), mfr_lgd.values, color="#457b9d", alpha=0.85)
    ax.set_yticks(range(len(mfr_lgd)))
    ax.set_yticklabels(mfr_lgd.index)
    ax.set_xlabel("Mean LGD (Base Case)")
    ax.set_title("LGD by Vehicle Manufacturer", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    for i, val in enumerate(mfr_lgd.values):
        ax.text(val + 0.005, i, f"{val:.3f}", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "lgd_by_manufacturer.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [PLOT] Saved: lgd_by_manufacturer.png")


def main():
    print("=" * 60)
    print("CREDIT RISK MODELLING — LGD ESTIMATION")
    print("=" * 60)

    # Load data (with PD if available, otherwise base cleaned data)
    pd_data_path = CLEANED_DATA_PATH.replace(".csv", "_with_pd.csv")
    if os.path.exists(pd_data_path):
        df = pd.read_csv(pd_data_path)
        print(f"Loaded data with PD: {len(df)} loans")
    else:
        df = pd.read_csv(CLEANED_DATA_PATH)
        print(f"Loaded cleaned data: {len(df)} loans")

    # Run scenario analysis
    df, results = run_scenario_analysis(df)

    # Analyze LGD drivers
    analyze_lgd_drivers(df)

    # Generate plots
    plot_lgd_results(df, results)

    # Save updated data
    output_path = CLEANED_DATA_PATH.replace(".csv", "_with_lgd.csv")
    df.to_csv(output_path, index=False)
    print(f"\nData with LGD saved to: {output_path}")

    print("\n" + "=" * 60)
    print("LGD ESTIMATION COMPLETE")
    for name, res in results.items():
        print(f"  {name}: Mean LGD = {res['mean_lgd']:.4f} ({res['mean_lgd']*100:.1f}%)")
    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
