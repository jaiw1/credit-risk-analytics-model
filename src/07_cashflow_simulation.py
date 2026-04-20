"""
Module 7: PD Curve & Cashflow Simulation
==========================================
Simulates cashflows from the loan pool:
  - PD term structure (marginal PD per period)
  - Prepayment modelling (CPR)
  - Monthly cashflow waterfall projection
  - Cumulative default curve

Run: python src/07_cashflow_simulation.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import CLEANED_DATA_PATH, FIGURES_DIR

# ============================================================
# ASSUMPTIONS
# ============================================================
CPR_ANNUAL = 0.07           # 7% Conditional Prepayment Rate (annual)
DISCOUNT_RATE = 0.10        # 10% annual discount rate for NPV
RECOVERY_LAG_MONTHS = 6     # Months between default and recovery
BASE_RECOVERY_RATE = 0.40   # Base recovery on defaulted loans
MAX_PROJECTION_MONTHS = 60  # Maximum months to project


def load_data():
    """Load the most recent dataset."""
    for suffix in ["_with_clusters.csv", "_with_el.csv", "_with_lgd.csv", "_with_pd.csv", ".csv"]:
        path = CLEANED_DATA_PATH.replace(".csv", suffix)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded: {path} ({len(df)} loans)")
            return df
    raise FileNotFoundError("No data found.")


def build_pd_term_structure(df, max_months=MAX_PROJECTION_MONTHS):
    """
    Build a PD term structure (marginal default rates per month).

    Uses vintage analysis: analyze how defaults develop over loan age.
    Approach: logistic-based hazard rate increasing with seasoning.
    """
    print("\n" + "=" * 50)
    print("PD TERM STRUCTURE")
    print("=" * 50)

    # Get portfolio average PD
    if "pd_final" in df.columns:
        lifetime_pd = df["pd_final"].mean()
    else:
        lifetime_pd = df["is_stressed"].mean()

    # Determine remaining tenor to project
    if "balance_tenure" in df.columns:
        avg_remaining = df["balance_tenure"].mean()
    else:
        avg_remaining = 24

    projection_months = min(int(avg_remaining) + 6, max_months)

    print(f"  Lifetime PD (avg):       {lifetime_pd:.4f}")
    print(f"  Avg remaining tenure:    {avg_remaining:.0f} months")
    print(f"  Projection horizon:      {projection_months} months")

    # Build marginal PD curve using a logistic hazard model
    # h(t) = h0 * exp(beta * t) — increasing hazard rate
    # Calibrated so cumulative PD = lifetime_pd over the tenure

    months = np.arange(1, projection_months + 1)

    # Use the seasoning data to calibrate: loans further into their term
    # have shown their stress behavior already
    if "seasoning_months" in df.columns and "peak_dpd" in df.columns:
        # Empirical: fraction of loans that reached stress by each seasoning month
        seasoning = pd.to_numeric(df["seasoning_months"], errors="coerce").dropna()
        peak = df.loc[seasoning.index, "peak_dpd"]

        # Group by seasoning bucket and compute cumulative default rate
        buckets = pd.cut(seasoning, bins=range(0, int(seasoning.max()) + 3, 3))
        vintage_dr = peak.groupby(buckets, observed=True).apply(lambda x: (x >= 30).mean())

    # Calibrate a simple logistic curve: PD(t) = lifetime_pd * CDF(t, loc, scale)
    from scipy.stats import lognorm

    # Parameters: mode at ~12 months, spread over tenure
    shape = 0.8
    scale_param = avg_remaining * 0.4

    # Marginal PD at each month
    cumulative_pd = lognorm.cdf(months, s=shape, scale=scale_param)
    cumulative_pd = cumulative_pd / cumulative_pd[-1] * lifetime_pd  # normalize

    # Marginal PD = difference in cumulative
    marginal_pd = np.diff(cumulative_pd, prepend=0)
    marginal_pd = np.clip(marginal_pd, 0, 1)

    # Survival probability
    survival = 1 - cumulative_pd

    print(f"\n  PD Term Structure (selected months):")
    for m in [1, 3, 6, 12, 18, 24, 36, 48]:
        if m <= projection_months:
            idx = m - 1
            print(f"    Month {m:>3}: Marginal PD = {marginal_pd[idx]:.6f}  "
                  f"Cumul PD = {cumulative_pd[idx]:.6f}  "
                  f"Survival = {survival[idx]:.6f}")

    return months, marginal_pd, cumulative_pd, survival, projection_months


def compute_smm(cpr_annual=CPR_ANNUAL):
    """Convert annual CPR to monthly SMM (Single Monthly Mortality)."""
    smm = 1 - (1 - cpr_annual) ** (1/12)
    return smm


def simulate_cashflows(df, months, marginal_pd, survival, projection_months):
    """
    Project monthly cashflows from the loan pool.

    Cashflow Waterfall:
    1. Beginning balance (outstanding principal)
    2. Scheduled principal payment
    3. Scheduled interest payment
    4. Prepayments (CPR-based)
    5. Defaults (PD curve-based)
    6. Recoveries (with lag)
    7. Net cashflow & ending balance
    """
    print("\n" + "=" * 50)
    print("CASHFLOW SIMULATION")
    print("=" * 50)

    # Portfolio aggregates
    initial_balance = df["total_pos"].sum()
    total_ios = df["total_ios"].sum()
    avg_interest_rate = df["interest_rate"].mean() / 100  # annual
    monthly_rate = avg_interest_rate / 12
    avg_installment = df["installment_amount"].mean()
    avg_tenure = df["no_of_installments"].mean()
    avg_remaining = df["balance_tenure"].mean()

    # SMM for prepayments
    smm = compute_smm(CPR_ANNUAL)

    print(f"  Initial Pool Balance:     ₹{initial_balance:,.0f}")
    print(f"  Avg Interest Rate:        {avg_interest_rate*100:.2f}% (monthly: {monthly_rate*100:.4f}%)")
    print(f"  CPR:                      {CPR_ANNUAL*100:.1f}% | SMM: {smm*100:.4f}%")
    print(f"  Recovery lag:             {RECOVERY_LAG_MONTHS} months")

    # Initialize cashflow table
    cf = pd.DataFrame(index=range(projection_months), columns=[
        "month", "beg_balance", "scheduled_principal", "scheduled_interest",
        "prepayments", "defaults", "recoveries", "total_collections",
        "net_loss", "end_balance", "cumulative_defaults", "cumulative_prepay"
    ])
    cf = cf.fillna(0.0)
    cf["month"] = months

    balance = float(initial_balance)
    cumul_defaults = 0.0
    cumul_prepay = 0.0
    default_queue = []  # (recovery_month, recovery_amount)

    for i in range(projection_months):
        cf.loc[i, "beg_balance"] = balance

        if balance <= 0:
            break

        # Scheduled payments (level amortization approximation)
        # Principal portion = balance / remaining_months
        remaining_months = max(avg_remaining - i, 1)
        sched_principal = balance / remaining_months
        sched_interest = balance * monthly_rate

        # Prepayments: SMM applied to remaining balance after scheduled payment
        remaining_after_sched = balance - sched_principal
        prepay = remaining_after_sched * smm

        # Defaults: apply marginal PD to current surviving balance
        default_amount = balance * marginal_pd[i]

        # Adjust if total exceeds balance
        total_outflow = sched_principal + prepay + default_amount
        if total_outflow > balance:
            scale = balance / total_outflow
            sched_principal *= scale
            prepay *= scale
            default_amount *= scale

        # Queue recovery for later
        recovery_amount = default_amount * BASE_RECOVERY_RATE
        recovery_month = i + RECOVERY_LAG_MONTHS
        default_queue.append((recovery_month, recovery_amount))

        # Process recoveries due this month
        recoveries = sum(amt for m, amt in default_queue if m == i)

        # Update accumulators
        cumul_defaults += default_amount
        cumul_prepay += prepay

        # Collections = scheduled payments + prepayments + recoveries
        total_collections = sched_principal + sched_interest + prepay + recoveries

        # Net loss = defaults - recoveries
        net_loss = default_amount - recoveries

        # End balance
        balance = balance - sched_principal - prepay - default_amount
        balance = max(balance, 0)

        # Record
        cf.loc[i, "scheduled_principal"] = sched_principal
        cf.loc[i, "scheduled_interest"] = sched_interest
        cf.loc[i, "prepayments"] = prepay
        cf.loc[i, "defaults"] = default_amount
        cf.loc[i, "recoveries"] = recoveries
        cf.loc[i, "total_collections"] = total_collections
        cf.loc[i, "net_loss"] = net_loss
        cf.loc[i, "end_balance"] = balance
        cf.loc[i, "cumulative_defaults"] = cumul_defaults
        cf.loc[i, "cumulative_prepay"] = cumul_prepay

    # Convert to numeric
    for col in cf.columns:
        cf[col] = pd.to_numeric(cf[col], errors="coerce").fillna(0)

    # NPV calculation
    discount_factors = (1 + DISCOUNT_RATE / 12) ** (-cf["month"])
    cf["pv_collections"] = cf["total_collections"] * discount_factors
    npv = cf["pv_collections"].sum()

    print(f"\n  Cashflow Summary:")
    print(f"    Total Scheduled Principal:  ₹{cf['scheduled_principal'].sum():,.0f}")
    print(f"    Total Scheduled Interest:   ₹{cf['scheduled_interest'].sum():,.0f}")
    print(f"    Total Prepayments:          ₹{cf['prepayments'].sum():,.0f}")
    print(f"    Total Defaults:             ₹{cf['defaults'].sum():,.0f}")
    print(f"    Total Recoveries:           ₹{cf['recoveries'].sum():,.0f}")
    print(f"    Total Net Loss:             ₹{cf['net_loss'].sum():,.0f}")
    print(f"    NPV of Collections:         ₹{npv:,.0f}")
    print(f"    NPV as % of initial:        {npv/initial_balance*100:.2f}%")

    return cf, initial_balance


def run_stress_cashflows(df, months, marginal_pd, survival, projection_months):
    """Run cashflow simulation under stressed assumptions."""
    print("\n" + "=" * 50)
    print("STRESS SCENARIO CASHFLOWS")
    print("=" * 50)

    scenarios = {
        "Base": {"pd_mult": 1.0, "cpr": 0.07, "recovery": 0.40},
        "Moderate Stress": {"pd_mult": 2.0, "cpr": 0.03, "recovery": 0.30},
        "Severe Stress": {"pd_mult": 3.0, "cpr": 0.01, "recovery": 0.20},
    }

    results = {}
    for name, params in scenarios.items():
        print(f"\n--- {name} (PD×{params['pd_mult']}, CPR={params['cpr']*100}%, Recovery={params['recovery']*100}%) ---")

        # Adjust marginal PD
        stressed_mpd = np.clip(marginal_pd * params["pd_mult"], 0, 0.5)

        global CPR_ANNUAL, BASE_RECOVERY_RATE
        orig_cpr = CPR_ANNUAL
        orig_recovery = BASE_RECOVERY_RATE
        CPR_ANNUAL = params["cpr"]
        BASE_RECOVERY_RATE = params["recovery"]

        cf, init_bal = simulate_cashflows(df, months, stressed_mpd, survival, projection_months)
        results[name] = {"cf": cf, "init_balance": init_bal}

        CPR_ANNUAL = orig_cpr
        BASE_RECOVERY_RATE = orig_recovery

    return results


def plot_cashflow_results(cf, initial_balance, stress_results=None):
    """Generate cashflow visualizations."""
    print("\n[PLOTS] Generating cashflow visualizations...")

    # 1. Cashflow waterfall
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Balance amortization
    ax = axes[0, 0]
    ax.fill_between(cf["month"], cf["beg_balance"], alpha=0.3, color="#457b9d")
    ax.plot(cf["month"], cf["beg_balance"], color="#457b9d", linewidth=2, label="Outstanding Balance")
    ax.set_title("Pool Balance Amortization")
    ax.set_xlabel("Month")
    ax.set_ylabel("Balance (₹)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Monthly cashflows
    ax = axes[0, 1]
    ax.stackplot(cf["month"],
                 cf["scheduled_principal"], cf["scheduled_interest"],
                 cf["prepayments"], cf["recoveries"],
                 labels=["Principal", "Interest", "Prepayments", "Recoveries"],
                 colors=["#457b9d", "#2a9d8f", "#f4a261", "#a8dadc"], alpha=0.8)
    ax.plot(cf["month"], cf["defaults"], "r-", linewidth=2, label="Defaults")
    ax.set_title("Monthly Cashflow Breakdown")
    ax.set_xlabel("Month")
    ax.set_ylabel("Amount (₹)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Cumulative default curve
    ax = axes[1, 0]
    cumdef_pct = cf["cumulative_defaults"] / initial_balance * 100
    ax.plot(cf["month"], cumdef_pct, "r-", linewidth=2.5, label="Cumulative Default %")
    ax.fill_between(cf["month"], cumdef_pct, alpha=0.2, color="#e63946")
    ax.set_title("Cumulative Default Curve")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative Default (% of initial balance)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Net loss over time
    ax = axes[1, 1]
    cumul_loss = cf["net_loss"].cumsum()
    cumul_collections = cf["total_collections"].cumsum()
    ax.plot(cf["month"], cumul_collections, color="#2a9d8f", linewidth=2, label="Cumulative Collections")
    ax.plot(cf["month"], cumul_loss, color="#e63946", linewidth=2, label="Cumulative Net Loss")
    ax.set_title("Cumulative Collections vs Net Loss")
    ax.set_xlabel("Month")
    ax.set_ylabel("Amount (₹)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Cashflow Simulation — Base Case", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "cashflow_waterfall.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [PLOT] Saved: cashflow_waterfall.png")

    # 2. Stress scenario comparison
    if stress_results:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        colors = {"Base": "#2a9d8f", "Moderate Stress": "#f4a261", "Severe Stress": "#e63946"}

        # Balance comparison
        ax = axes[0]
        for name, res in stress_results.items():
            scf = res["cf"]
            ax.plot(scf["month"], scf["beg_balance"], linewidth=2,
                    color=colors[name], label=name)
        ax.set_title("Balance Rundown — Scenario Comparison")
        ax.set_xlabel("Month")
        ax.set_ylabel("Balance (₹)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Cumulative default comparison
        ax = axes[1]
        for name, res in stress_results.items():
            scf = res["cf"]
            init = res["init_balance"]
            ax.plot(scf["month"], scf["cumulative_defaults"] / init * 100,
                    linewidth=2, color=colors[name], label=name)
        ax.set_title("Cumulative Defaults (% of Pool)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Cumulative Default %")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # NPV comparison
        ax = axes[2]
        npvs = {}
        for name, res in stress_results.items():
            scf = res["cf"]
            df_factors = (1 + DISCOUNT_RATE / 12) ** (-scf["month"])
            npv = (scf["total_collections"] * df_factors).sum()
            init = res["init_balance"]
            npvs[name] = npv / init * 100

        bars = ax.bar(npvs.keys(), npvs.values(),
                      color=[colors[n] for n in npvs.keys()], alpha=0.85)
        ax.set_title("NPV of Collections (% of Initial Balance)")
        ax.set_ylabel("NPV %")
        for bar, val in zip(bars, npvs.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

        plt.suptitle("Cashflow Stress Scenario Analysis", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "cashflow_stress_scenarios.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("  [PLOT] Saved: cashflow_stress_scenarios.png")


def main():
    print("=" * 60)
    print("CREDIT RISK MODELLING — CASHFLOW SIMULATION")
    print("=" * 60)

    df = load_data()

    # Build PD term structure
    months, marginal_pd, cumulative_pd, survival, proj_months = build_pd_term_structure(df)

    # Base case cashflow simulation
    cf, initial_balance = simulate_cashflows(df, months, marginal_pd, survival, proj_months)

    # Stress scenarios
    stress_results = run_stress_cashflows(df, months, marginal_pd, survival, proj_months)

    # Plots
    plot_cashflow_results(cf, initial_balance, stress_results)

    # Save cashflow table
    cf_path = os.path.join(os.path.dirname(CLEANED_DATA_PATH), "cashflow_projection.csv")
    cf.to_csv(cf_path, index=False)
    print(f"\n  Cashflow projection saved to: {cf_path}")

    # Save PD term structure
    pd_curve = pd.DataFrame({
        "month": months,
        "marginal_pd": marginal_pd,
        "cumulative_pd": cumulative_pd,
        "survival_prob": survival
    })
    pd_curve_path = os.path.join(os.path.dirname(CLEANED_DATA_PATH), "pd_term_structure.csv")
    pd_curve.to_csv(pd_curve_path, index=False)
    print(f"  PD term structure saved to: {pd_curve_path}")

    print("\n" + "=" * 60)
    print("CASHFLOW SIMULATION COMPLETE")
    print(f"  Total collections (base): ₹{cf['total_collections'].sum():,.0f}")
    print(f"  Total defaults (base):    ₹{cf['defaults'].sum():,.0f}")
    print(f"  NPV ratio:                {cf['pv_collections'].sum()/initial_balance*100:.1f}%")
    print("=" * 60)

    return cf


if __name__ == "__main__":
    main()
