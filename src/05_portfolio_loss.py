"""
Module 5: Portfolio Loss Distribution
=======================================
Computes Expected Loss (EL) and simulates loss distribution
using Monte Carlo simulation. Calculates VaR and CVaR tail risk metrics.

EL = PD × LGD × EAD

Run: python src/05_portfolio_loss.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from utils import CLEANED_DATA_PATH, FIGURES_DIR, PORTFOLIO_REPORT_PATH


# ============================================================
# CONFIGURATION
# ============================================================
N_SIMULATIONS = 50000         # Monte Carlo iterations
ASSET_CORRELATION = 0.15      # Default asset correlation (Vasicek-style)
RANDOM_SEED = 42
CONFIDENCE_LEVELS = [0.95, 0.99, 0.999]


def load_data():
    """Load the dataset with PD and LGD estimates."""
    # Try loading with LGD first, then PD, then base
    for suffix in ["_with_lgd.csv", "_with_pd.csv", ".csv"]:
        path = CLEANED_DATA_PATH.replace(".csv", suffix)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded: {path} ({len(df)} loans)")
            return df

    raise FileNotFoundError("No cleaned data found. Run previous modules first.")


def compute_expected_loss(df):
    """Compute Expected Loss for each loan and the portfolio."""
    print("\n" + "=" * 50)
    print("EXPECTED LOSS COMPUTATION")
    print("=" * 50)

    # EAD = Total Principal Outstanding
    df["ead"] = df["total_pos"].clip(lower=0)

    # Use PD from model (or fallback to historical rate)
    if "pd_recommended" in df.columns:
        df["pd_final"] = df["pd_recommended"]
        print("  Using model-predicted PD")
    else:
        # Fallback: use historical default rate
        hist_rate = df["is_stressed"].mean()
        df["pd_final"] = hist_rate
        print(f"  Using historical default rate: {hist_rate:.4f}")

    # Use LGD from scenario analysis (or fallback)
    if "lgd_base_case" in df.columns:
        df["lgd_final"] = df["lgd_base_case"]
        print("  Using base case LGD from Module 4")
    else:
        # Fallback: assume 60% LGD for unsecured two-wheeler
        df["lgd_final"] = 0.60
        print("  Using assumed LGD: 60%")

    # Expected Loss per loan
    df["expected_loss"] = df["pd_final"] * df["lgd_final"] * df["ead"]

    # Portfolio statistics
    total_ead = df["ead"].sum()
    total_el = df["expected_loss"].sum()
    el_rate = total_el / total_ead if total_ead > 0 else 0

    print(f"\n  Portfolio Exposure (EAD):    ₹{total_ead:,.0f}")
    print(f"  Portfolio Expected Loss:    ₹{total_el:,.0f}")
    print(f"  EL as % of exposure:        {el_rate*100:.4f}%")
    print(f"  Avg PD (portfolio):         {df['pd_final'].mean()*100:.4f}%")
    print(f"  Avg LGD (portfolio):        {df['lgd_final'].mean()*100:.2f}%")
    print(f"  Avg EAD per loan:           ₹{df['ead'].mean():,.0f}")

    return df, total_ead, total_el


def monte_carlo_simulation(df, n_simulations=N_SIMULATIONS, rho=ASSET_CORRELATION):
    """
    Simulate portfolio loss distribution using single-factor Gaussian copula model.

    For each simulation:
    1. Draw a systematic (market) factor Z ~ N(0,1)
    2. For each loan i, draw idiosyncratic factor eps_i ~ N(0,1)
    3. Compute asset return: A_i = sqrt(rho)*Z + sqrt(1-rho)*eps_i
    4. Loan defaults if A_i < Phi^{-1}(PD_i)
    5. For each default, draw LGD from Beta distribution around estimated LGD
    6. Loss = sum(default_i * LGD_i * EAD_i)
    """
    print("\n" + "=" * 50)
    print(f"MONTE CARLO SIMULATION ({n_simulations:,} iterations)")
    print("=" * 50)

    np.random.seed(RANDOM_SEED)

    n_loans = len(df)
    pds = df["pd_final"].values
    lgds = df["lgd_final"].values
    eads = df["ead"].values

    # Convert PDs to default thresholds (inverse normal)
    # Clip PDs to avoid infinities at 0 or 1
    pds_clipped = np.clip(pds, 1e-8, 1 - 1e-8)
    default_thresholds = stats.norm.ppf(pds_clipped)

    # Pre-compute LGD beta distribution parameters
    # Beta(alpha, beta) with mean = lgd, variance controlled
    lgd_var = 0.01  # small variance around point estimate
    lgd_alpha = lgds * ((lgds * (1 - lgds) / lgd_var) - 1)
    lgd_beta_param = (1 - lgds) * ((lgds * (1 - lgds) / lgd_var) - 1)
    # Clip to valid range
    lgd_alpha = np.clip(lgd_alpha, 0.1, 100)
    lgd_beta_param = np.clip(lgd_beta_param, 0.1, 100)

    # Simulation
    losses = np.zeros(n_simulations)
    default_counts = np.zeros(n_simulations)

    print("  Simulating...", end="", flush=True)

    # Batch computation for efficiency
    batch_size = 1000
    for batch_start in range(0, n_simulations, batch_size):
        batch_end = min(batch_start + batch_size, n_simulations)
        batch_n = batch_end - batch_start

        # Systematic factor (market shock)
        Z = np.random.normal(0, 1, batch_n)

        # Idiosyncratic factors for each loan
        eps = np.random.normal(0, 1, (batch_n, n_loans))

        # Asset returns: A_i = sqrt(rho)*Z + sqrt(1-rho)*eps_i
        sqrt_rho = np.sqrt(rho)
        sqrt_1_rho = np.sqrt(1 - rho)
        asset_returns = sqrt_rho * Z[:, np.newaxis] + sqrt_1_rho * eps

        # Default indicator: default if asset return < threshold
        defaults = (asset_returns < default_thresholds[np.newaxis, :]).astype(float)

        # Draw LGD for defaulted loans from Beta distribution
        lgd_draws = np.random.beta(
            lgd_alpha[np.newaxis, :].repeat(batch_n, axis=0),
            lgd_beta_param[np.newaxis, :].repeat(batch_n, axis=0)
        )

        # Loss for each simulation
        batch_losses = (defaults * lgd_draws * eads[np.newaxis, :]).sum(axis=1)
        losses[batch_start:batch_end] = batch_losses
        default_counts[batch_start:batch_end] = defaults.sum(axis=1)

        if (batch_start // batch_size) % 10 == 0:
            print(".", end="", flush=True)

    print(" Done!")

    return losses, default_counts


def compute_risk_metrics(losses, total_ead):
    """Compute VaR, CVaR, and other tail risk metrics."""
    print("\n" + "=" * 50)
    print("RISK METRICS")
    print("=" * 50)

    el = np.mean(losses)
    ul = np.std(losses)

    print(f"\n  Expected Loss (EL):         ₹{el:,.0f}  ({el/total_ead*100:.4f}% of EAD)")
    print(f"  Unexpected Loss (UL/σ):     ₹{ul:,.0f}  ({ul/total_ead*100:.4f}% of EAD)")
    print(f"  UL/EL ratio:                {ul/el:.2f}x")

    metrics = {"el": el, "ul": ul}

    for conf in CONFIDENCE_LEVELS:
        var = np.percentile(losses, conf * 100)
        # CVaR = Expected Shortfall = mean of losses above VaR
        cvar = losses[losses >= var].mean() if (losses >= var).any() else var
        ec = var - el   # Economic Capital

        pct_label = f"{conf*100:.1f}%"
        print(f"\n  VaR ({pct_label}):                ₹{var:,.0f}  ({var/total_ead*100:.4f}% of EAD)")
        print(f"  CVaR/ES ({pct_label}):            ₹{cvar:,.0f}  ({cvar/total_ead*100:.4f}% of EAD)")
        print(f"  Economic Capital ({pct_label}):   ₹{ec:,.0f}  ({ec/total_ead*100:.4f}% of EAD)")

        metrics[f"var_{pct_label}"] = var
        metrics[f"cvar_{pct_label}"] = cvar
        metrics[f"ec_{pct_label}"] = ec

    return metrics


def plot_loss_distribution(losses, metrics, total_ead):
    """Visualize the simulated loss distribution."""
    print("\n[PLOTS] Generating portfolio loss visualizations...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 1. Loss distribution histogram
    ax = axes[0]
    ax.hist(losses, bins=100, color="#457b9d", edgecolor="white", alpha=0.8, density=True)
    ax.axvline(metrics["el"], color="#2a9d8f", linewidth=2, linestyle="--",
               label=f"Expected Loss: ₹{metrics['el']:,.0f}")
    ax.axvline(metrics["var_95.0%"], color="#f4a261", linewidth=2, linestyle="--",
               label=f"VaR 95%: ₹{metrics['var_95.0%']:,.0f}")
    ax.axvline(metrics["var_99.0%"], color="#e63946", linewidth=2, linestyle="--",
               label=f"VaR 99%: ₹{metrics['var_99.0%']:,.0f}")

    ax.set_xlabel("Portfolio Loss (₹)")
    ax.set_ylabel("Density")
    ax.set_title("Simulated Portfolio Loss Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. Loss as percentage of EAD
    ax = axes[1]
    loss_pct = losses / total_ead * 100
    ax.hist(loss_pct, bins=100, color="#e63946", edgecolor="white", alpha=0.7, density=True)
    ax.axvline(metrics["el"]/total_ead*100, color="#2a9d8f", linewidth=2, linestyle="--",
               label=f"EL: {metrics['el']/total_ead*100:.3f}%")
    ax.axvline(metrics["var_99.0%"]/total_ead*100, color="black", linewidth=2, linestyle="--",
               label=f"VaR 99%: {metrics['var_99.0%']/total_ead*100:.3f}%")

    ax.set_xlabel("Portfolio Loss (% of EAD)")
    ax.set_ylabel("Density")
    ax.set_title("Loss Distribution (% of Exposure)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Monte Carlo Loss Simulation ({N_SIMULATIONS:,} scenarios, ρ={ASSET_CORRELATION})",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "portfolio_loss_distribution.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  [PLOT] Saved: portfolio_loss_distribution.png")

    # 3. Default count distribution
    # (Saved during simulation but not passed here — generate from losses)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Loss exceedance curve
    sorted_losses = np.sort(losses)[::-1]
    exceedance_prob = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)

    ax.plot(sorted_losses, exceedance_prob, color="#e63946", linewidth=1.5)
    ax.axhline(0.01, color="#f4a261", linestyle="--", alpha=0.7, label="1% exceedance")
    ax.axhline(0.05, color="#2a9d8f", linestyle="--", alpha=0.7, label="5% exceedance")
    ax.set_xlabel("Loss Amount (₹)")
    ax.set_ylabel("Exceedance Probability")
    ax.set_title("Loss Exceedance Curve", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "loss_exceedance_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [PLOT] Saved: loss_exceedance_curve.png")


def generate_portfolio_report(df, metrics, total_ead):
    """Generate a summary portfolio risk report."""
    report = pd.DataFrame([
        {"Metric": "Total Number of Loans", "Value": f"{len(df):,}"},
        {"Metric": "Total Exposure (EAD)", "Value": f"₹{total_ead:,.0f}"},
        {"Metric": "Portfolio Avg PD", "Value": f"{df['pd_final'].mean()*100:.4f}%"},
        {"Metric": "Portfolio Avg LGD", "Value": f"{df['lgd_final'].mean()*100:.2f}%"},
        {"Metric": "Expected Loss (EL)", "Value": f"₹{metrics['el']:,.0f}"},
        {"Metric": "EL % of EAD", "Value": f"{metrics['el']/total_ead*100:.4f}%"},
        {"Metric": "Unexpected Loss (σ)", "Value": f"₹{metrics['ul']:,.0f}"},
        {"Metric": "VaR 95%", "Value": f"₹{metrics['var_95.0%']:,.0f}"},
        {"Metric": "VaR 99%", "Value": f"₹{metrics['var_99.0%']:,.0f}"},
        {"Metric": "CVaR / ES 99%", "Value": f"₹{metrics['cvar_99.0%']:,.0f}"},
        {"Metric": "Economic Capital 99%", "Value": f"₹{metrics['ec_99.0%']:,.0f}"},
        {"Metric": "EC % of EAD", "Value": f"{metrics['ec_99.0%']/total_ead*100:.4f}%"},
        {"Metric": "Monte Carlo Simulations", "Value": f"{N_SIMULATIONS:,}"},
        {"Metric": "Asset Correlation (ρ)", "Value": f"{ASSET_CORRELATION}"},
    ])
    report.to_csv(PORTFOLIO_REPORT_PATH, index=False)
    print(f"\n  Portfolio report saved to: {PORTFOLIO_REPORT_PATH}")
    return report


def main():
    print("=" * 60)
    print("CREDIT RISK MODELLING — PORTFOLIO LOSS DISTRIBUTION")
    print("=" * 60)

    # Load data
    df = load_data()

    # Compute expected loss
    df, total_ead, total_el = compute_expected_loss(df)

    # Monte Carlo simulation
    losses, default_counts = monte_carlo_simulation(df)

    # Risk metrics
    metrics = compute_risk_metrics(losses, total_ead)

    # Visualizations
    plot_loss_distribution(losses, metrics, total_ead)

    # Portfolio report
    report = generate_portfolio_report(df, metrics, total_ead)
    print("\n  PORTFOLIO RISK REPORT:")
    print(report.to_string(index=False))

    # Save updated data
    output_path = CLEANED_DATA_PATH.replace(".csv", "_with_el.csv")
    df.to_csv(output_path, index=False)
    print(f"\n  Data with EL saved to: {output_path}")

    print("\n" + "=" * 60)
    print("PORTFOLIO LOSS ANALYSIS COMPLETE")
    print("=" * 60)

    return df, losses, metrics


if __name__ == "__main__":
    main()
