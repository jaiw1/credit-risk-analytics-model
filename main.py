"""
Credit Risk Modelling AI Project — Master Runner
===================================================
Runs the entire end-to-end pipeline in sequence.

Usage:
    python main.py           # Run all modules
    python main.py --from 3  # Resume from module 3

Modules:
    1. Data Cleaning & Preparation
    2. Exploratory Data Analysis (EDA)
    3. PD Modelling (Logistic Reg + Gradient Boosting + Neural Network)
    4. LGD Estimation (Scenario Analysis)
    5. Portfolio Loss Distribution (Monte Carlo)
    6. Default Correlation & Segmentation (incl. Macro Variables)
    7. Cashflow Simulation (PD Curve + Stress Scenarios)
    8. Excel Output (consolidated model outputs)
"""

import sys
import time

def run_module(module_num, module_name, import_path):
    """Run a single module and track timing."""
    print("\n" + "█" * 70)
    print(f"  MODULE {module_num}: {module_name}")
    print("█" * 70)

    start = time.time()
    try:
        mod = __import__(import_path)
        mod.main()
        elapsed = time.time() - start
        print(f"\n  ✓ Module {module_num} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ✗ Module {module_num} FAILED after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_excel_output():
    """Consolidate all model outputs into a formatted Excel file."""
    print("\n" + "█" * 70)
    print("  MODULE 9: EXCEL OUTPUT GENERATION")
    print("█" * 70)

    import pandas as pd
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
    from src.utils import (
        CLEANED_DATA_PATH, MODEL_OUTPUTS_PATH, PORTFOLIO_REPORT_PATH,
        EXCEL_OUTPUT_PATH, PROJECT_ROOT
    )

    try:
        writer = pd.ExcelWriter(EXCEL_OUTPUT_PATH, engine="openpyxl")

        # Sheet 1: PD per loan
        if os.path.exists(MODEL_OUTPUTS_PATH):
            pd_df = pd.read_csv(MODEL_OUTPUTS_PATH)
            pd_df.to_excel(writer, sheet_name="PD Per Loan", index=False)
            print(f"  Sheet 'PD Per Loan': {len(pd_df)} rows")

        # Sheet 2: Portfolio Risk Summary
        if os.path.exists(PORTFOLIO_REPORT_PATH):
            port_df = pd.read_csv(PORTFOLIO_REPORT_PATH)
            port_df.to_excel(writer, sheet_name="Portfolio Risk Summary", index=False)
            print(f"  Sheet 'Portfolio Risk Summary': {len(port_df)} rows")

        # Sheet 3: LGD per loan (subset)
        lgd_path = CLEANED_DATA_PATH.replace(".csv", "_with_lgd.csv")
        if os.path.exists(lgd_path):
            lgd_df = pd.read_csv(lgd_path)
            lgd_cols = [
                "loan_account_number",
                "lgd_pure_collateral_estimate",
                "lgd_base_case_20pct_floor",
                "lgd_moderate_stress",
                "lgd_severe_stress",
                "lgd_rbi_norms_65pct",
                "recovery_base_case_20pct_floor",
            ]
            existing_lgd = [c for c in lgd_cols if c in lgd_df.columns]
            if existing_lgd:
                lgd_df[existing_lgd].to_excel(writer, sheet_name="LGD Per Loan", index=False)
                print(f"  Sheet 'LGD Per Loan': {len(lgd_df)} rows")

        # Sheet 4: Cashflow Projection
        cf_path = os.path.join(os.path.dirname(CLEANED_DATA_PATH), "cashflow_projection.csv")
        if os.path.exists(cf_path):
            cf_df = pd.read_csv(cf_path)
            cf_df.to_excel(writer, sheet_name="Cashflow Projection", index=False)
            print(f"  Sheet 'Cashflow Projection': {len(cf_df)} rows")

        # Sheet 5: PD Term Structure
        pd_curve_path = os.path.join(os.path.dirname(CLEANED_DATA_PATH), "pd_term_structure.csv")
        if os.path.exists(pd_curve_path):
            pdcurve_df = pd.read_csv(pd_curve_path)
            pdcurve_df.to_excel(writer, sheet_name="PD Term Structure", index=False)
            print(f"  Sheet 'PD Term Structure': {len(pdcurve_df)} rows")

        # Sheet 6: Scored Analysis Pool
        analysis_pool_path = os.path.join(PROJECT_ROOT, "outputs", "analysis_pool_scored.csv")
        if os.path.exists(analysis_pool_path):
            analysis_df = pd.read_csv(analysis_pool_path)
            analysis_df.to_excel(writer, sheet_name="Scored Analysis Pool", index=False)
            print(f"  Sheet 'Scored Analysis Pool': {len(analysis_df)} rows")

        writer.close()
        print(f"\n  ✓ Excel output saved to: {EXCEL_OUTPUT_PATH}")
        return True

    except Exception as e:
        print(f"  ✗ Excel output FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    start_from = 1
    if len(sys.argv) > 2 and sys.argv[1] == "--from":
        start_from = int(sys.argv[2])

    # Add src to path
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

    print("=" * 70)
    print("  CREDIT RISK MODELLING AI PROJECT — FULL PIPELINE")
    print("=" * 70)
    print(f"  Starting from module {start_from}")

    modules = [
        (1, "DATA CLEANING & PREPARATION", "01_data_cleaning"),
        (2, "EXPLORATORY DATA ANALYSIS", "02_eda"),
        (3, "PD MODELLING", "03_pd_model"),
        (4, "LGD ESTIMATION", "04_lgd_model"),
        (5, "PORTFOLIO LOSS DISTRIBUTION", "05_portfolio_loss"),
        (6, "DEFAULT CORRELATION & SEGMENTATION", "06_correlation_model"),
        (7, "CASHFLOW SIMULATION", "07_cashflow_simulation"),
        (8, "SCORE ANALYSIS POOL", "08_score_analysis_pool"),
    ]

    results = {}
    overall_start = time.time()

    for num, name, path in modules:
        if num < start_from:
            print(f"\n  [SKIP] Module {num}: {name}")
            continue
        success = run_module(num, name, path)
        results[num] = success
        if not success:
            print(f"\n  ⚠ Module {num} failed. Continuing with remaining modules...")

    # Excel output (always runs)
    if start_from <= 9:
        results[9] = generate_excel_output()

    total_time = time.time() - overall_start

    # Summary
    print("\n\n" + "=" * 70)
    print("  PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    for num, success in sorted(results.items()):
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  Module {num}: {status}")
    print(f"\n  Total execution time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
