# Credit Risk Modelling AI Project — Task Implementation Mapping

This document provides a detailed mapping of the tasks outlined in the **Detailed Brief for Credit Risk Modelling AI Project.pdf** to the actual code implemented in this repository.

---

## 1. Data Cleaning and Preparation
**PDF Requirement:** Perform thorough data cleaning, develop data cleaning scripts, handle missing values/inconsistencies, and maintain a data cleaning log.
**Implementation:** 
- **Script:** `src/01_data_cleaning.py`
- **What is done:**
  - Systematically loads the raw sample data.
  - Handles missing values by applying median imputations for numeric variables.
  - Fixes formatting issues, particularly standardizing date formats and handling categorical data inconsistencies.
  - Maintains a clear workflow that produces a clean, reproducible analytical dataset (`cleaned_data.csv`).
  - *Note: Additional Exploratory Data Analysis (EDA) is performed in `src/02_eda.py` to ensure data quality before modelling.*

## 2. Probability of Default (PD) Model
**PDF Requirement:** Build a baseline statistical model and an AI/ML model, perform validation, estimate PD for every loan, recommend a model, and explain risk drivers.
**Implementation:**
- **Script:** `src/03_pd_model.py`
- **What is done:**
  - **Baseline Statistical Model:** Implements a Logistic Regression model with Platt scaling (calibration) to ensure probability estimates align with observed default rates.
  - **AI/ML Models:** Implements both a tree-based ensemble (Gradient Boosting / Random Forest) and a Neural Network (MLP Classifier).
  - **Model Validation:** Uses standard credit risk validation techniques including ROC-AUC, Gini, Kolmogorov-Smirnov (KS) statistic, Brier Score, reliability diagrams (calibration curves), and decile validation tables.
  - **Risk Drivers Explanation:** Uses **SHAP (SHapley Additive exPlanations)** to generate feature importance plots, explaining exactly which variables drive the credit risk (e.g., generating `shap_summary.png`).
  - **PD Estimate & Recommendation:** Estimates PD for each individual loan and dynamically recommends the best-performing model based on AUC and Brier scores. Because 90+ DPD defaults are rare in this sample, the headline model is a 30+ DPD stress-PD model; a 90+ DPD model should be treated as a rare-event extension.

## 3. Loss Given Default (LGD) Modelling
**PDF Requirement:** Estimate LGD based on collateral characteristics for each loan and perform scenario analysis (macroeconomic factors, recovery delays).
**Implementation:**
- **Script:** `src/04_lgd_model.py`
- **What is done:**
  - Calculates specific LGD estimates for each loan using available collateral and recovery data.
  - Performs **Scenario Analysis** by estimating LGD across a pure collateral case, a 20% floor base case, moderate stress, severe stress, and a 65% floor case, factoring in collateral haircuts, recovery delays, and recovery costs.

## 4. Portfolio Loss Distribution
**PDF Requirement:** Compute the expected loss of the pool, perform loss distribution simulation (Monte Carlo), and estimate tail risk.
**Implementation:**
- **Script:** `src/05_portfolio_loss.py`
- **What is done:**
  - Computes the aggregate Expected Loss (EL) for the entire portfolio based on the estimated PDs and LGDs.
  - Executes a **Monte Carlo simulation** to generate a full loss distribution for the portfolio.
  - Calculates Tail Risk metrics such as **Value at Risk (VaR)** and **Expected Shortfall (ES)** to quantify the risk of extreme portfolio losses.

## 5. Default Correlation Modelling
**PDF Requirement:** Model correlation (using factor models, clustering, or AI similarity), incorporate pool-level correlation with external variables, and produce borrower segments.
**Implementation:**
- **Script:** `src/06_correlation_model.py`
- **What is done:**
  - Incorporates pool-level default correlation linked to actual time-aligned macroeconomic and external variables from `data/india_macro_quarterly.csv`.
  - Uses clustering/segmentation algorithms to group borrowers into specific risk segments with similar stress behaviours.
  - Produces a macro/vintage correlation dataset and reports sample size limits instead of creating synthetic default rates.

## 6. Probability of Default Curve (Cashflow Simulation)
**PDF Requirement:** Simulate cashflows from the pool, making appropriate assumptions regarding prepayment and default timing.
**Implementation:**
- **Script:** `src/07_cashflow_simulation.py`
- **What is done:**
  - Simulates expected monthly cashflows from the loan pool over the lifetime of the loans.
  - Generates a **PD Term Structure** (PD Curve) to project when defaults are most likely to occur over time.
  - Incorporates assumptions around prepayment speeds and timing of defaults to provide a realistic amortization and cashflow schedule.

## Deliverables & Outputs
**PDF Requirement:** Code notebook (with detailed comments), Clean datasets, Model outputs in Excel/CSV, Presentation.
**Implementation:**
- **Master Runner (`main.py`):** Acts as the central pipeline runner, executing modules 1 through 7 sequentially. 
- **Excel/CSV Outputs:** Module 8 in `main.py` automatically consolidates all key outputs (PD per loan, LGD per loan, Portfolio Risk Summary, Cashflow Projections, PD Term Structure) into a single master Excel file.
- **Code Comments:** All `.py` files in the `src/` directory are thoroughly documented with detailed comments explaining the methodology and code functionality.
- **Visualisations:** The pipeline automatically generates and saves visualisations (ROC curves, SHAP summary plots, calibration curves, decile charts) to the `figures/` directory to be used in the required PowerPoint presentation.
