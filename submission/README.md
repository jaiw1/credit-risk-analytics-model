# Credit Risk Modelling AI Project

This repository is an end-to-end credit risk modelling project for a retail two-wheeler loan pool. It starts from the raw workbook, creates a clean analytical dataset, trains probability of default (PD) models, estimates loss given default (LGD), simulates portfolio loss distribution and cashflows, models segment/macro correlation, scores the current analysis pool, and consolidates all outputs into Excel and PowerPoint deliverables.

The project is designed around the requirements in `Detailed Brief for Credit Risk Modelling AI Project.pdf`:

- Clean and document the raw data.
- Build a baseline statistical PD model and an AI/ML PD model.
- Validate models, estimate PD for every loan, recommend a model, and explain risk drivers.
- Estimate LGD per loan using collateral characteristics and stress scenarios.
- Compute expected loss and simulate portfolio loss distribution.
- Estimate tail risk and default correlation.
- Create borrower segments and link pool risk to external variables.
- Simulate default timing, prepayment, and cashflows.
- Submit clean datasets, model outputs, Excel/CSV files, and a PowerPoint presentation.

## Executive Summary

The final implementation now uses two PD views:

1. `30+ DPD stress PD` as the headline operating target.
   This is used for the main downstream EL, LGD, portfolio loss, cashflow, and analysis-pool scoring workflow because the event count is large enough for stable model validation.

2. `90+ DPD hard-default PD` as a separate rare-event extension.
   This satisfies the strict hard-default definition and shows that 90+ DPD can be modelled, but the dataset has only 95 such cases, so it should be presented as a tail-surveillance model rather than the only portfolio calibration model.

The main pipeline now runs `src/03b_pd_model_xgboost_optuna.py` instead of `src/03_pd_model.py` because the XGBoost + Optuna model is the better headline model. In the refreshed run, it outperformed the baseline logistic regression and neural network on test AUC, KS, and Brier score.

Key refreshed outputs:

| Area | Result | Interpretation |
|---|---:|---|
| Historical modelling base | 34,390 loans | Clean historical loan-level dataset |
| Historical exposure | INR 2.492 billion | Total POS used as EAD for historical portfolio |
| 30+ DPD events | 1,505 loans, 4.38% | Suitable for stable stress-PD modelling |
| 90+ DPD events | 95 loans, 0.276% | Rare event; modelled separately |
| Selected stress-PD model | XGBoost + Optuna | Best validation among tested headline models |
| Stress-PD test AUC | 0.781 | Good rank-ordering for retail credit risk |
| 90+ DPD extension test AUC | 0.836 | Useful tail ranking despite sparse event base |
| Analysis pool loans | 3,221 loans | Current clean pool to score |
| Analysis pool exposure | INR 225.51 million | EAD for clean pool |
| Analysis pool mean stress PD | 7.25% | Higher than historical mean stress PD |
| Analysis pool base expected loss | INR 3.27 million | 1.45% of analysis-pool exposure |
| Historical base expected loss | INR 21.76 million | 0.8733% of historical EAD under 20% LGD floor |
| Historical base 99% VaR | INR 84.98 million | Tail loss under Monte Carlo simulation |

## Why 30+ DPD and 90+ DPD Are Both Included

The historical data has a very imbalanced strict default target:

- `is_stressed = 1` when `Peak DPD >= 30`: 1,505 events.
- `is_default = 1` when `Peak DPD >= 90`: 95 events.

Using only the 90+ DPD target would make the model statistically fragile because only 0.276% of the dataset is a positive hard-default case. The model can still be built using rare-event techniques, but validation and calibration uncertainty are much higher.

Therefore the project uses this governance stance:

- Use `30+ DPD` as the operational stress-PD model for portfolio EL and management reporting.
- Use `90+ DPD` as a strict hard-default sensitivity and tail-surveillance extension.
- Present both in the final PPT so the project is transparent rather than evasive about target-definition risk.

The 90+ DPD extension uses:

- Class-weighted logistic regression.
- Balanced random forest.
- XGBoost with `scale_pos_weight`.
- Probability calibration using `CalibratedClassifierCV`.
- PR-AUC, ROC-AUC, Brier score, and top-decile capture validation.

The selected hard-default model is rare-event logistic regression. It achieves test AUC 0.836 and captures 73.7% of test hard defaults in the top risk decile, but PR-AUC is low at 0.0566 because the underlying event is very rare.

## Repository Structure

```text
financial_prediction_model/
+-- main.py
+-- README.md
+-- requirements.txt
+-- Credit Risk Modelling_Sample Data.xlsb
+-- Credit Risk Modelling_Sample Data.xls
+-- Detailed Brief for Credit Risk Modelling AI Project.pdf
+-- data/
|   +-- india_macro_quarterly.csv
+-- docs/
|   +-- PROJECT_TASKS_DOCUMENTATION.md
|   +-- 01_data_cleaning_documentation.md
+-- src/
|   +-- 01_data_cleaning.py
|   +-- 02_eda.py
|   +-- 03_pd_model.py
|   +-- 03b_pd_model_xgboost_optuna.py
|   +-- 03c_pd_model_hard_default.py
|   +-- 04_lgd_model.py
|   +-- 05_portfolio_loss.py
|   +-- 06_correlation_model.py
|   +-- 07_cashflow_simulation.py
|   +-- 08_score_analysis_pool.py
|   +-- utils.py
+-- outputs/
    +-- credit_risk_model_outputs.xlsx
    +-- credit_risk_modelling_ai_project_presentation.pptx
    +-- cleaned_data.csv
    +-- model_outputs.csv
    +-- hard_default_model_outputs.csv
    +-- portfolio_report.csv
    +-- analysis_pool_scored.csv
    +-- cashflow_projection.csv
    +-- macro_correlation_results.csv
    +-- figures/
    +-- models/
```

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the complete pipeline:

```bash
python3 main.py
```

Resume from a specific module:

```bash
python3 main.py --from 4
```

Run the selected main PD model only:

```bash
python3 src/03b_pd_model_xgboost_optuna.py
```

Run the 90+ DPD rare-event extension only:

```bash
python3 src/03c_pd_model_hard_default.py
```

Run analysis-pool scoring only after models are trained:

```bash
python3 src/08_score_analysis_pool.py
```

Final deliverables:

- `outputs/credit_risk_model_outputs.xlsx`
- `outputs/credit_risk_modelling_ai_project_presentation.pptx`

## Main Pipeline

`main.py` runs these modules in order:

| Module | Script | Purpose |
|---:|---|---|
| 1 | `src/01_data_cleaning.py` | Clean raw workbook and create modelling dataset |
| 2 | `src/02_eda.py` | Exploratory visual analysis |
| 3 | `src/03b_pd_model_xgboost_optuna.py` | Main 30+ DPD stress-PD model |
| 3.5 | `src/03c_pd_model_hard_default.py` | 90+ DPD hard-default extension |
| 4 | `src/04_lgd_model.py` | LGD estimation and scenario analysis |
| 5 | `src/05_portfolio_loss.py` | Expected loss, Monte Carlo loss distribution, VaR, CVaR |
| 6 | `src/06_correlation_model.py` | Segment correlation, clustering, macro correlation |
| 7 | `src/07_cashflow_simulation.py` | PD curve and monthly cashflow simulation |
| 8 | `src/08_score_analysis_pool.py` | Score the current clean analysis pool |
| 9 | `main.py::generate_excel_output()` | Consolidate outputs into Excel |

The Excel workbook now includes a dedicated `Hard Default 90DPD PD` sheet generated from the 90+ DPD extension.

## Module 1: Data Cleaning and Preparation

Script: `src/01_data_cleaning.py`

Input:

- Raw workbook: `Credit Risk Modelling_Sample Data.xlsb`
- Sheet: `Historical Default Data`

Main actions:

- Loads the historical loan-level data with the correct workbook header row.
- Drops fully empty/unusable fields, including `Address1`.
- Renames raw Excel columns into consistent snake_case names.
- Standardizes business-source casing, for example normalizing branch-source labels.
- Converts Excel serial date columns into Python datetime columns:
  - `date_of_birth`
  - `disbursement_date`
  - `maturity_date`
  - `first_payment_date`
  - `cut_off_date`
- Cleans numeric columns such as bureau score and LTV.
- Treats negative bureau score values as missing/no-bureau-history cases.
- Imputes missing age from date of birth.
- Extracts `vehicle_type` from product details.
- Validates sourced-state versus borrower-state information.
- Creates the two target flags:
  - `is_stressed = 1` when `peak_dpd >= 30`
  - `is_default = 1` when `peak_dpd >= 90`
- Engineers model features:
  - `loan_to_income_ratio`
  - `emi_to_income_ratio`
  - `elapsed_ratio`
  - `bureau_score_band`
  - `age_band`
  - `income_band`
  - `state_group`
  - `loan_tenure_months`
  - `seasoning_months`

Outputs:

- `outputs/cleaned_data.csv`
- `outputs/cleaning_log.txt`

Refreshed output summary:

- Rows: 34,390
- Columns: 59
- Historical Total POS / EAD: INR 2,491,738,956
- 30+ DPD stressed loans: 1,505, or 4.38%
- 90+ DPD hard-default loans: 95, or 0.276%
- Critical EAD nulls after cleaning: 0

Inference:

The cleaned file is a usable modelling base. The event-rate split is the central modelling fact: 30+ DPD has enough observations for a stable PD model, while 90+ DPD is too sparse to carry the whole downstream portfolio framework by itself.

## Module 2: Exploratory Data Analysis

Script: `src/02_eda.py`

Input:

- `outputs/cleaned_data.csv`

Main actions:

- Creates distribution charts for core numeric fields.
- Plots peak DPD distribution and delinquent DPD distribution.
- Reviews default/stress rate by:
  - vehicle manufacturer
  - bureau-score band
  - age band
  - income band
  - business source
  - geography
- Creates a numeric correlation heatmap.
- Creates risk segmentation heatmaps using bureau score and income bands.
- Reviews LTV behaviour.

Important figures:

- `outputs/figures/dpd_distribution.png`
- `outputs/figures/risk_segmentation_heatmap.png`
- `outputs/figures/correlation_heatmap.png`
- `outputs/figures/geographic_default_rates.png`
- `outputs/figures/default_by_bureau_score.png`
- `outputs/figures/default_by_income.png`
- `outputs/figures/default_by_manufacturer.png`
- `outputs/figures/ltv_analysis.png`

Inference:

EDA confirms that the DPD distribution is heavily concentrated at low delinquency values with a small hard-default tail. That supports the modelling choice to use a stable 30+ stress target for the main model and a separate 90+ extension for strict defaults.

## Module 3: Main PD Model, XGBoost + Optuna

Script: `src/03b_pd_model_xgboost_optuna.py`

This is now the main PD module used by `main.py`.

Target:

- `is_stressed`, meaning `Peak DPD >= 30`.

Why this target is used for the main model:

- It has 1,505 positive cases.
- It gives a more stable train/test split.
- It is suitable for portfolio EL, cashflow, and management monitoring.
- It still captures meaningful early delinquency and stress behaviour.

Features used:

- Numeric:
  - `bureau_score`
  - `ltv`
  - `loan_to_income_ratio`
  - `emi_to_income_ratio`
  - `interest_rate`
  - `monthly_income`
  - `age`
  - `no_of_installments`
  - `sanctioned_loan_amount`
  - `property_value`
- Categorical:
  - `gender`
  - `manufacturer`
  - `state_group`
  - `insurance_status`
  - `reg_info_status`
  - `business_sourced_by`
  - `vehicle_type`

Leakage control:

- The model does not use `peak_dpd`, `current_dpd`, `overdue_amount`, or other direct outcome fields as predictors.
- Target labels are created from DPD, but DPD itself is excluded from the PD feature set.

Models trained:

- Logistic regression baseline.
- XGBoost with Optuna hyperparameter tuning and SMOTE.
- Neural network / MLP.

Techniques used:

- WOE encoding for categorical features.
- SMOTE for minority class support during XGBoost training.
- Optuna hyperparameter optimization.
- Probability calibration with `CalibratedClassifierCV`.
- ROC-AUC, KS, Brier score, precision-recall, and decile validation.
- SHAP analysis when available, with feature-importance fallback.

Validation summary:

| Model | Test AUC | KS | Brier | Decision |
|---|---:|---:|---:|---|
| Logistic regression | 0.760 | 0.381 | 0.0396 | Baseline |
| XGBoost + Optuna | 0.781 | 0.432 | 0.0391 | Selected |
| Neural network | 0.729 | 0.335 | 0.0402 | Not selected |

PD output summary:

- Mean stress PD: 4.36%
- Median stress PD: 2.98%
- Minimum stress PD: 0.00%
- Maximum stress PD: 49.49%
- Observed 30+ DPD rate: 4.38%

PD band distribution:

| Band | Loans |
|---|---:|
| AAA (<1%) | 2,804 |
| AA (1-3%) | 14,445 |
| A (3-5%) | 8,113 |
| BBB (5-10%) | 6,129 |
| BB (10-20%) | 2,066 |
| B+ (>20%) | 833 |

Outputs:

- `outputs/model_outputs.csv`
- `outputs/cleaned_data_with_pd.csv`
- `outputs/models/pd_best_model.pkl`
- `outputs/models/pd_best_scaler.pkl`
- `outputs/models/label_encoders.pkl`
- `outputs/figures/pd_model_comparison.png`
- `outputs/figures/calibration_curves.png`
- `outputs/figures/decile_validation.png`
- `outputs/figures/pd_distribution.png`
- `outputs/figures/shap_summary.png`
- `outputs/figures/shap_importance_bar.png`
- `outputs/figures/feature_importance.png`

Inference:

XGBoost + Optuna is the best operational stress-PD model because it has the strongest test discrimination and the best Brier score among the tested models. The mean predicted stress PD is also close to the observed 30+ DPD rate, which supports using the calibrated output for expected-loss calculations.

## Module 3.5: 90+ DPD Hard-Default Extension

Script: `src/03c_pd_model_hard_default.py`

Target:

- `is_default`, meaning `Peak DPD >= 90`.

Purpose:

- Adds the strict 90+ DPD default model requested for the project.
- Does not replace the 30+ DPD stress model.
- Gives a rare-event hard-default view for sensitivity, surveillance, and presentation completeness.

Why rare-event treatment is required:

- Only 95 of 34,390 loans reached 90+ DPD.
- The positive class rate is 0.276%.
- A normal model can appear accurate by predicting almost everything as non-default, so validation must include rare-event metrics.

Models trained:

- Rare-event logistic regression with class weighting.
- Balanced random forest.
- XGBoost rare-event model using `scale_pos_weight`.

Validation metrics:

| Model | Train AUC | Test AUC | PR-AUC | Brier | CV AUC |
|---|---:|---:|---:|---:|---:|
| Rare Event Logistic | 0.893 | 0.836 | 0.0566 | 0.002711 | 0.852 +/- 0.027 |
| Balanced Random Forest | 0.993 | 0.828 | 0.0433 | 0.002725 | 0.807 +/- 0.063 |
| XGBoost Rare Event | 0.988 | 0.815 | 0.0300 | 0.002734 | 0.785 +/- 0.053 |

Selected model:

- Rare-event logistic regression.

Hard-default PD output:

- Mean hard-default PD: about 0.275%
- Observed hard-default rate: about 0.276%
- Maximum hard-default PD: about 14.7%
- Top test decile captured 14 of 19 hard defaults, or 73.7%.

Hard-default band distribution:

| Band | Loans |
|---|---:|
| A (<0.10%) | 12,695 |
| B (0.10-0.30%) | 12,656 |
| C (0.30-0.50%) | 4,169 |
| D (0.50-1.00%) | 3,231 |
| E (1.00-2.00%) | 1,299 |
| F (>2.00%) | 340 |

Outputs:

- `outputs/hard_default_model_outputs.csv`
- `outputs/cleaned_data_with_hard_default_pd.csv`
- `outputs/hard_default_model_metrics.csv`
- `outputs/models/pd_hard_default_90dpd_best_model.pkl`
- `outputs/models/pd_hard_default_90dpd_metadata.pkl`
- `outputs/figures/hard_default_model_comparison.png`
- `outputs/figures/hard_default_decile_validation.png`
- `outputs/figures/hard_default_pd_distribution.png`

Inference:

The 90+ DPD model can be built and has useful rank-ordering power, but it should not be overclaimed as the only portfolio PD model. The correct presentation is: 30+ DPD is the stable operational stress-PD model; 90+ DPD is an additional hard-default extension for tail monitoring.

## Module 4: LGD Estimation

Script: `src/04_lgd_model.py`

Input:

- `outputs/cleaned_data_with_pd.csv`

Purpose:

- Estimate loss given default per loan.
- Use collateral characteristics rather than a flat assumption only.
- Present both a mathematical collateral estimate and conservative scenario/floor views.

Core LGD logic:

1. Start from `property_value`.
2. Apply depreciation based on vehicle age and expected recovery delay.
3. Apply collateral haircut under stress scenarios.
4. Deduct recovery costs.
5. Cap recovery rate between 0% and 100%.
6. Compute `LGD = 1 - recovery_rate`.
7. Apply floors for reporting/stress scenarios.

LGD scenarios:

| Scenario | Mean LGD | Logic |
|---|---:|---|
| Pure Collateral Estimate | 0.58% | Mathematical collateral recovery only |
| Base Case 20% Floor | 20.01% | Conservative headline reporting case |
| Moderate Stress | 35.62% | Recovery delay, 20% haircut, 35% floor |
| Severe Stress | 56.82% | Longer delay, 40% haircut, 50% floor |
| RBI Norms 65% | 65.00% | Prudential severe floor |

Outputs:

- `outputs/cleaned_data_with_lgd.csv`
- `outputs/figures/lgd_scenarios.png`
- `outputs/figures/lgd_scenario_comparison.png`
- `outputs/figures/lgd_by_manufacturer.png`

Inference:

The pure collateral model produces very low average LGD because many two-wheeler loans appear well collateralized. For a credible credit-risk presentation, the project therefore reports both the pure collateral estimate and conservative floored scenarios. The base 20% LGD floor is used for headline expected-loss numbers.

## Module 5: Portfolio Loss Distribution

Script: `src/05_portfolio_loss.py`

Input:

- `outputs/cleaned_data_with_lgd.csv`

Purpose:

- Compute expected loss by loan and scenario.
- Simulate full portfolio losses using Monte Carlo.
- Estimate tail risk, VaR, CVaR / expected shortfall, and economic capital.

Formula:

```text
Expected Loss = PD x LGD x EAD
```

Simulation method:

- Uses a Vasicek-style one-factor setup.
- Applies a portfolio asset correlation assumption of `rho = 0.15`.
- Runs 50,000 simulations.
- Produces loss distributions and exceedance curves.

Portfolio results:

| Scenario | EAD | Avg LGD | EL | EL / EAD | 99% VaR | 99% Economic Capital |
|---|---:|---:|---:|---:|---:|---:|
| Pure Collateral Estimate | INR 2.492B | 0.58% | INR 0.81M | 0.0325% | INR 3.36M | INR 2.55M |
| Base Case 20% Floor | INR 2.492B | 20.01% | INR 21.76M | 0.8733% | INR 84.98M | INR 63.22M |
| Moderate Stress | INR 2.492B | 35.62% | INR 38.94M | 1.5629% | INR 153.73M | INR 114.78M |
| Severe Stress | INR 2.492B | 56.82% | INR 63.27M | 2.5393% | INR 249.03M | INR 185.76M |
| RBI Norms 65% | INR 2.492B | 65.00% | INR 70.68M | 2.8366% | INR 269.87M | INR 199.19M |

Outputs:

- `outputs/cleaned_data_with_el.csv`
- `outputs/portfolio_report.csv`
- `outputs/figures/portfolio_loss_distribution_base_case_20_floor.png`
- `outputs/figures/loss_exceedance_curve_base_case_20_floor.png`
- Additional scenario loss-distribution and exceedance plots.

Inference:

Expected loss is modest relative to EAD under the base case, but tail risk is much larger. The gap between EL and 99% VaR is the capital buffer needed for adverse portfolio states. LGD conservatism drives the biggest movement in rupee loss.

## Module 6: Correlation, Segmentation, and Macro Evidence

Script: `src/06_correlation_model.py`

Input:

- `outputs/cleaned_data_with_el.csv`
- `data/india_macro_quarterly.csv`

Purpose:

- Estimate correlated risk across borrower segments.
- Build risk clusters.
- Link vintage stress rates to actual time-aligned macro variables.

Segment correlation:

- The module estimates Vasicek-style asset correlation using observed variation across segments.
- Geography/state grouping is the strongest observed concentration signal.
- Manufacturer, income band, and bureau band show weaker observed concentration in this sample.

Observed segment rho summary:

| Segment | Estimated rho | Interpretation |
|---|---:|---|
| State group | 0.0480 | Largest observed concentration signal |
| Manufacturer | 0.0051 | Weak segment dependence |
| Income band | 0.0010 | Minimal observed effect |
| Bureau band | 0.0010 | Minimal observed effect |

Clustering:

- K-means is used for borrower risk segmentation.
- The selected number of clusters is 2.
- Cluster outputs are saved into `outputs/cleaned_data_with_clusters.csv`.

Macro correlation:

- Earlier synthetic macro assumptions have been replaced with actual time-aligned macro data from `data/india_macro_quarterly.csv`.
- The model groups loans by disbursement quarter, filters thin vintages, and correlates remaining vintage stress rates with macro variables.

Macro results:

| Macro variable | Correlation r | p-value | Valid vintages |
|---|---:|---:|---:|
| CPI inflation YoY | 0.535 | 0.353 | 5 |
| Real GDP QoQ | -0.893 | 0.041 | 5 |
| Policy rate | 0.647 | 0.238 | 5 |
| Share-price QoQ return | 0.356 | 0.557 | 5 |

Outputs:

- `outputs/cleaned_data_with_clusters.csv`
- `outputs/macro_vintage_correlation_dataset.csv`
- `outputs/macro_correlation_results.csv`
- `outputs/figures/cluster_selection.png`
- `outputs/figures/cluster_profiles.png`
- `outputs/figures/pd_vs_lgd_clusters.png`
- `outputs/figures/macro_correlation.png`

Inference:

The macro relationship is now empirically defensible because it uses actual time-aligned data. However, only five valid vintages remain after filtering, so the macro output should be presented as directional stress evidence rather than a fully reliable econometric model. The direction of GDP is intuitive: weaker GDP growth is associated with higher stress.

## Module 7: PD Curve and Cashflow Simulation

Script: `src/07_cashflow_simulation.py`

Input:

- `outputs/cleaned_data_with_clusters.csv`

Purpose:

- Convert lifetime stress PD into a monthly default-timing curve.
- Simulate monthly collections, prepayments, defaults, recoveries, net loss, and present value.

Main assumptions:

- Lifetime PD comes from the calibrated stress-PD model.
- Monthly default timing uses a hazard curve.
- Constant prepayment rate is applied.
- Recoveries are included after defaults.
- Discounting is used to compute present value of collections.

Base cashflow results:

| Metric | Value |
|---|---:|
| Projection horizon | 28 months |
| Initial balance | INR 2.492B |
| Scheduled principal | INR 2.278B |
| Scheduled interest | INR 506.56M |
| Prepayments | INR 153.13M |
| Defaults | INR 60.69M |
| Recoveries | INR 23.71M |
| Net loss | INR 36.99M |
| Total collections | INR 2.961B |
| PV collections | INR 2.714B |
| NPV / initial balance | 108.94% |

Outputs:

- `outputs/cashflow_projection.csv`
- `outputs/pd_term_structure.csv`
- `outputs/figures/cashflow_waterfall.png`
- `outputs/figures/cashflow_stress_scenarios.png`

Inference:

The pool remains collection-positive in the base case because scheduled principal, interest, prepayments, and recoveries offset projected defaults. The cashflow result should be presented as a treasury-level portfolio projection, not a loan-by-loan servicing engine.

## Module 8: Score Loan Pool for Analysis

Script: `src/08_score_analysis_pool.py`

Input:

- Raw workbook: `Credit Risk Modelling_Sample Data.xlsb`
- Sheet: `Loan Pool for Analysis`
- Trained model artifacts from Module 3.

Purpose:

- Clean the current loan pool using the same cleaning logic as Module 1.
- Apply the selected main stress-PD model from Module 3.
- Estimate LGD using the same LGD logic as Module 4.
- Compute expected loss for each current clean loan.

Important point:

- The current analysis pool has no observed delinquency in the input. That is expected. The project is scoring currently clean loans using the trained historical stress-PD model.

Output summary:

| Metric | Value |
|---|---:|
| Analysis pool loans | 3,221 |
| Exposure at default | INR 225.51M |
| Mean stress PD | 7.25% |
| Maximum stress PD | 49.49% |
| Mean base LGD | 20.01% |
| Expected loss | INR 3.27M |
| EL / exposure | 1.45% |

Outputs:

- `outputs/analysis_pool_scored.csv`

Columns in the scored pool:

- `loan_account_number`
- `sanctioned_loan_amount`
- `exposure_at_default`
- `predicted_pd`
- `pure_collateral_lgd`
- `predicted_lgd`
- `expected_loss`

Inference:

The clean analysis pool has a higher mean stress PD than the historical portfolio average. Its base expected loss is INR 3.27M under the selected stress-PD model and conservative 20% LGD floor. This is the most management-ready number for the new pool.

## Module 9: Excel Output

Function: `generate_excel_output()` inside `main.py`

Purpose:

- Consolidate the most important outputs into one workbook for submission and review.

Output:

- `outputs/credit_risk_model_outputs.xlsx`

Workbook sheets:

- `PD Per Loan`
- `Portfolio Risk Summary`
- `Hard Default 90DPD PD`
- `LGD Per Loan`
- `Cashflow Projection`
- `PD Term Structure`
- `Scored Analysis Pool`

Inference:

This workbook is the main tabular deliverable. It allows a reviewer to inspect per-loan PD, strict 90+ DPD PD, LGD, portfolio loss, cashflow, and analysis-pool scoring without running the code.

## PowerPoint Deliverable

Final deck:

- `outputs/credit_risk_modelling_ai_project_presentation.pptx`

The PPT has 33 slides and covers:

- Project objective and brief mapping.
- Data estate and event-rate challenge.
- Full workflow.
- Data cleaning and EDA.
- Main stress-PD model and validation.
- 90+ DPD rare-event extension.
- LGD scenarios.
- Portfolio expected loss and tail risk.
- Correlation, clustering, macro evidence.
- Cashflow simulation.
- Analysis-pool scoring.
- Output inventory.
- Governance, limitations, and recommended talk track.
- Module map, formulas, and run commands.

The deck was rendered and checked with the presentation build workflow. The final layout check has 0 errors.

## Output Inventory

Core tabular outputs:

| File | What it contains |
|---|---|
| `outputs/cleaned_data.csv` | Clean historical modelling dataset |
| `outputs/cleaning_log.txt` | Cleaning actions and data-quality decisions |
| `outputs/model_outputs.csv` | Per-loan 30+ DPD stress PDs from Module 3 |
| `outputs/cleaned_data_with_pd.csv` | Full clean dataset with stress PDs |
| `outputs/hard_default_model_outputs.csv` | Per-loan 90+ DPD hard-default PDs |
| `outputs/hard_default_model_metrics.csv` | Rare-event model validation table |
| `outputs/cleaned_data_with_hard_default_pd.csv` | Full clean dataset with hard-default PD |
| `outputs/cleaned_data_with_lgd.csv` | Dataset with LGD scenario columns |
| `outputs/cleaned_data_with_el.csv` | Dataset with expected-loss calculations |
| `outputs/portfolio_report.csv` | EL, VaR, CVaR, and economic capital by scenario |
| `outputs/cleaned_data_with_clusters.csv` | Dataset with cluster assignments |
| `outputs/macro_correlation_results.csv` | Actual macro correlation results |
| `outputs/macro_vintage_correlation_dataset.csv` | Time-aligned vintage/macro dataset |
| `outputs/cashflow_projection.csv` | Monthly base-case cashflow projection |
| `outputs/pd_term_structure.csv` | Monthly marginal/cumulative PD curve |
| `outputs/analysis_pool_scored.csv` | Scored current clean pool |
| `outputs/credit_risk_model_outputs.xlsx` | Consolidated Excel workbook |
| `outputs/credit_risk_modelling_ai_project_presentation.pptx` | Final PowerPoint deck |

Model artifacts:

| File | Purpose |
|---|---|
| `outputs/models/pd_best_model.pkl` | Selected main stress-PD model |
| `outputs/models/pd_best_scaler.pkl` | Scaler placeholder or fitted scaler for selected model |
| `outputs/models/label_encoders.pkl` | Encoders/WOE transformer used for scoring |
| `outputs/models/pd_logistic_regression.pkl` | Logistic baseline |
| `outputs/models/pd_xgboost.pkl` | XGBoost model |
| `outputs/models/pd_neural_network.pkl` | Neural network model |
| `outputs/models/pd_hard_default_90dpd_best_model.pkl` | Selected 90+ DPD rare-event model |
| `outputs/models/pd_hard_default_90dpd_metadata.pkl` | Metadata for 90+ DPD model |

Important figures:

| Figure | Purpose |
|---|---|
| `outputs/figures/dpd_distribution.png` | Shows why 30+ and 90+ targets need separate treatment |
| `outputs/figures/risk_segmentation_heatmap.png` | EDA risk heatmap |
| `outputs/figures/pd_model_comparison.png` | PD model validation comparison |
| `outputs/figures/calibration_curves.png` | Probability calibration |
| `outputs/figures/decile_validation.png` | Risk rank-ordering by decile |
| `outputs/figures/pd_distribution.png` | Stress-PD distribution |
| `outputs/figures/hard_default_model_comparison.png` | Rare-event model comparison |
| `outputs/figures/hard_default_decile_validation.png` | 90+ DPD top-bucket capture |
| `outputs/figures/hard_default_pd_distribution.png` | Hard-default PD distribution |
| `outputs/figures/lgd_scenario_comparison.png` | LGD scenario comparison |
| `outputs/figures/portfolio_loss_distribution_base_case_20_floor.png` | Base loss distribution |
| `outputs/figures/loss_exceedance_curve_base_case_20_floor.png` | Tail loss exceedance |
| `outputs/figures/cluster_profiles.png` | Borrower segment profiles |
| `outputs/figures/macro_correlation.png` | Actual macro correlation evidence |
| `outputs/figures/cashflow_waterfall.png` | Base cashflow projection |
| `outputs/figures/cashflow_stress_scenarios.png` | Cashflow stress comparison |

## Assumptions and Limitations

Target definition:

- 30+ DPD is an early delinquency / stress target, not a strict legal default target.
- 90+ DPD is the stricter hard-default target, but only 95 historical observations meet it.
- The project addresses this by presenting both, with clear usage boundaries.

LGD:

- LGD is modelled from collateral and assumptions.
- Actual repossession, sale, and recovery cashflow data is not available.
- The project therefore shows both pure collateral recovery and conservative floored/stressed LGD scenarios.

Macro correlation:

- Actual macro data is used, not synthetic data.
- Only five valid time-aligned vintages remain after filtering thin quarters.
- The macro output is directionally useful but should not be presented as a high-confidence causal model.

Cashflow:

- The cashflow module is a portfolio-level projection.
- It uses aggregate assumptions for default timing, prepayment, recoveries, and discounting.
- It should not be described as a detailed loan servicing or collections engine.

Analysis pool:

- The analysis pool is currently clean with no observed DPD/default outcome.
- The scored PDs are model projections based on historical patterns.

## Presentation Talking Points

Use this narrative in the final presentation:

1. The project is not just a model score; it is a full credit risk analytics pipeline.
2. The target-definition choice is deliberate. 30+ DPD is used for stable operational stress PD; 90+ DPD is separately modelled as a strict rare-event extension.
3. XGBoost + Optuna is selected because it performs best on validation, not because it is the most complex model.
4. The 90+ DPD model is useful for tail ranking, but the low event count means it should not be overclaimed.
5. LGD is shown in two layers: transparent collateral mechanics and conservative reporting floors.
6. Portfolio risk is reported as both expected loss and tail loss, because VaR/economic capital answer a different question than EL.
7. Macro correlation is empirical and time-aligned, but sample size is a limitation.
8. The analysis pool result is the management-ready output: INR 3.27M expected loss on INR 225.51M exposure.

## Verification Performed

The refreshed project run completed successfully:

- `python3 -m py_compile main.py src/*.py`
- `python3 src/03b_pd_model_xgboost_optuna.py`
- `python3 src/03c_pd_model_hard_default.py`
- `python3 main.py --from 4`
- PowerPoint build generated `outputs/credit_risk_modelling_ai_project_presentation.pptx`
- PowerPoint layout check completed with 0 errors

The downstream run from Module 4 regenerated LGD, portfolio loss, correlation, cashflow, analysis-pool scoring, and Excel output using the selected XGBoost + Optuna stress-PD model and the new 90+ DPD extension.
