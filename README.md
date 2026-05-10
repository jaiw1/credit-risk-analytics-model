# Credit Risk Modelling AI Project

## Project Summary

This project is an end-to-end credit risk analytics pipeline designed to process, analyze, and predict the credit risk of a two-wheeler loan portfolio. The project spans the entire lifecycle of credit risk modeling, from raw data cleaning to probability of default (PD) modeling, loss given default (LGD) estimation, portfolio loss distribution, and cashflow stress testing. 

The primary goal is to assess historical default data (34,390 loans) to build predictive models, and then apply those models to a new "Loan Pool for Analysis" (3,221 loans) to project their expected losses and understand the overall portfolio risk under different economic scenarios.

---

## What Does a "Successful Prediction" Look Like?

In credit risk modeling, a successful prediction isn't about perfectly guessing exactly which individual person will default. Instead, success is defined by **discrimination** and **calibration**:

1. **Discrimination (ROC-AUC / Gini)**: Can the model rank-order risk? If the model assigns a high Probability of Default (PD) to Loan A and a low PD to Loan B, Loan A should be statistically much more likely to default. We measure this using the ROC-AUC score. An AUC above 0.70 is considered good for retail credit, and above 0.80 is excellent. Our model achieves strong discriminatory power, separating good borrowers from bad ones.
2. **Calibration (Brier Score / Reliability)**: If the model says a group of loans has a 5% PD, we want exactly 5% of those loans to actually default in the real world. A well-calibrated model means the *number* it outputs is a true probability that can be used directly in financial calculations (like Expected Loss). We ensure this using a technique called `CalibratedClassifierCV`.
3. **Financial Coherence (Expected Loss)**: The ultimate test of success is whether the model's outputs make financial sense. When we calculate `Expected Loss = PD × LGD × Exposure`, the total expected loss across the portfolio should align with historical write-off rates. If the model outputs extreme values (e.g., predicting 50% loss on a safe portfolio), it has failed financially even if it succeeds statistically.

---

## Pipeline Overview: Module by Module

### Module 1: Data Cleaning & Preparation (`01_data_cleaning.py`)
**What it does:** This module ingests the raw Excel data (`Credit Risk Modelling_Sample Data.xlsb`) and prepares it for machine learning. Real-world financial data is messy—it has missing values, inconsistent naming, and unformatted dates.
**How it works:** 
- It drops completely empty columns and rows.
- Standardizes categorical variables (e.g., ensuring "business sourced by" matches casing).
- Converts Excel serial dates into usable `datetime` objects.
- Imputes missing values (e.g., filling missing ages with the median age).
- Extracts useful features like "vehicle type" from the product descriptions.
- **Critical Assumption:** It defines a "Default" (`is_default`) as any loan that has hit a Peak DPD (Days Past Due) of 90 or more. This is the standard Basel II definition of default. It also defines a "Stressed" loan as DPD ≥ 30.
**Output:** A clean CSV file ready for modeling (`outputs/cleaned_data.csv`).

---

### Module 2: Exploratory Data Analysis (EDA) (`02_eda.py`)
**What it does:** Generates visualizations to help humans understand the data before modeling.
**What it looks for:** It searches for correlations between variables. For example, do younger borrowers default more? Do certain vehicle manufacturers have higher default rates? How does the Loan-to-Value (LTV) ratio affect stress rates?
**Output:** A series of PNG charts saved in `outputs/figures/`.

---

### Module 3: Probability of Default (PD) Modelling (`03_pd_model.py`)
**What it does:** This is the core machine learning engine. It attempts to predict the likelihood (`is_default`) of a loan going bad based on borrower characteristics.
**What models are used:**
- **Logistic Regression:** A standard, highly interpretable statistical model required by regulators.
- **Gradient Boosting / Random Forest:** Advanced tree-based AI models that find complex, non-linear patterns in the data. These typically perform the best.
- **Neural Network (MLP):** A deep learning model included to fulfill AI innovation requirements.
**Testing & Validation:** 
- The data is split into 80% training and 20% testing. 
- Models are evaluated on the test set using ROC-AUC (for ranking power), Precision/Recall, and Brier Score (for accuracy of probabilities).
- SHAP (SHapley Additive exPlanations) is used to explain *why* the AI made its decisions, ranking the most important risk drivers.
- Because 90+ DPD hard defaults are very sparse in this dataset, the headline model uses 30+ DPD as an early-delinquency/stress PD target. A 90+ DPD model can be built separately with rare-event techniques, but the 30+ target is more stable for validation on this sample.
**Outputs:** The trained models are saved (`outputs/models/*.pkl`) and the historical data is appended with a new `predicted_pd` column.

---

### Module 4: Loss Given Default (LGD) Estimation (`04_lgd_model.py`)
**What it does:** If a borrower defaults, how much money do we actually lose? This module calculates the Recovery Rate (how much we get back by selling the repossessed two-wheeler) and the LGD (`1 - Recovery Rate`).
**Assumptions & Logic:**
- Two-wheelers depreciate rapidly. We assume a 15% annual depreciation rate.
- We present both a pure collateral-recovery estimate and conservative floored LGD scenarios. The base reporting case applies a 20% LGD floor, while moderate/severe stress scenarios add collateral haircuts, longer recovery delays, higher costs, and higher floors.
- If a loan has no collateral, the model assumes a 100% LGD (0% recovery).
**Output:** The data is appended with estimated LGD values.

---

### Module 5: Portfolio Loss Distribution (`05_portfolio_loss.py`)
**What it does:** Simulates what could happen to the entire portfolio under thousands of different macroeconomic realities using Monte Carlo simulation.
**How it works:** It uses the Vasicek Single-Factor model (the foundation of the Basel II IRB framework). It generates 50,000 parallel universes where the economy is either booming or crashing, and calculates the total portfolio loss in each universe.
**Significance:** This tells us our **Value at Risk (VaR)**. If the 99% VaR is ₹50M, it means we are 99% confident that our losses will not exceed ₹50M in a given year. The bank must hold this amount as Economic Capital to survive a severe crash.

---

### Module 6: Default Correlation & Segmentation (`06_correlation_model.py`)
**What it does:** Analyzes how defaults are clustered together. If one loan defaults, does it mean others will too? 
**How it works:** 
- Uses K-Means clustering to segment borrowers into distinct risk profiles.
- Correlates internal loan vintages with actual time-aligned external macroeconomic factors from `data/india_macro_quarterly.csv` instead of synthetic macro defaults. Thin vintages are filtered so the output remains empirically defensible.
**Significance:** Helps the business understand if their risk is too concentrated in a specific borrower segment or highly vulnerable to a specific macroeconomic shock.

---

### Module 7: Cashflow Simulation (`07_cashflow_simulation.py`)
**What it does:** Projects the actual monthly EMI cashflows of the entire portfolio over time.
**How it works:** It uses the PDs to create a "survival curve" for the loans, factoring in a Constant Prepayment Rate (CPR) for people who pay off their loans early. It then simulates the cash waterfall month-by-month.
**Significance:** It shows the treasury department exactly how much cash they can expect to receive next month, next quarter, and next year, allowing them to calculate the Net Present Value (NPV) of the portfolio.

---

### Module 8: Score Analysis Pool (`08_score_analysis_pool.py`)
**What it does:** This is the culmination of the project. It takes the brand new "Loan Pool for Analysis" (~3,200 active loans), cleans them using the rules from Module 1, and applies the models trained in Module 3 and 4 to predict their specific risk.
**Output:** A definitive Expected Loss (EL) calculation for the new pool, calculating `EL = PD × LGD × Exposure`. This provides management with a clear, rupee-value estimate of future losses on the active book.

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (all modules in sequence)
python main.py

# Or run from a specific module (e.g., resume from module 8)
python main.py --from 8

# The final output is consolidated into a master Excel file:
# outputs/credit_risk_model_outputs.xlsx
```
