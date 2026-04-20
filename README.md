# Credit Risk Modelling AI Project

## Project Summary

End-to-end credit risk analytics pipeline for a **34,390 two-wheeler loan pool** using Python. The pipeline covers all 6 deliverables from the project brief across 7 executable modules, with 3 PD models (Logistic Regression, Gradient Boosting, Neural Network), SHAP explainability, Monte Carlo simulation, macro-factor correlation analysis, and cashflow stress testing.

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (all modules in sequence)
python main.py

# Or run from a specific module (e.g., resume from module 3)
python main.py --from 3

# Or run individual modules
python src/01_data_cleaning.py      # → outputs/cleaned_data.csv
python src/02_eda.py                # → outputs/figures/*.png
python src/03_pd_model.py           # → outputs/model_outputs.csv + outputs/models/
python src/04_lgd_model.py          # → outputs/cleaned_data_with_lgd.csv
python src/05_portfolio_loss.py     # → outputs/portfolio_report.csv
python src/06_correlation_model.py  # → outputs/cleaned_data_with_clusters.csv
python src/07_cashflow_simulation.py # → outputs/cashflow_projection.csv
```

---

## Pipeline Overview

### Module 1: Data Cleaning & Preparation
**File:** `src/01_data_cleaning.py`

- Processes 34,390 loans across 19 cleaning steps
- Converts Excel serial dates to datetime
- Standardizes categorical values (business source casing)
- Engineers 10+ derived features (LTV bands, income ratios, bureau score bands, etc.)
- Creates default flags: **is_default (DPD≥90)** and **is_stressed (DPD≥30)**
- Outputs: `outputs/cleaned_data.csv` + `outputs/cleaning_log.txt`

---

### Module 2: Exploratory Data Analysis (EDA)
**File:** `src/02_eda.py`

Generates 12 visualizations covering default rates by category, distributions, DPD analysis, correlation heatmap, geographic patterns, risk segmentation, and LTV analysis.

---

### Module 3: PD Modelling
**File:** `src/03_pd_model.py`

Builds **three models** as required by the project brief:

| Model | Type | Purpose |
|-------|------|---------|
| **Logistic Regression** | Baseline Statistical | Interpretable baseline with odds ratios |
| **Gradient Boosting / Random Forest** | AI/ML Model | Best discriminatory power |
| **Neural Network (MLP)** | Deep Learning | AI innovation requirement |

All models are **probability-calibrated** using `CalibratedClassifierCV` to ensure predicted PDs match observed default rates.

**Validation includes:**
- ROC/AUC, Gini, KS Statistic, Brier Score
- Cross-validation (5-fold stratified)
- Calibration plots (reliability diagrams)
- Decile validation tables (standard credit risk format)
- SHAP explainability (TreeExplainer for risk driver ranking)

**Outputs:** `outputs/model_outputs.csv`, `outputs/models/*.pkl`

---

### Module 4: LGD Estimation
**File:** `src/04_lgd_model.py`

Collateral-based LGD estimation with depreciation modelling (two-wheeler specific) and three stress scenarios:

| Scenario | Collateral Haircut | Recovery Delay | Recovery Costs |
|----------|:-----------------:|:--------------:|:--------------:|
| Base Case | 0% | 6 months | 10% |
| Moderate Stress | 20% | 12 months | 15% |
| Severe Stress | 40% | 18 months | 20% |

---

### Module 5: Portfolio Loss Distribution
**File:** `src/05_portfolio_loss.py`

50,000 Monte Carlo simulations using **single-factor Gaussian copula model** (Vasicek / Basel II IRB framework):
- Expected Loss (EL) = PD × LGD × EAD
- VaR at 95%, 99%, 99.9% confidence levels
- CVaR / Expected Shortfall
- Economic Capital computation
- Loss exceedance curve

---

### Module 6: Default Correlation & Segmentation
**File:** `src/06_correlation_model.py`

- **Part A:** Asset correlation estimation via Vasicek single-factor model
- **Part B:** K-Means borrower segmentation with silhouette analysis
- **Part C:** External macro variable correlation (CPI inflation, GDP growth, unemployment, BSE 500, two-wheeler index) with macro stress scenarios
- **Part D:** Intra vs inter-segment correlation analysis

---

### Module 7: Cashflow Simulation
**File:** `src/07_cashflow_simulation.py`

- PD term structure (marginal PD curve via log-normal hazard model)
- CPR-based prepayment modelling (SMM conversion)
- Monthly cashflow waterfall projection
- NPV computation at 10% discount rate
- Stress scenario comparison (Base / Moderate / Severe)

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/cleaned_data.csv` | Clean dataset (34,390 × 55) |
| `outputs/cleaning_log.txt` | 19-step cleaning documentation |
| `outputs/model_outputs.csv` | PD per loan (all 3 models + recommended + rating bands) |
| `outputs/models/` | Persisted trained models (joblib format) |
| `outputs/portfolio_report.csv` | VaR/CVaR/EC summary |
| `outputs/cashflow_projection.csv` | Monthly cashflow waterfall |
| `outputs/pd_term_structure.csv` | Marginal PD curve |
| `outputs/credit_risk_model_outputs.xlsx` | Consolidated Excel (multi-sheet) |
| `outputs/figures/` | 25+ visualization plots |

## Dependencies

See `requirements.txt`. Key libraries: pandas, numpy, scikit-learn, xgboost, lightgbm, shap, matplotlib, seaborn, scipy, openpyxl.
