# Credit Risk Modelling AI Project — Walkthrough

## Project Summary

Built an end-to-end credit risk analytics pipeline for a **34,390 two-wheeler loan pool** using Python. The pipeline covers all 6 deliverables from the project brief across 7 executable modules.

---

## Pipeline Results

### Module 1: Data Cleaning ✅
**File:** [01_data_cleaning.py](file:///Users/jaiwadhwa/Desktop/Jai/personal/financial_prediction_model/src/01_data_cleaning.py)

- Processed 34,390 loans across 19 cleaning steps
- Converted Excel serial dates to datetime
- Standardized categorical values (business source casing)
- Engineered 10+ new features (LTV bands, income ratios, bureau score bands, etc.)
- Created default flags: **is_default (DPD≥90): 95 loans (0.28%)** and **is_stressed (DPD≥30): 1,505 loans (4.38%)**
- Output: [cleaned_data.csv](file:///Users/jaiwadhwa/Desktop/Jai/personal/financial_prediction_model/outputs/cleaned_data.csv) (55 columns) + [cleaning_log.txt](file:///Users/jaiwadhwa/Desktop/Jai/personal/financial_prediction_model/outputs/cleaning_log.txt)

---

### Module 2: Exploratory Data Analysis ✅
**File:** [02_eda.py](file:///Users/jaiwadhwa/Desktop/Jai/personal/financial_prediction_model/src/02_eda.py)

Generated 12 visualizations covering default rates by category, distributions, DPD analysis, correlation heatmap, geographic patterns, risk segmentation, and LTV analysis.

**Key Finding:** Rajasthan has the highest stress rate (13.9%) despite being 5th by volume. Bureau score has limited discriminatory power for this pool.

![Risk Segmentation Heatmap](/Users/jaiwadhwa/.gemini/antigravity/brain/0df59837-6e45-4c25-9cc8-dd0f8dadf8e5/risk_segmentation_heatmap.png)

---

### Module 3: PD Modelling ✅
**File:** [03_pd_model.py](file:///Users/jaiwadhwa/Desktop/Jai/personal/financial_prediction_model/src/03_pd_model.py)

Built two models targeting **is_stressed (DPD≥30)** for better class balance:

| Metric | Logistic Regression | Random Forest ★ |
|--------|:------------------:|:--------------:|
| **Test AUC** | 0.7047 | **0.7714** |
| **Gini** | 0.4094 | **0.5429** |
| **KS Statistic** | 0.3309 | **0.4113** |
| **CV AUC** | 0.6847 ± 0.012 | **0.7380 ± 0.022** |

**★ Recommended: Random Forest** (higher AUC, better discrimination)

**Top Risk Drivers:**
1. Number of installments (19.8%)
2. State group (9.3%)
3. Sanctioned loan amount (7.9%)
4. Interest rate (7.3%)
5. EMI-to-income ratio (7.1%)

![PD Model Comparison](/Users/jaiwadhwa/.gemini/antigravity/brain/0df59837-6e45-4c25-9cc8-dd0f8dadf8e5/pd_model_comparison.png)

![Feature Importance](/Users/jaiwadhwa/.gemini/antigravity/brain/0df59837-6e45-4c25-9cc8-dd0f8dadf8e5/feature_importance.png)

---

### Module 4: LGD Estimation ✅
**File:** [04_lgd_model.py](file:///Users/jaiwadhwa/Desktop/Jai/personal/financial_prediction_model/src/04_lgd_model.py)

Collateral-based LGD estimation with depreciation modelling and three stress scenarios:

| Scenario | Mean LGD | Mean Recovery | Collateral Haircut |
|----------|:--------:|:-------------:|:-----------------:|
| Base Case | **0.1%** | 99.9% | 0% |
| Moderate Stress | **7.0%** | 93.0% | 20% |
| Severe Stress | **30.0%** | 70.0% | 40% |

> [!NOTE]
> The very low base-case LGD reflects strong collateral coverage — most two-wheeler loans have collateral values exceeding the outstanding balance even after depreciation.

![LGD Scenario Comparison](/Users/jaiwadhwa/.gemini/antigravity/brain/0df59837-6e45-4c25-9cc8-dd0f8dadf8e5/lgd_scenario_comparison.png)

---

### Module 5: Portfolio Loss Distribution ✅
**File:** [05_portfolio_loss.py](file:///Users/jaiwadhwa/Desktop/Jai/personal/financial_prediction_model/src/05_portfolio_loss.py)

50,000 Monte Carlo simulations using single-factor Gaussian copula (ρ=0.15):

| Metric | Value | % of EAD |
|--------|------:|:--------:|
| Total Exposure (EAD) | ₹2,491,738,956 | 100% |
| **Expected Loss (EL)** | ₹386,474,248 | 15.5% |
| Unexpected Loss (σ) | ₹151,207,946 | 6.1% |
| **VaR 95%** | ₹659,725,415 | 26.5% |
| **VaR 99%** | ₹782,304,174 | 31.4% |
| CVaR / ES 99% | ₹833,724,543 | 33.5% |
| Economic Capital 99% | ₹395,829,926 | 15.9% |

![Portfolio Loss Distribution](/Users/jaiwadhwa/.gemini/antigravity/brain/0df59837-6e45-4c25-9cc8-dd0f8dadf8e5/portfolio_loss_distribution.png)

---

### Module 6: Default Correlation & Segmentation ✅
**File:** [06_correlation_model.py](file:///Users/jaiwadhwa/Desktop/Jai/personal/financial_prediction_model/src/06_correlation_model.py)

**Asset Correlation (Vasicek model):**
| Segmentation | Estimated ρ |
|--------------|:-----------:|
| Geography (State) | **0.048** (highest) |
| Manufacturer | 0.005 |
| Income Band | 0.001 |
| Bureau Score Band | 0.001 |

**K-Means Segmentation:** Optimal k=2 (Silhouette=0.22)
- **Cluster 0** (24,542 loans): Higher income (₹69K), DR=4.4%
- **Cluster 1** (9,848 loans): Lower income (₹28K), DR=4.2%

> [!TIP]
> Geographic concentration is the strongest correlation driver — Rajasthan's 13.9% stress rate vs Kerala's 2.4% creates meaningful tail risk.

---

### Module 7: Cashflow Simulation ✅
**File:** [07_cashflow_simulation.py](file:///Users/jaiwadhwa/Desktop/Jai/personal/financial_prediction_model/src/07_cashflow_simulation.py)

28-month projection with PD term structure, CPR prepayment modelling, and recovery lag:

| Scenario | Total Defaults | Net Loss | NPV (% of initial) |
|----------|:-----------:|:--------:|:------------------:|
| **Base Case** | ₹398M | ₹239M | **100.1%** |
| Moderate Stress | ₹731M | ₹513M | **88.8%** |
| Severe Stress | ₹999M | ₹802M | **77.1%** |

![Cashflow Stress Scenarios](/Users/jaiwadhwa/.gemini/antigravity/brain/0df59837-6e45-4c25-9cc8-dd0f8dadf8e5/cashflow_stress_scenarios.png)

---

## Output Files

| File | Description |
|------|-------------|
| [cleaned_data.csv](file:///Users/jaiwadhwa/Desktop/Jai/personal/financial_prediction_model/outputs/cleaned_data.csv) | Clean dataset (34,390 × 55) |
| [cleaning_log.txt](file:///Users/jaiwadhwa/Desktop/Jai/personal/financial_prediction_model/outputs/cleaning_log.txt) | 19-step cleaning documentation |
| [model_outputs.csv](file:///Users/jaiwadhwa/Desktop/Jai/personal/financial_prediction_model/outputs/model_outputs.csv) | PD per loan + rating bands |
| [portfolio_report.csv](file:///Users/jaiwadhwa/Desktop/Jai/personal/financial_prediction_model/outputs/portfolio_report.csv) | VaR/CVaR/EC summary |
| [cashflow_projection.csv](file:///Users/jaiwadhwa/Desktop/Jai/personal/financial_prediction_model/outputs/cashflow_projection.csv) | Monthly cashflow waterfall |
| [pd_term_structure.csv](file:///Users/jaiwadhwa/Desktop/Jai/personal/financial_prediction_model/outputs/pd_term_structure.csv) | Marginal PD curve |
| [figures/](file:///Users/jaiwadhwa/Desktop/Jai/personal/financial_prediction_model/outputs/figures) | 24 visualization plots |

## How to Run

```bash
cd /Users/jaiwadhwa/Desktop/Jai/personal/financial_prediction_model

# Run each module in sequence:
python3 src/01_data_cleaning.py    # → outputs/cleaned_data.csv
python3 src/02_eda.py              # → outputs/figures/*.png
python3 src/03_pd_model.py         # → outputs/model_outputs.csv
python3 src/04_lgd_model.py        # → outputs/cleaned_data_with_lgd.csv
python3 src/05_portfolio_loss.py   # → outputs/portfolio_report.csv
python3 src/06_correlation_model.py # → outputs/cleaned_data_with_clusters.csv
python3 src/07_cashflow_simulation.py # → outputs/cashflow_projection.csv
```
