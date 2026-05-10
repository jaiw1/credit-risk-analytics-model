# Data Cleaning Pipeline Documentation: `01_data_cleaning.py`

This document provides a detailed explanation of the functions within `src/01_data_cleaning.py`, which serves as the first module in the financial prediction model pipeline. This module is responsible for loading raw financial data, cleaning it, and engineering features for subsequent modelling.

---

## 1. `load_raw_data(log)`
**Purpose:** Loads the raw dataset from an `.xlsb` file.
- **Reference:** Lines 22–32
- **Logic:** It uses `pd.read_excel` with the `pyxlsb` engine. It specifies `header=1` because the raw data has a descriptive row before the actual headers.
- **Logging:** Tracks the initial row count of the raw data.

## 2. `drop_empty_columns(df, log)`
**Purpose:** Removes columns that are mostly empty or contain no useful information.
- **Reference:** Lines 35–53
- **Logic:** 
    - Checks the last column; if it is more than 99.9% NaN, it is dropped (Line 42).
    - Specifically drops the `Address1` column as it is historically 99.9% null in this dataset (Line 49).
- **Logging:** Records the number of columns before and after the cleanup.

## 3. `rename_columns(df, log)`
**Purpose:** Standardizes column names into a clean `snake_case` format.
- **Reference:** Lines 56–63
- **Logic:** It uses a `COLUMN_RENAME_MAP` imported from `utils.py`. It iterates through the dataframe columns and renames only those present in the mapping (Line 60).

## 4. `standardize_business_sourced(df, log)`
**Purpose:** Normalizes inconsistent casing in the `business_sourced_by` categorical column.
- **Reference:** Lines 66–84
- **Logic:** It maps variations like "MFL BRANCH" or "MFL branch" to a unified "MFL Branch" using a dictionary mapping (Line 73–76).

## 5. `convert_dates(df, log)`
**Purpose:** Converts Excel serial date numbers into proper Python datetime objects.
- **Reference:** Lines 86–100
- **Logic:** It applies the `excel_serial_to_date` utility function to specific columns: `date_of_birth`, `disbursement_date`, `maturity_date`, `first_payment_date`, and `cut_off_date`.

## 6. `clean_numeric_columns(df, log)`
**Purpose:** Ensures numeric columns like `bureau_score` and `ltv` (Loan-to-Value) are of the correct float type.
- **Reference:** Lines 103–123
- **Logic:**
    - **Bureau Score:** Replaces negative values (like -1, often indicating "No Score") with `NaN` (Line 111).
    - **LTV:** Converts the column to float using `safe_float`.

## 7. `impute_age(df, log)`
**Purpose:** Fills missing values in the `age` column by calculating it from the `date_of_birth` (DOB).
- **Reference:** Lines 126–146
- **Logic:** If `age` is missing but `date_of_birth` is available, it calculates the age relative to the `cut_off_date` (Line 138–139). It uses `365.25` days per year to account for leap years.

## 8. `parse_product_detail(df, log)`
**Purpose:** Extracts the vehicle type (Scooter or Motorcycle) from a concatenated string in the `product_detail` column.
- **Reference:** Lines 149–177
- **Logic:** The `product_detail` usually follows a `MODEL_BRAND_Type` format. The function splits the string by underscores and classifies the last part into "Scooter", "Motorcycle", or the raw value if it doesn't match (Line 156–168).

## 9. `validate_states(df, log)`
**Purpose:** Ensures consistency between different state-related columns.
- **Reference:** Lines 180–190
- **Logic:** It compares `state_sourced` and `state`. While it identifies mismatches, the logic defaults to keeping the borrower's `state` for analysis (Line 188).

## 10. `create_default_flags(df, log)`
**Purpose:** Defines the target variables for the risk model based on Days Past Due (DPD).
- **Reference:** Lines 193–209
- **Logic:**
    - `is_default`: Flagged if `peak_dpd` >= 90 (or whatever `DEFAULT_DPD_THRESHOLD` is set to).
    - `is_stressed`: Flagged if `peak_dpd` >= 30 (or `STRESSED_DPD_THRESHOLD`).
- **Significance:** These binary flags (0 or 1) are the primary targets for the Probablity of Default (PD) model.

## 11. `engineer_features(df, log)`
**Purpose:** Creates new, high-value features from raw data to improve model performance.
- **Reference:** Lines 212–274
- **Key Features Created:**
    - **Ratios:** `loan_to_income_ratio`, `emi_to_income_ratio`, and `elapsed_ratio` (percentage of tenure completed).
    - **Bands:** Categorizes `bureau_score`, `age`, and `monthly_income` into discrete "bands" (e.g., "18-25", "26-35") using `pd.cut`.
    - **State Grouping:** Groups less frequent states into an "Other" category to reduce cardinality (Line 253–254).
    - **Tenure/Seasoning:** Calculates the total loan duration and how many months the loan has been active (seasoning) (Line 257–266).

## 12. `final_validation(df, log)`
**Purpose:** A safety check to ensure critical columns are not empty and to summarize the final dataset.
- **Reference:** Lines 277–297
- **Logic:** Checks for `NaN` values in critical columns like `loan_account_number` and `is_default` (Line 284). It logs the final default and stress rates.

## 13. `main()`
**Purpose:** The entry point that orchestrates the entire cleaning pipeline.
- **Reference:** Lines 300–335
- **Logic:** It initializes the `CleaningLog`, executes each cleaning function in sequence, saves the resulting dataframe to a CSV (`CLEANED_DATA_PATH`), and exports the cleaning log.
