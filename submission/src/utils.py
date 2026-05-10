"""
Shared utilities and constants for the Credit Risk Modelling project.
Contains file paths, helper functions, and configuration used across all modules.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================
# PATH CONFIGURATION
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "Credit Risk Modelling_Sample Data.xlsb")
CLEANED_DATA_PATH = os.path.join(PROJECT_ROOT, "outputs", "cleaned_data.csv")
MODEL_OUTPUTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "model_outputs.csv")
PORTFOLIO_REPORT_PATH = os.path.join(PROJECT_ROOT, "outputs", "portfolio_report.csv")
CLEANING_LOG_PATH = os.path.join(PROJECT_ROOT, "outputs", "cleaning_log.txt")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")
EXCEL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "credit_risk_model_outputs.xlsx")

# Ensure output directories exist
os.makedirs(os.path.join(PROJECT_ROOT, "outputs", "figures"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "outputs", "models"), exist_ok=True)

# ============================================================
# COLUMN NAME MAPPING (raw -> clean snake_case)
# ============================================================
COLUMN_RENAME_MAP = {
    "Loan Account Number": "loan_account_number",
    "Product Name": "product_name",
    "State": "state_sourced",
    "District#": "district_sourced",
    "Business Sourced By": "business_sourced_by",
    "Interest Rate": "interest_rate",
    "Branch Code": "branch_code",
    "Branch Name": "branch_name",
    "Product Name.1": "product_detail",
    "Product Name2": "product_detail",
    "AGE": "age",
    "Date of Birth": "date_of_birth",
    "Gender": "gender",
    "Monthly Income": "monthly_income",
    "Purpose of Loan": "purpose_of_loan",
    "Address1": "address1",
    "City and District": "city_and_district",
    "District": "district",
    "State.1": "state",
    "State3": "state",
    "Pin": "pin",
    "Sanctioned Loan Amount": "sanctioned_loan_amount",
    "Disbursed Loan Amount": "disbursed_loan_amount",
    "Full disbursed (Yes/No)": "full_disbursed",
    "Disbursement Date": "disbursement_date",
    "Maturity Date": "maturity_date",
    "No of Installments": "no_of_installments",
    "EMI_Paid": "emi_paid",
    "Balance Tenure": "balance_tenure",
    "First Payment Date": "first_payment_date",
    "Cut off Date": "cut_off_date",
    "Installment Amount": "installment_amount",
    "Instal_Type (Flat/Reducing)": "installment_type",
    "Overdue amount": "overdue_amount",
    "Over_Due": "over_due",
    "Peak DPD": "peak_dpd",
    "CURRENT_DPD": "current_dpd",
    "Total POS": "total_pos",
    "Total IOS": "total_ios",
    "EMI_Advance": "emi_advance",
    "LTV (Cal)": "ltv",
    "Property Value": "property_value",
    "INSURANCE_STATUS": "insurance_status",
    "REG_INFO_STATUS": "reg_info_status",
    "MANUFACTURER": "manufacturer",
    "Bureau Score": "bureau_score",
}

# ============================================================
# DEFAULT DEFINITION THRESHOLDS
# ============================================================
DEFAULT_DPD_THRESHOLD = 90    # Hard default: Peak DPD >= 90
STRESSED_DPD_THRESHOLD = 30   # Watch/stressed: Peak DPD >= 30

# ============================================================
# FEATURE GROUPS
# ============================================================
NUMERIC_FEATURES = [
    "interest_rate", "age", "monthly_income", "sanctioned_loan_amount",
    "no_of_installments", "emi_paid", "balance_tenure", "installment_amount",
    "peak_dpd", "total_pos", "total_ios", "ltv", "property_value",
    "bureau_score", "loan_to_income_ratio", "emi_to_income_ratio",
    "elapsed_ratio",
]

CATEGORICAL_FEATURES = [
    "gender", "business_sourced_by", "state_group", "manufacturer",
    "insurance_status", "reg_info_status", "bureau_score_band",
    "age_band", "income_band", "vehicle_type",
]

# Features to use for PD modelling (subset without target leakage)
# NOTE: Removed bureau_score_band, age_band, income_band — these are
# directly derived from bureau_score, age, monthly_income already in the list,
# which introduces multicollinearity and gives tree models redundant splits.
PD_MODEL_FEATURES = [
    "bureau_score", "ltv", "loan_to_income_ratio", "emi_to_income_ratio",
    "interest_rate", "monthly_income", "age", "no_of_installments",
    "sanctioned_loan_amount", "property_value",
    # categorical (will be encoded)
    "gender", "manufacturer", "state_group", "insurance_status",
    "reg_info_status", "business_sourced_by", "vehicle_type",
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def excel_serial_to_date(serial):
    """Convert Excel serial date number to Python datetime."""
    if pd.isna(serial):
        return pd.NaT
    try:
        serial = int(float(serial))
        # Excel epoch is 1900-01-01 with the 1900 leap year bug
        return datetime(1899, 12, 30) + timedelta(days=serial)
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def create_bins(series, bins, labels):
    """Create binned categories from a numeric series."""
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)


def safe_float(val):
    """Safely convert a value to float, returning NaN on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


class CleaningLog:
    """Logger for tracking data cleaning steps with before/after metrics."""

    def __init__(self):
        self.entries = []
        self.step_num = 0

    def log(self, action, details="", before_count=None, after_count=None):
        self.step_num += 1
        entry = f"Step {self.step_num}: {action}"
        if details:
            entry += f"\n    Details: {details}"
        if before_count is not None and after_count is not None:
            entry += f"\n    Before: {before_count} | After: {after_count} | Changed: {before_count - after_count}"
        self.entries.append(entry)
        print(f"  [CLEAN] {entry.split(chr(10))[0]}")

    def save(self, filepath):
        header = f"DATA CLEANING LOG\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n\n"
        with open(filepath, "w") as f:
            f.write(header)
            f.write("\n\n".join(self.entries))
            f.write(f"\n\n{'='*60}\nTotal steps: {self.step_num}\n")
        print(f"  [LOG] Cleaning log saved to {filepath}")
