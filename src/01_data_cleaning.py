"""
Module 1: Data Cleaning and Preparation
=========================================
Reads the raw .xlsb dataset, performs systematic cleaning,
produces a clean CSV and a detailed cleaning log.

Run: python src/01_data_cleaning.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from utils import (
    RAW_DATA_PATH, CLEANED_DATA_PATH, CLEANING_LOG_PATH,
    COLUMN_RENAME_MAP, DEFAULT_DPD_THRESHOLD, STRESSED_DPD_THRESHOLD,
    excel_serial_to_date, safe_float, CleaningLog
)


def load_raw_data(log):
    """Load raw .xlsb file with correct header row."""
    print("\n[1/12] Loading raw data...")
    df = pd.read_excel(RAW_DATA_PATH, engine="pyxlsb", header=1, sheet_name="Historical Default Data")
    log.log(
        "Loaded raw dataset",
        f"File: {RAW_DATA_PATH}",
        before_count=0, after_count=df.shape[0]
    )
    print(f"  Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def drop_empty_columns(df, log):
    """Drop the last empty/unnamed column and Address1 (99.9% null)."""
    print("\n[2/12] Dropping empty/useless columns...")
    before_cols = df.shape[1]

    # Drop last column if it's entirely NaN
    last_col = df.columns[-1]
    if df[last_col].isna().sum() > df.shape[0] * 0.99:
        df = df.drop(columns=[last_col])
        log.log("Dropped last empty column", f"Column '{last_col}' was 99.9%+ NaN")

    # Drop Address1 (almost entirely null)
    if "Address1" in df.columns:
        null_pct = df["Address1"].isna().sum() / len(df) * 100
        df = df.drop(columns=["Address1"])
        log.log("Dropped Address1 column", f"Was {null_pct:.1f}% null — no usable data")

    log.log("Column cleanup complete", before_count=before_cols, after_count=df.shape[1])
    return df


def rename_columns(df, log):
    """Rename columns to clean snake_case names."""
    print("\n[3/12] Renaming columns...")
    # Only rename columns that exist in our map
    rename_map = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    log.log("Renamed columns to snake_case", f"Renamed {len(rename_map)} columns")
    return df


def standardize_business_sourced(df, log):
    """Fix inconsistent casing in business_sourced_by."""
    print("\n[4/12] Standardizing business_sourced_by...")
    before = df["business_sourced_by"].nunique()

    # Standardize known duplicates
    mapping = {
        "MFL BRANCH": "MFL Branch",
        "MFL branch": "MFL Branch",
    }
    df["business_sourced_by"] = df["business_sourced_by"].replace(mapping)

    after = df["business_sourced_by"].nunique()
    log.log(
        "Standardized business_sourced_by casing",
        f"Mapped variations: {mapping}",
        before_count=before, after_count=after
    )
    return df

def convert_dates(df, log):
    """Convert Excel serial date numbers to proper datetime objects."""
    print("\n[5/12] Converting date columns...")
    date_cols = ["date_of_birth", "disbursement_date", "maturity_date",
                 "first_payment_date", "cut_off_date"]

    for col in date_cols:
        if col in df.columns:
            df[col] = df[col].apply(excel_serial_to_date)
            valid_count = df[col].notna().sum()
            log.log(
                f"Converted {col} from Excel serial to datetime",
                f"Valid dates: {valid_count}/{len(df)}"
            )
    return df


def clean_numeric_columns(df, log):
    """Convert bureau_score and ltv to proper numeric types."""
    print("\n[6/12] Cleaning numeric columns...")

    # Bureau Score: convert to float, handle -1 as "no score"
    if "bureau_score" in df.columns:
        df["bureau_score"] = df["bureau_score"].apply(safe_float)
        neg_count = (df["bureau_score"] < 0).sum()
        df.loc[df["bureau_score"] < 0, "bureau_score"] = np.nan
        log.log(
            "Cleaned bureau_score",
            f"Converted to float. Replaced {neg_count} negative values (-1) with NaN (no bureau history)"
        )

    # LTV: convert to float
    if "ltv" in df.columns:
        df["ltv"] = df["ltv"].apply(safe_float)
        null_count = df["ltv"].isna().sum()
        log.log("Cleaned LTV", f"Converted to float. {null_count} null values remaining")

    return df


def impute_age(df, log):
    """Fill missing AGE from date_of_birth if possible."""
    print("\n[7/12] Imputing missing age values...")
    null_before = df["age"].isna().sum()

    if null_before > 0 and "date_of_birth" in df.columns:
        # Convert age to numeric first
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

        mask = df["age"].isna() & df["date_of_birth"].notna()
        if mask.any():
            # Calculate age from DOB using cut_off_date as reference
            ref_date = df["cut_off_date"].dropna().iloc[0] if df["cut_off_date"].notna().any() else pd.Timestamp.now()
            df.loc[mask, "age"] = ((ref_date - df.loc[mask, "date_of_birth"]).dt.days / 365.25).astype(int)
    else:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

    null_after = df["age"].isna().sum()
    log.log("Imputed age", f"Calculated from DOB where missing",
            before_count=null_before, after_count=null_after)
    return df


def parse_product_detail(df, log):
    """Extract vehicle type (Scooter/Motorcycle) from product_detail."""
    print("\n[8/12] Parsing product detail...")

    if "product_detail" in df.columns:
        # product_detail format: "MODEL_BRAND_Type"
        # Extract the last component as vehicle type
        def extract_vehicle_type(val):
            if pd.isna(val):
                return "Unknown"
            parts = str(val).split("_")
            if len(parts) >= 3:
                vtype = parts[-1].strip()
                if "scooter" in vtype.lower():
                    return "Scooter"
                elif "motorcycle" in vtype.lower() or "bike" in vtype.lower():
                    return "Motorcycle"
                else:
                    return vtype
            return "Unknown"

        df["vehicle_type"] = df["product_detail"].apply(extract_vehicle_type)
        type_counts = df["vehicle_type"].value_counts().to_dict()
        log.log("Extracted vehicle_type from product_detail", f"Distribution: {type_counts}")
    else:
        df["vehicle_type"] = "Unknown"
        log.log("Vehicle type defaulted to Unknown", "product_detail column not found")

    return df


def validate_states(df, log):
    """Check consistency between state_sourced and state, keep borrower state."""
    print("\n[9/12] Validating state columns...")

    if "state_sourced" in df.columns and "state" in df.columns:
        mismatch = (df["state_sourced"].str.upper() != df["state"].str.upper()).sum()
        log.log(
            "Validated state columns",
            f"state_sourced vs state mismatch: {mismatch} rows. Keeping borrower 'state' column."
        )
    return df


def create_default_flags(df, log):
    """Create binary default and stress flags from Peak DPD."""
    print("\n[10/12] Creating default flags...")

    df["is_default"] = (df["peak_dpd"] >= DEFAULT_DPD_THRESHOLD).astype(int)
    df["is_stressed"] = (df["peak_dpd"] >= STRESSED_DPD_THRESHOLD).astype(int)

    default_count = df["is_default"].sum()
    stressed_count = df["is_stressed"].sum()
    total = len(df)

    log.log(
        "Created default flags",
        f"is_default (DPD>={DEFAULT_DPD_THRESHOLD}): {default_count} ({default_count/total*100:.2f}%)\n"
        f"    is_stressed (DPD>={STRESSED_DPD_THRESHOLD}): {stressed_count} ({stressed_count/total*100:.2f}%)"
    )
    return df


def engineer_features(df, log):
    """Create derived features for modelling."""
    print("\n[11/12] Engineering features...")

    # Loan-to-income ratio
    df["loan_to_income_ratio"] = df["sanctioned_loan_amount"] / df["monthly_income"].replace(0, np.nan)

    # EMI-to-income ratio
    df["emi_to_income_ratio"] = df["installment_amount"] / df["monthly_income"].replace(0, np.nan)

    # Elapsed ratio (how far through the loan)
    df["elapsed_ratio"] = df["emi_paid"] / df["no_of_installments"].replace(0, np.nan)

    # Bureau score bands
    df["bureau_score_band"] = pd.cut(
        df["bureau_score"],
        bins=[-1, 0, 300, 550, 700, 850],
        labels=["No Score", "Very Low (1-300)", "Low (301-550)", "Medium (551-700)", "High (701-850)"],
        include_lowest=True
    )
    # Handle NaN bureau scores
    df["bureau_score_band"] = df["bureau_score_band"].cat.add_categories("Missing")
    df.loc[df["bureau_score"].isna(), "bureau_score_band"] = "Missing"

    # Age bands
    df["age_band"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "55+"],
        include_lowest=True
    )

    # Income bands
    df["income_band"] = pd.cut(
        df["monthly_income"],
        bins=[0, 25000, 50000, 75000, 100000, float("inf")],
        labels=["<25K", "25K-50K", "50K-75K", "75K-1L", "1L+"],
        include_lowest=True
    )

    # State grouping (top 5 + Other)
    top_states = df["state"].value_counts().head(5).index.tolist()
    df["state_group"] = df["state"].apply(lambda x: x if x in top_states else "Other")

    # Loan tenure in months (from dates)
    if "disbursement_date" in df.columns and "maturity_date" in df.columns:
        df["loan_tenure_months"] = (
            (df["maturity_date"] - df["disbursement_date"]).dt.days / 30.44
        ).round(0)

    # Seasoning (months since disbursement to cutoff)
    if "disbursement_date" in df.columns and "cut_off_date" in df.columns:
        df["seasoning_months"] = (
            (df["cut_off_date"] - df["disbursement_date"]).dt.days / 30.44
        ).round(0)

    new_features = [
        "loan_to_income_ratio", "emi_to_income_ratio", "elapsed_ratio",
        "bureau_score_band", "age_band", "income_band", "state_group",
        "loan_tenure_months", "seasoning_months"
    ]
    log.log("Engineered features", f"Created: {', '.join(new_features)}")
    return df


def final_validation(df, log):
    """Run final checks on the cleaned dataset."""
    print("\n[12/12] Final validation...")

    # Check critical columns have no unexpected nulls
    critical_cols = ["loan_account_number", "peak_dpd", "is_default",
                     "sanctioned_loan_amount", "monthly_income", "interest_rate"]
    for col in critical_cols:
        nulls = df[col].isna().sum()
        if nulls > 0:
            log.log(f"WARNING: {col} has {nulls} null values after cleaning")

    # Summary stats
    log.log(
        "Final dataset summary",
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        f"    Default rate (DPD>=90): {df['is_default'].mean()*100:.2f}%\n"
        f"    Stress rate (DPD>=30): {df['is_stressed'].mean()*100:.2f}%\n"
        f"    Null summary:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}"
    )
    return df


def main():
    """Execute the full data cleaning pipeline."""
    print("=" * 60)
    print("CREDIT RISK MODELLING — DATA CLEANING PIPELINE")
    print("=" * 60)

    log = CleaningLog()

    # Pipeline steps
    df = load_raw_data(log)
    df = drop_empty_columns(df, log)
    df = rename_columns(df, log)
    df = standardize_business_sourced(df, log)
    df = convert_dates(df, log)
    df = clean_numeric_columns(df, log)
    df = impute_age(df, log)
    df = parse_product_detail(df, log)
    df = validate_states(df, log)
    df = create_default_flags(df, log)
    df = engineer_features(df, log)
    df = final_validation(df, log)

    # Save outputs
    print("\nSaving cleaned data...")
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"  Saved cleaned data to: {CLEANED_DATA_PATH}")

    log.save(CLEANING_LOG_PATH)

    print("\n" + "=" * 60)
    print(f"CLEANING COMPLETE: {df.shape[0]} loans, {df.shape[1]} features")
    print(f"Default rate: {df['is_default'].mean()*100:.2f}%")
    print(f"Stress rate:  {df['is_stressed'].mean()*100:.2f}%")
    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
