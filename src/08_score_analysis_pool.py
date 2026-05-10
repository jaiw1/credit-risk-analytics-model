"""
Module 8: Score Loan Pool for Analysis
========================================
Reads the 'Loan Pool for Analysis' sheet from the .xlsb file,
cleans the data using the same pipeline as the historical data,
applies the trained Probability of Default (PD) model,
calculates LGD, and computes expected loss for the new pool.

Run: python src/08_score_analysis_pool.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import joblib

from utils import (
    RAW_DATA_PATH, MODELS_DIR, PD_MODEL_FEATURES, PROJECT_ROOT,
    CleaningLog
)
from importlib import import_module
data_cleaning = import_module("01_data_cleaning")

def main():
    print("=" * 70)
    print("CREDIT RISK MODELLING — SCORING LOAN POOL FOR ANALYSIS")
    print("=" * 70)

    log = CleaningLog()

    # 1. Load data
    print("\n[1] Loading Loan Pool for Analysis...")
    df_raw = pd.read_excel(RAW_DATA_PATH, engine="pyxlsb", header=1, sheet_name="Loan Pool for Analysis")
    print(f"  Loaded {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

    # 2. Clean data using identical steps from module 1
    print("\n[2] Applying data cleaning pipeline...")
    df = data_cleaning.drop_empty_columns(df_raw, log)
    df = data_cleaning.rename_columns(df, log)
    df = data_cleaning.standardize_business_sourced(df, log)
    df = data_cleaning.convert_dates(df, log)
    df = data_cleaning.clean_numeric_columns(df, log)
    df = data_cleaning.impute_age(df, log)
    df = data_cleaning.parse_product_detail(df, log)
    df = data_cleaning.validate_states(df, log)
    df = data_cleaning.create_default_flags(df, log)
    df = data_cleaning.engineer_features(df, log)

    print(f"  Cleaning complete. Final shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 3. Prepare features for PD model
    print("\n[3] Encoding features for PD Model...")
    # Ensure all required features exist
    for feature in PD_MODEL_FEATURES:
        if feature not in df.columns:
            df[feature] = np.nan

    X = df[PD_MODEL_FEATURES].copy()
    
    # Fill NaN with median (same logic as 03_pd_model.py)
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median() if not pd.isna(X[col].median()) else 0)
        
    # Load label encoders and encode categoricals
    encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
    
    if "woe_encoder" in encoders:
        # We are using WOE encoding!
        woe = encoders["woe_encoder"]
        cat_cols = encoders.get("cat_cols", [])
        for col in cat_cols:
            if col in X.columns:
                X[col] = X[col].fillna("Unknown").astype(str)
        X = woe.transform(X)
    else:
        # Legacy LabelEncoding path
        cat_cols = X.select_dtypes(include=["object", "str", "category"]).columns.tolist()
        
        for col in cat_cols:
            X[col] = X[col].astype(str)
            le = encoders.get(col)
            if le is not None:
                known_classes = set(le.classes_)
                mode_val = le.classes_[0]
                if "Unknown" in known_classes:
                    mode_val = "Unknown"
                elif "Other" in known_classes:
                    mode_val = "Other"
                    
                X[col] = X[col].apply(lambda x: x if x in known_classes else mode_val)
                X[col] = le.transform(X[col])
            else:
                X[col] = 0

    # 4. Predict PD
    print("\n[4] Predicting Probability of Default (PD)...")
    
    # Dynamically load the single best model determined by 03_pd_model.py
    best_model_path = os.path.join(MODELS_DIR, "pd_best_model.pkl")
    best_scaler_path = os.path.join(MODELS_DIR, "pd_best_scaler.pkl")
    
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}. Please run 03_pd_model.py first.")
        
    model = joblib.load(best_model_path)
    scaler = joblib.load(best_scaler_path)
    
    # Check if a scaler is required for this model type
    if scaler != "NO_SCALER":
        X_processed = scaler.transform(X)
    else:
        X_processed = X
        
    df["predicted_pd"] = model.predict_proba(X_processed)[:, 1]
    
    print(f"  [VERIFICATION] Successfully loaded model from: pd_best_model.pkl")
    print(f"  [VERIFICATION] Model type being used: {type(model).__name__}")

    print(f"  Mean PD for Analysis Pool: {df['predicted_pd'].mean()*100:.2f}%")
    print(f"  Max PD for Analysis Pool:  {df['predicted_pd'].max()*100:.2f}%")

    # 5. Predict LGD (Exact replication from 04_lgd_model.py)
    print("\n[5] Estimating Loss Given Default (LGD)...")
    from importlib import import_module
    lgd_model = import_module("04_lgd_model")
    
    # Use the conservative base scenario for headline EL, while preserving the
    # pure collateral estimate in the output for transparency.
    pure_case = lgd_model.SCENARIOS["Pure Collateral Estimate"]
    base_case = lgd_model.SCENARIOS["Base Case (20% Floor)"]
    
    # ensure total_pos is not null for LGD calc
    df["total_pos"] = df["total_pos"].fillna(df["sanctioned_loan_amount"])
    
    pure_lgd, pure_recovery_rate = lgd_model.compute_lgd(df, pure_case)
    predicted_lgd, recovery_rate = lgd_model.compute_lgd(df, base_case)
    df["pure_collateral_lgd"] = pure_lgd
    df["pure_collateral_recovery_rate"] = pure_recovery_rate
    df["predicted_lgd"] = predicted_lgd
    df["recovery_rate"] = recovery_rate
    
    print(f"  Mean LGD for Analysis Pool: {df['predicted_lgd'].mean()*100:.2f}%")

    # 6. Compute Expected Loss
    print("\n[6] Computing Expected Loss (EL)...")
    df["exposure_at_default"] = df["total_pos"].fillna(df["sanctioned_loan_amount"])
    df["expected_loss"] = df["predicted_pd"] * df["predicted_lgd"] * df["exposure_at_default"]
    
    total_exposure = df["exposure_at_default"].sum()
    total_el = df["expected_loss"].sum()
    print(f"  Total Exposure: ₹{total_exposure:,.0f}")
    print(f"  Total Expected Loss: ₹{total_el:,.0f} ({total_el/total_exposure*100:.2f}% of Exposure)")

    # 7. Save outputs
    output_path = os.path.join(PROJECT_ROOT, "outputs", "analysis_pool_scored.csv")
    out_cols = ["loan_account_number", "sanctioned_loan_amount", "exposure_at_default",
                "predicted_pd", "pure_collateral_lgd", "predicted_lgd", "expected_loss"]
    df[out_cols].to_csv(output_path, index=False)
    print(f"\n[7] Saved scored pool to: {output_path}")

    print("=" * 70)
    print("ANALYSIS POOL SCORING COMPLETE")
    print("=" * 70)
    return True

if __name__ == "__main__":
    main()
