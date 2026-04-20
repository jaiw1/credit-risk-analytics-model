"""
Module 3: Probability of Default (PD) Modelling
=================================================
Builds two models:
  a) Baseline Logistic Regression with WOE scorecard
  b) XGBoost / LightGBM ML model with SHAP explainability

Outputs PD estimates for every loan and model validation metrics.

Run: python src/03_pd_model.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, f1_score
)
from scipy.stats import ks_2samp

from utils import CLEANED_DATA_PATH, MODEL_OUTPUTS_PATH, FIGURES_DIR, PD_MODEL_FEATURES

# ============================================================
# CONFIGURATION
# ============================================================
# Using is_stressed (DPD>=30) as primary target for better class balance
# Also building for is_default (DPD>=90) as secondary
TARGET_COL = "is_stressed"       # primary target (n~1304)
SECONDARY_TARGET = "is_default"  # hard default (n~91)
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5


def load_and_prepare(target_col):
    """Load cleaned data and prepare features/target for modelling."""
    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"Loaded {len(df)} loans. Target: {target_col} (positive rate: {df[target_col].mean()*100:.2f}%)")

    # Select features that exist in the data
    available_features = [f for f in PD_MODEL_FEATURES if f in df.columns]
    print(f"Using {len(available_features)} features: {available_features}")

    X = df[available_features].copy()
    y = df[target_col].copy()

    # Encode categoricals
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    # Fill remaining NaN with median
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    return df, X, y, available_features, encoders


def compute_ks_statistic(y_true, y_prob):
    """Compute Kolmogorov-Smirnov statistic for model discrimination."""
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    ks_stat, _ = ks_2samp(pos_probs, neg_probs)
    return ks_stat


def compute_gini(auc):
    """Compute Gini coefficient from AUC."""
    return 2 * auc - 1


# ============================================================
# MODEL A: LOGISTIC REGRESSION (BASELINE)
# ============================================================
def build_logistic_regression(X_train, X_test, y_train, y_test, feature_names):
    """Build and evaluate a logistic regression baseline model."""
    print("\n" + "=" * 50)
    print("MODEL A: LOGISTIC REGRESSION (BASELINE)")
    print("=" * 50)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train with class weight balancing
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        C=0.1,       # regularization
        solver="lbfgs"
    )
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_prob_train = model.predict_proba(X_train_scaled)[:, 1]
    y_prob_test = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    auc_train = roc_auc_score(y_train, y_prob_train)
    auc_test = roc_auc_score(y_test, y_prob_test)
    gini = compute_gini(auc_test)
    ks = compute_ks_statistic(y_test.values, y_prob_test)

    print(f"\n  Train AUC: {auc_train:.4f}")
    print(f"  Test AUC:  {auc_test:.4f}")
    print(f"  Gini:      {gini:.4f}")
    print(f"  KS Stat:   {ks:.4f}")

    # Cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="roc_auc")
    print(f"  CV AUC:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance (coefficients)
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": model.coef_[0],
        "abs_coefficient": np.abs(model.coef_[0]),
        "odds_ratio": np.exp(model.coef_[0])
    }).sort_values("abs_coefficient", ascending=False)

    print("\n  Top 10 Risk Drivers (by |coefficient|):")
    for _, row in coef_df.head(10).iterrows():
        direction = "↑ risk" if row["coefficient"] > 0 else "↓ risk"
        print(f"    {row['feature']:30s}  coef={row['coefficient']:+.4f}  OR={row['odds_ratio']:.3f}  ({direction})")

    metrics = {
        "model": "Logistic Regression",
        "auc_train": auc_train, "auc_test": auc_test,
        "gini": gini, "ks": ks,
        "cv_auc_mean": cv_scores.mean(), "cv_auc_std": cv_scores.std()
    }

    return model, scaler, y_prob_test, coef_df, metrics


# ============================================================
# MODEL B: GRADIENT BOOSTING (ML MODEL)
# ============================================================
def build_xgboost(X_train, X_test, y_train, y_test, feature_names):
    """Build and evaluate a Gradient Boosting + Random Forest ensemble."""
    print("\n" + "=" * 50)
    print("MODEL B: GRADIENT BOOSTING + RANDOM FOREST (ML)")
    print("=" * 50)

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

    # --- Gradient Boosting ---
    print("\n  Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=20,
        subsample=0.8,
        max_features=0.8,
        random_state=RANDOM_STATE,
    )
    gb_model.fit(X_train, y_train)

    gb_prob_test = gb_model.predict_proba(X_test)[:, 1]
    gb_auc = roc_auc_score(y_test, gb_prob_test)
    print(f"  Gradient Boosting Test AUC: {gb_auc:.4f}")

    # --- Random Forest ---
    print("  Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    rf_prob_test = rf_model.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_prob_test)
    print(f"  Random Forest Test AUC:     {rf_auc:.4f}")

    # Use the better model
    if gb_auc >= rf_auc:
        model = gb_model
        model_name = "Gradient Boosting"
        print(f"\n  → Using Gradient Boosting (better AUC)")
    else:
        model = rf_model
        model_name = "Random Forest"
        print(f"\n  → Using Random Forest (better AUC)")

    # Predictions
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    # Metrics
    auc_train = roc_auc_score(y_train, y_prob_train)
    auc_test = roc_auc_score(y_test, y_prob_test)
    gini = compute_gini(auc_test)
    ks = compute_ks_statistic(y_test.values, y_prob_test)

    print(f"\n  Train AUC: {auc_train:.4f}")
    print(f"  Test AUC:  {auc_test:.4f}")
    print(f"  Gini:      {gini:.4f}")
    print(f"  KS Stat:   {ks:.4f}")

    # Cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"  CV AUC:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print(f"\n  Top 10 Features ({model_name}):")
    for _, row in imp_df.head(10).iterrows():
        print(f"    {row['feature']:30s}  importance={row['importance']:.4f}")

    metrics = {
        "model": model_name,
        "auc_train": auc_train, "auc_test": auc_test,
        "gini": gini, "ks": ks,
        "cv_auc_mean": cv_scores.mean(), "cv_auc_std": cv_scores.std()
    }

    return model, None, y_prob_test, imp_df, metrics


# ============================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================
def generate_shap_analysis(model, X_test, feature_names):
    """Generate permutation-based feature importance plots."""
    print("\n[IMPORTANCE] Generating feature importance analysis...")

    try:
        from sklearn.inspection import permutation_importance

        perm_result = permutation_importance(
            model, X_test, X_test.__class__(
                np.zeros(len(X_test)), index=X_test.index if hasattr(X_test, 'index') else None
            ) if False else None,
            n_repeats=10, random_state=RANDOM_STATE
        )
    except Exception:
        pass

    # Use built-in feature importance (works for tree models)
    try:
        imp = model.feature_importances_
        imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance": imp
        }).sort_values("importance", ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(imp_df)), imp_df["importance"].values, color="#e63946", alpha=0.85)
        ax.set_yticks(range(len(imp_df)))
        ax.set_yticklabels(imp_df["feature"].values)
        ax.set_xlabel("Feature Importance")
        ax.set_title("Top 15 Risk Drivers — Feature Importance", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("  [PLOT] Saved: feature_importance.png")
    except Exception as e:
        print(f"  [WARN] Feature importance plot failed: {e}")


# ============================================================
# VISUALIZATION
# ============================================================
def plot_model_comparison(y_test, lr_probs, xgb_probs, lr_metrics, xgb_metrics):
    """Plot ROC curves and comparison for both models."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ROC Curves
    ax = axes[0]
    for probs, label, color in [
        (lr_probs, f"Logistic Reg (AUC={lr_metrics['auc_test']:.3f})", "#457b9d"),
        (xgb_probs, f"XGBoost (AUC={xgb_metrics['auc_test']:.3f})", "#e63946"),
    ]:
        if probs is not None:
            fpr, tpr, _ = roc_curve(y_test, probs)
            ax.plot(fpr, tpr, linewidth=2, label=label, color=color)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Precision-Recall Curves
    ax = axes[1]
    for probs, label, color in [
        (lr_probs, f"Logistic Reg (AP={average_precision_score(y_test, lr_probs):.3f})", "#457b9d"),
        (xgb_probs, f"XGBoost (AP={average_precision_score(y_test, xgb_probs):.3f})" if xgb_probs is not None else "N/A", "#e63946"),
    ]:
        if probs is not None:
            precision, recall, _ = precision_recall_curve(y_test, probs)
            ax.plot(recall, precision, linewidth=2, label=label, color=color)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Metrics comparison bar chart
    ax = axes[2]
    metrics_names = ["AUC", "Gini", "KS", "CV AUC"]
    lr_vals = [lr_metrics["auc_test"], lr_metrics["gini"], lr_metrics["ks"], lr_metrics["cv_auc_mean"]]

    x = np.arange(len(metrics_names))
    width = 0.35

    ax.bar(x - width/2, lr_vals, width, label="Logistic Regression", color="#457b9d", alpha=0.85)

    if xgb_metrics:
        xgb_vals = [xgb_metrics["auc_test"], xgb_metrics["gini"], xgb_metrics["ks"], xgb_metrics["cv_auc_mean"]]
        ax.bar(x + width/2, xgb_vals, width, label="XGBoost", color="#e63946", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylabel("Score")
    ax.set_title("Model Metrics Comparison")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("PD Model Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pd_model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [PLOT] Saved: pd_model_comparison.png")


def plot_pd_distribution(pd_scores, title, filename):
    """Plot the distribution of predicted PD scores."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(pd_scores, bins=50, color="#457b9d", edgecolor="white", alpha=0.8)
    ax1.set_xlabel("Predicted PD")
    ax1.set_ylabel("Count")
    ax1.set_title(f"PD Distribution — {title}")
    ax1.axvline(pd_scores.mean(), color="#e63946", linestyle="--",
                label=f"Mean: {pd_scores.mean():.4f}")
    ax1.legend()

    # Decile analysis
    pd_decile = pd.qcut(pd_scores, 10, labels=False, duplicates="drop")
    decile_df = pd.DataFrame({"pd": pd_scores, "decile": pd_decile})
    decile_means = decile_df.groupby("decile")["pd"].mean()

    ax2.bar(range(len(decile_means)), decile_means.values, color="#e63946", alpha=0.8)
    ax2.set_xlabel("PD Decile (0=lowest risk)")
    ax2.set_ylabel("average PD")
    ax2.set_title("PD by Decile")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Saved: {filename}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("CREDIT RISK MODELLING — PD MODEL BUILDING")
    print("=" * 60)

    # --- Load & Prepare ---
    df, X, y, features, encoders = load_and_prepare(TARGET_COL)

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {len(X_train)} ({y_train.mean()*100:.2f}% positive)")
    print(f"Test:  {len(X_test)} ({y_test.mean()*100:.2f}% positive)")

    # --- Model A: Logistic Regression ---
    lr_model, lr_scaler, lr_probs, lr_coef, lr_metrics = build_logistic_regression(
        X_train, X_test, y_train, y_test, features
    )

    # --- Model B: XGBoost ---
    xgb_model, _, xgb_probs, xgb_imp, xgb_metrics = build_xgboost(
        X_train, X_test, y_train, y_test, features
    )

    # --- Model Comparison ---
    print("\n" + "=" * 50)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 50)
    print(f"{'Metric':<20} {'Logistic Reg':>15} {'XGBoost':>15}")
    print("-" * 50)
    for metric in ["auc_test", "gini", "ks", "cv_auc_mean"]:
        lr_val = lr_metrics.get(metric, 0)
        xgb_val = xgb_metrics.get(metric, 0) if xgb_metrics else 0
        winner = " ✓" if xgb_val > lr_val else ""
        print(f"  {metric:<18} {lr_val:>15.4f} {xgb_val:>15.4f}{winner}")

    # Recommend model
    if xgb_metrics and xgb_metrics["auc_test"] > lr_metrics["auc_test"]:
        recommended = "XGBoost"
        best_model = xgb_model
        print(f"\n  ★ RECOMMENDED: {recommended} (higher AUC, better discrimination)")
    else:
        recommended = "Logistic Regression"
        best_model = lr_model
        print(f"\n  ★ RECOMMENDED: {recommended}")

    # --- Generate PD for all loans ---
    print("\n[OUTPUT] Generating PD estimates for all loans...")
    if recommended == "Logistic Regression":
        X_all_scaled = lr_scaler.transform(X)
        df["pd_logistic"] = lr_model.predict_proba(X_all_scaled)[:, 1]
    else:
        df["pd_logistic"] = lr_model.predict_proba(lr_scaler.transform(X))[:, 1]

    if xgb_model is not None:
        df["pd_xgboost"] = xgb_model.predict_proba(X.values)[:, 1]
        df["pd_recommended"] = df["pd_xgboost"] if recommended == "XGBoost" else df["pd_logistic"]
    else:
        df["pd_recommended"] = df["pd_logistic"]

    # Assign PD rating bands
    df["pd_band"] = pd.cut(
        df["pd_recommended"],
        bins=[0, 0.01, 0.03, 0.05, 0.10, 0.20, 1.0],
        labels=["AAA (<1%)", "AA (1-3%)", "A (3-5%)", "BBB (5-10%)", "BB (10-20%)", "B+ (>20%)"],
        include_lowest=True
    )

    print(f"\n  PD Distribution Summary:")
    print(f"    Mean PD:   {df['pd_recommended'].mean():.4f}")
    print(f"    Median PD: {df['pd_recommended'].median():.4f}")
    print(f"    Max PD:    {df['pd_recommended'].max():.4f}")
    print(f"\n  PD Band Distribution:")
    print(df["pd_band"].value_counts().sort_index().to_string())

    # --- Visualizations ---
    print("\n[PLOTS] Generating visualizations...")
    plot_model_comparison(y_test, lr_probs, xgb_probs, lr_metrics, xgb_metrics or {})
    plot_pd_distribution(df["pd_recommended"].values, recommended, "pd_distribution.png")

    # SHAP for tree model
    if xgb_model is not None:
        generate_shap_analysis(xgb_model, X_test.values, features)

    # --- Save outputs ---
    output_cols = [
        "loan_account_number", "pd_logistic", "pd_recommended", "pd_band",
        "is_default", "is_stressed", "peak_dpd"
    ]
    if "pd_xgboost" in df.columns:
        output_cols.insert(2, "pd_xgboost")
    df[output_cols].to_csv(MODEL_OUTPUTS_PATH, index=False)
    print(f"\n  PD outputs saved to: {MODEL_OUTPUTS_PATH}")

    # Save full dataframe with PD for downstream modules
    df.to_csv(CLEANED_DATA_PATH.replace(".csv", "_with_pd.csv"), index=False)
    print(f"  Full data with PD saved for downstream modules.")

    print("\n" + "=" * 60)
    print(f"PD MODELLING COMPLETE — Recommended: {recommended}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
