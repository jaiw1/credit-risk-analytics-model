"""
Module 3: Probability of Default (PD) Modelling
=================================================
Builds three models as required by the project brief:
  a) Baseline Logistic Regression (statistical model)
  b) Gradient Boosting / Random Forest (ML model)
  c) Neural Network (MLP Classifier)

All models are calibrated using CalibratedClassifierCV (Platt scaling)
to ensure predicted probabilities match observed default rates.

Includes:
  - SHAP-based explainability for the recommended model
  - Calibration plots (reliability diagrams)
  - Decile validation tables (standard credit risk validation)
  - Model persistence via joblib

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
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, f1_score, brier_score_loss
)
from scipy.stats import ks_2samp

from utils import (
    CLEANED_DATA_PATH, MODEL_OUTPUTS_PATH, FIGURES_DIR,
    MODELS_DIR, PD_MODEL_FEATURES
)

# ============================================================
# CONFIGURATION
# ============================================================
# Headline target uses DPD>=30 because 90+ DPD hard defaults are extremely
# sparse in this sample. Treat pd_recommended as a stress/early-delinquency PD.
# A separate 90+ DPD model should use rare-event techniques and wider validation.
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
# MODEL A: LOGISTIC REGRESSION (BASELINE STATISTICAL MODEL)
# ============================================================
def build_logistic_regression(X_train, X_test, y_train, y_test, feature_names):
    """
    Build and evaluate a logistic regression baseline model.
    Uses CalibratedClassifierCV (Platt scaling) for probability calibration.
    """
    print("\n" + "=" * 50)
    print("MODEL A: LOGISTIC REGRESSION (BASELINE)")
    print("=" * 50)

    # Scale features — LR is sensitive to feature scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train with class weight balancing for the imbalanced target
    base_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        C=0.1,       # L2 regularization strength
        solver="lbfgs"
    )

    # Calibrate probabilities using Platt scaling (sigmoid) via 5-fold CV
    # This ensures predicted PD aligns with observed default rates
    model = CalibratedClassifierCV(
        base_model, method="sigmoid", cv=5
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
    brier = brier_score_loss(y_test, y_prob_test)

    print(f"\n  Train AUC:    {auc_train:.4f}")
    print(f"  Test AUC:     {auc_test:.4f}")
    print(f"  Gini:         {gini:.4f}")
    print(f"  KS Stat:      {ks:.4f}")
    print(f"  Brier Score:  {brier:.4f}")

    # Cross-validation on base (uncalibrated) model for CV AUC
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(base_model, X_train_scaled, y_train, cv=cv, scoring="roc_auc")
    print(f"  CV AUC:       {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance: extract coefficients from the base estimators
    # CalibratedClassifierCV wraps the model, so we extract from calibrated_classifiers_
    try:
        coefs = np.mean([
            est.estimator.coef_[0] if hasattr(est, 'estimator') else est.base_estimator.coef_[0]
            for est in model.calibrated_classifiers_
        ], axis=0)
    except Exception:
        # Fallback: train a separate LR just for coefficients
        base_model.fit(X_train_scaled, y_train)
        coefs = base_model.coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
        "odds_ratio": np.exp(coefs)
    }).sort_values("abs_coefficient", ascending=False)

    print("\n  Top 10 Risk Drivers (by |coefficient|):")
    for _, row in coef_df.head(10).iterrows():
        direction = "↑ risk" if row["coefficient"] > 0 else "↓ risk"
        print(f"    {row['feature']:30s}  coef={row['coefficient']:+.4f}  OR={row['odds_ratio']:.3f}  ({direction})")

    metrics = {
        "model": "Logistic Regression",
        "auc_train": auc_train, "auc_test": auc_test,
        "gini": gini, "ks": ks, "brier": brier,
        "cv_auc_mean": cv_scores.mean(), "cv_auc_std": cv_scores.std()
    }

    return model, scaler, y_prob_test, coef_df, metrics


# ============================================================
# MODEL B: GRADIENT BOOSTING (ML MODEL)
# ============================================================
def build_gradient_boosting(X_train, X_test, y_train, y_test, feature_names):
    """
    Build and evaluate Gradient Boosting + Random Forest ensemble.
    Applies probability calibration via isotonic regression.
    """
    print("\n" + "=" * 50)
    print("MODEL B: GRADIENT BOOSTING + RANDOM FOREST (ML)")
    print("=" * 50)

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

    # --- Gradient Boosting ---
    print("\n  Training Gradient Boosting...")
    gb_base = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=20,
        subsample=0.8,
        max_features=0.8,
        random_state=RANDOM_STATE,
    )
    # Calibrate with isotonic regression (better for tree models)
    gb_model = CalibratedClassifierCV(gb_base, method="isotonic", cv=5)
    gb_model.fit(X_train, y_train)

    gb_prob_test = gb_model.predict_proba(X_test)[:, 1]
    gb_auc = roc_auc_score(y_test, gb_prob_test)
    print(f"  Gradient Boosting Test AUC: {gb_auc:.4f}")

    # --- Random Forest ---
    print("  Training Random Forest...")
    rf_base = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_model = CalibratedClassifierCV(rf_base, method="isotonic", cv=5)
    rf_model.fit(X_train, y_train)

    rf_prob_test = rf_model.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_prob_test)
    print(f"  Random Forest Test AUC:     {rf_auc:.4f}")

    # Select the better model
    if gb_auc >= rf_auc:
        model = gb_model
        base_model = gb_base
        model_name = "Gradient Boosting"
        print(f"\n  → Using Gradient Boosting (better AUC)")
    else:
        model = rf_model
        base_model = rf_base
        model_name = "Random Forest"
        print(f"\n  → Using Random Forest (better AUC)")

    # Predictions from selected model
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    # Metrics
    auc_train = roc_auc_score(y_train, y_prob_train)
    auc_test = roc_auc_score(y_test, y_prob_test)
    gini = compute_gini(auc_test)
    ks = compute_ks_statistic(y_test.values, y_prob_test)
    brier = brier_score_loss(y_test, y_prob_test)

    print(f"\n  Train AUC:    {auc_train:.4f}")
    print(f"  Test AUC:     {auc_test:.4f}")
    print(f"  Gini:         {gini:.4f}")
    print(f"  KS Stat:      {ks:.4f}")
    print(f"  Brier Score:  {brier:.4f}")

    # Cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(base_model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"  CV AUC:       {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance from base estimator
    # Fit the base model separately for importance extraction
    base_model.fit(X_train, y_train)
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": base_model.feature_importances_
    }).sort_values("importance", ascending=False)

    print(f"\n  Top 10 Features ({model_name}):")
    for _, row in imp_df.head(10).iterrows():
        print(f"    {row['feature']:30s}  importance={row['importance']:.4f}")

    metrics = {
        "model": model_name,
        "auc_train": auc_train, "auc_test": auc_test,
        "gini": gini, "ks": ks, "brier": brier,
        "cv_auc_mean": cv_scores.mean(), "cv_auc_std": cv_scores.std()
    }

    return model, base_model, None, y_prob_test, imp_df, metrics


# ============================================================
# MODEL C: NEURAL NETWORK (MLP CLASSIFIER)
# ============================================================
def build_neural_network(X_train, X_test, y_train, y_test, feature_names):
    """
    Build and evaluate a multi-layer perceptron (neural network) model.
    Architecture: 2 hidden layers with dropout-like regularization.
    Calibrated with Platt scaling for reliable probability estimates.
    """
    print("\n" + "=" * 50)
    print("MODEL C: NEURAL NETWORK (MLP CLASSIFIER)")
    print("=" * 50)

    # Neural networks require scaled input
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Two-layer MLP with ReLU activation
    # Architecture chosen for tabular credit data: not too deep, enough capacity
    nn_base = MLPClassifier(
        hidden_layer_sizes=(128, 64),   # 2 hidden layers
        activation="relu",
        solver="adam",
        alpha=0.001,                     # L2 regularization
        batch_size=256,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,             # Auto-stop when validation loss plateaus
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
        verbose=False,
    )

    # Calibrate probabilities
    model = CalibratedClassifierCV(nn_base, method="sigmoid", cv=5)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_prob_train = model.predict_proba(X_train_scaled)[:, 1]
    y_prob_test = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    auc_train = roc_auc_score(y_train, y_prob_train)
    auc_test = roc_auc_score(y_test, y_prob_test)
    gini = compute_gini(auc_test)
    ks = compute_ks_statistic(y_test.values, y_prob_test)
    brier = brier_score_loss(y_test, y_prob_test)

    print(f"\n  Train AUC:    {auc_train:.4f}")
    print(f"  Test AUC:     {auc_test:.4f}")
    print(f"  Gini:         {gini:.4f}")
    print(f"  KS Stat:      {ks:.4f}")
    print(f"  Brier Score:  {brier:.4f}")

    # Cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(nn_base, X_train_scaled, y_train, cv=cv, scoring="roc_auc")
    print(f"  CV AUC:       {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    metrics = {
        "model": "Neural Network (MLP)",
        "auc_train": auc_train, "auc_test": auc_test,
        "gini": gini, "ks": ks, "brier": brier,
        "cv_auc_mean": cv_scores.mean(), "cv_auc_std": cv_scores.std()
    }

    return model, scaler, y_prob_test, metrics


# ============================================================
# SHAP EXPLAINABILITY (actual SHAP, not broken fallback)
# ============================================================
def generate_shap_analysis(base_model, X_train, X_test, feature_names):
    """
    Generate SHAP (SHapley Additive exPlanations) feature importance.
    Uses TreeExplainer for tree-based models — the gold standard
    for explaining which variables drive credit risk predictions.
    """
    print("\n[SHAP] Generating SHAP explainability analysis...")

    try:
        import shap

        # TreeExplainer is efficient for tree-based models (GB, RF)
        explainer = shap.TreeExplainer(base_model)
        # Use a sample if dataset is large (SHAP can be slow)
        sample_size = min(2000, len(X_test))
        X_sample = X_test[:sample_size] if hasattr(X_test, 'iloc') else X_test[:sample_size]

        shap_values = explainer.shap_values(X_sample)

        # For binary classification, shap_values may be a list [class0, class1]
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]  # Use positive class
        else:
            shap_vals = shap_values

        # 1. SHAP Summary Plot (beeswarm)
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            shap_vals, X_sample,
            feature_names=feature_names,
            show=False, max_display=15
        )
        plt.title("SHAP Feature Importance — Risk Drivers", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("  [PLOT] Saved: shap_summary.png")

        # 2. SHAP Bar Plot (mean absolute SHAP value per feature)
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_vals, X_sample,
            feature_names=feature_names,
            plot_type="bar", show=False, max_display=15
        )
        plt.title("Mean |SHAP Value| — Feature Importance Ranking", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "shap_importance_bar.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("  [PLOT] Saved: shap_importance_bar.png")

        # 3. Mean absolute SHAP values for reporting
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False)

        print("\n  SHAP-based Risk Driver Ranking:")
        for i, (_, row) in enumerate(shap_df.head(10).iterrows(), 1):
            print(f"    {i:2d}. {row['feature']:30s}  SHAP={row['mean_abs_shap']:.4f}")

        return shap_df

    except ImportError:
        print("  [WARN] SHAP not installed. Falling back to feature_importances_.")
        return _fallback_feature_importance(base_model, feature_names)
    except Exception as e:
        print(f"  [WARN] SHAP failed ({e}). Falling back to feature_importances_.")
        return _fallback_feature_importance(base_model, feature_names)


def _fallback_feature_importance(model, feature_names):
    """Fallback to tree-based feature importance if SHAP fails."""
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
        ax.set_xlabel("Feature Importance (Gini)")
        ax.set_title("Top 15 Risk Drivers — Feature Importance", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("  [PLOT] Saved: feature_importance.png (Gini-based fallback)")
        return imp_df
    except Exception as e:
        print(f"  [WARN] Feature importance plot failed: {e}")
        return None


# ============================================================
# CALIBRATION PLOT (Reliability Diagram)
# ============================================================
def plot_calibration_curves(y_test, model_probs, model_names):
    """
    Plot calibration curves (reliability diagrams) for all models.
    A well-calibrated model's curve should align with the diagonal.
    This is a standard credit risk model validation technique.
    """
    print("\n[CALIBRATION] Generating calibration plots...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Calibration curve ---
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")

    for name, probs in zip(model_names, model_probs):
        if probs is None:
            continue
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
        brier = brier_score_loss(y_test, probs)
        ax1.plot(prob_pred, prob_true, "o-", linewidth=2,
                 label=f"{name} (Brier={brier:.4f})")

    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Observed Default Rate")
    ax1.set_title("Calibration Curve (Reliability Diagram)", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # --- Predicted PD histogram overlay ---
    for name, probs in zip(model_names, model_probs):
        if probs is None:
            continue
        ax2.hist(probs, bins=50, alpha=0.4, label=name)

    ax2.axvline(y_test.mean(), color="black", linestyle="--", linewidth=2,
                label=f"Actual default rate: {y_test.mean():.4f}")
    ax2.set_xlabel("Predicted PD")
    ax2.set_ylabel("Count")
    ax2.set_title("Predicted PD Distributions", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Model Calibration Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "calibration_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [PLOT] Saved: calibration_curves.png")


# ============================================================
# DECILE VALIDATION TABLE (Standard Credit Risk Validation)
# ============================================================
def generate_decile_analysis(y_test, y_prob, model_name):
    """
    Produce a decile validation table — the standard format for
    presenting PD model results in credit risk.
    Loans are sorted by predicted PD into 10 equal buckets.
    For each bucket, we compare predicted vs. actual default rates.
    """
    print(f"\n[DECILE] Generating decile analysis for {model_name}...")

    df_decile = pd.DataFrame({
        "actual": y_test.values,
        "predicted_pd": y_prob
    })

    # Create decile buckets (1 = lowest risk, 10 = highest risk)
    df_decile["decile"] = pd.qcut(df_decile["predicted_pd"], 10,
                                   labels=range(1, 11), duplicates="drop")

    decile_table = df_decile.groupby("decile", observed=True).agg(
        count=("actual", "count"),
        n_defaults=("actual", "sum"),
        actual_dr=("actual", "mean"),
        avg_pd=("predicted_pd", "mean"),
        min_pd=("predicted_pd", "min"),
        max_pd=("predicted_pd", "max"),
    ).reset_index()

    decile_table["cumul_defaults"] = decile_table["n_defaults"].cumsum()
    total_defaults = decile_table["n_defaults"].sum()
    decile_table["cumul_default_pct"] = (decile_table["cumul_defaults"] / total_defaults * 100).round(1)

    print(f"\n  {'Decile':>7} {'Count':>7} {'Defaults':>9} {'Act DR':>8} {'Pred PD':>9} "
          f"{'PD Range':>18} {'Cumul Def%':>11}")
    print("  " + "-" * 75)

    for _, row in decile_table.iterrows():
        print(f"  {int(row['decile']):>7} {int(row['count']):>7} {int(row['n_defaults']):>9} "
              f"{row['actual_dr']:>8.4f} {row['avg_pd']:>9.4f} "
              f"[{row['min_pd']:.4f}-{row['max_pd']:.4f}] "
              f"{row['cumul_default_pct']:>10.1f}%")

    # Plot decile chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = range(len(decile_table))
    width = 0.35

    # Actual vs Predicted by decile
    ax1.bar([i - width/2 for i in x], decile_table["actual_dr"] * 100, width,
            label="Actual Default Rate", color="#e63946", alpha=0.85)
    ax1.bar([i + width/2 for i in x], decile_table["avg_pd"] * 100, width,
            label="Predicted PD", color="#457b9d", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"D{int(d)}" for d in decile_table["decile"]])
    ax1.set_xlabel("PD Decile (D1=lowest risk)")
    ax1.set_ylabel("Default Rate (%)")
    ax1.set_title("Actual vs Predicted Default Rate by Decile")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Cumulative default capture
    ax2.plot(decile_table["decile"].astype(int), decile_table["cumul_default_pct"],
             "o-", color="#e63946", linewidth=2.5, markersize=8, label="Model")
    ax2.plot([1, 10], [10, 100], "k--", alpha=0.5, label="Random model")
    ax2.set_xlabel("PD Decile")
    ax2.set_ylabel("Cumulative Defaults Captured (%)")
    ax2.set_title("Cumulative Accuracy Profile (CAP Curve)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    plt.suptitle(f"Decile Validation — {model_name}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "decile_validation.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  [PLOT] Saved: decile_validation.png")

    return decile_table


# ============================================================
# MODEL COMPARISON VISUALIZATION
# ============================================================
def plot_model_comparison(y_test, all_probs, all_metrics):
    """Plot ROC curves and comparison for all models."""
    colors = {"Logistic Regression": "#457b9d",
              "Gradient Boosting": "#e63946", "Random Forest": "#e63946",
              "Neural Network (MLP)": "#2a9d8f"}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- ROC Curves ---
    ax = axes[0]
    for name, probs in all_probs.items():
        if probs is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = all_metrics[name]["auc_test"]
        color = colors.get(name, "#f4a261")
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc_val:.3f})", color=color)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Precision-Recall Curves ---
    ax = axes[1]
    for name, probs in all_probs.items():
        if probs is None:
            continue
        precision, recall, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        color = colors.get(name, "#f4a261")
        ax.plot(recall, precision, linewidth=2, label=f"{name} (AP={ap:.3f})", color=color)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Metrics comparison bar chart ---
    ax = axes[2]
    metric_keys = ["AUC", "Gini", "KS", "CV AUC", "Brier"]
    model_names = list(all_metrics.keys())
    x = np.arange(len(metric_keys))
    width = 0.8 / len(model_names)

    for i, name in enumerate(model_names):
        m = all_metrics[name]
        vals = [m["auc_test"], m["gini"], m["ks"], m["cv_auc_mean"], m["brier"]]
        offset = (i - len(model_names)/2 + 0.5) * width
        color = colors.get(name, "#f4a261")
        ax.bar(x + offset, vals, width, label=name, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys)
    ax.set_ylabel("Score")
    ax.set_title("Model Metrics Comparison")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("PD Model Comparison — All Three Models", fontsize=16, fontweight="bold")
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

    # --------------------------------------------------------
    # MODEL A: Logistic Regression (Baseline Statistical Model)
    # --------------------------------------------------------
    lr_model, lr_scaler, lr_probs, lr_coef, lr_metrics = build_logistic_regression(
        X_train, X_test, y_train, y_test, features
    )

    # --------------------------------------------------------
    # MODEL B: Gradient Boosting / Random Forest (ML Model)
    # --------------------------------------------------------
    gb_model, gb_base_model, _, gb_probs, gb_imp, gb_metrics = build_gradient_boosting(
        X_train, X_test, y_train, y_test, features
    )

    # --------------------------------------------------------
    # MODEL C: Neural Network (MLP Classifier)
    # --------------------------------------------------------
    nn_model, nn_scaler, nn_probs, nn_metrics = build_neural_network(
        X_train, X_test, y_train, y_test, features
    )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------
    all_metrics = {
        lr_metrics["model"]: lr_metrics,
        gb_metrics["model"]: gb_metrics,
        nn_metrics["model"]: nn_metrics,
    }
    all_probs = {
        lr_metrics["model"]: lr_probs,
        gb_metrics["model"]: gb_probs,
        nn_metrics["model"]: nn_probs,
    }

    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} {'Logistic Reg':>15} {gb_metrics['model']:>15} {'Neural Net':>15}")
    print("-" * 65)
    for metric_key, metric_label in [("auc_test", "AUC (test)"), ("gini", "Gini"),
                                       ("ks", "KS Statistic"), ("brier", "Brier Score"),
                                       ("cv_auc_mean", "CV AUC (mean)")]:
        lr_val = lr_metrics.get(metric_key, 0)
        gb_val = gb_metrics.get(metric_key, 0)
        nn_val = nn_metrics.get(metric_key, 0)
        # For Brier, lower is better
        if metric_key == "brier":
            best = min(lr_val, gb_val, nn_val)
        else:
            best = max(lr_val, gb_val, nn_val)

        lr_mark = " ★" if lr_val == best else ""
        gb_mark = " ★" if gb_val == best else ""
        nn_mark = " ★" if nn_val == best else ""
        print(f"  {metric_label:<18} {lr_val:>13.4f}{lr_mark} {gb_val:>13.4f}{gb_mark} {nn_val:>13.4f}{nn_mark}")

    # Recommend model (by AUC, then Brier as tiebreaker)
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["auc_test"])
    print(f"\n  ★ RECOMMENDED: {best_name} (highest AUC, best discrimination)")

    # --------------------------------------------------------
    # SHAP EXPLAINABILITY (on tree-based model)
    # --------------------------------------------------------
    if gb_base_model is not None:
        shap_df = generate_shap_analysis(
            gb_base_model,
            X_train.values if hasattr(X_train, 'values') else X_train,
            X_test.values if hasattr(X_test, 'values') else X_test,
            features
        )

    # --------------------------------------------------------
    # CALIBRATION ANALYSIS (all models)
    # --------------------------------------------------------
    plot_calibration_curves(
        y_test,
        [lr_probs, gb_probs, nn_probs],
        [lr_metrics["model"], gb_metrics["model"], nn_metrics["model"]]
    )

    # --------------------------------------------------------
    # DECILE VALIDATION (recommended model)
    # --------------------------------------------------------
    recommended_probs = all_probs[best_name]
    decile_table = generate_decile_analysis(y_test, recommended_probs, best_name)

    # --------------------------------------------------------
    # GENERATE PD FOR ALL LOANS
    # --------------------------------------------------------
    print("\n[OUTPUT] Generating PD estimates for all loans...")

    # Logistic Regression (needs scaling)
    X_all_scaled = lr_scaler.transform(X)
    df["pd_logistic"] = lr_model.predict_proba(X_all_scaled)[:, 1]

    # Gradient Boosting / Random Forest
    df["pd_gb"] = gb_model.predict_proba(X.values)[:, 1]

    # Neural Network (needs scaling)
    X_all_nn_scaled = nn_scaler.transform(X)
    df["pd_neural_net"] = nn_model.predict_proba(X_all_nn_scaled)[:, 1]

    # Recommended PD
    if best_name == lr_metrics["model"]:
        df["pd_recommended"] = df["pd_logistic"]
    elif best_name == nn_metrics["model"]:
        df["pd_recommended"] = df["pd_neural_net"]
    else:
        df["pd_recommended"] = df["pd_gb"]

    # PD rating bands based on calibrated PD
    df["pd_band"] = pd.cut(
        df["pd_recommended"],
        bins=[0, 0.01, 0.03, 0.05, 0.10, 0.20, 1.0],
        labels=["AAA (<1%)", "AA (1-3%)", "A (3-5%)", "BBB (5-10%)", "BB (10-20%)", "B+ (>20%)"],
        include_lowest=True
    )

    print(f"\n  PD Distribution Summary (calibrated):")
    print(f"    Mean PD:   {df['pd_recommended'].mean():.4f} ({df['pd_recommended'].mean()*100:.2f}%)")
    print(f"    Median PD: {df['pd_recommended'].median():.4f}")
    print(f"    Max PD:    {df['pd_recommended'].max():.4f}")
    print(f"    Actual stressed rate: {df['is_stressed'].mean()*100:.2f}%")
    print(f"\n  PD Band Distribution:")
    print(df["pd_band"].value_counts().sort_index().to_string())

    # --------------------------------------------------------
    # VISUALIZATIONS
    # --------------------------------------------------------
    print("\n[PLOTS] Generating visualizations...")
    plot_model_comparison(y_test, all_probs, all_metrics)
    plot_pd_distribution(df["pd_recommended"].values, best_name, "pd_distribution.png")

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------
    output_cols = [
        "loan_account_number", "pd_logistic", "pd_gb", "pd_neural_net",
        "pd_recommended", "pd_band",
        "is_default", "is_stressed", "peak_dpd"
    ]
    df[output_cols].to_csv(MODEL_OUTPUTS_PATH, index=False)
    print(f"\n  PD outputs saved to: {MODEL_OUTPUTS_PATH}")

    # Save full dataframe with PD for downstream modules
    df.to_csv(CLEANED_DATA_PATH.replace(".csv", "_with_pd.csv"), index=False)
    print(f"  Full data with PD saved for downstream modules.")

    # --------------------------------------------------------
    # MODEL PERSISTENCE (save trained models for later use)
    # --------------------------------------------------------
    print("\n[PERSIST] Saving trained models...")
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "pd_logistic_regression.pkl"))
    joblib.dump(lr_scaler, os.path.join(MODELS_DIR, "pd_lr_scaler.pkl"))
    joblib.dump(gb_model, os.path.join(MODELS_DIR, f"pd_{gb_metrics['model'].lower().replace(' ', '_')}.pkl"))
    joblib.dump(nn_model, os.path.join(MODELS_DIR, "pd_neural_network.pkl"))
    joblib.dump(nn_scaler, os.path.join(MODELS_DIR, "pd_nn_scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "label_encoders.pkl"))
    
    # Save the absolute best recommended model generically
    if best_name == lr_metrics["model"]:
        joblib.dump(lr_model, os.path.join(MODELS_DIR, "pd_best_model.pkl"))
        joblib.dump(lr_scaler, os.path.join(MODELS_DIR, "pd_best_scaler.pkl"))
    elif best_name == nn_metrics["model"]:
        joblib.dump(nn_model, os.path.join(MODELS_DIR, "pd_best_model.pkl"))
        joblib.dump(nn_scaler, os.path.join(MODELS_DIR, "pd_best_scaler.pkl"))
    else:
        # Tree models don't use scalers
        joblib.dump(gb_model, os.path.join(MODELS_DIR, "pd_best_model.pkl"))
        # Dump a dummy scaler placeholder for consistency
        joblib.dump("NO_SCALER", os.path.join(MODELS_DIR, "pd_best_scaler.pkl"))
        
    print(f"  Models saved to: {MODELS_DIR}/")

    print("\n" + "=" * 60)
    print(f"PD MODELLING COMPLETE — Recommended: {best_name}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
