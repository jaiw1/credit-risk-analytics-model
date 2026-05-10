"""
Module 3C: Hard Default PD Extension (90+ DPD)
==============================================
Builds a dedicated rare-event model for hard defaults, defined as
Peak DPD >= 90. This does not replace the headline 30+ DPD stress-PD model;
it provides a conservative extension for the actual hard-default definition.

Rare-event techniques used:
  - class-weighted logistic regression
  - class-weighted random forest
  - XGBoost with scale_pos_weight
  - probability calibration
  - PR-AUC, top-decile capture, and recall-focused validation

Run: python src/03c_pd_model_hard_default.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from utils import CLEANED_DATA_PATH, FIGURES_DIR, MODELS_DIR, PD_MODEL_FEATURES


TARGET_COL = "is_default"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 3
OUTPUT_PATH = os.path.join(os.path.dirname(CLEANED_DATA_PATH), "hard_default_model_outputs.csv")
FULL_OUTPUT_PATH = CLEANED_DATA_PATH.replace(".csv", "_with_hard_default_pd.csv")


def _one_hot_encoder():
    """Create a sklearn-version-compatible OneHotEncoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def load_data():
    """Load cleaned data and split feature columns by type."""
    df = pd.read_csv(CLEANED_DATA_PATH)
    features = [f for f in PD_MODEL_FEATURES if f in df.columns]
    X = df[features].copy()
    y = df[TARGET_COL].astype(int)

    categorical = X.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]

    print("=" * 70)
    print("HARD DEFAULT PD EXTENSION — 90+ DPD RARE-EVENT MODEL")
    print("=" * 70)
    print(f"Loaded {len(df):,} loans")
    print(f"Hard defaults: {y.sum():,} ({y.mean()*100:.3f}%)")
    print(f"Features: {len(features)} total ({len(numeric)} numeric, {len(categorical)} categorical)")

    return df, X, y, features, numeric, categorical


def build_preprocessor(numeric, categorical):
    """Build common preprocessing for all hard-default models."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", _one_hot_encoder()),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])


def fit_calibrated_model(name, estimator, X_train, y_train, X_test, y_test):
    """Fit a calibrated rare-event model and return validation metrics."""
    print("\n" + "-" * 70)
    print(f"Training {name}")

    model = CalibratedClassifierCV(estimator, method="sigmoid", cv=CV_FOLDS)
    model.fit(X_train, y_train)

    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "auc_train": roc_auc_score(y_train, train_prob),
        "auc_test": roc_auc_score(y_test, test_prob),
        "pr_auc_test": average_precision_score(y_test, test_prob),
        "brier_test": brier_score_loss(y_test, test_prob),
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(estimator, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1)
    metrics["cv_auc_mean"] = cv_scores.mean()
    metrics["cv_auc_std"] = cv_scores.std()

    print(f"  Train AUC: {metrics['auc_train']:.4f}")
    print(f"  Test AUC:  {metrics['auc_test']:.4f}")
    print(f"  PR-AUC:    {metrics['pr_auc_test']:.4f}")
    print(f"  Brier:     {metrics['brier_test']:.6f}")
    print(f"  CV AUC:    {metrics['cv_auc_mean']:.4f} +/- {metrics['cv_auc_std']:.4f}")

    return model, test_prob, metrics


def top_bucket_validation(y_true, y_prob):
    """Validate default capture in the highest-risk model buckets."""
    val = pd.DataFrame({"actual": y_true.values, "hard_default_pd": y_prob})
    val["decile"] = pd.qcut(val["hard_default_pd"], 10, labels=range(1, 11), duplicates="drop")
    table = val.groupby("decile", observed=True).agg(
        loans=("actual", "count"),
        hard_defaults=("actual", "sum"),
        default_rate=("actual", "mean"),
        avg_pd=("hard_default_pd", "mean"),
        min_pd=("hard_default_pd", "min"),
        max_pd=("hard_default_pd", "max"),
    ).reset_index()

    table = table.sort_values("decile", ascending=False)
    total_defaults = table["hard_defaults"].sum()
    if total_defaults:
        table["cum_defaults_captured_pct"] = table["hard_defaults"].cumsum() / total_defaults * 100
    else:
        table["cum_defaults_captured_pct"] = 0
    return table


def plot_outputs(y_test, probs_by_model, metrics_by_model, decile_table, all_pd):
    """Create rare-event validation visuals."""
    print("\n[PLOTS] Generating hard-default validation visuals...")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    for name, prob in probs_by_model.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} AUC={metrics_by_model[name]['auc_test']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title("ROC Curve — 90+ DPD")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for name, prob in probs_by_model.items():
        precision, recall, _ = precision_recall_curve(y_test, prob)
        ax.plot(recall, precision, linewidth=2, label=f"{name} AP={metrics_by_model[name]['pr_auc_test']:.3f}")
    ax.axhline(y_test.mean(), color="black", linestyle="--", alpha=0.5, label="Base rate")
    ax.set_title("Precision-Recall — Rare Event")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    labels = list(metrics_by_model)
    aucs = [metrics_by_model[n]["auc_test"] for n in labels]
    prs = [metrics_by_model[n]["pr_auc_test"] for n in labels]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, aucs, width, label="ROC-AUC", color="#457b9d")
    ax.bar(x + width/2, prs, width, label="PR-AUC", color="#e63946")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "hard_default_model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    plot_table = decile_table.sort_values("decile")
    ax.bar(plot_table["decile"].astype(int), plot_table["default_rate"] * 100, color="#e63946")
    ax.plot(plot_table["decile"].astype(int), plot_table["avg_pd"] * 100, "o-", color="#457b9d", label="Avg PD")
    ax.set_title("Actual 90+ DPD Rate by PD Decile")
    ax.set_xlabel("Decile (10 = highest risk)")
    ax.set_ylabel("Rate (%)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.plot(
        decile_table["decile"].astype(int),
        decile_table["cum_defaults_captured_pct"],
        "o-",
        color="#e63946",
        linewidth=2,
    )
    ax.set_title("Cumulative Hard Defaults Captured")
    ax.set_xlabel("Starting from highest-risk decile")
    ax.set_ylabel("Defaults captured (%)")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "hard_default_decile_validation.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_pd, bins=60, color="#457b9d", edgecolor="white", alpha=0.85)
    ax.axvline(all_pd.mean(), color="#e63946", linestyle="--", linewidth=2, label=f"Mean={all_pd.mean():.4%}")
    ax.set_title("Hard Default PD Distribution — 90+ DPD")
    ax.set_xlabel("Predicted hard-default PD")
    ax.set_ylabel("Loan count")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "hard_default_pd_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    df, X, y, features, numeric, categorical = load_data()

    if y.sum() < 50:
        print("[WARN] Fewer than 50 hard defaults; outputs should be treated as directional only.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train positives: {y_train.sum()} / {len(y_train):,}")
    print(f"Test positives:  {y_test.sum()} / {len(y_test):,}")

    preprocessor = build_preprocessor(numeric, categorical)
    neg_pos_ratio = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    estimators = {
        "Rare Event Logistic": Pipeline([
            ("preprocess", preprocessor),
            ("model", LogisticRegression(
                class_weight="balanced",
                C=0.05,
                max_iter=2000,
                random_state=RANDOM_STATE,
            )),
        ]),
        "Balanced Random Forest": Pipeline([
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=20,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }

    try:
        from xgboost import XGBClassifier
        estimators["XGBoost Rare Event"] = Pipeline([
            ("preprocess", preprocessor),
            ("model", XGBClassifier(
                n_estimators=250,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=5.0,
                min_child_weight=5,
                scale_pos_weight=neg_pos_ratio,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ])
    except ImportError:
        print("[WARN] xgboost not installed; skipping XGBoost hard-default model.")

    models = {}
    probs = {}
    metrics = {}
    for name, estimator in estimators.items():
        model, test_prob, model_metrics = fit_calibrated_model(
            name, estimator, X_train, y_train, X_test, y_test
        )
        models[name] = model
        probs[name] = test_prob
        metrics[name] = model_metrics

    best_name = max(metrics, key=lambda n: (metrics[n]["pr_auc_test"], metrics[n]["auc_test"]))
    best_model = models[best_name]
    print("\n" + "=" * 70)
    print(f"RECOMMENDED HARD-DEFAULT MODEL: {best_name}")
    print("Selection criterion: PR-AUC first, ROC-AUC second because 90+ DPD is rare.")
    print("=" * 70)

    all_pd = best_model.predict_proba(X)[:, 1]
    df["pd_hard_default_90dpd"] = all_pd
    df["hard_default_pd_band"] = pd.cut(
        df["pd_hard_default_90dpd"],
        bins=[0, 0.001, 0.003, 0.005, 0.01, 0.02, 1.0],
        labels=["A (<0.10%)", "B (0.10-0.30%)", "C (0.30-0.50%)",
                "D (0.50-1.00%)", "E (1.00-2.00%)", "F (>2.00%)"],
        include_lowest=True,
    )

    decile_table = top_bucket_validation(y_test, probs[best_name])
    print("\nTop-bucket validation, highest risk first:")
    print(decile_table.to_string(index=False))

    plot_outputs(y_test, probs, metrics, decile_table, all_pd)

    out = df[[
        "loan_account_number",
        "pd_hard_default_90dpd",
        "hard_default_pd_band",
        "is_default",
        "is_stressed",
        "peak_dpd",
    ]].copy()
    out.to_csv(OUTPUT_PATH, index=False)
    df.to_csv(FULL_OUTPUT_PATH, index=False)

    metrics_df = pd.DataFrame(metrics.values()).sort_values(
        ["pr_auc_test", "auc_test"], ascending=False
    )
    metrics_path = os.path.join(os.path.dirname(CLEANED_DATA_PATH), "hard_default_model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    joblib.dump(best_model, os.path.join(MODELS_DIR, "pd_hard_default_90dpd_best_model.pkl"))
    joblib.dump({"features": features, "numeric": numeric, "categorical": categorical, "best_model": best_name},
                os.path.join(MODELS_DIR, "pd_hard_default_90dpd_metadata.pkl"))

    print(f"\nSaved hard-default PD outputs: {OUTPUT_PATH}")
    print(f"Saved hard-default metrics: {metrics_path}")
    print(f"Mean 90+ DPD PD: {df['pd_hard_default_90dpd'].mean()*100:.4f}%")
    print(f"Observed 90+ DPD rate: {df['is_default'].mean()*100:.4f}%")

    return df


if __name__ == "__main__":
    main()
