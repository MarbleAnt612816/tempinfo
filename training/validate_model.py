"""
validate_model.py

Day 3 validation:
1. Group K-Fold cross-validation (grouped by source_file) -- gives a more
   reliable accuracy estimate than a single train/test split, since a
   single split's result can be lucky/unlucky depending on which runs
   happened to land in the test set.
2. Synthetic generalization check -- trains on ONLY your real logged
   data, tests on ONLY the synthetic overheat data. This answers a
   specific question: does the model recognize dangerous GPU behavior
   it never saw during training, or did it just memorize your real runs?
3. Saves the FINAL model, trained on ALL your data (real + synthetic),
   for actual use in the live app.

Usage:
    python3 validate_model.py Final.csv
"""

import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


NON_FEATURE_COLUMNS = ["data", "time", "Date", "Time", "scenario", "source_file", "label"]
RANDOM_SEED = 42
SYNTHETIC_PREFIX = "synthetic"


def _feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def _make_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["label"])
    df = df[df["label"].str.strip() != ""]
    return df


# ---------------------------------------------------------------------------
# 1. Group K-Fold cross-validation
# ---------------------------------------------------------------------------

def run_group_kfold(df: pd.DataFrame, feature_columns: list, n_splits: int = 4):
    print("=" * 70)
    print(f"GROUP K-FOLD CROSS-VALIDATION ({n_splits} folds, grouped by source_file)")
    print("=" * 70)

    groups = df["source_file"]
    n_groups = groups.nunique()
    n_splits = min(n_splits, n_groups)  # can't have more folds than groups

    gkf = GroupKFold(n_splits=n_splits)
    X = df[feature_columns]
    y = df["label"]

    fold_accuracies = []
    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        test_runs = sorted(groups.iloc[test_idx].unique())

        model = _make_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)

        print(f"\nFold {fold_i}: test runs = {test_runs}")
        print(f"  Accuracy: {acc:.3f}")

    print(f"\nMean accuracy across {n_splits} folds: {np.mean(fold_accuracies):.3f} "
          f"(std: {np.std(fold_accuracies):.3f})")
    print("A high std here means performance depends heavily on WHICH run got held out --")
    print("worth knowing if you see that, since it means results aren't fully stable yet.\n")


# ---------------------------------------------------------------------------
# 2. Synthetic generalization check
# ---------------------------------------------------------------------------

def run_synthetic_generalization_check(df: pd.DataFrame, feature_columns: list):
    print("=" * 70)
    print("SYNTHETIC GENERALIZATION CHECK")
    print("(train on REAL data only, test on SYNTHETIC overheat data only)")
    print("=" * 70)

    is_synthetic = df["source_file"].str.startswith(SYNTHETIC_PREFIX)
    real_df = df[~is_synthetic]
    synthetic_df = df[is_synthetic]

    if synthetic_df.empty:
        print("No synthetic rows found (source_file doesn't start with 'synthetic') -- skipping.")
        return

    X_train, y_train = real_df[feature_columns], real_df["label"]
    X_test, y_test = synthetic_df[feature_columns], synthetic_df["label"]

    model = _make_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\nTrained on {len(real_df)} real rows, tested on {len(synthetic_df)} synthetic rows\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Confusion matrix (rows = actual, cols = predicted):")
    print("Labels:", labels)
    print(cm)
    print()
    print("If 'bad' recall here is low, the model is NOT recognizing overheat patterns")
    print("it wasn't trained on -- it may be relying on exact values seen in your real")
    print("stress-test runs rather than the underlying relationship between sensors.\n")


# ---------------------------------------------------------------------------
# 3. Train and save the FINAL model on all available data
# ---------------------------------------------------------------------------

def train_final_model(df: pd.DataFrame, feature_columns: list, model_out: str):
    print("=" * 70)
    print("TRAINING FINAL MODEL (on all data: real + synthetic)")
    print("=" * 70)

    model = _make_model()
    model.fit(df[feature_columns], df["label"])

    joblib.dump({"model": model, "feature_columns": feature_columns}, model_out)
    print(f"\nSaved final model -> {model_out}")

    importances = pd.Series(model.feature_importances_, index=feature_columns)
    print("\nFinal feature importances:")
    print(importances.sort_values(ascending=False).to_string())


def check_final_model_on_synthetic(df: pd.DataFrame, feature_columns: list, model_out: str):
    """
    Sanity check: now that the FINAL model has been trained on real+synthetic
    combined, does it correctly predict the synthetic overheat rows it DID see
    during training? This isn't a generalization test (it saw these rows) --
    it's a basic correctness check that the extreme values are being learned
    at all, i.e. that combining the datasets actually fixed the problem shown
    by the earlier real-only-vs-synthetic check.
    """
    print("=" * 70)
    print("FINAL MODEL CHECK: predictions on synthetic rows (seen during training)")
    print("=" * 70)

    saved = joblib.load(model_out)
    model = saved["model"]

    is_synthetic = df["source_file"].str.startswith(SYNTHETIC_PREFIX)
    synthetic_df = df[is_synthetic]

    X = synthetic_df[feature_columns]
    y_true = synthetic_df["label"]
    y_pred = model.predict(X)

    print(f"\nEvaluated on {len(synthetic_df)} synthetic rows (in-sample, since final model trained on these)\n")
    print(classification_report(y_true, y_pred, zero_division=0))

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion matrix (rows = actual, cols = predicted):")
    print("Labels:", labels)
    print(cm)
    print()
    print("This should now look strong -- it confirms the model CAN learn extreme")
    print("value patterns when they're actually present in training data. It does")
    print("NOT prove generalization to overheat patterns it has never seen at all --")
    print("that remains an open limitation worth naming honestly in your writeup.\n")


def main(csv_path: str, model_out: str = "thermal_model_final.joblib"):
    df = load_data(csv_path)
    feature_columns = _feature_columns(df)

    print(f"Loaded {len(df)} rows, {len(feature_columns)} features, "
          f"{df['source_file'].nunique()} source runs\n")

    run_group_kfold(df, feature_columns)
    run_synthetic_generalization_check(df, feature_columns)
    train_final_model(df, feature_columns, model_out)
    check_final_model_on_synthetic(df, feature_columns, model_out)

if __name__ == "__main__":
    main("Final_plugged.csv")