"""
train_model.py

Trains a Random Forest classifier on the labeled sensor CSV to predict
label (good / warning / bad) from sensor readings.

KEY DECISION: split by source_file (i.e. by whole run), not by random
row. If you split randomly, rows from the SAME run end up in both train
and test sets -- since consecutive rows in one run are very similar
(e.g. temp barely changes second to second), the model can basically
"peek" at near-identical training examples and looks artificially
accurate. Splitting by whole run tests something more honest: can the
model generalize to a run it has never seen at all.

Usage:
    python3 train_model.py Final.csv
"""

import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


# Columns to exclude from training input (identifiers/metadata, not signal)
NON_FEATURE_COLUMNS = ["data", "time", "Date", "Time", "scenario", "source_file", "label"]

RANDOM_SEED = 42


def _resolve_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def split_by_source_file(df: pd.DataFrame, test_fraction: float = 0.25):
    """
    Split runs (source_file groups) into train/test, not individual rows.
    Ensures every row from a given run stays entirely in train OR test.
    """
    rng = np.random.RandomState(RANDOM_SEED)
    files = df["source_file"].unique()
    rng.shuffle(files)

    n_test = max(1, int(len(files) * test_fraction))
    test_files = set(files[:n_test])
    train_files = set(files[n_test:])

    train_df = df[df["source_file"].isin(train_files)]
    test_df = df[df["source_file"].isin(test_files)]

    print(f"Train runs ({len(train_files)}): {sorted(train_files)}")
    print(f"Test runs  ({len(test_files)}): {sorted(test_files)}")

    return train_df, test_df


def main(csv_path: str, model_out: str = "thermal_model.joblib"):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["label"])  # drop any rows you haven't labeled yet
    df = df[df["label"].str.strip() != ""]

    feature_columns = _resolve_feature_columns(df)
    print(f"Using {len(feature_columns)} feature columns: {feature_columns}\n")

    train_df, test_df = split_by_source_file(df)
    print(f"\nTrain rows: {len(train_df)}, Test rows: {len(test_df)}\n")

    X_train = train_df[feature_columns]
    y_train = train_df["label"]
    X_test = test_df[feature_columns]
    y_test = test_df["label"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",  # compensate for "warning" being rare (17 rows)
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("=== Classification Report (test = unseen runs) ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("=== Confusion Matrix ===")
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Rows = actual, Columns = predicted")
    print("Labels:", labels)
    print(cm)

    print("\n=== Feature Importance (top 10) ===")
    importances = pd.Series(model.feature_importances_, index=feature_columns)
    print(importances.sort_values(ascending=False).head(10).to_string())

    joblib.dump({"model": model, "feature_columns": feature_columns}, model_out)
    print(f"\nSaved trained model -> {model_out}")


if __name__ == "__main__":
    main("Final_plugged.csv")