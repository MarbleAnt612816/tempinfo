"""
stats_packaging.py

Takes a completed analysis run (a DataFrame of sensor readings, whether
from a CSV or eventually a live-polled buffer) and produces:
  1. Summary stats per sensor: max, avg, min, 1% low, spike count
  2. Model predictions (good/warning/bad) per row, using the trained
     Random Forest, aggregated into an overall verdict
  3. A single structured dict -- this is EXACTLY what gets handed to the
     LLM prompt in the next step. Keeping this as one clean function
     means the LLM integration step doesn't need to know anything about
     pandas, sensors, or the model -- it just reads this dict.

This module doesn't care whether the DataFrame came from a finished CSV
or a live polling buffer -- same function, same output shape, either way.
That's intentional: it's the shared interface between "training data
inspection," "testing against old logs," and "the live app," so all
three paths are guaranteed to produce identically-shaped summaries.
"""

import pandas as pd
import numpy as np
import joblib
from src.feature_engineering import engineer_features

# Sensor columns we report stats on. Each key maps to a LIST of possible
# column name matchers, checked in order -- first one found wins. This
# covers both naming conventions you may have in play:
#   1. The NEW standardized names (matching Program.cs's SensorReading
#      properties exactly: CpuTemp, GpuHotspot, etc.) -- checked first,
#      since this is where the project is headed.
#   2. The OLD raw HWiNFO names (e.g. "CPU (Tctl/Tdie)") -- kept as a
#      fallback so this still works against any not-yet-renamed CSV.
# Once every CSV is confirmed renamed, the old-name entries can be
# deleted, but leaving them costs nothing and prevents silent failures
# in the meantime.
STAT_SENSOR_MATCHERS = {
    "cpu_temp": ["CpuTemp", "CPU (Tctl/Tdie)"],
    "gpu_hotspot": ["GpuHotspot", "GPU Hot Spot Temperature"],
    "gpu_edge": ["GpuEdge", "GPU Temperature"],
    "cpu_clock": ["CpuClock", "Core Clocks (avg)"],
    "gpu_clock": ["GpuClock", "GPU Shader Clock"],
}

NON_FEATURE_COLUMNS = ["data", "time", "Date", "Time", "scenario", "source_file", "label"]


def _resolve_column(df: pd.DataFrame, matchers) -> str | None:
    """Accepts either a single matcher string or a list of matchers to try
    in order. Returns the first matching column found, or None."""
    if isinstance(matchers, str):
        matchers = [matchers]
    for matcher in matchers:
        matches = [c for c in df.columns if c.startswith(matcher)]
        if matches:
            return matches[0]
    return None


def _one_percent_low(values: pd.Series) -> float:
    """Average of the bottom 1% of readings (classic '1% low' gaming metric --
    captures worst-case dips/stutters rather than just the single min, which
    could be a one-off sensor glitch)."""
    values = values.dropna().sort_values()
    if len(values) == 0:
        return float("nan")
    cutoff = max(1, int(len(values) * 0.01))
    return float(values.iloc[:cutoff].mean())


def _spike_count(values: pd.Series, window: int = 5, spike_delta: float = 10.0) -> int:
    """Count readings that jump more than spike_delta above the rolling
    average of the previous `window` readings."""
    values = values.reset_index(drop=True)
    if len(values) < window + 1:
        return 0
    rolling_avg = values.rolling(window=window, min_periods=window).mean().shift(1)
    spikes = (values - rolling_avg) >= spike_delta
    return int(spikes.fillna(False).sum())


def compute_sensor_stats(df: pd.DataFrame) -> dict:
    """Returns {sensor_key: {max, min, avg, one_percent_low, spike_count}}"""
    stats = {}
    for key, matcher in STAT_SENSOR_MATCHERS.items():
        col = _resolve_column(df, matcher)
        if col is None:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        stats[key] = {
            "max": round(float(series.max()), 1),
            "min": round(float(series.min()), 1),
            "avg": round(float(series.mean()), 1),
            "one_percent_low": round(_one_percent_low(series), 1),
            "spike_count": _spike_count(series),
        }

    # GPU hotspot-to-edge delta, as a single extra stat (not per-row here,
    # just the worst observed gap during the run)
    hotspot_col = _resolve_column(df, STAT_SENSOR_MATCHERS["gpu_hotspot"])
    edge_col = _resolve_column(df, STAT_SENSOR_MATCHERS["gpu_edge"])
    if hotspot_col and edge_col:
        delta = (df[hotspot_col] - df[edge_col]).dropna()
        if not delta.empty:
            stats["gpu_hotspot_edge_delta"] = {
                "max": round(float(delta.max()), 1),
                "avg": round(float(delta.mean()), 1),
            }

    return stats


def run_model_predictions(df: pd.DataFrame, model_path: str) -> dict:
    """
    Runs the trained model on every row of this run and summarizes the
    result: overall verdict, % of time spent in each class, and the
    timestamps (row indices, since we may not always have real timestamps)
    of the worst moments.
    """
    saved = joblib.load(model_path)
    model = saved["model"]
    feature_columns = saved["feature_columns"]

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Data is missing columns the model expects: {missing}. "
            f"Make sure this run's data has the same columns as training data "
            f"(including engineered features, if the model was trained with them)."
        )

    X = df[feature_columns]
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    class_order = list(model.classes_)

    label_counts = pd.Series(predictions).value_counts(normalize=True).round(3).to_dict()

    # Overall verdict: worst label present, weighted by how much it shows up
    # (a single stray "bad" row out of 300 shouldn't dominate the verdict the
    # same way sustained "bad" readings should)
    severity_rank = {"good": 0, "warning": 1, "bad": 2}
    present_labels = [l for l in label_counts if label_counts[l] >= 0.05]  # at least 5% of the run
    if not present_labels:
        present_labels = [max(label_counts, key=label_counts.get)]
    overall_verdict = max(present_labels, key=lambda l: severity_rank.get(l, 0))

    # Find the row with highest predicted probability of being "bad" (or the
    # most severe class available), as a concrete "worst moment" reference
    worst_idx = None
    if "bad" in class_order:
        bad_col_idx = class_order.index("bad")
        worst_idx = int(np.argmax(probabilities[:, bad_col_idx]))

    return {
        "overall_verdict": overall_verdict,
        "label_distribution": label_counts,
        "worst_row_index": worst_idx,
        "total_rows_evaluated": len(df),
    }


def build_summary(df: pd.DataFrame, model_path: str, scenario_label: str = "analysis run") -> dict:
    """
    Top-level entry point: produces the final structured dict to hand to
    the LLM. This is the ONLY function the LLM integration step should
    need to call.
    """
    # CRITICAL FIX: Automatically inject engineered features (like GpuHotspotDelta)
    # so the model prediction code has every column it expects!
    df = engineer_features(df)
    
    stats = compute_sensor_stats(df)
    predictions = run_model_predictions(df, model_path)

    return {
        "scenario_label": scenario_label,
        "duration_rows": len(df),
        "sensor_stats": stats,
        "model_verdict": predictions,
    }


if __name__ == "__main__":
    import sys
    import json

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "Final.csv"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "thermal_model_final.joblib"

    df = pd.read_csv(csv_path)

    # Quick manual test: run this against one specific source_file at a time,
    # simulating what a single "Analyze My Computer" run's data would look like
    if "source_file" in df.columns:
        for source_file in df["source_file"].unique():
            run_df = df[df["source_file"] == source_file]
            summary = build_summary(run_df, model_path, scenario_label=source_file)
            print(f"\n=== {source_file} ===")
            print(json.dumps(summary, indent=2))
    else:
        summary = build_summary(df, model_path)
        print(json.dumps(summary, indent=2))