"""
parse_hwinfo_csv.py

Parses a RAW HWiNFO64 CSV export (as produced directly by HWiNFO's
"Log to CSV file" feature) into a clean DataFrame using the project's
STANDARDIZED column names -- the same names Program.cs's SensorReading
class outputs, and the same names stats_packaging.py expects.

This means any new log you collect going forward can go straight from
"raw HWiNFO export" to "ready for training/feature engineering" through
this one function, with no separate rename step needed.

What it handles:
- HWiNFO's footer rows (a repeated header row, then a
  "System: <motherboard model>" line) -- stripped automatically.
- Column lookup by stable substring match against HWiNFO's original
  verbose names, so it survives minor HWiNFO version differences and
  the degree-symbol encoding issue seen when files get saved/reopened
  through Excel.
- Adds 'scenario', 'source_file', and an empty 'label' column, ready
  for you to fill in (matching the format used throughout this project).

Usage:
    python3 parse_hwinfo_csv.py <raw_csv> <scenario_name> <output_csv>

Example:
    python3 parse_hwinfo_csv.py idle_02.CSV idle idle_02_clean.csv
"""

import sys
import pandas as pd


# Maps STANDARDIZED output name -> substring to find in the RAW HWiNFO
# header. Matching by substring (not exact string) survives the
# degree-symbol encoding issues seen before.
RAW_COLUMN_MATCHERS = {
    "CpuTemp": "CPU (Tctl/Tdie)",
    "CpuClock": "Core Clocks (avg)",
    "CpuPackagePower": "CPU Package Power",
    "GpuHotspot": "GPU Hot Spot Temperature",
    "GpuEdge": "GPU Temperature",
    "GpuClock": "GPU Shader Clock",
    "GpuFanRpm": "GPU Fan",
}


def _find_column(header: list, matcher: str) -> int | None:
    """Return the index of the first raw header column starting with
    `matcher`, or None if not found."""
    for i, col in enumerate(header):
        if col.strip().startswith(matcher):
            return i
    return None


def parse_hwinfo_csv(input_path: str, scenario: str) -> pd.DataFrame:
    """
    Parses a raw HWiNFO CSV export into a DataFrame with standardized
    columns: Date, Time, CpuTemp, CpuClock, CpuPackagePower, GpuHotspot,
    GpuEdge, GpuClock, GpuFanRpm, scenario, source_file, label (blank).
    """
    with open(input_path, newline="", encoding="utf-8-sig") as f:
        import csv
        reader = csv.reader(f)
        header = next(reader)
        raw_rows = list(reader)

    date_idx = _find_column(header, "Date")
    time_idx = _find_column(header, "Time")
    col_idx = {name: _find_column(header, matcher) for name, matcher in RAW_COLUMN_MATCHERS.items()}

    missing = [name for name, idx in col_idx.items() if idx is None]
    if missing:
        raise ValueError(
            f"Could not find expected columns in {input_path}: {missing}. "
            f"Check RAW_COLUMN_MATCHERS against this file's actual header row -- "
            f"HWiNFO's exact sensor names can vary slightly by hardware/version."
        )
    if date_idx is None or time_idx is None:
        raise ValueError(f"Could not find Date/Time columns in {input_path}.")

    clean_rows = []
    for row in raw_rows:
        if not row or len(row) <= max(list(col_idx.values()) + [date_idx, time_idx]):
            continue  # skip malformed/short rows

        # Skip HWiNFO's footer rows: a repeated header row, then a
        # "System: <motherboard model>" line.
        if row[date_idx].strip() in ("", "Date") or row[time_idx].strip() in ("", "Time"):
            continue

        clean_row = {
            "Date": row[date_idx],
            "Time": row[time_idx],
        }
        for name, idx in col_idx.items():
            clean_row[name] = row[idx]

        clean_row["scenario"] = scenario
        clean_row["source_file"] = input_path
        clean_row["label"] = ""  # left blank for you to fill in

        clean_rows.append(clean_row)

    df = pd.DataFrame(clean_rows)

    # Convert sensor columns to numeric (they come in as strings from the CSV)
    for name in RAW_COLUMN_MATCHERS:
        df[name] = pd.to_numeric(df[name], errors="coerce")

    return df


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 parse_hwinfo_csv.py <raw_csv> <scenario_name> <output_csv>")
        sys.exit(1)

    input_path, scenario, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
    df = parse_hwinfo_csv(input_path, scenario)
    df.to_csv(output_path, index=False)
    print(f"Parsed {len(df)} rows from {input_path} -> {output_path}")
    print(f"Columns: {df.columns.tolist()}")
