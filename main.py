"""
main.py
The central orchestration layer. Launches the C# sensor-bridge process,
tails live_readings.jsonl, runs ML triage, and packages diagnostic
telemetry summary stats using exact dataset headers.
"""
import os
import json
import time
import subprocess
import pandas as pd
import pprint

from src.stats_packaging import build_summary

MODEL_PATH = "training/thermal_model_final.joblib"
SENSOR_BRIDGE_DIR = "sensor-bridge"
READINGS_PATH = os.path.join(SENSOR_BRIDGE_DIR, "live_readings.jsonl")

# How long to wait for the C# process to start up and write its first
# line before we start counting the "real" analysis window. `dotnet run`
# has to JIT/build on first launch, which can take a few seconds -- this
# grace period keeps that startup time from eating into your actual
# monitoring duration.
STARTUP_GRACE_SECONDS = 15


def launch_sensor_bridge() -> subprocess.Popen:
    """
    Launches the C# sensor-bridge as a background process. Each call
    starts a FRESH process, which means Program.cs's own
    `new StreamWriter(path, append: false)` truncates live_readings.jsonl
    on its own -- this is what makes the old manual Python-side file wipe
    unnecessary (and removes the file-locking race condition that came
    with it).

    stdout/stderr are captured (not printed to your console) since
    Program.cs's DEBUG_MODE prints a line per sensor per poll, which
    would otherwise flood your terminal.
    """
    print(f"🔌 Launching sensor-bridge from {SENSOR_BRIDGE_DIR}/ ...")
    process = subprocess.Popen(
        ["dotnet", "run"],
        cwd=SENSOR_BRIDGE_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    return process


def wait_for_readings_file(timeout_seconds: int = STARTUP_GRACE_SECONDS) -> bool:
    """
    Polls for live_readings.jsonl to actually appear and contain at least
    one line, up to `timeout_seconds`. Returns True once ready, False if
    it timed out (meaning the C# process likely failed to start).
    """
    start = time.time()
    while time.time() - start < timeout_seconds:
        if os.path.exists(READINGS_PATH):
            try:
                with open(READINGS_PATH, "r") as f:
                    if f.readline().strip():
                        return True
            except IOError:
                pass  # file exists but not readable yet -- keep waiting
        time.sleep(0.5)
    return False


def monitor_examination_window(duration_seconds=10):
    """
    Launches the sensor-bridge, watches the C# streaming telemetry
    vector for the assessment window, then shuts the process down.
    """
    print(f"🎬 Initializing system examination engine ({duration_seconds}s scan)...")

    process = launch_sensor_bridge()

    try:
        ready = wait_for_readings_file()
        if not ready:
            stderr_output = process.stderr.read() if process.stderr else ""
            print(f"❌ Error: sensor-bridge did not produce readings in time.")
            if stderr_output:
                print(f"   dotnet stderr: {stderr_output.strip()[:500]}")
            print("   Common causes: .NET SDK not installed, or 'dotnet run' needs")
            print("   to be run manually once first to confirm it builds cleanly.")
            return None

        # Now count the ACTUAL analysis window, on top of whatever startup
        # time already elapsed while waiting for the file to appear.
        time.sleep(duration_seconds)

        # Collect streamed records
        records = []
        with open(READINGS_PATH, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not records:
            print("⚠️ Warning: Diagnostic stream contained no valid frames.")
            return None

        return pd.DataFrame(records)

    finally:
        # Always shut the sensor-bridge process down when we're done,
        # successful or not, so it doesn't keep running (and keep the
        # file open) in the background between scans.
        print("🔌 Stopping sensor-bridge...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()  # force-kill if it didn't stop cleanly


def execute_diagnostic_pipeline():
    # 1. Launch sensor-bridge and read live streamed telemetry from it
    df_raw = monitor_examination_window(duration_seconds=10)  # Set to 300 for a true 5-min scan!
    if df_raw is None:
        return

    # 2. Package everything using our stats packaging engine
    summary_dict = build_summary(df_raw, MODEL_PATH, scenario_label="Live Diagnostic Scan")

    print("\n📦 Structured Summary Package Generated for LLM Component:")
    pprint.pprint(summary_dict)

    return summary_dict


if __name__ == "__main__":
    execute_diagnostic_pipeline()