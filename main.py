"""
main.py
The central orchestration layer. Tails live_readings.jsonl, runs ML triage,
and packages diagnostic telemetry summary stats using exact dataset headers.
"""
import os
import json
import time
import pandas as pd
import pprint

from src.stats_packaging import build_summary

MODEL_PATH = "training/thermal_model_final.joblib"
READINGS_PATH = "sensor-bridge/live_readings.jsonl"

def monitor_examination_window(duration_seconds=10):
    """Watches the C# streaming telemetry vector for the assessment window."""
    print(f"🎬 Initializing system examination engine ({duration_seconds}s scan)...")
    
    # Wipe out old historic log data when a fresh scan starts
    if os.path.exists(READINGS_PATH):
        try:
            open(READINGS_PATH, 'w').close()
        except IOError:
            pass # File currently locked by running C# stream
            
    time.sleep(duration_seconds)
    
    # Collect streamed records
    records = []
    if not os.path.exists(READINGS_PATH):
        print(f"❌ Error: Could not locate telemetry stream at {READINGS_PATH}")
        return None
        
    with open(READINGS_PATH, 'r') as f:
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

def execute_diagnostic_pipeline():
    # 1. Read live streamed telemetry from the background C# engine
    df_raw = monitor_examination_window(duration_seconds=10) # Set to 300 for a true 5-min scan!
    if df_raw is None:
        return
        
    # 2. Package everything using our stats packaging engine
    summary_dict = build_summary(df_raw, MODEL_PATH, scenario_label="Live Live Diagnostic Scan")
    
    print("\n📦 Structured Summary Package Generated for LLM Component:")
    pprint.pprint(summary_dict)
    
    return summary_dict

if __name__ == "__main__":
    execute_diagnostic_pipeline()