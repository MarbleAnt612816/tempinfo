"""
src/feature_engineering.py
Handles data enrichment and thermal feature extraction matching the final dataset schema.
"""
import pandas as pd

def engineer_features(df):
    """
    Applies feature engineering on streaming or historical data using 
    exact column headers from Final_plugged.csv.
    """
    # Create a copy to prevent slicing warnings
    df = df.copy()
    
    # Mathematically isolate the cooler/paste breakdown delta using exact header casing
    if "GpuHotspot" in df.columns and "GpuEdge" in df.columns:
        df['GpuHotspotDelta'] = df['GpuHotspot'] - df['GpuEdge']
    else:
        # Fallback tracking if columns aren't present in streaming chunk
        df['GpuHotspotDelta'] = 0.0
        
    return df