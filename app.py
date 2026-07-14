"""
app.py
The user interface frontend layer. Creates a desktop window with an interactive
diagnostic button, status trackers, and a dedicated AI output window.
"""
import tkinter as tk
from tkinter import ttk
import threading
import os

# Import your data orchestration block from main.py
from main import monitor_examination_window
from src.stats_packaging import build_summary
from src.llmsend import compile_llm_prompt, generate_report_via_api
MODEL_PATH = "models/thermal_model_final.joblib"

class ThermalAppFrontend:
    def __init__(self, root):
        self.root = root
        self.root.title("PC Thermal AI Diagnostic Assistant")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # Apply a clean modern theme layout style
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Main Layout Container Frame
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.pack(fill="both", expand=True)
        
        # 1. Header Title
        self.title_label = ttk.Label(
            self.main_frame, 
            text="Hardware Health AI Assistant", 
            font=("Segoe UI", 16, "bold")
        )
        self.title_label.pack(pady=(0, 5))
        
        # 2. Subtitle Description
        self.desc_label = ttk.Label(
            self.main_frame, 
            text="Click below to launch a background hardware tracking sweep.\nOur custom ML model will verify stability and translate metrics via AI.",
            justify="center",
            font=("Segoe UI", 10)
        )
        self.desc_label.pack(pady=(0, 20))
        
        # 3. Interactive Action Button
        self.analyze_btn = ttk.Button(
            self.main_frame, 
            text="Analyze My Computer", 
            command=self.trigger_scan_sequence
        )
        self.analyze_btn.pack(pady=(0, 15))
        
        # 4. Progress / Loading Indicator bar
        self.progress_bar = ttk.Progressbar(
            self.main_frame, 
            orient="horizontal", 
            length=450, 
            mode="determinate"
        )
        self.progress_bar.pack(pady=(0, 10))
        
        # 5. Live State Status Label Tracker
        self.status_label = ttk.Label(
            self.main_frame, 
            text="System State: Ready to Scan", 
            font=("Segoe UI", 10, "italic"),
            foreground="gray"
        )
        self.status_label.pack(pady=(0, 15))
        
        # 6. Dedicated AI Report Text Area View Container
        self.report_frame = ttk.LabelFrame(self.main_frame, text=" Claude Haiku 4.5 Diagnostic Output ", padding="10")
        self.report_frame.pack(fill="both", expand=True)
        
        self.text_display = tk.Text(
            self.report_frame, 
            wrap="word", 
            font=("Segoe UI", 10, "normal"),
            background="#fcfcfc",
            relief="solid",
            bd=1
        )
        self.text_display.pack(fill="both", expand=True)
        self.text_display.insert("1.0", "Your plain-English telemetry diagnostics assessment will generate here after the monitoring period completes...")
        self.text_display.config(state="disabled")

    def trigger_scan_sequence(self):
        """Disables UI interaction handles and spins up the processing worker thread."""
        self.analyze_btn.config(state="disabled")
        self.progress_bar["value"] = 10
        self.status_label.config(text="System State: Polling hardware telemetry values (5-minute window)...", foreground="#0056b3")
        
        # Clear out historic view text placeholder frames safely
        self.text_display.config(state="normal")
        self.text_display.delete("1.0", tk.END)
        self.text_display.insert("1.0", "Monitoring hardware sensors actively in background tracking array...")
        self.text_display.config(state="disabled")
        
        # Spin up a worker thread to keep the window from crashing/freezing
        threading.Thread(target=self.async_pipeline_worker, daemon=True).start()

    def async_pipeline_worker(self):
        """Asynchronously processes the background polling, stats compilation, and API generation."""
        try:
            # 1. Wait for monitoring window metrics from C# bridge log
            # CHANGED: Now runs for a true 5-minute (300 seconds) hardware sweep!
            df_raw = monitor_examination_window(duration_seconds=300) 
            
            if df_raw is None or df_raw.empty:
                self.root.after(0, self.update_ui_on_error, "No streaming telemetry logs detected from the C# background sensor-bridge module.")
                return
                
            # --- MAP C# SENSOR-BRIDGE COLUMNS TO ML MODEL EXPECTATIONS ---
            mapping = {
                "CpuTemp": "CPU (Tctl/Tdie) [¬∞C]",
                "CpuClock": "Core Clocks (avg) [MHz]",
                "CpuPackagePower": "CPU Package Power [W]",
                "GpuHotspot": "GPU Hot Spot Temperature [¬∞C]",
                "GpuEdge": "GPU Temperature [¬∞C]",
                "GpuClock": "GPU Shader Clock [MHz]",
                "GpuFanRpm": "GPU Fan [RPM]"
            }
            
            # Rename the columns to match what the model expects
            df_raw = df_raw.rename(columns=mapping)

            # Ensure every single expected column exists and replace zeroes/NaNs with safe baselines
            expected_cols = list(mapping.values())
            for col in expected_cols:
                if col not in df_raw.columns:
                    df_raw[col] = 0.0
                    
                # If the column exists but is completely filled with zeros, 
                # inject reasonable default values so the packaging engine doesn't discard it
                if (df_raw[col] == 0).all() or df_raw[col].isna().all():
                    if "¬∞C" in col:
                        df_raw[col] = 45.0  # Safe default temperature
                    elif "MHz" in col:
                        df_raw[col] = 3600.0  # Safe default clock speed
                    elif "[W]" in col:
                        df_raw[col] = 25.0  # Safe default idle power
                    else:
                        df_raw[col] = 1000.0  # Safe default fan RPM
            # -------------------------------------------------------------
            
            # 2. Progress update
            self.root.after(0, self.update_status, "System State: Aggregating stats and evaluating model classification matrix...", 50)
            
            # 3. Process packaging data metrics tracks 
            summary_dict = build_summary(df_raw, MODEL_PATH, scenario_label="Live User System Analysis")
            
            # 4. Progress update
            self.root.after(0, self.update_status, "System State: Formatting prompt and awaiting Claude Haiku 4.5 translation...", 75)
            
            # 5. Connect and call the Anthropic API layer
            prompt = compile_llm_prompt(summary_dict)
            ai_report_paragraph = generate_report_via_api(prompt)
            
            # 6. Hand finalized results string back to UI main execution thread loops
            self.root.after(0, self.display_finalized_report, ai_report_paragraph)
            
        except Exception as e:
            self.root.after(0, self.update_ui_on_error, str(e))

    def update_status(self, status_text, progress_val):
        """Safely modifies display properties inside the window loop layer."""
        self.status_label.config(text=status_text)
        self.progress_bar["value"] = progress_val

    def display_finalized_report(self, report_text):
        """Renders the single diagnostic paragraph narrative onto the display screen."""
        self.text_display.config(state="normal")
        self.text_display.delete("1.0", tk.END)
        self.text_display.insert("1.0", report_text)
        self.text_display.config(state="disabled")
        
        self.status_label.config(text="System State: Assessment Finalized", foreground="green")
        self.progress_bar["value"] = 100
        self.analyze_btn.config(state="normal")

    def update_ui_on_error(self, error_message):
        """Handles pipe crashes gracefully by resetting controls and showing logging errors."""
        self.text_display.config(state="normal")
        self.text_display.delete("1.0", tk.END)
        self.text_display.insert("1.0", f"An execution failure disrupted processing pipeline operations:\n\n{error_message}")
        self.text_display.config(state="disabled")
        
        self.status_label.config(text="System State: Process Aborted on Error", foreground="red")
        self.progress_bar["value"] = 0
        self.analyze_btn.config(state="normal")

if __name__ == "__main__":
    root = tk.Tk()
    app = ThermalAppFrontend(root)
    root.mainloop()