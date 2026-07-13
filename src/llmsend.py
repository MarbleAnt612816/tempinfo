"""
src/llm_translator.py
Day 7: Integrates the Anthropic SDK to securely pass diagnostic telemetry summaries.
"""
import os
from dotenv import load_dotenv
from anthropic import Anthropic

# Load variables from the local .env file
load_dotenv()

def compile_llm_prompt(summary_dict: dict) -> str:
    """Unpacks the stats_packaging nested dictionary structure into a prompt string."""
    stats = summary_dict.get("sensor_stats", {})
    verdict_data = summary_dict.get("model_verdict", {})
    
    # Safely unpack dictionary sub-keys to prevent NoneType attribute crashes
    cpu_temp = stats.get("cpu_temp", {}).get("max", "N/A")
    cpu_1_low = stats.get("cpu_clock", {}).get("one_percent_low", "N/A")
    cpu_avg_clk = stats.get("cpu_clock", {}).get("avg", "N/A")
    gpu_hotspot = stats.get("gpu_hotspot", {}).get("max", "N/A")
    gpu_delta = stats.get("gpu_hotspot_edge_delta", {}).get("max", "N/A")
    
    overall_verdict = verdict_data.get("overall_verdict", "UNKNOWN").upper()
    
    prompt = f"""
You are an expert PC hardware diagnostic assistant. Translate this telemetry into a simple, non-technical paragraph:
- Classifier Verdict: {overall_verdict}
- Peak CPU: {cpu_temp}°C (Avg Clock: {cpu_avg_clk} MHz, 1% Low: {cpu_1_low} MHz)
- Peak GPU Hotspot: {gpu_hotspot}°C (Max Core-to-Hotspot Delta: {gpu_delta}°C)

Rules: 
1. Output exactly one cohesive paragraph. Do not use bullet points, tables, markdown headers, or raw json structures. 
2. Tell the user in plain English if their computer is healthy, running under standard heavy stress, or experiencing dangerous overheating. Explain to the user how this affects them in their real life (e.g. framing drops, safety shutoffs).
3. If the verdict is WARNING or BAD, use a simple real-world analogy to explain the specific offending metric (e.g., if the GPU Hotspot Delta is over 15-20°C, explain that the cooling paste might be drying up like old glue; if the 1% low CPU clock is significantly below average, explain that the processor is deliberately slamming on the brakes to keep from melting). 
4. Provide 1 or 2 concrete, household solutions (e.g., clearing dust from side vents, shifting the case off thick carpet, or verifying fans are spinning). 
5. Link a helpful tutorial video reference if actionable hardware adjustments are required.
"""
    return prompt

def generate_report_via_api(prompt: str) -> str:
    """Retrieves the API key securely from environment variables and calls Claude Haiku 4.5."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return "Fallback: API key missing. Check your local .env configuration settings."

    try:
        # Initialize official Anthropic client wrapper
        client = Anthropic()
        
        # CORRECTED OFFICIAL STR INTERFACE IDENTIFIER FOR HAIKU 4.5
        message = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=300,
            temperature=0.4, 
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    except Exception as e:
        return f"An error occurred while communicating with the AI platform: {str(e)}"