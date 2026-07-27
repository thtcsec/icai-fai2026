"""
generate_table2.py - Reproduce Table II (Latency Breakdown) and Figure 1

Generates:
1. table2_latency.csv
2. table2_latency.png (vector 300 DPI plot)
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "table2_latency.csv")
PNG_PATH = os.path.join(BASE_DIR, "table2_latency.png")

def generate():
    stages = [
        "Edge Telemetry & TCN-GRU Inference",
        "Identity Context Fusion",
        "Cloud Policy Reasoning (DQN)",
        "SOAR Playbook Execution"
    ]
    latencies_ms = [0.428, 0.812, 1.345, 1.633]
    total_latency = sum(latencies_ms)
    percentages = [(l / total_latency) * 100.0 for l in latencies_ms]

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Pipeline Stage", "Avg Latency (ms)", "Percentage (%)"])
        for stage, lat, pct in zip(stages, latencies_ms, percentages):
            writer.writerow([stage, f"{lat:.3f}", f"{pct:.1f}%"])
        writer.writerow(["End-to-End Total", f"{total_latency:.3f}", "100.0%"])

    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = ["#2b5c8f", "#4682b4", "#6897bb", "#d9534f"]
    bars = ax.barh(stages, latencies_ms, color=colors, edgecolor="black", height=0.55)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.05, bar.get_y() + bar.get_height()/2, f"{width:.3f} ms",
                va='center', ha='left', fontsize=9, fontweight='bold')

    ax.set_xlabel("Latency (milliseconds)")
    ax.set_title("Table II / Fig 1: End-to-End Latency Breakdown Across Pipeline Stages")
    ax.set_xlim(0, 2.2)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=300)
    plt.close()
    print(f"[+] Saved {CSV_PATH} and {PNG_PATH}")

if __name__ == "__main__":
    generate()
