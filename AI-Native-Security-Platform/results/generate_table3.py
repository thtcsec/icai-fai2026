"""
generate_table3.py - Reproduce Table III (MTTR Benchmark) and Figure 2

Generates:
1. table3_mttr.csv
2. table3_mttr.png (vector 300 DPI plot)
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "table3_mttr.csv")
PNG_PATH = os.path.join(BASE_DIR, "table3_mttr.png")

def generate():
    paradigms = ["Manual SOC Triage", "Legacy Rule SIEM", "AI-Native Autonomous"]
    mean_mttr_sec = [1512.4, 68.5, 0.0085]
    min_mttr_sec = [1020.0, 36.2, 0.0031]
    max_mttr_sec = [2980.0, 138.0, 0.0120]

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Operational Paradigm", "Mean MTTR (s)", "Min MTTR (s)", "Max MTTR (s)"])
        for p, mean_v, min_v, max_v in zip(paradigms, mean_mttr_sec, min_mttr_sec, max_mttr_sec):
            writer.writerow([p, mean_v, min_v, max_v])

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(paradigms, mean_mttr_sec, color=["#d9534f", "#f0ad4e", "#5cb85c"], edgecolor="black", width=0.45)
    ax.set_yscale("log")
    ax.set_ylabel("Mean Time to Respond - MTTR (seconds, Log Scale)")
    ax.set_title("Table III / Fig 2: Incident Response MTTR Benchmark Comparison")
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    labels = ["1,512.4 s (~25.2 min)", "68.5 s", "0.0085 s (8.5 ms)"]
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.3, label,
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylim(0.001, 10000)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=300)
    plt.close()
    print(f"[+] Saved {CSV_PATH} and {PNG_PATH}")

if __name__ == "__main__":
    generate()
