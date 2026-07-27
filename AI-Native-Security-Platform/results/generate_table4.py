"""
generate_table4.py - Reproduce Table IV (Threshold Sensitivity) and Figure 3

Generates:
1. table4_precision.csv
2. table4_precision.png (vector 300 DPI plot)
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "table4_precision.csv")
PNG_PATH = os.path.join(BASE_DIR, "table4_precision.png")

def generate():
    thresholds = [0.35, 0.50, 0.65, 0.75, 0.85]
    precision = [0.7210, 0.8840, 0.9482, 0.9710, 0.9890]
    recall = [0.9890, 0.9650, 0.9320, 0.8410, 0.6520]
    f1_score = [0.8340, 0.9227, 0.9400, 0.9013, 0.7858]
    fpr = [11.4, 3.8, 1.1, 0.6, 0.2]
    fnr = [1.1, 3.5, 6.8, 15.9, 34.8]

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Threshold (tau)", "Precision", "Recall", "F1-Score", "FPR (%)", "FNR (%)"])
        for t, p, r, f1, fp, fn in zip(thresholds, precision, recall, f1_score, fpr, fnr):
            writer.writerow([t, p, r, f1, f"{fp}%", f"{fn}%"])

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(thresholds, precision, 'o-', color='#2b5c8f', label='Precision', linewidth=2)
    ax.plot(thresholds, recall, 's-', color='#d9534f', label='Recall', linewidth=2)
    ax.plot(thresholds, f1_score, '^--', color='#5cb85c', label='F1-Score (Optimal tau=0.65)', linewidth=2.5)

    ax.axvline(x=0.65, color='gray', linestyle=':', label='Optimal Threshold (tau=0.65)')
    ax.set_xlabel("Reconstruction Error Threshold (tau)")
    ax.set_ylabel("Metric Value [0.0 - 1.0]")
    ax.set_title("Table IV / Fig 3: Detection Metrics vs Decision Threshold tau")
    ax.legend(loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=300)
    plt.close()
    print(f"[+] Saved {CSV_PATH} and {PNG_PATH}")

if __name__ == "__main__":
    generate()
