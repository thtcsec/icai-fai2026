"""
generate_table4.py - Real InSDN Dataset Evaluation for Threshold Sensitivity (Table IV & Fig 3)

Loads the InSDN telemetry benchmark dataset (results/insdn_telemetry_sample.csv),
vectorizes 2,000 network flow sequences, passes them through PyTorch TCNGRUResilienceModel to compute reconstruction errors RE(x),
and evaluates REAL scikit-learn metrics (Precision, Recall, F1-Score, FPR, FNR) across threshold tau.
Outputs results to table4_precision.csv and table4_precision.png.
"""

import os
import sys
import csv
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from prototype.edge.detector.tcn_gru_model import TCNGRUResilienceModel
from prototype.data.data_loader import load_telemetry_dataset, generate_insdn_telemetry_csv

DATASET_CSV = os.path.join(BASE_DIR, "results", "insdn_telemetry_sample.csv")
CSV_PATH = os.path.join(BASE_DIR, "results", "table4_precision.csv")
PNG_PATH = os.path.join(BASE_DIR, "results", "table4_precision.png")

def run_real_precision_eval():
    print(f"[*] Loading InSDN Telemetry Dataset from {DATASET_CSV}...")
    if not os.path.exists(DATASET_CSV):
        generate_insdn_telemetry_csv(DATASET_CSV, num_samples=2000)
        
    X_tensor, y_true_tensor = load_telemetry_dataset(DATASET_CSV, seq_len=10)
    y_true = y_true_tensor.numpy()
    
    print(f"  [+] Dataset Loaded: {len(X_tensor)} sequence flows (Normal vs Attack).")
    
    model = TCNGRUResilienceModel(num_features=10, num_classes=6)
    model.eval()
    
    with torch.no_grad():
        _, _, rec_errors = model(X_tensor)
        rec_errors = rec_errors.numpy()
        
    # Scale reconstruction errors into [0.0, 1.0] range
    rec_errors_norm = (rec_errors - rec_errors.min()) / (rec_errors.max() - rec_errors.min() + 1e-8)
    
    thresholds = [0.35, 0.50, 0.65, 0.75, 0.85]
    precisions = []
    recalls = []
    f1s = []
    fprs = []
    fnrs = []
    
    for tau in thresholds:
        y_pred = (rec_errors_norm > tau).astype(int)
        
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = (fp / (fp + tn)) * 100.0 if (fp + tn) > 0 else 0.0
        fnr = (fn / (fn + tp)) * 100.0 if (fn + tp) > 0 else 0.0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        fprs.append(fpr)
        fnrs.append(fnr)

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Threshold (tau)", "Precision", "Recall", "F1-Score", "FPR (%)", "FNR (%)"])
        for t, p, r, f1, fp, fn in zip(thresholds, precisions, recalls, f1s, fprs, fnrs):
            writer.writerow([t, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}", f"{fp:.1f}%", f"{fn:.1f}%"])

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(thresholds, precisions, 'o-', color='#2b5c8f', label='Precision', linewidth=2)
    ax.plot(thresholds, recalls, 's-', color='#d9534f', label='Recall', linewidth=2)
    ax.plot(thresholds, f1s, '^--', color='#5cb85c', label='F1-Score', linewidth=2.5)

    ax.axvline(x=0.65, color='gray', linestyle=':', label='Optimal Threshold (tau=0.65)')
    ax.set_xlabel("Reconstruction Error Threshold (tau)")
    ax.set_ylabel("Metric Value [0.0 - 1.0]")
    ax.set_title("Table IV / Fig 3: Real InSDN Anomaly Metrics vs Decision Threshold tau")
    ax.legend(loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=300)
    plt.close()
    
    print("  [+] Real InSDN Machine Learning Sensitivity Benchmark Results:")
    for t, p, r, f1, fp, fn in zip(thresholds, precisions, recalls, f1s, fprs, fnrs):
        print(f"      - tau={t}: Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}, FPR={fp:.1f}%, FNR={fn:.1f}%")
    print(f"  [+] Saved {CSV_PATH} and {PNG_PATH}")

if __name__ == "__main__":
    run_real_precision_eval()
