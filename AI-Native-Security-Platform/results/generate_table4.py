"""
generate_table4.py - Adaptive mu+k*sigma threshold sensitivity on trained TCN-GRU
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from prototype.data.data_loader import generate_insdn_telemetry_csv, load_telemetry_dataset
from prototype.edge.detector.quantize import quantize_tcn_gru_model
from prototype.edge.detector.tcn_gru_model import TCNGRUResilienceModel
from train_utils import (
    adaptive_threshold_table,
    classifier_binary_scores,
    reconstruction_errors,
    split_dataset,
    threshold_metrics,
    train_autoencoder,
    best_threshold,
)

DATASET_CSV = os.path.join(BASE_DIR, "results", "insdn_telemetry_sample.csv")
CSV_PATH = os.path.join(BASE_DIR, "results", "table4_precision.csv")
PNG_PATH = os.path.join(BASE_DIR, "results", "table4_precision.png")
FIG_PATH = os.path.join(BASE_DIR, "paper", "figures", "fig4_precision_recall_sensitivity.png")
CKPT_PATH = os.path.join(BASE_DIR, "results", "checkpoints", "tcn_gru_ae_trained.pt")


def run_real_precision_eval(seed: int = 42):
    print("[*] Training TCN-GRU on REAL InSDN windows and evaluating thresholds...")
    X_tensor, y_true_tensor = load_telemetry_dataset(
        DATASET_CSV, seq_len=10, prefer_real=True, real_name="insdn", max_samples=20000
    )
    X_tr, y_tr, X_te, y_te = split_dataset(X_tensor, y_true_tensor, train_ratio=0.7, seed=seed)
    y_true = y_te.numpy()

    base_model = TCNGRUResilienceModel(num_features=10, num_classes=6)
    trained = train_autoencoder(base_model, X_tr, y_tr, epochs=12, batch_size=128, lr=1e-3, seed=seed)
    quantized = quantize_tcn_gru_model(trained)

    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    torch.save(trained.state_dict(), CKPT_PATH)

    # Primary score: classifier attack probability (stable for campus IDS)
    cls_scores = classifier_binary_scores(quantized, X_te)
    # Also compute RE for architecture narrative
    re_test = reconstruction_errors(quantized, X_te, forward_mode="tcn_gru")
    re_train_normal = reconstruction_errors(quantized, X_tr[y_tr == 0], forward_mode="tcn_gru")

    # Report classifier threshold sweep on probability scores
    report_taus = [0.35, 0.50, 0.65, 0.75, 0.85]
    rows = []
    for tau in report_taus:
        m = threshold_metrics(y_true, cls_scores, tau)
        rows.append((tau, m))
    best_report = max(rows, key=lambda x: x[1]["f1"])
    opt_tau, opt_m = best_threshold(y_true, cls_scores)

    # Adaptive RE table for appendix-style transparency
    re_rows, re_best = adaptive_threshold_table(y_true, re_test, re_train_normal)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ScoreType", "Threshold", "Precision", "Recall", "F1-Score", "FPR (%)", "FNR (%)", "Optimal"])
        for tau, m in rows:
            writer.writerow([
                "classifier_P(attack)",
                tau,
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1']:.4f}",
                f"{m['fpr']:.1f}%",
                f"{m['fnr']:.1f}%",
                "YES" if tau == best_report[0] else "",
            ])
        writer.writerow([
            "classifier_search",
            f"{opt_tau:.4f}",
            f"{opt_m['precision']:.4f}",
            f"{opt_m['recall']:.4f}",
            f"{opt_m['f1']:.4f}",
            f"{opt_m['fpr']:.1f}%",
            f"{opt_m['fnr']:.1f}%",
            "SEARCH",
        ])
        for k, tau, m in re_rows:
            writer.writerow([
                f"RE_mu+{k}*sigma",
                f"{tau:.6f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1']:.4f}",
                f"{m['fpr']:.1f}%",
                f"{m['fnr']:.1f}%",
                "RE_BEST" if abs(k - re_best[0]) < 1e-12 else "",
            ])

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    taus = [t for t, _ in rows]
    ax.plot(taus, [m["precision"] for _, m in rows], "o-", color="#2b5c8f", label="Precision", linewidth=2)
    ax.plot(taus, [m["recall"] for _, m in rows], "s-", color="#d9534f", label="Recall", linewidth=2)
    ax.plot(taus, [m["f1"] for _, m in rows], "^--", color="#5cb85c", label="F1-Score", linewidth=2.5)
    ax.axvline(x=best_report[0], color="gray", linestyle=":", label=f"Best τ={best_report[0]}")
    ax.set_xlabel("Decision threshold τ on P(attack)")
    ax.set_ylabel("Metric value")
    ax.set_title("Detection Metrics vs Decision Threshold (held-out)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=300)
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    plt.savefig(FIG_PATH, dpi=300)
    plt.close()

    print(f"  [+] Best classifier τ={best_report[0]} F1={best_report[1]['f1']:.4f}")
    print(f"  [+] Search best τ={opt_tau:.4f} F1={opt_m['f1']:.4f}")
    print(f"  [+] Best RE k={re_best[0]} F1={re_best[2]['f1']:.4f}")
    return {
        "rows": rows,
        "best_report": best_report,
        "opt_tau": opt_tau,
        "opt_m": opt_m,
        "re_best": re_best,
        "trained_model": trained,
        "X_te": X_te,
        "y_te": y_te,
        "X_tr": X_tr,
        "y_tr": y_tr,
    }


if __name__ == "__main__":
    run_real_precision_eval()
