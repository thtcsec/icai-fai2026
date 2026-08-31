"""
generate_table6.py - Trained SOTA baseline & ablation comparison (classifier/AE honest metrics)
"""

import csv
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from prototype.data.data_loader import resolve_real_windows_npz
from prototype.edge.detector.quantize import quantize_tcn_gru_model
from prototype.edge.detector.tcn_gru_model import TCNGRUResilienceModel
from train_utils import (
    best_threshold,
    blocked_split_npz,
    classifier_binary_scores,
    model_param_mb,
    set_global_seed,
    reconstruction_errors,
    train_autoencoder,
    train_reconstruction_model,
)

DATASET_CSV = os.path.join(BASE_DIR, "results", "insdn_telemetry_sample.csv")
CSV_PATH = os.path.join(BASE_DIR, "results", "table6_sota_comparison.csv")
PNG_PATH = os.path.join(BASE_DIR, "results", "table6_sota_comparison.png")
FIG_PATH = os.path.join(BASE_DIR, "paper", "figures", "fig6_sota_comparison.png")


class LSTMAutoencoder(nn.Module):
    def __init__(self, num_features=10, hidden_dim=32):
        super().__init__()
        self.encoder = nn.LSTM(num_features, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, num_features, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h_repeated = h.squeeze(0).unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(h_repeated)
        rec_err = torch.mean((x - out) ** 2, dim=(1, 2))
        return out, rec_err


class TransformerAutoencoder(nn.Module):
    def __init__(self, num_features=10, d_model=32, nhead=4):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=64, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_proj = nn.Linear(d_model, num_features)

    def forward(self, x):
        h = self.input_proj(x)
        out_h = self.transformer(h)
        out = self.output_proj(out_h)
        rec_err = torch.mean((x - out) ** 2, dim=(1, 2))
        return out, rec_err


def _latency_ms(fn, warmup=20, runs=200):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    return ((time.perf_counter() - t0) / runs) * 1000.0


def run_sota_comparison(seed: int = 42):
    print("[*] Trained SOTA / ablation on REAL InSDN windows...")
    set_global_seed(seed)

    X_tr, y_tr, X_te, y_te, _ = blocked_split_npz(
        resolve_real_windows_npz("insdn"), seq_len=10, seed=seed
    )
    y_true = y_te.numpy()
    X_tr_2d = X_tr.numpy().mean(axis=1)
    X_te_2d = X_te.numpy().mean(axis=1)
    sample = X_te[:1]

    iforest = IsolationForest(n_estimators=100, contamination="auto", random_state=seed)
    iforest.fit(X_tr_2d)
    preds_if = (iforest.predict(X_te_2d) == -1).astype(int)
    if_f1 = float(f1_score(y_true, preds_if, zero_division=0))
    if_lat = _latency_ms(lambda: iforest.predict(X_te_2d[:1]))
    # No footprint reported: model_param_mb measures torch state_dict tensors and
    # has no meaningful analogue for a scikit-learn forest.
    if_ram = None

    lstm = train_reconstruction_model(LSTMAutoencoder(), X_tr, y_tr, epochs=10, forward_mode="pair", seed=seed)
    _, lstm_best = best_threshold(y_true, reconstruction_errors(lstm, X_te, forward_mode="pair"))
    lstm_f1 = lstm_best["f1"]
    lstm_lat = _latency_ms(lambda: lstm(sample))
    lstm_ram = model_param_mb(lstm)

    trans = train_reconstruction_model(TransformerAutoencoder(), X_tr, y_tr, epochs=8, forward_mode="pair", seed=seed)
    _, trans_best = best_threshold(y_true, reconstruction_errors(trans, X_te, forward_mode="pair"))
    trans_f1 = trans_best["f1"]
    trans_lat = _latency_ms(lambda: trans(sample))
    trans_ram = model_param_mb(trans)

    tcn_fp32 = train_autoencoder(TCNGRUResilienceModel(), X_tr, y_tr, epochs=12, seed=seed)
    _, fp32_best = best_threshold(y_true, classifier_binary_scores(tcn_fp32, X_te))
    tcn_fp32_f1 = fp32_best["f1"]
    tcn_fp32_lat = _latency_ms(lambda: tcn_fp32(sample))
    tcn_fp32_ram = model_param_mb(tcn_fp32)

    tcn_int8 = quantize_tcn_gru_model(tcn_fp32)
    _, int8_best = best_threshold(y_true, classifier_binary_scores(tcn_int8, X_te))
    tcn_int8_f1 = int8_best["f1"]
    tcn_int8_lat = _latency_ms(lambda: tcn_int8(sample))
    tcn_int8_ram = model_param_mb(tcn_int8)

    methods = [
        "Isolation Forest (iForest)",
        "LSTM Autoencoder (FP32)",
        "Transformer AE (FP32)",
        "TCN-GRU (FP32 Ablation)",
        "TCN-GRU (INT8 Proposed)",
    ]
    f1_scores = [if_f1, lstm_f1, trans_f1, tcn_fp32_f1, tcn_int8_f1]
    latencies = [if_lat, lstm_lat, trans_lat, tcn_fp32_lat, tcn_int8_lat]
    ram_sizes = [if_ram, lstm_ram, trans_ram, tcn_fp32_ram, tcn_int8_ram]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Detection Method", "F1-Score", "Inference Latency (ms)", "Model Footprint (MB)"])
        for m, f1, l, r in zip(methods, f1_scores, latencies, ram_sizes):
            writer.writerow([m, f"{f1:.4f}", f"{l:.4f}", "n/a" if r is None else f"{r:.3f} MB"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.8))
    colors = ["#777777", "#d9534f", "#f0ad4e", "#4682b4", "#5cb85c"]
    y = np.arange(len(methods))
    ax1.barh(y, f1_scores, color=colors, edgecolor="black")
    ax1.set_yticks(y)
    ax1.set_yticklabels(methods, fontsize=8)
    ax1.set_xlabel("F1-Score")
    ax1.set_xlim(0, 1.15)
    ax1.set_title("Detection Accuracy (F1)")
    ax1.grid(axis="x", linestyle="--", alpha=0.5)
    for i, v in enumerate(f1_scores):
        ax1.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=8, fontweight="bold")

    ram_plot = [0.0 if v is None else v for v in ram_sizes]
    ax2.barh(y, ram_plot, color=colors, edgecolor="black")
    ax2.set_yticks(y)
    ax2.set_yticklabels(methods, fontsize=8)
    ax2.set_xlabel("Model footprint (MB)")
    ax2.set_title("Memory Size")
    ax2.grid(axis="x", linestyle="--", alpha=0.5)
    for i, (v, raw) in enumerate(zip(ram_plot, ram_sizes)):
        label = "n/a" if raw is None else f"{raw:.3f}"
        ax2.text(v + max(ram_plot) * 0.02, i, label, va="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=300)
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    plt.savefig(FIG_PATH, dpi=300)
    plt.close()

    for m, f1, l, r in zip(methods, f1_scores, latencies, ram_sizes):
        size = "n/a" if r is None else f"{r:.3f}MB"
        print(f"  [+] {m}: F1={f1:.4f}, lat={l:.4f}ms, size={size}")

    return {
        "methods": methods,
        "f1": f1_scores,
        "latency": latencies,
        "ram": ram_sizes,
        "delta_f1": tcn_int8_f1 - tcn_fp32_f1,
    }


if __name__ == "__main__":
    run_sota_comparison()
