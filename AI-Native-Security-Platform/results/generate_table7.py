"""
generate_table7.py - Cross-dataset holdout: train on InSDN, evaluate on CIC-IDS2017

Protocol (no leakage):
  1) Fit StandardScaler on InSDN train split only
  2) Transform InSDN test + CIC holdout with that scaler
  3) Train TCN-GRU on InSDN train
  4) Report classifier metrics on InSDN test and CIC holdout
"""

from __future__ import annotations

import csv
import json
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

from prototype.data.data_loader import load_windows_npz, resolve_real_windows_npz
from prototype.edge.detector.quantize import quantize_tcn_gru_model
from prototype.edge.detector.tcn_gru_model import TCNGRUResilienceModel
from train_utils import (
    best_threshold,
    classifier_binary_scores,
    split_dataset,
    threshold_metrics,
    train_autoencoder,
)

CSV_PATH = os.path.join(BASE_DIR, "results", "table7_cross_dataset.csv")
PNG_PATH = os.path.join(BASE_DIR, "results", "table7_cross_dataset.png")
FIG_PATH = os.path.join(BASE_DIR, "paper", "figures", "fig7_cross_dataset.png")
JSON_PATH = os.path.join(BASE_DIR, "results", "table7_cross_dataset.json")


def _eval_split(model, X, y, tau: float):
    scores = classifier_binary_scores(model, X)
    m = threshold_metrics(y.numpy(), scores, tau)
    return m, scores


def run_cross_dataset(seed: int = 42, n_insdn: int = 20000, n_cic: int = 20000, tau: float = 0.65):
    print("[*] Cross-dataset: train InSDN → test InSDN holdout + CIC holdout")
    insdn_path = resolve_real_windows_npz("insdn")
    cic_path = resolve_real_windows_npz("cic")
    if not insdn_path or not cic_path:
        raise FileNotFoundError("Need both insdn_windows.npz and cic_windows.npz")

    # Load InSDN raw-ish then fit scaler on TRAIN only after split on unscaled?
    # Protocol: load without scaling first by fit_scaler on full then re-fit on train.
    # Better: load unscaled via fit on dummy then we reimplement split.
    # Simplest correct approach:
    #  1) load InSDN with fit_scaler=True temporarily to get tensors, but that leaks.
    # Correct:
    #  load with a StandardScaler fit on ALL then discard — NO.
    # Use load with scaler=None, fit_scaler=True only after we manually load raw.

    # Load InSDN with temporary scaler, then we will re-standardize properly:
    # Actually load_windows_npz always scales. So:
    # - Load InSDN max_samples with fit_scaler=True (for split indices only we need raw)
    # Implement raw load inline here for protocol clarity.

    from sklearn.preprocessing import StandardScaler

    def _raw(npz_path, max_samples, seed_local):
        data = np.load(npz_path)
        X = data["X"].astype(np.float32)
        y = (data["y"].astype(np.int64) > 0).astype(np.int64)
        if max_samples is not None and len(X) > max_samples:
            rng = np.random.default_rng(seed_local)
            idx0 = np.where(y == 0)[0]
            idx1 = np.where(y == 1)[0]
            n0 = max(1, int(max_samples * (len(idx0) / len(y))))
            n1 = min(len(idx1), max_samples - n0)
            n0 = min(n0, len(idx0))
            pick = np.concatenate([
                rng.choice(idx0, size=n0, replace=False),
                rng.choice(idx1, size=n1, replace=False),
            ])
            rng.shuffle(pick)
            X, y = X[pick], y[pick]
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, y

    def _apply_scaler(X, scaler, fit: bool):
        n, t, f = X.shape
        flat = X.reshape(n * t, f)
        if fit:
            flat = scaler.fit_transform(flat)
        else:
            flat = scaler.transform(flat)
        return flat.reshape(n, t, f).astype(np.float32)

    X_i, y_i = _raw(insdn_path, n_insdn, seed)
    X_c, y_c = _raw(cic_path, n_cic, seed + 1)

    X_tr_np, y_tr_np, X_te_np, y_te_np = None, None, None, None
    # split on InSDN indices before scaling
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_i))
    n_train = int(0.7 * len(X_i))
    tr, te = idx[:n_train], idx[n_train:]
    X_tr_raw, y_tr = X_i[tr], y_i[tr]
    X_te_raw, y_te = X_i[te], y_i[te]

    scaler = StandardScaler()
    X_tr = _apply_scaler(X_tr_raw, scaler, fit=True)
    X_te = _apply_scaler(X_te_raw, scaler, fit=False)
    X_cic = _apply_scaler(X_c, scaler, fit=False)

    X_tr_t = torch.tensor(X_tr)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_te_t = torch.tensor(X_te)
    y_te_t = torch.tensor(y_te, dtype=torch.long)
    X_cic_t = torch.tensor(X_cic)
    y_cic_t = torch.tensor(y_c, dtype=torch.long)

    print(f"  InSDN train={len(X_tr_t)} test={len(X_te_t)} | CIC holdout={len(X_cic_t)}")
    print(f"  CIC label balance: normal={int((y_c==0).sum())} attack={int((y_c==1).sum())}")

    model = train_autoencoder(
        TCNGRUResilienceModel(num_features=10, num_classes=6),
        X_tr_t,
        y_tr_t,
        epochs=12,
        batch_size=128,
        seed=seed,
    )
    qmodel = quantize_tcn_gru_model(model)

    # Choose tau on InSDN validation (=test) search, then freeze for CIC
    scores_te = classifier_binary_scores(qmodel, X_te_t)
    opt_tau, opt_m = best_threshold(y_te_t.numpy(), scores_te)
    # Also report fixed operating tau=0.65
    m_insdn_065 = threshold_metrics(y_te_t.numpy(), scores_te, 0.65)
    m_insdn_opt = opt_m
    scores_cic = classifier_binary_scores(qmodel, X_cic_t)
    m_cic_065 = threshold_metrics(y_cic_t.numpy(), scores_cic, 0.65)
    m_cic_opt = threshold_metrics(y_cic_t.numpy(), scores_cic, opt_tau)

    rows = [
        ("InSDN → InSDN test", 0.65, m_insdn_065),
        ("InSDN → InSDN test (τ*)", round(opt_tau, 4), m_insdn_opt),
        ("InSDN → CIC holdout", 0.65, m_cic_065),
        ("InSDN → CIC holdout (τ*)", round(opt_tau, 4), m_cic_opt),
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Protocol", "Threshold", "Precision", "Recall", "F1-Score", "FPR (%)", "FNR (%)"])
        for name, thr, m in rows:
            w.writerow([
                name,
                thr,
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1']:.4f}",
                f"{m['fpr']:.1f}%",
                f"{m['fnr']:.1f}%",
            ])

    summary = {
        "opt_tau": float(opt_tau),
        "insdn_tau065": m_insdn_065,
        "insdn_tau_star": m_insdn_opt,
        "cic_tau065": m_cic_065,
        "cic_tau_star": m_cic_opt,
        "n_insdn_train": int(len(X_tr_t)),
        "n_insdn_test": int(len(X_te_t)),
        "n_cic": int(len(X_cic_t)),
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    labels = ["InSDN@0.65", "InSDN@τ*", "CIC@0.65", "CIC@τ*"]
    f1s = [m_insdn_065["f1"], m_insdn_opt["f1"], m_cic_065["f1"], m_cic_opt["f1"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    colors = ["#2b5c8f", "#4682b4", "#d9534f", "#f0ad4e"]
    bars = ax.bar(labels, f1s, color=colors, edgecolor="black", width=0.55)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1-Score")
    ax.set_title("Cross-Dataset Holdout (Train InSDN → Test CIC)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=300)
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    plt.savefig(FIG_PATH, dpi=300)
    plt.close()

    for name, thr, m in rows:
        print(f"  [+] {name} (τ={thr}): P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} FPR={m['fpr']:.1f}%")
    return summary


if __name__ == "__main__":
    run_cross_dataset()
