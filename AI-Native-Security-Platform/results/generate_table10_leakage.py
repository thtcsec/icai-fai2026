"""generate_table10_leakage.py - Quantify how much the unsound protocol inflates results.

Reruns the detector under the *flawed* evaluation protocol (i.i.d. random split over
stride-1 windows, StandardScaler fit on the pooled corpus) and contrasts it with the
leakage-controlled protocol (temporally blocked, purged, train-only scaler) used in
train_utils.blocked_split_npz. Everything else -- architecture, seed, epochs, quantization,
scoring head -- is held fixed, so the delta isolates the split protocol.
"""

import csv
import json
import os
import sys

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

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
    set_global_seed,
    threshold_metrics,
    train_autoencoder,
)

CSV_PATH = os.path.join(BASE_DIR, "results", "table10_leakage_effect.csv")
JSON_PATH = os.path.join(BASE_DIR, "results", "table10_leakage_effect.json")

TAU_FIXED = 0.65
MAX_TRAIN, MAX_TEST = 14000, 6000
SEQ_LEN = 10


def leaky_split_npz(npz_path, seq_len=SEQ_LEN, seed=42):
    """The unsound protocol: pooled scaler + i.i.d. random split over overlapping windows."""
    data = np.load(npz_path)
    X = data["X"].astype(np.float32)
    y = (data["y"] > 0).astype(np.int64)

    n, t, f = X.shape
    flat = np.nan_to_num(X.reshape(-1, f), nan=0.0, posinf=0.0, neginf=0.0)
    # Leak #1: scaler sees the whole corpus, including the future test partition.
    flat = StandardScaler().fit_transform(flat).astype(np.float32)
    X = flat.reshape(n, t, f)

    # Leak #2: random split places windows overlapping by seq_len-1 on both sides.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    tr_idx, te_idx = idx[:MAX_TRAIN], idx[MAX_TRAIN:MAX_TRAIN + MAX_TEST]

    to_t = lambda a: torch.tensor(X[a], dtype=torch.float32)
    return to_t(tr_idx), torch.tensor(y[tr_idx]), to_t(te_idx), torch.tensor(y[te_idx])


def _eval(X_tr, y_tr, X_te, y_te, seed=42):
    fp32 = train_autoencoder(TCNGRUResilienceModel(), X_tr, y_tr, epochs=12, seed=seed)
    int8 = quantize_tcn_gru_model(fp32)
    y_true = y_te.numpy()
    scores = classifier_binary_scores(int8, X_te)
    fixed = threshold_metrics(y_true, scores, TAU_FIXED)
    opt_tau, opt = best_threshold(y_true, scores)
    return fixed, opt_tau, opt


def run(seed: int = 42):
    npz = resolve_real_windows_npz("insdn")
    if npz is None:
        raise SystemExit("Real InSDN window archive not found.")

    set_global_seed(seed)
    print("[*] Protocol A: leaky (random split + pooled scaler)...")
    a_fixed, a_tau, a_opt = _eval(*leaky_split_npz(npz, seed=seed), seed=seed)

    set_global_seed(seed)
    print("[*] Protocol B: leakage-controlled (blocked + purged + train-only scaler)...")
    X_tr, y_tr, X_te, y_te, _ = blocked_split_npz(npz, seq_len=SEQ_LEN, seed=seed)
    b_fixed, b_tau, b_opt = _eval(X_tr, y_tr, X_te, y_te, seed=seed)

    rows = [
        ("Random split + pooled scaler (unsound)", a_fixed, a_tau, a_opt),
        ("Blocked + purged + train-only scaler", b_fixed, b_tau, b_opt),
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Protocol", "F1 @ tau=0.65", "Precision @ 0.65", "Recall @ 0.65",
                    "FPR (%) @ 0.65", "tau*", "F1 @ tau*"])
        for name, fx, tau, op in rows:
            w.writerow([name, f"{fx['f1']:.4f}", f"{fx['precision']:.4f}", f"{fx['recall']:.4f}",
                        f"{fx['fpr']:.2f}", f"{tau:.4f}", f"{op['f1']:.4f}"])

    out = {
        "leaky": {"fixed": a_fixed, "opt_tau": float(a_tau), "opt": a_opt},
        "controlled": {"fixed": b_fixed, "opt_tau": float(b_tau), "opt": b_opt},
        "delta_f1_fixed": a_fixed["f1"] - b_fixed["f1"],
        "delta_f1_opt": a_opt["f1"] - b_opt["f1"],
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n  leaky      : F1@0.65={a_fixed['f1']:.4f}  tau*={a_tau:.4f}  F1@tau*={a_opt['f1']:.4f}")
    print(f"  controlled : F1@0.65={b_fixed['f1']:.4f}  tau*={b_tau:.4f}  F1@tau*={b_opt['f1']:.4f}")
    print(f"  inflation  : fixed={out['delta_f1_fixed']:+.4f}  optimal={out['delta_f1_opt']:+.4f}")
    return out


if __name__ == "__main__":
    run()
