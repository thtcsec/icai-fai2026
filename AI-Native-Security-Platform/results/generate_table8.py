"""
generate_table8.py - Component-wise ablation of the AI-native pipeline.

Detection quality is measured on the real InSDN test split. Identity context is
NOT present in InSDN, so it is supplied by a clearly-labelled simulated overlay
whose informativeness is controlled by rho = P(low-trust identity | attack).
rho = 0.5 makes the overlay pure noise; the sweep is written to CSV so the paper
can report a range rather than a single convenient operating point.

Transport cost is measured, not assumed: the asynchronous path appends to an
in-process stream buffer, while the synchronous path performs a real loopback
HTTP round-trip against a local server.
"""

import csv
import json
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from prototype.edge.detector.quantize import quantize_tcn_gru_model
from prototype.edge.detector.tcn_gru_model import TCNGRUResilienceModel
from prototype.edge.identity_fusion.fusion import IdentityFusionEngine
from prototype.data.data_loader import resolve_real_windows_npz
from train_utils import (
    best_threshold,
    blocked_split_npz,
    classifier_binary_scores,
    threshold_metrics,
    train_autoencoder,
)

DATASET_CSV = os.path.join(BASE_DIR, "results", "insdn_telemetry_sample.csv")
CSV_PATH = os.path.join(BASE_DIR, "results", "table8_ablation.csv")
SWEEP_PATH = os.path.join(BASE_DIR, "results", "table8_ablation_rho_sweep.csv")
PNG_PATH = os.path.join(BASE_DIR, "results", "table8_ablation.png")
FIG_PATH = os.path.join(BASE_DIR, "paper", "figures", "fig9_ablation.png")

TAU = 0.65
RISK_THRESHOLD = 0.70
W_ATTACK, W_TRUST, W_CRIT = 0.60, 0.25, 0.15
RHO_MAIN = 0.70
RHO_SWEEP = [0.50, 0.60, 0.70, 0.80, 0.90]

LOW_TRUST = 0.10
HIGH_TRUST = 0.90


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _measure_transport_ms(trials: int = 300):
    """Async in-process publish vs a real synchronous loopback HTTP round-trip."""
    stream_buffer = []
    payload = {"flow_id": "f-1", "p_attack": 0.91, "role": "STUDENT", "trust": 0.45}

    t0 = time.perf_counter()
    for _ in range(trials):
        stream_buffer.append(json.dumps(payload))
    async_ms = ((time.perf_counter() - t0) / trials) * 1000.0

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/event"
    data = json.dumps(payload).encode()
    try:
        for _ in range(20):
            urllib.request.urlopen(
                urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            ).read()
        t0 = time.perf_counter()
        for _ in range(trials):
            urllib.request.urlopen(
                urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            ).read()
        sync_ms = ((time.perf_counter() - t0) / trials) * 1000.0
    finally:
        server.shutdown()
        server.server_close()

    return async_ms, sync_ms


def _identity_overlay(y_true: np.ndarray, rho: float, seed: int) -> np.ndarray:
    """Assign a trust score per window; rho = P(low trust | attack)."""
    rng = np.random.default_rng(seed)
    draw = rng.uniform(size=len(y_true))
    is_attack = y_true == 1
    low_trust = np.where(is_attack, draw < rho, draw < (1.0 - rho))
    return np.where(low_trust, LOW_TRUST, HIGH_TRUST)


def _risk_scores(p_attack: np.ndarray, trust: np.ndarray, crit: np.ndarray, use_identity: bool) -> np.ndarray:
    if use_identity:
        return W_ATTACK * p_attack + W_TRUST * (1.0 - trust) + W_CRIT * crit
    # Without identity the trust term is unavailable; its weight folds into detection.
    return (W_ATTACK + W_TRUST) * p_attack + W_CRIT * crit


def run_ablation(seed: int = 42):
    print("[*] Component-wise ablation on REAL InSDN windows...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tr, y_tr, X_te, y_te, _ = blocked_split_npz(
        resolve_real_windows_npz("insdn"), seq_len=10, seed=seed
    )
    y_true = (y_te.numpy() > 0).astype(int)

    fp32 = train_autoencoder(TCNGRUResilienceModel(), X_tr, y_tr, epochs=12, seed=seed)
    int8 = quantize_tcn_gru_model(fp32)
    p_int8 = classifier_binary_scores(int8, X_te)
    p_fp32 = classifier_binary_scores(fp32, X_te)
    p_int8_tr = classifier_binary_scores(int8, X_tr)
    p_fp32_tr = classifier_binary_scores(fp32, X_tr)

    y_tr_bin = (y_tr.numpy() > 0).astype(int)
    rng = np.random.default_rng(seed + 1)
    crit = rng.uniform(0.0, 1.0, size=len(y_true))
    crit_tr = rng.uniform(0.0, 1.0, size=len(y_tr_bin))

    fusion = IdentityFusionEngine()
    sample = X_te[:1]
    async_ms, sync_ms = _measure_transport_ms()
    print(f"  [i] transport: async={async_ms:.4f}ms  sync-REST={sync_ms:.4f}ms")

    def _infer_ms(model, runs: int = 200) -> float:
        with torch.no_grad():
            for _ in range(20):
                model(sample)
            t0 = time.perf_counter()
            for _ in range(runs):
                model(sample)
            return ((time.perf_counter() - t0) / runs) * 1000.0

    int8_ms = _infer_ms(int8)
    fp32_ms = _infer_ms(fp32)

    t0 = time.perf_counter()
    for _ in range(2000):
        fusion.enrich("10.0.1.15")
    fusion_ms = ((time.perf_counter() - t0) / 2000) * 1000.0

    t0 = time.perf_counter()
    for i in range(2000):
        _ = W_ATTACK * 0.9 + W_TRUST * (1.0 - 0.45) + W_CRIT * 0.5
    risk_ms = ((time.perf_counter() - t0) / 2000) * 1000.0

    def evaluate(p_test, p_train, use_identity, use_risk_engine, rho):
        """Select the decision threshold on train, then apply it to test.

        Each configuration produces scores on a different scale, so comparing
        them at a shared fixed threshold would measure calibration mismatch
        rather than the contribution of the ablated component.
        """
        if use_risk_engine:
            trust_tr = _identity_overlay(y_tr_bin, rho, seed + 3)
            trust_te = _identity_overlay(y_true, rho, seed + 2)
            s_tr = _risk_scores(p_train, trust_tr, crit_tr, use_identity)
            s_te = _risk_scores(p_test, trust_te, crit, use_identity)
        else:
            s_tr, s_te = p_train, p_test

        tau, _ = best_threshold(y_tr_bin, s_tr)
        m = threshold_metrics(y_true, s_te, tau)
        m["tau"] = tau
        return m

    configs = [
        ("Full Proposed Pipeline", p_int8, p_int8_tr, True, True, int8_ms + fusion_ms + risk_ms + async_ms),
        ("--- w/o Identity Context Fusion", p_int8, p_int8_tr, False, True, int8_ms + risk_ms + async_ms),
        ("--- w/o Cloud Risk Engine (Direct Edge Actuation)", p_int8, p_int8_tr, False, False, int8_ms + async_ms),
        ("--- w/o INT8 Quantization (FP32 Baseline)", p_fp32, p_fp32_tr, True, True, fp32_ms + fusion_ms + risk_ms + async_ms),
        ("--- w/o Redis Streams (Synchronous REST Call)", p_int8, p_int8_tr, True, True, int8_ms + fusion_ms + risk_ms + sync_ms),
    ]

    rows = []
    for name, p_te, p_tr, use_id, use_risk, latency in configs:
        m = evaluate(p_te, p_tr, use_id, use_risk, RHO_MAIN)
        rows.append({"config": name, "f1": m["f1"], "fpr": m["fpr"], "latency": latency})
        print(f"  [+] {name}: F1={m['f1']:.4f} FPR={m['fpr']:.2f}% tau={m['tau']:.3f} lat={latency:.4f}ms")

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Architecture Configuration", "F1-Score", "FPR (%)", "Latency (ms)"])
        for r in rows:
            w.writerow([r["config"], f"{r['f1']:.4f}", f"{r['fpr']:.2f}", f"{r['latency']:.4f}"])

    sweep = []
    for rho in RHO_SWEEP:
        full = evaluate(p_int8, p_int8_tr, True, True, rho)
        no_id = evaluate(p_int8, p_int8_tr, False, True, rho)
        sweep.append({"rho": rho, "full_f1": full["f1"], "no_id_f1": no_id["f1"], "delta": full["f1"] - no_id["f1"]})
        print(f"  [~] rho={rho:.2f}: full F1={full['f1']:.4f}  w/o identity F1={no_id['f1']:.4f}  delta={full['f1']-no_id['f1']:+.4f}")

    with open(SWEEP_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rho = P(low trust | attack)", "Full F1", "w/o Identity F1", "Delta F1"])
        for s in sweep:
            w.writerow([f"{s['rho']:.2f}", f"{s['full_f1']:.4f}", f"{s['no_id_f1']:.4f}", f"{s['delta']:+.4f}"])

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    labels = [r["config"].replace("--- ", "") for r in rows]
    yv = np.arange(len(labels))
    ax.barh(yv, [r["f1"] for r in rows], color=["#5cb85c", "#4682b4", "#f0ad4e", "#6897bb", "#d9534f"], edgecolor="black")
    ax.set_yticks(yv)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("F1-Score")
    ax.set_xlim(0, 1.12)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    for i, r in enumerate(rows):
        ax.text(r["f1"] + 0.015, i, f"{r['f1']:.3f}", va="center", fontsize=8, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    plt.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    return {"rows": rows, "sweep": sweep, "async_ms": async_ms, "sync_ms": sync_ms}


if __name__ == "__main__":
    run_ablation()
