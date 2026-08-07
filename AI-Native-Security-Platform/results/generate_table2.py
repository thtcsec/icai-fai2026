"""
generate_table2.py - Measured pipeline latency breakdown (Mean +/- SD)
"""

import csv
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from prototype.cloud.policy_engine.drl_sdn_agent import DRLResilienceAgent
from prototype.edge.detector.quantize import quantize_tcn_gru_model
from prototype.edge.detector.tcn_gru_model import TCNGRUResilienceModel
from prototype.soar.playbooks import SOARPlaybooks

CSV_PATH = os.path.join(BASE_DIR, "results", "table2_latency.csv")
PNG_PATH = os.path.join(BASE_DIR, "results", "table2_latency.png")
FIG_PATH = os.path.join(BASE_DIR, "paper", "figures", "fig2_latency_breakdown.png")


def run_real_latency_benchmark(num_trials: int = 1000, seed: int = 42):
    print(f"[*] Latency benchmark ({num_trials} trials, seed={seed})...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    fp32_model = TCNGRUResilienceModel(num_features=10, num_classes=6)
    quantized_model = quantize_tcn_gru_model(fp32_model)
    dqn_agent = DRLResilienceAgent(state_dim=5, action_dim=4)
    soar_playbooks = SOARPlaybooks()
    sample_tensor = torch.randn(1, 10, 10)

    with torch.no_grad():
        for _ in range(50):
            _ = quantized_model(sample_tensor)

    edge_ms, fusion_ms, dqn_ms, soar_ms = [], [], [], []
    raw_event = {"src_ip": "10.0.1.15", "pkt_rate": 5000}
    state = [0.45, 1200.0, 0.78, 0.68, 0.92]

    for _ in range(num_trials):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = quantized_model(sample_tensor)
        edge_ms.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        enriched = dict(raw_event)
        enriched.update({"mac": "AA:BB:CC:DD:EE:01", "role": "STUDENT_BYOD", "trust_score": 0.45})
        fusion_ms.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        _ = dqn_agent.select_action(state, eval_mode=True)
        dqn_ms.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        _ = soar_playbooks.execute_playbook("TARGETED_FLOW_ISOLATION", enriched)
        soar_ms.append((time.perf_counter() - t0) * 1000.0)

    stages = [
        "Edge Telemetry & TCN-GRU Inference",
        "Identity Context Fusion",
        "Cloud Policy Reasoning (DQN)",
        "SOAR Playbook Execution",
    ]
    means = [float(np.mean(x)) for x in (edge_ms, fusion_ms, dqn_ms, soar_ms)]
    stds = [float(np.std(x, ddof=1)) for x in (edge_ms, fusion_ms, dqn_ms, soar_ms)]
    total_trials = np.array(edge_ms) + np.array(fusion_ms) + np.array(dqn_ms) + np.array(soar_ms)
    total_mean = float(np.mean(total_trials))
    total_std = float(np.std(total_trials, ddof=1))
    percentages = [(m / total_mean) * 100.0 for m in means]

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Pipeline Stage", "Mean Latency (ms)", "SD (ms)", "Percentage (%)"])
        for stage, mean, sd, pct in zip(stages, means, stds, percentages):
            writer.writerow([stage, f"{mean:.4f}", f"{sd:.4f}", f"{pct:.1f}%"])
        writer.writerow(["End-to-End Total", f"{total_mean:.4f}", f"{total_std:.4f}", "100.0%"])

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    colors = ["#2b5c8f", "#4682b4", "#6897bb", "#d9534f"]
    y = np.arange(len(stages))
    ax.barh(y, means, xerr=stds, color=colors, edgecolor="black", height=0.55, error_kw=dict(ecolor="#1A252C", lw=1.8, capsize=6, capthick=1.8))
    ax.set_yticks(y)
    ax.set_yticklabels(stages, fontsize=8)
    ax.set_xlabel("Latency (ms)")
    ax.set_title(f"Per-Stage Latency Breakdown (Total {total_mean:.3f} ± {total_std:.3f} ms)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + max(means) * 0.02, i, f"{m:.3f}±{s:.3f}", va="center", fontsize=8, fontweight="bold")
    ax.set_xlim(0, max(means) * 1.45)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    plt.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  [+] Total: {total_mean:.4f} ± {total_std:.4f} ms")
    for s, m, sd, p in zip(stages, means, stds, percentages):
        print(f"      - {s}: {m:.4f} ± {sd:.4f} ms ({p:.1f}%)")
    return {
        "means": means,
        "stds": stds,
        "total_mean": total_mean,
        "total_std": total_std,
        "percentages": percentages,
        "stages": stages,
    }


if __name__ == "__main__":
    run_real_latency_benchmark()
