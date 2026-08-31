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
    # The per-trial latency distribution is heavy-tailed: OS scheduling and GC
    # pauses produce rare trials orders of magnitude above the mode, which makes
    # the mean and SD unstable across runs and not reproducible. We therefore
    # report percentiles, which is also the conventional way to state a latency
    # budget for a control loop.
    stage_samples = (edge_ms, fusion_ms, dqn_ms, soar_ms)
    medians = [float(np.median(x)) for x in stage_samples]
    p95s = [float(np.percentile(x, 95)) for x in stage_samples]
    p99s = [float(np.percentile(x, 99)) for x in stage_samples]
    means = [float(np.mean(x)) for x in stage_samples]

    total_trials = np.array(edge_ms) + np.array(fusion_ms) + np.array(dqn_ms) + np.array(soar_ms)
    total_median = float(np.median(total_trials))
    total_p95 = float(np.percentile(total_trials, 95))
    total_p99 = float(np.percentile(total_trials, 99))
    total_mean = float(np.mean(total_trials))
    percentages = [(m / total_median) * 100.0 for m in medians]

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Pipeline Stage", "Median (ms)", "p95 (ms)", "p99 (ms)", "Share of Median Total (%)"])
        for stage, med, p95, p99, pct in zip(stages, medians, p95s, p99s, percentages):
            writer.writerow([stage, f"{med:.4f}", f"{p95:.4f}", f"{p99:.4f}", f"{pct:.1f}%"])
        writer.writerow(
            ["End-to-End Total", f"{total_median:.4f}", f"{total_p95:.4f}", f"{total_p99:.4f}", "100.0%"]
        )

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    colors = ["#2b5c8f", "#4682b4", "#6897bb", "#d9534f"]
    y = np.arange(len(stages))
    err_up = [p - m for p, m in zip(p95s, medians)]
    ax.barh(
        y, medians, xerr=[[0] * len(medians), err_up], color=colors, edgecolor="black", height=0.55,
        error_kw=dict(ecolor="#1A252C", lw=1.8, capsize=6, capthick=1.8),
    )
    ax.set_yticks(y)
    ax.set_yticklabels(stages, fontsize=8)
    ax.set_xlabel("Latency (ms), bar = median, whisker = p95")
    ax.set_title(f"Per-Stage Latency (Total median {total_median:.3f} ms, p95 {total_p95:.3f} ms)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    for i, (m, p) in enumerate(zip(medians, p95s)):
        ax.text(p + max(p95s) * 0.02, i, f"{m:.3f} / {p:.3f}", va="center", fontsize=8, fontweight="bold")
    ax.set_xlim(0, max(p95s) * 1.45)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    plt.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  [+] Total: median {total_median:.4f} ms, p95 {total_p95:.4f} ms, p99 {total_p99:.4f} ms")
    for s, med, p95, p99, pct in zip(stages, medians, p95s, p99s, percentages):
        print(f"      - {s}: median {med:.4f} p95 {p95:.4f} p99 {p99:.4f} ms ({pct:.1f}%)")
    return {
        "medians": medians,
        "p95s": p95s,
        "p99s": p99s,
        "means": means,
        "total_median": total_median,
        "total_p95": total_p95,
        "total_p99": total_p99,
        "total_mean": total_mean,
        "percentages": percentages,
        "stages": stages,
    }


if __name__ == "__main__":
    run_real_latency_benchmark()
