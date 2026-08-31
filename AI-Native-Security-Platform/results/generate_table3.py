"""
generate_table3.py - MTTR comparison (real AI loop vs literature-aligned baselines)
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

CSV_PATH = os.path.join(BASE_DIR, "results", "table3_mttr.csv")
PNG_PATH = os.path.join(BASE_DIR, "results", "table3_mttr.png")
FIG_PATH = os.path.join(BASE_DIR, "paper", "figures", "fig3_mttr_comparison.png")


def run_real_mttr_benchmark(num_trials: int = 500, seed: int = 42):
    print(f"[*] MTTR benchmark ({num_trials} trials)...")
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Stipulated (not measured) human/legacy baselines: truncated Gaussians whose
    # ranges bracket the timescales industry incident-response reporting associates
    # with manual triage and rule-based alerting. See the paper's provenance note.
    manual_soc_times = np.clip(np.random.normal(loc=1512.4, scale=350.0, size=num_trials), 900.0, 3200.0)
    legacy_siem_times = np.clip(np.random.normal(loc=68.5, scale=18.0, size=num_trials), 25.0, 150.0)

    fp32_model = TCNGRUResilienceModel(num_features=10, num_classes=6)
    quantized_model = quantize_tcn_gru_model(fp32_model)
    dqn_agent = DRLResilienceAgent(state_dim=5, action_dim=4)
    soar_playbooks = SOARPlaybooks()
    sample_tensor = torch.randn(1, 10, 10)

    with torch.no_grad():
        for _ in range(30):
            _ = quantized_model(sample_tensor)

    ai_native_times = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = quantized_model(sample_tensor)
        _ = dqn_agent.select_action([0.45, 1200.0, 0.78, 0.68, 0.92], eval_mode=True)
        _ = soar_playbooks.execute_playbook("TARGETED_FLOW_ISOLATION", {"src_ip": "10.0.1.15"})
        ai_native_times.append(time.perf_counter() - t0)

    ai_native_times = np.array(ai_native_times)
    paradigms = ["Manual SOC Triage", "Legacy Rule SIEM", "AI-Native Decision Path"]
    series = [manual_soc_times, legacy_siem_times, ai_native_times]
    means = [float(np.mean(s)) for s in series]
    stds = [float(np.std(s, ddof=1)) for s in series]
    mins = [float(np.min(s)) for s in series]
    maxs = [float(np.max(s)) for s in series]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Operational Paradigm", "Mean MTTR (s)", "SD (s)", "Min MTTR (s)", "Max MTTR (s)"])
        for p, mean_v, sd_v, min_v, max_v in zip(paradigms, means, stds, mins, maxs):
            writer.writerow([p, f"{mean_v:.4f}", f"{sd_v:.4f}", f"{min_v:.4f}", f"{max_v:.4f}"])

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bars = ax.bar(paradigms, means, color=["#d9534f", "#f0ad4e", "#5cb85c"], edgecolor="black", width=0.45)
    ax.set_yscale("log")
    ax.set_ylabel("Time-to-decision (seconds, log scale)")
    ax.set_title("Time-to-Decision Comparison (human baselines stipulated)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    labels = [
        f"{means[0]:.1f}±{stds[0]:.1f} s",
        f"{means[1]:.1f}±{stds[1]:.1f} s",
        f"{means[2]*1000:.2f}±{stds[2]*1000:.2f} ms",
    ]
    for bar, label in zip(bars, labels):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() * 1.35, label, ha="center", fontsize=8, fontweight="bold")
    ax.set_ylim(1e-4, 1e4)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=300)
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    plt.savefig(FIG_PATH, dpi=300)
    plt.close()

    print("  [+] MTTR results:")
    for p, m, s in zip(paradigms, means, stds):
        print(f"      - {p}: {m:.4f} ± {s:.4f} s")
    return {"paradigms": paradigms, "means": means, "stds": stds, "mins": mins, "maxs": maxs}


if __name__ == "__main__":
    run_real_mttr_benchmark()
