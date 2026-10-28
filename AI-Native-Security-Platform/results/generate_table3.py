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
from prototype.cloud.policy_engine.observation import observation_from_detector_window
from prototype.data.data_loader import resolve_real_windows_npz
from prototype.edge.detector.inference import infer_window, load_detector_checkpoint
from prototype.soar.playbooks import SOARPlaybooks

CSV_PATH = os.path.join(BASE_DIR, "results", "corrected", "table3_mttr.csv")
PNG_PATH = os.path.join(BASE_DIR, "results", "corrected", "table3_mttr.png")
FIG_PATH = os.path.join(BASE_DIR, "paper", "figures", "fig3_mttr_comparison.png")
DETECTOR_CKPT = os.path.join(BASE_DIR, "results", "checkpoints", "tcn_gru_ae_trained.pt")
DQN_CKPT = os.path.join(BASE_DIR, "results", "checkpoints", "dqn_policy_trained.pt")


def run_real_mttr_benchmark(num_trials: int = 500, seed: int = 42):
    print(f"[*] MTTR benchmark ({num_trials} trials)...")
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Stipulated (not measured) human/legacy baselines: truncated Gaussians whose
    # ranges bracket the timescales industry incident-response reporting associates
    # with manual triage and rule-based alerting. See the paper's provenance note.
    manual_soc_times = np.clip(np.random.normal(loc=1512.4, scale=350.0, size=num_trials), 900.0, 3200.0)
    legacy_siem_times = np.clip(np.random.normal(loc=68.5, scale=18.0, size=num_trials), 25.0, 150.0)

    detector, _meta = load_detector_checkpoint(DETECTOR_CKPT, quantize=True)
    dqn_agent = DRLResilienceAgent.from_checkpoint(DQN_CKPT)
    soar_playbooks = SOARPlaybooks()
    npz = resolve_real_windows_npz("insdn")
    if npz is None:
        raise FileNotFoundError("insdn_windows.npz required for MTTR software-path timing")
    data = np.load(npz)
    sample_tensor = torch.tensor(data["X"][:1], dtype=torch.float32)

    for _ in range(30):
        det = infer_window(detector, sample_tensor, checkpoint_path=DETECTOR_CKPT, quantized=True)
        obs = observation_from_detector_window(
            det.p_attack, det.reconstruction_error, sample_tensor[0, -1].tolist()
        )
        _ = dqn_agent.select_action(obs.as_list(), eval_mode=True)

    ai_native_times = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        det = infer_window(detector, sample_tensor, checkpoint_path=DETECTOR_CKPT, quantized=True)
        obs = observation_from_detector_window(
            det.p_attack, det.reconstruction_error, sample_tensor[0, -1].tolist()
        )
        action_id = dqn_agent.select_action(obs.as_list(), eval_mode=True)
        descriptor = soar_playbooks.execute_playbook(
            dqn_agent.get_action_name(action_id),
            {"principal_id": "bench-principal", "device": "BYOD"},
        )
        if descriptor.get("enforced") is True:
            raise RuntimeError("MTTR path must not claim SOAR enforcement")
        ai_native_times.append(time.perf_counter() - t0)

    ai_native_times = np.array(ai_native_times)
    paradigms = ["Manual SOC Triage", "Legacy Rule SIEM", "AI-Native Decision Path"]
    series = [manual_soc_times, legacy_siem_times, ai_native_times]
    means = [float(np.mean(s)) for s in series]
    stds = [float(np.std(s, ddof=1)) for s in series]
    mins = [float(np.min(s)) for s in series]
    maxs = [float(np.max(s)) for s in series]

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
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
    plt.close()

    print("  [+] MTTR results:")
    for p, m, s in zip(paradigms, means, stds):
        print(f"      - {p}: {m:.4f} ± {s:.4f} s")
    return {
        "paradigms": paradigms,
        "means": means,
        "stds": stds,
        "mins": mins,
        "maxs": maxs,
        "human_baselines": "stipulated_not_measured",
        "ai_path": "software_decision_time_loaded_detector_and_dqn",
        "soar_enforced": False,
    }


if __name__ == "__main__":
    run_real_mttr_benchmark()
