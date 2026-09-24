"""
generate_table2.py - Software control-path latency including Redis Streams.

Measures: edge INT8 inference → local identity enrichment → HMAC principal_id
→ Redis XADD + XREADGROUP → DQN action selection → SOAR playbook construction
using the *selected* DQN action name.

Excludes dataplane enforcement. Requires Redis 7.x on localhost:6379.
Cloud-bound payloads carry principal_id only (no cleartext IP/MAC).
"""

from __future__ import annotations

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
sys.path.insert(0, os.path.dirname(__file__))

from prototype.cloud.policy_engine.drl_sdn_agent import DRLResilienceAgent
from prototype.data.data_loader import resolve_real_windows_npz
from prototype.edge.detector.quantize import quantize_tcn_gru_model
from prototype.edge.detector.tcn_gru_model import TCNGRUResilienceModel
from prototype.edge.identity_fusion.fusion import IdentityFusionEngine
from prototype.edge.privacy.pseudonymize import assert_no_raw_endpoint
from prototype.edge.redis_stream.pubsub import require_redis
from prototype.soar.playbooks import SOARPlaybooks
from train_utils import blocked_split_npz, set_global_seed

CSV_PATH = os.path.join(BASE_DIR, "results", "table2_latency.csv")
PNG_PATH = os.path.join(BASE_DIR, "results", "table2_latency.png")
FIG_PATH = os.path.join(BASE_DIR, "paper", "figures", "fig2_latency_breakdown.png")
CKPT_PATH = os.path.join(BASE_DIR, "results", "checkpoints", "tcn_gru_ae_trained.pt")


def _cloud_event(ctx: dict, pkt_rate: int, p_attack: float) -> dict:
    event = {
        "principal_id": ctx["principal_id"],
        "pkt_rate": pkt_rate,
        "p_attack": p_attack,
        "role": ctx["role"],
        "trust_score": float(ctx["trust_score"]),
        "device": ctx["device"],
    }
    assert_no_raw_endpoint(event)
    return event


def run_real_latency_benchmark(num_trials: int = 1000, seed: int = 42, warmups: int = 50):
    print(f"[*] Latency benchmark with Redis Streams + HMAC principals ({num_trials} trials)...")
    set_global_seed(seed)
    torch.set_num_threads(1)

    bus = require_redis()
    bus.reset_stream()

    npz = resolve_real_windows_npz("insdn")
    if npz is None:
        raise FileNotFoundError("insdn_windows.npz required for latency microbenchmark")
    # Same blocked+purged protocol as Table IV+; use one real test window (not randn).
    _, _, X_te, _, _ = blocked_split_npz(npz, seq_len=10, seed=seed)
    sample_tensor = X_te[:1].contiguous()
    print(f"  [=] Prepared window: real InSDN X_te[0] shape={tuple(sample_tensor.shape)}")

    fp32_model = TCNGRUResilienceModel(num_features=10, num_classes=6)
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"Trained detector checkpoint required: {CKPT_PATH}")
    state = torch.load(CKPT_PATH, map_location="cpu")
    fp32_model.load_state_dict(state)
    fp32_model.eval()
    quantized_model = quantize_tcn_gru_model(fp32_model)
    quantized_model.eval()
    print(f"  [=] Loaded trained detector checkpoint: {os.path.basename(CKPT_PATH)}")
    fusion = IdentityFusionEngine()
    # Architecture-matched forward timing only; trained DQN weights live in Table IX.
    dqn_agent = DRLResilienceAgent(state_dim=5, action_dim=4)
    soar_playbooks = SOARPlaybooks()

    with torch.no_grad():
        for _ in range(warmups):
            _ = quantized_model(sample_tensor)
            ctx = fusion.enrich("10.0.1.15")
            event = _cloud_event(ctx, 5000, 0.91)
            _ = bus.publish_and_consume(event, block_ms=2000)
            state = [0.91, 1200.0, float(ctx["trust_score"]), 0.68, 0.92]
            action_id = dqn_agent.select_action(state, eval_mode=True)
            _ = soar_playbooks.execute_playbook(dqn_agent.get_action_name(action_id), event)

    bus.reset_stream()

    edge_ms, fusion_ms, redis_ms, dqn_ms, soar_ms = [], [], [], [], []

    for i in range(num_trials):
        src_ip = "10.0.1.15" if (i % 5) else "10.0.2.105"

        t0 = time.perf_counter()
        with torch.no_grad():
            _ = quantized_model(sample_tensor)
        edge_ms.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        ctx = fusion.enrich(src_ip)
        fusion_ms.append((time.perf_counter() - t0) * 1000.0)

        event = _cloud_event(ctx, 5000 + (i % 100), 0.91)

        t0 = time.perf_counter()
        _ = bus.publish_and_consume(event, block_ms=2000)
        redis_ms.append((time.perf_counter() - t0) * 1000.0)

        state = [0.91, 1200.0, float(ctx["trust_score"]), 0.68, 0.92]
        t0 = time.perf_counter()
        action_id = dqn_agent.select_action(state, eval_mode=True)
        action_name = dqn_agent.get_action_name(action_id)
        dqn_ms.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        _ = soar_playbooks.execute_playbook(action_name, event)
        soar_ms.append((time.perf_counter() - t0) * 1000.0)

    stages = [
        "Edge TCN-GRU Inference",
        "Identity Context Fusion",
        "Redis Streams (XADD+XREADGROUP)",
        "Cloud DQN Policy",
        "SOAR Playbook Construction",
    ]
    stage_samples = (edge_ms, fusion_ms, redis_ms, dqn_ms, soar_ms)
    medians = [float(np.median(x)) for x in stage_samples]
    p95s = [float(np.percentile(x, 95)) for x in stage_samples]
    p99s = [float(np.percentile(x, 99)) for x in stage_samples]
    means = [float(np.mean(x)) for x in stage_samples]

    total_trials = (
        np.array(edge_ms)
        + np.array(fusion_ms)
        + np.array(redis_ms)
        + np.array(dqn_ms)
        + np.array(soar_ms)
    )
    total_median = float(np.median(total_trials))
    total_p95 = float(np.percentile(total_trials, 95))
    total_p99 = float(np.percentile(total_trials, 99))
    total_mean = float(np.mean(total_trials))
    percentages = [(m / total_median) * 100.0 for m in medians]

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Pipeline Stage", "Median (ms)", "p95 (ms)", "p99 (ms)", "Share of Median Total (%)"]
        )
        for stage, med, p95, p99, pct in zip(stages, medians, p95s, p99s, percentages):
            writer.writerow([stage, f"{med:.4f}", f"{p95:.4f}", f"{p99:.4f}", f"{pct:.1f}"])
        writer.writerow(
            ["Control-Path Total", f"{total_median:.4f}", f"{total_p95:.4f}", f"{total_p99:.4f}", "100.0"]
        )

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    colors = ["#2b5c8f", "#4682b4", "#5b8c5a", "#6897bb", "#d9534f"]
    y = np.arange(len(stages))
    err_up = [p - m for p, m in zip(p95s, medians)]
    ax.barh(
        y,
        medians,
        xerr=[[0] * len(medians), err_up],
        color=colors,
        edgecolor="black",
        height=0.55,
        error_kw=dict(ecolor="#1A252C", lw=1.8, capsize=6, capthick=1.8),
    )
    ax.set_yticks(y)
    ax.set_yticklabels(stages, fontsize=7)
    ax.set_xlabel("Latency (ms), bar = median, whisker = p95")
    ax.set_title(
        f"Control-path latency with Redis + HMAC "
        f"(median {total_median:.3f} ms, p99 {total_p99:.3f} ms)"
    )
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    for i, (m, p) in enumerate(zip(medians, p95s)):
        ax.text(p + max(p95s) * 0.02, i, f"{m:.3f}", va="center", fontsize=7, fontweight="bold")
    ax.set_xlim(0, max(p95s) * 1.45)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    plt.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(
        f"  [+] Total: median {total_median:.4f} ms, p95 {total_p95:.4f} ms, p99 {total_p99:.4f} ms"
    )
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
        "redis_required": True,
        "hmac_principals": True,
    }


if __name__ == "__main__":
    run_real_latency_benchmark()
