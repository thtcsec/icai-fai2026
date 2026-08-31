"""
generate_table5.py - Edge throughput saturation and resource cost.

Two distinct quantities are measured and reported separately, because conflating
them is what makes per-inference latency look inconsistent with aggregate
throughput:

  1. Saturation throughput -- the maximum sustainable windows/s at a given batch
     size, obtained by running the detector flat out.
  2. Cost at a paced offered load -- CPU and RSS while the harness actually
     rate-limits itself to a target offered rate, so an unmet target is visible
     as a shortfall in achieved throughput rather than hidden.

No legacy-DPI baseline is reported: we have no DPI implementation to measure,
and synthesising one from a formula would not be an experimental result.
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

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from prototype.edge.detector.quantize import quantize_tcn_gru_model
from prototype.edge.detector.tcn_gru_model import TCNGRUResilienceModel
from train_utils import model_param_mb

CSV_PATH = os.path.join(BASE_DIR, "results", "table5_resource.csv")
SAT_PATH = os.path.join(BASE_DIR, "results", "table5_saturation.csv")
PNG_PATH = os.path.join(BASE_DIR, "results", "table5_resource.png")
FIG_PATH = os.path.join(BASE_DIR, "paper", "figures", "fig5_resource_overhead_scaling.png")

BATCH_SIZES = [1, 8, 32, 128, 512]
DEPLOY_BATCH = 128
TARGET_RATES = [100, 1000, 5000, 10000]


def _cpu_per_core(process: psutil.Process, interval: float) -> float:
    """Process CPU over the interval, normalised to a single logical core."""
    process.cpu_percent(interval=None)
    time.sleep(interval)
    return float(process.cpu_percent(interval=None)) / max(psutil.cpu_count(logical=True), 1)


def measure_saturation(model, batch_size: int, sustain_s: float = 2.0):
    x = torch.randn(batch_size, 10, 10)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        t0 = time.perf_counter()
        n = 0
        while time.perf_counter() - t0 < sustain_s:
            model(x)
            n += batch_size
        elapsed = time.perf_counter() - t0
    windows_per_s = n / elapsed
    ms_per_forward = (elapsed / (n / batch_size)) * 1000.0
    return windows_per_s, ms_per_forward


def measure_paced(model, target_rate: int, batch_size: int, sustain_s: float = 2.5):
    """Offer `target_rate` windows/s and report what was actually achieved."""
    process = psutil.Process(os.getpid())
    x = torch.randn(batch_size, 10, 10)
    batch_interval = batch_size / float(target_rate)

    with torch.no_grad():
        for _ in range(5):
            model(x)

        process.cpu_percent(interval=None)
        t_start = time.perf_counter()
        deadline = t_start
        n = 0
        while time.perf_counter() - t_start < sustain_s:
            deadline += batch_interval
            model(x)
            n += batch_size
            sleep_for = deadline - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
        elapsed = time.perf_counter() - t_start

    cpu = float(process.cpu_percent(interval=None)) / max(psutil.cpu_count(logical=True), 1)
    rss = process.memory_info().rss / (1024 * 1024)
    achieved = n / elapsed
    return achieved, cpu, rss


def run_real_resource_benchmark(seed: int = 42):
    print("[*] Throughput saturation and resource benchmark...")
    torch.manual_seed(seed)

    model = quantize_tcn_gru_model(TCNGRUResilienceModel())
    model.eval()
    model_mb = model_param_mb(model)

    sat_rows = []
    for bs in BATCH_SIZES:
        wps, ms = measure_saturation(model, bs)
        sat_rows.append({"batch": bs, "windows_per_s": wps, "ms_per_forward": ms})
        print(f"  [=] batch={bs:4d}: {wps:9.1f} windows/s ({ms:.3f} ms/forward)")

    with open(SAT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Batch Size", "Saturation Throughput (windows/s)", "Latency per Forward Pass (ms)"])
        for r in sat_rows:
            w.writerow([r["batch"], f"{r['windows_per_s']:.1f}", f"{r['ms_per_forward']:.3f}"])

    rows = []
    for rate in TARGET_RATES:
        achieved, cpu, rss = measure_paced(model, rate, DEPLOY_BATCH)
        met = achieved >= 0.95 * rate
        rows.append(
            {
                "target": rate,
                "achieved": achieved,
                "cpu": cpu,
                "rss": rss,
                "met": met,
            }
        )
        flag = "ok" if met else "SHORTFALL"
        print(f"  [=] offered {rate:6d} w/s -> achieved {achieved:9.1f} w/s  CPU={cpu:5.2f}%  RSS={rss:.1f}MB  [{flag}]")

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Offered Load (windows/s)",
                "Achieved Throughput (windows/s)",
                "Edge CPU (%/core)",
                "Process RSS (MB)",
                "INT8 Model Footprint (MB)",
                "Target Met",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["target"],
                    f"{r['achieved']:.1f}",
                    f"{r['cpu']:.2f}%",
                    f"{r['rss']:.1f} MB",
                    f"{model_mb:.3f} MB",
                    "yes" if r["met"] else "no",
                ]
            )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))

    bs_labels = [str(r["batch"]) for r in sat_rows]
    ax1.bar(bs_labels, [r["windows_per_s"] for r in sat_rows], color="#2b5c8f", edgecolor="black")
    ax1.axhline(10000, color="#d9534f", linestyle="--", lw=1.5, label="10,000 windows/s")
    ax1.set_xlabel("Inference batch size")
    ax1.set_ylabel("Saturation throughput (windows/s)")
    ax1.set_yscale("log")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    ax2.plot([r["target"] for r in rows], [r["cpu"] for r in rows], "o-", color="#2b5c8f", lw=2)
    ax2.set_xlabel(f"Offered load (windows/s), batch={DEPLOY_BATCH}")
    ax2.set_ylabel("CPU utilization (% / core)")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    plt.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "saturation": sat_rows,
        "throughputs": [r["target"] for r in rows],
        "achieved": [r["achieved"] for r in rows],
        "ai_cpu": [r["cpu"] for r in rows],
        "ai_ram": [r["rss"] for r in rows],
        "target_met": [r["met"] for r in rows],
        "model_mb": model_mb,
        "deploy_batch": DEPLOY_BATCH,
    }


if __name__ == "__main__":
    run_real_resource_benchmark()
