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

# Timing on a loaded commodity host is noisy enough that a single measurement is
# not a reproducible result: repeated runs of this benchmark have varied by more
# than 1.7x at batch=128. Every timing figure below is therefore the median of
# REPEATS independent measurements, and the spread is reported alongside it so a
# reader can see how much confidence a point estimate deserves.
REPEATS = 5


def _cpu_per_core(process: psutil.Process, interval: float) -> float:
    """Process CPU over the interval, normalised to a single logical core."""
    process.cpu_percent(interval=None)
    time.sleep(interval)
    return float(process.cpu_percent(interval=None)) / max(psutil.cpu_count(logical=True), 1)


def _peak_memory_mb(process: psutil.Process) -> float:
    """Peak working set since process start, in MB.

    Instantaneous RSS is not usable here: Windows trims the working set of a
    process that idles between paced batches, so a reading taken after the loop
    reported ~20 MB for a process whose actual demand was ~300 MB, and reported
    *less* memory at higher offered load. Peak working set is monotone and
    reflects demand rather than current residency. It is dominated by the
    Python/PyTorch runtime, not by the detector.
    """
    info = process.memory_info()
    peak = getattr(info, "peak_wset", None)
    return float(peak if peak is not None else info.rss) / (1024.0 * 1024.0)


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
    rss = _peak_memory_mb(process)
    achieved = n / elapsed
    return achieved, cpu, rss


def run_real_resource_benchmark(seed: int = 42):
    print("[*] Throughput saturation and resource benchmark...")
    torch.manual_seed(seed)

    model = quantize_tcn_gru_model(TCNGRUResilienceModel())
    model.eval()
    model_mb = model_param_mb(model)

    print(f"  [i] every timing figure is the median of {REPEATS} repeats")

    sat_rows = []
    for bs in BATCH_SIZES:
        trials = [measure_saturation(model, bs) for _ in range(REPEATS)]
        wps_all = sorted(t[0] for t in trials)
        ms_all = sorted(t[1] for t in trials)
        wps, ms = float(np.median(wps_all)), float(np.median(ms_all))
        sat_rows.append(
            {
                "batch": bs,
                "windows_per_s": wps,
                "ms_per_forward": ms,
                "windows_per_s_min": wps_all[0],
                "windows_per_s_max": wps_all[-1],
            }
        )
        print(
            f"  [=] batch={bs:4d}: {wps:9.1f} windows/s (median, range "
            f"{wps_all[0]:.0f}-{wps_all[-1]:.0f}), {ms:.3f} ms/forward"
        )

    with open(SAT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Batch Size",
                f"Saturation Throughput median of {REPEATS} (windows/s)",
                "Min (windows/s)",
                "Max (windows/s)",
                "Latency per Forward Pass median (ms)",
            ]
        )
        for r in sat_rows:
            w.writerow(
                [
                    r["batch"],
                    f"{r['windows_per_s']:.1f}",
                    f"{r['windows_per_s_min']:.1f}",
                    f"{r['windows_per_s_max']:.1f}",
                    f"{r['ms_per_forward']:.3f}",
                ]
            )

    rows = []
    for rate in TARGET_RATES:
        trials = [measure_paced(model, rate, DEPLOY_BATCH) for _ in range(REPEATS)]
        ach_all = sorted(t[0] for t in trials)
        cpu_all = sorted(t[1] for t in trials)
        rss_all = sorted(t[2] for t in trials)
        achieved, cpu, rss = float(np.median(ach_all)), float(np.median(cpu_all)), float(np.median(rss_all))
        # The target counts as met only if it was met in every repeat, so a
        # single lucky run cannot carry the claim.
        met = all(a >= 0.95 * rate for a in ach_all)
        rows.append(
            {
                "target": rate,
                "achieved": achieved,
                "achieved_min": ach_all[0],
                "achieved_max": ach_all[-1],
                "cpu": cpu,
                "cpu_min": cpu_all[0],
                "cpu_max": cpu_all[-1],
                "rss": rss,
                "met": met,
            }
        )
        flag = "ok" if met else "SHORTFALL"
        print(
            f"  [=] offered {rate:6d} w/s -> achieved {achieved:9.1f} w/s "
            f"(range {ach_all[0]:.0f}-{ach_all[-1]:.0f})  "
            f"CPU={cpu:5.2f}% (range {cpu_all[0]:.1f}-{cpu_all[-1]:.1f})  peakWS={rss:.0f}MB  [{flag}]"
        )

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Offered Load (windows/s)",
                f"Achieved Throughput median of {REPEATS} (windows/s)",
                "Edge CPU median (%/core)",
                "Peak Working Set (MB, runtime-dominated)",
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
        "achieved_min": [r["achieved_min"] for r in rows],
        "achieved_max": [r["achieved_max"] for r in rows],
        "ai_cpu": [r["cpu"] for r in rows],
        "ai_cpu_min": [r["cpu_min"] for r in rows],
        "ai_cpu_max": [r["cpu_max"] for r in rows],
        "ai_ram": [r["rss"] for r in rows],
        "target_met": [r["met"] for r in rows],
        "model_mb": model_mb,
        "deploy_batch": DEPLOY_BATCH,
        "repeats": REPEATS,
    }


if __name__ == "__main__":
    run_real_resource_benchmark()
