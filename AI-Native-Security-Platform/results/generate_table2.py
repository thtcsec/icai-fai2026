"""
generate_table2.py - Software control-path latency including Redis Streams.

Measures three separately labelled quantities:
  1. detector-only (INT8 TCN-GRU inference on a real InSDN window)
  2. policy-only (loaded DQN action selection on the encoded observation)
  3. end-to-end control path: preprocess + detector + identity + HMAC
     principal_id + Redis XADD/XREADGROUP + DQN + SOAR descriptor construction

Detector outputs are the real ``p_attack = 1 - P(normal)`` contract. Hardcoded
attack probabilities are forbidden on this path. DQN weights are loaded from
the trained checkpoint; a randomly initialized agent is a hard error.

Excludes dataplane enforcement. End-to-end Redis trials require Redis 7.x on
localhost:6379. Detector-only and policy-only still run if Redis is down.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from typing import Optional

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
from prototype.cloud.policy_engine.observation import observation_from_detector_window
from prototype.data.data_loader import resolve_real_windows_npz
from prototype.edge.detector.inference import infer_window, load_detector_checkpoint
from prototype.edge.identity_fusion.fusion import IdentityFusionEngine
from prototype.edge.privacy.pseudonymize import assert_no_raw_endpoint
from prototype.soar.playbooks import SOARPlaybooks
from train_utils import blocked_split_npz, set_global_seed

CORRECTED_DIR = os.path.join(BASE_DIR, "results", "corrected")
CSV_PATH = os.path.join(CORRECTED_DIR, "table2_latency.csv")
PNG_PATH = os.path.join(CORRECTED_DIR, "table2_latency.png")
JSON_PATH = os.path.join(CORRECTED_DIR, "table2_latency.json")
FIG_PATH = os.path.join(BASE_DIR, "paper", "figures", "fig2_latency_breakdown.png")
DETECTOR_CKPT = os.path.join(BASE_DIR, "results", "checkpoints", "tcn_gru_ae_trained.pt")
DQN_CKPT = os.path.join(BASE_DIR, "results", "checkpoints", "dqn_policy_trained.pt")

# Paper-submitted Table II (detector output discarded; p_attack hardcoded 0.91;
# DQN randomly initialized). Kept here as historical evidence, not as a target.
HISTORICAL_PAPER_TOTAL_MEDIAN_MS = 4.793
HISTORICAL_PAPER_NOTE = (
    "Submitted Table II discarded detector outputs, injected p_attack=0.91, "
    "and constructed DRLResilienceAgent without loading trained DQN weights."
)


def _cloud_event(ctx: dict, pkt_rate: float, p_attack: float) -> dict:
    event = {
        "principal_id": ctx["principal_id"],
        "pkt_rate": pkt_rate,
        "p_attack": float(p_attack),
        "role": ctx["role"],
        "trust_score": float(ctx["trust_score"]),
        "device": ctx["device"],
        "identity_source": ctx.get("identity_source", "unknown"),
        "identity_is_simulated": bool(ctx.get("is_simulated", True)),
    }
    assert_no_raw_endpoint(event)
    return event


def _host_info() -> dict:
    return {
        "cpu": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "os": f"{platform.system()} {platform.release()} {platform.version()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }


def _prepare_windows(seed: int):
    npz = resolve_real_windows_npz("insdn")
    if npz is None:
        raise FileNotFoundError("insdn_windows.npz required for latency microbenchmark")
    _, _, X_te, _, _ = blocked_split_npz(npz, seq_len=10, seed=seed)
    if len(X_te) == 0:
        raise RuntimeError("Blocked split produced an empty test set")
    return X_te.contiguous()


def run_real_latency_benchmark(
    num_trials: int = 1000,
    seed: int = 42,
    warmups: int = 50,
    redis_block_ms: int = 2000,
    write_paper_figure: bool = False,
):
    print(f"[*] Corrected latency benchmark ({num_trials} trials, seed={seed})...")
    set_global_seed(seed)
    torch.set_num_threads(1)

    os.makedirs(CORRECTED_DIR, exist_ok=True)
    X_te = _prepare_windows(seed)
    print(f"  [=] Prepared real InSDN test windows: {tuple(X_te.shape)}")

    detector, det_meta = load_detector_checkpoint(DETECTOR_CKPT, quantize=True)
    dqn_agent = DRLResilienceAgent.from_checkpoint(DQN_CKPT)
    fusion = IdentityFusionEngine()
    soar_playbooks = SOARPlaybooks()

    bus = None
    redis_ok = False
    redis_error: Optional[str] = None
    try:
        from prototype.edge.redis_stream.pubsub import require_redis

        bus = require_redis(retries=2, delay_s=0.2)
        bus.reset_stream()
        redis_ok = True
    except Exception as exc:
        redis_error = str(exc)
        print(f"  [!] Redis unavailable ({exc}); e2e Redis stage will be SKIPPED")

    src_ips = ("10.0.1.15", "10.0.2.105")

    def _one_window(i: int) -> torch.Tensor:
        return X_te[i % len(X_te)].unsqueeze(0).contiguous()

    # Warm-up (not timed)
    with torch.no_grad():
        for i in range(warmups):
            window = _one_window(i)
            det = infer_window(
                detector, window, checkpoint_path=DETECTOR_CKPT, quantized=True
            )
            ctx = fusion.enrich(src_ips[i % 2])
            obs = observation_from_detector_window(
                det.p_attack, det.reconstruction_error, window[0, -1].tolist()
            )
            if redis_ok:
                event = _cloud_event(ctx, float(window[0, -1, 1]), det.p_attack)
                _ = bus.publish_and_consume(event, block_ms=redis_block_ms)
            action_id = dqn_agent.select_action(obs.as_list(), eval_mode=True)
            _ = soar_playbooks.execute_playbook(
                dqn_agent.get_action_name(action_id),
                {"principal_id": ctx["principal_id"], "device": ctx["device"]},
            )
    if redis_ok:
        bus.reset_stream()

    preprocess_ms, edge_ms, fusion_ms, redis_ms, dqn_ms, soar_ms = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    p_attacks = []

    for i in range(num_trials):
        src_ip = src_ips[0] if (i % 5) else src_ips[1]
        raw = X_te[i % len(X_te)]

        t0 = time.perf_counter()
        window = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(0).contiguous()
        preprocess_ms.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        det = infer_window(
            detector, window, checkpoint_path=DETECTOR_CKPT, quantized=True
        )
        edge_ms.append((time.perf_counter() - t0) * 1000.0)
        p_attacks.append(det.p_attack)

        t0 = time.perf_counter()
        ctx = fusion.enrich(src_ip)
        fusion_ms.append((time.perf_counter() - t0) * 1000.0)

        event = _cloud_event(ctx, float(window[0, -1, 1]), det.p_attack)

        if redis_ok:
            t0 = time.perf_counter()
            _ = bus.publish_and_consume(event, block_ms=redis_block_ms)
            redis_ms.append((time.perf_counter() - t0) * 1000.0)
        else:
            redis_ms.append(float("nan"))

        obs = observation_from_detector_window(
            det.p_attack, det.reconstruction_error, window[0, -1].tolist()
        )
        t0 = time.perf_counter()
        action_id = dqn_agent.select_action(obs.as_list(), eval_mode=True)
        action_name = dqn_agent.get_action_name(action_id)
        dqn_ms.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        descriptor = soar_playbooks.execute_playbook(action_name, event)
        soar_ms.append((time.perf_counter() - t0) * 1000.0)
        if descriptor.get("enforced") is True:
            raise RuntimeError("SOAR path must not claim enforcement")

    def _stats(xs):
        arr = np.asarray(xs, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return {"median": None, "p95": None, "p99": None, "mean": None, "n": 0}
        return {
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(np.mean(arr)),
            "n": int(len(arr)),
        }

    detector_only = _stats(edge_ms)
    policy_only = _stats(dqn_ms)
    e2e_parts = [preprocess_ms, edge_ms, fusion_ms, dqn_ms, soar_ms]
    if redis_ok:
        e2e_parts.append(redis_ms)
    e2e_trials = np.sum([np.asarray(p, dtype=np.float64) for p in e2e_parts], axis=0)
    e2e = _stats(e2e_trials)

    stages = [
        ("Window preprocess (nan_to_num + tensor)", preprocess_ms),
        ("Edge TCN-GRU Inference (trained INT8)", edge_ms),
        ("Identity Context Fusion (simulated lease table)", fusion_ms),
        ("Redis Streams (XADD+XREADGROUP)", redis_ms),
        ("Cloud DQN Policy (loaded checkpoint)", dqn_ms),
        ("SOAR Playbook Descriptor (not enforced)", soar_ms),
    ]
    stage_stats = [(name, _stats(xs)) for name, xs in stages]

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Pipeline Stage",
                "Median (ms)",
                "p95 (ms)",
                "p99 (ms)",
                "n",
                "Notes",
            ]
        )
        for name, st in stage_stats:
            if st["n"] == 0:
                writer.writerow([name, "SKIP", "SKIP", "SKIP", 0, redis_error or "no samples"])
                continue
            writer.writerow(
                [
                    name,
                    f"{st['median']:.4f}",
                    f"{st['p95']:.4f}",
                    f"{st['p99']:.4f}",
                    st["n"],
                    "",
                ]
            )
        writer.writerow(
            [
                "Detector-only",
                f"{detector_only['median']:.4f}",
                f"{detector_only['p95']:.4f}",
                f"{detector_only['p99']:.4f}",
                detector_only["n"],
                "INT8 inference only",
            ]
        )
        writer.writerow(
            [
                "Policy-only",
                f"{policy_only['median']:.4f}",
                f"{policy_only['p95']:.4f}",
                f"{policy_only['p99']:.4f}",
                policy_only["n"],
                "Loaded DQN select_action only",
            ]
        )
        e2e_label = (
            "End-to-end control path (includes Redis)"
            if redis_ok
            else "End-to-end control path (Redis SKIPPED; not comparable to paper Table II)"
        )
        writer.writerow(
            [
                e2e_label,
                f"{e2e['median']:.4f}" if e2e["median"] is not None else "SKIP",
                f"{e2e['p95']:.4f}" if e2e["p95"] is not None else "SKIP",
                f"{e2e['p99']:.4f}" if e2e["p99"] is not None else "SKIP",
                e2e["n"],
                "Does not include capture/FE/windowing or dataplane install",
            ]
        )

    plot_names = []
    plot_med = []
    plot_p95 = []
    for name, st in stage_stats:
        if st["median"] is None:
            continue
        plot_names.append(name)
        plot_med.append(st["median"])
        plot_p95.append(st["p95"])
    if plot_names:
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        y = np.arange(len(plot_names))
        err_up = [p - m for p, m in zip(plot_p95, plot_med)]
        ax.barh(
            y,
            plot_med,
            xerr=[[0] * len(plot_med), err_up],
            color="#2b5c8f",
            edgecolor="black",
            height=0.55,
            error_kw=dict(ecolor="#1A252C", lw=1.8, capsize=6, capthick=1.8),
        )
        ax.set_yticks(y)
        ax.set_yticklabels(plot_names, fontsize=7)
        ax.set_xlabel("Latency (ms), bar = median, whisker = p95")
        title_e2e = f"{e2e['median']:.3f}" if e2e["median"] is not None else "n/a"
        ax.set_title(
            f"Corrected control-path latency "
            f"(e2e median {title_e2e} ms; real detector p_attack; loaded DQN)"
        )
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
        if write_paper_figure:
            os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
            plt.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
        plt.close()

    p_arr = np.asarray(p_attacks, dtype=np.float64)
    payload = {
        "historical_paper_total_median_ms": HISTORICAL_PAPER_TOTAL_MEDIAN_MS,
        "historical_paper_note": HISTORICAL_PAPER_NOTE,
        "detector_only_ms": detector_only,
        "policy_only_ms": policy_only,
        "end_to_end_ms": e2e,
        "stages": {name: st for name, st in stage_stats},
        "redis_included": redis_ok,
        "redis_error": redis_error,
        "p_attack": {
            "source": "trained_int8_tcn_gru_classifier_head",
            "scoring_rule": "1 - P(normal)",
            "mean": float(p_arr.mean()) if len(p_arr) else None,
            "min": float(p_arr.min()) if len(p_arr) else None,
            "max": float(p_arr.max()) if len(p_arr) else None,
            "hardcoded_override": False,
        },
        "detector_checkpoint": det_meta,
        "dqn_checkpoint": DQN_CKPT,
        "warmup": warmups,
        "num_trials": num_trials,
        "seed": seed,
        "host": _host_info(),
        "soar_enforced": False,
        "identity_table": "simulated_lease_fixture",
        "label": "corrected_real_detector_loaded_dqn",
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(
        f"  [+] Detector-only: median {detector_only['median']:.4f} ms "
        f"p95 {detector_only['p95']:.4f} p99 {detector_only['p99']:.4f}"
    )
    print(
        f"  [+] Policy-only:   median {policy_only['median']:.4f} ms "
        f"p95 {policy_only['p95']:.4f} p99 {policy_only['p99']:.4f}"
    )
    if e2e["median"] is not None:
        print(
            f"  [+] E2E:           median {e2e['median']:.4f} ms "
            f"p95 {e2e['p95']:.4f} p99 {e2e['p99']:.4f} redis={redis_ok}"
        )
    print(
        f"  [=] p_attack from detector: mean={payload['p_attack']['mean']:.4f} "
        f"range=[{payload['p_attack']['min']:.4f}, {payload['p_attack']['max']:.4f}]"
    )
    print(f"  [=] Wrote {CSV_PATH} and {JSON_PATH}")
    # Compatibility keys for run_all_and_summarize.py (corrected numbers, not paper Table II).
    payload["total_median"] = e2e["median"] if e2e["median"] is not None else float("nan")
    payload["total_p95"] = e2e["p95"] if e2e["p95"] is not None else float("nan")
    payload["total_p99"] = e2e["p99"] if e2e["p99"] is not None else float("nan")
    payload["total_mean"] = e2e["mean"] if e2e["mean"] is not None else float("nan")
    payload["stages"] = [name for name, _ in stage_stats]
    payload["medians"] = [st["median"] if st["median"] is not None else float("nan") for _, st in stage_stats]
    payload["p95s"] = [st["p95"] if st["p95"] is not None else float("nan") for _, st in stage_stats]
    payload["p99s"] = [st["p99"] if st["p99"] is not None else float("nan") for _, st in stage_stats]
    payload["percentages"] = [0.0 for _ in stage_stats]
    payload["hmac_principals"] = True
    payload["redis_required"] = True
    return payload


if __name__ == "__main__":
    run_real_latency_benchmark()
