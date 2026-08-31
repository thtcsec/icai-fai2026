"""Run all paper experiments and emit a JSON summary for LaTeX sync."""

from __future__ import annotations

import json
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
RESULTS = os.path.join(BASE_DIR, "results")
sys.path.insert(0, RESULTS)

from generate_architecture_figure import generate_architecture_figure
from generate_table2 import run_real_latency_benchmark
from generate_table3 import run_real_mttr_benchmark
from generate_table4 import run_real_precision_eval
from generate_table5 import run_real_resource_benchmark
from generate_table6 import run_sota_comparison
from generate_table7 import run_cross_dataset
from generate_table8 import run_ablation
from generate_table9 import run_policy_benchmark
from generate_table10_leakage import run as run_leakage_effect


def main():
    generate_architecture_figure()
    lat = run_real_latency_benchmark()
    mttr = run_real_mttr_benchmark()
    prec = run_real_precision_eval()
    res = run_real_resource_benchmark()
    sota = run_sota_comparison()
    cross = run_cross_dataset()
    ablation = run_ablation()
    policy = run_policy_benchmark()
    leakage = run_leakage_effect()

    best_tau, best_m = prec["best_report"]
    summary = {
        "latency_total_mean_ms": round(lat["total_mean"], 4),
        "latency_total_std_ms": round(lat["total_std"], 4),
        "latency_stages": [
            {
                "name": n,
                "mean": round(m, 4),
                "std": round(s, 4),
                "pct": round(p, 1),
            }
            for n, m, s, p in zip(lat["stages"], lat["means"], lat["stds"], lat["percentages"])
        ],
        "mttr_manual_mean_s": round(mttr["means"][0], 4),
        "mttr_manual_std_s": round(mttr["stds"][0], 4),
        "mttr_siem_mean_s": round(mttr["means"][1], 4),
        "mttr_siem_std_s": round(mttr["stds"][1], 4),
        "mttr_ai_mean_s": round(mttr["means"][2], 6),
        "mttr_ai_std_s": round(mttr["stds"][2], 6),
        "mttr_ai_mean_ms": round(mttr["means"][2] * 1000, 4),
        "precision_best_tau": best_tau,
        "precision": round(best_m["precision"], 4),
        "recall": round(best_m["recall"], 4),
        "f1": round(best_m["f1"], 4),
        "fpr": round(best_m["fpr"], 2),
        "fnr": round(best_m["fnr"], 2),
        "precision_rows": [
            {
                "tau": t,
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
                "fpr": round(m["fpr"], 2),
                "fnr": round(m["fnr"], 2),
            }
            for t, m in prec["rows"]
        ],
        "resource_10k_cpu": round(res["ai_cpu"][-1], 2),
        "resource_10k_rss_mb": round(res["ai_ram"][-1], 1),
        "resource_model_mb": round(res["model_mb"], 3),
        "resource_deploy_batch": res["deploy_batch"],
        "resource_saturation": [
            {
                "batch": s["batch"],
                "windows_per_s": round(s["windows_per_s"], 1),
                "ms_per_forward": round(s["ms_per_forward"], 3),
            }
            for s in res["saturation"]
        ],
        "resource_rows": [
            {
                "offered": tp,
                "achieved": round(a, 1),
                "cpu": round(c, 2),
                "rss": round(r, 1),
                "target_met": bool(m),
            }
            for tp, a, c, r, m in zip(
                res["throughputs"], res["achieved"], res["ai_cpu"], res["ai_ram"], res["target_met"]
            )
        ],
        "sota_methods": sota["methods"],
        "sota_f1": [round(x, 4) for x in sota["f1"]],
        "sota_latency_ms": [round(x, 4) for x in sota["latency"]],
        "sota_ram_mb": [None if x is None else round(x, 3) for x in sota["ram"]],
        "sota_delta_f1": round(sota["delta_f1"], 4),
        "cross_dataset": {
            "opt_tau": round(cross["opt_tau"], 4),
            "insdn_f1_065": round(cross["insdn_tau065"]["f1"], 4),
            "insdn_f1_star": round(cross["insdn_tau_star"]["f1"], 4),
            "cic_f1_065": round(cross["cic_tau065"]["f1"], 4),
            "cic_f1_star": round(cross["cic_tau_star"]["f1"], 4),
        },
        "ablation_transport_async_ms": round(ablation["async_ms"], 4),
        "ablation_transport_sync_ms": round(ablation["sync_ms"], 4),
        "ablation_rows": [
            {
                "config": r["config"],
                "f1": round(r["f1"], 4),
                "fpr": round(r["fpr"], 2),
                "latency_ms": round(r["latency"], 4),
            }
            for r in ablation["rows"]
        ],
        "ablation_rho_sweep": [
            {
                "rho": s["rho"],
                "full_f1": round(s["full_f1"], 4),
                "no_identity_f1": round(s["no_id_f1"], 4),
                "delta_f1": round(s["delta"], 4),
            }
            for s in ablation["sweep"]
        ],
        "policy_rows": [
            {
                "policy": r["policy"],
                "latency_ms": round(r["latency_ms"], 4),
                "qos_disruption": round(r["qos"], 4),
                "mean_regret": round(r["regret"], 4),
                "action_accuracy_pct": round(r["accuracy"], 2),
            }
            for r in policy["rows"]
        ],
        "leakage_effect": {
            "leaky_f1_065": round(leakage["leaky"]["fixed"]["f1"], 4),
            "leaky_opt_tau": round(leakage["leaky"]["opt_tau"], 4),
            "leaky_f1_opt": round(leakage["leaky"]["opt"]["f1"], 4),
            "controlled_f1_065": round(leakage["controlled"]["fixed"]["f1"], 4),
            "controlled_opt_tau": round(leakage["controlled"]["opt_tau"], 4),
            "controlled_f1_opt": round(leakage["controlled"]["opt"]["f1"], 4),
            "delta_f1_fixed": round(leakage["delta_f1_fixed"], 4),
            "delta_f1_opt": round(leakage["delta_f1_opt"], 4),
        },
    }

    out = os.path.join(RESULTS, "paper_metrics_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Wrote {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
