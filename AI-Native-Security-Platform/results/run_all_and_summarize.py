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


def main():
    generate_architecture_figure()
    lat = run_real_latency_benchmark()
    mttr = run_real_mttr_benchmark()
    prec = run_real_precision_eval()
    res = run_real_resource_benchmark()
    sota = run_sota_comparison()
    cross = run_cross_dataset()

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
        "resource_rows": [
            {
                "tp": tp,
                "cpu": round(c, 2),
                "rss": round(r, 1),
                "dpi_cpu": round(dc, 2),
                "dpi_rss": round(dr, 1),
            }
            for tp, c, r, dc, dr in zip(
                res["throughputs"], res["ai_cpu"], res["ai_ram"], res["dpi_cpu"], res["dpi_ram"]
            )
        ],
        "sota_methods": sota["methods"],
        "sota_f1": [round(x, 4) for x in sota["f1"]],
        "sota_latency_ms": [round(x, 4) for x in sota["latency"]],
        "sota_ram_mb": [round(x, 3) for x in sota["ram"]],
        "sota_cpu": [round(x, 2) for x in sota["cpu"]],
        "sota_delta_f1": round(sota["delta_f1"], 4),
        "cross_dataset": {
            "opt_tau": round(cross["opt_tau"], 4),
            "insdn_f1_065": round(cross["insdn_tau065"]["f1"], 4),
            "insdn_f1_star": round(cross["insdn_tau_star"]["f1"], 4),
            "cic_f1_065": round(cross["cic_tau065"]["f1"], 4),
            "cic_f1_star": round(cross["cic_tau_star"]["f1"], 4),
        },
    }

    out = os.path.join(RESULTS, "paper_metrics_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Wrote {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
