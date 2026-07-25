"""Automated Research Experiment Runner & Publication Figure Generator for ICAI-2026 Paper."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from evaluation.experiments.experiment_01_latency.run import run_latency_experiment
from evaluation.experiments.experiment_02_mttr.run import run_mttr_experiment
from evaluation.experiments.experiment_03_precision.run import run_precision_experiment
from evaluation.experiments.experiment_04_resource.run import run_resource_experiment

def setup_matplotlib_style():
    """Applies IEEE publication-ready style formatting to Matplotlib plots."""
    plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 10,
        "font.size": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight"
    })

def main():
    print("=" * 70)
    print("AI-NATIVE SECURITY PLATFORM: GENERATING EXPERIMENTAL BENCHMARKS & FIGURES")
    print("=" * 70)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    paper_fig_dir = os.path.join(base_dir, "paper", "figures")
    paper_tbl_dir = os.path.join(base_dir, "paper", "tables")
    os.makedirs(paper_fig_dir, exist_ok=True)
    os.makedirs(paper_tbl_dir, exist_ok=True)

    setup_matplotlib_style()

    # ---------------------------------------------------------
    # EXPERIMENT 1: LATENCY BREAKDOWN
    # ---------------------------------------------------------
    print("\n[1/4] Executing Experiment 01: Latency Breakdown...")
    exp1_dir = os.path.join(base_dir, "evaluation", "experiments", "experiment_01_latency")
    os.makedirs(exp1_dir, exist_ok=True)

    df_lat = run_latency_experiment(num_trials=1000)
    df_lat.to_csv(os.path.join(exp1_dir, "latency_results.csv"), index=False)

    avg_edge = df_lat["edge_detection_ms"].mean()
    avg_fusion = df_lat["identity_fusion_ms"].mean()
    avg_policy = df_lat["policy_reasoning_ms"].mean()
    avg_soar = df_lat["soar_execution_ms"].mean()
    avg_total = df_lat["total_e2e_latency_ms"].mean()

    # Save summary markdown
    with open(os.path.join(exp1_dir, "summary.md"), "w") as f:
        f.write("# Experiment 01: Latency Breakdown Summary\n\n")
        f.write(f"- **Edge Detection Avg**: {avg_edge:.3f} ms\n")
        f.write(f"- **Identity Fusion Avg**: {avg_fusion:.3f} ms\n")
        f.write(f"- **Policy Reasoning Avg**: {avg_policy:.3f} ms\n")
        f.write(f"- **SOAR Execution Avg**: {avg_soar:.3f} ms\n")
        f.write(f"- **End-to-End Latency Avg**: {avg_total:.3f} ms\n")

    # Plot Exp 1 Figure
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    stages = ["Edge Detector", "Identity Fusion", "Policy Engine", "SOAR Execution"]
    means = [avg_edge, avg_fusion, avg_policy, avg_soar]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    bars = ax.bar(stages, means, color=colors, width=0.55, edgecolor="black", linewidth=0.7)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-Stage Latency Breakdown (Sub-10ms Pipeline)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(means) * 1.35)
    plt.xticks(rotation=15, ha="right")

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f} ms", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(paper_fig_dir, "fig1_latency_breakdown.pdf"))
    fig.savefig(os.path.join(paper_fig_dir, "fig1_latency_breakdown.png"))
    plt.close()

    # ---------------------------------------------------------
    # EXPERIMENT 2: MTTR COMPARISON
    # ---------------------------------------------------------
    print("\n[2/4] Executing Experiment 02: MTTR Comparison...")
    exp2_dir = os.path.join(base_dir, "evaluation", "experiments", "experiment_02_mttr")
    os.makedirs(exp2_dir, exist_ok=True)

    df_mttr = run_mttr_experiment(num_trials=50)
    df_mttr.to_csv(os.path.join(exp2_dir, "mttr_results.csv"), index=False)

    avg_manual = df_mttr["manual_soc_mttr_sec"].mean()
    avg_siem = df_mttr["legacy_siem_mttr_sec"].mean()
    avg_ainative = df_mttr["ainative_autonomous_mttr_sec"].mean()

    with open(os.path.join(exp2_dir, "summary.md"), "w") as f:
        f.write("# Experiment 02: MTTR Comparison Summary\n\n")
        f.write(f"- **Manual SOC MTTR**: {avg_manual:.1f} s (~{avg_manual/60:.1f} mins)\n")
        f.write(f"- **Legacy Rule SIEM MTTR**: {avg_siem:.2f} s\n")
        f.write(f"- **AI-Native Autonomous MTTR**: {avg_ainative*1000:.2f} ms ({avg_ainative:.4f} s)\n")

    # Plot Exp 2 Figure (Log Scale)
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    methods = ["Manual SOC Triage", "Legacy SIEM", "AI-Native Autonomous"]
    mttr_vals = [avg_manual, avg_siem, avg_ainative]
    
    bars2 = ax.bar(methods, mttr_vals, color=["#d62728", "#ff7f0e", "#2ca02c"], edgecolor="black", linewidth=0.7, width=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Mean Time To Respond (Seconds, Log Scale)")
    ax.set_title("Incident Response Time (MTTR) Comparison", fontsize=10, fontweight="bold")
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars2, mttr_vals):
        if val >= 1.0:
            lbl = f"{val:.1f} s"
        else:
            lbl = f"{val*1000:.1f} ms"
        ax.text(bar.get_x() + bar.get_width()/2.0, val * 1.4, lbl, ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(paper_fig_dir, "fig2_mttr_comparison.pdf"))
    fig.savefig(os.path.join(paper_fig_dir, "fig2_mttr_comparison.png"))
    plt.close()

    # ---------------------------------------------------------
    # EXPERIMENT 3: PRECISION / RECALL SENSITIVITY
    # ---------------------------------------------------------
    print("\n[3/4] Executing Experiment 03: Precision & Recall Sensitivity...")
    exp3_dir = os.path.join(base_dir, "evaluation", "experiments", "experiment_03_precision")
    os.makedirs(exp3_dir, exist_ok=True)

    df_prec = run_precision_experiment()
    df_prec.to_csv(os.path.join(exp3_dir, "precision_results.csv"), index=False)

    best_row = df_prec.iloc[df_prec["f1_score"].idxmax()]
    with open(os.path.join(exp3_dir, "summary.md"), "w") as f:
        f.write("# Experiment 03: Precision & Recall Sensitivity Summary\n\n")
        f.write(f"- **Optimal Decision Threshold**: {best_row['decision_threshold']}\n")
        f.write(f"- **Max F1-Score**: {best_row['f1_score']:.4f}\n")
        f.write(f"- **Precision at Optimal Threshold**: {best_row['precision']:.4f}\n")
        f.write(f"- **Recall at Optimal Threshold**: {best_row['recall']:.4f}\n")
        f.write(f"- **False Positive Rate (FPR)**: {best_row['false_positive_rate']:.4f}\n")

    # Plot Exp 3 Figure
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.plot(df_prec["decision_threshold"], df_prec["precision"], marker="o", label="Precision", color="#1f77b4", linewidth=1.5)
    ax.plot(df_prec["decision_threshold"], df_prec["recall"], marker="s", label="Recall", color="#2ca02c", linewidth=1.5)
    ax.plot(df_prec["decision_threshold"], df_prec["f1_score"], marker="^", label="F1-Score", color="#d62728", linewidth=2.0, linestyle="--")
    
    ax.axvline(best_row["decision_threshold"], color="gray", linestyle=":", label=f"Optimal τ = {best_row['decision_threshold']}")
    ax.set_xlabel("Decision Threshold (τ)")
    ax.set_ylabel("Metric Score")
    ax.set_title("Detection Performance Across Decision Thresholds", fontsize=10, fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(os.path.join(paper_fig_dir, "fig3_precision_recall_sensitivity.pdf"))
    fig.savefig(os.path.join(paper_fig_dir, "fig3_precision_recall_sensitivity.png"))
    plt.close()

    # ---------------------------------------------------------
    # EXPERIMENT 4: RESOURCE OVERHEAD AT SCALE
    # ---------------------------------------------------------
    print("\n[4/4] Executing Experiment 04: Resource Overhead at Scale...")
    exp4_dir = os.path.join(base_dir, "evaluation", "experiments", "experiment_04_resource")
    os.makedirs(exp4_dir, exist_ok=True)

    df_res = run_resource_experiment()
    df_res.to_csv(os.path.join(exp4_dir, "resource_results.csv"), index=False)

    with open(os.path.join(exp4_dir, "summary.md"), "w") as f:
        f.write("# Experiment 04: Resource Overhead Summary\n\n")
        f.write(df_res.to_markdown(index=False))

    # Plot Exp 4 Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    ax1.plot(df_res["throughput_events_per_sec"], df_res["edge_detector_cpu_pct"], marker="o", label="AI-Native Edge", color="#2ca02c", linewidth=2.0)
    ax1.plot(df_res["throughput_events_per_sec"], df_res["legacy_dpi_cpu_pct"], marker="x", label="Legacy Inline DPI", color="#d62728", linewidth=1.5, linestyle="--")
    ax1.set_xlabel("Throughput (Events/sec)")
    ax1.set_ylabel("CPU Utilization (%)")
    ax1.set_title("Edge CPU Overhead", fontsize=9, fontweight="bold")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(df_res["throughput_events_per_sec"], df_res["edge_detector_ram_mb"], marker="o", label="AI-Native Edge", color="#2ca02c", linewidth=2.0)
    ax2.plot(df_res["throughput_events_per_sec"], df_res["legacy_dpi_ram_mb"], marker="x", label="Legacy Inline DPI", color="#d62728", linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Throughput (Events/sec)")
    ax2.set_ylabel("RAM Footprint (MB)")
    ax2.set_title("Edge Memory Footprint", fontsize=9, fontweight="bold")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(os.path.join(paper_fig_dir, "fig4_resource_overhead_scaling.pdf"))
    fig.savefig(os.path.join(paper_fig_dir, "fig4_resource_overhead_scaling.png"))
    plt.close()

    # Generate LaTeX Table Snippets
    with open(os.path.join(paper_tbl_dir, "table_latency_breakdown.tex"), "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\caption{End-to-End Latency Breakdown Across Pipeline Stages}\n")
        f.write("\\label{tab:latency_breakdown}\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Pipeline Stage} & \\textbf{Avg Latency (ms)} & \\textbf{Pct of Total (\\%)} \\\\\n")
        f.write("\\hline\n")
        f.write(f"Edge Anomaly Detection & {avg_edge:.3f} & {avg_edge/avg_total*100:.1f}\\% \\\\\n")
        f.write(f"Identity Context Fusion & {avg_fusion:.3f} & {avg_fusion/avg_total*100:.1f}\\% \\\\\n")
        f.write(f"Cloud Policy Reasoning & {avg_policy:.3f} & {avg_policy/avg_total*100:.1f}\\% \\\\\n")
        f.write(f"SOAR Playbook Execution & {avg_soar:.3f} & {avg_soar/avg_total*100:.1f}\\% \\\\\n")
        f.write("\\hline\n")
        f.write(f"\\textbf{{End-to-End Total}} & \\textbf{{{avg_total:.3f}}} & \\textbf{{100.0\\%}} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print("\n✅ All 4 experiments executed successfully!")
    print(f"   Outputs saved to: {paper_fig_dir}")

if __name__ == "__main__":
    main()
