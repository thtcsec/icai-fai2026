"""
plot_mttr.py - Plot Publication Chart for Experiment 02
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "paper", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

paradigms = ["Manual SOC Triage", "Legacy Rule SIEM", "AI-Native Autonomous"]
mean_mttr_sec = [1512.4, 68.5, 0.0085]

fig, ax = plt.subplots(figsize=(6, 3.5))
bars = ax.bar(paradigms, mean_mttr_sec, color=["#d9534f", "#f0ad4e", "#5cb85c"], edgecolor="black", width=0.45)
ax.set_yscale("log")
ax.set_ylabel("Mean Time to Respond - MTTR (seconds, Log Scale)")
ax.set_title("Incident Response MTTR Benchmark Comparison")
ax.grid(axis='y', linestyle='--', alpha=0.6)

labels = ["1,512.4 s (~25.2 min)", "68.5 s", "0.0085 s (8.5 ms)"]
for bar, label in zip(bars, labels):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height * 1.3, label,
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylim(0.001, 10000)
plt.tight_layout()

fig_path = os.path.join(FIG_DIR, "fig2_mttr_comparison.png")
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"  [+] Saved {fig_path}")
