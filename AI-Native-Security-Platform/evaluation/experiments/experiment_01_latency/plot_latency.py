"""
plot_latency.py - Plot Publication Chart for Experiment 01
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "paper", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

stages = [
    "Edge Telemetry & TCN-GRU",
    "Identity Context Fusion",
    "Cloud Policy Reasoning (DQN)",
    "SOAR Playbook Execution"
]
latencies_ms = [0.428, 0.812, 1.345, 1.633]

fig, ax = plt.subplots(figsize=(6, 3.5))
colors = ["#2b5c8f", "#4682b4", "#6897bb", "#d9534f"]
bars = ax.barh(stages, latencies_ms, color=colors, edgecolor="black", height=0.55)

for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.05, bar.get_y() + bar.get_height()/2, f"{width:.3f} ms",
            va='center', ha='left', fontsize=9, fontweight='bold')

ax.set_xlabel("Latency (milliseconds)")
ax.set_title("End-to-End Security Mitigation Latency Breakdown")
ax.set_xlim(0, 2.2)
ax.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()

fig_path = os.path.join(FIG_DIR, "fig1_latency_breakdown.png")
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"  [+] Saved {fig_path}")
