"""
plot_resource.py - Plot Publication Chart for Experiment 04
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "paper", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

throughput = [100, 1000, 5000, 10000]
ai_cpu = [2.12, 3.15, 7.75, 14.30]
ai_ram = [45.4, 45.4, 47.2, 49.0]
dpi_cpu = [8.78, 15.80, 47.00, 86.00]
dpi_ram = [121.8, 138.5, 212.5, 305.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

# CPU plot
ax1.plot(throughput, ai_cpu, 'o-', color='#2b5c8f', label='AI-Native Edge TCN-GRU', linewidth=2)
ax1.plot(throughput, dpi_cpu, 's--', color='#d9534f', label='Legacy Inline DPI Proxy', linewidth=2)
ax1.set_xlabel("Throughput (Events / sec)")
ax1.set_ylabel("CPU Utilization (%)")
ax1.set_title("Edge CPU Overhead vs Throughput")
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)

# RAM plot
ax2.plot(throughput, ai_ram, 'o-', color='#2b5c8f', label='AI-Native Edge TCN-GRU', linewidth=2)
ax2.plot(throughput, dpi_ram, 's--', color='#d9534f', label='Legacy Inline DPI Proxy', linewidth=2)
ax2.set_xlabel("Throughput (Events / sec)")
ax2.set_ylabel("RAM Footprint (MB)")
ax2.set_title("Edge RAM Footprint vs Throughput")
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

fig_path = os.path.join(FIG_DIR, "fig4_resource_overhead_scaling.png")
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"  [+] Saved {fig_path}")
