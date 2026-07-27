"""
plot_precision_recall.py - Plot Publication Chart for Experiment 03
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "paper", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

thresholds = [0.35, 0.50, 0.65, 0.75, 0.85]
precision = [0.7210, 0.8840, 0.9482, 0.9710, 0.9890]
recall = [0.9890, 0.9650, 0.9320, 0.8410, 0.6520]
f1_score = [0.8340, 0.9227, 0.9400, 0.9013, 0.7858]

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(thresholds, precision, 'o-', color='#2b5c8f', label='Precision', linewidth=2)
ax.plot(thresholds, recall, 's-', color='#d9534f', label='Recall', linewidth=2)
ax.plot(thresholds, f1_score, '^--', color='#5cb85c', label='F1-Score (Optimal tau=0.65)', linewidth=2.5)

ax.axvline(x=0.65, color='gray', linestyle=':', label='Optimal Threshold (tau=0.65)')
ax.set_xlabel("Reconstruction Error Threshold (tau)")
ax.set_ylabel("Metric Value [0.0 - 1.0]")
ax.set_title("Anomaly Detection Performance vs Decision Threshold")
ax.legend(loc='lower left')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

fig_path = os.path.join(FIG_DIR, "fig3_precision_recall_sensitivity.png")
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"  [+] Saved {fig_path}")
