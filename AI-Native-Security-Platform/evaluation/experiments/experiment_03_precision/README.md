# Experiment 03: Detection Performance & Decision Threshold (tau) Sensitivity

## Objective
Evaluate the unsupervised TCN-GRU sequence reconstruction anomaly detection performance across decision thresholds $\tau \in [0.35, 0.85]$.
Metrics tracked:
- **Precision**
- **Recall**
- **F1-Score**
- **False Positive Rate (FPR %)**
- **False Negative Rate (FNR %)**

## Execution
Run evaluation:
```bash
python evaluate_precision.py
```

Plot figure:
```bash
python plot_precision_recall.py
```

## Generated Files
- `metrics.csv`: Saved evaluation metrics table across decision thresholds $\tau$.
- `fig3_precision_recall_sensitivity.png`: Vector Precision-Recall-F1 sensitivity chart.
