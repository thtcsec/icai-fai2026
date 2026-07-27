# Experiment 04: Edge CPU Utilization & RAM Footprint Scaling

## Objective
Quantify edge node computational resource overhead under scaling telemetry throughput (100, 1,000, 5,000, and 10,000 events/sec).
Compares the lightweight **AI-Native Edge Gateway (PyTorch INT8 TCN-GRU)** against a **Legacy Inline Deep Packet Inspection (DPI) Proxy**.

## Execution
Run stress load benchmark:
```bash
python stress_test_resource.py
```

Plot figure:
```bash
python plot_resource.py
```

## Generated Files
- `results.csv`: Saved CPU (%) and RAM (MB) overhead metrics across throughput levels.
- `fig4_resource_overhead_scaling.png`: Vector CPU and RAM scaling chart.
