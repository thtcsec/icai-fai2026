# Experiment 02: Mean Time to Respond (MTTR) Benchmark Comparison

## Objective
Benchmark the incident response speedup of the proposed **AI-Native Autonomous Platform** against traditional security operations paradigms:
1. **Manual SOC Triage**: Human analyst manual investigation, ticket creation, and firewall rule editing.
2. **Legacy Rule-based SIEM**: Centralized SIEM polling and alert threshold triggers.
3. **AI-Native Autonomous Platform**: Closed-loop event-driven Redis Stream ingestion, TCN-GRU anomaly detection, DQN decision-making, and dynamic playbook execution.

## Execution
Run benchmark:
```bash
python benchmark_mttr.py
```

Plot figure:
```bash
python plot_mttr.py
```

## Generated Files
- `results.csv`: Saved Mean, Min, and Max MTTR benchmarks (seconds).
- `fig2_mttr_comparison.png`: Vector log-scale comparison chart.
