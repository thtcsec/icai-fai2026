# Experiment 01: End-to-End Latency Breakdown

## Objective
Quantify the execution latency across each component stage of the AI-Native Autonomous Security pipeline:
1. **Edge Telemetry Vectorization & TCN-GRU Anomaly Inference**
2. **Identity Context Fusion (RADIUS/DHCP Lease Mapping)**
3. **Cloud Policy Reasoning & Deep Q-Network (DQN) Action Selection**
4. **SOAR Playbook Execution (BGP Flowspec, 802.1X VLAN isolation, Token Revocation)**

## Execution
Run the latency benchmark:
```bash
python run_latency.py
```

Plot figure:
```bash
python plot_latency.py
```

## Generated Files
- `results.csv`: Per-stage latency measurements (ms) and percentage breakdown.
- `fig1_latency_breakdown.png`: Vector publication chart.
