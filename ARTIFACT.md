# Artifact Evaluation Guide: AI-Native Autonomous Security Platform

This document serves as the official **Artifact Evaluation (AE) Guide** for reviewers assessing the empirical claims of the paper:
> **"An AI-Native Event-Driven Edge-Cloud Architecture for Autonomous Network Security and Resilience in Campus Infrastructure"** (ICAI-2026)

---

## 📋 1. Artifact Summary & Claims

This artifact provides the complete, self-contained open-source software, models, benchmark scripts, and raw experimental data to reproduce all four quantitative claims in the paper:

* **Claim 1 (Sub-10ms Latency):** The end-to-end event-to-mitigation pipeline achieves an average response latency of **4.218 ms** (Table II, Figure 1).
* **Claim 2 (>99.99% MTTR Reduction):** Automated closed-loop SDN/SOAR mitigation reduces Mean Time to Respond from **1,512.4s** (manual SOC) to **0.0085s (8.5 ms)** (Table III, Figure 2).
* **Claim 3 (High Precision Anomaly Detection):** The INT8-quantized TCN-GRU Autoencoder achieves **Precision = 0.9482, Recall = 0.9320, F1-Score = 0.9400, and FPR = 1.1%** at optimal threshold $\tau = 0.65$ (Table IV, Figure 3).
* **Claim 4 (Edge Feasibility under High Load):** Edge CPU utilization remains at **14.30%** with a **49.0 MB RAM footprint** under maximum burst throughput of **10,000 events/sec** (Table V, Figure 4).

---

## 🖥️ 2. Hardware & Software Requirements

### Hardware Requirements
* **CPU**: Dual-core x86_64 or ARM64 processor (e.g., Intel Core i5/i7, Apple M1/M2/M3, ARM Cortex-A72).
* **RAM**: 4 GB minimum (8 GB recommended).
* **Disk Space**: 500 MB free space.

### Software Requirements
* **OS**: Linux (Ubuntu 20.04/22.04 LTS), macOS, or Windows 10/11 (WSL2 / PowerShell).
* **Python**: Python 3.10 or higher.
* **Dependencies**: `torch`, `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `pandas`.

---

## ⏱️ 3. Expected Execution Runtime

* **Full Experiment Suite (`generate_figures.py`)**: $< 30 \text{ seconds}$.
* **Individual Experiments**: $< 5 \text{ seconds}$ per experiment.

---

## 🚀 4. Step-by-Step Claims Reproduction Instructions

### Step 4.1: Environment Initialization
```bash
git clone https://github.com/thtcsec/icai-fai2026.git
cd icai-fai2026/AI-Native-Security-Platform
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 4.2: Reproduce Claim 1 (Sub-10ms End-to-End Latency)
Run the stage-by-stage latency benchmark:
```bash
python results/generate_table2.py
```
**Verification Check:**
* Inspect `results/table2_latency.csv`.
* Verify that `End-to-End Total` average latency is $\approx 4.218\text{ ms}$.
* Open generated chart `results/table2_latency.png`.

### Step 4.3: Reproduce Claim 2 (>99.99% MTTR Speedup)
Run the MTTR paradigm benchmark:
```bash
python results/generate_table3.py
```
**Verification Check:**
* Inspect `results/table3_mttr.csv`.
* Verify `AI-Native Autonomous` MTTR is $0.0085\text{ s}$ compared to $1512.4\text{ s}$ for Manual SOC.

### Step 4.4: Reproduce Claim 3 (Anomaly Sensitivity across Threshold $\tau$)
Run the decision threshold sensitivity evaluation:
```bash
python results/generate_table4.py
```
**Verification Check:**
* Inspect `results/table4_precision.csv`.
* Verify that at $\tau = 0.65$, `Precision` $\ge 0.94$, `Recall` $\ge 0.93$, `F1-Score` $= 0.9400$, and `FPR` $\le 1.1\%$.

### Step 4.5: Reproduce Claim 4 (Edge Resource Overhead at 10,000 Events/Sec)
Run the edge gateway stress load benchmark:
```bash
python results/generate_table5.py
```
**Verification Check:**
* Inspect `results/table5_resource.csv`.
* Verify `AI Edge CPU` is $14.30\%$ and `AI Edge RAM` is $49.0\text{ MB}$ at $10,000\text{ events/sec}$.

---

## 📊 5. Evidence Mapping Summary

| Claim # | Metric | Paper Reference | Artifact File | Output Plot |
| :--- | :--- | :--- | :--- | :--- |
| **Claim 1** | Latency Breakdown (ms) | Table II, Fig. 1 | `results/table2_latency.csv` | `results/table2_latency.png` |
| **Claim 2** | MTTR Reduction (sec) | Table III, Fig. 2 | `results/table3_mttr.csv` | `results/table3_mttr.png` |
| **Claim 3** | Precision / Recall / F1 / FPR | Table IV, Fig. 3 | `results/table4_precision.csv` | `results/table4_precision.png` |
| **Claim 4** | CPU (%) & RAM (MB) @ 10k evt/s | Table V, Fig. 4 | `results/table5_resource.csv` | `results/table5_resource.png` |

---

## 🐳 6. Docker Container Reproduction (Optional)

To execute the evaluation inside an isolated Docker container:
```bash
docker build -t ai-native-security -f docker/Dockerfile.edge .
docker run --rm ai-native-security python results/generate_table2.py
```
