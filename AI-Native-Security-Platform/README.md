# An AI-Native Event-Driven Edge-Cloud Architecture for Autonomous Network Security and Resilience in Campus Infrastructure

[![Conference](https://img.shields.io/badge/Conference-ICAI--2026-blue)](https://icai.cmcu.edu.vn)
[![Paper Category](https://img.shields.io/badge/Category-Applied%20AI%20%7C%20Information%20%26%20Communications-green)]()
[![License](https://img.shields.io/badge/License-MIT-amber.svg)](LICENSE)
[![Artifact Status](https://img.shields.io/badge/Artifact-100%25%20Reproducible-success)]()

Official Research Artifact repository for the paper:  
**"An AI-Native Event-Driven Edge-Cloud Architecture for Autonomous Network Security and Resilience in Campus Infrastructure"**  
Submitted to **The Second International Conference on AI: AI Native for University (ICAI-2026 / ICAI-FAI 2026)**, organized by CMC University in collaboration with Steinbeis University (Germany) and Tsinghua University Shenzhen International Graduate School (China).

---

## 📌 1. Research Motivation & Background

University campus networks present unique cybersecurity hurdles due to high-density IoT deployments, unmanaged student Bring-Your-Own-Device (BYOD) endpoints, open Wi-Fi architectures, and massive concurrent data flows. Traditional Security Operations Center (SOC) architectures rely on centralized, manual incident response workflows that exhibit long Mean Time to Respond (MTTR > 30 minutes) and severe alert burnout.

This repository provides an **AI-Native Autonomous Security Platform** engineered specifically for high-throughput, low-latency university environments. The platform replaces manual human-in-the-loop triage with an **event-driven Edge-Cloud architecture** combining lightweight anomaly sequence autoencoders at the network edge with real-time policy reasoning and automated closed-loop SDN response playbooks in the cloud.

---

## ❓ 2. Research Questions (RQs)

* **RQ1 (Detection & Compression Quality):** How effectively does a PyTorch INT8-quantized TCN-GRU Autoencoder detect network anomalies and control-plane attacks under edge gateway resource constraints?
* **RQ2 (Response Autonomy & Speedup):** How much MTTR reduction is achieved when transitioning from human-in-the-loop SOC triage to an automated, event-driven closed-loop SDN & SOAR mitigation pipeline?
* **RQ3 (Resource Feasibility at Scale):** What is the CPU, RAM, and throughput overhead of the edge telemetry ingestor and model inference pipeline under scaling event rates up to 10,000 events/sec?

---

## 🎯 3. Research Contributions

This repository serves as a **reproducible scientific artifact** validating three primary contributions:

1. **Contribution 1 (Lightweight AI-Native Edge-Cloud Pipeline):** An event-driven architecture combining PyTorch INT8-quantized TCN-GRU sequence reconstruction at the edge with Redis Stream event distribution and spatial-temporal identity context fusion.
2. **Contribution 2 (Closed-Loop Autonomous Mitigation Prototype):** A reproducible end-to-end prototype featuring a Deep Q-Network (DQN) controller and dynamic risk engine executing zero-trust containment (SDN `flow_mod` rate-limiting, path rerouting, IP containment, and token revocation).
3. **Contribution 3 (Empirical Evaluation on Benchmark Datasets):** Comprehensive quantitative evaluation using `InSDN`, `CSE-CIC-IDS2018`, and SUMO+Mininet-WiFi telemetry, measuring per-stage latency breakdown (4.218 ms), MTTR reduction (0.0085 s vs 1,512.4 s), threshold sensitivity $\tau$ (F1=0.9400 at $\tau=0.65$), and edge CPU/RAM scaling up to 10,000 events/sec.

---

## 🏗️ 4. System Architecture

```
                                [ UNIVERSITY CAMPUS EDGE ]
┌───────────────────────┐    ┌───────────────────────────────────┐    ┌─────────────────────────────┐
│ High-Density IoT /    │───>│ Edge Anomaly Detector             │───>│ Redis Stream Event Bus      │
│ BYOD Telemetry Flows  │    │ (Quantized TCN-GRU Autoencoder)   │    │ (security:telemetry:stream) │
└───────────────────────┘    └───────────────────────────────────┘    └──────────────┬──────────────┘
                                                                                     │
                                                                                     ▼
                                [ CLOUD CONTROL PLANE ]               ┌─────────────────────────────┐
┌───────────────────────┐    ┌───────────────────────────────────┐    │ Identity Context Fusion     │
│ Automated SOAR &      │<───│ Dynamic DQN Policy Controller     │<───│ (IP/MAC -> User/Role state) │
│ SDN Response Playbooks│    │ (Severity & Risk Score Resolver)  │    └─────────────────────────────┘
└───────────┬───────────┘    └───────────────────────────────────┘
            │
            ▼
┌─────────────────────────┐
│ Active Mitigation       │ (Zero-Trust Enforcement: BGP Flowspec, 802.1X VLAN, Token Revocation)
└─────────────────────────┘
```

---

## 📂 5. Repository Structure

```
AI-Native-Security-Platform/
├── README.md                           # Main IEEE-grade academic documentation
├── LICENSE                             # MIT Open Source License
├── CITATION.cff                        # Academic citation metadata
├── requirements.txt                    # Python environment requirements
├── docker-compose.yml                  # Microservice container orchestration
├── paper/                              # LaTeX paper files & assets
│   ├── paper.tex                       # IEEE conference paper LaTeX template
│   ├── references.bib                  # BibTeX references
│   └── figures/                        # Generated vector PDF/PNG figures
├── architecture/                       # Architectural diagrams & source specs
├── docs/                               # Detailed academic technical specs
│   ├── methodology.md                  # Mathematics of TCN-GRU Autoencoder & Quantization
│   └── threat-model.md                 # STRIDE & MITRE ATT&CK mapping
├── results/                            # CSV Evidence Files Mapping directly to Paper Tables
│   ├── table2_latency.csv              # Stage-by-stage latency breakdown data
│   ├── table3_mttr.csv                 # Incident response MTTR benchmark data
│   ├── table4_precision.csv            # Detection sensitivity across threshold tau
│   └── table5_resource.csv             # Edge CPU utilization and RAM footprint scaling data
├── prototype/                          # Working prototype modules (from tu_projects)
│   ├── edge/                           # TCN-GRU sequence reconstruction detector & quantizer
│   │   ├── detector/                   # TCN-GRU Autoencoder (tcn_gru_model.py, quantize.py)
│   ├── identity-fusion/                # Spatial-temporal identity context fusion module
│   ├── redis-stream/                   # Async Redis stream pub/sub client
│   ├── dqn-controller/                 # Deep Q-Network SDN resilience agent
│   └── soar/                           # Dynamic Risk Scorer & Autonomous Playbooks
├── evaluation/                         # Benchmarking & experiment suite
│   ├── datasets/                       # Real & synthetic telemetry dataset preprocessors
│   ├── experiments/                    # Isolated research experiment suites
│   │   ├── experiment_01_latency/      # run_latency.py, plot_latency.py, results.csv
│   │   ├── experiment_02_mttr/         # benchmark_mttr.py, plot_mttr.py, results.csv
│   │   ├── experiment_03_precision/    # evaluate_precision.py, plot_precision_recall.py, metrics.csv
│   │   └── experiment_04_resource/     # stress_test_resource.py, plot_resource.py, results.csv
│   └── scripts/                        # Master runner scripts
│       └── generate_figures.py         # Master benchmark & publication figure generator
└── docker/                             # Service Dockerfiles
    ├── Dockerfile.edge                 # Edge service container spec
    ├── Dockerfile.cloud                # Cloud engine container spec
    └── Dockerfile.sdn_controller       # SDN Controller & SOAR agent spec
```

---

## 🧪 6. Step-by-Step Reproduction Guide

### Step 1: Environment Setup
```bash
git clone https://github.com/your-org/AI-Native-Security-Platform.git
cd AI-Native-Security-Platform
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Reproduce Isolated Paper Experiments

#### Experiment 1: End-to-End Latency Breakdown
```bash
cd evaluation/experiments/experiment_01_latency
python run_latency.py
python plot_latency.py
```

#### Experiment 2: Incident Response MTTR Benchmark
```bash
cd evaluation/experiments/experiment_02_mttr
python benchmark_mttr.py
python plot_mttr.py
```

#### Experiment 3: Detection Threshold Sensitivity ($\tau$)
```bash
cd evaluation/experiments/experiment_03_precision
python evaluate_precision.py
python plot_precision_recall.py
```

#### Experiment 4: Edge Resource Scaling at 10,000 Events/Sec
```bash
cd evaluation/experiments/experiment_04_resource
python stress_test_resource.py
python plot_resource.py
```

### Step 3: Run Master Reproduction Script & Generate Paper Figures
To run the full suite of experiments and update all publication charts automatically:
```bash
python evaluation/scripts/generate_figures.py
```

---

## 📊 7. Empirical Evidence Mapping

| Paper Table | Paper Section | Artifact CSV Evidence | Experiment Folder | Output Chart |
| :--- | :--- | :--- | :--- | :--- |
| **Table II** | Sec IV-A | `results/table2_latency.csv` | `experiment_01_latency/` | `paper/figures/fig1_latency_breakdown.png` |
| **Table III** | Sec IV-B | `results/table3_mttr.csv` | `experiment_02_mttr/` | `paper/figures/fig2_mttr_comparison.png` |
| **Table IV** | Sec IV-C | `results/table4_precision.csv` | `experiment_03_precision/` | `paper/figures/fig3_precision_recall_sensitivity.png` |
| **Table V** | Sec IV-D | `results/table5_resource.csv` | `experiment_04_resource/` | `paper/figures/fig4_resource_overhead_scaling.png` |

---

## 📜 8. Citation

If you use this research artifact or reference our work in your research, please cite:

```bibtex
@inproceedings{tu2026ainative,
  author    = {Tu, Trinh Hoang},
  title     = {An AI-Native Event-Driven Edge-Cloud Architecture for Autonomous Network Security and Resilience in Campus Infrastructure},
  booktitle = {Proceedings of The Second International Conference on AI: AI Native for University (ICAI-2026)},
  year      = {2026},
  address   = {Hanoi, Vietnam},
  publisher = {IEEE}
}
```
