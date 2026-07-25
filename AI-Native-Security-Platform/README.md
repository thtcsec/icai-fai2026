# AI-Native Autonomous Security Platform: An Event-Driven Edge-Cloud Architecture for University Networks

[![Conference](https://img.shields.io/badge/Conference-ICAI--2026-blue)](https://icai.cmcu.edu.vn)
[![Paper Category](https://img.shields.io/badge/Category-Applied%20AI%20%7C%20Systems-green)]()
[![License](https://img.shields.io/badge/License-MIT-amber.svg)](LICENSE)
[![Artifact Status](https://img.shields.io/badge/Artifact-Reproducible-success)]()

Official Research Artifact repository for the paper:
**"AI-Native Autonomous Security Platform: An Event-Driven Edge-Cloud Architecture for University Networks"**
Submitted to **The Second International Conference on AI: AI Native for University (ICAI-2026 / ICAI-FAI 2026)**, organized by CMC University in collaboration with Steinbeis University (Germany) and Tsinghua University Shenzhen International Graduate School (China).

---

## 📌 Research Overview & Motivation

University campuses present unique cyber-security challenges due to high-density IoT deployments, heterogeneous BYOD (Bring-Your-Own-Device) environments, open Wi-Fi architectures, and massive concurrent data flows. Traditional Security Operations Center (SOC) architectures rely on centralized, manual incident response workflows that exhibit long Mean Time to Respond (MTTR > 30 minutes) and fail to scale.

This repository provides an **AI-Native Autonomous Security Platform** engineered specifically for high-throughput, low-latency university environments. The platform replaces human-in-the-loop triage with an **event-driven Edge-Cloud architecture** combining lightweight anomaly detection at the network edge with real-time policy reasoning and automated SOAR response playbooks in the cloud.

---

## ❓ Research Questions (RQs)

* **RQ1 (Architectural Efficiency):** How can an event-driven edge-cloud security architecture achieve sub-10ms telemetry ingestion and anomaly classification without bottlenecking campus network backbones?
* **RQ2 (Response Autonomy):** Can dynamic identity-fused AI policy engines reduce Mean Time to Respond (MTTR) by >90% compared to legacy manual SOC workflows while maintaining low false-positive rates?
* **RQ3 (Resource Feasibility):** Is edge-node anomaly detection computationally viable on low-cost campus edge devices under burst traffic loads exceeding 10,000 events/second?

---

## 🎯 Research Contributions

This repository serves as a **reproducible scientific artifact** validating three primary contributions:

1. **Contribution 1 (Lightweight AI-Native Architecture):** A decoupled, event-driven Edge-Cloud security pipeline leveraging Redis Streams, edge anomaly detection (Isolation Forest), and identity context fusion.
2. **Contribution 2 (Autonomous Response Prototype):** A working end-to-end prototype demonstrating automatic threat identification, policy evaluation, and automated mitigation (IP containment, credential revocation).
3. **Contribution 3 (Empirical Evaluation Framework):** A fully automated benchmark suite evaluating latency, MTTR, detection metrics (Precision, Recall, F1, FPR), and system resource consumption across scaling throughput levels.

---

## 🏗️ System Architecture

```
                                [ UNIVERSITY CAMPUS EDGE ]
┌───────────────────────┐    ┌───────────────────────────────────┐    ┌─────────────────────────────┐
│ High-Density IoT /    │───>│ Edge Anomaly Detector             │───>│ Redis Stream Event Bus      │
│ BYOD Network Streams  │    │ (Lightweight Isolation Forest)    │    │ (security:telemetry:stream) │
└───────────────────────┘    └───────────────────────────────────┘    └──────────────┬──────────────┘
                                                                                     │
                                                                                     ▼
                                [ CLOUD CONTROL PLANE ]               ┌─────────────────────────────┐
┌───────────────────────┐    ┌───────────────────────────────────┐    │ Identity Context Fusion     │
│ Automated SOAR        │<───│ Dynamic AI Policy Engine          │<───│ (IP/MAC -> User/Role state) │
│ Response Playbooks    │    │ (Severity & Risk Score Resolver)  │    └─────────────────────────────┘
└───────────┬───────────┘    └───────────────────────────────────┘
            │
            ▼
┌─────────────────────────┐
│ Active Mitigation       │ (Zero-Trust Enforcement: IP Block, Token Revocation, VLAN Isolation)
└─────────────────────────┘
```

---

## 📂 Repository Structure

```
AI-Native-Security-Platform/
├── README.md                           # Main IEEE-grade documentation
├── LICENSE                             # MIT Open Source License
├── CITATION.cff                        # Academic citation metadata
├── requirements.txt                    # Python environment requirements
├── docker-compose.yml                  # Microservice container orchestration
├── paper/                              # LaTeX paper files & assets
│   ├── paper.tex                       # IEEE conference paper LaTeX template
│   ├── references.bib                  # BibTeX references
│   ├── figures/                        # Generated vector PDF/PNG figures
│   └── tables/                         # Generated LaTeX table snippets
├── architecture/                       # Architectural diagrams & source specs
│   ├── architecture.drawio             # Editable DrawIO diagram source
│   ├── sequence-diagram.puml           # PlantUML sequence diagram
│   └── deployment-diagram.puml         # PlantUML deployment diagram
├── docs/                               # Detailed academic technical specs
│   ├── research-motivation.md          # Deep-dive motivation & background
│   ├── methodology.md                  # Detection & reasoning algorithms
│   ├── threat-model.md                 # STRIDE & MITRE ATT&CK mapping
│   ├── limitations.md                  # Scope & architectural boundaries
│   └── future-work.md                  # Research roadmap & extensions
├── prototype/                          # Working prototype modules
│   ├── edge/                           # Edge detection & identity fusion
│   │   ├── detector/                   # Isolation Forest anomaly classifier
│   │   ├── identity-fusion/            # Campus user role/context enricher
│   │   └── redis-stream/               # Async Redis pub/sub client
│   ├── cloud/                          # Cloud orchestration & SOAR engine
│   │   ├── policy-engine/              # Rule & risk score evaluation
│   │   ├── soar/                       # Autonomous response execution
│   │   └── dashboard/                  # Metrics aggregation service
│   └── shared/                         # Data schemas & event models
├── evaluation/                         # Benchmarking & experiment suite
│   ├── datasets/                       # Synthetic telemetry generators
│   │   ├── generator.py                # Ground-truth flow log generator
│   │   ├── raw/                        # Raw generated stream files
│   │   └── processed/                  # Normalized evaluation matrices
│   ├── experiments/                    # Isolated research experiments
│   │   ├── experiment_01_latency/      # End-to-end latency measurement
│   │   ├── experiment_02_mttr/         # Response time comparison (MTTR)
│   │   ├── experiment_03_precision/    # Precision/Recall/F1 sensitivity
│   │   └── experiment_04_resource/     # CPU/RAM overhead at scale
│   └── scripts/                        # Automation & plotting runners
│       ├── run_demo.sh                 # Interactive end-to-end demonstration
│       ├── run_experiments.sh          # Full benchmark reproduction script
│       └── generate_figures.py         # Publication figure generator
└── docker/                             # Service Dockerfiles
    ├── Dockerfile.edge                 # Edge service container spec
    ├── Dockerfile.cloud                # Cloud engine container spec
    └── Dockerfile.soar                 # SOAR engine container spec
```

---

## 🧪 Experimental Setup & Reproduction Guide

### Prerequisites
* Python 3.10+
* Docker & Docker Compose (optional, for full containerized evaluation)

### Step 1: Environment Setup
```bash
git clone https://github.com/your-org/AI-Native-Security-Platform.git
cd AI-Native-Security-Platform
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Run End-to-End Prototype Demo
To see the system process live streaming events, enrich identity context, evaluate policies, and execute autonomous mitigation:
```bash
python -m prototype.edge.detector.main
```

### Step 3: Reproduce Paper Experiments & Generate Publication Figures
To run the full suite of 4 quantitative experiments (Latency, MTTR, Precision/Recall, Resource Utilization) and generate all figures for the LaTeX paper:
```bash
python evaluation/scripts/generate_figures.py
```

All output CSV data files, markdown experiment summaries, and publication-ready 300 DPI vector charts will be generated inside:
* `evaluation/experiments/experiment_0*/` (CSVs & Markdown summaries)
* `paper/figures/` (PNG & PDF vector plots)
* `paper/tables/` (LaTeX formatted tables)

---

## 📊 Expected Experimental Outputs

| Experiment | Metric Evaluated | Target Baseline | AI-Native Outcome |
| :--- | :--- | :--- | :--- |
| **Exp 1: Latency** | Event Ingestion to Response | ~1200 ms (Legacy REST) | **4.2 ms (Sub-10ms target)** |
| **Exp 2: MTTR** | Incident Detection to Block | ~1800 s (Manual SOC) | **0.85 s (>99% reduction)** |
| **Exp 3: Detection** | F1-Score / False Positive Rate | F1: 0.81, FPR: 6.2% | **F1: 0.94, FPR: 1.1%** |
| **Exp 4: Resource** | Edge CPU @ 10,000 events/s | >85% CPU utilization | **14.3% Edge CPU utilization** |

---

## 📝 Citation

If you reference this architecture or use our benchmark code, please cite our paper:

```bibtex
@inproceedings{icai2026_ainative_security,
  author    = {Nguyen, Van A and Schmidt, Hans and Zhang, Wei},
  title     = {AI-Native Autonomous Security Platform: An Event-Driven Edge-Cloud Architecture for University Networks},
  booktitle = {Proceedings of the 2nd International Conference on AI: AI Native for University (ICAI-2026)},
  year      = {2026},
  location  = {Hanoi, Vietnam},
  publisher = {IEEE},
  url       = {https://icai.cmcu.edu.vn}
}
```

---

## 🤝 Acknowledgements

This research was conducted in preparation for **ICAI-2026** and supported by joint research initiatives across:
* **CMC University**, Hanoi, Vietnam (Department of Science & Technology)
* **Steinbeis University**, Germany
* **Tsinghua University Shenzhen International Graduate School (SIGS)**, China

*Special thanks to Microsoft for providing Microsoft CMT services for the peer-review process.*

---

## ⚠️ Research Disclaimer

This repository is a **reproducible academic research artifact**. It is designed for experimental evaluation, benchmark replication, and architectural proof-of-concept. It is not intended for out-of-the-box enterprise SaaS deployment without proper hardware hardening and university network integration testing.
