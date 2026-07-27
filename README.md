# An AI-Native Event-Driven Edge-Cloud Architecture for Autonomous Network Security and Resilience in Campus Infrastructure

[![Conference](https://img.shields.io/badge/Conference-ICAI--2026-blue)](https://icai.cmcu.edu.vn)
[![Peer Review Service](https://img.shields.io/badge/Peer--Review-Microsoft%20CMT-0078D4)](https://cmt3.research.microsoft.com)
[![IEEE Style Template](https://img.shields.io/badge/Template-IEEE%20Conference-orange)](https://www.ieee.org/conferences/publishing/templates)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-Zenodo--Ready-success)](.zenodo.json)

Open Research Artifact Repository and IEEE LaTeX Paper Source for **The Second International Conference on AI: AI Native for University (ICAI-2026 / ICAI-FAI 2026)**.

---

## 📌 1. Paper Overview & Title

**"An AI-Native Event-Driven Edge-Cloud Architecture for Autonomous Network Security and Resilience in Campus Infrastructure"**

### Author & Affiliation
* **Trịnh Hoàng Tú (Trinh Hoang Tu)** - *Lead Author*  
  *Department of Cybersecurity, Faculty of Information Technology*  
  *Ho Chi Minh City University of Foreign Languages - Information Technology (HUFLIT), Ho Chi Minh City, Vietnam*  
  Email: `tht.csec2005@gmail.com`

---

## 📝 2. Abstract

Modern university campus networks present extreme device heterogeneity, unmanaged BYOD endpoints, open research laboratories, and massive concurrent telemetry flows. Traditional Security Operations Center (SOC) architectures rely on centralized, manual incident response workflows that suffer from long Mean Time to Respond (MTTR $>$ 30 minutes) and severe alert fatigue. In this paper, we propose an **AI-Native Autonomous Security and Resilience Platform** engineered specifically for higher education campus networks. By decoupling threat detection into lightweight edge sequence autoencoders (PyTorch INT8-quantized TCN-GRU) and streaming high-confidence security events over an asynchronous Redis Stream bus to a cloud-based dynamic risk resolver and Deep Q-Network (DQN) SDN controller, our platform achieves sub-10ms event-to-mitigation latency. We construct a fully reproducible open-source research artifact and evaluate it empirically using benchmark intrusion datasets (InSDN, CSE-CIC-IDS2018) and TraCI SUMO/Mininet-WiFi traffic traces. Experimental results demonstrate an end-to-end mitigation latency of 4.218~ms, a 99.99\% reduction in MTTR compared to manual SOC triage (0.0085~s vs 1512.4~s), an optimal F1-score of 0.940 (Precision=0.9482, Recall=0.9320, FPR=1.1\%) at adaptive decision threshold ($\tau=0.65$), and minimal edge CPU overhead (14.3\% CPU and 49.0~MB RAM under 10,000 events/sec throughput).

---

## 🏗️ 3. System Architecture

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

## 📂 4. Repository Tree

```
icai-fai2026/
├── README.md                           # Main IEEE-grade academic documentation
├── ARTIFACT.md                         # Artifact Evaluation (AE) Committee Guide
├── SECURITY.md                         # Security vulnerability reporting policy
├── CITATION.cff                        # Citation metadata (Zenodo / GitHub DOI ready)
├── LICENSE                             # MIT Open Source License
├── icaifai.tex                         # Official IEEE double-column LaTeX source
├── icaifai.pdf                         # Compiled Camera-Ready PDF paper (4 pages)
├── IEEEtran.cls                        # IEEE Class formatting specification
├── IEEEtran.bst                        # IEEE Bibliography style specification
└── AI-Native-Security-Platform/        # Complete Reproducible Research Artifact
    ├── README.md                       # Artifact deep-dive documentation
    ├── CITATION.cff                    # BibTeX metadata
    ├── .zenodo.json                    # Zenodo 1-click DOI metadata
    ├── results/                        # Raw CSV results & standalone generators
    │   ├── table2_latency.csv          # Stage-by-stage latency breakdown data
    │   ├── table2_latency.png          # Latency breakdown plot
    │   ├── table2_latency.ipynb        # Interactive latency analysis notebook
    │   ├── generate_table2.py          # Standalone Table II & Fig 1 runner
    │   ├── table3_mttr.csv             # Incident response MTTR benchmark data
    │   ├── table3_mttr.png             # MTTR comparison plot
    │   ├── table3_mttr.ipynb           # Interactive MTTR analysis notebook
    │   ├── generate_table3.py          # Standalone Table III & Fig 2 runner
    │   ├── table4_precision.csv        # Anomaly threshold sensitivity metrics
    │   ├── table4_precision.png        # Precision/Recall plot
    │   ├── table4_precision.ipynb      # Interactive sensitivity notebook
    │   ├── generate_table4.py          # Standalone Table IV & Fig 3 runner
    │   ├── table5_resource.csv         # Edge CPU/RAM scaling data
    │   ├── table5_resource.png         # Resource scaling plot
    │   ├── table5_resource.ipynb       # Interactive resource notebook
    │   └── generate_table5.py          # Standalone Table V & Fig 4 runner
    ├── prototype/                      # Working prototype modules (from tu_projects)
    │   ├── edge/                       # TCN-GRU sequence detector & quantizer
    │   ├── identity-fusion/            # Spatial-temporal identity fusion module
    │   ├── redis-stream/               # Async Redis stream pub/sub client
    │   ├── dqn-controller/             # Deep Q-Network SDN resilience agent
    │   └── soar/                       # Dynamic Risk Scorer & Playbooks
    └── evaluation/                     # Benchmark suite & figure runners
        ├── experiments/                # Isolated experiment folders (01 to 04)
        └── scripts/                    # Master benchmark runner (generate_figures.py)
```

---

## ⚡ 5. Quick Start Reproduction Guide

### Environment Setup
```bash
git clone https://github.com/thtcsec/icai-fai2026.git
cd icai-fai2026/AI-Native-Security-Platform
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Reproduce Individual Paper Tables & Figures
```bash
python results/generate_table2.py   # Table II & Fig 1 (Latency Breakdown)
python results/generate_table3.py   # Table III & Fig 2 (MTTR Comparison)
python results/generate_table4.py   # Table IV & Fig 3 (Threshold Sensitivity)
python results/generate_table5.py   # Table V & Fig 4 (Edge Resource Scaling)
```

---

## 📊 6. Evidence Mapping Summary

| Paper Element | Paper Section | Artifact CSV | Generator Script | Output Chart |
| :--- | :--- | :--- | :--- | :--- |
| **Fig. 1** | Sec III / IV-A | `results/table2_latency.csv` | `results/generate_table2.py` | `results/table2_latency.png` |
| **Table I** | Sec IV | `docs/methodology.md` | N/A | N/A |
| **Table II** | Sec IV-A | `results/table2_latency.csv` | `results/generate_table2.py` | `results/table2_latency.png` |
| **Table III / Fig. 2** | Sec IV-B | `results/table3_mttr.csv` | `results/generate_table3.py` | `results/table3_mttr.png` |
| **Table IV / Fig. 3** | Sec IV-C | `results/table4_precision.csv` | `results/generate_table4.py` | `results/table4_precision.png` |
| **Table V / Fig. 4** | Sec IV-D | `results/table5_resource.csv` | `results/generate_table5.py` | `results/table5_resource.png` |

---

## 📜 7. Citation & BibTeX

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

---

## 🔒 8. Security Policy & License

* **License**: MIT Open Source License ([LICENSE](LICENSE))
* **Vulnerability Reporting**: See [SECURITY.md](SECURITY.md)
