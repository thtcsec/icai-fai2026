# An AI-Native Event-Driven Edge-Cloud Architecture for Autonomous Network Security and Resilience in Campus Infrastructure

[![Conference](https://img.shields.io/badge/Conference-ICAI--2026-blue)](https://icai.cmcu.edu.vn)
[![IEEE Style Template](https://img.shields.io/badge/Template-IEEE%20Conference-orange)](https://www.ieee.org/conferences/publishing/templates)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

IEEE LaTeX paper + research artifact for **ICAI-2026 / ICAI-FAI 2026** (Track 1: ICT).

## Detection data (real)

| Corpus | Local path | Notes |
| :--- | :--- | :--- |
| **InSDN (real)** | `AI-Native-Security-Platform/datasets/processed/insdn_windows.npz` | 49,991 windows from public InSDN CSV |
| **CIC-IDS2017** | `.../cic_windows.npz` | Available for cross-dataset; primary tables use InSDN |

Primary paper metrics use a stratified **20,000**-window InSDN subset (seed=42).

## Headline results (real InSDN, regenerated)

| Metric | Value |
| :--- | :--- |
| End-to-end latency | **3.393 ± 1.024 ms** |
| MTTR (AI-native) | **4.685 ± 0.576 ms** vs manual SOC **1518.5 ± 335.3 s** |
| Detection @ τ=0.65 | P **0.967**, R **0.953**, F1 **0.960**, FPR **13.3%** |
| INT8 TCN–GRU SOTA F1 | **0.9715** (vs LSTM AE 0.871 / Transformer AE 0.872) |
| INT8 model footprint | **0.048 MB** |
| Cross-dataset InSDN→CIC | In-domain F1 **0.961** → CIC zero-shot F1 **0.286** (@τ=0.65); **0.468** @τ* |

Reproduce cross-dataset only:

```bash
python results/generate_table7.py
```

## Reproduce

```bash
cd AI-Native-Security-Platform
pip install -r requirements.txt
python results/run_all_and_summarize.py
```

```bash
# from repo root
pdflatex icaifai.tex
pdflatex icaifai.tex
```

## Authors

* **Trịnh Hoàng Tú** — HUFLIT Cybersecurity (`tht.csec2005@gmail.com`)
* **Cao Tiến Thành** (Tien-Thanh Cao) — HUFLIT Faculty of Information Technology (`thanhct@huflit.edu.vn`)
