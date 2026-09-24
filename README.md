# An AI-Native Event-Driven Edge-Cloud Architecture for Autonomous Network Security in Campus Infrastructure

[![Conference](https://img.shields.io/badge/Conference-ICAI--2026-blue)](https://icai.cmcu.edu.vn)
[![IEEE Style Template](https://img.shields.io/badge/Template-IEEE%20Conference-orange)](https://www.ieee.org/conferences/publishing/templates)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](AI-Native-Security-Platform/LICENSE)

IEEE LaTeX paper + research artifact for **The Second International Conference on AI: AI-Native for Universities and Industry (ICAI-2026)** (Information Technology and Communications track). Conference date: **3 December 2026**, Hanoi.

All headline numbers below are transcribed from `AI-Native-Security-Platform/results/paper_metrics_summary.json` produced by `results/run_all_and_summarize.py` (seed `42`). Do not treat older README/ARTIFACT text as authoritative.

## Detection data (real)

| Corpus | Local path | Notes |
| :--- | :--- | :--- |
| **InSDN (real)** | `AI-Native-Security-Platform/datasets/processed/insdn_windows.npz` | 49,991 windows from public InSDN CSV |
| **CIC-IDS2017** | `.../cic_windows.npz` | Cross-dataset holdout; primary tables use InSDN |

Primary paper metrics use a stratified **20,000**-window InSDN subset under a blocked + purged split (14,000 train / 6,000 test).

## Evaluation protocol: leakage control

Windows use sequence length 10 and **stride 1**. All reported accuracy tables use `train_utils.blocked_split_npz` (20 temporal blocks, purge 9, train-only scaler). Comparing that protocol to a random split + pooled scaler on this run yields a **near-null** leakage effect (`|ΔF1| < 0.002` at fixed τ).

## Headline results (from `paper_metrics_summary.json`)

| Metric | Value |
| :--- | :--- |
| Software control-path latency (incl. Redis Streams localhost) | **median 4.194 ms**, p95 6.753 ms, **p99 7.470 ms** |
| Redis stage alone (XADD+XREADGROUP) | median **1.580 ms** (37.7% of median total) |
| Detection @ τ=0.65 | P **0.978**, R **0.894**, F1 **0.9340**, FPR **8.20%** |
| INT8 TCN–GRU SOTA F1 (test-tuned τ) | **0.9723** (FP32 0.9725; LightGBM **0.9886**) |
| INT8 model footprint | **0.048 MB** (`|ΔF1|` vs FP32 = **0.0002**) |
| Edge CPU @ 10k win/s (1 logical CPU, `torch.set_num_threads(1)`) | **36.0%** (range 35.0–49.1) — **H2 rejected** |
| Cross-dataset InSDN→CIC @ τ=0.65 | In-domain F1 **0.9340** → CIC **0.4564** |
| DQN vs oracle (analytic reward) | Action acc. **98.8%**, mean regret **0.0** |

> Absolute latency/CPU figures are hardware- and load-dependent. Redis must be running on `localhost:6379` (e.g. `docker run -p 6379:6379 redis:7.2.4`) before latency/ablation generators that claim Redis Streams.

## Reproduce

```bash
# Redis 7.x required for latency + Redis ablation arms
docker run -d --name icai-redis -p 6379:6379 redis:7.2.4

cd AI-Native-Security-Platform
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
python results/run_all_and_summarize.py
```

```bash
# from repo root
pdflatex icaifai.tex
pdflatex icaifai.tex
```

## Authors

* **Hoang Tu Trinh** — HUFLIT Faculty of Information Technology (`23dh113972@st.huflit.edu.vn`)
* **Tien Thanh Cao** — HUFLIT Faculty of Information Technology (`thanhct@huflit.edu.vn`)

## Citation

See `AI-Native-Security-Platform/CITATION.cff`. Publisher is **ICAI-2026 / CMC University proceedings** (not IEEE as the publication venue).
