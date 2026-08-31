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

## Evaluation protocol: leakage control

Windows are produced from InSDN flows with sequence length 10 and **stride 1**, so
adjacent windows share 9 of 10 timesteps. A random train/test split therefore places
near-duplicate windows on both sides. All results below instead use
`train_utils.blocked_split_npz`, which:

1. cuts the corpus into 20 contiguous temporal blocks and assigns whole blocks;
2. purges 9 windows from both ends of every block, guaranteeing that no train window
   shares a single timestep with any test window (verified: 0 duplicates);
3. stratified-subsamples to 14,000 train / 6,000 test independently per side; and
4. fits the `StandardScaler` on train only.

Switching from the random split to this protocol dropped fixed-threshold F1 from
0.960 to 0.886 but left optimal-threshold model comparison unchanged (0.9715 → 0.9725).

## Headline results (real InSDN, leakage-controlled)

| Metric | Value |
| :--- | :--- |
| Software control-path latency | **2.068 ± 0.545 ms** (ingestion → playbook emission; **excludes** dataplane enforcement) |
| Time-to-decision (automated) | **1.950 ± 0.320 ms** vs manual SOC **1518.5 ± 335.3 s** (stipulated, not measured) |
| Detection @ τ=0.65 | P **0.985**, R **0.806**, F1 **0.886**, FPR **4.94%** |
| INT8 TCN–GRU SOTA F1 (tuned τ) | **0.9725** (vs LSTM AE 0.869 / Transformer AE 0.868) |
| INT8 model footprint | **0.048 MB** (ΔF1 vs FP32 = **0.0000**) |
| Saturation throughput | **477** win/s unbatched → **28,341** win/s at batch=128 |
| Edge CPU @ 10k win/s | **55.86%** per core (batch=128) — **H3 is rejected** |
| Cross-dataset InSDN→CIC | In-domain F1 **0.9419** → CIC zero-shot F1 **0.4658** (@τ=0.65); **0.4677** @τ*=0.1218 |

> Absolute latency figures are hardware- and load-dependent: repeated runs of the same
> suite on the same machine produced end-to-end means between 1.9 ms and 7.9 ms.
> Values above come from the most recent full run of `run_all_and_summarize.py`.

## Ablation and policy control (real, threshold tuned on train)

| Configuration | F1 | FPR |
| :--- | :--- | :--- |
| Full pipeline | 0.9602 | **12.05%** |
| w/o identity fusion | 0.9628 | 20.67% |
| w/o cloud risk engine | 0.9680 | 25.19% |

Identity fusion and the risk engine do **not** raise F1 — they cut the false positive
rate by ~52% relative, which is the trade that matters when the controller auto-blocks.
Identity context is a simulated overlay (InSDN has no identity labels); the
`rho` sweep is in `results/table8_ablation_rho_sweep.csv`.

The DQN policy agent reaches **99.2%** action accuracy against a brute-force oracle
(static ladder 38.2%, heuristic table 34.8%) with near-zero regret, but inflicts more
raw QoS disruption than the conservative baselines.

Reproduce individual tables:

```bash
python results/generate_table7.py   # cross-dataset
python results/generate_table8.py   # ablation
python results/generate_table9.py   # policy benchmark
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

* **Trịnh Hoàng Tú** (Hoang Tu Trinh) — HUFLIT Faculty of Information Technology (`23dh113972@st.huflit.edu.vn`)
* **Cao Tiến Thành** (Tien-Thanh Cao) — HUFLIT Faculty of Information Technology (`thanhct@huflit.edu.vn`)
