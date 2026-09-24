# Artifact Evaluation Guide

Paper title (must match PDF):

> **An AI-Native Event-Driven Edge-Cloud Architecture for Autonomous Network Security in Campus Infrastructure**

Conference: **The Second International Conference on AI: AI-Native for Universities and Industry (ICAI-2026)**, 3 December 2026, Hanoi.

This guide verifies claims against `results/paper_metrics_summary.json`. **Do not** expect legacy figures such as 4.218 ms mean latency, F1=0.9400, or CPU=14.30% — those belong to an older draft.

---

## 1. Prerequisites

* Python 3.10+
* Redis **7.x** on `localhost:6379` (latency + Redis ablation require a live server; silent in-memory fallback is disabled for those benches)
* Optional: Docker (`docker run -d -p 6379:6379 redis:7.2.4`)

```bash
git clone https://github.com/thtcsec/icai-fai2026.git
cd icai-fai2026/AI-Native-Security-Platform
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

Full regeneration (writes all CSVs/figures + `results/paper_metrics_summary.json`):

```bash
python results/run_all_and_summarize.py
```

Approximate wall time: tens of minutes on a laptop CPU (training + 5-repeat resource loops).

---

## 2. Claims ↔ evidence (current run)

| Claim | Expected check (from summary JSON) | Generator |
| :--- | :--- | :--- |
| **RQ1 control-path latency** includes Redis Streams + HMAC principals | Total median ≈ **2.932 ms**, p99 ≈ **4.851 ms**; no cleartext IP on Redis | `results/generate_table2.py` → `table2_latency.csv` |
| **Detection @ τ=0.65** | F1 ≈ **0.9340** | `results/generate_table4.py` |
| **H2 edge CPU** | @10k win/s CPU ≈ **36.0% of one logical CPU** (H2 rejected vs 10%) | `results/generate_table5.py` |
| **Trees beat TCN–GRU** | LightGBM F1 **0.9886** > INT8 TCN–GRU **0.9723** | `results/generate_table6.py` |
| **Zero-shot CIC collapse** | CIC F1 @0.65 ≈ **0.4564** | `results/generate_table7.py` |
| **Redis transport is real** | Ablation prints `redis-streams=…ms` (not a Python list append) | `results/generate_table8.py` |
| **DQN vs analytic oracle** | Action accuracy ≈ **98.8%** | `results/generate_table9.py` |
| **Leakage near-null** | `|ΔF1|` at fixed τ ≈ **0.0018** | `results/generate_table10_leakage.py` |

MTTR table (`generate_table3.py`) still reports a **stipulated** manual-SOC baseline for contrast; the paper does not treat that baseline as a measured human study.

---

## 3. CPU metric definition (important)

`generate_table5.py` reports `psutil.Process.cpu_percent()` **without** dividing by `cpu_count()`. Under the psutil convention, **100% = one logical CPU fully busy**. PyTorch is pinned with `torch.set_num_threads(1)` for the H2 experiment. Low offered-load rows are sleep-dominated and can understate busy-period CPU; the H2 decision uses the 10,000 windows/s row.

---

## 4. Mapping to paper tables

| Paper content | Artifact file |
| :--- | :--- |
| Latency breakdown | `results/table2_latency.csv` |
| Threshold sensitivity | `results/table4_precision.csv` |
| Resource / CPU | `results/table5_resource.csv`, `table5_saturation.csv` |
| SOTA / quantization | `results/table6_sota_comparison.csv` |
| Cross-dataset | `results/table7_*.csv` (via generator) |
| Ablation + ρ sweep | `results/table8_ablation.csv`, `table8_ablation_rho_sweep.csv` |
| Policy oracle | `results/table9_policy_benchmark.csv` |
| Machine-readable digest | `results/paper_metrics_summary.json` |

---

## 5. Docker tip

```bash
docker run -d --name icai-redis -p 6379:6379 redis:7.2.4
python results/generate_table2.py
```

If Redis is down, latency/ablation generators **fail loudly** rather than silently substituting an in-process list.
