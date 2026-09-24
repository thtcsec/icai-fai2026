# AI-Native Security Platform (ICAI-2026 Artifact)

Companion code for:

**An AI-Native Event-Driven Edge-Cloud Architecture for Autonomous Network Security in Campus Infrastructure**

Submitted to **The Second International Conference on AI: AI-Native for Universities and Industry (ICAI-2026)** — 3 December 2026, Hanoi.

Authoritative metrics: [`results/paper_metrics_summary.json`](results/paper_metrics_summary.json). Root [`README.md`](../README.md) and [`ARTIFACT.md`](ARTIFACT.md) mirror that file.

## Quick start

```bash
# Required for Redis latency / ablation arms
docker run -d --name icai-redis -p 6379:6379 redis:7.2.4

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
python results/run_all_and_summarize.py
```

## What is measured

| Path | Includes | Excludes |
| :--- | :--- | :--- |
| Control-path latency (`generate_table2.py`) | INT8 inference, simulated identity + HMAC `principal_id`, Redis `XADD`+`XREADGROUP`, DQN→SOAR **selected** action descriptor | Dataplane `flow_mod`, capture/FE/windowing, cross-host RTT |
| CPU @ paced load (`generate_table5.py`) | `psutil` % of **one** logical CPU, `torch.set_num_threads(1)` | Division by `cpu_count()` (removed; that was mislabelled) |
| Redis vs REST ablation (`generate_table8.py`) | Live Redis Streams vs localhost HTTP | In-process list append labelled as Redis |

## Current headline numbers (seed 42)

* Latency median **2.932 ms**, p99 **4.851 ms** (HMAC principals; DQN→SOAR wired)
* F1 @ τ=0.65 **0.9340**
* CPU @ 10k windows/s **36.0%** of one logical CPU (H2 rejected vs 10%)
* LightGBM **0.9886** > INT8 TCN–GRU **0.9723**
* CIC zero-shot F1 **0.4564**
* DQN oracle action accuracy **98.8%**

## Layout

```
prototype/edge/detector/     TCN-GRU + INT8 quantize
prototype/edge/identity_fusion/
prototype/edge/redis_stream/ Redis Streams client (XADD/XREADGROUP)
prototype/cloud/policy_engine/ DQN agent
prototype/soar/              Playbook construction (no OpenFlow install)
results/                     Generators + CSV/JSON evidence
datasets/processed/          InSDN / CIC window tensors
```

## Citation

Use [`CITATION.cff`](CITATION.cff). Venue is ICAI-2026 proceedings (not IEEE as publisher).

## License

MIT — see [`LICENSE`](LICENSE).
