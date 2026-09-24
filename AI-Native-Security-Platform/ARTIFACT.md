# Artifact Evaluation Guide

Same content as the repository-root [`../ARTIFACT.md`](../ARTIFACT.md). Keep both files identical when updating claims.

Paper title:

> **An AI-Native Event-Driven Edge-Cloud Architecture for Autonomous Network Security in Campus Infrastructure**

Conference: **ICAI-2026** (3 December 2026, Hanoi).

Regenerate evidence:

```bash
# Redis 7.x on localhost:6379 required
python results/run_all_and_summarize.py
```

Verify numbers only against `results/paper_metrics_summary.json` (not older draft READMEs).

Current headline checks:

* Latency median ≈ **4.793 ms**, p99 ≈ **7.202 ms** (Redis XADD+XREADGROUP + HMAC `principal_id`; DQN action wired to SOAR)
* F1 @ τ=0.65 ≈ **0.9340**
* CPU @ 10k ≈ **36.0% of one logical CPU** (H2 rejected)
* LightGBM F1 **0.9886** > INT8 TCN–GRU **0.9723**
* CIC zero-shot F1 ≈ **0.4564**
* DQN oracle action accuracy ≈ **98.8%**
