<p align="center">
  <img src="https://img.shields.io/badge/ICAI-2026-003147.svg" alt="ICAI-2026" />
</p>

<h1 align="center">An AI-Native Event-Driven Edge–Cloud Architecture</h1>
<p align="center"><em>Autonomous network security in campus infrastructure</em></p>

<p align="center">
  <a href="https://icai.cmcu.edu.vn"><img src="https://img.shields.io/badge/Conference-ICAI--2026-blue.svg" alt="ICAI-2026" /></a>
  <a href="https://github.com/thtcsec/icai-fai2026/tree/icai2026-submission-artifact"><img src="https://img.shields.io/badge/artifact-submission-2ea44f.svg" alt="Submission artifact tag" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/code%20license-MIT-blue.svg" alt="MIT license" /></a>
  <a href="ARTIFACT.md"><img src="https://img.shields.io/badge/reproducibility-seed%2042-informational.svg" alt="Seeded artifact" /></a>
</p>

## Paper metadata

| Field | Record |
|---|---|
| **Title** | *An AI-Native Event-Driven Edge–Cloud Architecture for Autonomous Network Security in Campus Infrastructure* |
| **Authors** | Hoang Tu Trinh and Tien Thanh Cao |
| **Affiliation** | Faculty of Information Technology, HUFLIT |
| **Venue** | ICAI-2026, CMC University, Hanoi, 3 December 2026 |
| **Track** | Information Technology and Communications |
| **Status** | Submitted (CMT Paper ID 26). Under review. Not accepted. No DOI. |
| **Publisher** | ICAI-2026 / CMC University proceedings (IEEE template is formatting only) |
| **Artifact tag** | [`icai2026-submission-artifact`](https://github.com/thtcsec/icai-fai2026/tree/icai2026-submission-artifact) |
| **Code license** | MIT (`LICENSE`). InSDN and CIC-IDS2017 remain under their owners' terms. |

Manuscript PDF is submitted via CMT only and is **not** stored in this repository. Source: `icaifai.tex`.

## Overview

INT8-quantized TCN–GRU detectors score non-DPI flow telemetry at edge gateways. Redis Streams carries HMAC-pseudonymized events to a cloud risk resolver and a DQN SDN policy that **emits** SOAR descriptors. Dataplane installation is outside what we measure.

## Headline results

From `AI-Native-Security-Platform/results/paper_metrics_summary.json` (seed `42`):

| Metric | Value |
|---|---|
| Control-path latency (Redis + HMAC) | median **4.793 ms**, p99 **7.202 ms** |
| Detection @ τ=0.65 | F1 **0.9340** |
| INT8 footprint | **0.048 MB** (`\|ΔF1\|` vs FP32 = 0.0002) |
| LightGBM vs INT8 TCN–GRU | **0.9886** > **0.9723** |
| Edge CPU @ 10k win/s | **36.0%** of one logical CPU — **H2 rejected** |
| Zero-shot InSDN→CIC @ τ=0.65 | **0.4564** |

Absolute latency/CPU figures are hardware-dependent. Redis 7.x must be on `localhost:6379` for latency/ablation arms.

## Reproduce

```bash
docker run -d --name icai-redis -p 6379:6379 redis:7.2.4
cd AI-Native-Security-Platform
python -m venv .venv
# Windows: .venv\Scripts\activate
python -m pip install -r requirements-lock.txt
python results/run_all_and_summarize.py
```

## Limitations

Reconstruction-error scoring is unusable on this attack-heavy corpus. Localhost Redis is not faster than REST loopback; the bus is for the multi-consumer model. This repository does not claim IEEE as the publication venue.

## Citation

```bibtex
@inproceedings{trinh2026ainative,
  author    = {Hoang Tu Trinh and Tien Thanh Cao},
  title     = {An AI-Native Event-Driven Edge--Cloud Architecture for Autonomous Network Security in Campus Infrastructure},
  booktitle = {Proc. Second Int. Conf. AI: AI-Native for Universities and Industry (ICAI)},
  year      = {2026},
  note      = {Submitted manuscript (CMT 26). Artifact: https://github.com/thtcsec/icai-fai2026/tree/icai2026-submission-artifact}
}
```

Machine-readable citation: [`CITATION.cff`](CITATION.cff). Evaluation protocol: [`ARTIFACT.md`](ARTIFACT.md).

## License

Code is MIT. Do not redistribute third-party NIDS datasets from this repository.
