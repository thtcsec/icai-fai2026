<p align="center">
<table>
<tr>
<td align="center" valign="middle" width="200">
  <img src="assets/organizers/cmc.png" height="48" alt="CMC University, host of ICAI-2026" />
</td>
<td align="center" valign="middle" width="200">
  <img src="assets/organizers/tsinghua_shenzhen.png" height="64" alt="Tsinghua Shenzhen International Graduate School, ICAI-2026 collaborator" />
</td>
<td align="center" valign="middle" width="200">
  <img src="assets/organizers/steinbeis.png" height="46" alt="Steinbeis University – Schools of Next Practices, ICAI-2026 collaborator" />
</td>
</tr>
</table>
<sub>Conference host (CMC University) and collaborators (Tsinghua SIGS, Steinbeis). Authors are affiliated with HUFLIT, not CMC University, Tsinghua SIGS, or Steinbeis.</sub>
</p>

<h1 align="center">Evaluating an Event-Driven Edge–Cloud Intrusion Detection Prototype</h1>
<p align="center"><em>Host-day grouping and threshold transfer on an audited third-party InSDN extract</em></p>

<p align="center">
  <a href="https://icai.cmcu.edu.vn/icai-2026"><img src="https://img.shields.io/badge/ICAI-2026-003147.svg" alt="ICAI-2026" /></a>
  <a href="revisions/icai2026_20261028/PUBLIC_ARTIFACT.md"><img src="https://img.shields.io/badge/artifact-audited%20revision-2ea44f.svg" alt="Audited research artifact" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/code%20license-MIT-blue.svg" alt="MIT code license" /></a>
  <a href="ARTIFACT.md"><img src="https://img.shields.io/badge/reproducibility-checksums%20%2B%20evidence-informational.svg" alt="Checksums and limited evidence checks" /></a>
</p>

## Paper metadata

| Field | Record |
|---|---|
| **Title** | *Evaluating an Event-Driven Edge–Cloud Intrusion Detection Prototype: Host-Day Grouping and Threshold Transfer on InSDN* |
| **Authors** | Hoang Tu Trinh and Tien Thanh Cao |
| **Affiliation** | Faculty of Information Technology, Ho Chi Minh City University of Foreign Languages — Information Technology (HUFLIT) |
| **Conference** | ICAI-2026, CMC University, Hanoi, 3 December 2026 |
| **Paper ID / status** | 26; Submitted / Under Review, **not accepted**. Status supplied by the repository owner; live CMT remains author-controlled. |
| **Current artifact** | [Audited revision](revisions/icai2026_20261028/PUBLIC_ARTIFACT.md); tag `icai2026-revision-20261028-v2` |
| **Historical submission** | [icai2026-submission-artifact](https://github.com/thtcsec/icai-fai2026/tree/icai2026-submission-artifact), preserved unchanged |
| **Paper distribution** | Approved five-page PDF remains local pending confirmed CMT replacement and conference sharing clearance; no revised manuscript PDF or TeX is included in this code release. |
| **License** | MIT for project code; dataset and logo owners retain their rights. No publisher DOI, acceptance, indexing or camera-ready status is claimed. |

## Overview

This research artifact evaluates a TCN–GRU flow-window detector inside an event-driven edge–cloud software prototype. Its contribution is an extract-specific case study of host-day grouping and threshold transfer. High ranking on one held-out attack host-day coexists with poor binary F1 at a validation-frozen threshold.

The prototype implements local scoring, HMAC pseudonymization and in-memory composition with a one-step Q-regressor and recommended SOAR descriptors. It does not establish autonomous mitigation, live cloud receipt-to-policy integration or dataplane enforcement.

## Main results

P1 and P3 use separately trained models and different test populations. Their difference is descriptive, not a controlled causal leakage effect.

| Protocol / evidence | LightGBM | Dynamic INT8 TCN–GRU | Scope |
|---|---|---|---|
| P1 clock-forward F1 | **0.970** | **0.904** | Entity-overlapping host-days; 6,000 test windows |
| P3 validation-frozen F1 | **0.175** | **0.121** | 955 test windows; 32 positives from **one attack host-day**; zero group overlap |
| P3 continuous-score AP | 0.177 | **1.0** | Ranking of this narrow test realization, not population generalization |
| P3 always-normal / always-attack F1 | 0 / 0.065 | — | Explicit class baselines |
| One-step policy simulator | — | 99.28% analytic-oracle match | Immediate-reward regression, not sequential Bellman DQN |

Windowable host-days in this extract: **143 normal, 17 attack, zero mixed**. P3 consumes 14,000 training windows from 66 groups (nine attack), 3,000 validation windows from 16 groups (four attack), and all 955 test windows from 22 groups (one attack).

P3 headline whole-tensor CM is `[[457,466],[0,32]]`, F1 `0.12075471698113208`, at validation-selected threshold `0.002417233586311342`. The audited batch-1 CM is `[[454,469],[0,32]]`; maximum observed score change is about 0.1356. Streaming equivalence is unverified. Scores are uncalibrated.

## Data provenance and availability

The measured input is the **third-party InSDN derivative** [Sharukesh/INSDN](https://huggingface.co/datasets/Sharukesh/INSDN/tree/f59330c8ff0f1b231b382e05fc41b1e8fb0df6b4), `Dataset.csv`, revision `f59330c8ff0f1b231b382e05fc41b1e8fb0df6b4`, 343,889 rows, binary `target > 0` mapping. It is not an authenticated original publisher export. All grouping findings concern these exact bytes and mapping.

SHA-256: `a93d51e1c2debc7246728578f22d84a442f156628be64b8c9bae5ccc5dd88a42`.

Raw CSVs, row-level samples and derived NPZ datasets are **not bundled in this release**. Obtain datasets separately under their owners' terms. [Dataset documentation](AI-Native-Security-Platform/datasets/README.md) records provenance and the historical archive hashes. Historical tagged files remain unchanged; the new release tree excludes dataset payloads while preserving local copies.

## Quick verification — no datasets or Redis

```bash
python tools/verify_public_artifact.py
python -m venv .venv
# Windows: .venv\Scripts\activate
python -m pip install -r AI-Native-Security-Platform/requirements-lock.txt
python -m pip install -r requirements-test.txt
python tools/run_public_checks.py
```

The first command uses only the Python standard library and checks aggregate evidence, metadata and listed file hashes. It is a **limited consistency smoke check**, not independent scientific verification or a model-performance replay. The test runner uses synthetic fixtures and local code; dataset-dependent checks explicitly skip when the corresponding files are absent. Docker and Redis are not required.

The complete **62-test local closure suite** also requires the withheld manuscript packet and locally retained datasets. Its passed result is recorded in [freeze evidence](revisions/icai2026_20261028/evidence/scientific_freeze.json); this is distinguished from the clean public-checkout test run.

## Optional experimental reproduction

See [ARTIFACT.md](ARTIFACT.md) for data preparation, checkpoint paths, environments and commands. Existing builders and training runners are supplied for optional reproduction; model training can be expensive and is not part of quick verification. The release workflow does not rerun training or alter the frozen findings.

## Limitations and implementation boundaries

- One attack host-day and overlapping positive windows limit population inference; AP=1 is not broad detector generalization.
- Host-day disjointness is not globally unseen-host evaluation; no session-level sequence claim is made.
- Partial dynamic INT8 quantizes Linear/GRU modules while convolutions stay FP32; whole-tensor results do not establish streaming equivalence.
- Unknown addresses map to UNKNOWN, UNREGISTERED, identity_source=none, authenticated=False, trust=0.10. Identity fixtures do not provide campus AAA.
- The scored harness uses dictionary events. Legacy SecurityEvent lacks first-class principal_id/p_attack fields. Timed Redis returns are discarded; verified cloud receipt-to-policy integration is absent.
- Corrected Redis-inclusive timing and dataplane enforcement are **unmeasured**. Historical Table II's 4.793 ms used hardcoded detector input and unloaded policy weights.
- The historical 0.987/0.968 model comparison is withdrawn as independent selection evidence because the TCN's later validation carve was in-sample and effective training pools differed.
- Fixed seed-42 splitting and the fixed-list ten-seed support census do not establish preregistration or repeated model performance.

## Historical versus current versions

The original `icaifai.tex`, historical tables and submission tag are provenance, not the current scientific headline. The current audited manuscript remains under `revisions/icai2026_20261028/` in the author's local packet. This code release supplies portable aggregate evidence and owned detector/policy checkpoints without exposing that paper packet. [Paper locations](paper/README.md) explains this boundary.

## Citation

```bibtex
@misc{trinh2026icaiartifact,
  author = {Hoang Tu Trinh and Tien Thanh Cao},
  title = {Evaluating an Event-Driven Edge--Cloud Intrusion Detection Prototype: Host-Day Grouping and Threshold Transfer on {InSDN}},
  year = {2026},
  howpublished = {ICAI-2026 Paper 26 research software artifact},
  note = {Submitted / Under Review; not accepted. Audited revision 2026-10-28},
  url = {https://github.com/thtcsec/icai-fai2026}
}
```

Canonical citation: [CITATION.cff](CITATION.cff). [Zenodo metadata](.zenodo.json) is unpublished preparation only; no upload or DOI minting is part of this release.

## License and security

Project code is [MIT](LICENSE). Third-party datasets and institutional logos retain their owners' rights; logos identify conference roles, not paper endorsement. Do not redistribute campus logs or dataset payloads. Report security issues using [SECURITY.md](SECURITY.md).
