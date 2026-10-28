# ICAI-2026 Paper 26 — Audited Research Artifact (Under Review)

**Status:** draft pre-release preparation. Paper 26 is Submitted / Under Review, not accepted. No publisher DOI, indexing, IEEE publication or camera-ready status is claimed. Zenodo metadata is unpublished preparation only.

## Research contribution

This is the software/evidence artifact for *Evaluating an Event-Driven Edge–Cloud Intrusion Detection Prototype: Host-Day Grouping and Threshold Transfer on InSDN*, by Hoang Tu Trinh and Tien Thanh Cao (HUFLIT), ICAI-2026, CMC University. It examines host-day grouping and threshold transfer for a TCN–GRU flow-window detector on one pinned third-party InSDN derivative.

## Historical versus corrected results

The original tag `icai2026-submission-artifact` and commit `4540dbf8c7b8b50ac638c034b1c6c8fb090e06b9` are preserved. The current artifact distinguishes chronological P1 (entity overlap) from group-disjoint P3, withdraws the contaminated historical model-selection comparison, and bounds identity/event/Redis/policy/SOAR claims to executable evidence.

P1 F1: LightGBM 0.970, dynamic INT8 TCN–GRU 0.904. P3: 955 test windows, 32 overlapping positives from one attack host-day, zero group overlap; TCN validation-frozen F1 0.121 and AP 1.0; LightGBM F1 0.175. P1/P3 have different populations and trained models, not a controlled causal comparison. The extract has 143 normal/17 attack windowable host-days, zero mixed.

## Reproducibility and dependencies

Run `python tools/verify_public_artifact.py` for standard-library checks of stored evidence/checksums, then follow ARTIFACT.md for dependencies, pytest and fixture tests. Dataset-dependent reconstruction/training is optional, expensive and was not run for this release. The local audited closure suite passed 62 tests; clean public-checkout checks are recorded separately in release/verification.json.

Detector verification uses Python 3.11.9/Torch 2.13.0+cpu; historical repaired no-Redis latency records Python 3.13.5/Torch 2.8.0+cpu. Immutable training execution traces are unavailable. Raw CSVs, row-level samples and derived NPZ files are excluded from the new release tree. Owned model checkpoints and aggregate evidence are checksummed. Local protected data files and historical tables remain unchanged; the historical tag is not rewritten.

## Scientific and software limits

One test attack host-day, overlapping windows, third-party extract, no globally unseen-host/session claim, uncalibrated batch-sensitive scores, simulator-only one-step Q-regression and unenforced recommendations. Corrected live Redis receipt-to-policy integration and dataplane mitigation remain unverified. This is research software, not a deployment release.

## Paper distribution and citation

The approved five-page local PDF SHA-256 is `ccba97053f3c93ba27e7a8cf99ce422246ea63f698b835092f7d10a547f2b83d`. Revised PDF/TeX are withheld pending author confirmation of CMT replacement and conference sharing clearance. GitHub Release remains **draft** while those gates are unverified; no manuscript asset is attached.

CITATION.cff identifies the audited hybrid title and approved author order. No ORCID or DOI is invented. Project code is MIT; datasets and logos retain their owners' rights. The publication manifest enumerates inclusions and exclusions for GitHub automatic source archives.

## Packaging revision v2

Tag `icai2026-revision-20261028-v2` corrects JSON checksum newline portability identified by independent GitHub archive verification. The first revision tag is preserved and its draft is superseded. Scientific values, model bytes and the frozen local manuscript are unchanged. JSON hashes identify LF-canonical public Git bytes; Windows CRLF checkout conversion is handled explicitly.
