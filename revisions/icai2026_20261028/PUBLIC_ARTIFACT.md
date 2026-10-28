# ICAI-2026 Paper 26 — audited research artifact

Current title: **Evaluating an Event-Driven Edge–Cloud Intrusion Detection Prototype: Host-Day Grouping and Threshold Transfer on InSDN**. Authors: **Hoang Tu Trinh; Tien Thanh Cao**, HUFLIT. Submitted / Under Review; not accepted. Conference host: CMC University. Institutional logos identify organizer/collaborator roles, not author affiliation or endorsement.

The approved five-page PDF is frozen at SHA-256 `ccba97053f3c93ba27e7a8cf99ce422246ea63f698b835092f7d10a547f2b83d`. It and its TeX remain local pending confirmed CMT replacement and conference sharing clearance. This directory publishes aggregate evidence, not a public manuscript candidate.

## Evidence map

| Evidence | Portable copy |
|---|---|
| Scientific freeze and closure summary | [scientific_freeze.json](evidence/scientific_freeze.json) |
| Protocol/model/dataset provenance | [experiment_manifest.json](evidence/experiment_manifest.json) |
| Full and consumed support | [consumed_support.json](evidence/consumed_support.json) |
| P3 operating point and ranking diagnostics | [metric_integrity.json](evidence/metric_integrity.json) |
| P1 / P3 model results | [P1](evidence/p1_model_comparison.json), [P3](evidence/p3_model_comparison.json) |
| Fixed-list support census | [fixed_seed_support.json](evidence/fixed_seed_support.json) |
| Simulator-only policy evaluation | [policy_evaluation.json](evidence/policy_evaluation.json) |
| Historical repaired no-Redis latency | [historical_repaired_latency.json](evidence/historical_repaired_latency.json) |
| Artifact, model and logo checksums | [checksums.json](evidence/checksums.json) |

JSON copies preserve scientific values and normalize machine-specific paths. Logical paths to local datasets or the withheld packet are provenance references, not claims that those files are bundled. Dataset payloads and row-level samples are excluded. Owned checkpoints are supplied at the nested platform's existing paths and identified by the checksum manifest.

Closure audit: PASS WITH CAVEATS, 9/9 original P1 fixed, zero remaining confirmed P0/P1; local full suite 62 passed. Public checks are separately documented in [ARTIFACT.md](../../ARTIFACT.md). Limits remain: one test attack host-day, overlapping windows, third-party extract, no causal P1/P3 claim, no globally unseen-host/session claim, batch sensitivity, simulator policy and unenforced recommendations.

The historical submission tag remains unchanged. Release publication is held as a draft while author-only CMT and sharing checks are pending.
